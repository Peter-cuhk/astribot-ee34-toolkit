"""LeRobot writer that embeds already-verified JPEG bytes without a PNG round-trip.

LeRobot v0.1's recording API accepts decoded arrays, writes them as temporary
PNG files, then reads those files back while creating each episode Parquet.
For offline conversion the source JPEG already exists, so that path spends most
of its time recompressing pixels that were decoded only for validation.

This module keeps the public LeRobot dataset layout and loader contract.  The
only difference is the construction path: every image carries both the exact
JPEG bytes to store and the decoded RGB pixels used for shape validation and
the same sampled image statistics as the upstream writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from lerobot.common.datasets.compute_stats import auto_downsample_height_width, get_feature_stats, sample_indices
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import (
    check_timestamps_sync,
    get_episode_data_index,
    validate_episode_buffer,
    validate_frame,
)


@dataclass(frozen=True)
class VerifiedJpeg:
    """Exact encoded JPEG plus the RGB pixels produced by a successful decode."""

    encoded: bytes
    pixels: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.encoded, bytes) or not self.encoded:
            raise ValueError("encoded JPEG must be non-empty bytes")
        if not isinstance(self.pixels, np.ndarray):
            raise TypeError("decoded JPEG pixels must be a NumPy array")
        if self.pixels.dtype != np.uint8 or self.pixels.ndim != 3:
            raise ValueError(f"decoded JPEG must be HWC/CHW uint8, got {self.pixels.shape}, {self.pixels.dtype}")


def _image_stats(decoded_images: list[np.ndarray]) -> dict[str, np.ndarray]:
    if not decoded_images:
        raise ValueError("cannot compute image statistics without decoded images")
    indices = sample_indices(len(decoded_images))
    sampled: np.ndarray | None = None
    for sample_index, image_index in enumerate(indices):
        image = decoded_images[image_index]
        if image.shape[0] == 3:
            channel_first = image
        elif image.shape[-1] == 3:
            channel_first = image.transpose(2, 0, 1)
        else:
            raise ValueError(f"decoded image does not have three channels: {image.shape}")
        downsampled = auto_downsample_height_width(channel_first)
        if sampled is None:
            sampled = np.empty((len(indices), *downsampled.shape), dtype=np.uint8)
        sampled[sample_index] = downsampled
    assert sampled is not None
    stats = get_feature_stats(sampled, axis=(0, 2, 3), keepdims=True)
    return {key: value if key == "count" else np.squeeze(value / 255.0, axis=0) for key, value in stats.items()}


def _compute_episode_stats(
    episode_buffer: dict[str, Any],
    features: dict[str, dict[str, Any]],
    decoded_images: dict[str, list[np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, data in episode_buffer.items():
        feature = features[key]
        if feature["dtype"] == "string":
            continue
        if feature["dtype"] == "image":
            stats[key] = _image_stats(decoded_images[key])
            continue
        if feature["dtype"] == "video":
            raise NotImplementedError("embedded JPEG writer does not support video features")
        stats[key] = get_feature_stats(data, axis=0, keepdims=data.ndim == 1)
    return stats


class EmbeddedJpegLeRobotDataset(LeRobotDataset):
    """A LeRobotDataset creation path that stores verified JPEG bytes directly."""

    def _decoded_image_buffer(self) -> dict[str, list[np.ndarray]]:
        buffer = getattr(self, "_embedded_jpeg_decoded_images", None)
        if buffer is None:
            buffer = {key: [] for key in self.meta.image_keys}
            self._embedded_jpeg_decoded_images = buffer
        return buffer

    def _reset_decoded_image_buffer(self) -> None:
        self._embedded_jpeg_decoded_images = {key: [] for key in self.meta.image_keys}

    def add_frame(self, frame: dict[str, Any]) -> None:
        """Add one frame while retaining encoded JPEG bytes in memory."""
        frame = dict(frame)
        for name, value in frame.items():
            if isinstance(value, torch.Tensor):
                frame[name] = value.cpu().numpy()

        validation_frame = dict(frame)
        for key in self.meta.image_keys:
            value = frame.get(key)
            if not isinstance(value, VerifiedJpeg):
                raise TypeError(f"{key} must be VerifiedJpeg when using the embedded JPEG writer")
            validation_frame[key] = value.pixels
        validate_frame(validation_frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        decoded_images = self._decoded_image_buffer()
        for key, value in frame.items():
            if key == "task":
                self.episode_buffer["task"].append(value)
                continue
            if key not in self.features:
                raise ValueError(f"An element of the frame is not in the features: {key}")
            dtype = self.features[key]["dtype"]
            if dtype == "image":
                assert isinstance(value, VerifiedJpeg)
                self.episode_buffer[key].append({"bytes": value.encoded, "path": None})
                decoded_images[key].append(value.pixels)
            elif dtype == "video":
                raise NotImplementedError("embedded JPEG writer does not support video features")
            else:
                self.episode_buffer[key].append(value)
        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None) -> None:
        """Save an episode without creating temporary image files."""
        if episode_data is not None:
            raise NotImplementedError("explicit episode_data is not supported by the embedded JPEG writer")
        episode_buffer = self.episode_buffer
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]
        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            if self.meta.get_task_index(task) is None:
                self.meta.add_task(task)
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, feature in self.features.items():
            if key in {"index", "episode_index", "task_index"} or feature["dtype"] in {"image", "video"}:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        episode_stats = _compute_episode_stats(
            episode_buffer,
            self.features,
            self._decoded_image_buffer(),
        )
        self._save_episode_table(episode_buffer, episode_index)
        self.meta.save_episode(episode_index, episode_length, episode_tasks, episode_stats)

        episode_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        episode_data_index_np = {key: value.numpy() for key, value in episode_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            episode_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        video_files = list(self.root.rglob("*.mp4"))
        if video_files:
            raise AssertionError("embedded JPEG dataset unexpectedly produced video files")
        parquet_files = list(self.root.rglob("*.parquet"))
        if len(parquet_files) != self.num_episodes:
            raise AssertionError(f"expected {self.num_episodes} episode Parquets, found {len(parquet_files)}")

        self.episode_buffer = self.create_episode_buffer()
        self._reset_decoded_image_buffer()

    def clear_episode_buffer(self) -> None:
        self.episode_buffer = self.create_episode_buffer()
        self._reset_decoded_image_buffer()
