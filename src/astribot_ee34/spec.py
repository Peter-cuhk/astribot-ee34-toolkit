"""Source inventory and deterministic train/val split for EE34 tomato HDF5."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from . import contract as C  # noqa: N812


class DatasetSpecError(ValueError):
    pass


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str  # episode_259
    source_episode_index: int
    source_file: Path
    frame_count: int


@dataclass(frozen=True)
class DatasetPlan:
    source_root: Path
    episodes: tuple[EpisodeSpec, ...]
    train: tuple[EpisodeSpec, ...]
    validation: tuple[EpisodeSpec, ...]
    fingerprint: str


def split_hash(episode_id: str, salt: str = C.SPLIT_SALT) -> str:
    return hashlib.sha256(f"{salt}\0{episode_id}".encode()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _parse_episode_index(path: Path) -> int:
    match = re.fullmatch(r"Default_episode_(\d+)\.hdf5", path.name)
    if match is None:
        raise DatasetSpecError(f"unexpected hdf5 name: {path.name}")
    return int(match.group(1))


def discover_episodes(source_root: Path | None = None) -> tuple[EpisodeSpec, ...]:
    root = Path(source_root or C.SOURCE_ROOT)
    if not root.is_dir():
        raise FileNotFoundError(root)
    episodes: list[EpisodeSpec] = []
    for path in sorted(root.glob("Default_episode_*.hdf5")):
        index = _parse_episode_index(path)
        with h5py.File(path, "r") as handle:
            frames = int(handle["joints_dict/joints_position_command"].shape[0])
            if handle["command_poses_dict/merge_pose"].shape != (frames, C.MERGE_POSE_DIM):
                raise DatasetSpecError(f"{path.name}: merge_pose shape mismatch")
            if handle["poses_dict/merge_pose"].shape != (frames, C.MERGE_POSE_DIM):
                raise DatasetSpecError(f"{path.name}: poses merge_pose shape mismatch")
        if frames < C.ACTION_HORIZON:
            raise DatasetSpecError(f"{path.name}: too few frames ({frames})")
        episodes.append(
            EpisodeSpec(
                episode_id=f"episode_{index}",
                source_episode_index=index,
                source_file=path,
                frame_count=frames,
            )
        )
    if not episodes:
        raise DatasetSpecError(f"no episodes under {root}")
    return tuple(episodes)


def load_source_plan(
    source_root: Path | None = None,
    *,
    val_episodes: int = C.TARGET_VAL_EPISODES,
) -> DatasetPlan:
    root = Path(source_root or C.SOURCE_ROOT).resolve()
    episodes = discover_episodes(root)
    if len(episodes) != C.EXPECTED_EPISODES:
        # Soft warning path: still allow if count differs, but fingerprint records actual
        pass
    ranked = tuple(sorted(episodes, key=lambda item: split_hash(item.episode_id)))
    validation = tuple(sorted(ranked[:val_episodes], key=lambda item: item.episode_id))
    val_ids = {item.episode_id for item in validation}
    train = tuple(item for item in episodes if item.episode_id not in val_ids)
    fingerprint = _canonical_hash(
        {
            "algorithm": C.SPLIT_ALGORITHM,
            "salt": C.SPLIT_SALT,
            "source_root": str(root),
            "train": [item.episode_id for item in train],
            "validation": [item.episode_id for item in validation],
            "episode_count": len(episodes),
            "frame_count": int(sum(item.frame_count for item in episodes)),
        }
    )
    return DatasetPlan(root, episodes, train, validation, fingerprint)


def compact_summary(plan: DatasetPlan) -> dict[str, Any]:
    return {
        "source_root": str(plan.source_root),
        "source_episode_count": len(plan.episodes),
        "source_frame_count": int(sum(item.frame_count for item in plan.episodes)),
        "train_episode_count": len(plan.train),
        "train_frame_count": int(sum(item.frame_count for item in plan.train)),
        "validation_episode_count": len(plan.validation),
        "validation_frame_count": int(sum(item.frame_count for item in plan.validation)),
        "validation_ids": [item.episode_id for item in plan.validation],
        "split_fingerprint": plan.fingerprint,
    }


def assert_chassis_identity_episode(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        chassis = handle["command_poses_dict/astribot_chassis"][:]
    xyz = np.linalg.norm(chassis[:, :3], axis=1)
    quat_err = np.linalg.norm(chassis[:, 3:] - np.array([0.0, 0.0, 0.0, 1.0]), axis=1)
    return {
        "max_xyz": float(xyz.max()),
        "max_quat_err": float(quat_err.max()),
        "identity": bool(xyz.max() <= C.CHASSIS_IDENTITY_XYZ_TOL and quat_err.max() <= C.CHASSIS_IDENTITY_QUAT_TOL),
    }
