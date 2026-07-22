"""Convert Astribot HDF5 (pose7+joints25) to LeRobot v2.1 EE34 datasets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

from . import contract as C  # noqa: N812
from . import pack
from . import spec as ee34_spec
from .embedded_jpeg_lerobot import EmbeddedJpegLeRobotDataset, VerifiedJpeg

FORMAT_VERSION = "astribot-ee34-pick-tomato-v1-conversion/1"


def _features() -> dict[str, dict[str, Any]]:
    names = list(C.EE34_NAMES)
    features: dict[str, dict[str, Any]] = {}
    for cam, shape in C.IMAGE_SHAPES.items():
        h, w, _ = shape
        key = C.CAMERA_MAP[cam]
        features[key] = {
            "dtype": "image",
            "shape": (3, h, w),
            "names": ["channels", "height", "width"],
        }
    features["observation.state"] = {"dtype": "float32", "shape": (C.EE34_DIM,), "names": [names]}
    features["action"] = {"dtype": "float32", "shape": (C.EE34_DIM,), "names": [names]}
    return features


def _validate_repo_id(repo_id: str) -> None:
    path = Path(repo_id)
    if path.is_absolute() or not repo_id or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe LeRobot repo ID: {repo_id!r}")


def _repo_root(output_root: Path, repo_id: str) -> Path:
    _validate_repo_id(repo_id)
    return output_root.joinpath(*repo_id.split("/"))


def _staging(path: Path) -> Path:
    return path.with_name(f".{path.name}.staging")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _bounded_error(error: Exception) -> dict[str, str]:
    return {"error_type": type(error).__name__, "detail": " ".join(str(error).split())[:400]}


def _split_jpeg_blob(blob: np.ndarray, sizes: np.ndarray) -> list[bytes]:
    sizes_i = sizes.astype(np.int64)
    if int(sizes_i.sum()) != int(blob.shape[0]):
        raise ValueError("rgb blob length != sum(rgb_size)")
    out: list[bytes] = []
    offset = 0
    for size in sizes_i:
        chunk = bytes(blob[offset : offset + int(size)])
        if len(chunk) < 3 or chunk[:3] != b"\xff\xd8\xff":
            raise ValueError("invalid JPEG header in rgb blob")
        out.append(chunk)
        offset += int(size)
    return out


def _decode_verified(encoded: bytes, expected_hwc: tuple[int, int, int]) -> VerifiedJpeg:
    with Image.open(BytesIO(encoded)) as image:
        image.load()
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if pixels.shape != expected_hwc:
        raise ValueError(f"unexpected image shape {pixels.shape}, expected {expected_hwc}")
    return VerifiedJpeg(encoded=encoded, pixels=pixels)


def _create_dataset(repo_id: str, root: Path) -> EmbeddedJpegLeRobotDataset:
    return EmbeddedJpegLeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="astribot",
        fps=C.FPS,
        features=_features(),
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )


def convert_episode(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        joints_cmd = handle["joints_dict/joints_position_command"][:]
        joints_state = handle["joints_dict/joints_position_state"][:]
        merge_cmd = handle["command_poses_dict/merge_pose"][:]
        merge_state = handle["poses_dict/merge_pose"][:]
        timestamps = handle["time"][:].astype(np.float64)
        camera_bytes: dict[str, list[bytes]] = {}
        for cam in C.SOURCE_CAMERAS:
            blob = handle[f"images_dict/{cam}/rgb"][:]
            sizes = handle[f"images_dict/{cam}/rgb_size"][:]
            camera_bytes[cam] = _split_jpeg_blob(blob, sizes)

    frames = merge_cmd.shape[0]
    if joints_cmd.shape[0] != frames or merge_state.shape[0] != frames:
        raise ValueError("frame count mismatch among joints/poses")
    for cam, chunks in camera_bytes.items():
        if len(chunks) != frames:
            raise ValueError(f"{cam}: image count {len(chunks)} != frames {frames}")

    actions = pack.pack_episode_ee34(merge_cmd, joints_cmd)
    states = pack.pack_episode_ee34(merge_state, joints_state)
    if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(states)):
        raise ValueError("packed ee34 contains NaN/Inf")

    images: dict[str, list[VerifiedJpeg]] = {cam: [] for cam in C.SOURCE_CAMERAS}
    for cam in C.SOURCE_CAMERAS:
        expected = C.IMAGE_SHAPES[cam]
        images[cam] = [_decode_verified(chunk, expected) for chunk in camera_bytes[cam]]

    if frames >= 2:
        dt = np.diff(timestamps)
        rel = (
            np.arange(frames, dtype=np.float64) / float(C.FPS)
            if np.median(dt) > 1.0
            else timestamps - timestamps[0]
        )
    else:
        rel = np.zeros((frames,), dtype=np.float64)

    chassis_info = ee34_spec.assert_chassis_identity_episode(path)
    return {
        "states": states,
        "actions": actions,
        "images": images,
        "timestamps": rel.astype(np.float64),
        "chassis_identity": chassis_info,
        "right_gripper_abs_max": float(np.max(np.abs(actions[:, C.EE34_RIGHT_GRIPPER]))),
        "sample_action0": actions[0].tolist(),
        "sample_state0": states[0].tolist(),
    }


def _write_episode(dataset: EmbeddedJpegLeRobotDataset, payload: dict[str, Any], *, prompt: str) -> None:
    frames = payload["actions"].shape[0]
    for index in range(frames):
        frame = {
            "observation.state": payload["states"][index],
            "action": payload["actions"][index],
            "task": prompt,
        }
        for cam in C.SOURCE_CAMERAS:
            frame[C.CAMERA_MAP[cam]] = payload["images"][cam][index]
        dataset.add_frame(frame)
        for cam in C.SOURCE_CAMERAS:
            payload["images"][cam][index] = None  # type: ignore[assignment]
    dataset.save_episode()
    payload["images"].clear()


def convert(args: argparse.Namespace) -> None:
    plan = ee34_spec.load_source_plan(args.source_root, val_episodes=args.val_episodes)
    summary = ee34_spec.compact_summary(plan)
    if args.dry_run:
        print(json.dumps({"status": "dry-run", **summary}, indent=2, sort_keys=True))
        return

    output_root = args.output_root.resolve()
    train_final = _repo_root(output_root, args.train_repo_id)
    val_final = _repo_root(output_root, args.val_repo_id)
    audit_final = _repo_root(output_root, args.audit_repo_id)
    finals = (train_final, val_final, audit_final)
    staging = tuple(_staging(path) for path in finals)

    for final in finals:
        if final.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {final}; pass --overwrite")
    for path in staging:
        if path.exists():
            if not args.overwrite:
                raise FileExistsError(f"staging exists: {path}")
            shutil.rmtree(path)

    train_eps = list(plan.train)
    val_eps = list(plan.validation)
    if args.limit_episodes is not None:
        train_eps = train_eps[: args.limit_episodes]
        val_eps = val_eps[: min(args.limit_episodes, len(val_eps))]

    quarantine: list[dict[str, Any]] = []
    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    reference_vectors: dict[str, Any] = {}

    try:
        started = time.monotonic()

        val_dataset = _create_dataset(args.val_repo_id, staging[1])
        for index, episode in enumerate(val_eps, start=1):
            try:
                payload = convert_episode(episode.source_file)
                _write_episode(val_dataset, payload, prompt=C.DEFAULT_PROMPT)
            except Exception as error:
                val_dataset.clear_episode_buffer()
                quarantine.append({"episode_id": episode.episode_id, "stage": "validation", **_bounded_error(error)})
                print(
                    json.dumps(
                        {
                            "event": "quarantine",
                            "split": "validation",
                            "episode_id": episode.episode_id,
                            **_bounded_error(error),
                        }
                    ),
                    flush=True,
                )
                continue
            if payload["right_gripper_abs_max"] < 1e-8:
                warnings.append(f"{episode.episode_id}: right gripper action ~0")
            if not payload["chassis_identity"]["identity"]:
                warnings.append(f"{episode.episode_id}: chassis pose not identity")
            record = {
                "episode_id": episode.episode_id,
                "source_episode_index": episode.source_episode_index,
                "source_file": str(episode.source_file),
                "frame_count": episode.frame_count,
                "split": "validation",
                "split_hash": ee34_spec.split_hash(episode.episode_id),
                "sample_action0": payload["sample_action0"],
                "sample_state0": payload["sample_state0"],
            }
            val_records.append(record)
            reference_vectors[episode.episode_id] = {
                "action0": payload["sample_action0"],
                "state0": payload["sample_state0"],
            }
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "split": "validation",
                        "processed": index,
                        "total": len(val_eps),
                        "episode_id": episode.episode_id,
                        "elapsed_s": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
        val_dataset.stop_image_writer()

        train_dataset = _create_dataset(args.train_repo_id, staging[0])
        for index, episode in enumerate(train_eps, start=1):
            try:
                payload = convert_episode(episode.source_file)
                _write_episode(train_dataset, payload, prompt=C.DEFAULT_PROMPT)
            except Exception as error:
                train_dataset.clear_episode_buffer()
                quarantine.append({"episode_id": episode.episode_id, "stage": "train", **_bounded_error(error)})
                print(
                    json.dumps(
                        {
                            "event": "quarantine",
                            "split": "train",
                            "episode_id": episode.episode_id,
                            **_bounded_error(error),
                        }
                    ),
                    flush=True,
                )
                continue
            if payload["right_gripper_abs_max"] < 1e-8:
                warnings.append(f"{episode.episode_id}: right gripper action ~0")
            record = {
                "episode_id": episode.episode_id,
                "source_episode_index": episode.source_episode_index,
                "source_file": str(episode.source_file),
                "frame_count": episode.frame_count,
                "split": "train",
                "split_hash": ee34_spec.split_hash(episode.episode_id),
                "sample_action0": payload["sample_action0"],
                "sample_state0": payload["sample_state0"],
            }
            train_records.append(record)
            if episode.episode_id not in reference_vectors:
                reference_vectors[episode.episode_id] = {
                    "action0": payload["sample_action0"],
                    "state0": payload["sample_state0"],
                }
            if index % 5 == 0 or index == len(train_eps):
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "split": "train",
                            "processed": index,
                            "total": len(train_eps),
                            "episode_id": episode.episode_id,
                            "elapsed_s": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
        train_dataset.stop_image_writer()

        if not train_records or not val_records:
            raise RuntimeError("conversion produced empty train or validation set")

        staging[2].mkdir(parents=True, exist_ok=False)
        manifest = {
            "format_version": FORMAT_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "partial": args.limit_episodes is not None,
            "contract_id": C.CONTRACT_ID,
            "contract_version": C.CONTRACT_VERSION,
            "language_source": C.LANGUAGE_SOURCE,
            "prompt": C.DEFAULT_PROMPT,
            "asset_id": C.ASSET_ID,
            "source_root": str(plan.source_root),
            "split_fingerprint": plan.fingerprint,
            "split": {
                "algorithm": C.SPLIT_ALGORITHM,
                "salt": C.SPLIT_SALT,
                "validation_ids": [item["episode_id"] for item in val_records],
                "train_ids": [item["episode_id"] for item in train_records],
            },
            "repositories": {
                "train": {"repo_id": args.train_repo_id, "root": str(train_final)},
                "validation": {"repo_id": args.val_repo_id, "root": str(val_final)},
                "audit": {"repo_id": args.audit_repo_id, "root": str(audit_final)},
            },
            "counts": {
                "source_episodes": len(plan.episodes),
                "source_frames": int(sum(item.frame_count for item in plan.episodes)),
                "train_episodes": len(train_records),
                "train_frames": int(sum(item["frame_count"] for item in train_records)),
                "validation_episodes": len(val_records),
                "validation_frames": int(sum(item["frame_count"] for item in val_records)),
                "quarantine": len(quarantine),
            },
            "warnings": sorted(set(warnings)),
            "quarantine": quarantine,
            "reference_vectors": reference_vectors,
            "toolkit_source_commit": os.environ.get("ASTRIBOT_EE34_SOURCE_COMMIT"),
        }
        _write_json(staging[2] / "conversion_manifest.json", manifest)
        _write_jsonl(staging[2] / "episodes.jsonl", train_records + val_records)
        for root, records in ((staging[0], train_records), (staging[1], val_records)):
            _write_json(root / "meta" / "ee34_conversion.json", manifest)
            _write_jsonl(root / "meta" / "ee34_conversion_episodes.jsonl", records)

        if args.overwrite:
            for final in finals:
                if final.exists():
                    shutil.rmtree(final)
        for temporary, final in zip(staging, finals, strict=True):
            temporary.replace(final)

        reports = C.REPORT_ROOT
        reports.mkdir(parents=True, exist_ok=True)
        _write_json(reports / "ee34_conversion_manifest.json", manifest)

        print(
            json.dumps(
                {
                    "status": "conversion complete",
                    "train_root": str(train_final),
                    "validation_root": str(val_final),
                    "audit_root": str(audit_final),
                    "train_episodes": len(train_records),
                    "validation_episodes": len(val_records),
                    "quarantine": len(quarantine),
                    "warnings": len(set(warnings)),
                    "elapsed_s": round(time.monotonic() - started, 1),
                },
                indent=2,
                sort_keys=True,
            )
        )
    except BaseException:
        for path in staging:
            if path.exists():
                shutil.rmtree(path)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=C.SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=C.OUTPUT_ROOT)
    parser.add_argument("--train-repo-id", default=C.TRAIN_REPO_ID)
    parser.add_argument("--val-repo-id", default=C.VAL_REPO_ID)
    parser.add_argument("--audit-repo-id", default="astribot/ee34_pick_tomato_v1_audit")
    parser.add_argument("--val-episodes", type=int, default=C.TARGET_VAL_EPISODES)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(_parse_args())
