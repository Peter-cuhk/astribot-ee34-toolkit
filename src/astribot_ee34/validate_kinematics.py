"""Phase-1.5 gate: validate EE34 conversion parity and URDF kinematics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.parquet as pq

from . import contract as C  # noqa: N812
from . import kinematics as kin
from . import pack

FORMAT_VERSION = "astribot-ee34-kinematics-validation/1"
DEFAULT_REPORT_DIR = C.REPORT_ROOT
SPLIT_REPO_IDS = {
    "train": C.TRAIN_REPO_ID,
    "validation": C.VAL_REPO_ID,
}
STREAM_SOURCES = {
    "state": ("observation.state", "poses_dict/merge_pose", "joints_dict/joints_position_state"),
    "action": ("action", "command_poses_dict/merge_pose", "joints_dict/joints_position_command"),
}


class AuditInputError(ValueError):
    pass


@dataclass(frozen=True)
class Thresholds:
    conversion_atol: float = 1e-6
    warning_p95_position_mm: float = 5.0
    warning_p95_rotation_deg: float = 1.0
    fail_p95_position_mm: float = 20.0
    fail_p95_rotation_deg: float = 3.0
    fail_max_position_mm: float = 50.0
    fail_max_rotation_deg: float = 10.0


@dataclass(frozen=True)
class EpisodeTask:
    split: str
    local_episode_index: int
    episode_id: str
    source_file: str
    frame_count: int
    parquet_file: str
    max_frames: int | None


_WORKER_KINEMATICS: kin.AstribotKinematics | None = None


def _init_worker(urdf_path: str, torso_config_path: str) -> None:
    global _WORKER_KINEMATICS
    _WORKER_KINEMATICS = kin.AstribotKinematics(Path(urdf_path), Path(torso_config_path))


def _read_vectors(table: Any, column_name: str) -> np.ndarray:
    if column_name not in table.column_names:
        raise AuditInputError(f"missing parquet column: {column_name}")
    array = np.asarray(table[column_name].combine_chunks().to_pylist(), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != C.EE34_DIM:
        raise AuditInputError(f"{column_name} must be (T,{C.EE34_DIM}), got {array.shape}")
    return array


def _selected_indices(frame_count: int, max_frames: int | None) -> np.ndarray:
    if max_frames is None or max_frames >= frame_count:
        return np.arange(frame_count, dtype=np.int64)
    if max_frames <= 0:
        raise AuditInputError("--max-frames-per-episode must be positive")
    return np.unique(np.linspace(0, frame_count - 1, max_frames, dtype=np.int64))


def _empty_metric() -> dict[str, Any]:
    return {
        "position_mm": [],
        "rotation_deg": [],
        "worst_position": None,
        "worst_rotation": None,
    }


def _audit_episode(task: EpisodeTask) -> dict[str, Any]:
    result: dict[str, Any] = {
        "split": task.split,
        "local_episode_index": task.local_episode_index,
        "episode_id": task.episode_id,
        "source_file": task.source_file,
        "parquet_file": task.parquet_file,
        "frame_count": task.frame_count,
        "sampled_frames": 0,
        "conversion": {},
        "kinematics": {
            stream: {label: _empty_metric() for label in kin.LINK_LABELS}
            for stream in STREAM_SOURCES
        },
        "errors": [],
    }
    try:
        model = _WORKER_KINEMATICS
        if model is None:
            raise RuntimeError("worker kinematics was not initialized")
        parquet_path = Path(task.parquet_file)
        source_path = Path(task.source_file)
        table = pq.read_table(
            parquet_path,
            columns=["observation.state", "action", "frame_index", "episode_index"],
        )
        frames = table.num_rows
        if frames != task.frame_count:
            raise AuditInputError(f"parquet rows {frames} != provenance frame_count {task.frame_count}")
        frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        episode_index = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        if not np.array_equal(frame_index, np.arange(frames, dtype=np.int64)):
            raise AuditInputError("parquet frame_index is not contiguous 0..T-1")
        if not np.all(episode_index == task.local_episode_index):
            raise AuditInputError(
                f"parquet episode_index mismatch: expected {task.local_episode_index}, "
                f"got {np.unique(episode_index).tolist()}"
            )
        stored = {stream: _read_vectors(table, column) for stream, (column, _, _) in STREAM_SOURCES.items()}
        indices = _selected_indices(frames, task.max_frames)
        result["sampled_frames"] = int(indices.size)

        with h5py.File(source_path, "r") as handle:
            for stream, (_, merge_key, joints_key) in STREAM_SOURCES.items():
                merge = np.asarray(handle[merge_key][:], dtype=np.float64)
                joints = np.asarray(handle[joints_key][:], dtype=np.float64)
                if merge.shape != (frames, C.MERGE_POSE_DIM):
                    raise AuditInputError(f"{merge_key} shape {merge.shape} != ({frames},{C.MERGE_POSE_DIM})")
                if joints.shape != (frames, C.JOINT_DIM):
                    raise AuditInputError(f"{joints_key} shape {joints.shape} != ({frames},{C.JOINT_DIM})")

                expected = pack.pack_episode_ee34(merge[indices], joints[indices])
                actual = stored[stream][indices]
                difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
                if not np.all(np.isfinite(difference)):
                    raise AuditInputError(f"{stream}: non-finite conversion comparison")
                flat_index = int(np.argmax(difference))
                local_row, dimension = np.unravel_index(flat_index, difference.shape)
                result["conversion"][stream] = {
                    "max_abs": float(difference[local_row, dimension]),
                    "worst_frame": int(indices[local_row]),
                    "worst_dimension": int(dimension),
                }

                for sample_row, frame in enumerate(indices):
                    q20 = kin.joints25_to_urdf20(joints[frame])
                    fk = model.fk(q20)
                    targets = kin.ee34_to_target_matrices(actual[sample_row])
                    for label in kin.LINK_LABELS:
                        error = kin.se3_error(fk[label], targets[label])
                        metric = result["kinematics"][stream][label]
                        metric["position_mm"].append(error.position_mm)
                        metric["rotation_deg"].append(error.rotation_deg)
                        location = {
                            "episode_id": task.episode_id,
                            "local_episode_index": task.local_episode_index,
                            "frame": int(frame),
                        }
                        if metric["worst_position"] is None or error.position_mm > metric["worst_position"]["value"]:
                            metric["worst_position"] = {**location, "value": error.position_mm}
                        if metric["worst_rotation"] is None or error.rotation_deg > metric["worst_rotation"]["value"]:
                            metric["worst_rotation"] = {**location, "value": error.rotation_deg}
    except Exception as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    return result


def _find_parquet(split_root: Path, local_episode_index: int) -> Path:
    matches = sorted(split_root.glob(f"data/chunk-*/episode_{local_episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise AuditInputError(
            f"expected exactly one parquet for local episode {local_episode_index} under {split_root}, got {matches}"
        )
    return matches[0]


def _load_tasks(args: argparse.Namespace) -> tuple[list[EpisodeTask], dict[str, str]]:
    requested_ids = set(args.episode_id or [])
    found_ids: set[str] = set()
    tasks: list[EpisodeTask] = []
    fingerprints: dict[str, str] = {}
    for split in args.split:
        repo_id = SPLIT_REPO_IDS[split]
        split_root = args.output_root.joinpath(*repo_id.split("/"))
        records_path = split_root / "meta/ee34_conversion_episodes.jsonl"
        metadata_path = split_root / "meta/ee34_conversion.json"
        if not records_path.is_file() or not metadata_path.is_file():
            raise AuditInputError(f"missing conversion provenance under {split_root}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("contract_id") != C.CONTRACT_ID:
            raise AuditInputError(f"{split}: contract_id mismatch")
        fingerprints[split] = str(metadata.get("split_fingerprint", ""))
        records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
        for local_index, record in enumerate(records):
            episode_id = str(record["episode_id"])
            if requested_ids and episode_id not in requested_ids:
                continue
            found_ids.add(episode_id)
            tasks.append(
                EpisodeTask(
                    split=split,
                    local_episode_index=local_index,
                    episode_id=episode_id,
                    source_file=str(record["source_file"]),
                    frame_count=int(record["frame_count"]),
                    parquet_file=str(_find_parquet(split_root, local_index)),
                    max_frames=args.max_frames_per_episode,
                )
            )
    missing_ids = requested_ids - found_ids
    if missing_ids:
        raise AuditInputError(f"episode IDs not found in selected splits: {sorted(missing_ids)}")
    if not tasks:
        raise AuditInputError("no episodes selected")
    return tasks, fingerprints


def _percentiles(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _aggregate(
    results: list[dict[str, Any]],
    thresholds: Thresholds,
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    grouped: dict[str, Any] = {}
    for split in sorted({result["split"] for result in results}):
        grouped[split] = {}
        split_results = [result for result in results if result["split"] == split]
        for stream in STREAM_SOURCES:
            grouped[split][stream] = {}
            conversion_worst = max(
                (
                    (result["conversion"].get(stream, {}).get("max_abs", -1.0), result)
                    for result in split_results
                    if not result["errors"]
                ),
                default=(-1.0, None),
                key=lambda item: item[0],
            )
            if conversion_worst[1] is not None:
                conversion_detail = conversion_worst[1]["conversion"][stream]
                grouped[split][stream]["conversion"] = {
                    **conversion_detail,
                    "episode_id": conversion_worst[1]["episode_id"],
                    "local_episode_index": conversion_worst[1]["local_episode_index"],
                }
                if conversion_detail["max_abs"] > thresholds.conversion_atol:
                    errors.append(
                        f"{split}/{stream}: conversion max_abs={conversion_detail['max_abs']:.3g} "
                        f"> {thresholds.conversion_atol:.3g}"
                    )
            for label in kin.LINK_LABELS:
                position: list[float] = []
                rotation: list[float] = []
                worst_position = None
                worst_rotation = None
                for result in split_results:
                    metric = result["kinematics"][stream][label]
                    position.extend(metric["position_mm"])
                    rotation.extend(metric["rotation_deg"])
                    candidate_position = metric["worst_position"]
                    candidate_rotation = metric["worst_rotation"]
                    if candidate_position is not None and (
                        worst_position is None or candidate_position["value"] > worst_position["value"]
                    ):
                        worst_position = candidate_position
                    if candidate_rotation is not None and (
                        worst_rotation is None or candidate_rotation["value"] > worst_rotation["value"]
                    ):
                        worst_rotation = candidate_rotation
                if not position:
                    continue
                position_stats = _percentiles(position)
                rotation_stats = _percentiles(rotation)
                grouped[split][stream][label] = {
                    "position_mm": position_stats,
                    "rotation_deg": rotation_stats,
                    "worst_position": worst_position,
                    "worst_rotation": worst_rotation,
                }
                group_name = f"{split}/{stream}/{label}"
                if (
                    position_stats["p95"] > thresholds.fail_p95_position_mm
                    or rotation_stats["p95"] > thresholds.fail_p95_rotation_deg
                    or position_stats["max"] > thresholds.fail_max_position_mm
                    or rotation_stats["max"] > thresholds.fail_max_rotation_deg
                ):
                    errors.append(
                        f"{group_name}: kinematics hard threshold exceeded "
                        f"(pos p95/max={position_stats['p95']:.3f}/{position_stats['max']:.3f} mm, "
                        f"rot p95/max={rotation_stats['p95']:.3f}/{rotation_stats['max']:.3f} deg)"
                    )
                elif (
                    position_stats["p95"] > thresholds.warning_p95_position_mm
                    or rotation_stats["p95"] > thresholds.warning_p95_rotation_deg
                ):
                    warnings.append(
                        f"{group_name}: kinematics warning threshold exceeded "
                        f"(pos p95={position_stats['p95']:.3f} mm, rot p95={rotation_stats['p95']:.3f} deg)"
                    )
    return grouped, errors, warnings


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# EE34 kinematics validation ({report['contract_id']})",
        "",
        f"- ok: **{str(report['ok']).lower()}**",
        f"- partial: `{str(report['partial']).lower()}`",
        f"- episodes: `{report['counts']['episodes']}`",
        f"- source frames: `{report['counts']['source_frames']}`",
        f"- audited stream-frames: `{report['counts']['audited_stream_frames']}`",
        f"- URDF: `{report['model']['urdf']}`",
        f"- weld pose: `{report['model']['weld_to_base_pose']}`",
        "",
        "## Metrics",
        "",
        "| split | stream | link | conversion max abs | pos p50/p95/p99/max mm | rot p50/p95/p99/max deg |",
        "|---|---|---|---:|---:|---:|",
    ]
    for split, streams in report["metrics"].items():
        for stream, values in streams.items():
            conversion = values.get("conversion", {}).get("max_abs", float("nan"))
            for label in kin.LINK_LABELS:
                if label not in values:
                    continue
                position = values[label]["position_mm"]
                rotation = values[label]["rotation_deg"]
                lines.append(
                    f"| {split} | {stream} | {label} | {conversion:.3g} | "
                    f"{position['p50']:.3f}/{position['p95']:.3f}/{position['p99']:.3f}/{position['max']:.3f} | "
                    f"{rotation['p50']:.3f}/{rotation['p95']:.3f}/{rotation['p99']:.3f}/{rotation['max']:.3f} |"
                )
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in report["warnings"]] or ["- None"])
    lines.extend(["", "## Errors", ""])
    lines.extend([f"- {error}" for error in report["errors"]] or ["- None"])
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=C.OUTPUT_ROOT)
    parser.add_argument("--split", action="append", choices=tuple(SPLIT_REPO_IDS), help="Repeat to select splits")
    parser.add_argument("--episode-id", action="append", help="Repeat to select source episode IDs")
    parser.add_argument("--all-frames", action="store_true", help="Explicitly request the default full-frame audit")
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--urdf", type=Path, default=kin.DEFAULT_URDF_PATH)
    parser.add_argument("--torso-config", type=Path, default=kin.DEFAULT_TORSO_CONFIG_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--conversion-atol", type=float, default=Thresholds.conversion_atol)
    parser.add_argument("--warning-p95-position-mm", type=float, default=Thresholds.warning_p95_position_mm)
    parser.add_argument("--warning-p95-rotation-deg", type=float, default=Thresholds.warning_p95_rotation_deg)
    parser.add_argument("--fail-p95-position-mm", type=float, default=Thresholds.fail_p95_position_mm)
    parser.add_argument("--fail-p95-rotation-deg", type=float, default=Thresholds.fail_p95_rotation_deg)
    parser.add_argument("--fail-max-position-mm", type=float, default=Thresholds.fail_max_position_mm)
    parser.add_argument("--fail-max-rotation-deg", type=float, default=Thresholds.fail_max_rotation_deg)
    args = parser.parse_args()
    if args.split is None:
        args.split = list(SPLIT_REPO_IDS)
    else:
        args.split = list(dict.fromkeys(args.split))
    if args.all_frames and args.max_frames_per_episode is not None:
        parser.error("--all-frames conflicts with --max-frames-per-episode")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def main() -> int:
    args = _parse_args()
    started = time.monotonic()
    thresholds = Thresholds(
        conversion_atol=args.conversion_atol,
        warning_p95_position_mm=args.warning_p95_position_mm,
        warning_p95_rotation_deg=args.warning_p95_rotation_deg,
        fail_p95_position_mm=args.fail_p95_position_mm,
        fail_p95_rotation_deg=args.fail_p95_rotation_deg,
        fail_max_position_mm=args.fail_max_position_mm,
        fail_max_rotation_deg=args.fail_max_rotation_deg,
    )
    try:
        # Validate model inputs before starting a worker pool so configuration failures map to exit 2.
        model = kin.AstribotKinematics(args.urdf, args.torso_config)
        tasks, fingerprints = _load_tasks(args)
    except (AuditInputError, kin.KinematicsConfigError, OSError, ValueError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(tasks)),
        initializer=_init_worker,
        initargs=(str(args.urdf), str(args.torso_config)),
    ) as executor:
        futures = {executor.submit(_audit_episode, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "split": task.split,
                    "local_episode_index": task.local_episode_index,
                    "episode_id": task.episode_id,
                    "errors": [f"worker failure: {type(error).__name__}: {error}"],
                    "conversion": {},
                    "kinematics": {
                        stream: {label: _empty_metric() for label in kin.LINK_LABELS}
                        for stream in STREAM_SOURCES
                    },
                    "frame_count": task.frame_count,
                    "sampled_frames": 0,
                }
            results.append(result)
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "completed": completed,
                        "total": len(tasks),
                        "split": task.split,
                        "episode_id": task.episode_id,
                        "errors": len(result["errors"]),
                    }
                ),
                flush=True,
            )

    results.sort(key=lambda item: (item["split"], item["local_episode_index"]))
    metrics, threshold_errors, warnings = _aggregate(results, thresholds)
    episode_errors = [
        f"{result['split']}/{result['episode_id']}: {error}"
        for result in results
        for error in result["errors"]
    ]
    errors = episode_errors + threshold_errors
    report = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "contract_id": C.CONTRACT_ID,
        "contract_version": C.CONTRACT_VERSION,
        "ok": not errors,
        "partial": bool(
            args.episode_id
            or args.max_frames_per_episode is not None
            or set(args.split) != set(SPLIT_REPO_IDS)
        ),
        "counts": {
            "episodes": len(results),
            "source_frames": int(sum(result["frame_count"] for result in results)),
            "audited_stream_frames": int(sum(result["sampled_frames"] for result in results) * len(STREAM_SOURCES)),
        },
        "split_fingerprints": fingerprints,
        "model": {
            "urdf": str(args.urdf.resolve()),
            "urdf_sha256": kin.sha256_file(args.urdf),
            "torso_config": str(args.torso_config.resolve()),
            "torso_config_sha256": kin.sha256_file(args.torso_config),
            "weld_to_base_pose": model.weld_pose6.tolist(),
            "joint_names": list(model.joint_names),
            "end_effector_frames": kin.END_EFFECTOR_FRAMES,
        },
        "thresholds": asdict(thresholds),
        "metrics": metrics,
        "warnings": warnings,
        "errors": errors,
        "episode_results": [
            {
                "split": result["split"],
                "local_episode_index": result["local_episode_index"],
                "episode_id": result["episode_id"],
                "frame_count": result["frame_count"],
                "sampled_frames": result["sampled_frames"],
                "conversion": result["conversion"],
                "errors": result["errors"],
            }
            for result in results
        ],
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / "ee34_kinematics_validation.json"
    markdown_path = args.report_dir / "ee34_kinematics_validation.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    markdown_path.write_text(_render_markdown(report))
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "partial": report["partial"],
                "episodes": report["counts"]["episodes"],
                "warnings": len(warnings),
                "errors": len(errors),
                "json_report": str(json_path),
                "markdown_report": str(markdown_path),
                "elapsed_s": report["elapsed_s"],
            },
            indent=2,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
