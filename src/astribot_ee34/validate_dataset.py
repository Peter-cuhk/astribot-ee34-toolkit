"""Validate exported EE34 LeRobot datasets against contract + conversion manifest."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from . import contract as C  # noqa: N812
from . import pack

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:  # pragma: no cover
    LeRobotDataset = None  # type: ignore


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _check_repo(root: Path, expected_dim: int = C.EE34_DIM) -> dict[str, Any]:
    report: dict[str, Any] = {"root": str(root), "ok": False, "errors": [], "warnings": [], "stats": {}}
    if not root.is_dir():
        report["errors"].append("missing dataset root")
        return report
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        report["errors"].append("missing meta/info.json")
        return report
    info = _load_json(info_path)
    features = info.get("features", {})
    for key in (
        "observation.state",
        "action",
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ):
        if key not in features:
            report["errors"].append(f"missing feature: {key}")
    for key in ("observation.state", "action"):
        shape = tuple(features.get(key, {}).get("shape", ()))
        if shape != (expected_dim,):
            report["errors"].append(f"{key} shape {shape} != ({expected_dim},)")

    if LeRobotDataset is None:
        report["errors"].append("lerobot not importable")
        return report

    parts = root.parts
    repo_id = "/".join(parts[-2:])
    dataset = LeRobotDataset(repo_id, root=root)
    n = len(dataset)
    report["stats"]["frames"] = int(n)
    report["stats"]["episodes"] = int(dataset.meta.total_episodes)
    if n == 0:
        report["errors"].append("empty dataset")
        return report

    rng = np.random.default_rng(0)
    sample_idx = sorted({int(x) for x in rng.choice(n, size=min(16, n), replace=False)})
    max_orth = 0.0
    for idx in sample_idx:
        item = dataset[idx]
        state = np.asarray(item["observation.state"], dtype=np.float64).reshape(-1)
        action = np.asarray(item["action"], dtype=np.float64).reshape(-1)
        if state.shape != (expected_dim,) or action.shape != (expected_dim,):
            report["errors"].append(f"frame {idx}: bad state/action shape")
            continue
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
            report["errors"].append(f"frame {idx}: non-finite state/action")
            continue
        for sl in (C.EE34_TORSO, C.EE34_LEFT, C.EE34_RIGHT):
            max_orth = max(max_orth, pack.so3_block_orthonormality_error(action[sl]))
            max_orth = max(max_orth, pack.so3_block_orthonormality_error(state[sl]))
        for cam_key in (
            "observation.images.head",
            "observation.images.left_wrist",
            "observation.images.right_wrist",
        ):
            image = np.asarray(item[cam_key])
            if image.ndim != 3 or (image.shape[0] != 3 and image.shape[-1] != 3):
                report["errors"].append(f"frame {idx}: bad image {cam_key} shape {image.shape}")
    report["stats"]["max_so3_orth_error"] = max_orth
    if max_orth > C.SO3_ORTH_TOL:
        report["errors"].append(f"SO3 orthonormality error too large: {max_orth}")

    tasks_path = root / "meta" / "tasks.jsonl"
    if tasks_path.is_file():
        tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line]
        prompts = {item.get("task") for item in tasks}
        if prompts and prompts != {C.DEFAULT_PROMPT}:
            report["warnings"].append(f"unexpected prompts: {sorted(prompts)}")

    report["ok"] = len(report["errors"]) == 0
    return report


def _check_reference_vectors(manifest: dict[str, Any], train_root: Path, val_root: Path) -> list[str]:
    errors: list[str] = []
    refs = manifest.get("reference_vectors") or {}
    if not refs:
        return errors
    checked = 0
    for split_root in (train_root, val_root):
        ep_path = split_root / "meta" / "ee34_conversion_episodes.jsonl"
        if not ep_path.is_file():
            continue
        for line in ep_path.read_text().splitlines():
            if not line:
                continue
            record = json.loads(line)
            episode_id = record["episode_id"]
            if episode_id not in refs:
                continue
            source = Path(record["source_file"])
            with h5py.File(source, "r") as handle:
                action0 = pack.pack_ee34_from_merge_and_joints(
                    handle["command_poses_dict/merge_pose"][0],
                    handle["joints_dict/joints_position_command"][0],
                )
                state0 = pack.pack_ee34_from_merge_and_joints(
                    handle["poses_dict/merge_pose"][0],
                    handle["joints_dict/joints_position_state"][0],
                )
            ref = refs[episode_id]
            if not np.allclose(action0, np.asarray(ref["action0"], dtype=np.float32), atol=1e-6):
                errors.append(f"{episode_id}: action0 drifted vs conversion reference")
            if not np.allclose(state0, np.asarray(ref["state0"], dtype=np.float32), atol=1e-6):
                errors.append(f"{episode_id}: state0 drifted vs conversion reference")
            checked += 1
            if checked >= 5:
                return errors
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=C.OUTPUT_ROOT)
    parser.add_argument("--train-repo-id", default=C.TRAIN_REPO_ID)
    parser.add_argument("--val-repo-id", default=C.VAL_REPO_ID)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=C.REPORT_ROOT / "ee34_conversion_manifest.json",
    )
    args = parser.parse_args()

    train_root = args.output_root.joinpath(*args.train_repo_id.split("/"))
    val_root = args.output_root.joinpath(*args.val_repo_id.split("/"))
    train_report = _check_repo(train_root)
    val_report = _check_repo(val_root)

    errors: list[str] = []
    warnings: list[str] = []
    if args.manifest.is_file():
        manifest = _load_json(args.manifest)
        if manifest.get("contract_id") != C.CONTRACT_ID:
            errors.append("manifest contract_id mismatch")
        if manifest.get("partial"):
            warnings.append("manifest marked partial=true")
        warnings.extend(manifest.get("warnings") or [])
        if manifest.get("quarantine"):
            warnings.append(f"quarantine count={len(manifest['quarantine'])}")
        errors.extend(_check_reference_vectors(manifest, train_root, val_root))
        counts = manifest.get("counts", {})
        if train_report["stats"].get("episodes") != counts.get("train_episodes"):
            errors.append("train episode count != manifest")
        if val_report["stats"].get("episodes") != counts.get("validation_episodes"):
            errors.append("val episode count != manifest")
        if train_report["stats"].get("frames") != counts.get("train_frames"):
            errors.append("train frame count != manifest")
        if val_report["stats"].get("frames") != counts.get("validation_frames"):
            errors.append("val frame count != manifest")
    else:
        errors.append(f"missing conversion manifest: {args.manifest}")

    if not train_report["ok"]:
        errors.extend([f"train: {e}" for e in train_report["errors"]])
    if not val_report["ok"]:
        errors.extend([f"val: {e}" for e in val_report["errors"]])

    ok = len(errors) == 0
    summary = {
        "ok": ok,
        "created_at": datetime.now(UTC).isoformat(),
        "contract_id": C.CONTRACT_ID,
        "errors": errors,
        "warnings": warnings,
        "train": train_report,
        "validation": val_report,
        "manifest": str(args.manifest),
    }
    out_json = C.REPORT_ROOT / "ee34_dataset_validation.json"
    out_md = C.REPORT_ROOT / "ee34_dataset_validation.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    warn_preview = warnings[:20]
    out_md.write_text(
        "\n".join(
            [
                "# EE34 dataset validation",
                "",
                f"- ok: **{ok}**",
                f"- errors: {errors or '[]'}",
                f"- warnings: {warn_preview}{'...' if len(warnings) > 20 else ''}",
                f"- train episodes/frames: {train_report['stats']}",
                f"- val episodes/frames: {val_report['stats']}",
                "",
            ]
        )
        + "\n"
    )
    for root in (train_root, val_root):
        if root.is_dir():
            (root / "validation_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps({"ok": ok, "json": str(out_json), "md": str(out_md)}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
