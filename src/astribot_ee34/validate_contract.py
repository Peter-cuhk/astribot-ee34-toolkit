"""Phase-0 gate: validate EE34 packing on sample HDF5 episodes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from . import contract as C  # noqa: N812
from . import pack


def _episode_path(source_root: Path, episode_id: int) -> Path:
    return source_root / f"Default_episode_{episode_id}.hdf5"


def _list_episode_ids(source_root: Path) -> list[int]:
    ids: list[int] = []
    for path in sorted(source_root.glob("Default_episode_*.hdf5")):
        stem = path.stem  # Default_episode_259
        ids.append(int(stem.rsplit("_", 1)[-1]))
    return ids


def _pick_sample_ids(all_ids: list[int]) -> list[int]:
    if len(all_ids) < 3:
        return list(all_ids)
    mid = all_ids[len(all_ids) // 2]
    return [all_ids[0], mid, all_ids[-1]]


def _check_fields(f: h5py.File) -> list[str]:
    required = [
        "joints_dict/joints_position_state",
        "joints_dict/joints_position_command",
        "poses_dict/merge_pose",
        "command_poses_dict/merge_pose",
        "images_dict/head/rgb",
        "images_dict/head/rgb_size",
        "images_dict/left/rgb",
        "images_dict/left/rgb_size",
        "images_dict/right/rgb",
        "images_dict/right/rgb_size",
    ]
    return [f"missing dataset: {key}" for key in required if key not in f]


def _check_merge_consistency(f: h5py.File, which: str) -> list[str]:
    errors: list[str] = []
    group = f[which]
    merge = group["merge_pose"][:]
    torso = group["astribot_torso"][:]
    left = group["astribot_arm_left"][:]
    right = group["astribot_arm_right"][:]
    chassis = group["astribot_chassis"][:]
    head = group["astribot_head"][:]
    lg = group["astribot_gripper_left"][:].reshape(-1, 1)
    rg = group["astribot_gripper_right"][:].reshape(-1, 1)
    rebuilt = np.concatenate([chassis, torso, left, lg, right, rg, head], axis=1)
    if not np.allclose(merge, rebuilt, atol=1e-9, rtol=0):
        errors.append(f"{which}: merge_pose != concat(parts)")
    return errors


def _check_gripper_vs_joints(f: h5py.File) -> list[str]:
    errors: list[str] = []
    j_state = f["joints_dict/joints_position_state"][:]
    j_cmd = f["joints_dict/joints_position_command"][:]
    if not np.allclose(f["poses_dict/astribot_gripper_left"][:].reshape(-1), j_state[:, C.J_LEFT_GRIPPER], atol=1e-6):
        errors.append("poses Lg != joints state[14]")
    if not np.allclose(f["poses_dict/astribot_gripper_right"][:].reshape(-1), j_state[:, C.J_RIGHT_GRIPPER], atol=1e-6):
        errors.append("poses Rg != joints state[22]")
    if not np.allclose(
        f["command_poses_dict/astribot_gripper_left"][:].reshape(-1), j_cmd[:, C.J_LEFT_GRIPPER], atol=1e-6
    ):
        errors.append("command Lg != joints command[14]")
    if not np.allclose(
        f["command_poses_dict/astribot_gripper_right"][:].reshape(-1), j_cmd[:, C.J_RIGHT_GRIPPER], atol=1e-6
    ):
        errors.append("command Rg != joints command[22]")
    return errors


def _check_quat_norms(merge: np.ndarray) -> list[str]:
    errors: list[str] = []
    for name, sl in (
        ("chassis", C.MP_CHASSIS),
        ("torso", C.MP_TORSO),
        ("left", C.MP_LEFT),
        ("right", C.MP_RIGHT),
        ("head", C.MP_HEAD),
    ):
        norms = np.linalg.norm(merge[:, sl][:, 3:], axis=1)
        bad = np.where(np.abs(norms - 1.0) > C.QUAT_NORM_TOL)[0]
        if bad.size:
            errors.append(f"{name} quat norm outliers: {bad[:5].tolist()} (count={bad.size})")
    return errors


def _infra_compare_sample(hybrid28: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False}
    try:
        eval_root_value = os.environ.get("ASTRIBOT_EVAL_ROOT")
        if not eval_root_value:
            return {"available": False, "reason": "ASTRIBOT_EVAL_ROOT is not set"}
        eval_root = Path(eval_root_value)
        if str(eval_root) not in sys.path:
            sys.path.insert(0, str(eval_root))
        from tools.util import convert_rotation_format_list  # type: ignore

        segments = [
            hybrid28[0:7].tolist(),
            hybrid28[7:14].tolist(),
            hybrid28[14:15].tolist(),
            hybrid28[15:22].tolist(),
            hybrid28[22:23].tolist(),
            hybrid28[23:25].tolist(),
            hybrid28[25:28].tolist(),
        ]
        infra = np.asarray(
            convert_rotation_format_list(segments, dim_list=None, from_format="quat", to_format="so3"),
            dtype=np.float64,
        )
        ours = pack.quat_hybrid28_to_so3_34(hybrid28)
        result = {
            "available": True,
            "max_abs_diff": float(np.max(np.abs(infra - ours))),
            "ok": bool(np.allclose(infra, ours, atol=1e-9, rtol=0)),
        }
    except Exception as error:
        result = {"available": False, "error": f"{type(error).__name__}: {error}"}
    return result


def validate_episode(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "ok": False,
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    if not path.exists():
        report["errors"].append("file missing")
        return report

    with h5py.File(path, "r") as f:
        report["errors"].extend(_check_fields(f))
        if report["errors"]:
            return report

        report["errors"].extend(_check_merge_consistency(f, "poses_dict"))
        report["errors"].extend(_check_merge_consistency(f, "command_poses_dict"))
        report["errors"].extend(_check_gripper_vs_joints(f))

        merge_cmd = f["command_poses_dict/merge_pose"][:]
        merge_state = f["poses_dict/merge_pose"][:]
        joints_cmd = f["joints_dict/joints_position_command"][:]
        joints_state = f["joints_dict/joints_position_state"][:]
        t = merge_cmd.shape[0]
        report["stats"]["frames"] = int(t)

        report["errors"].extend(_check_quat_norms(merge_cmd))
        report["errors"].extend(_check_quat_norms(merge_state))

        chassis_cmd = merge_cmd[:, C.MP_CHASSIS]
        identity_mask = np.array([pack.is_chassis_identity(row) for row in chassis_cmd], dtype=bool)
        report["stats"]["chassis_identity_fraction"] = float(identity_mask.mean())
        if not bool(identity_mask.all()):
            report["warnings"].append(
                "command chassis pose is not identity on all frames; world→chassis will be applied"
            )

        # Pack a few frames + full episode SO3 checks
        sample_idx = sorted({0, t // 2, t - 1})
        max_orth = 0.0
        max_rt_pos = 0.0
        max_rt_rot = 0.0
        infra_cmp = None
        for idx in sample_idx:
            hybrid = pack.pack_hybrid28_from_merge_and_joints(merge_cmd[idx], joints_cmd[idx])
            ee34 = pack.quat_hybrid28_to_so3_34(hybrid)
            if ee34.shape != (C.EE34_DIM,):
                report["errors"].append(f"frame {idx}: ee34 shape {ee34.shape}")
                continue
            # dim_list rebuild length check
            rebuilt_len = sum(C.EE34_DIM_LIST)
            if rebuilt_len != C.EE34_DIM:
                report["errors"].append("EE34_DIM_LIST does not sum to 34")
            for sl in (C.EE34_TORSO, C.EE34_LEFT, C.EE34_RIGHT):
                max_orth = max(max_orth, pack.so3_block_orthonormality_error(ee34[sl]))
            back = pack.so3_34_to_quat_hybrid28(ee34)
            # compare EE pose7 parts (indices in hybrid)
            for a, b in ((0, 7), (7, 14), (15, 22)):
                max_rt_pos = max(max_rt_pos, float(np.linalg.norm(hybrid[a : a + 3] - back[a : a + 3])))
                # quat up to sign
                q1, q2 = hybrid[a + 3 : b], back[a + 3 : b]
                if float(np.dot(q1, q2)) < 0:
                    q2 = -q2
                max_rt_rot = max(max_rt_rot, float(np.linalg.norm(q1 - q2)))
            if infra_cmp is None:
                infra_cmp = _infra_compare_sample(hybrid)

        report["stats"]["max_so3_orth_error"] = max_orth
        report["stats"]["max_roundtrip_pos"] = max_rt_pos
        report["stats"]["max_roundtrip_rot"] = max_rt_rot
        report["stats"]["infra_compare"] = infra_cmp

        if max_orth > C.SO3_ORTH_TOL:
            report["errors"].append(f"SO3 orthonormality error too large: {max_orth}")
        if max_rt_pos > C.ROUNDTRIP_POS_TOL:
            report["errors"].append(f"roundtrip position error too large: {max_rt_pos}")
        if max_rt_rot > C.ROUNDTRIP_ROT_TOL:
            report["errors"].append(f"roundtrip rotation error too large: {max_rt_rot}")
        if infra_cmp and infra_cmp.get("available") and not infra_cmp.get("ok", False):
            report["errors"].append(f"Infra SO3 mismatch: {infra_cmp}")

        # Full pack sanity
        actions = pack.pack_episode_ee34(merge_cmd, joints_cmd)
        states = pack.pack_episode_ee34(merge_state, joints_state)
        if actions.shape != (t, C.EE34_DIM) or states.shape != (t, C.EE34_DIM):
            report["errors"].append("episode pack shape mismatch")
        if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(states)):
            report["errors"].append("packed ee34 contains NaN/Inf")

        rg = actions[:, C.EE34_RIGHT_GRIPPER]
        if float(np.max(np.abs(rg))) < 1e-8:
            report["warnings"].append("right gripper action is constantly ~0")
        report["stats"]["action_right_gripper_abs_max"] = float(np.max(np.abs(rg)))
        report["stats"]["action_left_gripper_range"] = [
            float(np.min(actions[:, C.EE34_LEFT_GRIPPER])),
            float(np.max(actions[:, C.EE34_LEFT_GRIPPER])),
        ]
        report["stats"]["sample_action0"] = actions[0].tolist()

        # camera sizes
        for cam in C.SOURCE_CAMERAS:
            sizes = f[f"images_dict/{cam}/rgb_size"][:].astype(np.int64)
            if sizes.shape[0] != t:
                report["errors"].append(f"{cam}: rgb_size length {sizes.shape[0]} != frames {t}")
            if int(sizes.sum()) != int(f[f"images_dict/{cam}/rgb"].shape[0]):
                report["errors"].append(f"{cam}: rgb blob size != sum(rgb_size)")

    report["ok"] = len(report["errors"]) == 0
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=C.SOURCE_ROOT)
    parser.add_argument("--episode-ids", type=int, nargs="*", default=None)
    args = parser.parse_args()

    all_ids = _list_episode_ids(args.source_root)
    sample_ids = args.episode_ids or _pick_sample_ids(all_ids)
    reports = [validate_episode(_episode_path(args.source_root, eid)) for eid in sample_ids]
    ok = all(item["ok"] for item in reports)
    summary = {
        "contract_id": C.CONTRACT_ID,
        "contract_version": C.CONTRACT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_root": str(args.source_root),
        "sample_episode_ids": sample_ids,
        "ok": ok,
        "episodes": reports,
        "notes": {
            "mapping": "merge_pose torso/left/right + joints gripper/head/chassis → SO3 34D",
            "chassis_frame": "tomato dump expected chassis identity; pack still supports world→chassis",
        },
    }

    C.REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = C.REPORT_ROOT / "ee34_contract_validation.json"
    md_path = C.REPORT_ROOT / "ee34_contract_validation.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    lines = [
        f"# EE34 contract validation ({C.CONTRACT_ID})",
        "",
        f"- ok: **{ok}**",
        f"- source: `{args.source_root}`",
        f"- samples: {sample_ids}",
        "",
    ]
    for item in reports:
        lines.append(f"## episode `{Path(item['path']).name}`")
        lines.append(f"- ok: {item['ok']}")
        lines.append(f"- errors: {item['errors'] or '[]'}")
        lines.append(f"- warnings: {item['warnings'] or '[]'}")
        lines.append(f"- stats: `{json.dumps(item['stats'], ensure_ascii=False)}`")
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n")

    print(json.dumps({"ok": ok, "json": str(json_path), "md": str(md_path)}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
