"""Pack HDF5 pose7+joints25 into absolute EE SO3 34D (Infra-aligned)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from . import contract as C  # noqa: N812


def _as_pose7(value: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (7,):
        raise ValueError(f"{name} must have shape (7,), got {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} contains NaN or Inf")
    return pose


def quat_norm(pose7: np.ndarray) -> float:
    return float(np.linalg.norm(_as_pose7(pose7, "pose7")[3:]))


def is_chassis_identity(
    chassis7: np.ndarray,
    *,
    xyz_tol: float = C.CHASSIS_IDENTITY_XYZ_TOL,
    quat_tol: float = C.CHASSIS_IDENTITY_QUAT_TOL,
) -> bool:
    chassis = _as_pose7(chassis7, "chassis7")
    xyz_ok = float(np.linalg.norm(chassis[:3])) <= xyz_tol
    quat_ok = float(np.linalg.norm(chassis[3:] - np.array([0.0, 0.0, 0.0, 1.0]))) <= quat_tol
    return xyz_ok and quat_ok


def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    pose = _as_pose7(pose7, "pose7")
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    mat[:3, 3] = pose[:3]
    return mat


def matrix_to_pose7(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"expected 4x4 matrix, got {mat.shape}")
    quat = Rotation.from_matrix(mat[:3, :3]).as_quat()  # xyzw
    return np.concatenate([mat[:3, 3], quat]).astype(np.float64, copy=False)


def world_to_chassis(chassis7: np.ndarray, link7: np.ndarray) -> np.ndarray:
    """Express link pose in chassis frame: T_c_l = inv(T_w_c) @ T_w_l.

    When chassis is identity (common for this tomato HDF5 dump), returns link7.
    """
    chassis = _as_pose7(chassis7, "chassis7")
    link = _as_pose7(link7, "link7")
    if is_chassis_identity(chassis):
        return link.copy()
    t_w_c = pose7_to_matrix(chassis)
    t_w_l = pose7_to_matrix(link)
    t_c_l = np.linalg.inv(t_w_c) @ t_w_l
    return matrix_to_pose7(t_c_l)


def quat7_to_so39(pose7: np.ndarray) -> np.ndarray:
    """xyz + first two rotation-matrix rows (Infra quat→so3)."""
    pose = _as_pose7(pose7, "pose7")
    rot = Rotation.from_quat(pose[3:]).as_matrix()
    return np.concatenate([pose[:3], rot[0], rot[1]]).astype(np.float64, copy=False)


def so39_to_quat7(so39: np.ndarray) -> np.ndarray:
    """Inverse of quat7_to_so39 (matches astribot_vla_execute.so3_to_quat)."""
    so3 = np.asarray(so39, dtype=np.float64).reshape(-1)
    if so3.shape != (9,):
        raise ValueError(f"so39 must have shape (9,), got {so3.shape}")
    x, y, z, r11, r12, r13, r21, r22, r23 = so3
    r3 = np.cross([r11, r12, r13], [r21, r22, r23])
    mat = np.array([[r11, r12, r13], [r21, r22, r23], r3], dtype=np.float64)
    # Re-orthonormalize lightly for numerical stability
    u, _, vt = np.linalg.svd(mat)
    mat = u @ vt
    if np.linalg.det(mat) < 0:
        u[:, -1] *= -1
        mat = u @ vt
    quat = Rotation.from_matrix(mat).as_quat()
    return np.array([x, y, z, quat[0], quat[1], quat[2], quat[3]], dtype=np.float64)


def so3_block_orthonormality_error(so39: np.ndarray) -> float:
    so3 = np.asarray(so39, dtype=np.float64).reshape(-1)
    r1 = so3[3:6]
    r2 = so3[6:9]
    n1 = abs(float(np.linalg.norm(r1)) - 1.0)
    n2 = abs(float(np.linalg.norm(r2)) - 1.0)
    dot = abs(float(np.dot(r1, r2)))
    return max(n1, n2, dot)


def pack_hybrid28_from_merge_and_joints(merge_pose37: np.ndarray, joints25: np.ndarray) -> np.ndarray:
    """Build hybrid 28D quat vector used before SO3 conversion.

    EE links from merge_pose (after world→chassis); grippers/head/chassis from joints.
    """
    mp = np.asarray(merge_pose37, dtype=np.float64).reshape(-1)
    joints = np.asarray(joints25, dtype=np.float64).reshape(-1)
    if mp.shape != (C.MERGE_POSE_DIM,):
        raise ValueError(f"merge_pose must be ({C.MERGE_POSE_DIM},), got {mp.shape}")
    if joints.shape != (C.JOINT_DIM,):
        raise ValueError(f"joints must be ({C.JOINT_DIM},), got {joints.shape}")
    if not np.all(np.isfinite(mp)) or not np.all(np.isfinite(joints)):
        raise ValueError("merge_pose/joints contain NaN or Inf")

    chassis7 = mp[C.MP_CHASSIS]
    torso7 = world_to_chassis(chassis7, mp[C.MP_TORSO])
    left7 = world_to_chassis(chassis7, mp[C.MP_LEFT])
    right7 = world_to_chassis(chassis7, mp[C.MP_RIGHT])
    left_g = joints[C.J_LEFT_GRIPPER : C.J_LEFT_GRIPPER + 1]
    right_g = joints[C.J_RIGHT_GRIPPER : C.J_RIGHT_GRIPPER + 1]
    head2 = joints[C.J_HEAD]
    chassis3 = joints[C.J_CHASSIS]
    hybrid = np.concatenate([torso7, left7, left_g, right7, right_g, head2, chassis3])
    if hybrid.shape != (C.HYBRID_QUAT_DIM,):
        raise ValueError(f"hybrid28 shape drift: {hybrid.shape}")
    return hybrid.astype(np.float64, copy=False)


def quat_hybrid28_to_so3_34(hybrid28: np.ndarray) -> np.ndarray:
    h = np.asarray(hybrid28, dtype=np.float64).reshape(-1)
    if h.shape != (C.HYBRID_QUAT_DIM,):
        raise ValueError(f"hybrid28 must be ({C.HYBRID_QUAT_DIM},), got {h.shape}")
    torso9 = quat7_to_so39(h[0:7])
    left9 = quat7_to_so39(h[7:14])
    left_g = h[14:15]
    right9 = quat7_to_so39(h[15:22])
    right_g = h[22:23]
    head2 = h[23:25]
    chassis3 = h[25:28]
    out = np.concatenate([torso9, left9, left_g, right9, right_g, head2, chassis3])
    if out.shape != (C.EE34_DIM,):
        raise ValueError(f"ee34 shape drift: {out.shape}")
    return out.astype(np.float64, copy=False)


def so3_34_to_quat_hybrid28(ee34: np.ndarray) -> np.ndarray:
    v = np.asarray(ee34, dtype=np.float64).reshape(-1)
    if v.shape != (C.EE34_DIM,):
        raise ValueError(f"ee34 must be ({C.EE34_DIM},), got {v.shape}")
    return np.concatenate(
        [
            so39_to_quat7(v[C.EE34_TORSO]),
            so39_to_quat7(v[C.EE34_LEFT]),
            v[C.EE34_LEFT_GRIPPER : C.EE34_LEFT_GRIPPER + 1],
            so39_to_quat7(v[C.EE34_RIGHT]),
            v[C.EE34_RIGHT_GRIPPER : C.EE34_RIGHT_GRIPPER + 1],
            v[C.EE34_HEAD],
            v[C.EE34_CHASSIS],
        ]
    ).astype(np.float64, copy=False)


def pack_ee34_from_merge_and_joints(merge_pose37: np.ndarray, joints25: np.ndarray) -> np.ndarray:
    hybrid = pack_hybrid28_from_merge_and_joints(merge_pose37, joints25)
    return quat_hybrid28_to_so3_34(hybrid).astype(np.float32, copy=False)


def pack_episode_ee34(
    merge_poses: np.ndarray,
    joints: np.ndarray,
) -> np.ndarray:
    """Vectorized episode packing: (T,37)+(T,25) -> (T,34) float32."""
    merge_poses = np.asarray(merge_poses, dtype=np.float64)
    joints = np.asarray(joints, dtype=np.float64)
    if merge_poses.ndim != 2 or merge_poses.shape[1] != C.MERGE_POSE_DIM:
        raise ValueError(f"merge_poses must be (T,{C.MERGE_POSE_DIM}), got {merge_poses.shape}")
    if joints.ndim != 2 or joints.shape[1] != C.JOINT_DIM:
        raise ValueError(f"joints must be (T,{C.JOINT_DIM}), got {joints.shape}")
    if merge_poses.shape[0] != joints.shape[0]:
        raise ValueError(f"T mismatch: merge {merge_poses.shape[0]} vs joints {joints.shape[0]}")
    out = np.empty((merge_poses.shape[0], C.EE34_DIM), dtype=np.float32)
    for index in range(merge_poses.shape[0]):
        out[index] = pack_ee34_from_merge_and_joints(merge_poses[index], joints[index])
    return out
