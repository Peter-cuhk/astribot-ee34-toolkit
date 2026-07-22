"""Offline Astribot S1 kinematics helpers for EE34 auditing and visualization."""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from . import contract as C  # noqa: N812
from . import pack

SDK_MODEL_ROOT = Path(os.environ.get("ASTRIBOT_S1_CONFIG_ROOT", "third_party/astribot_s1"))
DEFAULT_URDF_PATH = SDK_MODEL_ROOT / "model/astribot_whole_body_with_head.urdf"
DEFAULT_TORSO_CONFIG_PATH = SDK_MODEL_ROOT / "astribot_torso.yaml"

LINK_LABELS = ("torso", "left", "right")
END_EFFECTOR_FRAMES = {
    "torso": "astribot_torso_end_effector",
    "left": "astribot_arm_left_end_effector",
    "right": "astribot_arm_right_end_effector",
}
EXPECTED_JOINT_NAMES = (
    *(f"astribot_torso_joint_{index}" for index in range(1, 5)),
    *(f"astribot_head_joint_{index}" for index in range(1, 3)),
    *(f"astribot_arm_left_joint_{index}" for index in range(1, 8)),
    *(f"astribot_arm_right_joint_{index}" for index in range(1, 8)),
)

# q20 is torso4 + head2 + left7 + right7. IK excludes head.
IK_Q20_INDICES = np.asarray([*range(4), *range(6, 13), *range(13, 20)], dtype=np.int64)


class KinematicsConfigError(ValueError):
    """Raised when robot model inputs do not satisfy the frozen EE34 contract."""


@dataclass(frozen=True)
class PoseError:
    position_m: float
    rotation_rad: float

    @property
    def position_mm(self) -> float:
        return self.position_m * 1000.0

    @property
    def rotation_deg(self) -> float:
        return float(np.rad2deg(self.rotation_rad))


@dataclass(frozen=True)
class IkResult:
    q20: np.ndarray
    valid: bool
    success: bool
    cost: float
    nfev: int
    max_joint_delta_rad: float
    errors: dict[str, PoseError]
    message: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_vector(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def pose_xyz_rpy_to_matrix(pose6: np.ndarray) -> np.ndarray:
    """Convert SDK [x,y,z,roll,pitch,yaw] into a homogeneous transform."""
    pose = _require_vector(pose6, (6,), "pose6")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def load_weld_transform(config_path: Path = DEFAULT_TORSO_CONFIG_PATH) -> tuple[np.ndarray, np.ndarray]:
    path = Path(config_path)
    if not path.is_file():
        raise KinematicsConfigError(f"missing torso config: {path}")
    payload = yaml.safe_load(path.read_text())
    try:
        pose = np.asarray(payload["transform"]["weld_to_base_pose"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise KinematicsConfigError(f"invalid transform.weld_to_base_pose in {path}") from error
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise KinematicsConfigError(f"weld_to_base_pose must be finite 6D, got {pose}")
    return pose, pose_xyz_rpy_to_matrix(pose)


def joints25_to_urdf20(joints25: np.ndarray) -> np.ndarray:
    joints = _require_vector(joints25, (C.JOINT_DIM,), "joints25")
    return np.concatenate(
        [
            joints[C.J_TORSO],
            joints[C.J_HEAD],
            joints[C.J_LEFT_ARM],
            joints[C.J_RIGHT_ARM],
        ]
    )


def ee34_to_target_matrices(ee34: np.ndarray) -> dict[str, np.ndarray]:
    value = _require_vector(ee34, (C.EE34_DIM,), "ee34")
    return {
        "torso": pack.pose7_to_matrix(pack.so39_to_quat7(value[C.EE34_TORSO])),
        "left": pack.pose7_to_matrix(pack.so39_to_quat7(value[C.EE34_LEFT])),
        "right": pack.pose7_to_matrix(pack.so39_to_quat7(value[C.EE34_RIGHT])),
    }


def chassis_xyyaw_to_matrix(chassis3: np.ndarray) -> np.ndarray:
    chassis = _require_vector(chassis3, (3,), "chassis3")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("z", chassis[2]).as_matrix()
    transform[:2, 3] = chassis[:2]
    return transform


def se3_error(actual: np.ndarray, target: np.ndarray) -> PoseError:
    actual = np.asarray(actual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if actual.shape != (4, 4) or target.shape != (4, 4):
        raise ValueError(f"SE3 inputs must be 4x4, got {actual.shape} and {target.shape}")
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    relative = actual[:3, :3].T @ target[:3, :3]
    rotation = float(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec()))
    return PoseError(position_m=position, rotation_rad=rotation)


def matrix_to_wxyz_position(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {transform.shape}")
    xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    wxyz = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    return wxyz, transform[:3, 3].copy()


class AstribotKinematics:
    """Validated, local-only FK and numerical IK backed by yourdfpy."""

    def __init__(
        self,
        urdf_path: Path = DEFAULT_URDF_PATH,
        torso_config_path: Path = DEFAULT_TORSO_CONFIG_PATH,
    ) -> None:
        try:
            import yourdfpy
        except ImportError as error:  # pragma: no cover - dependency error path
            raise KinematicsConfigError(
                "yourdfpy is required; install with `uv sync --extra visualization`"
            ) from error

        self.urdf_path = Path(urdf_path)
        self.torso_config_path = Path(torso_config_path)
        if not self.urdf_path.is_file():
            raise KinematicsConfigError(f"missing URDF: {self.urdf_path}")
        self._validate_mesh_files()
        self.weld_pose6, self.t_chassis_urdf_root = load_weld_transform(self.torso_config_path)
        self.urdf = yourdfpy.URDF.load(
            self.urdf_path,
            load_meshes=False,
            build_scene_graph=True,
        )
        actual_names = tuple(self.urdf.actuated_joint_names)
        if actual_names != EXPECTED_JOINT_NAMES:
            raise KinematicsConfigError(
                "URDF actuated joint order drifted:\n"
                f"expected={EXPECTED_JOINT_NAMES}\nactual={actual_names}"
            )
        if self.urdf.scene is None:
            raise KinematicsConfigError("URDF scene graph was not built")
        for frame in END_EFFECTOR_FRAMES.values():
            try:
                self.urdf.scene.graph.get(frame)
            except (KeyError, ValueError) as error:
                raise KinematicsConfigError(f"missing end-effector frame: {frame}") from error

        lower: list[float] = []
        upper: list[float] = []
        for joint in self.urdf.actuated_joints:
            if joint.limit is None or joint.limit.lower is None or joint.limit.upper is None:
                raise KinematicsConfigError(f"joint has no finite position limit: {joint.name}")
            lower.append(float(joint.limit.lower))
            upper.append(float(joint.limit.upper))
        self.lower20 = np.asarray(lower, dtype=np.float64)
        self.upper20 = np.asarray(upper, dtype=np.float64)
        if not np.all(self.lower20 < self.upper20):
            raise KinematicsConfigError("invalid URDF position limits")

    def _validate_mesh_files(self) -> None:
        try:
            root = ET.parse(self.urdf_path).getroot()
        except ET.ParseError as error:
            raise KinematicsConfigError(f"invalid URDF XML: {self.urdf_path}") from error
        missing: list[str] = []
        for element in root.findall(".//mesh"):
            filename = element.attrib.get("filename")
            if not filename:
                raise KinematicsConfigError("URDF mesh element is missing filename")
            if filename.startswith("package://"):
                missing.append(filename)
                continue
            candidates = (self.urdf_path.parent / filename, self.urdf_path.parent / "meshes" / filename)
            if not any(candidate.is_file() for candidate in candidates):
                missing.append(filename)
        if missing:
            raise KinematicsConfigError(f"URDF mesh files are missing: {sorted(set(missing))}")

    @property
    def joint_names(self) -> tuple[str, ...]:
        return EXPECTED_JOINT_NAMES

    @property
    def ik_lower(self) -> np.ndarray:
        return self.lower20[IK_Q20_INDICES]

    @property
    def ik_upper(self) -> np.ndarray:
        return self.upper20[IK_Q20_INDICES]

    def fk(self, q20: np.ndarray, *, apply_weld: bool = True) -> dict[str, np.ndarray]:
        q = _require_vector(q20, (20,), "q20")
        self.urdf.update_cfg(q)
        prefix = self.t_chassis_urdf_root if apply_weld else np.eye(4, dtype=np.float64)
        return {
            label: prefix @ np.asarray(self.urdf.scene.graph.get(frame)[0], dtype=np.float64)
            for label, frame in END_EFFECTOR_FRAMES.items()
        }

    def skeleton_segments(self, q20: np.ndarray) -> np.ndarray:
        """Return parent-to-child link segments in the URDF-root frame."""
        q = _require_vector(q20, (20,), "q20")
        self.urdf.update_cfg(q)
        segments = []
        for joint in self.urdf.joint_map.values():
            parent = np.asarray(self.urdf.scene.graph.get(joint.parent)[0], dtype=np.float64)
            child = np.asarray(self.urdf.scene.graph.get(joint.child)[0], dtype=np.float64)
            segments.append((parent[:3, 3], child[:3, 3]))
        return np.asarray(segments, dtype=np.float32)

    @staticmethod
    def q20_to_ik18(q20: np.ndarray) -> np.ndarray:
        q = _require_vector(q20, (20,), "q20")
        return q[IK_Q20_INDICES].copy()

    @staticmethod
    def ik18_to_q20(q18: np.ndarray, head2: np.ndarray) -> np.ndarray:
        q = _require_vector(q18, (18,), "q18")
        head = _require_vector(head2, (2,), "head2")
        out = np.empty((20,), dtype=np.float64)
        out[:4] = q[:4]
        out[4:6] = head
        out[6:13] = q[4:11]
        out[13:20] = q[11:18]
        return out

    def _ik_residual(
        self,
        q18: np.ndarray,
        *,
        head2: np.ndarray,
        targets: Mapping[str, np.ndarray],
        seed18: np.ndarray,
    ) -> np.ndarray:
        q20 = self.ik18_to_q20(q18, head2)
        actual = self.fk(q20)
        residuals: list[np.ndarray] = []
        for label in LINK_LABELS:
            current = actual[label]
            target = np.asarray(targets[label], dtype=np.float64)
            residuals.append((current[:3, 3] - target[:3, 3]) / 0.005)
            relative = current[:3, :3].T @ target[:3, :3]
            residuals.append(Rotation.from_matrix(relative).as_rotvec() / np.deg2rad(1.0))
        joint_range = np.maximum(self.ik_upper - self.ik_lower, 1e-6)
        residuals.append(np.sqrt(1e-3) * (q18 - seed18) / joint_range)
        return np.concatenate(residuals)

    def solve_ik(
        self,
        targets: Mapping[str, np.ndarray],
        seed_q20: np.ndarray,
        *,
        head2: np.ndarray,
        alternate_q20: np.ndarray | None = None,
        max_nfev: int = 100,
    ) -> IkResult:
        if tuple(targets) != LINK_LABELS:
            raise ValueError(f"targets must be ordered {LINK_LABELS}, got {tuple(targets)}")
        seed20 = _require_vector(seed_q20, (20,), "seed_q20")
        head = _require_vector(head2, (2,), "head2")
        seed18 = np.clip(self.q20_to_ik18(seed20), self.ik_lower, self.ik_upper)
        starts = [seed18]
        if alternate_q20 is not None:
            alternate = np.clip(
                self.q20_to_ik18(_require_vector(alternate_q20, (20,), "alternate_q20")),
                self.ik_lower,
                self.ik_upper,
            )
            if not np.allclose(alternate, seed18, atol=1e-9, rtol=0.0):
                starts.append(alternate)

        results: list[IkResult] = []
        for start in starts:
            solution = least_squares(
                self._ik_residual,
                start,
                bounds=(self.ik_lower, self.ik_upper),
                kwargs={"head2": head, "targets": targets, "seed18": seed18},
                max_nfev=max_nfev,
                xtol=1e-9,
                ftol=1e-9,
                gtol=1e-9,
            )
            q20 = self.ik18_to_q20(solution.x, head)
            actual = self.fk(q20)
            errors = {label: se3_error(actual[label], targets[label]) for label in LINK_LABELS}
            valid = all(
                error.position_m <= 0.005 and error.rotation_rad <= np.deg2rad(1.0)
                for error in errors.values()
            )
            results.append(
                IkResult(
                    q20=q20,
                    valid=valid,
                    success=bool(solution.success),
                    cost=float(solution.cost),
                    nfev=int(solution.nfev),
                    max_joint_delta_rad=float(np.max(np.abs(solution.x - seed18))),
                    errors=errors,
                    message=str(solution.message),
                )
            )
        return min(
            results,
            key=lambda result: (
                not result.valid,
                result.max_joint_delta_rad,
                result.cost,
            ),
        )
