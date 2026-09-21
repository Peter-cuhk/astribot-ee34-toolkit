"""Low-poly solid robot geometry for the offline renderers.

The Astribot visual STLs are ~1.2M triangles across 24 links, more than a
645x988 panel can show and too slow to redraw 500 times. Each link is split into
its connected shells and every shell is replaced by its convex hull, which keeps
each part's silhouette -- torso column, shoulders, arm segments, wrists, the head
dome and its camera bar -- at ~122k triangles. Concave interior detail is
deliberately given up; shells are *not* dropped by size, because the small ones
are what fill the gaps between the big ones.

The hulls of one link are merged into a single vertex/face array at build time so
a frame costs one transform per link rather than one per shell.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from . import kinematics as kin

# Lambert light direction in world coordinates, and the ambient floor that keeps
# faces pointing away from it readable rather than black. A white body needs a
# wide shading range to read at all against the near-white panel, so the floor
# sits lower than a mid-tone body would want.
LIGHT_DIRECTION = np.asarray([0.4, -0.7, 0.6], dtype=np.float64)
AMBIENT = 0.16
DEFAULT_COLOR = "#FFFFFF"

# The parallel gripper is a four-bar per finger with no <mimic> in the URDF, so
# its six joints are driven from one angle. The signs were read off the model:
# L1/L2 lead, the distal L11 counter-rotates to keep the tip parallel, and the R
# finger mirrors all three. THETA_CLOSED is where the fingers meet and
# THETA_OPEN is the URDF zero pose.
#
# The recorded 0-100 stream is *closure*, not opening: in the wrist camera the
# fingers are spread at 0, clamped on the tomato at 31 and shut at 97. There is
# no published angle calibration, so the two endpoints are a visual
# approximation, not a kinematic claim -- but the direction is measured.
GRIPPER_JOINT_SIGNS = {"L1": 1.0, "L2": 1.0, "L11": -1.0, "R1": -1.0, "R2": -1.0, "R11": 1.0}
THETA_CLOSED = -1.0
THETA_OPEN = 0.0
MOUNT_SIDES = ("left", "right")


@dataclass(frozen=True)
class LinkGeometry:
    """Every convex shell of one link, merged, in that link's frame."""

    vertices: np.ndarray
    faces: np.ndarray


class GripperMesh:
    """Gripper links lifted from a second URDF and hung off each arm's last link.

    The SDK's deployment URDF has no gripper. The published
    ``astribot_whole_body_maniskill.urdf`` does, but its torso and head use
    different meshes that hull into visible junk, so only the gripper subtree is
    taken from it and mounted on the body model's own arm link. Poses are
    returned relative to that mount, so the donor URDF's arm joints stay at zero
    and never have to agree with the body model's.
    """

    MOUNTS: ClassVar[dict[str, str]] = {
        "left": "astribot_arm_left_link_7",
        "right": "astribot_arm_right_link_7",
    }

    def __init__(self, urdf_path: Path) -> None:
        self.urdf = kin.load_visual_urdf(urdf_path)
        actuated = set(self.urdf.actuated_joint_names)
        self.joints = {
            side: {finger: f"astribot_gripper_{side}_joint_{finger}" for finger in GRIPPER_JOINT_SIGNS}
            for side in MOUNT_SIDES
        }
        missing = [
            name
            for joints in self.joints.values()
            for name in joints.values()
            if name not in actuated
        ]
        if missing:
            raise kin.KinematicsConfigError(f"gripper URDF {urdf_path} is missing joints: {missing}")
        parents = self.urdf.scene.graph.transforms.parents
        self.links: dict[str, list[tuple[str, LinkGeometry]]] = {side: [] for side in MOUNT_SIDES}
        for node in self.urdf.scene.graph.nodes_geometry:
            parent = str(parents.get(node, ""))
            for side in MOUNT_SIDES:
                if parent.startswith(f"astribot_gripper_{side}_"):
                    geometry = _hull_geometry(self.urdf.scene.geometry[self.urdf.scene.graph[node][1]])
                    if geometry is not None:
                        self.links[side].append((node, geometry))
        for side in MOUNT_SIDES:
            if not self.links[side]:
                raise kin.KinematicsConfigError(f"gripper URDF {urdf_path} has no {side} gripper geometry")
        self.face_count = sum(len(link.faces) for side in MOUNT_SIDES for _, link in self.links[side])
        self._zero = {name: 0.0 for name in self.urdf.actuated_joint_names}

    def posed(self, side: str, percent: float) -> list[tuple[LinkGeometry, np.ndarray]]:
        """Each gripper link's geometry and its pose relative to the arm mount."""
        closure = float(np.clip(percent, 0.0, 100.0)) / 100.0
        theta = THETA_OPEN + (THETA_CLOSED - THETA_OPEN) * closure
        configuration = dict(self._zero)
        for finger, sign in GRIPPER_JOINT_SIGNS.items():
            configuration[self.joints[side][finger]] = sign * theta
        self.urdf.update_cfg(configuration)
        mount = np.asarray(self.urdf.scene.graph.get(self.MOUNTS[side])[0], dtype=np.float64)
        inverse = np.linalg.inv(mount)
        return [
            (link, inverse @ np.asarray(self.urdf.scene.graph.get(node)[0], dtype=np.float64))
            for node, link in self.links[side]
        ]


class RobotMesh:
    """Per-link convex shells plus the per-frame world-space triangle build."""

    def __init__(
        self,
        urdf_path: Path,
        color: str = DEFAULT_COLOR,
        gripper_urdf_path: Path | None = None,
    ) -> None:
        self.urdf = kin.load_visual_urdf(urdf_path)
        actuated = set(self.urdf.actuated_joint_names)
        missing = [name for name in kin.EXPECTED_JOINT_NAMES if name not in actuated]
        if missing:
            raise kin.KinematicsConfigError(
                f"visual URDF {urdf_path} is missing actuated joints: {missing}"
            )
        self.gripper = None if gripper_urdf_path is None else GripperMesh(gripper_urdf_path)
        self.links: dict[str, LinkGeometry] = {}
        for node in self.urdf.scene.graph.nodes_geometry:
            geometry = _hull_geometry(self.urdf.scene.geometry[self.urdf.scene.graph[node][1]])
            if geometry is not None:
                self.links[node] = geometry
        self.face_count = sum(len(link.faces) for link in self.links.values())
        if self.face_count == 0:
            raise kin.KinematicsConfigError(f"no renderable visual geometry in {urdf_path}")
        self.light = LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)
        self.color = _parse_color(color)

    def configuration(self, q20: np.ndarray) -> dict[str, float]:
        """Joint-name -> angle for this visual URDF.

        Addressing joints by name rather than by index keeps the visual model
        independent of the FK model's joint vector layout.
        """
        return dict(zip(kin.EXPECTED_JOINT_NAMES, np.asarray(q20, dtype=np.float64)))

    def triangles(
        self, q20: np.ndarray, base: np.ndarray, grippers: tuple[float, float] | None = None
    ) -> np.ndarray:
        """World-space (n, 3, 3) triangles for one configuration."""
        self.urdf.update_cfg(self.configuration(q20))
        batches: list[np.ndarray] = []
        for node, link in self.links.items():
            transform = base @ np.asarray(self.urdf.scene.graph.get(node)[0], dtype=np.float64)
            world = link.vertices @ transform[:3, :3].T + transform[:3, 3]
            batches.append(world[link.faces])
        if self.gripper is not None:
            percents = (0.0, 0.0) if grippers is None else grippers
            for side, percent in zip(("left", "right"), percents):
                mount = base @ np.asarray(
                    self.urdf.scene.graph.get(GripperMesh.MOUNTS[side])[0], dtype=np.float64
                )
                for link, relative in self.gripper.posed(side, percent):
                    transform = mount @ relative
                    world = link.vertices @ transform[:3, :3].T + transform[:3, 3]
                    batches.append(world[link.faces])
        return np.concatenate(batches, axis=0)

    def bounds(
        self,
        configurations: list[np.ndarray],
        bases: list[np.ndarray],
        grippers: list[tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """World-space AABB of the body over a whole episode.

        Framing the view on the joint-centre skeleton cuts the top of the head
        off: the head shell reaches ~0.13 m above the highest joint centre. This
        walks the episode transforming each link's own AABB corners -- 8 points
        per link rather than every vertex -- which is a slight over-estimate of
        the true hull but cheap enough to run per frame.
        """
        corners = {node: _aabb_corners(link.vertices) for node, link in self.links.items()}
        low = np.full(3, np.inf)
        high = np.full(3, -np.inf)
        if grippers is None:
            grippers = [(0.0, 0.0)] * len(configurations)
        for q20, base, gripper in zip(configurations, bases, grippers):
            self.urdf.update_cfg(self.configuration(q20))
            for node, box in corners.items():
                transform = base @ np.asarray(self.urdf.scene.graph.get(node)[0], dtype=np.float64)
                world = box @ transform[:3, :3].T + transform[:3, 3]
                low = np.minimum(low, world.min(axis=0))
                high = np.maximum(high, world.max(axis=0))
            if self.gripper is None:
                continue
            for side, percent in zip(("left", "right"), gripper):
                mount = base @ np.asarray(
                    self.urdf.scene.graph.get(GripperMesh.MOUNTS[side])[0], dtype=np.float64
                )
                for link, relative in self.gripper.posed(side, percent):
                    transform = mount @ relative
                    world = _aabb_corners(link.vertices) @ transform[:3, :3].T + transform[:3, 3]
                    low = np.minimum(low, world.min(axis=0))
                    high = np.maximum(high, world.max(axis=0))
        return low, high

    def shade(self, triangles: np.ndarray) -> np.ndarray:
        """Flat Lambert shading, one RGB per triangle."""
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        lit = AMBIENT + (1.0 - AMBIENT) * np.abs(normals @ self.light)
        return np.clip(lit[:, None] * self.color[None, :], 0.0, 1.0)


def _aabb_corners(vertices: np.ndarray) -> np.ndarray:
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    return np.asarray(
        [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])],
        dtype=np.float64,
    )


def _parse_color(value: str) -> np.ndarray:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"robot color must be a 6-digit hex string, got {value!r}")
    try:
        channels = [int(text[index : index + 2], 16) / 255.0 for index in (0, 2, 4)]
    except ValueError as error:
        raise ValueError(f"robot color must be a 6-digit hex string, got {value!r}") from error
    return np.asarray(channels, dtype=np.float64)


def _hull_geometry(geometry) -> LinkGeometry | None:
    vertex_blocks: list[np.ndarray] = []
    face_blocks: list[np.ndarray] = []
    offset = 0
    for shell in list(geometry.split(only_watertight=False)) or [geometry]:
        if len(shell.vertices) < 4:
            continue
        try:
            hull = shell.convex_hull
        except Exception:
            continue
        vertices = np.asarray(hull.vertices, dtype=np.float64)
        vertex_blocks.append(vertices)
        face_blocks.append(np.asarray(hull.faces, dtype=np.int64) + offset)
        offset += len(vertices)
    if not vertex_blocks:
        return None
    return LinkGeometry(
        vertices=np.concatenate(vertex_blocks, axis=0),
        faces=np.concatenate(face_blocks, axis=0),
    )
