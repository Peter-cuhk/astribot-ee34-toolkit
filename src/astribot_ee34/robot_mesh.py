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

import numpy as np

from . import kinematics as kin

# Lambert light direction in world coordinates, and the ambient floor that keeps
# faces pointing away from it readable rather than black.
LIGHT_DIRECTION = np.asarray([0.4, -0.7, 0.6], dtype=np.float64)
AMBIENT = 0.34
DEFAULT_COLOR = "#B0A99F"


@dataclass(frozen=True)
class LinkGeometry:
    """Every convex shell of one link, merged, in that link's frame."""

    vertices: np.ndarray
    faces: np.ndarray


class RobotMesh:
    """Per-link convex shells plus the per-frame world-space triangle build."""

    def __init__(self, urdf_path: Path, color: str = DEFAULT_COLOR) -> None:
        self.urdf = kin.load_visual_urdf(urdf_path)
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

    def triangles(self, q20: np.ndarray, base: np.ndarray) -> np.ndarray:
        """World-space (n, 3, 3) triangles for one configuration."""
        self.urdf.update_cfg(np.asarray(q20, dtype=np.float64))
        batches: list[np.ndarray] = []
        for node, link in self.links.items():
            transform = base @ np.asarray(self.urdf.scene.graph.get(node)[0], dtype=np.float64)
            world = link.vertices @ transform[:3, :3].T + transform[:3, 3]
            batches.append(world[link.faces])
        return np.concatenate(batches, axis=0)

    def bounds(
        self, configurations: list[np.ndarray], bases: list[np.ndarray]
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
        for q20, base in zip(configurations, bases):
            self.urdf.update_cfg(np.asarray(q20, dtype=np.float64))
            for node, box in corners.items():
                transform = base @ np.asarray(self.urdf.scene.graph.get(node)[0], dtype=np.float64)
                world = box @ transform[:3, :3].T + transform[:3, 3]
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
