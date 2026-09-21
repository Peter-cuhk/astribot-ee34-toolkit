"""Low-poly solid robot geometry for the offline renderers.

The Astribot visual STLs are ~1.2M triangles across 24 links, which is far more
than a 645x988 panel can show and far too slow to redraw 500 times. Each link is
therefore split into its connected shells, the negligible ones (bolt heads,
washers) are dropped, and every remaining shell is replaced by its convex hull.
That keeps the silhouette a viewer actually reads -- torso column, shoulders,
upper and lower arms, wrists, head -- at ~79k triangles, while deliberately
giving up concave interior detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import kinematics as kin

# A shell is kept when its longest extent is at least this fraction of the
# largest shell in the same link.
DEFAULT_KEEP_RATIO = 0.12
# Lambert light direction in world coordinates, and the ambient floor so faces
# pointing away from it stay readable rather than going black.
LIGHT_DIRECTION = np.asarray([0.4, -0.7, 0.6], dtype=np.float64)
AMBIENT = 0.38
BASE_COLOR = np.asarray([0.36, 0.55, 0.86], dtype=np.float64)


@dataclass(frozen=True)
class Chunk:
    """One convex shell of a link, in that link's frame."""

    vertices: np.ndarray
    faces: np.ndarray


class RobotMesh:
    """Per-link convex shells plus the per-frame world-space triangle build."""

    def __init__(self, urdf_path: Path, keep_ratio: float = DEFAULT_KEEP_RATIO) -> None:
        self.urdf = kin.load_visual_urdf(urdf_path)
        self.parts: dict[str, list[Chunk]] = {}
        for node in self.urdf.scene.graph.nodes_geometry:
            geometry = self.urdf.scene.geometry[self.urdf.scene.graph[node][1]]
            self.parts[node] = _hull_chunks(geometry, keep_ratio)
        self.face_count = sum(len(chunk.faces) for chunks in self.parts.values() for chunk in chunks)
        if self.face_count == 0:
            raise kin.KinematicsConfigError(f"no renderable visual geometry in {urdf_path}")
        self.light = LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)

    def triangles(self, q20: np.ndarray, base: np.ndarray) -> np.ndarray:
        """World-space (n, 3, 3) triangles for one configuration."""
        self.urdf.update_cfg(np.asarray(q20, dtype=np.float64))
        batches: list[np.ndarray] = []
        for node, chunks in self.parts.items():
            transform = base @ np.asarray(self.urdf.scene.graph.get(node)[0], dtype=np.float64)
            rotation, translation = transform[:3, :3], transform[:3, 3]
            for chunk in chunks:
                world = chunk.vertices @ rotation.T + translation
                batches.append(world[chunk.faces])
        return np.concatenate(batches, axis=0)

    def shade(self, triangles: np.ndarray) -> np.ndarray:
        """Flat Lambert shading, one RGB per triangle."""
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        lit = AMBIENT + (1.0 - AMBIENT) * np.abs(normals @ self.light)
        return np.clip(lit[:, None] * BASE_COLOR[None, :], 0.0, 1.0)


def _hull_chunks(geometry, keep_ratio: float) -> list[Chunk]:
    shells = list(geometry.split(only_watertight=False)) or [geometry]
    extents = np.asarray([float(shell.extents.max()) for shell in shells])
    chunks: list[Chunk] = []
    for shell, extent in zip(shells, extents):
        if extent < keep_ratio * extents.max() or len(shell.vertices) < 4:
            continue
        try:
            hull = shell.convex_hull
        except Exception:
            continue
        chunks.append(
            Chunk(
                vertices=np.asarray(hull.vertices, dtype=np.float64),
                faces=np.asarray(hull.faces, dtype=np.int64),
            )
        )
    return chunks
