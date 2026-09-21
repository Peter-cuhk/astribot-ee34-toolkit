from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("imageio")

from astribot_ee34.render_rollout import split_jpeg_blob
from astribot_ee34.render_video import build_layout


def test_split_jpeg_blob_uses_sizes_as_lengths() -> None:
    blob = np.frombuffer(b"aaabbcccc", dtype=np.uint8)
    assert split_jpeg_blob(blob, np.array([3.0, 2.0, 4.0])) == [b"aaa", b"bb", b"cccc"]


def test_split_jpeg_blob_rejects_length_mismatch() -> None:
    blob = np.frombuffer(b"aaabb", dtype=np.uint8)
    with pytest.raises(ValueError):
        split_jpeg_blob(blob, np.array([3.0, 3.0]))


def _fits(box, width: int, height: int) -> bool:
    left, top, w, h = box
    return left >= 0 and top >= 0 and left + w <= width and top + h <= height


def test_head_only_layout_drops_the_wrist_row() -> None:
    layout = build_layout(1920, 1080, 0.66, wrists=False)

    assert layout.left_wrist is None
    assert layout.right_wrist is None
    assert [label for label, _ in layout.camera_panels()] == ["head"]
    assert _fits(layout.head, 1920, 1080)
    assert _fits(layout.skeleton, 1920, 1080)
    # head is 16:9 and letterboxed; the skeleton column runs the full body height
    assert layout.head[3] == round(layout.head[2] * 9 / 16)
    assert layout.skeleton[3] == 1080 - 76 - 2 * 8
    assert layout.head[1] > layout.skeleton[1]


def test_default_layout_keeps_all_three_cameras() -> None:
    layout = build_layout(1920, 1080, 0.66)

    assert [label for label, _ in layout.camera_panels()] == ["head", "left_wrist", "right_wrist"]
    for box in (layout.head, layout.left_wrist, layout.right_wrist, layout.skeleton):
        assert _fits(box, 1920, 1080)


def test_head_only_head_panel_is_taller_than_the_three_camera_one() -> None:
    assert build_layout(1920, 1080, 0.66, wrists=False).head[3] > build_layout(1920, 1080, 0.66).head[3]


def test_hull_geometry_merges_every_shell_of_a_link() -> None:
    import trimesh

    from astribot_ee34.robot_mesh import _hull_geometry

    big = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    small = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    small.apply_translation((3.0, 0.0, 0.0))

    merged = _hull_geometry(trimesh.util.concatenate([big, small]))

    # Both shells survive -- the small ones are what fill the gaps between the
    # big ones -- and their faces index into one shared vertex array.
    assert merged.vertices.shape == (16, 3)
    assert merged.faces.shape[1] == 3
    assert merged.faces.max() == len(merged.vertices) - 1
    assert merged.vertices[merged.faces].reshape(-1, 3)[:, 0].max() > 2.0


def test_parse_color_accepts_hex_with_or_without_hash() -> None:
    from astribot_ee34.robot_mesh import _parse_color

    assert _parse_color("#ff8000") == pytest.approx([1.0, 128 / 255, 0.0])
    assert _parse_color("FF8000") == pytest.approx([1.0, 128 / 255, 0.0])


def test_parse_color_rejects_a_bad_string() -> None:
    from astribot_ee34.robot_mesh import _parse_color

    with pytest.raises(ValueError, match="6-digit hex"):
        _parse_color("blue")


def test_aabb_corners_span_the_vertex_box() -> None:
    from astribot_ee34.robot_mesh import _aabb_corners

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.5, 1.0, 1.0]])

    corners = _aabb_corners(vertices)

    assert corners.shape == (8, 3)
    assert corners.min(axis=0) == pytest.approx([0.0, 0.0, 0.0])
    assert corners.max(axis=0) == pytest.approx([1.0, 2.0, 3.0])
