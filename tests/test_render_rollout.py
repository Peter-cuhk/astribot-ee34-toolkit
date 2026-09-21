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


def test_hull_chunks_drop_negligible_shells() -> None:
    import trimesh

    from astribot_ee34.robot_mesh import _hull_chunks

    big = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    bolt = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    bolt.apply_translation((3.0, 0.0, 0.0))
    link = trimesh.util.concatenate([big, bolt])

    chunks = _hull_chunks(link, keep_ratio=0.12)

    assert len(chunks) == 1
    assert chunks[0].vertices.shape[1] == 3
    assert chunks[0].faces.shape[1] == 3


def test_hull_chunks_keep_every_significant_shell() -> None:
    import trimesh

    from astribot_ee34.robot_mesh import _hull_chunks

    first = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    second = trimesh.creation.box(extents=(0.8, 0.8, 0.8))
    second.apply_translation((3.0, 0.0, 0.0))

    chunks = _hull_chunks(trimesh.util.concatenate([first, second]), keep_ratio=0.12)

    assert len(chunks) == 2
