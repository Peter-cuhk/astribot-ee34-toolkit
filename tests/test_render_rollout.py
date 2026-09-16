from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("imageio")

from astribot_ee34.render_rollout import split_jpeg_blob


def test_split_jpeg_blob_uses_sizes_as_lengths() -> None:
    blob = np.frombuffer(b"aaabbcccc", dtype=np.uint8)
    assert split_jpeg_blob(blob, np.array([3.0, 2.0, 4.0])) == [b"aaa", b"bb", b"cccc"]


def test_split_jpeg_blob_rejects_length_mismatch() -> None:
    blob = np.frombuffer(b"aaabb", dtype=np.uint8)
    with pytest.raises(ValueError):
        split_jpeg_blob(blob, np.array([3.0, 3.0]))
