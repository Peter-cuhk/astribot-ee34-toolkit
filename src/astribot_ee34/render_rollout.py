"""Render raw Astribot rollout recordings with the :mod:`astribot_ee34.render_video` layout.

``render_video`` needs a converted LeRobot split. Robot rollouts are uploaded as
a directory holding ``robot_episode.hdf5`` (joints plus the three cameras as
concatenated JPEG blobs) and ``recording.json``; this module renders those
directly, one MP4 per rollout, without a conversion step.
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import contract as C  # noqa: N812
from . import kinematics as kin
from . import render_video as rv

CAMERAS = {"head": "head", "left_wrist": "left", "right_wrist": "right"}


def split_jpeg_blob(blob: np.ndarray, sizes: np.ndarray) -> list[bytes]:
    """Cut ``images_dict/<cam>/rgb`` into per-frame JPEG payloads using ``rgb_size``."""
    ends = np.cumsum(np.asarray(sizes, dtype=np.int64))
    if len(ends) == 0 or ends[-1] != len(blob):
        raise ValueError("rgb blob length != sum(rgb_size)")
    starts = np.concatenate([[0], ends[:-1]])
    return [bytes(blob[start:end]) for start, end in zip(starts, ends)]


def resolve_hdf5(path: Path) -> Path:
    return path / "robot_episode.hdf5" if path.is_dir() else path


def default_prompt(hdf5_path: Path) -> str:
    recording = hdf5_path.parent / "recording.json"
    if recording.is_file():
        task = json.loads(recording.read_text()).get("task")
        if task:
            return str(task).replace("_", " ")
    return C.DEFAULT_PROMPT


def render_rollout(hdf5_path: Path, output: Path, args: argparse.Namespace) -> Path:
    key = rv.JOINT_STATE_KEY if args.stream == "state" else rv.JOINT_ACTION_KEY
    with h5py.File(hdf5_path, "r") as handle:
        joints = np.asarray(handle[key][:], dtype=np.float64)
        images = {
            label: split_jpeg_blob(handle[f"images_dict/{cam}/rgb"][:], handle[f"images_dict/{cam}/rgb_size"][:])
            for label, cam in CAMERAS.items()
        }
    if joints.ndim != 2 or joints.shape[1] != C.JOINT_DIM:
        raise ValueError(f"{key} shape {joints.shape} != (N, {C.JOINT_DIM})")
    for label, frames in images.items():
        if len(frames) != len(joints):
            raise ValueError(f"{label}: {len(frames)} images != {len(joints)} joint frames")
    prompt = args.prompt if args.prompt is not None else default_prompt(hdf5_path)

    model = kin.AstribotKinematics(args.urdf, args.torso_config)
    world_segments: list[np.ndarray] = []
    world_frames: list[dict[str, np.ndarray]] = []
    for joints25 in joints:
        q20 = kin.joints25_to_urdf20(joints25)
        world = kin.chassis_xyyaw_to_matrix(joints25[C.J_CHASSIS])
        base = world @ model.t_chassis_urdf_root
        world_segments.append(model.skeleton_segments(q20).astype(np.float64) @ base[:3, :3].T + base[:3, 3])
        fk = model.fk(q20)
        world_frames.append({link: world @ fk[link] for link in rv.FRAME_LABELS})

    layout = rv.build_layout(args.width, args.height, args.camera_fraction)
    renderer = rv.SkeletonRenderer(
        model, world_segments, (layout.skeleton[2], layout.skeleton[3]),
        args.elev, args.azim, args.axis_length, args.zoom,
    )
    title_font = ImageFont.truetype(str(rv.FONT_DIR / "DejaVuSans-Bold.ttf"), 34)
    label_font = ImageFont.truetype(str(rv.FONT_DIR / "DejaVuSans.ttf"), 15)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output, fps=args.fps, codec="libx264", quality=None, bitrate=None,
        macro_block_size=1, pixelformat="yuv420p", output_params=["-crf", str(args.crf)],
    )
    total = len(joints) if args.max_frames is None else min(len(joints), args.max_frames)
    try:
        for frame in range(total):
            canvas = Image.new("RGB", layout.size, rv.BG)
            draw = ImageDraw.Draw(canvas)
            draw.rectangle(layout.title, fill=rv.TITLE_BG)
            draw.text((rv.PAD + 16, rv.TITLE_H // 2), prompt, font=title_font, fill=rv.TEXT, anchor="lm")
            for label, box in (
                ("head", layout.head),
                ("left_wrist", layout.left_wrist),
                ("right_wrist", layout.right_wrist),
            ):
                rv._paste(canvas, Image.open(BytesIO(images[label][frame])).convert("RGB"), box)
                rv._draw_label(draw, label, box, label_font)
            rv._paste(canvas, renderer.render(world_segments[frame], world_frames[frame]), layout.skeleton)
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        renderer.close()
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path, nargs="+", help="rollout directories or robot_episode.hdf5 files")
    parser.add_argument("--output-dir", type=Path, required=True, help="writes <rollout dir name>.mp4 here")
    parser.add_argument("--prompt", default=None, help="title text (default: recording.json task)")
    parser.add_argument("--stream", choices=("state", "action"), default="state")
    parser.add_argument("--urdf", type=Path, default=kin.DEFAULT_URDF_PATH)
    parser.add_argument("--torso-config", type=Path, default=kin.DEFAULT_TORSO_CONFIG_PATH)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--camera-fraction", type=float, default=0.66)
    parser.add_argument("--elev", type=float, default=16.0)
    parser.add_argument("--azim", type=float, default=-72.0)
    parser.add_argument("--axis-length", type=float, default=0.16)
    parser.add_argument("--zoom", type=float, default=1.6, help="3D panel fill factor")
    parser.add_argument("--fps", type=int, default=C.FPS)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--max-frames", type=int, default=None, help="smoke-test a prefix of each rollout")
    args = parser.parse_args()
    if not 0.3 <= args.camera_fraction <= 0.9:
        parser.error("--camera-fraction must be within [0.3, 0.9]")
    if args.width % 2 or args.height % 2:
        parser.error("--width and --height must be even for yuv420p")
    return args


def main() -> int:
    args = _parse_args()
    for rollout in args.rollouts:
        hdf5_path = resolve_hdf5(rollout)
        output = args.output_dir / f"{hdf5_path.parent.name}.mp4"
        print(f"wrote {render_rollout(hdf5_path, output, args)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
