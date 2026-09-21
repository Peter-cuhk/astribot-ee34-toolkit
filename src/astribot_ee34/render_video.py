"""Offline MP4 renderer: camera panels as the main view, skeleton as the side view.

Unlike :mod:`astribot_ee34.visualize` (an interactive Viser server that has to be
screen-recorded), this module composites frames headlessly and writes an MP4
directly. The layout puts the recorded cameras in the large left block and the
URDF skeleton in the narrow right block, with the episode language prompt in the
title bar. ``--head-only`` keeps just the head camera on the left.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib import get_data_path
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image, ImageDraw, ImageFont

from . import contract as C  # noqa: N812
from . import kinematics as kin
from .robot_mesh import RobotMesh

IMAGE_COLUMNS = {
    "head": "observation.images.head",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}
# EE34 end-effector frames, labelled with the camera they carry so the skeleton
# panel and the camera panels use one vocabulary.
FRAME_LABELS = {"torso": "head", "left": "left_wrist", "right": "right_wrist"}
JOINT_STATE_KEY = "joints_dict/joints_position_state"
JOINT_ACTION_KEY = "joints_dict/joints_position_command"

BG = (12, 16, 24)
TITLE_BG = (12, 16, 24)
LABEL_BG = (28, 36, 50)
TEXT = (236, 240, 246)
PANEL_BG = (247, 249, 252)
SKELETON_COLOR = "#1668d8"
AXIS_COLORS = ("#d02020", "#12a012", "#1a4fd8")

TITLE_H = 76
LABEL_H = 26
PAD = 8
GAP = 4
FONT_DIR = Path(get_data_path()) / "fonts/ttf"


@dataclass(frozen=True)
class Layout:
    """Pixel boxes for one composited frame, as (left, top, width, height).

    ``left_wrist`` and ``right_wrist`` are ``None`` in head-only layouts.
    """

    size: tuple[int, int]
    title: tuple[int, int, int, int]
    head: tuple[int, int, int, int]
    left_wrist: tuple[int, int, int, int] | None
    right_wrist: tuple[int, int, int, int] | None
    skeleton: tuple[int, int, int, int]

    def camera_panels(self) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
        """The camera boxes to draw, in paste order."""
        panels = [("head", self.head)]
        if self.left_wrist is not None and self.right_wrist is not None:
            panels.append(("left_wrist", self.left_wrist))
            panels.append(("right_wrist", self.right_wrist))
        return tuple(panels)


def build_layout(width: int, height: int, camera_fraction: float, *, wrists: bool = True) -> Layout:
    """Split the canvas into a title bar, a camera block and a skeleton block.

    All three cameras are 16:9, so the camera block height is fixed by its width:
    one full-width view plus a half-width row is ``27/32`` of the block width,
    or ``9/16`` when the wrist row is dropped. The camera block width is capped
    by ``camera_fraction`` and by the height that is actually left over,
    whichever binds first; the block is then centred in the body area, which
    matters for head-only layouts where ``camera_fraction`` usually binds.
    """
    body_top = TITLE_H
    body_h = height - TITLE_H - 2 * PAD
    max_cam_w = int((width - 3 * PAD) * camera_fraction)
    if wrists:
        cam_w = min(max_cam_w, int((body_h - 2 * LABEL_H - GAP) * 32 / 27))
        head_h = round(cam_w * 9 / 16)
        wrist_w = (cam_w - GAP) // 2
        wrist_h = round(wrist_w * 9 / 16)
        block_h = LABEL_H + head_h + GAP + LABEL_H + wrist_h
    else:
        cam_w = min(max_cam_w, int((body_h - LABEL_H) * 16 / 9))
        head_h = round(cam_w * 9 / 16)
        block_h = LABEL_H + head_h
    by = body_top + PAD + (body_h - block_h) // 2
    skel_x = PAD + cam_w + PAD
    skel_w = width - skel_x - PAD
    if not wrists:
        # A 16:9 head panel wide enough to leave room for the skeleton cannot
        # also fill the body height, so it is letterboxed and centred while the
        # skeleton takes the full height of its column.
        return Layout(
            size=(width, height),
            title=(0, 0, width, TITLE_H),
            head=(PAD, by + LABEL_H, cam_w, head_h),
            left_wrist=None,
            right_wrist=None,
            skeleton=(skel_x, body_top + PAD, skel_w, body_h),
        )
    wrist_top = by + LABEL_H + head_h + GAP + LABEL_H
    return Layout(
        size=(width, height),
        title=(0, 0, width, TITLE_H),
        head=(PAD, by + LABEL_H, cam_w, head_h),
        left_wrist=(PAD, wrist_top, wrist_w, wrist_h),
        right_wrist=(PAD + wrist_w + GAP, wrist_top, wrist_w, wrist_h),
        skeleton=(skel_x, by, skel_w, block_h),
    )


def _load_provenance(split_root: Path, episode_id: str) -> tuple[int, int, Path]:
    records_path = split_root / "meta/ee34_conversion_episodes.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError(f"missing conversion provenance: {records_path}")
    for local_index, line in enumerate(records_path.read_text().splitlines()):
        if not line:
            continue
        payload = json.loads(line)
        if str(payload["episode_id"]) == episode_id:
            return local_index, int(payload["frame_count"]), Path(payload["source_file"])
    raise ValueError(f"episode {episode_id!r} is not in {split_root}")


def _load_prompt(split_root: Path) -> str:
    tasks_path = split_root / "meta/tasks.jsonl"
    if not tasks_path.is_file():
        return C.DEFAULT_PROMPT
    lines = [line for line in tasks_path.read_text().splitlines() if line]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one task in {tasks_path}, got {len(lines)}")
    return str(json.loads(lines[0])["task"])


def _find_parquet(split_root: Path, local_index: int) -> Path:
    matches = sorted(split_root.glob(f"data/chunk-*/episode_{local_index:06d}.parquet"))
    if len(matches) != 1:
        raise ValueError(f"expected one parquet for episode index {local_index}, got {matches}")
    return matches[0]


def _decode(payload: dict, parquet_file: Path) -> Image.Image:
    encoded = payload.get("bytes")
    if encoded is None:
        relative = payload.get("path")
        if not relative:
            raise ValueError("image payload contains neither bytes nor path")
        candidates = (parquet_file.parent / relative, parquet_file.parents[2] / relative)
        image_path = next((path for path in candidates if path.is_file()), None)
        if image_path is None:
            raise FileNotFoundError(f"image path not found: {relative}")
        encoded = image_path.read_bytes()
    return Image.open(BytesIO(bytes(encoded))).convert("RGB")


def _load_joints(hdf5_path: Path, key: str, frame_count: int) -> np.ndarray:
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        joints = np.asarray(handle[key][:], dtype=np.float64)
    if joints.shape != (frame_count, C.JOINT_DIM):
        raise ValueError(f"{key} shape {joints.shape} != ({frame_count}, {C.JOINT_DIM})")
    return joints


class SkeletonRenderer:
    """Fixed-camera matplotlib view of the joint-FK skeleton and its EE frames."""

    def __init__(
        self,
        model: kin.AstribotKinematics,
        world_segments: list[np.ndarray],
        size: tuple[int, int],
        elev: float,
        azim: float,
        axis_length: float,
        zoom: float,
        mesh: RobotMesh | None = None,
    ) -> None:
        self.model = model
        self.mesh = mesh
        self.axis_length = axis_length
        width, height = size
        dpi = 100.0
        self.figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self.figure.patch.set_facecolor(np.asarray(PANEL_BG) / 255.0)
        self.ax = self.figure.add_axes((0.0, 0.0, 1.0, 1.0), projection="3d")
        self.ax.set_facecolor(np.asarray(PANEL_BG) / 255.0)
        self.ax.set_axis_off()
        self.ax.view_init(elev=elev, azim=azim)

        # One fixed cube around the whole episode: the camera never moves, and
        # equal half-extents on every axis keep the robot undistorted.
        points = np.concatenate([segments.reshape(-1, 3) for segments in world_segments], axis=0)
        low = points.min(axis=0)
        high = points.max(axis=0)
        low[2] = 0.0
        center = (low + high) / 2.0
        # Link segments only bound the joint centres; the solid body sticks out
        # past them, so the mesh view needs a wider cube.
        radius = float(np.max(high - low)) / 2.0 * (1.12 if mesh is not None else 1.08)
        self.ax.set_xlim(center[0] - radius, center[0] + radius)
        self.ax.set_ylim(center[1] - radius, center[1] + radius)
        self.ax.set_zlim(center[2] - radius, center[2] + radius)
        self.ax.set_box_aspect((1.0, 1.0, 1.0), zoom=zoom)

        if mesh is not None:
            # A solid body is one collection, so matplotlib's per-artist depth
            # sort puts the floor grid on top of the base. The camera is fixed
            # and the robot always stands on the floor, so fix the order by hand.
            self.ax.computed_zorder = False
        grid = Line3DCollection(
            _floor_grid(center, radius), colors="#c8ced8", linewidths=0.8, zorder=0
        )
        self.ax.add_collection3d(grid)
        if mesh is None:
            self.skeleton = Line3DCollection(np.zeros((1, 2, 3)), colors=SKELETON_COLOR, linewidths=7.0)
            self.ax.add_collection3d(self.skeleton)
            self.solid = None
        else:
            self.skeleton = None
            self.solid = Poly3DCollection(
                np.zeros((1, 3, 3)), edgecolors="none", shade=False, zorder=1
            )
            self.ax.add_collection3d(self.solid)
        self.axes_art = Line3DCollection(
            np.zeros((1, 2, 3)), colors=["#000000"], linewidths=2.6, zorder=2
        )
        self.ax.add_collection3d(self.axes_art)
        self.texts = {
            label: self.ax.text(0.0, 0.0, 0.0, label, color="#101820", fontsize=11, zorder=3)
            for label in FRAME_LABELS.values()
        }

    def render(
        self,
        segments: np.ndarray,
        frames: dict[str, np.ndarray],
        q20: np.ndarray | None = None,
        base: np.ndarray | None = None,
    ) -> Image.Image:
        if self.mesh is None:
            self.skeleton.set_segments(list(segments))
        else:
            if q20 is None or base is None:
                raise ValueError("mesh rendering needs the per-frame q20 and base transform")
            triangles = self.mesh.triangles(q20, base)
            self.solid.set_verts(triangles)
            self.solid.set_facecolor(self.mesh.shade(triangles))
        axis_segments: list[np.ndarray] = []
        colors: list[str] = []
        for link, label in FRAME_LABELS.items():
            pose = frames[link]
            origin = pose[:3, 3]
            for axis in range(3):
                axis_segments.append(np.stack([origin, origin + pose[:3, axis] * self.axis_length]))
                colors.append(AXIS_COLORS[axis])
            self.texts[label].set_position_3d((origin[0], origin[1], origin[2] - 0.06))
        self.axes_art.set_segments(axis_segments)
        self.axes_art.set_color(colors)
        self.figure.canvas.draw()
        return Image.fromarray(np.asarray(self.figure.canvas.buffer_rgba())[:, :, :3])

    def close(self) -> None:
        plt.close(self.figure)


def _floor_grid(center: np.ndarray, radius: float, cells: int = 10) -> list[np.ndarray]:
    half = radius * 0.8
    ticks = np.linspace(-half, half, cells + 1)
    x, y = center[0], center[1]
    segments = []
    for tick in ticks:
        segments.append(np.array([[x + tick, y - half, 0.0], [x + tick, y + half, 0.0]]))
        segments.append(np.array([[x - half, y + tick, 0.0], [x + half, y + tick, 0.0]]))
    return segments


def _paste(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    left, top, width, height = box
    canvas.paste(image.resize((width, height), Image.LANCZOS), (left, top))


def _draw_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.FreeTypeFont,
) -> None:
    left, top, width, _ = box
    draw.rectangle((left, top - LABEL_H, left + width, top), fill=LABEL_BG)
    draw.text((left + 8, top - LABEL_H + LABEL_H // 2), text, font=font, fill=TEXT, anchor="lm")


def render_episode(args: argparse.Namespace) -> Path:
    split_root = args.dataset_root
    local_index, frame_count, source_file = _load_provenance(split_root, args.episode_id)
    hdf5_path = args.hdf5 if args.hdf5 is not None else source_file
    if not hdf5_path.is_file():
        raise FileNotFoundError(
            f"joint source HDF5 not found: {hdf5_path}\n"
            "pass --hdf5 to point at a relocated copy of the recording"
        )
    prompt = args.prompt if args.prompt is not None else _load_prompt(split_root)
    parquet_file = _find_parquet(split_root, local_index)

    layout = build_layout(args.width, args.height, args.camera_fraction, wrists=not args.head_only)
    wanted = {label: IMAGE_COLUMNS[label] for label, _ in layout.camera_panels()}
    table = pq.read_table(parquet_file, columns=[*wanted.values(), "frame_index"])
    if table.num_rows != frame_count:
        raise ValueError(f"parquet length {table.num_rows} != provenance {frame_count}")
    images = {label: table[column].combine_chunks().to_pylist() for label, column in wanted.items()}

    key = JOINT_STATE_KEY if args.stream == "state" else JOINT_ACTION_KEY
    joints = _load_joints(hdf5_path, key, frame_count)
    model = kin.AstribotKinematics(args.urdf, args.torso_config)

    world_segments: list[np.ndarray] = []
    world_frames: list[dict[str, np.ndarray]] = []
    configurations: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    for joints25 in joints:
        q20 = kin.joints25_to_urdf20(joints25)
        world = kin.chassis_xyyaw_to_matrix(joints25[C.J_CHASSIS])
        local = model.skeleton_segments(q20).astype(np.float64)
        base = world @ model.t_chassis_urdf_root
        world_segments.append(local @ base[:3, :3].T + base[:3, 3])
        fk = model.fk(q20)
        world_frames.append({link: world @ fk[link] for link in FRAME_LABELS})
        configurations.append(q20)
        bases.append(base)

    mesh = RobotMesh(args.urdf) if args.robot_style == "mesh" else None

    skeleton_size = (layout.skeleton[2], layout.skeleton[3])
    renderer = SkeletonRenderer(
        model, world_segments, skeleton_size, args.elev, args.azim, args.axis_length, args.zoom, mesh
    )
    title_font = ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), 34)
    label_font = ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), 15)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output, fps=args.fps, codec="libx264", quality=None, bitrate=None,
        macro_block_size=1, pixelformat="yuv420p", output_params=["-crf", str(args.crf)],
    )
    total = frame_count if args.max_frames is None else min(frame_count, args.max_frames)
    try:
        for frame in range(total):
            canvas = Image.new("RGB", layout.size, BG)
            draw = ImageDraw.Draw(canvas)
            draw.rectangle(layout.title, fill=TITLE_BG)
            draw.text((PAD + 16, TITLE_H // 2), prompt, font=title_font, fill=TEXT, anchor="lm")
            for label, box in layout.camera_panels():
                _paste(canvas, _decode(images[label][frame], parquet_file), box)
                _draw_label(draw, label, box, label_font)
            panel = renderer.render(
                world_segments[frame], world_frames[frame], configurations[frame], bases[frame]
            )
            _paste(canvas, panel, layout.skeleton)
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        renderer.close()
    return args.output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="LeRobot split directory")
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, default=None, help="override the recorded joint source")
    parser.add_argument("--prompt", default=None, help="override the title text (default: meta/tasks.jsonl)")
    parser.add_argument("--stream", choices=("state", "action"), default="state")
    parser.add_argument("--urdf", type=Path, default=kin.DEFAULT_URDF_PATH)
    parser.add_argument("--torso-config", type=Path, default=kin.DEFAULT_TORSO_CONFIG_PATH)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--camera-fraction", type=float, default=0.66)
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="drop the left_wrist/right_wrist row: head camera left, skeleton right",
    )
    parser.add_argument(
        "--robot-style",
        choices=("skeleton", "mesh"),
        default="skeleton",
        help="3D panel: joint skeleton lines, or the solid low-poly robot body",
    )
    parser.add_argument("--elev", type=float, default=16.0)
    parser.add_argument("--azim", type=float, default=-72.0)
    parser.add_argument("--axis-length", type=float, default=0.16)
    parser.add_argument("--zoom", type=float, default=1.6, help="3D panel fill factor")
    parser.add_argument("--fps", type=int, default=C.FPS)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--max-frames", type=int, default=None, help="smoke-test a prefix of the episode")
    args = parser.parse_args()
    if not 0.3 <= args.camera_fraction <= 0.9:
        parser.error("--camera-fraction must be within [0.3, 0.9]")
    if args.width % 2 or args.height % 2:
        parser.error("--width and --height must be even for yuv420p")
    return args


def main() -> int:
    print(f"wrote {render_episode(_parse_args())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
