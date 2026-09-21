"""Interactive Viser player for joint/FK/EE34/IK comparison."""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.parquet as pq
import viser
import yourdfpy
from PIL import Image
from viser.extras import ViserUrdf

from . import contract as C  # noqa: N812
from . import kinematics as kin

SPLIT_REPO_IDS = {
    "train": C.TRAIN_REPO_ID,
    "validation": C.VAL_REPO_ID,
}
IMAGE_COLUMNS = {
    "head": "observation.images.head",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}
STREAM_COLUMNS = {"state": "observation.state", "action": "action"}
STREAM_JOINT_KEYS = {
    "state": "joints_dict/joints_position_state",
    "action": "joints_dict/joints_position_command",
}


@dataclass(frozen=True)
class EpisodeRecord:
    split: str
    local_episode_index: int
    episode_id: str
    source_file: Path
    parquet_file: Path
    frame_count: int


@dataclass
class LoadedEpisode:
    record: EpisodeRecord
    state: np.ndarray
    action: np.ndarray
    joints_state: np.ndarray
    joints_action: np.ndarray
    images: dict[str, list[dict[str, Any]]]


def _find_parquet(split_root: Path, local_episode_index: int) -> Path:
    matches = sorted(split_root.glob(f"data/chunk-*/episode_{local_episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise ValueError(f"expected one parquet for episode index {local_episode_index}, got {matches}")
    return matches[0]


def load_catalog(output_root: Path) -> dict[str, list[EpisodeRecord]]:
    catalog: dict[str, list[EpisodeRecord]] = {}
    for split, repo_id in SPLIT_REPO_IDS.items():
        split_root = output_root.joinpath(*repo_id.split("/"))
        records_path = split_root / "meta/ee34_conversion_episodes.jsonl"
        metadata_path = split_root / "meta/ee34_conversion.json"
        if not records_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"missing conversion provenance under {split_root}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("contract_id") != C.CONTRACT_ID:
            raise ValueError(f"{split}: contract_id mismatch")
        records: list[EpisodeRecord] = []
        for local_index, line in enumerate(records_path.read_text().splitlines()):
            if not line:
                continue
            payload = json.loads(line)
            records.append(
                EpisodeRecord(
                    split=split,
                    local_episode_index=local_index,
                    episode_id=str(payload["episode_id"]),
                    source_file=Path(payload["source_file"]),
                    parquet_file=_find_parquet(split_root, local_index),
                    frame_count=int(payload["frame_count"]),
                )
            )
        catalog[split] = records
    return catalog


def _fixed_vectors(table: Any, column: str) -> np.ndarray:
    values = np.asarray(table[column].combine_chunks().to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != C.EE34_DIM:
        raise ValueError(f"{column} must be (T,{C.EE34_DIM}), got {values.shape}")
    return values


def load_episode(record: EpisodeRecord) -> LoadedEpisode:
    columns = [*STREAM_COLUMNS.values(), *IMAGE_COLUMNS.values(), "frame_index", "episode_index"]
    table = pq.read_table(record.parquet_file, columns=columns)
    if table.num_rows != record.frame_count:
        raise ValueError(f"parquet length {table.num_rows} != provenance {record.frame_count}")
    frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    episode_index = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    if not np.array_equal(frame_index, np.arange(record.frame_count)):
        raise ValueError("frame_index is not contiguous")
    if not np.all(episode_index == record.local_episode_index):
        raise ValueError("episode_index does not match provenance order")
    with h5py.File(record.source_file, "r") as handle:
        joints_state = np.asarray(handle[STREAM_JOINT_KEYS["state"]][:], dtype=np.float64)
        joints_action = np.asarray(handle[STREAM_JOINT_KEYS["action"]][:], dtype=np.float64)
    if joints_state.shape != (record.frame_count, C.JOINT_DIM):
        raise ValueError(f"joints_state shape drifted: {joints_state.shape}")
    if joints_action.shape != (record.frame_count, C.JOINT_DIM):
        raise ValueError(f"joints_action shape drifted: {joints_action.shape}")
    return LoadedEpisode(
        record=record,
        state=_fixed_vectors(table, STREAM_COLUMNS["state"]),
        action=_fixed_vectors(table, STREAM_COLUMNS["action"]),
        joints_state=joints_state,
        joints_action=joints_action,
        images={label: table[column].combine_chunks().to_pylist() for label, column in IMAGE_COLUMNS.items()},
    )


def _decode_image(payload: dict[str, Any], parquet_file: Path) -> np.ndarray:
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
    with Image.open(BytesIO(bytes(encoded))) as image:
        return np.asarray(image.convert("RGB"))


def _root_pose(chassis3: np.ndarray, weld: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return kin.matrix_to_wxyz_position(kin.chassis_xyyaw_to_matrix(chassis3) @ weld)


def _world_pose(chassis3: np.ndarray, local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return kin.matrix_to_wxyz_position(kin.chassis_xyyaw_to_matrix(chassis3) @ local)


def _load_visual_urdf(path: Path) -> yourdfpy.URDF:
    return kin.load_visual_urdf(path)


class KinematicsViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.catalog = load_catalog(args.output_root)
        self.model = kin.AstribotKinematics(args.urdf, args.torso_config)
        self.report = self._load_report(args.report)
        self.lock = threading.RLock()
        self.loaded: LoadedEpisode | None = None
        self.trajectory_points: dict[str, np.ndarray] = {}
        self.stream = args.stream
        self.last_ik_q20: np.ndarray | None = None
        self.last_ik_valid = False
        self.last_tick = time.monotonic()

        initial_record = self._select_initial_record(args.split, args.episode_id)
        self.server = viser.ViserServer(host=args.host, port=args.port, label="Astribot EE34 audit")
        self.server.gui.configure_theme(
            control_layout="fixed",
            control_width="large",
            show_share_button=False,
        )
        self.server.initial_camera.position = (0.0, -3.0, 1.35)
        self.server.initial_camera.look_at = (0.1, 0.0, 0.85)
        self.server.scene.add_grid("/grid", width=4.0, height=4.0, cell_size=0.25)

        self.source_root = self.server.scene.add_frame("/robots/source", show_axes=False)
        self.ghost_valid_root = self.server.scene.add_frame("/robots/ghost_valid", show_axes=False)
        self.ghost_invalid_root = self.server.scene.add_frame("/robots/ghost_invalid", show_axes=False, visible=False)
        visual_model = None if args.no_meshes else _load_visual_urdf(args.urdf)
        self.source_urdf = ViserUrdf(
            self.server,
            args.urdf if args.no_meshes else visual_model,
            root_node_name="/robots/source",
            mesh_color_override=(70, 130, 230),
            load_meshes=not args.no_meshes,
        )
        empty_skeleton = self.model.skeleton_segments(np.zeros(20, dtype=np.float64))
        self.source_skeleton = self.server.scene.add_line_segments(
            "/robots/source/skeleton",
            empty_skeleton,
            colors=(70, 130, 230),
            line_width=6.0,
            visible=args.no_meshes,
        )
        self.ghost_valid_skeleton = self.server.scene.add_line_segments(
            "/robots/ghost_valid/skeleton",
            empty_skeleton,
            colors=(255, 140, 30),
            line_width=6.0,
        )
        self.ghost_invalid_skeleton = self.server.scene.add_line_segments(
            "/robots/ghost_invalid/skeleton",
            empty_skeleton,
            colors=(230, 45, 45),
            line_width=6.0,
            visible=False,
        )

        self.target_frames = {
            label: self.server.scene.add_frame(
                f"/frames/target/{label}",
                axes_length=0.12,
                axes_radius=0.006,
            )
            for label in kin.LINK_LABELS
        }
        self.fk_frames = {
            label: self.server.scene.add_frame(
                f"/frames/fk/{label}",
                axes_length=0.075,
                axes_radius=0.003,
            )
            for label in kin.LINK_LABELS
        }
        self.trajectory = self.server.scene.add_spline_catmull_rom(
            "/trajectory/selected",
            np.zeros((2, 3), dtype=np.float32),
            line_width=3.0,
            color=(230, 45, 45),
        )
        self._build_gui(initial_record)
        self._load_record(initial_record)
        if args.start_at_worst:
            self._jump_to_worst()
        elif args.start_frame is not None:
            self._set_frame(args.start_frame)

        @self.server.on_client_connect
        def _reset_new_client_camera(client: viser.ClientHandle) -> None:
            self._reset_camera(client)

    @staticmethod
    def _load_report(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
        if payload.get("contract_id") != C.CONTRACT_ID:
            raise ValueError(f"audit report contract mismatch: {path}")
        return payload

    def _select_initial_record(self, split: str, episode_id: str | None) -> EpisodeRecord:
        records = self.catalog[split]
        if episode_id is None:
            return records[0]
        matches = [record for record in records if record.episode_id == episode_id]
        if len(matches) != 1:
            raise ValueError(f"episode {episode_id!r} is not in split {split!r}")
        return matches[0]

    def _build_gui(self, initial: EpisodeRecord) -> None:
        max_frames = max(record.frame_count for records in self.catalog.values() for record in records)
        with self.server.gui.add_folder("Playback"):
            self.split_gui = self.server.gui.add_dropdown(
                "Split", tuple(SPLIT_REPO_IDS), initial_value=initial.split
            )
            self.episode_gui = self.server.gui.add_dropdown(
                "Episode",
                tuple(record.episode_id for record in self.catalog[initial.split]),
                initial_value=initial.episode_id,
            )
            self.frame_gui = self.server.gui.add_slider(
                "Frame", min=0, max=max_frames - 1, step=1, initial_value=0
            )
            self.stream_gui = self.server.gui.add_dropdown(
                "Data", ("State", "Action"), initial_value=self.stream.title()
            )
            self.playing_gui = self.server.gui.add_checkbox("Playing", initial_value=False)
            self.speed_gui = self.server.gui.add_dropdown(
                "Speed", ("0.25x", "0.5x", "1x", "2x"), initial_value="1x"
            )
            self.previous_gui = self.server.gui.add_button("Previous")
            self.next_gui = self.server.gui.add_button("Next")
            self.worst_gui = self.server.gui.add_button("Jump to worst frame")
            self.reset_camera_gui = self.server.gui.add_button("Reset camera")
            self.camera_view_gui = self.server.gui.add_dropdown(
                "Camera view",
                ("Side", "Diagonal", "Front", "Opposite side"),
                initial_value="Side",
            )
        with self.server.gui.add_folder("Visibility"):
            self.show_source_gui = self.server.gui.add_checkbox("Joint robot", initial_value=True)
            self.show_ghost_gui = self.server.gui.add_checkbox("IK ghost", initial_value=True)
            self.show_targets_gui = self.server.gui.add_checkbox("EE targets", initial_value=True)
            self.show_fk_gui = self.server.gui.add_checkbox("FK frames", initial_value=True)
            self.show_trajectory_gui = self.server.gui.add_checkbox("EE trajectory", initial_value=True)
            self.trajectory_gui = self.server.gui.add_dropdown(
                "Trajectory link",
                ("Left", "Right", "Torso", "None"),
                initial_value=self.args.trajectory.title(),
            )
        with self.server.gui.add_folder("Frame diagnostics"):
            self.status_gui = self.server.gui.add_markdown("Loading…")
            placeholder = np.zeros((32, 48, 3), dtype=np.uint8)
            self.image_gui = {
                label: self.server.gui.add_image(placeholder, label=label, format="jpeg")
                for label in IMAGE_COLUMNS
            }

        @self.split_gui.on_update
        def _(_) -> None:
            with self.lock:
                split = str(self.split_gui.value)
                options = tuple(record.episode_id for record in self.catalog[split])
                self.episode_gui.options = options
                self.episode_gui.value = options[0]
                self._load_record(self.catalog[split][0])

        @self.episode_gui.on_update
        def _(_) -> None:
            with self.lock:
                split = str(self.split_gui.value)
                episode_id = str(self.episode_gui.value)
                record = next(record for record in self.catalog[split] if record.episode_id == episode_id)
                self._load_record(record)

        @self.frame_gui.on_update
        def _(_) -> None:
            with self.lock:
                self._render_frame(self._clamped_frame())

        @self.stream_gui.on_update
        def _(_) -> None:
            with self.lock:
                self.stream = str(self.stream_gui.value).lower()
                self.last_ik_q20 = None
                self._update_trajectories()
                self._render_frame(self._clamped_frame())

        @self.previous_gui.on_click
        def _(_) -> None:
            self._set_frame(self._clamped_frame() - 1)

        @self.next_gui.on_click
        def _(_) -> None:
            self._set_frame(self._clamped_frame() + 1)

        @self.worst_gui.on_click
        def _(_) -> None:
            self._jump_to_worst()

        @self.reset_camera_gui.on_click
        def _(_) -> None:
            self._reset_camera()

        @self.camera_view_gui.on_update
        def _(_) -> None:
            self._set_camera_view(str(self.camera_view_gui.value))

        @self.trajectory_gui.on_update
        def _(_) -> None:
            self._update_trajectory_tail(self._clamped_frame())
            self._apply_visibility()

        for checkbox in (
            self.show_source_gui,
            self.show_ghost_gui,
            self.show_targets_gui,
            self.show_fk_gui,
            self.show_trajectory_gui,
        ):
            checkbox.on_update(lambda _: self._apply_visibility())

    def _load_record(self, record: EpisodeRecord) -> None:
        loaded = load_episode(record)
        self.loaded = loaded
        self.last_ik_q20 = None
        self.last_ik_valid = False
        self.frame_gui.value = 0
        self._update_trajectories()
        self._render_frame(0)

    def _clamped_frame(self) -> int:
        if self.loaded is None:
            return 0
        return int(np.clip(int(self.frame_gui.value), 0, self.loaded.record.frame_count - 1))

    def _set_frame(self, frame: int) -> None:
        with self.lock:
            if self.loaded is None:
                return
            frame = int(np.clip(frame, 0, self.loaded.record.frame_count - 1))
            if int(self.frame_gui.value) == frame:
                self._render_frame(frame)
            else:
                self.frame_gui.value = frame

    def _stream_values(self) -> tuple[np.ndarray, np.ndarray]:
        if self.loaded is None:
            raise RuntimeError("no episode loaded")
        if self.stream == "state":
            return self.loaded.state, self.loaded.joints_state
        return self.loaded.action, self.loaded.joints_action

    def _update_trajectories(self) -> None:
        if self.loaded is None:
            return
        ee_values, _ = self._stream_values()
        points = {label: [] for label in kin.LINK_LABELS}
        for ee34 in ee_values:
            chassis = ee34[C.EE34_CHASSIS]
            targets = kin.ee34_to_target_matrices(ee34)
            for label in kin.LINK_LABELS:
                points[label].append((kin.chassis_xyyaw_to_matrix(chassis) @ targets[label])[:3, 3])
        for label in kin.LINK_LABELS:
            self.trajectory_points[label] = np.asarray(points[label], dtype=np.float32)
        self._apply_visibility()

    def _update_trajectory_tail(self, frame: int) -> None:
        label = str(self.trajectory_gui.value).lower()
        if label == "none":
            return
        start = max(0, frame - self.args.trajectory_frames + 1)
        points = self.trajectory_points[label][start : frame + 1]
        if points.shape[0] == 1:
            points = np.repeat(points, 2, axis=0)
        self.trajectory.positions = points

    def _apply_visibility(self) -> None:
        self.source_root.visible = bool(self.show_source_gui.value)
        ghost_visible = bool(self.show_ghost_gui.value)
        self.ghost_valid_root.visible = ghost_visible and self.last_ik_valid
        self.ghost_invalid_root.visible = ghost_visible and not self.last_ik_valid
        for handle in self.target_frames.values():
            handle.visible = bool(self.show_targets_gui.value)
        for handle in self.fk_frames.values():
            handle.visible = bool(self.show_fk_gui.value)
        self.trajectory.visible = bool(self.show_trajectory_gui.value) and str(
            self.trajectory_gui.value
        ).lower() != "none"

    def _render_frame(self, frame: int) -> None:
        if self.loaded is None:
            return
        ee_values, joints_values = self._stream_values()
        ee34 = np.asarray(ee_values[frame], dtype=np.float64)
        joints25 = np.asarray(joints_values[frame], dtype=np.float64)
        source_q20 = kin.joints25_to_urdf20(joints25)
        targets = kin.ee34_to_target_matrices(ee34)
        fk = self.model.fk(source_q20)
        ik_result = self.model.solve_ik(
            targets,
            source_q20,
            head2=ee34[C.EE34_HEAD],
            alternate_q20=self.last_ik_q20,
            max_nfev=self.args.ik_max_nfev,
        )
        if ik_result.valid:
            self.last_ik_q20 = ik_result.q20.copy()
        self.last_ik_valid = ik_result.valid
        self._update_trajectory_tail(frame)
        source_skeleton = self.model.skeleton_segments(source_q20)
        ghost_skeleton = self.model.skeleton_segments(ik_result.q20)

        source_wxyz, source_position = _root_pose(joints25[C.J_CHASSIS], self.model.t_chassis_urdf_root)
        ghost_wxyz, ghost_position = _root_pose(ee34[C.EE34_CHASSIS], self.model.t_chassis_urdf_root)
        with self.server.atomic():
            self.source_root.wxyz = source_wxyz
            self.source_root.position = source_position
            self.source_urdf.update_cfg(source_q20)
            self.source_skeleton.points = source_skeleton
            for root in (self.ghost_valid_root, self.ghost_invalid_root):
                root.wxyz = ghost_wxyz
                root.position = ghost_position
            self.ghost_valid_skeleton.points = ghost_skeleton
            self.ghost_invalid_skeleton.points = ghost_skeleton
            self.ghost_valid_root.visible = bool(self.show_ghost_gui.value) and ik_result.valid
            self.ghost_invalid_root.visible = bool(self.show_ghost_gui.value) and not ik_result.valid
            for label in kin.LINK_LABELS:
                target_wxyz, target_position = _world_pose(ee34[C.EE34_CHASSIS], targets[label])
                fk_wxyz, fk_position = _world_pose(joints25[C.J_CHASSIS], fk[label])
                self.target_frames[label].wxyz = target_wxyz
                self.target_frames[label].position = target_position
                self.fk_frames[label].wxyz = fk_wxyz
                self.fk_frames[label].position = fk_position
        self._apply_visibility()

        source_errors = {label: kin.se3_error(fk[label], targets[label]) for label in kin.LINK_LABELS}
        rows = "\n".join(
            f"| {label} | {source_errors[label].position_mm:.3f} | "
            f"{source_errors[label].rotation_deg:.3f} | {ik_result.errors[label].position_mm:.3f} | "
            f"{ik_result.errors[label].rotation_deg:.3f} |"
            for label in kin.LINK_LABELS
        )
        self.status_gui.content = (
            f"**{self.loaded.record.split}/{self.loaded.record.episode_id}** — frame `{frame}` — "
            f"`{self.stream}`\n\n"
            "| link | FK pos mm | FK rot deg | IK pos mm | IK rot deg |\n"
            "|---|---:|---:|---:|---:|\n"
            f"{rows}\n\n"
            f"IK: **{'valid' if ik_result.valid else 'INVALID'}**, solver success={ik_result.success}, "
            f"nfev={ik_result.nfev}, cost={ik_result.cost:.4g}, "
            f"max Δq={ik_result.max_joint_delta_rad:.4f} rad  \n"
            f"gripper L/R: `{ee34[C.EE34_LEFT_GRIPPER]:.3f}` / `{ee34[C.EE34_RIGHT_GRIPPER]:.3f}`"
        )
        for label, handle in self.image_gui.items():
            handle.image = _decode_image(self.loaded.images[label][frame], self.loaded.record.parquet_file)

    def _jump_to_worst(self) -> None:
        if self.loaded is None or self.report is None:
            self.status_gui.content = "No compatible kinematics report was found."
            return
        split_metrics = self.report.get("metrics", {}).get(self.loaded.record.split, {})
        stream_metrics = split_metrics.get(self.stream, {})
        candidates: list[tuple[float, dict[str, Any]]] = []
        for label in kin.LINK_LABELS:
            values = stream_metrics.get(label, {})
            for key, scale in (("worst_position", 1.0 / 5.0), ("worst_rotation", 1.0)):
                item = values.get(key)
                if item is not None:
                    candidates.append((float(item["value"]) * scale, item))
        if not candidates:
            self.status_gui.content = "The report has no worst-frame entry for this split/stream."
            return
        _, worst = max(candidates, key=lambda item: item[0])
        episode_id = str(worst["episode_id"])
        if episode_id != self.loaded.record.episode_id:
            record = next(
                record
                for record in self.catalog[self.loaded.record.split]
                if record.episode_id == episode_id
            )
            self.episode_gui.value = episode_id
            self._load_record(record)
        self._set_frame(int(worst["frame"]))

    def _reset_camera(self, client: viser.ClientHandle | None = None) -> None:
        self._set_camera_view("Side", client)

    def _set_camera_view(self, view: str, client: viser.ClientHandle | None = None) -> None:
        poses = {
            "Side": ((0.0, -3.0, 1.35), (0.1, 0.0, 0.85)),
            "Diagonal": ((2.8, -2.8, 2.0), (0.0, 0.0, 0.9)),
            "Front": ((3.2, 0.0, 1.35), (0.0, 0.0, 0.85)),
            "Opposite side": ((0.0, 3.0, 1.35), (0.1, 0.0, 0.85)),
        }
        if view not in poses:
            raise ValueError(f"unknown camera view: {view}")
        position, look_at = poses[view]
        clients = (client,) if client is not None else tuple(self.server.get_clients().values())
        for current in clients:
            current.camera.position = position
            current.camera.look_at = look_at
            current.camera.up_direction = (0.0, 0.0, 1.0)

    def smoke_test(self) -> None:
        if self.loaded is None:
            raise RuntimeError("no episode loaded")
        for frame in (0, self.loaded.record.frame_count // 2, self.loaded.record.frame_count - 1):
            self._render_frame(frame)
        self.server.flush()
        self.server.stop()

    def run(self) -> None:
        speed_map = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0}
        try:
            while True:
                time.sleep(0.005)
                if not self.playing_gui.value or self.loaded is None:
                    self.last_tick = time.monotonic()
                    continue
                now = time.monotonic()
                interval = 1.0 / (C.FPS * speed_map[str(self.speed_gui.value)])
                if now - self.last_tick < interval:
                    continue
                self.last_tick = now
                current = self._clamped_frame()
                next_frame = 0 if current + 1 >= self.loaded.record.frame_count else current + 1
                self._set_frame(next_frame)
        except KeyboardInterrupt:
            self.server.stop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=C.OUTPUT_ROOT)
    parser.add_argument("--split", choices=tuple(SPLIT_REPO_IDS), default="validation")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--stream", choices=tuple(STREAM_COLUMNS), default="state")
    parser.add_argument("--urdf", type=Path, default=kin.DEFAULT_URDF_PATH)
    parser.add_argument("--torso-config", type=Path, default=kin.DEFAULT_TORSO_CONFIG_PATH)
    parser.add_argument("--report", type=Path, default=C.REPORT_ROOT / "ee34_kinematics_validation.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ik-max-nfev", type=int, default=100)
    parser.add_argument("--trajectory-frames", type=int, default=60)
    parser.add_argument("--trajectory", choices=(*kin.LINK_LABELS, "none"), default="left")
    parser.add_argument("--start-frame", type=int, help="Initial frame after loading the episode")
    parser.add_argument(
        "--start-at-worst",
        action="store_true",
        help="Jump to the highest normalized FK error in the audit report after startup",
    )
    parser.add_argument("--no-meshes", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("the viewer must bind to loopback only")
    if args.trajectory_frames < 2:
        parser.error("--trajectory-frames must be at least 2")
    return args


def main() -> int:
    args = _parse_args()
    viewer = KinematicsViewer(args)
    if args.smoke_test:
        viewer.smoke_test()
    else:
        viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
