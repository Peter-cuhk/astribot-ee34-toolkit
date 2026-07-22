"""Frozen contract constants for Astribot S1 absolute EE SO3 34D (route B)."""

from __future__ import annotations

import os
from pathlib import Path

CONTRACT_ID = "astribot_s1_ee34_contract_v1"
CONTRACT_VERSION = "1.0.0"

SOURCE_ROOT = Path(os.environ.get("ASTRIBOT_HDF5_ROOT", "data/astribot/hdf5"))
OUTPUT_ROOT = Path(os.environ.get("HF_LEROBOT_HOME", "data/lerobot"))
REPORT_ROOT = Path(os.environ.get("ASTRIBOT_REPORT_DIR", "reports"))
TRAIN_REPO_ID = "astribot/ee34_pick_tomato_v1_train"
VAL_REPO_ID = "astribot/ee34_pick_tomato_v1_val"
ASSET_ID = "astribot_ee34_pick_tomato_v1"
CONFIG_NAME = "pi05_astribot_ee34_pick_tomato_v1"

DEFAULT_PROMPT = "pick the tomato"
LANGUAGE_SOURCE = "fixed_v1"
FPS = 30
ACTION_HORIZON = 30

# Source dimensions
JOINT_DIM = 25
MERGE_POSE_DIM = 37
HYBRID_QUAT_DIM = 28  # torso7+left7+Lg1+right7+Rg1+head2+chassis3
EE34_DIM = 34
MODEL_ACTION_DIM = 34

# merge_pose 37D layout
MP_CHASSIS = slice(0, 7)
MP_TORSO = slice(7, 14)
MP_LEFT = slice(14, 21)
MP_LEFT_GRIPPER = 21
MP_RIGHT = slice(22, 29)
MP_RIGHT_GRIPPER = 29
MP_HEAD = slice(30, 37)

# joints 25D layout
J_CHASSIS = slice(0, 3)
J_TORSO = slice(3, 7)
J_LEFT_ARM = slice(7, 14)
J_LEFT_GRIPPER = 14
J_RIGHT_ARM = slice(15, 22)
J_RIGHT_GRIPPER = 22
J_HEAD = slice(23, 25)

# EE34 layout / Infra execution dim_list
EE34_DIM_LIST = (9, 9, 1, 9, 1, 2, 3)
EE34_TORSO = slice(0, 9)
EE34_LEFT = slice(9, 18)
EE34_LEFT_GRIPPER = 18
EE34_RIGHT = slice(19, 28)
EE34_RIGHT_GRIPPER = 28
EE34_HEAD = slice(29, 31)
EE34_CHASSIS = slice(31, 34)

# Camera keys
SOURCE_CAMERAS = ("head", "left", "right")
LEROBOT_CAMERAS = ("head", "left_wrist", "right_wrist")
CAMERA_MAP = {
    "head": "observation.images.head",
    "left": "observation.images.left_wrist",
    "right": "observation.images.right_wrist",
}
IMAGE_SHAPES = {
    "head": (720, 1280, 3),
    "left": (360, 640, 3),
    "right": (360, 640, 3),
}

# Split
SPLIT_SALT = "astribot_ee34_pick_tomato_v1_val_v1"
SPLIT_ALGORITHM = "sha256-sorted-episode-id/1"
EXPECTED_EPISODES = 264
TARGET_VAL_EPISODES = 5

EE34_NAMES = (
    *(f"torso_so3_{i}" for i in range(9)),
    *(f"left_so3_{i}" for i in range(9)),
    "left_gripper",
    *(f"right_so3_{i}" for i in range(9)),
    "right_gripper",
    "head_joint_0",
    "head_joint_1",
    "chassis_x",
    "chassis_y",
    "chassis_z_rot",
)

CHASSIS_IDENTITY_XYZ_TOL = 1e-6
CHASSIS_IDENTITY_QUAT_TOL = 1e-6
QUAT_NORM_TOL = 1e-3
SO3_ORTH_TOL = 1e-3
ROUNDTRIP_POS_TOL = 1e-5
ROUNDTRIP_ROT_TOL = 1e-5
