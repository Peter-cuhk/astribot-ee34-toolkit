# Astribot S1 EE34 Contract v1

**Contract ID:** `astribot_s1_ee34_contract_v1`

**Version:** `1.0.0`
**Purpose:** Absolute SO3 EE 34D state/action aligned with Astribot_infra default EE inference.

## Source dataset

| Item | Value |
|------|-------|
| Path | `$ASTRIBOT_HDF5_ROOT` (default: `data/astribot/hdf5`) |
| Episodes | 264 |
| FPS | 30 |
| Keys | `joints_dict`, `poses_dict`, `command_poses_dict`, `images_dict`, `time` |

### Source dims

- `joints_*`: `(T, 25)` absolute joint positions
- `*_poses_dict/merge_pose`: `(T, 37)` = chassis7+torso7+left7+Lg1+right7+Rg1+head7
- pose7 = `[x,y,z,qx,qy,qz,qw]` (ROS xyzw)
- Images: concatenated JPEG bytes + `rgb_size` for `head` (1280×720), `left`/`right` (640×360)

### Joint 25D index

```text
0:3 chassis | 3:7 torso | 7:14 left_arm | 14 left_gripper
15:22 right_arm | 22 right_gripper | 23:25 head
```

## Target EE34

```text
[0:9] torso_so3 | [9:18] left_so3 | [18] Lg | [19:28] right_so3
[28] Rg | [29:31] head_joints | [31:34] chassis_joints
```

Semantics: **absolute** commands/states (no delta). Infra wire `dim_list=[9,9,1,9,1,2,3]`.

### Mapping (adopted)

| EE34 | Source | Processing |
|------|--------|------------|
| `[0:9]` torso | `merge_pose[7:14]` | world→chassis, quat→rotation-6D |
| `[9:18]` left | `merge_pose[14:21]` | same |
| `[18]` Lg | `joints[:,14]` | direct |
| `[19:28]` right | `merge_pose[22:29]` | same |
| `[28]` Rg | `joints[:,22]` | direct |
| `[29:31]` head | `joints[:,23:25]` | direct; **not** head7 |
| `[31:34]` chassis | `joints[:,0:3]` | direct; **not** chassis7 |

state ← `poses_dict` + `joints_position_state`

action ← `command_poses_dict` + `joints_position_command`

## Three rules

1. Online frame is **chassis**. Apply `T_c_l = inv(T_w_c) @ T_w_l`. On this tomato dump, chassis pose is identity → transform is identity, but conversion still asserts.
2. head/chassis come from **joints**, never merge_pose 7D.
3. Pair state/action dicts correctly (poses vs command_poses).

## LeRobot export (phase 1)

- train: `astribot/ee34_pick_tomato_v1_train`
- val: `astribot/ee34_pick_tomato_v1_val`
- under `$HF_LEROBOT_HOME` (default: `data/lerobot`)
- cameras: `observation.images.{head,left_wrist,right_wrist}`
- task: fixed `"pick the tomato"` (`language_source=fixed_v1`)
- no mean/std normalization at export time

## Phase 1.5 offline kinematics audit

The independent audit uses `astribot_whole_body_with_head.urdf` for FK and
`astribot_torso.yaml: transform.weld_to_base_pose` for the chassis-to-URDF-root
transform. The current S1 configuration contributes a required `+0.097 m` Z
translation. It is loaded from YAML rather than duplicated in code.

```bash
uv run astribot-ee34-validate-kinematics --all-frames
uv run astribot-ee34-visualize \
  --split validation --episode-id episode_269 --host 127.0.0.1 --port 8080
```

Conversion parity is a hard `rtol=0, atol=1e-6` gate. FK consistency warns at
P95 `5 mm / 1 deg` and fails at P95 `20 mm / 3 deg` or max `50 mm / 10 deg`.
IK is diagnostic only: its joint solution is not required to equal the recorded
redundant-arm configuration.
