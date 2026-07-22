from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from astribot_ee34 import contract as C  # noqa: N812
from astribot_ee34 import kinematics as kin
from astribot_ee34 import pack


@pytest.fixture(scope="module")
def model() -> kin.AstribotKinematics:
    if not kin.DEFAULT_URDF_PATH.is_file() or not kin.DEFAULT_TORSO_CONFIG_PATH.is_file():
        pytest.skip("set ASTRIBOT_S1_CONFIG_ROOT to an Astribot S1 SDK config directory")
    return kin.AstribotKinematics()


def test_joint_mapping_and_names(model: kin.AstribotKinematics) -> None:
    joints = np.arange(C.JOINT_DIM, dtype=np.float64)
    expected = np.r_[joints[3:7], joints[23:25], joints[7:14], joints[15:22]]
    np.testing.assert_array_equal(kin.joints25_to_urdf20(joints), expected)
    assert model.joint_names == kin.EXPECTED_JOINT_NAMES


def test_weld_transform_comes_from_yaml(model: kin.AstribotKinematics) -> None:
    np.testing.assert_allclose(model.weld_pose6, [0.0, 0.0, 0.097, 0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(model.t_chassis_urdf_root[:3, 3], [0.0, 0.0, 0.097], atol=1e-12)
    np.testing.assert_allclose(model.t_chassis_urdf_root[:3, :3], np.eye(3), atol=1e-12)
    q20 = (model.lower20 + model.upper20) / 2.0
    corrected = model.fk(q20)
    uncorrected = model.fk(q20, apply_weld=False)
    for label in kin.LINK_LABELS:
        assert kin.se3_error(uncorrected[label], corrected[label]).position_mm == pytest.approx(97.0)


def test_chassis_xyyaw_and_se3_error() -> None:
    transform = kin.chassis_xyyaw_to_matrix(np.asarray([1.0, -2.0, np.pi / 2.0]))
    np.testing.assert_allclose(transform[:2, 3], [1.0, -2.0], atol=1e-12)
    np.testing.assert_allclose(
        transform[:3, :3],
        Rotation.from_euler("z", np.pi / 2.0).as_matrix(),
        atol=1e-12,
    )
    target = transform.copy()
    target[0, 3] += 0.003
    target[:3, :3] = target[:3, :3] @ Rotation.from_euler("x", 2.0, degrees=True).as_matrix()
    error = kin.se3_error(transform, target)
    assert error.position_mm == pytest.approx(3.0)
    assert error.rotation_deg == pytest.approx(2.0)


def test_fk_ee34_decode_and_ik(model: kin.AstribotKinematics) -> None:
    q20 = (model.lower20 + model.upper20) / 2.0
    fk = model.fk(q20)
    ee34 = np.zeros((C.EE34_DIM,), dtype=np.float64)
    for label, output_slice in (
        ("torso", C.EE34_TORSO),
        ("left", C.EE34_LEFT),
        ("right", C.EE34_RIGHT),
    ):
        pose7 = pack.matrix_to_pose7(fk[label])
        ee34[output_slice] = pack.quat7_to_so39(pose7)
    ee34[C.EE34_HEAD] = q20[4:6]
    decoded = kin.ee34_to_target_matrices(ee34)
    for label in kin.LINK_LABELS:
        error = kin.se3_error(fk[label], decoded[label])
        assert error.position_m < 1e-12
        assert error.rotation_rad < 1e-12

    result = model.solve_ik(decoded, q20, head2=ee34[C.EE34_HEAD], max_nfev=30)
    assert result.success
    assert result.valid
    for error in result.errors.values():
        assert error.position_m < 1e-9
        assert error.rotation_rad < 1e-9


def test_ik_joint_redundancy_is_not_an_equality_gate(model: kin.AstribotKinematics) -> None:
    q20 = (model.lower20 + model.upper20) / 2.0
    targets = model.fk(q20)
    seed = q20.copy()
    seed[6] += 0.05
    result = model.solve_ik(targets, seed, head2=q20[4:6], max_nfev=100)
    assert result.valid
    assert all(error.position_mm <= 5.0 and error.rotation_deg <= 1.0 for error in result.errors.values())


def test_left_right_swap_is_a_gross_kinematics_failure(model: kin.AstribotKinematics) -> None:
    q20 = (model.lower20 + model.upper20) / 2.0
    fk = model.fk(q20)
    swapped = {"torso": fk["torso"], "left": fk["right"], "right": fk["left"]}
    errors = {label: kin.se3_error(fk[label], swapped[label]) for label in kin.LINK_LABELS}
    assert errors["left"].position_mm > 50.0
    assert errors["right"].position_mm > 50.0
