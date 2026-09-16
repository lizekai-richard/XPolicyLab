"""CPU tests for sana_wam_min/eef.py: URDF forward kinematics against recorded RoboDojo poses, the E remap, rot6d,
the anchor-relative reconstruction and the conversion into RoboDojo ``ee`` action dicts."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_SANA_WAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SANA_WAM_DIR not in sys.path:
    sys.path.insert(0, _SANA_WAM_DIR)

from sana_wam_min import eef  # noqa: E402

# (side, joint state [6], recorded state/<side>_ee_poses row [x y z qw qx qy qz]) from the public RoboDojo corpus,
# insert_key episode 0 at t = 0 / 60 / 120: the evaluator's link6 pose in the env-relative world frame.
RECORDED = [
    ("left", [0.0, -0.0, 0.0, 0.0, -0.0, 0.0], [-0.299529, -0.3523, 0.9215, 0.707, -0.0, -0.0, 0.707214]),
    ("left", [-0.020568, 0.091248, 0.040309, 0.048234, 0.063774, -0.073363], [-0.295603, -0.348796, 0.932747, 0.735655, -0.027975, -0.023885, 0.676357]),
    ("left", [-0.572899, 1.975945, 0.924668, 1.02197, 1.214516, -1.567636], [-0.075155, -0.152211, 0.920974, 0.694756, -0.711069, 0.080852, -0.071812]),
    ("right", [0.0, -0.0, 0.0, 0.0, -0.0, 0.0], [0.300471, -0.3523, 0.9215, 0.707, -0.0, -0.0, 0.707214]),
    ("right", [0.6532, 1.847324, 1.821942, -1.570069, -0.02928, 2.431581], [0.095426, -0.183412, 1.072576, 0.701634, 0.059825, 0.704707, -0.086712]),
    ("right", [0.471568, 1.711331, 1.656437, -1.545695, -0.024348, 2.249674], [0.162634, -0.181271, 1.072722, 0.701689, 0.059652, 0.704669, -0.086692]),
]


@pytest.fixture(scope="module")
def kin():
    return eef.ArxX5Kinematics()


@pytest.fixture(scope="module")
def roots():
    return eef.root_transforms()


def _joint_row(side, joints):
    row = np.zeros(80, dtype=np.float32)
    row[eef.ARM_FIELDS[0][1] if side == "left" else eef.ARM_FIELDS[1][1]] = np.asarray(joints, dtype=np.float32)
    return row


@pytest.mark.parametrize("side,joints,recorded", RECORDED)
def test_fk_with_root_pose_reproduces_the_recorded_world_link6_pose(kin, roots, side, joints, recorded):
    world = roots[side] @ kin.link6_in_base(joints)
    recorded_t = eef.pose_to_transform(recorded)
    assert np.linalg.norm(world[:3, 3] - recorded_t[:3, 3]) < 5e-4            # 0.5 mm
    assert eef.rotation_angle_deg(world[:3, :3], recorded_t[:3, :3]) < 0.1     # 0.1 deg (recorded quats are rounded)


@pytest.mark.parametrize("side,joints,recorded", RECORDED)
def test_state_slots_round_trip_to_the_world_pose(kin, roots, side, joints, recorded):
    row = np.zeros(80, dtype=np.float32)
    row[0:6] = joints if side == "left" else 0.0
    row[29:35] = joints if side == "right" else 0.0
    state, mask = eef.fill_eef_state_slots(row, eef.eef_state_slot_mask(True), kin)
    assert mask[list(eef.EEF_SLOTS)].all() and mask.sum() == 32
    back = eef.world_link6_pose_from_slots(state, side, roots[side])
    recorded_t = eef.pose_to_transform(recorded)
    assert np.linalg.norm(back[:3] - recorded_t[:3, 3]) < 5e-4
    assert eef.rotation_angle_deg(eef.quat_wxyz_to_matrix(back[3:]), recorded_t[:3, :3]) < 0.1


def test_e_remap_is_rotation_only_and_invertible():
    assert np.allclose(eef.LINK6_FROM_E[:3, 3], 0) and np.allclose(eef.LINK6_FROM_E @ eef.E_FROM_LINK6, np.eye(4))
    r = eef.LINK6_FROM_E[:3, :3]
    assert np.allclose(r.T @ r, np.eye(3)) and np.isclose(np.linalg.det(r), 1.0)
    # +X approach kept, E's +Y maps to link6 -Z... i.e. the Sana _STANDARD_E_ROWS rows verbatim
    assert np.allclose(eef.LINK6_FROM_E[:3, :3], [[1, 0, 0], [0, 0, 1], [0, -1, 0]])


def test_rot6d_round_trip_and_column_convention():
    rot = eef.quat_wxyz_to_matrix([0.9, 0.1, -0.3, 0.2])
    r6 = eef.matrix_to_rot6d(rot)
    np.testing.assert_allclose(r6[:3], rot[:, 0]); np.testing.assert_allclose(r6[3:], rot[:, 1])
    np.testing.assert_allclose(eef.rot6d_to_matrix(r6), rot, atol=1e-12)
    assert eef.rotation_angle_deg(rot, eef.rot6d_to_matrix(r6 + 1e-3)) < 0.2   # Gram-Schmidt tolerates noise


def test_quaternion_round_trip_prefers_positive_w():
    for q in ([0.707, 0.0, 0.0, 0.707], [-0.5, 0.5, 0.5, 0.5], [0.0, 1.0, 0.0, 0.0]):
        back = eef.matrix_to_quat_wxyz(eef.quat_wxyz_to_matrix(q))
        assert back[0] >= 0
        assert np.allclose(eef.quat_wxyz_to_matrix(back), eef.quat_wxyz_to_matrix(q), atol=1e-9)


def test_reconstruct_absolute_eef_applies_anchor_relative_deltas(kin):
    anchor, mask = eef.fill_eef_state_slots(_joint_row("left", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) + _joint_row("right", [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]), eef.eef_state_slot_mask(True), kin)
    delta_r = eef.quat_wxyz_to_matrix([0.99, 0.0, 0.1, 0.0])
    rows = np.zeros((2, 80), dtype=np.float32)
    for _, _, pos, rot, _ in eef.ARM_FIELDS:
        rows[:, pos] = [0.01, -0.02, 0.03]
        rows[0, rot] = eef.matrix_to_rot6d(np.eye(3))
        rows[1, rot] = eef.matrix_to_rot6d(delta_r)
    out = eef.reconstruct_absolute_eef(torch.from_numpy(rows), np.tile(mask, (2, 1)), anchor, mask).numpy()
    for _, _, pos, rot, _ in eef.ARM_FIELDS:
        np.testing.assert_allclose(out[:, pos], np.tile(anchor[pos] + [0.01, -0.02, 0.03], (2, 1)), atol=1e-6)
        r_anchor = eef.rot6d_to_matrix(anchor[rot])
        np.testing.assert_allclose(eef.rot6d_to_matrix(out[0, rot]), r_anchor, atol=1e-6)               # identity delta
        np.testing.assert_allclose(eef.rot6d_to_matrix(out[1, rot]), delta_r @ r_anchor, atol=1e-6)     # R_t = dR @ R_anchor
    joints_untouched = [i for i in range(80) if i not in eef.EEF_SLOTS]
    np.testing.assert_array_equal(out[:, joints_untouched], rows[:, joints_untouched])


def test_reconstruct_refuses_an_anchor_without_eef_slots(kin):
    rows = np.zeros((1, 80), dtype=np.float32); mask = np.tile(eef.eef_state_slot_mask(True), (1, 1))
    with pytest.raises(ValueError, match="anchor state has no valid EEF slots"):
        eef.reconstruct_absolute_eef(rows, mask, np.zeros(80, np.float32), eef.eef_state_slot_mask(False))


def test_native_ee_actions_carry_world_link6_poses_and_gripper_openings(kin, roots):
    row, mask = eef.fill_eef_state_slots(_joint_row("left", [0.0] * 6) + _joint_row("right", [0.0] * 6), eef.eef_state_slot_mask(True), kin)
    row[16], row[45] = 0.25, 1.0
    actions = eef.native_ee_actions_from_action80(np.stack([row, row]), roots)
    assert len(actions) == 2 and sorted(actions[0]) == ["left_ee_joint_state", "left_ee_pose", "right_ee_joint_state", "right_ee_pose"]
    np.testing.assert_allclose(actions[0]["left_ee_pose"][:3], [-0.299529, -0.3523, 0.9215], atol=5e-4)   # rest pose of the corpus
    np.testing.assert_allclose(actions[0]["right_ee_pose"][:3], [0.300471, -0.3523, 0.9215], atol=5e-4)
    assert actions[0]["left_ee_pose"].dtype == np.float32 and actions[0]["left_ee_pose"].shape == (7,)
    np.testing.assert_allclose(actions[0]["left_ee_joint_state"], [0.75]); np.testing.assert_allclose(actions[0]["right_ee_joint_state"], [0.0])


def test_observed_discrepancy_is_zero_for_consistent_observations(kin, roots):
    joints = RECORDED[2][1]
    row, _ = eef.fill_eef_state_slots(_joint_row("left", joints) + _joint_row("right", [0.0] * 6), eef.eef_state_slot_mask(True), kin)
    obs_state = {"left_ee_pose": np.asarray(RECORDED[2][2], dtype=np.float32), "right_ee_pose": np.asarray(RECORDED[3][2], dtype=np.float32)}
    gaps = eef.observed_link6_discrepancy(obs_state, row, roots)
    assert gaps["left"][0] < 5e-4 and gaps["left"][1] < 0.1 and gaps["right"][0] < 5e-4
    obs_state["left_ee_pose"][0] += 0.02
    assert abs(eef.observed_link6_discrepancy(obs_state, row, roots)["left"][0] - 0.02) < 1e-3


def test_root_transforms_validate_sides():
    with pytest.raises(ValueError, match="missing side"):
        eef.root_transforms({"left": [0, 0, 0, 1, 0, 0, 0]})
