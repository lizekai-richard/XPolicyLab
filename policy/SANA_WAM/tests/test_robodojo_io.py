"""CPU tests for sana_wam_min.robodojo_io (RoboDojo observation/action codecs)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

# policy/SANA_WAM (plain ``sana_wam_min`` imports) and the XPolicyLab parent (``XPolicyLab.policy.SANA_WAM`` imports).
ADAPTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_XPOLICYLAB_PARENT = os.path.abspath(os.path.join(ADAPTER_DIR, "..", "..", ".."))
for _p in (ADAPTER_DIR, _XPOLICYLAB_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sana_wam_min import robodojo_io  # noqa: E402

ACTIVE_SLOTS = [0, 1, 2, 3, 4, 5, 16, 29, 30, 31, 32, 33, 34, 45]


def test_package_import_path():
    from XPolicyLab.policy.SANA_WAM.sana_wam_min import robodojo_io as via_pkg

    assert via_pkg.VIEW_SLOT_IDS == (0, 2, 3)


def test_constants():
    assert robodojo_io.VIEW_SLOT_IDS == (0, 2, 3)
    assert robodojo_io.ROBODOJO_VIEW_ORDER == ("cam_head", "cam_left_wrist", "cam_right_wrist")
    assert robodojo_io.JOINT_ONLY_ACTIVE_SLOTS == tuple(ACTIVE_SLOTS)
    assert robodojo_io.ROBODOJO_MODEL_FPS_HZ == 25.0
    assert robodojo_io.target_offsets_ns(24)[:3] == (40_000_000, 80_000_000, 120_000_000)
    assert robodojo_io.target_offsets_ns(24)[-1] == 960_000_000
    assert robodojo_io.ROBODOJO_FROZEN_INTRINSICS[0, 0] == pytest.approx(288.13254096884566)
    assert robodojo_io.ROBODOJO_NATIVE_ACTION_KEYS == (
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
    )


def test_view_order_matches_text_module():
    from sana_wam_min import text

    assert text.ROBODOJO_VIEW_ORDER == robodojo_io.ROBODOJO_VIEW_ORDER


def test_joint_slot_mask():
    mask = robodojo_io.joint_slot_mask(6, 6)
    assert mask.dtype == np.bool_ and mask.shape == (80,)
    assert np.flatnonzero(mask).tolist() == ACTIVE_SLOTS
    assert np.array_equal(robodojo_io.joint_slot_mask(), mask)
    with pytest.raises(ValueError):
        robodojo_io.joint_slot_mask(7, 6)


def test_state80_from_obs():
    left = np.arange(6, dtype=np.float64) * 0.1
    right = -np.arange(6, dtype=np.float64) * 0.2
    obs_state = {
        "left_arm_joint_state": left,
        "right_arm_joint_state": right.tolist(),
        "left_ee_joint_state": np.array([0.25], dtype=np.float32),
        "right_ee_joint_state": [1.3],  # overshoot above 1 -> clipped before inversion
        "left_ee_pose": np.ones(7),  # privileged, never read
    }
    state80, mask = robodojo_io.state80_from_obs(obs_state)
    assert state80.dtype == np.float32 and state80.shape == (80,)
    assert mask.dtype == np.bool_ and np.array_equal(mask, robodojo_io.joint_slot_mask(6, 6))
    np.testing.assert_array_equal(state80[0:6], left.astype(np.float32))
    np.testing.assert_array_equal(state80[29:35], right.astype(np.float32))
    assert state80[16] == np.float32(0.75)
    assert state80[45] == np.float32(0.0)
    assert np.all(state80[~mask] == 0.0)
    assert mask.flags.writeable


def test_state80_from_obs_rejects_bad_inputs():
    good = {
        "left_arm_joint_state": np.zeros(6),
        "right_arm_joint_state": np.zeros(6),
        "left_ee_joint_state": [1.0],
        "right_ee_joint_state": [1.0],
    }
    bad_dim = dict(good, left_arm_joint_state=np.zeros(7))
    with pytest.raises(ValueError):
        robodojo_io.state80_from_obs(bad_dim)
    bad_nan = dict(good, right_ee_joint_state=[float("nan")])
    with pytest.raises(ValueError):
        robodojo_io.state80_from_obs(bad_nan)
    bad_inf = dict(good, right_arm_joint_state=np.array([0, 0, 0, 0, 0, np.inf]))
    with pytest.raises(ValueError):
        robodojo_io.state80_from_obs(bad_inf)


def test_upstream_actions_from_action80():
    K = 24
    action = np.zeros((K, 80), dtype=np.float32)
    action[:, 0:6] = np.arange(6, dtype=np.float32)[None, :] + np.arange(K, dtype=np.float32)[:, None]
    action[:, 29:35] = -1.0
    action[:, 16] = np.linspace(0.0, 1.0, K, dtype=np.float32)
    action[:, 45] = 0.3
    dicts = robodojo_io.upstream_actions_from_action80(action)
    assert len(dicts) == K
    for k, d in enumerate(dicts):
        assert tuple(d.keys()) == robodojo_io.ROBODOJO_NATIVE_ACTION_KEYS
        assert d["left_arm_joint_state"].shape == (6,) and d["left_arm_joint_state"].dtype == np.float32
        assert d["right_arm_joint_state"].shape == (6,) and d["right_arm_joint_state"].dtype == np.float32
        assert d["left_ee_joint_state"].shape == (1,) and d["left_ee_joint_state"].dtype == np.float32
        assert d["right_ee_joint_state"].shape == (1,) and d["right_ee_joint_state"].dtype == np.float32
        np.testing.assert_array_equal(d["left_arm_joint_state"], action[k, 0:6])
        np.testing.assert_array_equal(d["right_arm_joint_state"], action[k, 29:35])
        assert d["left_ee_joint_state"][0] == pytest.approx(1.0 - action[k, 16], abs=1e-7)
        assert d["right_ee_joint_state"][0] == pytest.approx(0.7, abs=1e-7)
        assert d["left_arm_joint_state"].flags.c_contiguous
    # torch input and out-of-range closedness (defensive clip on the opening).
    t = torch.zeros(2, 80)
    t[:, 16] = 1.5
    t[:, 45] = -0.5
    d = robodojo_io.upstream_actions_from_action80(t)
    assert d[0]["left_ee_joint_state"][0] == 0.0 and d[1]["right_ee_joint_state"][0] == 1.0
    with pytest.raises(ValueError):
        robodojo_io.upstream_actions_from_action80(np.zeros((80,), dtype=np.float32))
    with pytest.raises(ValueError):
        robodojo_io.upstream_actions_from_action80(np.full((2, 80), np.nan, dtype=np.float32))


def test_gripper_convention_round_trip():
    for opening in (0.0, 0.25, 1.0):
        closedness = robodojo_io.closedness_from_opening(opening)
        assert robodojo_io.opening_from_closedness(closedness) == np.float32(opening)


def test_instruction_from_obs():
    assert robodojo_io.instruction_from_obs({"instruction": "make toast"}) == "make toast"
    assert robodojo_io.instruction_from_obs({"instructions": ["pour water", "other"]}) == "pour water"
    with pytest.raises(KeyError):
        robodojo_io.instruction_from_obs({})
