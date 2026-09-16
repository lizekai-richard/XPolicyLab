"""CPU tests for sana_wam_min.actions (denormalize -> anchor reconstruction -> gripper clip -> joint limits)."""

from __future__ import annotations

import math
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

from sana_wam_min import actions, robot80  # noqa: E402

# The packaged copy shipped with the adapter (policy/SANA_WAM/normalization/); byte-identical to the
# checkpoint artifact (sha256 983fbd46...).
ARTIFACT = os.path.join(ADAPTER_DIR, "normalization", "robodojo_arx_x5_model_fps_25_f25_normalization.json")
JOINT_SLOTS = [0, 1, 2, 3, 4, 5, 29, 30, 31, 32, 33, 34]
ACTIVE_SLOTS = [0, 1, 2, 3, 4, 5, 16, 29, 30, 31, 32, 33, 34, 45]
K = 24


def _mask14() -> torch.Tensor:
    mask = torch.zeros(80, dtype=torch.bool)
    mask[ACTIVE_SLOTS] = True
    return mask


def _chunk_mask() -> torch.Tensor:
    return _mask14()[None, :].expand(K, 80).contiguous()


@pytest.fixture(scope="module")
def norm() -> robot80.Normalization:
    return robot80.load_normalization(ARTIFACT)


def test_package_import_path():
    from XPolicyLab.policy.SANA_WAM.sana_wam_min import actions as via_pkg

    assert via_pkg.JOINT_LIMIT_MODES == ("clip", "reject")


def test_reconstruct_adds_anchor_on_masked_joint_slots_only():
    gen = torch.Generator().manual_seed(0)
    delta = torch.zeros(K, 80)
    delta[:, ACTIVE_SLOTS] = torch.randn(K, len(ACTIVE_SLOTS), generator=gen)
    anchor = torch.zeros(80)
    anchor[JOINT_SLOTS] = torch.arange(12, dtype=torch.float32) + 1.0
    anchor[16] = 0.9  # gripper anchor must be ignored
    anchor[6] = 100.0  # slot 6 inactive: must not leak
    mask = _chunk_mask()
    anchor_mask = _mask14()
    out = actions.reconstruct_absolute_joints(delta, mask, anchor, anchor_mask)
    assert out.shape == (K, 80) and out.dtype == torch.float32
    torch.testing.assert_close(out[:, JOINT_SLOTS], delta[:, JOINT_SLOTS] + anchor[JOINT_SLOTS][None, :])
    torch.testing.assert_close(out[:, [16, 45]], delta[:, [16, 45]])
    assert torch.all(out[~mask] == 0.0)
    # Inactive joint slot inside the slice keeps its (zero) delta.
    assert torch.all(out[:, 6] == 0.0) and torch.all(out[:, 35] == 0.0)


def test_reconstruct_partial_mask_only_touches_active_rows_slots():
    delta = torch.ones(K, 80)
    mask = torch.zeros(K, 80, dtype=torch.bool)
    mask[:, 3] = True
    anchor = torch.full((80,), 2.0)
    anchor_mask = torch.ones(80, dtype=torch.bool)
    out = actions.reconstruct_absolute_joints(delta, mask, anchor, anchor_mask)
    assert torch.all(out[:, 3] == 3.0)
    other = [s for s in range(80) if s != 3]
    assert torch.all(out[:, other] == 1.0)


def test_reconstruct_raises_on_missing_anchor():
    delta = torch.zeros(K, 80)
    mask = _chunk_mask()
    anchor = torch.zeros(80)
    anchor_mask = _mask14()
    anchor_mask[30] = False
    with pytest.raises(ValueError, match="slot 30 .*row 0"):
        actions.reconstruct_absolute_joints(delta, mask, anchor, anchor_mask)


def test_reconstruct_accepts_numpy_inputs():
    delta = np.zeros((K, 80), dtype=np.float32)
    delta[:, 0] = 0.5
    mask = np.zeros((K, 80), dtype=np.bool_)
    mask[:, 0] = True
    anchor = np.zeros(80, dtype=np.float32)
    anchor[0] = 1.0
    out = actions.reconstruct_absolute_joints(delta, mask, anchor, mask[0])
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32
    assert torch.all(out[:, 0] == 1.5)


def test_clip_gripper_closedness_clamps_only_16_45():
    action = torch.zeros(K, 80)
    action[:, 16] = 1.7
    action[:, 45] = -0.3
    action[:, 0] = 9.0  # joint out of range stays untouched here
    action[:, 20] = 5.0
    mask = _chunk_mask()
    out = actions.clip_gripper_closedness(action, mask)
    assert torch.all(out[:, 16] == 1.0) and torch.all(out[:, 45] == 0.0)
    assert torch.all(out[:, 0] == 9.0) and torch.all(out[:, 20] == 5.0)
    # Unmasked gripper keeps its original value.
    mask2 = mask.clone()
    mask2[:, 45] = False
    out2 = actions.clip_gripper_closedness(action, mask2)
    assert torch.all(out2[:, 45] == -0.3) and torch.all(out2[:, 16] == 1.0)
    # Input is not mutated.
    assert torch.all(action[:, 16] == 1.7)


def test_apply_joint_limits_reject_and_clip():
    slots = [0, 1, 2, 3, 4, 5, 29, 30, 31, 32, 33, 34]
    lower = [-math.pi] * 12
    upper = [math.pi] * 12
    action = torch.zeros(K, 80)
    action[:, 31] = 3.0
    out, clipped = actions.apply_joint_limits(action, slots, lower, upper, mode="reject")
    assert clipped == () and torch.equal(out, action)

    action[5, 31] = 3.5
    action[2, 0] = -4.0
    with pytest.raises(ValueError, match="slot 0 .*clipping is forbidden"):
        actions.apply_joint_limits(action, slots, lower, upper, mode="reject")
    out, clipped = actions.apply_joint_limits(action, slots, lower, upper, mode="clip")
    assert clipped == (0, 31)
    pi32 = np.float32(math.pi)
    # float32(pi) > pi, so the upper bound is nudged one ulp inward; float32(-pi) < -pi likewise.
    hi = np.nextafter(pi32, np.float32(-np.inf))
    lo = np.nextafter(-pi32, np.float32(np.inf))
    assert out[5, 31].item() == float(hi)
    assert out[2, 0].item() == float(lo)
    assert out[0, 31].item() == 3.0
    # The gripper and other slots are untouched, and the input is not mutated.
    assert torch.all(out[:, 16] == 0.0) and action[5, 31].item() == 3.5
    with pytest.raises(ValueError):
        actions.apply_joint_limits(action, slots, lower, upper, mode="warn")


def test_model_action_to_absolute_order(norm):
    gen = torch.Generator().manual_seed(3)
    model = torch.randn(K, 80, generator=gen)
    model[:, 16] = 1.4  # exceeds [0,1] -> clipped last
    model[:, 45] = -0.2
    mask = _chunk_mask()
    anchor = torch.zeros(80)
    anchor[JOINT_SLOTS] = torch.linspace(-1.0, 1.0, 12)
    anchor_mask = _mask14()

    out = actions.model_action_to_absolute(model, mask, norm, anchor, anchor_mask)

    # Step 1: denormalize with the ACTION statistics (not state).
    denorm = robot80.denormalize_action(model, mask, norm)
    # Step 2: anchor added to the joints after denormalization (delta in radians, not normalized units).
    expected_joints = denorm[:, JOINT_SLOTS] + anchor[JOINT_SLOTS][None, :]
    torch.testing.assert_close(out[:, JOINT_SLOTS], expected_joints)
    wrong_order = robot80.denormalize_action(model + anchor[None, :], mask, norm)[:, JOINT_SLOTS]
    assert not torch.allclose(out[:, JOINT_SLOTS], wrong_order)
    # Step 3: grippers clipped to [0, 1] in the physical domain.
    assert torch.all(out[:, 16] == 1.0) and torch.all(out[:, 45] == 0.0)
    assert torch.all(out[~mask] == 0.0)
    assert out.dtype == torch.float32 and out.shape == (K, 80)


def test_model_action_to_absolute_skips_anchor_for_absolute_mode(norm):
    absolute_norm = robot80.Normalization(
        state_center80=norm.state_center80,
        state_scale80=norm.state_scale80,
        state_normalization_mask80=norm.state_normalization_mask80,
        action_center80=norm.action_center80,
        action_scale80=norm.action_scale80,
        action_normalization_mask80=norm.action_normalization_mask80,
        joint_target_mode="absolute",
        action_representation=norm.action_representation,
        sha256=None,
    )
    model = torch.zeros(K, 80)
    mask = _chunk_mask()
    anchor = torch.full((80,), 1.0)
    out = actions.model_action_to_absolute(model, mask, absolute_norm, anchor, _mask14())
    torch.testing.assert_close(out[:, JOINT_SLOTS], torch.from_numpy(norm.action_center80[JOINT_SLOTS])[None, :].expand(K, 12))
