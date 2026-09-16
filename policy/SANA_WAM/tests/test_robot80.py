"""CPU tests for sana_wam_min.robot80 against the pinned RoboDojo normalization artifact."""

from __future__ import annotations

import hashlib
import json
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

from sana_wam_min import robot80  # noqa: E402

# The packaged copy shipped with the adapter (policy/SANA_WAM/normalization/); byte-identical to the
# checkpoint artifact (sha256 983fbd46...).
ARTIFACT = os.path.join(ADAPTER_DIR, "normalization", "robodojo_arx_x5_model_fps_25_f25_normalization.json")
ARTIFACT_SHA = "983fbd46df6af34e2048ed6806ae9ce49cd8bd3af72cfba7d8610069959ea1da"
JOINT_SLOTS = [0, 1, 2, 3, 4, 5, 29, 30, 31, 32, 33, 34]
ACTIVE_SLOTS = [0, 1, 2, 3, 4, 5, 16, 29, 30, 31, 32, 33, 34, 45]
NORMALIZED_SLOTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 29, 30, 31, 32, 33, 34, 36, 37, 38]


def _mask14() -> torch.Tensor:
    mask = torch.zeros(80, dtype=torch.bool)
    mask[ACTIVE_SLOTS] = True
    return mask


@pytest.fixture(scope="module")
def norm() -> robot80.Normalization:
    return robot80.load_normalization(ARTIFACT, expected_sha256=ARTIFACT_SHA)


def test_package_import_path():
    from XPolicyLab.policy.SANA_WAM.sana_wam_min import robot80 as via_pkg

    assert via_pkg.ROBOT80_DIM == 80


def test_constants():
    assert robot80.ROBOT80_DIM == 80
    assert robot80.LEFT_JOINT == slice(0, 7) and robot80.RIGHT_JOINT == slice(29, 36)
    assert robot80.LEFT_GRIPPER == 16 and robot80.RIGHT_GRIPPER == 45
    assert robot80.ROBOT80_GRIPPER_SLOTS == (16, 45)
    assert robot80.JOINT_TARGET_MODES == ("anchor_delta", "absolute")
    assert robot80.validate_joint_target_mode("Anchor_Delta") == "anchor_delta"
    with pytest.raises(ValueError):
        robot80.validate_joint_target_mode("relative")


def test_load_normalization_parses_pinned_artifact(norm):
    assert norm.sha256 == ARTIFACT_SHA
    assert norm.joint_target_mode == "anchor_delta"
    assert norm.action_representation == "robot_base_qwen"
    assert norm.action_mode == "robot_base_eef"
    assert norm.normalization_domain_id == "robodojo_arx_x5"
    assert norm.datasets == ("RoboDojo-ARX-X5",)
    assert norm.model_fps == 25 and norm.num_frames == 25
    for kind in ("state", "action"):
        center = getattr(norm, f"{kind}_center80")
        scale = getattr(norm, f"{kind}_scale80")
        nmask = getattr(norm, f"{kind}_normalization_mask80")
        q01 = getattr(norm, f"{kind}_q01_80")
        q99 = getattr(norm, f"{kind}_q99_80")
        assert center.dtype == np.float32 and scale.dtype == np.float32 and nmask.dtype == np.bool_
        assert center.shape == scale.shape == nmask.shape == (80,)
        assert sorted(np.flatnonzero(nmask).tolist()) == NORMALIZED_SLOTS
        assert nmask[JOINT_SLOTS].all()
        assert not nmask[16] and not nmask[45]
        np.testing.assert_allclose(center[nmask], ((q01 + q99) / 2.0)[nmask], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(scale[nmask], ((q99 - q01) / 2.0)[nmask], rtol=1e-6, atol=1e-7)
        assert np.all(center[~nmask] == 0.0) and np.all(scale[~nmask] == 1.0)
        assert np.all(scale > 0)
    # Anchor-delta action statistics differ from the absolute state statistics on the joints.
    assert not np.allclose(norm.action_center80[JOINT_SLOTS], norm.state_center80[JOINT_SLOTS])
    # Paired left/right pooling: slot i and slot 29+i share statistics.
    for i in range(6):
        assert norm.state_center80[i] == norm.state_center80[29 + i]
        assert norm.action_scale80[i] == norm.action_scale80[29 + i]


def test_spot_values_match_spec_table(norm):
    assert norm.state_center80[1] == pytest.approx(1.205615520477295, rel=1e-6)
    assert norm.state_scale80[5] == pytest.approx(1.9673399925231934, rel=1e-6)
    assert norm.action_center80[0] == pytest.approx(0.026520848274230957, rel=1e-6)
    assert norm.action_scale80[4] == pytest.approx(0.5588216185569763, rel=1e-6)


def test_sha_mismatch_and_prefix_form(tmp_path):
    with pytest.raises(ValueError, match="sha256 mismatch"):
        robot80.load_normalization(ARTIFACT, expected_sha256="0" * 64)
    ok = robot80.load_normalization(ARTIFACT, expected_sha256="sha256:" + ARTIFACT_SHA)
    assert ok.sha256 == ARTIFACT_SHA
    unpinned = robot80.load_normalization(ARTIFACT)
    assert unpinned.sha256 == hashlib.sha256(open(ARTIFACT, "rb").read()).hexdigest()


def test_parse_rejects_inconsistent_affine(tmp_path):
    artifact = json.load(open(ARTIFACT))
    artifact["action"]["center80"][0] += 1.0
    with pytest.raises(ValueError, match="disagree"):
        robot80.parse_normalization_artifact(artifact)
    artifact = json.load(open(ARTIFACT))
    artifact["joint_target_mode"] = "bogus"
    with pytest.raises(ValueError):
        robot80.parse_normalization_artifact(artifact)


def test_parse_derives_center_scale_from_quantiles(norm):
    artifact = json.load(open(ARTIFACT))
    for kind in ("state", "action"):
        del artifact[kind]["center80"]
        del artifact[kind]["scale80"]
    derived = robot80.parse_normalization_artifact(artifact)
    nm = norm.action_normalization_mask80
    np.testing.assert_allclose(derived.action_center80[nm], norm.action_center80[nm], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(derived.action_scale80[nm], norm.action_scale80[nm], rtol=1e-6, atol=1e-7)


def test_normalize_state_round_trip_and_gripper_passthrough(norm):
    gen = torch.Generator().manual_seed(0)
    raw = torch.zeros(80)
    raw[JOINT_SLOTS] = torch.rand(12, generator=gen) * 4 - 2
    raw[16] = 0.73
    raw[45] = 0.05
    raw[[6, 35, 7, 60]] = 5.0  # inactive junk that must be zeroed
    mask = _mask14()
    normalized = robot80.normalize_state(raw, mask, norm)
    assert normalized.dtype == torch.float32 and normalized.shape == (80,)
    expected = (raw[JOINT_SLOTS] - torch.from_numpy(norm.state_center80[JOINT_SLOTS])) / torch.from_numpy(
        norm.state_scale80[JOINT_SLOTS]
    )
    torch.testing.assert_close(normalized[JOINT_SLOTS], expected)
    assert normalized[16].item() == pytest.approx(0.73) and normalized[45].item() == pytest.approx(0.05)
    assert torch.all(normalized[~mask] == 0.0)
    recovered = robot80.denormalize_state(normalized, mask, norm)
    torch.testing.assert_close(recovered[mask], raw[mask], rtol=1e-5, atol=1e-6)
    assert torch.all(recovered[~mask] == 0.0)


def test_normalize_state_accepts_numpy(norm):
    raw = np.zeros(80, dtype=np.float32)
    raw[0] = 1.0
    mask = _mask14().numpy()
    out = robot80.normalize_state(raw, mask, norm)
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32
    assert out[0].item() == pytest.approx((1.0 - norm.state_center80[0]) / norm.state_scale80[0], rel=1e-6)


def test_denormalize_action_round_trip(norm):
    gen = torch.Generator().manual_seed(1)
    model = torch.randn(24, 80, generator=gen)
    model[:, [16, 45]] = torch.rand(24, 2, generator=gen)
    mask = _mask14()[None, :].expand(24, 80).contiguous()
    raw = robot80.denormalize_action(model, mask, norm)
    assert raw.shape == (24, 80) and raw.dtype == torch.float32
    expected = model[:, JOINT_SLOTS] * torch.from_numpy(norm.action_scale80[JOINT_SLOTS]) + torch.from_numpy(
        norm.action_center80[JOINT_SLOTS]
    )
    torch.testing.assert_close(raw[:, JOINT_SLOTS], expected)
    torch.testing.assert_close(raw[:, [16, 45]], model[:, [16, 45]])
    assert torch.all(raw[~mask] == 0.0)
    back = robot80.normalize_action(raw, mask, norm)
    torch.testing.assert_close(back[mask], model[mask], rtol=1e-5, atol=1e-6)
    assert torch.all(back[~mask] == 0.0)


def test_affine_zeroes_invalid_slots_unconditionally(norm):
    values = torch.full((3, 80), 7.0)
    mask = torch.zeros(3, 80, dtype=torch.bool)
    mask[:, 0] = True
    out = robot80.denormalize_action(values, mask, norm)
    assert torch.count_nonzero(out) == 3
    assert torch.all(out[:, 1:] == 0.0)


def test_affine_shape_and_finite_guards(norm):
    with pytest.raises(ValueError):
        robot80.normalize_state(torch.zeros(79), torch.zeros(79, dtype=torch.bool), norm)
    with pytest.raises(ValueError):
        robot80.denormalize_action(torch.zeros(2, 80), torch.zeros(3, 80, dtype=torch.bool), norm)
    bad = torch.zeros(80)
    bad[0] = float("nan")
    with pytest.raises(ValueError):
        robot80.normalize_state(bad, _mask14(), norm)
