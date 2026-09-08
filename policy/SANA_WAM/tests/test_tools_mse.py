"""CPU tests for the verification tools: MSE reduction, noise draw order, tensor comparison verdicts."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_SANA_WAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TOOLS_DIR = os.path.join(_SANA_WAM_DIR, "tools")
for _p in (_SANA_WAM_DIR, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compare_to_sana_reference as compare_tool  # noqa: E402
import holdout_replay  # noqa: E402
import import_isolation_check  # noqa: E402

ACTIVE_SLOTS = [0, 1, 2, 3, 4, 5, 16, 29, 30, 31, 32, 33, 34, 45]


def _mask(k: int = 24) -> torch.Tensor:
    mask = torch.zeros(1, k, 80, dtype=torch.bool)
    mask[..., ACTIVE_SLOTS] = True
    return mask


def test_masked_action_mse_matches_hand_computation():
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(1, 24, 80, generator=g)
    target = torch.randn(1, 24, 80, generator=g)
    mask = _mask()
    got = holdout_replay.masked_action_mse(pred, target, mask)
    p = pred.numpy()[0][:, ACTIVE_SLOTS]
    t = target.numpy()[0][:, ACTIVE_SLOTS]
    expected = float(((p - t) ** 2).sum() / (24 * len(ACTIVE_SLOTS)))
    assert got == pytest.approx(expected, rel=1e-6)
    # masked-out slots never contribute
    pred2 = pred.clone()
    pred2[..., 7] += 100.0
    assert holdout_replay.masked_action_mse(pred2, target, mask) == pytest.approx(got, rel=0, abs=0)


def test_masked_action_mse_float32_reduction_from_bf16_inputs():
    pred = torch.full((1, 24, 80), 0.1, dtype=torch.bfloat16)
    target = torch.zeros(1, 24, 80)
    mask = _mask()
    got = holdout_replay.masked_action_mse(pred, target, mask)
    assert got == pytest.approx(float(torch.tensor(0.1, dtype=torch.bfloat16).float() ** 2), rel=1e-6)


def test_masked_action_mse_rejects_empty_mask_and_shape_mismatch():
    with pytest.raises(ValueError):
        holdout_replay.masked_action_mse(torch.zeros(1, 24, 80), torch.zeros(1, 24, 80), torch.zeros(1, 24, 80, dtype=torch.bool))
    with pytest.raises(ValueError):
        holdout_replay.masked_action_mse(torch.zeros(1, 24, 80), torch.zeros(1, 23, 80), _mask())


def test_regenerate_noise_order_and_dtypes():
    device = torch.device("cpu")
    v1, a1 = holdout_replay.regenerate_noise(7, (1, 4, 3, 1, 5), torch.bfloat16, (1, 24, 80), torch.float32, device)
    v2, a2 = holdout_replay.regenerate_noise(7, (1, 4, 3, 1, 5), torch.bfloat16, (1, 24, 80), torch.float32, device)
    assert v1.dtype == torch.bfloat16 and a1.dtype == torch.float32
    assert torch.equal(v1, v2) and torch.equal(a1, a2)
    # video is drawn first: the action draw differs from a fresh generator's first draw
    fresh = torch.randn((1, 24, 80), generator=torch.Generator().manual_seed(7))
    assert not torch.equal(fresh, a1)


def test_move_data_info_keeps_non_tensors():
    info = {"rwm_task": "policy", "x": torch.ones(2), "n": 3}
    moved = holdout_replay.move_data_info(info, torch.device("cpu"))
    assert moved["rwm_task"] == "policy" and moved["n"] == 3 and torch.equal(moved["x"], info["x"])
    assert moved is not info


def test_summarize_and_reference_constants():
    s = holdout_replay.summarize([0.1, 0.3, 0.2])
    assert s == {"count": 3, "mean": pytest.approx(0.2), "median": 0.2, "max": 0.3, "min": 0.1}
    assert holdout_replay.summarize([]) == {"count": 0}
    assert holdout_replay.REFERENCE_AGGREGATES_S35000 == {"mean": 0.0219, "median": 0.0041}


def test_compare_gate_semantics():
    ref = torch.tensor([1.0, -2.0, 0.5])
    # Bitwise gates: any difference fails, however small.
    assert compare_tool.bitwise_gate(ref.clone(), ref)["verdict"] == "PASS"
    tiny = compare_tool.bitwise_gate(ref + 1e-7, ref)
    assert tiny["verdict"] == "FAIL" and tiny["bitwise"] is False and tiny["max_abs_diff"] > 0
    assert compare_tool.bitwise_gate(torch.zeros(2), ref)["verdict"] == "FAIL"
    # A bf16 round trip of a value bf16 cannot represent is a bitwise failure.
    rounded = torch.tensor([1.001, -2.0, 0.5])
    assert compare_tool.bitwise_gate(rounded.to(torch.bfloat16).float(), rounded)["verdict"] == "FAIL"
    # Kernel-level comparison: max|d| reported, gated only against an explicit tolerance.
    close = compare_tool.kernel_level_compare(ref + 1e-3, ref, 5e-2)
    assert close["verdict"] == "PASS" and close["gate"] == "kernel_level"
    assert close["max_abs_diff"] == pytest.approx(1e-3, rel=1e-3) and close["tol"] == 5e-2
    assert compare_tool.kernel_level_compare(ref + 0.1, ref, 5e-2)["verdict"] == "FAIL"
    assert compare_tool.kernel_level_compare(ref + 0.1, ref, None)["verdict"] == "INFO"
    assert compare_tool.kernel_level_compare(torch.zeros(2), ref, 5e-2)["verdict"] == "FAIL"
    assert compare_tool.bool_gate(True)["verdict"] == "PASS" and compare_tool.bool_gate(False)["verdict"] == "FAIL"
    assert compare_tool.all_pass([{"verdict": "PASS"}, {"verdict": "INFO"}]) == "PASS"
    assert compare_tool.all_pass([{"verdict": "PASS"}, {"verdict": "FAIL"}]) == "FAIL"
    assert compare_tool.all_pass([]) == "PASS"


def test_frames_chw_to_hwc():
    frame = torch.arange(2 * 3 * 4, dtype=torch.uint8).reshape(3, 2, 4)
    hwc = compare_tool.frames_chw_to_hwc([frame])[0]
    assert hwc.shape == (2, 4, 3) and hwc.flags["C_CONTIGUOUS"]
    assert np.array_equal(hwc[:, :, 1], frame[1].numpy())


def test_forbidden_module_classifier():
    names = [
        "torch", "diffusers.models", "sana_wam_min.sampler", "XPolicyLab.policy.SANA_WAM.model",
        "XPolicyLab.policy.SANA_WAM.sana_wam_min.vae", "dev.rwm.x", "diffusion.model", "sana", "sana.utils",
        "XPolicyLab.policy.OtherPolicy.model", "devtools", "development",
    ]
    bad = import_isolation_check.forbidden_modules(names)
    assert bad == sorted(["dev.rwm.x", "diffusion.model", "sana", "sana.utils", "XPolicyLab.policy.OtherPolicy.model"])
    roots = import_isolation_check.third_party_roots(["torch.nn", "os", "sana_wam_min.x", "XPolicyLab.policy", "numpy"])
    assert roots == ["numpy", "torch"]
