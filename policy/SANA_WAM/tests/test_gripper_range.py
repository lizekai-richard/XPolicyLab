"""The deploy key ``gripper_range`` on SANA_WAM: the model-domain range of the gripper closedness.

The parser and the slot rewrite are robot80's (SANA_WAM_CAUSAL/tests/test_gripper_range.py covers them on the 1fe3b7e7
artifact and its Sana 41cf94114 [0, 1] twin). Here: the session serves the requested map (auto = the artifact's own),
normalizes the state with it, clamps the sampled gripper to its image of raw [0, 1] and names it in every receipt;
from_paths refuses a bad value before the weight load and records the label in its load report.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from test_session_cpu import (  # noqa: E402  (puts policy/SANA_WAM on sys.path)
    CAP_L,
    SNAPSHOT_CONFIG,
    StubModel,
    StubTextEncoder,
    StubTokenizer,
    fake_frames,
    fake_state,
    make_session,
    stub_vae,
)

from sana_wam_min import session as session_mod  # noqa: E402
from sana_wam_min.config import load_train_config  # noqa: E402
from sana_wam_min.robodojo_io import state80_from_obs  # noqa: E402
from sana_wam_min.robot80 import (  # noqa: E402
    GRIPPER_RANGE_SIGNED,
    GRIPPER_RANGE_UNIT,
    load_normalization,
    normalized_gripper_bounds,
    with_gripper_range,
)
from sana_wam_min.session import PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256  # noqa: E402

GRIPPERS = (16, 45)
SIGNED_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0))
UNIT_BOUNDS = ((0.0, 1.0), (0.0, 1.0))


def _packaged():
    """The packaged f25 artifact: grippers identity (the pre-2026-09-20 scheme), the model sees [0, 1]."""

    return load_normalization(PACKAGED_NORMALIZATION_PATH, TRAINING_NORMALIZATION_SHA256)


def _signed():
    """The packaged artifact with the 2026-09-20 gripper map and its stamp: the model sees [-1, 1]."""

    return dataclasses.replace(with_gripper_range(_packaged(), GRIPPER_RANGE_SIGNED), gripper_normalization="statistics")


@pytest.mark.parametrize(
    "artifact, value, label, bounds",
    [
        (_packaged, None, "[0, 1](artifact)", UNIT_BOUNDS),
        (_packaged, "auto", "[0, 1](artifact)", UNIT_BOUNDS),
        (_packaged, "[0, 1]", "[0, 1](deploy = artifact)", UNIT_BOUNDS),
        (_packaged, [-1, 1], "[-1, 1](deploy OVERRIDES artifact [0, 1])", SIGNED_BOUNDS),
        (_signed, "auto", "[-1, 1](artifact)", SIGNED_BOUNDS),
        (_signed, "[-1, 1]", "[-1, 1](deploy = artifact)", SIGNED_BOUNDS),
        (_signed, "[0, 1]", "[0, 1](deploy OVERRIDES artifact [-1, 1])", UNIT_BOUNDS),
    ],
)
def test_the_session_serves_the_requested_map(artifact, value, label, bounds):
    norm = artifact()
    session = session_mod.PolicyInferenceSession(**{**_session_kwargs(norm), "gripper_range": value})
    assert session.gripper_range == label
    assert normalized_gripper_bounds(session.normalization) == bounds
    if "OVERRIDES" not in label:
        assert session.normalization is norm                                     # auto / equal: the artifact untouched
    other = [slot for slot in range(80) if slot not in GRIPPERS]
    for kind in ("state", "action"):
        for field in ("center80", "scale80", "normalization_mask80"):
            served, given = (np.asarray(getattr(n, f"{kind}_{field}")) for n in (session.normalization, norm))
            np.testing.assert_array_equal(served[other], given[other])


def test_the_session_refuses_a_bad_range():
    with pytest.raises(ValueError, match="gripper_range must be auto"):
        make_session(gripper_range="[0, 2]")


@pytest.mark.parametrize("value, expected", [(None, (0.75, 0.0)), ("[-1, 1]", (0.5, -1.0))])
def test_state_clamp_and_receipt_follow_the_served_map(monkeypatch, value, expected):
    """fake_state: left closedness 0.75, right 0.0 -> the model reads them through the served map; the sampler clamps
    the gripper to the served image of raw [0, 1]; the receipt names the map."""

    seen = []
    real = session_mod.sample_policy

    def spy(*args, **kwargs):
        seen.append((args[11]["initial_state80"].detach().cpu().float(), kwargs.get("gripper_bounds")))
        return real(*args, **kwargs)

    monkeypatch.setattr(session_mod, "sample_policy", spy)
    state80, mask80 = state80_from_obs(fake_state())
    session = make_session(gripper_range=value)
    result = session.predict(fake_frames(), state80, mask80, "Open the drawer.", torch.Generator().manual_seed(1))
    (state, bounds), = seen
    torch.testing.assert_close(state.reshape(-1, 80)[0, list(GRIPPERS)], torch.tensor(expected), rtol=0, atol=1e-7)
    assert bounds == (UNIT_BOUNDS if value is None else SIGNED_BOUNDS)
    assert result.receipt["gripper_range"] == session.gripper_range


def _session_kwargs(norm) -> dict:
    train_cfg = load_train_config(SNAPSHOT_CONFIG)
    train_cfg["text_encoder"]["model_max_length"] = CAP_L
    return dict(
        model=StubModel(), vae=stub_vae(), tokenizer=StubTokenizer(), text_encoder=StubTextEncoder(),
        normalization=norm, train_config=train_cfg, device="cpu", steps=3, cfg_scale=1.0, flow_shift=3.5,
        checkpoint_path="stub",
    )


def _patch_loaders(monkeypatch, weights_loaded: list):
    train_cfg = load_train_config(SNAPSHOT_CONFIG)
    train_cfg["text_encoder"]["model_max_length"] = CAP_L
    monkeypatch.setattr(session_mod, "find_train_config", lambda checkpoint_dir: SNAPSHOT_CONFIG)
    monkeypatch.setattr(session_mod, "load_train_config", lambda path: train_cfg)
    monkeypatch.setattr(session_mod, "checkpoint_is_condition_causal", lambda checkpoint_dir: False)
    monkeypatch.setattr(session_mod, "load_checked_normalization", lambda *args: _packaged())
    monkeypatch.setattr(session_mod, "build_policy_model", lambda *args, **kwargs: StubModel())

    def load_weights(model, checkpoint_dir, device):
        weights_loaded.append(checkpoint_dir)
        return {"missing": 0}

    monkeypatch.setattr(session_mod, "load_policy_weights", load_weights)
    monkeypatch.setattr(session_mod, "load_vae", lambda *args, **kwargs: stub_vae())
    monkeypatch.setattr(session_mod, "load_text_encoder", lambda *args, **kwargs: (StubTokenizer(), StubTextEncoder()))
    monkeypatch.setattr(session_mod, "resolve_checkpoint_file", lambda checkpoint_dir: "stub")


def test_from_paths_applies_the_key_and_reports_it(monkeypatch):
    weights_loaded = []
    _patch_loaders(monkeypatch, weights_loaded)
    session = session_mod.PolicyInferenceSession.from_paths(
        "ckpt", "gemma", "vae", device="cpu", steps=3, cfg_scale=1.0, flow_shift=3.5, gripper_range="[-1, 1]"
    )
    assert weights_loaded == ["ckpt"]
    assert session.gripper_range == session.load_report["gripper_range"] == "[-1, 1](deploy OVERRIDES artifact [0, 1])"
    assert normalized_gripper_bounds(session.normalization) == SIGNED_BOUNDS


def test_from_paths_refuses_a_bad_range_before_the_weights(monkeypatch):
    weights_loaded = []
    _patch_loaders(monkeypatch, weights_loaded)
    with pytest.raises(ValueError, match="gripper_range must be auto"):
        session_mod.PolicyInferenceSession.from_paths("ckpt", "gemma", "vae", device="cpu", gripper_range="0..1")
    assert weights_loaded == []
