"""Mirror-only tests of the condition-causal memory-less policy (rwm/zekai-merge f6a257d4a / 8e0639b16 / 89eaa78f6):
the checkpoint decides the attention form (no yaml switch), the options are refused on a bidirectional model, and the
sampler's condition prefix cache equals the plain sampler (guidance included). Live bit parity:
test_condition_causal_live_parity.py."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from _tiny_policy import ROBOT_DIM, tiny_inputs, tiny_policy_config  # noqa: E402

from sana_wam_min.checkpoint import checkpoint_is_condition_causal  # noqa: E402
from sana_wam_min.policy_model.condition_chunks import (  # noqa: E402
    GDN_RECURRENCE_KEYS,
    condition_chunk_layout,
)
from sana_wam_min.policy_model.model import PolicyModel  # noqa: E402
from sana_wam_min.sampler import sample_policy  # noqa: E402


def _model(**changes) -> PolicyModel:
    torch.manual_seed(5)
    model = PolicyModel(dataclasses.replace(tiny_policy_config(), **changes).validate())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05)
    return model.eval()


def test_the_checkpoint_names_the_attention_form(tmp_path):
    bidirectional = _model().state_dict()
    condition = _model(condition_causal=True).state_dict()
    assert not set(condition) - set(bidirectional) - {
        key for key in condition if any(key.endswith("attn." + name) for name in GDN_RECURRENCE_KEYS)
    }
    for name, state, expected in (("bidir.bin", bidirectional, False), ("cond.bin", condition, True)):
        torch.save(state, tmp_path / name)
        assert checkpoint_is_condition_causal(str(tmp_path / name)) is expected
    partial = dict(condition)
    dropped = next(key for key in partial if key.endswith("attn.A_log"))
    partial.pop(dropped)
    torch.save(partial, tmp_path / "partial.bin")
    with pytest.raises(ValueError, match="carry the condition-causal recurrence"):
        checkpoint_is_condition_causal(str(tmp_path / "partial.bin"))


def test_a_bidirectional_model_refuses_the_condition_options():
    base = tiny_policy_config()
    for changes in ({"state_cross_attention": True}, {"gdn_output_normalizer": False}):
        with pytest.raises(ValueError, match="condition-causal policy"):
            dataclasses.replace(base, **changes).validate()
    dataclasses.replace(base, condition_causal=True, state_cross_attention=True, gdn_output_normalizer=False).validate()


def test_the_layout_puts_frame_zero_and_the_state_first():
    frame_ids = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2])          # two views of 2 and 1 tokens per frame
    layout = condition_chunk_layout(frame_ids, action_rows=4)
    assert layout.bounds == (0, 4, 14)                               # 3 frame-0 tokens + the state row | 6 + 4 actions
    assert layout.order[:4].tolist() == [0, 1, 6, 9]
    assert condition_chunk_layout(frame_ids, action_rows=4, state_row=False).bounds == (0, 3, 13)


def _sample(model, inputs, *, prefix_cache: bool, cfg: float):
    data_info = dict(inputs["data_info"])
    video, action = inputs["x"], data_info.pop("action80")
    action_mask = data_info.pop("action_mask80")
    data_info.pop("action_timestep")
    generator = torch.Generator().manual_seed(3)
    y, mask = inputs["y"], inputs["mask"]
    return sample_policy(
        model, video, None, None, torch.zeros_like(action), action_mask, y, mask,
        torch.zeros_like(y) if cfg > 1 else None, torch.ones_like(mask) if cfg > 1 else None,
        cfg, data_info, steps=4, flow_shift=3.0, generator=generator, action_cfg_scale=1.0,
        gripper_bounds=None, prefix_cache=prefix_cache,
    )


@pytest.mark.parametrize("state_cross", [False, True])
@pytest.mark.parametrize("cfg", [1.0, 4.0])
def test_the_cached_sampler_equals_the_plain_one(cfg, state_cross):
    model = _model(condition_causal=True, state_cross_attention=state_cross).double()
    inputs = tiny_inputs(batch=1, views=3, fps=25.0, seed=9)
    inputs["x"], inputs["y"] = inputs["x"].double(), inputs["y"].double()
    for key in ("initial_state80", "action80"):
        inputs["data_info"][key] = inputs["data_info"][key].double()
    plain = _sample(model, inputs, prefix_cache=False, cfg=cfg)
    cached = _sample(model, inputs, prefix_cache=True, cfg=cfg)
    for ours, reference in zip(cached, plain):
        torch.testing.assert_close(ours, reference, rtol=0, atol=1e-10)
    assert plain[1].shape[-1] == ROBOT_DIM


def test_the_prefix_cache_refuses_a_bidirectional_model():
    inputs = tiny_inputs(batch=1, views=3, fps=25.0, seed=9)
    with pytest.raises(ValueError, match="condition-causal policy only"):
        _sample(_model(), inputs, prefix_cache=True, cfg=1.0)
