"""CPU tests for the canvas-policy flags of the vendored policy_model: the shared prompt (G = 1, one query span over
video + robot tail) and the opt-in state-as-cross-attention conditioning (actions-only tail, one appended text key,
independent local 1D action RoPE), plus the checkpoint loader's state-projector guard."""

from __future__ import annotations

import copy
import dataclasses
import os
import sys
from types import SimpleNamespace

import pytest
import torch

# policy/SANA_WAM (plain ``sana_wam_min`` imports) and the XPolicyLab parent (``XPolicyLab.policy.SANA_WAM`` imports).
ADAPTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_XPOLICYLAB_PARENT = os.path.abspath(os.path.join(ADAPTER_DIR, "..", "..", ".."))
for _p in (ADAPTER_DIR, _XPOLICYLAB_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _tiny_policy import (  # noqa: E402
    CAP_CH,
    FRAMES,
    HIDDEN,
    IN_CH,
    MML,
    ROBOT_DIM,
    VIEW_H,
    VIEW_W,
    tiny_inputs,
    tiny_policy_config,
)
from sana_wam_min.checkpoint import CHECKPOINT_RELATIVE_PATH, build_policy_model, load_policy_weights  # noqa: E402
from sana_wam_min.policy_model.config import PolicyConfig  # noqa: E402
from sana_wam_min.policy_model.model import _independent_action_rope  # noqa: E402

STEPS = (FRAMES - 1) * 8
VIDEO_TOKENS = FRAMES * VIEW_H * VIEW_W


def canvas_config(state_as_cross_attention: bool = False) -> PolicyConfig:
    return dataclasses.replace(
        tiny_policy_config(), shared_prompt=True, state_as_cross_attention=state_as_cross_attention
    ).validate()


def canvas_model(state_as_cross_attention: bool = False, seed: int = 20260916):
    torch.manual_seed(seed)
    model = build_policy_model(canvas_config(state_as_cross_attention), dtype=torch.float32, device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05)
    return model


def canvas_inputs(batch: int = 2, seed: int = 3, groups: int = 1) -> dict:
    inputs = tiny_inputs(batch=batch, views=1, fps=25.0, seed=seed)
    torch.manual_seed(seed + 100)
    inputs["y"] = torch.randn(batch, groups, 1, MML, CAP_CH)
    mask = torch.ones(batch, groups, 1, 1, MML, dtype=torch.int64)
    mask[..., 5:] = 0
    inputs["mask"] = mask
    return inputs


def _run(model, inputs: dict) -> dict:
    with torch.no_grad():
        return model(
            inputs["x"], inputs["timestep"], inputs["y"], mask=inputs["mask"], data_info=copy.deepcopy(inputs["data_info"])
        )


def test_shared_prompt_forward_uses_one_span_over_video_state_and_actions():
    model = canvas_model()
    assert model.shared_prompt and not model.state_as_cross_attention and hasattr(model, "state_embed")
    assert model._prompt_group_spans((VIDEO_TOKENS,), STEPS) == ((0, VIDEO_TOKENS + 1 + STEPS),)
    out = _run(model, canvas_inputs())
    assert out["x"].shape == (2, IN_CH, FRAMES, VIEW_H, VIEW_W) and out["action_pred"].shape == (2, STEPS, ROBOT_DIM)
    assert torch.isfinite(out["x"]).all() and torch.isfinite(out["action_pred"]).all()
    assert out["action_pred"].abs().sum() > 0
    assert torch.equal(out["action_pred"][:, :, 66:], torch.zeros_like(out["action_pred"][:, :, 66:]))
    again = _run(model, canvas_inputs())
    torch.testing.assert_close(again["x"], out["x"], rtol=0, atol=0)
    torch.testing.assert_close(again["action_pred"], out["action_pred"], rtol=0, atol=0)


def test_shared_prompt_refuses_several_text_groups_and_several_views():
    model = canvas_model()
    with pytest.raises(ValueError, match="ONE prompt"):
        _run(model, canvas_inputs(groups=2))
    with pytest.raises(ValueError, match="ONE composite visual stream"):
        _run(model, tiny_inputs(batch=1, views=3, fps=25.0, seed=1))
    # the three-view model still routes V + 1 groups (unchanged behaviour)
    default = build_policy_model(tiny_policy_config(), dtype=torch.float32, device="cpu")
    assert not default.shared_prompt and not default.state_as_cross_attention
    assert default._prompt_group_spans((VIDEO_TOKENS,) * 3, STEPS) == (
        (0, VIDEO_TOKENS), (VIDEO_TOKENS, VIDEO_TOKENS), (2 * VIDEO_TOKENS, VIDEO_TOKENS), (3 * VIDEO_TOKENS, 1 + STEPS),
    )


def test_state_as_cross_attention_replaces_the_state_token_by_a_text_key():
    model = canvas_model(state_as_cross_attention=True)
    assert hasattr(model, "state_context_embed") and not hasattr(model, "state_embed")
    keys = set(model.state_dict())
    assert "state_context_embed.proj.weight" in keys and not any(k.startswith("state_embed.") for k in keys)
    assert model._prompt_group_spans((VIDEO_TOKENS,), STEPS) == ((0, VIDEO_TOKENS + STEPS),)
    inputs = canvas_inputs()
    out = _run(model, inputs)
    assert out["x"].shape == (2, IN_CH, FRAMES, VIEW_H, VIEW_W) and out["action_pred"].shape == (2, STEPS, ROBOT_DIM)
    assert torch.isfinite(out["x"]).all() and out["action_pred"].abs().sum() > 0
    # the state still conditions the prediction (through the appended key) ...
    perturbed = copy.deepcopy(inputs)
    perturbed["data_info"]["initial_state80"] = inputs["data_info"]["initial_state80"] + 1.0
    assert not torch.equal(_run(model, perturbed)["action_pred"], out["action_pred"])
    # ... and a masked-out state slot (58:66 in tiny_inputs) does not
    masked = copy.deepcopy(inputs)
    masked["data_info"]["initial_state80"][:, 60] += 5.0
    torch.testing.assert_close(_run(model, masked)["action_pred"], out["action_pred"], rtol=0, atol=0)


def test_splice_appends_one_masked_in_column_for_the_last_group():
    model = canvas_model(state_as_cross_attention=True)
    inputs = canvas_inputs(batch=2)
    text = torch.randn(2, 1, MML, HIDDEN)
    mask = torch.ones(2, 1, MML, dtype=torch.int16)
    mask[:, :, 5:] = 0
    spliced, spliced_mask = model._splice_state_context(text, mask, inputs["data_info"])
    assert spliced.shape == (2, 1, MML + 1, HIDDEN) and spliced_mask.shape == (2, 1, MML + 1)
    assert torch.equal(spliced[:, :, :MML], text) and torch.equal(spliced_mask[:, :, :MML], mask)
    assert spliced_mask[:, -1, -1].tolist() == [1, 1] and spliced_mask.dtype == torch.int16
    state = inputs["data_info"]["initial_state80"]
    state_mask = inputs["data_info"]["initial_state_condition_mask80"]
    with torch.no_grad():
        expected = model.state_context_embed(state, state_mask)
    torch.testing.assert_close(spliced[:, 0, -1], expected, rtol=0, atol=0)
    # no key mask: an all-ones base mask is synthesized
    spliced, spliced_mask = model._splice_state_context(text, None, inputs["data_info"])
    assert spliced_mask.shape == (2, 1, MML + 1) and bool((spliced_mask == 1).all())
    # several groups: only the last one takes the state key
    _, two_mask = model._splice_state_context(torch.randn(2, 2, MML, HIDDEN), None, inputs["data_info"])
    assert two_mask[:, 0, -1].tolist() == [0, 0] and two_mask[:, 1, -1].tolist() == [1, 1]


def test_independent_action_rope_is_a_full_head_dim_local_clock():
    model = canvas_model(state_as_cross_attention=True)
    device = torch.device("cpu")
    for rope in (model.rope_linear, model.rope_softmax):
        head_dim = sum(rope.axis_dims)
        freqs = _independent_action_rope(rope, STEPS, 2, device)
        assert freqs.shape == (2, 1, STEPS, head_dim // 2) and freqs.dtype == torch.complex128
        torch.testing.assert_close(freqs[0, 0, 0], torch.ones(head_dim // 2, dtype=torch.complex128), rtol=0, atol=0)
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
        expected = torch.polar(torch.ones(head_dim // 2, dtype=torch.float64), 3.0 * rope.theta ** (-exponent))
        torch.testing.assert_close(freqs[1, 0, 3], expected, rtol=0, atol=0)
        # the joint RoPE is the video grid followed by the action rows; no state row, no fps scaling of the tail
        fps = torch.full((2,), 25.0, dtype=torch.float64)
        model.f, model.h, model.w = FRAMES, VIEW_H, VIEW_W
        with rope.use_model_fps(fps, batch_size=2, device=device):
            joint = model._robot_rope(rope, {}, fps, ((VIEW_H, VIEW_W),), STEPS, device)
        assert joint.shape[2] == VIDEO_TOKENS + STEPS
        torch.testing.assert_close(joint[:, :, VIDEO_TOKENS:], freqs, rtol=0, atol=0)


def test_state_as_cross_attention_requires_the_shared_prompt():
    with pytest.raises(ValueError, match="shared-prompt"):
        dataclasses.replace(tiny_policy_config(), state_as_cross_attention=True).validate()
    config = SimpleNamespace(
        model=SimpleNamespace(
            extra={"state_as_cross_attention": True},
            multiview_spatial_rope_layout="local_reset",
            multiview_spatial_rope_tile_shape=(15, 30),
        ),
        vae=SimpleNamespace(vae_stride=[8, 32, 32]),
    )
    kwargs = dict(
        depth=4, hidden_size=64, patch_size=(1, 1, 1), num_heads=2, input_size=2, in_channels=4, caption_channels=32,
        model_max_length=8, linear_head_dim=32, softmax_head_dim=32, softmax_ratio=0.5, attn_res_block_size=2,
        use_time_conditioning=False, pred_sigma=False, y_norm=True, cross_norm=True, config=config,
    )
    with pytest.raises(ValueError, match="shared-prompt"):
        PolicyConfig.from_sana_kwargs(**kwargs)  # model.extra opts in, but the prompt is not shared
    resolved = PolicyConfig.from_sana_kwargs(shared_prompt=True, **kwargs)
    assert resolved.shared_prompt and resolved.state_as_cross_attention
    assert not PolicyConfig.from_sana_kwargs(shared_prompt=True, state_as_cross_attention=False, **kwargs).state_as_cross_attention


def test_checkpoint_loader_guards_the_state_projector(tmp_path):
    token_model, context_model = canvas_model(False), canvas_model(True)
    for source, target in ((context_model, canvas_model(False, seed=1)), (token_model, canvas_model(True, seed=1))):
        ckpt = tmp_path / f"mismatch_{int(source.state_as_cross_attention)}"
        (ckpt / "model").mkdir(parents=True)
        torch.save(source.state_dict(), ckpt / CHECKPOINT_RELATIVE_PATH)
        with pytest.raises(ValueError, match="state conditioning disagrees"):
            load_policy_weights(target, str(ckpt))
    for source in (token_model, context_model):
        ckpt = tmp_path / f"match_{int(source.state_as_cross_attention)}"
        (ckpt / "model").mkdir(parents=True)
        torch.save(source.state_dict(), ckpt / CHECKPOINT_RELATIVE_PATH)
        target = canvas_model(source.state_as_cross_attention, seed=2)
        report = load_policy_weights(target, str(ckpt))
        assert report["tensors_loaded"] == len(source.state_dict()) and report["stripped"] == []
        for key, tensor in source.state_dict().items():
            assert torch.equal(target.state_dict()[key], tensor), key
