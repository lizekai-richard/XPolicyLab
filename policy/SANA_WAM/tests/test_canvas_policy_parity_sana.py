"""Bitwise parity of the vendored mirror against Sana's live ``SanaRWMOpenWAMCanvasPolicy`` (skipped when the
rwm/openwam checkout is not importable; ``SANA_OPENWAM_REPO``, default ~/zekail/Sana_openwam).

A tiny live canvas policy and the mirror resolved from the same builder kwargs share one randomized state_dict
through the mirror's strict loader; ``x`` and ``action_pred`` must agree with rtol=atol=0 on CPU fp32 in both
state-conditioning modes (the default self-attention state token and the opt-in cross-attention key)."""

from __future__ import annotations

import copy
import os
import sys
from types import SimpleNamespace

import pytest
import torch

_SANA_WAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SANA_WAM_DIR not in sys.path:
    sys.path.insert(0, _SANA_WAM_DIR)
SANA_REPO = os.environ.get("SANA_OPENWAM_REPO", os.path.expanduser("~/zekail/Sana_openwam"))
if SANA_REPO not in sys.path and os.path.isdir(SANA_REPO):
    sys.path.insert(0, SANA_REPO)
# The live cross-attention takes xformers whenever it is installed and then builds its block mask on CUDA; this is a
# CPU fp32 parity test, so the live class must run SDPA like the mirror (dev/rwm/kernel/tests/conftest.py does the same).
os.environ["DISABLE_XFORMERS"] = "1"

live_module = pytest.importorskip(
    "dev.rwm.diffusion.model.nets.sana_qwennext_openwam_canvas_policy",
    reason="Sana rwm/openwam checkout not importable (set SANA_OPENWAM_REPO)",
)
Live = live_module.SanaRWMOpenWAMCanvasPolicy

from _tiny_policy import CAP_CH, DEPTH, FRAMES, HEADS, HIDDEN, IN_CH, LHD, MML, NOISY_T, ROBOT_DIM, SHD, VIEW_H, VIEW_W  # noqa: E402
from sana_wam_min.policy_model.checkpoint import load_policy_state_dict, strip_unmodeled_state  # noqa: E402
from sana_wam_min.policy_model.config import PolicyConfig  # noqa: E402
from sana_wam_min.policy_model.model import PolicyModel  # noqa: E402


class _Cfg(SimpleNamespace):
    """Unset fields read as None (dev/rwm/tests/test_policy_model_ownership.py)."""

    def __getattr__(self, name):
        return None


def _builder_kwargs(state_as_cross_attention: bool) -> dict:
    config = _Cfg(
        model=_Cfg(
            extra={"action_dim": ROBOT_DIM, "state_dim": ROBOT_DIM, "state_as_cross_attention": state_as_cross_attention},
            multiview_spatial_rope_layout="local_reset",
            multiview_spatial_rope_tile_shape=(15, 30),
        )
    )
    return dict(
        depth=DEPTH, hidden_size=HIDDEN, patch_size=(1, 1, 1), num_heads=HEADS, in_channels=IN_CH,
        caption_channels=CAP_CH, model_max_length=MML, linear_head_dim=LHD, softmax_head_dim=SHD, softmax_ratio=0.5,
        attn_res_block_size=2, use_time_conditioning=False, pred_sigma=False, y_norm=True, cross_norm=True,
        config=config,
    )


def _twins(state_as_cross_attention: bool):
    torch.manual_seed(20260916)
    kwargs = _builder_kwargs(state_as_cross_attention)
    live = Live(**kwargs)
    if getattr(live, "use_xformers_cross_attention", False):
        pytest.skip("the live model kept xformers cross-attention (imported before DISABLE_XFORMERS=1 took effect)")
    with torch.no_grad():
        for parameter in live.parameters():
            parameter.uniform_(-0.05, 0.05)
    live.eval()
    mirror = PolicyModel(PolicyConfig.from_sana_kwargs(shared_prompt=True, **kwargs))
    assert mirror.state_as_cross_attention == state_as_cross_attention == live.state_as_cross_attention
    load_policy_state_dict(mirror, live.state_dict())
    mirror.eval()
    return live, mirror


def _inputs(batch: int, fps: float, seed: int) -> dict:
    torch.manual_seed(seed)
    steps = (FRAMES - 1) * 8
    x = torch.randn(batch, IN_CH, FRAMES, VIEW_H, VIEW_W)
    timestep = torch.zeros(batch, 1, FRAMES)
    timestep[:, :, 1:] = NOISY_T
    y = torch.randn(batch, 1, 1, MML, CAP_CH)  # ONE prompt group shared by every token
    mask = torch.ones(batch, 1, MML, dtype=torch.int16)
    mask[:, :, 5:] = 0
    action_mask = torch.ones(batch, steps, ROBOT_DIM, dtype=torch.bool)
    action_mask[:, :, 66:] = False
    state_mask = torch.ones(batch, ROBOT_DIM, dtype=torch.bool)
    state_mask[:, 58:66] = False
    data_info = {
        "rwm_task": "policy",
        "view_count": torch.full((batch,), 1, dtype=torch.long),
        "view_latent_shape": torch.tensor([[[VIEW_H, VIEW_W]]] * batch),
        "view_slot_ids": torch.tensor((0,)),
        "model_fps": torch.full((batch,), fps),
        "initial_state80": torch.randn(batch, ROBOT_DIM),
        "initial_state_condition_mask80": state_mask,
        "action80": torch.randn(batch, steps, ROBOT_DIM),
        "action_mask80": action_mask,
        "action_timestep": torch.full((batch, steps), NOISY_T),
        "camera_conditioning_enabled": False,
    }
    return {"x": x, "timestep": timestep, "y": y, "mask": mask, "data_info": data_info}


def _run(model, inputs: dict) -> dict:
    with torch.no_grad():
        return model(
            inputs["x"].clone(), inputs["timestep"].clone(), inputs["y"].clone(), mask=inputs["mask"].clone(),
            data_info=copy.deepcopy(inputs["data_info"]),
        )


@pytest.mark.parametrize("state_as_cross_attention", [False, True])
def test_strict_load_after_documented_strip(state_as_cross_attention: bool) -> None:
    live, mirror = _twins(state_as_cross_attention)
    state = live.state_dict()
    stripped = strip_unmodeled_state(
        state, input_size=mirror.policy_config.input_size, hidden_size=HIDDEN, model_max_length=MML, caption_channels=CAP_CH
    )
    assert sorted(set(state) - set(stripped)) == ["plucker_embed.weight", "pos_embed", "y_embedder.y_embedding"]
    assert set(stripped) == set(mirror.state_dict())
    projector = "state_context_embed" if state_as_cross_attention else "state_embed"
    assert f"{projector}.proj.weight" in stripped
    for key, tensor in stripped.items():
        assert torch.equal(mirror.state_dict()[key], tensor), key


@pytest.mark.parametrize("state_as_cross_attention", [False, True])
@pytest.mark.parametrize("fps", [25.0, 16.0])
def test_forward_bitwise(state_as_cross_attention: bool, fps: float) -> None:
    live, mirror = _twins(state_as_cross_attention)
    inputs = _inputs(batch=2, fps=fps, seed=11 + int(state_as_cross_attention))
    out_live = _run(live, inputs)
    out_mirror = _run(mirror, inputs)
    assert out_mirror["x"].shape == out_live["x"].shape == (2, IN_CH, FRAMES, VIEW_H, VIEW_W)
    assert out_mirror["action_pred"].shape == out_live["action_pred"].shape
    torch.testing.assert_close(out_mirror["x"], out_live["x"], rtol=0, atol=0)
    torch.testing.assert_close(out_mirror["action_pred"], out_live["action_pred"], rtol=0, atol=0)
    assert out_live["action_pred"].abs().sum() > 0


def test_live_and_mirror_both_refuse_two_prompt_groups() -> None:
    live, mirror = _twins(False)
    inputs = _inputs(batch=1, fps=25.0, seed=5)
    inputs["y"] = torch.randn(1, 2, 1, MML, CAP_CH)
    inputs["mask"] = torch.ones(1, 2, MML, dtype=torch.int16)
    for model in (live, mirror):
        with pytest.raises(ValueError, match="ONE prompt"):
            _run(model, inputs)
