"""Bitwise parity of the policy port with the live one-policy class of rwm/zekai-merge at the 2026-09-23 contracts.

The live model (``SANA_ZEKAI_MERGE_REPO``, default ~/zekail/Sana, checked out at >= b13415841) is built with
``model.extra.rope`` aligned / independent / undeclared and compared with the mirror on:

* the three-view strip (V = 3, G = 4) at video frame strides 1, 2 and 4 -- ``aligned`` folds the stride into the video
  clock, ``independent`` keeps the video clock and puts the actions at 1 .. A;
* a single visual stream (V = 1, the canvas modes ``openwam`` / ``sana_pixel``) carrying G = 2 rows (view, robot tail:
  the 2026-09-21 payload) or ONE row (G = 1: one span over the whole sequence, the payload since 2026-09-22).

The spatial RoPE of V > 1 is the fixed semantic 2x2 tile (15, 30) on this tree (the yaml key is gone), which the mirror
resolves from a yaml without the key. Run this file on its own (``pytest tests/test_zekai_merge_head_parity.py``): other
parity files import other Sana checkouts, and the first one owns the ``dev`` package of the interpreter.
"""

from __future__ import annotations

import copy
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from _tiny_policy import CAP_CH, DEPTH, FRAMES, HEADS, HIDDEN, IN_CH, LHD, MML, NOISY_T, ROBOT_DIM, SHD  # noqa: E402

from sana_wam_min.policy_model.checkpoint import load_policy_state_dict  # noqa: E402
from sana_wam_min.policy_model.config import PolicyConfig  # noqa: E402
from sana_wam_min.policy_model.model import PolicyModel  # noqa: E402

# pinned to the contract these tests check (b13415841, before the (8, 16) strip tile of 6da565230 and the later moves):
# the moving ~/zekail/Sana head no longer matches it
SANA_ZEKAI_MERGE_REPO = os.environ.get("SANA_ZEKAI_MERGE_REPO", os.path.expanduser("~/zekail/Sana_b13415841"))
STEPS_PER_FRAME = 8
VIEW_H, VIEW_W = 2, 3          # one tile of the strip; fits the fixed (15, 30) semantic tile
CANVAS_H, CANVAS_W = 3, 4      # a single-stream grid (the canvases are 12x10 / 10x15 at real scale)


class _Cfg(SimpleNamespace):
    """Unset fields read as None (dev/rwm/tests/test_policy_model_ownership.py)."""

    def __getattr__(self, name):
        return None


def _live_class():
    if not os.path.isdir(SANA_ZEKAI_MERGE_REPO):
        pytest.skip(f"no rwm/zekai-merge checkout at {SANA_ZEKAI_MERGE_REPO} (set SANA_ZEKAI_MERGE_REPO)")
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    if SANA_ZEKAI_MERGE_REPO not in sys.path:
        sys.path.insert(0, SANA_ZEKAI_MERGE_REPO)
    module = pytest.importorskip(
        "dev.rwm.diffusion.model.nets.sana_qwennext_action_policy",
        reason="Sana rwm/zekai-merge checkout not importable (set SANA_ZEKAI_MERGE_REPO)",
    )
    if not os.path.abspath(module.__file__).startswith(os.path.abspath(SANA_ZEKAI_MERGE_REPO)):
        pytest.skip(
            f"the dev package is already imported from another Sana checkout ({module.__file__}); "
            "run this file on its own for the live parity"
        )
    live = module.SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewPolicy
    if not hasattr(live, "_rope_mode"):
        pytest.skip("this Sana checkout predates model.extra.rope (b13415841)")
    return live


def _twins(rope):
    """(live, mirror) with identical random weights; ``rope`` None = undeclared."""

    Live = _live_class()
    torch.manual_seed(20260923)
    kwargs = dict(
        depth=DEPTH, hidden_size=HIDDEN, patch_size=(1, 1, 1), num_heads=HEADS, in_channels=IN_CH,
        caption_channels=CAP_CH, model_max_length=MML, linear_head_dim=LHD, softmax_head_dim=SHD, softmax_ratio=0.5,
        attn_res_block_size=2, use_time_conditioning=False, pred_sigma=False, y_norm=True, cross_norm=True,
    )
    extra = {"action_dim": ROBOT_DIM, "state_dim": ROBOT_DIM}
    if rope is not None:
        extra["rope"] = rope
    # a 2026-09-23 yaml: no multiview_spatial_rope_* keys (pyrallis rejects them on this tree)
    config = _Cfg(model=_Cfg(extra=extra), vae=_Cfg(vae_stride=[8, 32, 32]))
    live = Live(config=config, **kwargs)
    if getattr(live, "use_xformers_cross_attention", False):
        pytest.skip("the live model kept xformers cross-attention (imported before DISABLE_XFORMERS=1 took effect)")
    with torch.no_grad():
        for parameter in live.parameters():
            parameter.uniform_(-0.05, 0.05)
    live.eval()
    policy_config = PolicyConfig.from_sana_kwargs(config=config, **kwargs)
    assert policy_config.rope == rope and policy_config.multiview_spatial_rope_layout == "semantic_2x2"
    assert policy_config.multiview_spatial_rope_tile_shape == (15, 30)
    mirror = PolicyModel(policy_config)
    load_policy_state_dict(mirror, live.state_dict())
    return live, mirror.eval()


def _inputs(*, views: int, groups: int, stride: int = 1, batch: int = 2, fps: float = 25.0, seed: int = 7) -> dict:
    torch.manual_seed(seed)
    steps = (FRAMES - 1) * STEPS_PER_FRAME * stride
    if views == 1:
        x = torch.randn(batch, IN_CH, FRAMES, CANVAS_H, CANVAS_W)
        shapes = [[CANVAS_H, CANVAS_W]]
    else:
        x = torch.randn(batch, IN_CH, FRAMES, 1, views * VIEW_H * VIEW_W)
        shapes = [[VIEW_H, VIEW_W]] * views
    timestep = torch.zeros(batch, 1, FRAMES)
    timestep[:, :, 1:] = NOISY_T
    y = torch.randn(batch, groups, 1, MML, CAP_CH)
    mask = torch.ones(batch, groups, MML, dtype=torch.int16)
    mask[..., 5:] = 0
    action_mask = torch.ones(batch, steps, ROBOT_DIM, dtype=torch.bool)
    action_mask[:, :, 66:] = False
    state_mask = torch.ones(batch, ROBOT_DIM, dtype=torch.bool)
    state_mask[:, 58:66] = False
    data_info = {
        "rwm_task": "policy",
        "view_count": torch.full((batch,), views, dtype=torch.long),
        "view_latent_shape": torch.tensor([shapes] * batch),
        "view_slot_ids": torch.tensor((0, 2, 3)[:views]),
        "model_fps": torch.full((batch,), fps),
        "initial_state80": torch.randn(batch, ROBOT_DIM),
        "initial_state_condition_mask80": state_mask,
        "action80": torch.randn(batch, steps, ROBOT_DIM),
        "action_mask80": action_mask,
        "action_timestep": torch.full((batch, steps), NOISY_T),
        "camera_conditioning_enabled": False,
    }
    if stride != 1:
        data_info["video_frame_stride"] = torch.full((batch,), stride, dtype=torch.long)
    return {"x": x, "timestep": timestep, "y": y, "mask": mask, "data_info": data_info}


def _run(model, inputs: dict) -> dict:
    with torch.no_grad():
        return model(
            inputs["x"].clone(), inputs["timestep"].clone(), inputs["y"].clone(), mask=inputs["mask"].clone(),
            data_info=copy.deepcopy(inputs["data_info"]),
        )


def _assert_bitwise(live, mirror, inputs):
    out_live, out_mirror = _run(live, inputs), _run(mirror, inputs)
    assert out_mirror["x"].shape == out_live["x"].shape == inputs["x"].shape
    assert out_mirror["action_pred"].shape == out_live["action_pred"].shape
    torch.testing.assert_close(out_mirror["x"], out_live["x"], rtol=0, atol=0)
    torch.testing.assert_close(out_mirror["action_pred"], out_live["action_pred"], rtol=0, atol=0)
    assert out_live["action_pred"].abs().sum() > 0
    return out_live


@pytest.mark.parametrize("rope", ["aligned", "independent"])
@pytest.mark.parametrize("stride", [1, 2, 4])
def test_strip_forward_is_bitwise_in_both_rope_modes(rope, stride):
    live, mirror = _twins(rope)
    _assert_bitwise(live, mirror, _inputs(views=3, groups=4, stride=stride, seed=11 + stride))


@pytest.mark.parametrize("rope", ["aligned", "independent"])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("stride", [1, 4])
def test_single_stream_forward_is_bitwise_for_one_and_two_text_groups(rope, groups, stride):
    live, mirror = _twins(rope)
    _assert_bitwise(live, mirror, _inputs(views=1, groups=groups, stride=stride, seed=3 + groups + stride))


def test_the_modes_really_differ_and_undeclared_dense_is_aligned():
    live_aligned, mirror_aligned = _twins("aligned")
    _, mirror_independent = _twins("independent")
    live_undeclared, mirror_undeclared = _twins(None)
    for stride in (1, 4):
        inputs = _inputs(views=3, groups=4, stride=stride)
        aligned = _run(mirror_aligned, inputs)["action_pred"]
        independent = _run(mirror_independent, inputs)["action_pred"]
        assert not torch.equal(aligned, independent)
    dense = _inputs(views=3, groups=4)
    torch.testing.assert_close(_run(mirror_undeclared, dense)["x"], _run(live_undeclared, dense)["x"], rtol=0, atol=0)
    torch.testing.assert_close(_run(mirror_undeclared, dense)["action_pred"], _run(mirror_aligned, dense)["action_pred"], rtol=0, atol=0)
    # the live model refuses an undeclared strided batch; the mirror serves it with the pre-mode table of the run's era
    strided = _inputs(views=3, groups=4, stride=2)
    with pytest.raises(ValueError, match="undeclared"):
        _run(live_undeclared, strided)
    assert torch.isfinite(_run(mirror_undeclared, strided)["action_pred"]).all()


def test_g1_single_stream_is_not_the_g2_payload():
    live, mirror = _twins("independent")
    one, two = _inputs(views=1, groups=1), _inputs(views=1, groups=2)
    two["y"][:, :1] = one["y"]
    two["mask"][:, :1] = one["mask"]
    assert not torch.equal(_run(mirror, one)["action_pred"], _run(mirror, two)["action_pred"])


def test_bad_group_counts_are_refused_like_the_live_model():
    live, mirror = _twins("aligned")
    for views, groups in ((3, 2), (3, 1), (1, 3)):
        inputs = _inputs(views=views, groups=groups)
        with pytest.raises(ValueError):
            _run(live, inputs)
        with pytest.raises(ValueError):
            _run(mirror, inputs)
