"""Live parity of the sana_latent STRIP (V = 3 views encoded on their own, G = 4 text rows) against a robot_sft training tree
(``SANA_LATENT_REPO``, default the worktree ~/zekail/Sana_87e34d163 = the runtime commit of
logits/sft_robodojo_eefabs_sana_latent_256x320_vanilla34k_aligned_f33fps8_fixed_resize).

Since rwm/zekai-merge 6da565230 the strip's semantic 2x2 tile is the code constant (8, 16) with no yaml key; the mirror takes
it from ``strip_spatial_rope_tile_shape_from_train_config`` (8, 16 for a ``data.extra.robot_sft`` config), as the server does.
The live policy and the mirror get identical random weights and one strip batch at the real per-view latent grid (a 256x320
view = 8 x 10 latents) and a small one; the video prediction and the action prediction must agree bit for bit with the
aligned RoPE at video frame strides 1 and 4 (f33 at video_fps 8). test_zekai_merge_head_parity.py checks the (15, 30) era
(b13415841) instead. Run on its own like the other live parity files (the first file of a session owns ``dev``).
"""

from __future__ import annotations

import copy
import dataclasses
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from _tiny_policy import CAP_CH, DEPTH, FRAMES, HEADS, HIDDEN, IN_CH, LHD, MML, NOISY_T, ROBOT_DIM, SHD  # noqa: E402

from sana_wam_min import config as wam_config  # noqa: E402
from sana_wam_min.policy_model.checkpoint import load_policy_state_dict  # noqa: E402
from sana_wam_min.policy_model.config import PolicyConfig  # noqa: E402
from sana_wam_min.policy_model.model import PolicyModel  # noqa: E402

SANA_LATENT_REPO = os.environ.get("SANA_LATENT_REPO", os.path.expanduser("~/zekail/Sana_87e34d163"))
STEPS_PER_FRAME = 8
# the tile a robot_sft strip config resolves to (the server's path); a 2026-10 robot_sft yaml carries no tile key
ROBOT_SFT_TILE = wam_config.strip_spatial_rope_tile_shape_from_train_config({"data": {"extra": {"robot_sft": {}}}})


class _Cfg(SimpleNamespace):
    """Unset fields read as None (dev/rwm/tests/test_policy_model_ownership.py)."""

    def __getattr__(self, name):
        return None


def _live_module():
    if not os.path.isdir(SANA_LATENT_REPO):
        pytest.skip(f"no Sana checkout at {SANA_LATENT_REPO} (set SANA_LATENT_REPO)")
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    if SANA_LATENT_REPO not in sys.path:
        sys.path.insert(0, SANA_LATENT_REPO)
    module = pytest.importorskip(
        "dev.rwm.diffusion.model.nets.sana_qwennext_action_policy",
        reason="Sana checkout not importable (set SANA_LATENT_REPO)",
    )
    if not os.path.abspath(module.__file__).startswith(os.path.abspath(SANA_LATENT_REPO)):
        pytest.skip(
            f"the dev package is already imported from another Sana checkout ({module.__file__}); "
            "run this file on its own for the live parity"
        )
    return module


def _twins(rope: str = "aligned"):
    """(live, mirror) with identical random weights; the mirror's strip tile is the robot_sft one the server resolves."""

    module = _live_module()
    Live = module.SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewPolicy
    torch.manual_seed(20261001)
    kwargs = dict(
        depth=DEPTH, hidden_size=HIDDEN, patch_size=(1, 1, 1), num_heads=HEADS, in_channels=IN_CH,
        caption_channels=CAP_CH, model_max_length=MML, linear_head_dim=LHD, softmax_head_dim=SHD, softmax_ratio=0.5,
        attn_res_block_size=2, use_time_conditioning=False, pred_sigma=False, y_norm=True, cross_norm=True,
    )
    extra = {"action_dim": ROBOT_DIM, "state_dim": ROBOT_DIM, "rope": rope}
    config = _Cfg(model=_Cfg(extra=extra), data=_Cfg(extra={"multiview": "sana_latent"}), vae=_Cfg(vae_stride=[8, 32, 32]))
    live = Live(config=config, **kwargs)
    if getattr(live, "use_xformers_cross_attention", False):
        pytest.skip("the live model kept xformers cross-attention (imported before DISABLE_XFORMERS=1 took effect)")
    with torch.no_grad():
        for parameter in live.parameters():
            parameter.uniform_(-0.05, 0.05)
    live.eval()
    policy_config = dataclasses.replace(
        PolicyConfig.from_sana_kwargs(config=config, **kwargs), multiview_spatial_rope_tile_shape=ROBOT_SFT_TILE
    ).validate()
    assert policy_config.rope == rope and policy_config.multiview_spatial_rope_layout == "semantic_2x2"
    mirror = PolicyModel(policy_config)
    load_policy_state_dict(mirror, live.state_dict())
    return live, mirror.eval()


def _inputs(view_hw: tuple[int, int], *, stride: int, batch: int = 2, seed: int = 7) -> dict:
    torch.manual_seed(seed)
    views, groups = 3, 4
    height, width = view_hw
    steps = (FRAMES - 1) * STEPS_PER_FRAME * stride
    timestep = torch.zeros(batch, 1, FRAMES)
    timestep[:, :, 1:] = NOISY_T
    mask = torch.ones(batch, groups, MML, dtype=torch.int16)
    mask[..., 5:] = 0
    action_mask = torch.ones(batch, steps, ROBOT_DIM, dtype=torch.bool)
    action_mask[:, :, 66:] = False
    state_mask = torch.ones(batch, ROBOT_DIM, dtype=torch.bool)
    state_mask[:, 58:66] = False
    data_info = {
        "rwm_task": "policy",
        "view_count": torch.full((batch,), views, dtype=torch.long),
        "view_latent_shape": torch.tensor([[[height, width]] * views] * batch),
        "view_slot_ids": torch.tensor((0, 2, 3)),
        "model_fps": torch.full((batch,), 25.0),
        "initial_state80": torch.randn(batch, ROBOT_DIM),
        "initial_state_condition_mask80": state_mask,
        "action80": torch.randn(batch, steps, ROBOT_DIM),
        "action_mask80": action_mask,
        "action_timestep": torch.full((batch, steps), NOISY_T),
        "camera_conditioning_enabled": False,
    }
    if stride != 1:
        data_info["video_frame_stride"] = torch.full((batch,), stride, dtype=torch.long)
    return {
        "x": torch.randn(batch, IN_CH, FRAMES, 1, views * height * width),
        "timestep": timestep,
        "y": torch.randn(batch, groups, 1, MML, CAP_CH),
        "mask": mask,
        "data_info": data_info,
    }


def _run(model, inputs: dict) -> dict:
    with torch.no_grad():
        return model(
            inputs["x"].clone(), inputs["timestep"].clone(), inputs["y"].clone(), mask=inputs["mask"].clone(),
            data_info=copy.deepcopy(inputs["data_info"]),
        )


def test_the_live_tree_uses_the_robot_sft_tile():
    _live_module()
    from dev.rwm.diffusion import multiview_utils

    assert ROBOT_SFT_TILE == (8, 16) == tuple(multiview_utils.MULTIVIEW_SPATIAL_ROPE_TILE_SHAPE)


@pytest.mark.parametrize("view_hw", [(8, 10), (2, 3)])
@pytest.mark.parametrize("stride", [1, 4])
def test_the_strip_forward_is_bitwise(view_hw, stride):
    live, mirror = _twins("aligned")
    inputs = _inputs(view_hw, stride=stride, seed=11 + stride + view_hw[0])
    expected, ours = _run(live, inputs), _run(mirror, inputs)
    assert ours["x"].shape == expected["x"].shape == inputs["x"].shape
    torch.testing.assert_close(ours["x"], expected["x"], rtol=0, atol=0)
    torch.testing.assert_close(ours["action_pred"], expected["action_pred"], rtol=0, atol=0)
    assert expected["action_pred"].abs().sum() > 0


def test_the_legacy_tile_would_not_match():
    """Control: the (15, 30) tile of the pre-6da565230 strips gives other positions on this tree."""

    live, mirror = _twins("aligned")
    legacy = PolicyModel(dataclasses.replace(mirror.policy_config, multiview_spatial_rope_tile_shape=(15, 30)).validate())
    load_policy_state_dict(legacy, live.state_dict())
    inputs = _inputs((8, 10), stride=4)
    assert not torch.equal(_run(legacy.eval(), inputs)["action_pred"], _run(live, inputs)["action_pred"])
