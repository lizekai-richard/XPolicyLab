"""Live parity of ``model.extra.sana_pixel_pad: masked`` (rwm/zekai-merge 319d3f666) against the tree the padmask
checkpoints trained on (``SANA_PADMASK_REPO``, default the worktree ~/zekail/Sana_2ee450e05 = the runtime commit of
logits/sft_robodojo_eefabs_sana_pixel_320x512_vanilla34k_aligned_f33fps8_fixed_resize[_padmask]).

The live bidirectional policy and this mirror get identical random weights and one V = 1 sana_pixel canvas batch (G = 1
text, aligned RoPE, strided video); the masked and the unmasked forwards must agree bit for bit -- the video prediction
(zero on the black quadrant when masked) and the action prediction -- on a tiny even grid and on the real 10 x 16 grid of
the 320x512 canvas. Run on its own like the other live parity files (the first file of a session owns ``dev``).
"""

from __future__ import annotations

import copy
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from _tiny_policy import CAP_CH, DEPTH, HEADS, HIDDEN, IN_CH, LHD, MML, NOISY_T, ROBOT_DIM, SHD  # noqa: E402

from sana_wam_min.policy_model.checkpoint import load_policy_state_dict  # noqa: E402
from sana_wam_min.policy_model.config import PolicyConfig  # noqa: E402
from sana_wam_min.policy_model.geometry import sana_pixel_real_token_mask  # noqa: E402
from sana_wam_min.policy_model.model import PolicyModel  # noqa: E402

SANA_PADMASK_REPO = os.environ.get("SANA_PADMASK_REPO", os.path.expanduser("~/zekail/Sana_2ee450e05"))
FRAMES, STRIDE, STEPS_PER_FRAME = 3, 4, 8          # video_fps 8 of a 25-row tier: every 4th row, 8 action rows each


class _Cfg(SimpleNamespace):
    """Unset fields read as None (dev/rwm/tests/test_policy_model_ownership.py)."""

    def __getattr__(self, name):
        return None


def _live_class():
    if not os.path.isdir(SANA_PADMASK_REPO):
        pytest.skip(f"no Sana checkout at {SANA_PADMASK_REPO} (set SANA_PADMASK_REPO)")
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    if SANA_PADMASK_REPO not in sys.path:
        sys.path.insert(0, SANA_PADMASK_REPO)
    module = pytest.importorskip(
        "dev.rwm.diffusion.model.nets.sana_qwennext_action_policy",
        reason="Sana checkout not importable (set SANA_PADMASK_REPO)",
    )
    if not os.path.abspath(module.__file__).startswith(os.path.abspath(SANA_PADMASK_REPO)):
        pytest.skip(
            f"the dev package is already imported from another Sana checkout ({module.__file__}); "
            "run this file on its own for the live parity"
        )
    live = module.SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewPolicy
    if not getattr(live, "_sana_pixel_pad_maskable", False):
        pytest.skip("this Sana checkout predates model.extra.sana_pixel_pad (319d3f666)")
    return live


def _twins(pad: str):
    """(live, mirror) with identical random weights for ``model.extra.sana_pixel_pad: <pad>``."""

    Live = _live_class()
    torch.manual_seed(20260927)
    kwargs = dict(
        depth=DEPTH, hidden_size=HIDDEN, patch_size=(1, 1, 1), num_heads=HEADS, in_channels=IN_CH,
        caption_channels=CAP_CH, model_max_length=MML, linear_head_dim=LHD, softmax_head_dim=SHD, softmax_ratio=0.5,
        attn_res_block_size=2, use_time_conditioning=False, pred_sigma=False, y_norm=True, cross_norm=True,
    )
    extra = {"action_dim": ROBOT_DIM, "state_dim": ROBOT_DIM, "rope": "aligned", "sana_pixel_pad": pad}
    config = _Cfg(model=_Cfg(extra=extra), data=_Cfg(extra={"multiview": "sana_pixel"}), vae=_Cfg(vae_stride=[8, 32, 32]))
    live = Live(config=config, **kwargs)
    assert live.sana_pixel_pad == pad
    if getattr(live, "use_xformers_cross_attention", False):
        pytest.skip("the live model kept xformers cross-attention (imported before DISABLE_XFORMERS=1 took effect)")
    with torch.no_grad():
        for parameter in live.parameters():
            parameter.uniform_(-0.05, 0.05)
    live.eval()
    policy_config = PolicyConfig.from_sana_kwargs(config=config, **kwargs)
    assert (policy_config.sana_pixel_pad, policy_config.rope) == (pad, "aligned")
    mirror = PolicyModel(policy_config)
    load_policy_state_dict(mirror, live.state_dict())
    return live, mirror.eval()


def _inputs(grid: tuple[int, int], *, batch: int = 2, seed: int = 11) -> dict:
    torch.manual_seed(seed)
    height, width = grid
    steps = (FRAMES - 1) * STEPS_PER_FRAME * STRIDE
    timestep = torch.zeros(batch, 1, FRAMES)
    timestep[:, :, 1:] = NOISY_T
    mask = torch.ones(batch, 1, MML, dtype=torch.int16)
    mask[..., 5:] = 0
    action_mask = torch.ones(batch, steps, ROBOT_DIM, dtype=torch.bool)
    action_mask[:, :, 66:] = False
    state_mask = torch.ones(batch, ROBOT_DIM, dtype=torch.bool)
    state_mask[:, 58:66] = False
    data_info = {
        "rwm_task": "policy",
        "view_count": torch.full((batch,), 1, dtype=torch.long),
        "view_latent_shape": torch.tensor([[[height, width]]] * batch),
        "view_slot_ids": torch.tensor((0,)),
        "model_fps": torch.full((batch,), 25.0),
        "video_frame_stride": torch.full((batch,), STRIDE, dtype=torch.long),
        "initial_state80": torch.randn(batch, ROBOT_DIM),
        "initial_state_condition_mask80": state_mask,
        "action80": torch.randn(batch, steps, ROBOT_DIM),
        "action_mask80": action_mask,
        "action_timestep": torch.full((batch, steps), NOISY_T),
        "camera_conditioning_enabled": False,
    }
    return {
        "x": torch.randn(batch, IN_CH, FRAMES, height, width),
        "timestep": timestep,
        "y": torch.randn(batch, 1, 1, MML, CAP_CH),
        "mask": mask,
        "data_info": data_info,
    }


def _run(model, inputs: dict) -> dict:
    with torch.no_grad():
        return model(
            inputs["x"].clone(), inputs["timestep"].clone(), inputs["y"].clone(), mask=inputs["mask"].clone(),
            data_info=copy.deepcopy(inputs["data_info"]),
        )


@pytest.mark.parametrize("grid", [(4, 6), (10, 16)])
@pytest.mark.parametrize("pad", ["masked", "unmasked"])
def test_the_canvas_forward_is_bitwise(pad, grid):
    live, mirror = _twins(pad)
    inputs = _inputs(grid)
    expected, ours = _run(live, inputs), _run(mirror, inputs)
    torch.testing.assert_close(ours["action_pred"], expected["action_pred"], rtol=0, atol=0)
    torch.testing.assert_close(ours["x"], expected["x"], rtol=0, atol=0)
    if pad == "masked":
        assert torch.count_nonzero(ours["x"][..., ~sana_pixel_real_token_mask(*grid)]) == 0


def test_masked_outputs_do_not_see_the_pad_in_either_model():
    live, mirror = _twins("masked")
    inputs = _inputs((10, 16))
    poked = copy.deepcopy(inputs)
    pad = ~sana_pixel_real_token_mask(10, 16)
    poked["x"][..., pad] = 10 * torch.randn_like(poked["x"][..., pad])
    for model in (live, mirror):
        base, again = _run(model, inputs), _run(model, poked)
        assert torch.equal(base["action_pred"], again["action_pred"]) and torch.equal(base["x"], again["x"])
