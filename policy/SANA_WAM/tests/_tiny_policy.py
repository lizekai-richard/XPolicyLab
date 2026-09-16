"""Tiny PolicyConfig and forward inputs shared by the CPU tests (mirror parity test shapes)."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

# policy/SANA_WAM (plain ``sana_wam_min`` imports) and the XPolicyLab parent (``XPolicyLab.policy.SANA_WAM`` imports).
ADAPTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_XPOLICYLAB_PARENT = os.path.abspath(os.path.join(ADAPTER_DIR, "..", "..", ".."))
for _p in (ADAPTER_DIR, _XPOLICYLAB_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sana_wam_min.policy_model.config import PolicyConfig  # noqa: E402

HIDDEN, DEPTH, HEADS, IN_CH, CAP_CH, MML = 64, 4, 2, 4, 32, 8
LHD = SHD = 32
VIEW_H, VIEW_W = 2, 3
FRAMES = 3
NOISY_T = 700.0
ROBOT_DIM = 80
INPUT_SIZE = 2
TILE = (VIEW_H + 1, VIEW_W + 2)


def tiny_policy_config(tile: tuple[int, int] = TILE) -> PolicyConfig:
    config = SimpleNamespace(
        model=SimpleNamespace(
            multiview_spatial_rope_layout="semantic_2x2",
            multiview_spatial_rope_tile_shape=tile,
        ),
        vae=SimpleNamespace(vae_stride=[8, 32, 32]),
    )
    return PolicyConfig.from_sana_kwargs(
        depth=DEPTH,
        hidden_size=HIDDEN,
        patch_size=(1, 1, 1),
        num_heads=HEADS,
        input_size=INPUT_SIZE,
        in_channels=IN_CH,
        caption_channels=CAP_CH,
        model_max_length=MML,
        linear_head_dim=LHD,
        softmax_head_dim=SHD,
        softmax_ratio=0.5,
        attn_res_block_size=2,
        use_time_conditioning=False,
        pred_sigma=False,
        y_norm=True,
        cross_norm=True,
        config=config,
    )


def tiny_inputs(batch: int, views: int, fps: float, seed: int) -> dict:
    torch.manual_seed(seed)
    steps = (FRAMES - 1) * 8
    if views == 1:
        x = torch.randn(batch, IN_CH, FRAMES, VIEW_H, VIEW_W)
    else:
        x = torch.randn(batch, IN_CH, FRAMES, 1, views * VIEW_H * VIEW_W)
    timestep = torch.zeros(batch, 1, FRAMES)
    timestep[:, :, 1:] = NOISY_T
    y = torch.randn(batch, views + 1, 1, MML, CAP_CH)
    # Text mask in the encoder's [B, G, 1, 1, L] layout.
    mask = torch.ones(batch, views + 1, 1, 1, MML, dtype=torch.int64)
    mask[..., 5:] = 0
    action_mask = torch.ones(batch, steps, ROBOT_DIM, dtype=torch.bool)
    action_mask[:, :, 66:] = False
    state_mask = torch.ones(batch, ROBOT_DIM, dtype=torch.bool)
    state_mask[:, 58:66] = False
    data_info = {
        "rwm_task": "policy",
        "view_count": torch.full((batch,), views, dtype=torch.long),
        "view_latent_shape": torch.tensor([[[VIEW_H, VIEW_W]] * views] * batch),
        "view_slot_ids": torch.tensor((0, 2, 3)[:views]),
        "model_fps": torch.full((batch,), fps),
        "initial_state80": torch.randn(batch, ROBOT_DIM),
        "initial_state_condition_mask80": state_mask,
        "action80": torch.randn(batch, steps, ROBOT_DIM),
        "action_mask80": action_mask,
        "action_timestep": torch.full((batch, steps), NOISY_T),
        "camera_conditioning_enabled": False,
    }
    return {"x": x, "timestep": timestep, "y": y, "mask": mask, "data_info": data_info}
