"""Tiny MoT configs and forward inputs shared by the CPU tests (the dims of Sana's manual_cpu_selfcheck_mot_policy.py)."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

# policy/SANA_MOT (plain ``sana_mot_min`` imports) and the XPolicyLab parent (``XPolicyLab.policy.SANA_MOT`` imports).
ADAPTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_XPOLICYLAB_PARENT = os.path.abspath(os.path.join(ADAPTER_DIR, "..", "..", ".."))
for _p in (ADAPTER_DIR, _XPOLICYLAB_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sana_mot_min  # noqa: E402,F401  (puts policy/SANA_WAM on sys.path)
from sana_wam_min.multiview import pack_spatial_views  # noqa: E402
from sana_wam_min.policy_model.config import PolicyConfig  # noqa: E402
from sana_mot_min.mot_model.model import MoTConfig  # noqa: E402

HIDDEN, DEPTH, HEADS, IN_CH, CAP_CH, MML = 64, 4, 4, 4, 32, 8
LHD = SHD = 16
ACTION_HIDDEN, ACTION_CROSS_HEADS = 32, 4
B, F, Hh, Ww = 2, 2, 2, 3
A = (F - 1) * 8
VIEWS = 3
SLOTS = (0, 2, 3)
TILE = (15, 30)
NOISY_T = 500.0
ROBOT_DIM = 80
INPUT_SIZE = 2
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
CONFIG_MULTIVIEW = os.path.join(FIXTURES, "config_multiview.yaml")
CONFIG_CANVAS = os.path.join(FIXTURES, "config_canvas.yaml")
CONFIG_F33FPS8 = os.path.join(FIXTURES, "config_f33fps8.yaml")
CONFIG_NSC_F25 = os.path.join(FIXTURES, "config_nsc_f25_openwam.yaml")
CONFIG_NSC_F25_SAC = os.path.join(FIXTURES, "config_nsc_f25_openwam_sac.yaml")


def tiny_video_config(fp32_attention: bool = False) -> PolicyConfig:
    config = SimpleNamespace(
        model=SimpleNamespace(multiview_spatial_rope_layout="local_reset", multiview_spatial_rope_tile_shape=(15, 30)),
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
        mlp_ratio=2.0,
        linear_head_dim=LHD,
        softmax_head_dim=SHD,
        softmax_layer_indices=[1, 3],
        attn_res_block_size=2,
        qk_norm=True,
        cross_norm=True,
        y_norm=True,
        pred_sigma=False,
        use_fp32_attention=fp32_attention,
        config=config,
    )


def tiny_mot_config(
    state_as_context: bool = False,
    fp32_attention: bool = False,
    *,
    video_layout: str = "multiview",
    rope_layout: str = "semantic_2x2",
    context_layout: str = "context_embedder",
) -> MoTConfig:
    return MoTConfig(
        video=tiny_video_config(fp32_attention),
        action_hidden_size=ACTION_HIDDEN,
        action_mlp_ratio=4.0,
        action_cross_attn_heads=ACTION_CROSS_HEADS,
        action_rope_theta=10000.0,
        action_attn_res_block_size=2,
        action_state_as_context=state_as_context,
        video_layout=video_layout,
        multiview_spatial_rope_layout=rope_layout,
        multiview_spatial_rope_tile_shape=TILE,
        context_layout=context_layout,
    )


def tiny_inputs(
    view_count: int = VIEWS,
    groups: int | None = None,
    seed: int = 1,
    slots=SLOTS,
    stride: int = 1,
    dtype: torch.dtype = torch.float64,
) -> tuple:
    """``(x, timestep, y, mask, data_info)``: V views packed as the strip (V > 1) or the native grid, token-group text
    ``[B, G, 1, L, C]`` (G defaults to V + 1), ``(F - 1) * 8 * stride`` action rows."""

    g = torch.Generator().manual_seed(seed)
    groups = view_count + 1 if groups is None else groups
    views = [torch.randn(B, IN_CH, F, Hh, Ww, dtype=dtype, generator=g) for _ in range(view_count)]
    x = views[0] if view_count == 1 else pack_spatial_views(views)[0]
    timestep = torch.zeros(B, 1, F, dtype=dtype)
    timestep[:, :, 1:] = NOISY_T
    y = torch.randn(B, groups, 1, MML, CAP_CH, dtype=dtype, generator=g)
    mask = torch.ones(B, groups, 1, 1, MML, dtype=dtype)
    mask[..., -2:] = 0
    rows = (F - 1) * 8 * stride
    data_info = {
        "model_fps": 16.0,
        "action80": torch.randn(B, rows, ROBOT_DIM, dtype=dtype, generator=g),
        "action_mask80": torch.ones(B, rows, ROBOT_DIM, dtype=torch.bool),
        "action_timestep": torch.full((B, rows), NOISY_T, dtype=dtype),
        "initial_state80": torch.randn(B, ROBOT_DIM, dtype=dtype, generator=g),
        "initial_state_condition_mask80": torch.ones(B, ROBOT_DIM, dtype=torch.bool),
    }
    data_info["action_mask80"][:, :, 60:] = False
    if stride != 1:
        data_info["video_frame_stride"] = torch.full((B,), stride, dtype=torch.int64)
    if view_count > 1:
        data_info.update(
            view_count=torch.full((B,), view_count, dtype=torch.int64),
            view_latent_shape=torch.tensor([[[Hh, Ww]] * view_count] * B, dtype=torch.int64),
            view_slot_ids=torch.tensor([list(slots)] * B, dtype=torch.int64),
            camera_conditioning_enabled=torch.zeros(B, dtype=torch.bool),
        )
    return x, timestep, y, mask, data_info
