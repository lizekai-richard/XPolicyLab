"""The video expert: the Sana-Video AttnRes trunk's non-block modules plus its (later detached) block stack.

Mirrors the parts of ``SanaRWMVideoQwenNextSubAttnResV2SelfFlow`` the MoT policy drives itself
(``_video_prepare`` in ``sana_qwennext_mot_policy.py``): patch embedding, timestep MLPs, the two physical-time RoPE
modules, AttnRes and the video output head. In the legacy (606e48dd9) layout the expert also owns its text path
(``y_embedder`` + ``attention_y_norm``, ``with_text=True``); in the current layout those modules were re-parented to
``context_embedder`` and the expert carries none. The blocks are built here with the vendored ``PolicyBlock`` (same
parameter names as the training block) and re-parented into the :class:`MoTLayer` units by :class:`MoTPolicyModel`.
Checkpoint keys: ``video_dit.<child>.*``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sana_wam_min.policy_model.attnres import BlockAttnResV2
from sana_wam_min.policy_model.block import PolicyBlock
from sana_wam_min.policy_model.config import PolicyConfig
from sana_wam_min.policy_model.embeddings import (
    CaptionEmbedder,
    PatchEmbedMS3D,
    RMSNorm,
    T2IFinalLayer,
    TimestepEmbedder,
    t2i_modulate,
)
from sana_wam_min.policy_model.rope import PhysicalTimeWanRotaryPosEmbed, WanRotaryPosEmbed


def final_layer_frame_aware(module: T2IFinalLayer, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``T2IFinalLayer.forward_frame_aware``: per-frame modulation, ``t`` is ``[B, 1, F, D]``, ``x`` is ``[B, N, C]`` with ``N`` a multiple of ``F``."""

    batch, tokens, channels = x.shape
    num_frames = t.shape[2]
    shift, scale = (module.scale_shift_table[None, None, :, :] + t.transpose(1, 2)).chunk(2, dim=-2)
    x = t2i_modulate(module.norm_final(x).reshape(batch, num_frames, -1, channels), shift, scale).reshape(
        batch, tokens, channels
    )
    return module.linear(x)


class VideoExpert(nn.Module):
    """The MoT policy's ``video_dit``: the trunk without a forward of its own."""

    def __init__(self, config: PolicyConfig, *, with_text: bool = False) -> None:
        super().__init__()
        config = config.validate()
        self.with_text = bool(with_text)
        self.policy_config = config
        self.hidden_size = config.hidden_size
        self.depth = config.depth
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.patch_size = tuple(config.patch_size)
        self.softmax_attn_type = "GatedSoftmaxAttention"
        self.block_attn_types = list(config.block_attn_types)
        self.attn_res_block_size = config.attn_res_block_size
        self.timestep_norm_scale_factor = config.timestep_norm_scale_factor
        self.y_norm = config.y_norm
        self.use_pe = True

        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True))
        if self.with_text:
            self.y_embedder = CaptionEmbedder(config.caption_channels, config.hidden_size)
            self.attention_y_norm = RMSNorm(config.hidden_size, scale_factor=config.y_norm_scale_factor, eps=config.norm_eps)
        self.x_embedder = PatchEmbedMS3D(config.patch_size, config.in_channels, config.hidden_size)
        self.final_layer = T2IFinalLayer(config.hidden_size, config.patch_size, config.out_channels)
        self.attn_res = BlockAttnResV2(config.hidden_size, config.depth, config.attn_res_block_size)
        self.rope_linear = PhysicalTimeWanRotaryPosEmbed(
            WanRotaryPosEmbed(config.linear_head_dim, max_seq_len=1024), attention_head_dim=config.linear_head_dim
        )
        self.rope_softmax = PhysicalTimeWanRotaryPosEmbed(
            WanRotaryPosEmbed(config.softmax_head_dim, max_seq_len=1024), attention_head_dim=config.softmax_head_dim
        )
        self.blocks = nn.ModuleList(
            PolicyBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                attention_kind=kind,
                linear_head_dim=config.linear_head_dim,
                softmax_head_dim=config.softmax_head_dim,
                fp32_attention=config.fp32_attention,
            )
            for kind in self.block_attn_types
        )
        for block in self.blocks:
            block.cross_attn.set_use_xformers(False)
        self.f = self.h = self.w = 0

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[B, F*H*W, C_out] -> [B, C_out, F, H, W]`` for the P1 patch."""

        patch_f, patch_h, patch_w = self.patch_size
        tokens = tokens.reshape(tokens.shape[0], self.f, self.h, self.w, patch_f, patch_h, patch_w, self.out_channels)
        tokens = torch.einsum("nfhwopqc->ncfohpwq", tokens)
        return tokens.reshape(
            tokens.shape[0], self.out_channels, self.f * patch_f, self.h * patch_h, self.w * patch_w
        )


__all__ = ["VideoExpert", "final_layer_frame_aware"]
