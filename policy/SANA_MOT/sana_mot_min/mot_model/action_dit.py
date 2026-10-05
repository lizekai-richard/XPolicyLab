"""The action expert (``ActionDiT``) and its per-layer parameter holder (``MoTActionBlock``).

Port of ``dev/rwm/diffusion/model/layers/mot_action_dit.py``: the fresh ~1B expert at hidden 1024 with the same depth
and softmax-layer schedule as the video expert and its own AttnRes. Two checkpoint layouts are modeled:

* current (rwm/mot @ b3b9e0e9e .. 71ac93f43): the expert owns no text path -- ``ContextEmbedder`` (``context.py``)
  projects its cross-attention context, at the expert's own width (1024; ``multiview: sana_latent``) or, since
  2026-09-22, at the video width (2560) of the ONE caption embedding the canvas modes share (``context_dim`` of every
  block's ``cross_attn.kv_linear``); ``state_embed`` exists only in the default state mode. The per-layer attention
  projections are ``attn`` (rwm/mot 3058e5785, 2026-09-18; ``attn_head`` before -- the loader renames old keys).
* legacy (rwm/mot @ 606e48dd9, ``legacy_context=True``): the expert owns its context MLP ``context_mlp`` over the raw
  prompt features (``project_context``) and, under ``state_as_context``, ``state_context_embed`` (``build_context``);
  the checkpoints of runs launched before the ContextEmbedder refactor carry these keys.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from sana_wam_min.policy_model.attention import MultiHeadCrossAttention
from sana_wam_min.policy_model.attnres import BlockAttnResV2
from sana_wam_min.policy_model.embeddings import ActionFinalLayer, MaskedProjector, TimestepEmbedder, t2i_modulate
from sana_wam_min.policy_model.feedforward import SwiGLU
from sana_wam_min.robot80 import ROBOT80_DIM

from .heads import MoTGDNExpertHead, MoTSoftmaxExpertHead

ACTION_TEMPORAL_COMPRESSION = 8


def independent_action_rope_at(head_dim: int, positions: torch.Tensor, batch: int, theta: float) -> torch.Tensor:
    """Full-head-dim 1D rotary phases of the action expert's own clock at ``positions`` (``[N]``), shaped
    ``[batch, 1, N, head_dim // 2]`` complex128 (``mot_action_dit.independent_action_rope_at``, 2026-09-21)."""

    positions = positions.to(torch.float64)
    exponent = torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float64) / head_dim
    inverse_frequency = theta ** (-exponent)
    phase = torch.outer(positions, inverse_frequency)
    freqs = torch.polar(torch.ones_like(phase), phase)
    return freqs.view(1, 1, positions.numel(), -1).expand(batch, 1, positions.numel(), -1)


def independent_action_rope(head_dim: int, action_steps: int, batch: int, theta: float, device) -> torch.Tensor:
    """Full-head-dim 1D rotary phases at positions ``0..action_steps-1``, shaped ``[batch, 1, action_steps, head_dim // 2]`` complex128."""

    positions = torch.arange(action_steps, device=device, dtype=torch.float64)
    return independent_action_rope_at(head_dim, positions, batch, theta)


class MoTActionBlock(nn.Module):
    """One action-expert layer's parameters (AdaLN, attention projections, cross-attention, SwiGLU), driven by :class:`MoTLayer`."""

    def __init__(
        self,
        hidden_size: int,
        attn_kind: str,
        shared_dim: int,
        shared_heads: int,
        shared_head_dim: int,
        *,
        cross_attn_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        cross_norm: bool = True,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        if attn_kind not in ("gdn", "softmax"):
            raise ValueError(f"attn_kind must be 'gdn' or 'softmax', got {attn_kind!r}")
        if not cross_norm:
            raise ValueError("the vendored MultiHeadCrossAttention always carries q/k RMSNorm; cross_norm=False is not modeled")
        self.attn_kind = attn_kind
        self.hidden_size = hidden_size
        self.context_dim = hidden_size if context_dim is None else int(context_dim)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        head_cls = MoTGDNExpertHead if attn_kind == "gdn" else MoTSoftmaxExpertHead
        self.attn = head_cls(
            expert_dim=hidden_size,
            shared_dim=shared_dim,
            heads=shared_heads,
            head_dim=shared_head_dim,
            qk_norm=qk_norm,
        )

        self.cross_attn = MultiHeadCrossAttention(
            hidden_size, cross_attn_heads, use_xformers=False, context_dim=self.context_dim
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLU(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            out_features=hidden_size,
        )
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, *args, **kwargs):
        """Refuse a standalone call; :class:`MoTLayer` drives this block's submodules."""

        raise RuntimeError("MoTActionBlock has no standalone forward; it is driven by MoTLayer (mot_model/block.py)")

    def modulation(self, t: torch.Tensor, batch: int):
        """Six-way plain per-batch AdaLN table (state_as_context=True path)."""

        return (self.scale_shift_table[None] + t.reshape(batch, 6, -1)).chunk(6, dim=1)

    def mlp_sublayer(self, h: torch.Tensor, post_state: dict) -> torch.Tensor:
        """MLP residual delta for ``h``, frame-aware when ``post_state`` carries ``num_frames``."""

        if "num_frames" in post_state:
            batch, num_frames, channels = post_state["h"].shape[0], post_state["num_frames"], post_state["channels"]
            tokens = post_state["tokens"]
            mlp_out = self.mlp(
                t2i_modulate(
                    self.norm2(h).reshape(batch, num_frames, -1, channels),
                    post_state["shift_mlp"],
                    post_state["scale_mlp"],
                ).reshape(batch, tokens, channels)
            )
            return (post_state["gate_mlp"] * mlp_out.reshape(batch, num_frames, -1, channels)).reshape(
                batch, tokens, channels
            )
        mlp_out = self.mlp(t2i_modulate(self.norm2(h), post_state["shift_mlp"], post_state["scale_mlp"]))
        return post_state["gate_mlp"] * mlp_out


class ActionDiT(nn.Module):
    """The action expert: robot IO, time/context conditioning, the hybrid block stack and its own AttnRes."""

    def __init__(
        self,
        hidden_size: int = 1024,
        depth: int = 32,
        shared_dim: int = 2560,
        gdn_heads: int = 20,
        gdn_head_dim: int = 128,
        softmax_heads: int = 10,
        softmax_head_dim: int = 256,
        *,
        raw_context_dim: int = 2304,
        cross_attn_heads: int = 8,
        mlp_ratio: float = 4.0,
        softmax_layer_indices: Sequence[int],
        rope_theta: float = 10000.0,
        attn_res_block_size: int = 8,
        qk_norm: bool = True,
        cross_norm: bool = True,
        state_as_context: bool = False,
        legacy_context: bool = False,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.legacy_context = bool(legacy_context)
        self.hidden_size = hidden_size
        self.context_dim = hidden_size if context_dim is None else int(context_dim)
        self.depth = depth
        self.gdn_heads, self.gdn_head_dim = gdn_heads, gdn_head_dim
        self.softmax_heads, self.softmax_head_dim = softmax_heads, softmax_head_dim
        self.state_as_context = bool(state_as_context)
        self.rope_theta = float(rope_theta)

        self.action_embed = MaskedProjector(ROBOT80_DIM, hidden_size)
        self.action_head = ActionFinalLayer(hidden_size, ROBOT80_DIM)
        if not state_as_context:
            self.state_embed = MaskedProjector(ROBOT80_DIM, hidden_size)
        elif self.legacy_context:
            self.state_context_embed = MaskedProjector(ROBOT80_DIM, raw_context_dim)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        if self.legacy_context:
            self.context_mlp = nn.Sequential(
                nn.Linear(raw_context_dim, hidden_size),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_size, hidden_size),
            )

        softmax_layers = set(int(i) for i in softmax_layer_indices)
        self.softmax_layer_indices = tuple(sorted(softmax_layers))
        self.blocks = nn.ModuleList(
            [
                MoTActionBlock(
                    hidden_size,
                    "softmax" if i in softmax_layers else "gdn",
                    shared_dim,
                    softmax_heads if i in softmax_layers else gdn_heads,
                    softmax_head_dim if i in softmax_layers else gdn_head_dim,
                    cross_attn_heads=cross_attn_heads,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    cross_norm=cross_norm,
                    context_dim=self.context_dim,
                )
                for i in range(depth)
            ]
        )

        self.attn_res = BlockAttnResV2(hidden_size, depth, block_size=attn_res_block_size)
        self.initialize()

    def initialize(self) -> None:
        """Mirror the training initialization (irrelevant once a checkpoint is loaded; keeps random tests well scaled)."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        if self.legacy_context:
            nn.init.normal_(self.context_mlp[0].weight, std=0.02)
            nn.init.normal_(self.context_mlp[2].weight, std=0.02)
        nn.init.zeros_(self.attn_res.attn_proj.weight)
        nn.init.zeros_(self.attn_res.mlp_proj.weight)
        nn.init.zeros_(self.attn_res.final_proj.weight)
        for projector in (
            self.action_embed,
            getattr(self, "state_embed", None),
            getattr(self, "state_context_embed", None),
        ):
            if projector is not None:
                nn.init.zeros_(projector.proj.weight)
                nn.init.zeros_(projector.proj.bias)
        nn.init.zeros_(self.action_head.linear.weight)
        nn.init.zeros_(self.action_head.linear.bias)

    @staticmethod
    def expand_for_action_head(t_flat: torch.Tensor, action_steps: int) -> torch.Tensor:
        """``[B, hidden] -> [B, 1, A, hidden]``, the per-token shape ``ActionFinalLayer`` requires."""

        return t_flat.unsqueeze(1).unsqueeze(1).expand(-1, 1, action_steps, -1)

    def project_context(self, raw_context: torch.Tensor) -> torch.Tensor:
        """Project raw text embeddings to the action expert's hidden width (legacy layout only)."""

        if not self.legacy_context:
            raise RuntimeError("the current MoT layout projects the action context in ContextEmbedder, not in ActionDiT")
        return self.context_mlp(raw_context)

    def build_context(
        self,
        raw_text: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        state80: torch.Tensor,
        state_mask80: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append one projected state token to the raw text context and project it (legacy layout, state_as_context=True).

        Returns ``(context [B, L+1, hidden], mask [B, L+1] bool)``.
        """

        if not self.legacy_context:
            raise RuntimeError("the current MoT layout appends the state token in ContextEmbedder, not in ActionDiT")
        state_token = self.state_context_embed(state80, state_mask80)
        raw_context = torch.cat((raw_text, state_token), dim=1)
        if text_mask is None:
            text_mask = raw_text.new_ones(raw_text.shape[0], raw_text.shape[1], dtype=torch.bool)
        state_mask = text_mask.new_ones(text_mask.shape[0], 1)
        context_mask = torch.cat((text_mask, state_mask), dim=1)
        return self.project_context(raw_context), context_mask


__all__ = [
    "ACTION_TEMPORAL_COMPRESSION",
    "ActionDiT",
    "MoTActionBlock",
    "independent_action_rope",
    "independent_action_rope_at",
]
