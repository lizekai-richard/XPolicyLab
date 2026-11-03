"""One resolved QwenNext policy block, split at profileable sublayers."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import (
    ConditionChunkGatedDeltaNet,
    ConditionChunkGatedSoftmaxAttention,
    GatedDeltaNet,
    GatedSoftmaxAttention,
    MultiHeadCrossAttention,
)
from .embeddings import t2i_modulate
from .feedforward import SwiGLU


class PolicyBlock(nn.Module):
    """The frame-aware attention and MLP deltas used by AttnRes inference.

    The scalar-timestep, non-AttnRes, causal, image-cross-attention, and drop-path
    branches from the general training model are intentionally absent.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        attention_kind: str,
        linear_head_dim: int,
        softmax_head_dim: int,
        fp32_attention: bool,
        condition_causal: bool = False,
        gdn_output_normalizer: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_type = attention_kind
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        # condition-causal (rwm/zekai-merge f6a257d4a): the live policy swaps every self-attention for its
        # ``[obs; state] | [prediction; actions]`` subclass (condition_chunks.py); the GDN layers gain the recurrence
        self.condition_causal = bool(condition_causal)
        linear_cls = ConditionChunkGatedDeltaNet if self.condition_causal else GatedDeltaNet
        softmax_cls = ConditionChunkGatedSoftmaxAttention if self.condition_causal else GatedSoftmaxAttention
        linear_kwargs = {"output_normalizer": gdn_output_normalizer} if self.condition_causal else {}
        if attention_kind == "GatedDeltaNet":
            self.attn = linear_cls(
                hidden_size,
                hidden_size,
                heads=hidden_size // linear_head_dim,
                eps=1e-8,
                fp32_attention=fp32_attention,
                **linear_kwargs,
            )
        elif attention_kind == "GatedSoftmaxAttention":
            self.attn = softmax_cls(
                hidden_size,
                hidden_size,
                heads=hidden_size // softmax_head_dim,
                eps=1e-8,
                fp32_attention=fp32_attention,
            )
        else:
            raise ValueError(f"unsupported policy attention kind: {attention_kind}")
        self.cross_attn = MultiHeadCrossAttention(
            hidden_size,
            num_heads,
        )
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.mlp = SwiGLU(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            out_features=hidden_size,
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5
        )
        # Stable observation seam used by existing self-flow/profile hooks.
        self.sf_hook_point = nn.Identity()

    def forward_attn_sublayer(
        self,
        hidden_states: torch.Tensor,
        text: torch.Tensor,
        modulation: torch.Tensor,
        text_mask=None,
        rotary_emb=None,
        prompt_group_spans=None,
        chunk_layout=None,
    ) -> torch.Tensor:
        """Return self-attention plus text cross-attention delta.

        ``chunk_layout`` is the condition layout (or a prefix-cache pass) of a condition-causal block, None otherwise.

        ``text`` is ``[B,G,L,C]`` with ``text_mask`` ``[B,G,L]`` when
        ``prompt_group_spans`` routes one text group per query span; without
        spans it is the single-block ``[B,L,C]`` condition.
        """

        batch, tokens, channels = hidden_states.shape
        token_count = modulation.shape[2]
        if token_count != tokens:
            raise ValueError(
                f"per-token modulation has {token_count} rows for {tokens} tokens"
            )
        mod = modulation.reshape(batch, token_count, 6, channels)
        shift, scale, gate, _, _, _ = (
            self.scale_shift_table[None, None] + mod
        ).chunk(6, dim=-2)

        normalized = self.norm1(hidden_states).reshape(
            batch, token_count, -1, channels
        )
        attention_input = t2i_modulate(normalized, shift, scale).reshape(
            batch, tokens, channels
        )
        if self.condition_causal:
            self_attention = self.attn(attention_input, rotary_emb=rotary_emb, chunk_layout=chunk_layout)
        elif chunk_layout is not None:
            raise ValueError("a bidirectional block takes no condition layout")
        else:
            self_attention = self.attn(attention_input, rotary_emb=rotary_emb)
        self_attention = self_attention.reshape(batch, token_count, -1, channels)
        self_delta = (gate * self_attention).reshape(batch, tokens, channels)
        cross_delta = self.cross_attn(
            hidden_states + self_delta,
            text,
            mask=text_mask,
            prompt_group_spans=prompt_group_spans,
        )
        return self_delta + cross_delta

    def forward_mlp_sublayer(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
    ) -> torch.Tensor:
        """Return the AdaLN-modulated SwiGLU delta."""

        batch, tokens, channels = hidden_states.shape
        token_count = modulation.shape[2]
        if token_count != tokens:
            raise ValueError(
                f"per-token modulation has {token_count} rows for {tokens} tokens"
            )
        mod = modulation.reshape(batch, token_count, 6, channels)
        _, _, _, shift, scale, gate = (
            self.scale_shift_table[None, None] + mod
        ).chunk(6, dim=-2)
        normalized = self.norm2(hidden_states).reshape(
            batch, token_count, -1, channels
        )
        mlp_input = t2i_modulate(normalized, shift, scale).reshape(
            batch, tokens, channels
        )
        mlp_output = self.mlp(mlp_input).reshape(
            batch, token_count, -1, channels
        )
        delta = (gate * mlp_output).reshape(batch, tokens, channels)
        self.sf_hook_point(hidden_states + delta)
        return delta


__all__ = ["PolicyBlock"]
