"""Self-contained attention primitives for the bidirectional policy.

This module is the resolved baseline inference path used by the policy:

* global-bilinear ``GatedDeltaNet`` for 24/32 transformer blocks;
* full-head-RoPE ``GatedSoftmaxAttention`` for the other 8 blocks;
* ``MultiHeadCrossAttention`` for text conditioning.

It intentionally has no dependency on ``diffusion`` or ``dev``.  Registry
decorators, model-discovery monkey patches, causal-attention variants,
visualization buffers and alternate positional-encoding implementations are not
part of this inference-only copy.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .condition_chunks import (
    A_LOG_INIT,
    DT_BIAS_INIT,
    TARGET,
    ConditionPass,
    chunk_causal_attention,
    chunk_means,
    gated_delta_chunk_scan,
    gated_delta_condition_pass,
    require_condition_layout,
)
from .embeddings import RMSNorm


# xformers is never imported in this vendored copy: SDPA is the reference path.
_xformers_ops = None


def apply_rotary_emb_fp64(
    hidden_states: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply the baseline complex128 RoPE to GDN ``[B,H,D,N]`` data."""

    original_dtype = hidden_states.dtype
    values = torch.view_as_complex(
        hidden_states.permute(0, 1, 3, 2)
        .to(torch.float64)
        .unflatten(-1, (-1, 2))
    )
    values = torch.view_as_real(values * freqs).flatten(-2)
    return values.permute(0, 1, 3, 2).to(original_dtype)


def apply_rotary_emb_fp64_softmax(
    hidden_states: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply the baseline complex128 RoPE to ``[B,N,H,D]`` data.

    RoPE owns the complete per-head feature dimension.  This intentionally
    mirrors the current policy baseline instead of reserving or neutralizing a
    camera-pose subspace.
    """

    original_dtype = hidden_states.dtype
    values = torch.view_as_complex(
        hidden_states.transpose(1, 2)
        .to(torch.float64)
        .unflatten(-1, (-1, 2))
    )
    values = torch.view_as_real(values * freqs).flatten(-2)
    return values.transpose(1, 2).to(original_dtype)


class _QKVAttentionBase(nn.Module):
    """The small subset of timm Attention that these policy modules consume."""

    def __init__(self, dim: int, num_heads: int, *, qkv_bias: bool) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)


class GatedDeltaNet(_QKVAttentionBase):
    """Current bidirectional GDN: global gated bilinear linear attention.

    Despite the historical class name, this path has no recurrent delta-rule
    state.  It computes two global matrix products.  Q/K content features used
    by the denominator are deliberately taken before RoPE.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        heads: int,
        eps: float = 1e-6,
        norm_eps: float = 1e-5,
        fp32_attention: bool = False,
    ) -> None:
        super().__init__(in_dim, heads, qkv_bias=False)

        if in_dim != out_dim:
            raise ValueError(
                "the policy checkpoint requires equal input/output dimensions; "
                f"got {in_dim} and {out_dim}"
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads
        self.eps = eps
        self.fp32_attention = fp32_attention

        self.kernel_func = nn.ReLU(inplace=False)
        self.q_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
        self.k_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)

        self.beta_proj = nn.Linear(in_dim, heads, bias=True)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.ones_(self.beta_proj.bias)

        self.output_gate = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.output_gate.weight)
        nn.init.zeros_(self.output_gate.bias)

        self.o_norm = RMSNorm(
            self.dim,
            scale_factor=1.0,
            eps=norm_eps,
            norm_dim=-2,
        )

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape

        q, k, v = self.qkv(x).reshape(batch, tokens, 3, channels).unbind(2)
        output_dtype = q.dtype

        q = self.q_norm(q).transpose(-1, -2)
        k = self.k_norm(k).transpose(-1, -2)
        v = v.transpose(-1, -2)

        q = q.reshape(batch, self.heads, self.dim, tokens)
        k = k.reshape(batch, self.heads, self.dim, tokens)
        v = v.reshape(batch, self.heads, self.dim, tokens)

        if rotary_emb is None:
            q_rotated, k_rotated = q, k
        else:
            q_rotated = apply_rotary_emb_fp64(q, rotary_emb)
            k_rotated = apply_rotary_emb_fp64(k, rotary_emb)

        # These must remain pre-RoPE features: they define the normalization
        # denominator of the current checkpoint's global-bilinear attention.
        q_kernel = self.kernel_func(q)
        k_kernel = self.kernel_func(k)

        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2).unsqueeze(2)
        k_gated = k_rotated * beta

        if self.fp32_attention:
            q_rotated = q_rotated.float()
            k_gated = k_gated.float()
            v = v.float()

        z = 1.0 / (
            k_kernel.sum(dim=-1, keepdim=True).transpose(-2, -1) @ q_kernel
            + self.eps
        )
        vk = torch.matmul(v, k_gated.transpose(-1, -2))
        out = torch.matmul(vk, q_rotated) * z

        out = self.o_norm(out.to(output_dtype))
        out = out.reshape(batch, channels, tokens).permute(0, 2, 1)
        out = out * torch.sigmoid(self.output_gate(x))
        return self.proj(out)


class GatedSoftmaxAttention(_QKVAttentionBase):
    """Dense gated attention with RoPE over the complete head dimension."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        heads: int,
        eps: float = 1e-6,
        norm_eps: float = 1e-5,
        fp32_attention: bool = False,
    ) -> None:
        super().__init__(in_dim, heads, qkv_bias=False)

        if in_dim != out_dim:
            raise ValueError(
                "the policy checkpoint requires equal input/output dimensions; "
                f"got {in_dim} and {out_dim}"
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads
        self.eps = eps
        self.fp32_attention = fp32_attention

        self.q_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
        self.k_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)

        self.output_gate = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.output_gate.weight)
        nn.init.zeros_(self.output_gate.bias)

    def _run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if _xformers_ops is not None:
            return _xformers_ops.memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=None,
            )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        return out.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, channels).unbind(2)
        output_dtype = q.dtype

        q = self.q_norm(q).reshape(batch, tokens, self.heads, self.dim)
        k = self.k_norm(k).reshape(batch, tokens, self.heads, self.dim)
        v = v.reshape(batch, tokens, self.heads, self.dim)

        if rotary_emb is not None:
            q = apply_rotary_emb_fp64_softmax(q, rotary_emb)
            k = apply_rotary_emb_fp64_softmax(k, rotary_emb)

        if self.fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        out = self._run_attention(q, k, v)
        out = out.reshape(batch, tokens, channels).to(output_dtype)
        out = out * torch.sigmoid(self.output_gate(x))
        return self.proj(out)


class ConditionChunkGatedDeltaNet(GatedDeltaNet):
    """GatedDeltaNet of the condition-causal policy (Sana condition_chunks.py ``_ConditionChunkGDNMixin``, f6a257d4a):
    the two-chunk gated delta-rule scan with a per-chunk decay from ``gate_proj`` / ``A_log`` / ``dt_bias`` and a
    beta-weighted write. Projections, q/k norms, fp64 RoPE, beta, the output norm and gate are the parent's;
    ``output_normalizer`` False (``model.memory_gdn_output_normalizer``) returns the numerator alone. ``recall_gate``
    is a persistent buffer the forward does not read (the live class loads and saves it)."""

    def __init__(self, *args, output_normalizer: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output_normalizer = bool(output_normalizer)
        self.gate_proj = nn.Linear(self.in_dim, self.heads, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        self.A_log = nn.Parameter(torch.full((self.heads,), A_LOG_INIT))
        self.dt_bias = nn.Parameter(torch.full((self.heads,), DT_BIAS_INIT))
        self.register_buffer("recall_gate", torch.zeros(1), persistent=True)

    def _chunk_decay(self, x: torch.Tensor, slices) -> torch.Tensor:
        """Per-chunk decay ``[B, H, M]`` from the chunk mean of ``gate_proj`` over chunk-major ``x``."""

        gate = chunk_means(self.gate_proj(x), slices).float()
        decay = (-self.A_log.float().exp() * F.softplus(gate + self.dt_bias.float())).exp()
        return decay.transpose(1, 2)

    def forward(self, x: torch.Tensor, rotary_emb: torch.Tensor | None = None, chunk_layout=None) -> torch.Tensor:
        layout = require_condition_layout(chunk_layout)
        batch, tokens, channels = x.shape
        if tokens != layout.num_tokens:
            raise ValueError(f"linear attention has {tokens} tokens, the condition layout covers {layout.num_tokens}")
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, channels).unbind(2)
        dtype = q.dtype

        def head_major(tensor):
            return tensor.transpose(-1, -2).reshape(batch, channels // self.dim, self.dim, tokens)

        q = head_major(self.q_norm(q))
        k = head_major(self.k_norm(k))
        if rotary_emb is None:
            q_num, k_num = q, k
        else:
            q_num, k_num = apply_rotary_emb_fp64(q, rotary_emb), apply_rotary_emb_fp64(k, rotary_emb)
        v = v.transpose(-1, -2).reshape(batch, channels // self.dim, self.dim, tokens)
        order = layout.order

        def chunk_major(tensor):
            return tensor.transpose(-1, -2).index_select(2, order)

        normalizer = self.output_normalizer
        q_den = chunk_major(self.kernel_func(q)) if normalizer else None
        k_den = chunk_major(self.kernel_func(k)) if normalizer else None
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2).index_select(2, order)
        slices = layout.slices()
        decay = self._chunk_decay(x.index_select(1, order), slices)
        if isinstance(layout, ConditionPass):
            reads = (chunk_major(q_num), chunk_major(k_num), chunk_major(v), q_den, k_den, beta, decay, self.eps)
            if layout.phase == TARGET:
                out = gated_delta_condition_pass(*reads, normalizer=normalizer, state=layout.cache.load(self))
            else:
                out, written = gated_delta_condition_pass(*reads, normalizer=normalizer)
                layout.cache.store(self, written)
            out = out.to(dtype)
        else:
            out = gated_delta_chunk_scan(
                chunk_major(q_num), chunk_major(k_num), chunk_major(v), q_den, k_den, beta, decay, slices, self.eps,
                normalizer=normalizer,
            ).to(dtype)
        out = out.index_select(2, layout.restore).transpose(-1, -2)
        out = self.o_norm(out).reshape(batch, channels, tokens).transpose(1, 2)
        return self.proj(out * torch.sigmoid(self.output_gate(x)))


class ConditionChunkGatedSoftmaxAttention(GatedSoftmaxAttention):
    """Softmax attention whose queries read only their own and earlier chunks (Sana condition_chunks.py
    ``_ConditionChunkSoftmaxMixin``): under a prefix-cache pass the condition chunk stores its keys and values and the
    prediction chunk reads them ahead of its own."""

    _condition_layout = None

    def forward(self, x: torch.Tensor, rotary_emb: torch.Tensor | None = None, chunk_layout=None) -> torch.Tensor:
        self._condition_layout = require_condition_layout(chunk_layout)
        try:
            return super().forward(x, rotary_emb=rotary_emb)
        finally:
            self._condition_layout = None

    def _run_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        layout = self._condition_layout
        if layout is None:  # the chunk calls below run the parent's attention on their slices
            return super()._run_attention(q, k, v)
        if isinstance(layout, ConditionPass):
            if layout.phase == TARGET:
                keys, values = layout.cache.load(self)
                return super()._run_attention(q, torch.cat((keys, k), dim=1), torch.cat((values, v), dim=1))
            layout.cache.store(self, (k, v))
        parent = super()._run_attention
        return chunk_causal_attention(parent, q, k, v, layout)


class MultiHeadCrossAttention(nn.Module):
    """Text cross-attention over one text block or per-group query spans."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_xformers: bool | None = None,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # width of the keys/values' source (layers/cross_attention.py ``context_dim``, 2026-09-22): the MoT canvas
        # builds feed the action expert (d_model 1024) the video-width (2560) shared caption embedding
        self.context_dim = d_model if context_dim is None else int(context_dim)

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(self.context_dim, d_model * 2)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(0.0)

        self.q_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6)
        self.k_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6)

        available = _xformers_ops is not None
        self.use_xformers = (
            available
            if use_xformers is None
            else bool(use_xformers and available)
        )

    def set_use_xformers(self, enabled: bool) -> None:
        self.use_xformers = bool(enabled and _xformers_ops is not None)

    @staticmethod
    def _validate_group_spans(x, cond, prompt_group_spans):
        """Static per-group (offset, length) spans that tile the query tokens.

        Port of layers/cross_attention.py MultiHeadCrossAttention
        ._validate_group_spans: a gapped or permuted layout is a routing bug.
        """

        if cond.ndim != 4:
            raise ValueError(
                "grouped cross-attention condition must be [B,G,L,C], got "
                f"{tuple(cond.shape)}"
            )
        spans = tuple(
            (int(offset), int(length)) for offset, length in prompt_group_spans
        )
        if len(spans) != cond.shape[1]:
            raise ValueError(
                f"prompt_group_spans carries {len(spans)} groups; the "
                f"condition declares G={cond.shape[1]}"
            )
        expected_offset = 0
        for group, (offset, length) in enumerate(spans):
            if offset != expected_offset or length <= 0:
                raise ValueError(
                    "prompt_group_spans must tile the query tokens "
                    f"contiguously in group order, got {spans} at group {group}"
                )
            expected_offset = offset + length
        if expected_offset != x.shape[1]:
            raise ValueError(
                f"prompt_group_spans covers {expected_offset} query tokens; "
                f"the sequence has {x.shape[1]}"
            )
        return spans

    def _forward_grouped(self, x, cond, mask, prompt_group_spans):
        """One cross-attention per text group over its contiguous query span.

        Port of layers/cross_attention.py MultiHeadCrossAttention
        ._forward_grouped.
        """

        if cond.shape[0] != x.shape[0] or cond.shape[-1] != self.context_dim:
            raise ValueError("grouped condition batch/channels must match query tokens")
        if mask is not None and tuple(mask.shape[:2]) != tuple(cond.shape[:2]):
            raise ValueError("grouped text mask must start with [B,G]")
        spans = self._validate_group_spans(x, cond, prompt_group_spans)
        outputs = []
        for group, (offset, length) in enumerate(spans):
            group_condition = cond[:, group]
            group_mask = None if mask is None else mask[:, group]
            if self.use_xformers:
                if group_mask is None:
                    group_lens = [group_condition.shape[1]] * x.shape[0]
                    group_condition = group_condition.reshape(
                        1, -1, group_condition.shape[-1]
                    )
                else:
                    group_mask = group_mask.to(torch.bool)
                    group_lens = group_mask.sum(dim=1).tolist()
                    group_condition = group_condition.masked_select(
                        group_mask.unsqueeze(-1)
                    ).reshape(1, -1, group_condition.shape[-1])
                group_mask = group_lens
            outputs.append(
                self.forward(
                    # The span slice is strided for B > 1; a contiguous query
                    # keeps q_linear on the fused addmm.
                    x[:, offset : offset + length].contiguous(),
                    group_condition,
                    mask=group_mask,
                )
            )
        return torch.cat(outputs, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask=None,
        prompt_group_spans=None,
    ) -> torch.Tensor:
        if prompt_group_spans is not None:
            return self._forward_grouped(x, cond, mask, prompt_group_spans)
        batch, tokens, channels = x.shape
        first_dim = 1 if self.use_xformers else batch

        q = self.q_linear(x)
        k, v = self.kv_linear(cond).view(first_dim, -1, 2, channels).unbind(2)
        q = self.q_norm(q).view(
            first_dim,
            -1,
            self.num_heads,
            self.head_dim,
        )
        k = self.k_norm(k).view(
            first_dim,
            -1,
            self.num_heads,
            self.head_dim,
        )
        v = v.view(first_dim, -1, self.num_heads, self.head_dim)

        if self.use_xformers:
            attention_bias = None
            if mask is not None:
                attention_bias = _xformers_ops.fmha.BlockDiagonalMask.from_seqlens(
                    [tokens] * batch,
                    mask,
                )
            out = _xformers_ops.memory_efficient_attention(
                q,
                k,
                v,
                p=self.attn_drop.p,
                attn_bias=attention_bias,
            )
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            attention_mask = mask
            if attention_mask is not None and attention_mask.ndim == 2:
                attention_mask = (
                    1 - attention_mask.to(q.dtype)
                ) * -10000.0
                attention_mask = attention_mask[:, None, None].repeat(
                    1,
                    self.num_heads,
                    1,
                    1,
                )
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)

        out = out.view(batch, -1, channels)
        out = self.proj(out)
        return self.proj_drop(out)


__all__ = [
    "ConditionChunkGatedDeltaNet",
    "ConditionChunkGatedSoftmaxAttention",
    "GatedDeltaNet",
    "GatedSoftmaxAttention",
    "MultiHeadCrossAttention",
    "apply_rotary_emb_fp64",
    "apply_rotary_emb_fp64_softmax",
]
