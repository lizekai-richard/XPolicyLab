"""Chunk-causal attention shells and block of the causal policy (inference subset).

Ports of Sana rwm/zekai-merge (c391260a4):

* ``CausalGDNAttention`` <- ``layers/chunk_causal_gdn.py`` ``CachedChunkCausalPolicyGDNAttention`` with one chunk
  segment per forward (every deploy window is one segment): the donor's ``GatedDeltaNet`` projections plus the
  recurrence parameters ``gate_proj`` / ``A_log`` / ``dt_bias`` (and the never-read ``recall_gate`` buffer), the
  segment key scale ``(dim * n)^-1/2``, the per-segment decay ``exp(-e^A_log * softplus(gate_proj(mean x) + dt_bias))``,
  the read ``q (gS + K^T diag(beta) V) / (q_den (gz + K_den^T beta) + eps)`` and the delta-rule write of ``(S, z)``
  (``gdn_scan`` / ``_chunk_slices`` / ``_readout``, same einsum contractions, fp32 compute, complex64 RoPE on the
  numerator path only).
* ``CausalSoftmaxAttention`` <- ``layers/chunk_causal_softmax.py`` ``CachedChunkCausalPolicySoftmaxAttention``: q/k
  RMSNorm, fp64 RoPE cast back to the activation dtype, then attention over ``[context entries | this window]``. The
  context entries are post-RoPE keys rotated at their physical positions when they were committed, which is what
  the two-stream training forward reads (its prefix keys are rotated once over the whole canvas); the training
  spans, not Sana's deploy KV manager, are the reference (the manager always reads the observation entry).
* ``CausalPolicyBlock`` <- ``layers/chunk_causal_block.py`` ``ChunkWiseCausalPolicyBlock``: the bidirectional
  block's AdaLN / cross-attention / SwiGLU with the two shells above.

A forward receives an optional ``LayerCache``: absent = the cache-free chunk-0 window (zero GDN state, the softmax
reads only the window); present = a cached window that reads ``cache.gdn_state`` / ``cache.softmax_context`` and,
when ``cache.commit`` is set, leaves the next GDN state and this window's K/V in ``cache.gdn_next`` /
``cache.softmax_new`` for the session to keep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared import SANA_WAM_POLICY_DIR  # noqa: F401  (sys.path bridge first)

from sana_wam_min.policy_model.attention import (  # noqa: E402
    GatedDeltaNet,
    GatedSoftmaxAttention,
    MultiHeadCrossAttention,
    apply_rotary_emb_fp64_softmax,
)
from sana_wam_min.policy_model.block import PolicyBlock  # noqa: E402
from sana_wam_min.policy_model.embeddings import t2i_modulate  # noqa: E402
from sana_wam_min.policy_model.feedforward import SwiGLU  # noqa: E402

# eps of the donor's attention modules (sana_wam_min PolicyBlock); the causal shells inherit it (``eps=old.eps``)
ATTENTION_EPS = 1e-8
GDN_KIND = "GatedDeltaNet"
SOFTMAX_KIND = "GatedSoftmaxAttention"


@dataclass
class LayerCache:
    """One layer's view of the session memory for one forward (one text stream)."""

    commit: bool = False
    gdn_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None       # (S [B,H,D,D], z [B,H,D]); None = zeros
    gdn_next: Optional[tuple[torch.Tensor, torch.Tensor]] = None        # written on commit
    softmax_context: Optional[tuple[torch.Tensor, torch.Tensor]] = None  # (k, v) [B,T,H,D] post-RoPE, oldest first
    softmax_new: Optional[tuple[torch.Tensor, torch.Tensor]] = None      # this window's (k, v), written on commit


def rope_bhnd_complex64(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate ``[B,H,N,D]`` by ``freqs`` in complex64 (Sana ``chunk_causal_gdn._rope_bhnd``), back in ``x``'s dtype."""

    rotated = torch.view_as_complex(x.to(torch.float32).contiguous().unflatten(-1, (-1, 2)))
    freqs_c = freqs.to(torch.complex64) if freqs.is_complex() else freqs.float()
    return torch.view_as_real(rotated * freqs_c).flatten(-2).type_as(x)


class CausalGDNAttention(GatedDeltaNet):
    """GDN with a chunk-granular delta-rule state; every forward is ONE chunk segment (the deploy windows)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        heads: int,
        eps: float = ATTENTION_EPS,
        norm_eps: float = 1e-5,
        fp32_attention: bool = False,
        dt_bias_init: float = -5.0,
        a_log_init: float = 0.0,
    ) -> None:
        super().__init__(in_dim, out_dim, heads=heads, eps=eps, norm_eps=norm_eps, fp32_attention=fp32_attention)
        self.gate_proj = nn.Linear(in_dim, heads, bias=True)
        self.A_log = nn.Parameter(torch.full((heads,), float(a_log_init), dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.full((heads,), float(dt_bias_init), dtype=torch.float32))
        self.register_buffer("recall_gate", torch.zeros(1), persistent=True)
        self.dt_bias_init = float(dt_bias_init)
        self.a_log_init = float(a_log_init)
        self.init_recurrence_parameters()

    def init_recurrence_parameters(self) -> None:
        """Sana ``ChunkGDNLinearAttention._init_new_recurrence_params`` (a donor checkpoint without them): decay
        ``exp(-softplus(dt_bias_init))`` ~ 0.9933, i.e. near pass-through."""

        with torch.no_grad():
            self.recall_gate.zero_()
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)
            self.dt_bias.fill_(self.dt_bias_init)
            self.A_log.fill_(self.a_log_init)

    def chunk_decay(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, H]`` decay of the one segment from the mean of the attention input (``_chunk_decay``)."""

        gate = self.gate_proj(x.mean(dim=1)).float()
        return (-self.A_log.float().exp() * F.softplus(gate + self.dt_bias.float())).exp()

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        cache: Optional[LayerCache] = None,
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, channels)
        q_raw, k_raw, v_raw = qkv.unbind(2)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(batch, tokens, self.heads, self.dim).permute(0, 2, 1, 3)

        q_num = heads(self.q_norm(q_raw))
        k_num = heads(self.k_norm(k_raw))
        q_den, k_den = torch.relu(q_num), torch.relu(k_num)
        v = v_raw.reshape(batch, tokens, self.heads, self.dim).permute(0, 2, 1, 3)
        beta = self.beta_proj(x).sigmoid().permute(0, 2, 1).contiguous()
        if rotary_emb is not None:
            q_num = rope_bhnd_complex64(q_num, rotary_emb)
            k_num = rope_bhnd_complex64(k_num, rotary_emb)
        decay_raw = self.chunk_decay(x)

        compute = torch.float64 if q_num.dtype == torch.float64 else torch.float32
        q_c = q_num.to(compute)
        k_c = k_num.to(compute)
        v_c = v.to(compute)
        q_den_c = q_den.to(compute)
        k_den_c = k_den.to(compute)
        beta_c = beta.to(compute)
        n_valid = torch.full((batch,), float(tokens), dtype=compute, device=x.device)
        key_scale = (self.dim * n_valid).rsqrt()[:, None, None, None]
        k_c = k_c * key_scale
        k_den_c = k_den_c * key_scale
        decay = decay_raw.to(compute)

        seed = None if cache is None else cache.gdn_state
        if seed is None:
            state = torch.zeros(batch, self.heads, self.dim, self.dim, dtype=compute, device=x.device)
            state_z = torch.zeros(batch, self.heads, self.dim, dtype=compute, device=x.device)
        else:
            state, state_z = seed[0].to(compute), seed[1].to(compute)
        decayed_state = state * decay[..., None, None]
        decayed_z = state_z * decay[..., None]

        # _readout: the decayed past plus the chunk's own beta-gated sum-linear attention
        intra_read = torch.einsum("bhld,bhl,bhlv->bhdv", k_c, beta_c, v_c)
        num = torch.einsum("bhld,bhdv->bhlv", q_c, decayed_state + intra_read)
        intra_z = torch.einsum("bhld,bhl->bhd", k_den_c, beta_c)
        den = torch.einsum("bhld,bhd->bhl", q_den_c, decayed_z + intra_z) + self.eps
        out = (num / den[..., None]).to(v.dtype)

        if cache is not None and cache.commit:
            v_pred = torch.einsum("bhld,bhdv->bhlv", k_c, decayed_state)
            dv = (v_c - v_pred) * beta_c[..., None]
            state_new = decayed_state + torch.einsum("bhld,bhlv->bhdv", k_c, dv)
            z_pred = torch.einsum("bhld,bhd->bhl", k_den_c, decayed_z)
            dz = (1.0 - z_pred) * beta_c
            z_new = decayed_z + torch.einsum("bhld,bhl->bhd", k_den_c, dz)
            cache.gdn_next = (state_new.detach(), z_new.detach())

        out = out.permute(0, 1, 3, 2)
        out = self.o_norm(out)
        out = out.permute(0, 3, 1, 2).reshape(batch, tokens, channels)
        out = out * torch.sigmoid(self.output_gate(x)).to(out.dtype)
        return self.proj(out.to(self.proj.weight.dtype))


class CausalSoftmaxAttention(GatedSoftmaxAttention):
    """Gated softmax attention over ``[context entries | this window]``."""

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        cache: Optional[LayerCache] = None,
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
        if cache is not None and cache.commit:
            cache.softmax_new = (k.detach(), v.detach())
        if cache is not None and cache.softmax_context is not None:
            context_k, context_v = cache.softmax_context
            k = torch.cat((context_k.to(k.dtype), k), dim=1)
            v = torch.cat((context_v.to(v.dtype), v), dim=1)
        if self.fp32_attention:
            q, k, v = q.float(), k.float(), v.float()
        out = self._run_attention(q, k, v)
        out = out.reshape(batch, tokens, channels).to(output_dtype)
        out = out * torch.sigmoid(self.output_gate(x))
        return self.proj(out)


class CausalPolicyBlock(PolicyBlock):
    """``PolicyBlock`` with the chunk-causal attention shells; the cross-attention, norms, SwiGLU and AdaLN table
    keep the donor's keys and computation."""

    def __init__(  # noqa: D401 - mirrors PolicyBlock.__init__ without building the bidirectional attention first
        self,
        *,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        attention_kind: str,
        linear_head_dim: int,
        softmax_head_dim: int,
        fp32_attention: bool,
        dt_bias_init: float = -5.0,
        a_log_init: float = 0.0,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.attn_type = attention_kind
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if attention_kind == GDN_KIND:
            self.attn = CausalGDNAttention(
                hidden_size,
                hidden_size,
                heads=hidden_size // linear_head_dim,
                eps=ATTENTION_EPS,
                fp32_attention=fp32_attention,
                dt_bias_init=dt_bias_init,
                a_log_init=a_log_init,
            )
        elif attention_kind == SOFTMAX_KIND:
            self.attn = CausalSoftmaxAttention(
                hidden_size,
                hidden_size,
                heads=hidden_size // softmax_head_dim,
                eps=ATTENTION_EPS,
                fp32_attention=fp32_attention,
            )
        else:
            raise ValueError(f"unsupported policy attention kind: {attention_kind}")
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLU(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            out_features=hidden_size,
        )
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
        self.sf_hook_point = nn.Identity()

    def forward_attn_sublayer(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        text: torch.Tensor,
        modulation: torch.Tensor,
        text_mask=None,
        rotary_emb=None,
        prompt_group_spans=None,
        cache: Optional[LayerCache] = None,
    ) -> torch.Tensor:
        """Self-attention delta (the causal shell, reading / writing ``cache``) plus the text cross-attention delta."""

        batch, tokens, channels = hidden_states.shape
        token_count = modulation.shape[2]
        if token_count != tokens:
            raise ValueError(f"per-token modulation has {token_count} rows for {tokens} tokens")
        mod = modulation.reshape(batch, token_count, 6, channels)
        shift, scale, gate, _, _, _ = (self.scale_shift_table[None, None] + mod).chunk(6, dim=-2)
        normalized = self.norm1(hidden_states).reshape(batch, token_count, -1, channels)
        attention_input = t2i_modulate(normalized, shift, scale).reshape(batch, tokens, channels)
        self_attention = self.attn(attention_input, rotary_emb=rotary_emb, cache=cache).reshape(
            batch, token_count, -1, channels
        )
        self_delta = (gate * self_attention).reshape(batch, tokens, channels)
        cross_delta = self.cross_attn(
            hidden_states + self_delta,
            text,
            mask=text_mask,
            prompt_group_spans=prompt_group_spans,
        )
        return self_delta + cross_delta


__all__ = [
    "ATTENTION_EPS",
    "CausalGDNAttention",
    "CausalPolicyBlock",
    "CausalSoftmaxAttention",
    "GDN_KIND",
    "LayerCache",
    "SOFTMAX_KIND",
    "rope_bhnd_complex64",
]
