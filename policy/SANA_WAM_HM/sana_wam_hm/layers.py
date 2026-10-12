"""Chunk-causal attention shells and block of the hierarchical-memory policy (inference subset).

Ports of Sana rwm/sparse-memory (e8dde6b56), applied to one block's token subset with explicit token-space chunk
bounds (every forward of this package; the bounds come from ``memory.memory_layout``):

* ``HMGDNAttention`` <- ``layers/chunk_causal_gdn.py`` ``CachedChunkCausalPolicyGDNAttention._plain`` /
  ``gdn_scan`` over ALL tokens valid: the donor's ``GatedDeltaNet`` projections plus the recurrence parameters
  ``gate_proj`` / ``A_log`` / ``dt_bias`` (and the never-read ``recall_gate`` buffer); per chunk the decay
  ``exp(-e^A_log * softplus(gate_proj(chunk mean) + dt_bias))``, the key scale ``(dim * chunk tokens)^-1/2``, the read
  ``q (gS + K^T diag(beta) V) / (q_den (gz + K_den^T beta) + eps)`` and the delta-rule write of ``(S, z)``, in the
  same einsum contractions, fp32 compute (fp64 under an fp64 model), complex64 RoPE on the numerator path only. The
  scan is seeded with a carried ``(S, z)`` (a committed state, else zeros) and returns its final state for a commit.
* ``HMSoftmaxAttention`` <- ``layers/attention_softmax.py`` ``GatedSoftmaxAttention.forward`` with ``chunk_causal``
  (the causal block's ``configure_chunk_causal(chunk_size=1)`` and ``HW = (N, 1, 1)``): q/k RMSNorm, fp64 RoPE cast
  back to the activation dtype, then chunk c's queries attend to the keys of chunks 0..c (``_run_chunk_causal_attention``)
  -- the prefix-causal window over the block's subset. A ``SoftmaxCache`` adds the commit-once read: the committed
  post-RoPE K / V of the memory before this forward's tokens (every chunk of the forward reads all of it, as the
  memory's chunks precede it in the window), and on commit keeps that prefix extended by this forward's own K / V.
* ``HMPolicyBlock`` <- ``layers/chunk_causal_block.py`` ``ChunkWiseCausalPolicyBlock``: the bidirectional block's AdaLN /
  cross-attention / SwiGLU with the two shells above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

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
class GDNCache:
    """One GDN layer's carried state for one forward of one text stream."""

    seed: Optional[tuple[torch.Tensor, torch.Tensor]] = None   # (S [B,H,D,D], z [B,H,D]) read first; None = zeros
    commit: bool = False                                        # keep the final state in ``out`` (a commit forward)
    out: Optional[tuple[torch.Tensor, torch.Tensor]] = None     # the scan's final (S, z), written on commit


@dataclass
class SoftmaxCache:
    """One softmax layer's committed memory K / V for one forward of one text stream."""

    prefix: Optional[tuple[torch.Tensor, torch.Tensor]] = None  # (k post-RoPE, v) [B, T, H, D] before the forward's keys
    commit: bool = False                                        # keep prefix + this forward's (k, v) in ``out``
    out: Optional[tuple[torch.Tensor, torch.Tensor]] = None


def rope_bhnd_complex64(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate ``[B,H,N,D]`` by ``freqs`` in complex64 (Sana ``chunk_causal_gdn._rope_bhnd``), back in ``x``'s dtype."""

    rotated = torch.view_as_complex(x.to(torch.float32).contiguous().unflatten(-1, (-1, 2)))
    freqs_c = freqs.to(torch.complex64) if freqs.is_complex() else freqs.float()
    return torch.view_as_real(rotated * freqs_c).flatten(-2).type_as(x)


def _check_bounds(bounds: Sequence[int], tokens: int) -> list[int]:
    bounds = [int(b) for b in bounds]
    if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != tokens or any(b <= a for a, b in zip(bounds, bounds[1:])):
        raise ValueError(f"chunk bounds {bounds} must run 0 < .. < {tokens} strictly increasing")
    return bounds


class HMGDNAttention(GatedDeltaNet):
    """GDN with a chunk-granular delta-rule state over explicit token-space chunks."""

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
        """Sana ``ChunkGDNLinearAttention._init_new_recurrence_params`` (a donor checkpoint without them)."""

        with torch.no_grad():
            self.recall_gate.zero_()
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)
            self.dt_bias.fill_(self.dt_bias_init)
            self.A_log.fill_(self.a_log_init)

    def decay_from_gate(self, gate: torch.Tensor) -> torch.Tensor:
        """Decay g = exp(-exp(A_log) * softplus(gate + dt_bias)) (Sana ``_decay_from_gate``)."""

        return (-self.A_log.float().exp() * F.softplus(gate + self.dt_bias.float())).exp()

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        bounds: Optional[Sequence[int]] = None,
        cache: Optional[GDNCache] = None,
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape
        bounds = _check_bounds([0, tokens] if bounds is None else bounds, tokens)
        qkv = self.qkv(x).reshape(batch, tokens, 3, channels)
        q_raw, k_raw, v_raw = qkv.unbind(2)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(batch, tokens, self.heads, self.dim).permute(0, 2, 1, 3)

        q_num = heads(self.q_norm(q_raw))
        k_num = heads(self.k_norm(k_raw))
        q_den, k_den = torch.relu(q_num), torch.relu(k_num)
        v = v_raw.reshape(batch, tokens, self.heads, self.dim).permute(0, 2, 1, 3)
        beta = self.beta_proj(x).sigmoid().permute(0, 2, 1).contiguous()
        # Sana computes the per-chunk decays from the attention input before rotating (``_chunk_decay``)
        decays = [self.decay_from_gate(self.gate_proj(x[:, lo:hi].mean(dim=1)).float()) for lo, hi in zip(bounds, bounds[1:])]
        if rotary_emb is not None:
            q_num = rope_bhnd_complex64(q_num, rotary_emb)
            k_num = rope_bhnd_complex64(k_num, rotary_emb)

        compute = torch.float64 if q_num.dtype == torch.float64 else torch.float32
        q_c_all, k_c_all, v_c_all = q_num.to(compute), k_num.to(compute), v.to(compute)
        q_den_all, k_den_all, beta_all = q_den.to(compute), k_den.to(compute), beta.to(compute)
        seed = None if cache is None else cache.seed
        if seed is None:
            state = torch.zeros(batch, self.heads, self.dim, self.dim, dtype=compute, device=x.device)
            state_z = torch.zeros(batch, self.heads, self.dim, dtype=compute, device=x.device)
        else:
            state, state_z = seed[0].to(compute), seed[1].to(compute)

        outs = []
        for c, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
            n_valid = torch.full((batch,), float(hi - lo), dtype=compute, device=x.device)
            key_scale = (self.dim * n_valid).rsqrt()[:, None, None, None]
            q_c = q_c_all[:, :, lo:hi]
            k_c = k_c_all[:, :, lo:hi] * key_scale
            v_c = v_c_all[:, :, lo:hi]
            q_den_c = q_den_all[:, :, lo:hi]
            k_den_c = k_den_all[:, :, lo:hi] * key_scale
            beta_c = beta_all[:, :, lo:hi]
            decay = decays[c].to(compute)
            decayed_state = state * decay[..., None, None]
            decayed_z = state_z * decay[..., None]
            # _readout: the decayed past plus the chunk's own beta-gated sum-linear attention
            intra_read = torch.einsum("bhld,bhl,bhlv->bhdv", k_c, beta_c, v_c)
            num = torch.einsum("bhld,bhdv->bhlv", q_c, decayed_state + intra_read)
            intra_z = torch.einsum("bhld,bhl->bhd", k_den_c, beta_c)
            den = torch.einsum("bhld,bhd->bhl", q_den_c, decayed_z + intra_z) + self.eps
            outs.append(num / den[..., None])
            # the delta-rule write (every chunk writes; the last one's survives only in a commit)
            v_pred = torch.einsum("bhld,bhdv->bhlv", k_c, decayed_state)
            dv = (v_c - v_pred) * beta_c[..., None]
            state = decayed_state + torch.einsum("bhld,bhlv->bhdv", k_c, dv)
            z_pred = torch.einsum("bhld,bhd->bhl", k_den_c, decayed_z)
            dz = (1.0 - z_pred) * beta_c
            state_z = decayed_z + torch.einsum("bhld,bhl->bhd", k_den_c, dz)
        if cache is not None and cache.commit:
            cache.out = (state.detach(), state_z.detach())

        out = torch.cat(outs, dim=2).to(v.dtype)
        out = out.permute(0, 1, 3, 2)
        out = self.o_norm(out)
        out = out.permute(0, 3, 1, 2).reshape(batch, tokens, channels)
        out = out * torch.sigmoid(self.output_gate(x)).to(out.dtype)
        return self.proj(out.to(self.proj.weight.dtype))


class HMSoftmaxAttention(GatedSoftmaxAttention):
    """Gated softmax attention, chunk-causal over explicit token-space chunks (the prefix-causal window)."""

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        bounds: Optional[Sequence[int]] = None,
        cache: Optional[SoftmaxCache] = None,
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape
        bounds = _check_bounds([0, tokens] if bounds is None else bounds, tokens)
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, channels).unbind(2)
        output_dtype = q.dtype
        q = self.q_norm(q).reshape(batch, tokens, self.heads, self.dim)
        k = self.k_norm(k).reshape(batch, tokens, self.heads, self.dim)
        v = v.reshape(batch, tokens, self.heads, self.dim)
        if rotary_emb is not None:
            q = apply_rotary_emb_fp64_softmax(q, rotary_emb)
            k = apply_rotary_emb_fp64_softmax(k, rotary_emb)
        prefix = 0
        if cache is not None and cache.prefix is not None:
            # kept in the activation dtype (the fp32 cast below is exact, so cast-after-concat == cast-before)
            k = torch.cat((cache.prefix[0].to(k.dtype), k), dim=1)
            v = torch.cat((cache.prefix[1].to(v.dtype), v), dim=1)
            prefix = int(cache.prefix[0].shape[1])
        if cache is not None and cache.commit:
            cache.out = (k.detach(), v.detach())
        if self.fp32_attention:
            q, k, v = q.float(), k.float(), v.float()
        out = torch.cat(
            [
                self._run_attention(q[:, lo:hi], k[:, : prefix + hi], v[:, : prefix + hi])
                for lo, hi in zip(bounds, bounds[1:])
            ],
            dim=1,
        )
        out = out.reshape(batch, tokens, channels).to(output_dtype)
        out = out * torch.sigmoid(self.output_gate(x))
        return self.proj(out)


class HMPolicyBlock(PolicyBlock):
    """``PolicyBlock`` with the hierarchical-memory attention shells; the cross-attention, norms, SwiGLU and AdaLN table
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
            self.attn = HMGDNAttention(
                hidden_size,
                hidden_size,
                heads=hidden_size // linear_head_dim,
                eps=ATTENTION_EPS,
                fp32_attention=fp32_attention,
                dt_bias_init=dt_bias_init,
                a_log_init=a_log_init,
            )
        elif attention_kind == SOFTMAX_KIND:
            self.attn = HMSoftmaxAttention(
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
        bounds: Optional[Sequence[int]] = None,
        cache: Optional[GDNCache | SoftmaxCache] = None,
    ) -> torch.Tensor:
        """Self-attention delta (the shell over ``bounds``, reading / writing ``cache``: a GDN's ``GDNCache``, a softmax
        layer's ``SoftmaxCache``) plus the text cross-attention delta, on the block's token subset."""

        batch, tokens, channels = hidden_states.shape
        token_count = modulation.shape[2]
        if token_count != tokens:
            raise ValueError(f"per-token modulation has {token_count} rows for {tokens} tokens")
        mod = modulation.reshape(batch, token_count, 6, channels)
        shift, scale, gate, _, _, _ = (self.scale_shift_table[None, None] + mod).chunk(6, dim=-2)
        normalized = self.norm1(hidden_states).reshape(batch, token_count, -1, channels)
        attention_input = t2i_modulate(normalized, shift, scale).reshape(batch, tokens, channels)
        expected = GDNCache if self.attn_type == GDN_KIND else SoftmaxCache
        if cache is not None and not isinstance(cache, expected):
            raise TypeError(f"a {self.attn_type} block takes a {expected.__name__}, got {type(cache).__name__}")
        attention = self.attn(attention_input, rotary_emb=rotary_emb, bounds=bounds, cache=cache)
        self_attention = attention.reshape(batch, token_count, -1, channels)
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
    "GDNCache",
    "GDN_KIND",
    "HMGDNAttention",
    "HMPolicyBlock",
    "HMSoftmaxAttention",
    "SOFTMAX_KIND",
    "SoftmaxCache",
    "rope_bhnd_complex64",
]
