"""The per-layer math of one MoT layer: joint self-attention halves and the AdaLN sublayer helpers.

Ports of ``dev/rwm/diffusion/model/layers/mot_block.py`` (rwm/mot @ b3b9e0e9e; the math is unchanged since 606e48dd9
except that the video cross-attention can be routed per text group). The attention halves
reproduce ``GatedDeltaNet.forward`` / ``GatedSoftmaxAttention.forward`` around the attention op
(``gdn_qkv`` / ``gdn_ffn``, ``softmax_qkv`` / ``softmax_ffn``); ``mixed_*_attention`` runs the attention op
ONCE over the concatenated tokens of every expert (unrestricted mutual attention). The rotary embeddings
are the vendored fp64 kernels of ``sana_wam_min.policy_model.attention`` (the same math as Sana's
``_rope_gdn_layout`` / ``_rope_softmax_layout``). Drop-path and the self-flow hook point of the training
blocks are identities at inference and are not modeled.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

from sana_wam_min.policy_model.attention import apply_rotary_emb_fp64, apply_rotary_emb_fp64_softmax
from sana_wam_min.policy_model.embeddings import t2i_modulate


def gdn_qkv(module, x: torch.Tensor, rotary_emb) -> Tuple[torch.Tensor, ...]:
    """GatedDeltaNet.forward up to the attention op; returns ``(q_rotated, k_gated, v, q_kernel, k_kernel)``, each ``[B, heads, head_dim, N]``."""

    batch, tokens, _ = x.shape
    q, k, v = module.qkv(x).reshape(batch, tokens, 3, -1).unbind(2)

    q = module.q_norm(q).transpose(-1, -2)
    k = module.k_norm(k).transpose(-1, -2)
    v = v.transpose(-1, -2)

    q = q.reshape(batch, module.heads, module.dim, tokens)
    k = k.reshape(batch, module.heads, module.dim, tokens)
    v = v.reshape(batch, module.heads, module.dim, tokens)

    if rotary_emb is not None:
        q_rotated = apply_rotary_emb_fp64(q, rotary_emb)
        k_rotated = apply_rotary_emb_fp64(k, rotary_emb)
    else:
        q_rotated = q
        k_rotated = k

    q_kernel = module.kernel_func(q)
    k_kernel = module.kernel_func(k)

    beta = torch.sigmoid(module.beta_proj(x))
    beta = beta.transpose(1, 2).unsqueeze(2)
    k_gated = k_rotated * beta

    return q_rotated, k_gated, v, q_kernel, k_kernel


def mixed_gdn_attention(
    projected: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    eps: float = 1e-6,
    fp32_attention: bool = False,
) -> List[torch.Tensor]:
    """Run GatedDeltaNet's global linear-attention formula once over every expert's tokens.

    Args:
        projected: one :func:`gdn_qkv` output per expert; concatenated along the token axis.
        eps: the normalizer epsilon of the video GatedDeltaNet instance.
        fp32_attention: upcast q_rotated / k_gated / v as GatedDeltaNet.forward does.

    Returns:
        One ``[B, heads, head_dim, N_i]`` tensor per expert, in input order.
    """

    dtype = projected[0][0].dtype
    q_rotated = torch.cat([p[0] for p in projected], dim=-1)
    k_gated = torch.cat([p[1] for p in projected], dim=-1)
    v = torch.cat([p[2] for p in projected], dim=-1)
    q_kernel = torch.cat([p[3] for p in projected], dim=-1)
    k_kernel = torch.cat([p[4] for p in projected], dim=-1)

    if fp32_attention:
        q_rotated, k_gated, v = q_rotated.float(), k_gated.float(), v.float()

    z = 1 / (k_kernel.sum(dim=-1, keepdim=True).transpose(-2, -1) @ q_kernel + eps)
    vk = torch.matmul(v, k_gated.transpose(-1, -2))
    out = torch.matmul(vk, q_rotated)
    out = (out * z).to(dtype)

    sizes = [p[0].shape[-1] for p in projected]
    return list(torch.split(out, sizes, dim=-1))


def gdn_ffn(module, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """GatedDeltaNet.forward after the attention op: per-head output norm, sigmoid output gate from ``x``, output projection."""

    batch, heads, dim, tokens = out.shape
    dtype = x.dtype
    out = module.o_norm(out)
    out = out.reshape(batch, heads * dim, tokens).permute(0, 2, 1).to(dtype)
    gate = torch.sigmoid(module.output_gate(x))
    out = out * gate
    return module.proj(out)


def softmax_qkv(module, x: torch.Tensor, rotary_emb) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GatedSoftmaxAttention.forward up to the attention op; returns ``(q, k, v)``, each ``[B, N, heads, head_dim]``."""

    batch, tokens, _ = x.shape
    q, k, v = module.qkv(x).reshape(batch, tokens, 3, -1).unbind(2)

    q = module.q_norm(q)
    k = module.k_norm(k)

    q = q.reshape(batch, tokens, module.heads, module.dim)
    k = k.reshape(batch, tokens, module.heads, module.dim)
    v = v.reshape(batch, tokens, module.heads, module.dim)

    if rotary_emb is not None:
        q = apply_rotary_emb_fp64_softmax(q, rotary_emb)
        k = apply_rotary_emb_fp64_softmax(k, rotary_emb)

    return q, k, v


def mixed_softmax_attention(
    projected: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    fp32_attention: bool = False,
) -> List[torch.Tensor]:
    """Run one ``scaled_dot_product_attention`` over every expert's tokens; one ``[B, N_i, heads, head_dim]`` per expert."""

    q = torch.cat([p[0] for p in projected], dim=1)
    k = torch.cat([p[1] for p in projected], dim=1)
    v = torch.cat([p[2] for p in projected], dim=1)
    if fp32_attention:
        q, k, v = q.float(), k.float(), v.float()
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=None, dropout_p=0.0, is_causal=False
    ).transpose(1, 2)
    sizes = [p[0].shape[1] for p in projected]
    return list(torch.split(out, sizes, dim=1))


def softmax_ffn(module, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """GatedSoftmaxAttention.forward after the attention op: reshape, sigmoid output gate from ``x``, output projection."""

    batch, tokens, heads, dim = out.shape
    dtype = x.dtype
    out = out.reshape(batch, tokens, heads * dim).to(dtype)
    gate = torch.sigmoid(module.output_gate(x))
    out = out * gate
    return module.proj(out)


def frame_aware_pre_attn(block, h: torch.Tensor, t0: torch.Tensor, num_frames: int):
    """Per-row AdaLN modulation up to the attention op; ``t0`` is the frame-aware ``[B, 1, num_frames, 6*hidden]`` table."""

    batch, tokens, channels = h.shape
    t_mod = t0.reshape(batch, num_frames, 6, -1)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        block.scale_shift_table[None, None, :, :] + t_mod
    ).chunk(6, dim=-2)
    x_mod = t2i_modulate(
        block.norm1(h).reshape(batch, num_frames, -1, channels), shift_msa, scale_msa
    ).reshape(batch, tokens, channels)
    post_state = {
        "h": h,
        "gate_msa": gate_msa,
        "shift_mlp": shift_mlp,
        "scale_mlp": scale_mlp,
        "gate_mlp": gate_mlp,
        "num_frames": num_frames,
        "tokens": tokens,
        "channels": channels,
    }
    return x_mod, post_state


def frame_aware_post_attn(
    block, attn_out: torch.Tensor, post_state: dict, context: torch.Tensor, context_mask, prompt_group_spans=None
) -> torch.Tensor:
    """Per-row gate, residual add and cross-attention after the attention op; returns the attention sublayer's residual delta.

    ``prompt_group_spans`` routes a grouped context ``[B, G, L, C]`` span by span (the multiview video expert: view g's
    tokens cross-attend to text group g); None is the plain ``[B, L, C]`` cross-attention.
    """

    batch, num_frames, channels = post_state["h"].shape[0], post_state["num_frames"], post_state["channels"]
    tokens = post_state["tokens"]
    attn_delta = (post_state["gate_msa"] * attn_out.reshape(batch, num_frames, -1, channels)).reshape(
        batch, tokens, channels
    )
    h_after_attn = post_state["h"] + attn_delta
    cross_delta = block.cross_attn(h_after_attn, context, mask=context_mask, prompt_group_spans=prompt_group_spans)
    return attn_delta + cross_delta


def frame_aware_mlp(block, h: torch.Tensor, t0: torch.Tensor, num_frames: int) -> torch.Tensor:
    """The video block's frame-aware MLP residual delta (``_forward_mlp_sublayer_frame_aware`` without drop-path / hook)."""

    batch, tokens, channels = h.shape
    t_mod = t0.reshape(batch, num_frames, 6, -1)
    _, _, _, shift_mlp, scale_mlp, gate_mlp = (block.scale_shift_table[None, None, :, :] + t_mod).chunk(6, dim=-2)
    mlp_out = block.mlp(
        t2i_modulate(block.norm2(h).reshape(batch, num_frames, -1, channels), shift_mlp, scale_mlp).reshape(
            batch, tokens, channels
        )
    )
    return (gate_mlp * mlp_out.reshape(batch, num_frames, -1, channels)).reshape(batch, tokens, channels)


def action_pre_attn(block, h: torch.Tensor, t0: torch.Tensor):
    """Plain per-batch AdaLN modulation up to the attention op (action block, state_as_context=True)."""

    batch = h.shape[0]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.modulation(t0, batch)
    x_mod = t2i_modulate(block.norm1(h), shift_msa, scale_msa)
    post_state = {
        "h": h,
        "gate_msa": gate_msa,
        "shift_mlp": shift_mlp,
        "scale_mlp": scale_mlp,
        "gate_mlp": gate_mlp,
    }
    return x_mod, post_state


def action_post_attn(block, attn_out: torch.Tensor, post_state: dict, context: torch.Tensor, context_mask) -> torch.Tensor:
    """Gate, residual add and cross-attention after the attention op (action block, state_as_context=True); returns the residual delta."""

    attn_delta = post_state["gate_msa"] * attn_out
    h_after_attn = post_state["h"] + attn_delta
    cross_delta = block.cross_attn(h_after_attn, context, mask=context_mask)
    return attn_delta + cross_delta


__all__ = [
    "action_post_attn",
    "action_pre_attn",
    "frame_aware_mlp",
    "frame_aware_post_attn",
    "frame_aware_pre_attn",
    "gdn_ffn",
    "gdn_qkv",
    "mixed_gdn_attention",
    "mixed_softmax_attention",
    "softmax_ffn",
    "softmax_qkv",
]
