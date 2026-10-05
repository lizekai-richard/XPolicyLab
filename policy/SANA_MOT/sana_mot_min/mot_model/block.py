"""One MoT layer (``MoTLayer`` = Sana's ``SanaPolicyMoTBlock``): both experts' sublayers with one joint attention between them."""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from .joint_attention import (
    action_post_attn,
    action_pre_attn,
    frame_aware_mlp,
    frame_aware_post_attn,
    frame_aware_pre_attn,
    gdn_ffn,
    gdn_qkv,
    mixed_gdn_attention,
    mixed_softmax_attention,
    softmax_ffn,
    softmax_qkv,
)


class Contexts(NamedTuple):
    """The two experts' cross-attention contexts, fixed for a whole forward.

    ``video`` is ``[B, L, Cv]`` (one text group) or ``[B, V, L, Cv]`` routed per view through ``prompt_group_spans``;
    ``action`` is ``[B, L, Ca]``; the masks are int16 / bool key masks (1 = valid) or None.
    """

    video: torch.Tensor
    action: torch.Tensor
    video_mask: Optional[torch.Tensor]
    action_mask: Optional[torch.Tensor]
    prompt_group_spans: Optional[tuple] = None


class MoTLayer(nn.Module):
    """The video block, the action block and their joint self-attention mix, run as one module.

    Checkpoint keys: ``blocks.<i>.video_block.*`` / ``blocks.<i>.action_block.*``. The attention and MLP
    sublayers are exposed separately because AttnRes re-aggregates the trunk inputs between them.
    """

    def __init__(self, video_block: nn.Module, action_block: nn.Module, *, is_softmax: bool) -> None:
        super().__init__()
        self.video_block = video_block
        self.action_block = action_block
        self.is_softmax = bool(is_softmax)

    def attention_sublayer(self, h_attn_v: torch.Tensor, h_attn_a: torch.Tensor, video: dict, action: dict, contexts: Contexts):
        """Both experts' attention sublayers on their AttnRes aggregates; returns ``(delta_v, delta_a, post_v, post_a)``."""

        video_block, action_block = self.video_block, self.action_block
        rope_v = video["rope_softmax"] if self.is_softmax else video["rope_linear"]
        rope_a = action["rope_softmax"] if self.is_softmax else action["rope_gdn"]

        x_mod_v, post_v = frame_aware_pre_attn(video_block, h_attn_v, video["t0"], video["num_frames"])
        if action["frame_aware"]:
            x_mod_a, post_a = frame_aware_pre_attn(action_block, h_attn_a, action["t0"], action["num_rows"])
        else:
            x_mod_a, post_a = action_pre_attn(action_block, h_attn_a, action["t0"])

        attn_v = video_block.attn
        attn_a = action_block.attn
        fp32_attention = bool(getattr(attn_v, "fp32_attention", False))
        if self.is_softmax:
            proj_v = softmax_qkv(attn_v, x_mod_v, rope_v)
            proj_a = softmax_qkv(attn_a, x_mod_a, rope_a)
            mixed_v, mixed_a = mixed_softmax_attention([proj_v, proj_a], fp32_attention=fp32_attention)
            out_v = softmax_ffn(attn_v, mixed_v, x_mod_v)
            out_a = softmax_ffn(attn_a, mixed_a, x_mod_a)
        else:
            proj_v = gdn_qkv(attn_v, x_mod_v, rope_v)
            proj_a = gdn_qkv(attn_a, x_mod_a, rope_a)
            mixed_v, mixed_a = mixed_gdn_attention([proj_v, proj_a], eps=attn_v.eps, fp32_attention=fp32_attention)
            out_v = gdn_ffn(attn_v, mixed_v, x_mod_v)
            out_a = gdn_ffn(attn_a, mixed_a, x_mod_a)

        attn_delta_v = frame_aware_post_attn(
            video_block, out_v, post_v, contexts.video, contexts.video_mask, contexts.prompt_group_spans
        )
        if action["frame_aware"]:
            attn_delta_a = frame_aware_post_attn(action_block, out_a, post_a, contexts.action, contexts.action_mask)
        else:
            attn_delta_a = action_post_attn(action_block, out_a, post_a, contexts.action, contexts.action_mask)
        return attn_delta_v, attn_delta_a, post_v, post_a

    def mlp_sublayer(self, h_mlp_v: torch.Tensor, h_mlp_a: torch.Tensor, post_v: dict, post_a: dict, video: dict):
        """Both experts' MLP sublayers on their AttnRes aggregates; returns ``(delta_v, delta_a)``."""

        mlp_delta_v = frame_aware_mlp(self.video_block, h_mlp_v, video["t0"], video["num_frames"])
        mlp_delta_a = self.action_block.mlp_sublayer(h_mlp_a, post_a)
        return mlp_delta_v, mlp_delta_a


__all__ = ["Contexts", "MoTLayer"]
