"""The token-group text of both experts (``ContextEmbedder``), the current rwm/mot text layouts.

Port of ``dev/rwm/diffusion/model/layers/mot_context_embedder.py`` (rwm/mot @ b3b9e0e9e .. 71ac93f43). Two builds:

* separate (``shared=False``; ``multiview: sana_latent``, every canvas run before 2026-09-22 and every canvas run from
  rwm/mot b010eb9a5 on): two identically built text paths, one per expert width (the OpenWAM dual-system pattern) --
  the donor's caption projection ``y_embedder`` + ``y_norm`` (2304 -> 2560) for the video expert and an action-width
  twin ``action_y_embedder`` + ``action_y_norm`` (2304 -> 1024). With G = V + 1 groups the video path embeds the V view
  groups and the action path the robot group (for a canvas that is G = 2: the composite-view row and the robot row,
  2026-09-21..22); with ONE group (``shared_prompt``: the canvas contract before 2026-09-21, 2026-09-22 briefly, and
  b010eb9a5's ``single_prompt`` since 2026-09-23) both paths embed that group.
* shared (``shared=True``; the canvas modes ``openwam`` / ``sana_pixel`` from rwm/mot 4e67e1e1d to b010eb9a5): ONE caption embedder,
  the donor's ``y_embedder`` + ``y_norm``, projects the ONE prompt (G = 1) once, and both experts cross-attend to that
  identical 2560-wide embedding (the action blocks' ``kv_linear`` takes ``context_dim`` 2560).

Under ``state_as_context`` one projected state token (``state_proj``, 80 -> raw width) is appended to the robot group,
or to the one group both experts read. Checkpoint keys: ``context_embedder.*``. The caption-dropout null tables
(``*.y_embedding``) are training-only and are stripped at load (``uncond_prob`` is 0 in the policy; CFG runs on the
instruction-free prompt), so this mirror carries only the projections and norms.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from sana_wam_min.policy_model.embeddings import CaptionEmbedder, MaskedProjector, RMSNorm
from sana_wam_min.robot80 import ROBOT80_DIM


class ContextEmbedder(nn.Module):
    """Both experts' text paths over the token-group prompt (+ the state token under state_as_context)."""

    def __init__(
        self,
        raw_dim: int,
        video_hidden_size: int,
        action_hidden_size: int,
        *,
        norm_eps: float = 1e-5,
        state_as_context: bool = False,
        shared: bool = False,
    ) -> None:
        super().__init__()
        self.raw_dim = int(raw_dim)
        self.video_hidden_size = int(video_hidden_size)
        self.action_hidden_size = int(action_hidden_size)
        self.state_as_context = bool(state_as_context)
        self.shared = bool(shared)
        self.y_embedder = CaptionEmbedder(self.raw_dim, self.video_hidden_size)
        self.y_norm = RMSNorm(self.video_hidden_size, scale_factor=1.0, eps=norm_eps)
        if self.shared:
            self.action_y_embedder = None
            self.action_y_norm = None
        else:
            self.action_y_embedder = CaptionEmbedder(self.raw_dim, self.action_hidden_size)
            self.action_y_norm = RMSNorm(self.action_hidden_size, scale_factor=1.0, eps=norm_eps)
        self.state_proj = MaskedProjector(ROBOT80_DIM, self.raw_dim) if self.state_as_context else None

    @property
    def context_width(self) -> int:
        """Width of the context the action expert cross-attends to: the video width when shared, its own otherwise."""

        return self.video_hidden_size if self.shared else self.action_hidden_size

    @staticmethod
    def prompt_groups(y: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return the prompt as token groups ``[B, G, 1, L, C]`` with a ``[B, G, 1, 1, L]`` mask (base layout ``[B, 1, L, C]`` = G 1)."""

        if y.ndim == 4:
            y = y.unsqueeze(1)
            if mask is not None:
                if mask.ndim != 4:
                    raise ValueError(f"base-layout text mask must be [B, 1, 1, L]; got {tuple(mask.shape)}")
                mask = mask.unsqueeze(1)
        if y.ndim != 5 or y.shape[2] != 1:
            raise ValueError(f"y must be [B, 1, L, C] or token-group [B, G, 1, L, C]; got {tuple(y.shape)}")
        if mask is not None:
            if mask.ndim != 5 or mask.shape[1] != y.shape[1] or mask.shape[-1] != y.shape[-2]:
                raise ValueError(f"token-group text mask must be [B, G={y.shape[1]}, 1, 1, L={y.shape[-2]}]; got {tuple(mask.shape)}")
            if mask.shape[0] != y.shape[0]:
                if mask.shape[0] != 1:
                    raise ValueError(f"text mask batch {mask.shape[0]} does not match y batch {y.shape[0]}")
                mask = mask.expand(y.shape[0], -1, -1, -1, -1)
        return y, mask

    @staticmethod
    def _embed_groups(
        embedder: CaptionEmbedder,
        norm: RMSNorm,
        rows: torch.Tensor,
        mask: Optional[torch.Tensor],
        state_token: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """One expert's text path over ``[B, G, 1, L, C]`` rows: ``(context [B, G, L', H], key_mask [B, G, L'] int16 or None)``, L' = L + 1 with a state token."""

        batch, groups, _, length, channels = rows.shape
        raw = rows.reshape(batch * groups, 1, length, channels).squeeze(1)
        if state_token is not None:
            raw = torch.cat((raw, state_token.repeat_interleave(groups, dim=0).to(raw.dtype)), dim=1)
        out = norm(embedder.y_proj(raw))
        out = out.reshape(batch, groups, out.shape[1], out.shape[2])
        key_mask = None if mask is None else mask.to(torch.int16).reshape(batch, groups, length)
        if state_token is not None:
            if key_mask is None:
                key_mask = torch.ones(batch, groups, length, dtype=torch.int16, device=rows.device)
            key_mask = torch.cat((key_mask, key_mask.new_ones(batch, groups, 1)), dim=2)
        return out, key_mask

    def forward(
        self,
        y: torch.Tensor,
        mask: Optional[torch.Tensor],
        state80: torch.Tensor,
        state_mask80: torch.Tensor,
        *,
        shared_prompt: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build both experts' contexts from the token-group prompt.

        Args:
            y: Prompt embeddings ``[B, G, 1, L, C]`` (token groups) or the base layout ``[B, 1, L, C]``.
            mask: Prompt mask ``[B, G, 1, 1, L]`` (or the base ``[B, 1, 1, L]``), nonzero = valid, or None.
            state80: Initial state row ``[B, 1, 80]``.
            state_mask80: Validity of the state row ``[B, 1, 80]`` (bool).
            shared_prompt: True = G must be 1 and the one group serves both experts (the canvas contract); False =
                groups ``0 .. G-2`` are the view groups (video expert) and group ``G-1`` the robot group (action expert).

        Returns:
            ``(context_video [B, Gv, L', Cv], context_action [B, L', Ca], video_key_mask [B, Gv, L'] or None,
            action_key_mask [B, L'] or None)``; Gv = 1 under ``shared_prompt``, else G - 1.
        """

        caption, mask = self.prompt_groups(y, mask)
        caption = caption.to(self.y_embedder.y_proj.fc1.weight.dtype)
        groups = caption.shape[1]
        if self.shared:
            if groups != 1:
                raise ValueError(
                    "the shared caption embedder serves ONE prompt group (a canvas mode ships one prompt naming its "
                    f"layout); got G={groups}"
                )
            state_token = self.state_proj(state80.to(caption.dtype), state_mask80) if self.state_as_context else None
            context, key_mask = self._embed_groups(self.y_embedder, self.y_norm, caption, mask, state_token)
            return context, context[:, 0], key_mask, None if key_mask is None else key_mask[:, 0]
        if shared_prompt:
            if groups != 1:
                raise ValueError(f"the shared-prompt contract reads ONE token group for both experts (G=1); got G={groups}")
            video_rows, action_rows = caption, caption
            video_mask, action_mask = mask, mask
        else:
            if groups < 2:
                raise ValueError(
                    "token-group text must carry the view groups followed by the robot group (G = V + 1 >= 2); "
                    f"got G={groups}"
                )
            video_rows, action_rows = caption[:, :-1], caption[:, -1:]
            video_mask = None if mask is None else mask[:, :-1]
            action_mask = None if mask is None else mask[:, -1:]
        state_token = self.state_proj(state80.to(caption.dtype), state_mask80) if self.state_as_context else None
        context_video, video_key_mask = self._embed_groups(
            self.y_embedder, self.y_norm, video_rows, video_mask, state_token if shared_prompt else None
        )
        context_action, action_key_mask = self._embed_groups(
            self.action_y_embedder, self.action_y_norm, action_rows, action_mask, state_token
        )
        return (
            context_video,
            context_action[:, 0],
            video_key_mask,
            None if action_key_mask is None else action_key_mask[:, 0],
        )


__all__ = ["ContextEmbedder"]
