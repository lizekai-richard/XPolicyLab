# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Multi-view latent packing: V views of ``[..., h, w]`` grids become one ``[..., 1, V*h*w]`` strip.

Vendored verbatim from Sana ``dev/rwm/diffusion/multiview_utils.py``: ``ViewShapes``,
``pack_spatial_views``, ``unpack_spatial_views``, ``pack_multiview_latents`` and
``unpack_multiview_latents``. The RoPE-position helper of that module
(``semantic_2x2_position_ids``) is intentionally omitted here: its port lives in
``policy_model.rope`` next to the rotary embedding that consumes it. The token reorder helpers
``strip_to_view_tokens`` / ``view_to_strip_tokens`` are omitted because nothing in the inference
path reorders strip tokens.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


ViewShapes = tuple[tuple[int, int], ...]


def pack_spatial_views(
    views: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ViewShapes]:
    """Flatten each view's H/W and concatenate them into ``[..., 1, S]``."""

    shapes = tuple((int(view.shape[-2]), int(view.shape[-1])) for view in views)
    strip = torch.cat([view.flatten(-2) for view in views], dim=-1).unsqueeze(-2)
    return strip, shapes


def unpack_spatial_views(
    strip: torch.Tensor,
    shapes: Sequence[tuple[int, int]],
) -> tuple[torch.Tensor, ...]:
    """Invert :func:`pack_spatial_views`."""

    shapes = tuple((int(height), int(width)) for height, width in shapes)
    chunks = strip.squeeze(-2).split(
        [height * width for height, width in shapes], dim=-1
    )
    return tuple(
        chunk.unflatten(-1, (height, width))
        for chunk, (height, width) in zip(chunks, shapes, strict=True)
    )


def pack_multiview_latents(
    views: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ViewShapes]:
    """Pack V>1 latent grids while preserving the native V=1 layout."""

    shapes = tuple((int(view.shape[-2]), int(view.shape[-1])) for view in views)
    return (views[0], shapes) if len(views) == 1 else pack_spatial_views(views)


def unpack_multiview_latents(
    latent: torch.Tensor,
    shapes: Sequence[tuple[int, int]],
) -> tuple[torch.Tensor, ...]:
    """Invert :func:`pack_multiview_latents`."""

    return (latent,) if len(shapes) == 1 else unpack_spatial_views(latent, shapes)

__all__ = [
    "ViewShapes",
    "pack_multiview_latents",
    "pack_spatial_views",
    "unpack_multiview_latents",
    "unpack_spatial_views",
]
