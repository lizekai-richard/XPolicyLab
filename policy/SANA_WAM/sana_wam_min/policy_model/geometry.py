# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Inference-active multi-view token-layout primitives."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def strip_to_view_tokens(
    tokens: torch.Tensor,
    frames: int,
    shapes: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Reorder ``[frame, strip]`` tokens to ``[view, frame, y, x]``."""

    areas = [int(height) * int(width) for height, width in shapes]
    tokens = tokens.unflatten(1, (int(frames), sum(areas)))
    return torch.cat(
        [view.flatten(1, 2) for view in tokens.split(areas, dim=2)], dim=1
    )


def view_to_strip_tokens(
    tokens: torch.Tensor,
    frames: int,
    shapes: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Reorder ``[view, frame, y, x]`` tokens to ``[frame, strip]``."""

    areas = [int(height) * int(width) for height, width in shapes]
    views = [
        view.unflatten(1, (int(frames), area))
        for view, area in zip(
            tokens.split(
                [int(frames) * area for area in areas], dim=1
            ),
            areas,
            strict=True,
        )
    ]
    return torch.cat(views, dim=2).flatten(1, 2)


# ``model.extra.sana_pixel_pad: masked`` (rwm/zekai-merge 319d3f666, dev/rwm/diffusion/data/sana_pixel_multiview_layout.py):
# the 2x2 sana_pixel canvas's black quadrant -- slot 1, top-right -- leaves the policy's token sequence.
SANA_PIXEL_GRID = (2, 2)
SANA_PIXEL_PAD_SLOTS = (1,)


def sana_pixel_real_token_mask(latent_height: int, latent_width: int, *, device=None) -> torch.Tensor:
    """Bool ``[H, W]`` of the canvas latent grid's real cells, False on the black quadrants ``SANA_PIXEL_PAD_SLOTS``.

    Every quadrant edge must fall between latent cells: the 320x512 canvas (10 x 16), not 320x480 (10 x 15), which the
    live layout refuses the same way."""

    rows, columns = SANA_PIXEL_GRID
    height, width = int(latent_height), int(latent_width)
    if height % rows or width % columns:
        raise ValueError(
            f"the {height}x{width} latent grid does not split into whole {rows}x{columns} quadrants; masking the "
            "sana_pixel pad needs every tile edge on the latent grid (the 320x512 canvas, not 320x480)"
        )
    tile_height, tile_width = height // rows, width // columns
    mask = torch.ones(height, width, dtype=torch.bool, device=device)
    for slot in SANA_PIXEL_PAD_SLOTS:
        row, column = divmod(slot, columns)
        mask[row * tile_height : (row + 1) * tile_height, column * tile_width : (column + 1) * tile_width] = False
    return mask


def sana_pixel_real_token_index(frames: int, latent_height: int, latent_width: int, *, device=None) -> torch.Tensor:
    """Ascending int64 positions of the real cells in the frame-major ``F * H * W`` token order."""

    mask = sana_pixel_real_token_mask(latent_height, latent_width).reshape(1, -1).expand(int(frames), -1)
    return mask.reshape(-1).nonzero(as_tuple=True)[0].to(device)


__all__ = [
    "SANA_PIXEL_GRID",
    "SANA_PIXEL_PAD_SLOTS",
    "sana_pixel_real_token_index",
    "sana_pixel_real_token_mask",
    "strip_to_view_tokens",
    "view_to_strip_tokens",
]
