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


__all__ = [
    "strip_to_view_tokens",
    "view_to_strip_tokens",
]
