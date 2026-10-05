# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Physical-time three-axis MRoPE used by policy inference."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
import torch.nn as nn


def _complex_frequency_table(
    dim: int,
    length: int,
    theta: float,
) -> torch.Tensor:
    if dim <= 0 or dim % 2:
        raise ValueError(f"RoPE dimension must be positive and even, got {dim}")
    positions = torch.arange(length)
    # Same expression as layers/embedders.py get_1d_rotary_pos_embed so the
    # regular-grid table is bitwise identical to the live one.
    inverse_frequency = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    )
    phase = torch.outer(positions, inverse_frequency)
    return torch.polar(torch.ones_like(phase), phase)


def semantic_2x2_position_ids(
    frames: int,
    view_shapes: Sequence[tuple[int, int]],
    view_slot_ids: Sequence[int],
    fps: torch.Tensor,
    base_fps: float,
    tile_shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """View-major ``[time, y, x]`` ids placing each view on its TL/TR/BL/BR tile.

    A view smaller than the tile is centered in it with fractional offsets.
    Port of dev/rwm/diffusion/multiview_utils.py semantic_2x2_position_ids.
    """

    frames = int(frames)
    tile_height, tile_width = (int(value) for value in tile_shape)
    fps = torch.as_tensor(fps, device=device, dtype=torch.float64).reshape(-1)
    frame_ids = torch.arange(frames, device=device, dtype=torch.float64)
    time = float(base_fps) * frame_ids[None] / fps[:, None]

    position_ids = []
    for (height, width), slot in zip(view_shapes, view_slot_ids, strict=True):
        height, width, slot = int(height), int(width), int(slot)
        # An oversized view would spill into a neighbouring tile silently.
        if height > tile_height or width > tile_width:
            raise ValueError(
                f"view shape {(height, width)} must fit semantic tile {tile_shape}"
            )
        row, column = divmod(slot, 2)
        y_offset = row * tile_height + (tile_height - height) / 2.0
        x_offset = column * tile_width + (tile_width - width) / 2.0
        view_time = time[:, :, None, None].expand(-1, -1, height, width)
        view_y = (
            torch.arange(height, device=device, dtype=torch.float64) + y_offset
        )[None, None, :, None].expand_as(view_time)
        view_x = (
            torch.arange(width, device=device, dtype=torch.float64) + x_offset
        )[None, None, None, :].expand_as(view_time)
        position_ids.append(
            torch.stack((view_time, view_y, view_x), dim=-1).reshape(
                fps.numel(), frames * height * width, 3
            )
        )
    return torch.cat(position_ids, dim=1)


class WanRotaryPosEmbed(nn.Module):
    """Legacy regular-grid Wan MRoPE frequency generator."""

    def __init__(
        self,
        attention_head_dim: int,
        max_seq_len: int = 1024,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.attention_head_dim = int(attention_head_dim)
        self.max_seq_len = int(max_seq_len)

        height_dim = width_dim = 2 * (self.attention_head_dim // 6)
        temporal_dim = self.attention_head_dim - height_dim - width_dim
        self.axis_dims = (temporal_dim, height_dim, width_dim)
        self.freqs = torch.cat(
            tuple(
                _complex_frequency_table(dim, self.max_seq_len, theta)
                for dim in self.axis_dims
            ),
            dim=1,
        )

    def forward(
        self,
        fhw: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        frames, height, width = (int(value) for value in fhw)
        if min(frames, height, width) <= 0:
            raise ValueError(f"F/H/W must be positive, got {fhw}")
        self.freqs = self.freqs.to(device)
        temporal, vertical, horizontal = self.freqs.split(
            tuple(dim // 2 for dim in self.axis_dims), dim=1
        )
        temporal = temporal[:frames].view(frames, 1, 1, -1).expand(
            frames, height, width, -1
        )
        vertical = vertical[:height].view(1, height, 1, -1).expand(
            frames, height, width, -1
        )
        horizontal = horizontal[:width].view(1, 1, width, -1).expand(
            frames, height, width, -1
        )
        return torch.cat((temporal, vertical, horizontal), dim=-1).reshape(
            1, 1, frames * height * width, -1
        )


class PhysicalTimeWanRotaryPosEmbed(nn.Module):
    """Evaluate Wan MRoPE on a physical clock, including fractional positions."""

    def __init__(
        self,
        legacy_rope: WanRotaryPosEmbed,
        *,
        attention_head_dim: int,
        base_fps: float = 16.0,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.legacy_rope = legacy_rope
        self.base_fps = float(base_fps)
        self.theta = float(theta)
        self.axis_dims = self._axis_dims(attention_head_dim)
        self._model_fps: torch.Tensor | None = None

    @staticmethod
    def _axis_dims(attention_head_dim: int) -> tuple[int, int, int]:
        height_dim = width_dim = 2 * (attention_head_dim // 6)
        dims = (
            attention_head_dim - height_dim - width_dim,
            height_dim,
            width_dim,
        )
        if any(dim <= 0 or dim % 2 for dim in dims):
            raise ValueError(
                f"Wan RoPE axis dimensions must be positive and even: {dims}"
            )
        return dims

    @contextmanager
    def use_model_fps(
        self,
        model_fps: float | Sequence[float] | torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Iterator[None]:
        previous = self._model_fps
        self._model_fps = self._normalize_fps(
            model_fps, batch_size, device
        )
        try:
            yield
        finally:
            self._model_fps = previous

    @staticmethod
    def _normalize_fps(
        model_fps: float | Sequence[float] | torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        fps = (
            model_fps.detach()
            if isinstance(model_fps, torch.Tensor)
            else torch.as_tensor(model_fps)
        )
        if fps.dtype == torch.bool or torch.is_complex(fps):
            raise TypeError("model_fps must contain real numeric values")
        if fps.ndim == 0 or (fps.ndim == 1 and fps.numel() == 1):
            fps = fps.reshape(1).expand(batch_size)
        elif fps.ndim != 1 or fps.numel() != batch_size:
            raise ValueError(
                f"model_fps must be scalar or have batch length {batch_size}, "
                f"got shape {tuple(fps.shape)}"
            )
        fps = fps.to(device=device, dtype=torch.float64)
        if not bool(torch.isfinite(fps).all()) or not bool((fps > 0).all()):
            raise ValueError("model_fps must contain positive finite values")
        return fps

    def forward(
        self,
        fhw: tuple[int, int, int],
        device: torch.device,
        *,
        frame_stride: int = 1,
    ) -> torch.Tensor:
        """Rotary table of an ``(F, H, W)`` grid; ``frame_stride`` source frames separate consecutive latent frames'
        time ids (``rope: aligned`` folds the video frame stride in: latent frame j at ``base_fps * j * s / fps``,
        layers/wan_mrope.py ``forward(..., frame_stride=)``). 1 is the plain context clock."""

        if self._model_fps is None:
            raise RuntimeError(
                "model_fps context must be set before evaluating Wan RoPE"
            )
        frames, height, width = (int(value) for value in fhw)
        if min(frames, height, width) <= 0:
            raise ValueError(f"F/H/W must be positive, got {fhw}")
        frame_stride = int(frame_stride)
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

        fps = self._model_fps.to(device=device)
        if frame_stride == 1 and bool(torch.all(fps == self.base_fps)):
            return self.legacy_rope((frames, height, width), device)
        return self.from_position_ids(
            self._position_ids(frames, height, width, fps / frame_stride, device)
        )

    def from_position_ids(self, position_ids: torch.Tensor) -> torch.Tensor:
        if position_ids.ndim != 3 or position_ids.shape[-1] != 3:
            raise ValueError(
                "position_ids must have shape [batch, tokens, 3], "
                f"got {tuple(position_ids.shape)}"
            )
        if not bool(torch.isfinite(position_ids).all()):
            raise ValueError("position_ids must be finite")

        phases = []
        for axis, dim in enumerate(self.axis_dims):
            exponent = torch.arange(
                0,
                dim,
                2,
                device=position_ids.device,
                dtype=torch.float64,
            ) / dim
            inverse_frequency = self.theta ** (-exponent)
            phases.append(
                position_ids[..., axis : axis + 1].to(torch.float64)
                * inverse_frequency
            )
        phase = torch.cat(phases, dim=-1)
        return torch.polar(torch.ones_like(phase), phase).unsqueeze(1)

    def _position_ids(
        self,
        frames: int,
        height: int,
        width: int,
        fps: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = fps.numel()
        frame_ids = torch.arange(frames, device=device, dtype=torch.float64)
        time_ids = self.base_fps * frame_ids[None] / fps[:, None]
        time_ids = time_ids[:, :, None, None].expand(
            batch_size, frames, height, width
        )
        height_ids = torch.arange(height, device=device, dtype=torch.float64)
        height_ids = height_ids[None, None, :, None].expand_as(time_ids)
        width_ids = torch.arange(width, device=device, dtype=torch.float64)
        width_ids = width_ids[None, None, None, :].expand_as(time_ids)
        return torch.stack((time_ids, height_ids, width_ids), dim=-1).reshape(
            batch_size, frames * height * width, 3
        )


def independent_action_rope(
    rope: PhysicalTimeWanRotaryPosEmbed,
    action_steps: int,
    batch: int,
    device: torch.device,
    first_position: int = 1,
) -> torch.Tensor:
    """Full-head-dim 1D rotary phases at the local positions ``first_position .. first_position + A - 1``.

    ``first_position=1`` is the live ``rope: independent`` table (layers/wan_mrope.py ``independent_action_rope``):
    the state row keeps the zero phase and the robot tail ``[state; action_1 .. action_A]`` is one consecutive local
    sequence ``0, 1 .. A`` (user ruling 2026-09-22). ``first_position=0`` is the table every strided checkpoint trained
    before that ruling used (rwm/strided_video ``d27304e82`` .. zekai-merge ``8a61ae18a``: state and the first action
    both at 0) and the one the OpenWAM canvas policy's ``state_as_cross_attention`` option has always used.
    No three-axis split and no ``model_fps`` scaling; ``[batch, 1, A, head_dim // 2]`` complex128.
    """

    head_dim = sum(rope.axis_dims)
    start = int(first_position)
    positions = torch.arange(start, start + int(action_steps), device=device, dtype=torch.float64)
    exponent = torch.arange(0, head_dim, 2, device=device, dtype=torch.float64) / head_dim
    inverse_frequency = rope.theta ** (-exponent)
    phase = torch.outer(positions, inverse_frequency)
    freqs = torch.polar(torch.ones_like(phase), phase)
    return freqs.view(1, 1, int(action_steps), -1).expand(int(batch), 1, int(action_steps), -1)


MULTIVIEW_SPATIAL_ROPE_TILE_SHAPE = (15, 30)


__all__ = [
    "MULTIVIEW_SPATIAL_ROPE_TILE_SHAPE",
    "PhysicalTimeWanRotaryPosEmbed",
    "WanRotaryPosEmbed",
    "independent_action_rope",
    "semantic_2x2_position_ids",
]
