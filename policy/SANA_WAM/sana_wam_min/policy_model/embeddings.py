# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Inference-only embedding and output modules for the bidirectional policy.

The classes in this file intentionally preserve the checkpoint-facing child
module names used by the training model while omitting its training-only
caption dropout and unrelated image-model utilities.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def t2i_modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply SANA's adaptive scale and shift."""

    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    """Checkpoint-compatible learnable RMSNorm."""

    def __init__(
        self,
        dim: int,
        *,
        scale_factor: float = 1.0,
        eps: float = 1e-6,
        norm_dim: int = -1,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.norm_dim = norm_dim
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_shape = [1] * x.ndim
        weight_shape[self.norm_dim] = -1
        weight = self.weight.view(*weight_shape)
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.pow(2).mean(self.norm_dim, keepdim=True) + self.eps
        )
        return (weight * normalized).type_as(x)


class PatchEmbedMS3D(nn.Module):
    """3D latent patch projection followed by ``BCTHW -> BNC`` flattening."""

    def __init__(
        self,
        patch_size=(1, 1, 1),
        in_chans: int = 128,
        embed_dim: int = 2560,
    ) -> None:
        super().__init__()
        patch_size = tuple(patch_size)
        if patch_size != (1, 1, 1):
            raise ValueError(
                "the resolved policy patch projection is exactly P1 Conv3d"
            )
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    """Embed scalar (including fractional) flow timesteps."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        timestep: torch.Tensor,
        dim: int,
        max_period: int = 10_000,
    ) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(
                0,
                half,
                dtype=torch.float32,
                device=timestep.device,
            )
            / half
        )
        angles = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        if dim % 2:
            embedding = torch.cat(
                (embedding, torch.zeros_like(embedding[:, :1])), dim=-1
            )
        return embedding

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        frequency = self.timestep_embedding(
            timestep, self.frequency_embedding_size
        ).to(self.dtype)
        return self.mlp(frequency)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


class CaptionProjection(nn.Module):
    """The inference-active portion of timm's two-layer caption MLP.

    Attribute names deliberately remain ``fc1`` and ``fc2`` so trained policy
    checkpoints retain the ``y_embedder.y_proj.*`` namespace.
    """

    def __init__(self, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_size)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class CaptionEmbedder(nn.Module):
    """No-CFG, eval-only caption projection.

    The legacy unconditional ``y_embedding`` is intentionally absent: it is
    never read by the deployed no-CFG policy. Checkpoint loading owns the
    explicit removal and validation of that legacy key.
    """

    def __init__(self, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.y_proj = CaptionProjection(in_channels, hidden_size)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        return self.y_proj(caption)


class MaskedProjector(nn.Module):
    """Project a canonical value/mask row to one policy token."""

    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * input_dim, hidden_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        value = value.masked_fill(~mask, 0)
        return self.proj(torch.cat((value, mask.to(value.dtype)), dim=-1))


class T2IFinalLayer(nn.Module):
    """Per-token AdaLN and projection for latent-video velocity."""

    def __init__(self, hidden_size: int, patch_size, out_channels: int) -> None:
        super().__init__()
        if tuple(patch_size) != (1, 1, 1):
            raise ValueError("the resolved policy output head is exactly P1")
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size**0.5
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim <= 2:
            raise ValueError(
                "joint policy final modulation requires per-token timestep "
                f"embeddings, got {tuple(timestep.shape)}"
            )
        batch, tokens, channels = x.shape
        num_rows = timestep.shape[2]
        if num_rows != tokens:
            raise ValueError(
                f"timestep rows {num_rows} must match token count {tokens}"
            )
        shift, scale = (
            self.scale_shift_table[None, None, :, :]
            + timestep.transpose(1, 2)
        ).chunk(2, dim=-2)
        normalized = self.norm_final(x).reshape(
            batch, num_rows, -1, channels
        )
        return self.linear(
            t2i_modulate(normalized, shift, scale).reshape(
                batch, tokens, channels
            )
        )


class ActionFinalLayer(T2IFinalLayer):
    """Per-action-token AdaLN and projection to the 80D robot row.

    Same modulation as the video head; the projection is zero-initialized.
    Port of sana_qwennext_pretrain.py ActionFinalLayer.
    """

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__(hidden_size, (1, 1, 1), out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        expected = (hidden.shape[0], 1, hidden.shape[1], hidden.shape[2])
        if tuple(timestep.shape) != expected:
            raise ValueError(
                "action timestep embedding must be [B,1,A,D], got "
                f"{tuple(timestep.shape)} for hidden {tuple(hidden.shape)}"
            )
        return super().forward(hidden, timestep)


__all__ = [
    "ActionFinalLayer",
    "CaptionEmbedder",
    "CaptionProjection",
    "MaskedProjector",
    "PatchEmbedMS3D",
    "RMSNorm",
    "T2IFinalLayer",
    "TimestepEmbedder",
    "t2i_modulate",
]
