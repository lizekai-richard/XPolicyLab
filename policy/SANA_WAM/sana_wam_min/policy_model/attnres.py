# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Inference-only Attention Residual aggregation for the bid policy."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthRMSNorm(nn.Module):
    """Parameter-free RMSNorm for depth-attention keys."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class BlockAttnResV2(nn.Module):
    """Buffer-based AttnRes v2 used by the eval-only policy trunk.

    The resolved bidirectional-policy config has timestep conditioning disabled;
    supporting the training-only conditioned variant here would add parameters
    that do not exist in its checkpoint.
    """

    def __init__(
        self,
        hidden_size: int,
        depth: int,
        block_size: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.block_size = block_size
        self.num_blocks = math.ceil(depth / block_size)
        self.attn_proj = nn.Linear(hidden_size, 1, bias=False)
        self.mlp_proj = nn.Linear(hidden_size, 1, bias=False)
        self.final_proj = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.attn_proj.weight)
        nn.init.zeros_(self.mlp_proj.weight)
        nn.init.zeros_(self.final_proj.weight)

        self.key_norm = DepthRMSNorm(eps=1e-6)

    def _attend_buffer(
        self,
        projection: nn.Linear,
        value_buffer: torch.Tensor,
        key_buffer: torch.Tensor,
        n_active: int,
        partial_block: torch.Tensor | None,
    ) -> torch.Tensor:
        if partial_block is not None:
            num_sources = n_active + 1
            value_buffer[n_active] = partial_block
            key_buffer[n_active] = self.key_norm(
                partial_block.unsqueeze(0)
            ).squeeze(0)
        else:
            num_sources = n_active

        if num_sources == 1:
            return value_buffer[0]

        values = value_buffer[:num_sources]
        keys = key_buffer[:num_sources]
        query = projection.weight.squeeze()
        logits = torch.einsum("d,nbtd->nbt", query, keys)
        alpha = F.softmax(logits, dim=0)
        return torch.einsum("nbt,nbtd->btd", alpha, values)


__all__ = ["BlockAttnResV2", "DepthRMSNorm"]
