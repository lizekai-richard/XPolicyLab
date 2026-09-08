"""Checkpoint-compatible pointwise SwiGLU used by policy inference."""

from __future__ import annotations

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """The resolved two-projection QwenNext feed-forward module."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=True)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=True)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=True)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.act(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(hidden)


__all__ = ["SwiGLU"]
