"""The action expert's attention-width adapters (``MoTGDNExpertHead`` / ``MoTSoftmaxExpertHead``).

Ports of ``dev/rwm/diffusion/model/layers/mot_block.py`` (rwm/mot @ b3b9e0e9e, unchanged since 606e48dd9): the projection and output
parameters of one expert at ``(expert_dim, shared_dim)``, carrying GatedDeltaNet's / GatedSoftmaxAttention's
attribute names so the joint-attention functions of :mod:`.joint_attention` duck-type on them exactly as on
the video expert's real attention modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sana_wam_min.policy_model.embeddings import RMSNorm


class MoTGDNExpertHead(nn.Module):
    """GDN projection/output parameters for one MoT expert, mirroring GatedDeltaNet's attribute names."""

    def __init__(
        self,
        expert_dim: int,
        shared_dim: int,
        heads: int,
        head_dim: int,
        *,
        qk_norm: bool = True,
        norm_eps: float = 1e-5,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if heads * head_dim != shared_dim:
            raise ValueError(f"heads*head_dim ({heads}*{head_dim}) must equal shared_dim ({shared_dim})")
        self.heads = heads
        self.dim = head_dim

        self.qkv = nn.Linear(expert_dim, shared_dim * 3, bias=use_bias)
        if qk_norm:
            self.q_norm = RMSNorm(shared_dim, scale_factor=1.0, eps=norm_eps)
            self.k_norm = RMSNorm(shared_dim, scale_factor=1.0, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.beta_proj = nn.Linear(expert_dim, heads, bias=True)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.ones_(self.beta_proj.bias)
        self.kernel_func = nn.ReLU(inplace=False)

        self.output_gate = nn.Linear(expert_dim, shared_dim, bias=True)
        nn.init.xavier_uniform_(self.output_gate.weight)
        nn.init.zeros_(self.output_gate.bias)

        self.o_norm = RMSNorm(self.dim, scale_factor=1.0, eps=norm_eps, norm_dim=-2)
        self.proj = nn.Linear(shared_dim, expert_dim, bias=True)


class MoTSoftmaxExpertHead(nn.Module):
    """Softmax projection/output parameters for one MoT expert, mirroring GatedSoftmaxAttention's attribute names."""

    def __init__(
        self,
        expert_dim: int,
        shared_dim: int,
        heads: int,
        head_dim: int,
        *,
        qk_norm: bool = True,
        norm_eps: float = 1e-5,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if heads * head_dim != shared_dim:
            raise ValueError(f"heads*head_dim ({heads}*{head_dim}) must equal shared_dim ({shared_dim})")
        self.heads = heads
        self.dim = head_dim

        self.qkv = nn.Linear(expert_dim, shared_dim * 3, bias=use_bias)
        if qk_norm:
            self.q_norm = RMSNorm(shared_dim, scale_factor=1.0, eps=norm_eps)
            self.k_norm = RMSNorm(shared_dim, scale_factor=1.0, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.output_gate = nn.Linear(expert_dim, shared_dim, bias=True)
        nn.init.xavier_uniform_(self.output_gate.weight)
        nn.init.zeros_(self.output_gate.bias)
        self.proj = nn.Linear(shared_dim, expert_dim, bias=True)


__all__ = ["MoTGDNExpertHead", "MoTSoftmaxExpertHead"]
