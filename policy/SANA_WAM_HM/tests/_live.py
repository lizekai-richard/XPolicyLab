"""The live Sana checkout the parity tests compare against (``SANA_HM_REPO``, default the pinned rwm/sparse-memory worktree
~/zekail/Sana_sparse_memory_e8dde6b56): its hierarchical-memory module and its causal policy class. A test module that
imports this one owns the ``dev`` package for its pytest session (run the live files apart from other Sana checkouts).
"""

from __future__ import annotations

import os
import sys
import types

import pytest

SANA_HM_REPO = os.path.expanduser(os.environ.get("SANA_HM_REPO", "~/zekail/Sana_sparse_memory_e8dde6b56"))


def reference_varlen(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0):
    """fp64 varlen attention over packed (T, H, D): one (query span, key span) pair at a time (Sana
    ``tests/causal_test_helpers._reference_varlen``)."""

    import torch

    assert dropout_p == 0.0
    scale = q.shape[-1] ** -0.5
    q_bounds = [int(b) for b in cu_seqlens_q.tolist()]
    k_bounds = [int(b) for b in cu_seqlens_k.tolist()]
    outs = []
    for (q_lo, q_hi), (k_lo, k_hi) in zip(zip(q_bounds, q_bounds[1:]), zip(k_bounds, k_bounds[1:])):
        qs, ks, vs = q[q_lo:q_hi], k[k_lo:k_hi], v[k_lo:k_hi]
        att = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        outs.append(torch.einsum("hqk,khd->qhd", att.softmax(-1), vs))
    return torch.cat(outs, dim=0)


def live_hm():
    """Sana's ``hierarchical_memory`` module and the causal net module, or skip."""

    if not os.path.isdir(os.path.join(SANA_HM_REPO, "dev", "rwm")):
        pytest.skip(f"no Sana checkout at {SANA_HM_REPO} (set SANA_HM_REPO to an rwm/sparse-memory checkout)")
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    if SANA_HM_REPO not in sys.path:
        sys.path.insert(0, SANA_HM_REPO)
    memory = pytest.importorskip("dev.rwm.diffusion.model.utils.hierarchical_memory")
    if not os.path.abspath(memory.__file__).startswith(os.path.abspath(SANA_HM_REPO)):
        pytest.skip(f"dev is already imported from another checkout ({memory.__file__}); run the live files on their own")
    return memory


def live_causal():
    live_hm()
    net = pytest.importorskip("dev.rwm.diffusion.model.nets.sana_qwennext_policy_causal")
    if not hasattr(net, "memory_layout"):
        pytest.skip(f"{SANA_HM_REPO} predates the hierarchical memory (rwm/sparse-memory)")
    return net


def bind_reference_varlen(monkeypatch) -> None:
    """``flash_attn.flash_attn_varlen_func`` -> the fp64 reference (Sana's causal cross-attention imports it at call time)."""

    module = types.ModuleType("flash_attn")
    module.flash_attn_varlen_func = reference_varlen
    monkeypatch.setitem(sys.modules, "flash_attn", module)


__all__ = ["SANA_HM_REPO", "bind_reference_varlen", "live_causal", "live_hm", "reference_varlen"]
