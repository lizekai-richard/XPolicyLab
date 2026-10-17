"""The live Sana checkout the parity tests compare against (``SANA_SM_REPO``, default the pinned rwm/sparse-memory tree
~/zekail/Sana_sparse_memory_cc18ad78f, e.g. ``git -C <Sana-sparse-memory clone> worktree add --detach <dir> cc18ad78f``):
its sparse-memory module, its causal policy class and its sparse-memory deploy session. A test module that imports this
one owns the ``dev`` package for its pytest session (run the live files apart from other Sana checkouts).
"""

from __future__ import annotations

import os
import sys
import types

import pytest

SANA_SM_REPO = os.path.expanduser(os.environ.get("SANA_SM_REPO", "~/zekail/Sana_sparse_memory_cc18ad78f"))


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


def live_sparse_memory():
    """Sana's ``sparse_memory`` module, or skip."""

    if not os.path.isdir(os.path.join(SANA_SM_REPO, "dev", "rwm")):
        pytest.skip(f"no Sana checkout at {SANA_SM_REPO} (set SANA_SM_REPO to an rwm/sparse-memory checkout)")
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    if SANA_SM_REPO not in sys.path:
        sys.path.insert(0, SANA_SM_REPO)
    try:
        module = __import__("dev.rwm.diffusion.model.utils.sparse_memory", fromlist=["sparse_memory"])
    except ImportError:
        pytest.skip(f"{SANA_SM_REPO} predates the sparse memory (rwm/sparse-memory 3f6f8daa5)")
    if not os.path.abspath(module.__file__).startswith(os.path.abspath(SANA_SM_REPO)):
        pytest.skip(f"dev is already imported from another checkout ({module.__file__}); run the live files on their own")
    if not hasattr(module, "global_rows"):
        pytest.skip(f"{SANA_SM_REPO} predates memory_type global (rwm/sparse-memory eef7beeab)")
    return module


def live_causal():
    """The causal net module with the sparse-memory deploy entries, or skip."""

    live_sparse_memory()
    net = pytest.importorskip("dev.rwm.diffusion.model.nets.sana_qwennext_policy_causal")
    cls = net.SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewCausal
    if not hasattr(cls, "forward_memory_target"):
        pytest.skip(f"{SANA_SM_REPO} predates the sparse-memory deploy form (rwm/sparse-memory da9693c1a)")
    return net


def live_session():
    """Sana's ``sparse_memory_session`` module (``SparseMemoryEpisode``, ``SparseMemoryPolicySession``,
    ``encode_prefix``), or skip."""

    live_causal()
    return pytest.importorskip("dev.rwm.deploy.causal.sparse_memory_session")


def bind_reference_varlen(monkeypatch) -> None:
    """``flash_attn.flash_attn_varlen_func`` -> the fp64 reference (Sana's causal cross-attention imports it at call time)."""

    module = types.ModuleType("flash_attn")
    module.flash_attn_varlen_func = reference_varlen
    monkeypatch.setitem(sys.modules, "flash_attn", module)


__all__ = [
    "SANA_SM_REPO",
    "bind_reference_varlen",
    "live_causal",
    "live_session",
    "live_sparse_memory",
    "reference_varlen",
]
