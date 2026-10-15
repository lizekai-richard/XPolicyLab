"""The live Sana checkout the parity tests compare against (``SANA_HM_REPO``, default the pinned rwm/sparse-memory worktree
~/zekail/Sana_sparse_memory_e8dde6b56): its hierarchical-memory module and its causal policy class. A test module that
imports this one owns the ``dev`` package for its pytest session (run the live files apart from other Sana checkouts).

Two API generations are served: e8dde6b56 (``memory_streams``, ``frame_layout``, ``offsets_x8``, the net's
``memory_layout(form, k, memory)``, ``data_info[...]['frame_offsets_x8']``) and the renamed one of 057412e5e / d83cd2480
(``memory_types_read``, ``frame_indices``, ``offset_eighths``, ``train_layout`` / ``commit_layout`` /
``generate_layout``, ``frame_offset_eighths``), which from 6564adc38 parses ``chunk_observation`` into the memory.
"""

from __future__ import annotations

import inspect
import os
import sys
import types

import pytest

SANA_HM_REPO = os.path.expanduser(os.environ.get("SANA_HM_REPO", "~/zekail/Sana_sparse_memory_e8dde6b56"))
# the mirror's name -> Sana's name after the renames (the same function either way)
RENAMED = {"memory_streams": "memory_types_read", "frame_layout": "frame_indices", "offsets_x8": "offset_eighths"}


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
    memory = live_hm()
    net = pytest.importorskip("dev.rwm.diffusion.model.nets.sana_qwennext_policy_causal")
    if not (hasattr(net, "memory_layout") or hasattr(memory, "train_layout")):
        pytest.skip(f"{SANA_HM_REPO} predates the hierarchical memory (rwm/sparse-memory)")
    return net


def live_function(name: str):
    """The live geometry function the mirror calls ``name`` (under its renamed name where Sana renamed it)."""

    memory = live_hm()
    return getattr(memory, RENAMED.get(name, name), None) or getattr(memory, name)


def has_chunk_observation() -> bool:
    """Whether the live checkout parses ``chunk_causal_policy.chunk_observation`` (Sana 6564adc38 on)."""

    return "chunk_observation" in inspect.signature(live_hm().parse_memory).parameters


def require_chunk_observation(chunk_observation: bool) -> None:
    if not chunk_observation and not has_chunk_observation():
        pytest.skip(f"{SANA_HM_REPO} predates chunk_observation (Sana 6564adc38)")


def live_memory(block, chunk_observation: bool = True) -> dict:
    """``block`` ({gdn, softmax, sparse_frames}) parsed by the live checkout: with ``chunk_observation`` from 6564adc38 on."""

    require_chunk_observation(chunk_observation)
    if has_chunk_observation():
        return live_hm().parse_memory(dict(block), chunk_observation)
    return live_hm().parse_memory(dict(block))


def live_layout(form: str, k: int, memory) -> dict:
    """Sana's layout of a forward of ``form``: the net's ``memory_layout`` before 057412e5e, then the module's
    ``train_layout`` / ``commit_layout`` / ``generate_layout``; ``memory`` is a ``live_memory``."""

    module = live_hm()
    if not hasattr(module, "train_layout"):
        return live_causal().memory_layout(form, k, memory)
    if form == "train":
        return module.train_layout(k, memory)
    if form == "commit":
        return module.commit_layout(memory)
    if form == "generate":
        return module.generate_layout(memory)
    raise ValueError(f"no Sana layout for form {form!r}")


def offsets_key() -> str:
    """The key of the frame offsets in ``data_info['hierarchical_memory']`` (renamed by d83cd2480)."""

    return "frame_offset_eighths" if hasattr(live_hm(), "offset_eighths") else "frame_offsets_x8"


def bind_reference_varlen(monkeypatch) -> None:
    """``flash_attn.flash_attn_varlen_func`` -> the fp64 reference (Sana's causal cross-attention imports it at call time)."""

    module = types.ModuleType("flash_attn")
    module.flash_attn_varlen_func = reference_varlen
    monkeypatch.setitem(sys.modules, "flash_attn", module)


__all__ = [
    "RENAMED",
    "SANA_HM_REPO",
    "bind_reference_varlen",
    "has_chunk_observation",
    "live_causal",
    "live_function",
    "live_hm",
    "live_layout",
    "live_memory",
    "offsets_key",
    "reference_varlen",
    "require_chunk_observation",
]
