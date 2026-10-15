"""Sana-free inference of the SANA sparse-memory policy (Sana rwm/sparse-memory ``sana_qwennext_policy_causal.py``).

The trunk is the bidirectional 5B policy's (``sana_wam_min.policy_model``) with the chunk-causal shells of the causal
line; this package adds the sparse memory: every decision encodes [sink | memory rows] -- the episode's first frame and
rows sampled from its history (``memory_type`` fixed: m rows uniformly over it; strided: every s rows back from the
observation) -- in one causal VAE pass, runs it once into per-layer caches and denoises the target chunk [T | state |
actions] against them (``sparse_memory.py`` holds the geometry, ``model.py`` the trunk and its three forwards,
``session.py`` the decision).
"""

from . import shared  # noqa: F401  (puts policy/SANA_WAM on sys.path before anything imports sana_wam_min)
