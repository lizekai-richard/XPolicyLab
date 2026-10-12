"""Sana-free inference of the SANA hierarchical-memory policy (Sana rwm/sparse-memory ``sana_qwennext_policy_causal.py``).

The trunk is the bidirectional 5B policy's (``sana_wam_min.policy_model``) with the chunk-causal shells of the causal
line; this package adds the hierarchical memory: two memory streams of causal-VAE latents (a dense one, one latent per
executed chunk, and a sparse one, the first frame plus summaries of rows sampled uniformly over the history), each block
kind reading one of them through block-level routing, the streaming commit / generate forms and the per-episode deploy
session (``memory.py`` holds the geometry, ``model.py`` the routed trunk, ``session.py`` the decision cycle).
"""

from . import shared  # noqa: F401  (puts policy/SANA_WAM on sys.path before anything imports sana_wam_min)
