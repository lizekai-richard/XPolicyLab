# XPolicyLab adapter package for the SANA hierarchical-memory policy (chunk-causal GDN + softmax blocks, each reading
# its own memory stream of causal-VAE latents: dense per-chunk latents or uniformly sampled sparse summaries).
#
# Intentionally import-free: setup_policy_server.py imports ``XPolicyLab.policy.SANA_WAM_HM.model``
# directly, and importing the model here would pull torch and the vendored packages into every
# ``XPolicyLab.policy`` scan (including the registry scan of other adapters' environments).
