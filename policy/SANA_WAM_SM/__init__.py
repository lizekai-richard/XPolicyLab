# XPolicyLab adapter package for the SANA sparse-memory policy (chunk-causal GDN + softmax blocks over one sequence:
# the episode's first frame, a memory of frames sampled from its history and the target chunk).
#
# Intentionally import-free: setup_policy_server.py imports ``XPolicyLab.policy.SANA_WAM_SM.model``
# directly, and importing the model here would pull torch and the vendored packages into every
# ``XPolicyLab.policy`` scan (including the registry scan of other adapters' environments).
