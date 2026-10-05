# XPolicyLab adapter package for the SANA chunk-causal policy (teacher-forced GDN recurrence + softmax sliding window).
#
# Intentionally import-free: setup_policy_server.py imports ``XPolicyLab.policy.SANA_WAM_CAUSAL.model``
# directly, and importing the model here would pull torch and the vendored packages into every
# ``XPolicyLab.policy`` scan (including the registry scan of other adapters' environments).
