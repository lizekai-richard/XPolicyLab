# XPolicyLab adapter package for the SANA MoT (dual-expert video + action) policy.
#
# Intentionally import-free: setup_policy_server.py imports ``XPolicyLab.policy.SANA_MOT.model``
# directly, and importing the model here would pull torch and the vendored packages into every
# ``XPolicyLab.policy`` scan (including the registry scan of other adapters' environments).
