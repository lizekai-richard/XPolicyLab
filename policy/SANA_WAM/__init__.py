# XPolicyLab adapter package for the SANA unified world-action policy.
#
# Intentionally import-free: setup_policy_server.py imports ``XPolicyLab.policy.SANA_WAM.model``
# directly, and importing the model here would pull torch and the vendored package into every
# ``XPolicyLab.policy`` scan (including the registry scan of other adapters' environments).
