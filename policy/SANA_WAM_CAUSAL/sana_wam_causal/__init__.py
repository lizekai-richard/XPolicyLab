"""Sana-free inference of the SANA chunk-causal policy (Sana rwm/zekai-merge ``sana_qwennext_policy_causal.py``).

The trunk is the bidirectional 5B policy's (``sana_wam_min.policy_model``); this package adds the chunk-causal
attention shells (GDN chunk recurrence with an ``(S, z)`` state, softmax attention over a sliding window of committed
chunks), the three deploy windows of the streaming session (observation prefill, the chunk-0 window that carries the
observation, the cached chunk window) and the teacher-forced deploy session (commit the executed chunk, generate the
next one).
"""

from . import shared  # noqa: F401  (puts policy/SANA_WAM on sys.path before anything imports sana_wam_min)
