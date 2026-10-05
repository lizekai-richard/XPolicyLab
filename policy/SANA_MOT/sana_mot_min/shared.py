"""Bridge to the runtime the MoT adapter shares with SANA_WAM: the sibling adapter's vendored ``sana_wam_min``.

SANA_MOT re-uses the Sana-free runtime of ``policy/SANA_WAM/sana_wam_min`` (LTX-2.3 causal VAE encoder,
Gemma-2-2B text conditioning, Robot80 normalization and action post-processing, the RoboDojo wire codecs,
the Flow-Euler sampler and the inference-only layer primitives of the video trunk). Importing this module
puts ``policy/SANA_WAM`` on ``sys.path`` so ``sana_wam_min`` resolves as the same top-level package that
SANA_WAM's own ``model.py`` uses (one module identity per process; never the ``XPolicyLab.policy.SANA_WAM``
spelling next to it, which would create a second copy of every class).
"""

from __future__ import annotations

import sys
from pathlib import Path

SANA_WAM_POLICY_DIR = Path(__file__).resolve().parents[2] / "SANA_WAM"

if not (SANA_WAM_POLICY_DIR / "sana_wam_min" / "__init__.py").is_file():
    raise ImportError(
        "SANA_MOT needs the sibling adapter policy/SANA_WAM (its vendored sana_wam_min runtime); "
        f"nothing found at {SANA_WAM_POLICY_DIR / 'sana_wam_min'}"
    )
if str(SANA_WAM_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(SANA_WAM_POLICY_DIR))

import sana_wam_min  # noqa: E402,F401

__all__ = ["SANA_WAM_POLICY_DIR"]
