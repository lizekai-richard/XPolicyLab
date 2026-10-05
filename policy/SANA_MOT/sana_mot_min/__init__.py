"""Sana-free inference package for the SANA MoT (AttnRes dual-system) policy.

The MoT model mirror lives in ``mot_model/``; the canvas front-end, the shared-prompt renderer, the
config resolver and the inference session are the top-level modules. Everything the MoT line shares
with the single-stream SANA_WAM line (LTX-2.3 causal VAE encoder, Gemma-2-2B text conditioning,
Robot80 normalization and action post-processing, the RoboDojo codecs, the Flow-Euler sampler and
the inference-only trunk layers) is imported from the sibling adapter's ``sana_wam_min`` package
through :mod:`sana_mot_min.shared`, which this package imports first.
"""

from . import shared  # noqa: F401  (puts policy/SANA_WAM on sys.path before any sana_wam_min import)

__version__ = "0.1.0"
