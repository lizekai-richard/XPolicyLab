"""sys.path for the CPU tests: policy/SANA_WAM_CAUSAL (``sana_wam_causal``), policy/SANA_WAM (``sana_wam_min``) and the
XPolicyLab parent (``XPolicyLab.*`` imports of the adapter module)."""

from __future__ import annotations

import os
import sys

ADAPTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SANA_WAM_DIR = os.path.abspath(os.path.join(ADAPTER_DIR, "..", "SANA_WAM"))
XPOLICYLAB_PARENT = os.path.abspath(os.path.join(ADAPTER_DIR, "..", "..", ".."))
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
for _path in (XPOLICYLAB_PARENT, SANA_WAM_DIR, ADAPTER_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# XPolicyLab.utils.load_file imports h5py at module import; the policy test env does not need HDF5 (as SANA_WAM's tests).
try:
    import h5py  # noqa: F401
except ImportError:
    import types as _types

    sys.modules["h5py"] = _types.ModuleType("h5py")
