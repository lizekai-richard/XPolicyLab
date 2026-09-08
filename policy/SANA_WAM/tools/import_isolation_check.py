#!/usr/bin/env python
"""Assert that the XPolicyLab SANA_WAM adapter and sana_wam_min import nothing from the Sana repo.

Imports ``XPolicyLab.policy.SANA_WAM.model`` and every ``sana_wam_min`` submodule, then scans
``sys.modules`` for forbidden roots (``dev``, ``diffusion*``, ``sana*`` other than ``sana_wam_min``,
any ``XPolicyLab.policy.<other adapter>``) and prints the third-party top-level packages that were
pulled in.

Exit codes: 0 clean; 1 isolation violation (or a sana_wam_min submodule failed to import);
2 the adapter module itself failed to import (environment gap such as a missing dependency).

Run with the Sana repo NOT on PYTHONPATH; the tool puts the XPolicyLab parent dir and the adapter
dir on sys.path itself so both import spellings are exercised.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import traceback
from pathlib import Path

_ADAPTER_DIR = Path(__file__).resolve().parents[1]          # .../XPolicyLab/policy/SANA_WAM
_XPOLICYLAB_ROOT = _ADAPTER_DIR.parents[1]                  # .../XPolicyLab
_XPOLICYLAB_PARENT = _XPOLICYLAB_ROOT.parent                # dir that makes `import XPolicyLab` a package

FORBIDDEN_EXACT = ("dev", "sana")
FORBIDDEN_PREFIXES = ("dev.", "diffusion", "sana.", "sana_")
ALLOWED_OWN = ("sana_wam_min",)
ADAPTER_PACKAGE = "XPolicyLab.policy.SANA_WAM"


def forbidden_modules(module_names) -> list[str]:
    """Return the sorted module names that violate the isolation contract."""

    bad = []
    for name in module_names:
        root = name.split(".")[0]
        if name.startswith(ADAPTER_PACKAGE + ".") and any(part in ALLOWED_OWN for part in name.split(".")):
            continue
        if root in ALLOWED_OWN:
            continue
        if root in FORBIDDEN_EXACT or any(name.startswith(p) for p in FORBIDDEN_PREFIXES):
            bad.append(name)
        elif name.startswith("XPolicyLab.policy.") and not name.startswith(ADAPTER_PACKAGE):
            bad.append(name)
    return sorted(bad)


def third_party_roots(module_names) -> list[str]:
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    roots = set()
    for name in module_names:
        root = name.split(".")[0]
        if root in stdlib or root.startswith("_") or root in ("XPolicyLab", "__main__", "__mp_main__") or root in ALLOWED_OWN:
            continue
        roots.add(root)
    return sorted(roots)


def import_all_submodules(package_name: str) -> list[str]:
    package = importlib.import_module(package_name)
    failures = []
    for info in pkgutil.walk_packages(package.__path__, prefix=package_name + "."):
        try:
            importlib.import_module(info.name)
        except Exception:  # noqa: BLE001
            failures.append(f"{info.name}: {traceback.format_exc(limit=3)}")
    return failures


def main() -> int:
    for path in (str(_ADAPTER_DIR), str(_XPOLICYLAB_PARENT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    before = set(sys.modules)
    rc = 0

    try:
        importlib.import_module(ADAPTER_PACKAGE + ".model")
        print(f"imported {ADAPTER_PACKAGE}.model")
    except Exception:  # noqa: BLE001
        print(f"ADAPTER IMPORT ERROR: {ADAPTER_PACKAGE}.model\n{traceback.format_exc()}")
        rc = 2

    failures = []
    for spelling in ("sana_wam_min", ADAPTER_PACKAGE + ".sana_wam_min"):
        try:
            failures += import_all_submodules(spelling)
            print(f"imported {spelling} and all submodules")
        except Exception:  # noqa: BLE001
            failures.append(f"{spelling}: {traceback.format_exc(limit=3)}")
    if failures:
        print("SUBMODULE IMPORT FAILURES:")
        for failure in failures:
            print("  " + failure.replace("\n", "\n  "))
        rc = max(rc, 1)

    bad = forbidden_modules(sys.modules)
    if bad:
        print("ISOLATION VIOLATION - forbidden modules present in sys.modules:")
        for name in bad:
            module = sys.modules.get(name)
            print(f"  {name}  ({getattr(module, '__file__', None)})")
        rc = max(rc, 1)
    else:
        print("isolation OK: no dev./diffusion*/sana*/other-adapter modules imported")

    new_roots = third_party_roots(set(sys.modules) - before)
    print("third-party top-level packages imported: " + ", ".join(new_roots))
    print(f"exit {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
