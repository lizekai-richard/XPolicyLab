"""Site configuration for the python helpers of cluster_eval: the CE_* variables exported by lib/site.sh, or -- when a
helper is started on its own -- the same values loaded by sourcing lib/site.sh in a bash subprocess."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_cache: dict[str, str] | None = None


def _load() -> dict[str, str]:
    global _cache
    if _cache is not None:
        return _cache
    env = {k: v for k, v in os.environ.items() if k.startswith("CE_")}
    if "CE_SSH_USER" not in env:                      # not started from a cluster_eval shell entry point
        out = subprocess.run(["bash", "-c", f"source {_HERE / 'site.sh'} >/dev/null && env -0"], capture_output=True, check=False)
        if out.returncode != 0:
            raise SystemExit(f"cluster_eval: cannot load the site configuration ({out.stderr.decode(errors='replace').strip()})")
        for item in out.stdout.split(b"\0"):
            k, _, v = item.decode(errors="replace").partition("=")
            if k.startswith("CE_"):
                env[k] = v
        os.environ.update(env)
    _cache = env
    return env


def get(name: str) -> str:
    """One CE_* value, e.g. get('CE_EVAL_ROOT')."""
    v = _load().get(name)
    if v is None:
        raise SystemExit(f"cluster_eval: {name} is not set (lib/site.sh)")
    return v


def host(node: str) -> str:
    """ssh target of a node: <user>@<node><domain>."""
    return f"{get('CE_SSH_USER')}@{node}{get('CE_SSH_DOMAIN')}"


def gateway() -> str:
    return host(get("CE_GATEWAY"))
