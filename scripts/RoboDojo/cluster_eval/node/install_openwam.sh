#!/usr/bin/env bash
# =============================================================================
# install_openwam.sh -- build the node-local `openwam` conda env (OpenWAM policy runtime)
# on ONE MIT node. Mirrors install_sana_wam.sh's conventions: node-local prefix under
# $ROOT/envs, caches under $ROOT/.cache, logs under $ROOT/logs + a mirror in the shared home.
#
# DELIBERATE CHOICE -- dependencies only, the `openwam` package itself is NOT pip-installed:
#   * the XPolicyLab OpenWAM adapter resolves `openwam_root` and inserts it at the FRONT of
#     sys.path, then REFUSES to run if `openwam` was imported from anywhere else
#     (model.py _resolve_openwam_root). So the vendored NFS copy must win the import.
#   * `pip install -e` would write openwam.egg-info/ INTO that NFS source tree; six nodes
#     installing at once would race on it (the same hazard install_sana_wam.sh flocks around).
#   Result: each node gets the deps, every node imports the one shared openwam source.
#
# Usage:  bash install_openwam.sh
# Env:    ROBODOJO_ROOT (default /dataset2/jiamingt/robodojo)  CONDA_BIN  OPENWAM_PREFIX
#         OPENWAM_ROOT (default /homes/jiamingt/xpl_zekai/XPolicyLab/policy/OpenWAM/OpenWAM)
#         XPL_ROOT     (default /homes/jiamingt/xpl_zekai/XPolicyLab)
# Logs:   $ROOT/logs/openwam/install.log, mirror ~/robodojo_install/logs/<host>.openwam.{log,status}
# Final line on success: "OPENWAM ENV READY".
# =============================================================================
set -Eeuo pipefail
SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
ROOT=${ROBODOJO_ROOT:-/dataset2/jiamingt/robodojo}
CONDA_BIN=${CONDA_BIN:-$HOME/miniconda3/bin/conda}
ENV_PREFIX=${OPENWAM_PREFIX:-$ROOT/envs/openwam}
XPL_ROOT=${XPL_ROOT:-/homes/jiamingt/xpl_zekai/XPolicyLab}
OPENWAM_ROOT=${OPENWAM_ROOT:-$XPL_ROOT/policy/OpenWAM/OpenWAM}
BENCH_ROOT=$(dirname "$XPL_ROOT")
CACHE=$ROOT/.cache
LOGS=$ROOT/logs/openwam
HOST=$(hostname -s)
MIRROR=$SCRIPT_DIR/logs
PY=$ENV_PREFIX/bin/python
MIN_FREE_GB=40
PYVER=3.11
TORCH_INDEX=https://download.pytorch.org/whl/cu128

mkdir -p "$LOGS" "$MIRROR" "$CACHE"
exec > >(tee -a "$LOGS/install.log" "$MIRROR/$HOST.openwam.log") 2>&1
export PIP_USER=0 PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR=$CACHE/pip TMPDIR=$CACHE/tmp
export XDG_CACHE_HOME=$CACHE/xdg HF_HOME=$CACHE/huggingface CONDA_PKGS_DIRS=$CACHE/conda_pkgs
# one healthy GPU is enough for the install-time check; seeing every device makes torch die on a
# node with a "[GPU requires reset]" card (l2/l4/l6, 2026-09-09)
export CUDA_VISIBLE_DEVICES=${OPENWAM_INSTALL_GPU:-0}
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

log(){ printf '[%s] [%s] %s\n' "$(date '+%F %T')" "$HOST" "$*"; }
die(){ log "FATAL: $*"; echo "FAILED $(date -u +%FT%TZ)" > "$MIRROR/$HOST.openwam.status"; exit 1; }
trap 'die "aborted at line $LINENO"' ERR

log "=== openwam env install on $HOST ==="
log "prefix=$ENV_PREFIX openwam_root=$OPENWAM_ROOT xpl_root=$XPL_ROOT"
[[ -x $CONDA_BIN ]] || die "conda not found at $CONDA_BIN"
[[ -d $OPENWAM_ROOT/openwam ]] || die "vendored openwam package not found at $OPENWAM_ROOT/openwam"
[[ -f $OPENWAM_ROOT/pyproject.toml ]] || die "pyproject.toml missing at $OPENWAM_ROOT"
free=$(df -BG "$ROOT" | tail -1 | awk '{gsub("G","",$4); print $4}')
(( free >= MIN_FREE_GB )) || die "only ${free}G free on $ROOT, need ${MIN_FREE_GB}G"
log "disk ok: ${free}G free"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  gpu /'

if [[ -x $PY ]]; then
  log "env already exists at $ENV_PREFIX ($("$PY" -V 2>&1)); reusing (pip steps are idempotent)"
else
  log "creating conda env python=$PYVER"
  "$CONDA_BIN" create -y -p "$ENV_PREFIX" "python=$PYVER" >/dev/null
fi
log "python: $("$PY" -V 2>&1)"

log "installing torch/torchvision from $TORCH_INDEX"
"$PY" -m pip install -q --index-url "$TORCH_INDEX" torch torchvision
"$PY" -c "import torch; print('  torch', torch.__version__, 'cuda', torch.version.cuda)"

# Dependencies straight from the vendored pyproject, minus torch/torchvision (pinned above to the
# cu128 wheels) -- so this list can never drift from the package's own metadata.
log "resolving dependencies from $OPENWAM_ROOT/pyproject.toml"
mapfile -t DEPS < <("$PY" - "$OPENWAM_ROOT/pyproject.toml" <<'PYEOF'
import re, sys, tomllib
data = tomllib.load(open(sys.argv[1], "rb"))
for dep in data["project"]["dependencies"]:
    name = re.split(r"[<>=!\[ ]", dep.strip(), 1)[0].lower()
    if name in {"torch", "torchvision"}:
        continue
    print(dep)
PYEOF
)
log "  ${#DEPS[@]} dependencies: ${DEPS[*]}"
"$PY" -m pip install -q "${DEPS[@]}"

# XPolicyLab's OWN dependencies. The policy server imports client_server.ws.*, whose codec needs
# msgpack_numpy -- declared by XPolicyLab, NOT by OpenWAM, and absent from the openwam env (the SANA
# lane got it from the pydeps overlay, which we deliberately do not put on this env's PYTHONPATH).
# XPolicyLab itself is NOT pip-installed: it is imported from the NFS checkout via PYTHONPATH, and an
# editable install would race six nodes on an egg-info dir in that shared tree.
log "resolving XPolicyLab dependencies from $XPL_ROOT/pyproject.toml"
mapfile -t XDEPS < <("$PY" - "$XPL_ROOT/pyproject.toml" <<'PYEOF'
import re, subprocess, sys, tomllib
data = tomllib.load(open(sys.argv[1], "rb"))
have_cv2 = subprocess.run([sys.executable, "-c", "import cv2"], capture_output=True).returncode == 0
for dep in data["project"]["dependencies"]:
    name = re.split(r"[<>=!\[ ]", dep.strip(), 1)[0].lower()
    if name in {"torch", "torchvision"}:
        continue
    # opencv-python (already installed for OpenWAM) and opencv-python-headless both provide cv2 and
    # conflict when co-installed; the headless variant is only a packaging preference.
    if name == "opencv-python-headless" and have_cv2:
        continue
    print(dep)
PYEOF
)
log "  ${#XDEPS[@]} XPolicyLab dependencies: ${XDEPS[*]}"
[[ ${#XDEPS[@]} -eq 0 ]] || "$PY" -m pip install -q "${XDEPS[@]}"

log "--- import probe (openwam MUST come from the vendored root, not site-packages) ---"
PYTHONPATH="$BENCH_ROOT:$OPENWAM_ROOT:$XPL_ROOT" TOKENIZERS_PARALLELISM=false "$PY" - <<'PYEOF'
import os, sys, torch
root = os.environ["PYTHONPATH"].split(":")[1]
import openwam
assert os.path.realpath(openwam.__file__).startswith(os.path.realpath(root)), \
    f"openwam imported from {openwam.__file__}, expected under {root}"
print("  openwam      :", openwam.__file__)
from openwam.deploy.server import _load_deploy_yaml, build_server_from_config   # noqa: F401
from openwam.deploy.model_loader import load_from_checkpoint_dir                # noqa: F401
from openwam.dataloader.robodojo_contract import arx_x5_calibration
from openwam.dataloader.transforms.multiview import format_prompt_for_inference # noqa: F401
print("  deploy/loader/dataloader imports OK; arx_x5 arms:", list(arx_x5_calibration()["arms"]))
# the websocket policy-server stack: this is what the first OpenWAM smoke died on (msgpack_numpy)
from client_server.ws.model_server import PolicyServer, PolicyServerConfig  # noqa: F401
import msgpack, msgpack_numpy, websockets, pydantic, cv2                    # noqa: F401
print("  ws server stack OK: msgpack", msgpack.version, "msgpack_numpy", msgpack_numpy.__version__,
      "websockets", websockets.__version__, "pydantic", pydantic.VERSION, "cv2", cv2.__version__)
print("  torch.cuda   :", torch.cuda.is_available(), torch.cuda.device_count(), "device(s)")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    t = torch.zeros(8, device="cuda") + 1
    print(f"  gpu0         : {p.name} {p.total_memory>>20} MiB, sm_{p.major}{p.minor}, tensor ok sum={t.sum().item():.0f}")
    b = torch.zeros(4, dtype=torch.bfloat16, device="cuda") + 1
    print(f"  bf16 on gpu  : ok sum={b.float().sum().item():.0f}")
PYEOF

log "--- XPolicyLab adapter import probe ---"
PYTHONPATH="$BENCH_ROOT:$OPENWAM_ROOT:$XPL_ROOT" TOKENIZERS_PARALLELISM=false "$PY" - <<'PYEOF'
import XPolicyLab.policy.OpenWAM.model as adapter
print("  adapter      :", adapter.__file__)
from XPolicyLab.utils.process_data import get_robot_action_dim_info
print("  arx_x5 dims  :", get_robot_action_dim_info("arx_x5"))
PYEOF

cat > "$ROOT/activate_openwam.sh" <<EOF
# source this to use the node-local OpenWAM env
export OPENWAM_ENV_PREFIX=$ENV_PREFIX
export OPENWAM_PY=$ENV_PREFIX/bin/python
export OPENWAM_ROOT=$OPENWAM_ROOT
export XPL_ROOT=$XPL_ROOT
export PYTHONPATH=$BENCH_ROOT:$OPENWAM_ROOT:$XPL_ROOT\${PYTHONPATH:+:\$PYTHONPATH}
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
EOF
log "wrote $ROOT/activate_openwam.sh"
"$PY" -m pip list --format=freeze 2>/dev/null | sort > "$MIRROR/$HOST.openwam.pip.txt"
log "pip freeze -> $MIRROR/$HOST.openwam.pip.txt ($(wc -l < "$MIRROR/$HOST.openwam.pip.txt") packages)"
log "env size: $(du -sh "$ENV_PREFIX" | cut -f1)"
echo "OK $(date -u +%FT%TZ)" > "$MIRROR/$HOST.openwam.status"
log "OPENWAM ENV READY: $ENV_PREFIX"
