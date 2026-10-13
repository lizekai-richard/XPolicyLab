#!/usr/bin/env bash
# =============================================================================
# install_sana_wam.sh -- build the `sana-wam` conda env (SANA model runtime,
# used by the RoboDojo policy server) on one node, node-local.
#
# Recipe = Sana/environment_setup.sh (rwm/zekai @ bd9bf130d) with three
# deliberate deltas, each justified in the step that applies it:
#   * no conda cuda-toolkit: nothing here compiles CUDA (mmcv is installed
#     without ops, flash-attn is optional in the code and not installed)
#   * mmcv==1.7.2 with MMCV_WITH_OPS=0: the repo only uses Registry/Config/
#     runner; the nodes have no compiler and no nvcc
#   * the editable `pip install -e <Sana>` is serialized across nodes with an
#     NFS flock because the repo lives in the shared home
#
# Usage:  bash install_sana_wam.sh [--from STEP | --only STEP | --list]
# Env:    ROBODOJO_ROOT (default /dataset2/jiamingt/robodojo)  CONDA_BIN
#         SANA_REPO (default $HOME/Sana)  SANA_WAM_PREFIX (default $ROOT/envs/sana-wam)
# Logs:   $ROOT/logs/sana_wam/<step>.log, mirror <this dir>/logs/<host>.sana_wam.{log,status}
# =============================================================================
set -Eeuo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
ROOT=${ROBODOJO_ROOT:-/dataset2/jiamingt/robodojo}
CONDA_BIN=${CONDA_BIN:-$HOME/miniconda3/bin/conda}
SANA_REPO=${SANA_REPO:-$HOME/Sana}
ENV_NAME=sana-wam
ENV_PREFIX=${SANA_WAM_PREFIX:-$ROOT/envs/$ENV_NAME}
CACHE=$ROOT/.cache
LOGS=$ROOT/logs/sana_wam
STATE=$ROOT/state/sana_wam
HOST=$(hostname -s)
MIRROR_DIR=$SCRIPT_DIR/logs
TAG=$HOST.sana_wam
PY=$ENV_PREFIX/bin/python
SANA_EXPECTED_COMMIT=bd9bf130d65ed347c5e569b505db4871d6c77f30
MIN_FREE_GB=60

STEPS=(preflight conda_env pip_base torch xformers mmcv sana_editable extras
       import_probe contract_tests activation audit)

export PIP_USER=0 PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR=$CACHE/pip TMPDIR=$CACHE/tmp
export XDG_CACHE_HOME=$CACHE/xdg XDG_DATA_HOME=$CACHE/xdg_data
export HF_HOME=$CACHE/huggingface CONDA_PKGS_DIRS=$CACHE/conda_pkgs
export GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1
export TERM=${TERM:-xterm-256color}
# The server launcher passes only PYTHONNOUSERSITE=1 PYTHONPATH=$SANA_REPO to
# python; SANA_DEV_USERS (dev-module loading) is left to the caller.
# one healthy GPU is enough for the install-time checks; seeing every device makes torch 2.9 die on nodes with a
# "[GPU requires reset]" card (l2/l4/l6, 2026-09-09)
export CUDA_VISIBLE_DEVICES=${SANA_WAM_INSTALL_GPU:-0}

log()    { printf '[%s] [%s] %s\n' "$(date '+%F %T')" "$HOST" "$*"; }
die()    { log "FATAL: $*"; exit 1; }
status() { local line; line="$(date -Is) $HOST $1 $2 ${3:-}"; echo "$line" >> "$STATE/status.txt"; echo "$line" >> "$MIRROR_DIR/$TAG.status" 2>/dev/null || true; }
conda_root() { cd "$(dirname "$CONDA_BIN")/.." && pwd; }
use_env() { export PATH="$ENV_PREFIX/bin:$(conda_root)/bin:$PATH"; }
pip() { "$PY" -m pip "$@"; }

# ----------------------------------------------------------------------------
step_preflight() {
  log "root=$ROOT env=$ENV_PREFIX sana_repo=$SANA_REPO conda=$CONDA_BIN"
  [[ -x $CONDA_BIN ]] || die "conda not found at $CONDA_BIN"
  command -v git >/dev/null || die "git missing"
  [[ -d $SANA_REPO/.git ]] || die "$SANA_REPO is not a git checkout"
  [[ -f $SANA_REPO/pyproject.toml && -f $SANA_REPO/environment_setup.sh ]] || die "$SANA_REPO lacks pyproject.toml/environment_setup.sh"
  local head; head=$(git -C "$SANA_REPO" rev-parse HEAD)
  if [[ $head == "$SANA_EXPECTED_COMMIT" ]]; then log "Sana HEAD $head (expected)"; else log "WARN: Sana HEAD $head != expected $SANA_EXPECTED_COMMIT (env recipe files are identical across these commits, continuing)"; fi
  local free_gb; free_gb=$(df -P --block-size=1G "$ROOT" | awk 'NR==2{print $4}')
  log "free space on $(df -P "$ROOT" | awk 'NR==2{print $6}'): ${free_gb}G"; (( free_gb >= MIN_FREE_GB )) || die "need >= ${MIN_FREE_GB}G free"
  command -v nvidia-smi >/dev/null && log "gpu0='$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)' driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  local u code
  for u in https://github.com https://pypi.org/simple/pip/ https://download.pytorch.org/whl/cu128/torch/; do
    code=$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$u" || echo 000); log "net $u -> $code"; [[ $code != 000 ]] || die "cannot reach $u"
  done
}

step_conda_env() {
  local croot; croot=$(conda_root)
  if [[ ! -x $ENV_PREFIX/bin/python ]]; then
    log "creating $ENV_PREFIX (python 3.11, conda-forge)"
    "$CONDA_BIN" create -y --override-channels -c conda-forge -p "$ENV_PREFIX" python=3.11 pip
  fi
  if [[ "$(readlink -f "$ENV_PREFIX")" != "$(readlink -f "$croot/envs/$ENV_NAME" 2>/dev/null)" ]]; then
    mkdir -p "$croot/envs"
    if [[ -e $croot/envs/$ENV_NAME && ! -L $croot/envs/$ENV_NAME ]]; then die "$croot/envs/$ENV_NAME exists and is not a symlink"; fi
    ln -sfn "$ENV_PREFIX" "$croot/envs/$ENV_NAME"; log "registered name: $croot/envs/$ENV_NAME -> $ENV_PREFIX"
  fi
  use_env; log "python: $("$PY" --version) at $PY"
}

step_pip_base() { use_env; pip install -U pip wheel; pip install "setuptools<80"; }

step_torch() {
  use_env
  pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
  "$PY" - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
assert torch.__version__.startswith("2.9.1+cu128"), torch.__version__
assert torch.cuda.is_available()
PY
}

step_xformers() {
  use_env
  pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 xformers==0.0.33.post2
  "$PY" - <<'PY'
import torch, xformers, xformers.ops  # noqa: F401
assert torch.__version__.startswith("2.9.1+cu128"), "xformers install changed torch: " + torch.__version__
print("xformers", xformers.__version__, "torch still", torch.__version__)
PY
}

step_mmcv() {
  use_env
  # Registry/Config/runner only; no compiled ops anywhere in the repo, and the
  # nodes have neither a compiler nor nvcc, so build the ops-free variant.
  MMCV_WITH_OPS=0 pip install mmcv==1.7.2 --no-build-isolation --no-cache-dir
  "$PY" - <<'PY'
import mmcv
from mmcv import Registry, Config  # noqa: F401
from mmcv.runner import get_dist_info  # noqa: F401
from mmcv.utils import _BatchNorm  # noqa: F401
print("mmcv", mmcv.__version__, "ops-free import OK")
PY
}

step_sana_editable() {
  use_env
  # The repo is on the shared NFS home: serialize the editable install across
  # nodes (it writes egg-info/build metadata into the repo).
  local lock=$SCRIPT_DIR/.sana_editable.lock
  log "waiting for NFS lock $lock"
  exec 8>"$lock"; flock -w 3600 8 || die "could not acquire $lock in 3600s"
  log "lock acquired; pip install -e $SANA_REPO"
  pip install -e "$SANA_REPO"
  flock -u 8
  "$PY" - <<'PY'
import torch, sana  # noqa: F401
assert torch.__version__.startswith("2.9.1+cu128"), "editable install changed torch: " + torch.__version__
import transformers, diffusers, accelerate, timm, einops, pyrallis, imageio, yaml, numpy, decord, peft
print("transformers", transformers.__version__, "diffusers", diffusers.__version__, "accelerate", accelerate.__version__,
      "timm", timm.__version__, "peft", peft.__version__, "numpy", numpy.__version__)
PY
}

step_extras() { use_env; pip install pytest; "$PY" -m pytest --version; }

step_import_probe() {
  use_env
  cd "$SANA_REPO"
  PYTHONPATH=$SANA_REPO timeout 900 "$PY" - <<'PY'
import importlib
import diffusion.model.builder  # mmcv Registry path
import diffusion.utils.misc     # mmcv Config path
import diffusion.model.registry  # registry module itself
for mod in ["dev.rwm.diffusion.policy_inference",
            "dev.rwm.simulation_evaluation.policy_runtime.in_process",
            "dev.rwm.simulation_evaluation.policy_runtime.rpc_server_main",
            "dev.rwm.simulation_evaluation.service.server",
            "dev.rwm.simulation_evaluation.robodojo.server.main"]:
    importlib.import_module(mod); print("import OK", mod)
print("imports OK")
PY
}

step_contract_tests() {
  # simulator-free contract suite (EVAL/README.md): must pass without any
  # benchmark installed. WARN-only: failures here are for a human to read.
  use_env; cd "$SANA_REPO"
  local out=$LOGS/contract_tests_output.txt rc=0
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SANA_REPO timeout 1800 "$PY" -m pytest -q -p no:cacheprovider dev/rwm/tests/simulation_evaluation > "$out" 2>&1 || rc=$?
  tail -n 5 "$out"
  if (( rc == 0 )); then log "contract tests PASSED: $(tail -n 1 "$out")"; else log "WARN: contract tests rc=$rc: $(tail -n 1 "$out")"; echo "contract_tests rc=$rc" > "$STATE/contract_tests.WARN"; fi
}

step_activation() {
  cat > "$ROOT/activate_sana_wam.sh" <<EOF
# source this to use the sana-wam runtime on this node
export SANA_REPO=$SANA_REPO
export SANA_PY=$PY
export PYTHONPATH=$SANA_REPO\${PYTHONPATH:+:\$PYTHONPATH}
export HF_HOME=$CACHE/huggingface XDG_CACHE_HOME=$CACHE/xdg TMPDIR=$CACHE/tmp
export PIP_USER=0 PYTHONNOUSERSITE=1
export PATH=$ENV_PREFIX/bin:$(conda_root)/bin:\$PATH
# HF token lives once in the shared home (chmod 600); HF_HOME above is node-local
export HF_TOKEN_PATH=\$HOME/.hf_token
# served checkpoint + VAE defaults (names = run_learned_server_only.sbatch env contract); the eval master
# overrides the checkpoint per run (SANA_EVAL_CHECKPOINT_ROOT) and run_unit.sh points the input-cache dir
# at <checkpoint>/json_cache when the snapshot ships one
export SANA_POLICY_CHECKPOINT_ROOT=\$HOME/checkpoints/huggingface/logits/sana_rwm_policy_from_pretrained_robodojo_320px_epoch4_step25000
export SANA_POLICY_VAE_PRETRAINED=$SANA_REPO/output/pretrained_models/LTX-2.3-Diffusers
export SANA_NORMALIZATION_INPUT_CACHE_DIR=\$SANA_POLICY_CHECKPOINT_ROOT/json_cache
EOF
  if [[ -f $ROOT/activate.sh ]] && ! grep -q 'SANA_PY=' "$ROOT/activate.sh"; then
    printf '\n# sana-wam runtime (added by install_sana_wam.sh)\nexport SANA_PY=%s\nexport SANA_REPO=%s\n' "$PY" "$SANA_REPO" >> "$ROOT/activate.sh"
    log "appended SANA_PY/SANA_REPO to $ROOT/activate.sh"
  fi
  log "wrote $ROOT/activate_sana_wam.sh"; cat "$ROOT/activate_sana_wam.sh"
}

step_audit() {
  use_env
  pip freeze --all > "$LOGS/freeze.txt"; log "pip check:"; pip check || true
  if [[ -f $SCRIPT_DIR/reference_freeze_sana_wam_x5.txt ]]; then
    grep -E '^[A-Za-z0-9][A-Za-z0-9._-]*==' "$LOGS/freeze.txt" | sort -f > "$LOGS/freeze.pinned.txt"
    sort -f "$SCRIPT_DIR/reference_freeze_sana_wam_x5.txt" > "$LOGS/reference.sorted.txt"
    diff "$LOGS/reference.sorted.txt" "$LOGS/freeze.pinned.txt" > "$LOGS/freeze_diff_vs_x5.txt" || true
    log "package diff vs the x5 reference build: $(grep -c '^[<>]' "$LOGS/freeze_diff_vs_x5.txt" || true) lines ($LOGS/freeze_diff_vs_x5.txt)"
  fi
  log "disk: env $(du -sh "$ENV_PREFIX" | cut -f1)"
  { echo "host=$HOST"; echo "date=$(date -Is)"; echo "sana_py=$PY"; echo "sana_repo=$SANA_REPO"; echo "sana_commit=$(git -C "$SANA_REPO" rev-parse HEAD)"
    echo "contract_tests=$([[ -f $STATE/contract_tests.WARN ]] && echo WARN || echo OK)"; } > "$STATE/INSTALL_COMPLETE"
  cp "$STATE/INSTALL_COMPLETE" "$MIRROR_DIR/$TAG.INSTALL_COMPLETE" 2>/dev/null || true
  log "SANA-WAM INSTALL COMPLETE on $HOST"
}

# ----------------------------------------------------------------------------
if [[ ${1:-} == --run-step ]]; then step="${2:?}"; "step_$step"; exit 0; fi
if [[ ${1:-} == --list ]]; then printf '%s\n' "${STEPS[@]}"; exit 0; fi

mkdir -p "$ROOT" "$LOGS" "$STATE" "$CACHE"/{pip,tmp,xdg,xdg_data,huggingface,conda_pkgs} "$MIRROR_DIR"
exec 9>"$STATE/.lock"; flock -n 9 || die "another install_sana_wam.sh is running on $HOST"
exec > >(tee -a "$LOGS/install.log" "$MIRROR_DIR/$TAG.log") 2>&1

ONLY=""; FROM=""
case "${1:-}" in --only) ONLY=${2:?};; --from) FROM=${2:?};; "") ;; *) die "unknown argument: $1";; esac
if [[ -n $FROM ]]; then c=0; for s in "${STEPS[@]}"; do [[ $s == "$FROM" ]] && c=1; (( c )) && rm -f "$STATE/$s.done"; done; (( c )) || die "unknown step: $FROM"; fi
if [[ -n $ONLY ]]; then printf '%s\n' "${STEPS[@]}" | grep -qx "$ONLY" || die "unknown step: $ONLY"; fi

log "===== install_sana_wam.sh start (env=$ENV_PREFIX, args: ${*:-none}) ====="
status all START
for s in "${STEPS[@]}"; do
  if [[ -n $ONLY && $ONLY != "$s" ]]; then continue; fi
  if [[ -z $ONLY && -f $STATE/$s.done ]]; then log "skip $s (done $(date -r "$STATE/$s.done" '+%F %T'))"; continue; fi
  log "=== BEGIN $s ==="; status "$s" START; t0=$(date +%s)
  if bash "$SCRIPT_PATH" --run-step "$s" > >(tee -a "$LOGS/$s.log") 2>&1; then
    touch "$STATE/$s.done"; status "$s" DONE "$(( $(date +%s) - t0 ))s"; log "=== DONE $s in $(( $(date +%s) - t0 ))s ==="
  else
    status "$s" FAIL "$(( $(date +%s) - t0 ))s"; log "=== FAILED $s (log: $LOGS/$s.log) ==="; log "fix, then re-run: bash $SCRIPT_PATH   (resumes at $s)"; exit 1
  fi
done
status all DONE
log "===== all steps finished on $HOST ====="
