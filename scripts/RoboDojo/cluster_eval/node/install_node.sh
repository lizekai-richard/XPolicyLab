#!/usr/bin/env bash
# =============================================================================
# install_node.sh -- RoboDojo simulator stack, per-node installer for the GPU
# nodes of the site in site.env (written for the MIT lN RTX A6000 nodes).
#
# Everything lands on NODE-LOCAL disk (default CE_NODE_ROOT from site.env) so
# the four nodes never contend on the shared NFS home. Only this script
# directory and a small per-host mirror of the log/status live in ~/.
#
# Sources of truth:
#   * Sana dev/rwm/simulation_evaluation/robodojo/INSTALL.md (rwm/zekai
#     @ bd9bf130d), Sections 2-6, and launch/remote_client/README.md
#   * the 19-step log of the working RTX 5090 install (2026-08-30)
#
# Usage:
#   bash install_node.sh                 run all steps, skipping finished ones
#   bash install_node.sh --from STEP     clear markers from STEP onward, then run
#   bash install_node.sh --only STEP     run exactly one step (marker ignored)
#   bash install_node.sh --list          list steps in order
# Env overrides: ROBODOJO_ROOT (default CE_NODE_ROOT from site.env), CONDA_BIN
#
# Re-running is safe: each step writes a marker in $ROOT/state when it
# finishes and is skipped next time. Logs: $ROOT/logs/<step>.log plus
# $ROOT/logs/install.log; a mirror of install.log and the status stream is
# kept in <this dir>/logs/<host>.log and <host>.status on NFS.
# =============================================================================
set -Eeuo pipefail
_CE_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
[[ -f $_CE_HERE/site.env ]] && source "$_CE_HERE/site.env"   # the site configuration deploy.sh copies next to the node scripts

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
ROOT=${ROBODOJO_ROOT:-${CE_NODE_ROOT:?no site.env next to this script (deploy.sh) and no ROBODOJO_ROOT}}
CONDA_BIN=${CONDA_BIN:-$HOME/miniconda3/bin/conda}

CHECKOUT=$ROOT/RoboDojo-official-clone-20260823
ENV_NAME=robodojo-sana-eval
ENV_PREFIX=$ROOT/envs/$ENV_NAME
WS_PREFIX=$ROOT/envs/robodojo-ws
CACHE=$ROOT/.cache
LOGS=$ROOT/logs
STATE=$ROOT/state
HOST=$(hostname -s)
MIRROR_DIR=$SCRIPT_DIR/logs
PY=$ENV_PREFIX/bin/python

PIN_ROOT=2184bf8844ea9d205382c4aefa3a694311418251
PIN_XPOLICYLAB=432f82b1758c5b1202e42a3dfe014546dbc50871
PIN_ISAACLAB=afca7b09d60d8beb9c1cb28b43066499940b969b
PIN_CUROBO=d17b54ce32cba095c0b000c4c58777075d11de0e

# Reference numbers from the complete 5090 asset tree, EXCLUDING the two
# curobo.yml files that update_embodiment_config_path.py renders later (they
# are not part of the HF dataset): find -L Assets -type f ! -name curobo.yml
REF_ASSET_FILES=15365
REF_ASSET_BYTES=41269490671
MIN_FREE_GB=200

# conda_env comes BEFORE checkout on purpose: it supplies git-lfs, which the
# XPolicyLab submodule needs at checkout time (the nodes have no system git-lfs).
STEPS=(preflight conda_env checkout base_pip torch isaacsim pins_isaacsim
       isaaclab pins_isaaclab curobo pins_curobo runtime_libs import_probe
       assets render_configs ws_env doctor render_probe activation audit)

# ----------------------------------------------------------------------------
# shared environment (exported so --run-step children inherit it)
# ----------------------------------------------------------------------------
export PIP_USER=0 PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR=$CACHE/pip TMPDIR=$CACHE/tmp
export XDG_CACHE_HOME=$CACHE/xdg XDG_DATA_HOME=$CACHE/xdg_data
export HF_HOME=$CACHE/huggingface CONDA_PKGS_DIRS=$CACHE/conda_pkgs
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export GIT_TERMINAL_PROMPT=0
# Never let a plain checkout invoke the lfs smudge filter (it fails hard when
# git-lfs is missing); LFS content is fetched explicitly with `git lfs pull`.
export GIT_LFS_SKIP_SMUDGE=1
export TERM=${TERM:-xterm-256color}
unset CUDA_VISIBLE_DEVICES || true

log()    { printf '[%s] [%s] %s\n' "$(date '+%F %T')" "$HOST" "$*"; }
die()    { log "FATAL: $*"; exit 1; }
status() { # status STEP STATE [note]
  local line; line="$(date -Is) $HOST $1 $2 ${3:-}"
  echo "$line" >> "$STATE/status.txt"
  echo "$line" >> "$MIRROR_DIR/$HOST.status" 2>/dev/null || true
}
conda_root() { cd "$(dirname "$CONDA_BIN")/.." && pwd; }
use_env() {   # put the sim env (and conda) first on PATH
  export PATH="$ENV_PREFIX/bin:$(conda_root)/bin:$PATH"
}
conda_activate_env() { # real activation (CONDA_PREFIX etc.), needed by isaaclab.sh
  set +u
  # shellcheck disable=SC1091
  source "$(conda_root)/etc/profile.d/conda.sh"
  conda activate "$ENV_PREFIX"
  set -u
}
apply_pins() {
  log "re-applying pin set (INSTALL.md section 3)"
  "$PY" -m pip install -r "$SCRIPT_DIR/pins.txt"
  "$PY" -m pip uninstall -y python-discovery 2>/dev/null || true
}
pip_check_report() {
  log "pip check:"
  "$PY" -m pip check || true
  log "(the isaaclab/starlette==0.49.1 complaint is upstream-intended; anything else is worth a look)"
}

# ----------------------------------------------------------------------------
# steps
# ----------------------------------------------------------------------------
step_preflight() {
  log "root=$ROOT conda=$CONDA_BIN"
  [[ -x $CONDA_BIN ]] || die "conda not found at $CONDA_BIN"
  log "conda version: $("$CONDA_BIN" --version 2>&1)"
  command -v git >/dev/null || die "git missing"
  command -v nvidia-smi >/dev/null || die "nvidia-smi missing"
  local drv gpu major
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  major=${drv%%.*}
  log "gpu0='$gpu' driver=$drv gpus=$(nvidia-smi -L | wc -l)"
  (( major >= 570 )) || die "driver $drv below the >=570 floor"
  if [[ $gpu =~ (A100|H100|H200|B200) ]]; then die "GPU '$gpu' has no RT cores; Isaac Sim cannot render"; fi
  local free_gb
  free_gb=$(df -P --block-size=1G "$ROOT" | awk 'NR==2{print $4}')
  log "free space on $(df -P "$ROOT" | awk 'NR==2{print $6}'): ${free_gb}G"
  (( free_gb >= MIN_FREE_GB )) || die "need >= ${MIN_FREE_GB}G free, have ${free_gb}G"
  log "cpus=$(nproc) mem_total_gb=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)"
  local u code
  for u in https://github.com https://huggingface.co https://pypi.org/simple/pip/ \
           https://pypi.nvidia.com https://download.pytorch.org/whl/cu128/torch/; do
    code=$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$u" || echo "000")
    log "net $u -> $code"
    [[ $code != 000 ]] || die "cannot reach $u"
  done
  log "vulkan ICDs: $(ls /usr/share/vulkan/icd.d/ 2>/dev/null | tr '\n' ' ')"
  log "system libGLU entries: $(ldconfig -p | grep -c libGLU || true) (0 is fine; conda libglu is installed later)"
}

git_retry() {
  # GitHub over the campus network hung up mid-clone on all four nodes
  # (2026-09-01): force HTTP/1.1, big buffers, and retry up to 5 times.
  local attempt
  for attempt in 1 2 3 4 5; do
    if git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 \
           -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=120 "$@"; then return 0; fi
    log "git ${*:1:3} failed (attempt $attempt/5); retrying in 20s"; sleep 20
  done
  return 1
}
step_checkout() {
  use_env   # git-lfs from the conda env (conda_env step runs first)
  command -v git-lfs >/dev/null || log "WARN: git-lfs not on PATH; LFS-tracked files stay pointers until it is"
  local stage=$SCRIPT_DIR/stage/RoboDojo-official-clone-20260823.tar
  local subs=(XPolicyLab third_party/IsaacLab third_party/curobo)
  if [[ -f $stage ]]; then
    # Preferred: restore the pinned tree (root + the three submodules' git
    # data) from the tarball staged in the shared home; no GitHub involved.
    if [[ ! -f $STATE/checkout.extracted ]]; then
      if [[ -f $stage.sha256 ]]; then
        # compare digests directly so the path recorded in the .sha256 file is irrelevant
        local want have
        want=$(awk '{print $1}' "$stage.sha256"); have=$(sha256sum "$stage" | awk '{print $1}')
        [[ $want == "$have" ]] || die "staged tarball checksum mismatch (want $want have $have)"
        log "tarball checksum OK"
      fi
      log "restoring pinned checkout from $stage ($(du -h "$stage" | cut -f1))"
      if [[ -e $CHECKOUT ]]; then log "removing partial $CHECKOUT"; rm -rf "$CHECKOUT"; fi
      mkdir -p "$ROOT"
      tar -xf "$stage" -C "$ROOT"
      touch "$STATE/checkout.extracted"
    else
      log "staged tarball already extracted"
    fi
  else
    if [[ ! -d $CHECKOUT/.git ]]; then
      log "cloning RoboDojo into $CHECKOUT"
      git_retry clone --quiet https://github.com/RoboDojo-Benchmark/RoboDojo.git "$CHECKOUT"
    fi
    git_retry -C "$CHECKOUT" fetch --quiet origin
  fi
  git -C "$CHECKOUT" checkout --quiet "$PIN_ROOT"
  # NEVER pass --remote here: it floats IsaacLab/curobo to their branch heads.
  git -C "$CHECKOUT" submodule sync --quiet "${subs[@]}"
  git_retry -C "$CHECKOUT" submodule update --init "${subs[@]}"
  local got
  got=$(git -C "$CHECKOUT" rev-parse HEAD);                      [[ $got == "$PIN_ROOT" ]]       || die "root commit $got != $PIN_ROOT"
  got=$(git -C "$CHECKOUT/XPolicyLab" rev-parse HEAD);           [[ $got == "$PIN_XPOLICYLAB" ]] || die "XPolicyLab $got != $PIN_XPOLICYLAB"
  got=$(git -C "$CHECKOUT/third_party/IsaacLab" rev-parse HEAD); [[ $got == "$PIN_ISAACLAB" ]]   || die "IsaacLab $got != $PIN_ISAACLAB"
  got=$(git -C "$CHECKOUT/third_party/curobo" rev-parse HEAD);   [[ $got == "$PIN_CUROBO" ]]     || die "curobo $got != $PIN_CUROBO"
  [[ -f $CHECKOUT/XPolicyLab/client_server/ws/model_client.py ]] || die "XPolicyLab tree incomplete"
  # Fetch LFS content for any submodule that still holds pointer files.
  local sub pointers
  for sub in "${subs[@]}"; do
    pointers=$(git -C "$CHECKOUT/$sub" lfs ls-files 2>/dev/null | grep -c ' - ' || true)
    if (( pointers > 0 )); then
      log "$sub: $pointers LFS pointer file(s) -> git lfs pull"
      git_retry -C "$CHECKOUT/$sub" lfs pull || log "WARN: lfs pull failed for $sub (pointer files remain)"
    fi
  done
  log "checkout pinned OK: root=$PIN_ROOT"
}

step_conda_env() {
  local croot; croot=$(conda_root)
  if [[ ! -x $ENV_PREFIX/bin/python ]]; then
    log "creating $ENV_PREFIX (python 3.11, conda-forge)"
    "$CONDA_BIN" create -y --override-channels -c conda-forge -p "$ENV_PREFIX" python=3.11
  fi
  log "installing build/runtime tools from conda-forge (cmake ninja ffmpeg git-lfs libglu libopengl)"
  "$CONDA_BIN" install -y --override-channels -c conda-forge -p "$ENV_PREFIX" \
      cmake ninja ffmpeg git-lfs libglu libopengl
  # Register the prefix env under its NAME so `conda run -n robodojo-sana-eval`
  # (used by the upstream doctor) resolves. The symlink lives in the shared
  # home but points at the identical node-local path on every node.
  mkdir -p "$croot/envs"
  if [[ -e $croot/envs/$ENV_NAME && ! -L $croot/envs/$ENV_NAME ]]; then
    die "$croot/envs/$ENV_NAME exists and is not a symlink; refusing to touch it"
  fi
  ln -sfn "$ENV_PREFIX" "$croot/envs/$ENV_NAME"
  use_env
  log "python: $("$PY" --version) at $PY"
  log "git-lfs: $(git lfs version)"; log "cmake: $(cmake --version | head -1)"
}

step_base_pip() {
  use_env
  "$PY" -m pip install --upgrade pip
  if ! "$PY" -m pip install -r "$CHECKOUT/scripts/requirements.txt"; then
    log "bulk requirements install failed; falling back to per-package (INSTALL.md section 11)"
    while read -r pkg; do
      [[ -z $pkg || $pkg == \#* ]] && continue
      "$PY" -m pip install "$pkg" || log "WARN: could not install '$pkg'"
    done < "$CHECKOUT/scripts/requirements.txt"
  fi
  "$PY" -m pip install opencv-python-headless==4.11.0.86 pillow matplotlib "scipy==1.15.3" scikit-learn
  "$PY" -m pip install numpy==1.26.0
}

step_torch() {
  use_env
  "$PY" -m pip install "numpy==1.26.0" "typing_extensions==4.12.2" "filelock==3.13.1"
  "$PY" -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
      --index-url https://download.pytorch.org/whl/cu128
  "$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      "devices", torch.cuda.device_count())
assert torch.__version__.startswith("2.7.0+cu128"), torch.__version__
assert torch.cuda.is_available()
PY
}

step_isaacsim() {
  use_env
  log "installing isaacsim[all,extscache]==5.1.0 (large download, 10+ GB)"
  "$PY" -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
}

step_pins_isaacsim() {
  use_env
  apply_pins
  # The 5090 install needed one extra compat move here: isaacsim pulls a
  # huggingface_hub that wants click>=8.4.2, which the pin set forbids.
  if "$PY" -m pip check 2>&1 | grep -qiE 'huggingface[-_]hub .* click'; then
    log "huggingface_hub/click conflict -> installing huggingface_hub<1.17 (5090 step 07)"
    "$PY" -m pip install "huggingface_hub<1.17"
  fi
  pip_check_report
}

step_isaaclab() {
  use_env
  conda_activate_env   # isaaclab.sh picks python from CONDA_PREFIX
  [[ ${CONDA_PREFIX:-} == "$ENV_PREFIX" ]] || die "conda activate did not yield $ENV_PREFIX (got '${CONDA_PREFIX:-}')"
  command -v cmake >/dev/null || die "cmake not on PATH; isaaclab.sh would try sudo apt"
  cd "$CHECKOUT/third_party/IsaacLab"
  export OMNI_KIT_ACCEPT_EULA=YES TERM=xterm-256color
  ./isaaclab.sh --install none
}

step_pins_isaaclab() { use_env; apply_pins; pip_check_report; }

step_curobo() {
  use_env
  cd "$CHECKOUT/third_party/curobo"
  "$PY" -m pip uninstall -y nvidia-curobo curobo 2>/dev/null || true
  # Kernels compile at runtime via cuda.core (CUROBO_USE_PYBIND=0 default):
  # no nvcc, no GPU needed now. Constraints keep the resolver from drifting.
  "$PY" -m pip install -e ".[cu12]" --no-build-isolation --constraint "$SCRIPT_DIR/pins.txt"
}

step_pins_curobo() { use_env; apply_pins; pip_check_report; }

step_runtime_libs() {
  use_env
  # 5090 step 16: conda libglu/libopengl (done in conda_env) + chardet<6 to
  # silence the requests/chardet version-mismatch warning kit prints.
  "$PY" -m pip install "chardet<6"
  "$PY" - <<'PY'
import ctypes
ctypes.CDLL("libGLU.so.1"); print("libGLU load OK")
ctypes.CDLL("libOpenGL.so.0"); print("libOpenGL load OK")
PY
}

step_import_probe() {
  use_env
  timeout 900 "$PY" - <<'PY'
import importlib.metadata as m
import isaacsim, isaaclab  # noqa: F401
print("isaacsim", m.version("isaacsim"))
print("isaaclab", m.version("isaaclab"))
print("nvidia-curobo", m.version("nvidia-curobo"))
from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg  # noqa: F401
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg  # noqa: F401
from curobo.motion_planner import MotionPlanner  # noqa: F401
from curobo.types import ToolPoseCriteria  # noqa: F401
print("imports OK")
PY
}

assets_ok() {
  local d=$CHECKOUT/Assets s
  [[ -d $d ]] || return 1
  for s in Robots Object Material Eval_Layout; do [[ -d $d/$s ]] || return 1; done
}
assets_report() {
  local files bytes
  # rendered curobo.yml files are excluded so the numbers match the dataset itself
  files=$(find -L "$CHECKOUT/Assets" -type f ! -name 'curobo.yml' | wc -l)
  bytes=$(find -L "$CHECKOUT/Assets" -type f ! -name 'curobo.yml' -printf '%s\n' | awk '{s+=$1} END{printf "%d", s}')
  log "assets: files=$files (ref $REF_ASSET_FILES) bytes=$bytes (ref $REF_ASSET_BYTES)"
  if (( files < REF_ASSET_FILES || bytes < REF_ASSET_BYTES )); then
    log "WARN: asset tree smaller than the 5090 reference; download may be incomplete"
    return 1
  fi
  log "assets complete (matches or exceeds the 5090 reference)"
  return 0
}
step_assets() {
  use_env
  cd "$CHECKOUT"
  if assets_ok && assets_report; then log "assets already complete"; return 0; fi
  local attempt ok=0
  for attempt in 1 2 3 4 5; do
    log "init_assets.sh attempt $attempt/5 (sparse git-lfs clone of RoboDojo-Benchmark/RoboDojo, ~39 GB)"
    if bash scripts/init_assets.sh && assets_ok && assets_report; then ok=1; break; fi
    log "attempt $attempt did not complete; git-lfs pull is resumable, retrying in 30s"
    sleep 30
  done
  if (( ! ok )); then
    log "git-lfs route failed; falling back to per-file huggingface_hub downloads"
    # (snapshot_download is avoided: with this env's huggingface_hub/tqdm pair it
    #  dies in tqdm.thread_map with 'min() arg is an empty sequence')
    local hfdir=$CHECKOUT/.cache/robodojo_assets_hf
    "$PY" - "$hfdir" <<'PY'
import sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download
repo, dest = "RoboDojo-Benchmark/RoboDojo", sys.argv[1]
files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith("Assets/")]
print(f"{len(files)} dataset files under Assets/ to fetch", flush=True)
def get(f):
    err = None
    for _ in range(5):
        try:
            hf_hub_download(repo, f, repo_type="dataset", local_dir=dest)
            return None
        except Exception as e:  # noqa: BLE001
            err = e
    return (f, str(err))
with ThreadPoolExecutor(max_workers=8) as ex:
    bad = [r for r in ex.map(get, files) if r]
print(f"per-file download finished, failed={len(bad)}", flush=True)
for f, e in bad[:20]:
    print("FAILED", f, e)
sys.exit(1 if bad else 0)
PY
    [[ -e $CHECKOUT/Assets && ! -L $CHECKOUT/Assets ]] && mv "$CHECKOUT/Assets" "$CHECKOUT/Assets.partial.$(date +%s)"
    ln -sfn "$hfdir/Assets" "$CHECKOUT/Assets"
    assets_ok || die "assets still incomplete after HF fallback"
    assets_report || die "asset tree smaller than reference after HF fallback"
  fi
  log "Assets -> $(readlink -f "$CHECKOUT/Assets")"
}

step_render_configs() {
  use_env
  cd "$CHECKOUT"
  # bakes ABSOLUTE paths from $PWD; must run from the checkout root.
  "$PY" utils/update_embodiment_config_path.py < /dev/null
  local f=$CHECKOUT/Assets/Robots/x5/curobo.yml
  [[ -f $f ]] || die "$f not rendered"
  grep -q "urdf_path: $CHECKOUT/Assets/Robots/x5/X5A.urdf" "$f" || die "urdf_path in $f does not point at $CHECKOUT"
  log "rendered $f OK"
}

step_ws_env() {
  # websocket no-op responder interpreter: websockets>=14, which the sim env
  # (websockets==12.0) cannot host. Package list from remote_client/README.md.
  if [[ ! -x $WS_PREFIX/bin/python ]]; then
    "$CONDA_BIN" create -y --override-channels -c conda-forge -p "$WS_PREFIX" python=3.11 pip
  fi
  "$WS_PREFIX/bin/python" -m pip install "websockets>=14" msgpack msgpack-numpy pydantic numpy \
      opencv-python-headless h5py pyyaml
  "$WS_PREFIX/bin/python" - <<'PY'
import websockets, msgpack, msgpack_numpy, pydantic, numpy, cv2, h5py, yaml  # noqa: F401
major = int(str(websockets.__version__).split(".")[0])
assert major >= 14, websockets.__version__
print("ws env OK, websockets", websockets.__version__)
PY
}

step_doctor() {
  use_env
  set +u; source "$(conda_root)/etc/profile.d/conda.sh"; set -u
  cd "$CHECKOUT"
  local out=$LOGS/doctor_output.txt
  bash scripts/robodojo.sh doctor --sim-env "$ENV_NAME" --summary "$LOGS/doctor_summary.json" 2>&1 | tee "$out"
  grep -q 'fail=0' "$out" || die "doctor reported failures (see $out)"
  log "doctor: $(grep -o 'pass=[0-9]* warn=[0-9]* fail=[0-9]*' "$out" | tail -1)"
}

step_render_probe() {
  # The one thing doctor cannot prove: that THIS node's GPU renders headless.
  # Probe from remote_client/README.md section 5.2. Non-fatal: a failure is
  # logged as WARN for a human to look at (dmesg Xid, kit/logs).
  use_env
  local out=$LOGS/render_probe_output.txt rc=0
  timeout 900 "$PY" - <<'PY' > "$out" 2>&1 || rc=$?
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})
print("APP_READY", flush=True)
for _ in range(60):
    app.update()
print("RENDER_LOOP_OK", flush=True)
app.close()
PY
  if grep -q RENDER_LOOP_OK "$out" && ! grep -qi 'GPU crash is detected' "$out"; then
    log "render probe OK (APP_READY + RENDER_LOOP_OK, rc=$rc)"
  else
    log "WARN: render probe did not pass (rc=$rc). Tail of $out:"
    tail -n 25 "$out" || true
    log "WARN: check dmesg for Xid lines and the kit log under $ENV_PREFIX/lib/python3.11/site-packages/isaacsim/kit/logs/"
    echo "render_probe_failed rc=$rc" > "$STATE/render_probe.WARN"
  fi
}

step_activation() {
  mkdir -p "$CHECKOUT/.cache/kit_home"
  cat > "$ROOT/activate.sh" <<EOF
# source this before running RoboDojo / Sana remote client on this node
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PIP_USER=0 PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=$CACHE/pip TMPDIR=$CACHE/tmp
export XDG_CACHE_HOME=$CACHE/xdg XDG_DATA_HOME=$CACHE/xdg_data HF_HOME=$CACHE/huggingface
export ROBODOJO_CHECKOUT=$CHECKOUT
export ROBODOJO_PY=$PY
export ROBODOJO_KIT_HOME=$CHECKOUT/.cache/kit_home
export SANA_SIM_PY=$PY
export SANA_WS_PY=$WS_PREFIX/bin/python
export PATH=$ENV_PREFIX/bin:$(conda_root)/bin:\$PATH
# HF token lives once in the shared home (chmod 600); HF_HOME above is node-local
export HF_TOKEN_PATH=\$HOME/.hf_token
# The Sana remote client defaults VK_ICD_FILENAMES to /etc/vulkan/icd.d/nvidia_icd.json (the 5090
# workstation's path); on these nodes the NVIDIA Vulkan ICD lives under /usr/share.
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
unset CUDA_VISIBLE_DEVICES
EOF
  log "wrote $ROOT/activate.sh"; cat "$ROOT/activate.sh"
}

step_audit() {
  use_env
  "$PY" -m pip freeze --all > "$LOGS/freeze.txt"
  grep -E '^[A-Za-z0-9][A-Za-z0-9._-]*==' "$LOGS/freeze.txt" | sort -f > "$LOGS/freeze.pinned.txt"
  sort -f "$SCRIPT_DIR/reference_freeze_5090.txt" > "$LOGS/reference.sorted.txt"
  diff "$LOGS/reference.sorted.txt" "$LOGS/freeze.pinned.txt" > "$LOGS/freeze_diff_vs_5090.txt" || true
  local n; n=$(grep -c '^[<>]' "$LOGS/freeze_diff_vs_5090.txt" || true)
  log "package diff vs the working 5090 env: $n lines (see $LOGS/freeze_diff_vs_5090.txt; '<' = 5090 only, '>' = this node only)"
  head -n 40 "$LOGS/freeze_diff_vs_5090.txt" || true
  pip_check_report
  log "disk: $(du -sh "$ROOT" 2>/dev/null | cut -f1) total; env $(du -sh "$ENV_PREFIX" | cut -f1); assets $(du -sh -L "$CHECKOUT/Assets" | cut -f1)"
  {
    echo "host=$HOST"; echo "date=$(date -Is)"; echo "root=$ROOT"; echo "checkout=$CHECKOUT"
    echo "sim_py=$PY"; echo "ws_py=$WS_PREFIX/bin/python"
    echo "render_probe=$([[ -f $STATE/render_probe.WARN ]] && echo WARN || echo OK)"
  } > "$STATE/INSTALL_COMPLETE"
  cp "$STATE/INSTALL_COMPLETE" "$MIRROR_DIR/$HOST.INSTALL_COMPLETE" 2>/dev/null || true
  log "INSTALL COMPLETE on $HOST"
}

# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
if [[ ${1:-} == --run-step ]]; then   # internal: one step in its own process so set -e is honoured
  step="${2:?}"; "step_$step"; exit 0
fi
if [[ ${1:-} == --list ]]; then printf '%s\n' "${STEPS[@]}"; exit 0; fi

mkdir -p "$ROOT" "$LOGS" "$STATE" "$CACHE"/{pip,tmp,xdg,xdg_data,huggingface,conda_pkgs} "$MIRROR_DIR"
exec 9>"$STATE/.lock"; flock -n 9 || die "another install_node.sh is running on $HOST (lock $STATE/.lock)"
exec > >(tee -a "$LOGS/install.log" "$MIRROR_DIR/$HOST.log") 2>&1

ONLY=""; FROM=""
case "${1:-}" in
  --list) printf '%s\n' "${STEPS[@]}"; exit 0 ;;
  --only) ONLY=${2:?}; ;;
  --from) FROM=${2:?}; ;;
  "") ;;
  *) die "unknown argument: $1" ;;
esac
if [[ -n $FROM ]]; then
  clearing=0
  for s in "${STEPS[@]}"; do
    [[ $s == "$FROM" ]] && clearing=1
    (( clearing )) && rm -f "$STATE/$s.done"
  done
  (( clearing )) || die "unknown step: $FROM"
fi
if [[ -n $ONLY ]]; then printf '%s\n' "${STEPS[@]}" | grep -qx "$ONLY" || die "unknown step: $ONLY"; fi

log "===== install_node.sh start (root=$ROOT, args: ${*:-none}) ====="
status all START
for s in "${STEPS[@]}"; do
  if [[ -n $ONLY && $ONLY != "$s" ]]; then continue; fi
  if [[ -z $ONLY && -f $STATE/$s.done ]]; then log "skip $s (done $(date -r "$STATE/$s.done" '+%F %T'))"; continue; fi
  log "=== BEGIN $s ==="; status "$s" START; t0=$(date +%s)
  if bash "$SCRIPT_PATH" --run-step "$s" > >(tee -a "$LOGS/$s.log") 2>&1; then
    touch "$STATE/$s.done"; status "$s" DONE "$(( $(date +%s) - t0 ))s"; log "=== DONE $s in $(( $(date +%s) - t0 ))s ==="
  else
    status "$s" FAIL "$(( $(date +%s) - t0 ))s"; log "=== FAILED $s (log: $LOGS/$s.log) ==="
    log "fix, then re-run: bash $SCRIPT_PATH            (resumes at $s)"
    exit 1
  fi
done
status all DONE
log "===== all steps finished on $HOST ====="
