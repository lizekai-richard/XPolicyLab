#!/usr/bin/env bash
# =============================================================================
# run_unit_xpl.sh RUN_ROOT UNIT_ID TASK SEED EPS SERVER_GPU CLIENT_GPU RPC_PORT WS_PORT [ATTEMPT]
#
# XPolicyLab-lane worker: SAME file contract as run_unit.sh (status.txt, heartbeat, result.json,
# client/<i>_<task>/{upstream_eval_result/_result.json, run_status.json, client.*.log}), so master.py and
# summarize_eval.py drive it unchanged, but the policy is served by the OFFICIAL XPolicyLab websocket
# PolicyServer running an XPolicyLab adapter IN-PROCESS -- policy/SANA_WAM by default (sana_wam_min: Sana-free
# policy mirror, SDPA attention), policy/SANA_MOT (the MoT dual-expert policy, same sana-wam env) or policy/OpenWAM
# -- and the simulator is driven by the official scripts/eval_policy.sh + XPolicyLab/policy/<POLICY>/deploy.py loop
# (get_obs after EVERY action, no lazy render, no SANA RPC).
# Purpose: rule the SANA RPC server/client integration in or out as the cause of the low success rates.
#
#   RPC_PORT   the slot port used for the websocket policy server (WS_PORT is unused but reclaimed)
# Env (from master.py or the operator):
#   SANA_EVAL_XPL_POLICY         the adapter under policy/: SANA_WAM (default) | SANA_MOT | SANA_WAM_CAUSAL | SANA_WAM_SM |
#                                OpenWAM. The four SANA adapters share the sana-wam env and the Sana snapshot layout + text
#                                encoder / VAE keys below
#   SANA_EVAL_CHECKPOINT_ROOT    HF-style snapshot (config.yaml + model/pytorch_model_fsdp.bin [+ normalization/])
#   SANA_EVAL_XPL_ROOT           the XPolicyLab checkout carrying policy/<POLICY> (default CE_REMOTE_XPL)
#   SANA_EVAL_XPL_WS             node-local RoboDojo workspace built from symlinks to the pinned checkout with
#                                XPolicyLab -> SANA_EVAL_XPL_ROOT (default CE_NODE_XPL_WS)
#   SANA_EVAL_XPL_PYDEPS         directory with the websocket/msgpack/h5py wheels the sana-wam env lacks (default
#                                CE_REMOTE_PYDEPS for the SANA adapters, none otherwise), prepended to PYTHONPATH of
#                                the policy server only
#   SANA_WAM_TEXT_ENCODER_PATH / SANA_WAM_VAE_PATH   default CE_TEXT_ENCODER_PATH / CE_VAE_PATH, else the Sana layout
#                                $SANA_REPO/output/pretrained_models/{text_encoder/gemma-2-2b-it,LTX-2.3-Diffusers}
#   SANA_EVAL_XPL_EXTRA_OVERRIDES  extra space-separated key=value deploy.yml overrides for the policy server
#                                  (the token @ATTEMPT_DIR@ becomes this attempt's NFS dir, e.g.
#                                  prediction_latent_dir=@ATTEMPT_DIR@/pred_latents)
#                                (e.g. "diffusion_seed_base=20260803 sampling_steps=50", or for SANA_MOT
#                                "video_layout=openwam_canvas" to assert the checkpoint's video layout); recorded in
#                                resolved_inputs
#   SANA_EVAL_LAYOUT_IDS / SANA_EVAL_LAYOUT_TOTAL   layout shard (master --shards): comma-separated layout ids this
#                                unit evaluates (EPS of them) out of LAYOUT_TOTAL. The official client's resume
#                                manifest is pre-seeded with every other layout marked 'abandoned', so it runs
#                                exactly these ids; _result.json then holds only these layouts.
#   ROBODOJO_ROOT (default CE_NODE_ROOT), INSTALL_DIR (default this script's parent dir), SERVER_STARTUP_TIMEOUT (3000)
# Site values: <INSTALL_DIR>/site.env (deploy.sh copies the active site configuration there; CE_SITE_NODE overrides).
# Node runtime contract: $ROBODOJO_ROOT/activate.sh exports SANA_SIM_PY (the Isaac/RoboDojo python) and ROBODOJO_CHECKOUT
# (the pinned RoboDojo clone); $ROBODOJO_ROOT/activate_sana_wam.sh exports SANA_PY (the policy-server python).
# Budget: 1800 + EPS * max(EPISODE_SECONDS, step_lim * SEC_PER_STEP + RESET_SECONDS); SEC_PER_STEP defaults to 4
# here (SDPA policy + a render per step is slower than the fused-kernel-free lazy-render lane's 0.85 s/step).
# =============================================================================
set -Eeuo pipefail
(( $# == 9 || $# == 10 )) || { echo "usage: run_unit_xpl.sh RUN_ROOT UNIT_ID TASK SEED EPS SERVER_GPU CLIENT_GPU RPC_PORT WS_PORT [ATTEMPT]" >&2; exit 64; }
RUN_ROOT=$1; UNIT_ID=$2; TASK=$3; SEED=$4; EPS=$5; SERVER_GPU=$6; CLIENT_GPU=$7; RPC_PORT=$8; WS_PORT=$9; ATTEMPT=${10:-1}
SITE_ENV=${CE_SITE_NODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/site.env}
# shellcheck disable=SC1090
if [[ -f $SITE_ENV ]]; then source "$SITE_ENV"; fi
ROOT=${ROBODOJO_ROOT:-${CE_NODE_ROOT:?no site.env next to the worker (run deploy.sh) and no ROBODOJO_ROOT}}
INSTALL_DIR=${INSTALL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SERVER_STARTUP_TIMEOUT=${SERVER_STARTUP_TIMEOUT:-3000}
EPISODE_SECONDS=${EPISODE_SECONDS:-600}
SEC_PER_STEP=${SEC_PER_STEP:-4}
RESET_SECONDS=${RESET_SECONDS:-120}
STALL_SECONDS=${STALL_SECONDS:-2700}
STALL_CRASH_SECONDS=${STALL_CRASH_SECONDS:-600}
STALL_CONFIRM=${STALL_CONFIRM:-300}
STALL_POLL=${STALL_POLL:-60}
DRY=${EVAL_DRY_RUN:-0}
XPL_ROOT=${SANA_EVAL_XPL_ROOT:-${CE_REMOTE_XPL:?no CE_REMOTE_XPL in site.env and no SANA_EVAL_XPL_ROOT}}
XPL_WS=${SANA_EVAL_XPL_WS:-${CE_NODE_XPL_WS:-$ROOT/xpl_ws}}
XPL_EXTRA_OVERRIDES=${SANA_EVAL_XPL_EXTRA_OVERRIDES:-}
XPL_ACTION_TYPE=${SANA_EVAL_XPL_ACTION_TYPE:-joint}     # joint | ee: the adapter's action_type (ee needs a robot_base_eef checkpoint)
case $XPL_ACTION_TYPE in joint|ee) ;; *) echo "SANA_EVAL_XPL_ACTION_TYPE must be joint or ee, got '$XPL_ACTION_TYPE'" >&2; exit 64;; esac
LAYOUT_IDS=${SANA_EVAL_LAYOUT_IDS:-}
LAYOUT_TOTAL=${SANA_EVAL_LAYOUT_TOTAL:-}
read -r -a XPL_EXTRA_OVERRIDE_ARGS <<<"$XPL_EXTRA_OVERRIDES"
POLICY_NAME=${SANA_EVAL_XPL_POLICY:-SANA_WAM}          # SANA_WAM (default) | SANA_MOT | SANA_WAM_CAUSAL | SANA_WAM_SM | OpenWAM
# pydeps only exist to patch the sana-wam env's missing websockets/msgpack/h5py, and it is PREPENDED to the
# server's PYTHONPATH -- for a policy whose own env already ships them it would shadow the correct versions.
# Hence: the policies served from the sana-wam env (SANA_WAM, SANA_MOT, SANA_WAM_CAUSAL, SANA_WAM_SM) get the historical
# default, any other none.
case $POLICY_NAME in
  SANA_WAM|SANA_MOT|SANA_WAM_CAUSAL|SANA_WAM_SM) XPL_PYDEPS=${SANA_EVAL_XPL_PYDEPS:-${CE_REMOTE_PYDEPS:-}} ;;
  *)                 XPL_PYDEPS=${SANA_EVAL_XPL_PYDEPS:-} ;;
esac
ENV_CFG_TYPE=arx_x5
[[ $RUN_ROOT == /*robodojo* ]] || { echo "RUN_ROOT must be an absolute path containing 'robodojo'" >&2; exit 64; }
[[ $ATTEMPT =~ ^[0-9]+$ ]] || { echo "ATTEMPT must be an integer" >&2; exit 64; }

U=$RUN_ROOT/units/$UNIT_ID
A=$U/a$ATTEMPT
mkdir -p "$A"
exec > >(tee -a "$A/unit.log") 2>&1
exec 9>"$U/.lock"; flock -n 9 || { echo "unit $UNIT_ID already has a runner"; exit 1; }
echo $$ >"$U/unit.pid"
HOST=$(hostname -s); STARTED=$(date -Is)
log()    { printf '[%s] [%s %s a%s] %s\n' "$(date '+%F %T')" "$HOST" "$UNIT_ID" "$ATTEMPT" "$*"; }
status() { echo "$(date -Is) $UNIT_ID a$ATTEMPT $*" >>"$A/status.txt"; }

# ---- heartbeat (immune to set -e, no lock fd) --------------------------------------
( set +e; exec 9>&-; while :; do { date -u +%FT%T >"$A/heartbeat.tmp" && mv -f "$A/heartbeat.tmp" "$A/heartbeat"; } || true; sleep 30; done ) &
HB_PID=$!
stop_heartbeat() { pkill -P "$HB_PID" 2>/dev/null || true; kill "$HB_PID" 2>/dev/null || true; }

# ---- result (python3 from PATH until the env is sourced) ----------------------------
PYBIN=$(command -v python3 || echo python3)
RESULT_WRITTEN=0
write_result() { # write_result STATUS [episodes successes score_sum note]
  (( RESULT_WRITTEN )) && return 0; RESULT_WRITTEN=1
  "$PYBIN" - "$A/result.json" "$1" "${2:-0}" "${3:-0}" "${4:-0}" "${5:-}" "$UNIT_ID" "$TASK" "$SEED" "$EPS" "$HOST" "$SERVER_GPU" "$CLIENT_GPU" "$RPC_PORT" "$STARTED" "$ATTEMPT" "${CKPT:-${SANA_EVAL_CHECKPOINT_ROOT:-}}" <<'PY' || echo "{\"status\": \"$1\", \"unit_id\": \"$UNIT_ID\", \"attempt\": $ATTEMPT, \"note\": \"result writer failed\"}" >"$A/result.json"
import json, sys, datetime, os
(out, st, eps_rec, succ, score, note, uid, task, seed, eps, host, sg, cg, rpc, started, attempt, ckpt) = sys.argv[1:]
tmp = out + ".tmp"
json.dump({"unit_id": uid, "attempt": int(attempt), "task": task, "eval_seed": int(seed), "episodes_per_task": int(eps),
           "status": st, "episodes_recorded": int(eps_rec), "successes": int(succ), "score_sum": float(score), "note": note,
           "node": host, "server_gpu": int(sg), "client_gpu": int(cg), "rpc_port": int(rpc), "checkpoint_root": ckpt,
           "lane": "xpolicylab", "started": started, "finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds")},
          open(tmp, "w"), indent=2, sort_keys=True)
os.replace(tmp, out)
PY
  status "RESULT $1"; log "result: $1 ${5:-}"
}
trap 'rc=$?; write_result prologue_fail 0 0 0 "prologue failed at line $LINENO (rc $rc)"; stop_heartbeat; exit $rc' ERR

# ---- runtime env -------------------------------------------------------------------
if [[ $DRY != 1 ]]; then
  # a failed `source` of a missing file exits bash WITHOUT running the ERR trap (the heartbeat loop would outlive the worker
  # and keep the unit looking alive), so check the node runtime first
  for f in "$ROOT/activate.sh" "$ROOT/activate_sana_wam.sh"; do
    [[ -f $f ]] || { trap - ERR; write_result prologue_fail 0 0 0 "node runtime missing: $f (RUNBOOK section 1.3)"; stop_heartbeat; exit 78; }
  done
  # shellcheck disable=SC1091
  source "$ROOT/activate.sh"
  # shellcheck disable=SC1091
  source "$ROOT/activate_sana_wam.sh"
  : "${SANA_PY:?}" "${SANA_SIM_PY:?}" "${ROBODOJO_CHECKOUT:?}"
  PYBIN=$SANA_PY
  # The SIMULATOR always runs in $SANA_SIM_PY; only the POLICY SERVER's interpreter varies, because a
  # second policy may need a conflicting dependency set (OpenWAM wants timm>=1.0.20 / transformers>=5.5,
  # sana-wam is pinned to timm 0.6.13). Default keeps the SANA lane exactly as before.
  SERVER_PY=${SANA_EVAL_XPL_SERVER_PY:-$SANA_PY}
  [[ -x $SERVER_PY ]] || { trap - ERR; write_result prologue_fail 0 0 0 "policy-server python not executable: $SERVER_PY"; stop_heartbeat; exit 78; }
  # ---- policy checkpoint (same rules as run_unit.sh) ----------------------------------------------
  if [[ -n ${SANA_EVAL_CHECKPOINT_ROOT:-} ]]; then export SANA_POLICY_CHECKPOINT_ROOT=$SANA_EVAL_CHECKPOINT_ROOT; fi
  CKPT=${SANA_POLICY_CHECKPOINT_ROOT:?no checkpoint: the master passes SANA_EVAL_CHECKPOINT_ROOT}
  missing=()
  if [[ $POLICY_NAME == OpenWAM ]]; then
    [[ -f $CKPT/config.yaml ]] || missing+=("config.yaml")
    compgen -G "$CKPT/checkpoint_step_*.safetensors" >/dev/null || missing+=("checkpoint_step_*.safetensors")
    [[ -f $CKPT/normalization_stats.npy ]] || missing+=("normalization_stats.npy")
  else
    for f in config.yaml model/pytorch_model_fsdp.bin; do [[ -f $CKPT/$f ]] || missing+=("$f"); done
  fi
  inc=""
  if [[ -d $CKPT/.cache ]]; then inc=$(find "$CKPT/.cache" -name '*.incomplete' 2>/dev/null | head -1 || true); fi
  [[ -z $inc ]] || missing+=("download still in progress: $inc")
  if (( ${#missing[@]} )); then
    trap - ERR; write_result ckpt_missing 0 0 0 "checkpoint $CKPT unusable: ${missing[*]}"; stop_heartbeat; exit 78
  fi
  CKPT_LABEL=$(basename "$CKPT")
  # ---- XPolicyLab checkout + portable deps ------------------------------------------------------
  xmissing=()
  for f in setup_policy_server.py policy/$POLICY_NAME/model.py policy/$POLICY_NAME/deploy.py policy/$POLICY_NAME/deploy.yml client_server/ws/model_server.py; do
    [[ -f $XPL_ROOT/$f ]] || xmissing+=("$f")
  done
  [[ -z $XPL_PYDEPS || -d $XPL_PYDEPS ]] || xmissing+=("pydeps dir $XPL_PYDEPS")
  if (( ${#xmissing[@]} )); then
    trap - ERR; write_result prologue_fail 0 0 0 "XPolicyLab lane inputs missing under $XPL_ROOT: ${xmissing[*]}"; stop_heartbeat; exit 78
  fi
  # Per-policy deploy.yml keys. The SANA adapters point at the shared Sana encoders; OpenWAM's bundle is
  # self-contained (its Wan2.2 DiT/VAE/text-encoder weights are all inside the one safetensors) and
  # takes `ckpt_dir` + the vendored source root instead.
  POLICY_OVERRIDE_ARGS=()
  if [[ $POLICY_NAME == OpenWAM ]]; then
    POLICY_OVERRIDE_ARGS+=("ckpt_dir=$CKPT" "openwam_root=$XPL_ROOT/policy/OpenWAM/OpenWAM")
    [[ -d $XPL_ROOT/policy/OpenWAM/OpenWAM/openwam ]] || { trap - ERR; write_result prologue_fail 0 0 0 "vendored openwam package missing under $XPL_ROOT/policy/OpenWAM/OpenWAM"; stop_heartbeat; exit 78; }
    if [[ -n ${SANA_EVAL_XPL_LATENT_DIR:-} ]]; then
      LATENT_DIR=$A/latents; mkdir -p "$LATENT_DIR"
      POLICY_OVERRIDE_ARGS+=("prediction_latent_dir=$LATENT_DIR")
    fi
  else
    TE_PATH=${SANA_WAM_TEXT_ENCODER_PATH:-${CE_TEXT_ENCODER_PATH:-${SANA_REPO:-/nonexistent}/output/pretrained_models/text_encoder/gemma-2-2b-it}}
    VAE_PATH=${SANA_WAM_VAE_PATH:-${CE_VAE_PATH:-${SANA_REPO:-/nonexistent}/output/pretrained_models/LTX-2.3-Diffusers}}
    [[ -f $TE_PATH/config.json ]] || { trap - ERR; write_result prologue_fail 0 0 0 "text encoder not found at $TE_PATH"; stop_heartbeat; exit 78; }
    [[ -f $VAE_PATH/vae/config.json ]] || { trap - ERR; write_result prologue_fail 0 0 0 "VAE not found at $VAE_PATH/vae"; stop_heartbeat; exit 78; }
    POLICY_OVERRIDE_ARGS+=("checkpoint_dir=$CKPT" "text_encoder_path=$TE_PATH" "vae_path=$VAE_PATH")
    # Legacy view resize: the adapters default to stretch, the only SFT resize since rwm/zekai-merge 1061b16f0 / rwm/mot
    # 034e55dca (2026-09-27); a checkpoint listed in pre_stretch_labels.txt (next to this script) was trained with the
    # old scale-to-cover + centre crop and is served view_resize=crop. A driver's own view_resize override (later) wins.
    VR_REGISTRY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pre_stretch_labels.txt
    [[ -f $VR_REGISTRY ]] || { trap - ERR; write_result prologue_fail 0 0 0 "legacy view-resize registry missing: $VR_REGISTRY"; stop_heartbeat; exit 78; }
    if awk -v l="$(basename "$CKPT")" '!/^[[:space:]]*(#|$)/ && $1 == l {f = 1} END {exit !f}' "$VR_REGISTRY"; then
      POLICY_OVERRIDE_ARGS+=("view_resize=crop")
    fi
  fi
fi
CKPT_LABEL=${CKPT_LABEL:-dry}
trap - ERR

# ---- process control ---------------------------------------------------------------
SERVER_PID=""; CLIENT_PID=""
kill_tree() { local pid=$1 c; for c in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$c"; done; kill -TERM "$pid" 2>/dev/null || true; }
reap_group() { # reap_group PGID LABEL: TERM then KILL every process still in the client's process group (Kit helpers
  # outlive src/eval_client/main.py and keep 6-9 GB of GPU memory each -- 15-18 of them per node were found on 2026-09-09)
  local pgid=$1 label=$2 left mygid
  mygid=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ' || true)
  if [[ -z $pgid || $pgid == "$mygid" ]]; then log "$label shares the worker's process group; no group reap"; return 0; fi
  # NOTE: every command here must be set -e safe -- an unguarded `pgrep` with no match (exit 1) killed the
  # whole worker right after the client exited (2026-09-09 05:31, l2/l4 smokes -> STALE, no result written)
  left=$(pgrep -g "$pgid" 2>/dev/null | tr '\n' ' ' || true)
  [[ -n ${left// /} ]] || return 0
  log "$label process group $pgid still has pid(s) $left after exit; TERM"
  kill -TERM -- "-$pgid" 2>/dev/null || true; sleep 15
  left=$(pgrep -g "$pgid" 2>/dev/null | tr '\n' ' ' || true)
  if [[ -n ${left// /} ]]; then log "$label group $pgid still alive ($left); SIGKILL"; kill -KILL -- "-$pgid" 2>/dev/null || true; sleep 2; fi
  status REAPED "$label group $pgid"
}
wait_pid_bounded() { # wait_pid_bounded PID SECONDS -> KILL if still alive afterwards
  local pid=$1 limit=$2 t=0
  while kill -0 "$pid" 2>/dev/null && (( t < limit )); do sleep 5; t=$((t + 5)); done
  if kill -0 "$pid" 2>/dev/null; then log "process $pid still alive after ${limit}s; SIGKILL"; kill_tree "$pid"; sleep 2; kill -KILL "$pid" 2>/dev/null || true; fi
}
stop_server_now() {
  [[ -n $SERVER_PID ]] || return 0
  kill -0 "$SERVER_PID" 2>/dev/null || { SERVER_PID=""; return 0; }
  log "stopping policy server pid $SERVER_PID"
  kill_tree "$SERVER_PID"; wait_pid_bounded "$SERVER_PID" 60
  SERVER_PID=""
}
STALL_PID=""
stop_stall_watch() { [[ -n $STALL_PID ]] || return 0; pkill -P "$STALL_PID" 2>/dev/null || true; kill "$STALL_PID" 2>/dev/null || true; STALL_PID=""; }
client_progress() { # prints "<simulation steps taken so far>": the only signal that the evaluation is advancing
  # occurrences, not lines: the official client prints the per-step counter with carriage returns, so a
  # whole episode can sit on ONE line (seen 2026-09-08 in the XPolicyLab lane)
  cat "$A"/client/*/client.stdout.log 2>/dev/null | grep -a -o 'env0 step' | wc -l
}
client_crashed() { # crash signatures Isaac/Kit prints before it hangs in its (GUI-less) crash handler
  grep -a -q -E 'GPU pagefault|aftermath dump|Fatal Python error|Segmentation fault|cudaErrorIllegalAddress|CUDA error' \
    "$A"/client/*/client.stdout.log "$A"/client/*/client.stderr.log 2>/dev/null
}
stall_watch() { # stall_watch CLIENT_PID: TERM the client when the simulation stops advancing (progress-based, see run_unit.sh)
  set +e; exec 9>&-
  local last_steps=-1 last_change now steps limit
  last_change=$(date +%s)
  while kill -0 "$1" 2>/dev/null; do
    sleep "$STALL_POLL"
    now=$(date +%s); steps=$(client_progress)
    if (( steps != last_steps )); then last_steps=$steps; last_change=$now; continue; fi
    limit=$STALL_SECONDS
    client_crashed && limit=$STALL_CRASH_SECONDS
    if (( now - last_change > limit )); then
      status STALL_SUSPECTED "no simulation progress for $((now - last_change))s at step count $steps; confirming for ${STALL_CONFIRM}s"
      log "no simulation progress for $((now - last_change))s (step lines $steps$(client_crashed && echo ', crash signature in the client log')); confirming for ${STALL_CONFIRM}s before TERM"
      sleep "$STALL_CONFIRM"
      kill -0 "$1" 2>/dev/null || break
      if (( $(client_progress) != steps )); then
        status STALL_CLEARED "simulation progressed again"; log "simulation progressed again; stall not confirmed"
        last_steps=-1; continue
      fi
      status STALL "no simulation progress for $(( $(date +%s) - last_change ))s"; log "client stalled (no progress for $(( $(date +%s) - last_change ))s); TERM"
      kill_tree "$1"; sleep 90; kill -0 "$1" 2>/dev/null && { log "client still alive after TERM; SIGKILL"; pkill -KILL -P "$1" 2>/dev/null; kill -KILL "$1" 2>/dev/null; }
      break
    fi
  done
}
cleanup() { stop_stall_watch; stop_heartbeat; }
on_stop() {
  log "stop signal received; aborting unit"
  if [[ -n $CLIENT_PID ]]; then
    kill_tree "$CLIENT_PID"
    # the client's group outlives kill_tree (reparented main.py, Kit helpers, ffmpeg writers: 6-7 GB left on the GPU after
    # every `tail assemble`, 2026-09-27 l1 GPU1 and 2026-09-28 l1 GPU0+1) -> reap it like a normal client exit does
    reap_group "$CLIENT_PID" "client"
  fi
  stop_server_now
  write_result aborted 0 0 0 "TERM/INT received"
  cleanup; exit 130
}
trap on_stop TERM INT
trap cleanup EXIT
port_free() { ! timeout 2 bash -c ">/dev/tcp/127.0.0.1/$1" 2>/dev/null; }
reclaim_port() { # reclaim_port PORT: the port belongs to this slot, so any listener is an orphan of an earlier unit
  local port=$1 pids t=0
  pids=$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u | tr '\n' ' ')
  [[ -n $pids ]] || return 0
  status PORT_RECLAIM "port=$port pids=$pids"; log "port $port held by pid(s) $pids (orphan of an earlier unit on this slot); TERM"
  kill -TERM $pids 2>/dev/null || true
  while (( t < 45 )) && ! port_free "$port"; do sleep 3; t=$((t + 3)); done
  port_free "$port" || { log "port $port still held after ${t}s; SIGKILL"; kill -KILL $pids 2>/dev/null || true; sleep 5; }
}

judge() { # prints "episodes successes score_sum"; returns 0 when all EPS episodes are recorded
  local d; d=$(ls -d "$A"/client/*_"$TASK" 2>/dev/null | head -n1) || true
  [[ -n $d && -f $d/upstream_eval_result/_result.json && -f $d/run_status.json ]] || { echo "0 0 0"; return 1; }
  "$PYBIN" - "$d" "$EPS" <<'PY'
import json, sys
d, eps = sys.argv[1], int(sys.argv[2])
rs = json.load(open(f"{d}/run_status.json")); det = json.load(open(f"{d}/upstream_eval_result/_result.json")).get("details") or {}
rows = [v for k, v in det.items() if str(k).isdigit()][:eps]
succ = sum(1 for v in rows if v.get("success")); score = sum(float(v.get("score", 0.0) or 0.0) for v in rows)
print(len(det), succ, round(score, 4))
sys.exit(0 if (len(det) >= eps and not rs.get("client_watchdog_fired")) else 1)
PY
}

build_workspace() { # node-local RoboDojo workspace = the pinned checkout by symlinks, XPolicyLab -> the user's checkout
  mkdir -p "$XPL_WS"
  local e name
  for e in "$ROBODOJO_CHECKOUT"/* "$ROBODOJO_CHECKOUT"/.cache; do
    [[ -e $e ]] || continue
    name=$(basename "$e")
    [[ $name == XPolicyLab || $name == eval_result ]] && continue
    ln -sfn "$e" "$XPL_WS/$name"
  done
  ln -sfn "$XPL_ROOT" "$XPL_WS/XPolicyLab"
  mkdir -p "$XPL_WS/eval_result"
  [[ -f $XPL_WS/env_cfg/$ENV_CFG_TYPE.yml && -f $XPL_WS/scripts/eval_policy.sh && -f $XPL_WS/src/eval_client/main.py && -f $XPL_WS/XPolicyLab/policy/$POLICY_NAME/deploy.yml ]]
}

# ---- run -----------------------------------------------------------------------------
log "===== unit $UNIT_ID attempt $ATTEMPT (XPolicyLab lane): task=$TASK seed=$SEED eps=$EPS gpus=$SERVER_GPU/$CLIENT_GPU ws_policy_port=$RPC_PORT checkpoint=${CKPT:-dry} dry=$DRY ====="
status START "task=$TASK seed=$SEED eps=$EPS node=$HOST gpus=$SERVER_GPU/$CLIENT_GPU checkpoint=$CKPT_LABEL lane=xpolicylab policy=$POLICY_NAME action_type=$XPL_ACTION_TYPE xpl_root=$XPL_ROOT"
if [[ $DRY == 1 ]]; then
  sleep "${DRY_SLEEP:-2}"; mkdir -p "$A/client/0_$TASK/upstream_eval_result"
  "$PYBIN" - "$A/client/0_$TASK" "$TASK" "$EPS" "$SEED" "$ATTEMPT" <<'PY'
import json, sys, random, os
d, task, eps, attempt = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[5]); random.seed()
ids = [int(x) for x in os.environ.get("SANA_EVAL_LAYOUT_IDS", "").split(",") if x.strip()] or list(range(eps))
if attempt == 1 and random.random() < 0.2: ids = ids[:-1]          # a short first attempt now and then
json.dump({"details": {str(k): {"layout_id": k, "success": random.random() < 0.3, "score": round(random.random(), 3)} for k in ids}}, open(f"{d}/upstream_eval_result/_result.json", "w"))
json.dump({"task": task, "eval_num": str(eps), "eval_seed": int(sys.argv[4]), "client_rc": 0, "client_watchdog_fired": False, "lane": "xpolicylab",
           "run_id": "dry-" + __import__("datetime").datetime.utcnow().strftime("%Y%m%d-%H%M%S%f")}, open(f"{d}/run_status.json", "w"))
PY
else
  for p in "$RPC_PORT" "$WS_PORT"; do port_free "$p" || reclaim_port "$p"; done
  if ! port_free "$RPC_PORT"; then write_result port_busy 0 0 0 "policy port $RPC_PORT in use on $HOST"; exit 75; fi
  if ! build_workspace; then write_result prologue_fail 0 0 0 "could not build the XPolicyLab workspace at $XPL_WS"; exit 78; fi
  mkdir -p "$A/server"
  {
    echo "lane=xpolicylab"; echo "policy_name=$POLICY_NAME"; echo "xpl_root=$XPL_ROOT"; echo "xpl_workspace=$XPL_WS"; echo "pydeps=$XPL_PYDEPS"
    echo "checkpoint_root=$CKPT"; echo "text_encoder_path=${TE_PATH:-}"; echo "vae_path=${VAE_PATH:-}"; echo "env_cfg_type=$ENV_CFG_TYPE"
    echo "policy_overrides=${POLICY_OVERRIDE_ARGS[*]:-}"
    echo "ws_policy_port=$RPC_PORT"; echo "server_python=$SERVER_PY"; echo "sim_python=$SANA_SIM_PY"; echo "eval_num=$EPS"; echo "seed=$SEED"
    echo "extra_overrides=$XPL_EXTRA_OVERRIDES"; echo "action_type=$XPL_ACTION_TYPE"; echo "layout_ids=$LAYOUT_IDS"; echo "layout_total=$LAYOUT_TOTAL"
    echo "deploy_yml_sha256=$(sha256sum "$XPL_ROOT/policy/$POLICY_NAME/deploy.yml" | cut -d' ' -f1)"
    echo "model_py_sha256=$(sha256sum "$XPL_ROOT/policy/$POLICY_NAME/model.py" | cut -d' ' -f1)"
  } >"$A/server/resolved_inputs.txt"
  # ---- policy server: the official ws PolicyServer with the $POLICY_NAME adapter in-process ----------------
  # sys.path[0] = XPL_ROOT (script dir) resolves `client_server`; PYTHONPATH = workspace resolves
  # `XPolicyLab.policy.<POLICY>.model` (+ pydeps for websockets/msgpack/h5py the sana-wam env lacks).
  SERVER_PYTHONPATH=$XPL_WS${XPL_PYDEPS:+:$XPL_PYDEPS}
  [[ $POLICY_NAME == OpenWAM ]] && SERVER_PYTHONPATH=$SERVER_PYTHONPATH:$XPL_ROOT/policy/OpenWAM/OpenWAM
  ( cd "$XPL_WS" && exec env -u PYTHONPATH PYTHONPATH="$SERVER_PYTHONPATH" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
      PYTHONWARNINGS=ignore::UserWarning CUDA_VISIBLE_DEVICES="$SERVER_GPU" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
      "$SERVER_PY" "$XPL_ROOT/setup_policy_server.py" --config_path "$XPL_ROOT/policy/$POLICY_NAME/deploy.yml" \
        --overrides "port=$RPC_PORT" host=127.0.0.1 bench_name=RoboDojo "task_name=$TASK" "ckpt_name=$CKPT_LABEL" \
          "env_cfg_type=$ENV_CFG_TYPE" "seed=$SEED" "policy_name=$POLICY_NAME" "action_type=$XPL_ACTION_TYPE" \
          "${POLICY_OVERRIDE_ARGS[@]}" "${XPL_EXTRA_OVERRIDE_ARGS[@]//@ATTEMPT_DIR@/$A}" ) \
    >"$A/server/server.stdout.log" 2>"$A/server/server.stderr.log" 9>&- &
  SERVER_PID=$!
  waited=0
  until ! port_free "$RPC_PORT"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      write_result server_fail 0 0 0 "policy server exited before listening (see server/server.stderr.log)"; exit 70
    fi
    if (( waited >= SERVER_STARTUP_TIMEOUT )); then
      stop_server_now; write_result server_timeout 0 0 0 "policy server not listening after ${SERVER_STARTUP_TIMEOUT}s"; exit 70
    fi
    sleep 5; waited=$((waited + 5))
  done
  status SERVER_UP "wait=${waited}s"; log "policy server listening on 127.0.0.1:$RPC_PORT after ${waited}s"
  KIT_HOME=$ROOT/kit_home/slot_${SERVER_GPU}_${CLIENT_GPU}; mkdir -p "$KIT_HOME"
  if [[ -z $(ls -A "$KIT_HOME") && -d ${ROBODOJO_KIT_HOME:-/nonexistent} ]]; then cp -a "${ROBODOJO_KIT_HOME}/." "$KIT_HOME/" || true; fi
  # ---- client budget: EPS episodes that may all run to the task's step limit ----
  STEP_LIM=$(grep -o -m1 'self.step_lim = [0-9]*' "$ROBODOJO_CHECKOUT/task/RoboDojo/tasks/$TASK.py" 2>/dev/null | grep -o '[0-9]*$' || true)
  STEP_LIM=${STEP_LIM:-2000}
  PER_EPISODE=$(awk -v l="$STEP_LIM" -v s="$SEC_PER_STEP" -v r="$RESET_SECONDS" -v e="$EPISODE_SECONDS" 'BEGIN { v = l * s + r; if (v < e) v = e; printf "%d", v }')
  WATCHDOG=$(( 1800 + EPS * PER_EPISODE ))
  CD=$A/client/0_$TASK; mkdir -p "$CD"
  RUN_ID=xpl-$(date -u +%Y%m%d-%H%M%S)-$$
  ADDITIONAL_INFO="ckpt_name=$CKPT_LABEL,action_type=$XPL_ACTION_TYPE"
  if [[ -n $LAYOUT_IDS ]]; then
    # Layout shard: pre-seed the official client's resume manifest (read by src/eval_client/main.py from
    # eval_result/RoboDojo/<task>/<policy>/<env>/<seed>_<info>/_resume_<run_id>.json) with the layouts outside this
    # unit's range marked abandoned; SeedManager.init_eval then evaluates only the range, in layout order.
    MANIFEST_DIR=$XPL_WS/eval_result/RoboDojo/$TASK/$POLICY_NAME/$ENV_CFG_TYPE/${SEED}_${ADDITIONAL_INFO}
    mkdir -p "$MANIFEST_DIR"
    # The layout directory may hold MORE files than the task's eval_num (fold_clothes: 55 for 25 evaluated):
    # SeedManager lists them all, so every file outside the shard is marked abandoned, not just 0..eval_num-1.
    N_LAYOUT_FILES=$(ls "$XPL_WS/Assets/Eval_Layout/RoboDojo/$ENV_CFG_TYPE/$SEED/${TASK}_"[0-9]*.json 2>/dev/null | wc -l)
    shard_note=$("$PYBIN" - "$MANIFEST_DIR/_resume_$RUN_ID.json" "$LAYOUT_IDS" "${LAYOUT_TOTAL:-0}" "$EPS" "$N_LAYOUT_FILES" <<'PY'
import json, sys
path, ids_s, total, eps, n_files = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
ids = sorted({int(x) for x in ids_s.split(",") if x.strip()})
if not ids or len(ids) != eps:
    sys.exit(f"layout shard {ids} does not match eps={eps}")
total = max(total, max(ids) + 1, n_files)
abandoned = [i for i in range(total) if i not in set(ids)]
json.dump({"success_nums": 0, "fail_nums": 0, "total_score": 0.0, "details": {}, "completed_layout_ids": [],
           "abandoned_layout_ids": abandoned, "restart_count": 0,
           "layout_shard": {"layout_ids": ids, "layout_total": total,
                            "note": "pre-seeded by run_unit_xpl.sh: layouts outside this unit's shard are marked abandoned"}},
          open(path, "w"), indent=1)
print(f"{len(ids)} layouts kept, {len(abandoned)} of {total} marked abandoned")
PY
) || { write_result prologue_fail 0 0 0 "layout shard manifest: $shard_note"; exit 78; }
    status LAYOUT_SHARD "ids=$LAYOUT_IDS total=$LAYOUT_TOTAL $shard_note"
    log "layout shard: ids=$LAYOUT_IDS of $LAYOUT_TOTAL -> $MANIFEST_DIR/_resume_$RUN_ID.json ($shard_note)"
  fi
  SIM_ENV_BIN=$(dirname -- "$SANA_SIM_PY")
  set +e
  # The official evaluator: eval_policy.sh -> src/eval_client/main.py with XPolicyLab/policy/<POLICY>/deploy.py.
  # Same Isaac posture as the SANA lane (node-local HOME for Kit, EULA, Vulkan ICD, one bash-level attempt).
  ( cd "$XPL_WS" && exec env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PATH="$SIM_ENV_BIN:$PATH" \
      HOME="$KIT_HOME" WANDB_MODE=disabled OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y \
      VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}" \
      ROBODOJO_RUN_ID="$RUN_ID" ROBODOJO_MAX_BASH_RETRIES=1 EVAL_NUM="$EPS" \
      setsid timeout --signal=TERM --kill-after=60s "$WATCHDOG" \
      bash scripts/eval_policy.sh --root_dir "$XPL_WS" --task_name "$TASK" --env_cfg_type "$ENV_CFG_TYPE" \
        --device_id "$CLIENT_GPU" --policy_name "$POLICY_NAME" --host 127.0.0.1 --port "$RPC_PORT" --protocol ws \
        --additional_info "$ADDITIONAL_INFO" --seed "$SEED" ) \
    >"$CD/client.stdout.log" 2>"$CD/client.stderr.log" 9>&- &
  CLIENT_PID=$!; status CLIENT_START "watchdog=${WATCHDOG}s step_lim=$STEP_LIM per_episode=${PER_EPISODE}s stall=${STALL_SECONDS}s run_id=$RUN_ID"
  log "client started: run_id=$RUN_ID step_lim=$STEP_LIM per-episode budget ${PER_EPISODE}s, hard watchdog ${WATCHDOG}s, stall watchdog ${STALL_SECONDS}s"
  stall_watch "$CLIENT_PID" 9>&- &
  STALL_PID=$!
  wait "$CLIENT_PID"; CLIENT_RC=$?
  stop_stall_watch
  set -e
  reap_group "$CLIENT_PID" "client"; CLIENT_PID=""; log "client finished rc=$CLIENT_RC (preliminary)"
  WD_FIRED=false; (( CLIENT_RC == 124 || CLIENT_RC == 137 )) && WD_FIRED=true
  RES_DIR=$XPL_WS/eval_result/RoboDojo/$TASK/$POLICY_NAME/$ENV_CFG_TYPE/${SEED}_${ADDITIONAL_INFO}/$RUN_ID
  if [[ -d $RES_DIR ]]; then cp -r -- "$RES_DIR" "$CD/upstream_eval_result" || true; else log "no upstream result dir at $RES_DIR"; fi
  "$PYBIN" - "$CD/run_status.json" "$TASK" "$EPS" "$SEED" "$CLIENT_RC" "$WD_FIRED" "$RUN_ID" "$POLICY_NAME" <<'PY'
import json, sys
out, task, eps, seed, rc, wd, run_id, policy_name = sys.argv[1:]
json.dump({"task": task, "eval_num": eps, "eval_seed": int(seed), "client_rc": int(rc), "client_watchdog_fired": wd == "true",
           "run_id": run_id, "lane": "xpolicylab", "policy_name": policy_name}, open(out, "w"), indent=2)
PY
  stop_server_now
fi

set +e; counts=$(judge); jrc=$?; set -e
read -r eps_rec succ score <<<"$counts"
if (( jrc == 0 )); then write_result complete "$eps_rec" "$succ" "$score" "all $EPS episodes recorded (XPolicyLab lane)"; exit 0
else write_result incomplete "$eps_rec" "$succ" "$score" "recorded $eps_rec/$EPS episodes (watchdog or crash); see client/"; exit 2; fi
