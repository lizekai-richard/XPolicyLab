#!/usr/bin/env bash
# =============================================================================
# run_unit.sh RUN_ROOT UNIT_ID TASK SEED EPS SERVER_GPU CLIENT_GPU RPC_PORT WS_PORT [ATTEMPT]
#
# Worker side of the master/worker evaluation: ONE unit = one task at one eval
# seed with its full episode budget, run as one policy-server session plus one
# benchmark-client run on this node, on the given GPU pair.  Dispatched by
# master.py (on x5) over ssh, detached with nohup/setsid; reports back only
# through files on the shared home.  Every ATTEMPT gets its own directory
# (the server and client launchers refuse to reuse an evidence dir):
#   RUN_ROOT/units/UNIT_ID/.lock, unit.pid          one runner per unit at a time
#   RUN_ROOT/units/UNIT_ID/a<ATTEMPT>/status.txt    append-only status stream
#                                    heartbeat      touched every 30 s while alive
#                                    result.json    written once at the end: complete|incomplete|
#                                                   server_fail|server_timeout|port_busy|ckpt_missing|aborted|prologue_fail
#                                    server/ client/ unit.log
# Env: ROBODOJO_ROOT (/dataset2/jiamingt/robodojo), INSTALL_DIR (~/robodojo_install),
#      SERVER_STARTUP_TIMEOUT (3000), EVAL_DRY_RUN=1 (no GPU),
#      SANA_EVAL_CHECKPOINT_ROOT (master-selected policy checkpoint snapshot; overrides the node's
#      activate default SANA_POLICY_CHECKPOINT_ROOT; validated here before any GPU work)
# Client budget: the hard watchdog is 1800 + EPS * max(EPISODE_SECONDS(600), step_lim * SEC_PER_STEP(2.5)
#      + RESET_SECONDS(120)) where step_lim is the task's per-episode step limit (task/RoboDojo/tasks/<task>.py);
#      a STALL watchdog TERMs the client when it has printed nothing for STALL_SECONDS (2700).
# =============================================================================
set -Eeuo pipefail
(( $# == 9 || $# == 10 )) || { echo "usage: run_unit.sh RUN_ROOT UNIT_ID TASK SEED EPS SERVER_GPU CLIENT_GPU RPC_PORT WS_PORT [ATTEMPT]" >&2; exit 64; }
RUN_ROOT=$1; UNIT_ID=$2; TASK=$3; SEED=$4; EPS=$5; SERVER_GPU=$6; CLIENT_GPU=$7; RPC_PORT=$8; WS_PORT=$9; ATTEMPT=${10:-1}
ROOT=${ROBODOJO_ROOT:-/dataset2/jiamingt/robodojo}
INSTALL_DIR=${INSTALL_DIR:-$HOME/robodojo_install}
SERVER_STARTUP_TIMEOUT=${SERVER_STARTUP_TIMEOUT:-3000}
HANDOFF_TIMEOUT=${HANDOFF_TIMEOUT:-$((SERVER_STARTUP_TIMEOUT + 1200))}
EPISODE_SECONDS=${EPISODE_SECONDS:-600}      # floor per episode
SEC_PER_STEP=${SEC_PER_STEP:-2.5}            # observed ~0.85 s/step with 4 client+server pairs per node -> 3x margin
RESET_SECONDS=${RESET_SECONDS:-120}          # per-episode reset/video budget
STALL_SECONDS=${STALL_SECONDS:-2700}         # no simulation progress for this long = suspected hung client
STALL_CRASH_SECONDS=${STALL_CRASH_SECONDS:-600}  # ... only this long when the client log shows a crash signature
STALL_CONFIRM=${STALL_CONFIRM:-300}          # ... and still no progress after this long = confirmed (survives NFS outages)
STALL_POLL=${STALL_POLL:-60}
DRY=${EVAL_DRY_RUN:-0}
POLICY_ACCEL=${RWM_EVAL_POLICY_ACCELERATION:-off}   # off | fused | fused_fp8 (SM90 only) | fused_portable (A6000); not comparable across presets
PRED_VIDEOS=${SANA_PREDICTION_VIDEOS:-1}            # 0 = no server-side diagnostic prediction videos (their VAE decode sits inside the timed inference)
LAZY_RENDER=${SANA_ROBODOJO_LAZY_RENDER:-0}         # 0 (default since 2026-09-09): get_obs after every action like the official evaluator. 1 skips
                                                    # mid-chunk get_obs -- NOT behaviour-neutral: it changes the simulator cadence between actions
                                                    # (insert_key ep7: 11-12/50 stage-1 hits with 1 vs 25/50 with 0)
[[ $RUN_ROOT == /*robodojo* ]] || { echo "RUN_ROOT must be an absolute path containing 'robodojo'" >&2; exit 64; }
[[ $ATTEMPT =~ ^[0-9]+$ ]] || { echo "ATTEMPT must be an integer" >&2; exit 64; }

U=$RUN_ROOT/units/$UNIT_ID
A=$U/a$ATTEMPT
mkdir -p "$A"
# log first, lock second: tee must NOT inherit the lock descriptor (it outlives us by a moment)
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
           "started": started, "finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds")},
          open(tmp, "w"), indent=2, sort_keys=True)
os.replace(tmp, out)
PY
  status "RESULT $1"; log "result: $1 ${5:-}"
}
trap 'rc=$?; write_result prologue_fail 0 0 0 "prologue failed at line $LINENO (rc $rc)"; stop_heartbeat; exit $rc' ERR

# ---- runtime env -------------------------------------------------------------------
if [[ $DRY != 1 ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/activate.sh"
  # shellcheck disable=SC1091
  source "$ROOT/activate_sana_wam.sh"
  : "${SANA_PY:?}" "${SANA_REPO:?}" "${SANA_SIM_PY:?}" "${SANA_WS_PY:?}" "${ROBODOJO_CHECKOUT:?}"
  PYBIN=$SANA_WS_PY
  # ---- policy checkpoint: the master's choice overrides the node's activate default -------------
  if [[ -n ${SANA_EVAL_CHECKPOINT_ROOT:-} ]]; then export SANA_POLICY_CHECKPOINT_ROOT=$SANA_EVAL_CHECKPOINT_ROOT; fi
  CKPT=${SANA_POLICY_CHECKPOINT_ROOT:-$HOME/checkpoints/huggingface/logits/sana_rwm_policy_from_pretrained_robodojo_320px_epoch4_step25000}
  export SANA_POLICY_CHECKPOINT_ROOT=$CKPT
  missing=()
  for f in config.yaml README.md model/pytorch_model_fsdp.bin; do [[ -f $CKPT/$f ]] || missing+=("$f"); done
  norms=("$CKPT"/normalization/*.json)
  [[ -f ${norms[0]} ]] || missing+=("normalization/*.json")
  inc=""
  if [[ -d $CKPT/.cache ]]; then inc=$(find "$CKPT/.cache" -name '*.incomplete' 2>/dev/null | head -1 || true); fi
  [[ -z $inc ]] || missing+=("download still in progress: $inc")
  if (( ${#missing[@]} )); then
    trap - ERR; write_result ckpt_missing 0 0 0 "checkpoint $CKPT unusable: ${missing[*]}"; stop_heartbeat; exit 78
  fi
  # the server launcher's default normalization file name is the epoch4 one; a snapshot whose single
  # normalization artifact is named differently is served all the same
  if [[ -z ${SANA_POLICY_NORMALIZATION:-} && ! -f $CKPT/normalization/robodojo_arx_x5_model_fps_25_f25_normalization.json && ${#norms[@]} -eq 1 ]]; then
    export SANA_POLICY_NORMALIZATION=${norms[0]}
  fi
  # the server resolves the normalization receipt's input cache (sha256-checked) from
  # SANA_NORMALIZATION_INPUT_CACHE_DIR first: point it at this snapshot's own copy when it ships one
  if [[ -d $CKPT/json_cache ]]; then export SANA_NORMALIZATION_INPUT_CACHE_DIR=$CKPT/json_cache; fi
  CKPT_LABEL=$(basename "$CKPT")
fi
CKPT_LABEL=${CKPT_LABEL:-dry}
trap - ERR

# ---- process control ---------------------------------------------------------------
SERVER_LAUNCHER_PID=""; CLIENT_PID=""
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
  [[ -s $A/server/launcher.pid ]] || return 0
  [[ $DRY == 1 ]] && return 0
  timeout 1000 bash "$INSTALL_DIR/stop_server.sh" "$A/server" >>"$A/unit.log" 2>&1 || true
  [[ -n $SERVER_LAUNCHER_PID ]] && wait_pid_bounded "$SERVER_LAUNCHER_PID" 120
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
stall_watch() { # stall_watch CLIENT_PID: TERM the client when the simulation stops advancing
  # Progress = number of policy steps in client.stdout.log, NOT file mtimes: a crashed Kit process keeps
  # spamming stderr (`sh: 1: zenity: not found`, aftermath polling) for hours, so mtimes never go stale
  # (2026-09-03: hang_mugs_random hung 4 h after a GPU pagefault with the mtime-based watchdog silent).
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
      # A shared-home (NFS) outage looks like a stall (the client's writes and our reads both block);
      # confirm before killing: still no new steps after STALL_CONFIRM more seconds.
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
  [[ -n $CLIENT_PID ]] && kill_tree "$CLIENT_PID"
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

# ---- run -----------------------------------------------------------------------------
log "===== unit $UNIT_ID attempt $ATTEMPT: task=$TASK seed=$SEED eps=$EPS gpus=$SERVER_GPU/$CLIENT_GPU rpc=$RPC_PORT ws=$WS_PORT checkpoint=${CKPT:-dry} dry=$DRY ====="
status START "task=$TASK seed=$SEED eps=$EPS node=$HOST gpus=$SERVER_GPU/$CLIENT_GPU checkpoint=$CKPT_LABEL policy_acceleration=$POLICY_ACCEL prediction_videos=$PRED_VIDEOS lazy_render=$LAZY_RENDER disable_xformers=${DISABLE_XFORMERS:-0}"
if [[ $DRY == 1 ]]; then
  sleep "${DRY_SLEEP:-2}"; mkdir -p "$A/client/0_$TASK/upstream_eval_result"
  "$PYBIN" - "$A/client/0_$TASK" "$TASK" "$EPS" "$SEED" "$ATTEMPT" <<'PY'
import json, sys, random
d, task, eps, attempt = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[5]); random.seed()
n = eps - 1 if (attempt == 1 and random.random() < 0.2) else eps                    # ~20% of FIRST attempts incomplete; retries succeed
json.dump({"details": {str(k): {"success": random.random() < 0.3, "score": round(random.random(), 3)} for k in range(n)}}, open(f"{d}/upstream_eval_result/_result.json", "w"))
json.dump({"task": task, "eval_num": str(eps), "eval_seed": int(sys.argv[4]), "client_rc": 0, "client_watchdog_fired": False, "policy_step_evidence_line_count": 10 * n,
           "run_id": "dry-" + __import__("datetime").datetime.utcnow().strftime("%Y%m%d-%H%M%S%f")}, open(f"{d}/run_status.json", "w"))
PY
else
  for p in "$RPC_PORT" "$WS_PORT"; do port_free "$p" || reclaim_port "$p"; done
  if ! port_free "$RPC_PORT"; then write_result port_busy 0 0 0 "rpc port $RPC_PORT in use on $HOST"; exit 75; fi
  if ! port_free "$WS_PORT"; then write_result port_busy 0 0 0 "ws port $WS_PORT in use on $HOST"; exit 75; fi
  LIST=$A/task_list.txt; echo "$TASK" >"$LIST"
  PRED=$ROOT/prediction_videos/$(basename "$RUN_ROOT")/$UNIT_ID/a$ATTEMPT   # node-local: ~34 mp4 per episode
  [[ $PRED_VIDEOS == 1 ]] && mkdir -p "$PRED"
  SANA_SERVER_GPU=$SERVER_GPU SANA_RPC_PORT=$RPC_PORT SANA_EVAL_SEED=$SEED SANA_TASK_LIST_FILE=$LIST SANA_EPISODES_PER_TASK=$EPS \
    SANA_PREDICTION_VIDEO_DIR=$PRED SANA_PREDICTION_VIDEOS=$PRED_VIDEOS ROBODOJO_SERVER_STARTUP_TIMEOUT_SECONDS=$SERVER_STARTUP_TIMEOUT \
    RWM_EVAL_POLICY_ACCELERATION=$POLICY_ACCEL \
    bash "$INSTALL_DIR/run_learned_server_only_local.sh" "$A/server" >"$A/server.launcher.log" 2>&1 9>&- &
  SERVER_LAUNCHER_PID=$!
  waited=0
  until [[ -f $A/server/client_handoff/CLIENT_README.txt ]]; do
    if ! kill -0 "$SERVER_LAUNCHER_PID" 2>/dev/null; then
      # Errno 98 at bind = something grabbed the slot's rpc port between our port_free check and the
      # server's bind ~100 s later (41000-42150 sit inside Linux's ephemeral range 32768-60999, so any
      # outgoing connection on the node can collide; Isaac/Kit opens many at start-up).  Record who.
      why="server exited before publishing (see server.launcher.log / server/server.stderr.log)"
      if grep -q 'Errno 98' "$A/server/server.stderr.log" 2>/dev/null; then
        why="server could not bind rpc port $RPC_PORT (Errno 98 address in use; transient, see unit.log for the holder)"
        log "rpc port $RPC_PORT holders at failure time:"; ss -tanp "( sport = :$RPC_PORT or dport = :$RPC_PORT )" 2>/dev/null | head -n 20
      fi
      write_result server_fail 0 0 0 "$why"; exit 70
    fi
    if (( waited >= HANDOFF_TIMEOUT )); then
      stop_server_now; write_result server_timeout 0 0 0 "no endpoint after ${HANDOFF_TIMEOUT}s"; exit 70
    fi
    sleep 5; waited=$((waited + 5))
  done
  status SERVER_UP "wait=${waited}s"; log "server up after ${waited}s"
  KIT_HOME=$ROOT/kit_home/slot_${SERVER_GPU}_${CLIENT_GPU}; mkdir -p "$KIT_HOME"
  if [[ -z $(ls -A "$KIT_HOME") && -d ${ROBODOJO_KIT_HOME:-/nonexistent} ]]; then cp -a "${ROBODOJO_KIT_HOME}/." "$KIT_HOME/" || true; fi
  # ---- client budget: EPS episodes that may all run to the task's step limit ----
  STEP_LIM=$(grep -o -m1 'self.step_lim = [0-9]*' "$ROBODOJO_CHECKOUT/task/RoboDojo/tasks/$TASK.py" 2>/dev/null | grep -o '[0-9]*$' || true)
  STEP_LIM=${STEP_LIM:-2000}
  PER_EPISODE=$(awk -v l="$STEP_LIM" -v s="$SEC_PER_STEP" -v r="$RESET_SECONDS" -v e="$EPISODE_SECONDS" 'BEGIN { v = l * s + r; if (v < e) v = e; printf "%d", v }')
  WATCHDOG=$(( 1800 + EPS * PER_EPISODE ))
  set +e
  ( cd "$SANA_REPO" && EVIDENCE_DIR=$A/client SANA_RPC_PORT=$RPC_PORT SANA_ENV_GPU=$CLIENT_GPU WS_PORT=$WS_PORT \
      ROBODOJO_KIT_HOME=$KIT_HOME ROBODOJO_CLIENT_WATCHDOG_SECONDS=$WATCHDOG SANA_ROBODOJO_LAZY_RENDER=$LAZY_RENDER \
      VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json} \
      exec setsid bash dev/rwm/simulation_evaluation/robodojo/launch/remote_client/run_remote_benchmark_client.sh --keep-going "$A/server/client_handoff" ) \
      >"$A/client.runner.log" 2>&1 9>&- &
  CLIENT_PID=$!; status CLIENT_START "watchdog=${WATCHDOG}s step_lim=$STEP_LIM per_episode=${PER_EPISODE}s stall=${STALL_SECONDS}s"
  log "client started: step_lim=$STEP_LIM per-episode budget ${PER_EPISODE}s, hard watchdog ${WATCHDOG}s, stall watchdog ${STALL_SECONDS}s"
  stall_watch "$CLIENT_PID" 9>&- &
  STALL_PID=$!
  wait "$CLIENT_PID"; CLIENT_RC=$?
  stop_stall_watch
  set -e
  reap_group "$CLIENT_PID" "client"; CLIENT_PID=""; log "client finished rc=$CLIENT_RC (preliminary)"
  stop_server_now; SERVER_LAUNCHER_PID=""
fi

set +e; counts=$(judge); jrc=$?; set -e
read -r eps_rec succ score <<<"$counts"
if (( jrc == 0 )); then write_result complete "$eps_rec" "$succ" "$score" "all $EPS episodes recorded"; exit 0
else write_result incomplete "$eps_rec" "$succ" "$score" "recorded $eps_rec/$EPS episodes (watchdog or crash); see client/"; exit 2; fi
