#!/usr/bin/env bash
# =============================================================================
# run_eval_job.sh PLAN_JSON JOB_ID -- run one two-GPU evaluation job on THIS node.
#
# For every session of the job (one per (eval seed, episodes_per_task) group):
#   1. start the SANA policy server on the job's server GPU with the session's
#      task list (run_learned_server_only_local.sh, task-list lane);
#   2. wait for the client handoff bundle (endpoint loaded + listening);
#   3. run the upstream-replaying benchmark client on the job's client GPU
#      (run_remote_benchmark_client.sh --keep-going), one fresh Isaac Sim
#      process per task, sharing the one server session;
#   4. stop the server; mark every task whose upstream _result.json carries
#      all episodes_per_task episodes as DONE; anything else is re-queued.
# Up to MAX_ROUNDS passes; the job ends with JOB_COMPLETE or JOB_INCOMPLETE.
# Re-running the same command resumes (done units are remembered in state/).
#
# Layout (RUN_ROOT = dirname PLAN_JSON, on the shared home):
#   RUN_ROOT/jobs/<job>/runner.log, status.txt, runner.pid, state/done_units.txt
#   RUN_ROOT/jobs/<job>/sessions/<sid>_r<round>/server   (server evidence + client_handoff)
#   RUN_ROOT/jobs/<job>/sessions/<sid>_r<round>/client   (per-task client evidence)
# Env: ROBODOJO_ROOT (default /dataset2/jiamingt/robodojo), INSTALL_DIR (default
#      ~/robodojo_install), MAX_ROUNDS (3), EVAL_DRY_RUN=1 (fabricate results, no GPU)
# =============================================================================
set -Eeuo pipefail

PLAN=$(readlink -f "${1:?usage: run_eval_job.sh PLAN_JSON JOB_ID}")
JOB_ID=${2:?usage: run_eval_job.sh PLAN_JSON JOB_ID}
RUN_ROOT=$(dirname "$PLAN")
ROOT=${ROBODOJO_ROOT:-/dataset2/jiamingt/robodojo}
INSTALL_DIR=${INSTALL_DIR:-$HOME/robodojo_install}
MAX_ROUNDS=${MAX_ROUNDS:-3}
SERVER_STARTUP_TIMEOUT=${SERVER_STARTUP_TIMEOUT:-3000}   # eager load budget; 4 concurrent loads per node are slower than 1
HANDOFF_TIMEOUT=${HANDOFF_TIMEOUT:-$((SERVER_STARTUP_TIMEOUT + 1200))}
EPISODE_SECONDS=${EPISODE_SECONDS:-600}                # client watchdog budget per episode; a full 800-step failing episode took ~8 min in the smoke
DRY=${EVAL_DRY_RUN:-0}
[[ $RUN_ROOT == *robodojo* ]] || { echo "RUN_ROOT must contain 'robodojo' (server evidence rule): $RUN_ROOT" >&2; exit 64; }

JOB_DIR=$RUN_ROOT/jobs/$JOB_ID
STATE=$JOB_DIR/state
mkdir -p "$JOB_DIR" "$STATE" "$JOB_DIR/sessions"
exec 9>"$STATE/.lock"; flock -n 9 || { echo "another runner holds $STATE/.lock"; exit 1; }
echo $$ >"$JOB_DIR/runner.pid"
exec > >(tee -a "$JOB_DIR/runner.log") 2>&1
log()    { printf '[%s] [%s %s] %s\n' "$(date '+%F %T')" "$(hostname -s)" "$JOB_ID" "$*"; }
status() { echo "$(date -Is) $JOB_ID $*" >>"$JOB_DIR/status.txt"; }

# ---- runtime env (sim client + sana-wam server) --------------------------------
if [[ $DRY != 1 ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/activate.sh"
  # shellcheck disable=SC1091
  source "$ROOT/activate_sana_wam.sh"
  : "${SANA_PY:?}" "${SANA_REPO:?}" "${SANA_SIM_PY:?}" "${SANA_WS_PY:?}" "${ROBODOJO_CHECKOUT:?}"
  PYBIN=$SANA_WS_PY
else
  PYBIN=$(command -v python3)
fi

# ---- job record from the plan --------------------------------------------------
mapfile -t JOBREC < <("$PYBIN" - "$PLAN" "$JOB_ID" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1])); job = next(j for j in plan["jobs"] if j["job_id"] == sys.argv[2])
print(job["node"]); print(job["server_gpu"]); print(job["client_gpu"]); print(job["rpc_port"]); print(job["ws_port"])
for s in job["sessions"]:
    print(f"{s['session_id']}|{s['eval_seed']}|{s['episodes_per_task']}|{s['task_list']}|{' '.join(s['tasks'])}")
PY
)
(( ${#JOBREC[@]} >= 5 )) || { log "job $JOB_ID not found in $PLAN"; exit 64; }
JOB_NODE=${JOBREC[0]}; SERVER_GPU=${JOBREC[1]}; CLIENT_GPU=${JOBREC[2]}; RPC_PORT=${JOBREC[3]}; WS_PORT=${JOBREC[4]}
SESSIONS=("${JOBREC[@]:5}")
[[ $(hostname -s) == "$JOB_NODE" || $DRY == 1 ]] || log "WARN: plan places $JOB_ID on $JOB_NODE but this is $(hostname -s)"
TOTAL_UNITS=0; for s in "${SESSIONS[@]}"; do IFS='|' read -r _ _ _ _ tasks <<<"$s"; for _t in $tasks; do TOTAL_UNITS=$((TOTAL_UNITS+1)); done; done
touch "$STATE/done_units.txt"
done_count() { sort -u "$STATE/done_units.txt" | wc -l; }
is_done() { grep -qx "$1" "$STATE/done_units.txt"; }

# ---- process control -----------------------------------------------------------
SERVER_LAUNCHER_PID=""; CLIENT_PID=""; CUR_SERVER_DIR=""
port_free() { ! timeout 2 bash -c ">/dev/tcp/127.0.0.1/$1" 2>/dev/null; }
sweep_orphans() { # a previous runner died without stopping its server: stop it, then require a free rpc port
  local pf pid
  for pf in "$JOB_DIR"/sessions/*/server/launcher.pid; do
    [[ -s $pf ]] || continue; pid=$(<"$pf")
    if kill -0 "$pid" 2>/dev/null; then
      log "orphaned server launcher $pid from $(dirname "$pf"); stopping it"
      [[ $DRY == 1 ]] || bash "$INSTALL_DIR/stop_server.sh" "$(dirname "$pf")" >>"$JOB_DIR/runner.log" 2>&1 || kill_tree "$pid"
    fi
  done
  if [[ $DRY != 1 ]] && ! port_free "$RPC_PORT"; then log "rpc port $RPC_PORT is still in use on this node; refusing to start"; exit 75; fi
}
stop_server_now() {
  [[ -n $CUR_SERVER_DIR && -s $CUR_SERVER_DIR/launcher.pid ]] || return 0
  if [[ $DRY == 1 ]]; then return 0; fi
  bash "$INSTALL_DIR/stop_server.sh" "$CUR_SERVER_DIR" >>"$JOB_DIR/runner.log" 2>&1 || true
}
kill_tree() { # TERM a process and all of its descendants (deepest first)  (defined before sweep_orphans is called)
  local pid=$1 child
  for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$child"; done
  kill -TERM "$pid" 2>/dev/null || true
}
on_stop() {
  log "stop signal received; stopping client and server"
  [[ -n $CLIENT_PID ]] && kill_tree "$CLIENT_PID"
  stop_server_now
  if [[ -n $SERVER_LAUNCHER_PID ]] && kill -0 "$SERVER_LAUNCHER_PID" 2>/dev/null; then kill_tree "$SERVER_LAUNCHER_PID"; fi
  status all ABORTED "done=$(done_count)/$TOTAL_UNITS"
  exit 130
}
trap on_stop TERM INT

# ---- helpers -------------------------------------------------------------------
task_complete() { # task_complete CLIENT_EVIDENCE_DIR TASK EPISODES -> 0 if all episodes recorded
  local cdir=$1 task=$2 eps=$3 d
  d=$(ls -d "$cdir"/*_"$task" 2>/dev/null | head -n1) || true
  [[ -n $d && -f $d/upstream_eval_result/_result.json && -f $d/run_status.json ]] || return 1
  "$PYBIN" - "$d" "$eps" <<'PY'
import json, sys
d, eps = sys.argv[1], int(sys.argv[2])
rs = json.load(open(f"{d}/run_status.json")); res = json.load(open(f"{d}/upstream_eval_result/_result.json"))
details = res.get("details") or {}
ok = (not rs.get("client_watchdog_fired")) and len(details) >= eps
sys.exit(0 if ok else 1)
PY
}
fabricate_dry_results() { # EVAL_DRY_RUN: pretend the client ran (2 tasks per session fail on round 1 to exercise re-queue)
  local cdir=$1 eps=$2 round=$3; shift 3; local i=0 t
  mkdir -p "$cdir"
  for t in "$@"; do
    mkdir -p "$cdir/${i}_$t/upstream_eval_result"
    local n=$eps; if (( round == 1 && i < 2 )); then n=$((eps - 3)); fi
    "$PYBIN" - "$cdir/${i}_$t" "$t" "$n" "$eps" <<'PY'
import json, sys, random
d, task, n, eps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
import zlib; random.seed(zlib.crc32(task.encode()) % 1000)
json.dump({"details": {str(k): {"success": random.random() < 0.3, "score": round(random.random(), 3)} for k in range(n)}},
          open(f"{d}/upstream_eval_result/_result.json", "w"), indent=1)
json.dump({"task": task, "eval_num": str(eps), "client_rc": 0, "client_watchdog_fired": False,
           "policy_step_evidence_line_count": 10 * n}, open(f"{d}/run_status.json", "w"), indent=1)
PY
    i=$((i+1))
  done
}

# ---- main loop -----------------------------------------------------------------
rm -f "$JOB_DIR/JOB_COMPLETE" "$JOB_DIR/JOB_INCOMPLETE"   # a fresh (re)run re-derives the verdict
log "===== $JOB_ID start: node=$JOB_NODE server_gpu=$SERVER_GPU client_gpu=$CLIENT_GPU rpc=$RPC_PORT ws=$WS_PORT sessions=${#SESSIONS[@]} units=$TOTAL_UNITS done=$(done_count) dry=$DRY ====="
status all START "done=$(done_count)/$TOTAL_UNITS"
sweep_orphans
for (( round = 1; round <= MAX_ROUNDS; round++ )); do
  (( $(done_count) < TOTAL_UNITS )) || break
  for srec in "${SESSIONS[@]}"; do
    IFS='|' read -r SID SEED EPS LIST_REL TASKS <<<"$srec"
    REMAINING=(); for t in $TASKS; do is_done "$t@$SEED" || REMAINING+=("$t"); done
    (( ${#REMAINING[@]} > 0 )) || continue
    SDIR=$JOB_DIR/sessions/${SID}_r${round}
    if [[ -e $SDIR ]]; then SDIR=${SDIR}_$(date +%H%M%S); fi
    mkdir -p "$SDIR"
    LIST=$SDIR/task_list.txt; printf '%s\n' "${REMAINING[@]}" >"$LIST"
    log "--- round $round session $SID: seed=$SEED eps=$EPS tasks=${#REMAINING[@]} (${REMAINING[*]}) ---"
    status "$SID" START "round=$round tasks=${#REMAINING[@]} done=$(done_count)/$TOTAL_UNITS"

    if [[ $DRY == 1 ]]; then
      sleep 1; fabricate_dry_results "$SDIR/client" "$EPS" "$round" "${REMAINING[@]}"
    else
      # 1) server ------------------------------------------------------------
      CUR_SERVER_DIR=$SDIR/server
      PRED_VIDEO_DIR=$ROOT/prediction_videos/$JOB_ID/$(basename "$SDIR"); mkdir -p "$PRED_VIDEO_DIR"   # node-local: ~34 mp4 per episode
      SANA_SERVER_GPU=$SERVER_GPU SANA_RPC_PORT=$RPC_PORT SANA_EVAL_SEED=$SEED \
        SANA_TASK_LIST_FILE=$LIST SANA_EPISODES_PER_TASK=$EPS \
        SANA_PREDICTION_VIDEO_DIR=$PRED_VIDEO_DIR \
        ROBODOJO_SERVER_STARTUP_TIMEOUT_SECONDS=$SERVER_STARTUP_TIMEOUT \
        bash "$INSTALL_DIR/run_learned_server_only_local.sh" "$CUR_SERVER_DIR" \
        >"$SDIR/server.launcher.log" 2>&1 9>&- &
      SERVER_LAUNCHER_PID=$!
      waited=0
      until [[ -f $CUR_SERVER_DIR/client_handoff/CLIENT_README.txt ]]; do
        if ! kill -0 "$SERVER_LAUNCHER_PID" 2>/dev/null; then
          log "server launcher exited before publishing the endpoint; see $SDIR/server.launcher.log and $CUR_SERVER_DIR/server.stderr.log"
          status "$SID" SERVER_FAIL "round=$round"; CUR_SERVER_DIR=""; SERVER_LAUNCHER_PID=""; continue 2
        fi
        if (( waited >= HANDOFF_TIMEOUT )); then
          log "timed out after ${HANDOFF_TIMEOUT}s waiting for the endpoint"; stop_server_now
          status "$SID" SERVER_TIMEOUT "round=$round"; CUR_SERVER_DIR=""; SERVER_LAUNCHER_PID=""; continue 2
        fi
        sleep 5; waited=$((waited + 5))
      done
      log "server up after ${waited}s (rpc $RPC_PORT, gpu $SERVER_GPU)"; status "$SID" SERVER_UP "round=$round wait=${waited}s"

      # 2) client ------------------------------------------------------------
      KIT_HOME=$ROOT/kit_home/$JOB_ID; mkdir -p "$KIT_HOME"
      if [[ -z $(ls -A "$KIT_HOME") && -d ${ROBODOJO_KIT_HOME:-/nonexistent} ]]; then
        log "seeding $KIT_HOME from the warm kit cache ${ROBODOJO_KIT_HOME}"; cp -a "${ROBODOJO_KIT_HOME}/." "$KIT_HOME/" || true
      fi
      WATCHDOG=$(( (1800 + EPS * EPISODE_SECONDS) * round ))   # per task; grows with the retry round
      set +e
      ( cd "$SANA_REPO" && EVIDENCE_DIR=$SDIR/client SANA_RPC_PORT=$RPC_PORT SANA_ENV_GPU=$CLIENT_GPU WS_PORT=$WS_PORT \
          ROBODOJO_KIT_HOME=$KIT_HOME ROBODOJO_CLIENT_WATCHDOG_SECONDS=$WATCHDOG \
          VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json} \
          bash dev/rwm/simulation_evaluation/robodojo/launch/remote_client/run_remote_benchmark_client.sh --keep-going \
          "$CUR_SERVER_DIR/client_handoff" ) >"$SDIR/client.runner.log" 2>&1 9>&- &
      CLIENT_PID=$!
      wait "$CLIENT_PID"; CLIENT_RC=$?
      set -e
      CLIENT_PID=""
      log "benchmark client finished rc=$CLIENT_RC (preliminary; completion is judged per task below)"

      # 3) server down ---------------------------------------------------------
      stop_server_now; wait "$SERVER_LAUNCHER_PID" 2>/dev/null || true
      SERVER_LAUNCHER_PID=""; CUR_SERVER_DIR=""
    fi

    # 4) per-task completion --------------------------------------------------
    newly=0
    for t in "${REMAINING[@]}"; do
      if task_complete "$SDIR/client" "$t" "$EPS"; then echo "$t@$SEED" >>"$STATE/done_units.txt"; newly=$((newly+1)); else log "incomplete: $t (seed $SEED) -> re-queued"; fi
    done
    log "session $SID round $round: $newly/${#REMAINING[@]} tasks complete; job done=$(done_count)/$TOTAL_UNITS"
    status "$SID" DONE "round=$round complete=$newly/${#REMAINING[@]} done=$(done_count)/$TOTAL_UNITS"
  done
done

trap - TERM INT
if (( $(done_count) >= TOTAL_UNITS )); then
  { echo "date=$(date -Is)"; echo "job=$JOB_ID"; echo "units=$TOTAL_UNITS"; } >"$JOB_DIR/JOB_COMPLETE"
  status all COMPLETE "done=$(done_count)/$TOTAL_UNITS"; log "===== $JOB_ID COMPLETE ====="; exit 0
else
  { echo "date=$(date -Is)"; echo "job=$JOB_ID"; echo "done=$(done_count)/$TOTAL_UNITS"; echo "missing:";
    for s in "${SESSIONS[@]}"; do IFS='|' read -r _ seed _ _ tasks <<<"$s"; for t in $tasks; do is_done "$t@$seed" || echo "  $t@$seed"; done; done; } >"$JOB_DIR/JOB_INCOMPLETE"
  status all INCOMPLETE "done=$(done_count)/$TOTAL_UNITS"; log "===== $JOB_ID INCOMPLETE after $MAX_ROUNDS rounds; re-run to retry the missing units ====="; exit 2
fi
