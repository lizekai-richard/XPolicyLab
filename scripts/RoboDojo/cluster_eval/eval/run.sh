#!/usr/bin/env bash
# run.sh -- the per-run driver of the RoboDojo cluster evaluation (control host): ONE checkpoint x ONE eval seed over the
# COMPLETE official protocol (42 tasks / 54 task configs / 2100 episodes; master --support-arm-embodiments on), served through
# an XPolicyLab adapter on single-GPU slots (policy server and Isaac client share one board), layout-sharded (SHARD_PLAN),
# dispatched one capability category at a time, finished with the tail template and summarized with the official
# summarize_result.py. Everything that defines a run comes from the environment -- eval/campaign.sh exports it from a
# campaign definition (eval/defs/*.env); by hand, export the variables below:
#   ADAPTER      SANA_WAM (default) | SANA_MOT | SANA_WAM_CAUSAL | SANA_WAM_SM   (ports, worker env, synced adapter dirs:
#                lib/adapters.sh)
#   CKPT_LABEL   the checkpoint snapshot <CE_CKPT_ROOT>/<label> (REQUIRED); CKPT overrides the directory
#   NORM_SHA     sha256 of the normalization artifact the snapshot's training yaml pins (REQUIRED to launch: the adapter refuses
#                a snapshot whose normalization differs; read it from <snapshot>/config.yaml normalization_sha256)
#   SEED         eval seed (0)
#   NFE (10) VIDEO_CFG (6) ACTION_CFG (1)           sampling steps and the per-stream guidance (VIDEO_CFG=1 ACTION_CFG=1 = guidance
#                                                   off, no unconditional pass: checkpoints trained without condition dropout)
#   VIDEO_FS ACTION_FS                              flow shifts (SANA_WAM / SANA_WAM_CAUSAL / SANA_WAM_SM 5 / 1; SANA_MOT 8 /
#                                                   shared)
#   NAS          n_action_steps: execute the first NAS of the 32 predicted targets, then replan (unset = the whole chunk;
#                refused for SANA_WAM_CAUSAL / SANA_WAM_SM, whose memory clock advances by whole 32-action chunks)
#   ACTION_TYPE  ee (default: end-effector poses through the evaluator's IK) | joint
#   VIDEO_LAYOUT SANA_MOT only (default sana_pixel_canvas): asserts the checkpoint's video layout
#   VIEW_RESIZE  crop | stretch: ONLY to force a view resize against the checkpoint's own (an A/B; run name _eval<mode>). By
#                default the adapter stretches, and every label in eval/pre_stretch_labels.txt is served with the legacy crop
#   ORGANIZE_TABLE_PROMPT  new (default: the instruction the simulator gives) | old: organize_table gets the OLD instruction -- the long
#                sentence of RoboDojo before 726e9aa and of the HF RoboDojo data <= cfb06d1d (2026-08-17), the training prompt of the
#                checkpoints trained on it (run name _oldorg; the worker edits a node-local copy of the pinned task/, see run_unit_xpl.sh)
#   EXTRA_OVERRIDES  more space-separated deploy.yml key=value overrides (diagnostics, e.g. prediction_latent_dir=@ATTEMPT_DIR@/pred)
#   RUN          the run name; default official_seed<SEED>_<label>_xpl_vcfg<V>_acfg<A>_nfe<N>_<vfs<v>_afs<a> | fs<v>>_<action
#                type>[_oldorg][_eval<resize>][_nas<NAS>] -- the names of every published run
#   SHARD_PLAN DIM_ORDER PIN_LONGEST EXTRA_MASTER_ARGS   master planning knobs (defaults below)
#   BASELINES    run names `compare` puts next to this run
#
#   bash run.sh name | show                   the run name / every resolved setting (overrides, master arguments, ports)
#   bash run.sh check-sync                    the worker scripts, site.env and adapter dirs on the shared home == this checkout
#   bash run.sh free-spec                     every free single-GPU slot of CE_NODES (healthy, < 1 GiB, ports closed, no master)
#   bash run.sh smoke-gpu NODE:GPU [TASK] [EPS]   direct worker on ONE free GPU (default insert_key x 1) -> run <RUN>_smoke1gpu
#   bash run.sh smoke-status [TASK]           its status / server log / result
#   bash run.sh smoke-verify [TASK]           judge it: exit 0 passed, 1 not finished, 2 failed (the adapter's ready line must
#                                             carry every token of `expect-ready` plus the '|'-separated EXPECT tokens)
#   bash run.sh expect-ready                  the ready-line tokens implied by the settings above
#   bash run.sh dry ['<spec>']                local --dry-run of the master's plan (no ssh)
#   bash run.sh start ['<spec>']              the full run; default spec = every free single-GPU slot (free-spec)
#   bash run.sh resume ['<spec>']             restart this run's stopped master (--resume) on the slots its state.json holds
#   bash run.sh status | report | watch | server-log UNIT [RUN] [ATTEMPT] | compare
#   bash run.sh tail plan|launch|status|more-substitutes|assemble|invalidate|fill|fill-status [TASK ...] [--opts]
#                                             the TAIL TEMPLATE (eval/tail_shards.py): finish long unsharded units as
#                                             one-layout shards + substitutes on idle GPUs once the queue is empty
#   bash run.sh official-summary              official summarize_result.py table -> <CE_RESULTS_ROOT>/runs/<RUN>/OFFICIAL_SUMMARY_<ts>.md;
#                                             SUMMARY_RUNS="<run> <run> ..." lays several runs of ONE checkpoint into one table
#                                             (the 3-seed table: OFFICIAL_SUMMARY_combined_<ts>.md)
set -euo pipefail
EVAL=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../lib/site.sh
source "$EVAL/../lib/site.sh"
# shellcheck source=../lib/adapters.sh
source "$EVAL/../lib/adapters.sh"
# shellcheck source=../lib/slots.sh
source "$EVAL/../lib/slots.sh"
ADAPTER=${ADAPTER:-SANA_WAM}; ce_adapter_profile "$ADAPTER"
PY=$CE_PY
log(){ echo "[$(date -u +%FT%TZ)] $*"; }

CKPT_LABEL=${CKPT_LABEL:-}
CKPT=${CKPT:-${CKPT_LABEL:+$CE_CKPT_ROOT/$CKPT_LABEL}}
NORM_SHA=${NORM_SHA:-}
SEED=${SEED:-0}; [[ $SEED =~ ^[0-9]+$ ]] || { echo "SEED must be an integer, got '$SEED'"; exit 2; }
NFE=${NFE:-10}; VIDEO_CFG=${VIDEO_CFG:-6}; ACTION_CFG=${ACTION_CFG:-1}
VIDEO_FS=${VIDEO_FS:-$VIDEO_FS_DEFAULT}; ACTION_FS=${ACTION_FS-$ACTION_FS_DEFAULT}
NAS=${NAS:-}; [[ -z $NAS || $NAS =~ ^[0-9]+$ ]] || { echo "NAS must be an integer, got '$NAS'"; exit 2; }
[[ -n $NAS && ( $ADAPTER == SANA_WAM_CAUSAL || $ADAPTER == SANA_WAM_SM ) ]] && {
  echo "NAS: $ADAPTER executes whole 32-action chunks (its memory clock; the adapter has no n_action_steps): unset NAS"; exit 2; }
ACTION_TYPE=${ACTION_TYPE:-ee}; [[ $ACTION_TYPE == ee || $ACTION_TYPE == joint ]] || { echo "ACTION_TYPE must be ee or joint"; exit 2; }
VIDEO_LAYOUT=${VIDEO_LAYOUT:-$VIDEO_LAYOUT_DEFAULT}
# How each camera view reaches its bucket (adapter deploy key view_resize). The adapters stretch by default -- the only SFT resize
# since the upstream change of 2026-09-27 -- and the worker serves every label in eval/pre_stretch_labels.txt (verified trained
# before it) with the legacy crop, so nothing is set by default. VIEW_RESIZE forces a mode against the checkpoint's own (a
# train/eval-mismatch A/B): the run name gets _evalcrop / _evalstretch, and a launch whose mode disagrees with its run is refused.
VR_REGISTRY=$EVAL/pre_stretch_labels.txt
[[ -f $VR_REGISTRY ]] || { echo "missing $VR_REGISTRY"; exit 2; }
TRAINED_RESIZE=stretch
awk -v l="$CKPT_LABEL" '!/^[[:space:]]*(#|$)/ && $1 == l {f = 1} END {exit !f}' "$VR_REGISTRY" && TRAINED_RESIZE=crop
VIEW_RESIZE=${VIEW_RESIZE:-}
[[ -z $VIEW_RESIZE || $VIEW_RESIZE == crop || $VIEW_RESIZE == stretch ]] || { echo "VIEW_RESIZE must be crop or stretch, got '$VIEW_RESIZE'"; exit 2; }
[[ " ${EXTRA_OVERRIDES:-} " == *" view_resize="* ]] && { echo "set VIEW_RESIZE instead of view_resize= in EXTRA_OVERRIDES"; exit 2; }
[[ " ${EXTRA_OVERRIDES:-} " == *" n_action_steps="* ]] && { echo "set NAS instead of n_action_steps= in EXTRA_OVERRIDES"; exit 2; }
VR_TAG=""; [[ -n $VIEW_RESIZE && $VIEW_RESIZE != "$TRAINED_RESIZE" ]] && VR_TAG=_eval$VIEW_RESIZE
if [[ -n $VIEW_RESIZE ]]; then VR_EXPECT="$VIEW_RESIZE (deploy)"; elif [[ $TRAINED_RESIZE == crop ]]; then VR_EXPECT="crop (deploy)"; else VR_EXPECT="stretch (default)"; fi
FS_TAG=fs$VIDEO_FS; [[ -n $ACTION_FS ]] && FS_TAG=vfs${VIDEO_FS}_afs$ACTION_FS
ORGANIZE_TABLE_PROMPT=${ORGANIZE_TABLE_PROMPT:-new}
[[ $ORGANIZE_TABLE_PROMPT == new || $ORGANIZE_TABLE_PROMPT == old ]] || { echo "ORGANIZE_TABLE_PROMPT must be new or old, got '$ORGANIZE_TABLE_PROMPT'"; exit 2; }
OT_TAG=""; if [[ $ORGANIZE_TABLE_PROMPT == old ]]; then OT_TAG=_oldorg; WORKER_ENV="${WORKER_ENV:+$WORKER_ENV }SANA_EVAL_ORGANIZE_TABLE_PROMPT=old"; fi
RUN=${RUN:-official_seed${SEED}_${CKPT_LABEL}_xpl_vcfg${VIDEO_CFG}_acfg${ACTION_CFG}_nfe${NFE}_${FS_TAG}_${ACTION_TYPE}${OT_TAG}${VR_TAG}${NAS:+_nas$NAS}}
OVERRIDES="sampling_steps=$NFE cfg_scale=$VIDEO_CFG action_cfg_scale=$ACTION_CFG flow_shift=$VIDEO_FS"
[[ -n $ACTION_FS ]] && OVERRIDES+=" action_flow_shift=$ACTION_FS"
OVERRIDES+=" normalization_sha256=$NORM_SHA"
[[ $ADAPTER == SANA_MOT && -n $VIDEO_LAYOUT ]] && OVERRIDES+=" video_layout=$VIDEO_LAYOUT"
[[ -n $NAS ]] && OVERRIDES+=" n_action_steps=$NAS"
[[ -n $VIEW_RESIZE ]] && OVERRIDES+=" view_resize=$VIEW_RESIZE"
[[ -n ${EXTRA_OVERRIDES:-} ]] && OVERRIDES+=" $EXTRA_OVERRIDES"
# Layout shards (master --shard-plan): every task config expected to take > 3.2 h as ONE unit is split into contiguous layout
# ranges run as separate units (counts = ceil(hours / 2.5) from the slowest comparable run). Tasks whose layouts the official
# client abandons stay UNSHARDED -- a shard marks every layout outside its range abandoned, so it cannot draw the substitutes:
# deposit_coin, fill_egg_holder, general_pickup (abandon fixed layouts) and play_tic_tac_toe / imitate_sorting_sequence (abandon
# policy-dependent layouts; they become the run's tail, finished by the tail template).
SHARD_PLAN=${SHARD_PLAN-make_kong=2,fasten_screws=5,pour_by_language=4,play_stacking_toy=3,fill_pen_holder=3,classify_objects_by_language=3,organize_table=3,classify_objects=3,build_tower=3,store_tools_in_toolbox=3,make_toast_random=2,cover_blocks=2,make_toast=2,swap_blocks=2,pick_from_conveyor_by_image=2,match_and_pick_from_conveyor=2,pack_objects_into_box=2,pack_objects_into_box_random=2,pour_balls_into_vase=2,arrange_largest_number_random=2,sweep_blocks_random=2,arrange_largest_number=2,sort_nesting_dolls_by_size=2,insert_tubes=2,put_bottles_into_dustbin=2,sweep_blocks=2,press_by_number=2,sort_nesting_dolls_by_size_random=2,play_Xylophone=2}
DIM_ORDER=${DIM_ORDER:-$DIM_ORDER_DEFAULT}   # categories one after another, so per-category comparisons arrive early
PIN_LONGEST=${PIN_LONGEST:-2}                # the longest units dispatched first regardless of category
STATE=$CE_MASTER_ROOT/$RUN
SCRATCH=${SCRATCH:-$CE_MASTER_ROOT/.scratch/$RUN}; mkdir -p "$SCRATCH"
SMOKE_RUN_ROOT=${SMOKE_RUN_ROOT:-$CE_EVAL_ROOT/${RUN}_smoke1gpu}
SMOKE_RPC_PORT=${SMOKE_RPC_PORT:-$SMOKE_RPC}; SMOKE_WS_PORT=${SMOKE_WS_PORT:-$SMOKE_WS}
GW=$(ce_gw)
SHARD_ARGS=(); [[ -n $SHARD_PLAN ]] && SHARD_ARGS=(--shard-plan "$SHARD_PLAN")
WORKER_ENV_ARGS=(); [[ -n $WORKER_ENV ]] && WORKER_ENV_ARGS=(--worker-env "$WORKER_ENV")
COMMON=(--worker run_unit_xpl.sh --xpl-root "$CE_REMOTE_XPL" --xpl-action-type "$ACTION_TYPE" --sec-per-step 4 --checkpoint-root "$CKPT"
        --xpl-overrides "$OVERRIDES" "${WORKER_ENV_ARGS[@]}" --order dimension --pin-longest "$PIN_LONGEST" "${SHARD_ARGS[@]}"
        --support-arm-embodiments on --seeds "$SEED" --dimension-order "$DIM_ORDER" --rpc-port-base "$RPC_BASE" --ws-port-base "$WS_BASE")

# OFFLINE TEST HOOK (never set it for a real run): CE_TEST_DRY_MASTER=1 makes start / resume launch the master with --dry-run
# (fake local workers, no ssh) and skip the shared-home and GPU checks, so eval/campaign.sh can be exercised end to end
DRY_MASTER=(); [[ ${CE_TEST_DRY_MASTER:-0} == 1 ]] && DRY_MASTER=(--dry-run --dry-sleep "${CE_TEST_DRY_SLEEP:-0.05}" --poll 2)
need_launch(){ # what every launch path needs
  [[ -n $CKPT_LABEL || -n ${CKPT:-} ]] || { echo "set CKPT_LABEL (the snapshot <CE_CKPT_ROOT>/<label>)"; exit 2; }
  [[ $NORM_SHA =~ ^[0-9a-f]{64}$ ]] || { echo "set NORM_SHA to the snapshot's normalization_sha256 (64 hex; <snapshot>/config.yaml)"; exit 2; }
}
check_sync(){ # node-side copies on the shared home == this checkout (deploy.sh writes them)
  local diff="" pol
  diff=$(rsync -n -c -i -e "ssh $CE_SSH_OPTS" "$EVAL/run_unit_xpl.sh" "$EVAL/run_unit.sh" "$EVAL/pre_stretch_labels.txt" "$GW:$CE_REMOTE_INSTALL/eval/" | grep -v '^\.' | sed 's#^#  eval/: #' || true)
  diff+=$(rsync -n -c -i -L -e "ssh $CE_SSH_OPTS" "$CE_SITE" "$GW:$CE_REMOTE_INSTALL/site.env" | grep -v '^\.' | sed 's#^#  site.env: #' || true)
  for pol in $SYNC_POLICIES; do
    diff+=$(rsync -n -c -i -r -e "ssh $CE_SSH_OPTS" --exclude __pycache__ --exclude .pytest_cache "$CE_LOCAL_XPL/policy/$pol/" "$GW:$CE_REMOTE_XPL/policy/$pol/" | grep -v '^\.d\|^cd\|^\.f' | sed "s#^#  policy/$pol/: #" || true)
  done
  diff+=$(rsync -n -c -i -r -e "ssh $CE_SSH_OPTS" --exclude __pycache__ "$CE_LOCAL_XPL/utils/" "$GW:$CE_REMOTE_XPL/utils/" | grep -v '^\.d\|^cd\|^\.f' | sed 's#^#  utils/: #' || true)
  if [[ -n $diff ]]; then echo "the shared-home copies differ from this checkout (run deploy.sh first):"; echo "$diff"; exit 1; fi
  log "worker scripts, site.env, utils/ and the adapters ($SYNC_POLICIES) are in sync with the shared home"
}
check_view_resize(){ # a shard or retry is served like the run it lands in
  local run_mode=$TRAINED_RESIZE serve=${VIEW_RESIZE:-$TRAINED_RESIZE}
  case $RUN in *_evalcrop) run_mode=crop ;; *_evalstretch) run_mode=stretch ;; esac
  [[ $serve == "$run_mode" ]] && return
  echo "REFUSE: VIEW_RESIZE=${VIEW_RESIZE:-unset} serves $CKPT_LABEL with $serve, but run $RUN is served $run_mode"; exit 2
}
pyfloat(){ "$PY" -c 'import sys; print(float(sys.argv[1]))' "$1"; }   # the adapters print floats with python's repr: 6 -> 6.0
expect_ready(){ # the ready-line tokens these settings imply, one per line
  echo "ckpt=$CKPT"; echo "steps=$NFE"
  case $ADAPTER in
    SANA_WAM|SANA_MOT)
      echo "cfg_scale=$(pyfloat "$VIDEO_CFG")"; echo "video_cfg_scale=$(pyfloat "$VIDEO_CFG")"; echo "action_cfg_scale=$(pyfloat "$ACTION_CFG")"
      echo "flow_shift=$(pyfloat "$VIDEO_FS")"; if [[ -n $ACTION_FS ]]; then echo "action_flow_shift=$(pyfloat "$ACTION_FS")"; else echo "action_flow_shift=shared"; fi
      echo "n_action_steps=${NAS:-all}"
      [[ $ADAPTER == SANA_MOT && -n $VIDEO_LAYOUT ]] && echo "video_layout=$VIDEO_LAYOUT" ;;
    SANA_WAM_CAUSAL|SANA_WAM_SM)
      echo "video_cfg_scale=$(pyfloat "$VIDEO_CFG")"; echo "action_cfg_scale=$(pyfloat "$ACTION_CFG")"
      echo "flow_shift=$(pyfloat "$VIDEO_FS")"; echo "action_flow_shift=$(pyfloat "${ACTION_FS:-$VIDEO_FS}")"
      echo "normalization_sha256=$NORM_SHA" ;;
  esac
  echo "action_type=$ACTION_TYPE"; echo "view_resize=$VR_EXPECT"
}
smoke_dir(){ echo "$SCRATCH/smoke1gpu_${SMOKE_UNIT_ID:-${1}__s${SMOKE_SEED:-$SEED}}"; }
smoke_pull(){ # TASK -> the smoke unit's a1/ (no videos) under smoke_dir
  local task=$1 uid dst
  uid=${SMOKE_UNIT_ID:-${task}__s${SMOKE_SEED:-$SEED}}; dst=$(smoke_dir "$task"); mkdir -p "$dst"
  [[ ${SMOKE_NO_PULL:-0} == 1 ]] && return 0
  rsync -az -e "ssh $CE_SSH_OPTS" --exclude 'upstream_eval_result/**/*.mp4' --exclude '*.mp4' "$GW:$SMOKE_RUN_ROOT/units/$uid/a1/" "$dst/" 2>/dev/null || true
}
smoke_verify(){ # TASK [EPS] -> 0 passed, 1 not finished, 2 failed
  local task=${1:-insert_key} eps=${2:-1} dst ready need missing=""
  smoke_pull "$task"; dst=$(smoke_dir "$task")
  if ! grep -q '"status": "complete"' "$dst/result.json" 2>/dev/null; then
    if grep -q '"status"' "$dst/result.json" 2>/dev/null; then echo "FAILED: $(tr -d '\n' <"$dst/result.json" | cut -c1-300)"; return 2; fi
    if grep -q "Traceback" "$dst/server/server.stderr.log" 2>/dev/null; then echo "FAILED: server traceback ($dst/server/server.stderr.log)"; return 2; fi
    echo "not finished ($dst)"; return 1
  fi
  ready=$(grep -h "\[$POLICY\] ready:" "$dst/server/server.stdout.log" 2>/dev/null | tail -n 1 || true)
  while IFS= read -r need; do
    [[ -z $need ]] && continue
    [[ " $ready " == *" $need "* ]] || missing+=" [$need]"
  done < <(expect_ready; tr '|' '\n' <<<"${EXPECT:-}")
  grep -q "\"episodes_recorded\": $eps\b" "$dst/result.json" || missing+=" [episodes_recorded $eps]"
  if [[ -n $NAS && $ADAPTER != SANA_WAM_CAUSAL ]]; then
    grep -q "n_action_steps=$NAS: executing the first $NAS of 32 predicted targets per inference" "$dst/server/server.stdout.log" 2>/dev/null \
      || missing+=" [server log: executing the first $NAS of 32]"
  fi
  if [[ -n $missing ]]; then echo "FAILED: ready line lacks$missing: ${ready:0:1500}"; return 2; fi
  echo "PASSED: $(tr -d '\n' <"$dst/result.json" | grep -o '"score_sum": [0-9.]*\|"successes": [0-9]*' | paste -sd' ') | ${ready:0:1500}"
}

cmd=${1:-help}; shift || true
case $cmd in
  name) echo "$RUN" ;;
  show)
    printf '%s\n' "RUN=$RUN" "ADAPTER=$ADAPTER POLICY=$POLICY CKPT=$CKPT NORM_SHA=${NORM_SHA:-<unset>} SEED=$SEED" \
      "trained view resize: $TRAINED_RESIZE; served: ${VIEW_RESIZE:-$TRAINED_RESIZE} (ready line: view_resize=$VR_EXPECT)" \
      "organize_table prompt: $ORGANIZE_TABLE_PROMPT" \
      "overrides: $OVERRIDES" "ports: slots $RPC_BASE/$WS_BASE + 50 + g, smoke $SMOKE_RPC_PORT/$SMOKE_WS_PORT${WORKER_ENV:+; worker env $WORKER_ENV}" \
      "master: ${COMMON[*]} ${EXTRA_MASTER_ARGS:-}" "state: $STATE; run root: $CE_EVAL_ROOT/$RUN" ;;
  expect-ready) expect_ready ;;
  check-sync) check_sync ;;
  free-spec) spec=$(ce_free_spec "$RPC_BASE" "$WS_BASE"); echo "$spec"; log "$(ce_spec_count "$spec") free GPUs" >&2 ;;
  smoke-gpu)
    spec=${1:?'NODE:GPU, e.g. node3:1'}; task=${2:-insert_key}; eps=${3:-1}
    node=${spec%%:*}; gpu=${spec##*:}
    [[ $gpu =~ ^[0-9]+$ ]] || { echo "GPU index must be an integer, got '$gpu'"; exit 2; }
    need_launch; check_sync; check_view_resize
    if ce_live_slots | grep -qx "$node:$gpu"; then echo "REFUSE: $node GPU $gpu belongs to a slot of a live master"; exit 3; fi
    [[ " $CE_RESERVE " == *" $node:$gpu "* ]] && { echo "REFUSE: $node:$gpu is in CE_RESERVE"; exit 3; }
    seed=${SMOKE_SEED:-$SEED}; [[ $seed =~ ^[0-9]+$ ]] || { echo "SMOKE_SEED must be an integer, got '$seed'"; exit 2; }
    uid=${SMOKE_UNIT_ID:-${task}__s$seed}; udir=$SMOKE_RUN_ROOT/units/$uid/a1
    layout_env=""; [[ -n ${SMOKE_LAYOUT_IDS:-} ]] && layout_env="SANA_EVAL_LAYOUT_IDS='$SMOKE_LAYOUT_IDS' SANA_EVAL_LAYOUT_TOTAL='${SMOKE_LAYOUT_TOTAL:-0}'"
    # $gpu is a CUDA index = the gpu-th HEALTHY board
    remote="set -u
healthy=(\$(nvidia-smi --query-gpu=index,pstate --format=csv,noheader | awk -F', ' '\$2 !~ /reset/ {print \$1}'))
phys=\${healthy[$gpu]:-}; [ -n \"\$phys\" ] || { echo \"REFUSE: \$(hostname -s) has no healthy board for CUDA device $gpu\"; exit 3; }
used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i \$phys | tr -d ' ')
[ \"\$used\" -lt 1024 ] || { echo \"REFUSE: CUDA $gpu (board \$phys) on \$(hostname -s) has \${used} MiB in use\"; exit 3; }
for p in $SMOKE_RPC_PORT $SMOKE_WS_PORT; do [ -z \"\$(ss -ltnH \"sport = :\$p\")\" ] || { echo \"REFUSE: port \$p is listening on \$(hostname -s)\"; exit 3; }; done
# the worker's runner lock (units/<uid>/.lock) is taken only AFTER this rm -rf: probe it first, or a relaunch wipes a1/ of a
# unit whose runner is still alive
mkdir -p \"\$(dirname '$udir')\"; exec 8>\"\$(dirname '$udir')/.lock\"; flock -n 8 || { echo \"REFUSE: unit $uid already has a live runner\"; exit 3; }; exec 8>&-
rm -rf '$udir'; mkdir -p '$udir' || { echo mkdir_failed; exit 1; }
$WORKER_ENV SANA_EVAL_CHECKPOINT_ROOT='$CKPT' SANA_EVAL_XPL_ROOT='$CE_REMOTE_XPL' \
SANA_EVAL_XPL_EXTRA_OVERRIDES='$OVERRIDES' SANA_EVAL_XPL_ACTION_TYPE=$ACTION_TYPE SEC_PER_STEP=4 $layout_env \
  nohup setsid bash '$CE_REMOTE_INSTALL/eval/run_unit_xpl.sh' '$SMOKE_RUN_ROOT' '$uid' '$task' $seed '$eps' $gpu $gpu $SMOKE_RPC_PORT $SMOKE_WS_PORT 1 \
  > '$udir/nohup.log' 2>&1 < /dev/null &
sleep 2; kill -0 \$! 2>/dev/null && echo \"dispatched \$! on \$(hostname -s) GPU $gpu (server+client), ports $SMOKE_RPC_PORT/$SMOKE_WS_PORT\" || echo died"
    ssh -n $CE_SSH_OPTS "$(ce_host "$node")" "$remote"
    printf '%s\n' "node=$node gpu=$gpu task=$task eps=$eps run_root=$SMOKE_RUN_ROOT unit=$uid overrides=$OVERRIDES started=$(date -u +%FT%TZ)" >"$SCRATCH/smoke1gpu_$task.txt"
    log "smoke-gpu launched; follow with: bash $0 smoke-status $task  (judge: smoke-verify $task)" ;;
  smoke-status)
    task=${1:-insert_key}; smoke_pull "$task"; dst=$(smoke_dir "$task")
    echo "== status.txt"; cat "$dst/status.txt" 2>/dev/null || echo "(none yet)"
    echo "== server log ([$POLICY] lines, errors)"
    grep -h "\[$POLICY\]\|Traceback\|Error\|error:" "$dst/server/server.stdout.log" "$dst/server/server.stderr.log" 2>/dev/null | cut -c1-400 | head -n 30 || true
    echo "== client progress: $(cat "$dst"/client/*/client.stdout.log 2>/dev/null | grep -a -o 'env0 step' | wc -l) simulation steps"
    echo "== result.json"; cat "$dst/result.json" 2>/dev/null || echo "(none yet)" ;;
  smoke-verify) smoke_verify "${1:-insert_key}" "${2:-1}" ;;
  start)
    need_launch; (( ${#DRY_MASTER[@]} )) || check_sync; check_view_resize
    spec=${1:-$(ce_free_spec "$RPC_BASE" "$WS_BASE")}; [[ -n $spec ]] || { echo "no free GPU"; exit 3; }
    if (( ! ${#DRY_MASTER[@]} )); then ce_check_gpus_free "$spec"; log "requested GPUs are free of live masters and empty on their nodes: $spec"; fi
    # EXTRA_MASTER_ARGS: e.g. "--tasks put_bottles_into_dustbin --max-attempts 1" (with RUN / SHARD_PLAN overrides)
    # shellcheck disable=SC2086
    bash "$EVAL/start_master.sh" "$RUN" "${COMMON[@]}" "${DRY_MASTER[@]}" ${EXTRA_MASTER_ARGS:-} --nodes "$(ce_nodes_of "$spec")" --node-pairs "$spec"
    log "started $RUN on $(ce_spec_count "$spec") GPUs: $spec" ;;
  resume)   # restart the stopped master of this run on its own slots (state.json keeps units, attempts, slots, checkpoint)
    [[ -f $STATE/state.json ]] || { echo "no master state for $RUN at $STATE"; exit 2; }
    ce_master_alive "$RUN" && { echo "the master of $RUN is alive (pid $(cat "$STATE/master.pid"))"; exit 1; }
    need_launch; (( ${#DRY_MASTER[@]} )) || check_sync; check_view_resize
    if [[ -n ${1:-} ]]; then spec=$1; else
      spec=$("$PY" - "$STATE/state.json" "$CE_NODES" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1])); order = sys.argv[2].split(); by = {}
for sl in s["slots"]:
    by.setdefault(sl["node"], []).append(f"s{sl['server_gpu']}" if sl["server_gpu"] == sl["client_gpu"] else str(sl["server_gpu"] // 2))
nodes = [n for n in order if n in by] + [n for n in by if n not in order]
print(";".join(f"{n}:{','.join(dict.fromkeys(by[n]))}" for n in nodes))
PYEOF
)
    fi
    # shellcheck disable=SC2086
    bash "$EVAL/start_master.sh" "$RUN" "${COMMON[@]}" "${DRY_MASTER[@]}" ${EXTRA_MASTER_ARGS:-} --resume --nodes "$(ce_nodes_of "$spec")" --node-pairs "$spec"
    log "resumed $RUN ($(ce_spec_count "$spec") slots)" ;;
  dry)
    [[ -n $CKPT ]] || { echo "set CKPT_LABEL"; exit 2; }
    if [[ -n ${1:-} ]]; then spec=$1; else   # default: GPUs 0-7 of every CE_NODES node minus CE_RESERVE (no ssh; the plan only)
      spec=$("$PY" - "$CE_NODES" "$CE_RESERVE" <<'PYEOF'
import sys
nodes, reserve = sys.argv[1].split(), set(sys.argv[2].split())
print(";".join(f"{n}:" + ",".join(f"s{g}" for g in range(8) if f"{n}:{g}" not in reserve) for n in nodes))
PYEOF
)
    fi
    st=$SCRATCH/dry_state; rm -rf "$st"; mkdir -p "$st"
    # shellcheck disable=SC2086
    "$PY" "$EVAL/master.py" --run-root "$CE_EVAL_ROOT/${RUN}_dry" --checkout "$CE_ROBODOJO_CHECKOUT" --state-dir "$st" \
         --dry-run --dry-sleep 0.05 --poll 1 "${COMMON[@]}" ${EXTRA_MASTER_ARGS:-} --nodes "$(ce_nodes_of "$spec")" --node-pairs "$spec" >"$st/stdout.log" 2>&1 || true
    grep -h "planned\|dispatch order\|ALL UNITS SETTLED" "$st/master.log" | cut -c1-260 | head -6
    grep -h "Traceback\|Error\|error:" "$st/stdout.log" | head -5 || true
    "$PY" - "$st/state.json" <<'PYEOF'
import collections, json, sys
s = json.load(open(sys.argv[1]))
print("slots:", len(s["slots"]), [(x["slot_id"], x["server_gpu"], x["client_gpu"], x["rpc_port"], x["ws_port"]) for x in s["slots"][:4]], "...")
by = collections.Counter(u["state"] for u in s["units"])
print("units", len(s["units"]), "task configs", len({(u["task"], u["eval_seed"]) for u in s["units"]}),
      "episodes", sum(u["episodes_per_task"] for u in s["units"]), "states", dict(by))
print("dimension order of dispatch:", list(dict.fromkeys(u["dimension"] for u in sorted(s["units"], key=lambda u: (u.get("history") or [{}])[0].get("dispatched") or ""))))
PYEOF
    ;;
  report)
    echo "== $RUN  $(date -u +%FT%TZ)"; grep -m1 "^units:" "$STATE/STATUS.md" || true
    "$PY" - "$STATE/state.json" "$SEED" <<'PYEOF'
import collections, json, sys, time
s = json.load(open(sys.argv[1])); now = time.time(); tag = f"__s{sys.argv[2]}"
by = collections.defaultdict(collections.Counter)
for u in s["units"]: by[u["dimension"]][u["state"]] += 1
for d in ("precision", "memory", "long-horizon", "generalization", "open"):
    print(f"{d:15s} done {by[d]['done']:2d} running {by[d]['running']:2d} queued {by[d]['queued']:2d} failed {by[d]['failed']:2d}")
run = []
for u in s["units"]:
    if u["state"] == "running":
        sl = next(x for x in s["slots"] if x["slot_id"] == u["slot"]); run.append(f"{u['unit_id'].replace(tag, '')}@{u['slot']}({int((now - sl['dispatched_at']) / 60)}m)")
print("slots:", len(s["slots"]))
print("running:", " ".join(run))
PYEOF
    if [[ ${1:-} != --no-progress ]]; then XPL_POLICY=$POLICY "$PY" "$EVAL/progress_xpl.py" "$STATE" 2>/dev/null | grep "running" | awk '{printf "  %-40s %s eps, succ %s, score_sum %s\n",$1,$3,$4,$5}' || true; fi
    echo "== last completions"; grep -h "DONE " "$STATE/master.log" | tail -n 8 | cut -c1-140 || true ;;
  compare)
    [[ -n ${BASELINES:-} ]] || { echo "set BASELINES='<run> <run> ...' (master state dirs under CE_MASTER_ROOT)"; exit 2; }
    ts=$(date -u +%Y%m%dT%H%M%S); out=$CE_RESULTS_ROOT/per_task_${RUN}_vs_baselines_$ts.md; mkdir -p "$CE_RESULTS_ROOT"
    # shellcheck disable=SC2086
    if ! "$PY" "$EVAL/compare_runs.py" "$RUN" $BASELINES --results-root "$CE_MASTER_ROOT" --runs-subdir . --out "$out" >"$SCRATCH/compare.stdout" 2>&1; then
      tail -n 5 "$SCRATCH/compare.stdout"; exit 1
    fi
    echo "columns: this run | $BASELINES   (SR% / score%)"
    grep -h "tasks compared" "$SCRATCH/compare.stdout" | cut -c1-120 || true
    grep "^| [A-Za-z_]* | [A-Za-z-]* | [0-9]" "$out" | cut -c1-200 || true
    grep "^| Precision\|^| Memory\|^| Open\|^| Long-Horizon\|^| Generalization\|overall" "$out" | cut -c1-200 || true
    echo "-> $out" ;;
  tail)
    sub=${1:?plan|launch|status|more-substitutes|assemble|invalidate|fill|fill-status}; shift
    targs=(); topts=()
    while [[ $# -gt 0 ]]; do case $1 in --dry-run|--force) topts+=("$1"); shift ;; --*) topts+=("$1" "$2"); shift 2 ;; *) targs+=(--task "$1"); shift ;; esac; done
    [[ ${#targs[@]} -gt 0 ]] || targs=(--task play_tic_tac_toe --task imitate_sorting_sequence)
    # the shards are started through `bash run.sh smoke-gpu`, which inherits this environment (ADAPTER, CKPT_LABEL, ...)
    export ADAPTER CKPT_LABEL CKPT NORM_SHA SEED NFE VIDEO_CFG ACTION_CFG VIDEO_FS ACTION_FS NAS ACTION_TYPE VIDEO_LAYOUT VIEW_RESIZE ORGANIZE_TABLE_PROMPT
    export EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}
    "$PY" "$EVAL/tail_shards.py" --run "$RUN" --driver "$EVAL/run.sh" --policy "$POLICY" "${targs[@]}" "${topts[@]}" "$sub" ;;
  status) sed -n 1,40p "$STATE/STATUS.md"; echo ...; tail -n 6 "$STATE/master.log" ;;
  watch) tail -n 0 -F "$STATE/master.log" | grep -E --line-buffered "DONE |FAIL|failed [1-9]|quarantin|ALL UNITS SETTLED|Traceback|WARN slot|stale|server_timeout|server_fail|prologue|control: |requeue" ;;
  server-log)
    uid=${1:?unit id}; run=${2:-$RUN}; a=${3:-1}; dst=$SCRATCH/serverlog_${run}_${uid}_a$a; mkdir -p "$dst"
    rsync -az -e "ssh $CE_SSH_OPTS" "$GW:$CE_EVAL_ROOT/$run/units/$uid/a$a/server/" "$dst/"
    grep -h "\[$POLICY\]\|WARNING\|Traceback\|Error\|error" "$dst"/server.stdout.log "$dst"/server.stderr.log 2>/dev/null | head -n 30 ;;
  official-summary)   # the official summarize_result.py aggregation (consolidates first) -> <CE_RESULTS_ROOT>/runs/<RUN>/OFFICIAL_SUMMARY_<ts>.md
    "$PY" "$EVAL/consolidate_results.py" >/dev/null 2>&1 || true
    # SUMMARY_RUNS="<run> <run> ..." lays several runs of ONE checkpoint into one tree (the seeds of a campaign, or a main run plus
    # a back-fill) so the official table covers all of them; default: this RUN alone
    runs=${SUMMARY_RUNS:-$RUN}; out=$SCRATCH/official_$RUN; rm -rf "$out"
    "$PY" - "${runs// /,}" "$POLICY" "$out" "$CE_RESULTS_ROOT" <<'PYEOF3'
import collections, json, os, re, sys
runs, policy, out, results = [r for r in sys.argv[1].split(",") if r], sys.argv[2], sys.argv[3], sys.argv[4]
for run in runs:
    rd = f"{results}/runs/{run}"
    st = json.load(open(f"{rd}/state.json")); label = os.path.basename(st["checkpoint_root"].rstrip("/"))
    action_type = st.get("xpl_action_type") or "ee"
    # summarize_result.py reads ONE timestamp folder per (task, policy, embodiment, seed) -- the latest -- and counts the
    # first N detail entries by KEY (the episode index; a layout the client abandoned is replaced by a later layout id, so
    # layout ids need not be 0..N-1). The layout shards of one task config are therefore concatenated in range order into
    # ONE _result.json re-keyed 0..N-1 -- for an unsharded unit exactly its own file.
    parts = collections.defaultdict(list)
    for u in st["units"]:
        parts[(u["task"], u["eval_seed"])].append(u)
    laid, waiting, bad, notes = 0, [], [], []
    for (task, seed), us in sorted(parts.items()):
        if any(u["state"] != "done" for u in us):
            waiting.append(f"{task} ({sum(u['state'] == 'done' for u in us)}/{len(us)} parts)"); continue
        us = sorted(us, key=lambda u: min(u.get("layout_ids") or [0]))
        total = int(us[0].get("layout_total") or us[0]["episodes_per_task"])
        episodes, rids, ok = [], [], True
        for u in us:
            a = f"{rd}/per_unit/{u['unit_id']}/a{u['result']['attempt']}"
            if not os.path.isfile(f"{a}/_result.json"):
                bad.append(f"{u['unit_id']}: no _result.json"); ok = False; break
            m = re.search(r"run_id=(\S+)", open(f"{a}/status.txt").read()); rids.append(m.group(1) if m else f"a{u['result']['attempt']}")
            details = json.load(open(f"{a}/_result.json")).get("details") or {}
            rows = sorted(((int(k), v) for k, v in details.items() if str(k).isdigit()), key=lambda r: r[0])
            need = int(u["episodes_per_task"])
            if len(rows) < need:
                bad.append(f"{u['unit_id']}: {len(rows)}/{need} episodes"); ok = False; break
            allowed = set(int(x) for x in u["layout_ids"]) if u.get("layout_ids") else set(range(total))
            for _, v in rows[:need]:
                lid = int(v.get("layout_id", -1))
                if lid not in allowed:
                    notes.append(f"{u['unit_id']}: layout {lid} replaced an abandoned one")
                episodes.append({"layout_id": lid, "success": bool(v.get("success", False)), "score": float(v.get("score", 0.0) or 0.0)})
        if not ok:
            continue
        if len(episodes) != total:
            bad.append(f"{task}: {len(episodes)}/{total} episodes"); continue
        n = len(episodes); succ = sum(e["success"] for e in episodes)
        doc = {"success_rate": 100.0 * succ / n, "eval_time": n, "score": 100.0 * sum(e["score"] for e in episodes) / n,
               "details": {str(i): e for i, e in enumerate(episodes)}, "merged_from": rids}
        d = f"{out}/eval_result/RoboDojo/{task}/{policy}/arx_x5/{seed}_ckpt_name={label},action_type={action_type}/{max(rids)}"
        os.makedirs(d, exist_ok=True); json.dump(doc, open(f"{d}/_result.json", "w"), indent=1); laid += 1
    print(f"{laid} task configs laid out (layout shards concatenated)" + (f"; still running: {', '.join(waiting)}" if waiting else "")
          + (f"; PROBLEMS: {bad}" if bad else "") + (f"; notes: {len(notes)} substituted layouts" if notes else ""))
PYEOF3
    ROBODOJO_EVAL_ROOT=$out/eval_result/RoboDojo "$PY" "$CE_ROBODOJO_CHECKOUT/scripts/internal/summarize_result.py" 2>&1 | { grep -v '^[^|].*\(MISMATCH\|mp4\|video\)' || true; } | tail -2
    ts=$(date -u +%Y%m%dT%H%M%S); mkdir -p "$CE_RESULTS_ROOT/runs/$RUN"
    name=OFFICIAL_SUMMARY_$ts.md; [[ $runs == *" "* ]] && name=OFFICIAL_SUMMARY_combined_$ts.md
    { [[ $runs == *" "* ]] && printf '<!-- runs laid out together: %s -->\n' "$runs"; cat "$out/eval_result/RoboDojo/_summary.md"; } \
      >"$CE_RESULTS_ROOT/runs/$RUN/$name"
    echo "-> $CE_RESULTS_ROOT/runs/$RUN/$name" ;;
  *) sed -n 2,49p "$0"; exit 1 ;;
esac
