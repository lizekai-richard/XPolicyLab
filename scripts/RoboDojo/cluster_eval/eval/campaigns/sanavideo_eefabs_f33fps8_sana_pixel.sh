#!/usr/bin/env bash
# Campaign driver (runs on x5): the SINGLE-STREAM Sana policy, checkpoint `sanavideo_eefabs_f33fps8_sana_pixel` (HF
# logits/sft_robodojo_sanavideo_eefabs_f33fps8_sana_pixel, root weights, 17.87 GB, epoch 10 / step 36,180 = training final;
# snapshot ~/checkpoints/huggingface/logits/sanavideo_eefabs_f33fps8_sana_pixel) on the XPolicyLab lane through policy/SANA_WAM.
# The FIRST checkpoint of the 2026-09-20..23 contracts (rwm/zekai-merge 167d45af1): Sana-Video donor, frame-aligned data
# (needs RoboDojo >= 6e3190f; the 726e9aa pin has it), data.extra.multiview sana_pixel = ONE 320x480 2x2 pixel canvas
# (10x15 latent, V = 1), 33-row window at video_fps 8 -> stride 4, model.extra.rope independent, G = 1 tiling prompt,
# EEF-ONLY robot_base_eef with ABSOLUTE targets (no joint slot fed or predicted), normalization 1fe3b7e7 (forced scheme).
# EXECUTED AS EE POSES (--xpl-action-type ee): the adapter refuses joint execution for an EEF-only checkpoint.
# Sampling identical to every other f33fps8 run (CFG 6 @ 10 NFE, flow_shift 8).
#
#   bash sanavideo_eefabs_f33fps8_sana_pixel.sh check-sync           worker scripts + SANA_WAM on NFS == x5
#   bash sanavideo_eefabs_f33fps8_sana_pixel.sh free-pairs           usable GPU pairs no live master can dispatch onto
#   bash sanavideo_eefabs_f33fps8_sana_pixel.sh smoke-gpu 'l3:1' [TASK] [EPS]
#                                                    direct worker on ONE free GPU (policy server + Isaac client share it);
#                                                    refuses a GPU any live master slot covers, a GPU with >1 GiB in use,
#                                                    or a listening port; run <RUN>_smoke1gpu, unit <TASK>__s0
#   bash sanavideo_eefabs_f33fps8_sana_pixel.sh smoke-status [TASK]  status / log tail / result of that direct smoke
#   bash sanavideo_eefabs_f33fps8_sana_pixel.sh smoke 'l5:1'         1 episode of insert_key on a free PAIR through the master -> <RUN>_smoke
#   bash sanavideo_eefabs_f33fps8_sana_pixel.sh dry ['<spec>']       local --dry-run of the plan on that slot spec (default 'l3:s1;l26:s0')
#   bash sanavideo_eefabs_f33fps8_sana_pixel.sh start ['<spec>']     full run (category order precision, memory, long-horizon, generalization, open;
#                                                    SHARD_PLAN layout shards); default spec = every free GPU as a single-GPU slot
#                                                    (SINGLE_SPEC); pairs 'l5:0,1' and single GPUs 'l3:s1' also accepted
#   bash sanavideo_eefabs_f33fps8_sana_pixel.sh autogrow start|stop|status
#                                                    background loop (every 5 min): every USABLE pair no live master holds
#                                                    (no unit, not draining, its master has nothing dispatchable) is appended
#                                                    to this run through control/add_slots; stops when this run's master ends
#   bash sanavideo_eefabs_f33fps8_sana_pixel.sh compare               per-task + dimension means over the tasks this run has finished, vs the
#                                                    sanavideo f33fps8 joint run, eefabs_nsc_s30k (ee) and the MoT canvas run
#   bash sanavideo_eefabs_f33fps8_sana_pixel.sh tail plan|launch|status|more-substitutes|assemble [TASK ...]
#                                                    the TAIL TEMPLATE: when the queue is empty, finish the unsharded tail units
#                                                    (default play_tic_tac_toe + imitate_sorting_sequence) as one-layout shards
#                                                    + substitutes on idle GPUs, then assemble them into the run (eval/tail_shards.py)
#   bash sanavideo_eefabs_f33fps8_sana_pixel.sh status | report | watch | server-log UNIT [RUN] [ATTEMPT]
#   ACTION_CFG=6 bash ...                            historical single-knob CFG (default ACTION_CFG=1: video-only guidance)
set -euo pipefail
CKPT_LABEL=${CKPT_LABEL:-sanavideo_eefabs_f33fps8_sana_pixel}
# _gripfix: the run of 2026-09-24 04:38 (official_seed0_..._ee, x5 state renamed .gripperbug_20260924T0635) served the
# model with the normalized gripper clamped to [0, 1] (opening <= 0.5) and is invalid; this name gets a fresh run root
# action-stream CFG: DEFAULT 1 (user rule 2026-09-25, also the adapters' deploy.yml default) = video-only guidance;
# ACTION_CFG=6 (or the old VIDEO_ONLY_CFG=0) reproduces the historical single knob of the runs before 2026-09-24
ACTION_CFG=${ACTION_CFG:-1}; [[ ${VIDEO_ONLY_CFG:-} == 0 ]] && ACTION_CFG=6
RUN=${RUN:-official_seed0_${CKPT_LABEL}_xpl_vcfg6_acfg${ACTION_CFG}_nfe10_fs8_ee_gripfix}
CKPT=/homes/jiamingt/checkpoints/huggingface/logits/$CKPT_LABEL
# The artifact this snapshot's own training yaml pins (grep normalization_sha256 <snapshot>/config.yaml after any re-download).
NORM_SHA=${NORM_SHA:-1fe3b7e72e8fa93eca5efb8a8deeace1daf4434b4b97e9f4445670ac758287ed}
OVERRIDES="sampling_steps=10 cfg_scale=6 action_cfg_scale=$ACTION_CFG flow_shift=8 normalization_sha256=$NORM_SHA"
[[ -n ${EXTRA_OVERRIDES:-} ]] && OVERRIDES="$OVERRIDES $EXTRA_OVERRIDES"   # e.g. trajectory_dump_dir=<NFS dir> (diagnostics)
# Layout shards (master --shard-plan, XPL lane): every task config expected to take > 3.2 h as ONE unit is split into
# contiguous layout ranges run as separate units, so 46 slots are not left waiting on a 7-12 h fasten_screws tail.
# Counts = ceil(hours / 2.5) from the slowest comparable run, the ee run eefabs_nsc_s30k (176 unit-hours): 51 configs
# -> 92 units; a list-scheduling simulation on the dry-run order gives ~4.6 h on 46 slots in category order (3.2 h
# shards: 5.6 h; 2.0 h shards: 4.9 h; unsharded: the 11.6 h fasten_screws unit). Results merge by layout id
# (summarize_eval.py, compare_runs.py; official-summary concatenates the parts per task config).
# Tasks whose layouts the official client abandons (seed 0: deposit_coin 35,47 -> 51,52; fill_egg_holder 28,35,37,46 ->
# 50-53; general_pickup 29 -> 50) stay UNSHARDED: a shard marks every layout outside its range abandoned, so it cannot
# draw the substitutes and never completes (fill_egg_holder p2of2, 2026-09-24: 3 x 21/25). play_tic_tac_toe and
# imitate_sorting_sequence abandon layouts in EVERY run, but WHICH ones depends on the policy (back-fills: [39], [], [20, 43],
# [2, 11, ..]) -> they stay unsharded too (play_tic_tac_toe is then the run's tail, 4-8 h); make_kong never abandoned any.
# COMPLETE PROTOCOL (user rule 2026-09-24): --support-arm-embodiments on = 42 tasks / 54 configs / 2100 episodes per seed.
SHARD_PLAN=${SHARD_PLAN-make_kong=2,fasten_screws=5,pour_by_language=4,play_stacking_toy=3,fill_pen_holder=3,classify_objects_by_language=3,organize_table=3,classify_objects=3,build_tower=3,store_tools_in_toolbox=3,make_toast_random=2,cover_blocks=2,make_toast=2,swap_blocks=2,pick_from_conveyor_by_image=2,match_and_pick_from_conveyor=2,pack_objects_into_box=2,pack_objects_into_box_random=2,pour_balls_into_vase=2,arrange_largest_number_random=2,sweep_blocks_random=2,arrange_largest_number=2,sort_nesting_dolls_by_size=2,insert_tubes=2,put_bottles_into_dustbin=2,sweep_blocks=2,press_by_number=2,sort_nesting_dolls_by_size_random=2,play_Xylophone=2}
DIM_ORDER=${DIM_ORDER:-precision,memory,long-horizon,generalization,open}   # master --dimension-order
PIN_LONGEST=${PIN_LONGEST:-2}     # the longest units dispatched first regardless of category (master --pin-longest)
# ONE GPU per slot (policy server + Isaac client on the same board; user 2026-09-24): every free GPU of the six 726e9aa
# nodes -- l3 GPU0 and l26 GPU1 host another user's vLLM.
SINGLE_SPEC="l5:s0,s1,s2,s3,s4,s5,s6,s7;l7:s0,s1,s2,s3,s4,s5,s6,s7;l8:s0,s1,s2,s3,s4,s5,s6,s7;l25:s0,s1,s2,s3,s4,s5,s6,s7;l3:s1,s2,s3,s4,s5,s6,s7;l26:s0,s2,s3,s4,s5,s6,s7"
# Pair k of a node uses rpc base + 10k: 62300..62330 / 64300..64330, disjoint from every other campaign's range
# (61000..61900, 62400.., 62500.., 62800.., 63000.., 63500.., 64000.., 64500.., 65000..). The direct single-GPU smoke uses 62390 / 64390.
# Every bound port must be <= 65535 (a worker binds rpc_port_base + 10 * slot).
PORT_ARGS=(--rpc-port-base 62300 --ws-port-base 64300)
SMOKE_RPC_PORT=${SMOKE_RPC_PORT:-62390}; SMOKE_WS_PORT=${SMOKE_WS_PORT:-64390}
XPL_ROOT=/homes/jiamingt/xpl_zekai/XPolicyLab
POLICY=SANA_WAM          # adapter dir segment of the official eval_result layout
USABLE="l5:0,1,2,3;l7:0,1,2,3;l8:0,1,2,3;l25:0,1,2,3;l3:1,2,3;l26:1,2,3"
NFS_USER_HOST=jiamingt@l5.mit.edu
EVAL=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STATE=$HOME/robodojo_master/$RUN
SCRATCH=${SCRATCH:-$HOME/robodojo_master/.scratch/$RUN}; mkdir -p "$SCRATCH"
PY=${MASTER_PY:-$HOME/miniforge3/envs/sana-wam/bin/python}
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
nodes_of(){ echo "$1" | tr ';' '\n' | cut -d: -f1 | sed '/^$/d' | paste -sd, -; }

check_sync(){
  local diff
  diff=$(rsync -n -c -i "$EVAL/run_unit_xpl.sh" "$EVAL/run_unit.sh" "$NFS_USER_HOST:/homes/jiamingt/robodojo_install/eval/" | grep -v '^\.' || true)
  for pol in SANA_WAM; do
    diff+=$(rsync -n -c -i -r --exclude __pycache__ --exclude .pytest_cache "$HOME/zekail/XPolicyLab/policy/$pol/" "$NFS_USER_HOST:$XPL_ROOT/policy/$pol/" | grep -v '^\.d\|^cd\|^\.f' || true)
  done
  if [[ -n $diff ]]; then echo "NFS copies differ from x5 (rsync them first):"; echo "$diff"; exit 1; fi
  log "worker scripts and the SANA_WAM adapter are in sync with NFS"
}
live_slots(){ # every GPU a live master may use, as node:gpu (server and client of each slot it holds or can dispatch onto)
  "$PY" - "$HOME/robodojo_master" <<'PYEOF'
import json, os, pathlib, sys
root = pathlib.Path(sys.argv[1]); out = set()
for pidf in root.glob("*/master.pid"):
    try: pid = int(pidf.read_text().strip()); os.kill(pid, 0)
    except Exception: continue
    st = pidf.with_name("state.json")
    if not st.exists(): continue
    for sl in json.loads(st.read_text())["slots"]:
        out.update({f'{sl["node"]}:{sl["server_gpu"]}', f'{sl["node"]}:{sl["client_gpu"]}'})
print("\n".join(sorted(out)))
PYEOF
}
busy_pairs(){
  "$PY" - "$HOME/robodojo_master" <<'PYEOF'
import json, os, sys, pathlib
root = pathlib.Path(sys.argv[1]); busy = set()
for pidf in root.glob("*/master.pid"):
    try: pid = int(pidf.read_text().strip())
    except Exception: continue
    try: os.kill(pid, 0)
    except OSError: continue
    st = pidf.with_name("state.json")
    if not st.exists(): continue
    s = json.loads(st.read_text()); cdir = pidf.parent / "control"
    paused_all = (cdir / "pause").exists()
    paused_labels = {f.name.split(".", 1)[1] for f in cdir.glob("pause.*") if f.name != "pause"} if cdir.is_dir() else set()
    dispatchable = sum(1 for u in s["units"] if u["state"] == "queued" and not paused_all and u.get("setting") not in paused_labels)
    for sl in s["slots"]:
        if sl.get("unit") or sl.get("draining") or dispatchable:
            busy.add(f'{sl["node"]}:{sl["server_gpu"] // 2}')
print("\n".join(sorted(busy)))
PYEOF
}
free_pairs(){
  local busy; busy=$(busy_pairs)
  "$PY" - "$USABLE" "$busy" <<'PYEOF'
import sys
usable, busy = sys.argv[1], set(sys.argv[2].split()); out = []
for item in usable.split(";"):
    node, ks = item.split(":"); free = [k for k in ks.split(",") if f"{node}:{k}" not in busy]
    if free: out.append(f"{node}:{','.join(free)}")
print(";".join(out))
PYEOF
}
busy_gpus(){ # node:gpu of every GPU a live master holds (a slot with a unit, draining, or while it has dispatchable work)
  "$PY" - "$HOME/robodojo_master" <<'PYEOF'
import json, os, sys, pathlib
root = pathlib.Path(sys.argv[1]); busy = set()
for pidf in root.glob("*/master.pid"):
    try: pid = int(pidf.read_text().strip()); os.kill(pid, 0)
    except Exception: continue
    st = pidf.with_name("state.json")
    if not st.exists(): continue
    s = json.loads(st.read_text()); cdir = pidf.parent / "control"
    paused_all = (cdir / "pause").exists()
    paused_labels = {f.name.split(".", 1)[1] for f in cdir.glob("pause.*") if f.name != "pause"} if cdir.is_dir() else set()
    dispatchable = sum(1 for u in s["units"] if u["state"] == "queued" and not paused_all and u.get("setting") not in paused_labels)
    for sl in s["slots"]:
        if sl.get("unit") or sl.get("draining") or dispatchable:
            busy.update({f'{sl["node"]}:{sl["server_gpu"]}', f'{sl["node"]}:{sl["client_gpu"]}'})
print("\n".join(sorted(busy)))
PYEOF
}
spec_gpus(){ # 'l3:s1;l5:0' -> lines 'node gpu' (a pair k covers GPUs 2k and 2k+1)
  "$PY" - "$1" <<'PYEOF'
import sys
for item in sys.argv[1].split(";"):
    item = item.strip()
    if not item: continue
    node, ks = item.split(":")
    for k in ks.split(","):
        k = k.strip()
        gpus = [int(k[1:])] if k[:1] in "sS" else [2 * int(k), 2 * int(k) + 1]
        for g in gpus: print(node, g)
PYEOF
}
check_pairs_free(){ # refuse GPUs a live master holds, then check on each node that the GPUs are healthy and empty
  local spec=$1 busy node g bad="" nodes
  busy=$(busy_gpus)
  while read -r node g; do
    if grep -qx "$node:$g" <<<"$busy"; then bad+=" $node:$g"; fi
  done < <(spec_gpus "$spec")
  [[ -z $bad ]] || { echo "REFUSE: GPUs held by a live master (occupied, draining, or reachable by its queue):$bad"; exit 3; }
  nodes=$(spec_gpus "$spec" | awk '{print $1}' | sort -u)
  for node in $nodes; do
    gl=$(spec_gpus "$spec" | awk -v n="$node" '$1 == n {printf "%s ", $2}')
    # CUDA numbers only the healthy GPUs (a '[GPU requires reset]' board is skipped), so CUDA device g is the g-th
    # healthy board in nvidia-smi (PCI) order; check the memory of THAT board.
    ssh -n -o BatchMode=yes -o ConnectTimeout=20 "jiamingt@$node.mit.edu" "
      healthy=(\$(nvidia-smi --query-gpu=index,pstate --format=csv,noheader | awk -F', ' '\$2 !~ /reset/ {print \$1}'))
      rc=0
      for g in $gl; do
        phys=\${healthy[\$g]:-}
        if [ -z \"\$phys\" ]; then echo \"REFUSE: $node has no healthy board for CUDA device \$g\"; rc=3; continue; fi
        used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i \$phys | tr -d ' ')
        if [ \"\$used\" -ge 1024 ]; then echo \"REFUSE: $node CUDA \$g (board \$phys) has \${used} MiB in use\"; rc=3; fi
      done
      exit \$rc" || exit 3
  done
  log "requested GPUs are free of live masters and empty on their nodes: $spec"
}
SHARD_ARGS=(); [[ -n $SHARD_PLAN ]] && SHARD_ARGS=(--shard-plan "$SHARD_PLAN")
COMMON=(--worker run_unit_xpl.sh --xpl-root "$XPL_ROOT" --xpl-action-type ee --sec-per-step 4 --checkpoint-root "$CKPT"
        --xpl-overrides "$OVERRIDES" --order dimension --pin-longest "$PIN_LONGEST" "${SHARD_ARGS[@]}"
        --support-arm-embodiments on
        --dimension-order "$DIM_ORDER" "${PORT_ARGS[@]}")
SMOKE_RUN_ROOT=${SMOKE_RUN_ROOT:-/homes/jiamingt/robodojo_eval/${RUN}_smoke1gpu}
# smoke-gpu extras: SMOKE_UNIT_ID (default <task>__s0), SMOKE_LAYOUT_IDS='50,51,52,53' (+ SMOKE_LAYOUT_TOTAL) = evaluate exactly
# those layouts (run_unit_xpl.sh layout shard), e.g. the substitutes the official client draws for abandoned layouts
# Baselines: the same Sana-Video donor + f33fps8 on the 3-view strip with absolute JOINT targets (jointabs_f33fps8_nsc),
# the robot_base_eef absolute line executed as ee poses (eefabs_nsc_s30k, f25 3-view strip), and the MoT canvas run
# (mot_canvas_nsc_s36k, the most successes on record).
BASELINES=(official_seed0_jointabs_f33fps8_nsc_xpl_cfg6_nfe10_fs8 official_seed0_eefabs_nsc_s30k_xpl_cfg6_nfe10_fs8_ee@fs8
           official_seed0_mot_canvas_nsc_s36k_xpl_cfg6_nfe10_fs8)

cmd=${1:-help}; shift || true
case $cmd in
  check-sync) check_sync ;;
  free-pairs) free_pairs ;;
  smoke-gpu)
    spec=${1:?'NODE:GPU, e.g. l3:1'}; task=${2:-insert_key}; eps=${3:-1}
    node=${spec%%:*}; gpu=${spec##*:}
    [[ $gpu =~ ^[0-7]$ ]] || { echo "GPU index must be 0..7, got '$gpu'"; exit 2; }
    check_sync
    if live_slots | grep -qx "$node:$gpu"; then echo "REFUSE: $node GPU $gpu belongs to a slot of a live master"; exit 3; fi
    uid=${SMOKE_UNIT_ID:-${task}__s0}; udir=$SMOKE_RUN_ROOT/units/$uid/a1
    layout_env=""; [[ -n ${SMOKE_LAYOUT_IDS:-} ]] && layout_env="SANA_EVAL_LAYOUT_IDS='$SMOKE_LAYOUT_IDS' SANA_EVAL_LAYOUT_TOTAL='${SMOKE_LAYOUT_TOTAL:-0}'"
    # $gpu is a CUDA index = the gpu-th HEALTHY board (a '[GPU requires reset]' board is not enumerated; l2/l4/l6)
    remote="set -u
healthy=(\$(nvidia-smi --query-gpu=index,pstate --format=csv,noheader | awk -F', ' '\$2 !~ /reset/ {print \$1}'))
phys=\${healthy[$gpu]:-}; [ -n \"\$phys\" ] || { echo \"REFUSE: \$(hostname -s) has no healthy board for CUDA device $gpu\"; exit 3; }
used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i \$phys | tr -d ' ')
[ \"\$used\" -lt 1024 ] || { echo \"REFUSE: CUDA $gpu (board \$phys) on \$(hostname -s) has \${used} MiB in use\"; exit 3; }
for p in $SMOKE_RPC_PORT $SMOKE_WS_PORT; do [ -z \"\$(ss -ltnH \"sport = :\$p\")\" ] || { echo \"REFUSE: port \$p is listening on \$(hostname -s)\"; exit 3; }; done
rm -rf '$udir'; mkdir -p '$udir' || { echo mkdir_failed; exit 1; }
SANA_EVAL_CHECKPOINT_ROOT='$CKPT' SANA_EVAL_XPL_ROOT='$XPL_ROOT' \
SANA_EVAL_XPL_EXTRA_OVERRIDES='$OVERRIDES' SANA_EVAL_XPL_ACTION_TYPE=ee SEC_PER_STEP=4 $layout_env \
  nohup setsid bash ~/robodojo_install/eval/run_unit_xpl.sh '$SMOKE_RUN_ROOT' '$uid' '$task' 0 '$eps' $gpu $gpu $SMOKE_RPC_PORT $SMOKE_WS_PORT 1 \
  > '$udir/nohup.log' 2>&1 < /dev/null &
sleep 2; kill -0 \$! 2>/dev/null && echo \"dispatched \$! on \$(hostname -s) GPU $gpu (server+client), ports $SMOKE_RPC_PORT/$SMOKE_WS_PORT\" || echo died"
    ssh -n -o BatchMode=yes -o ConnectTimeout=20 "jiamingt@$node.mit.edu" "$remote"
    printf '%s\n' "node=$node gpu=$gpu task=$task eps=$eps run_root=$SMOKE_RUN_ROOT unit=$uid overrides=$OVERRIDES started=$(date -u +%FT%TZ)" >"$SCRATCH/smoke1gpu_$task.txt"
    log "smoke-gpu launched; follow with: bash $0 smoke-status $task" ;;
  smoke-status)
    task=${1:-insert_key}; uid=${task}__s0; dst=$SCRATCH/smoke1gpu_$uid; mkdir -p "$dst"
    rsync -az --exclude 'upstream_eval_result/**/*.mp4' "$NFS_USER_HOST:$SMOKE_RUN_ROOT/units/$uid/a1/" "$dst/" 2>/dev/null || true
    echo "== status.txt"; cat "$dst/status.txt" 2>/dev/null || echo "(none yet)"
    echo "== server log ([SANA_WAM] lines, errors)"
    grep -h "\[SANA_WAM\]\|Traceback\|Error\|error:" "$dst/server/server.stdout.log" "$dst/server/server.stderr.log" 2>/dev/null | cut -c1-400 | head -n 30 || true
    echo "== client progress: $(cat "$dst"/client/*/client.stdout.log 2>/dev/null | grep -a -o 'env0 step' | wc -l) simulation steps"
    echo "== result.json"; cat "$dst/result.json" 2>/dev/null || echo "(none yet)" ;;
  smoke)
    spec=${1:?pairs spec, e.g. 'l5:1'}; check_sync; check_pairs_free "$spec"
    bash "$EVAL/start_master.sh" "${RUN}_smoke" "${COMMON[@]}" --shard-plan "" --nodes "$(nodes_of "$spec")" --node-pairs "$spec" --tasks insert_key --episodes-per-task 1 --pin-longest 0 --poll 20
    echo "expect in the server log: '[SANA_WAM] ready: ... steps=10 cfg_scale=6.0 ... flow_shift=8.0 ... action_type=ee ... visual_layout=sana_pixel_canvas"
    echo "  ... video_fps=8 video_frame_stride=4 ... rope=independent text_groups=1 canvas_prompt=tiling robot_base_eef_layout=eef_only' -> bash $0 server-log insert_key__s0 ${RUN}_smoke" ;;
  start)
    spec=${1:-$SINGLE_SPEC}; check_sync; check_pairs_free "$spec"
    # EXTRA_MASTER_ARGS: e.g. "--seeds 1 --tasks put_bottles_into_dustbin --max-attempts 1" (with RUN / SHARD_PLAN overrides)
    bash "$EVAL/start_master.sh" "$RUN" "${COMMON[@]}" ${EXTRA_MASTER_ARGS:-} --nodes "$(nodes_of "$spec")" --node-pairs "$spec"
    log "started $RUN on '$spec'" ;;
  dry)
    spec=${1:-$SINGLE_SPEC}; st=$SCRATCH/dry_state; rm -rf "$st"; mkdir -p "$st"
    CHECKOUT=${ROBODOJO_CHECKOUT_X5:-$HOME/zekail/sana_wam_sim_benchmarks/RoboDojo-official-clone-20260823}
    "$PY" "$EVAL/master.py" --run-root "/homes/jiamingt/robodojo_eval/${RUN}_dry" --checkout "$CHECKOUT" --state-dir "$st" \
         --dry-run --dry-sleep 0.05 --poll 1 "${COMMON[@]}" ${EXTRA_MASTER_ARGS:-} --nodes "$(nodes_of "$spec")" --node-pairs "$spec" >"$st/stdout.log" 2>&1 || true
    grep -h "planned\|dispatch order\|ALL UNITS SETTLED" "$st/master.log" | cut -c1-260 | head -6
    grep -h "Traceback\|Error\|error:" "$st/stdout.log" | head -5 || true
    "$PY" - "$st/state.json" <<'PYEOF'
import collections, json, sys
s = json.load(open(sys.argv[1]))
print("slots:", [(x["slot_id"], x["server_gpu"], x["client_gpu"], x["rpc_port"], x["ws_port"]) for x in s["slots"]])
by = collections.Counter(u["state"] for u in s["units"])
print("units", len(s["units"]), "episodes", sum(u["episodes_per_task"] for u in s["units"]), "states", dict(by))
print("dimension order of dispatch:", list(dict.fromkeys(u["dimension"] for u in sorted(s["units"], key=lambda u: (u.get("history") or [{}])[0].get("dispatched") or ""))))
PYEOF
    ;;
  autogrow)
    sub=${1:-status}; pidf=$STATE/autogrow.pid
    case $sub in
      start)
        [[ -s $STATE/master.pid ]] && kill -0 "$(cat "$STATE/master.pid")" 2>/dev/null || { echo "no live master for $RUN"; exit 1; }
        if [[ -s $pidf ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then echo "autogrow already running (pid $(cat "$pidf"))"; exit 0; fi
        nohup setsid bash "$0" autogrow-loop >>"$STATE/autogrow.log" 2>&1 </dev/null &
        echo $! >"$pidf"; log "autogrow started (pid $!), log $STATE/autogrow.log" ;;
      stop) if [[ -s $pidf ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then kill -TERM "$(cat "$pidf")"; log "autogrow stopped"; else echo "autogrow not running"; fi ;;
      status) if [[ -s $pidf ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then echo "autogrow running (pid $(cat "$pidf"))"; else echo "autogrow not running"; fi
              tail -n 5 "$STATE/autogrow.log" 2>/dev/null || true ;;
      *) echo "autogrow start|stop|status"; exit 2 ;;
    esac ;;
  autogrow-loop)
    interval=${AUTOGROW_INTERVAL:-300}
    log "autogrow: every ${interval}s, append every USABLE pair no live master holds to $RUN (USABLE $USABLE)"
    while :; do
      if ! { [[ -s $STATE/master.pid ]] && kill -0 "$(cat "$STATE/master.pid")" 2>/dev/null; }; then log "autogrow: the master of $RUN is gone; exiting"; break; fi
      if [[ ! -e $STATE/control/add_slots ]]; then
        free=$(free_pairs || true)
        # drop the pairs this run already owns (its own idle slots also read as free once its queue is empty)
        free=$("$PY" - "$STATE/state.json" "$free" <<'PYEOF' || true
import json, sys
own = {(x["node"], x["server_gpu"] // 2) for x in json.load(open(sys.argv[1]))["slots"] if x["server_gpu"] != x["client_gpu"]}
out = []
for item in sys.argv[2].split(";"):
    if not item.strip(): continue
    node, ks = item.split(":"); ks = [k for k in ks.split(",") if (node, int(k)) not in own]
    if ks: out.append(f"{node}:{','.join(ks)}")
print(";".join(out))
PYEOF
)
        if [[ -n $free ]]; then
          mkdir -p "$STATE/control"; printf '%s\n' "$free" >"$STATE/control/add_slots.tmp" && mv -f "$STATE/control/add_slots.tmp" "$STATE/control/add_slots"
          log "autogrow: add_slots '$free'"
        fi
      fi
      sleep "$interval"
    done ;;
  report)
    echo "== $RUN  $(date -u +%FT%TZ)"; grep -m1 "^units:" "$STATE/STATUS.md" || true
    "$PY" - "$STATE/state.json" <<'PYEOF'
import collections, json, sys, time
s = json.load(open(sys.argv[1])); now = time.time()
by = collections.defaultdict(collections.Counter)
for u in s["units"]: by[u["dimension"]][u["state"]] += 1
for d in ("precision", "memory", "long-horizon", "generalization", "open"):
    print(f"{d:15s} done {by[d]['done']:2d} running {by[d]['running']:2d} queued {by[d]['queued']:2d} failed {by[d]['failed']:2d}")
run = []
for u in s["units"]:
    if u["state"] == "running":
        sl = next(x for x in s["slots"] if x["slot_id"] == u["slot"]); run.append(f"{u['unit_id'].replace('__s0', '')}@{u['slot']}({int((now - sl['dispatched_at']) / 60)}m)")
print("slots:", len(s["slots"]), [x["slot_id"] for x in s["slots"]])
print("running:", " ".join(run))
PYEOF
    if [[ ${1:-} != --no-progress ]]; then "$PY" "$EVAL/progress_xpl.py" "$STATE" 2>/dev/null | grep "running" | sed 's/__s0//' | awk '{printf "  %-40s %s eps, succ %s, score_sum %s\n",$1,$3,$4,$5}' || true; fi
    echo "== last completions"; grep -h "DONE " "$STATE/master.log" | tail -n 8 | cut -c1-140 || true ;;
  compare)
    ts=$(date -u +%Y%m%dT%H%M); out=$HOME/zekail/robodojo_results/per_task_${RUN}_vs_baselines_$ts.md
    if ! "$PY" "$EVAL/compare_runs.py" "$RUN" "${BASELINES[@]}" --results-root "$HOME/robodojo_master" --runs-subdir . --out "$out" >"$SCRATCH/compare.stdout" 2>&1; then
      tail -n 5 "$SCRATCH/compare.stdout"; exit 1
    fi
    echo "columns: this run | jointabs f33fps8 sanavideo | eefabs_nsc_s30k ee | MoT canvas s36k   (SR% / score%)"
    grep -h "tasks compared" "$SCRATCH/compare.stdout" | cut -c1-120
    grep "^| [A-Za-z_]* | [A-Za-z-]* | [0-9]" "$out" | cut -c1-200       # tasks this run has finished
    grep "^| Precision\|^| Memory\|^| Open\|^| Long-Horizon\|^| Generalization\|overall" "$out" | cut -c1-200
    echo "-> $out" ;;
  tail)   # the TAIL TEMPLATE (user 2026-09-25): finish the long unsharded units (play_tic_tac_toe, imitate_sorting_sequence, ...)
           # in parallel once the queue is empty -- one-layout shards above a cut + substitutes for abandoned layouts; see
           # eval/tail_shards.py. bash $0 tail plan|launch|status|more-substitutes|assemble TASK [TASK ...] [--cut N --spare N ...]
    sub=${1:?plan|launch|status|more-substitutes|assemble}; shift
    targs=(); topts=()
    while [[ $# -gt 0 ]]; do case $1 in --dry-run|--force) topts+=("$1"); shift ;; --*) topts+=("$1" "$2"); shift 2 ;; *) targs+=(--task "$1"); shift ;; esac; done
    [[ ${#targs[@]} -gt 0 ]] || targs=(--task play_tic_tac_toe --task imitate_sorting_sequence)
    "$PY" "$EVAL/tail_shards.py" --run "$RUN" --driver "$0" "${targs[@]}" "${topts[@]}" "$sub" ;;
  status) sed -n 1,40p "$STATE/STATUS.md"; echo ...; tail -n 6 "$STATE/master.log" ;;
  watch) tail -n 0 -F "$STATE/master.log" | grep -E --line-buffered "DONE |FAIL|failed [1-9]|quarantin|ALL UNITS SETTLED|Traceback|WARN slot|stale|server_timeout|server_fail|prologue|control: |requeue" ;;
  server-log)
    uid=${1:?unit id}; run=${2:-$RUN}; a=${3:-1}; dst=$SCRATCH/serverlog_${run}_${uid}_a$a; mkdir -p "$dst"
    rsync -az "$NFS_USER_HOST:/homes/jiamingt/robodojo_eval/$run/units/$uid/a$a/server/" "$dst/"
    grep -h "\[SANA_WAM\]\|WARNING\|Traceback\|Error\|error" "$dst"/server.stdout.log "$dst"/server.stderr.log 2>/dev/null | head -n 30 ;;
  official-summary)   # official summarize_result.py aggregation (consolidates first) -> runs/<RUN>/OFFICIAL_SUMMARY_<ts>.md
    # This run carries NO --settings label, so units have setting=None and unit ids are '<task>__s0'; POLICY names the
    # adapter directory segment of the layout summarize_result.py walks.
    "$PY" "$EVAL/consolidate_results.py" >/dev/null 2>&1 || true
    # SUMMARY_RUNS="<main run> <back-fill run> ..." lays several runs of ONE checkpoint into one tree (e.g. the 51-config run
    # plus the back-fill of the 3 support-arm tasks) so the official table covers all 42 tasks; default: this RUN alone
    runs=${SUMMARY_RUNS:-$RUN}; out=$SCRATCH/official_$RUN; rm -rf "$out"
    "$PY" - "${runs// /,}" "$POLICY" "$out" <<'PYEOF3'
import collections, json, os, re, sys
runs, policy, out = sys.argv[1].split(","), sys.argv[2], sys.argv[3]
for run in runs:
  rd = os.path.expanduser(f"~/zekail/robodojo_results/runs/{run}")
  st = json.load(open(f"{rd}/state.json")); label = os.path.basename(st["checkpoint_root"])
  if True:
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
        d = f"{out}/eval_result/RoboDojo/{task}/{policy}/arx_x5/{seed}_ckpt_name={label},action_type=ee/{max(rids)}"
        os.makedirs(d, exist_ok=True); json.dump(doc, open(f"{d}/_result.json", "w"), indent=1); laid += 1
    print(f"{laid} task configs laid out (layout shards concatenated)" + (f"; still running: {', '.join(waiting)}" if waiting else "")
          + (f"; PROBLEMS: {bad}" if bad else "") + (f"; notes: {len(notes)} substituted layouts" if notes else ""))
PYEOF3
    CHECKOUT=${ROBODOJO_CHECKOUT_X5:-$HOME/zekail/sana_wam_sim_benchmarks/RoboDojo-official-clone-20260823}
    ROBODOJO_EVAL_ROOT=$out/eval_result/RoboDojo "$PY" "$CHECKOUT/scripts/internal/summarize_result.py" 2>&1 | { grep -v '^[^|].*\(MISMATCH\|mp4\|video\)' || true; } | tail -2
    ts=$(date -u +%Y%m%dT%H%M); mkdir -p "$HOME/zekail/robodojo_results/runs/$RUN"
    name=OFFICIAL_SUMMARY_$ts.md; [[ $runs == *" "* ]] && name=OFFICIAL_SUMMARY_combined_$ts.md
    { [[ $runs == *" "* ]] && printf '<!-- runs laid out together: %s -->\n' "$runs"; cat "$out/eval_result/RoboDojo/_summary.md"; } \
      >"$HOME/zekail/robodojo_results/runs/$RUN/$name"
    echo "-> ~/zekail/robodojo_results/runs/$RUN/$name" ;;
  *) sed -n 2,33p "$0"; exit 1 ;;
esac
