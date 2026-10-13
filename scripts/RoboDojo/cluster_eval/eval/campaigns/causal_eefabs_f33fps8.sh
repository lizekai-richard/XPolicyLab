#!/usr/bin/env bash
# Campaign driver (runs on x5): the CHUNK-CAUSAL policy on the XPolicyLab lane through policy/SANA_WAM_CAUSAL (worker env
# SANA_EVAL_XPL_POLICY=SANA_WAM_CAUSAL; a copy of vanilla34k_eefabs_f33fps8.sh, 2026-10-06):
#   CKPT_LABEL=causal_eefabs_sana_pixel_320x512_vanilla52k_aligned_f33fps8_m48n24_s7646 (default; HF
#     logits/causal_robodojo_eefabs_sana_pixel_320x512_vanilla52k_aligned_f33fps8_m48n24 at epoch 2 / step 7,646, training in
#     progress; trained on zekai-merge d8dba0654 -> 958be1587 = the adapter's c391260a4 model / data contract)
# Teacher-forced two-stream causal policy (GDN recurrence + softmax window 24 chunks, chunk 32 rows, obs_in_first_chunk) on the
# vanilla52k sana_pixel 320x512 STRETCH donor: EEF-only robot_base_eef ABSOLUTE targets, f33fps8 (stride 4), rope ALIGNED, G = 1,
# normalization 1fe3b7e7, video flow shift 5 + action flow shift 1. Every key below is a policy/SANA_WAM_CAUSAL deploy key.
# Sampling (user, 2026-09-26): video CFG 6, action CFG 1, 10 NFE, VIDEO flow shift 5, ACTION flow shift 1, executed as ee poses.
# Same slots, shard plan, complete protocol (42 tasks / 54 configs / 2100 eps, seed 0) and tail template as the other drivers.
#
#   bash causal_eefabs_f33fps8.sh check-sync           worker scripts + SANA_WAM + SANA_WAM_CAUSAL on NFS == x5
#   bash causal_eefabs_f33fps8.sh free-pairs           usable GPU pairs no live master can dispatch onto
#   bash causal_eefabs_f33fps8.sh smoke-gpu 'l3:1' [TASK] [EPS]
#                                                    direct worker on ONE free GPU (policy server + Isaac client share it);
#                                                    refuses a GPU any live master slot covers, a GPU with >1 GiB in use,
#                                                    or a listening port; run <RUN>_smoke1gpu, unit <TASK>__s0
#   bash causal_eefabs_f33fps8.sh smoke-status [TASK]  status / log tail / result of that direct smoke
#   bash causal_eefabs_f33fps8.sh smoke 'l5:1'         1 episode of insert_key on a free PAIR through the master -> <RUN>_smoke
#   bash causal_eefabs_f33fps8.sh dry ['<spec>']       local --dry-run of the plan on that slot spec (default 'l3:s1;l26:s0')
#   bash causal_eefabs_f33fps8.sh start ['<spec>']     full run (category order precision, memory, long-horizon, generalization, open;
#                                                    SHARD_PLAN layout shards); default spec = every free GPU as a single-GPU slot
#                                                    (SINGLE_SPEC); pairs 'l5:0,1' and single GPUs 'l3:s1' also accepted
#   bash causal_eefabs_f33fps8.sh autogrow start|stop|status
#                                                    background loop (every 5 min): every USABLE pair no live master holds
#                                                    (no unit, not draining, its master has nothing dispatchable) is appended
#                                                    to this run through control/add_slots; stops when this run's master ends
#   bash causal_eefabs_f33fps8.sh compare               per-task + dimension means over the tasks this run has finished, vs the
#                                                    single-stream sana_pixel aligned run and the MoT openwam / pixel 320x512 pretrained runs
#   bash causal_eefabs_f33fps8.sh tail plan|launch|status|more-substitutes|assemble [TASK ...]
#                                                    the TAIL TEMPLATE: when the queue is empty, finish the unsharded tail units
#                                                    (default play_tic_tac_toe + imitate_sorting_sequence) as one-layout shards
#                                                    + substitutes on idle GPUs, then assemble them into the run (eval/tail_shards.py)
#   bash causal_eefabs_f33fps8.sh status | report | watch | server-log UNIT [RUN] [ATTEMPT]
#   ACTION_CFG=6 bash ...                            historical single-knob CFG (default ACTION_CFG=1: video-only guidance)
#   VIEW_RESIZE=crop|stretch bash ...                force a view resize against the checkpoint's own (A/B only: run _evalcrop / _evalstretch)
set -euo pipefail
CKPT_LABEL=${CKPT_LABEL:-causal_eefabs_sana_pixel_320x512_vanilla52k_aligned_f33fps8_m48n24_s7646}
# How each camera view reaches its bucket (adapter deploy key view_resize; the OpenWAM canvas stretches either way). The
# adapters default to stretch, the only SFT resize since rwm/zekai-merge 1061b16f0 / rwm/mot 034e55dca (2026-09-27; user
# 2026-09-29), and the node worker serves every label in eval/pre_stretch_labels.txt (verified trained before them) with the
# legacy crop -- so nothing is set here by default. VIEW_RESIZE=crop|stretch forces a mode against the checkpoint's own
# (a train/eval-mismatch A/B): the run name then gets _evalcrop / _evalstretch, and a launch whose mode disagrees with its
# run (tail shard, retry) is refused.
VR_REGISTRY=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/pre_stretch_labels.txt
[[ -f $VR_REGISTRY ]] || { echo "missing $VR_REGISTRY"; exit 2; }
TRAINED_RESIZE=stretch
awk -v l="$CKPT_LABEL" '!/^[[:space:]]*(#|$)/ && $1 == l {f = 1} END {exit !f}' "$VR_REGISTRY" && TRAINED_RESIZE=crop
VIEW_RESIZE=${VIEW_RESIZE:-}
[[ -z $VIEW_RESIZE || $VIEW_RESIZE == crop || $VIEW_RESIZE == stretch ]] || { echo "VIEW_RESIZE must be crop or stretch, got '$VIEW_RESIZE'"; exit 2; }
[[ " ${EXTRA_OVERRIDES:-} " == *" view_resize="* ]] && { echo "set VIEW_RESIZE instead of view_resize= in EXTRA_OVERRIDES"; exit 2; }
VR_TAG=""; [[ -n $VIEW_RESIZE && $VIEW_RESIZE != "$TRAINED_RESIZE" ]] && VR_TAG=_eval$VIEW_RESIZE
# what the adapter's ready line must print for this launch
if [[ -n $VIEW_RESIZE ]]; then VR_EXPECT="$VIEW_RESIZE (deploy)"; elif [[ $TRAINED_RESIZE == crop ]]; then VR_EXPECT="crop (deploy)"; else VR_EXPECT="stretch (default)"; fi
# action-stream CFG: DEFAULT 1 (user rule 2026-09-25) = video-only guidance
ACTION_CFG=${ACTION_CFG:-1}
# flow shifts (user, 2026-09-26): video 5 (what these checkpoints were trained with), action 1 (their separate action schedule)
VIDEO_FS=${VIDEO_FS:-5}; ACTION_FS=${ACTION_FS:-1}
RUN=${RUN:-official_seed0_${CKPT_LABEL}_xpl_vcfg6_acfg${ACTION_CFG}_nfe10_vfs${VIDEO_FS}_afs${ACTION_FS}_ee${VR_TAG}}
CKPT=/homes/jiamingt/checkpoints/huggingface/logits/$CKPT_LABEL
# The artifact this snapshot's own training yaml pins (grep normalization_sha256 <snapshot>/config.yaml after any re-download).
NORM_SHA=${NORM_SHA:-1fe3b7e72e8fa93eca5efb8a8deeace1daf4434b4b97e9f4445670ac758287ed}
OVERRIDES="sampling_steps=10 cfg_scale=6 action_cfg_scale=$ACTION_CFG flow_shift=$VIDEO_FS action_flow_shift=$ACTION_FS normalization_sha256=$NORM_SHA"
[[ -n $VIEW_RESIZE ]] && OVERRIDES="$OVERRIDES view_resize=$VIEW_RESIZE"
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
DIM_ORDER=${DIM_ORDER:-memory,long-horizon,precision,generalization,open}   # master --dimension-order (user 2026-10-06: memory + long-horizon first)
PIN_LONGEST=${PIN_LONGEST:-2}     # the longest units dispatched first regardless of category (master --pin-longest)
# ONE GPU per slot (policy server + Isaac client on the same board; user 2026-09-24): every free GPU of the six 726e9aa
# nodes -- l3 GPU0 and l26 GPU1 host another user's vLLM.
SINGLE_SPEC="l5:s0,s1,s2,s3,s4,s5,s6,s7;l7:s0,s1,s2,s3,s4,s5,s6,s7;l8:s0,s1,s2,s3,s4,s5,s6,s7;l25:s0,s1,s2,s3,s4,s5,s6,s7;l3:s1,s2,s3,s4,s5,s6,s7;l26:s0,s2,s3,s4,s5,s6,s7"
# Single-GPU slot g binds rpc/ws base + 50 + g: 63250.. / 65250.., disjoint from every other campaign's range (61000..61900,
# 62300.. single-stream sana_pixel, 62600.. MoT, 62700.. tail shards, 62400.., 62500.., 62800.., 63000.., 63500.., 64000..,
# 64300.., 64500.., 64600.., 64700.., 65000..); both vanilla34k runs share the base (a port is per GPU, a GPU is in one run).
# The direct single-GPU smoke uses 63290 / 65290.
# The causal driver: 63600 / 65300 (slot g -> 63650 + g / 65350 + g; smoke 63690 / 65390), disjoint from the bases above.
PORT_ARGS=(--rpc-port-base 63600 --ws-port-base 65300)
SMOKE_RPC_PORT=${SMOKE_RPC_PORT:-63690}; SMOKE_WS_PORT=${SMOKE_WS_PORT:-65390}
XPL_ROOT=/homes/jiamingt/xpl_zekai/XPolicyLab
POLICY=SANA_WAM_CAUSAL   # adapter dir segment of the official eval_result layout (tail_shards.py reads this line)
WORKER_ENV="SANA_EVAL_XPL_POLICY=SANA_WAM_CAUSAL"
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
  diff=$(rsync -n -c -i "$EVAL/run_unit_xpl.sh" "$EVAL/run_unit.sh" "$EVAL/pre_stretch_labels.txt" "$NFS_USER_HOST:/homes/jiamingt/robodojo_install/eval/" | grep -v '^\.' || true)
  for pol in SANA_WAM SANA_WAM_CAUSAL; do   # SANA_WAM_CAUSAL imports policy/SANA_WAM/sana_wam_min
    diff+=$(rsync -n -c -i -r --exclude __pycache__ --exclude .pytest_cache "$HOME/zekail/XPolicyLab/policy/$pol/" "$NFS_USER_HOST:$XPL_ROOT/policy/$pol/" | grep -v '^\.d\|^cd\|^\.f' || true)
  done
  if [[ -n $diff ]]; then echo "NFS copies differ from x5 (rsync them first):"; echo "$diff"; exit 1; fi
  log "worker scripts and the SANA_WAM + SANA_WAM_CAUSAL adapters are in sync with NFS"
}
check_view_resize(){ # launch paths (start / smoke / smoke-gpu, hence tail): a shard or retry is served like the run it lands in
  local run_mode=$TRAINED_RESIZE serve=${VIEW_RESIZE:-$TRAINED_RESIZE}
  case $RUN in *_evalcrop) run_mode=crop ;; *_evalstretch) run_mode=stretch ;; esac
  [[ $serve == "$run_mode" ]] && return
  echo "REFUSE: VIEW_RESIZE=${VIEW_RESIZE:-unset} serves $CKPT_LABEL with $serve, but run $RUN is served $run_mode"; exit 2
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
        --xpl-overrides "$OVERRIDES" --worker-env "$WORKER_ENV" --order dimension --pin-longest "$PIN_LONGEST" "${SHARD_ARGS[@]}"
        --support-arm-embodiments on
        --dimension-order "$DIM_ORDER" "${PORT_ARGS[@]}")
SMOKE_RUN_ROOT=${SMOKE_RUN_ROOT:-/homes/jiamingt/robodojo_eval/${RUN}_smoke1gpu}
# smoke-gpu extras: SMOKE_UNIT_ID (default <task>__s0), SMOKE_LAYOUT_IDS='50,51,52,53' (+ SMOKE_LAYOUT_TOTAL) = evaluate exactly
# those layouts (run_unit_xpl.sh layout shard), e.g. the substitutes the official client draws for abandoned layouts
BASELINES=(official_seed0_eefabs_sana_pixel_320x512_vanilla52k_aligned_f33fps8_fixed_resize_xpl_vcfg6_acfg1_nfe10_vfs5_afs1_ee)   # the causal run's donor (stretch vanilla52k sana_pixel)

cmd=${1:-help}; shift || true
# RETIRED 2026-10-10: policy/SANA_WAM_CAUSAL serves the per-chunk observation protocol only (Sana 61aad2d58 / 3a6b2bf07) and refuses the s7646 checkpoint (chunk-0-only observation); the adapter that served it is XPolicyLab commit 69cfb8e. Read-only subcommands still work; the pantheon14k driver is causal_p14k_eefabs_f33fps8.sh.
case "$cmd ${1:-}" in smoke-gpu*|"smoke "*|smoke|start*|"autogrow start"*|autogrow-loop*|"tail launch"*|"tail more-substitutes"*) echo "$0: RETIRED 2026-10-10: policy/SANA_WAM_CAUSAL serves the per-chunk observation protocol only (Sana 61aad2d58 / 3a6b2bf07) and refuses the s7646 checkpoint (chunk-0-only observation); the adapter that served it is XPolicyLab commit 69cfb8e. Read-only subcommands still work; the pantheon14k driver is causal_p14k_eefabs_f33fps8.sh." >&2; exit 2 ;; esac
case $cmd in
  check-sync) check_sync ;;
  free-pairs) free_pairs ;;
  smoke-gpu)
    spec=${1:?'NODE:GPU, e.g. l3:1'}; task=${2:-insert_key}; eps=${3:-1}
    node=${spec%%:*}; gpu=${spec##*:}
    [[ $gpu =~ ^[0-7]$ ]] || { echo "GPU index must be 0..7, got '$gpu'"; exit 2; }
    check_sync; check_view_resize
    if live_slots | grep -qx "$node:$gpu"; then echo "REFUSE: $node GPU $gpu belongs to a slot of a live master"; exit 3; fi
    seed=${SMOKE_SEED:-0}; [[ $seed =~ ^[0-9]+$ ]] || { echo "SMOKE_SEED must be an integer, got '$seed'"; exit 2; }
    uid=${SMOKE_UNIT_ID:-${task}__s$seed}; udir=$SMOKE_RUN_ROOT/units/$uid/a1
    layout_env=""; [[ -n ${SMOKE_LAYOUT_IDS:-} ]] && layout_env="SANA_EVAL_LAYOUT_IDS='$SMOKE_LAYOUT_IDS' SANA_EVAL_LAYOUT_TOTAL='${SMOKE_LAYOUT_TOTAL:-0}'"
    # $gpu is a CUDA index = the gpu-th HEALTHY board (a '[GPU requires reset]' board is not enumerated; l2/l4/l6)
    remote="set -u
healthy=(\$(nvidia-smi --query-gpu=index,pstate --format=csv,noheader | awk -F', ' '\$2 !~ /reset/ {print \$1}'))
phys=\${healthy[$gpu]:-}; [ -n \"\$phys\" ] || { echo \"REFUSE: \$(hostname -s) has no healthy board for CUDA device $gpu\"; exit 3; }
used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i \$phys | tr -d ' ')
[ \"\$used\" -lt 1024 ] || { echo \"REFUSE: CUDA $gpu (board \$phys) on \$(hostname -s) has \${used} MiB in use\"; exit 3; }
for p in $SMOKE_RPC_PORT $SMOKE_WS_PORT; do [ -z \"\$(ss -ltnH \"sport = :\$p\")\" ] || { echo \"REFUSE: port \$p is listening on \$(hostname -s)\"; exit 3; }; done
# the worker's own runner lock (units/<uid>/.lock) is taken only AFTER this rm -rf: probe it first, or a relaunch wipes a1/ of
# a unit whose runner is still alive (2026-09-28: tail relaunches wiped a running shard's results 20 times)
mkdir -p \"\$(dirname '$udir')\"; exec 8>\"\$(dirname '$udir')/.lock\"; flock -n 8 || { echo \"REFUSE: unit $uid already has a live runner\"; exit 3; }; exec 8>&-
rm -rf '$udir'; mkdir -p '$udir' || { echo mkdir_failed; exit 1; }
SANA_EVAL_XPL_POLICY=SANA_WAM_CAUSAL SANA_EVAL_CHECKPOINT_ROOT='$CKPT' SANA_EVAL_XPL_ROOT='$XPL_ROOT' \
SANA_EVAL_XPL_EXTRA_OVERRIDES='$OVERRIDES' SANA_EVAL_XPL_ACTION_TYPE=ee SEC_PER_STEP=4 $layout_env \
  nohup setsid bash ~/robodojo_install/eval/run_unit_xpl.sh '$SMOKE_RUN_ROOT' '$uid' '$task' $seed '$eps' $gpu $gpu $SMOKE_RPC_PORT $SMOKE_WS_PORT 1 \
  > '$udir/nohup.log' 2>&1 < /dev/null &
sleep 2; kill -0 \$! 2>/dev/null && echo \"dispatched \$! on \$(hostname -s) GPU $gpu (server+client), ports $SMOKE_RPC_PORT/$SMOKE_WS_PORT\" || echo died"
    ssh -n -o BatchMode=yes -o ConnectTimeout=20 "jiamingt@$node.mit.edu" "$remote"
    printf '%s\n' "node=$node gpu=$gpu task=$task eps=$eps run_root=$SMOKE_RUN_ROOT unit=$uid overrides=$OVERRIDES started=$(date -u +%FT%TZ)" >"$SCRATCH/smoke1gpu_$task.txt"
    log "smoke-gpu launched; follow with: bash $0 smoke-status $task" ;;
  smoke-status)
    task=${1:-insert_key}; uid=${SMOKE_UNIT_ID:-${task}__s${SMOKE_SEED:-0}}; dst=$SCRATCH/smoke1gpu_$uid; mkdir -p "$dst"
    rsync -az --exclude 'upstream_eval_result/**/*.mp4' "$NFS_USER_HOST:$SMOKE_RUN_ROOT/units/$uid/a1/" "$dst/" 2>/dev/null || true
    echo "== status.txt"; cat "$dst/status.txt" 2>/dev/null || echo "(none yet)"
    echo "== server log ([SANA_WAM_CAUSAL] lines, errors)"
    grep -h "\[SANA_WAM_CAUSAL\]\|Traceback\|Error\|error:" "$dst/server/server.stdout.log" "$dst/server/server.stderr.log" 2>/dev/null | cut -c1-400 | head -n 30 || true
    echo "== client progress: $(cat "$dst"/client/*/client.stdout.log 2>/dev/null | grep -a -o 'env0 step' | wc -l) simulation steps"
    echo "== result.json"; cat "$dst/result.json" 2>/dev/null || echo "(none yet)" ;;
  smoke)
    spec=${1:?pairs spec, e.g. 'l5:1'}; check_sync; check_view_resize; check_pairs_free "$spec"
    bash "$EVAL/start_master.sh" "${RUN}_smoke" "${COMMON[@]}" --shard-plan "" --nodes "$(nodes_of "$spec")" --node-pairs "$spec" --tasks insert_key --episodes-per-task 1 --pin-longest 0 --poll 20
    echo "expect in the server log: '[SANA_WAM_CAUSAL] ready: ... actions_per_chunk=32 video_frame_stride=4 ... sliding_window_chunks=24"
    echo "  ... rope=aligned softmax_window=training steps=10 video_cfg_scale=6.0 action_cfg_scale=1.0 flow_shift=5.0 action_flow_shift=1.0"
    echo "  view_resize=$VR_EXPECT commit_actions=executed gdn_recurrence=checkpoint ... action_type=ee' -> bash $0 server-log insert_key__s0 ${RUN}_smoke" ;;
  start)
    spec=${1:-$SINGLE_SPEC}; check_sync; check_view_resize; check_pairs_free "$spec"
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
    echo "columns: this run | single-stream sana_pixel aligned (sanavideo) | MoT openwam pretrained | MoT pixel 320x512 pretrained   (SR% / score%)"
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
    grep -h "\[SANA_WAM_CAUSAL\]\|WARNING\|Traceback\|Error\|error" "$dst"/server.stdout.log "$dst"/server.stderr.log 2>/dev/null | head -n 30 ;;
  official-summary)   # official summarize_result.py aggregation (consolidates first) -> runs/<RUN>/OFFICIAL_SUMMARY_<ts>.md
    # This run carries NO --settings label, so units have setting=None and unit ids are '<task>__s0'; POLICY names the
    # adapter directory segment of the layout summarize_result.py walks.
    "$PY" "$EVAL/consolidate_results.py" >/dev/null 2>&1 || true
    # SUMMARY_RUNS="<main run> <back-fill run> ..." lays several runs of ONE checkpoint into one tree (e.g. the 51-config run
    # plus the back-fill of the 3 support-arm tasks) so the official table covers all 42 tasks; default: this RUN alone
    runs=${SUMMARY_RUNS:-$RUN}; out=$SCRATCH/official_$RUN; rm -rf "$out"
    "$PY" - "${runs// /,}" "$POLICY" "$out" <<'PYEOF3'
import collections, json, os, re, sys
runs, policy, out = [r for r in sys.argv[1].split(",") if r], sys.argv[2], sys.argv[3]   # a trailing space in SUMMARY_RUNS left an empty name (2026-10-01)
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
    ts=$(date -u +%Y%m%dT%H%M%S); mkdir -p "$HOME/zekail/robodojo_results/runs/$RUN"   # seconds: the chain writes several tables a minute
    name=OFFICIAL_SUMMARY_$ts.md; [[ $runs == *" "* ]] && name=OFFICIAL_SUMMARY_combined_$ts.md
    { [[ $runs == *" "* ]] && printf '<!-- runs laid out together: %s -->\n' "$runs"; cat "$out/eval_result/RoboDojo/_summary.md"; } \
      >"$HOME/zekail/robodojo_results/runs/$RUN/$name"
    echo "-> ~/zekail/robodojo_results/runs/$RUN/$name" ;;
  *) sed -n 2,33p "$0"; exit 1 ;;
esac
