#!/usr/bin/env bash
# Campaign driver (runs on x5): the FIXED-STEP DiT CACHE on the complete RoboDojo protocol (user, 2026-09-30).
#   Schedule: the RoboDojo velocity calibration of 2026-09-29 (~/zekail/robodojo_results/analysis/velocity_calib_20260929)
#     picked, on the 10-NFE grid (video flow shift 5 / action 1, video CFG 6 / action CFG 1), NFE 4 = COMPUTE steps
#     0, 3, 6, 8 and replay the last computed step on 1, 2, 4, 5, 7, 9 (STEP_CACHE_SKIP 1-2,4-5,7,9; step 9 reused: its dt
#     is tiny). First-order action error 0.66% of the displacement (NFE 3 = 0,3,6: 0.94%).
#   Checkpoint: HF logits/sft_robodojo_eefabs_sana_pixel_320x512_vanilla34k_aligned_f33fps8_fixed_resize (stretch, pad
#     unmasked, norm 9f8b98e); cache-free 3 seeds 13.75+-0.66 / 19.82+-0.41 (309, 276, 296 successes), seed 0 14.52 / 20.30.
#   Adapter: a COPY of XPolicyLab carrying the step-cache switch (deploy keys step_cache / step_cache_skip /
#     step_cache_compensation / step_cache_delta_scale; sana_wam_min/step_cache.py), x5 ~/zekail/xpl_fixedstep ->
#     NFS /homes/jiamingt/xpl_fixedstep, node workspace /dataset2/jiamingt/robodojo/xpl_fixedstep_ws. The shared
#     ~/xpl_zekai adapter and its workspace are untouched (live runs serve from them).
#   Everything else = the cache-free seed runs of this checkpoint: 42 tasks / 54 configs / 2100 eps per seed,
#     --support-arm-embodiments on, single-GPU slots, SHARD_PLAN, dimension order, pin-longest 2, tail template.
#   THREE SEEDS IN PARALLEL (user, 2026-09-30): one master per seed (SEED=0|1|2 -> run official_seed<N>_..., master
#     --seeds N, unit ids <task>__s<N>), each on a round-robin share of the free GPUs; every per-run command below takes
#     SEED (default 0), e.g. `SEED=2 bash <driver> tail launch play_tic_tac_toe`.
#
#   bash fixedstep_cache_eefabs_sana_pixel.sh check-sync      worker scripts + the SANA_WAM copy on NFS == x5
#   bash fixedstep_cache_eefabs_sana_pixel.sh free-spec       every usable GPU right now as single-GPU slots ('l1:s0,s2;l5:s1')
#   bash fixedstep_cache_eefabs_sana_pixel.sh smoke-gpu 'l25:5' [TASK] [EPS]
#                                                    direct worker on ONE free GPU (server + client share it) -> <RUN>_smoke1gpu
#   bash fixedstep_cache_eefabs_sana_pixel.sh smoke-status [TASK]
#   bash fixedstep_cache_eefabs_sana_pixel.sh dry ['<spec>']   local --dry-run of the plan
#   bash fixedstep_cache_eefabs_sana_pixel.sh start ['<spec>'] the SEED run now (default spec: free-spec)
#   bash fixedstep_cache_eefabs_sana_pixel.sh start-seeds ['<spec>']
#                                                    one master per seed of SEEDS (default 0 1 2) that has no run yet, the
#                                                    spec (default free-spec) dealt round-robin over them
#   bash fixedstep_cache_eefabs_sana_pixel.sh wait-start start|stop|status
#                                                    background waiter: once every WAIT_RUNS run is completely over (no live
#                                                    master, no worker on any node, WAIT_QUIET_CHECKS checks in a row) run
#                                                    start-seeds on EVERY free GPU (free-spec; at least MIN_GPUS)
#   bash fixedstep_cache_eefabs_sana_pixel.sh status-seeds | compare-seeds | official-summary-seeds
#                                                    the three seed runs at a glance / merged vs the merged cache-free seeds /
#                                                    the official 3-seed table
#   bash fixedstep_cache_eefabs_sana_pixel.sh wait-check      one probe: which waited-for runs are still active, which GPUs are free
#   bash fixedstep_cache_eefabs_sana_pixel.sh resume-seeds    resume every stopped seed run on its own slots (pause lifted), then deal the
#                                                    free GPUs round-robin to them; WAIT_ACTION=resume-seeds makes wait-start do this
#   REQUIRE_FINISHED=1 ...                           a WAIT_RUNS run that has not started yet / has unfinished units still blocks
#   bash fixedstep_cache_eefabs_sana_pixel.sh wait-grow start|stop|status
#                                                    background, one shot: once WAIT_RUNS are completely over, deal every free
#                                                    GPU round-robin to the live seed runs (control/add_slots), then exit
#   bash fixedstep_cache_eefabs_sana_pixel.sh watch-seeds [HOURS]
#                                                    block until a seed run needs attention (problem line, dead master, queue
#                                                    empty = tail time, settled) or HOURS (default 3) pass; print it and exit
#   bash fixedstep_cache_eefabs_sana_pixel.sh add-free        append every GPU free now to the live run (control/add_slots)
#   bash fixedstep_cache_eefabs_sana_pixel.sh resume ['<spec>']
#   bash fixedstep_cache_eefabs_sana_pixel.sh compare        per task + dimension means vs the three cache-free seeds
#   bash fixedstep_cache_eefabs_sana_pixel.sh tail plan|launch|status|more-substitutes|fill|fill-status|invalidate|assemble [TASK ...]
#                                                    the TAIL TEMPLATE (eval/tail_shards.py; shards start through smoke-gpu,
#                                                    hence with this run's adapter copy, workspace and step cache)
#   bash fixedstep_cache_eefabs_sana_pixel.sh official-summary | status | report | watch | server-log UNIT [RUN] [ATTEMPT]
#   STEP_CACHE_SKIP=1-2,4-5,7-9 bash ...             another schedule (the run name carries the computed steps: _fixedstep036)
#   STEP_CACHE_COMP=delta STEP_CACHE_DELTA=0.75 ...  fixed_step's delta compensation (run name + _delta0.75)
set -euo pipefail
PY=${MASTER_PY:-$HOME/miniforge3/envs/sana-wam/bin/python}
CKPT_LABEL=${CKPT_LABEL:-eefabs_sana_pixel_320x512_vanilla34k_aligned_f33fps8_fixed_resize}
SEED=${SEED:-0}; SEEDS=${SEEDS:-0 1 2}
[[ $SEED =~ ^[0-9]+$ ]] || { echo "SEED must be an integer, got '$SEED'"; exit 2; }
CKPT=/homes/jiamingt/checkpoints/huggingface/logits/$CKPT_LABEL
# the artifact this snapshot's training yaml pins (the cache-free seed runs of this label served 9f8b98e)
NORM_SHA=${NORM_SHA:-9f8b98ede445f554cafef99974a0ef67ce4d59292578ccbdeaba704c9cedd5ba}
# the calibration grid (user sampling rule 2026-09-26): 10 NFE, video CFG 6 / action CFG 1, video / action flow shift 5 / 1
STEPS=10
STEP_CACHE_SKIP=${STEP_CACHE_SKIP-1-2,4-5,7,9}   # an explicitly empty value = every step computes (the cache-free loop, bit for bit)
STEP_CACHE_COMP=${STEP_CACHE_COMP:-last}
STEP_CACHE_DELTA=${STEP_CACHE_DELTA:-0.0}
# the computed steps of the schedule (validated here with the adapter's own parser: a bad spec fails before anything runs)
XPL_SRC=${XPL_SRC:-$HOME/zekail/xpl_fixedstep/XPolicyLab}
COMPUTED=$(PYTHONPATH="$XPL_SRC/policy/SANA_WAM" "$PY" -c "
import sys
from sana_wam_min.step_cache import FixedStepCache
c = FixedStepCache(sys.argv[1], int(sys.argv[2]), compensation=sys.argv[3], delta_scale=float(sys.argv[4]))
print(''.join(map(str, c.computed_steps)) if c.steps <= 10 else '_'.join(map(str, c.computed_steps)))
" "$STEP_CACHE_SKIP" "$STEPS" "$STEP_CACHE_COMP" "$STEP_CACHE_DELTA") || { echo "bad step cache settings"; exit 2; }
CACHE_TAG=_fixedstep$COMPUTED; [[ $STEP_CACHE_COMP == delta ]] && CACHE_TAG+=_delta$STEP_CACHE_DELTA
run_of(){ echo "official_seed$1_${CKPT_LABEL}_xpl_vcfg6_acfg1_nfe10_vfs5_afs1_ee$CACHE_TAG"; }
RUN=${RUN:-$(run_of "$SEED")}
OVERRIDES="sampling_steps=$STEPS cfg_scale=6 action_cfg_scale=1 flow_shift=5 action_flow_shift=1 normalization_sha256=$NORM_SHA"
OVERRIDES+=" step_cache=fixed_step step_cache_skip=$STEP_CACHE_SKIP step_cache_compensation=$STEP_CACHE_COMP step_cache_delta_scale=$STEP_CACHE_DELTA"
# Layout shards, dimension order and pinning exactly as the cache-free seed-0 run of this checkpoint (95 units).
SHARD_PLAN=${SHARD_PLAN-make_kong=2,fasten_screws=5,pour_by_language=4,play_stacking_toy=3,fill_pen_holder=3,classify_objects_by_language=3,organize_table=3,classify_objects=3,build_tower=3,store_tools_in_toolbox=3,make_toast_random=2,cover_blocks=2,make_toast=2,swap_blocks=2,pick_from_conveyor_by_image=2,match_and_pick_from_conveyor=2,pack_objects_into_box=2,pack_objects_into_box_random=2,pour_balls_into_vase=2,arrange_largest_number_random=2,sweep_blocks_random=2,arrange_largest_number=2,sort_nesting_dolls_by_size=2,insert_tubes=2,put_bottles_into_dustbin=2,sweep_blocks=2,press_by_number=2,sort_nesting_dolls_by_size_random=2,play_Xylophone=2}
DIM_ORDER=${DIM_ORDER:-memory,long-horizon,precision,generalization,open}
PIN_LONGEST=${PIN_LONGEST:-2}
# Single-GPU slot g binds rpc/ws base + 50 + g = 64250.. / 65450.. (disjoint from every other campaign's range; tail shards
# use tail_shards.py's own 62700+g / 64700+g); the direct single-GPU smoke 64290 / 65490.
RPC_BASE=64200; WS_BASE=65400
PORT_ARGS=(--rpc-port-base $RPC_BASE --ws-port-base $WS_BASE)
SMOKE_RPC_PORT=${SMOKE_RPC_PORT:-64290}; SMOKE_WS_PORT=${SMOKE_WS_PORT:-65490}
XPL_ROOT=/homes/jiamingt/xpl_fixedstep/XPolicyLab
XPL_WS=/dataset2/jiamingt/robodojo/xpl_fixedstep_ws
POLICY=SANA_WAM          # adapter dir segment of the official eval_result layout (tail_shards.py reads this line)
# The twelve evaluation nodes (l0: root's job holds every board; l4: no usable board). l3 CUDA 0 and l26 CUDA 1 host
# another user's vLLM (never free anyway). RESERVE='l3:2,3;l9:0' keeps boards out of free-spec (CUDA indices; none by
# default -- the user released l25 5, 6, 7 on 2026-09-30).
NODES_ALL=${NODES_ALL:-l1 l2 l3 l5 l6 l7 l8 l9 l25 l26 l27 l30}
RESERVE=${RESERVE:-}
# wait-start: the runs whose complete end this run waits for (the user's live openwam seed runs, 2026-09-30)
WAIT_RUNS=${WAIT_RUNS:-official_seed1_vanilla34k_eefabs_f33fps8_openwam_aligned_xpl_vcfg6_acfg1_nfe10_vfs5_afs1_ee official_seed2_vanilla34k_eefabs_f33fps8_openwam_aligned_xpl_vcfg6_acfg1_nfe10_vfs5_afs1_ee}
WAIT_INTERVAL=${WAIT_INTERVAL:-300}; WAIT_QUIET_CHECKS=${WAIT_QUIET_CHECKS:-3}; MIN_GPUS=${MIN_GPUS:-12}
NFS_USER_HOST=jiamingt@l5.mit.edu
EVAL=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STATE=$HOME/robodojo_master/$RUN
SCRATCH=${SCRATCH:-$HOME/robodojo_master/.scratch/$RUN}; mkdir -p "$SCRATCH"
SEEDS_SCRATCH=$HOME/robodojo_master/.scratch/seeds_${CKPT_LABEL}$CACHE_TAG; mkdir -p "$SEEDS_SCRATCH"
BASELINES=(official_seed0_${CKPT_LABEL}_xpl_vcfg6_acfg1_nfe10_vfs5_afs1_ee
           official_seed1_${CKPT_LABEL}_xpl_vcfg6_acfg1_nfe10_vfs5_afs1_ee
           official_seed2_${CKPT_LABEL}_xpl_vcfg6_acfg1_nfe10_vfs5_afs1_ee)
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
nodes_of(){ echo "$1" | tr ';' '\n' | cut -d: -f1 | sed '/^$/d' | paste -sd, -; }

check_sync(){
  local diff
  diff=$(rsync -n -c -i "$EVAL/run_unit_xpl.sh" "$EVAL/run_unit.sh" "$EVAL/pre_stretch_labels.txt" "$NFS_USER_HOST:/homes/jiamingt/robodojo_install/eval/" | grep -v '^\.' || true)
  diff+=$(rsync -n -c -i -r --exclude __pycache__ --exclude .pytest_cache --exclude checkpoints "$XPL_SRC/policy/SANA_WAM/" "$NFS_USER_HOST:$XPL_ROOT/policy/SANA_WAM/" | grep -v '^\.d\|^cd\|^\.f' || true)
  if [[ -n $diff ]]; then echo "NFS copies differ from x5 (rsync them first):"; echo "$diff"; exit 1; fi
  log "worker scripts and the step-cache SANA_WAM copy are in sync with NFS"
}
check_cache_tag(){ # a launch (start, smoke, tail shard) must serve the schedule its run name carries
  [[ $RUN == *"$CACHE_TAG" ]] && return
  echo "REFUSE: run $RUN does not end with $CACHE_TAG (STEP_CACHE_SKIP=$STEP_CACHE_SKIP STEP_CACHE_COMP=$STEP_CACHE_COMP); set the env the run was started with"; exit 2
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
live_slots(){ # every GPU any live master's slot names (dispatchable or not)
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
free_spec(){ # every healthy CUDA device of NODES_ALL with < 1 GiB in use and both slot ports closed, that no live master
             # slot names and RESERVE does not hold, as single-GPU slots: 'l1:s0,s2;l5:s1' (CUDA index = g-th healthy board)
  local held node gl g out=""
  held=$(live_slots)
  for node in $NODES_ALL; do
    gl=$(timeout 60 ssh -n -o BatchMode=yes -o ConnectTimeout=15 "jiamingt@$node.mit.edu" "
      healthy=(\$(nvidia-smi --query-gpu=index,pstate --format=csv,noheader | awk -F', ' '\$2 !~ /reset/ {print \$1}'))
      for g in \"\${!healthy[@]}\"; do
        used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i \${healthy[\$g]} | tr -d ' ')
        [ \"\$used\" -lt 1024 ] || continue
        [ -z \"\$(ss -ltnH \"sport = :\$(($RPC_BASE + 50 + g))\")\" ] && [ -z \"\$(ss -ltnH \"sport = :\$(($WS_BASE + 50 + g))\")\" ] || continue
        printf '%s ' \"\$g\"
      done" 2>/dev/null) || continue
    local keep=()
    for g in $gl; do
      grep -qx "$node:$g" <<<"$held" && continue
      "$PY" -c "
import sys
node, g, reserve = sys.argv[1], sys.argv[2], sys.argv[3]
held = {(n, k) for item in reserve.split(';') if ':' in item for n, ks in [item.split(':', 1)] for k in ks.split(',')}
sys.exit(0 if (node, g) in held else 1)" "$node" "$g" "$RESERVE" && continue
      keep+=("s$g")
    done
    [[ ${#keep[@]} -gt 0 ]] && out+="${out:+;}$node:$(IFS=,; echo "${keep[*]}")"
  done
  echo "$out"
}
spec_count(){ echo "$1" | tr ';' '\n' | cut -d: -f2 | tr ',' '\n' | grep -c . || true; }
split_spec(){ # deal the single-GPU slots of a spec round-robin into $2 parts (node order kept): one spec per line
  "$PY" - "$1" "$2" <<'PYEOF'
import sys
spec, k = sys.argv[1], int(sys.argv[2])
slots = [(node, g) for item in spec.split(";") if ":" in item for node, gs in [item.split(":", 1)] for g in gs.split(",") if g]
parts = [dict() for _ in range(k)]
for i, (node, g) in enumerate(slots):
    parts[i % k].setdefault(node, []).append(g)
for part in parts:
    print(";".join(f"{node}:{','.join(gs)}" for node, gs in part.items()))
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
  local spec=$1 busy node g bad="" nodes gl
  busy=$(busy_gpus)
  while read -r node g; do
    if grep -qx "$node:$g" <<<"$busy"; then bad+=" $node:$g"; fi
  done < <(spec_gpus "$spec")
  [[ -z $bad ]] || { echo "REFUSE: GPUs held by a live master (occupied, draining, or reachable by its queue):$bad"; return 3; }
  nodes=$(spec_gpus "$spec" | awk '{print $1}' | sort -u)
  for node in $nodes; do
    gl=$(spec_gpus "$spec" | awk -v n="$node" '$1 == n {printf "%s ", $2}')
    ssh -n -o BatchMode=yes -o ConnectTimeout=20 "jiamingt@$node.mit.edu" "
      healthy=(\$(nvidia-smi --query-gpu=index,pstate --format=csv,noheader | awk -F', ' '\$2 !~ /reset/ {print \$1}'))
      rc=0
      for g in $gl; do
        phys=\${healthy[\$g]:-}
        if [ -z \"\$phys\" ]; then echo \"REFUSE: $node has no healthy board for CUDA device \$g\"; rc=3; continue; fi
        used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i \$phys | tr -d ' ')
        if [ \"\$used\" -ge 1024 ]; then echo \"REFUSE: $node CUDA \$g (board \$phys) has \${used} MiB in use\"; rc=3; fi
      done
      exit \$rc" || return 3
  done
  log "requested GPUs are free of live masters and empty on their nodes: $spec"
}
runs_pending(){ # the WAIT_RUNS that are not over yet: a live master, or a worker (run_unit_xpl.sh of its run root) on a node
  local run node pending="" pat
  for run in $WAIT_RUNS; do
    if [[ -s $HOME/robodojo_master/$run/master.pid ]] && kill -0 "$(cat "$HOME/robodojo_master/$run/master.pid")" 2>/dev/null; then
      pending+=" $run(master)"
    elif [[ ${REQUIRE_FINISHED:-0} == 1 ]]; then
      # a run that has not started yet (no state) or has unfinished units is still ahead of us in the queue
      st=$HOME/robodojo_master/$run/state.json
      if [[ ! -f $st ]]; then pending+=" $run(not started)"
      elif ! "$PY" -c "import json, sys; s = json.load(open(sys.argv[1])); sys.exit(0 if all(u['state'] in ('done', 'failed') for u in s['units']) else 1)" "$st"; then
        pending+=" $run(unfinished)"
      fi
    fi
  done
  # one ssh per node for every run: '[r]obodojo_eval/<run> ' never matches the remote shell's own command line
  pat=$(for run in $WAIT_RUNS; do printf '[r]obodojo_eval/%s |' "$run"; done); pat=${pat%|}
  for node in $NODES_ALL; do
    # an unreachable node cannot hold the wait forever (its GPUs are not in free-spec either): noted, not counted
    n=$(timeout 60 ssh -n -o BatchMode=yes -o ConnectTimeout=15 "jiamingt@$node.mit.edu" "ps -eo args | grep -E -c '$pat' || true" 2>/dev/null) \
      || { log "waiter: $node unreachable (not counted)" >&2; continue; }
    [[ ${n:-0} -gt 0 ]] && pending+=" $node(${n}workers)"
  done
  echo "${pending# }"
}
SHARD_ARGS=(); [[ -n $SHARD_PLAN ]] && SHARD_ARGS=(--shard-plan "$SHARD_PLAN")
COMMON=(--worker run_unit_xpl.sh --xpl-root "$XPL_ROOT" --xpl-action-type ee --sec-per-step 4 --checkpoint-root "$CKPT"
        --xpl-overrides "$OVERRIDES" --worker-env "SANA_EVAL_XPL_WS=$XPL_WS"
        --order dimension --pin-longest "$PIN_LONGEST" "${SHARD_ARGS[@]}" --support-arm-embodiments on
        --dimension-order "$DIM_ORDER" "${PORT_ARGS[@]}")
SMOKE_RUN_ROOT=${SMOKE_RUN_ROOT:-/homes/jiamingt/robodojo_eval/${RUN}_smoke1gpu}

cmd=${1:-help}; shift || true
case $cmd in
  check-sync) check_sync ;;
  free-spec) spec=$(free_spec); echo "$spec"; log "$(spec_count "$spec") free GPUs" >&2 ;;
  smoke-gpu)
    spec=${1:?'NODE:GPU, e.g. l25:5'}; task=${2:-insert_key}; eps=${3:-1}
    node=${spec%%:*}; gpu=${spec##*:}
    [[ $gpu =~ ^[0-7]$ ]] || { echo "GPU index must be 0..7, got '$gpu'"; exit 2; }
    check_sync; check_cache_tag
    if live_slots | grep -qx "$node:$gpu"; then echo "REFUSE: $node GPU $gpu belongs to a slot of a live master"; exit 3; fi
    seed=${SMOKE_SEED:-$SEED}; [[ $seed =~ ^[0-9]+$ ]] || { echo "SMOKE_SEED must be an integer, got '$seed'"; exit 2; }
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
SANA_EVAL_CHECKPOINT_ROOT='$CKPT' SANA_EVAL_XPL_ROOT='$XPL_ROOT' SANA_EVAL_XPL_WS='$XPL_WS' \
SANA_EVAL_XPL_EXTRA_OVERRIDES='$OVERRIDES' SANA_EVAL_XPL_ACTION_TYPE=ee SEC_PER_STEP=4 $layout_env \
  nohup setsid bash ~/robodojo_install/eval/run_unit_xpl.sh '$SMOKE_RUN_ROOT' '$uid' '$task' $seed '$eps' $gpu $gpu $SMOKE_RPC_PORT $SMOKE_WS_PORT 1 \
  > '$udir/nohup.log' 2>&1 < /dev/null &
sleep 2; kill -0 \$! 2>/dev/null && echo \"dispatched \$! on \$(hostname -s) GPU $gpu (server+client), ports $SMOKE_RPC_PORT/$SMOKE_WS_PORT\" || echo died"
    ssh -n -o BatchMode=yes -o ConnectTimeout=20 "jiamingt@$node.mit.edu" "$remote"
    printf '%s\n' "node=$node gpu=$gpu task=$task eps=$eps run_root=$SMOKE_RUN_ROOT unit=$uid overrides=$OVERRIDES started=$(date -u +%FT%TZ)" >"$SCRATCH/smoke1gpu_$task.txt"
    log "smoke-gpu launched; follow with: bash $0 smoke-status $task" ;;
  smoke-status)
    task=${1:-insert_key}; uid=${SMOKE_UNIT_ID:-${task}__s${SMOKE_SEED:-$SEED}}; dst=$SCRATCH/smoke1gpu_$uid; mkdir -p "$dst"
    rsync -az --exclude 'upstream_eval_result/**/*.mp4' "$NFS_USER_HOST:$SMOKE_RUN_ROOT/units/$uid/a1/" "$dst/" 2>/dev/null || true
    echo "== status.txt"; cat "$dst/status.txt" 2>/dev/null || echo "(none yet)"
    echo "== server log ([SANA_WAM] lines, errors)"
    grep -h "\[SANA_WAM\]\|Traceback\|Error\|error:" "$dst/server/server.stdout.log" "$dst/server/server.stderr.log" 2>/dev/null | cut -c1-600 | head -n 30 || true
    echo "== client progress: $(cat "$dst"/client/*/client.stdout.log 2>/dev/null | grep -a -o 'env0 step' | wc -l) simulation steps"
    echo "== result.json"; cat "$dst/result.json" 2>/dev/null || echo "(none yet)" ;;
  start)
    spec=${1:-}; [[ -n $spec ]] || spec=$(free_spec)
    [[ -n $spec ]] || { echo "no free GPU"; exit 3; }
    check_sync; check_cache_tag; check_pairs_free "$spec"
    # EXTRA_MASTER_ARGS: e.g. "--seeds 1" (with a RUN override)
    bash "$EVAL/start_master.sh" "$RUN" "${COMMON[@]}" --seeds "$SEED" ${EXTRA_MASTER_ARGS:-} --nodes "$(nodes_of "$spec")" --node-pairs "$spec"
    log "started $RUN on $(spec_count "$spec") GPUs: '$spec'" ;;
  resume-seeds)   # every seed of SEEDS with a state but no live master: lift its pause, resume it on its OWN slots, then deal
                  # the GPUs still free round-robin to the resumed runs (control/add_slots)
    todo=()
    for s in $SEEDS; do
      st=$HOME/robodojo_master/$(run_of "$s"); [[ -f $st/state.json ]] || continue
      [[ -s $st/master.pid ]] && kill -0 "$(cat "$st/master.pid")" 2>/dev/null && continue
      todo+=("$s")
    done
    [[ ${#todo[@]} -gt 0 ]] || { log "no stopped seed run to resume in [$SEEDS]"; exit 0; }
    rc=0
    for s in "${todo[@]}"; do
      rm -f "$HOME/robodojo_master/$(run_of "$s")/control/pause"
      if env -u RUN -u SCRATCH SEED="$s" bash "$0" resume; then log "seed $s: resumed on its own slots"; else log "seed $s: resume refused"; rc=3; fi
    done
    sleep 60   # the resumed masters sweep their slots before the free GPUs are measured
    spec=$(free_spec); n=$(spec_count "$spec")
    if (( n > 0 )); then
      mapfile -t parts < <(split_spec "$spec" "${#todo[@]}")
      for i in "${!todo[@]}"; do
        s=${todo[$i]}; part=${parts[$i]}; [[ -n $part ]] || continue
        ctl=$HOME/robodojo_master/$(run_of "$s")/control; mkdir -p "$ctl"
        printf '%s\n' "$part" >"$ctl/add_slots.tmp" && mv -f "$ctl/add_slots.tmp" "$ctl/add_slots"
        log "seed $s: add_slots '$part' ($(spec_count "$part") GPUs)"
      done
    fi
    exit $rc ;;
  start-seeds)
    spec=${1:-}; [[ -n $spec ]] || spec=$(free_spec)
    todo=(); for s in $SEEDS; do [[ -f $HOME/robodojo_master/$(run_of "$s")/state.json ]] || todo+=("$s"); done
    [[ ${#todo[@]} -gt 0 ]] || { log "every seed of [$SEEDS] already has a run"; exit 0; }
    (( $(spec_count "$spec") >= ${#todo[@]} )) || { echo "only $(spec_count "$spec") GPUs for ${#todo[@]} seeds"; exit 3; }
    mapfile -t parts < <(split_spec "$spec" "${#todo[@]}")
    rc=0
    for i in "${!todo[@]}"; do
      s=${todo[$i]}; part=${parts[$i]}
      if env -u RUN -u SCRATCH SEED="$s" bash "$0" start "$part"; then log "seed $s: started on $(spec_count "$part") GPUs"
      else log "seed $s: start refused"; rc=3; fi
    done
    exit $rc ;;
  resume)   # default spec = this run's OWN slots (a resumed seed never picks up another seed's GPUs); pass a spec to grow
    spec=${1:-}
    [[ -n $spec ]] || spec=$("$PY" -c "
import json, sys, collections
own = collections.OrderedDict()
for sl in json.load(open(sys.argv[1]))['slots']:
    own.setdefault(sl['node'], []).append(f\"s{sl['server_gpu']}\")
print(';'.join(f'{n}:{\",\".join(g)}' for n, g in own.items()))" "$STATE/state.json")
    [[ -n $spec ]] || { echo "no slots in $STATE/state.json"; exit 2; }
    check_sync; check_cache_tag
    bash "$EVAL/start_master.sh" "$RUN" --resume "${COMMON[@]}" --seeds "$SEED" ${EXTRA_MASTER_ARGS:-} --nodes "$(nodes_of "$spec")" --node-pairs "$spec" ;;
  add-free)
    [[ -s $STATE/master.pid ]] && kill -0 "$(cat "$STATE/master.pid")" 2>/dev/null || { echo "no live master for $RUN"; exit 1; }
    [[ ! -e $STATE/control/add_slots ]] || { echo "an add_slots request is still pending"; exit 1; }
    spec=$(free_spec); [[ -n $spec ]] || { echo "no free GPU"; exit 0; }
    mkdir -p "$STATE/control"; printf '%s\n' "$spec" >"$STATE/control/add_slots.tmp" && mv -f "$STATE/control/add_slots.tmp" "$STATE/control/add_slots"
    log "add_slots '$spec' ($(spec_count "$spec") GPUs)" ;;
  wait-check)   # one probe of what the waiter waits for
    pending=$(runs_pending); log "waited-for runs still active: ${pending:-none}"
    spec=$(free_spec); log "free now: $(spec_count "$spec") GPUs ($spec)" ;;
  wait-start)
    sub=${1:-status}; pidf=$SEEDS_SCRATCH/waiter.pid; wlog=$SEEDS_SCRATCH/waiter.log
    case $sub in
      start)
        if [[ -s $pidf ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then echo "waiter already running (pid $(cat "$pidf")), log $wlog"; exit 0; fi
        if [[ ${WAIT_ACTION:-start-seeds} == start-seeds ]]; then
          for s in $SEEDS; do [[ -f $HOME/robodojo_master/$(run_of "$s")/state.json ]] && { echo "seed $s already has a run ($(run_of "$s")); nothing to wait for (WAIT_ACTION=resume-seeds resumes stopped runs)"; exit 1; }; done
        fi
        check_sync; check_cache_tag
        nohup setsid bash "$0" wait-loop >>"$wlog" 2>&1 </dev/null &
        echo $! >"$pidf"; log "waiter started (pid $!): waits for [$WAIT_RUNS], then starts seeds [$SEEDS] ($(run_of "{N}")); log $wlog" ;;
      stop) if [[ -s $pidf ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then kill -TERM "$(cat "$pidf")"; log "waiter stopped"; else echo "waiter not running"; fi ;;
      status) if [[ -s $pidf ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then echo "waiter running (pid $(cat "$pidf"))"; else echo "waiter not running"; fi
              tail -n 8 "$wlog" 2>/dev/null || true ;;
      *) echo "wait-start start|stop|status"; exit 2 ;;
    esac ;;
  wait-loop)
    log "waiter: every ${WAIT_INTERVAL}s until [$WAIT_RUNS] are over for $WAIT_QUIET_CHECKS checks in a row, then start seeds [$SEEDS] on every free GPU (>= $MIN_GPUS), dealt round-robin"
    quiet=0; tries=0
    while :; do
      if [[ ${WAIT_ACTION:-start-seeds} == start-seeds ]]; then
        started=0; for s in $SEEDS; do [[ -f $HOME/robodojo_master/$(run_of "$s")/state.json ]] && started=$((started + 1)); done
        if (( started > 0 && tries == 0 )); then log "waiter: $started of the seed runs already exist; exiting"; break; fi
      else
        live=0; for s in $SEEDS; do st=$HOME/robodojo_master/$(run_of "$s"); [[ -s $st/master.pid ]] && kill -0 "$(cat "$st/master.pid")" 2>/dev/null && live=$((live + 1)); done
        if (( live > 0 && tries == 0 )); then log "waiter: $live seed masters already live; exiting"; break; fi
      fi
      pending=$(runs_pending)
      if [[ -n $pending ]]; then quiet=0; log "waiter: still running: $pending"
      else
        quiet=$((quiet + 1)); log "waiter: all waited-for runs are over ($quiet/$WAIT_QUIET_CHECKS quiet checks)"
        if (( quiet >= WAIT_QUIET_CHECKS )); then
          spec=$(free_spec); n=$(spec_count "$spec")
          if (( n < MIN_GPUS )); then log "waiter: only $n free GPUs ($spec) < MIN_GPUS $MIN_GPUS; waiting"
          else
            log "waiter: ${WAIT_ACTION:-start-seeds} for seeds [$SEEDS] with $n free GPUs: $spec"
            if bash "$0" "${WAIT_ACTION:-start-seeds}" "$spec"; then log "waiter: started; done"; break; fi
            tries=$((tries + 1)); log "waiter: start refused (attempt $tries); re-probing in 60 s"
            (( tries >= 5 )) && { log "waiter: giving up after 5 refused starts"; exit 1; }
            sleep 60; continue
          fi
        fi
      fi
      sleep "$WAIT_INTERVAL"
    done ;;
  wait-grow)   # background, one shot: once every WAIT_RUNS run is completely over (as wait-start judges it), deal every free
               # GPU round-robin to the LIVE seed runs of SEEDS through their control/add_slots, then exit
    sub=${1:-status}; pidf=$SEEDS_SCRATCH/grow.pid; glog=$SEEDS_SCRATCH/grow.log
    case $sub in
      start)
        if [[ -s $pidf ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then echo "grow waiter already running (pid $(cat "$pidf")), log $glog"; exit 0; fi
        nohup setsid bash "$0" grow-loop >>"$glog" 2>&1 </dev/null &
        echo $! >"$pidf"; log "grow waiter started (pid $!): waits for [$WAIT_RUNS], then adds the free GPUs to seeds [$SEEDS]; log $glog" ;;
      stop) if [[ -s $pidf ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then kill -TERM "$(cat "$pidf")"; log "grow waiter stopped"; else echo "grow waiter not running"; fi ;;
      status) if [[ -s $pidf ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then echo "grow waiter running (pid $(cat "$pidf"))"; else echo "grow waiter not running"; fi
              tail -n 6 "$glog" 2>/dev/null || true ;;
      *) echo "wait-grow start|stop|status"; exit 2 ;;
    esac ;;
  grow-loop)
    log "grow: every ${WAIT_INTERVAL}s until [$WAIT_RUNS] are over for $WAIT_QUIET_CHECKS checks in a row, then deal the free GPUs to the live seed runs of [$SEEDS]"
    quiet=0
    while :; do
      live=()
      for s in $SEEDS; do st=$HOME/robodojo_master/$(run_of "$s"); [[ -s $st/master.pid ]] && kill -0 "$(cat "$st/master.pid")" 2>/dev/null && live+=("$s"); done
      [[ ${#live[@]} -gt 0 ]] || { log "grow: no live seed run left; exiting"; break; }
      pending=$(runs_pending)
      if [[ -n $pending ]]; then quiet=0; log "grow: still running: $pending"
      else
        quiet=$((quiet + 1)); log "grow: waited-for runs are over ($quiet/$WAIT_QUIET_CHECKS quiet checks)"
        if (( quiet >= WAIT_QUIET_CHECKS )); then
          spec=$(free_spec); n=$(spec_count "$spec")
          if (( n == 0 )); then log "grow: no free GPU right now; waiting"
          else
            mapfile -t parts < <(split_spec "$spec" "${#live[@]}")
            for i in "${!live[@]}"; do
              s=${live[$i]}; part=${parts[$i]}; [[ -n $part ]] || continue
              ctl=$HOME/robodojo_master/$(run_of "$s")/control; mkdir -p "$ctl"
              if [[ -e $ctl/add_slots ]]; then log "grow: seed $s still has a pending add_slots; its share is skipped"; continue; fi
              printf '%s\n' "$part" >"$ctl/add_slots.tmp" && mv -f "$ctl/add_slots.tmp" "$ctl/add_slots"
              log "grow: seed $s add_slots '$part' ($(spec_count "$part") GPUs)"
            done
            log "grow: done"; break
          fi
        fi
      fi
      sleep "$WAIT_INTERVAL"
    done ;;
  dry)
    spec=${1:-l1:s0,s1;l5:s0}; st=$SCRATCH/dry_state; rm -rf "$st"; mkdir -p "$st"
    CHECKOUT=${ROBODOJO_CHECKOUT_X5:-$HOME/zekail/sana_wam_sim_benchmarks/RoboDojo-official-clone-20260823}
    "$PY" "$EVAL/master.py" --run-root "/homes/jiamingt/robodojo_eval/${RUN}_dry" --checkout "$CHECKOUT" --state-dir "$st" \
         --dry-run --dry-sleep 0.05 --poll 1 "${COMMON[@]}" --seeds "$SEED" ${EXTRA_MASTER_ARGS:-} --nodes "$(nodes_of "$spec")" --node-pairs "$spec" >"$st/stdout.log" 2>&1 || true
    grep -h "planned\|dispatch order\|ALL UNITS SETTLED" "$st/master.log" | cut -c1-400 | head -6
    grep -h "Traceback\|Error\|error:" "$st/stdout.log" | head -5 || true
    "$PY" - "$st/state.json" <<'PYEOF'
import collections, json, sys
s = json.load(open(sys.argv[1]))
print("slots:", [(x["slot_id"], x["server_gpu"], x["client_gpu"], x["rpc_port"], x["ws_port"]) for x in s["slots"]])
by = collections.Counter(u["state"] for u in s["units"])
print("units", len(s["units"]), "episodes", sum(u["episodes_per_task"] for u in s["units"]), "states", dict(by))
PYEOF
    ;;
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
        sl = next(x for x in s["slots"] if x["slot_id"] == u["slot"]); run.append(f"{u['unit_id'].rsplit('__s', 1)[0]}@{u['slot']}({int((now - sl['dispatched_at']) / 60)}m)")
print("slots:", len(s["slots"]), [x["slot_id"] for x in s["slots"]])
print("running:", " ".join(run))
PYEOF
    echo "== last completions"; grep -h "DONE " "$STATE/master.log" | tail -n 8 | cut -c1-140 || true ;;
  compare)
    ts=$(date -u +%Y%m%dT%H%M); out=$HOME/zekail/robodojo_results/per_task_${RUN}_vs_cachefree_seeds_$ts.md
    if ! "$PY" "$EVAL/compare_runs.py" "$RUN" "${BASELINES[@]}" --results-root "$HOME/robodojo_master" --runs-subdir . --out "$out" >"$SCRATCH/compare.stdout" 2>&1; then
      tail -n 5 "$SCRATCH/compare.stdout"; exit 1
    fi
    echo "columns: this run (fixed-step cache) | cache-free seed 0 | seed 1 | seed 2   (SR% / score%)"
    grep -h "tasks compared" "$SCRATCH/compare.stdout" | cut -c1-160
    grep "^| Precision\|^| Memory\|^| Open\|^| Long-Horizon\|^| Generalization\|↳\|overall" "$out" | cut -c1-240
    echo "-> $out" ;;
  watch-seeds)   # monitoring helper: block until a seed run needs attention, print why, exit. Events: a problem line in a
                 # master.log (failure, re-queue, traceback, server fail/timeout, stale, prologue), a master that died,
                 # a queue that ran empty for the first time (tail time), a run that settled, or $1 hours (default 3) elapsed
    hours=${1:-3}; deadline=$(( $(date +%s) + hours * 3600 ))
    # case-SENSITIVE (a case-insensitive FAIL matches every 'failed 0' status line); master.log spells RE-QUEUE and requeue
    pattern='FAIL|failed [1-9]|quarantin|Traceback|stale|server_timeout|server_fail|prologue|RE-QUEUE|requeue|ALL UNITS SETTLED'
    declare -A off
    for s in $SEEDS; do f=$HOME/robodojo_master/$(run_of "$s")/master.log; off[$s]=$( [[ -f $f ]] && wc -l <"$f" || echo 0 ); done
    while :; do
      events=""
      for s in $SEEDS; do
        r=$(run_of "$s"); st=$HOME/robodojo_master/$r; f=$st/master.log
        [[ -f $f ]] || continue
        total=$(wc -l <"$f")
        new=$(sed -n "$(( ${off[$s]} + 1 )),${total}p" "$f" | grep -v '^units:' | grep -E "$pattern" | cut -c1-260 | head -n 12 || true)
        off[$s]=$total   # every line is judged once (a filtered first retry is not re-read next minute)
        if [[ ${QUIET_FIRST_RETRY:-0} == 1 && -n $new ]]; then
          # an infrastructure episode (NFS / ssh stall) re-queues many units once or goes stale on an unreachable node:
          # note re-queues with <= 1 real and <= 4 infrastructure failures (and their slot-draining lines), wake on the rest
          mild='RE-QUEUE .*real failures [01]/3, infrastructure [0-4]/9|slot [^ ]+ quarantined after [0-9]+ consecutive infrastructure failures'
          noted=$(grep -E "$mild" <<<"$new" | cut -c1-200 || true)
          [[ -n $noted ]] && while read -r line; do log "watch-seeds: seed $s noted: $line"; done <<<"$noted" >>"$SEEDS_SCRATCH/first_retries.log"
          new=$(grep -v -E "$mild|slot [^ ]+ draining" <<<"$new" || true)
        fi
        [[ -n $new ]] && events+=$'\n'"seed $s log:"$'\n'"$new"
        if ! { [[ -s $st/master.pid ]] && kill -0 "$(cat "$st/master.pid")" 2>/dev/null; }; then
          grep -q "ALL UNITS SETTLED" "$f" || events+=$'\n'"seed $s: MASTER NOT RUNNING (no ALL UNITS SETTLED in its log)"
        fi
        if grep -q "^units: queued 0 " "$st/STATUS.md" 2>/dev/null && [[ ! -e $SEEDS_SCRATCH/queue_empty.$s ]]; then
          date -u +%FT%TZ >"$SEEDS_SCRATCH/queue_empty.$s"
          events+=$'\n'"seed $s: queue empty -> tail time ($(grep -m1 '^units:' "$st/STATUS.md" | cut -c1-120))"
        fi
      done
      if [[ -n $events ]]; then log "watch-seeds: attention$events"; bash "$0" status-seeds; exit 0; fi
      if (( $(date +%s) >= deadline )); then log "watch-seeds: ${hours} h heartbeat, nothing flagged"; bash "$0" status-seeds; exit 0; fi
      sleep 60
    done ;;
  status-seeds)
    for s in $SEEDS; do r=$(run_of "$s"); printf 'seed %s  %s\n' "$s" "$(grep -m1 '^units:' "$HOME/robodojo_master/$r/STATUS.md" 2>/dev/null || echo 'not started')"; done ;;
  compare-seeds)   # the cached seeds merged vs the cache-free seeds merged (compare_runs.py RUN_A+RUN_B merges episodes)
    ts=$(date -u +%Y%m%dT%H%M); out=$HOME/zekail/robodojo_results/per_task_seeds_${CKPT_LABEL}${CACHE_TAG}_vs_cachefree_$ts.md
    cached=$(for s in $SEEDS; do printf '%s+' "$(run_of "$s")"; done); cached=${cached%+}
    free=$(IFS=+; echo "${BASELINES[*]}")
    if ! "$PY" "$EVAL/compare_runs.py" "$cached" "$free" --results-root "$HOME/robodojo_master" --runs-subdir . --out "$out" >"$SEEDS_SCRATCH/compare.stdout" 2>&1; then
      tail -n 5 "$SEEDS_SCRATCH/compare.stdout"; exit 1
    fi
    echo "columns: fixed-step cache, seeds [$SEEDS] merged | cache-free seeds 0+1+2 merged   (SR% / score%)"
    grep -h "tasks compared" "$SEEDS_SCRATCH/compare.stdout" | cut -c1-200
    grep "^| Precision\|^| Memory\|^| Open\|^| Long-Horizon\|^| Generalization\|↳\|overall" "$out" | cut -c1-240
    echo "-> $out" ;;
  official-summary-seeds)
    runs=$(for s in $SEEDS; do printf '%s ' "$(run_of "$s")"; done)
    RUN=$(run_of "${SEEDS%% *}") SUMMARY_RUNS="${runs% }" CKPT_LABEL=$CKPT_LABEL SCRATCH=$SEEDS_SCRATCH \
      bash "$(dirname "$0")/vanilla34k_eefabs_f33fps8.sh" official-summary ;;
  tail)   # the TAIL TEMPLATE (user 2026-09-25); see eval/tail_shards.py
    sub=${1:?plan|launch|status|more-substitutes|fill|fill-status|invalidate|assemble}; shift
    targs=(); topts=()
    while [[ $# -gt 0 ]]; do case $1 in --dry-run|--force) topts+=("$1"); shift ;; --*) topts+=("$1" "$2"); shift 2 ;; *) targs+=(--task "$1"); shift ;; esac; done
    [[ ${#targs[@]} -gt 0 ]] || targs=(--task play_tic_tac_toe --task imitate_sorting_sequence)
    SANA_EVAL_XPL_WS=$XPL_WS "$PY" "$EVAL/tail_shards.py" --run "$RUN" --driver "$0" "${targs[@]}" "${topts[@]}" "$sub" ;;   # the tool reads the node-local partial results of THIS workspace
  status) sed -n 1,40p "$STATE/STATUS.md"; echo ...; tail -n 6 "$STATE/master.log" ;;
  watch) tail -n 0 -F "$STATE/master.log" | grep -E --line-buffered -i "DONE |FAIL|failed [1-9]|quarantin|ALL UNITS SETTLED|Traceback|WARN slot|stale|server_timeout|server_fail|prologue|control: |requeue" ;;
  server-log)
    uid=${1:?unit id}; run=${2:-$RUN}; a=${3:-1}; dst=$SCRATCH/serverlog_${run}_${uid}_a$a; mkdir -p "$dst"
    rsync -az "$NFS_USER_HOST:/homes/jiamingt/robodojo_eval/$run/units/$uid/a$a/server/" "$dst/"
    grep -h "\[SANA_WAM\]\|WARNING\|Traceback\|Error\|error" "$dst"/server.stdout.log "$dst"/server.stderr.log 2>/dev/null | cut -c1-600 | head -n 30 ;;
  official-summary)   # the vanilla34k driver's official aggregation (layout shards concatenated) for this RUN
    RUN=$RUN SUMMARY_RUNS=${SUMMARY_RUNS:-$RUN} CKPT_LABEL=$CKPT_LABEL SCRATCH=$SCRATCH bash "$(dirname "$0")/vanilla34k_eefabs_f33fps8.sh" official-summary ;;
  *) sed -n 2,34p "$0"; exit 1 ;;
esac
