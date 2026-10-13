#!/usr/bin/env bash
# Campaign driver (runs on x5): the robot_base_eef checkpoint `s55k_eef` (HF logits/..._eef_s157000_lr1e-4cos_wd1e-2_alw1_epoch9_step55000,
# run R12 at step 55000; snapshot ~/checkpoints/huggingface/logits/s55k_eef assembled by download_partial_ckpt.sh with the R12 bundle's
# resolved config) on the XPolicyLab lane, sampled like the s50k `fs8` baseline (CFG 6 @ 10 NFE, flow_shift 8) so the two checkpoints compare
# directly. ACTION_TYPE=joint (default) executes the predicted joint targets (the EEF slots are conditioned on, reconstructed, not emitted);
# ACTION_TYPE=ee sends {left,right}_ee_pose targets the evaluator solves by IK per tick. One master per action type (RUN name carries it).
#
#   bash s55k_eef.sh check-sync        worker scripts + adapter on NFS byte-identical to x5 (start/smoke refuse otherwise; the EEF adapter must be synced first)
#   bash s55k_eef.sh free-pairs        usable GPU pairs no live master can dispatch onto
#   bash s55k_eef.sh dry               local --dry-run of the plan (51 configs, category order)
#   bash s55k_eef.sh smoke 'l5:1'      1 episode of insert_key on that pair -> run <RUN>_smoke; then `server-log insert_key__s0 <RUN>_smoke`
#   bash s55k_eef.sh start '<pairs>'   full run on those pairs (precision first, then memory, open, long-horizon, generalization)
#   bash s55k_eef.sh status | report [--no-progress] | watch | server-log UNIT [RUN] [ATTEMPT]
#   ACTION_TYPE=ee bash s55k_eef.sh smoke 'l5:1'   (same subcommands for the ee lane; RUN becomes ..._ee)
set -euo pipefail
ACTION_TYPE=${ACTION_TYPE:-joint}
case $ACTION_TYPE in joint|ee) ;; *) echo "ACTION_TYPE must be joint or ee"; exit 1;; esac
RUN=${RUN:-official_seed0_s55k_eef_xpl_cfg6_nfe10_fs8_${ACTION_TYPE}}
BASE_RUN=official_seed0_s50k_xpl_cfg6_nfe10_fs3v8
CKPT=/homes/jiamingt/checkpoints/huggingface/logits/s55k_eef
NORM_SHA=817630bbd5d644ba7c663703a0dce7bfefb836011aba9abe7aeeb5e3f906aab6      # the R12 yaml pin (rot6d-normalized robot_base_eef artifact)
OVERRIDES="sampling_steps=10 cfg_scale=6 flow_shift=8 normalization_sha256=$NORM_SHA"
PORT_ARGS=(--rpc-port-base 61000 --ws-port-base 62000)
XPL_ROOT=/homes/jiamingt/xpl_zekai/XPolicyLab
USABLE="l5:0,1,2,3;l7:0,1,2,3;l8:0,1,2,3;l25:0,1,2,3;l3:1,2,3;l26:1,2,3"
NFS_USER_HOST=jiamingt@l5.mit.edu
EVAL=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STATE=$HOME/robodojo_master/$RUN
SCRATCH=${SCRATCH:-$HOME/robodojo_master/.scratch/$RUN}; mkdir -p "$SCRATCH"
PY=${MASTER_PY:-$HOME/miniforge3/envs/sana-wam/bin/python}
CHECKOUT=${ROBODOJO_CHECKOUT_X5:-$HOME/zekail/sana_wam_sim_benchmarks/RoboDojo-official-clone-20260823}
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
nodes_of(){ echo "$1" | tr ';' '\n' | cut -d: -f1 | sed '/^$/d' | paste -sd, -; }

check_sync(){
  local diff
  diff=$(rsync -n -c -i "$EVAL/run_unit_xpl.sh" "$EVAL/run_unit.sh" "$NFS_USER_HOST:/homes/jiamingt/robodojo_install/eval/" | grep -v '^\.' || true)
  diff+=$(rsync -n -c -i -r --exclude __pycache__ "$HOME/zekail/XPolicyLab/policy/SANA_WAM/" "$NFS_USER_HOST:$XPL_ROOT/policy/SANA_WAM/" | grep -v '^\.d\|^cd\|^\.f' || true)
  if [[ -n $diff ]]; then echo "NFS copies differ from x5 (rsync them first; the EEF adapter is needed on NFS):"; echo "$diff"; exit 1; fi
  log "worker scripts and adapter in sync with NFS"
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
check_pairs_free(){
  local spec=$1 busy; busy=$(busy_pairs)
  "$PY" - "$spec" "$busy" <<'PYEOF'
import sys
spec, busy = sys.argv[1], set(sys.argv[2].split()); bad = []
for item in spec.split(";"):
    node, ks = item.split(":"); bad += [f"{node}:{k}" for k in ks.split(",") if f"{node}:{k}" in busy]
if bad: sys.exit(f"pairs held by a live master (occupied, draining, or reachable by its queue): {bad}")
print("requested pairs are free of live masters")
PYEOF
}
COMMON=(--worker run_unit_xpl.sh --xpl-root "$XPL_ROOT" --xpl-action-type "$ACTION_TYPE" --sec-per-step 4 --checkpoint-root "$CKPT"
        --xpl-overrides "$OVERRIDES" --order dimension --dimension-order precision,memory,open,long-horizon,generalization "${PORT_ARGS[@]}")

cmd=${1:-help}; shift || true
case $cmd in
  check-sync) check_sync ;;
  free-pairs) free_pairs ;;
  dry)
    st=$SCRATCH/dry_state; rm -rf "$st"; mkdir -p "$st"
    "$PY" "$EVAL/master.py" --run-root "/homes/jiamingt/robodojo_eval/${RUN}_dry" --checkout "$CHECKOUT" --state-dir "$st" --dry-run --dry-sleep 0.05 --poll 1 \
         "${COMMON[@]}" --nodes "$(nodes_of "$USABLE")" --node-pairs "$USABLE" >"$st/stdout.log" 2>&1 || true
    grep -h "planned\|dispatch order\|ALL UNITS SETTLED" "$st/master.log" | cut -c1-260 | head -6; grep -h "Traceback\|Error\|error:" "$st/stdout.log" | head -5 || true
    "$PY" -c "import json,sys; u=json.load(open(sys.argv[1])); print('units', len(u), 'episodes', sum(x['episodes_per_task'] for x in u))" "$st/units.json" ;;
  smoke)
    spec=${1:?pairs spec, e.g. 'l5:1'}; check_sync; check_pairs_free "$spec"
    bash "$EVAL/start_master.sh" "${RUN}_smoke" "${COMMON[@]}" --nodes "$(nodes_of "$spec")" --node-pairs "$spec" --tasks insert_key --episodes-per-task 1 --pin-longest 0 --poll 20
    echo "expect in the server log: 'ready: ... steps=10 cfg_scale=6.0 flow_shift=8.0 ... action_type=$ACTION_TYPE state_profile=robot_base_eef include_eef=True'"
    echo "and '[SANA_WAM] eef check ep N: observed *_ee_pose vs FK(measured joints): left <1 mm ...' -> bash $0 server-log insert_key__s0 ${RUN}_smoke" ;;
  start)
    spec=${1:?pairs spec}; check_sync; check_pairs_free "$spec"
    bash "$EVAL/start_master.sh" "$RUN" "${COMMON[@]}" --nodes "$(nodes_of "$spec")" --node-pairs "$spec"
    log "started $RUN (action_type=$ACTION_TYPE) on '$spec'" ;;
  status) sed -n 1,40p "$STATE/STATUS.md"; echo ...; tail -n 6 "$STATE/master.log" ;;
  report)
    echo "== $RUN (action_type=$ACTION_TYPE)  $(date -u +%FT%TZ)"; grep -m1 "^units:" "$STATE/STATUS.md" || true
    "$PY" - "$STATE/state.json" <<'PYEOF'
import json, sys, time, collections
s = json.load(open(sys.argv[1])); now = time.time()
by = collections.defaultdict(collections.Counter)
for u in s["units"]: by[u["dimension"]][u["state"]] += 1
for d in ("precision", "memory", "open", "long-horizon", "generalization"): print(f"{d:15s} done {by[d]['done']:2d} running {by[d]['running']:2d} queued {by[d]['queued']:2d} failed {by[d]['failed']:2d}")
run = []
for u in s["units"]:
    if u["state"] == "running":
        sl = next(x for x in s["slots"] if x["slot_id"] == u["slot"]); run.append(f"{u['unit_id'].replace('__s0', '')}@{u['slot']}({int((now - sl['dispatched_at']) / 60)}m)")
print("running:", " ".join(run))
PYEOF
    if [[ ${1:-} != --no-progress ]]; then "$PY" "$EVAL/progress_xpl.py" "$STATE" 2>/dev/null | grep "running" | sed 's/__s0//' | awk '{printf "  %-40s %s eps, succ %s, score_sum %s\n",$1,$3,$4,$5}'; fi
    ts=$(date -u +%Y%m%dT%H%M); out=$HOME/zekail/robodojo_results/per_task_${RUN}_vs_s50k_$ts.md
    "$PY" "$EVAL/compare_runs.py" "$RUN" "$BASE_RUN@fs8" official_seed0_ep10clip1_xpl_nfe10_cfg6 --results-root "$HOME/robodojo_master" --runs-subdir . --out "$out" >/dev/null 2>&1 || echo "compare_runs failed"
    echo "== dimension means over tasks finished in both (columns: s55k_eef $ACTION_TYPE | s50k fs8 | ep10clip1 cfg6) -> $out"
    grep "^| Precision\|^| Memory\|^| Open\|^| Long-Horizon\|^| Generalization\|overall\|tasks compared" "$out" | cut -c1-200
    echo "== last completions"; grep -h "DONE " "$STATE/master.log" | tail -n 8 | cut -c1-140 ;;
  watch) tail -n 0 -F "$STATE/master.log" | grep -E --line-buffered "DONE |FAIL|failed [1-9]|quarantin|ALL UNITS SETTLED|Traceback|WARN slot|stale|server_timeout|server_fail|prologue|control: |requeue" ;;
  server-log)
    uid=${1:?unit id}; run=${2:-$RUN}; a=${3:-1}; dst=$SCRATCH/serverlog_${run}_${uid}_a$a; mkdir -p "$dst"
    rsync -az "$NFS_USER_HOST:/homes/jiamingt/robodojo_eval/$run/units/$uid/a$a/server/" "$dst/"
    grep -h "SANA_WAM\] ready\|SANA_WAM\] normalization\|EEF slots\|eef check\|anchor ep\|WARNING\|Error\|error" "$dst"/server.stdout.log "$dst"/server.stderr.log 2>/dev/null | head -n 30 ;;
  *) sed -n 2,14p "$0"; exit 1 ;;
esac
