#!/usr/bin/env bash
# Campaign driver (runs on x5): anchor_source A/B on the PRECISION category (robodojo_results/HANDOFF.md section 6b).
# Baseline = the `fs8` setting of official_seed0_s50k_xpl_cfg6_nfe10_fs3v8: checkpoint sft_p120k_1e-4cos_clip1_s50k,
# CFG 6 @ 10 NFE, flow_shift 8, XPolicyLab lane, measured-joint anchor. This run repeats the 8 precision tasks
# (50 episodes each, seed 0, same layouts) with the adapter anchoring the model's state input AND its joint deltas on
# the LAST EXECUTED COMMAND (deploy.yml `anchor_source=last_command`) -- the convention of the training corpus, whose
# recorded state is the previous frame's command. fasten_screws runs as 4 layout shards and build_tower as 2 so the
# category finishes in ~4 h on 12 GPU pairs (a shard restarts the per-episode diffusion seed counter, so those two tasks
# do not share per-layout noise seeds with the baseline; the other six do). `ANCHOR=last_command_clamped RUN=<name>`
# reuses the script for the clamped arm.
#
#   bash anchor_lastcmd_s50k_fs8_precision.sh check-sync        worker scripts + adapter on NFS byte-identical to x5 (start/smoke refuse otherwise)
#   bash anchor_lastcmd_s50k_fs8_precision.sh free-pairs        usable GPU pairs no live master can dispatch onto
#   bash anchor_lastcmd_s50k_fs8_precision.sh dry               local --dry-run of the plan (12 units, dispatch order, drop_slots control)
#   bash anchor_lastcmd_s50k_fs8_precision.sh smoke 'l3:3'      1 episode of insert_key on that pair -> run <RUN>_smoke; then `server-log insert_key__s0 <RUN>_smoke`
#   bash anchor_lastcmd_s50k_fs8_precision.sh start '<pairs>'   make the s50k master forget the pairs (control/drop_slots), then start this run on them
#   bash anchor_lastcmd_s50k_fs8_precision.sh status            STATUS.md head + last master.log lines
#   bash anchor_lastcmd_s50k_fs8_precision.sh report [--no-progress]   per-task table vs the fs8 / fs3 / cfg6 baselines (precision rows) + running units
#   bash anchor_lastcmd_s50k_fs8_precision.sh watch             stream completions / failures / controls (Ctrl-C to stop)
#   bash anchor_lastcmd_s50k_fs8_precision.sh server-log UNIT [RUN] [ATTEMPT]   pull a unit's policy-server log from NFS (ready line, anchor lag lines)
#   bash anchor_lastcmd_s50k_fs8_precision.sh release           when this run is finished: hand its pairs back to the s50k master (control/add_slots)
set -euo pipefail
ANCHOR=${ANCHOR:-last_command}
RUN=${RUN:-anchor_lastcmd_s50k_fs8_precision}
BASE_RUN=official_seed0_s50k_xpl_cfg6_nfe10_fs3v8
CKPT=/homes/jiamingt/checkpoints/huggingface/logits/sft_p120k_1e-4cos_clip1_s50k
NORM_SHA=b133f38f701533da72fa84c51b5cfe264b4ea237571b3a163660c0a92e97292c
OVERRIDES="sampling_steps=10 cfg_scale=6 normalization_sha256=$NORM_SHA flow_shift=8 anchor_source=$ANCHOR"
# the s50k campaign holds rpc 63000+10k / ws 64000+10k; the cfg6 master that used 61000/62000 has exited
PORT_ARGS=(--rpc-port-base 61000 --ws-port-base 62000)
XPL_ROOT=/homes/jiamingt/xpl_zekai/XPolicyLab
USABLE="l5:0,1,2,3;l7:0,1,2,3;l8:0,1,2,3;l25:0,1,2,3;l3:1,2,3;l26:1,2,3"   # l2/l4/l6 GPUs need reset, l0 taken, l3/l26 pair 0 taken
NFS_USER_HOST=jiamingt@l5.mit.edu
EVAL=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STATE=$HOME/robodojo_master/$RUN
BASE_STATE=$HOME/robodojo_master/$BASE_RUN
SCRATCH=${SCRATCH:-$HOME/robodojo_master/.scratch/$RUN}; mkdir -p "$SCRATCH"
PY=${MASTER_PY:-$HOME/miniforge3/envs/sana-wam/bin/python}
CHECKOUT=${ROBODOJO_CHECKOUT_X5:-$HOME/zekail/sana_wam_sim_benchmarks/RoboDojo-official-clone-20260823}
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
nodes_of(){ echo "$1" | tr ';' '\n' | cut -d: -f1 | sed '/^$/d' | paste -sd, -; }
master_alive(){ [[ -s $1/master.pid ]] && kill -0 "$(cat "$1/master.pid")" 2>/dev/null; }

check_sync(){   # worker + adapter execute from the NFS copies: refuse to dispatch with an unsynced edit
  local diff
  diff=$(rsync -n -c -i "$EVAL/run_unit_xpl.sh" "$EVAL/run_unit.sh" "$NFS_USER_HOST:/homes/jiamingt/robodojo_install/eval/" | grep -v '^\.' || true)
  diff+=$(rsync -n -c -i -r --exclude __pycache__ "$HOME/zekail/XPolicyLab/policy/SANA_WAM/" \
            "$NFS_USER_HOST:$XPL_ROOT/policy/SANA_WAM/" | grep -v '^\.d\|^cd\|^\.f' || true)
  if [[ -n $diff ]]; then echo "NFS copies differ from x5 (rsync them first):"; echo "$diff"; exit 1; fi
  log "worker scripts and adapter in sync with NFS"
}

# pairs held by live masters on x5: occupied/draining slots, plus EVERY slot of a live master that still has a queued
# unit it may dispatch (not behind control/pause or control/pause.<its setting>). Prints 'node:k' lines.
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
    s = json.loads(st.read_text())
    cdir = pidf.parent / "control"
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
usable, busy = sys.argv[1], set(sys.argv[2].split())
out = []
for item in usable.split(";"):
    node, ks = item.split(":")
    free = [k for k in ks.split(",") if f"{node}:{k}" not in busy]
    if free: out.append(f"{node}:{','.join(free)}")
print(";".join(out))
PYEOF
}

check_pairs_free(){   # abort if any requested pair is held by a live master
  local spec=$1 busy; busy=$(busy_pairs)
  "$PY" - "$spec" "$busy" <<'PYEOF'
import sys
spec, busy = sys.argv[1], set(sys.argv[2].split())
bad = []
for item in spec.split(";"):
    node, ks = item.split(":")
    bad += [f"{node}:{k}" for k in ks.split(",") if f"{node}:{k}" in busy]
if bad:
    sys.exit(f"pairs held by a live master (occupied, draining, or reachable by its queue): {bad}")
print("requested pairs are free of live masters")
PYEOF
}

slots_left_in(){   # slot ids of STATE_JSON that belong to a 'node:k,...' spec
  "$PY" - "$1" "$2" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1])); spec = sys.argv[2]
want = {f"{n}:g{2*int(k)}{2*int(k)+1}" for item in spec.split(";") for n, ks in [item.split(":")] for k in ks.split(",")}
print(" ".join(sorted(sl["slot_id"] for sl in s["slots"] if sl["slot_id"] in want)))
PYEOF
}

take_from_base(){   # the s50k master forgets the (idle) pairs, so a later `resume-setting fs3` can never dispatch onto them
  local spec=$1
  if ! master_alive "$BASE_STATE"; then log "$BASE_RUN master not alive; nothing to drop"; return; fi
  if [[ -z $(slots_left_in "$BASE_STATE/state.json" "$spec") ]]; then log "$BASE_RUN holds none of '$spec'; nothing to drop"; return; fi
  mkdir -p "$BASE_STATE/control"; echo "$spec" >"$BASE_STATE/control/drop_slots"
  log "control/drop_slots '$spec' written for $BASE_RUN; waiting for its master to consume it (it polls every 60 s)"
  for _ in $(seq 1 100); do [[ -e $BASE_STATE/control/drop_slots ]] || break; sleep 2; done
  if [[ -e $BASE_STATE/control/drop_slots ]]; then
    rm -f "$BASE_STATE/control/drop_slots"
    echo "the $BASE_RUN master did not consume control/drop_slots within 200 s: it predates the knob."
    echo "restart it in place first (workers untouched): bash $EVAL/campaigns/s50k_cfg6_nfe10_fs3v8.sh resume"; exit 1
  fi
  grep -h "control: drop_slots" "$BASE_STATE/master.log" | tail -n 1
  local left
  for _ in $(seq 1 45); do left=$(slots_left_in "$BASE_STATE/state.json" "$spec"); [[ -z $left ]] && break; sleep 2; done
  if [[ -n $left ]]; then echo "pairs still slots of $BASE_RUN (running or draining there): $left -- pick other pairs"; exit 1; fi
  log "$BASE_RUN no longer holds '$spec'"
}

# one array: --xpl-overrides takes ONE space-separated string
COMMON=(--worker run_unit_xpl.sh --xpl-root "$XPL_ROOT" --sec-per-step 4 --checkpoint-root "$CKPT"
        --xpl-overrides "$OVERRIDES" --dimensions precision --shard-plan "fasten_screws=4,build_tower=2" "${PORT_ARGS[@]}")

cmd=${1:-help}; shift || true
case $cmd in
  check-sync) check_sync ;;
  free-pairs) free_pairs ;;
  dry)
    st=$SCRATCH/dry_state; rm -rf "$st"; mkdir -p "$st/control"
    echo 'l26:1,2;l3:3' >"$st/control/drop_slots"     # exercise the drop_slots control (12 units settle in seconds, so it is pre-created)
    "$PY" "$EVAL/master.py" --run-root "/homes/jiamingt/robodojo_eval/${RUN}_dry" --checkout "$CHECKOUT" --state-dir "$st" --dry-run --dry-sleep 0.05 --poll 1 \
         "${COMMON[@]}" --nodes "$(nodes_of "$USABLE")" --node-pairs "$USABLE" >"$st/stdout.log" 2>&1 || true
    grep -h "planned\|dispatch order\|control: drop_slots\|ALL UNITS SETTLED" "$st/master.log" | head -12
    grep -h "Traceback\|Error\|error:" "$st/stdout.log" | head -5 || true
    echo "-- units"; "$PY" - "$st/units.json" <<'PYEOF'
import json, sys
u = json.load(open(sys.argv[1]))
for x in u:
    ids = x.get("layout_ids") or []
    print(f"  {x['unit_id']:40s} {x['dimension']:10s} eps {x['episodes_per_task']:3d}" + (f"  layouts {min(ids)}..{max(ids)}" if ids else ""))
print("  total", len(u), "units", sum(x["episodes_per_task"] for x in u), "episodes")
PYEOF
    echo "-- slots left after drop_slots: $("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))['slots']))" "$st/state.json")"
    ;;
  smoke)
    spec=${1:?pairs spec, e.g. 'l3:3'}; check_sync; check_pairs_free "$spec"
    bash "$EVAL/start_master.sh" "${RUN}_smoke" "${COMMON[@]}" --nodes "$(nodes_of "$spec")" --node-pairs "$spec" \
         --tasks insert_key --episodes-per-task 1 --pin-longest 0 --poll 20
    echo "expect in the server log: '[SANA_WAM] ready: ... steps=10 cfg_scale=6.0 flow_shift=8.0 ... anchor_source=$ANCHOR'"
    echo "and one '[SANA_WAM] anchor ep 0 chunk K src=$ANCHOR: |last_cmd-measured| ...' line per chunk after the first:"
    echo "  bash $0 server-log insert_key__s0 ${RUN}_smoke"
    ;;
  start)
    spec=${1:?pairs spec, e.g. 'l5:1,2,3;l7:0,1,2,3;l8:0,2,3;l25:0,1'}; check_sync; check_pairs_free "$spec"
    take_from_base "$spec"
    bash "$EVAL/start_master.sh" "$RUN" "${COMMON[@]}" --nodes "$(nodes_of "$spec")" --node-pairs "$spec"
    log "started $RUN (anchor_source=$ANCHOR) on '$spec'. Watch: bash $0 watch ; table: bash $0 report ; when finished: bash $0 release"
    ;;
  status) sed -n 1,40p "$STATE/STATUS.md"; echo ...; tail -n 6 "$STATE/master.log" ;;
  report)
    echo "== $RUN (anchor_source=$ANCHOR)  $(date -u +%FT%TZ)"; grep -m1 "^units:" "$STATE/STATUS.md" || true
    "$PY" - "$STATE/state.json" <<'PYEOF'
import json, sys, time
s = json.load(open(sys.argv[1])); now = time.time()
for st in ("done", "running", "queued", "failed"):
    ids = [u["unit_id"].replace("__s0", "") for u in s["units"] if u["state"] == st]
    if ids: print(f"{st:8s} {len(ids):2d}: " + " ".join(ids))
run = []
for u in s["units"]:
    if u["state"] == "running":
        sl = next(x for x in s["slots"] if x["slot_id"] == u["slot"]); run.append(f"{u['unit_id'].replace('__s0', '')}@{u['slot']}({int((now - sl['dispatched_at']) / 60)}m)")
print("running:", " ".join(run))
PYEOF
    if [[ ${1:-} != --no-progress ]]; then "$PY" "$EVAL/progress_xpl.py" "$STATE" 2>/dev/null | grep "running" | sed 's/__s0//' | awk '{printf "  %-40s %s eps, succ %s, score_sum %s\n",$1,$3,$4,$5}'; fi
    ts=$(date -u +%Y%m%dT%H%M); out=$HOME/zekail/robodojo_results/per_task_${RUN}_vs_fs8_$ts.md
    "$PY" "$EVAL/compare_runs.py" "$RUN" "$BASE_RUN@fs8" "$BASE_RUN@fs3" official_seed0_ep10clip1_xpl_nfe10_cfg6 \
        official_seed0_ep10clip1_xpl_nfe10+official_seed0_ep10clip1_xpl_nfe10_lang --results-root "$HOME/robodojo_master" --runs-subdir . --out "$out" >/dev/null 2>&1 || echo "compare_runs failed"
    echo "== precision (columns: $ANCHOR | s50k fs8 measured | s50k fs3 measured | ep10clip1 cfg6@10 | ep10clip1 nfe10) -> $out"
    grep "^| Precision\|^| build_tower\|^| deposit_coin\|^| fasten_screws\|^| insert_key\|^| insert_tubes\|^| play_Xylophone\|^| plug_in_charger\|^| pour_balls\|tasks compared" "$out" | cut -c1-230
    echo "== last completions"; grep -h "DONE " "$STATE/master.log" | tail -n 8 | cut -c1-140 ;;
  watch)
    tail -n 0 -F "$STATE/master.log" | grep -E --line-buffered "DONE |FAIL|failed [1-9]|quarantin|ALL UNITS SETTLED|Traceback|WARN slot|stale|server_timeout|prologue|control: |requeue" ;;
  server-log)
    uid=${1:?unit id, e.g. insert_key__s0}; run=${2:-$RUN}; a=${3:-1}
    dst=$SCRATCH/serverlog_${run}_${uid}_a$a; mkdir -p "$dst"
    rsync -az "$NFS_USER_HOST:/homes/jiamingt/robodojo_eval/$run/units/$uid/a$a/server/" "$dst/"
    ls -la "$dst"
    grep -h "SANA_WAM\] ready\|SANA_WAM\] normalization\|anchor ep" "$dst"/server.stdout.log | head -n 30
    echo "... $(grep -c 'anchor ep' "$dst"/server.stdout.log) anchor lines; stderr tail:"; tail -n 5 "$dst"/server.stderr.log ;;
  release)
    if master_alive "$STATE"; then
      running=$("$PY" -c "import json,sys;print(sum(1 for u in json.load(open(sys.argv[1]))['units'] if u['state'] in ('running','queued')))" "$STATE/state.json")
      [[ $running == 0 ]] || { echo "$RUN still has $running running/queued unit(s); release after it settles"; exit 1; }
    fi
    spec=$("$PY" -c "
import json,sys,collections; s=json.load(open(sys.argv[1])); d=collections.defaultdict(list)
for sl in s['slots']: d[sl['node']].append(str(sl['server_gpu']//2))
print(';'.join(f\"{n}:{','.join(sorted(set(k)))}\" for n,k in d.items()))" "$STATE/state.json")
    if master_alive "$BASE_STATE"; then
      mkdir -p "$BASE_STATE/control"; echo "$spec" >"$BASE_STATE/control/add_slots"; log "control/add_slots '$spec' written for $BASE_RUN (appended on its next poll)"
    else
      echo "$BASE_RUN master not alive; give the pairs back at its next resume with: --node-pairs '$spec'"
    fi ;;
  *) sed -n 2,24p "$0"; exit 1 ;;
esac
