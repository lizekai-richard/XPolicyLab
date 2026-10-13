#!/usr/bin/env bash
# Campaign driver (runs on x5): checkpoint sft_p120k_1e-4cos_clip1_s50k
# (HF logits/sana_rwm_policy_from_pretrained_robodojo_320px_1e-4_cosine_clip1_2node, work_dir/checkpoints/epoch_8_step_50000),
# CFG scale 6 @ 20 NFE, flow_shift 8 ONLY (single setting, label `fs8`, so results address as RUN@fs8 like the 10-NFE run),
# all 51 task configs (39 tasks incl. the 4 language tasks, seed 0), dispatched category by category
# (master --order dimension default: precision -> memory -> open -> long-horizon -> generalization).
# Reference: the same checkpoint at 10 NFE / flow_shift 8 scored 3.98 SR% / 7.45 score% (official_seed0_s50k_xpl_cfg6_nfe10_fs3v8@fs8).
#
#   bash s50k_cfg6_nfe20_fs8.sh verify-norm          normalization json on NFS: sha256 vs config.yaml + stats diff vs ep10clip1
#   bash s50k_cfg6_nfe20_fs8.sh free-pairs           usable GPU pairs no live master on x5 holds (spec you can pass to start/grow)
#   bash s50k_cfg6_nfe20_fs8.sh smoke 'l25:0'        1 episode of insert_key on that pair -> run <RUN>_smoke
#   bash s50k_cfg6_nfe20_fs8.sh dry                  local --dry-run of the full plan: unit count, dispatch order, add_slots
#   bash s50k_cfg6_nfe20_fs8.sh start 'l25:0,1,2,3;l3:1,2,3;l26:1,2,3'   start the campaign on those free pairs
#   bash s50k_cfg6_nfe20_fs8.sh grow  'l5:0,1'       append pairs to the running campaign (control/add_slots, no restart) as the anchor run frees l5/l7/l8
#   bash s50k_cfg6_nfe20_fs8.sh status               STATUS.md head + last master.log lines
#   bash s50k_cfg6_nfe20_fs8.sh resume               restart the master with its saved slots (workers untouched)
#   bash s50k_cfg6_nfe20_fs8.sh report [--no-progress]  progress per category, running units (+ in-progress eps), comparison table vs the 10-NFE run, last completions
#   bash s50k_cfg6_nfe20_fs8.sh watch                stream completions/failures (Ctrl-C to stop)
#   bash s50k_cfg6_nfe20_fs8.sh official-summary fs8 official summarize_result.py summary
#
# Results: ~/robodojo_master/<RUN>/{STATUS.md,RESULTS_fs8.md}; consolidate with
# eval/consolidate_results.py as for the other runs (see ~/zekail/robodojo_results/HANDOFF.md).
set -euo pipefail
RUN=${RUN:-official_seed0_s50k_xpl_cfg6_nfe20_fs8}
CKPT=/homes/jiamingt/checkpoints/huggingface/logits/sft_p120k_1e-4cos_clip1_s50k
EP10=/homes/jiamingt/checkpoints/huggingface/logits/sana_rwm_policy_from_pretrained_robodojo_320px_epoch10_step65690_lr5e-5_clip1
# normalization_sha256 recorded in the checkpoint's config.yaml; passed to the adapter as a pin so a wrong stats file fails fast
NORM_SHA=b133f38f701533da72fa84c51b5cfe264b4ea237571b3a163660c0a92e97292c
GLOBAL_OVERRIDES="sampling_steps=20 cfg_scale=6 normalization_sha256=$NORM_SHA"
SETTINGS="fs8:flow_shift=8"      # single setting; the label keeps unit ids and RUN@fs8 addressing consistent with the 10-NFE run
# anchor_lastcmd_s50k_fs8_precision holds rpc 61000+10k / ws 62000+10k on l5/l7/l8; disjoint bases mean the two
# masters can never collide on a port even once this run grows onto those pairs
PORT_ARGS=(--rpc-port-base 63000 --ws-port-base 64000)
XPL_ROOT=/homes/jiamingt/xpl_zekai/XPolicyLab
USABLE="l5:0,1,2,3;l7:0,1,2,3;l8:0,1,2,3;l25:0,1,2,3;l3:1,2,3;l26:1,2,3"   # l2/l4/l6 GPUs need reset, l0 taken, l3/l26 pair 0 taken
NFS_USER_HOST=jiamingt@l5.mit.edu
EVAL=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); INST=$(cd "$EVAL/.." && pwd)
STATE=$HOME/robodojo_master/$RUN
SCRATCH=${SCRATCH:-$HOME/robodojo_master/.scratch/$RUN}; mkdir -p "$SCRATCH"
PY=${MASTER_PY:-$HOME/miniforge3/envs/sana-wam/bin/python}
CHECKOUT=${ROBODOJO_CHECKOUT_X5:-$HOME/zekail/sana_wam_sim_benchmarks/RoboDojo-official-clone-20260823}
log(){ echo "[$(date -u +%FT%TZ)] $*"; }

nodes_of(){ echo "$1" | tr ';' '\n' | cut -d: -f1 | sed '/^$/d' | paste -sd, -; }

check_sync(){   # worker + adapter execute from the NFS copies: refuse to dispatch with an unsynced edit
  local diff
  diff=$(rsync -n -c -i "$EVAL/run_unit_xpl.sh" "$EVAL/run_unit.sh" "$NFS_USER_HOST:/homes/jiamingt/robodojo_install/eval/" | grep -v '^\.' || true)
  diff+=$(rsync -n -c -i -r --exclude __pycache__ --exclude tests "$HOME/zekail/XPolicyLab/policy/SANA_WAM/" \
            "$NFS_USER_HOST:$XPL_ROOT/policy/SANA_WAM/" | grep -v '^\.d\|^cd\|^\.f' || true)
  if [[ -n $diff ]]; then echo "NFS copies differ from x5 (rsync them first):"; echo "$diff"; exit 1; fi
  log "worker scripts and adapter in sync with NFS"
}

# pairs held by live masters on x5: occupied/draining slots, plus EVERY slot of a live, unpaused master that still has queued
# units (it may dispatch onto any free slot of its own). Prints 'node:k' lines.
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
    # a queued unit the master may still dispatch (not behind control/pause or control/pause.<its setting>)
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

# one array (not a word-split string): --xpl-overrides takes ONE space-separated string
COMMON=(--worker run_unit_xpl.sh --xpl-root "$XPL_ROOT" --sec-per-step 4 --checkpoint-root "$CKPT"
        --xpl-overrides "$GLOBAL_OVERRIDES" --settings "$SETTINGS" "${PORT_ARGS[@]}")

cmd=${1:-help}; shift || true
case $cmd in
  verify-norm)
    rsync -q "$NFS_USER_HOST:$CKPT/config.yaml" "$SCRATCH/config.yaml"
    rsync -q "$NFS_USER_HOST:$CKPT/normalization/robodojo_arx_x5_model_fps_25_f25_normalization.json" "$SCRATCH/norm_s50k.json"
    # The ep10clip1 snapshot was deleted from NFS 2026-09-15; its normalization artifact survives in the local
    # metadata archive, which is all this stats diff needs. NFS first, archive second, skip the diff if neither.
    EP10_ARCHIVE=$HOME/zekail/robodojo_results/checkpoints_meta/$(basename "$EP10")/normalization/robodojo_arx_x5_model_fps_25_f25_normalization.json
    rsync -q "$NFS_USER_HOST:$EP10/normalization/robodojo_arx_x5_model_fps_25_f25_normalization.json" "$SCRATCH/norm_ep10.json" 2>/dev/null \
      || { [[ -f $EP10_ARCHIVE ]] && cp "$EP10_ARCHIVE" "$SCRATCH/norm_ep10.json" && echo "ep10clip1 reference: local archive (its NFS copy was deleted 2026-09-15)"; } \
      || echo "ep10clip1 reference unavailable -- skipping the stats diff"
    # the artifact the user uploaded to the HF repo (normalization/action_mode/robot_base_eef/...), fetched by
    # ~/zekail/robodojo_install/launch_fetch_repo_subdir.sh into the snapshot's .upload_normalization/ staging dir
    UPLOAD="$CKPT/.upload_normalization/normalization/action_mode/robot_base_eef/robodojo_arx_x5_model_fps_25_f25_normalization.json"
    if rsync -q "$NFS_USER_HOST:$UPLOAD" "$SCRATCH/norm_upload.json" 2>/dev/null; then
      echo "HF-uploaded original sha256      : $(sha256sum "$SCRATCH/norm_upload.json" | cut -d' ' -f1)"
    fi
    cfg_sha=$(grep -m1 "normalization_sha256:" "$SCRATCH/config.yaml" | awk '{print $2}')
    file_sha=$(sha256sum "$SCRATCH/norm_s50k.json" | cut -d' ' -f1)
    echo "config.yaml normalization_sha256 : $cfg_sha"; echo "NFS normalization file sha256   : $file_sha"; echo "pin in this script (NORM_SHA)  : $NORM_SHA"
    [[ $cfg_sha == "$file_sha" && $cfg_sha == "$NORM_SHA" ]] || { echo "MISMATCH -- do not start"; exit 1; }
    if [[ -f $SCRATCH/norm_upload.json ]]; then cmp -s "$SCRATCH/norm_upload.json" "$SCRATCH/norm_s50k.json" && echo "OK: byte-identical to the HF-uploaded original" || { echo "MISMATCH vs the HF upload -- do not start"; exit 1; }; fi
    echo "OK: file matches config and the pin"
    [[ -f $SCRATCH/norm_ep10.json ]] || { echo "OK (stats diff skipped: no reference artifact)"; exit 0; }
    "$PY" - "$SCRATCH/norm_ep10.json" "$SCRATCH/norm_s50k.json" <<'PYEOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
diff = []
def walk(x, y, path):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)): walk(x.get(k), y.get(k), path + [k])
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y) and x and isinstance(x[0], dict):
        for i, (p, q) in enumerate(zip(x, y)): walk(p, q, path + [f"[{i}]"])
    elif x != y:
        diff.append(".".join(path))
walk(a, b, [])
stats = [d for d in diff if d.startswith("action.") or d in ("formula", "joint_target_mode", "action_mode", "action_representation", "eef_origin", "gripper")]
print(f"fields differing vs ep10clip1: {len(diff)} -> {diff}")
print("ACTION STATISTICS DIFFER" if stats else "action statistics identical to ep10clip1 (only cache metadata differs)")
PYEOF
    ;;
  free-pairs) free_pairs ;;
  smoke)
    spec=${1:?pairs spec, e.g. 'l8:0,2'}; check_sync; check_pairs_free "$spec"
    bash "$EVAL/start_master.sh" "${RUN}_smoke" "${COMMON[@]}" --nodes "$(nodes_of "$spec")" --node-pairs "$spec" \
         --tasks insert_key --episodes-per-task 1 --pin-longest 0 --poll 20
    echo "expect in the server logs: '[SANA_WAM] ready: ... steps=20 cfg_scale=6.0 flow_shift=8.0' and no normalization sha error"
    ;;
  dry)
    st=$SCRATCH/dry_state; rm -rf "$st"; mkdir -p "$st"
    "$PY" "$EVAL/master.py" --run-root "/homes/jiamingt/robodojo_eval/${RUN}_dry" --checkout "$CHECKOUT" --state-dir "$st" --dry-run --dry-sleep 0.05 --poll 1 \
         "${COMMON[@]}" --nodes "$(nodes_of "$USABLE")" --node-pairs "$USABLE" >"$st/stdout.log" 2>&1 &
    mpid=$!; sleep 6; mkdir -p "$st/control"; echo 'l3:0;l26:0' >"$st/control/add_slots"     # exercise runtime slot growth
    wait $mpid || true
    grep -h "settings:\|dispatch order\|planned\|control: add_slots\|ALL UNITS SETTLED" "$st/master.log" | head -12
    grep -h "Traceback\|Error\|error:" "$st/stdout.log" | head -5 || true
    echo "-- first 12 dispatches"; grep -h "dispatch " "$st/master.log" | head -12
    echo "-- units per dimension x setting"; "$PY" -c "
import json,collections,sys; u=json.load(open('$st/units.json')); c=collections.Counter((x['dimension'],x['setting']) for x in u)
[print(f'  {d:15s} {s:4s} {n:3d} units') for (d,s),n in sorted(c.items())]; print('  total',len(u),'units',sum(x['episodes_per_task'] for x in u),'episodes')"
    ls "$st" | grep -i "RESULTS_\|COMPARE" || true
    ;;
  start)
    spec=${1:?pairs spec, e.g. 'l8:0,2;l25:0,2;l3:1;l5:2'}; check_sync; check_pairs_free "$spec"
    bash "$0" verify-norm >/dev/null || { echo "verify-norm failed"; exit 1; }; log "normalization pin verified"
    bash "$EVAL/start_master.sh" "$RUN" "${COMMON[@]}" --nodes "$(nodes_of "$spec")" --node-pairs "$spec"
    log "started $RUN on pairs '$spec'. Grow later with: bash $0 grow '<spec>'   (free ones: bash $0 free-pairs)"
    ;;
  grow)
    spec=${1:?pairs spec, e.g. 'l7:1,3'}; check_pairs_free "$spec"
    [[ -s $STATE/master.pid ]] && kill -0 "$(cat "$STATE/master.pid")" 2>/dev/null || { echo "$RUN master is not running"; exit 1; }
    mkdir -p "$STATE/control"; echo "$spec" >"$STATE/control/add_slots"
    log "control/add_slots written; the master appends the pairs on its next poll (see master.log 'control: add_slots')"
    ;;
  resume)          # restart the master (e.g. after a master.py change) keeping every slot: workers are untouched, state.json is re-read
    if [[ -s $STATE/master.pid ]] && kill -0 "$(cat "$STATE/master.pid")" 2>/dev/null; then
      pid=$(cat "$STATE/master.pid"); kill -TERM "$pid"; log "TERM sent to master $pid; waiting for it to exit"
      for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
      kill -0 "$pid" 2>/dev/null && { echo "master $pid still alive after 120 s; not restarting"; exit 1; }
    fi
    spec=$("$PY" -c "
import json,sys,collections; s=json.load(open(sys.argv[1])); d=collections.defaultdict(list)
for sl in s['slots']: d[sl['node']].append(str(sl['server_gpu']//2))
print(';'.join(f\"{n}:{','.join(sorted(set(k)))}\" for n,k in d.items()))" "$STATE/state.json")
    log "resuming $RUN with its saved slots: $spec"
    bash "$EVAL/start_master.sh" "$RUN" --resume "${COMMON[@]}" --nodes "$(nodes_of "$spec")" --node-pairs "$spec" ;;
  status) sed -n 1,40p "$STATE/STATUS.md"; echo ...; tail -n 6 "$STATE/master.log" ;;
  report)   # one-shot progress report: per-category x setting counts, running units, comparison table, recent completions
            # 'report --no-progress' skips the rsync of in-progress episode counts from the nodes (~1-2 min)
    echo "== $RUN  $(date -u +%FT%TZ)"; grep -m1 "^units:" "$STATE/STATUS.md" || true
    "$PY" - "$STATE/state.json" <<'PYEOF2'
import json, sys, time, collections
s = json.load(open(sys.argv[1])); now = time.time()
labels = sorted({u.get("setting") or "-" for u in s["units"]})      # a run without --settings has one nameless column
by = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
for u in s["units"]: by[u["dimension"]][u.get("setting") or "-"][u["state"]] += 1
print(f"{'dimension':15s} " + "  ".join(f"{lbl:>4s} done/run/queue" for lbl in labels))
for d in ("precision", "memory", "open", "long-horizon", "generalization"):
    if d not in by: continue
    print(f"{d:15s} " + "  ".join(f"{lbl:>4s} {by[d][lbl]['done']:2d}/{by[d][lbl]['running']:2d}/{by[d][lbl]['queued']:2d}      " for lbl in labels))
run = []
for u in s["units"]:
    if u["state"] == "running":
        sl = next(x for x in s["slots"] if x["slot_id"] == u["slot"]); run.append(f"{u['unit_id'].replace('__s0', '')}@{u['slot']}({int((now - sl['dispatched_at']) / 60)}m)")
print("running:", " ".join(run))
PYEOF2
    if [[ ${1:-} != --no-progress ]]; then "$PY" "$EVAL/progress_xpl.py" "$STATE" 2>/dev/null | grep "running" | sed 's/__s0//' | awk '{printf "  %-40s %s eps, succ %s, score_sum %s\n",$1,$3,$4,$5}'; fi
    ts=$(date -u +%Y%m%dT%H%M); out=$HOME/zekail/robodojo_results/per_task_s50k_fs3_vs_fs8_partial_$ts.md
    mapfile -t LBL < <("$PY" -c "import json,sys;s=json.load(open(sys.argv[1]));print('\n'.join(sorted({u.get('setting') for u in s['units'] if u.get('setting')})))" "$STATE/state.json" | sed '/^$/d')
    COLS=(); if [[ ${#LBL[@]} -gt 0 ]]; then for l in "${LBL[@]}"; do COLS+=("$RUN@$l"); done; else COLS=("$RUN"); fi
    # the finished s50k fs8 run is the standing reference; skip it when RUN is that run itself
    [[ $RUN == official_seed0_s50k_xpl_cfg6_nfe10_fs3v8 ]] || COLS+=("official_seed0_s50k_xpl_cfg6_nfe10_fs3v8@fs8")
    "$PY" "$EVAL/compare_runs.py" "${COLS[@]}" official_seed0_ep10clip1_xpl_nfe10_cfg6 \
        official_seed0_ep10clip1_xpl_nfe10+official_seed0_ep10clip1_xpl_nfe10_lang official_seed0_ep10clip1_xpl_nfe50+official_seed0_ep10clip1_xpl_nfe50_lang \
        official_seed0_ep7_xpl+official_seed0_ep7_xpl_lang --results-root "$HOME/robodojo_master" --runs-subdir . --out "$out" >/dev/null 2>&1 || echo "compare_runs failed"
    echo "== dimension means over tasks finished in BOTH settings (columns: fs3 | fs8 | ep10clip1 cfg6@10 | ep10clip1 nfe10 | ep10clip1 nfe50 | ep7@50) -> $out"
    grep "^| Generalization\|^| Precision\|^| Long-Horizon\|^| Memory\|^| Open\|\u21b3 Generalization\|overall\|tasks compared" "$out" | cut -c1-200
    echo "== last completions"; grep -h "DONE " "$STATE/master.log" | tail -n 8 | cut -c1-140 ;;
  watch)    # stream completions/failures/slot changes (what the Claude session's Monitor did); Ctrl-C to stop
    tail -n 0 -F "$STATE/master.log" | grep -E --line-buffered "DONE |FAIL|failed [1-9]|quarantin|ALL UNITS SETTLED|Traceback|WARN slot|stale|server_timeout|prologue|control: |requeue" ;;
  official-summary)   # official summarize_result.py format for ONE setting (consolidates first); -> runs/<RUN>/OFFICIAL_SUMMARY_<label>_<ts>.md
    lbl=${1:?fs3|fs8}; "$PY" "$EVAL/consolidate_results.py" >/dev/null 2>&1 || true
    out=$SCRATCH/official_${RUN}_$lbl; rm -rf "$out"
    "$PY" - "$RUN" "$lbl" "$out" <<'PYEOF3'
import json, os, re, shutil, sys
run, lbl, out = sys.argv[1:4]; rd = os.path.expanduser(f"~/zekail/robodojo_results/runs/{run}")
st = json.load(open(f"{rd}/state.json")); n = 0
for u in st["units"]:
    if u["state"] != "done" or u.get("setting") != lbl: continue
    label = os.path.basename(u.get("checkpoint_root") or st["checkpoint_root"]) + "_" + lbl
    a = f"{rd}/per_unit/{u['unit_id']}/a{u['result']['attempt']}"
    m = re.search(r"run_id=(\S+)", open(f"{a}/status.txt").read()); rid = m.group(1) if m else f"a{u['result']['attempt']}"
    d = f"{out}/eval_result/RoboDojo/{u['task']}/SANA_WAM/arx_x5/{u['eval_seed']}_ckpt_name={label},action_type=joint/{rid}"
    os.makedirs(d, exist_ok=True); shutil.copy2(f"{a}/_result.json", d); n += 1
print(f"{lbl}: {n} finished units laid out")
PYEOF3
    ROBODOJO_EVAL_ROOT=$out/eval_result/RoboDojo "$PY" "$CHECKOUT/scripts/internal/summarize_result.py" 2>&1 | { grep -v '^[^|].*\(MISMATCH\|mp4\|video\)' || true; } | tail -2
    ts=$(date -u +%Y%m%dT%H%M); mkdir -p "$HOME/zekail/robodojo_results/runs/$RUN"; cp "$out/eval_result/RoboDojo/_summary.md" "$HOME/zekail/robodojo_results/runs/$RUN/OFFICIAL_SUMMARY_${lbl}_$ts.md"
    echo "-> ~/zekail/robodojo_results/runs/$RUN/OFFICIAL_SUMMARY_${lbl}_$ts.md" ;;
  *) sed -n 2,20p "$0"; exit 1 ;;
esac
