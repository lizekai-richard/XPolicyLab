#!/usr/bin/env bash
# Campaign driver (runs on x5): the FIRST non-SANA policy -- OpenWAM (HF logits/openwam_robodojo), fine-tuned from
# OpenWAM-Alpha-Pretrain-Foundation-Model on RoboDojo arx_x5/sim (35 tasks / 3500 episodes -- the SAME corpus as the SANA line),
# 60k micro-steps / 15k optimizer updates, bf16. Video backbone Wan2.2-TI2V-5B, action backbone 1.0 B, 12.407 B params total.
# EE-space policy (action_type=ee, 80-dim unified action -> EEF20), 32-step chunks, the bundle's own denoise_steps: 10.
# The checkpoint is self-contained (DiT/VAE/text-encoder weights all inside checkpoint_step_60000.safetensors).
#
# Two kinds of video are kept for every episode:
#   * SIMULATOR video -- the evaluator's 3 camera mp4s per episode, as in every previous run (client/.../upstream_eval_result/).
#   * The WORLD MODEL's own predicted VIDEO LATENTS -- <attempt>/latents/batch<RRRR>/env<EE>_chunk<CCC>.npz, fp16 (48,3,24,20),
#     ~102 kB compressed per env per chunk (~3-4 GB for the whole campaign). NO VAE decode happens during the run: the batch
#     path never decodes, and dumping the latent costs one device->host copy per chunk. Decode offline with the bundle's Wan2.2 VAE.
#
# Runs in the node-local `openwam` conda env (/dataset2/jiamingt/robodojo/envs/openwam), NOT sana-wam -- the two dependency
# sets conflict (timm 0.6.13 vs >=1.0.20). The worker selects policy + interpreter through --worker-env; with none of those set
# run_unit_xpl.sh behaves exactly as it always has for SANA_WAM.
#
# References on the same 39 tasks (SANA lane, 10 NFE): s65k 4.76 / 8.42, s65k video-only-CFG 4.61 / 8.58, s50k@fs8 3.98 / 7.45,
# ep10clip1+CFG6 3.45 / 6.00, ep10clip1 no CFG 1.77 / 5.01.
# All 51 task configs (39 tasks incl. the 4 language tasks, seed 0), dispatched category by category
# (master --order dimension default: precision -> memory -> open -> long-horizon -> generalization).
#
#   bash openwam_robodojo.sh check                 checkpoint bundle on NFS + node-local openwam env + adapter import
#   bash openwam_robodojo.sh free-pairs           usable GPU pairs no live master on x5 holds (spec you can pass to start/grow)
#   bash openwam_robodojo.sh smoke 'l25:0'        1 episode of insert_key on that pair -> run <RUN>_smoke
#   bash openwam_robodojo.sh dry                  local --dry-run of the full plan: unit count, dispatch order, add_slots
#   bash openwam_robodojo.sh start 'l25:0,1,2,3;l3:1,2,3;l26:1,2,3'   start the campaign on those free pairs
#   bash openwam_robodojo.sh grow  'l5:0,1'       append pairs to the running campaign (control/add_slots, no restart) as the anchor run frees l5/l7/l8
#   bash openwam_robodojo.sh status               STATUS.md head + last master.log lines
#   bash openwam_robodojo.sh resume               restart the master with its saved slots (workers untouched)
#   bash openwam_robodojo.sh report [--no-progress]  progress per category, running units (+ in-progress eps), comparison table vs the 10-NFE run, last completions
#   bash openwam_robodojo.sh watch                stream completions/failures (Ctrl-C to stop)
#   bash openwam_robodojo.sh official-summary fs8 official summarize_result.py summary
#
# Results: ~/robodojo_master/<RUN>/{STATUS.md,RESULTS_fs8.md}; consolidate with
# eval/consolidate_results.py as for the other runs (see ~/zekail/robodojo_results/HANDOFF.md).
set -euo pipefail
RUN=${RUN:-official_seed0_openwam_xpl_ee}
CKPT=/homes/jiamingt/checkpoints/huggingface/logits/openwam_robodojo
OPENWAM_PY=/dataset2/jiamingt/robodojo/envs/openwam/bin/python
EP10=/homes/jiamingt/checkpoints/huggingface/logits/sana_rwm_policy_from_pretrained_robodojo_320px_epoch10_step65690_lr5e-5_clip1
# OpenWAM takes its sampling knobs from the bundle's own config + configs/deploy.yaml (denoise_steps 10, 32-step chunk,
# dit_cache/compile/decode_video forced off by the adapter), so there is nothing to override here and no --settings label.
GLOBAL_OVERRIDES=""
# policy, interpreter and the latent dump reach run_unit_xpl.sh through the master's generic --worker-env passthrough
WORKER_ENV="SANA_EVAL_XPL_POLICY=OpenWAM SANA_EVAL_XPL_SERVER_PY=$OPENWAM_PY SANA_EVAL_XPL_LATENT_DIR=1"
# rpc 61000+10k / ws 62000+10k belong to the (paused) anchor_lastcmd master; this base keeps us disjoint from it
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
        --xpl-action-type ee --worker-env "$WORKER_ENV" "${PORT_ARGS[@]}")

cmd=${1:-help}; shift || true
case $cmd in
  check)        # bundle completeness on NFS + the node-local openwam env + the adapter's own import guard
    echo "-- checkpoint bundle: $CKPT"
    rsync -n --list-only -r "$NFS_USER_HOST:$CKPT/" 2>/dev/null \
      | awk '$1 ~ /^-/ {gsub(",","",$2); printf "   %10.2f MB  %s\n", $2/1048576, $5}' | sort -k2 -r | head -12
    for f in config.yaml normalization_stats.npy SHA256SUMS; do
      rsync -n --list-only "$NFS_USER_HOST:$CKPT/$f" >/dev/null 2>&1 && echo "   OK  $f" || { echo "   MISSING $f"; exit 1; }
    done
    echo "-- node-local openwam env + adapter import (on the first node of USABLE)"
    h=$(echo "$USABLE" | cut -d: -f1)
    ssh -n -o BatchMode=yes -o ConnectTimeout=15 "jiamingt@$h.mit.edu" \
      "source /dataset2/jiamingt/robodojo/activate_openwam.sh 2>/dev/null || { echo '   activate_openwam.sh MISSING on '\$(hostname -s); exit 1; }; \
       \$OPENWAM_PY -c \"
import os, sys, torch, openwam, XPolicyLab.policy.OpenWAM.model as m
print('   python      :', sys.version.split()[0], '| torch', torch.__version__)
print('   openwam     :', openwam.__file__)
print('   adapter     :', m.__file__)
print('   cuda        :', torch.cuda.is_available(), torch.cuda.device_count(), 'device(s)')
\"" || { echo "   env check FAILED"; exit 1; }
    echo "OK: bundle and node env ready"
    ;;
  free-pairs) free_pairs ;;
  smoke)
    spec=${1:?pairs spec, e.g. 'l8:0,2'}; check_sync; check_pairs_free "$spec"
    bash "$EVAL/start_master.sh" "${RUN}_smoke" "${COMMON[@]}" --nodes "$(nodes_of "$spec")" --node-pairs "$spec" \
         --tasks insert_key --episodes-per-task 1 --pin-longest 0 --poll 20
    echo "expect in the server logs: '[OpenWAM] video-latent dump ON -> .../latents' then '[OpenWAM] initialized | {\'denoise_steps\': 10, ...}' and latents under <attempt>/latents/"
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
    bash "$0" check >/dev/null || { echo "check failed -- run: bash $0 check"; exit 1; }; log "bundle + node env verified"
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
