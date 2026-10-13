#!/usr/bin/env bash
# Campaign driver (runs on x5): the Sana MoT dual-expert policy, checkpoint `mot_canvas_nsc_s36k` (HF
# logits/sft_robodojo_mot_jointabs_f25_openwam_nsc, the NSC 4-node run at its FINAL milestone epoch 10 / step 36,250;
# snapshot ~/checkpoints/huggingface/logits/mot_canvas_nsc_s36k, sha-verified) on the XPolicyLab lane through the
# policy/SANA_MOT adapter (--worker-env SANA_EVAL_XPL_POLICY=SANA_MOT).
# VIDEO LAYOUT: this checkpoint is the OpenWAM CANVAS (three cameras stretched into ONE 384x320 L-shape, encoded once, one
# prompt shared by both experts), NOT the 2x2 multiview default of rwm/mot. Its yaml predates model.extra.video_layout (it
# was trained at 295f48f15, inside the canvas-only window); the adapter resolves the canvas from data.type, and every
# launch here also passes `video_layout=openwam_canvas`, which the adapter ASSERTS against the checkpoint before any
# weight is read (a mismatch stops the server; it can never re-route the checkpoint).
# Weights: context_embedder text layout, action_state_as_context false, absolute joint targets (ACTION_TYPE joint).
# Sampling like the other NSC canvas runs (CFG 6 @ 10 NFE, flow_shift 8) so scores compare directly.
#
#   bash acfg1_foldclothes_canvas.sh check-sync           worker scripts + SANA_MOT + SANA_WAM (its runtime) on NFS == x5
#   bash acfg1_foldclothes_canvas.sh free-pairs           usable GPU pairs no live master can dispatch onto
#   bash acfg1_foldclothes_canvas.sh smoke-gpu 'l3:1' [TASK] [EPS]
#                                                    direct worker on ONE free GPU (policy server + Isaac client share it);
#                                                    refuses a GPU any live master slot covers, a GPU with >1 GiB in use,
#                                                    or a listening port; run <RUN>_smoke1gpu, unit <TASK>__s0
#   bash acfg1_foldclothes_canvas.sh smoke-status [TASK]  status / log tail / result of that direct smoke
#   bash acfg1_foldclothes_canvas.sh smoke 'l5:1'         1 episode of insert_key on a free PAIR through the master -> <RUN>_smoke
#   bash acfg1_foldclothes_canvas.sh dry ['<spec>']       local --dry-run of the plan on that slot spec (default 'l3:s1;l26:s0')
#   bash acfg1_foldclothes_canvas.sh start '<spec>'       full run (category order precision, memory, long-horizon, generalization, open);
#                                                    spec = pairs 'l5:0,1' and/or single GPUs 'l3:s1' (server + client on GPU 1)
#   bash acfg1_foldclothes_canvas.sh autogrow start|stop|status
#                                                    background loop (every 5 min): every USABLE pair no live master holds
#                                                    (no unit, not draining, its master has nothing dispatchable) is appended
#                                                    to this run through control/add_slots; stops when this run's master ends
#   bash acfg1_foldclothes_canvas.sh compare               per-task + dimension means over the tasks this run has finished, vs the
#                                                    single-stream canvas s30000, abs_nsc_s30k and the best run r18_s61k
#   bash acfg1_foldclothes_canvas.sh status | report | watch | server-log UNIT [RUN] [ATTEMPT]
#   VIDEO_ONLY_CFG=1 bash ...                        guide the video stream only (action_cfg_scale=1)
set -euo pipefail
CKPT_LABEL=${CKPT_LABEL:-mot_canvas_nsc_s36k}
RUN=${RUN:-acfg1_fold_${CKPT_LABEL}_xpl_cfg6_nfe10_fs8}
CKPT=/homes/jiamingt/checkpoints/huggingface/logits/$CKPT_LABEL
# The artifact this snapshot's own training yaml pins (grep normalization_sha256 <snapshot>/config.yaml after any re-download).
NORM_SHA=${NORM_SHA:-3cd3ce1d949e68ae5164373ebcc1161c01eb44c145f01ef09c4a3225bba480be}
OVERRIDES="sampling_steps=10 cfg_scale=6 action_cfg_scale=1 flow_shift=8 normalization_sha256=$NORM_SHA video_layout=openwam_canvas"
[[ ${VIDEO_ONLY_CFG:-0} == 1 ]] && OVERRIDES="$OVERRIDES action_cfg_scale=1"
WORKER_ENV="SANA_EVAL_XPL_POLICY=SANA_MOT"
# Pair k of a node uses rpc base + 10k: 61500..61530 / 62500..62530, disjoint from the live ranges of the other campaigns
# (61000.., 63000.., 65000..). The direct single-GPU smoke uses 61590 / 62590.
PORT_ARGS=(--rpc-port-base 61900 --ws-port-base 63900)
SMOKE_RPC_PORT=${SMOKE_RPC_PORT:-61990}; SMOKE_WS_PORT=${SMOKE_WS_PORT:-63990}
XPL_ROOT=/homes/jiamingt/xpl_zekai/XPolicyLab
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
  for pol in SANA_MOT SANA_WAM; do
    diff+=$(rsync -n -c -i -r --exclude __pycache__ --exclude .pytest_cache "$HOME/zekail/XPolicyLab/policy/$pol/" "$NFS_USER_HOST:$XPL_ROOT/policy/$pol/" | grep -v '^\.d\|^cd\|^\.f' || true)
  done
  if [[ -n $diff ]]; then echo "NFS copies differ from x5 (rsync them first):"; echo "$diff"; exit 1; fi
  log "worker scripts and the SANA_MOT + SANA_WAM adapters are in sync with NFS"
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
COMMON=(--worker run_unit_xpl.sh --xpl-root "$XPL_ROOT" --xpl-action-type joint --sec-per-step 4 --checkpoint-root "$CKPT"
        --xpl-overrides "$OVERRIDES" --worker-env "$WORKER_ENV" --order dimension
        --dimension-order precision,memory,long-horizon,generalization,open "${PORT_ARGS[@]}")
SMOKE_RUN_ROOT=/homes/jiamingt/robodojo_eval/${RUN}_smoke1gpu
# Baselines: the single-stream SANA_WAM canvas line at step 30000 (same NSC canvas data), the single-stream 3-view absolute
# line on the same NSC data (abs_nsc_s30k, same normalization 3cd3ce1d), and the best official score so far (r18_s61k).
BASELINES=(official_seed0_openwam_canvas_s30000_xpl_cfg6_nfe10_fs8 official_seed0_abs_nsc_s30k_xpl_cfg6_nfe10_fs8@fs8
           official_seed0_r18_s61k_xpl_cfg6_nfe10_fs8@fs8)

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
    uid=${task}__s0; udir=$SMOKE_RUN_ROOT/units/$uid/a1
    remote="set -u
used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu | tr -d ' ')
[ \"\$used\" -lt 1024 ] || { echo \"REFUSE: GPU $gpu on \$(hostname -s) has \${used} MiB in use\"; exit 3; }
for p in $SMOKE_RPC_PORT $SMOKE_WS_PORT; do [ -z \"\$(ss -ltnH \"sport = :\$p\")\" ] || { echo \"REFUSE: port \$p is listening on \$(hostname -s)\"; exit 3; }; done
rm -rf '$udir'; mkdir -p '$udir' || { echo mkdir_failed; exit 1; }
SANA_EVAL_XPL_POLICY=SANA_MOT SANA_EVAL_CHECKPOINT_ROOT='$CKPT' SANA_EVAL_XPL_ROOT='$XPL_ROOT' \
SANA_EVAL_XPL_EXTRA_OVERRIDES='$OVERRIDES' SANA_EVAL_XPL_ACTION_TYPE=joint SEC_PER_STEP=4 \
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
    echo "== server log ([SANA_MOT] lines, errors)"
    grep -h "\[SANA_MOT\]\|Traceback\|Error\|error:" "$dst/server/server.stdout.log" "$dst/server/server.stderr.log" 2>/dev/null | cut -c1-400 | head -n 30 || true
    echo "== client progress: $(cat "$dst"/client/*/client.stdout.log 2>/dev/null | grep -a -o 'env0 step' | wc -l) simulation steps"
    echo "== result.json"; cat "$dst/result.json" 2>/dev/null || echo "(none yet)" ;;
  smoke)
    spec=${1:?pairs spec, e.g. 'l5:1'}; check_sync; check_pairs_free "$spec"
    bash "$EVAL/start_master.sh" "${RUN}_smoke" "${COMMON[@]}" --nodes "$(nodes_of "$spec")" --node-pairs "$spec" --tasks insert_key --episodes-per-task 1 --pin-longest 0 --poll 20
    echo "expect in the server log: '[SANA_MOT] ready: ... steps=10 cfg_scale=6.0 ... flow_shift=8.0 ... video_layout=openwam_canvas"
    echo "(requested openwam_canvas) context_layout=context_embedder state_as_context=False' -> bash $0 server-log insert_key__s0 ${RUN}_smoke" ;;
  start)
    spec=${1:?pairs spec}; check_sync; check_pairs_free "$spec"
    bash "$EVAL/start_master.sh" "$RUN" "${COMMON[@]}" --nodes "$(nodes_of "$spec")" --node-pairs "$spec" \
         --tasks fold_clothes --episodes-per-task 25 --pin-longest 0
    log "started $RUN on '$spec'" ;;
  dry)
    spec=${1:-l3:s1;l26:s0}; st=$SCRATCH/dry_state; rm -rf "$st"; mkdir -p "$st"
    CHECKOUT=${ROBODOJO_CHECKOUT_X5:-$HOME/zekail/sana_wam_sim_benchmarks/RoboDojo-official-clone-20260823}
    "$PY" "$EVAL/master.py" --run-root "/homes/jiamingt/robodojo_eval/${RUN}_dry" --checkout "$CHECKOUT" --state-dir "$st" \
         --dry-run --dry-sleep 0.05 --poll 1 "${COMMON[@]}" --nodes "$(nodes_of "$spec")" --node-pairs "$spec" >"$st/stdout.log" 2>&1 || true
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
    echo "columns: MoT s36k | canvas s30000 | abs_nsc_s30k | r18_s61k   (SR% / score%)"
    grep -h "tasks compared" "$SCRATCH/compare.stdout" | cut -c1-120
    grep "^| [A-Za-z_]* | [A-Za-z-]* | [0-9]" "$out" | cut -c1-200       # tasks this run has finished
    grep "^| Precision\|^| Memory\|^| Open\|^| Long-Horizon\|^| Generalization\|overall" "$out" | cut -c1-200
    echo "-> $out" ;;
  status) sed -n 1,40p "$STATE/STATUS.md"; echo ...; tail -n 6 "$STATE/master.log" ;;
  watch) tail -n 0 -F "$STATE/master.log" | grep -E --line-buffered "DONE |FAIL|failed [1-9]|quarantin|ALL UNITS SETTLED|Traceback|WARN slot|stale|server_timeout|server_fail|prologue|control: |requeue" ;;
  server-log)
    uid=${1:?unit id}; run=${2:-$RUN}; a=${3:-1}; dst=$SCRATCH/serverlog_${run}_${uid}_a$a; mkdir -p "$dst"
    rsync -az "$NFS_USER_HOST:/homes/jiamingt/robodojo_eval/$run/units/$uid/a$a/server/" "$dst/"
    grep -h "\[SANA_MOT\]\|WARNING\|Traceback\|Error\|error" "$dst"/server.stdout.log "$dst"/server.stderr.log 2>/dev/null | head -n 30 ;;
  *) sed -n 2,33p "$0"; exit 1 ;;
esac
