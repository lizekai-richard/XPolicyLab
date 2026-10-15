#!/usr/bin/env bash
# Campaign CHAIN (runs on x5): the OpenWAM-canvas SFT on the Pantheon14k donor, THREE SEEDS on ALL usable GPUs, one seed at a time,
# QUEUED behind the whole no-CFG Pantheon14k pixel series (user, 2026-10-12: "请你再下载logits/sft_robodojo_eefabs_openwam_384x320_vanilla_
# pantheon14k_aligned_f33fps8，这是我又用openwam模式sft的结果，请你排队测试"):
#   a = logits/sft_robodojo_eefabs_openwam_384x320_vanilla_pantheon14k_aligned_f33fps8  (Sana 9da55937e; OpenWAM L-shaped 384x320
#       canvas on the one policy class, donor sanapixel_pantheon_e4s14173 = the Pantheon14k pixel's donor, condition dropout 0.1,
#       norm 9f8b98e, epoch 10 / step 36,180; serves like the vanilla34k openwam: test_the_pantheon14k_openwam_serves_like_the_vanilla34k_openwam)
# Default sampling (video CFG 6, action CFG 1, 10 steps, flow shifts 5 / 1), whole 32-target chunks.
# Baselines: the Pantheon14k pixel on the same donor (956/6300), vanilla34k openwam (922 = 3 seeds 14.36/20.18), MoT-49.5k openwam 1007.
# WAIT_RUNS = all three no-CFG runs; a waited-for run that has not STARTED yet counts as pending (the gap between two seeds of that
# chain must not let this one in: user rule "queue behind the whole planned series").
#
#   bash pantheon14k_openwam_allgpu_seeds.sh chain start|stop|status | watch [HOURS] | status | wait-check | free-spec | smoke-check | combined
#   bash pantheon14k_openwam_allgpu_seeds.sh run SEED a ARGS...    the per-run driver with this chain's env
# ATTENTION about a run also creates $CHAIN_DIR/hold.<RUN>: no automatic action on that run until the file is removed.
set -uo pipefail
PY=${MASTER_PY:-$HOME/miniforge3/envs/sana-wam/bin/python}
EVAL=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF=$EVAL/campaigns/$(basename "${BASH_SOURCE[0]}")
RUN_DRIVER=$EVAL/campaigns/vanilla34k_eefabs_f33fps8.sh
KEYS="a"
LABEL_a=eefabs_openwam_384x320_vanilla_pantheon14k_aligned_f33fps8
NORM_a=9f8b98ede445f554cafef99974a0ef67ce4d59292578ccbdeaba704c9cedd5ba   # the snapshot's own artifact (RUN_ID normalization_sha256)
VR_a="stretch (default)"   # the ready line must show this (the label is not in the crop registry)
PAD_a=unmasked             # model.extra.sana_pixel_pad
SEEDS=${SEEDS:-0 1 2}
PER_RUN=${PER_RUN:-109}           # GPUs per seed = every usable slot (111 boards minus the reserved l3:0, l26:1 and l30:5)
MIN_PER_RUN=${MIN_PER_RUN:-40}    # never start a run on fewer
START_GRACE=${START_GRACE:-300}   # wait up to this long for PER_RUN free GPUs before starting on fewer (topped up later)
TOPUP_WINDOW=${TOPUP_WINDOW:-10800}  # top a run up to PER_RUN slots during this long after its start, while it has queued units
_nocfg=eefabs_sana_pixel_320x512_vanilla_pantheon14k_aligned_f33fps8_nocfg_xpl_vcfg1_acfg1_nfe10_vfs5_afs1_ee
WAIT_RUNS=${WAIT_RUNS-official_seed0_$_nocfg official_seed1_$_nocfg official_seed2_$_nocfg}   # the whole no-CFG series (not-started = pending)
WAIT_INTERVAL=${WAIT_INTERVAL:-300}; CONFIRM_INTERVAL=${CONFIRM_INTERVAL:-120}; CONFIRM_CHECKS=${CONFIRM_CHECKS:-1}
TICK=${TICK:-90}
NODES_ALL=${NODES_ALL:-l7 l0 l25 l1 l5 l6 l2 l9 l8 l30 l3 l26 l27}   # 2026-10-14 18:45 PDT (seeds 1-2, after the 46 h NFS outage): every reachable node, lowest load first (master pull order), l27 last (slowest NFS); l4 still unreachable; page cache prewarmed per node. Before: # 2026-10-13: l2 l9 l25 l30 OUT (policy servers hung 50 min -> server_timeout on every
# slot; raw NFS reads 13-14 MB/s), l4 OUT (unreachable since 10-12 14:30 PDT, its 8 units went stale)   # l5 / l0 LAST: their NFS clients are degraded (2026-10-12, load ~700,
# hundreds of D-state processes of another user); the master pulls through its nodes in this order, 300 s per timed-out node
RPC_BASE=63200; WS_BASE=65200     # the vanilla34k driver's slot ports: base + 50 + g
RESERVE=${RESERVE:-l3:0 l26:1 l30:5}    # never taken: the vLLM boards (l3, l26) and l30 CUDA 5 (devices busy or unavailable since 2026-10-09)
TAIL_TASKS=(play_tic_tac_toe imitate_sorting_sequence fill_egg_holder general_pickup)
MIN_IDLE=${MIN_IDLE:-4}; MAX_RELAUNCH_ROUNDS=${MAX_RELAUNCH_ROUNDS:-6}
CHAIN_DIR=$HOME/robodojo_master/.scratch/chain_pantheon14k_openwam_allgpu; mkdir -p "$CHAIN_DIR"
LOG=$CHAIN_DIR/chain.log; PIDF=$CHAIN_DIR/chain.pid
NFS_USER_HOST=jiamingt@${NFS_GATEWAY:-l7}.mit.edu   # l5 degraded 2026-10-12
CKPT_DIR=/homes/jiamingt/checkpoints/huggingface/logits

log(){ echo "[$(date -u +%FT%TZ)] $*" >&2; }
event(){ log "EVENT $*"; }
attention(){ # a key naming a run holds it (one ATTENTION per hold); any other key is reported once
  local key=$1; shift; local f
  if [[ $key == official_* ]]; then
    [[ -e $CHAIN_DIR/hold.$key ]] && return 0
    printf '%s\n' "$*" >"$CHAIN_DIR/hold.$key"
  else
    f=$CHAIN_DIR/att.$(printf '%s' "$key" | md5sum | cut -c1-16); [[ -e $f ]] && return 0; printf '%s\n' "$*" >"$f"
  fi
  log "ATTENTION $*"
}
label_of(){ local v="LABEL_$1"; echo "${!v}"; }
norm_of(){ local v="NORM_$1"; echo "${!v}"; }
run_of(){ echo "official_seed$1_$(label_of "$2")_xpl_vcfg6_acfg1_nfe10_vfs5_afs1_ee"; }
drv(){ # the per-run driver with exactly this chain's settings (never inherit VIEW_RESIZE / overrides / scratch from the caller)
  local seed=$1 k=$2; shift 2
  env -u VIEW_RESIZE -u EXTRA_OVERRIDES -u SCRATCH -u SMOKE_SEED -u SMOKE_RUN_ROOT -u SUMMARY_RUNS -u SHARD_PLAN \
      CKPT_LABEL="$(label_of "$k")" NORM_SHA="$(norm_of "$k")" VIDEO_FS=5 ACTION_FS=1 VIDEO_CFG=6 ACTION_CFG=1 \
      RUN="$(run_of "$seed" "$k")" EXTRA_MASTER_ARGS="--seeds $seed" "$@"
}
drive(){ local seed=$1 k=$2; shift 2; drv "$seed" "$k" bash "$RUN_DRIVER" "$@"; }
master_alive(){ local m=$HOME/robodojo_master/$1; [[ -s $m/master.pid ]] && kill -0 "$(cat "$m/master.pid")" 2>/dev/null; }

live_slots(){ # every GPU any live master's slot names, plus every GPU a live master's tail plan dropped for its shards
  "$PY" - "$HOME/robodojo_master" <<'PYEOF'
import glob, json, os, pathlib, sys
root = pathlib.Path(sys.argv[1]); out = set()
for pidf in root.glob("*/master.pid"):
    try: pid = int(pidf.read_text().strip()); os.kill(pid, 0)
    except Exception: continue
    st = pidf.with_name("state.json")
    if st.exists():
        for sl in json.loads(st.read_text())["slots"]:
            out.update({f'{sl["node"]}:{sl["server_gpu"]}', f'{sl["node"]}:{sl["client_gpu"]}'})
    for plan in glob.glob(str(pidf.parent / "tail_shards" / "*.json")):
        try: dropped = json.load(open(plan)).get("dropped_slots", "")
        except Exception: continue
        for part in filter(None, dropped.split(";")):
            node, gs = part.split(":")
            out.update(f"{node}:{int(g.lstrip('s'))}" for g in filter(None, gs.split(",")))
print("\n".join(sorted(out)))
PYEOF
}
free_spec(){ # every healthy CUDA device of NODES_ALL with < 1 GiB in use and both slot ports closed that no live master holds
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
    for g in $gl; do grep -qx "$node:$g" <<<"$held" && continue; [[ " $RESERVE " == *" $node:$g "* ]] && continue; keep+=("s$g"); done
    [[ ${#keep[@]} -gt 0 ]] && out+="${out:+;}$node:$(IFS=,; echo "${keep[*]}")"
  done
  echo "$out"
}
spec_count(){ echo "$1" | tr ';' '\n' | cut -d: -f2 | tr ',' '\n' | grep -c . || true; }
deal(){ # deal the slots of spec $1 round-robin (node order kept) into $2 parts of at most $3 slots each: one spec per line
  "$PY" - "$1" "$2" "$3" <<'PYEOF'
import sys
spec, k, cap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
slots = [(n, g) for item in spec.split(";") if ":" in item for n, gs in [item.split(":", 1)] for g in gs.split(",") if g]
parts = [dict() for _ in range(k)]; sizes = [0] * k; i = 0
for n, g in slots:
    open_ = [j for j in range(k) if sizes[j] < cap]
    if not open_: break
    j = open_[i % len(open_)]; i += 1
    parts[j].setdefault(n, []).append(g); sizes[j] += 1
for p in parts:
    print(";".join(f"{n}:{','.join(gs)}" for n, gs in p.items()))
PYEOF
}
runs_pending(){ # the WAIT_RUNS not over: a live master, a worker of its run root on a node, or units not done (dead master)
  local run pending="" pat n node
  [[ -n ${WAIT_RUNS// /} ]] || { echo ""; return 0; }   # nothing to wait for (an empty grep pattern would match every process)
  for run in $WAIT_RUNS; do master_alive "$run" && pending+=" $run(master)"; done
  pat=$(for run in $WAIT_RUNS; do printf '[r]obodojo_eval/%s |' "$run"; done); pat=${pat%|}
  for node in $NODES_ALL; do
    n=$(timeout 60 ssh -n -o BatchMode=yes -o ConnectTimeout=15 "jiamingt@$node.mit.edu" "ps -eo args | grep -E -c '$pat' || true" 2>/dev/null) \
      || { log "wait: $node unreachable (not counted)" >&2; continue; }
    [[ ${n:-0} -gt 0 ]] && pending+=" $node(${n}workers)"
  done
  if [[ -z $pending ]]; then
    for run in $WAIT_RUNS; do
      st=$HOME/robodojo_master/$run/state.json
      [[ -f $st ]] || { pending+=" $run(not-started)"; continue; }
      nd=$("$PY" -c "import json,sys; print(sum(u['state'] != 'done' for u in json.load(open(sys.argv[1]))['units']))" "$st" 2>/dev/null || echo "?")
      if [[ $nd != 0 ]]; then
        pending+=" $run(dead-master,$nd-units-not-done)"
        attention "dit-incomplete-$run" "waited-for run $run: master not running but $nd units are not done (crashed or stopped?) -- the chain keeps waiting"
      fi
    done
  fi
  echo "${pending# }"
}
ckpts_ready(){ # every checkpoint downloaded and sha256-verified on the node (marker written after `sha256sum -c SHA256SUMS`)
  local k; for k in $KEYS; do [[ -e $CHAIN_DIR/verified.$(label_of "$k") ]] || return 1; done
}

# ---- seed 0 smoke gate -----------------------------------------------------------------------------------------------------
smoke_check(){ # $1 = a key: 0 passed, 1 not yet, 2 failed
  local k=$1 label vr pad run dst out ready need missing=""
  label=$(label_of "$k"); vr=VR_$k; vr=${!vr}; pad=PAD_$k; pad=${!pad}; run=$(run_of 0 "$k")
  out=$(drive 0 "$k" smoke-status insert_key 2>&1)
  dst=$HOME/robodojo_master/.scratch/$run/smoke1gpu_insert_key__s0
  grep -q '"status": "complete"' "$dst/result.json" 2>/dev/null || {
    grep -q '"status"' "$dst/result.json" 2>/dev/null && { log "smoke $k result: $(tr -d '\n' <"$dst/result.json" | cut -c1-300)"; return 2; }
    grep -q "Traceback" "$dst/server/server.stderr.log" 2>/dev/null && { log "smoke $k: server traceback"; return 2; }
    return 1; }
  ready=$(grep -h "\[SANA_WAM\] ready:" "$dst/server/server.stdout.log" 2>/dev/null | tail -n 1)
  for need in "ckpt=$CKPT_DIR/$label " "steps=10 " " cfg_scale=6.0 " "video_cfg_scale=6.0 " "action_cfg_scale=1.0 " " flow_shift=5.0 " \
              "action_flow_shift=1.0 " "n_action_steps=all " "action_type=ee " "visual_layout=openwam_canvas " \
              "state_as_cross_attention=False " "video_fps=8 " "video_frame_stride=4 " "multiview=openwam " "rope=aligned " \
              "text_groups=1 " "canvas_prompt=two_rows " "robot_base_eef_layout=eef_only"; do
    [[ $ready == *"$need"* ]] || missing+=" [$need]"
  done
  grep -q '"episodes_recorded": 1' "$dst/result.json" || missing+=" [episodes_recorded 1]"
  if [[ -n $missing ]]; then log "smoke $k: ready line lacks$missing: ${ready:0:1200}"; return 2; fi
  log "smoke $k PASSED: $(tr -d '\n' <"$dst/result.json" | grep -o '"score_sum": [0-9.]*\|"successes": [0-9]*' | paste -sd' ') | ${ready:0:1200}"
  return 0
}
smoke_gate(){ # 0 passed (or passed before), 1 retry later, 2 failed; one insert_key episode per key, on different nodes if possible
  [[ -e $CHAIN_DIR/smoke_passed ]] && return 0
  local spec picks k i=0 rc all_ok failed t0 ng msg="" nk
  nk=$(wc -w <<<"$KEYS")
  if [[ ! -e $CHAIN_DIR/smoke_started ]]; then
    spec=$(free_spec)
    picks=$("$PY" - "$spec" "$nk" <<'PYEOF'
import sys
slots = [(n, g.lstrip("s")) for item in sys.argv[1].split(";") if ":" in item for n, gs in [item.split(":", 1)] for g in gs.split(",") if g]
k = int(sys.argv[2]); nodes = list(dict.fromkeys(n for n, _ in slots))
first = [(n, next(g for m, g in slots if m == n)) for n in nodes]
picks = first[:k] if len(first) >= k else slots[:k]
if len(picks) == k: print(" ".join(f"{n}:{g}" for n, g in picks))
PYEOF
)
    [[ -n $picks ]] || { log "smoke: fewer than $nk free GPUs ($spec)"; return 1; }
    local arr=($picks)
    for k in $KEYS; do
      ng=${arr[$i]}
      log "smoke: $k on $ng (insert_key x 1, seed 0)"
      drv 0 "$k" env SMOKE_SEED=0 SMOKE_RPC_PORT=$((63290 + i)) SMOKE_WS_PORT=$((65290 + i)) bash "$RUN_DRIVER" smoke-gpu "$ng" insert_key 1 2>&1 | tail -n 3 | sed 's/^/   /' >&2
      msg+="$k $ng, "; i=$((i + 1))
    done
    date +%s >"$CHAIN_DIR/smoke_started"
    event "smoke dispatched (${msg%, })"
  fi
  t0=$(cat "$CHAIN_DIR/smoke_started")
  while :; do
    all_ok=1; failed=""
    for k in $KEYS; do smoke_check "$k"; rc=$?; (( rc == 0 )) || all_ok=0; (( rc == 2 )) && failed+=" $k"; done
    if (( all_ok )); then touch "$CHAIN_DIR/smoke_passed"; event "smoke PASSED (ready line as expected)"; return 0; fi
    if [[ -n $failed ]]; then attention smoke-failed "smoke FAILED ($failed): see the lines above and bash $SELF run 0 a smoke-status"; return 2; fi
    if (( $(date +%s) - t0 > 2400 )); then attention smoke-timeout "smoke not finished after 40 min"; return 2; fi
    sleep 60
  done
}

# ---- one run ----------------------------------------------------------------------------------------------------------------
run_info(){ # idle (single-GPU, outside l2/l6), others-not-done, queued, failed, slots, tail unit states
  "$PY" - "$1" "$2" "${TAIL_TASKS[@]}" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1])); seed = sys.argv[2]; tasks = sys.argv[3:]; tails = {f"{t}__s{seed}" for t in tasks}
idle = sum(1 for sl in s["slots"] if not sl.get("unit") and not sl.get("draining") and sl["server_gpu"] == sl["client_gpu"]
           and sl["node"] not in ("l2", "l6"))
st = {u["unit_id"]: u["state"] for u in s["units"]}
others = sum(1 for u, v in st.items() if u not in tails and v != "done")
print(idle, others, sum(v == "queued" for v in st.values()), sum(v == "failed" for v in st.values()), len(s["slots"]),
      " ".join(f"{t}={st.get(f'{t}__s{seed}', 'absent')}" for t in tasks))
PYEOF
}
run_totals(){ # done units / units, episodes, successes
  "$PY" - "$1" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1])); us = s["units"]; done = [u for u in us if u["state"] == "done"]
eps = sum((u.get("result") or {}).get("episodes_recorded", 0) for u in done)
succ = sum((u.get("result") or {}).get("successes", 0) for u in done)
print(f"{len(done)}/{len(us)} units, {eps}/{sum(u['episodes_per_task'] for u in us)} eps, {succ} successes")
PYEOF
}
summary_line(){ # the overview row of an official summary: SR / score per dimension and overall
  awk '/^\| SANA_WAM \|/ {print; exit}' "$1" | sed 's/ ±0\.00//g; s/<nobr>//g; s/<\/nobr>//g' | cut -c1-400
}
manage_run(){ # $1 seed $2 key -> 0 complete, 1 in progress / held
  local seed=$1 k=$2 R M idle others queued failed nslots rest t ust out resolved=0 launched=0 summ="" asm=() rr f
  R=$(run_of "$seed" "$k"); M=$HOME/robodojo_master/$R
  [[ -e $CHAIN_DIR/complete.$R ]] && return 0
  [[ -e $CHAIN_DIR/hold.$R ]] && return 1
  [[ -f $M/state.json ]] || return 1
  if ! master_alive "$R"; then
    if [[ $("$PY" -c "import json,sys; print(sum(u['state'] != 'done' for u in json.load(open(sys.argv[1]))['units']))" "$M/state.json") != 0 ]]; then
      attention "$R" "$R: master not running but units are not done ($(run_totals "$M/state.json")) -- held"; return 1
    fi
    out=$(drive "$seed" "$k" official-summary 2>&1); f=$(grep -o '[^ ]*OFFICIAL_SUMMARY_[0-9T]*\.md' <<<"$out" | tail -n 1)
    f=${f/#\~/$HOME}
    if [[ -z $f || ! -s $f ]]; then attention "$R" "$R: official-summary failed: $(tail -n 3 <<<"$out" | tr '\n' ' ')"; return 1; fi
    grep -q "PROBLEMS\|still running" <<<"$out" && { attention "$R" "$R: official-summary reports problems: $(grep -m1 'laid out' <<<"$out")"; return 1; }
    printf '%s\n' "$f" >"$CHAIN_DIR/complete.$R"
    event "RUN COMPLETE seed $seed $k: $(run_totals "$M/state.json") | $(grep -m1 'laid out' <<<"$out") | summary $f"
    log "SUMMARY seed $seed $k $(summary_line "$f")"
    return 0
  fi
  read -r idle others queued failed nslots rest <<<"$(run_info "$M/state.json" "$seed")"
  if (( failed > 0 )); then attention "$R" "$R: $failed failed unit(s) -- held (retry / tail fill by hand)"; return 1; fi
  for t in "${TAIL_TASKS[@]}"; do
    ust=$(grep -o "$t=[a-z]*" <<<"$rest" | cut -d= -f2)
    if [[ $ust == done || $ust == absent ]]; then resolved=$((resolved + 1)); summ+=" $t:seq-$ust"; continue; fi
    if [[ -f $M/tail_shards/$t.json ]]; then
      out=$(drv "$seed" "$k" timeout 1200 bash "$RUN_DRIVER" tail status "$t" 2>&1)
      summ+=" $t:$(grep -o '[0-9]*/[0-9]* episodes, [0-9]* successes; COVERAGE [A-Za-z]*' <<<"$out" | sed 's/ episodes, /eps,/; s/ successes; COVERAGE /succ,/')"
      if grep -q "COVERAGE COMPLETE" <<<"$out"; then resolved=$((resolved + 1)); asm+=("$t"); continue; fi
      if grep -q "NEED [0-9]* MORE\|NOT STARTED\|FAILED \[[0-9]" <<<"$out"; then
        if grep -q "FAILED \[[0-9]" <<<"$out"; then
          rr=$(( $(cat "$CHAIN_DIR/relaunch.$R.$t" 2>/dev/null || echo 0) + 1 )); echo "$rr" >"$CHAIN_DIR/relaunch.$R.$t"
          (( rr > MAX_RELAUNCH_ROUNDS )) && { attention "$R" "$R: $t had $rr failed-shard relaunch rounds -- held (look at the shards)"; return 1; }
        fi
        log "$R: $t: $(grep -o 'NOT STARTED.*\|NEED [0-9]* MORE\|FAILED \[[^]]*\]' <<<"$out" | head -n 3 | tr '\n' ' ')-> more-substitutes"
        drv "$seed" "$k" timeout 1800 bash "$RUN_DRIVER" tail more-substitutes "$t" 2>&1 | grep -v "^$" | tail -n 40 | sed 's/^/   /' >&2
      fi
    elif [[ $ust == running && $queued -eq 0 && $launched -eq 0 && $idle -ge $MIN_IDLE ]]; then
      event "seed $seed $k: queue empty, tail launch $t (idle slots $idle, other units not done $others)"
      drv "$seed" "$k" timeout 1800 bash "$RUN_DRIVER" tail launch "$t" 2>&1 | grep -v "^$" | tail -n 60 | sed 's/^/   /' >&2
      launched=1
    else
      summ+=" $t:seq-$ust"
    fi
  done
  printf '%s\n' "$(date -u +%FT%TZ) idle $idle others $others queued $queued slots $nslots |$summ" >"$CHAIN_DIR/progress.$R"
  if (( resolved == ${#TAIL_TASKS[@]} && others == 0 && ${#asm[@]} > 0 )); then
    out=$("$PY" "$EVAL/tail_overlap_check.py" "$R" 2>&1)
    if ! grep -q "involved in an overlap: \[\]" <<<"$out" || ! grep -q "BIND-ERR shards: \[\]" <<<"$out"; then
      attention "$R" "$R: tail overlap check NOT clean -- held: $(grep -m3 'OVERLAP\|involved\|BIND' <<<"$out" | tr '\n' ' ' | cut -c1-600)"; return 1
    fi
    log "$R: overlap check clean ($(grep -m1 'shards:' <<<"$out" | cut -c1-80)); tail assemble ${asm[*]}"
    out=$(drv "$seed" "$k" timeout 3600 bash "$RUN_DRIVER" tail assemble "${asm[@]}" 2>&1); printf '%s\n' "$out" | tail -n 12 | sed 's/^/   /' >&2
    if [[ ! -s $M/tail_shards/ASSEMBLED.json ]] || ! grep -q "^ASSEMBLED " <<<"$out"; then
      attention "$R" "$R: tail assemble failed -- held: $(tail -n 3 <<<"$out" | tr '\n' ' ' | cut -c1-400)"; return 1
    fi
    event "seed $seed $k assembled: $(grep '^ASSEMBLED ' <<<"$out" | tr '\n' ' ' | cut -c1-600)"
  fi
  return 1
}

# ---- one seed -----------------------------------------------------------------------------------------------------------
start_seed(){ # $1 seed: start the runs of this seed that do not exist yet; 0 started (or nothing to start), 1 retry later, 2 refused
  local seed=$1 k todo=() spec n parts i rc=0 rc_k first now
  for k in $KEYS; do [[ -f $HOME/robodojo_master/$(run_of "$seed" "$k")/state.json ]] || todo+=("$k"); done
  [[ ${#todo[@]} -gt 0 ]] || return 0
  spec=$(free_spec); n=$(spec_count "$spec")
  first=$(cat "$CHAIN_DIR/start_wait.$seed" 2>/dev/null || { date +%s | tee "$CHAIN_DIR/start_wait.$seed"; }); now=$(date +%s)
  if (( n < ${#todo[@]} * PER_RUN && now - first < START_GRACE )); then log "seed $seed: $n free GPUs < ${#todo[@]} x $PER_RUN; waiting (grace $(( START_GRACE - now + first )) s)"; return 1; fi
  if (( n < ${#todo[@]} * MIN_PER_RUN )); then
    (( now - first > 3600 )) && attention "few-gpus-$seed" "seed $seed: only $n free GPUs for ${#todo[@]} run(s) for over an hour ($spec)"
    log "seed $seed: only $n free GPUs ($spec); waiting"; return 1
  fi
  mapfile -t parts < <(deal "$spec" "${#todo[@]}" "$PER_RUN")
  local msg="" out
  for i in "${!todo[@]}"; do
    k=${todo[$i]}
    log "seed $seed $k: start on $(spec_count "${parts[$i]}") GPUs: ${parts[$i]}"
    out=$(drive "$seed" "$k" start "${parts[$i]}" 2>&1); rc_k=$?
    printf '%s\n' "$out" | tail -n 4 | sed 's/^/   /' >&2
    if (( rc_k != 0 )); then attention "start-refused-$seed-$k" "seed $seed $k: start refused (rc $rc_k): $(tail -n 2 <<<"$out" | tr '\n' ' ' | cut -c1-300)"; rc=2
    else date +%s >"$CHAIN_DIR/started.$(run_of "$seed" "$k")"; msg+="$k $(spec_count "${parts[$i]}") GPUs; "; fi
  done
  sleep 30
  for k in "${todo[@]}"; do
    [[ -e $CHAIN_DIR/started.$(run_of "$seed" "$k") ]] || continue
    master_alive "$(run_of "$seed" "$k")" || { attention "start-dead-$seed-$k" "seed $seed $k: no live master 30 s after the start"; rc=2; }
  done
  (( rc == 0 )) && event "seed $seed STARTED: $msg"
  return $rc
}
topup(){ # $1 seed: a run of this seed below PER_RUN slots, with queued units, inside TOPUP_WINDOW of its start -> add free GPUs
  local seed=$1 k R M nslots queued spec part need t0
  for k in $KEYS; do
    R=$(run_of "$seed" "$k"); M=$HOME/robodojo_master/$R
    master_alive "$R" || continue
    [[ -e $CHAIN_DIR/hold.$R || -e $M/control/add_slots ]] && continue
    t0=$(cat "$CHAIN_DIR/started.$R" 2>/dev/null || echo 0); (( $(date +%s) - t0 < TOPUP_WINDOW )) || continue
    read -r nslots queued <<<"$("$PY" -c "import json,sys; s=json.load(open(sys.argv[1])); print(len(s['slots']), sum(u['state']=='queued' for u in s['units']))" "$M/state.json")"
    (( nslots < PER_RUN && queued > 0 )) || continue
    (( $(date +%s) - $(cat "$CHAIN_DIR/topup_probe" 2>/dev/null || echo 0) >= 600 )) || continue   # probe the nodes at most every 10 min
    date +%s >"$CHAIN_DIR/topup_probe"
    spec=$(free_spec); [[ -n $spec ]] || continue
    need=$(( PER_RUN - nslots )); part=$(deal "$spec" 1 "$need" | head -n 1)
    [[ -n $part ]] || continue
    mkdir -p "$M/control"; printf '%s\n' "$part" >"$M/control/add_slots.tmp" && mv -f "$M/control/add_slots.tmp" "$M/control/add_slots"
    log "seed $seed $k: top-up add_slots '$part' ($(spec_count "$part") GPUs; had $nslots)"
  done
}

chain_loop(){
  log "chain: label a=$LABEL_a; seeds [$SEEDS]; up to $PER_RUN GPUs per run; waits for [${WAIT_RUNS:-nothing}]"
  local seed pending ok i k all r done_runs last_hb=0 rc s0_exists=0
  for k in $KEYS; do [[ -f $HOME/robodojo_master/$(run_of "${SEEDS%% *}" "$k")/state.json ]] && s0_exists=1; done
  if (( ! s0_exists )); then
    # 1. the waited-for runs (none by default) must be completely over
    while :; do
      pending=$(runs_pending)
      if [[ -z $pending ]]; then
        ok=1
        for (( i = 1; i < CONFIRM_CHECKS; i++ )); do
          log "wait: the waited-for runs look over (check $i/$CONFIRM_CHECKS); confirming in ${CONFIRM_INTERVAL}s"
          sleep "$CONFIRM_INTERVAL"; pending=$(runs_pending); [[ -z $pending ]] || { ok=0; break; }
        done
        (( ok )) && { event "the waited-for runs are over ($CONFIRM_CHECKS quiet checks)"; break; }
      fi
      log "wait: still active: $pending"
      sleep "$WAIT_INTERVAL"
    done
    # 2. the checkpoint verified, then the seed-0 smoke
    until ckpts_ready; do attention ckpts-not-verified "checkpoints not marked verified in $CHAIN_DIR (verified.<label>); waiting"; sleep 300; done
    while :; do
      smoke_gate; rc=$?
      (( rc == 0 )) && break
      if (( rc == 2 )); then log "smoke gate failed; waiting for a decision (rm $CHAIN_DIR/smoke_started to re-smoke, or touch smoke_passed)"; sleep 600; else sleep 120; fi
    done
  fi
  # 3. the seeds, one at a time
  for seed in $SEEDS; do
    while :; do
      start_seed "$seed"; rc=$?
      if (( rc == 1 )); then sleep 300; continue; fi
      (( rc == 2 )) && sleep 600            # a refused start is retried, but not every tick
      all=1
      for k in $KEYS; do r=$(run_of "$seed" "$k"); [[ -f $HOME/robodojo_master/$r/state.json ]] || all=0; manage_run "$seed" "$k" || all=0; done
      (( all )) && { event "SEED $seed COMPLETE: run assembled and summarized"; break; }
      topup "$seed"
      if (( $(date +%s) - last_hb >= 3600 )); then
        last_hb=$(date +%s)
        for k in $KEYS; do r=$(run_of "$seed" "$k"); [[ -f $HOME/robodojo_master/$r/state.json ]] && log "hourly: seed $seed $k $(run_totals "$HOME/robodojo_master/$r/state.json") | $(cut -d'|' -f2- "$CHAIN_DIR/progress.$r" 2>/dev/null)"; done
      fi
      sleep "$TICK"
    done
  done
  # 4. the official 3-seed tables
  for k in $KEYS; do
    out=$(drv "${SEEDS%% *}" "$k" env SUMMARY_RUNS="$(for s in $SEEDS; do run_of "$s" "$k"; done | paste -sd' ')" bash "$RUN_DRIVER" official-summary 2>&1)
    f=$(grep -o '[^ ]*OFFICIAL_SUMMARY_combined_[0-9T]*\.md' <<<"$out" | tail -n 1); f=${f/#\~/$HOME}
    event "COMBINED $k: $f"; [[ -s $f ]] && log "SUMMARY 3 seeds $k $(summary_line "$f")"
  done
  event "CHAIN DONE: seeds [$SEEDS] complete"
}

cmd=${1:-help}; shift || true
case $cmd in
  chain)
    sub=${1:-status}
    case $sub in
      start)
        if [[ -s $PIDF ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "chain already running (pid $(cat "$PIDF")), log $LOG"; exit 0; fi
        nohup setsid bash "$SELF" chain-loop >>"$LOG" 2>&1 </dev/null &
        echo $! >"$PIDF"; sleep 3; wc -l <"$LOG" >"$CHAIN_DIR/watch.offset"; log "chain started (pid $(cat "$PIDF")), log $LOG" ;;
      stop) if [[ -s $PIDF ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then kill -TERM "$(cat "$PIDF")"; log "chain stopped (its runs and any child command keep running)"; else echo "chain not running"; fi ;;
      status) if [[ -s $PIDF ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "chain running (pid $(cat "$PIDF"))"; else echo "chain NOT running"; fi
              tail -n 12 "$LOG" 2>/dev/null | cut -c1-300 || true ;;
      *) echo "chain start|stop|status"; exit 2 ;;
    esac ;;
  chain-loop) chain_loop ;;
  watch)
    hours=${1:-4}; deadline=$(( $(date +%s) + ${hours%.*} * 3600 ))
    off=$(cat "$CHAIN_DIR/watch.offset" 2>/dev/null || echo 0)
    while :; do
      total=$(wc -l <"$LOG" 2>/dev/null || echo 0)
      new=$(sed -n "$(( off + 1 )),${total}p" "$LOG" 2>/dev/null | grep -E "EVENT|ATTENTION|SUMMARY" | cut -c1-700 || true)
      if [[ -n $new ]]; then echo "$total" >"$CHAIN_DIR/watch.offset"; echo "$new"; exit 0; fi
      if ! { [[ -s $PIDF ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }; then echo "$total" >"$CHAIN_DIR/watch.offset"; echo "chain daemon NOT running"; tail -n 5 "$LOG"; exit 3; fi
      if (( $(date +%s) >= deadline )); then echo "$total" >"$CHAIN_DIR/watch.offset"; echo "heartbeat ${hours} h: nothing flagged"; tail -n 3 "$LOG" | cut -c1-300; exit 0; fi
      off=$total; sleep 60
    done ;;
  status)
    echo "waited-for runs:"; for r in $WAIT_RUNS; do printf '  %s  %s\n' "$(master_alive "$r" && echo live || echo ended)" "$(grep -m1 '^units:' "$HOME/robodojo_master/$r/STATUS.md" 2>/dev/null | cut -c1-110)"; done
    for s in $SEEDS; do for k in $KEYS; do r=$(run_of "$s" "$k"); m=$HOME/robodojo_master/$r
      st="not started"; [[ -f $m/state.json ]] && st="$(master_alive "$r" && echo live || echo ended) $(run_totals "$m/state.json")"
      [[ -e $CHAIN_DIR/complete.$r ]] && st+=" COMPLETE"; [[ -e $CHAIN_DIR/hold.$r ]] && st+=" HELD"
      printf 'seed %s %s  %s\n' "$s" "$k" "$st"; done; done
    ls "$CHAIN_DIR"/att.* >/dev/null 2>&1 && { echo "attention notes:"; cat "$CHAIN_DIR"/att.* | sed 's/^/  /' | cut -c1-300; } ;;
  wait-check) pending=$(runs_pending); log "waited-for runs still active: ${pending:-none}"; spec=$(free_spec); log "free now: $(spec_count "$spec") GPUs ($spec)" ;;
  free-spec) spec=$(free_spec); echo "$spec"; log "$(spec_count "$spec") free GPUs" >&2 ;;
  run) seed=${1:?seed}; k=${2:?a}; shift 2; drive "$seed" "$k" "$@" ;;
  smoke-check)   # judge the seed-0 smokes (however they were dispatched) with the gate's own rule; all passed -> smoke_passed
    rcs=""; ok=1; for k in $KEYS; do smoke_check "$k"; rc=$?; rcs+=" $k=$rc"; (( rc == 0 )) || ok=0; done
    log "smoke-check:$rcs (0 passed, 1 not finished, 2 failed)"
    (( ok )) && { touch "$CHAIN_DIR/smoke_passed"; log "all passed -> $CHAIN_DIR/smoke_passed"; } ;;
  combined)
    for k in $KEYS; do drv "${SEEDS%% *}" "$k" env SUMMARY_RUNS="$(for s in $SEEDS; do run_of "$s" "$k"; done | paste -sd' ')" bash "$RUN_DRIVER" official-summary; done ;;
  *) sed -n 2,20p "$0"; exit 1 ;;
esac
