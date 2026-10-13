#!/usr/bin/env bash
# Campaign CHAIN (runs on x5): the chunk-causal checkpoint s7646 with the OBSERVATION SINK (deploy override
# softmax_window=sink) on the 28 task configs whose episodes can pass 24 chunks (step_lim > 768), THREE SEEDS as parallel
# lanes on every free GPU (user, 2026-10-06: "用sink测一下那28个config，3个seed"). Runs ..._sink28; the other 26 configs
# never reach the window boundary, so the sink cannot change them. A/B partner: the training-rule s7646 runs
# (causal_s7646_allgpu_lanes.sh; their adapter still kept softmax keys post-RoPE at physical positions -- the same
# logits in exact arithmetic, see tools/teacher_forcing_parity.py).
# Plan: smoke (one pour_by_language episode = 25 chunks: the ready line must show softmax_window=sink and chunk 24 must
# read the observation with origin 1), then the three seeds at once.
#
#   bash causal_s7646_sink28_lanes.sh chain start|stop|status | watch [HOURS] | status | free-spec | smoke-check | combined
#   bash causal_s7646_sink28_lanes.sh run SEED sink28 ARGS...     the per-run driver with this chain's env
# ATTENTION about a run also creates $CHAIN_DIR/hold.<RUN>: no automatic action on that run until the file is removed.
set -uo pipefail
EVAL=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${MASTER_PY:-$HOME/miniforge3/envs/sana-wam/bin/python}
SELF=$EVAL/campaigns/$(basename "${BASH_SOURCE[0]}")
RUN_DRIVER=$EVAL/campaigns/causal_eefabs_f33fps8.sh
LABEL=causal_eefabs_sana_pixel_320x512_vanilla52k_aligned_f33fps8_m48n24_s7646
NORM=1fe3b7e72e8fa93eca5efb8a8deeace1daf4434b4b97e9f4445670ac758287ed   # the snapshot's own artifact (= its yaml pin)
PHASES="sink28"
TASKS_sink28="arrange_largest_number,arrange_largest_number_random,build_tower,classify_objects,classify_objects_by_language,cover_blocks,fasten_screws,fill_pen_holder,hang_mugs,hang_mugs_random,imitate_sorting_sequence,make_toast,make_toast_random,organize_table,pack_objects_into_box,pack_objects_into_box_random,play_stacking_toy,play_tic_tac_toe,pour_by_language,sort_nesting_dolls_by_size,sort_nesting_dolls_by_size_random,stack_bowls,stack_bowls_random,store_laptop_and_headphones,store_laptop_and_headphones_random,store_tools_in_toolbox,sweep_blocks,sweep_blocks_random"
SMOKE_TASK=pour_by_language   # 800 steps = 25 chunks, never succeeded: always reaches chunk 24
SEEDS=${SEEDS:-0 1 2}
PER_RUN=${PER_RUN:-31}            # GPUs per run: 93 usable single-GPU slots / 3 lanes
MIN_PER_RUN=${MIN_PER_RUN:-16}    # never start a run on fewer
START_GRACE=${START_GRACE:-300}   # wait up to this long for PER_RUN free GPUs before starting on fewer (topped up later)
TOPUP_WINDOW=${TOPUP_WINDOW:-10800}  # top a run up to PER_RUN slots during this long after its start, while it has queued units
TICK=${TICK:-90}
NODES_ALL=${NODES_ALL:-l0 l1 l2 l3 l5 l6 l7 l8 l9 l25 l26 l27 l30}   # l0 included (user 2026-10-06: "用所有的卡")
RPC_BASE=63600; WS_BASE=65300     # the causal driver's slot ports: base + 50 + g
RESERVE=${RESERVE:-l3:0 l26:1}    # CUDA devices never taken: another user's vLLM (l3, l26)
TAIL_TASKS=(play_tic_tac_toe imitate_sorting_sequence)
MIN_IDLE=${MIN_IDLE:-4}; MAX_RELAUNCH_ROUNDS=${MAX_RELAUNCH_ROUNDS:-6}
CHAIN_DIR=$HOME/robodojo_master/.scratch/chain_causal_s7646_sink28_lanes; mkdir -p "$CHAIN_DIR"
LOG=$CHAIN_DIR/chain.log; PIDF=$CHAIN_DIR/chain.pid
NFS_USER_HOST=jiamingt@l5.mit.edu
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
tasks_of(){ local v="TASKS_$1"; echo "${!v}"; }
run_of(){ echo "official_seed$1_${LABEL}_xpl_vcfg6_acfg1_nfe10_vfs5_afs1_ee_$2"; }   # $1 seed $2 phase
drv(){ # the per-run driver with exactly this chain's settings (never inherit VIEW_RESIZE / overrides / scratch from the caller)
  local seed=$1 phase=$2; shift 2
  env -u VIEW_RESIZE -u EXTRA_OVERRIDES -u SCRATCH -u SMOKE_SEED -u SMOKE_RUN_ROOT -u SUMMARY_RUNS -u SHARD_PLAN -u DIM_ORDER \
      CKPT_LABEL="$LABEL" NORM_SHA="$NORM" VIDEO_FS=5 ACTION_FS=1 ACTION_CFG=1 RUN="$(run_of "$seed" "$phase")" \
      EXTRA_OVERRIDES="softmax_window=sink" EXTRA_MASTER_ARGS="--seeds $seed --tasks $(tasks_of "$phase")" "$@"
}
drive(){ local seed=$1 phase=$2; shift 2; drv "$seed" "$phase" bash "$RUN_DRIVER" "$@"; }
master_alive(){ local m=$HOME/robodojo_master/$1; [[ -s $m/master.pid ]] && kill -0 "$(cat "$m/master.pid")" 2>/dev/null; }
exists(){ [[ -f $HOME/robodojo_master/$1/state.json ]]; }
complete(){ [[ -e $CHAIN_DIR/complete.$1 ]]; }
all_complete(){ local s; for s in $SEEDS; do complete "$(run_of "$s" "$1")" || return 1; done; }   # $1 phase

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
    specs = []
    for plan in glob.glob(str(pidf.parent / "tail_shards" / "*.json")):
        try: specs.append(json.load(open(plan)).get("dropped_slots", ""))
        except Exception: continue
    pending = pidf.parent / "control" / "add_slots"      # a top-up the master has not read yet (it polls every minute)
    if pending.exists():
        specs.append(pending.read_text().strip())
    for spec in specs:
        for part in filter(None, spec.split(";")):
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
ckpt_ready(){ [[ -e $CHAIN_DIR/verified.$LABEL ]]; }   # written after verify_ckpt_sha.sh (3/3 against the repo's MANIFEST.json)

# ---- smoke gate (seed 0, through the _sink28 run's driver env: a 25-chunk episode crosses the window) ----------------
smoke_check(){ # 0 passed, 1 not yet, 2 failed
  local run dst out ready need missing=""
  run=$(run_of 0 sink28)
  out=$(drive 0 sink28 smoke-status "$SMOKE_TASK" 2>&1)
  dst=$HOME/robodojo_master/.scratch/$run/smoke1gpu_${SMOKE_TASK}__s0
  grep -q '"status": "complete"' "$dst/result.json" 2>/dev/null || {
    grep -q '"status"' "$dst/result.json" 2>/dev/null && { log "smoke result: $(tr -d '\n' <"$dst/result.json" | cut -c1-300)"; return 2; }
    grep -q "Traceback" "$dst/server/server.stderr.log" 2>/dev/null && { log "smoke: server traceback"; return 2; }
    return 1; }
  ready=$(grep -h "\[SANA_WAM_CAUSAL\] ready:" "$dst/server/server.stdout.log" 2>/dev/null | tail -n 1)
  for need in "ckpt=$CKPT_DIR/$LABEL " "actions_per_chunk=32 " "video_frame_stride=4 " "latent_frames_per_chunk=1 " \
              "sliding_window_chunks=24 " "obs_in_first_chunk=True " "multiview=sana_pixel " "rope=aligned " \
              "softmax_window=sink " "steps=10 " "video_cfg_scale=6.0 " "action_cfg_scale=1.0 " " flow_shift=5.0 " \
              "action_flow_shift=1.0 " "view_resize=stretch (default) " "commit_actions=executed " "gdn_recurrence=checkpoint " \
              "normalization_sha256=$NORM " "action_type=ee"; do
    [[ $ready == *"$need"* ]] || missing+=" [$need]"
  done
  grep -q '"episodes_recorded": 1' "$dst/result.json" || missing+=" [episodes_recorded 1]"
  grep -q "eef check ep 1: observed \*_ee_pose vs FK(measured joints): left 0.0" "$dst/server/server.stdout.log" 2>/dev/null \
    || missing+=" [eef check sub-mm]"
  grep -q "ep 1 chunk 24: read_entries=\[0, 2, 3, .*24\] origin=1 " "$dst/server/server.stdout.log" 2>/dev/null \
    || missing+=" [chunk 24 reads the observation sink, origin 1]"
  if [[ -n $missing ]]; then log "smoke: ready line lacks$missing: ${ready:0:1200}"; return 2; fi
  log "smoke PASSED: $(tr -d '\n' <"$dst/result.json" | grep -o '"score_sum": [0-9.]*\|"successes": [0-9]*' | paste -sd' ') | ${ready:0:1200}"
  return 0
}
smoke_gate(){ # 0 passed (or passed before), 1 retry later, 2 failed
  [[ -e $CHAIN_DIR/smoke_passed ]] && return 0
  local spec ng rc t0
  if [[ ! -e $CHAIN_DIR/smoke_started ]]; then
    spec=$(free_spec)
    ng=$("$PY" -c "
import sys
s = [(n, g.lstrip('s')) for it in sys.argv[1].split(';') if ':' in it for n, gs in [it.split(':', 1)] for g in gs.split(',') if g]
print(f'{s[0][0]}:{s[0][1]}' if s else '')" "$spec")
    [[ -n $ng ]] || { log "smoke: no free GPU ($spec)"; return 1; }
    log "smoke: $ng ($SMOKE_TASK x 1, seed 0)"
    drv 0 sink28 env SMOKE_SEED=0 bash "$RUN_DRIVER" smoke-gpu "$ng" "$SMOKE_TASK" 1 2>&1 | tail -n 3 | sed 's/^/   /' >&2
    date +%s >"$CHAIN_DIR/smoke_started"
    event "smoke dispatched ($ng)"
  fi
  t0=$(cat "$CHAIN_DIR/smoke_started")
  while :; do
    smoke_check; rc=$?
    if (( rc == 0 )); then touch "$CHAIN_DIR/smoke_passed"; event "smoke PASSED (causal ready line as expected)"; return 0; fi
    if (( rc == 2 )); then attention smoke-failed "smoke FAILED: see the lines above and bash $SELF run 0 sink28 smoke-status $SMOKE_TASK"; return 2; fi
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
  awk '/^\| SANA_WAM_CAUSAL \|/ {print; exit}' "$1" | sed 's/ ±0\.00//g; s/<nobr>//g; s/<\/nobr>//g' | cut -c1-400
}
manage_run(){ # $1 seed $2 phase -> 0 complete, 1 in progress / held
  local seed=$1 ph=$2 R M idle others queued failed nslots rest t ust out resolved=0 launched=0 summ="" asm=() rr f
  R=$(run_of "$seed" "$ph"); M=$HOME/robodojo_master/$R
  complete "$R" && return 0
  [[ -e $CHAIN_DIR/hold.$R ]] && return 1
  [[ -f $M/state.json ]] || return 1
  if ! master_alive "$R"; then
    if [[ $("$PY" -c "import json,sys; print(sum(u['state'] != 'done' for u in json.load(open(sys.argv[1]))['units']))" "$M/state.json") != 0 ]]; then
      attention "$R" "$R: master not running but units are not done ($(run_totals "$M/state.json")) -- held"; return 1
    fi
    out=$(drive "$seed" "$ph" official-summary 2>&1); f=$(grep -o '[^ ]*OFFICIAL_SUMMARY_[0-9T]*\.md' <<<"$out" | tail -n 1)
    f=${f/#\~/$HOME}
    if [[ -z $f || ! -s $f ]]; then attention "$R" "$R: official-summary failed: $(tail -n 3 <<<"$out" | tr '\n' ' ')"; return 1; fi
    grep -q "PROBLEMS\|still running" <<<"$out" && { attention "$R" "$R: official-summary reports problems: $(grep -m1 'laid out' <<<"$out")"; return 1; }
    printf '%s\n' "$f" >"$CHAIN_DIR/complete.$R"
    event "RUN COMPLETE seed $seed $ph: $(run_totals "$M/state.json") | $(grep -m1 'laid out' <<<"$out") | summary $f"
    log "SUMMARY seed $seed $ph $(summary_line "$f")"
    return 0
  fi
  read -r idle others queued failed nslots rest <<<"$(run_info "$M/state.json" "$seed")"
  if (( failed > 0 )); then attention "$R" "$R: $failed failed unit(s) -- held (retry / tail fill by hand)"; return 1; fi
  for t in "${TAIL_TASKS[@]}"; do
    ust=$(grep -o "$t=[a-z]*" <<<"$rest" | cut -d= -f2)
    if [[ $ust == done || $ust == absent ]]; then resolved=$((resolved + 1)); [[ $ust == absent ]] || summ+=" $t:seq-$ust"; continue; fi
    if [[ -f $M/tail_shards/$t.json ]]; then
      out=$(drv "$seed" "$ph" timeout 1200 bash "$RUN_DRIVER" tail status "$t" 2>&1)
      summ+=" $t:$(grep -o '[0-9]*/[0-9]* episodes, [0-9]* successes; COVERAGE [A-Za-z]*' <<<"$out" | sed 's/ episodes, /eps,/; s/ successes; COVERAGE /succ,/')"
      if grep -q "COVERAGE COMPLETE" <<<"$out"; then resolved=$((resolved + 1)); asm+=("$t"); continue; fi
      if grep -q "NEED [0-9]* MORE\|NOT STARTED\|FAILED \[[0-9]" <<<"$out"; then
        if grep -q "FAILED \[[0-9]" <<<"$out"; then
          rr=$(( $(cat "$CHAIN_DIR/relaunch.$R.$t" 2>/dev/null || echo 0) + 1 )); echo "$rr" >"$CHAIN_DIR/relaunch.$R.$t"
          (( rr > MAX_RELAUNCH_ROUNDS )) && { attention "$R" "$R: $t had $rr failed-shard relaunch rounds -- held (look at the shards)"; return 1; }
        fi
        log "$R: $t: $(grep -o 'NOT STARTED.*\|NEED [0-9]* MORE\|FAILED \[[^]]*\]' <<<"$out" | head -n 3 | tr '\n' ' ')-> more-substitutes"
        drv "$seed" "$ph" timeout 1800 bash "$RUN_DRIVER" tail more-substitutes "$t" 2>&1 | grep -v "^$" | tail -n 40 | sed 's/^/   /' >&2
      fi
    elif [[ $ust == running && $queued -eq 0 && $launched -eq 0 && $idle -ge $MIN_IDLE ]]; then
      event "seed $seed $ph: queue empty, tail launch $t (idle slots $idle, other units not done $others)"
      drv "$seed" "$ph" timeout 1800 bash "$RUN_DRIVER" tail launch "$t" 2>&1 | grep -v "^$" | tail -n 60 | sed 's/^/   /' >&2
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
    out=$(drv "$seed" "$ph" timeout 3600 bash "$RUN_DRIVER" tail assemble "${asm[@]}" 2>&1); printf '%s\n' "$out" | tail -n 12 | sed 's/^/   /' >&2
    if [[ ! -s $M/tail_shards/ASSEMBLED.json ]] || ! grep -q "^ASSEMBLED " <<<"$out"; then
      attention "$R" "$R: tail assemble failed -- held: $(tail -n 3 <<<"$out" | tr '\n' ' ' | cut -c1-400)"; return 1
    fi
    event "seed $seed $ph assembled: $(grep '^ASSEMBLED ' <<<"$out" | tr '\n' ' ' | cut -c1-600)"
  fi
  return 1
}

# ---- starting runs ------------------------------------------------------------------------------------------------------------
start_runs(){ # $1 phase, $2.. seeds: start these runs (none may exist yet) on the free GPUs dealt evenly; 0 started, 1 later, 2 refused
  local ph=$1; shift; local seeds=("$@") spec n parts i rc=0 rc_k first now key out msg="" s
  spec=$(free_spec); n=$(spec_count "$spec")
  key=start_wait.$ph.$(IFS=_; echo "${seeds[*]}")
  first=$(cat "$CHAIN_DIR/$key" 2>/dev/null || { date +%s | tee "$CHAIN_DIR/$key"; }); now=$(date +%s)
  if (( n < ${#seeds[@]} * PER_RUN && now - first < START_GRACE )); then
    log "$ph [${seeds[*]}]: $n free GPUs < ${#seeds[@]} x $PER_RUN; waiting (grace $(( START_GRACE - now + first )) s)"; return 1
  fi
  if (( n < ${#seeds[@]} * MIN_PER_RUN )); then
    (( now - first > 3600 )) && attention "few-gpus-$ph-${seeds[*]}" "$ph [${seeds[*]}]: only $n free GPUs for over an hour ($spec)"
    log "$ph [${seeds[*]}]: only $n free GPUs ($spec); waiting"; return 1
  fi
  mapfile -t parts < <(deal "$spec" "${#seeds[@]}" "$PER_RUN")
  for i in "${!seeds[@]}"; do
    s=${seeds[$i]}
    log "seed $s $ph: start on $(spec_count "${parts[$i]}") GPUs: ${parts[$i]}"
    out=$(drive "$s" "$ph" start "${parts[$i]}" 2>&1); rc_k=$?
    printf '%s\n' "$out" | tail -n 4 | sed 's/^/   /' >&2
    if (( rc_k != 0 )); then attention "start-refused-$s-$ph" "seed $s $ph: start refused (rc $rc_k): $(tail -n 2 <<<"$out" | tr '\n' ' ' | cut -c1-300)"; rc=2
    else date +%s >"$CHAIN_DIR/started.$(run_of "$s" "$ph")"; msg+="seed $s $(spec_count "${parts[$i]}") GPUs; "; fi
  done
  for s in "${seeds[@]}"; do   # the next free_spec must see these slots: wait for each master's state.json (written after planning)
    [[ -e $CHAIN_DIR/started.$(run_of "$s" "$ph") ]] || continue
    for i in $(seq 1 60); do exists "$(run_of "$s" "$ph")" && break; master_alive "$(run_of "$s" "$ph")" || break; sleep 5; done
    master_alive "$(run_of "$s" "$ph")" && exists "$(run_of "$s" "$ph")" \
      || { attention "start-dead-$s-$ph" "seed $s $ph: no live master with a state.json 5 min after the start"; rc=2; }
  done
  (( rc == 0 )) && event "$ph STARTED: $msg"
  return $rc
}
spec_minus(){ # spec $1 without the slots of spec $2
  "$PY" - "$1" "$2" <<'PYEOF'
import sys
def slots(spec): return [(n, g) for it in spec.split(";") if ":" in it for n, gs in [it.split(":", 1)] for g in gs.split(",") if g]
gone = set(slots(sys.argv[2])); out = {}
for n, g in slots(sys.argv[1]):
    if (n, g) not in gone: out.setdefault(n, []).append(g)
print(";".join(f"{n}:{','.join(gs)}" for n, gs in out.items()))
PYEOF
}
topup(){ # every live run below PER_RUN slots, with queued units, inside TOPUP_WINDOW of its start -> add free GPUs
  local s ph R M nslots queued spec="" probed=0 part need t0
  (( $(date +%s) - $(cat "$CHAIN_DIR/topup_probe" 2>/dev/null || echo 0) >= 600 )) || return 0   # probe the nodes at most every 10 min
  for s in $SEEDS; do for ph in $PHASES; do
    R=$(run_of "$s" "$ph"); M=$HOME/robodojo_master/$R
    master_alive "$R" || continue
    [[ -e $CHAIN_DIR/hold.$R || -e $M/control/add_slots ]] && continue
    t0=$(cat "$CHAIN_DIR/started.$R" 2>/dev/null || echo 0); (( $(date +%s) - t0 < TOPUP_WINDOW )) || continue
    read -r nslots queued <<<"$("$PY" -c "import json,sys; s=json.load(open(sys.argv[1])); print(len(s['slots']), sum(u['state']=='queued' for u in s['units']))" "$M/state.json")"
    (( nslots < PER_RUN && queued > 0 )) || continue
    (( probed )) || { date +%s >"$CHAIN_DIR/topup_probe"; spec=$(free_spec); probed=1; }
    [[ -n $spec ]] || return 0
    need=$(( PER_RUN - nslots )); part=$(deal "$spec" 1 "$need" | head -n 1)
    [[ -n $part ]] || continue
    mkdir -p "$M/control"; printf '%s\n' "$part" >"$M/control/add_slots.tmp" && mv -f "$M/control/add_slots.tmp" "$M/control/add_slots"
    log "seed $s $ph: top-up add_slots '$part' ($(spec_count "$part") GPUs; had $nslots)"
    spec=$(spec_minus "$spec" "$part")   # these GPUs are taken
  done; done
}
combined(){ # $1 tag, $2.. runs: one official table over these runs (written next to the first run)
  local tag=$1; shift; local first=$1 out f seed ph
  seed=$(sed -E 's/^official_seed([0-9]+)_.*/\1/' <<<"$first"); ph=${first##*_}
  out=$(drv "$seed" "$ph" env SUMMARY_RUNS="$*" bash "$RUN_DRIVER" official-summary 2>&1)
  f=$(grep -o '[^ ]*OFFICIAL_SUMMARY_combined_[0-9T]*\.md' <<<"$out" | tail -n 1); f=${f/#\~/$HOME}
  if [[ -s $f ]]; then event "COMBINED $tag: $f"; log "SUMMARY $tag $(summary_line "$f")"; printf '%s\n' "$f" >"$CHAIN_DIR/combined.$tag"
  else attention "combined-$tag" "combined table $tag failed: $(tail -n 3 <<<"$out" | tr '\n' ' ' | cut -c1-300)"; fi
}

chain_loop(){
  log "chain: $LABEL; seeds [$SEEDS] as parallel lanes (sink28, softmax_window=sink); up to $PER_RUN GPUs per run; nodes $NODES_ALL"
  local s rc all todo=() last_hb=0 R
  until ckpt_ready; do attention ckpt-not-verified "checkpoint not marked verified ($CHAIN_DIR/verified.$LABEL); waiting"; sleep 300; done
  while :; do
    smoke_gate; rc=$?
    (( rc == 0 )) && break
    if (( rc == 2 )); then log "smoke gate failed; waiting for a decision (rm $CHAIN_DIR/smoke_started to re-smoke, or touch smoke_passed)"; sleep 600; else sleep 120; fi
  done
  while :; do
    todo=(); for s in $SEEDS; do exists "$(run_of "$s" sink28)" || todo+=("$s"); done
    if (( ${#todo[@]} > 0 )); then
      start_runs sink28 "${todo[@]}"; rc=$?
      if (( rc == 1 )); then sleep 300; continue; fi
      (( rc == 2 )) && sleep 600
    fi
    all=1
    for s in $SEEDS; do
      exists "$(run_of "$s" sink28)" || { all=0; continue; }
      manage_run "$s" sink28 || all=0
    done
    (( all )) && break
    topup
    if (( $(date +%s) - last_hb >= 3600 )); then
      last_hb=$(date +%s)
      for s in $SEEDS; do R=$(run_of "$s" sink28); exists "$R" && ! complete "$R" && \
        log "hourly: seed $s sink28 $(run_totals "$HOME/robodojo_master/$R/state.json") | $(cut -d'|' -f2- "$CHAIN_DIR/progress.$R" 2>/dev/null)"; done
    fi
    sleep "$TICK"
  done
  [[ -e $CHAIN_DIR/combined.sink28 ]] || combined sink28 $(for s in $SEEDS; do run_of "$s" sink28; done)
  event "CHAIN DONE: seeds [$SEEDS] complete (sink28)"
}

cmd=${1:-help}; shift || true
# RETIRED 2026-10-10: policy/SANA_WAM_CAUSAL serves the per-chunk observation protocol only (Sana 61aad2d58 / 3a6b2bf07) and refuses the s7646 checkpoint (chunk-0-only observation); the adapter that served it is XPolicyLab commit 69cfb8e. Read-only subcommands still work; the pantheon14k driver is causal_p14k_eefabs_f33fps8.sh.
case "$cmd ${1:-}" in "chain start"*|chain-loop*|run*) echo "$0: RETIRED 2026-10-10: policy/SANA_WAM_CAUSAL serves the per-chunk observation protocol only (Sana 61aad2d58 / 3a6b2bf07) and refuses the s7646 checkpoint (chunk-0-only observation); the adapter that served it is XPolicyLab commit 69cfb8e. Read-only subcommands still work; the pantheon14k driver is causal_p14k_eefabs_f33fps8.sh." >&2; exit 2 ;; esac
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
    for s in $SEEDS; do for ph in $PHASES; do r=$(run_of "$s" "$ph"); m=$HOME/robodojo_master/$r
      st="not started"; [[ -f $m/state.json ]] && st="$(master_alive "$r" && echo live || echo ended) $(run_totals "$m/state.json")"
      complete "$r" && st+=" COMPLETE"; [[ -e $CHAIN_DIR/hold.$r ]] && st+=" HELD"
      printf 'seed %s %-5s  %s\n' "$s" "$ph" "$st"; done; done
    for f in "$CHAIN_DIR"/progress.*; do [[ -e $f ]] && printf '  %s: %s\n' "${f##*_ee_}" "$(cut -c1-200 "$f")"; done
    ls "$CHAIN_DIR"/att.* >/dev/null 2>&1 && { echo "attention notes:"; cat "$CHAIN_DIR"/att.* | sed 's/^/  /' | cut -c1-300; } ;;
  free-spec) spec=$(free_spec); echo "$spec"; log "$(spec_count "$spec") free GPUs" >&2 ;;
  run) seed=${1:?seed}; ph=${2:?sink28}; shift 2; drive "$seed" "$ph" "$@" ;;
  smoke-check)   # judge the seed-0 smoke (however it was dispatched) with the gate's own rule; passed -> smoke_passed
    smoke_check; rc=$?; log "smoke-check: $rc (0 passed, 1 not finished, 2 failed)"
    (( rc == 0 )) && { touch "$CHAIN_DIR/smoke_passed"; log "passed -> $CHAIN_DIR/smoke_passed"; } ;;
  combined)
    combined sink28 $(for s in $SEEDS; do run_of "$s" sink28; done) ;;
  *) sed -n 2,22p "$0"; exit 1 ;;
esac
