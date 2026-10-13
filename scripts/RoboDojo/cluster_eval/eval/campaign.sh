#!/usr/bin/env bash
# campaign.sh DEF VERB ... -- a multi-seed evaluation CAMPAIGN (control host) driven by a definition file (eval/defs/*.env):
# one or more checkpoints (KEYS), each evaluated on the complete official protocol once per eval seed (SEEDS, default 0 1 2),
# one seed at a time on every free GPU, through eval/run.sh. The background daemon (`chain start`) does, in order:
#   1. wait until every run in WAIT_RUNS is completely over (a live master, a worker on a node, a dead master with units not
#      done; with WAIT_NOT_STARTED=1 a waited-for run that has not started yet counts as pending too: queue behind a whole
#      planned series, never into the gap between two of its seeds)
#   2. wait for the checkpoints to be verified (`verify` writes verified.<label> after sha256 + normalization checks)
#   3. seed-0 SMOKE GATE: one insert_key episode per checkpoint on a free GPU; the adapter's ready line must carry every token
#      `run.sh expect-ready` derives from the settings plus the def's EXPECT_<key> tokens
#   4. per seed: start each checkpoint's run on its share of the free GPUs (up to PER_RUN each, never fewer than MIN_PER_RUN;
#      topped up from GPUs freed later during TOPUP_WINDOW), finish the long unsharded units with the TAIL TEMPLATE once the
#      queue is empty, check the tail for overlaps, assemble, write the run's official summary
#   5. after the last seed: the combined 3-seed official summary per checkpoint, then the 3-seed report against BASELINE_<key>
# Anything unexpected about a run writes an ATTENTION line and the hold file <chain dir>/hold.<RUN>: the daemon takes no further
# automatic action on that run until the file is removed (or moved away). Chain dir: <CE_MASTER_ROOT>/.scratch/chain_<CAMPAIGN>.
#
#   bash campaign.sh DEF show                       resolved settings, run names, overrides, smoke tokens
#   bash campaign.sh DEF download [KEY ...]         fetch HF_<key> into <CE_CKPT_ROOT>/<label> on the gateway (detached)
#   bash campaign.sh DEF verify [KEY ...]           sha256 (SHA256SUMS / MANIFEST.json) + NORM_<key> == the snapshot's
#                                                   normalization_sha256 -> <chain dir>/verified.<label>
#   bash campaign.sh DEF chain start|stop|status    the daemon (log <chain dir>/chain.log)
#   bash campaign.sh DEF watch [HOURS]              block until the log shows EVENT / ATTENTION / SUMMARY lines (or HOURS pass)
#   bash campaign.sh DEF status | wait-check | free-spec | smoke-check | combined | report
#   bash campaign.sh DEF run SEED KEY VERB ...      eval/run.sh with exactly this campaign's settings, e.g. `run 0 a report`
# Environment overrides (win over the def): SEEDS PER_RUN MIN_PER_RUN START_GRACE TOPUP_WINDOW WAIT_RUNS WAIT_NOT_STARTED NODES
# RESERVE TICK WAIT_INTERVAL CONFIRM_INTERVAL CONFIRM_CHECKS MIN_IDLE MAX_RELAUNCH_ROUNDS (e.g. SEEDS=0 = stop after seed 0).
set -uo pipefail
EVAL=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SELF=$EVAL/campaign.sh; RUN_DRIVER=$EVAL/run.sh
DEF=${1:-}; shift || true
if [[ -z $DEF ]]; then sed -n 2,31p "$SELF"; exit 1; fi
[[ -f $DEF ]] || DEF=$EVAL/defs/$DEF.env
[[ -f $DEF ]] || { echo "no campaign definition $DEF" >&2; exit 2; }
DEF=$(cd "$(dirname "$DEF")" && pwd)/$(basename "$DEF")
# shellcheck source=../lib/site.sh
source "$EVAL/../lib/site.sh" || exit 2
# shellcheck source=../lib/adapters.sh
source "$EVAL/../lib/adapters.sh"
# shellcheck source=../lib/slots.sh
source "$EVAL/../lib/slots.sh"
PY=$CE_PY

# run knobs come from the def ONLY (a caller's stray VIDEO_CFG must not leak into a run); chain knobs may be overridden
KNOBS=(ADAPTER NFE VIDEO_CFG ACTION_CFG VIDEO_FS ACTION_FS NAS ACTION_TYPE VIDEO_LAYOUT VIEW_RESIZE EXTRA_OVERRIDES SHARD_PLAN DIM_ORDER PIN_LONGEST)
unset "${KNOBS[@]}" RUN CKPT CKPT_LABEL NORM_SHA SEED EXPECT BASELINES SCRATCH SUMMARY_RUNS EXTRA_MASTER_ARGS \
      SMOKE_SEED SMOKE_RUN_ROOT SMOKE_UNIT_ID SMOKE_LAYOUT_IDS SMOKE_LAYOUT_TOTAL SMOKE_RPC_PORT SMOKE_WS_PORT SMOKE_NO_PULL
CHAIN_KNOBS=(SEEDS PER_RUN MIN_PER_RUN START_GRACE TOPUP_WINDOW WAIT_RUNS WAIT_NOT_STARTED NODES RESERVE TICK WAIT_INTERVAL
             CONFIRM_INTERVAL CONFIRM_CHECKS MIN_IDLE MAX_RELAUNCH_ROUNDS)
declare -A _over=()
for _v in "${CHAIN_KNOBS[@]}"; do [[ -v $_v ]] && _over[$_v]=${!_v}; done
# shellcheck disable=SC1090
source "$DEF"
for _v in "${!_over[@]}"; do printf -v "$_v" '%s' "${_over[$_v]}"; done
CAMPAIGN=${CAMPAIGN:-$(basename "$DEF" .env)}
ADAPTER=${ADAPTER:-SANA_WAM}; ce_adapter_profile "$ADAPTER" || exit 2
KEYS=${KEYS:-a}
SEEDS=${SEEDS:-0 1 2}
PER_RUN=${PER_RUN:-999}           # GPUs per checkpoint and seed (999 = every free GPU)
MIN_PER_RUN=${MIN_PER_RUN:-40}    # never start a run on fewer
START_GRACE=${START_GRACE:-300}   # wait up to this long for PER_RUN free GPUs per run before starting on fewer (topped up later)
TOPUP_WINDOW=${TOPUP_WINDOW:-10800}
WAIT_RUNS=${WAIT_RUNS:-}; WAIT_NOT_STARTED=${WAIT_NOT_STARTED:-1}
WAIT_INTERVAL=${WAIT_INTERVAL:-300}; CONFIRM_INTERVAL=${CONFIRM_INTERVAL:-120}; CONFIRM_CHECKS=${CONFIRM_CHECKS:-1}
TICK=${TICK:-90}
NODES=${NODES:-$CE_NODES}; RESERVE=${RESERVE-$CE_RESERVE}
read -r -a TAIL_TASKS <<<"${TAIL_TASKS:-play_tic_tac_toe imitate_sorting_sequence fill_egg_holder general_pickup}"
MIN_IDLE=${MIN_IDLE:-4}; MAX_RELAUNCH_ROUNDS=${MAX_RELAUNCH_ROUNDS:-6}
FIRST_SEED=${SEEDS%% *}
CHAIN_DIR=$CE_MASTER_ROOT/.scratch/chain_$CAMPAIGN; mkdir -p "$CHAIN_DIR"
LOG=$CHAIN_DIR/chain.log; PIDF=$CHAIN_DIR/chain.pid
for k in $KEYS; do
  v=LABEL_$k; [[ -n ${!v:-} ]] || { echo "$DEF: $v is not set" >&2; exit 2; }
  v=NORM_$k; [[ ${!v:-} =~ ^[0-9a-f]{64}$ ]] || { echo "$DEF: $v must be the snapshot's normalization_sha256 (64 hex)" >&2; exit 2; }
done

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
kv(){ local v="${1}_$2"; echo "${!v:-}"; }       # kv LABEL a -> $LABEL_a
label_of(){ kv LABEL "$1"; }
drv(){ # SEED KEY CMD... : CMD with exactly this campaign's run settings for that checkpoint and seed (RUN = the run.sh name)
  local seed=$1 k=$2 envs=() n; shift 2
  envs+=(ADAPTER="$ADAPTER" CKPT_LABEL="$(label_of "$k")" NORM_SHA="$(kv NORM "$k")" SEED="$seed")
  [[ -n $(kv CKPT "$k") ]] && envs+=(CKPT="$(kv CKPT "$k")")
  for n in "${KNOBS[@]}"; do [[ $n != ADAPTER && -v $n ]] && envs+=("$n=${!n}"); done
  env "${envs[@]}" "$@"
}
drive(){ local seed=$1 k=$2; shift 2; drv "$seed" "$k" bash "$RUN_DRIVER" "$@"; }
declare -A RUN_NAME=()
run_of(){ # SEED KEY -> the run name (eval/run.sh name)
  local key="$1/$2"
  [[ -n ${RUN_NAME[$key]:-} ]] || RUN_NAME[$key]=$(drive "$1" "$2" name)
  echo "${RUN_NAME[$key]}"
}
for s in $SEEDS; do for k in $KEYS; do run_of "$s" "$k" >/dev/null; done; done
master_alive(){ ce_master_alive "$1"; }
free_spec(){ ce_free_spec "$RPC_BASE" "$WS_BASE" "$NODES" "$RESERVE"; }

runs_pending(){ # the WAIT_RUNS not over: a live master, a worker of its run root on a node, or units not done (dead master)
  local run pending="" pat n node st nd
  [[ -n ${WAIT_RUNS// /} ]] || { echo ""; return 0; }   # nothing to wait for (an empty grep pattern would match every process)
  for run in $WAIT_RUNS; do master_alive "$run" && pending+=" $run(master)"; done
  pat=$(for run in $WAIT_RUNS; do printf '[r]obodojo_eval/%s |' "$run"; done); pat=${pat%|}
  for node in $NODES; do
    n=$(timeout 60 ssh -n $CE_SSH_OPTS "$(ce_host "$node")" "ps -eo args | grep -E -c '$pat' || true" 2>/dev/null) \
      || { log "wait: $node unreachable (not counted)"; continue; }
    [[ ${n:-0} -gt 0 ]] && pending+=" $node(${n}workers)"
  done
  if [[ -z $pending ]]; then
    for run in $WAIT_RUNS; do
      st=$CE_MASTER_ROOT/$run/state.json
      if [[ ! -f $st ]]; then (( WAIT_NOT_STARTED )) && pending+=" $run(not-started)"; continue; fi
      nd=$("$PY" -c "import json,sys; print(sum(u['state'] != 'done' for u in json.load(open(sys.argv[1]))['units']))" "$st" 2>/dev/null || echo "?")
      if [[ $nd != 0 ]]; then
        pending+=" $run(dead-master,$nd-units-not-done)"
        attention "dit-incomplete-$run" "waited-for run $run: master not running but $nd units are not done (crashed or stopped?) -- the chain keeps waiting"
      fi
    done
  fi
  echo "${pending# }"
}
ckpts_ready(){ local k; for k in $KEYS; do [[ -e $CHAIN_DIR/verified.$(label_of "$k") ]] || return 1; done; }

# ---- checkpoints --------------------------------------------------------------------------------------------------------
download(){ # KEY: download_hf_repo.sh (node/) on the gateway, detached
  local k=$1 label repo
  label=$(label_of "$k"); repo=$(kv HF "$k")
  [[ -n $repo ]] || { echo "$k: no HF_$k in the def (download $label by hand into $CE_CKPT_ROOT/$label)"; return 1; }
  ssh -n $CE_SSH_OPTS "$(ce_gw)" "mkdir -p '$CE_REMOTE_INSTALL/logs'; cd '$CE_REMOTE_INSTALL' || exit 1
    nohup setsid env REPO='$repo' LABEL='$label' DEST='$CE_CKPT_ROOT/$label' bash ./download_hf_repo.sh > 'logs/hfrepo_$label.nohup.log' 2>&1 < /dev/null &
    sleep 2; echo \"launched on \$(hostname -s), pid \$!\"" || { echo "launch FAILED on $CE_GATEWAY"; return 1; }
  echo "$k: $repo -> $CE_CKPT_ROOT/$label; log $CE_REMOTE_INSTALL/logs/hfrepo_$label.nohup.log (wait for 'HF REPO READY'), then: verify $k"
}
verify(){ # KEY: digests + normalization -> verified.<label>
  local k=$1 label norm out rc
  label=$(label_of "$k"); norm=$(kv NORM "$k")
  out=$(timeout 3600 ssh -n $CE_SSH_OPTS "$(ce_gw)" "set -u; cd '$CE_CKPT_ROOT/$label' || exit 4
    ls *.incomplete .cache/huggingface/download/*.incomplete 2>/dev/null | head -3 | sed 's/^/INCOMPLETE /'
    if [ -s SHA256SUMS ]; then sha256sum -c SHA256SUMS || exit 3
    elif [ -s MANIFEST.json ]; then python3 - <<'PY' || exit 3
import hashlib, json, sys
bad = 0
for f in json.load(open('MANIFEST.json')).get('files', []):
    if not f.get('sha256'): continue
    h = hashlib.sha256()
    with open(f['name'], 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 24), b''): h.update(c)
    ok = h.hexdigest() == f['sha256']; bad += not ok; print(f\"{f['name']}: {'OK' if ok else 'FAILED'}\")
sys.exit(1 if bad else 0)
PY
    else echo 'no SHA256SUMS / MANIFEST.json: nothing to verify against'; exit 3; fi
    echo \"normalization_sha256 \$(grep -m1 -o 'normalization_sha256: *[0-9a-f]*' config.yaml | awk '{print \$2}')\"" 2>&1); rc=$?
  printf '%s\n' "$out" | tail -n 12
  if (( rc != 0 )) || grep -q '^INCOMPLETE ' <<<"$out"; then echo "$k: $label NOT verified (rc $rc)"; return 1; fi
  if ! grep -q "^normalization_sha256 $norm\$" <<<"$out"; then
    echo "$k: NORM_$k ($norm) != the snapshot's config.yaml normalization_sha256 -- fix the def"; return 1
  fi
  { date -u +%FT%TZ; printf '%s\n' "$out"; } >"$CHAIN_DIR/verified.$label"
  echo "$k: $label verified -> $CHAIN_DIR/verified.$label"
}

# ---- seed-0 smoke gate --------------------------------------------------------------------------------------------------
smoke_check(){ # KEY: 0 passed, 1 not yet, 2 failed
  local k=$1 out rc
  out=$(drv "$FIRST_SEED" "$k" env EXPECT="$(kv EXPECT "$k")" SMOKE_SEED="$FIRST_SEED" bash "$RUN_DRIVER" smoke-verify insert_key 1 2>&1); rc=$?
  case $rc in
    0) log "smoke $k ${out:0:1600}" ;;
    1) ;;
    *) log "smoke $k: ${out:0:1600}"; rc=2 ;;
  esac
  return $rc
}
smoke_gate(){ # 0 passed (or passed before), 1 retry later, 2 failed; one insert_key episode per key, on different nodes if possible
  [[ -e $CHAIN_DIR/smoke_passed ]] && return 0
  local spec picks k i=0 rc all_ok failed t0 ng msg="" nk arr
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
    read -r -a arr <<<"$picks"
    for k in $KEYS; do
      ng=${arr[$i]}
      log "smoke: $k on $ng (insert_key x 1, seed $FIRST_SEED)"
      drv "$FIRST_SEED" "$k" env SMOKE_SEED="$FIRST_SEED" SMOKE_RPC_PORT=$((SMOKE_RPC + i)) SMOKE_WS_PORT=$((SMOKE_WS + i)) \
        bash "$RUN_DRIVER" smoke-gpu "$ng" insert_key 1 2>&1 | tail -n 3 | sed 's/^/   /' >&2
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
    if [[ -n $failed ]]; then attention smoke-failed "smoke FAILED ($failed): see the lines above and bash $SELF $DEF run $FIRST_SEED <key> smoke-status"; return 2; fi
    if (( $(date +%s) - t0 > 2400 )); then attention smoke-timeout "smoke not finished after 40 min"; return 2; fi
    sleep 60
  done
}

# ---- one run ------------------------------------------------------------------------------------------------------------
run_info(){ # STATE SEED -> idle single-GPU slots (outside CE_TAIL_EXCLUDE_NODES), others-not-done, queued, failed, slots, tail unit states
  "$PY" - "$1" "$2" "$CE_TAIL_EXCLUDE_NODES" "${TAIL_TASKS[@]}" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1])); seed = sys.argv[2]; excl = set(sys.argv[3].split()); tasks = sys.argv[4:]
tails = {f"{t}__s{seed}" for t in tasks}
idle = sum(1 for sl in s["slots"] if not sl.get("unit") and not sl.get("draining") and sl["server_gpu"] == sl["client_gpu"]
           and sl["node"] not in excl)
st = {u["unit_id"]: u["state"] for u in s["units"]}
others = sum(1 for u, v in st.items() if u not in tails and v != "done")
print(idle, others, sum(v == "queued" for v in st.values()), sum(v == "failed" for v in st.values()), len(s["slots"]),
      " ".join(f"{t}={st.get(f'{t}__s{seed}', 'absent')}" for t in tasks))
PYEOF
}
run_totals(){ # STATE -> done units / units, episodes, successes
  "$PY" - "$1" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1])); us = s["units"]; done = [u for u in us if u["state"] == "done"]
eps = sum((u.get("result") or {}).get("episodes_recorded", 0) for u in done)
succ = sum((u.get("result") or {}).get("successes", 0) for u in done)
print(f"{len(done)}/{len(us)} units, {eps}/{sum(u['episodes_per_task'] for u in us)} eps, {succ} successes")
PYEOF
}
summary_line(){ # the overview row of an official summary: SR / score per dimension and overall
  awk -v p="| $POLICY |" 'index($0, p) == 1 {print; exit}' "$1" | sed 's/ ±0\.00//g; s/<nobr>//g; s/<\/nobr>//g' | cut -c1-400
}
manage_run(){ # SEED KEY -> 0 complete, 1 in progress / held
  local seed=$1 k=$2 R M idle others queued failed nslots rest t ust out resolved=0 launched=0 summ="" asm=() rr f
  R=$(run_of "$seed" "$k"); M=$CE_MASTER_ROOT/$R
  [[ -e $CHAIN_DIR/complete.$R ]] && return 0
  [[ -e $CHAIN_DIR/hold.$R ]] && return 1
  [[ -f $M/state.json ]] || return 1
  if ! master_alive "$R"; then
    if [[ $("$PY" -c "import json,sys; print(sum(u['state'] != 'done' for u in json.load(open(sys.argv[1]))['units']))" "$M/state.json") != 0 ]]; then
      attention "$R" "$R: master not running but units are not done ($(run_totals "$M/state.json")) -- held"; return 1
    fi
    out=$(drive "$seed" "$k" official-summary 2>&1); f=$(grep -o '[^ ]*OFFICIAL_SUMMARY_[0-9T]*\.md' <<<"$out" | tail -n 1)
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
start_seed(){ # SEED: start the runs of this seed that do not exist yet; 0 started (or nothing to start), 1 retry later, 2 refused
  local seed=$1 k todo=() spec n parts i rc=0 rc_k first now msg="" out
  for k in $KEYS; do [[ -f $CE_MASTER_ROOT/$(run_of "$seed" "$k")/state.json ]] || todo+=("$k"); done
  [[ ${#todo[@]} -gt 0 ]] || return 0
  spec=$(free_spec); n=$(ce_spec_count "$spec")
  first=$(cat "$CHAIN_DIR/start_wait.$seed" 2>/dev/null || { date +%s | tee "$CHAIN_DIR/start_wait.$seed"; }); now=$(date +%s)
  if (( n < ${#todo[@]} * PER_RUN && now - first < START_GRACE )); then log "seed $seed: $n free GPUs < ${#todo[@]} x $PER_RUN; waiting (grace $(( START_GRACE - now + first )) s)"; return 1; fi
  if (( n < ${#todo[@]} * MIN_PER_RUN )); then
    (( now - first > 3600 )) && attention "few-gpus-$seed" "seed $seed: only $n free GPUs for ${#todo[@]} run(s) for over an hour ($spec)"
    log "seed $seed: only $n free GPUs ($spec); waiting"; return 1
  fi
  mapfile -t parts < <(ce_deal "$spec" "${#todo[@]}" "$PER_RUN")
  for i in "${!todo[@]}"; do
    k=${todo[$i]}
    log "seed $seed $k: start on $(ce_spec_count "${parts[$i]}") GPUs: ${parts[$i]}"
    out=$(drive "$seed" "$k" start "${parts[$i]}" 2>&1); rc_k=$?
    printf '%s\n' "$out" | tail -n 4 | sed 's/^/   /' >&2
    if (( rc_k != 0 )); then attention "start-refused-$seed-$k" "seed $seed $k: start refused (rc $rc_k): $(tail -n 2 <<<"$out" | tr '\n' ' ' | cut -c1-300)"; rc=2
    else date +%s >"$CHAIN_DIR/started.$(run_of "$seed" "$k")"; msg+="$k $(ce_spec_count "${parts[$i]}") GPUs; "; fi
  done
  sleep 30
  for k in "${todo[@]}"; do
    [[ -e $CHAIN_DIR/started.$(run_of "$seed" "$k") ]] || continue
    master_alive "$(run_of "$seed" "$k")" || { attention "start-dead-$seed-$k" "seed $seed $k: no live master 30 s after the start"; rc=2; }
  done
  (( rc == 0 )) && event "seed $seed STARTED: $msg"
  return $rc
}
topup(){ # SEED: a run of this seed below PER_RUN slots, with queued units, inside TOPUP_WINDOW of its start -> add free GPUs
  local seed=$1 k R M nslots queued spec part need t0
  for k in $KEYS; do
    R=$(run_of "$seed" "$k"); M=$CE_MASTER_ROOT/$R
    master_alive "$R" || continue
    [[ -e $CHAIN_DIR/hold.$R || -e $M/control/add_slots ]] && continue
    t0=$(cat "$CHAIN_DIR/started.$R" 2>/dev/null || echo 0); (( $(date +%s) - t0 < TOPUP_WINDOW )) || continue
    read -r nslots queued <<<"$("$PY" -c "import json,sys; s=json.load(open(sys.argv[1])); print(len(s['slots']), sum(u['state']=='queued' for u in s['units']))" "$M/state.json")"
    (( nslots < PER_RUN && queued > 0 )) || continue
    (( $(date +%s) - $(cat "$CHAIN_DIR/topup_probe" 2>/dev/null || echo 0) >= 600 )) || continue   # probe the nodes at most every 10 min
    date +%s >"$CHAIN_DIR/topup_probe"
    spec=$(free_spec); [[ -n $spec ]] || continue
    need=$(( PER_RUN - nslots )); part=$(ce_deal "$spec" 1 "$need" | head -n 1)
    [[ -n $part ]] || continue
    mkdir -p "$M/control"; printf '%s\n' "$part" >"$M/control/add_slots.tmp" && mv -f "$M/control/add_slots.tmp" "$M/control/add_slots"
    log "seed $seed $k: top-up add_slots '$part' ($(ce_spec_count "$part") GPUs; had $nslots)"
  done
}

combined(){ # KEY -> the combined official summary over SEEDS (prints the summary path)
  local k=$1 out f
  out=$(drv "$FIRST_SEED" "$k" env SUMMARY_RUNS="$(for s in $SEEDS; do run_of "$s" "$k"; done | paste -sd' ')" bash "$RUN_DRIVER" official-summary 2>&1)
  f=$(grep -o '[^ ]*OFFICIAL_SUMMARY_combined_[0-9T]*\.md' <<<"$out" | tail -n 1)
  [[ -s $f ]] || { printf '%s\n' "$out" | tail -n 3 >&2; return 1; }
  echo "$f"
}
report(){ # KEY: three_seed_report.py of this checkpoint's latest combined summary vs BASELINE_<key>
  local k=$1 r0 a b base ts out
  r0=$(run_of "$FIRST_SEED" "$k"); base=$(kv BASELINE "$k")
  [[ -n $base ]] || { echo "$k: no BASELINE_$k in the def"; return 1; }
  a=$(ls -t "$CE_RESULTS_ROOT/runs/$r0"/OFFICIAL_SUMMARY_combined_*.md 2>/dev/null | head -n 1)
  [[ -n $a ]] || { echo "$k: no combined summary under $CE_RESULTS_ROOT/runs/$r0 (bash $SELF $DEF combined)"; return 1; }
  if [[ $base == *.md ]]; then b=$base; [[ $b == /* ]] || b=$CE_RESULTS_ROOT/$b
  else b=$(ls -t "$CE_RESULTS_ROOT/runs/$base"/OFFICIAL_SUMMARY_combined_*.md 2>/dev/null | head -n 1); fi
  [[ -s $b ]] || { echo "$k: baseline summary not found ($base)"; return 1; }
  ts=$(date -u +%Y%m%dT%H%M%S); out=$CE_RESULTS_ROOT/per_task_${CAMPAIGN}_${k}_3seeds_$ts.md
  "$PY" "$EVAL/three_seed_report.py" "$a" "$b" "$CE_RESULTS_ROOT/runs/$r0/tasks.json" --out "$out" \
    --labels "$(n=$(kv NAME "$k"); echo "${n:-$(label_of "$k")}")" "$(n=$(kv BASELINE_NAME "$k"); echo "${n:-$(basename "${base%.md}")}")" && echo "-> $out"
}

chain_loop(){
  local seed pending ok i k all r last_hb=0 rc s0_exists=0 f
  log "chain $CAMPAIGN ($DEF): adapter $ADAPTER, keys [$(for k in $KEYS; do printf '%s=%s ' "$k" "$(label_of "$k")"; done)], seeds [$SEEDS], up to $PER_RUN GPUs per run, waits for [${WAIT_RUNS:-nothing}]"
  for k in $KEYS; do [[ -f $CE_MASTER_ROOT/$(run_of "$FIRST_SEED" "$k")/state.json ]] && s0_exists=1; done
  if (( ! s0_exists )); then
    # 1. the waited-for runs must be completely over
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
    # 2. the checkpoints verified, then the smoke
    until ckpts_ready; do attention ckpts-not-verified "checkpoints not verified (bash $SELF $DEF verify); waiting"; sleep 300; done
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
      for k in $KEYS; do r=$(run_of "$seed" "$k"); [[ -f $CE_MASTER_ROOT/$r/state.json ]] || all=0; manage_run "$seed" "$k" || all=0; done
      (( all )) && { event "SEED $seed COMPLETE: run(s) assembled and summarized"; break; }
      topup "$seed"
      if (( $(date +%s) - last_hb >= 3600 )); then
        last_hb=$(date +%s)
        for k in $KEYS; do r=$(run_of "$seed" "$k"); [[ -f $CE_MASTER_ROOT/$r/state.json ]] && log "hourly: seed $seed $k $(run_totals "$CE_MASTER_ROOT/$r/state.json") | $(cut -d'|' -f2- "$CHAIN_DIR/progress.$r" 2>/dev/null)"; done
      fi
      sleep "$TICK"
    done
  done
  # 4. the combined official tables over the seeds, 5. the reports
  if [[ $SEEDS == *" "* ]]; then
    for k in $KEYS; do
      f=$(combined "$k") && { event "COMBINED $k: $f"; log "SUMMARY $(wc -w <<<"$SEEDS") seeds $k $(summary_line "$f")"; }
      [[ -n $(kv BASELINE "$k") ]] && { out=$(report "$k" 2>&1); event "REPORT $k: $(tail -n 1 <<<"$out")"; }
    done
  fi
  event "CHAIN DONE: seeds [$SEEDS] complete"
}

cmd=${1:-help}; shift || true
case $cmd in
  show)
    echo "campaign $CAMPAIGN ($DEF); chain dir $CHAIN_DIR"
    echo "adapter $ADAPTER; seeds [$SEEDS]; PER_RUN $PER_RUN (min $MIN_PER_RUN); nodes [$NODES]; reserve [$RESERVE]; waits for [${WAIT_RUNS:-nothing}] (not-started pending: $WAIT_NOT_STARTED)"
    echo "tail tasks: ${TAIL_TASKS[*]}"
    for k in $KEYS; do
      hf=HF_$k
      echo "== key $k: $(label_of "$k")  HF ${!hf:-<none>}  baseline $(kv BASELINE "$k")"
      for s in $SEEDS; do echo "   seed $s: $(run_of "$s" "$k")"; done
      drive "$FIRST_SEED" "$k" show | sed -n '2,5p' | sed 's/^/   /'
      t=$(kv EXPECT "$k"); echo "   smoke ready-line tokens: $(drive "$FIRST_SEED" "$k" expect-ready | paste -sd'|')${t:+|$t}"
    done ;;
  download) for k in ${*:-$KEYS}; do download "$k"; done ;;
  verify) rc=0; for k in ${*:-$KEYS}; do verify "$k" || rc=1; done; exit $rc ;;
  chain)
    sub=${1:-status}
    case $sub in
      start)
        if [[ -s $PIDF ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "chain already running (pid $(cat "$PIDF")), log $LOG"; exit 0; fi
        nohup setsid bash "$SELF" "$DEF" chain-loop >>"$LOG" 2>&1 </dev/null &
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
    echo "waited-for runs:"; for r in $WAIT_RUNS; do printf '  %s  %s\n' "$(master_alive "$r" && echo live || echo ended)" "$(grep -m1 '^units:' "$CE_MASTER_ROOT/$r/STATUS.md" 2>/dev/null | cut -c1-110)"; done
    for s in $SEEDS; do for k in $KEYS; do r=$(run_of "$s" "$k"); m=$CE_MASTER_ROOT/$r
      st="not started"; [[ -f $m/state.json ]] && st="$(master_alive "$r" && echo live || echo ended) $(run_totals "$m/state.json")"
      [[ -e $CHAIN_DIR/complete.$r ]] && st+=" COMPLETE"; [[ -e $CHAIN_DIR/hold.$r ]] && st+=" HELD"
      printf 'seed %s %s  %s\n' "$s" "$k" "$st"; done; done
    for k in $KEYS; do [[ -e $CHAIN_DIR/verified.$(label_of "$k") ]] || echo "key $k: checkpoint not verified"; done
    [[ -e $CHAIN_DIR/smoke_passed ]] && echo "smoke passed" || echo "smoke not passed"
    ls "$CHAIN_DIR"/att.* >/dev/null 2>&1 && { echo "attention notes:"; cat "$CHAIN_DIR"/att.* | sed 's/^/  /' | cut -c1-300; }
    true ;;
  wait-check) pending=$(runs_pending); log "waited-for runs still active: ${pending:-none}"; spec=$(free_spec); log "free now: $(ce_spec_count "$spec") GPUs ($spec)" ;;
  free-spec) spec=$(free_spec); echo "$spec"; log "$(ce_spec_count "$spec") free GPUs" ;;
  run) seed=${1:?seed}; k=${2:?key}; shift 2; drive "$seed" "$k" "$@" ;;
  smoke-check)   # judge the seed-0 smokes (however they were dispatched) with the gate's own rule; all passed -> smoke_passed
    rcs=""; ok=1; for k in $KEYS; do smoke_check "$k"; rc=$?; rcs+=" $k=$rc"; (( rc == 0 )) || ok=0; done
    log "smoke-check:$rcs (0 passed, 1 not finished, 2 failed)"
    (( ok )) && { touch "$CHAIN_DIR/smoke_passed"; log "all passed -> $CHAIN_DIR/smoke_passed"; } ;;
  combined) for k in $KEYS; do f=$(combined "$k") && echo "$k: $f" && summary_line "$f"; done ;;
  report) for k in $KEYS; do report "$k"; done ;;
  *) sed -n 2,31p "$SELF"; exit 1 ;;
esac
