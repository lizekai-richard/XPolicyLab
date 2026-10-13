#!/usr/bin/env bash
# Grow one or more running masters onto GPUs that free up, as single-GPU slots (control/add_slots 'node:sG').
# A GPU is taken only if: healthy board, < 1 GiB used CONTINUOUSLY for >= MIN_FREE s (default 600; other schedulers on
# this account -- robolab_* -- grab a freed GPU within minutes, so a GPU free that long is one they are not waiting for;
# 2026-09-26 12:52 a 20 s check collided with a robolab job on l3 GPU4/5), still free on a look 20 s later, no live master
# holds it, and nothing listens on the slot's ports (rpc PORT_RPC+g / ws PORT_WS+g) -- the master's port sweep would TERM it.
# Each addition goes to the run with fewer slots that still has more queued units than idle slots. Never kills anything.
# Exits when both masters have ended.
set -u
PY=$HOME/miniforge3/envs/sana-wam/bin/python
# usage: RUNS="<run> <run> ..." [PORT_RPC=63250 PORT_WS=65250 (the runs' base + 50)] bash grow_single_gpu_slots.sh
read -r -a RUNS <<<"${RUNS:?RUNS=\"run1 run2\" (dirs under ~/robodojo_master)}"
PORT_RPC=${PORT_RPC:-63250}; PORT_WS=${PORT_WS:-65250}
NODES=${NODES:-"l1 l2 l3 l5 l6 l7 l8 l9 l25 l26 l27 l30"}
INTERVAL=${INTERVAL:-120}
MIN_FREE=${MIN_FREE:-600}
declare -A free_since
log(){ echo "[$(date -u +%FT%TZ)] $*"; }

scan(){ # prints 'node g usedMiB listeners' for every healthy CUDA device of every node
  for h in $NODES; do
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "jiamingt@$h.mit.edu" '
      healthy=($(nvidia-smi --query-gpu=index,pstate --format=csv,noheader | awk -F", " "\$2 !~ /reset/ {print \$1}"))
      for g in $(seq 0 $((${#healthy[@]} - 1))); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${healthy[$g]} | tr -d " ")
        lis=$(ss -ltnH 2>/dev/null | awk "{print \$4}" | grep -cE ":($(('"$PORT_RPC"' + g))|$(('"$PORT_WS"' + g)))$")
        echo "$(hostname -s) $g $used $lis"
      done' 2>/dev/null
  done
}
held(){ # node:gpu of every slot of every live master
  "$PY" - "$HOME/robodojo_master" <<'PYEOF'
import json, os, pathlib, sys
out = set()
for pidf in pathlib.Path(sys.argv[1]).glob("*/master.pid"):
    try: os.kill(int(pidf.read_text().strip()), 0)
    except Exception: continue
    st = pidf.with_name("state.json")
    if st.exists():
        for sl in json.loads(st.read_text())["slots"]:
            out.update({f'{sl["node"]}:{sl["server_gpu"]}', f'{sl["node"]}:{sl["client_gpu"]}'})
print("\n".join(sorted(out)))
PYEOF
}
need(){ # per run: 'run slots queued idle alive'
  for r in "${RUNS[@]}"; do
    "$PY" - "$HOME/robodojo_master/$r" <<'PYEOF'
import json, os, pathlib, sys
d = pathlib.Path(sys.argv[1]); alive = 0
try: os.kill(int((d / "master.pid").read_text().strip()), 0); alive = 1
except Exception: pass
s = json.loads((d / "state.json").read_text())
q = sum(u["state"] == "queued" for u in s["units"]); idle = sum(1 for sl in s["slots"] if not sl.get("unit") and not sl.get("draining"))
pending = (d / "control" / "add_slots").exists()
print(d.name, len(s["slots"]), q, idle, alive, int(pending))
PYEOF
  done
}

log "grow loop started: nodes $NODES, every ${INTERVAL}s, a GPU must stay free ${MIN_FREE}s"
while :; do
  mapfile -t st < <(need)
  alive=0; for l in "${st[@]}"; do read -r _ _ _ _ a _ <<<"$l"; alive=$((alive + a)); done
  (( alive == 0 )) && { log "both masters ended; grow loop exits"; exit 0; }
  h=$(held)
  now=$(date +%s)
  seen=$(scan | awk '$3 < 1024 && $4 == 0 {print $1":"$2}' | grep -vxF -f <(echo "$h") || true)
  for k in "${!free_since[@]}"; do grep -qxF "$k" <<<"$seen" || unset "free_since[$k]"; done
  for k in $seen; do [[ -n ${free_since[$k]:-} ]] || free_since[$k]=$now; done
  cand1=$(for k in "${!free_since[@]}"; do (( now - ${free_since[$k]} >= MIN_FREE )) && echo "$k"; done | sort)
  if [[ -n $cand1 ]]; then
    sleep 20
    h=$(held)
    cand=$(comm -12 <(echo "$cand1" | sort) <(scan | awk '$3 < 1024 && $4 == 0 {print $1":"$2}' | grep -vxF -f <(echo "$h") | sort) || true)
    declare -A slots queued idle alivev pend add
    for l in "${st[@]}"; do read -r r n q i a p <<<"$l"; slots[$r]=$n; queued[$r]=$q; idle[$r]=$i; alivev[$r]=$a; pend[$r]=$p; add[$r]=""; done
    for c in $cand; do
      best=""
      for r in "${RUNS[@]}"; do
        (( ${alivev[$r]} == 1 && ${pend[$r]} == 0 && ${queued[$r]} > ${idle[$r]} )) || continue
        if [[ -z $best ]] || (( ${slots[$r]} < ${slots[$best]} )); then best=$r; fi
      done
      [[ -n $best ]] || break
      node=${c%%:*}; g=${c##*:}
      add[$best]+="$node:s$g "; slots[$best]=$((${slots[$best]} + 1)); idle[$best]=$((${idle[$best]} + 1)); unset "free_since[$c]"
    done
    for r in "${RUNS[@]}"; do
      [[ -n ${add[$r]} ]] || continue
      spec=$(echo "${add[$r]}" | tr ' ' '\n' | sed '/^$/d' | awk -F: '{a[$1]=a[$1] (a[$1] ? "," : "") $2} END {for (n in a) printf "%s:%s;", n, a[n]}' | sed 's/;$//')
      d=$HOME/robodojo_master/$r/control; mkdir -p "$d"
      if [[ -n ${DRY:-} ]]; then log "[dry] would add_slots $spec to $r"; continue; fi
      printf '%s\n' "$spec" >"$d/add_slots.tmp" && mv -f "$d/add_slots.tmp" "$d/add_slots"
      log "add_slots $(echo "$r" | sed 's/official_seed0_//; s/_xpl.*//') '$spec' (slots now ${slots[$r]})"
    done
    unset slots queued idle alivev pend add
  fi
  [[ -n ${ONCE:-} ]] && exit 0
  sleep "$INTERVAL"
done
