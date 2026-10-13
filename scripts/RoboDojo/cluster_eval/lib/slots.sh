# shellcheck shell=bash
# GPU slot helpers shared by eval/run.sh and eval/campaign.sh (source lib/site.sh first). A slot spec is
# 'node:s0,s3;node2:s1' (single-GPU slots: CUDA device g runs the policy server AND the Isaac client) -- the form every
# run since 2026-09-24 uses; a bare k ('node:1') is the historical two-GPU pair (devices 2k, 2k+1).
# CUDA numbers only HEALTHY boards (a board in '[GPU requires reset]' is skipped), so CUDA device g is the g-th healthy
# board in nvidia-smi order: every probe below maps g through that list before reading a board's memory.

ce_master_alive(){ # RUN -> 0 when the master of RUN is alive
  local m=$CE_MASTER_ROOT/$1; [[ -s $m/master.pid ]] && kill -0 "$(cat "$m/master.pid")" 2>/dev/null
}

ce_live_slots(){ # node:gpu of every GPU a live master's slot names, plus every GPU a live master's tail plan dropped for its shards
  "$CE_PY" - "$CE_MASTER_ROOT" <<'PYEOF'
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

ce_busy_gpus(){ # node:gpu of every GPU a live master holds: a slot with a unit, draining, or while its master has dispatchable work
  "$CE_PY" - "$CE_MASTER_ROOT" <<'PYEOF'
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

ce_spec_gpus(){ # SPEC -> lines 'node gpu'
  "$CE_PY" - "$1" <<'PYEOF'
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

ce_nodes_of(){ echo "$1" | tr ';' '\n' | cut -d: -f1 | sed '/^$/d' | paste -sd, -; }   # SPEC -> 'n1,n2' (spec order)
ce_spec_count(){ echo "$1" | tr ';' '\n' | cut -d: -f2 | tr ',' '\n' | grep -c . || true; }

ce_deal(){ # SPEC K CAP -> K specs (one per line), the slots dealt round-robin (node order kept), at most CAP slots each
  "$CE_PY" - "$1" "$2" "$3" <<'PYEOF'
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

ce_free_spec(){ # RPC_BASE WS_BASE [NODES] [RESERVE] -> every healthy CUDA device of NODES (default CE_NODES) with < 1 GiB in use,
  # both of its slot ports (base + 50 + g) closed, that no live master holds and RESERVE ('node:g ...', default CE_RESERVE)
  # does not name, as a single-GPU spec in node order. An unreachable node contributes nothing.
  local rpc=$1 ws=$2 nodes=${3:-$CE_NODES} reserve=${4-$CE_RESERVE} held node gl g out="" keep
  held=$(ce_live_slots)
  if [[ -n ${CE_TEST_FREE_SPEC:-} ]]; then   # OFFLINE TEST HOOK (never set it for a real run): this spec minus held / reserved GPUs, no ssh
    "$CE_PY" - "$CE_TEST_FREE_SPEC" "$held" "$reserve" <<'PYEOF'
import sys
held, reserve = set(sys.argv[2].split()), set(sys.argv[3].split()); out = []
for item in filter(None, sys.argv[1].split(";")):
    node, gs = item.split(":"); keep = [g for g in gs.split(",") if f"{node}:{g.lstrip('s')}" not in held | reserve]
    if keep: out.append(f"{node}:{','.join(keep)}")
print(";".join(out))
PYEOF
    return
  fi
  for node in $nodes; do
    gl=$(timeout 60 ssh -n $CE_SSH_OPTS "$(ce_host "$node")" "
      healthy=(\$(nvidia-smi --query-gpu=index,pstate --format=csv,noheader | awk -F', ' '\$2 !~ /reset/ {print \$1}'))
      for g in \"\${!healthy[@]}\"; do
        used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i \${healthy[\$g]} | tr -d ' ')
        [ \"\$used\" -lt 1024 ] || continue
        [ -z \"\$(ss -ltnH \"sport = :\$(($rpc + 50 + g))\")\" ] && [ -z \"\$(ss -ltnH \"sport = :\$(($ws + 50 + g))\")\" ] || continue
        printf '%s ' \"\$g\"
      done" 2>/dev/null) || continue
    keep=()
    for g in $gl; do grep -qx "$node:$g" <<<"$held" && continue; [[ " $reserve " == *" $node:$g "* ]] && continue; keep+=("s$g"); done
    [[ ${#keep[@]} -gt 0 ]] && out+="${out:+;}$node:$(IFS=,; echo "${keep[*]}")"
  done
  echo "$out"
}

ce_check_gpus_free(){ # SPEC: refuse GPUs a live master holds or RESERVE names, then check on each node that they are healthy and empty
  local spec=$1 busy node g bad="" gl
  busy=$(ce_busy_gpus)
  while read -r node g; do
    grep -qx "$node:$g" <<<"$busy" && bad+=" $node:$g(live master)"
    [[ " $CE_RESERVE " == *" $node:$g "* ]] && bad+=" $node:$g(CE_RESERVE)"
  done < <(ce_spec_gpus "$spec")
  [[ -z $bad ]] || { echo "REFUSE: GPUs held by a live master (occupied, draining, or reachable by its queue) or reserved:$bad"; return 3; }
  for node in $(ce_spec_gpus "$spec" | awk '{print $1}' | sort -u); do
    gl=$(ce_spec_gpus "$spec" | awk -v n="$node" '$1 == n {printf "%s ", $2}')
    ssh -n $CE_SSH_OPTS "$(ce_host "$node")" "
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
}
