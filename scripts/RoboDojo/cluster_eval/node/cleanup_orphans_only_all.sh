#!/usr/bin/env bash
# Kill ONLY orphaned Isaac clients (src/eval_client/main.py of user CE_SSH_USER whose parent is init, i.e. PPID 1)
# on each host, even while evaluation workers are running there: a live client's parent is the worker's
# `timeout`/`bash`, never init. Run on the control host. Report: <CE_REMOTE_INSTALL>/logs/<host>.cleanup_orphans_only.txt
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
HOSTS=${HOSTS:-$CE_NODES}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$(ce_host "$h")" 'me=$(id -un); out='"$CE_REMOTE_INSTALL"'/logs/$(hostname -s).cleanup_orphans_only.txt; {
    echo "== $(hostname -s) $(date -u +%FT%TZ)"
    own="^($$|$PPID)$"
    pids=""
    for p in $(pgrep -u "$me" -f "src/eval_client/main.py" | grep -v -E "$own"); do
      pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d " "); [ "$pp" = "1" ] && pids="$pids $p"
    done
    if [ -z "${pids// /}" ]; then echo "no orphaned Isaac clients"; else
      echo "orphans (PPID 1):$pids"; ps -o pid=,etimes=,args= -p ${pids// /,} 2>/dev/null | cut -c1-150
      kill -TERM $pids 2>/dev/null; sleep 20
      left=""; for p in $pids; do kill -0 "$p" 2>/dev/null && left="$left $p"; done
      [ -n "${left// /}" ] && { echo "still alive after TERM:$left -> KILL"; kill -KILL $left 2>/dev/null; sleep 5; }
    fi
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  } >"$out" 2>&1; echo "orphans-only cleanup done on $(hostname -s)"' || echo "FAILED on $h"
done
