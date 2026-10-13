#!/usr/bin/env bash
# Kill OUR leaked Isaac client processes (src/eval_client/main.py, user CE_SSH_USER) on nodes where no evaluation
# worker is running any more. Run on the control host. Refuses to touch a node that still has a run_unit*.sh worker alive.
# Report: <CE_REMOTE_INSTALL>/logs/<host>.cleanup.txt
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
HOSTS=${HOSTS:-$CE_NODES}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$(ce_host "$h")" 'out='"$CE_REMOTE_INSTALL"'/logs/$(hostname -s).cleanup.txt; {
    echo "== $(hostname -s) $(date -u +%FT%TZ)"
    me=$(id -un)
    if pgrep -u "$me" -f "run_unit(_xpl)?\.sh " >/dev/null; then echo "REFUSED: a worker is still running here"; exit 0; fi
    # exclude this very shell and its parent: their command line contains the patterns below
    own="^($$|$PPID)$"
    pids=$(pgrep -u "$me" -f "src/eval_client/main.py" | grep -v -E "$own" | tr "\n" " ")
    srv=$(pgrep -u "$me" -f "robodojo/server/main.py|client_server.ws.model_server|setup_policy_server.py" | grep -v -E "$own" | tr "\n" " ")
    echo "orphan clients: $pids"; echo "orphan servers: $srv"
    before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
    [ -n "$pids$srv" ] && kill -TERM $pids $srv 2>/dev/null; sleep 20
    left=$(pgrep -u "$me" -f "src/eval_client/main.py|robodojo/server/main.py|client_server.ws.model_server|setup_policy_server.py" | grep -v -E "$own" | tr "\n" " ")
    [ -n "$left" ] && { echo "still alive after TERM: $left -> KILL"; kill -KILL $left 2>/dev/null; sleep 5; }
    after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
    echo "GPU memory used total MiB: before=$before after=$after"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  } > "$out" 2>&1; echo "cleanup done on $(hostname -s)"' || echo "FAILED on $h"
done
