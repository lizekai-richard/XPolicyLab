#!/usr/bin/env bash
# FORCE variant of cleanup_orphans_all.sh for nodes being withdrawn from a run: kills OUR eval workers too
# (run_unit*.sh, heartbeat/stall subshells, policy servers, Isaac clients) for user CE_SSH_USER on the given hosts.
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
HOSTS=${HOSTS:?set HOSTS explicitly, e.g. HOSTS="l2 l4 l6"}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$(ce_host "$h")" 'out='"$CE_REMOTE_INSTALL"'/logs/$(hostname -s).killours.txt; {
    echo "== $(hostname -s) $(date -u +%FT%TZ)"; me=$(id -un); own="^($$|$PPID)$"
    pat="run_unit(_xpl)?\.sh |src/eval_client/main.py|robodojo/server/main.py|client_server.ws.model_server|setup_policy_server.py|run_remote_(sim|benchmark)_client.sh|eval_policy.sh|robodojo.sh client"
    pids=$(pgrep -u "$me" -f "$pat" | grep -v -E "$own" | tr "\n" " "); echo "killing: $pids"
    [ -n "$pids" ] && kill -TERM $pids 2>/dev/null; sleep 15
    left=$(pgrep -u "$me" -f "$pat" | grep -v -E "$own" | tr "\n" " "); [ -n "$left" ] && { echo "KILL: $left"; kill -KILL $left 2>/dev/null; sleep 3; }
    echo "remaining: $(pgrep -u "$me" -f "$pat" | grep -v -E "$own" | tr "\n" " ")"
    nvidia-smi --query-gpu=index,memory.used,pstate --format=csv,noheader | tr "\n" ";"
  } > "$out" 2>&1; echo "done on $(hostname -s)"' || echo "FAILED on $h"
done
