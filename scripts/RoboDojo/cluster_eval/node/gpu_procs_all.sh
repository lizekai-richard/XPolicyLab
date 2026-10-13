#!/usr/bin/env bash
# Read-only: who holds GPU memory on each node (pid -> user, elapsed, command). Results: <CE_REMOTE_INSTALL>/logs/<host>.gpuprocs.txt
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
HOSTS=${HOSTS:-$CE_NODES}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$(ce_host "$h")" 'out='"$CE_REMOTE_INSTALL"'/logs/$(hostname -s).gpuprocs.txt; {
    echo "== $(hostname -s) $(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    echo "-- processes on GPUs"
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
      printf "%s | " "$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader | grep "^$p," | head -1)"
      ps -o user=,etimes=,ppid=,pgid=,cmd= -p "$p" 2>/dev/null | cut -c1-230 || echo "pid $p gone"
    done
    echo "-- our eval processes (any)"; pgrep -u "$(id -un)" -af "eval_client/main.py|robodojo/server/main.py|client_server.ws|run_unit|robodojo.sh|eval_policy.sh" | cut -c1-200
  } > "$out" 2>&1; echo "written on $(hostname -s)"' || echo "FAILED on $h"
done
