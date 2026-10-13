#!/usr/bin/env bash
# Read-only GPU health probe for the nodes (run on the control host): per GPU nvidia-smi state + a one-tensor CUDA init in
# the sana-wam env for each GPU index. Results: <CE_REMOTE_INSTALL>/logs/<host>.gpuprobe.txt (shared home).
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
HOSTS=${HOSTS:-$CE_NODES}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$(ce_host "$h")" 'mkdir -p '"$CE_REMOTE_INSTALL"'/logs; out='"$CE_REMOTE_INSTALL"'/logs/$(hostname -s).gpuprobe.txt; {
    echo "== $(hostname -s) $(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,compute_mode,pstate --format=csv
    echo "-- compute apps"; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
    echo "-- uuid map"; nvidia-smi --query-gpu=index,uuid --format=csv,noheader
    PY='"$CE_NODE_ROOT"'/envs/sana-wam/bin/python; [ -x "$PY" ] || PY='"$CE_NODE_ROOT"'/envs/robodojo-sana-eval/bin/python
    echo "-- per-GPU CUDA init with $PY"
    for i in 0 1 2 3 4 5 6 7; do
      r=$(CUDA_VISIBLE_DEVICES=$i timeout 60 "$PY" -c "import torch; torch.zeros(1).cuda(); p=torch.cuda.get_device_properties(0); print(\"ok\", p.name, p.total_memory//2**20, \"MiB\")" 2>&1 | tail -n 1)
      echo "gpu$i: $r"
    done
  } > "$out" 2>&1; echo "probe written on $(hostname -s)"' || echo "probe FAILED on $h"
done
