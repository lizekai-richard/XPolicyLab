#!/usr/bin/env bash
# Read-only: how the sana-wam torch enumerates CUDA devices vs nvidia-smi, default order vs PCI_BUS_ID, two runs.
set -u
HOSTS=${HOSTS:-"l2 l4 l6"}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "jiamingt@$h.mit.edu" 'out=~/robodojo_install/logs/$(hostname -s).cudaenum.txt; {
    echo "== $(hostname -s) $(date -u +%FT%TZ)"; echo "-- nvidia-smi index,uuid,pci,state"
    nvidia-smi --query-gpu=index,uuid,pci.bus_id,pstate --format=csv,noheader
    PY=/dataset2/jiamingt/robodojo/envs/sana-wam/bin/python
    for order in default PCI_BUS_ID; do for run in 1 2; do
      echo "-- torch enumeration order=$order run=$run"
      if [ "$order" = default ]; then unset CUDA_DEVICE_ORDER; else export CUDA_DEVICE_ORDER=PCI_BUS_ID; fi
      timeout 120 "$PY" - <<PYEOF 2>&1 | tail -n 12
import torch
n = torch.cuda.device_count(); print("device_count", n)
for i in range(n):
    try:
        p = torch.cuda.get_device_properties(i); print(i, "uuid", str(p.uuid), "mem", p.total_memory // 2**20)
    except Exception as e:
        print(i, "ERROR", type(e).__name__, str(e)[:100])
PYEOF
    done; done
  } > "$out" 2>&1; echo "written on $(hostname -s)"' || echo "FAILED on $h"
done
