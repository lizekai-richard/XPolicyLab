#!/usr/bin/env bash
# Read-only: which physical board CUDA_VISIBLE_DEVICES=g selects on a node with "[GPU requires reset]" boards
# (the sana-wam torch, one process per g; prints the selected device's UUID next to nvidia-smi's board list).
set -u
HOSTS=${HOSTS:-"l2 l4 l6"}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "jiamingt@$h.mit.edu" '
    echo "== $(hostname -s)"; nvidia-smi --query-gpu=index,uuid,pstate --format=csv,noheader | sed "s/^/   smi /"
    PY=/dataset2/jiamingt/robodojo/envs/sana-wam/bin/python
    for g in 0 1 2 3 4 5 6 7; do
      r=$(CUDA_VISIBLE_DEVICES=$g timeout 90 "$PY" -c "
import torch
try:
    p = torch.cuda.get_device_properties(0); x = torch.ones(4, device=\"cuda\").sum().item(); print(\"GPU-\" + str(p.uuid), \"ok\", x)
except Exception as e:
    print(\"ERROR\", type(e).__name__, str(e)[:60])
" 2>&1 | tail -1)
      echo "   CUDA_VISIBLE_DEVICES=$g -> $r"
    done' 2>&1 || echo "FAILED on $h"
done
