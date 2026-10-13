#!/usr/bin/env bash
# Run on x5: [HOST=l5] bash probe_nfs_disk.sh
# Prints the free space of the shared NFS home and the size of every checkpoint snapshot under it.
set -u
HOST=${HOST:-l5}
ssh -n -o BatchMode=yes -o ConnectTimeout=15 "jiamingt@$HOST.mit.edu" '
  echo "== df of the shared home =="; df -h /homes/jiamingt | tail -2
  echo "== checkpoint snapshots =="; du -sh /homes/jiamingt/checkpoints/huggingface/logits/* 2>/dev/null || echo "   (none)"
  echo "== total =="; du -sh /homes/jiamingt/checkpoints 2>/dev/null
' 2>&1
