#!/usr/bin/env bash
# Run on the control host: [HOST=<node>, default CE_GATEWAY] bash probe_nfs_disk.sh
# Prints the free space of the shared NFS home and the size of every checkpoint snapshot under it.
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
HOST=${HOST:-$CE_GATEWAY}
ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$(ce_host "$HOST")" '
  echo "== df of the shared home =="; df -h '"$CE_SHARED_HOME"' | tail -2
  echo "== checkpoint snapshots =="; du -sh '"$CE_CKPT_ROOT"'/* 2>/dev/null || echo "   (none)"
  echo "== total =="; du -sh '"$CE_SHARED_HOME"'/checkpoints 2>/dev/null
' 2>&1
