#!/usr/bin/env bash
# Run on the control host. Starts download_weights.sh for one HF checkpoint repo in the background on ONE
# node (default CE_GATEWAY); the script and its log live in the shared NFS home (CE_REMOTE_INSTALL).
#   CKPT_REPO=logits/<repo> [HOST=<node>, default CE_GATEWAY] bash launch_download.sh
# Progress: <CE_REMOTE_INSTALL>/logs/download_weights.log (appended by the script itself) and
# logs/download_<repo>.nohup.log; the terminal line is "ALL WEIGHTS DOWNLOADED AND VERIFIED".
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
: "${CKPT_REPO:?CKPT_REPO=logits/<repo> required}"
HOST=${HOST:-$CE_GATEWAY}
name=$(basename "$CKPT_REPO")
ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$(ce_host "$HOST")" \
  "mkdir -p $CE_REMOTE_INSTALL/logs; cd $CE_REMOTE_INSTALL || exit 1; nohup setsid env CKPT_REPO='$CKPT_REPO' bash ./download_weights.sh > 'logs/download_$name.nohup.log' 2>&1 < /dev/null & sleep 2; echo \"launched on \$(hostname -s), pid \$!\"; pgrep -af download_weights.sh | grep -v pgrep | head -n 3" \
  || echo "launch FAILED on $HOST"
