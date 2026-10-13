#!/usr/bin/env bash
# Run on the control host: CKPT_REPO=logits/<repo> REPO_SUBDIR=normalization DEST=<CE_SHARED_HOME>/... LABEL=<tag> [HOST=<node>, default CE_GATEWAY] bash launch_fetch_repo_subdir.sh
# Starts fetch_repo_subdir.sh on one node in the background; log: <CE_REMOTE_INSTALL>/logs/fetch_<LABEL>.nohup.log ("FETCH DONE").
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
: "${CKPT_REPO:?}"; : "${REPO_SUBDIR:?}"; : "${DEST:?}"; : "${LABEL:?}"; HOST=${HOST:-$CE_GATEWAY}
ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$(ce_host "$HOST")" \
  "mkdir -p $CE_REMOTE_INSTALL/logs; cd $CE_REMOTE_INSTALL || exit 1; nohup setsid env CKPT_REPO='$CKPT_REPO' REPO_SUBDIR='$REPO_SUBDIR' DEST='$DEST' bash ./fetch_repo_subdir.sh > 'logs/fetch_$LABEL.nohup.log' 2>&1 < /dev/null & sleep 2; echo \"launched on \$(hostname -s), pid \$!\"" \
  || echo "launch FAILED on $HOST"
