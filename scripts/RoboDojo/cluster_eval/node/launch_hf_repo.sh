#!/usr/bin/env bash
# Run on the control host: REPO=logits/<repo> [LABEL=<name>] [HOST=<node>, default CE_GATEWAY] bash launch_hf_repo.sh
# Starts download_hf_repo.sh on one node in the background; logs:
#   <CE_REMOTE_INSTALL>/logs/hfrepo_<label>.nohup.log  and  logs/download_hf_repo.log
# Terminal line to wait for: "HF REPO READY".
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
: "${REPO:?REPO=logits/<repo> required}"; LABEL=${LABEL:-$(basename "$REPO")}; HOST=${HOST:-$CE_GATEWAY}
ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$(ce_host "$HOST")" \
  "mkdir -p $CE_REMOTE_INSTALL/logs; cd $CE_REMOTE_INSTALL || exit 1; nohup setsid env REPO='$REPO' LABEL='$LABEL' ALLOW='${ALLOW:-}' bash ./download_hf_repo.sh > 'logs/hfrepo_$LABEL.nohup.log' 2>&1 < /dev/null & sleep 2; echo \"launched on \$(hostname -s), pid \$!\"" \
  || echo "launch FAILED on $HOST"
