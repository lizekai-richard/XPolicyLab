#!/usr/bin/env bash
# Run on the control host: CKPT_REPO=logits/<repo> [CKPT_SUBDIR=work_dir/checkpoints/<epoch_step> | auto] CKPT_LABEL=<name> [HOST=<node>, default CE_GATEWAY] bash launch_partial_ckpt.sh
# Starts download_partial_ckpt.sh on one node in the background; log: <CE_REMOTE_INSTALL>/logs/partial_<label>.nohup.log
# and download_partial_ckpt.log; terminal line "PARTIAL CHECKPOINT READY".
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
: "${CKPT_REPO:?}"; CKPT_SUBDIR=${CKPT_SUBDIR:-auto}; : "${CKPT_LABEL:?}"; HOST=${HOST:-$CE_GATEWAY}
ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$(ce_host "$HOST")" \
  "mkdir -p $CE_REMOTE_INSTALL/logs; cd $CE_REMOTE_INSTALL || exit 1; nohup setsid env CKPT_REPO='$CKPT_REPO' CKPT_SUBDIR='$CKPT_SUBDIR' CKPT_LABEL='$CKPT_LABEL' CONFIG_SRC='${CONFIG_SRC:-}' bash ./download_partial_ckpt.sh > 'logs/partial_$CKPT_LABEL.nohup.log' 2>&1 < /dev/null & sleep 2; echo \"launched on \$(hostname -s), pid \$!\"" \
  || echo "launch FAILED on $HOST"
