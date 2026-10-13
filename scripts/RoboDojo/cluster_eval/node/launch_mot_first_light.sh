#!/usr/bin/env bash
# Run on the control host: CKPT_LABEL=<snapshot> [HOST=<node>, default CE_GATEWAY] [GPU=0] [EXTRA="--skip-forward"] bash launch_mot_first_light.sh
# Runs mot_first_light.py on one node in the sana-wam env against the NFS checkpoint snapshot and the
# NFS XPolicyLab copy (CE_REMOTE_XPL). Foreground: prints the probe's output here.
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
: "${CKPT_LABEL:?CKPT_LABEL=<snapshot dir name> required}"; HOST=${HOST:-$CE_GATEWAY}; GPU=${GPU:-0}; EXTRA=${EXTRA:-}
ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$(ce_host "$HOST")" "
  set -u
  source $CE_NODE_ROOT/activate_sana_wam.sh
  cd $CE_REMOTE_INSTALL || exit 1
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning \
    \"\$SANA_PY\" mot_first_light.py \
      --ckpt $CE_CKPT_ROOT/$CKPT_LABEL \
      --xpl-root $CE_REMOTE_XPL $EXTRA
" 2>&1
