#!/usr/bin/env bash
# Run on the control host: [HOST=<node>, default CE_GATEWAY] [DUMP=<node dir>] bash launch_ckpt_key_probe.sh <source> [<source> ...]
# A source is hf://<owner>/<repo>/<file> (only the zip directory + data.pkl are range-read, no weight download)
# or a node-visible local path. Runs probe_ckpt_keys.py in the node's sana-wam env (which provides HF_TOKEN_PATH);
# foreground, prints the probe's output here.
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
HOST=${HOST:-$CE_GATEWAY}
[[ $# -ge 1 ]] || { echo "usage: bash launch_ckpt_key_probe.sh <source> ..."; exit 2; }
args=$(printf " '%s'" "$@")
ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$(ce_host "$HOST")" "
  set -u
  source $CE_NODE_ROOT/activate_sana_wam.sh
  cd $CE_REMOTE_INSTALL || exit 1
  export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1
  PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \"\$SANA_PY\" probe_ckpt_keys.py ${DUMP:+--dump $DUMP} $args
" 2>&1
