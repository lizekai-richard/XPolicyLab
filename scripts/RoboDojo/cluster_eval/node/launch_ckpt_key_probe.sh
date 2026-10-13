#!/usr/bin/env bash
# Run on x5: [HOST=l5] [DUMP=<node dir>] bash launch_ckpt_key_probe.sh <source> [<source> ...]
# A source is hf://<owner>/<repo>/<file> (only the zip directory + data.pkl are range-read, no weight download)
# or a node-visible local path. Runs probe_ckpt_keys.py in the node's sana-wam env (which provides HF_TOKEN_PATH);
# foreground, prints the probe's output here.
set -u
HOST=${HOST:-l5}
[[ $# -ge 1 ]] || { echo "usage: bash launch_ckpt_key_probe.sh <source> ..."; exit 2; }
args=$(printf " '%s'" "$@")
ssh -n -o BatchMode=yes -o ConnectTimeout=20 "jiamingt@$HOST.mit.edu" "
  set -u
  source /dataset2/jiamingt/robodojo/activate_sana_wam.sh
  cd ~/robodojo_install || exit 1
  export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1
  PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \"\$SANA_PY\" probe_ckpt_keys.py ${DUMP:+--dump $DUMP} $args
" 2>&1
