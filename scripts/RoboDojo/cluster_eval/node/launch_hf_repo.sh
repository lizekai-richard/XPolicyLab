#!/usr/bin/env bash
# Run on x5: REPO=logits/<repo> [LABEL=<name>] [HOST=l5] bash launch_hf_repo.sh
# Starts download_hf_repo.sh on one MIT node in the background; logs:
#   ~/robodojo_install/logs/hfrepo_<label>.nohup.log  and  logs/download_hf_repo.log
# Terminal line to wait for: "HF REPO READY".
set -u
: "${REPO:?REPO=logits/<repo> required}"; LABEL=${LABEL:-$(basename "$REPO")}; HOST=${HOST:-l5}
ssh -n -o BatchMode=yes -o ConnectTimeout=15 "jiamingt@$HOST.mit.edu" \
  "mkdir -p ~/robodojo_install/logs; cd ~/robodojo_install || exit 1; nohup setsid env REPO='$REPO' LABEL='$LABEL' ALLOW='${ALLOW:-}' bash ./download_hf_repo.sh > 'logs/hfrepo_$LABEL.nohup.log' 2>&1 < /dev/null & sleep 2; echo \"launched on \$(hostname -s), pid \$!\"" \
  || echo "launch FAILED on $HOST"
