#!/usr/bin/env bash
# Run on x5. Starts download_weights.sh for one HF checkpoint repo in the background on ONE MIT
# node (default l5); the script and its log live in the shared NFS home (~/robodojo_install).
#   CKPT_REPO=logits/<repo> [HOST=l5] bash launch_download.sh
# Progress: ~/robodojo_install/logs/download_weights.log (appended by the script itself) and
# logs/download_<repo>.nohup.log; the terminal line is "ALL WEIGHTS DOWNLOADED AND VERIFIED".
set -u
: "${CKPT_REPO:?CKPT_REPO=logits/<repo> required}"
HOST=${HOST:-l5}
name=$(basename "$CKPT_REPO")
ssh -n -o BatchMode=yes -o ConnectTimeout=15 "jiamingt@$HOST.mit.edu" \
  "mkdir -p ~/robodojo_install/logs; cd ~/robodojo_install || exit 1; nohup setsid env CKPT_REPO='$CKPT_REPO' bash ./download_weights.sh > 'logs/download_$name.nohup.log' 2>&1 < /dev/null & sleep 2; echo \"launched on \$(hostname -s), pid \$!\"; pgrep -af download_weights.sh | grep -v pgrep | head -n 3" \
  || echo "launch FAILED on $HOST"
