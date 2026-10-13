#!/usr/bin/env bash
# Run on x5: CKPT_REPO=logits/<repo> REPO_SUBDIR=normalization DEST=/homes/jiamingt/... LABEL=<tag> [HOST=l5] bash launch_fetch_repo_subdir.sh
# Starts fetch_repo_subdir.sh on one MIT node in the background; log: ~/robodojo_install/logs/fetch_<LABEL>.nohup.log ("FETCH DONE").
set -u
: "${CKPT_REPO:?}"; : "${REPO_SUBDIR:?}"; : "${DEST:?}"; : "${LABEL:?}"; HOST=${HOST:-l5}
ssh -n -o BatchMode=yes -o ConnectTimeout=15 "jiamingt@$HOST.mit.edu" \
  "mkdir -p ~/robodojo_install/logs; cd ~/robodojo_install || exit 1; nohup setsid env CKPT_REPO='$CKPT_REPO' REPO_SUBDIR='$REPO_SUBDIR' DEST='$DEST' bash ./fetch_repo_subdir.sh > 'logs/fetch_$LABEL.nohup.log' 2>&1 < /dev/null & sleep 2; echo \"launched on \$(hostname -s), pid \$!\"" \
  || echo "launch FAILED on $HOST"
