#!/usr/bin/env bash
# Run on x5 (as yuyang): start install_sana_wam.sh in the background on each
# node. Only the nohup command is backgrounded (see launch_all.sh note).
set -u
HOSTS=${HOSTS:-"l5 l7 l8 l25"}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "jiamingt@$h.mit.edu" \
    'mkdir -p ~/robodojo_install/logs; cd ~/robodojo_install || exit 1; nohup setsid bash ./install_sana_wam.sh > logs/$(hostname -s).sana_wam.nohup.log 2>&1 < /dev/null & sleep 1; echo "launched sana-wam install on $(hostname -s), pid $!"' \
  || echo "launch FAILED on $h"
done
