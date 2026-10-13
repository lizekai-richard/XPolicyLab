#!/usr/bin/env bash
# Run on x5: start install_openwam.sh in the background on each MIT node (node-local envs).
#   [HOSTS="l5 l7 ..."] bash launch_openwam_all.sh
# Progress: ~/robodojo_install/logs/<host>.openwam.log  (shared home; also <host>.openwam.status)
set -u
HOSTS=${HOSTS:-"l5 l7 l8 l25 l3 l26"}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "jiamingt@$h.mit.edu" \
    'mkdir -p ~/robodojo_install/logs; cd ~/robodojo_install || exit 1; nohup setsid bash ./install_openwam.sh > logs/$(hostname -s).openwam.nohup.log 2>&1 < /dev/null & sleep 1; echo "launched openwam install on $(hostname -s), pid $!"' \
  || echo "launch FAILED on $h"
done
