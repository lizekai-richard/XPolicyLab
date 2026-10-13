#!/usr/bin/env bash
# Run this on x5 (as yuyang). Starts install_node.sh in the background on each
# node; the installer itself lives in the shared NFS home (~/robodojo_install).
# Logs: node-local /dataset2/jiamingt/robodojo/logs/ and a mirror in
# ~/robodojo_install/logs/<host>.log + <host>.status (readable from any node).
#
# Note: only the nohup command is backgrounded (commands are separated by ';',
# not '&&'). An earlier version backgrounded the whole && chain, which left a
# subshell holding the ssh channel open until the installer exited, so ssh
# hung and the loop never reached the later hosts.
set -u
HOSTS=${HOSTS:-"l5 l7 l8 l25"}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "jiamingt@$h.mit.edu" \
    'mkdir -p ~/robodojo_install/logs; cd ~/robodojo_install || exit 1; nohup setsid bash ./install_node.sh > logs/$(hostname -s).nohup.log 2>&1 < /dev/null & sleep 1; echo "launched on $(hostname -s), pid $!"' \
  || echo "launch FAILED on $h"
done
echo "status: bash $(dirname "$0")/status_all.sh"
