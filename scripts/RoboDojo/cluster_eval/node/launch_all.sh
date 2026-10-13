#!/usr/bin/env bash
# Run this on the control host. Starts install_node.sh in the background on each
# node; the installer itself lives in the shared NFS home (CE_REMOTE_INSTALL).
# Logs: node-local <CE_NODE_ROOT>/logs/ and a mirror in
# <CE_REMOTE_INSTALL>/logs/<host>.log + <host>.status (readable from any node).
#
# Note: only the nohup command is backgrounded (commands are separated by ';',
# not '&&'). An earlier version backgrounded the whole && chain, which left a
# subshell holding the ssh channel open until the installer exited, so ssh
# hung and the loop never reached the later hosts.
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
HOSTS=${HOSTS:-$CE_NODES}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$(ce_host "$h")" \
    'mkdir -p '"$CE_REMOTE_INSTALL"'/logs; cd '"$CE_REMOTE_INSTALL"' || exit 1; nohup setsid bash ./install_node.sh > logs/$(hostname -s).nohup.log 2>&1 < /dev/null & sleep 1; echo "launched on $(hostname -s), pid $!"' \
  || echo "launch FAILED on $h"
done
echo "status: bash $(dirname "$0")/status_all.sh"
