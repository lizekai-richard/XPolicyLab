#!/usr/bin/env bash
# Run on the control host: start install_openwam.sh in the background on each node (node-local envs).
#   [HOSTS="<node> <node> ...", default CE_NODES] bash launch_openwam_all.sh
# Progress: <CE_REMOTE_INSTALL>/logs/<host>.openwam.log  (shared home; also <host>.openwam.status)
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
HOSTS=${HOSTS:-$CE_NODES}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$(ce_host "$h")" \
    'mkdir -p '"$CE_REMOTE_INSTALL"'/logs; cd '"$CE_REMOTE_INSTALL"' || exit 1; nohup setsid bash ./install_openwam.sh > logs/$(hostname -s).openwam.nohup.log 2>&1 < /dev/null & sleep 1; echo "launched openwam install on $(hostname -s), pid $!"' \
  || echo "launch FAILED on $h"
done
