#!/usr/bin/env bash
# Run on the control host: start install_sana_wam.sh in the background on each
# node. Only the nohup command is backgrounded (see launch_all.sh note).
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
HOSTS=${HOSTS:-$CE_NODES}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$(ce_host "$h")" \
    'mkdir -p '"$CE_REMOTE_INSTALL"'/logs; cd '"$CE_REMOTE_INSTALL"' || exit 1; nohup setsid bash ./install_sana_wam.sh > logs/$(hostname -s).sana_wam.nohup.log 2>&1 < /dev/null & sleep 1; echo "launched sana-wam install on $(hostname -s), pid $!"' \
  || echo "launch FAILED on $h"
done
