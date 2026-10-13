#!/usr/bin/env bash
# Read-only: process forest of our eval workers on each node -> <CE_REMOTE_INSTALL>/logs/<host>.psprobe.txt
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
HOSTS=${HOSTS:-$CE_NODES}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$(ce_host "$h")" 'out='"$CE_REMOTE_INSTALL"'/logs/$(hostname -s).psprobe.txt; {
    echo "== $(hostname -s) $(date -u +%FT%TZ)"
    ps -u "$(id -un)" -o pid,ppid,pgid,sess,stat,etimes,wchan:20,cmd --forest | grep -v -E "sshd|ps -u|bash -c out=" | cut -c1-220
  } > "$out" 2>&1; echo "ps probe written on $(hostname -s)"' || echo "FAILED on $h"
done
