#!/usr/bin/env bash
# Read-only: process forest of our eval workers on each node -> ~/robodojo_install/logs/<host>.psprobe.txt
set -u
HOSTS=${HOSTS:-"l2 l4"}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "jiamingt@$h.mit.edu" 'out=~/robodojo_install/logs/$(hostname -s).psprobe.txt; {
    echo "== $(hostname -s) $(date -u +%FT%TZ)"
    ps -u "$(id -un)" -o pid,ppid,pgid,sess,stat,etimes,wchan:20,cmd --forest | grep -v -E "sshd|ps -u|bash -c out=" | cut -c1-220
  } > "$out" 2>&1; echo "ps probe written on $(hostname -s)"' || echo "FAILED on $h"
done
