#!/usr/bin/env bash
# Read-only pre-install check: CE_NODE_ROOT's parent dir state + write access, free space, conda, driver, net to github/pypi.
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
NODE_PARENT=$(dirname "$CE_NODE_ROOT"); NODE_DISK=$(dirname "$NODE_PARENT")   # CE_NODE_ROOT's parent (the user's dir) and its disk
HOSTS=${HOSTS:-$CE_NODES}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$(ce_host "$h")" '
    d='"$NODE_PARENT"'; w=no
    if [ -d $d ]; then [ -w $d ] && w=yes; else [ -w '"$NODE_DISK"' ] && w="yes(parent)"; fi
    echo "$(hostname -s): '"$NODE_PARENT"' exists=$([ -d $d ] && echo yes || echo no) writable=$w contents=[$(ls $d 2>/dev/null | head -5 | tr "\n" " ")] free=$(df -h '"$NODE_DISK"' | awk "NR==2{print \$4}") conda=$([ -x ~/miniconda3/bin/conda ] && echo ok || echo MISSING) driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1) github=$(curl -s -o /dev/null -m 10 -w %{http_code} https://github.com) pypi=$(curl -s -o /dev/null -m 10 -w %{http_code} https://pypi.org/simple/pip/) cpus=$(nproc) mem=$(free -g | awk "/^Mem:/{print \$2}")G"' 2>&1 || echo "FAILED on $h"
done
