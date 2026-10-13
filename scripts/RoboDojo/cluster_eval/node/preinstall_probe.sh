#!/usr/bin/env bash
# Read-only pre-install check: /dataset2/jiamingt state + write access, free space, conda, driver, net to github/pypi.
set -u
HOSTS=${HOSTS:-"l1 l9 l27 l30"}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "jiamingt@$h.mit.edu" '
    d=/dataset2/jiamingt; w=no
    if [ -d $d ]; then [ -w $d ] && w=yes; else [ -w /dataset2 ] && w="yes(parent)"; fi
    echo "$(hostname -s): /dataset2/jiamingt exists=$([ -d $d ] && echo yes || echo no) writable=$w contents=[$(ls $d 2>/dev/null | head -5 | tr "\n" " ")] free=$(df -h /dataset2 | awk "NR==2{print \$4}") conda=$([ -x ~/miniconda3/bin/conda ] && echo ok || echo MISSING) driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1) github=$(curl -s -o /dev/null -m 10 -w %{http_code} https://github.com) pypi=$(curl -s -o /dev/null -m 10 -w %{http_code} https://pypi.org/simple/pip/) cpus=$(nproc) mem=$(free -g | awk "/^Mem:/{print \$2}")G"' 2>&1 || echo "FAILED on $h"
done
