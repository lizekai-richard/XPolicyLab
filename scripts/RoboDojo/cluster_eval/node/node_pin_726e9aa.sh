#!/usr/bin/env bash
# Fast-forward the node-local RoboDojo checkout (the one that simulates) to EXACTLY 726e9aa, as done on l3/l5/l7/l8/l25/l26
# on 2026-09-23. Refuses a dirty tree or a HEAD that is not an ancestor of 726e9aa; superproject only (no submodule
# update, same as the six nodes). HOSTS="l2 l4 l6" bash node_pin_726e9aa.sh
set -u
HOSTS=${HOSTS:-"l2 l4 l6"}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "jiamingt@$h.mit.edu" '
    set -u; C=/dataset2/jiamingt/robodojo/RoboDojo-official-clone-20260823; PIN=726e9aa
    cd "$C" || { echo "$(hostname -s): no checkout at $C"; exit 1; }
    before=$(git rev-parse --short HEAD)
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "$(hostname -s): REFUSE, tracked changes:"; git status --porcelain --untracked-files=no | head; exit 1; fi
    if ! git cat-file -e "$PIN^{commit}" 2>/dev/null; then
      git -c http.version=HTTP/1.1 fetch --quiet origin || { echo "$(hostname -s): fetch FAILED"; exit 1; }
    fi
    git merge-base --is-ancestor HEAD "$PIN" || { echo "$(hostname -s): REFUSE, HEAD $before is not an ancestor of $PIN"; exit 1; }
    git merge --ff-only --quiet "$PIN" || { echo "$(hostname -s): ff-merge FAILED"; exit 1; }
    echo "$(hostname -s): $before -> $(git rev-parse --short HEAD) ($(git log -1 --format=%s | cut -c1-60))"' 2>&1 || echo "FAILED on $h"
done
