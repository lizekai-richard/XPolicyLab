#!/usr/bin/env bash
# Fast-forward the node-local RoboDojo checkout (the one that simulates) to EXACTLY 09bd5f5 (upstream RoboDojo of 2026-10-12:
# 726e9aa + PR #48, the SpringButton reset fix 487e425; without it swap_blocks cannot be solved, see RUNBOOK.md "Pin history").
# Refuses a dirty tree or a HEAD that is not an ancestor of 09bd5f5 (726e9aa and the 2184bf8 install pin both are); superproject
# only (no submodule update, same as node_pin_726e9aa.sh). [HOSTS="<node> ...", default CE_NODES] bash node_pin_09bd5f5.sh
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
HOSTS=${HOSTS:-$CE_NODES}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$(ce_host "$h")" '
    set -u; C='"$CE_NODE_ROOT"'/RoboDojo-official-clone-20260823; PIN=09bd5f5
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
