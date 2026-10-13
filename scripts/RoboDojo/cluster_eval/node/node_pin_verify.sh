#!/usr/bin/env bash
# Read-only: RoboDojo node checkout HEAD + tracked changes (submodules ignored) + submodule pointer, per host.
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
HOSTS=${HOSTS:-$CE_NODES}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$(ce_host "$h")" '
    C='"$CE_NODE_ROOT"'/RoboDojo-official-clone-20260823; cd "$C" || exit 1
    echo "$(hostname -s): HEAD $(git rev-parse --short HEAD) | tracked changes: $(git status --porcelain --untracked-files=no --ignore-submodules=all | wc -l) | submodule XPolicyLab: $(git ls-tree HEAD XPolicyLab | awk "{print substr(\$3,1,9)}") checked out $(git -C XPolicyLab rev-parse --short HEAD 2>/dev/null)"' 2>&1 || echo "FAILED on $h"
done
