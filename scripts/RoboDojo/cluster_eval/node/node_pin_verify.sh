#!/usr/bin/env bash
# Read-only: RoboDojo node checkout HEAD + tracked changes (submodules ignored) + submodule pointer, per host.
set -u
HOSTS=${HOSTS:-"l2 l4 l6 l5"}
for h in $HOSTS; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "jiamingt@$h.mit.edu" '
    C=/dataset2/jiamingt/robodojo/RoboDojo-official-clone-20260823; cd "$C" || exit 1
    echo "$(hostname -s): HEAD $(git rev-parse --short HEAD) | tracked changes: $(git status --porcelain --untracked-files=no --ignore-submodules=all | wc -l) | submodule XPolicyLab: $(git ls-tree HEAD XPolicyLab | awk "{print substr(\$3,1,9)}") checked out $(git -C XPolicyLab rev-parse --short HEAD 2>/dev/null)"' 2>&1 || echo "FAILED on $h"
done
