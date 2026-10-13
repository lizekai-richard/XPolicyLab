#!/usr/bin/env bash
# Refresh this snapshot from the live deployment (default ~/zekail/robodojo_install). Private Sana artefacts (bundles,
# patches), old kits, backups and caches are never copied. Review `git status` and scan for secrets before committing.
#   bash sync_from_install.sh [SRC]
set -euo pipefail
SRC=${1:-$HOME/zekail/robodojo_install}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXCL=(--exclude '__pycache__' --exclude '*.pyc' --exclude '*.bak*' --exclude '*.bundle' --exclude '*.patch' --exclude '*.tar.sha256' --exclude 'logs/')
rsync -a --delete "${EXCL[@]}" "$SRC/eval/" "$HERE/eval/"
rsync -a --delete "${EXCL[@]}" --exclude 'eval/' --exclude 'patches/' --exclude 'sana_patches/' --exclude 'xpolicylab_sana_wam/' "$SRC/" "$HERE/node/"
grep -rIn -E "hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{20,}|github_pat_|BEGIN (RSA|OPENSSH|EC) PRIVATE" "$HERE" && { echo "possible secret above -- do not commit"; exit 1; }
echo "synced from $SRC; now: git -C $HERE status"
