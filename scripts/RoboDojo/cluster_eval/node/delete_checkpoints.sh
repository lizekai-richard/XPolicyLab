#!/usr/bin/env bash
# Run on the control host: [HOST=<node>, default CE_GATEWAY] CONFIRM=yes bash delete_checkpoints.sh <label> [<label> ...]
# Permanently removes checkpoint directories under CE_CKPT_ROOT (site.env) on the shared home.
# Labels are the directory names only -- a '/' or '..' in a label is refused, and the parent dir is fixed (CE_CKPT_ROOT), so
# this can never touch anything outside the logits tree. Without CONFIRM=yes it only prints what it WOULD delete.
# Metadata (config.yaml, normalization, json_cache) of every evaluated checkpoint is archived at
# <CE_RESULTS_ROOT>/checkpoints_meta/ with a restore recipe in its RESTORE.md -- back one up BEFORE deleting.
set -uo pipefail
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
ROOT=$CE_CKPT_ROOT
HOST=${HOST:-$CE_GATEWAY}
[[ $# -gt 0 ]] || { sed -n 2,7p "$0"; exit 1; }
for l in "$@"; do
  case "$l" in */*|*..*|""|.|-*) echo "REFUSED unsafe label: '$l'"; exit 1;; esac
done
printf 'target host : %s%s\ntarget parent: %s\nlabels (%d): %s\n\n' "$HOST" "$CE_SSH_DOMAIN" "$ROOT" "$#" "$*"
size_of(){ rsync -n --list-only -r "$(ce_host "$HOST"):$ROOT/$1/" 2>/dev/null \
    | awk '$1 ~ /^-/ {gsub(",","",$2); s+=$2; n++} END {printf "%d files, %.2f GB", n, s/1073741824}'; }
total=0
for l in "$@"; do printf '  %-78s %s\n' "$l" "$(size_of "$l")"; done
if [[ ${CONFIRM:-no} != yes ]]; then echo; echo "DRY RUN -- re-run with CONFIRM=yes to delete."; exit 0; fi
rm_list=""; for l in "$@"; do rm_list+=" '$ROOT/$l'"; done
echo; echo "[$(date -u +%FT%TZ)] deleting ..."
ssh -n -o BatchMode=yes -o ConnectTimeout=30 "$(ce_host "$HOST")" \
    "set -u; for d in $rm_list; do if [ -d \"\$d\" ]; then rm -rf -- \"\$d\" && echo \"deleted \$d\"; else echo \"absent  \$d\"; fi; done; echo '--- remaining in $ROOT:'; ls -1 '$ROOT' 2>/dev/null || echo '(empty or gone)'; df -h '$ROOT' | tail -1" \
  || { echo "ssh/rm FAILED"; exit 1; }
echo "[$(date -u +%FT%TZ)] done"
