#!/usr/bin/env bash
# Read-only status of all nodes via ONE ssh to VIA (default CE_GATEWAY): the installer mirrors its
# status stream and master log into the shared home, so that node sees every node.
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
VIA=${VIA:-$CE_GATEWAY}
ssh -o BatchMode=yes -o ConnectTimeout=10 "$(ce_host "$VIA")" '
cd '"$CE_REMOTE_INSTALL"'/logs 2>/dev/null || { echo "no logs yet"; exit 0; }
for s in *.status; do
  [ -e "$s" ] || continue
  h=${s%.status}
  echo "=== $h ==="
  echo "  steps done : $(grep -c " DONE " "$s")   failed: $(grep -c " FAIL " "$s")"
  echo "  last status: $(tail -n 1 "$s")"
  [ -e "$h.INSTALL_COMPLETE" ] && echo "  INSTALL_COMPLETE: $(grep render_probe "$h.INSTALL_COMPLETE")"
  echo "  last log   : $(tail -n 1 "$h.log" 2>/dev/null | cut -c1-200)"
done'
