#!/usr/bin/env bash
# Read-only (run on the control host): HOST=<node> PORT=<ws policy port> bash probe_unit_server.sh UNIT_ATTEMPT_DIR
# Shows, on the node itself, the policy-server process listening on PORT (pid, fd 1/2 targets), the attempt's server
# log sizes as the node sees them, the tail of its stdout and the client's simulation-step count -- to tell NFS
# write-back lag from lost output, and a stepping client from a stuck one.
set -u
# shellcheck source=../lib/site.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/site.sh" || exit 2
HOST=${HOST:?HOST=<node>}; PORT=${PORT:?PORT=<ws policy port>}; DIR=${1:?attempt dir}
ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$(ce_host "$HOST")" "
  pid=\$(ss -ltnpH 'sport = :$PORT' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  echo \"listener on :$PORT -> pid \${pid:-none}\"
  if [ -n \"\$pid\" ]; then
    ps -o pid=,etimes=,rss=,cmd= -p \$pid | cut -c1-200
    echo \"fd1 -> \$(readlink /proc/\$pid/fd/1)\"; echo \"fd2 -> \$(readlink /proc/\$pid/fd/2)\"
  fi
  ls -la '$DIR/server/'
  echo '--- stdout tail (node view)'; tail -c 1500 '$DIR/server/server.stdout.log'
  # the official client's resume manifest (node-local workspace): episodes recorded so far, successes, score
  task=\$(ls '$DIR/client' 2>/dev/null | head -1 | sed 's/^[0-9]*_//')
  rid=\$(grep -o 'run_id=[^ ]*' '$DIR/status.txt' 2>/dev/null | tail -1 | cut -d= -f2)
  man=\$(ls -t $CE_NODE_XPL_WS/eval_result/RoboDojo/\$task/*/arx_x5/*/_resume_\$rid.json 2>/dev/null | head -1)
  if [ -n \"\$man\" ]; then python3 -c \"import json,sys; d=json.load(open(sys.argv[1])); det=d.get('details') or {}; print('--- manifest: %d episodes recorded, %d successes, score_sum %.2f' % (len(det), sum(1 for v in det.values() if v.get('success')), sum(float(v.get('score') or 0) for v in det.values())))\" \"\$man\"; fi
  for f in '$DIR'/client/*/client.stdout.log; do
    [ -f \"\$f\" ] || continue
    echo \"--- client \$(basename \$(dirname \$f)): \$(grep -a -o 'env0 step' \$f | wc -l) simulation steps; last counter: \$(tr '\\r' '\\n' < \$f | grep -a 'env0 step' | tail -1 | sed 's/\\x1b\\[[0-9;]*m//g')\"
  done
" 2>&1
