#!/usr/bin/env bash
# start_master.sh RUN_NAME [master.py args...] -- run on the control host. Starts master.py detached; state/logs under
# <CE_MASTER_ROOT>/RUN_NAME, run dir on the shared home at <CE_EVAL_ROOT>/RUN_NAME (site.env). Refuses to start a second
# master for the same RUN_NAME while one is alive, and requires --resume to continue an existing state.
#   bash start_master.sh official_seed0_dyn ...
#   bash start_master.sh official_seed0_dyn --resume ...   (the same arguments as the first start, plus --resume)
set -euo pipefail
RUN=${1:?usage: start_master.sh RUN_NAME [master.py args]}; shift || true
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../lib/site.sh
source "$HERE/../lib/site.sh"
STATE=$CE_MASTER_ROOT/$RUN; mkdir -p "$STATE"
if [[ -s $STATE/master.pid ]] && kill -0 "$(cat "$STATE/master.pid")" 2>/dev/null; then
  echo "a master for $RUN is already running (pid $(cat "$STATE/master.pid")); stop it first: kill -TERM $(cat "$STATE/master.pid")" >&2; exit 1
fi
if [[ -f $STATE/state.json && " $* " != *" --resume "* ]]; then
  echo "$STATE/state.json exists: add --resume to continue that run, or pick a new RUN_NAME" >&2; exit 1
fi
nohup setsid "$CE_PY" "$HERE/master.py" --run-root "$CE_EVAL_ROOT/$RUN" --checkout "$CE_ROBODOJO_CHECKOUT" \
  --state-dir "$STATE" --ssh-user "$CE_SSH_USER" --ssh-domain "$CE_SSH_DOMAIN" --remote-install "$CE_REMOTE_INSTALL" \
  "$@" >>"$STATE/master.stdout.log" 2>&1 </dev/null &
echo $! >"$STATE/master.pid"
echo "master pid $! ; log: $STATE/master.log ; status: $STATE/STATUS.md ; stop: kill -TERM $!"
