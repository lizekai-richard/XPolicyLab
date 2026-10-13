#!/usr/bin/env bash
# start_master.sh RUN_NAME [master.py args...] -- run on x5. Starts master.py
# detached; state/logs under ~/robodojo_master/RUN_NAME, run dir on the shared
# home at /homes/jiamingt/robodojo_eval/RUN_NAME.  Refuses to start a second
# master for the same RUN_NAME while one is alive, and requires --resume to
# continue an existing state.
#   bash start_master.sh official_seed0_dyn
#   bash start_master.sh smoke_dyn --tasks stack_bowls,insert_key --episodes-per-task 1 --nodes l5 --slots-per-node 2
#   bash start_master.sh official_seed0_dyn --resume
set -euo pipefail
RUN=${1:?usage: start_master.sh RUN_NAME [master.py args]}; shift || true
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${MASTER_PY:-$HOME/miniforge3/envs/sana-wam/bin/python}      # any python3 with pyyaml
CHECKOUT=${ROBODOJO_CHECKOUT_X5:-$HOME/zekail/sana_wam_sim_benchmarks/RoboDojo-official-clone-20260823}
STATE=$HOME/robodojo_master/$RUN; mkdir -p "$STATE"
if [[ -s $STATE/master.pid ]] && kill -0 "$(cat "$STATE/master.pid")" 2>/dev/null; then
  echo "a master for $RUN is already running (pid $(cat "$STATE/master.pid")); stop it first: kill -TERM $(cat "$STATE/master.pid")" >&2; exit 1
fi
if [[ -f $STATE/state.json && " $* " != *" --resume "* ]]; then
  echo "$STATE/state.json exists: add --resume to continue that run, or pick a new RUN_NAME" >&2; exit 1
fi
nohup setsid "$PY" "$HERE/master.py" --run-root "/homes/jiamingt/robodojo_eval/$RUN" --checkout "$CHECKOUT" \
  --state-dir "$STATE" "$@" >>"$STATE/master.stdout.log" 2>&1 </dev/null &
echo $! >"$STATE/master.pid"
echo "master pid $! ; log: $STATE/master.log ; status: $STATE/STATUS.md ; stop: kill -TERM $!"
