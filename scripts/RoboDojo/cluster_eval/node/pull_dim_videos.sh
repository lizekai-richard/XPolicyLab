#!/usr/bin/env bash
# Pull the rollout videos of the given capability dimensions of one finished run from the MIT NFS to x5.
# Usage (x5): [DRY=1] bash pull_dim_videos.sh <RUN> <dim,dim,...>     e.g. memory,long-horizon
# Takes every unit of those dimensions' tasks (layout shards, tail-template sub-units and every attempt), keeping only the
# rollout mp4s and the result json files, into ~/zekail/robodojo_results/runs/<RUN>/videos/<unit>/a<N>/client/<i>_<task>/
# upstream_eval_result/ (the layout of the earlier video archives). Re-runnable; a second pass verifies with rsync -n.
set -euo pipefail
RUN=${1:?run name}; DIMS=${2:?comma-separated dimensions}
STATE=$HOME/robodojo_master/$RUN/state.json
[[ -f $STATE ]] || { echo "no state.json for $RUN"; exit 2; }
SRC=jiamingt@l5.mit.edu:/homes/jiamingt/robodojo_eval/$RUN/units/
DST=$HOME/zekail/robodojo_results/runs/$RUN/videos/
mapfile -t TASKS < <("$HOME/miniforge3/envs/sana-wam/bin/python" - "$STATE" "$DIMS" <<'PY'
import json, sys
dims = set(sys.argv[2].split(","))
for task in sorted({u["task"] for u in json.load(open(sys.argv[1]))["units"] if u["dimension"] in dims}):
    print(task)
PY
)
[[ ${#TASKS[@]} -gt 0 ]] || { echo "no task of dimensions $DIMS in $RUN"; exit 2; }
FILTER=()
for t in "${TASKS[@]}"; do FILTER+=(--include="/${t}__s0*/"); done
FILTER+=(--exclude='/*/' --include='*/' --include='*.mp4' --include='_result.json' --include='result.json' --exclude='*')
mkdir -p "$DST"
echo "$RUN: ${#TASKS[@]} tasks of [$DIMS]: ${TASKS[*]}"
if [[ ${DRY:-0} == 1 ]]; then
  rsync -a -m -n --stats "${FILTER[@]}" "$SRC" "$DST" | grep -E "Number of (regular )?files|Total file size"
  exit 0
fi
rsync -a -m "${FILTER[@]}" "$SRC" "$DST"
left=$(rsync -a -m -n -i "${FILTER[@]}" "$SRC" "$DST" | grep -c '^>f' || true)
echo "$RUN: pulled $(find "$DST" -name '*.mp4' | wc -l) mp4 ($(du -sh "$DST" | cut -f1)); files still differing after the pass: $left"
