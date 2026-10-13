#!/usr/bin/env bash
# launch_eval.sh RUN_ROOT [job ...] -- run on x5. Starts every job of the plan
# (or only the named ones) on its node, detached.  RUN_ROOT is the path AS SEEN
# ON THE NODES (shared home), e.g. /homes/jiamingt/robodojo_eval/<run_id>.
# Jobs on the same node are staggered so their first Isaac/bridge syncs do not
# collide.  Only the nohup command is backgrounded (see launch_all.sh note).
set -euo pipefail
RUN_ROOT=${1:?usage: launch_eval.sh RUN_ROOT [job_id ...]}; shift || true
[[ $RUN_ROOT == /*robodojo* ]] || { echo "RUN_ROOT must be an absolute node-side path containing 'robodojo'" >&2; exit 64; }
ONLY=("$@")
USER_AT=${SSH_USER:-jiamingt}; DOMAIN=${SSH_DOMAIN:-.mit.edu}
STAGGER=${STAGGER:-20}
PLAN_LOCAL=${PLAN_LOCAL:-}   # optional local copy of plan.json for reading; else fetched via ssh cat
if [[ -z $PLAN_LOCAL ]]; then
  PLAN_LOCAL=$(mktemp); trap 'rm -f "$PLAN_LOCAL"' EXIT; ssh -n -o BatchMode=yes "$USER_AT@l5$DOMAIN" "cat '$RUN_ROOT/plan.json'" >"$PLAN_LOCAL"
fi
mapfile -t JOBS < <(python3 -c '
import json,sys; p=json.load(open(sys.argv[1]))
for j in p["jobs"]: print(j["job_id"], j["node"])' "$PLAN_LOCAL")
prev_node=""
for rec in "${JOBS[@]}"; do
  read -r jid node <<<"$rec"
  if (( ${#ONLY[@]} )) && [[ ! " ${ONLY[*]} " == *" $jid "* ]]; then continue; fi
  [[ $node == "$prev_node" ]] && sleep "$STAGGER"; prev_node=$node
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$USER_AT@$node$DOMAIN" \
    "mkdir -p '$RUN_ROOT/jobs/$jid'; nohup setsid bash ~/robodojo_install/eval/run_eval_job.sh '$RUN_ROOT/plan.json' '$jid' > '$RUN_ROOT/jobs/$jid/nohup.log' 2>&1 < /dev/null & sleep 1; echo \"launched $jid on \$(hostname -s), pid \$!\"" \
    || echo "launch FAILED: $jid on $node"
done
echo "status: bash $(dirname "$0")/status_eval.sh $RUN_ROOT"
