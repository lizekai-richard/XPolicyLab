#!/usr/bin/env bash
# stop_eval.sh RUN_ROOT [job ...] -- run on x5. Sends TERM to each job runner
# (it stops its client and asks its server for a graceful shutdown).
set -euo pipefail
RUN_ROOT=${1:?usage: stop_eval.sh RUN_ROOT [job_id ...]}; shift || true
ONLY=("$@")
USER_AT=${SSH_USER:-jiamingt}; DOMAIN=${SSH_DOMAIN:-.mit.edu}
PLAN_LOCAL=$(mktemp); ssh -n -o BatchMode=yes "$USER_AT@l5$DOMAIN" "cat '$RUN_ROOT/plan.json'" >"$PLAN_LOCAL"
mapfile -t JOBS < <(python3 -c '
import json,sys; p=json.load(open(sys.argv[1]))
for j in p["jobs"]: print(j["job_id"], j["node"])' "$PLAN_LOCAL")
for rec in "${JOBS[@]}"; do
  read -r jid node <<<"$rec"
  if (( ${#ONLY[@]} )) && [[ ! " ${ONLY[*]} " == *" $jid "* ]]; then continue; fi
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$USER_AT@$node$DOMAIN" \
    "p=\$(cat '$RUN_ROOT/jobs/$jid/runner.pid' 2>/dev/null); if [ -n \"\$p\" ] && kill -0 \"\$p\" 2>/dev/null; then kill -TERM \"\$p\" && echo \"TERM sent to $jid (pid \$p)\"; else echo \"$jid not running\"; fi" \
    || echo "stop FAILED: $jid on $node"
done
