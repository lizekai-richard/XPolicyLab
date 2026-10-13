#!/usr/bin/env bash
# status_eval.sh RUN_ROOT [LOCAL_MIRROR] -- run on x5. Pulls the run's status
# files (no videos) with rsync via l5 and prints one line per job.
set -euo pipefail
RUN_ROOT=${1:?usage: status_eval.sh RUN_ROOT [LOCAL_MIRROR]}
MIRROR=${2:-$HOME/robodojo_eval_mirror/$(basename "$RUN_ROOT")}
USER_AT=${SSH_USER:-jiamingt}; DOMAIN=${SSH_DOMAIN:-.mit.edu}
mkdir -p "$MIRROR"
rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
  --include='*/' --include='plan.json' --include='tasks.json' --include='status.txt' --include='JOB_*' \
  --include='runner.log' --include='run_status.json' --include='_result.json' --include='benchmark_client_status.json' \
  --exclude='*' "$USER_AT@l5$DOMAIN:$RUN_ROOT/" "$MIRROR/"
echo "run: $RUN_ROOT   (mirror $MIRROR, $(date '+%F %T'))"
python3 - "$MIRROR" <<'PY'
import json, sys, glob, os, re
root = sys.argv[1]; plan = json.load(open(f"{root}/plan.json"))
tot_done = tot_units = 0
for j in plan["jobs"]:
    jd = f"{root}/jobs/{j['job_id']}"; units = sum(len(s["tasks"]) for s in j["sessions"])
    st = open(f"{jd}/status.txt").read().splitlines() if os.path.exists(f"{jd}/status.txt") else []
    last = st[-1] if st else "(not started)"
    m = re.search(r"done=(\d+)/(\d+)", last); done = int(m.group(1)) if m else 0
    flag = "COMPLETE" if os.path.exists(f"{jd}/JOB_COMPLETE") else "INCOMPLETE" if os.path.exists(f"{jd}/JOB_INCOMPLETE") else ""
    tot_done += done; tot_units += units
    print(f"{j['job_id']} {j['node']:4s} gpus {j['server_gpu']}/{j['client_gpu']}  tasks {done:3d}/{units:<3d} {flag:10s} {last[20:140]}")
print(f"TOTAL task-units done {tot_done}/{tot_units}")
PY
