#!/usr/bin/env bash
# start_server.sh -- start the RoboDojo SANA policy server on THIS node in the
# background (no Slurm) and print where to watch it.
#
#   bash <CE_REMOTE_INSTALL>/start_server.sh                # single-task lane, insert_key
#   SANA_TASK_SET=A SANA_EPISODES_PER_TASK=25 bash <CE_REMOTE_INSTALL>/start_server.sh
#   SANA_TASK_LIST_FILE=~/tasks.txt SANA_EPISODES_PER_TASK=10 bash <CE_REMOTE_INSTALL>/start_server.sh
#   SANA_SERVER_GPU=3 SANA_RPC_PORT=41000 bash <CE_REMOTE_INSTALL>/start_server.sh
#
# Knobs are the sbatch's (SANA_EVAL_TASK / SANA_EVAL_INSTRUCTION / SANA_EVAL_SEED /
# SANA_TASK_SET / SANA_TASK_LIST_FILE / SANA_EPISODES_PER_TASK / SANA_RPC_PORT ...)
# plus SANA_SERVER_GPU (default 0) and SANA_EVIDENCE_ROOT (default ~/robodojo_evidence).
set -euo pipefail
_CE_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
[[ -f $_CE_HERE/site.env ]] && source "$_CE_HERE/site.env"   # the site configuration deploy.sh copies next to the node scripts
ROOT=${ROBODOJO_ROOT:-${CE_NODE_ROOT:?no site.env next to this script (deploy.sh) and no ROBODOJO_ROOT}}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${ROOT}/activate_sana_wam.sh"
EVIDENCE_ROOT=${SANA_EVIDENCE_ROOT:-${HOME}/robodojo_evidence}
mkdir -p "${EVIDENCE_ROOT}"
EVIDENCE_DIR=${EVIDENCE_ROOT}/robodojo_server_$(date +%Y%m%d-%H%M%S)_$$
LAUNCH_LOG=${EVIDENCE_DIR}.launcher.log
nohup setsid bash "${HERE}/run_learned_server_only_local.sh" "${EVIDENCE_DIR}" \
  >"${LAUNCH_LOG}" 2>&1 </dev/null &
LAUNCHER_PID=$!
sleep 2
if ! kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
  echo "launcher exited immediately; see ${LAUNCH_LOG}" >&2
  tail -n 20 "${LAUNCH_LOG}" >&2 || true
  exit 1
fi
cat <<EOF
server launcher started on $(hostname -s): pid ${LAUNCHER_PID}, gpu ${SANA_SERVER_GPU:-0}
  evidence : ${EVIDENCE_DIR}
  launcher : ${LAUNCH_LOG}
  server   : ${EVIDENCE_DIR}/server.stdout.log  (+ server.stderr.log)
  handoff  : ${EVIDENCE_DIR}/client_handoff/CLIENT_README.txt  (appears once the endpoint is up, ~10-25 min for the eager load)
  stop     : bash ${HERE}/stop_server.sh ${EVIDENCE_DIR}
EOF
