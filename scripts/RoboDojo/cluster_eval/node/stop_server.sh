#!/usr/bin/env bash
# stop_server.sh EVIDENCE_DIR -- ask the launcher for a graceful shutdown (it
# forwards TERM to the server, waits the 90 s grace, then writes run_status.json
# and checksums.sha256) and wait for it to finish.
set -euo pipefail
EVIDENCE_DIR=${1:?usage: stop_server.sh EVIDENCE_DIR}
PID_FILE=${EVIDENCE_DIR}/launcher.pid
[[ -s "${PID_FILE}" ]] || { echo "no launcher.pid under ${EVIDENCE_DIR}" >&2; exit 1; }
PID=$(<"${PID_FILE}")
if ! kill -0 "${PID}" 2>/dev/null; then
  echo "launcher ${PID} is not running (already stopped?)"; exit 0
fi
kill -TERM "${PID}"
echo "TERM sent to launcher ${PID}; waiting for the server to drain (grace 90 s)..."
for _ in $(seq 1 900); do
  kill -0 "${PID}" 2>/dev/null || break
  sleep 1
done
if kill -0 "${PID}" 2>/dev/null; then
  echo "launcher still alive after 900 s; inspect ${EVIDENCE_DIR}/server.stderr.log" >&2; exit 1
fi
echo "stopped. run_status:"; cat "${EVIDENCE_DIR}/run_status.json" 2>/dev/null || echo "(no run_status.json)"
