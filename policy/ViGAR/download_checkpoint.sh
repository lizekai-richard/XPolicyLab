#!/bin/bash
set -euo pipefail

# Downloads the released ViGAR policy and subgoal planner (about 60 GB) from
# https://huggingface.co/DAGroup-PKU/ViGAR.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_REPO_ID=${VIGAR_HF_REPO:-DAGroup-PKU/ViGAR}
TARGET_DIR=${1:-${SCRIPT_DIR}/checkpoints/ViGAR}
VIGAR_ROOT="${VIGAR_ROOT:-${SCRIPT_DIR}/source_vigar}"
HF_BIN="${VIGAR_ROOT}/.venv-model/bin/hf"
[[ -x "${HF_BIN}" ]] || HF_BIN=hf

mkdir -p "${TARGET_DIR}"
"${HF_BIN}" download "${HF_REPO_ID}" \
    --include "weights/policy/robotwin_c2r/*" "weights/subgoal_planner/robotwin_subgoal_planner/*" \
    --local-dir "${TARGET_DIR}"
test -f "${TARGET_DIR}/weights/policy/robotwin_c2r/complete.json"
test -f "${TARGET_DIR}/weights/subgoal_planner/robotwin_subgoal_planner/inference_config.json"
echo "[ViGAR] weights ready: ${TARGET_DIR}"
