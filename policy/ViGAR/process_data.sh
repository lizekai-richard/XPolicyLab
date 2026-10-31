#!/bin/bash
set -euo pipefail

# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>
# Downloads ViGAR's prepared RoboTwin training data (50 tasks x 50 demo_clean episodes
# with stage annotations, 49-D actions and generated goal images) and the base
# checkpoint to data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>.
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="${POLICY_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
HF_REPO_ID=${VIGAR_HF_REPO:-DAGroup-PKU/ViGAR}
VIGAR_ROOT="${VIGAR_ROOT:-${POLICY_DIR}/source_vigar}"
HF_BIN="${VIGAR_ROOT}/.venv-model/bin/hf"
[[ -x "${HF_BIN}" ]] || HF_BIN=hf

mkdir -p "${data_dir}"
"${HF_BIN}" download "${HF_REPO_ID}" \
    --include "data/robotwin_segmented_2500/*" "weights/Cosmos3-Nano-dcp/*" \
    --local-dir "${data_dir}"
test -f "${data_dir}/data/robotwin_segmented_2500/dataset.json"
test -d "${data_dir}/data/robotwin_segmented_2500/goal_cache"
echo "[ViGAR] training data ready: ${data_dir}"
