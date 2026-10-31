#!/bin/bash
set -euo pipefail

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
# Post-trains the ViGAR policy from Cosmos3-Nano with ViGAR's scripts/train_policy.sh:
# 8 GPUs (gpu_id lists eight ids), global batch 256, 50,000 updates. Extra
# arguments go to ViGAR's trainer. Output: checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/runs/
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6
shift 6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIGAR_ROOT="${VIGAR_ROOT:-${POLICY_DIR}/source_vigar}"
data_dir="${POLICY_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
weights="${VIGAR_WEIGHTS:-${POLICY_DIR}/checkpoints/ViGAR}/weights/policy/robotwin_c2r"
ckpt_dir="${POLICY_DIR}/checkpoints/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"

if [[ "$(tr ',' '\n' <<< "${gpu_id}" | grep -c .)" != 8 ]]; then
    echo "[ViGAR] train.sh runs on 8 GPUs; pass eight ids, e.g. 0,1,2,3,4,5,6,7" >&2
    exit 1
fi
for path in "${data_dir}/data/robotwin_segmented_2500/dataset.json" "${weights}/vae" "${weights}/text_tokenizer"; do
    [[ -e "${path}" ]] || { echo "[ViGAR] missing ${path}; run process_data.sh and download_checkpoint.sh" >&2; exit 1; }
done

mkdir -p "${ckpt_dir}"
# ViGAR's recipe fixes its own training seed; <seed> only names the run directory.
exec env \
    VIGAR_ENV_FILE=/dev/null \
    PYTHON_BIN="${VIGAR_ROOT}/.venv-model/bin/python" \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    VIGAR_POLICY_WORKSPACE="${ckpt_dir}" \
    VIGAR_DATASET_MANIFEST="${data_dir}/data/robotwin_segmented_2500/dataset.json" \
    VIGAR_GENERATED_GOAL_CACHE="${data_dir}/data/robotwin_segmented_2500/goal_cache" \
    BASE_CHECKPOINT_PATH="${data_dir}/weights/Cosmos3-Nano-dcp" \
    WAN_VAE_PATH="${weights}/vae/Wan2.2_VAE.pth" \
    QWEN_TOKENIZER_PATH="${weights}/text_tokenizer" \
    bash "${VIGAR_ROOT}/scripts/train_policy.sh" "$@"
