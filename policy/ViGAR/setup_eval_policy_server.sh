#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_env=${8}
policy_server_port=${9}
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

# <policy_env>: `uv` for the environment made by install.sh, a ViGAR checkout,
# a Python environment directory, or a conda environment.
vigar_root="${VIGAR_ROOT:-}"
if [[ "${policy_env}" == "uv" ]]; then
    vigar_root="${vigar_root:-${SCRIPT_DIR}/source_vigar}"
    python_bin="${vigar_root}/.venv-model/bin/python"
elif [[ -x "${policy_env}/.venv-model/bin/python" ]]; then
    vigar_root="$(cd "${policy_env}" && pwd)"
    python_bin="${vigar_root}/.venv-model/bin/python"
elif [[ -x "${policy_env}/bin/python" ]]; then
    python_bin="${policy_env}/bin/python"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_env}"
    python_bin="$(command -v python)"
fi
if [[ ! -x "${python_bin}" ]]; then
    echo "[SERVER][ERROR] Python not found: ${python_bin}; run bash ${SCRIPT_DIR}/install.sh" >&2
    exit 1
fi
overrides=()
if [[ -n "${vigar_root}" ]]; then
    overrides+=(vigar_root="${vigar_root}")
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, python=${python_bin}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONPATH="${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${python_bin}" "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            ${overrides[@]+"${overrides[@]}"}
