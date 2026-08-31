#!/bin/bash
# Start the FlashVLA policy server in its conda env.
#
# Unlike Pi_05 (uv + OpenPI), flashvla is a plain conda env, so this activates
# the env named by policy_conda_env in deploy.yml and puts the flashvla repo on
# PYTHONPATH. The server only needs a GPU and a port -- the simulator runs in a
# separate Isaac Sim container and talks to it over websockets.
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=${8:-""}
policy_server_port=$9
policy_server_host=${10:-"localhost"}

# Resolved physically, on purpose. RoboDojo reaches this tree through a symlinked
# XPolicyLab/, and upstream helpers locate the benchmark config as
# `XPolicyLab/utils/../../env_cfg` -- the kernel expands `..` after following the
# symlink, so a logical path here would send those lookups into the wrong tree.
# Physical paths keep the server anchored at the flashvla-side bench root no
# matter which of the two paths the caller used.
SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd -P "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd -P "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}, action_dim=${action_dim}"

# `conda` is a valid value for the caller's env slot (robodojo.sh passes the
# --policy-env verbatim), and deploy.yml is only readable once an env with
# PyYAML is active, so the env name cannot come from the yaml itself.
if [[ -z "${policy_conda_env}" || "${policy_conda_env}" == "conda" ]]; then
    policy_conda_env=flashvla
fi
echo "[SERVER] Activating conda environment: ${policy_conda_env}"
conda activate "${policy_conda_env}"
PYTHON_BIN="$(command -v python)"
echo "[SERVER] Using python: ${PYTHON_BIN}"

read_cfg() {
    "${PYTHON_BIN}" - "${yaml_file}" "$1" "$2" <<'PYCFG'
import sys
import yaml

yaml_path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
with open(yaml_path, encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
value = cfg.get(key)
print(default if value is None else value)
PYCFG
}

flashvla_repo="$(read_cfg flashvla_repo "")"
# This adapter ships inside the flashvla repo at sim_eval/robodojo/XPolicyLab,
# so when deploy.yml pins nothing the repo root is three levels above
# XPolicyLab. Resolve physically: RoboDojo reaches this tree through a symlink,
# and the logical path would walk up into the RoboDojo checkout instead.
if [[ -z "${flashvla_repo}" ]]; then
    flashvla_repo="$(cd -P "${XPL_ROOT}/../../.." && pwd)"
fi

# BENCH_ROOT makes `XPolicyLab.*` importable, XPL_ROOT makes `client_server.*`
# importable without pip-installing XPolicyLab into the policy env. The repo
# entry is a fallback for envs where `pip install -e` (install.sh) was skipped.
PYTHONPATH_PARTS=("${BENCH_ROOT}" "${XPL_ROOT}")
if [[ -d "${flashvla_repo}/flashvla" ]]; then
    PYTHONPATH_PARTS+=("${flashvla_repo}")
else
    echo "[SERVER] no flashvla checkout at ${flashvla_repo}; relying on the env's installed package"
fi

overrides=(
    port="${policy_server_port}"
    host="${policy_server_host}"
    bench_name="${bench_name}"
    task_name="${task_name}"
    ckpt_name="${ckpt_name}"
    env_cfg_type="${env_cfg_type}"
    seed="${seed}"
    policy_name="${policy_name}"
    action_type="${action_type}"
    action_dim="${action_dim}"
)

exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${PYTHON_BIN}" "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides "${overrides[@]}"
