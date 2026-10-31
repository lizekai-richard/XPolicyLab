#!/bin/bash
set -euo pipefail

# Clones ViGAR and builds its Python 3.13 / PyTorch 2.10 model environment
# (<vigar_root>/.venv-model, needs uv), which runs the planner, the policy and
# this adapter. The simulator uses RoboTwin's own environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VIGAR_ROOT="${VIGAR_ROOT:-${SCRIPT_DIR}/source_vigar}"
revision=c8813a6464f95600e793f3816df6b860ec46775e

if [[ ! -e "${VIGAR_ROOT}" ]]; then
    git clone https://github.com/DAGroup-PKU/ViGAR.git "${VIGAR_ROOT}"
    git -C "${VIGAR_ROOT}" checkout "${revision}"
fi
test -f "${VIGAR_ROOT}/evaluation/native_async_rollout.py"

if [[ ! -x "${VIGAR_ROOT}/.venv-model/bin/python" ]]; then
    bash "${VIGAR_ROOT}/scripts/setup_models.sh" "${VIGAR_ROOT}/.venv-model"
fi
PYTHON_BIN="${VIGAR_ROOT}/.venv-model/bin/python"
uv pip install --python "${PYTHON_BIN}" websockets msgpack msgpack-numpy h5py pydantic pyyaml
uv pip install --python "${PYTHON_BIN}" --no-deps -e "${XPL_ROOT}"
