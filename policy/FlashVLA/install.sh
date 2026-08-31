#!/bin/bash
# Install the policy side of the FlashVLA adapter into the active environment.
#
# Create the environment from flashvla's own environment.yml first (it pins
# torch and lerobot); this script only wires the two repos into it, so pip has
# little left to resolve and cannot churn a working training env.
#
#   conda env create -f <flashvla>/environment.yml   # once
#   conda activate flashvla
#   bash install.sh
set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd -P "${SCRIPT_DIR}/../.." && pwd)"

# This adapter is vendored in the flashvla repo at sim_eval/robodojo/XPolicyLab,
# so the repo root is three levels above XPolicyLab. Resolving physically keeps
# this correct when the tree is reached through the RoboDojo symlink.
FLASHVLA_REPO="${FLASHVLA_REPO:-$(cd -P "${XPL_ROOT}/../../.." && pwd)}"
if [[ ! -f "${FLASHVLA_REPO}/flashvla/__init__.py" ]]; then
    echo "[INSTALL] ${FLASHVLA_REPO} is not a flashvla checkout." >&2
    echo "[INSTALL] Set FLASHVLA_REPO=/path/to/flashvla and re-run." >&2
    exit 1
fi

echo "[INSTALL] flashvla repo: ${FLASHVLA_REPO}"
echo "[INSTALL] XPolicyLab:    ${XPL_ROOT}"

python -m pip install -e "${FLASHVLA_REPO}"
# Brings in the websocket/msgpack transport and utils the policy server imports.
python -m pip install -e "${XPL_ROOT}"

python - <<'PY'
import flashvla
import client_server
from XPolicyLab.utils.process_data import get_robot_action_dim_info  # noqa: F401

print(f"[INSTALL] ok: flashvla {getattr(flashvla, '__version__', 'dev')}")
PY
