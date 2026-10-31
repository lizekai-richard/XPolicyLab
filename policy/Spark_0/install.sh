#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${SCRIPT_DIR}/source_starvla"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m pip install torch==2.6.0 torchvision==0.21.0
python -m pip install -r "${STARVLA_ROOT}/requirements.txt"
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
# Pillow is pinned because train (dataloader _pack_sample) and eval (model.py)
# both resize through PIL's default resample kernel; the pin keeps the kernel
# identical across environments. Installed after requirements.txt (which lists
# an unpinned `pillow`) so this version wins.
python -m pip install h5py pandas numpy==1.26.4 opencv-python==4.10.0.84 pillow==12.2.0
python -m pip install -e "${STARVLA_ROOT}"

# The adapter (model.py, process_data) imports `XPolicyLab.*`; install the
# XPolicyLab package itself when this policy lives under XPolicyLab/policy/.
if [[ -f "${XPL_ROOT}/pyproject.toml" ]]; then
    python -m pip install -e "${XPL_ROOT}"
else
    echo "[install][WARN] XPolicyLab root not found at ${XPL_ROOT};" >&2
    echo "[install][WARN] install it manually (pip install -e <XPolicyLab>) or model.py imports will fail." >&2
fi
