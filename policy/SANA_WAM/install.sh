#!/bin/bash
set -euo pipefail

# Install the SANA_WAM runtime into the ACTIVE environment.
#
# torch/torchvision are installed first from the CUDA 12.8 wheel index because the generic PyPI
# wheel may pull a different CUDA runtime; the validated pair is torch 2.9.1 + torchvision 0.24.1.
# Set SANA_WAM_SKIP_TORCH=1 when the environment already carries a CUDA-enabled torch.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${SANA_WAM_SKIP_TORCH:-}" ]]; then
    python -m pip install "torch==2.9.1" "torchvision==0.24.1" --index-url https://download.pytorch.org/whl/cu128
fi

python -m pip install \
    "diffusers>=0.38" \
    "transformers>=4.46,<5" \
    "safetensors>=0.4" \
    "numpy>=1.26" \
    "pillow>=10" \
    "pyyaml>=6" \
    "huggingface_hub>=0.25"

# XPolicyLab itself (websocket server, utils) so `XPolicyLab.policy.SANA_WAM.model` and
# `client_server.ws` resolve in the policy environment.
python -m pip install -e "${XPL_ROOT}"

python - <<'PY'
import diffusers, torch, transformers
assert tuple(int(x) for x in diffusers.__version__.split(".")[:2]) >= (0, 38), diffusers.__version__
assert tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 46), transformers.__version__
print("torch", torch.__version__, "cuda", torch.version.cuda, "diffusers", diffusers.__version__, "transformers", transformers.__version__)
PY
