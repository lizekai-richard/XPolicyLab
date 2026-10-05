#!/bin/bash
set -euo pipefail

# Install the SANA_MOT runtime into the ACTIVE environment.
#
# SANA_MOT runs in the same environment as SANA_WAM: its vendored package ``sana_mot_min`` holds only the MoT model
# mirror, the OpenWAM canvas front-end and the shared-prompt renderer, and imports everything else (VAE, Gemma,
# Robot80, sampler, RoboDojo codecs) from the sibling adapter's ``sana_wam_min``. The SANA_WAM installer therefore
# is the SANA_MOT installer (torch 2.9.1 / torchvision 0.24.1 from the cu128 index unless SANA_WAM_SKIP_TORCH=1,
# diffusers>=0.38, transformers, safetensors, numpy, pillow, pyyaml, huggingface_hub, and XPolicyLab itself).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANA_WAM_DIR="${SCRIPT_DIR}/../SANA_WAM"

if [[ ! -f "${SANA_WAM_DIR}/install.sh" ]]; then
    echo "[SANA_MOT] policy/SANA_WAM/install.sh not found; SANA_MOT shares the SANA_WAM runtime and environment" >&2
    exit 1
fi
bash "${SANA_WAM_DIR}/install.sh"

echo "[SANA_MOT] runtime installed (shared with SANA_WAM)"
