#!/bin/bash
set -euo pipefail

# SANA_WAM_CAUSAL is an eval-only submission (CONTRIBUTING.md eval-only rule): data conversion lives in the
# Sana repository (the causal chunk-window dataset of the bidirectional SFT pipeline).
echo "[SANA_WAM_CAUSAL] eval-only adapter: data processing is not provided here (see README 'Data Processing')."
exit 0
