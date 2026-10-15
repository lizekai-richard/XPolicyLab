#!/bin/bash
set -euo pipefail

# SANA_WAM_SM is an eval-only submission (CONTRIBUTING.md eval-only rule): data conversion lives in the Sana repository
# (the frame-anchor samples of the causal chunk-window dataset and the sparse-memory latent store, branch
# rwm/sparse-memory).
echo "[SANA_WAM_SM] eval-only adapter: data processing is not provided here (see README 'Data Processing')."
exit 0
