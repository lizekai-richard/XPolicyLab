#!/bin/bash
set -euo pipefail

# SANA_WAM_HM is an eval-only submission (CONTRIBUTING.md eval-only rule): data conversion lives in the Sana repository
# (the frame-anchor samples of the causal chunk-window dataset, branch rwm/sparse-memory).
echo "[SANA_WAM_HM] eval-only adapter: data processing is not provided here (see README 'Data Processing')."
exit 0
