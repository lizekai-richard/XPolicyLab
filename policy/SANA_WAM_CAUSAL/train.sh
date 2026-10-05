#!/bin/bash
set -euo pipefail

# SANA_WAM_CAUSAL is an eval-only submission (CONTRIBUTING.md eval-only rule): causal post-training lives in the
# Sana repository (train_video_rwm_policy_causal.py).
echo "[SANA_WAM_CAUSAL] eval-only adapter: training is not provided here (see README 'Training')."
exit 0
