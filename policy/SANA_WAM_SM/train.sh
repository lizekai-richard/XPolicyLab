#!/bin/bash
set -euo pipefail

# SANA_WAM_SM is an eval-only submission (CONTRIBUTING.md eval-only rule): sparse-memory post-training lives in
# the Sana repository (train_video_rwm_policy_causal.py, branch rwm/sparse-memory).
echo "[SANA_WAM_SM] eval-only adapter: training is not provided here (see README 'Training')."
exit 0
