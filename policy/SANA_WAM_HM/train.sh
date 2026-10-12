#!/bin/bash
set -euo pipefail

# SANA_WAM_HM is an eval-only submission (CONTRIBUTING.md eval-only rule): hierarchical-memory post-training lives in
# the Sana repository (train_video_rwm_policy_causal.py, branch rwm/sparse-memory).
echo "[SANA_WAM_HM] eval-only adapter: training is not provided here (see README 'Training')."
exit 0
