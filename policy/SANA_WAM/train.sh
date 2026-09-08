#!/bin/bash
set -e

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
# SANA_WAM is an eval-only submission (CONTRIBUTING.md eval-only rule): training lives in the Sana
# repository. Place or symlink an HF-style checkpoint dir under checkpoints/ instead (see README).
echo "[SANA_WAM] eval-only adapter: training is not provided here (see README 'Training')."
exit 0
