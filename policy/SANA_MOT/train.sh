#!/bin/bash
set -e

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
# SANA_MOT is an eval-only submission (CONTRIBUTING.md eval-only rule): training lives in the Sana repository
# (branch rwm/mot, dfw/run_sft_robodojo_mot_jointabs_f25_openwam.sbatch). Place or symlink an HF-style checkpoint
# dir under checkpoints/ instead (see README).
echo "[SANA_MOT] eval-only adapter: training is not provided here (see README 'Training')."
exit 0
