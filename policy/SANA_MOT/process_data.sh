#!/bin/bash
set -e

# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
# SANA_MOT is an eval-only submission (CONTRIBUTING.md eval-only rule): data conversion (the OpenWAM canvas
# dataset over the RoboDojo HDF5 release) lives in the Sana training repository and is not part of this adapter.
echo "[SANA_MOT] eval-only adapter: data processing is not provided here (see README 'Data Processing')."
exit 0
