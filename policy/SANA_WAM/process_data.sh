#!/bin/bash
set -e

# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
# SANA_WAM is an eval-only submission (CONTRIBUTING.md eval-only rule): data conversion lives in the
# Sana training repository and is not part of this adapter.
echo "[SANA_WAM] eval-only adapter: data processing is not provided here (see README 'Data Processing')."
exit 0
