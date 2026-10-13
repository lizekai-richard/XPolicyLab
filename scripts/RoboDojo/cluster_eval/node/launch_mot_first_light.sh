#!/usr/bin/env bash
# Run on x5: CKPT_LABEL=<snapshot> [HOST=l5] [GPU=0] [EXTRA="--skip-forward"] bash launch_mot_first_light.sh
# Runs mot_first_light.py on one MIT node in the sana-wam env against the NFS checkpoint snapshot and the
# NFS XPolicyLab copy (~/xpl_zekai/XPolicyLab). Foreground: prints the probe's output here.
set -u
: "${CKPT_LABEL:?CKPT_LABEL=<snapshot dir name> required}"; HOST=${HOST:-l5}; GPU=${GPU:-0}; EXTRA=${EXTRA:-}
ssh -n -o BatchMode=yes -o ConnectTimeout=20 "jiamingt@$HOST.mit.edu" "
  set -u
  source /dataset2/jiamingt/robodojo/activate_sana_wam.sh
  cd ~/robodojo_install || exit 1
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning \
    \"\$SANA_PY\" mot_first_light.py \
      --ckpt /homes/jiamingt/checkpoints/huggingface/logits/$CKPT_LABEL \
      --xpl-root /homes/jiamingt/xpl_zekai/XPolicyLab $EXTRA
" 2>&1
