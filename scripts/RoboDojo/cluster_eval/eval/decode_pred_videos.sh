#!/usr/bin/env bash
# Decode every unit attempt of a run that was served with prediction_latent_dir=@ATTEMPT_DIR@/pred_latents into side-by-side
# MP4s (real | predicted, per camera) with policy/SANA_WAM/tools/decode_prediction_latents.py, one decode process per GPU.
#   bash decode_pred_videos.sh RUN 'l25:0,1;l26:0,2'      -> NFS /homes/jiamingt/robodojo_eval/RUN/pred_videos/*.mp4
# The unit attempts come from the master's state.json (final attempt of every done unit). Logs: .../pred_videos/logs/.
set -euo pipefail
RUN=${1:?RUN}; SPEC=${2:?"'node:g,g;node:g'"}
PY=${MASTER_PY:-$HOME/miniforge3/envs/sana-wam/bin/python}
ST=$HOME/robodojo_master/$RUN/state.json
ROOT=/homes/jiamingt/robodojo_eval/$RUN
OUT=$ROOT/pred_videos
mapfile -t ATTEMPTS < <("$PY" - "$ST" "$ROOT" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
for u in s["units"]:
    if u["state"] == "done":
        print(f"{sys.argv[2]}/units/{u['unit_id']}/a{u.get('attempts', 1)}")
PYEOF
)
mapfile -t GPUS < <(echo "$SPEC" | tr ';' '\n' | while IFS=: read -r n gs; do for g in ${gs//,/ }; do echo "$n:${g#s}"; done; done)
(( ${#GPUS[@]} > 0 )) || { echo "no GPUs in '$SPEC'"; exit 2; }
echo "$(date -u +%FT%TZ) $RUN: ${#ATTEMPTS[@]} attempts over ${#GPUS[@]} GPUs -> $OUT"
declare -A LISTS
for i in "${!ATTEMPTS[@]}"; do k=${GPUS[$(( i % ${#GPUS[@]} ))]}; LISTS[$k]+="${ATTEMPTS[$i]} "; done
for k in "${!LISTS[@]}"; do
  node=${k%%:*}; gpu=${k##*:}
  ssh -n -o BatchMode=yes -o ConnectTimeout=15 "jiamingt@$node.mit.edu" "mkdir -p $OUT/logs; nohup setsid bash -c '
    source /dataset2/jiamingt/robodojo/activate_sana_wam.sh >/dev/null 2>&1
    cd /homes/jiamingt/xpl_zekai/XPolicyLab/policy/SANA_WAM
    for a in ${LISTS[$k]}; do
      CUDA_VISIBLE_DEVICES=$gpu \"\$SANA_PY\" tools/decode_prediction_latents.py --attempt-dir \$a --out $OUT || echo \"FAILED \$a\"
    done; echo DONE' >$OUT/logs/${node}_g${gpu}.log 2>&1 </dev/null &" && echo "  $node:$gpu <- $(wc -w <<<"${LISTS[$k]}") attempts"
done
