#!/usr/bin/env bash
# Velocity calibration run (an analysis, NOT an official score; user request 2026-09-29): the fixed-step DiT cache is
# calibrated on RoboDojo itself. The vanilla34k OpenWAM checkpoint (the dit_cache study's default) is served at the
# deploy sampling (10 NFE, video CFG 6 / action CFG 1, video / action flow shift 5 / 1, ee actions) through a COPY of
# the SANA_WAM adapter that dumps every denoising step's integrated velocities (deploy key velocity_dump_dir), for
# 10 episodes of every task config (42 tasks / 54 configs, seed 0, no layout shards).
#   copy of XPolicyLab   x5 ~/zekail/xpl_velocity/{XPolicyLab,env_cfg}  ->  NFS /homes/jiamingt/xpl_velocity/...
#                        (the deployed ~/xpl_zekai adapter is untouched; check-sync compares this copy instead)
#   node workspace       /dataset2/jiamingt/robodojo/xpl_velocity_ws (the shared xpl_zekai_ws keeps pointing at xpl_zekai)
#   ports                rpc / ws base 63900 / 64900 -> single-GPU slot g binds 63950+g / 64950+g (ports stop at
#                        65535: the first try with bases 66000 / 67000 could not bind)
#   dumps                /homes/jiamingt/robodojo_eval/<RUN>/velocity/<task>/<UTC start>_<pid>/ep<N>_c<M>.npz
#
#   bash velocity_calib.sh check-sync        the NFS adapter copy == x5 ~/zekail/xpl_velocity
#   bash velocity_calib.sh smoke ['l25:s5']  1 episode of insert_key -> <RUN>_smoke
#   bash velocity_calib.sh start ['<spec>']  the calibration run (default spec l25:s5,s6,s7, the GPUs the user freed)
#   bash velocity_calib.sh resume ['<spec>'] continue a stopped master (repeats the non-persisted flags)
#   bash velocity_calib.sh status | watch
set -euo pipefail
CKPT_LABEL=${CKPT_LABEL:-vanilla34k_eefabs_f33fps8_openwam_aligned}
NORM_SHA=${NORM_SHA:-1fe3b7e72e8fa93eca5efb8a8deeace1daf4434b4b97e9f4445670ac758287ed}
RUN=${RUN:-velocity_calib_${CKPT_LABEL}_vcfg6_acfg1_nfe10_vfs5_afs1_ee_ep10}
EPISODES=${EPISODES:-10}
SPEC_DEFAULT=${SPEC_DEFAULT:-l25:s5,s6,s7}
CKPT=/homes/jiamingt/checkpoints/huggingface/logits/$CKPT_LABEL
XPL_ROOT=/homes/jiamingt/xpl_velocity/XPolicyLab
XPL_WS=/dataset2/jiamingt/robodojo/xpl_velocity_ws
NFS_USER_HOST=jiamingt@l5.mit.edu
EVAL=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STATE=$HOME/robodojo_master/$RUN
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
nodes_of(){ echo "$1" | tr ';' '\n' | cut -d: -f1 | sed '/^$/d' | paste -sd, -; }
overrides(){ # $1 = the run whose root receives the dumps
  echo "sampling_steps=10 cfg_scale=6 action_cfg_scale=1 flow_shift=5 action_flow_shift=1" \
       "normalization_sha256=$NORM_SHA velocity_dump_dir=/homes/jiamingt/robodojo_eval/$1/velocity"
}
common(){ # $1 = run name
  COMMON=(--worker run_unit_xpl.sh --xpl-root "$XPL_ROOT" --xpl-action-type ee --sec-per-step 4 --checkpoint-root "$CKPT"
          --xpl-overrides "$(overrides "$1")" --worker-env "SANA_EVAL_XPL_WS=$XPL_WS"
          --support-arm-embodiments on --shard-plan "" --order dimension --rpc-port-base 63900 --ws-port-base 64900)
}
check_sync(){
  local diff
  diff=$(rsync -n -c -i -r --exclude __pycache__ --exclude .pytest_cache --exclude checkpoints \
    "$HOME/zekail/xpl_velocity/XPolicyLab/policy/SANA_WAM/" "$NFS_USER_HOST:$XPL_ROOT/policy/SANA_WAM/" | grep -v '^\.d\|^cd' || true)
  diff+=$(rsync -n -c -i "$EVAL/run_unit_xpl.sh" "$NFS_USER_HOST:/homes/jiamingt/robodojo_install/eval/" | grep -v '^\.' || true)
  if [[ -n $diff ]]; then echo "NFS copies differ from x5 (rsync them first):"; echo "$diff"; exit 1; fi
  log "the velocity adapter copy and the worker are in sync with NFS"
}
check_free(){ # refuse GPUs that hold memory and slot ports that listen, on each node of the spec
  local spec=$1 node gl ports
  for node in $(nodes_of "$spec" | tr ',' ' '); do
    gl=$(echo "$spec" | tr ';' '\n' | awk -F: -v n="$node" '$1 == n {print $2}' | tr ',' ' ' | sed 's/s//g')
    ports=""; for g in $gl; do ports+=" $((63950 + g)) $((64950 + g))"; done
    ssh -n -o BatchMode=yes -o ConnectTimeout=20 "jiamingt@$node.mit.edu" "
      rc=0
      for g in $gl; do
        used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i \$g | tr -d ' ')
        if [ \"\$used\" -ge 1024 ]; then echo \"REFUSE: $node GPU \$g has \${used} MiB in use\"; rc=3; fi
      done
      for p in $ports; do [ -z \"\$(ss -ltnH \"sport = :\$p\")\" ] || { echo \"REFUSE: port \$p listens on $node\"; rc=3; }; done
      exit \$rc" || exit 3
  done
  log "GPUs and ports free on $(nodes_of "$spec"): $spec"
}

cmd=${1:-help}; shift || true
case $cmd in
  check-sync) check_sync ;;
  smoke)
    spec=${1:-l25:s5}; check_sync; check_free "$spec"; common "${RUN}_smoke"
    bash "$EVAL/start_master.sh" "${RUN}_smoke" "${COMMON[@]}" --nodes "$(nodes_of "$spec")" --node-pairs "$spec" \
      --tasks insert_key --episodes-per-task 1 --pin-longest 0 --poll 20
    echo "expect '[SANA_WAM] ready: ... velocity_dump_dir=.../${RUN}_smoke/velocity/insert_key/...' in the server log" ;;
  start)
    spec=${1:-$SPEC_DEFAULT}; check_sync; check_free "$spec"; common "$RUN"
    # poll 20 s: a slot re-dispatches within seconds, so the robolab queue (same account) does not take the GPU between units
    bash "$EVAL/start_master.sh" "$RUN" "${COMMON[@]}" --nodes "$(nodes_of "$spec")" --node-pairs "$spec" \
      --episodes-per-task "$EPISODES" --poll 20
    log "started $RUN on '$spec'" ;;
  resume)
    spec=${1:-$SPEC_DEFAULT}; check_sync; common "$RUN"
    bash "$EVAL/start_master.sh" "$RUN" --resume "${COMMON[@]}" --nodes "$(nodes_of "$spec")" --node-pairs "$spec" --poll 20 ;;
  status) sed -n 1,40p "$STATE/STATUS.md"; echo ...; tail -n 6 "$STATE/master.log" ;;
  watch) tail -n 0 -F "$STATE/master.log" | grep -E --line-buffered "DONE |FAIL|failed [1-9]|quarantin|ALL UNITS SETTLED|Traceback|stale|server_timeout|server_fail|RE-QUEUE|requeue" ;;
  *) sed -n 2,20p "$0" ;;
esac
