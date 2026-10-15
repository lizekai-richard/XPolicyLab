# shellcheck shell=bash
# Adapter profiles of the per-run driver (eval/run.sh): what differs between serving policy/SANA_WAM, policy/SANA_MOT,
# policy/SANA_WAM_CAUSAL and policy/SANA_WAM_SM through the XPolicyLab lane (eval/run_unit_xpl.sh). `ce_adapter_profile
# ADAPTER` sets
#   POLICY              the adapter directory under policy/ = the policy segment of the official eval_result layout
#                       (eval_result/RoboDojo/<task>/<POLICY>/arx_x5/...) and the row name of the official summary
#   RPC_BASE WS_BASE    master --rpc-port-base / --ws-port-base: single-GPU slot g binds base + 50 + g on its node. The ranges
#                       are disjoint per adapter, from every historical campaign and from the tail shards (62700 / 64700 + g),
#                       so runs of different adapters can share a node (a GPU belongs to one run at a time)
#   SMOKE_RPC SMOKE_WS  ports of the direct one-GPU smoke (run.sh smoke-gpu); a campaign adds i for its i-th checkpoint
#   WORKER_ENV          exported into every worker: selects the adapter in run_unit_xpl.sh (SANA_WAM is its default)
#   SYNC_POLICIES       adapter dirs that must be identical in this checkout and on the shared home (run.sh check-sync)
#   DIM_ORDER_DEFAULT   master --dimension-order: the order the capability categories are dispatched in
#   VIDEO_FS_DEFAULT ACTION_FS_DEFAULT   flow shifts; an empty action flow shift = the shared schedule (no override)
#   VIDEO_LAYOUT_DEFAULT                 SANA_MOT only: the video_layout override that asserts the checkpoint's layout
ce_adapter_profile(){
  case $1 in
    SANA_WAM)          # the single-stream SANA world-action model (sana_pixel canvas, sana_latent strip, openwam canvas)
      POLICY=SANA_WAM; RPC_BASE=63200; WS_BASE=65200; SMOKE_RPC=63290; SMOKE_WS=65290
      WORKER_ENV=""; SYNC_POLICIES="SANA_WAM"
      DIM_ORDER_DEFAULT=precision,memory,long-horizon,generalization,open
      VIDEO_FS_DEFAULT=5; ACTION_FS_DEFAULT=1; VIDEO_LAYOUT_DEFAULT="" ;;
    SANA_MOT)          # the mixture-of-transformers policy (video + action experts); imports policy/SANA_WAM helpers
      POLICY=SANA_MOT; RPC_BASE=62600; WS_BASE=64600; SMOKE_RPC=62690; SMOKE_WS=64690
      WORKER_ENV="SANA_EVAL_XPL_POLICY=SANA_MOT"; SYNC_POLICIES="SANA_MOT SANA_WAM"
      DIM_ORDER_DEFAULT=precision,memory,long-horizon,generalization,open
      VIDEO_FS_DEFAULT=8; ACTION_FS_DEFAULT=""; VIDEO_LAYOUT_DEFAULT=sana_pixel_canvas ;;
    SANA_WAM_CAUSAL)   # the chunk-causal policy (episode protocol, single form: sink + memory chunks, no chunk observation);
                       # imports policy/SANA_WAM/sana_wam_min
      POLICY=SANA_WAM_CAUSAL; RPC_BASE=63600; WS_BASE=65300; SMOKE_RPC=63690; SMOKE_WS=65390
      WORKER_ENV="SANA_EVAL_XPL_POLICY=SANA_WAM_CAUSAL"; SYNC_POLICIES="SANA_WAM SANA_WAM_CAUSAL"
      DIM_ORDER_DEFAULT=memory,long-horizon,precision,generalization,open    # memory + long-horizon first (user 2026-10-06)
      VIDEO_FS_DEFAULT=5; ACTION_FS_DEFAULT=1; VIDEO_LAYOUT_DEFAULT="" ;;
    SANA_WAM_SM)       # the sparse-memory policy (one decision per chunk: [sink | memory | target], the prefix re-encoded
                       # from the episode's frames); imports policy/SANA_WAM/sana_wam_min
      POLICY=SANA_WAM_SM; RPC_BASE=63700; WS_BASE=65500; SMOKE_RPC=63790; SMOKE_WS=65590
      WORKER_ENV="SANA_EVAL_XPL_POLICY=SANA_WAM_SM"; SYNC_POLICIES="SANA_WAM SANA_WAM_SM"
      DIM_ORDER_DEFAULT=memory,long-horizon,precision,generalization,open    # as SANA_WAM_CAUSAL: the memory policies
      VIDEO_FS_DEFAULT=5; ACTION_FS_DEFAULT=1; VIDEO_LAYOUT_DEFAULT="" ;;
    *) echo "unknown ADAPTER '$1' (SANA_WAM | SANA_MOT | SANA_WAM_CAUSAL | SANA_WAM_SM)" >&2; return 2 ;;
  esac
}
