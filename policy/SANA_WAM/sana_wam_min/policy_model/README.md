# Unified policy kernel surface

This directory is a self-contained, inference-only mirror of the unified
world model's policy branch
(`SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewPolicy`,
factory `..._5B_P1_D36`). It exists so individual model stages can be
profiled and replaced by fused kernels without registry side effects or
imports from the general SANA training stack.

## Boundaries

- Runtime dependencies: Python, PyTorch, and optional xFormers.
- Internal dependencies: relative imports inside this directory only.
- Model dependencies: no imports from `diffusion.*`, `dev.junsongc.*`, or the
  live `dev.rwm.diffusion.*` implementation; needed math is copied with its
  source stated in a comment.
- Integration: the production inference loader is not switched to this model
  until strict checkpoint and end-to-end numerical parity are accepted.

## What is modeled

The 5B P1 unified policy on the policy task: 32 blocks (24 global-bilinear
GDN, 8 full-head-RoPE softmax), pointwise SwiGLU, per-token AdaLN, AttnResV2,
physical-time three-axis MRoPE, and the joint `[video; state; action]` token
sequence.

- Forward dialect: `forward(x, timestep, y, mask=None, **kwargs)`. The robot
  stream rides `data_info['action80' / 'action_mask80' / 'initial_state80' /
  'initial_state_condition_mask80' / 'action_timestep']`; `data_info` must
  carry `rwm_task='policy'`, `view_count`, `model_fps`, and for V>1
  `view_latent_shape` (`[B,V,2]`) and `view_slot_ids`. Explicit `action` /
  `action_mask` keyword arguments are refused with a `TypeError`. One action
  row per source-frame transition is required (`K == (F-1) * 8`).
- Video layout: V=1 is the regular `[B,C,F,H,W]` grid with the regular-grid
  RoPE; V>1 is the padding-free strip `[B,C,F,1,sum(H_v*W_v)]` reordered to
  view-major tokens, with the spatial RoPE layout selected by
  `multiview_spatial_rope_layout` (`semantic_2x2` places each view on its
  TL/TR/BL/BR tile of `multiview_spatial_rope_tile_shape`, centered;
  `local_reset` restarts the grid per view).
- Text: `y` is `[B,G,1,L,C]` with `mask` `[B,G,L]`, `G = V + 1` token
  groups (one per view, one for the robot tail). Each group is
  cross-attended by its own contiguous query span (`prompt_group_spans`).
- Output heads: `final_layer` (video velocity) and `action_head`
  (LayerNorm + per-token timestep AdaLN + Linear to 80D, masked by
  `action_mask80`).

## Condition-causal form (rwm/zekai-merge f6a257d4a)

`PolicyConfig.condition_causal` swaps every block's self-attention for its
condition-causal subclass (`condition_chunks.py`, `ConditionChunkGatedDeltaNet`,
`ConditionChunkGatedSoftmaxAttention`): the condition chunk (latent frame 0 of
every view and the state row) never reads the prediction chunk (later frames
and the action rows). The GDN blocks gain the recurrence tensors `gate_proj`,
`A_log`, `dt_bias` and `recall_gate`, which is how a checkpoint names its form.
`gdn_output_normalizer` (`model.memory_gdn_output_normalizer`) and
`state_cross_attention` (`model.extra.state_cross_attention`: the state as one
caption token of the last text group, no state row) exist only in this form.
`forward(..., condition_prefix=(ConditionPrefix, PREFILL | TARGET))` runs one
chunk alone against the other's cached reads (the prefix cache); bit parity with
the live class: `tests/test_condition_causal_live_parity.py`.

## What is not modeled

- The camera channel: `plucker_embed` and the CaPE softmax QKVO transform.
  A batch with the camera gate on (`first_frame_plucker` present or
  `camera_conditioning_enabled` true) is refused.
- The packed flash-varlen grouped-text path (`set_packed_grouped_text`).
- CFG null embeddings (`y_embedder.y_embedding`) and checkpoint wrapper
  parsing (`state_dict` / `module.` prefixes).
- Dual AttnRes routing (`use_dual_attn_res_routing`) and AttnRes time
  conditioning (`use_time_conditioning`).
- Deploy fused-kernel switches (`set_fused_*`, blockglue driver, static
  AttnRes buffers), the fps-verdict capture cache, and the training-time
  RTC clean-prefix arm (`rtc_prefix_rows`).
- Kernel-variant classes: `*FP32Rope`, `*FlashRope`, and `SwiGLUFusedAct`
  names are accepted as aliases (same state_dict) and run the fp64-rope base
  math; `SwiGLUPacked*` (different keys) is refused.

## Checkpoint contract

`checkpoint.py` removes exactly three tensors after validating their
shapes — `pos_embed`, `y_embedder.y_embedding`, and
`plucker_embed.weight` — and then strict-loads every remaining tensor.
`tests/test_policy_mirror_parity.py` pins strict load plus bitwise `x` and
`action_pred` parity against a tiny live policy.

## Profiling seams

- `embeddings.py`: video, timestep, caption, state/action embedding and both
  output heads.
- `rope.py` and `geometry.py`: physical-time MRoPE, the semantic 2x2 view
  layout, and the multi-view token layout.
- `attention.py`: GDN, dense softmax attention, and grouped text
  cross-attention.
- `feedforward.py`: checkpoint-compatible baseline SwiGLU.
- `block.py`: per-token AdaLN attention and MLP sublayers.
- `attnres.py`: inference-only depth aggregation.
- `model.py`: joint sequence assembly, trunk driver, and output split.
