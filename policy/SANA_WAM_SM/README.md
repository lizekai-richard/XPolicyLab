# SANA_WAM_SM

**Contributor:** NVIDIA (zekail) | **Paper:** SANA-WAM (to appear) | **arXiv:** pending | **Original code:** Sana (NVlabs), internal RoboDojo sparse-memory policy line (branch `rwm/sparse-memory`)

`SANA_WAM_SM` adapts the SANA **sparse-memory policy** to XPolicyLab. The network is the chunk-causal class `SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewCausal_5B_P1_D36` (the 32-layer hybrid trunk of the bidirectional SANA_WAM policy: 24 Gated DeltaNet blocks, 8 softmax blocks at 3, 7, ..., 31) trained in the sparse-memory form of Sana `rwm/sparse-memory` (`3f6f8daa5` on, which replaced the hierarchical memory). A training sample anchored at the observation row t is ONE causal-VAE encode of the episode's own camera canvases, the sink (the span's first row r_0), the memory rows and the 8 target rows t + 4i, into the latents **[S | M_1..M_n | T]**, followed by the 33 robot rows [state at t | 32 actions]. `model.extra.chunk_causal_policy.memory_type` picks the memory rows:

* `strided` with `memory_stride` s: t, t - s, t - 2s, ... above r_0, led by copies of r_0 up to a multiple of 8, so n = ceil(ceil((t - r_0) / s) / 8) grows with t (recipe `..._sana_pixel_aligned_sparse_memory`: s = 25, rungs up to n = 8);
* `fixed` with `memory_frames` m (a multiple of 8): m rows sampled uniformly over [r_0 + 1, t], led by copies of r_0 while the span is shorter, so n = m / 8 at every row (recipe `..._vanilla_pantheon14k_..._sparse_memory_fixed`: m = 16, n = 2, the hierarchical memory's noobs rows).

Either memory's last row is t once the span has a row past r_0: the current frame reaches the policy through the memory, there is no observation latent. **Every block (GDN and softmax) runs on the whole sequence**, one chunk per sink / memory latent and the target chunk [T | state | actions] last (no routing); RoPE counts latents (latent i at frame i, the target at n + 1) and starts the robot rows at action step 32n.

Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=ee`; the EEF-only absolute `robot_base_eef` line on the one-view `sana_pixel` 320x512 canvas (160 tokens per latent), `rope: aligned`, video frame stride 4, no AttnRes time conditioning, the causal LTX encoder. The model runs **in-process** through the vendored, Sana-free package `sana_wam_sm/`, a mirror of Sana `rwm/sparse-memory` checked against `3fa3e9689`: `dev/rwm/diffusion/model/utils/sparse_memory.py`, the sparse-memory paths of `sana_qwennext_policy_causal.py` (`_forward_multiview`, `_sparse_memory_layers`, `_run_causal_layers`, `_layer_step`, `forward_memory_prefix`, `forward_memory_target`), `layers/chunk_causal_{gdn,softmax,block}.py` and the deploy session `dev/rwm/deploy/causal/sparse_memory_session.py`. It sits on top of the sibling adapter's runtime `policy/SANA_WAM/sana_wam_min/` (trunk modules, Flow-Euler sampler, LTX-2.3 causal VAE encoder, Gemma-2-2B text conditioning, the sana_pixel canvas, Robot80 normalization, EEF kinematics, RoboDojo codecs), which is not modified. This is an **eval-only** submission: training and data conversion live in the Sana repository.

This directory was `SANA_WAM_HM` (the hierarchical memory: routed dense / sparse memories, `hierarchical_memory` / `memory_uniform` keys, `chunk_observation`) up to XPolicyLab `d57a617`; Sana removed that design and so did this adapter. Hierarchical-memory checkpoints (e.g. `hm_uniform_noobs`) are refused here with that pointer; serve them from a checkout at `d57a617`. Checkpoints without a `memory_type` are `SANA_WAM_CAUSAL`'s, which refuses sparse-memory checkpoints, and vice versa.

## Installation

Same environment as SANA_WAM / SANA_WAM_CAUSAL (`bash install.sh` installs the pinned stack into the active environment). `policy/SANA_WAM/` must sit next to this directory (`sana_wam_sm/shared.py` puts its `sana_wam_min` on `sys.path`). `deploy.yml` points `text_encoder_path` / `vae_path` at copies of `google/gemma-2-2b-it` and `Efficient-Large-Model/LTX-2.3-Diffusers`; null reads `SANA_WAM_TEXT_ENCODER_PATH` / `SANA_WAM_VAE_PATH`.

## Data Processing / Training

Not supported here (eval-only). Training is Sana's `train_video_rwm_policy_causal.py` (`sparse_memory_train_step`) on the frame-anchor samples of `CausalRoboDojoSFTDataset` (`SparseMemoryDatasetMixin`: every row t of every RoboDojo span is one sample, with `robot_sft.tail_windows` {freeze, min_rows 5} the rows up to E - 5 too, bucketed by n), recipes `dev/rwm/configs/causal/robot_pretrain/causal_robodojo_pantheon14k_eefabs_f33fps8_sana_pixel_aligned_sparse_memory[_cached].yaml` (strided) and `causal_robodojo_vanilla_pantheon14k_eefabs_f33fps8_sana_pixel_aligned_sparse_memory_fixed[_cached].yaml` (fixed); the `_cached` twins read the sequence latents from a prebuilt store (`sparse_memory_latent_store`), which the deploy does not need. Normalization pin `1fe3b7e7`. `process_data.sh` / `train.sh` print a notice and exit 0.

## Model Assets

An HF-style checkpoint dir of a sparse-memory run: `config.yaml` (the trainer dump of a sparse-memory recipe) + `model/pytorch_model_fsdp.bin` [+ `normalization/*.json`]. The state dict has the causal line's schema (the bidirectional policy's tensors plus every GDN block's recurrence `gate_proj`, `A_log`, `dt_bias`, `recall_gate`), loaded strictly (`sana_wam_sm/checkpoint.py`). `allow_donor_checkpoint: true` serves an SFT checkpoint as the step-0 model (recurrence initialized as the trainer does) for pipeline checks only.

## The deploy cycle

Decision c anchors at tick t = 32c (tick 0 = the episode's first observation = the span's first row r_0). Every `update_obs` keeps that tick's canvas, quantized to uint8 exactly as the training payload (`frames_to_uint8`, then `pixels_from_uint8` into the encoder): the memory grid moves with t, so a decision may read any earlier frame and every tick is kept (Sana `SparseMemoryEpisode`; ~0.5 MB per tick). Every `get_action`:

1. (c >= 1) the previous decision is scored: its commanded vs reached EE poses (`cmd_vs_reached`), and its predicted target latent vs the target latent of its training sample, the causal encode of its `sequence_rows` now that the 8 target frames exist (`world_model_rel_rmse`);
2. the prefix rows [tick 0 | the memory rows] (`sparse_memory.prefix_rows`, Sana `SparseMemoryEpisode.prefix_rows`) are encoded in ONE causal pass into [S | M_1..M_n] (Sana `encode_prefix`: the causal encoder gives a frame group nothing of the later ones, so the prefix alone yields the training's sink and memory latents);
3. the target chunk is sampled (10 Euler steps, video CFG 6 / action CFG 1 by default, the video and action streams on their own flow-match schedules) in one of two forms (`deploy_form`):
   * `cached` (the default, Sana `SparseMemoryPolicySession.decide`): the clean prefix runs ONCE per text stream (cond and uncond keep separate prefixes: the caption enters every block through cross-attention) and fills every layer's cache, a GDN layer its state after M_n, a softmax layer the prefix's rotated K and V; every denoising step then runs the 193-token target chunk alone against them, T at RoPE frame n + 1 and the rows from step 32n, the training sequence's last chunk token for token. Exact up to round-off: the prefix never reads the target (the GDN scan reads its chunks in order, the softmax attention is chunk-causal), its t is 0, and without AttnRes time conditioning its depth mixing is its own;
   * `uncached` (reference, Sana `decide_uncached`): every step runs the whole training sequence [S | M_1..M_n | T] + rows.

Two decisions share nothing (the memory rows move with t), so nothing is streamed or committed across decisions. A strided memory past the recipe's top rung (n > 8: t > 1,600 rows) warns once. `reset` (sent twice per episode by RoboDojo) drops the canvases.

## Configuration

| key | default | meaning |
|---|---|---|
| `checkpoint_dir` | null | the run dir above; null = XPolicyLab resolution of `ckpt_name` |
| `allow_donor_checkpoint` | false | serve an SFT checkpoint as step 0 (pipeline checks only) |
| `normalization_path` / `normalization_sha256` | null / null | explicit artifact and/or sha pin |
| `text_encoder_path` / `vae_path` | null | local Gemma-2-2B / LTX-2.3-Diffusers copies |
| `sampling_steps` | 10 | Euler steps per decision |
| `cfg_scale` / `video_cfg_scale` / `action_cfg_scale` | 6.0 / null / 1.0 | per-stream text CFG (one unconditional forward per step either way) |
| `flow_shift` / `action_flow_shift` | null / null | null = the yaml's inference shifts (5 / 1) |
| `deploy_form` | null | `cached` (prefix once, target per step; the default) / `uncached` (the training sequence per step, reference) |
| `max_chunks_per_episode` | null | refuse longer episodes |
| `log_world_model_error` | true | per-decision `world_model_rel_rmse` (one more causal encode of 1 + 8n + 8 canvases per decision) |
| `view_resize` | null (= stretch) | as in SANA_WAM |
| `diffusion_seed_base` | 20261012 | noise seed = sha256(domain, base, eval seed, episode index, decision index) |
| `eef_pose_check` / `urdf_path` / `robot_root_poses` | true / null / null | the SANA_WAM EEF kinematics settings |

The ready line prints `protocol=sparse_memory memory_type=... memory_frames=... | memory_stride=... trained_memory_latents=... deploy_form=...` next to the sampling knobs, `gdn_recurrence` and `normalization_sha256`.

## Validation

Against Sana `rwm/sparse-memory` `3fa3e9689` (`SANA_SM_REPO`): 102 passed, 0 skipped.

* `tests/test_sparse_memory_rows.py`: every row function (`memory_from_policy` and its refusals, `fixed_rows`, `strided_rows`, `strided_latents`, `memory_rows`, `memory_latents`, `target_rows` with the tail freeze, `sequence_rows`, `sampler_rung`, `chunk_bounds`, `pixels_from_uint8`, `video_timesteps`) equal to live Sana's for t - r_0 up to 1,600 and six memories; the episode's prefix rows equal to Sana's `SparseMemoryEpisode`; the three forwards' layouts split the sequence exactly; the deploy clock reads the current frame through the memory.
* `tests/test_live_parity.py` (tiny fp64 twins of the live class and the mirror, identical random weights): the training forward equal to the live one (both memory types, n = 0, 1, 3, rtol 1e-9); the prefix + target forms equal to Sana's `forward_memory_prefix` / `forward_memory_target`, outputs and caches (every GDN (S, z), every softmax K / V); prefix + target equal to the sequence's last chunk; the sink and memory latents never read the last chunk; a whole decision (Euler loop, CFG off / video 2 + action 1.5 / video 6) in both forms equal to Sana's `decide` / `decide_uncached` on the same generator.
* `tests/test_vae_encode.py`: on a tiny random LTX-2.3 encoder, the prefix encoded alone is the first n + 1 latents of the whole sample's encode (bit for bit), the target latent depends on its memory context, the encode equals Sana's `encode_prefix`; the pixel quantization is Sana's.
* `tests/test_contract.py`: the four recipes resolve (the latent-store twins to the same contract); the keys Sana's sparse-memory forms never read are ignored; the refusals (hierarchical-memory keys and the released `hm_uniform_noobs` config with the `d57a617` pointer, no `memory_type`, Sana's own `memory_from_policy` refusals, a fixed memory off the rung ladder, a non-positive stride, independent rope, time conditioning, ...); SANA_WAM_CAUSAL refuses every recipe.
* `tests/test_adapter_state_machine.py`: the decision cycle on a fake runtime for both memory types (the ticks of every prefix and world-model encode, the order, the double reset, the refusals, the strided warning past the trained rung).
* `tools/first_light.py`: the real-dimension model with random weights on a GPU (bf16 autocast) in both forms, and the real LTX-2.3 encoder's prefix encode vs the whole sample's encode on synthetic 320x512 canvases (not run yet for this adapter).

No sparse-memory checkpoint has been evaluated through this adapter yet.

## Files

`model.py` (XPolicyLab entry: ticks, canvases, the decision cycle) | `deploy.yml` | `sana_wam_sm/{sparse_memory,contract,layers,model,session,vae_encode,runtime,checkpoint,shared}.py` | `tools/first_light.py` | `tests/` (`SANA_SM_REPO` points the live tests at an rwm/sparse-memory checkout; run the live files apart from other Sana checkouts).
