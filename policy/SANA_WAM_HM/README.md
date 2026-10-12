# SANA_WAM_HM

**Contributor:** NVIDIA (zekail) | **Paper:** SANA-WAM (to appear) | **arXiv:** pending | **Original code:** Sana (NVlabs), internal RoboDojo hierarchical-memory policy line (branch `rwm/sparse-memory`)

`SANA_WAM_HM` adapts the SANA **hierarchical-memory policy** to XPolicyLab. The network is the chunk-causal class `SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewCausal_5B_P1_D36` (the 32-layer hybrid trunk of the bidirectional SANA_WAM policy: 24 Gated DeltaNet blocks, 8 softmax blocks at 3, 7, ..., 31) trained in the hierarchical-memory form of Sana `rwm/sparse-memory`: every training sample is ONE noisy chunk [obs | target] + [state | 32 actions] at an anchor row t, preceded by memory latents of the episode's own camera canvases, and **each block kind reads one memory**:

* the **dense** memory: the stride-4 rows of the whole history, encoded as one causal-VAE sequence: L_0 (the first frame alone), then one latent per 32-row chunk;
* the **sparse** memory: the first frame plus 16 rows sampled uniformly over the history [r0 + 1, t], encoded as one causal sequence: A_0 (the first frame) and two summary latents of 8 rows each.

`model.extra.chunk_causal_policy.hierarchical_memory: {gdn: dense|sparse, softmax: dense|sparse, sparse_frames: 16}` picks which memory the GDN blocks and the softmax blocks read; every block computes on (its memory + the noisy chunk), a GDN block as a chunk-wise scan, a softmax block as a prefix-causal window, and the other memory's tokens pass it unchanged (block-level routing). Two recipes exist: `hm_dense_sparse` (GDN dense, softmax sparse) and `hm_sparse_sparse` (both sparse).

Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=ee`; the EEF-only absolute `robot_base_eef` line on the one-view `sana_pixel` 320x512 canvas, `rope: aligned`, `memory_content: video_only`, video frame stride 4, no AttnRes time conditioning. The model runs **in-process** through the vendored, Sana-free package `sana_wam_hm/`, a mirror of Sana `rwm/sparse-memory` (checked against `e8dde6b56`): `dev/rwm/diffusion/model/utils/hierarchical_memory.py`, the hierarchical-memory paths of `sana_qwennext_policy_causal.py` (`memory_layout`, `_memory_core`, `_run_causal_layers`, `_layer_step`) and `layers/chunk_causal_{gdn,softmax,block}.py`. It sits on top of the sibling adapter's runtime `policy/SANA_WAM/sana_wam_min/` (trunk modules, Flow-Euler sampler, LTX-2.3 causal VAE encoder, Gemma-2-2B text conditioning, the sana_pixel canvas, Robot80 normalization, EEF kinematics, RoboDojo codecs), which is not modified. Sana has **no deploy session** for this mode yet (its plan lists the inference side as requirements only); `sana_wam_hm/session.py` implements those requirements. This is an **eval-only** submission: training and data conversion live in the Sana repository.

Checkpoints of the per-chunk observation protocol (no `hierarchical_memory` block) are served by `SANA_WAM_CAUSAL`, which refuses hierarchical-memory checkpoints, and vice versa.

## Installation

Same environment as SANA_WAM / SANA_WAM_CAUSAL (`bash install.sh` installs the pinned stack into the active environment). `policy/SANA_WAM/` must sit next to this directory (`sana_wam_hm/shared.py` puts its `sana_wam_min` on `sys.path`). `deploy.yml` points `text_encoder_path` / `vae_path` at copies of `google/gemma-2-2b-it` and `Efficient-Large-Model/LTX-2.3-Diffusers`; null reads `SANA_WAM_TEXT_ENCODER_PATH` / `SANA_WAM_VAE_PATH`.

## Data Processing / Training

Not supported here (eval-only). Training is Sana's `train_video_rwm_policy_causal.py` (`hierarchical_memory_train_step`) on the frame-anchor samples of `CausalRoboDojoSFTDataset` (every row t of every RoboDojo span is one sample with its whole history), recipes `dev/rwm/configs/causal/robot_pretrain/causal_robodojo_pantheon14k_eefabs_f33fps8_sana_pixel_aligned_hm_{dense_sparse,sparse_sparse}_videoonly.yaml` (donor: the pantheon14k sana_pixel 320x512 SFT checkpoint; normalization pin `1fe3b7e7`; 10 epochs; the dense recipe reads its dense latents from a prebuilt store, which the deploy does not need). `process_data.sh` / `train.sh` print a notice and exit 0.

## Model Assets

An HF-style checkpoint dir of a hierarchical-memory run: `config.yaml` (the trainer dump of an `hm_*` recipe) + `model/pytorch_model_fsdp.bin` [+ `normalization/*.json`]. The state dict has the causal line's schema (the bidirectional policy's tensors plus every GDN block's recurrence `gate_proj`, `A_log`, `dt_bias`, `recall_gate`), loaded strictly (`sana_wam_hm/checkpoint.py`). `allow_donor_checkpoint: true` serves an SFT checkpoint as the step-0 model (recurrence initialized as the trainer does) for pipeline checks only.

## The deploy cycle

Decision c anchors at tick t = 32c (tick 0 = the episode's first observation = training's first paired row r0). Every `update_obs` keeps that tick's canvas, quantized to uint8 exactly as the training payload (`frames_to_uint8`, then `pixels_from_uint8` into the encoder): the sparse memory samples arbitrary rows of the whole history, so every tick is kept (~0.5 MB per tick, ~0.6 GB for a 1,100-step episode). Every `get_action`:

1. (c >= 1) the dense stream (dense recipes) encodes the executed chunk's 8 stride-4 canvases (ticks 32(c-1)+4 .. 32c) as its next latent L_c, incrementally on its own causal-VAE feature cache (bit for bit the one-shot encode of the whole stream); the previous prediction is scored against the latent training regressed it to (L_c, or for sparse-only recipes the target encoded after the decision's sparse rows) -> `world_model_rel_rmse`;
2. the row-t canvas is encoded alone as the observation;
3. (sparse recipes) the sparse memory [tick 0, s_0..s_15] is encoded as one causal sequence -> [A_0, A_1, A_2];
4. the chunk is denoised (10 Euler steps, video CFG 6 / action CFG 1 by default) in one of two forms (`deploy_form`):
   * `streaming` (the default, every memory choice): every memory latent is **committed once** through the blocks that read its memory -- a dense latent when it is produced (the dense memory's GDN state and/or softmax K/V grow over the episode), the sparse memory once per decision (its GDN state and/or softmax K/V start afresh, since its sampled rows move with t); cond and uncond text streams keep separate memories (the caption enters every block through cross-attention). Every denoising step then runs the 353-token noisy chunk alone, each block reading its memory's committed GDN state or softmax K/V. This is exact: a memory token never reads the noisy chunk (the GDN scan reads memory first, the softmax window is prefix-causal), its t is 0, and without AttnRes time conditioning its depth mixing is its own. With a dense memory every position sits on the episode clock (rows from r0: L_j at 32j, the sparse latents at their sampled rows, obs at 32c, the robot rows from step 32c), the training geometry shifted by t - r0 (equal up to floating-point round-off); without one, on the training clock exactly. (Sana's own deploy forms instead commit the dense latents and run the sparse memory inside every step's `generate` window; the mirror keeps them for the parity tests.)
   * `full` (reference): every step runs the training forward of the decision's sample [L_0..L_c | A_0..A_2 | obs | target] + rows on the training clock -- the training computation itself; with a dense memory its cost grows with c.

`reset` (sent twice per episode by RoboDojo) drops the memory and the canvases.

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
| `deploy_form` | null | `streaming` (commit-once, the default) / `full` (the training forward per step, reference) |
| `max_chunks_per_episode` | null | refuse longer episodes; training covers K <= 47 history chunks |
| `log_world_model_error` | true | per-decision `world_model_rel_rmse` (sparse-only recipes pay one extra 8-frame stream encode for it) |
| `view_resize` | null (= stretch) | as in SANA_WAM |
| `diffusion_seed_base` | 20261012 | noise seed = sha256(domain, base, eval seed, episode index, decision index) |
| `eef_pose_check` / `urdf_path` / `robot_root_poses` | true / null / null | the SANA_WAM EEF kinematics settings |

## Validation

* `tests/test_memory_geometry.py`: every geometry function (rows, sparse sampling, frame layout, offsets, block subsets and bounds, the three forms' layouts) equal to live Sana's for t - r0 up to 3,000 and all four memory choices; the plan's worked examples; the episode clock = the training clock shifted by t - r0.
* `tests/test_live_parity.py` (tiny fp64 twins of the live class and the mirror, identical random weights): the training forward of all four memory choices equal to the live one (rtol 1e-9); Sana's `forward_chunk` commit / generate on the episode clock equal to the mirror's (outputs and committed GDN states); the commit-once forms equal to the training forward for all four memory choices (exactly on the training clock, within the episode clock's round-off with a dense memory: max 3e-10, flat in c, the GDN's complex64 RoPE on shifted positions) and to Sana's generate form on the same clock; a whole decision cycle of the session (Euler loop, CFG, both forms, every memory choice) equal to the same loop driven by the live training forward (training clock 1e-9; episode clock within a few float32 ulps, the diffusers Euler step running in float32).
* `tests/test_vae_stream.py`: on a tiny random LTX-2.3 encoder, the dense / sparse streams fed 1 frame then 8 per call equal the one-shot encode bit for bit, with no crosstalk between streams and the one-shot encodes in between; the pixel quantization is Sana's.
* `tests/test_contract.py`: both recipes resolve; 15 refusals; SANA_WAM_CAUSAL refuses both recipes.
* `tests/test_adapter_state_machine.py`: the decision cycle on a fake runtime (ticks of every encode, order, double reset, refusals).
* `tools/first_light.py`: the real-dimension model with random weights on a GPU (bf16 autocast) in both forms, and the real LTX-2.3 encoder's streams vs its one-shot encode on synthetic 320x512 canvases. On an RTX 5090: `hm_dense_sparse` streaming 0.075 s per dense commit and ~0.95 s per decision (10 steps x CFG, the sparse commit included), 8.8 GiB peak; full 1.42 s at c = 0 growing to 2.47 s at c = 5; `hm_sparse_sparse` streaming ~1.04 s per decision vs 1.73 s full; streaming vs full actions within bf16 round-off (0.3 % relative RMSE, flat in c); both VAE streams maxdiff 0 vs the one-shot encode.

**Block AttnRes and routing.** In training a memory token passes a block that does not read its memory with a ZERO delta (the scatter): under `hm_dense_sparse` the GDN blocks see the sparse latents A with zero deltas and the softmax blocks see the dense latents L with zero deltas (`sparse_dense` mirrored); under `hm_sparse_sparse` (and `dense_dense`) every memory token is in every block. A zero delta opens the AttnRes group's partial, and a zero partial is an extra depth source (key_norm(0) = 0, logit 0), so "skip the block" and "zero delta" mix differently. The mirror applies the zero delta (a commit's skipped block opens the partial when it is still empty); `test_memory_tokens_see_zero_deltas_in_the_blocks_that_do_not_read_them` checks, block by block, that the commits reproduce the training forward's memory-token inputs, on the real layout (GDN blocks open every AttnRes group) and on one where softmax blocks open the groups (a mutation back to skipping fails exactly the mixed memory choices). Sana's commit form skips (`continue`): on its real trunk every AttnRes group of 8 opens with a GDN block, so a skipped softmax block always finds the partial open and the result is exact (its deploy chain equals its training forward to 3e-16), but with softmax blocks opening the groups its chain diverges from its training forward (2.5e-7 in fp64 on the tiny model).

No hierarchical-memory checkpoint has been evaluated through this adapter yet.

## Files

`model.py` (XPolicyLab entry: ticks, canvases, the decision cycle) | `deploy.yml` | `sana_wam_hm/{memory,contract,layers,model,session,vae_stream,runtime,checkpoint,shared}.py` | `tools/first_light.py` | `tests/` (`SANA_HM_REPO` points the live tests at an rwm/sparse-memory checkout; run the live files apart from other Sana checkouts).
