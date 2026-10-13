# SANA_WAM_CAUSAL

**Contributor:** NVIDIA (zekail) | **Paper:** SANA-WAM (to appear) | **arXiv:** pending | **Original code:** Sana (NVlabs), internal RoboDojo chunk-causal policy line (branch `rwm/zekai-merge`)

`SANA_WAM_CAUSAL` adapts the SANA **chunk-causal policy** `SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewCausal_5B_P1_D36` to XPolicyLab. It is the 32-layer hybrid trunk of the bidirectional SANA_WAM policy made causal over action chunks. The 24 Gated DeltaNet layers carry a chunk-level delta-rule recurrence from chunk to chunk. The 8 softmax layers (3, 7, ..., 31) attend over a sliding window of the newest committed chunks. The model trains with **two-stream teacher forcing** and is deployed the same way: every chunk is generated from its own observation (the chunk-start frame) against the memory, and after it executes, its observed outcome (camera frames, the commands actually sent, the chunk-start state) is committed into memory before the next chunk is generated. This is the **per-chunk observation protocol** of Sana `61aad2d58` / `3a6b2bf07`; checkpoints of the earlier chunk-0-only observation layout (the vanilla52k line) are refused, and the adapter that served them is archived in this repository's history (commit `69cfb8e`). Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=ee`. The checkpoint must be the EEF-only absolute `robot_base_eef` line on the one-view `sana_pixel` 320x512 canvas with `rope: aligned` and strided video. The model runs **in-process** through the vendored, Sana-free package `sana_wam_causal/`, a mirror of Sana `rwm/zekai-merge` (checked against `9ee9c2d9d`): `sana_qwennext_policy_causal.py`, `layers/chunk_causal_{gdn,softmax,block}.py`, and the inference subset of `deploy/causal/session.py`. It sits on top of the sibling adapter's runtime `policy/SANA_WAM/sana_wam_min/`, which supplies the bidirectional trunk modules, the Flow-Euler sampler, the LTX-2.3 causal VAE encoder, Gemma-2-2B text conditioning, the sana_pixel canvas, Robot80 normalization, the EEF kinematics and the RoboDojo codecs. `policy/SANA_WAM` itself is not modified. This is an **eval-only** submission: training and data conversion live in the Sana repository, and `process_data.sh` / `train.sh` print a notice and exit 0.

Shared conventions (argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE`) are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

SANA_WAM_CAUSAL runs in the **same environment as SANA_WAM**. `install.sh` installs the same pinned stack into the active environment:

```bash
conda create -n sana_wam python=3.11 && conda activate sana_wam
cd XPolicyLab/policy/SANA_WAM_CAUSAL && bash install.sh
# = torch 2.9.1 / torchvision 0.24.1 (cu128; SANA_WAM_SKIP_TORCH=1 keeps an existing CUDA torch), diffusers>=0.38,
#   transformers>=4.46,<5, safetensors, numpy, pillow, pyyaml, huggingface_hub, and XPolicyLab itself (pip install -e).
```

`policy/SANA_WAM/` must sit next to this directory, because `sana_wam_causal` imports its `sana_wam_min` package (`sana_wam_causal/shared.py` puts it on `sys.path`; the import fails loudly otherwise). No `flash_attn` is needed: Sana's causal cross-attention hard-imports it, whereas this port uses the bidirectional mirror's SDPA cross-attention. The text encoder and VAE weights are not part of the checkpoint. `deploy.yml` points `text_encoder_path` / `vae_path` at copies of `google/gemma-2-2b-it` and `Efficient-Large-Model/LTX-2.3-Diffusers`; when the keys are null, `SANA_WAM_TEXT_ENCODER_PATH` / `SANA_WAM_VAE_PATH` are read.

## Data Processing

Not supported in this adapter (eval-only). The training dataset is Sana's `CausalRoboDojoSFTDataset`. Every 25 Hz row of a RoboDojo episode starts a window of up to 48 chunks; chunk c of a window is the bidirectional SFT sample at its start + 32c, whose own encode supplies both the chunk's observation latent and its target latent. `bash process_data.sh ...` prints a notice and exits 0.

## Training

Not supported in this adapter (eval-only). Training lives on Sana branch `rwm/zekai-merge`. The recipes warm-start from a bidirectional SFT checkpoint (the *donor*) and add the GDN recurrence parameters:

| recipe (`dev/rwm/train_video_scripts/dfw/`) | yaml (`dev/rwm/configs/causal/robot_pretrain/`) | donor | served here |
|---|---|---|---|
| `run_causal_robodojo_pantheon14k_eefabs_f33fps8_sana_pixel_aligned_m48n24.sbatch` | `causal_robodojo_pantheon14k_eefabs_f33fps8_sana_pixel_aligned_m48n24.yaml` | pantheon14k sana_pixel 320x512 stretch SFT | yes (`memory_content` video_and_action; the campaign's config has no key, which reads the same) |
| `run_causal_robodojo_pantheon14k_eefabs_f33fps8_sana_pixel_aligned_m48n24_videoonly.sbatch` | `causal_robodojo_pantheon14k_eefabs_f33fps8_sana_pixel_aligned_m48n24_videoonly.yaml` | the same | yes (`memory_content: video_only`) |

The vanilla52k / vanilla34k recipes trained the retired chunk-0-only observation layout (`obs_in_first_chunk: true`) and are refused.

Recipe `m48n24`:

- C = 32 action rows per chunk (1.28 s at 25 Hz).
- `video_fps` 8, i.e. video frame stride 4 and one latent frame per chunk.
- The two-stream canvas [clean targets 0..L-2 | noisy chunks 0..L-1], every noisy chunk [its observation latent (t = 0, no loss) | its targets]; training windows of up to 48 chunks (`window_repeats` enters the 35-47-chunk windows three times per epoch); softmax window W = 24 chunks.
- `memory_content`: what a clean (memory) chunk carries besides its target latent -- `video_and_action` (its anchor state and actions too) or `video_only`.
- Flow shifts: video 5, action 1.

`bash train.sh ...` prints a notice and exits 0.

## Model Assets

Checkpoint layout (the accelerate FSDP full state dict of the causal policy, as the trainer saves it):

```text
<ckpt_dir>/
├── config.yaml                                   # the training yaml: model.model = the causal class above,
│                                                 #   model.extra.chunk_causal_policy (memory_content), data.type CausalRoboDojoSFTDataset
├── model/pytorch_model_fsdp.bin                  # fp32 full state dict: the bidirectional trunk's keys + per GDN block
│                                                 #   attn.gate_proj.{weight,bias}, attn.A_log, attn.dt_bias (attn.recall_gate)
└── normalization/robodojo_arx_x5_model_fps_25_f33_normalization.json   # the f33 Robot80 statistics the yaml pins
```

`config.yaml` may also sit one or two levels above the checkpoint dir, as in the `checkpoints/epoch_X_step_Y/` layout of a training run. `ckpt_name` resolves through `XPolicyLab.utils.checkpoint_resolver.resolve_checkpoint_root` exactly as for SANA_WAM: explicit `checkpoint_dir` / `checkpoint_path` / `ckpt_dir` / `model_dir` first, then `ckpt_name` as a path, then `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`, then `checkpoints/<ckpt_name>/`.

**Contract.** `sana_wam_causal/contract.py` reads everything from the yaml and refuses anything it does not implement, before the weights are read:

- **Geometry.** Actions per chunk come from `model.extra.chunk_causal_policy.actions_per_chunk` at the model fps; the dataset's chunk must agree. The tier is `data.extra.robot_sft.tier_num_frames = C + 1` (33). The video stride comes from `video_fps`. The window is `sliding_window_chunks`. `memory_content` is `video_and_action` (also when the key is absent, as Sana reads it) or `video_only`.
- **Refused contracts:** any `obs_in_first_chunk` key (a retired observation layout: the vanilla52k line); a bidirectional class or dataset; a strip or multiview layout; `sana_pixel_pad: masked`; `rope` other than `aligned`; `use_time_conditioning: true`; dual AttnRes routing.

The non-model half (canvas, prompt, VAE encode, normalization, action denormalization) is SANA_WAM's `PolicyInferenceSession`, built on the yaml's **bidirectional view**: the same yaml with the SFT policy class and the 33-frame tier. It resolves exactly as it would for the donor checkpoint.

**Strict load** (`sana_wam_causal/checkpoint.py`):

- The state dict must carry the donor's trunk, the state token (`state_embed`), and the softmax placement of the config.
- The buffers that SANA_WAM's loader strips are dropped after a shape check. Any other unexpected or missing key is refused.
- The GDN recurrence parameters must be present for **every** GDN block. A checkpoint missing them for only some blocks is refused. `recall_gate` is a buffer the forward never reads and may be absent.

**Normalization** resolves as in SANA_WAM, in this order:

1. an explicit `normalization_path`;
2. `<ckpt_dir>/normalization/<f33 name>`.

The artifact must be the one the yaml pins in `data.extra.robot_sft.normalization_sha256`; for the m48n24 recipes that is `1fe3b7e7...`, the same statistics as the sana_pixel donors' `9f8b98ed...`. EEF-only absolute statistics:

- xyz: q01/q99 of both arms pooled;
- rot6d: the fixed [-1, 1] range (identity);
- gripper closedness: [0, 1] mapped to [-1, 1].

The action EEF slots share the state statistics.

**Step-0 pipeline checks.** An SFT checkpoint has no recurrence parameters. With `allow_donor_checkpoint: true`, a dir whose `config.yaml` is the causal yaml and whose `model/` + `normalization/` are the donor's serves the donor as the causal policy at step 0. The recurrence parameters are initialized as the causal trainer initializes them on its first load: `gate_proj` zero, and `dt_bias` / `A_log` from the yaml's `dt_bias_init` / `a_log_init` (-5 / 0 for m48n24). This gives a decay of about 0.993 per chunk. The ready line then says `gdn_recurrence=initialized (SFT donor, causal step 0)`. Chunk 0 of such a model is the donor's single window exactly; later chunks read an untrained memory. Use this for wiring and latency checks only, never as an evaluation.

## Evaluation

```bash
cd XPolicyLab/policy/SANA_WAM_CAUSAL
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# offline wiring check, no simulator (needs a GPU for the real model):
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls causal_pantheon14k_m48n24_stepNNNNN arx_x5 ee 0 0 0 sana_wam base
# same, exercising the server-side image decode path:
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh RoboDojo stack_bowls causal_pantheon14k_m48n24_stepNNNNN arx_x5 ee 0 0 0 sana_wam base
# simulator, one GPU for both (the policy peaks at ~18 GiB on an RTX A6000 with the softmax window full):
bash eval.sh RoboDojo insert_key causal_pantheon14k_m48n24_stepNNNNN arx_x5 ee 0 0 0 sana_wam robodojo
```

The debug client stops every episode after 20 actions, so it exercises chunk 0 and `reset` only; the commit path needs the simulator. Its fake `*_ee_pose` (all ones) makes the once-per-episode pose check report a gap of about 1.5-1.9 m and warn, which is expected there.

The same conda-activation (`NVCC_PREPEND_FLAGS`) and `pyyaml` notes as in the [SANA_WAM README](../SANA_WAM/README.md#evaluation) apply, as does the parent-directory `env_cfg/` requirement. The adapter checks `get_robot_action_dim_info(env_cfg_type)` against the dual ARX-X5 contract (`arm_dim [6, 6]`, `ee_dim [1, 1]`). `action_type` must be `ee`: the causal line predicts EE poses only.

Server log lines to check:

- `[SANA_WAM_CAUSAL] ready: ...`: the resolved contract (`protocol=per_chunk_obs`, `memory_content`), the softmax position clock, the sampling knobs, the view resize, the commit and memory modes, the GDN recurrence source and the normalization sha.
- One line per `get_action`:

  ```text
  [SANA_WAM_CAUSAL] ep E chunk c: read_chunks=[...] origin=O encode_s=.. commit_s=.. obs_encode_s=.. generate_s=.. total_s=.. world_model_rel_rmse(chunk c-1)=.. cmd_vs_reached(chunk c-1): L pos ../.. mm rot .. deg grip .. | R ... obs_anchor_gap(chunk c-1)=..
  ```

  `read_chunks` lists the committed chunks the generation reads (the newest `sliding_window_chunks`), `origin` the oldest of them, where the softmax clock of the read starts at 0. `encode_s` / `commit_s` cover the executed chunk, `obs_encode_s` the new chunk's observation. `world_model_rel_rmse` is the relative RMSE between the latent the model predicted for the previous chunk and the latent of what was actually observed. `cmd_vs_reached(chunk c-1)` compares the previous chunk's EE commands with the poses reached after them, per arm (`L`/`R`): mean/max position gap in mm, max rotation gap in degrees, max gripper closedness gap. `obs_anchor_gap` is the relative RMSE between the previous chunk's observation (its start frame encoded alone, what it was generated from) and the anchor latent of its 9-frame clip encode: training's observation is the latter, so this checks the causal-VAE equality the protocol relies on (expected at bf16 noise).

The RoboDojo client times every RPC out after 120 s whatever `request_timeout_s` says. One `get_action` takes about 2 s.

## Configuration

`deploy.yml` carries the standard keys (`policy_name protocol host port bench_name task_name ckpt_name env_cfg_type seed action_type gpu_id eval_batch`) plus:

| key | default | meaning |
|---|---|---|
| `request_timeout_s` | 600.0 | ws client timeout (the RoboDojo client uses its own 120 s) |
| `checkpoint_dir` | null | explicit checkpoint dir (highest priority; `checkpoint_path` / `ckpt_dir` / `model_dir` are aliases) |
| `allow_donor_checkpoint` | false | serve an SFT checkpoint as the causal step-0 model (see Model Assets); pipeline checks only |
| `normalization_path` / `normalization_sha256` | null / null | explicit artifact and/or sha pin (see Model Assets) |
| `text_encoder_path` / `vae_path` | null | local copies of `google/gemma-2-2b-it` and `Efficient-Large-Model/LTX-2.3-Diffusers` (a repo root loads its `vae/` subfolder); null reads `SANA_WAM_TEXT_ENCODER_PATH` / `SANA_WAM_VAE_PATH`, then falls back to the Hub ids |
| `sampling_steps` | 10 | Euler steps per chunk |
| `cfg_scale` / `video_cfg_scale` / `action_cfg_scale` | 6.0 / null / 1.0 | per-stream text CFG as in SANA_WAM: video CFG 6, action CFG 1 (one unconditional forward per step either way). Each must be >= 1 |
| `flow_shift` / `action_flow_shift` | null / null | null = the yaml's `scheduler.inference_flow_shift` / `inference_action_flow_shift` (5 / 1 for m48n24) |
| `memory_mode` | full | `full` = the deployment above; `none` = the **memory ablation**: nothing is committed and every chunk reads an empty memory, so it is generated exactly like chunk 0 (the [observation \| targets] window every training window opens with, i.e. the donor's single window on the causal weights). The executed chunk is still encoded for `world_model_rel_rmse`, and the noise seed keeps the episode's chunk index |
| `commit_actions` | executed | what enters memory as an executed chunk's actions (`memory_content: video_and_action` only; a `video_only` memory keeps no action rows and refuses the other values): `executed` = the EE commands actually sent, re-normalized; `predicted` = the raw model output (A/B only); `reached` = the poses the arms actually reached (URDF FK of the joints measured after each action, the observed gripper), re-normalized. In the training data the two coincide (the dataset state is the previous command, `state[i+1] == action[i]`), so only `reached` keeps training's invariants in closed loop: the action rows agree with the observed video, and the last row equals the next chunk's state slot |
| `max_chunks_per_episode` | null | refuse an episode longer than this many chunks; null = no limit (GDN extrapolates past the 48-chunk training windows) |
| `log_world_model_error` | true | the per-chunk `world_model_rel_rmse` (and `obs_anchor_gap`) in the server log; `memory_mode: none` without it encodes no executed chunk |
| `attention_dump_dir` | null | diagnostics (`sana_wam_causal/attention_probe.py`): one `ep{E:03d}_chunk{C:03d}.npz` per chunk with, for every denoising step and text stream of its generation, the softmax layers' attention mass of each query group (obs / target / state / action) on each key group (every committed chunk read: its targets, state, actions; the window's own four groups), the norm of the attention output read from the memory vs the window, and the GDN layers' readout split into the decayed past state and the window (the past's share of the normalizer, both parts' norms). Read-only: the forwards are unchanged; the commits are not probed. The worker token `@ATTEMPT_DIR@` puts it in the unit's attempt dir |
| `view_resize` | null (= stretch) | as in SANA_WAM: the yaml's `data.extra.robot_sft.view_resize` when declared (a contradicting value is refused), else `stretch`. Every causal recipe trains on Sana code where stretch is the only SFT view resize, whatever resize its donor saw |
| `diffusion_seed_base` | 20261003 | noise seed = sha256(domain, base, eval seed, episode index, chunk index), 53 bits |
| `eef_pose_check` / `urdf_path` / `robot_root_poses` | true / null / null | the SANA_WAM EEF kinematics settings; once per episode the observed `*_ee_pose` is compared with FK(measured joints) |
| `device` / `weight_dtype` | cuda / bfloat16 | only bf16 weights are supported |
| `strict_image_size` | false | true rejects frames that are not 480x640x3; false warns once |
| `default_instruction` | follow the instruction | used only when the observation carries no instruction |

The chunk length, the video stride, the softmax window, the memory content and the canvas are properties of the checkpoint (see Model Assets), so no `deploy.yml` key selects them. The softmax read is always training's (the retired `softmax_window` key is refused). There is no `n_action_steps`: the memory only ever holds complete executed chunks (see Notes).

## Notes

**Episode cycle.** One `get_action` = one commit + one observation encode + one chunk generation:

1. **First `get_action`.**
   1. The three cameras are composited into the sana_pixel canvas and VAE-encoded as one frame (a 10x16 latent grid, 160 tokens): chunk 0's observation.
   2. Chunk 0 is denoised as the window [observation (t = 0) | chunk-0 target latent] + [state | 32 actions] (353 tokens) against an empty memory: the bidirectional single window.
2. **During a chunk.** RoboDojo calls `update_obs` once after each executed action; the last call arrives right before the next `get_action`. Each call advances the chunk clock tau and keeps the frames of ticks 0, 4, ..., 32 (stride 4). It encodes nothing.
3. **Every later `get_action` (chunk c >= 1).**
   1. Commit the executed chunk c - 1. Its 9 frames (ticks 0..32) are encoded as one VAE clip and latent 0 (the anchor) is dropped, which is exactly how training's clean target latent is built. One clean forward (t = 0) over [that latent] + [chunk-start state | executed actions] (193 tokens; `video_only`: the latent alone, 160 tokens) reads the memory like the generation of that chunk did, then writes the next GDN state and keeps its K/V as the softmax entry of chunk c - 1.
   2. Encode the current frame (tick 32 of chunk c - 1 = tick 0 of chunk c) alone: chunk c's observation.
   3. Generate chunk c: 10 Euler steps of [observation | targets] + [state | actions] against the frozen memory (the GDN state after the commit of chunk c - 1; softmax over the newest 24 committed chunks plus its own tokens).
   4. The observation is never committed; the conditional and unconditional text streams each keep their own memory.
4. **`reset`.** RoboDojo sends `reset` twice per episode; it drops the memory and is idempotent. An episode that ends mid-chunk commits nothing.

**Refusals.** A chunk executed for fewer than 32 ticks is never committed, because the causal policy never trained on a partial context chunk; the next `get_action` raises. An instruction change inside an episode also raises, since the memory is caption-conditioned.

**Clock.** Under `rope: aligned` the video latents and the robot rows share one physical clock (Sana `_physical_frame_ids` / `forward_chunk`):

- chunk c's observation is latent frame c (the instant of chunk c - 1's last target) and its target latent frame c + 1 (one latent frame per chunk);
- its state row sits at step 32c, its actions at steps 32c+1 .. 32c+32.

**Softmax read and positions.**
- Chunk c's forwards (its denoising steps and its commit) read the newest min(c, 24) committed chunks, c - 24 .. c - 1: training's spans (noisy chunk c reads clean chunks max(0, c - W) .. c - 1, clean chunk j reads max(0, j - W) .. j), which is also the default read of Sana's deploy session (W - 1 older chunks plus the most recent one).
- The softmax entries are kept BEFORE the rotary embedding and rotated on the fly at every read, on a clock numbered from 0 at the read's origin O, the oldest chunk read: chunk k in chunk slot k - O, the window of chunk c in slot c - O. A read is rotated once per chunk; the 10 denoising steps reuse it.
- While nothing has slid out (O = 0), this is the episode's physical clock. From chunk 24 on, every read sits where training's noisy chunk 24 sits in a window that starts at its first chunk, so the softmax layers never see a position id outside a training window.
- This is Sana's deploy `kv_position_mode="rank"`. Sana's default `"absolute"` differs by one common shift, which the relative RoPE of the aligned clock does not see: the tests match the port against both, and Sana's own test pins both against the training forward.
- **GDN layers** are never windowed. Their state keeps keys rotated at the physical positions of the whole episode, and their window runs on that clock.
- **Precision:** GDN numerators use complex64 RoPE; softmax layers use fp64 RoPE cast back to the activation dtype and fp32 attention, as in Sana.

**State and actions** (the D5 contract):

- **Anchor.** Measured joints -> URDF FK -> base-frame link6 pose (xyz + rot6d), and gripper closedness from the observed opening. Joints are masked: EEF-only, 20 active slots, 7-16 and 36-45.
- **Returned actions.** The 32 sampled rows are denormalized as absolute EEF targets. Gripper closedness is clipped to [0, 1]. Each rot6d is replaced by the rotation it encodes (Gram-Schmidt), and the rows are returned as `{left,right}_ee_pose` (link6 in the env-relative world frame, xyz + wxyz) plus `{left,right}_ee_joint_state` (opening).
- **Commit.** These *executed* rows, re-normalized with the same artifact, are what the next commit feeds the model. On an unchanged slot they equal the model output to fp32 rounding.

**Train/deploy gaps that remain:**

- The context is the policy's own execution, not demonstrations: the teacher-forcing distribution shift.
- Simulator frames are raw renders, whereas training frames went through x264.
- The gripper observation is RoboDojo's commanded, rate-limited opening.

The last two are shared with the bidirectional lane. Two RoboDojo tasks run past the 48-chunk training windows: fasten_screws (1,900 steps, 60 chunks) and imitate_sorting_sequence (1,600 steps, 50 chunks). There, only the GDN recurrence extrapolates; the softmax read set stays at 24 chunks. And whereas a training window starts at any row of an episode, the deployed memory always starts at the episode's first chunk.

**Cost per chunk** (RTX A6000, bf16, 10 steps, video CFG 6; first light 2026-10-10):

| stage | time |
|---|---|
| VAE encode of the executed chunk | ~0.27 s |
| commit (2 forwards, one per text stream) | ~0.18 s |
| single-frame encode of the new chunk's observation | ~0.12 s |
| generation (10 x 2 forwards of 353 tokens) | ~1.65-1.75 s |

The first chunk pays only the observation encode and the generation. The softmax memory is bf16 pre-RoPE K/V of at most 24 chunks x 193 tokens (160 under `video_only`) x 8 layers per text stream, plus one rotated copy per chunk; the GDN state is per layer.

**Known limitations:**

- Batch-1 sampler (`eval_batch: false`).
- Synchronous deployment only: no asynchronous or speculative chunk generation, no replanning inside a chunk.
- Only the one-view sana_pixel canvas with `rope: aligned`.
- No trained checkpoint of the per-chunk observation protocol has been evaluated yet (see Validation).

## Validation

Run the CPU tests from `policy/SANA_WAM_CAUSAL` with the policy env. `SANA_CAUSAL_REPO` points at a Sana `rwm/zekai-merge` checkout at or after `3a6b2bf07` for the live parity tests (default `~/zekail/Sana`); those tests skip without one, or with an older one.

```bash
DISABLE_XFORMERS=1 CUDA_VISIBLE_DEVICES="" SANA_CAUSAL_REPO=~/zekail/Sana python -m pytest -q tests
```

95 tests (2026-10-10, Sana `9ee9c2d9d`):

- **Deploy = training** (`tests/test_causal_live_parity.py`, against live Sana; tiny architecture with identical random weights; CPU fp64; `flash_attn` bound to an fp64 reference varlen as Sana's own tests do).
  - The decisive check: Sana's two-stream training forward over [clean targets 0..L-2 | noisy (obs_c | targets_c) 0..L-1] equals, chunk by chunk, the streaming chain of this port (for every chunk a denoising window [obs_c | targets_c] against the memory, then the commit of the clean chunk). It holds for both memory contents, softmax windows None / 1 / 2 / 3 (so the read slides) and video strides 1 / 4, to 1e-9 relative; the first chunk equals Sana's one-chunk forward under split and shared action shifts.
  - The full session (10-step sampler, per-stream CFG, memory) matches Sana's `CausalPolicyChunkSession` under no CFG, video CFG and both-stream CFG across the window boundary (5 chunks, W = 2), on both of Sana's position clocks (`rank` and `absolute`), for both memory contents.
  - Negative controls: an observation moves its own chunk only (it never enters memory), in training and in deploy; the training forward reads the clean chunks' actions under `video_and_action` and ignores them under `video_only`; a read window one chunk short or long matches training only while the history fits both.
  - One GDN layer's scan matches Sana's `gdn_scan` / `_readout` to 1e-12; the fp32-attention chain matches at fp32 tolerance.
  - Every rotary table of a 6-chunk chain (W = 3): the softmax positions stay within chunk slot W (the read's chunks in slots 0..W-1), while the GDN windows run on the physical clock.
- **Normalization (D5)** (`tests/test_normalization_commit.py`, on the `1fe3b7e7` artifact the causal recipes pin).
  - The EEF-only slots and schemes.
  - Executed commands commit as the model output when nothing was clipped.
  - Clipped grippers commit at the bound, and a non-orthonormal rot6d commits the rotation the evaluator received.
  - Parity with Sana's deploy robot I/O: `PolicyRobotIO.state80` / `normalize_state80` / `actions80` and `normalize_robot80_affine`.
- **Contract and memory** (`tests/test_contract_and_memory.py`).
  - Both pantheon14k yamls resolve (`video_and_action` from the absent key, `video_only`), and the bidirectional view builds the 32-layer / 2560 / 20-head trunk with softmax layers (3, 7, ..., 31).
  - The real `config.yaml` of the retired vanilla52k checkpoint s7646 is refused for its `obs_in_first_chunk`, and the other refused contracts listed in Model Assets are refused.
  - The read rule (the newest W committed chunks, numbered from 0 at the oldest, pre-RoPE keys rotated per read), a deterministic CPU episode for both memory contents, `reset_memory`, and the session's misuse refusals.
  - The test yamls (`tests/fixtures/config_causal_pantheon14k_m48n24_{videoandaction,videoonly}_s5000.yaml`) are the real `config.yaml` files of the two step-5000 checkpoints (2026-10-10), copied verbatim; the `video_and_action` one has no `memory_content` key (its training pin `3e05fe760` predates the key).
- **Adapter state machine** (`tests/test_adapter_state_machine.py`, fake runtime). The RoboDojo call order: the frames a commit encodes (ticks 0, 4, ..., 32 of the right chunk), commit, then the observation encode of the new chunk's start frame, then generate; the chunk-start anchor; `memory_mode: none` (no commit, an empty memory every chunk); the logged gaps; the double reset; a partial chunk, an instruction change and `commit_actions` other than `executed` on a `video_only` memory refused.
- **Real-weight teacher-forcing parity** (`tools/teacher_forcing_parity.py`, 2026-10-10, RTX A6000; the pantheon14k donor served as the causal step-0 model through `/homes/jiamingt/checkpoints/causal_step0/pantheon14k_sana_pixel_donor{,_videoonly}`).
  - Setup: Sana's own causal class (git archive of `9ee9c2d9d`), built and loaded as its validator does, runs ONE two-stream training forward over [29 clean targets | 30 noisy (obs | targets) chunks]. Against it, the adapter's KV-cache chain: the cached denoising window [obs_c | targets_c] of each noisy chunk, then `commit_chunk` of each clean chunk; chunks 25-29 read a sliding window.
  - Inputs: the same Gemma caption, VAE latents of smooth synthetic frames (each chunk's observation its start frame encoded alone), EEF-only states and actions, and per-chunk video / action noise levels 0.15 .. 0.85.
  - **fp32 (TF32 off):** relative error at most 3.1e-5 (video velocity) and 3.2e-6 (action velocity) under `video_and_action`, 3.8e-5 / 6.5e-6 under `video_only`, with no growth along the episode or past the window boundary.
  - **bf16:** 4.8e-2 / 2.0e-2 (`video_and_action`) and 5.5e-2 / 1.8e-2 (`video_only`): the size of the training forward's own bf16-vs-fp32 noise (4.5e-2 / 2.0e-2 and 5.2e-2 / 1.9e-2).
  - **fp64** (`--precisions float64 --model-device cpu`, 8 chunks, attention and the GDN rotary product in fp64 on both sides): video <= 2.5e-6 / action <= 2.4e-7. This is the floor of the fp32 islands both implementations keep in fp64 (every RMSNorm computes in fp32, the chunk decay in fp32): chunk 0, which reads no memory, sits at the same level (3.8e-7 / 3.2e-8) as the chunks that do, so the memory path adds no error; the tiny-model fp64 tests above are exact to 1e-9.
  - **`obs_anchor_gap` = 0 exactly:** every chunk's single-frame observation encode equals the anchor latent of its 9-frame clip encode, the causal-VAE equality the deploy observation relies on.
- **Real-weight first light** (`tools/first_light.py`, RTX A6000, 2026-10-10; the same step-0 model, both memory contents):
  - load 69 s, 802 of 805 tensors, recurrence initialized for the 24 GDN blocks;
  - a 60-chunk run (the fasten_screws length) produces finite actions throughout; the read set grows to [0..23] at chunk 24 and slides from chunk 25 (origin c - 24) through chunk 59 ([35..58]); `obs_anchor_gap` 0 at every commit;
  - per chunk: clip encode 0.25-0.29 s, commit 0.18 s, observation encode 0.11-0.12 s, generate 1.65-1.75 s;
  - GPU peak 17.8 GiB (`video_and_action`) / 17.4 GiB (`video_only`) with the window full, including the VAE and Gemma;
  - chunk 0 replays bit-identically with the same seed and is the same under both memory contents.
- **RoboDojo smoke** (2026-10-10, the same step-0 models, simulator and policy on one A6000 through `causal_p14k_eefabs_f33fps8.sh smoke-gpu`): `video_and_action` insert_key (300 steps, 10 chunks) and `video_only` stack_blocks (550 steps, 18 chunks); every later chunk committed at its boundary; 2.1-2.4 s per `get_action`; observed `*_ee_pose` vs FK(measured joints) 0.03 mm / 0.017 deg on both arms; `obs_anchor_gap` 0; no errors. A step-0 model is not expected to solve tasks.
- **Not yet done:** a trained checkpoint of either memory content, and the full protocol.
