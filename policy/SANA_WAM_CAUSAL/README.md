# SANA_WAM_CAUSAL

**Contributor:** NVIDIA (zekail) | **Paper:** SANA-WAM (to appear) | **arXiv:** pending | **Original code:** Sana (NVlabs), internal RoboDojo chunk-causal policy line (branch `rwm/zekai-merge`)

`SANA_WAM_CAUSAL` adapts the SANA **chunk-causal policy** `SanaRWMVideoQwenNextSubAttnResV2SelfFlowWorldModelCameraConditionMultiViewCausal_5B_P1_D36` to XPolicyLab. It is the 32-layer hybrid trunk of the bidirectional SANA_WAM policy made causal over action chunks. The 24 Gated DeltaNet layers carry a chunk-level delta-rule recurrence from chunk to chunk. The 8 softmax layers (3, 7, ..., 31) attend over a sliding window of the newest committed chunks. The model trains with **two-stream teacher forcing** and is deployed the same way: after each executed chunk, its observed outcome (camera frames, the commands actually sent, the chunk-start state) is committed into memory, and then the next chunk is generated. Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=ee`. The checkpoint must be the EEF-only absolute `robot_base_eef` line on the one-view `sana_pixel` 320x512 canvas with `rope: aligned` and strided video. The model runs **in-process** through the vendored, Sana-free package `sana_wam_causal/`, a mirror of Sana `rwm/zekai-merge` @ `c391260a4`: `sana_qwennext_policy_causal.py`, `layers/chunk_causal_{gdn,softmax,block}.py`, and the inference subset of `deploy/causal/session.py`. It sits on top of the sibling adapter's runtime `policy/SANA_WAM/sana_wam_min/`, which supplies the bidirectional trunk modules, the Flow-Euler sampler, the LTX-2.3 causal VAE encoder, Gemma-2-2B text conditioning, the sana_pixel canvas, Robot80 normalization, the EEF kinematics and the RoboDojo codecs. `policy/SANA_WAM` itself is not modified. This is an **eval-only** submission: training and data conversion live in the Sana repository, and `process_data.sh` / `train.sh` print a notice and exit 0.

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

Not supported in this adapter (eval-only). The training dataset is Sana's `CausalRoboDojoSFTDataset`. It cuts the RoboDojo episodes into windows of up to 48 chunks, each window starting from an observation frame. `bash process_data.sh ...` prints a notice and exits 0.

## Training

Not supported in this adapter (eval-only). Training lives on Sana branch `rwm/zekai-merge`. The recipes warm-start from a bidirectional SFT checkpoint (the *donor*) and add the GDN recurrence parameters:

| recipe (`dev/rwm/train_video_scripts/dfw/`) | yaml (`dev/rwm/configs/causal/robot_pretrain/`) | donor | served here |
|---|---|---|---|
| `run_causal_robodojo_vanilla52k_eefabs_f33fps8_sana_pixel_aligned_m48n24.sbatch` | `causal_robodojo_vanilla52k_eefabs_f33fps8_sana_pixel_aligned_m48n24.yaml` | vanilla52k sana_pixel 320x512 stretch SFT | yes |
| `run_causal_robodojo_vanilla34k_eefabs_f33fps8_sana_pixel_aligned_m48n24.sbatch` | `causal_robodojo_vanilla34k_eefabs_f33fps8_sana_pixel_aligned_m48n24.yaml` | vanilla34k sana_pixel 320x512 SFT | yes (same contract) |

Recipe `m48n24`:

- C = 32 action rows per chunk (1.28 s at 25 Hz).
- `video_fps` 8, i.e. video frame stride 4 and one latent frame per chunk.
- Training windows of up to 48 chunks; softmax window N = 24 chunks.
- `obs_in_first_chunk: true`.
- Flow shifts: video 5, action 1.

`bash train.sh ...` prints a notice and exits 0.

## Model Assets

Checkpoint layout (the accelerate FSDP full state dict of the causal policy, as the trainer saves it):

```text
<ckpt_dir>/
├── config.yaml                                   # the training yaml: model.model = the causal class above,
│                                                 #   model.extra.chunk_causal_policy, data.type CausalRoboDojoSFTDataset
├── model/pytorch_model_fsdp.bin                  # fp32 full state dict: the bidirectional trunk's keys + per GDN block
│                                                 #   attn.gate_proj.{weight,bias}, attn.A_log, attn.dt_bias (attn.recall_gate)
└── normalization/robodojo_arx_x5_model_fps_25_f33_normalization.json   # the f33 Robot80 statistics the yaml pins
```

`config.yaml` may also sit one or two levels above the checkpoint dir, as in the `checkpoints/epoch_X_step_Y/` layout of a training run. `ckpt_name` resolves through `XPolicyLab.utils.checkpoint_resolver.resolve_checkpoint_root` exactly as for SANA_WAM: explicit `checkpoint_dir` / `checkpoint_path` / `ckpt_dir` / `model_dir` first, then `ckpt_name` as a path, then `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`, then `checkpoints/<ckpt_name>/`.

**Contract.** `sana_wam_causal/contract.py` reads everything from the yaml and refuses anything it does not implement, before the weights are read:

- **Geometry.** Actions per chunk come from `model.extra.chunk_causal_policy.actions_per_chunk` at the model fps; the dataset's chunk must agree. The tier is `data.extra.robot_sft.tier_num_frames = C + 1` (33). The video stride comes from `video_fps`. The window is `sliding_window_chunks`, and `obs_in_first_chunk` must be true.
- **Refused contracts:** a bidirectional class or dataset; a strip or multiview layout; `sana_pixel_pad: masked`; `rope` other than `aligned`; `use_time_conditioning: true`; dual AttnRes routing.

The non-model half (canvas, prompt, VAE encode, normalization, action denormalization) is SANA_WAM's `PolicyInferenceSession`, built on the yaml's **bidirectional view**: the same yaml with the SFT policy class and the 33-frame tier. It resolves exactly as it would for the donor checkpoint.

**Strict load** (`sana_wam_causal/checkpoint.py`):

- The state dict must carry the donor's trunk, the state token (`state_embed`), and the softmax placement of the config.
- The buffers that SANA_WAM's loader strips are dropped after a shape check. Any other unexpected or missing key is refused.
- The GDN recurrence parameters must be present for **every** GDN block. A checkpoint missing them for only some blocks is refused. `recall_gate` is a buffer the forward never reads and may be absent.

**Normalization** resolves as in SANA_WAM, in this order:

1. an explicit `normalization_path`;
2. `<ckpt_dir>/normalization/<f33 name>`.

The artifact must be the one the yaml pins in `data.extra.robot_sft.normalization_sha256`; for the m48n24 recipes that is `1fe3b7e7...`, the same statistics as the vanilla52k donor's `9f8b98ed...`. EEF-only absolute statistics:

- xyz: q01/q99 of both arms pooled;
- rot6d: the fixed [-1, 1] range (identity);
- gripper closedness: [0, 1] mapped to [-1, 1].

The action EEF slots share the state statistics.

**Step-0 pipeline checks.** An SFT checkpoint has no recurrence parameters. With `allow_donor_checkpoint: true`, a dir whose `config.yaml` is the causal yaml and whose `model/` + `normalization/` are the donor's serves the donor as the causal policy at step 0. The recurrence parameters are initialized as the causal trainer initializes them on its first load: `gate_proj` zero, and `dt_bias` / `A_log` from the yaml's `dt_bias_init` / `a_log_init` (-5 / 0 for m48n24). This gives a decay of about 0.993 per chunk. The ready line then says `gdn_recurrence=initialized (SFT donor, causal step 0)`. Use this for wiring and latency checks only, never as an evaluation.

## Evaluation

```bash
cd XPolicyLab/policy/SANA_WAM_CAUSAL
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# offline wiring check, no simulator (needs a GPU for the real model):
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls causal_vanilla52k_m48n24_stepNNNNN arx_x5 ee 0 0 0 sana_wam base
# same, exercising the server-side image decode path:
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh RoboDojo stack_bowls causal_vanilla52k_m48n24_stepNNNNN arx_x5 ee 0 0 0 sana_wam base
# simulator, one GPU for both (the policy peaks at ~18 GiB on an RTX A6000 with the softmax window full):
bash eval.sh RoboDojo insert_key causal_vanilla52k_m48n24_stepNNNNN arx_x5 ee 0 0 0 sana_wam robodojo
```

The debug client stops every episode after 20 actions, so it exercises chunk 0 and `reset` only; the commit path needs the simulator. Its fake `*_ee_pose` (all ones) makes the once-per-episode pose check report a gap of about 1.5-1.9 m and warn, which is expected there.

The same conda-activation (`NVCC_PREPEND_FLAGS`) and `pyyaml` notes as in the [SANA_WAM README](../SANA_WAM/README.md#evaluation) apply, as does the parent-directory `env_cfg/` requirement. The adapter checks `get_robot_action_dim_info(env_cfg_type)` against the dual ARX-X5 contract (`arm_dim [6, 6]`, `ee_dim [1, 1]`). `action_type` must be `ee`: the causal line predicts EE poses only.

Server log lines to check:

- `[SANA_WAM_CAUSAL] ready: ...`: the resolved contract, the sampling knobs, the view resize, the commit mode, the GDN recurrence source and the normalization sha.
- One line per `get_action`:

  ```text
  [SANA_WAM_CAUSAL] ep E chunk c: read_entries=[...] encode_s=.. commit_s=.. generate_s=.. total_s=.. world_model_rel_rmse(chunk c-1)=..
  ```

  `read_entries` lists the memory entries the generation reads: 0 = the observation, k + 1 = chunk k. `world_model_rel_rmse` is the relative RMSE between the latent the model predicted for the previous chunk and the latent of what was actually observed.

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
| `softmax_window` | training | read rule of the softmax memory: `training` = the newest `sliding_window_chunks` entries of [observation, chunk 0, chunk 1, ...], so the observation evicts like any other entry, which is the two-stream training forward's rule; `keep_obs` = the observation plus the newest N - 1 chunks, as Sana's deploy KV manager reads (A/B only) |
| `commit_actions` | executed | what enters memory as an executed chunk's actions: `executed` = the EE commands actually sent, re-normalized; `predicted` = the raw model output (A/B only) |
| `max_chunks_per_episode` | null | refuse an episode longer than this many chunks; null = no limit (GDN extrapolates past the 48-chunk training windows) |
| `log_world_model_error` | true | the per-chunk `world_model_rel_rmse` in the server log |
| `view_resize` | null (= stretch) | as in SANA_WAM: the yaml's `data.extra.robot_sft.view_resize` when declared (a contradicting value is refused), else `stretch`. Every causal recipe trains on Sana code where stretch is the only SFT view resize, whatever resize its donor saw |
| `diffusion_seed_base` | 20261003 | noise seed = sha256(domain, base, eval seed, episode index, chunk index), 53 bits |
| `eef_pose_check` / `urdf_path` / `robot_root_poses` | true / null / null | the SANA_WAM EEF kinematics settings; once per episode the observed `*_ee_pose` is compared with FK(measured joints) |
| `device` / `weight_dtype` | cuda / bfloat16 | only bf16 weights are supported |
| `strict_image_size` | false | true rejects frames that are not 480x640x3; false warns once |
| `default_instruction` | follow the instruction | used only when the observation carries no instruction |

The chunk length, the video stride, the window and the canvas are properties of the checkpoint (see Model Assets), so no `deploy.yml` key selects them. There is no `n_action_steps`: the memory only ever holds complete executed chunks (see Notes).

## Notes

**Episode cycle.** One `get_action` = one commit + one chunk generation:

1. **First `get_action`.**
   1. The three cameras are composited into the sana_pixel canvas and VAE-encoded as one frame (a 10x16 latent grid, 160 tokens).
   2. The frame enters memory as the video-only clean entry E_0 (`commit_observation`: the GDN state and the softmax K/V of entry 0).
   3. Chunk 0 is denoised as the cache-free window [observation (t = 0) | chunk-0 target latent] + [state | 32 actions].
2. **During a chunk.** RoboDojo calls `update_obs` once after each executed action; the last call arrives right before the next `get_action`. Each call advances the chunk clock tau and keeps the frames of ticks 0, 4, ..., 32 (stride 4). It encodes nothing.
3. **Every later `get_action`.**
   1. Commit the executed chunk. Its 9 frames (ticks 0..32) are encoded as one VAE clip and latent 0 (the anchor) is dropped, which is exactly how the training window's clean target latent is built.
   2. One clean forward (t = 0) over [that latent] + [chunk-start state | executed actions] reads the memory like a generation would. It then writes the next GDN state and appends its K/V as entry E_{c+1} (193 tokens).
   3. Generate chunk c + 1: 10 Euler steps against the frozen memory (GDN state after the last commit; softmax over the window rule's entries plus its own tokens).
   4. The conditional and unconditional text streams each keep their own memory.
4. **`reset`.** RoboDojo sends `reset` twice per episode; it drops the memory and is idempotent. An episode that ends mid-chunk commits nothing.

**Refusals.** A chunk executed for fewer than 32 ticks is never committed, because the causal policy never trained on a partial context chunk; the next `get_action` raises. An instruction change inside an episode also raises, since the memory is caption-conditioned.

**Clock.** Under `rope: aligned` the video latents and the robot rows share one physical clock:

- chunk c's state row sits at step 32c;
- its actions sit at steps 32c+1 .. 32c+32;
- its target latent is frame c+1.

The softmax K/V are stored post-RoPE at their physical positions, as the two-stream forward rotates its prefix once over the whole canvas. GDN numerators use complex64 RoPE; softmax layers use fp64 RoPE cast back to the activation dtype and fp32 attention, as in Sana.

**State and actions** (the D5 contract):

- **Anchor.** Measured joints -> URDF FK -> base-frame link6 pose (xyz + rot6d), and gripper closedness from the observed opening. Joints are masked: EEF-only, 20 active slots, 7-16 and 36-45.
- **Returned actions.** The 32 sampled rows are denormalized as absolute EEF targets. Gripper closedness is clipped to [0, 1]. Each rot6d is replaced by the rotation it encodes (Gram-Schmidt), and the rows are returned as `{left,right}_ee_pose` (link6 in the env-relative world frame, xyz + wxyz) plus `{left,right}_ee_joint_state` (opening).
- **Commit.** These *executed* rows, re-normalized with the same artifact, are what the next commit feeds the model. On an unchanged slot they equal the model output to fp32 rounding.

**Departures from Sana's own deploy session** (`dev/rwm/deploy/causal/session.py`, `CausalPolicyChunkSession`). Its KV cache is append-only, it always reads the observation entry, and by default it renumbers positions by rank. From chunk 24 on (more than 768 steps), its read set and positions therefore differ from the training forward's. This adapter follows the training spans and keeps Sana's always-read-the-observation rule as `softmax_window: keep_obs`, at physical positions. Sana's chunk-0 forward also raises whenever `sliding_window_chunks` is set (its validator unsets it), whereas the port serves the cache-free chunk-0 window directly.

**Train/deploy gaps that remain:**

- The context is the policy's own execution, not demonstrations: the teacher-forcing distribution shift.
- Simulator frames are raw renders, whereas training frames went through x264.
- The gripper observation is RoboDojo's commanded, rate-limited opening.

The last two are shared with the bidirectional lane. Two RoboDojo tasks run past the 48-chunk training windows: fasten_screws (1,900 steps, 60 chunks) and imitate_sorting_sequence (1,600 steps, 50 chunks). There, only the GDN recurrence extrapolates; the softmax read set stays at 24 entries.

**Cost per chunk** (RTX A6000, bf16, 10 steps, video CFG 6):

| stage | time |
|---|---|
| VAE encode of the executed chunk | ~0.27 s |
| commit (2 forwards, one per text stream) | ~0.17 s |
| generation (10 x 2 forwards) | ~1.53 s |

The first chunk instead pays a 0.1 s single-frame encode and the observation commit. The softmax memory is bf16 post-RoPE K/V of at most 24 entries x 193 tokens x 8 layers per text stream; the GDN state is per layer.

**Known limitations:**

- Batch-1 sampler (`eval_batch: false`).
- Synchronous deployment only: no asynchronous or speculative chunk generation, no replanning inside a chunk.
- Only the one-view sana_pixel canvas with `rope: aligned`.
- No trained causal checkpoint has been evaluated yet (see Validation).

## Validation

Run the CPU tests from `policy/SANA_WAM_CAUSAL` with the policy env. `SANA_CAUSAL_REPO` points at a Sana `rwm/zekai-merge` checkout for the live parity tests (default `~/zekail/Sana`); those tests skip without one.

```bash
DISABLE_XFORMERS=1 CUDA_VISIBLE_DEVICES="" SANA_CAUSAL_REPO=~/zekail/Sana python -m pytest -q tests
```

- **Deploy = training** (`tests/test_causal_live_parity.py`, against live Sana @ `c391260a4`; tiny architecture with identical random weights; CPU fp64; `flash_attn` bound to an fp64 reference varlen as Sana's own tests do).
  - The decisive check: Sana's two-stream training forward over [obs | clean 0..L-2 | noisy 0..L-1] equals, chunk by chunk, the streaming chain of this port (observation prefill, cache-free chunk-0 window, then for every chunk a cached denoising window followed by the commit of the clean chunk). It holds for softmax windows None / 1 / 2 / 3 (so the observation evicts) and video strides 1 / 4, to 1e-9 relative.
  - The chunk-0 window equals Sana's single-window forward.
  - `keep_obs` equals `training` while the observation fits the window and differs once it is evicted (negative control).
  - One GDN layer's scan matches Sana's `gdn_scan` / `_readout` to 1e-12.
  - The fp32-attention chain matches at fp32 tolerance.
  - The full session (10-step sampler, per-stream CFG, memory) matches Sana's `CausalPolicyChunkSession` under no CFG, video CFG and both-stream CFG over the chunks where both read the same entries.
- **Normalization (D5)** (`tests/test_normalization_commit.py`, on the vanilla52k f33 artifact).
  - The EEF-only slots and schemes.
  - Executed commands commit as the model output when nothing was clipped.
  - Clipped grippers commit at the bound, and a non-orthonormal rot6d commits the rotation the evaluator received.
  - Parity with Sana's deploy robot I/O: `PolicyRobotIO.state80` / `normalize_state80` / `actions80` and `normalize_robot80_affine`.
- **Contract and memory** (`tests/test_contract_and_memory.py`).
  - The m48n24 yaml resolves, and its bidirectional view builds the 32-layer / 2560 / 20-head trunk with softmax layers (3, 7, ..., 31).
  - The refused contracts listed in Model Assets are refused.
  - The read rules of both window modes, including eviction.
  - A deterministic CPU episode, and the session's misuse refusals.
  - The test yaml (`tests/fixtures/config_causal_vanilla52k_m48n24_derived.yaml`) is derived from the donor's trainer-written yaml. Its `model` / `data` / `scheduler` / `vae` / `text_encoder` sections equal the upstream vanilla52k recipe at `c391260a4` except for cluster paths and the normalization pin: it carries the donor's `9f8b98ed...`, whose artifact the fixtures ship. Both upstream m48n24 yamls (vanilla34k, vanilla52k) were checked once to resolve to the same contract.
- **Adapter state machine** (`tests/test_adapter_state_machine.py`, fake runtime). The RoboDojo call order: the frames a commit encodes (ticks 0, 4, ..., 32 of the right chunk), commit before generate, the chunk-start anchor, the double reset, a partial chunk and an instruction change refused.
- **Real-weight first light** (`tools/first_light.py`, RTX A6000, 2026-10-03/04). The vanilla52k sana_pixel donor served as the causal step-0 model:
  - load 47 s, 802 of 805 tensors, recurrence initialized for the 24 GDN blocks;
  - a 60-chunk run (the fasten_screws length) produces finite actions throughout. The read set grows to [0..23] at chunk 23, the observation evicts at chunk 24 ([1..24]), and the read set stays at 24 entries through chunk 59 ([36..59]);
  - per chunk: encode 0.25-0.29 s, commit 0.17 s, generate 1.56 s mean (1.65 s max), flat over the 60 chunks;
  - GPU peak 17.8 GiB with the window full, including the VAE and Gemma;
  - chunk 0 replays bit-identically with the same seed.
- **Debug closed loop** (2026-10-04, the same step-0 model). `setup_policy_server.py` + `debug_env_client.py`, run by hand because a donor dir needs `allow_donor_checkpoint=true`, which `eval.sh` cannot pass. Plain arrays and `DEBUG_OBS_ENCODED=1`: 4 episodes, every action dict validated, no tracebacks.
- **RoboDojo smoke** (2026-10-03/04, the same step-0 model, one `insert_key` episode, simulator and policy on one A6000):
  - 300 steps over 10 chunks, each later chunk committed at its boundary;
  - 1.9-2.1 s per `get_action`;
  - observed `*_ee_pose` vs FK(measured joints): 0.03 mm / 0.017 deg on both arms;
  - no errors.

  A step-0 model is not expected to solve tasks; the episode failed, which is expected.
- **Not yet done:** a trained causal checkpoint on real weights, the full protocol, and a real-weight forward-parity probe against the live Sana model (the tiny-model parity above is fp64).
