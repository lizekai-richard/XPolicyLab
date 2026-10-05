# SANA_MOT

**Contributor:** NVIDIA (zekail) | **Paper:** SANA-WAM (to appear) | **arXiv:** pending | **Original code:** Sana (NVlabs), internal RoboDojo MoT SFT line (branch `rwm/mot`)

`SANA_MOT` adapts the SANA **bidirectional MoT (AttnRes dual-system) policy** `SanaRWMMoTAttnResPolicy_5B_P1_D36` to XPolicyLab. The policy has two independently parameterized experts: the Sana-Video hybrid trunk for the video stream and a fresh ~1B action expert. One joint self-attention over the concatenated video and robot tokens couples them at every layer. The adapter covers the RoboDojo ARX-X5 joint-space line with **absolute joint targets**, in both video layouts the policy trains on: the **2x2 multiview strip** (the default since Sana `33b220373`) and the **OpenWAM 384x320 composite canvas** (a switch; the only layout before that commit). It runs the model **in-process** through the vendored, Sana-free package `sana_mot_min/`. That package mirrors the bidirectional policy of Sana `rwm/mot` @ `b3b9e0e9e`, and also covers the text layout of checkpoints trained at `606e48dd9`; it adds the canvas compositor and the prompt renderers. It runs on top of the sibling adapter's runtime `policy/SANA_WAM/sana_wam_min/`: Flow-Euler sampler, LTX-2.3 causal VAE encoder, Gemma-2-2B text conditioning, Robot80 normalization and the RoboDojo codecs. The chunk-causal MoT line on the same branch (`SanaRWMMoTAttnResCausalPolicy`) is out of scope. Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5` (dual six-joint arms, one gripper channel each), `action_type=joint`. This is an **eval-only** submission: training and data conversion live in the Sana repository (`process_data.sh` / `train.sh` print a notice and exit 0).

Shared conventions (argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE`) are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

SANA_MOT runs in the **same environment as SANA_WAM**; `install.sh` delegates to `policy/SANA_WAM/install.sh`:

```bash
conda create -n sana_wam python=3.11 && conda activate sana_wam
cd XPolicyLab/policy/SANA_MOT && bash install.sh
# = torch 2.9.1 / torchvision 0.24.1 (cu128; SANA_WAM_SKIP_TORCH=1 keeps an existing CUDA torch), diffusers>=0.38,
#   transformers, safetensors, numpy, pillow, pyyaml, huggingface_hub, and XPolicyLab itself (pip install -e).
```

`policy/SANA_WAM/` must be present next to this directory: `sana_mot_min` imports its `sana_wam_min` package (the import fails loudly otherwise). Text encoder and VAE weights are not part of the checkpoint; `deploy.yml` points `text_encoder_path` / `vae_path` at the cluster copies of `google/gemma-2-2b-it` and `Efficient-Large-Model/LTX-2.3-Diffusers` (`SANA_WAM_TEXT_ENCODER_PATH` / `SANA_WAM_VAE_PATH` are read when the yaml keys are null).

## Data Processing

Not supported in this adapter (eval-only). The training datasets live in the Sana repository: `RoboDojoSFTDataset` (per-camera frames, packed into the multiview strip by the trainer) and `RoboDojoOpenWAMCanvasSFTDataset` (raw RoboDojo HDF5 JPEGs composited into the L-shape canvas). `bash process_data.sh ...` prints a notice and exits 0.

## Training

Not supported in this adapter (eval-only). Training lives on Sana branch `rwm/mot`, under `dev/rwm/train_video_scripts/dfw/` with the yamls in `dev/rwm/configs/ablation/mot/`:

| recipe | yaml | video layout | served here |
|---|---|---|---|
| `run_sft_robodojo_mot_jointabs_f25_320px.sbatch` | `sft_robodojo_mot_jointabs_f25_320px.yaml` | multiview (default) | yes |
| `run_sft_robodojo_mot_jointabs_f25_openwam.sbatch` | `sft_robodojo_mot_jointabs_f25_openwam.yaml` | `openwam_canvas` | yes |
| `run_sft_robodojo_mot_jointabs_f33fps8_320px.sbatch` | `sft_robodojo_mot_jointabs_f33fps8_320px.yaml` | multiview, strided video (`video_fps: 8`) | served (stride 4, 2 latent frames, 32 action rows) |

`bash train.sh ...` prints a notice and exits 0.

## Model Assets

Checkpoint layout (the accelerate FSDP full state dict of `SanaRWMMoTAttnResPolicy_5B_P1_D36`):

```text
<ckpt_dir>/
├── config.yaml                                   # the training yaml (model.model + model.extra MoT knobs + sampling defaults)
├── model/pytorch_model_fsdp.bin                  # fp32 full state dict, paired layout:
│                                                 #   video_dit.* / action_dit.* (expert-level modules),
│                                                 #   blocks.<i>.video_block.* / blocks.<i>.action_block.* (per layer),
│                                                 #   context_embedder.* (text path, current layout)
└── normalization/robodojo_arx_x5_model_fps_25_f25_normalization.json   # optional; the absolute-target Robot80 statistics
```

`config.yaml` may also sit one or two levels above the checkpoint dir (the `checkpoints/epoch_X_step_Y/` layout of a training run).

**Text layout** is detected from the state dict. The yaml does not record it.

| text layout | trained by | keys | stripped before the strict load |
|---|---|---|---|
| `context_embedder` | rwm/mot from `8fd95e219` on (the tip's `sana_latent` builds; every canvas run before `4e67e1e1d`) | 1,595: `context_embedder.{y_embedder, y_norm, action_y_embedder, action_y_norm[, state_proj]}` | `video_dit.pos_embed`, `context_embedder.y_embedder.y_embedding`, `context_embedder.action_y_embedder.y_embedding` |
| `shared_caption_embedder` | rwm/mot from `4e67e1e1d` on, the canvas modes (`openwam`, `sana_pixel`) | 1,589: `context_embedder.{y_embedder, y_norm[, state_proj]}` only; every action block's `cross_attn.kv_linear` is `[2048, 2560]` (it reads the one 2560-wide caption embedding both experts share) | `video_dit.pos_embed`, `context_embedder.y_embedder.y_embedding` |
| `legacy_action_mlp` | rwm/mot up to `606e48dd9` | 1,593: `video_dit.{y_embedder, attention_y_norm}` + `action_dit.context_mlp[, state_context_embed]` | `video_dit.pos_embed`, `video_dit.y_embedder.y_embedding` |

The action expert's per-layer attention projections are `blocks.<i>.action_block.attn.*` since rwm/mot `3058e5785` (2026-09-18) and `attn_head.*` before it; the loader renames old keys (a pure key rename, verified bit-identical upstream), so neither naming needs a converted copy of the weights.

The stripped tensors are buffers the inference forward never reads: the position table is unused under wan_rope, and the `y_embedding` tables serve caption dropout during training only. Each one is shape-checked before it is dropped. The transient layouts between the two, `context_embedder.text_proj` or a shared 2560-wide action context, were never used for an evaluation run and are refused. The loader then checks the softmax-layer placement of **both** experts against the config, and checks that the state-conditioning projector matches `model.extra.action_state_as_context`. Everything else is strict-loaded.

**Contracts of 2026-09-20 .. 23 (rwm/mot up to 71ac93f43).** The video layout is a DATA knob since `a26807821`:
`data.extra.multiview` = `sana_latent` (the strip; spelled `sana` for a day) / `openwam` (the 384x320 canvas) /
`sana_pixel` (new: every camera resize-cropped to 320x480 and halved into its semantic 2x2 quadrant of ONE 320x480
canvas, top-right black, a 10x15 latent grid; `sana_wam_min/sana_pixel_canvas.py`). The RoPE is `model.extra.rope`
(`71ac93f43`, required by the live factory): `aligned` = the video frame stride folded into the video clock and the
robot rows on it through video_dit's RoPE modules; `independent` = the action expert's own 1D clock, state 0 /
actions 1..S (0..S-1 under state_as_context), the video on its own clock. Yamls without the key trained the table of
their era: `legacy` before `42aee4fa9` (physical clock at stride 1; strided: state and first action both at 0 -- the
pinned canvas campaign 19045548 and every NSC run), the independent clock after it. A canvas reads ONE composite-view
row (G = 1): through both experts' embedders before `a26807821`, through the ONE shared caption embedder since
`4e67e1e1d`; separate embedders with `data.extra.multiview` are the 2026-09-21..22 G = 2 payload. The canvas row's
Observation View sentence changed twice (`composite_view` -> `l_shape` at `2b4a4dc8d` -> `two_rows` at `3ea1f9af9`;
sana_pixel `tiling`). `robot_base_eef` is EEF-only (EEF pose + grippers, `2434e9199`) and is served through EE poses.
Normalization artifacts since 2026-09-20 normalize the grippers by statistics and Rot6D by the fixed [-1, 1] range (the
masked affine map handles both schemes). The loader resolves every one of these from the yaml and the weights, prints it
in the ready line, and the deploy keys `rope_mode` / `text_groups` / `canvas_prompt` / `robot_base_eef_layout` override
an `auto` resolution that cannot be decided from the files.

**Video layout (older yamls)** comes from `model.extra.video_layout` when the yaml sets it. When the key is absent, `data.type` decides: a `*Canvas*` dataset means `openwam_canvas`, and any other dataset means `multiview`. Every yaml from before `33b220373` omits the key, because the canvas was the only MoT video layout then. That covers `606e48dd9`-era runs and `context_embedder` runs launched in the canvas-only window `8fd95e219..33b220373`. The NSC f25 canvas runs are an example: their step-36,250 uploads carry `context_embedder` weights and a canvas yaml without the key. An explicit key that contradicts `data.type` is refused. The legacy text layout exists only with the canvas.

The normalization artifact is resolved in this order: (1) an explicit `normalization_path`; (2) `<ckpt_dir>/normalization/robodojo_arx_x5_model_fps_25_f25_normalization.json`; (3) the packaged copy `normalization/`. The packaged copy is the **absolute** joint-target artifact that the local-cluster MoT yamls (multiview and canvas) pin, sha256 `802f8fe90cbe35688859d2ed2b058e04fd33a57261124422e7dc7e686db3d7ad`. Every trained checkpoint so far ships its own copy, and that copy wins. The NSC line's artifact is `3cd3ce1d949e68ae5164373ebcc1161c01eb44c145f01ef09c4a3225bba480be`. It has the same schema, masks and absolute mode, but its cache rebuild moved the quantiles slightly (up to 0.005 rad over the 12 joint slots, 0.2 % on `scale80`). The two are therefore **not** interchangeable, and the resolver refuses a fallback that the training yaml does not pin. The packaged copy is always pinned to its sha. An implicitly resolved artifact must also be the one the training yaml pins in `data.extra.robotwin_sft.normalization_sha256`. The artifact's `joint_target_mode` and `num_frames` are cross-checked against the yaml before the model is built. The adapter reads the joint-target mode from the artifact: `absolute` (this line) emits the denormalized targets as they are, and `anchor_delta` would add the observed joints.

Place or symlink checkpoints under `checkpoints/` (git-ignored). `ckpt_name` resolves through `XPolicyLab.utils.checkpoint_resolver.resolve_checkpoint_root` exactly as for SANA_WAM: explicit `checkpoint_dir` / `checkpoint_path` / `ckpt_dir` / `model_dir` first, then `ckpt_name` as a path, then `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`, then `checkpoints/<ckpt_name>/`.

**View resize since 2026-09-27 (rwm/mot 034e55dca / f06615b2f).** Every SFT view is **stretched** whole into its
bucket (`StretchResize`, bilinear, no aspect-ratio preservation) -- the sana_pixel canvas and the sana_latent per-view
bucket; the OpenWAM canvas always stretched. f06615b2f removed the `robot_sft.view_resize` key, so a newer yaml does not
record it: the adapter defaults to `view_resize: stretch` (a declared yaml key still wins) and keeps `crop`
(`ResizeCrop`) only as the legacy resize of checkpoints trained before 034e55dca, served with `view_resize: crop`. The
resize is the shared `sana_wam_min` code, byte-checked in `policy/SANA_WAM/tests/test_view_resize_live_parity.py`.

## Evaluation

```bash
cd XPolicyLab/policy/SANA_MOT
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# offline wiring check, no simulator (needs a GPU for the real model):
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls sana_mot_robodojo_stepNNNNN arx_x5 joint 0 0 0 sana_wam base
# simulator:
bash eval.sh RoboDojo stack_bowls sana_mot_robodojo_stepNNNNN arx_x5 joint 0 0 1 sana_wam robodojo
```

The same conda-activation (`NVCC_PREPEND_FLAGS`) and `pyyaml` notes as in the [SANA_WAM README](../SANA_WAM/README.md#evaluation) apply. The XPolicyLab checkout must live in a parent directory that carries `env_cfg/`; the adapter checks `get_robot_action_dim_info(env_cfg_type)` against the dual ARX-X5 contract (`arm_dim [6, 6]`, `ee_dim [1, 1]`). The server log's ready line names the resolved `video_layout=` (with the requested value) and `context_layout=`.

## Configuration

`deploy.yml` carries the standard keys (`policy_name protocol host port bench_name task_name ckpt_name env_cfg_type seed action_type gpu_id eval_batch`) plus:

| key | default | meaning |
|---|---|---|
| `request_timeout_s` | 600.0 | ws client timeout |
| `checkpoint_dir` | null | explicit checkpoint dir (highest priority; `checkpoint_path` / `ckpt_dir` / `model_dir` are aliases) |
| `video_layout` | auto | `auto` takes the checkpoint's layout; an explicit `multiview` / `openwam_canvas` / `sana_pixel_canvas` must agree with it and fails before the weights are read otherwise. It asserts, it never re-routes |
| `rope_mode` | auto | `auto` / `legacy` / `aligned` / `independent` (see Model Assets); a value contradicting a declared `model.extra.rope` is refused |
| `text_groups` / `canvas_prompt` | auto / auto | canvas layouts only: 1 or 2 prompt rows and the Observation View descriptor (`composite_view` / `l_shape` / `two_rows`, sana_pixel `composite_view` / `tiling`) |
| `robot_base_eef_layout` | auto | robot_base_eef checkpoints: `full` / `eef_only`; `auto` from the run's post-2026-09-20 markers. An EEF-only checkpoint requires `action_type: ee` |
| `eef_pose_check` / `urdf_path` / `robot_root_poses` | true / null / null | the SANA_WAM EEF kinematics settings, used by robot_base_eef checkpoints |
| `normalization_path` / `normalization_sha256` | null / null | explicit artifact and/or sha pin (see Model Assets) |
| `text_encoder_path` / `vae_path` | cluster copies | `google/gemma-2-2b-it` and `Efficient-Large-Model/LTX-2.3-Diffusers` (a repo root loads its `vae/` subfolder) |
| `sampling_steps` | 10 | Euler steps (null -> the yaml's `train.extra.rwm_validation_steps`) |
| `cfg_scale` | 1.0 | text CFG scale shared by the video and action streams |
| `video_cfg_scale` / `action_cfg_scale` | null / null | per-stream CFG scale, null inherits `cfg_scale`; `cfg_scale: 6` + `action_cfg_scale: 1` guides the video expert only (one unconditional forward per step either way). Each must be >= 1 |
| `flow_shift` | 3.5 | `scheduler.inference_flow_shift` |
| `view_resize` | null (= stretch) | how each view reaches its bucket: the yaml's `data.extra.robot_sft.view_resize` when declared (a contradicting value is refused), else `stretch` (the only SFT resize since rwm/mot 034e55dca); `crop` = the legacy scale-to-cover + centre crop of every checkpoint trained before it. The OpenWAM canvas always stretches |
| `n_action_steps` | null | targets returned per inference: `null` / `0` / `all` = the whole 24-target chunk; a positive `n` = the first `n` only (receding-horizon replanning) |
| `trajectory_dump_dir` | null | diagnostic only, same format as `policy/SANA_WAM`: per-episode `ep<N>.npz` with every `update_obs` tick's measured Robot80 joints + grippers (`obs_state`) and, per inference, the conditioning row and the absolute chunk returned (`chunk_actions`, `chunk_step` = the tick it was predicted from). Leave `null` in evaluations |
| `device` / `weight_dtype` | cuda / bfloat16 | only bf16 weights are supported |
| `joint_limit_mode` / `joint_lower` / `joint_upper` | clip / 12 x -pi / 12 x +pi | per-joint envelope gate (left 6 then right 6, radians); the shipped +-pi values only catch runaway targets |
| `diffusion_seed_base` | 20260802 | noise seed = sha256(base, eval seed, episode index, chunk index) |
| `strict_image_size` | false | true rejects frames that are not 480x640x3; false warns once |
| `default_instruction` | follow the instruction | used only when the observation carries no instruction |

The video layout, the multiview RoPE layout and the text layout are properties of the checkpoint (see Model Assets), so no `deploy.yml` key selects them; `video_layout` only asserts the first one, which is worth doing for a canvas checkpoint whose yaml predates the key. There is no `anchor_source` and no `state_profile`: the MoT RoboDojo joint lines predict absolute targets conditioned on the observed (measured) joint state, which is what the evaluator provides. A `robot_base_eef` yaml (the eefabs line) is served through EE poses (`action_type: ee`): its EEF state slots are FK(measured joints) exactly as SANA_WAM derives them, absolute EEF targets are emitted as they are, anchor-delta ones are reconstructed first. `qwen_canonical` and mixed action-mode ratios are refused.

## Notes

- **Observation, multiview.** The three cameras (`cam_head`, `cam_left_wrist`, `cam_right_wrist`, decoded RGB uint8 HxWx3, never channel-swapped) each go through the training transform `ToTensorVideo -> ResizeCrop 256x320 -> Normalize(0.5, 0.5)`. Each one is VAE-encoded on its own as frame 0 of a zero-filled 4-frame latent window (8x10 latent per view). The three windows are packed into one strip `[1, C, 4, 1, 240]` with `view_count` 3, `view_latent_shape` and `view_slot_ids` `(0, 2, 3)`. That is the data contract of Sana's bidirectional deploy session.
- **Observation, canvas.** The three frames are composited into ONE 384x320 canvas exactly as the canvas dataset does (`sana_mot_min/canvas.py`, a port of Sana's `openwam_multiview_layout.py`). The head frame is stretched to the 256x320 top slot and the wrists to the 128x160 bottom-left and bottom-right slots, with PIL BILINEAR and no aspect-ratio preservation. Then `ToTensorVideo -> Normalize(0.5, 0.5)` runs with no resize-crop. The canvas is encoded as frame 0 of the latent window on the native 12x10 grid (`view_count` 1).
- **Text.** Multiview uses G = V + 1 = 4 token-group rows (`sana_mot_min/prompt.py`). Each view row carries `Embodiment Type` / `Action Mode` / `Observation View` (the camera's descriptor) / `Instruction` and conditions its own video span through the grouped cross-attention. The robot row drops the `Observation View` line and conditions the action expert. The canvas uses ONE row shared by both experts, whose `Observation View` is the composite descriptor. Both renderings are byte-identical to Sana's `build_token_group_prompts`. The CFG unconditional rows drop the `Instruction` line.
- **Model.** `sana_mot_min/mot_model/` mirrors `sana_qwennext_mot_policy.py`, `mot_block.py`, `mot_action_dit.py` and `mot_context_embedder.py` at `b3b9e0e9e`. The video expert (vendored trunk layers, SDPA attention) and the action expert (hidden 1024, expert heads projecting into the shared 20x128 / 10x256 attention width) run their AdaLN, q/k/v projections, cross-attention and SwiGLU separately. Each layer runs one GDN global formula or one `scaled_dot_product_attention` over the concatenated tokens, and AttnRes runs per expert. Under multiview the video tokens are reordered view-major and get the `semantic_2x2` spatial RoPE (tile 15x30; `local_reset` is also modeled), and each view's timesteps are repeated. Both state-conditioning modes are modeled. With `action_state_as_context` false, one clean state token sits ahead of the action tokens with physical-time RoPE. With it true, the state is one text token: it joins the robot group only under multiview, and the shared group (so both experts' text) under the canvas. The action rows then get an independent 1D RoPE. Per-stream CFG applies to the two output heads exactly as for SANA_WAM.
- **Actions.** 24 absolute joint targets per inference (0.96 s at 25 Hz), grippers as closedness clamped to [0, 1] and inverted back to the RoboDojo opening on the boundary.
- **Strided video is served.** A yaml with `data.extra.robotwin_sft.video_fps` (the f33fps8 recipe: 33 source rows, `video_fps` 8, so stride `(33 - 1) / 8 = 4`, 9 video frames, 2 latent frames, 32 dense action rows) is accepted. Only the observation frame is real at deployment, so a stride changes exactly two things here: the observation latent is placed in the SHORTER window (`1 + video_fps / 8` frames instead of `1 + (rows - 1) / 8`), and the batch carries `data_info['video_frame_stride']`, which the mirror already reads to expect `(F - 1) * 8 * s` action rows and to take the independent 1D action RoPE (zero-phase state row first). The action rows, the normalization artifact and the prompt stay keyed on the source rows. Sana's own bidirectional deploy session still refuses such a yaml -- it builds the dense window of the tier -- so there is no upstream session to diff against; the forwards themselves are parity-tested at stride 2 and 4.
- **Known limitations.** Batch-1 sampler (`eval_batch: false`; `get_action_batch` loops sequentially); only the dense 25 fps / 25-frame tier; consumer GPUs unvalidated; no simulator run yet; no dense multiview checkpoint exists yet to exercise the multiview path on real weights. See Validation.

## Validation

- **Parity with the rwm/mot tip** (`tests/test_mot_head_parity.py`, live `71ac93f43`, CPU fp64, 1e-10): `sana_latent` strips under `rope: aligned` / `independent` at strides 1, 2, 4, the `openwam` and `sana_pixel` canvases through the shared caption embedder under both modes at strides 1 and 4, both state-conditioning modes, fps 16 and 25; the canvas build's key set (no action caption path; every action `kv_linear` reads the video-width embedding) strict-loads and the other build refuses it; both models refuse the same text group counts.
- **Bitwise parity with the live Sana model at `b3b9e0e9e`** (`tests/test_mot_parity_sana.py`; run it with `SANA_MOT_REPO` at a worktree of that revision -- the tip dropped `video_layout` -- CPU fp64, the tiny architecture of Sana's `manual_cpu_selfcheck_mot_policy.py`). The live state dict strict-loads into the mirror after the three buffers are stripped. `x` and `action_pred` agree to 1e-10 in both state-conditioning modes for multiview with V=3 under both RoPE layouts, V=1, the canvas switch and strided forwards at stride 2 and 4. An explicit stride of 1 equals the dense forward, and the fp32-attention upcast matches under both video layouts. Both models refuse the same malformed inputs.
- **Parity with recorded past revisions** (`tests/test_mot_reference_parity.py`, fixtures from `tools/make_reference_parity_fixture.py`, both state modes, 1e-10):
  - `606e48dd9` is the legacy text layout, recorded once from that commit. The generator reproduces the committed fixture bit for bit.
  - `20c4d63d9` is the `context_embedder` layout while the canvas was the only layout. This is the forward the NSC f25 step-36,250 checkpoints were trained with. Their source commit `295f48f15` was never pushed, and `20c4d63d9` is the last pushed commit before the switch, which Sana's `33b220373` reports as a bit-identical canvas forward.
- **Canvas parity**: `canvas_from_frames` equals Sana's `assemble_openwam_canvas` pixel for pixel on random 480x640 frames (`tests/test_canvas.py`).
- **Prompt parity**: byte-identical to `build_token_group_prompts` for the canvas row and the multiview rows, in both joint-target modes, conditional and unconditional (`tests/test_prompt.py`).
- **Config resolution** (`tests/test_mot_model_cpu.py`): the multiview, canvas, f33fps8 and both NSC f25 yamls resolve to the 32-layer / 2560 / 20-head trunk with softmax layers (3, 7, ..., 31) and a 1024-wide action expert. They count 5.436B parameters on the meta device, the figure the trainer smoke reported, and 1,595 / 1,593 checkpoint keys for the two text layouts.
- **Offline strict-load rehearsal against real checkpoints** (2026-09-18). The key and shape lists of four trained checkpoints were read from their zip `data.pkl` without downloading the weights. Each one went through `load_mot_weights` against a meta-device mirror built from its yaml, and every key and shape matched:

  | checkpoint | resolved layouts | tensors loaded |
  |---|---|---|
  | `mot_canvas_nsc_s15k` | legacy / canvas | 1,591 of 1,593 |
  | `sft_robodojo_mot_jointabs_f25_openwam_nsc` step 36,250 | context_embedder / canvas | 1,592 of 1,595 |
  | `..._state_as_context_true` step 36,250 | context_embedder / canvas, state as context | 1,592 of 1,595 |
  | `sft_robodojo_mot_jointabs_f33fps8_320px_nsc` step 10,000 | context_embedder / multiview | 1,592 of 1,595; the session serves its strided yaml at stride 4 |
- **Real-weight first light** (RTX A6000, 10 Euler steps, CFG 6 / action CFG 1, 24 finite absolute joint targets, bit-identical across repeated calls with the same seed):

  | checkpoint | layouts | load | predict warm / cold | GPU peak incl. VAE + Gemma |
  |---|---|---|---|---|
  | `logits/sft_robodojo_mot_jointabs_f25_openwam_nsc` epoch 5 / step 15000 (2026-09-16; rerun with the current code 2026-09-18, outputs identical) | legacy / canvas | 1,591 tensors | 2.7-2.8 s / 3.3 s | 17.2 GiB |
  | the same repo, step 36,250 (2026-09-18) | context_embedder / canvas | 1,592 tensors | 2.8 s / 3.5 s | 17.2 GiB |

  Both are 5.436 B parameters in bf16 with the softmax placement (3, 7, ..., 31) agreeing on both experts. No missing and no unexpected keys.
- **Not yet done**: a multiview checkpoint on real weights, a forward-parity probe against the live Sana model on real weights (the tiny-model parity above is bitwise), a holdout replay, the `EVAL_ENV_TYPE=debug` wiring check and a simulator run. The SANA_WAM `tools/` scripts are the template for the first two.

Run the CPU tests from `policy/SANA_MOT` with the policy env. `SANA_MOT_REPO` points at a Sana rwm/mot checkout for the live parity tests, which skip without one; the recorded-revision fixtures need no checkout.

```bash
DISABLE_XFORMERS=1 SANA_MOT_REPO=~/zekail/Sana_mot python -m pytest -q tests
```
