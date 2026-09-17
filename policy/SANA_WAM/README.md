# SANA_WAM

**Contributor:** NVIDIA (zekail) | **Paper:** SANA-WAM (to appear) | **arXiv:** pending | **Original code:** Sana (NVlabs), internal RoboDojo SFT line

`SANA_WAM` adapts the SANA unified world-action policy (bidirectional video + action diffusion, joint space, 320 px, RoboDojo ARX-X5 SFT) to XPolicyLab. The adapter runs the model **in-process** through the vendored, Sana-free package `sana_wam_min/` (policy transformer mirror, flow-matching Euler sampler, LTX-2.3 causal VAE encoder, Gemma-2-2B text conditioning, Robot80 normalization and the RoboDojo codecs). Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5` (dual six-joint arms, one gripper channel each), `action_type=joint`. This is an **eval-only** submission: training and data conversion live in the Sana repository (`process_data.sh` / `train.sh` print a notice and exit 0). Two visual front-ends are served, auto-detected from the checkpoint's training yaml (`visual_layout`): the 320px three-view lines (`three_view_strip`: every camera 256x320, encoded on its own, packed as a strip) and the `rwm/openwam` canvas line (`openwam_canvas`: the three cameras composited into one 384x320 L-shaped canvas, encoded once, one shared prompt -- see [OpenWAM canvas checkpoints](#openwam-canvas-checkpoints)).

Shared conventions (argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE`) are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

See `INSTALLATION.md` for the full setup: environment, the three model assets that do **not** ship with the
checkpoint (LTX-2.3 VAE, Gemma-2-2b-it, the checkpoint itself) with download commands, and troubleshooting.

```bash
conda create -n sana_wam python=3.11 && conda activate sana_wam
cd XPolicyLab/policy/SANA_WAM && bash install.sh
# install.sh installs torch 2.9.1 / torchvision 0.24.1 from the cu128 index first (set SANA_WAM_SKIP_TORCH=1 to keep an existing CUDA torch),
# then diffusers>=0.38, transformers>=4.46,<5, safetensors, numpy, pillow, pyyaml, huggingface_hub, and XPolicyLab itself (pip install -e).
```

The policy itself (`sana_wam_min/`) needs only torch, diffusers, transformers, safetensors, numpy and pillow. The XPolicyLab side of the same environment (websocket `PolicyServer`, `utils/`, the debug client) additionally needs these runtime packages, which `pip install -e XPolicyLab` pulls in; install them explicitly when the editable install is skipped:

```text
h5py  opencv-python-headless  pyyaml  websockets>=14  msgpack  msgpack-numpy  pydantic
```

Text encoder and VAE weights are not part of the checkpoint. `deploy.yml` points `text_encoder_path` / `vae_path` at the cluster copies of `google/gemma-2-2b-it` and `Efficient-Large-Model/LTX-2.3-Diffusers`; outside the cluster set them to local snapshots or the HF ids (Gemma is gated and needs an HF token). `SANA_WAM_TEXT_ENCODER_PATH` / `SANA_WAM_VAE_PATH` are read when the yaml keys are null.

## Data Processing

Not supported in this adapter (eval-only). `bash process_data.sh ...` prints a notice and exits 0.

## Training

Not supported in this adapter (eval-only; training release timeline: to be announced with the paper). `bash train.sh ...` prints a notice and exits 0.

## Model Assets

Checkpoint layout (HF-style directory):

```text
<ckpt_dir>/
├── config.yaml                                   # the training yaml (architecture + sampling defaults)
├── model/pytorch_model_fsdp.bin                  # fp32 FSDP full state dict, 805 tensors, 17.9 GB
└── normalization/robodojo_arx_x5_model_fps_25_f25_normalization.json   # optional; Robot80 affine statistics
```

`config.yaml` may also sit one or two levels above the checkpoint dir (the `checkpoints/epoch_X_step_Y/` layout of a training run); the adapter searches the dir, its parent and its grandparent and fails loudly otherwise.

The normalization artifact is resolved in this order, without any directory scan: (1) an explicit `normalization_path`; (2) `<ckpt_dir>/normalization/robodojo_arx_x5_model_fps_25_f25_normalization.json`; (3) the packaged copy `normalization/robodojo_arx_x5_model_fps_25_f25_normalization.json` (sha256 `983fbd46df6af34e2048ed6806ae9ce49cd8bd3af72cfba7d8610069959ea1da`). The packaged copy is always pinned to that sha; the other two are pinned only when `normalization_sha256` is set. The artifact is loaded and cross-checked against the training yaml (`joint_target_mode`, `num_frames`) before the 4.47B model is built, so a wrong artifact fails in seconds rather than after the weight load. After loading, the server prints `[SANA_WAM] normalization: path=... sha256=... action_mode=... joint_target_mode=... action_slots_normalized=[...]` and checks the layout against the ARX-X5 joint-only contract (6 joints per arm in Robot80 slots 0-5 / 29-34, grippers 16 / 45 identity, 7th-joint slots 6 / 35 unused): `normalization layout OK` or a `WARNING` line.

### OpenWAM canvas checkpoints

The `rwm/openwam` canvas line (`SanaRWMOpenWAMCanvasPolicy_5B_P1_D36` trained on `RoboDojoOpenWAMCanvasSFTDataset`) keeps the same checkpoint layout. Its resolved `config.yaml` declares `model.model: SanaRWMOpenWAMCanvasPolicy_5B_P1_D36`, `data.type: RoboDojoOpenWAMCanvasSFTDataset`, `data.extra.openwam_canvas` (`layout: openwam_lshape_rgb_v1`, `encode_mode: joint_rgb_canvas`), `joint_target_mode: absolute` and the absolute normalization pin `802f8fe9...`, which is **not** the packaged artifact: ship the run's json at `<ckpt_dir>/normalization/robodojo_arx_x5_model_fps_25_f25_normalization.json` (the yaml pin refuses the packaged anchor_delta copy). The adapter detects the line (`visual_layout: auto`) and serves it as trained (`sana_wam_min/openwam_canvas.py` and the `shared_prompt` / `state_as_cross_attention` flags of `sana_wam_min/policy_model`, mirrored from Sana `dev/rwm/diffusion/data/openwam_multiview_layout.py`, `datasets/robodojo_openwam_canvas_sft_data.py` and `diffusion/model/nets/sana_qwennext_openwam_canvas_policy.py`):

- the three cameras are composited into **one** 384x320 L-shaped RGB canvas -- `cam_head` stretched to 256x320 on top, `cam_left_wrist` / `cam_right_wrist` to 128x160 below (PIL BILINEAR, no aspect-preserving crop, no gaps) -- and the canvas is encoded **once** by the LTX-2.3 VAE into a `[128, 4, 12, 10]` window: 480 video tokens on the native 12x10 grid, `view_count 1`, `view_keys ["openwam_canvas"]`, `view_slot_ids [0]`. The compositor is pixel-identical to OpenWAM's `assemble_multiview_layout` (tested against the copy vendored under `policy/OpenWAM/OpenWAM`) and to Sana's port;
- **one** prompt row is shared by the video and action tokens (G = 1): `Embodiment Type` / `Action Mode` / `Observation View: a composite view combining the head camera above the left and right wrist cameras` / `Instruction`; the CFG unconditional row drops the instruction;
- `model.extra.state_as_cross_attention` (default false) selects the state conditioning: the clean state token in the self-attention robot tail (`state_embed`, as every other line) or the opt-in cross-attention key (`state_context_embed`, actions-only tail, independent local 1D action RoPE). The checkpoint's own keys must match the flag (the loader refuses the other projector);
- absolute joint targets (no anchor addition; `anchor_source` only picks the conditioning row), `joint_only` state profile, `action_type: joint` only.

Verification: `tests/test_canvas_policy_parity_sana.py` pins the mirror bitwise against the live `SanaRWMOpenWAMCanvasPolicy` in both state modes on a tiny CPU model (needs a `rwm/openwam` checkout, `SANA_OPENWAM_REPO`), `tests/test_openwam_canvas*.py` pin the compositor, the pixel transform and the prompt rows against OpenWAM's and Sana's functions. No real canvas checkpoint had been served when this was written (the line had only been smoke-trained).

Place or symlink checkpoints under `checkpoints/` (git-ignored):

```bash
mkdir -p checkpoints
ln -sfn /path/to/sana_wam_robodojo_320px_stepNNNNN checkpoints/sana_wam_robodojo_320px_stepNNNNN
```

`ckpt_name` resolves through `XPolicyLab.utils.checkpoint_resolver.resolve_checkpoint_root` in this order: explicit keys `checkpoint_dir` / `checkpoint_path` / `ckpt_dir` / `model_dir` in `deploy.yml` (relative paths against `policy/SANA_WAM/`), then `ckpt_name` given as a path, then `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`, then `checkpoints/<ckpt_name>/`. The first existing candidate wins.

## Evaluation

```bash
cd XPolicyLab/policy/SANA_WAM
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# offline wiring check, no simulator (needs a GPU for the real model):
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls sana_wam_robodojo_320px_stepNNNNN arx_x5 joint 0 0 0 sana_wam base
# same, exercising the server-side image decode path (JPEG buffer / raw bytes / plain array across the 3 cameras):
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh RoboDojo stack_bowls sana_wam_robodojo_320px_stepNNNNN arx_x5 joint 0 0 0 sana_wam base
# simulator:
bash eval.sh RoboDojo stack_bowls sana_wam_robodojo_320px_stepNNNNN arx_x5 joint 0 0 1 sana_wam robodojo
```

> Conda activation under `set -u`: `setup_eval_policy_server.sh` (the standard XPolicyLab launcher) runs
> `conda activate` with `set -u` enabled. Environments that ship the `cuda-nvcc` activation hook reference
> `NVCC_PREPEND_FLAGS` before defining it and abort the launcher. Export the two variables before `eval.sh`:
> `export NVCC_PREPEND_FLAGS="" NVCC_APPEND_FLAGS=""`.
>
> `utils/setup_env_client.sh` also reads `deploy.yml` with whatever `python` is first on `PATH` *before* activating
> the eval env, so that interpreter needs `pyyaml` (put the policy env's `bin/` first on `PATH`, or install pyyaml in the base env).


The XPolicyLab checkout must live in a parent directory that carries `env_cfg/` (the RoboDojo workspace layout): `get_robot_action_dim_info(env_cfg_type)` reads `../../env_cfg/<env_cfg_type>.yml` and the adapter checks the result against the dual ARX-X5 contract (`arm_dim [6, 6]`, `ee_dim [1, 1]`). `eval.sh` waits up to 1200 s for the server; loading the 17.9 GB weights plus Gemma and the VAE takes a few minutes from cold storage.

## Configuration

`deploy.yml` carries the standard keys (`policy_name protocol host port bench_name task_name ckpt_name env_cfg_type seed action_type gpu_id eval_batch`) plus:

| key | default | meaning |
|---|---|---|
| `request_timeout_s` | 600.0 | ws client timeout; one inference is ~15 s at 50 steps on an H100 |
| `checkpoint_dir` | null | explicit HF-style checkpoint dir (highest priority; `checkpoint_path` / `ckpt_dir` / `model_dir` are aliases) |
| `normalization_path` | null | explicit normalization json (highest priority); null -> the checkpoint copy, then the packaged copy |
| `normalization_sha256` | null | sha256 pin enforced on whichever normalization json is used; the packaged copy is always pinned to `983fbd46...` |
| `text_encoder_path` | cluster Gemma-2-2b-it | local dir or HF id of `google/gemma-2-2b-it` |
| `vae_path` | cluster LTX-2.3-Diffusers | HF id of `Efficient-Large-Model/LTX-2.3-Diffusers`, a local repo root (loads its `vae/` subfolder), or the VAE folder itself (basename `vae` or a `config.json` with `_class_name: AutoencoderKLLTX2Video`) |
| `sampling_steps` | 50 | Euler steps (`train.extra.rwm_validation_steps`) |
| `cfg_scale` | 1.0 | text CFG scale shared by the video and action streams; the holdout validation ran at 1.0, the yaml's `inference_cfg_scale: 6.0` is a training-time visualization knob and is not used |
| `video_cfg_scale` / `action_cfg_scale` | null / null | per-stream CFG scale; `null` inherits `cfg_scale`. The transformer denoises video and action chunk jointly, so the unconditional forward runs once per step whenever either scale is above 1 (same cost as plain CFG); `cfg_scale: 6` + `action_cfg_scale: 1` guides the video only and the action stream integrates the exact conditional velocity. Each must be >= 1; both effective values are printed in the ready line and recorded in the predict receipt |
| `flow_shift` | 3.5 | `scheduler.inference_flow_shift` |
| `n_action_steps` | null | joint targets returned per inference. `null` / `0` / `all` returns the whole predicted chunk (24 for the 25-frame tier); a positive `n` returns only its first `n`, so the RoboDojo loop executes `n` control ticks, re-observes and asks for a new chunk -- receding-horizon replanning every `n` ticks at `chunk / n` times the inferences per episode. Values above the chunk length warn once and behave like `all` |
| `anchor_source` | measured | the joint state the model is conditioned on **and** the anchor its joint deltas are added to (one and the same row in training). `measured` = the evaluator's measured joints (historical behaviour); `last_command` = the last joint target this adapter returned for the previous chunk -- the training corpus records the previous frame's command as the state (`state[i+1] == action[i]`, verified on the public HDF5), so a measured anchor that lags the command it tracks shifts every target of a chunk by that lag; `last_command_clamped` = measured + clip(last_command - measured, +-`anchor_clamp_rad`), a guard against command wind-up under blocking contact. Grippers follow the same source; the first chunk of an episode always uses the measured state; the lag `|last_command - measured|` is printed per chunk in every mode |
| `anchor_clamp_rad` | 0.05 | radians; only used by `last_command_clamped` |
| `state_profile` | auto | `joint_only` / `robot_base_eef` / `auto` (= from the training yaml's `data.extra.action_mode_sample_ratio`). `robot_base_eef` checkpoints are conditioned on the flange (`link6`) pose per arm in the arm's `base_link` frame -- position + column rot6d after the mechanical-E axis remap -- which the adapter derives from the row's joints by URDF forward kinematics (`sana_wam_min/eef.py`, vendored `assets/robotwin2_arx_x5.urdf`), exactly as the corpus was packed (`FK(joint drive target)`) |
| `visual_layout` | auto | `three_view_strip` / `openwam_canvas` / `auto` (= from the training yaml's `model.model` + `data.type`). Strip: each camera resized/cropped to 256x320, encoded on its own, packed as a strip (the 320px lines). Canvas: the three cameras stretched into one 384x320 L-shaped canvas, encoded once, one shared prompt (the `rwm/openwam` line; the ready log prints `visual_layout=` and a `visual layout openwam_canvas:` line). An explicit value must agree with the checkpoint; it cannot re-route one |
| `action_type` | joint | `joint` = 24 joint-target dicts (works for every checkpoint; a robot_base_eef checkpoint's predicted EEF slots are reconstructed but not emitted); `ee` = 24 `left_ee_pose` / `right_ee_pose` (+ gripper) dicts -- the predicted anchor-relative E pose made absolute, mapped back to `link6` and into the env-relative world frame through `robot_root_poses`; the evaluator solves cuRobo IK per tick. Needs a robot_base_eef checkpoint and `anchor_source: measured` |
| `eef_pose_check` | true | once per episode, log the gap between the evaluator's `*_ee_pose` observation and FK(measured joints); warn above 5 mm / 1 deg |
| `urdf_path` / `robot_root_poses` | null / null | overrides for the packaged URDF and the ARX-X5 root poses (`env_cfg/robot/dual_x5.yml`: left (-0.3, -0.45, 0.765), right (0.3, -0.45, 0.765), quaternion wxyz (0.707, 0, 0, 0.707)) |
| `device` / `weight_dtype` | cuda / bfloat16 | only bf16 weights are validated |
| `joint_limit_mode` | clip | `clip` / `reject` / `none`; applied only when `joint_lower` / `joint_upper` (12 radians each: left 6 then right 6) are set, otherwise a warning is printed and no gating happens |
| `joint_lower` / `joint_upper` | 12 x -pi / 12 x +pi | per-joint envelope in radians. The shipped +-pi values are the smoke envelope the debug harness ran with, **not** the ARX-X5 URDF limits; they keep `clip` active and only catch runaway targets. Replace them with the robot's limits for hardware or strict simulator use |
| `diffusion_seed_base` | 20260802 | noise seed = sha256(base, eval seed, episode index, chunk index) |
| `strict_image_size` | false | true rejects frames that are not 480x640x3; false warns once and resizes/crops to the trained 256x320 bucket |
| `default_instruction` | follow the instruction | used only when the observation carries no instruction |

## Notes

- Actions: 24 absolute joint targets per inference (0.96 s at 25 Hz), all executed before the next inference unless `n_action_steps` is set, in which case only the first `n` are returned and the loop replans after them (the environment re-observes after every tick either way; the model itself is Markov in the current observation, so truncation needs no history bookkeeping). Joint targets are `anchor + delta` with the anchor being the raw joint state of the observation used for the chunk (`anchor_source: measured`), or the last target returned for the previous chunk (`last_command`, see Configuration): in the training corpus the recorded state IS the previous command, so the model learned command increments relative to the previous command, not to a lagging measurement.
- Robot-base EEF (the `..._unified_eef_...` SFT line, `action_mode_sample_ratio [0, 1, 0]`): the state row carries, per arm, `T_base_E = FK_base->link6(joints) @ LINK6_FROM_E` (rotation-only remap +X approach / +Y down / +Z left) as position (slots 7-9 / 36-38) and column rot6d (10-15 / 39-44); targets are `p_t - p_anchor` and `R_t R_anchor^T` -> rot6d, reconstructed by `eef.reconstruct_absolute_eef`. Verified: FK(recorded joint state) composed with the root poses reproduces the dataset's recorded `state/*_ee_poses` to 0.14 mm / 0.017 deg. All 32 dual_arm32 slots are supervised by such a checkpoint, so it can be deployed either through joints (`action_type: joint`) or through EE poses (`action_type: ee`).
- Prompt: the `Action Mode` sentence of every prompt row follows the checkpoint's training yaml (`data.extra.action_mode_sample_ratio`, `joint_target_mode`, `eef_target_mode`; `sana_wam_min/text.py::action_mode_text`, byte-identical to Sana's `build_token_group_prompts` for every mode) and the ready log prints it. Before 2026-09-16 the adapter always rendered the joint_only / anchor_delta sentence, so the `s55k_video` (absolute joints) and `s55k_eef` (robot_base_eef) evaluations ran with an `Action Mode` line their training never showed.
- Grippers: XPolicyLab `*_ee_joint_state` is the normalized opening (1 = open); the model works in closedness (1 = closed); the adapter inverts on both boundaries and clamps the model output to [0, 1].
- Images: RGB end to end. The checkpoint was trained on RGB frames; no channel conversion anywhere. Frames must arrive as decoded uint8 HxWx3 arrays; any other dtype is rejected with a `ValueError` naming the camera (a float frame's value range is unknowable, so it is never rescaled).
- Latency: ~15 s per chunk at 50 steps on an H100 (bf16, SDPA attention, no fused kernels); the first call also pays CUDA warm-up. Consumer GPUs are unvalidated and expected to be several times slower.
- Known limitations: batch-1 sampler (`eval_batch: false`; `get_action_batch` loops sequentially); only the 25 fps / 25-frame tier is trained (`additional_info.frequency` is ignored); instruction strings outside the RoboDojo training set are untested; the noise seed derivation differs from the Sana harness (per eval seed / episode / chunk instead of run/episode/request ids).
- Validation: see the section below.

## Validation

Measured on the checkpoint `epoch_6_step_35000` of run `SANA_Policy_SFT_RoboDojo_ArxX5_f25rgb_320px_unified_joint_only_lr5e-5_clip1` (H100, bf16, 50 Euler steps, cfg_scale 1.0, flow_shift 3.5):

- Strict load: 805 -> 802 tensors; `pos_embed`, `y_embedder.y_embedding` and `plucker_embed.weight` are stripped (not modeled by the inference mirror), everything else loads strictly.
- 35-sample holdout replay (`tools/holdout_replay.py`, one held-out episode per task): action MSE mean 0.021911 / median 0.004050 vs the trainer's 0.021878 / 0.004055 (documented 0.0219 / 0.0041); per-sample MSE delta mean 3.3e-5, max 5.2e-4.
- Bitwise components vs the Sana reference: Gemma text embeds, LTX2 causal VAE latents (frame-0 and the full strip), noise regeneration, and session vs direct sampler (35/35).
- Transformer forward is not bitwise vs the Sana live class on GPU (Sana runs flash/fla attention kernels, this package runs SDPA): final normalized action max|d| median 4.5e-3, mean 8.5e-3, max 0.117 over 50 Euler steps. Single-step probe numbers: to be appended. `tools/compare_to_sana_reference.py` therefore gates the final action at `--action-tol` (default 5e-2 normalized units) and treats the holdout MSE parity above as the functional gate.
- Import isolation (`tools/import_isolation_check.py`): passes; no `dev.`, `diffusion` or `sana` modules are imported by the adapter or `sana_wam_min`.

Offline wiring check (no simulator, real model on one GPU), run from `XPolicyLab/policy/SANA_WAM` with `checkpoint_dir` set in `deploy.yml` or the checkpoint linked under `checkpoints/<ckpt_name>/`:

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls <ckpt_name> arx_x5 joint 0 0 0 sana_wam base
```

### Forward-pass parity probe (Sana live class vs this port, real weights, one H100)

Identical inputs (bundle sample 000, sampler step 0 at t=1000 and a t=700 mixture point):

- fp32 weights, no autocast: `x` and `action_pred` are bitwise identical (`torch.equal`), and all 100 hooked
  sub-modules (embedders, 32 x {attn, cross_attn, mlp}, final_layer, action_head) match bitwise.
- bf16 weights + autocast (production): the first divergence is `blocks.0.cross_attn`; the Sana live class uses
  `xformers.ops.memory_efficient_attention` for softmax self- and cross-attention when xformers is installed,
  while this port always uses `F.scaled_dot_product_attention`. Single-forward `action_pred` max|d| 1.56e-2
  (masked-slot mean|d| 1.9e-3, mean|a| 0.81).
- With `DISABLE_XFORMERS=1` on the live class (both sides on SDPA) the bf16 outputs are bitwise identical as well.
- Control: live bf16 vs live fp32 `action_pred` max|d| 1.49e-2 -- the port-vs-live bf16 gap does not exceed the
  live model's own precision noise.

Therefore the residual holdout-level difference (final normalized action max|d| median 4.5e-3 over 50 Euler
steps) is attention-kernel arithmetic, not a porting defect; holdout action-MSE parity is the functional gate.
