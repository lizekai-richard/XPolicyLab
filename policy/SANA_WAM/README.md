# SANA_WAM

**Contributor:** NVIDIA (zekail) | **Paper:** SANA-WAM (to appear) | **arXiv:** pending | **Original code:** Sana (NVlabs), internal RoboDojo SFT line

`SANA_WAM` adapts the SANA unified world-action policy (bidirectional video + action diffusion, joint space, 320 px, RoboDojo ARX-X5 SFT) to XPolicyLab. The adapter runs the model **in-process** through the vendored, Sana-free package `sana_wam_min/` (policy transformer mirror, flow-matching Euler sampler, LTX-2.3 causal VAE encoder, Gemma-2-2B text conditioning, Robot80 normalization and the RoboDojo codecs). Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5` (dual six-joint arms, one gripper channel each), `action_type=joint`. This is an **eval-only** submission: training and data conversion live in the Sana repository (`process_data.sh` / `train.sh` print a notice and exit 0).

Shared conventions (argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE`) are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

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

The normalization artifact is resolved in this order, without any directory scan: (1) an explicit `normalization_path`; (2) `<ckpt_dir>/normalization/robodojo_arx_x5_model_fps_25_f25_normalization.json`; (3) the packaged copy `normalization/robodojo_arx_x5_model_fps_25_f25_normalization.json` (sha256 `983fbd46df6af34e2048ed6806ae9ce49cd8bd3af72cfba7d8610069959ea1da`). The packaged copy is always pinned to that sha; the other two are pinned only when `normalization_sha256` is set. The artifact is loaded and cross-checked against the training yaml (`joint_target_mode`, `num_frames`) before the 4.47B model is built, so a wrong artifact fails in seconds rather than after the weight load.

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
| `cfg_scale` | 1.0 | text CFG scale; the holdout validation ran at 1.0, the yaml's `inference_cfg_scale: 6.0` is a training-time visualization knob and is not used |
| `flow_shift` | 3.5 | `scheduler.inference_flow_shift` |
| `device` / `weight_dtype` | cuda / bfloat16 | only bf16 weights are validated |
| `joint_limit_mode` | clip | `clip` / `reject` / `none`; applied only when `joint_lower` / `joint_upper` (12 radians each: left 6 then right 6) are set, otherwise a warning is printed and no gating happens |
| `joint_lower` / `joint_upper` | 12 x -pi / 12 x +pi | per-joint envelope in radians. The shipped +-pi values are the smoke envelope the debug harness ran with, **not** the ARX-X5 URDF limits; they keep `clip` active and only catch runaway targets. Replace them with the robot's limits for hardware or strict simulator use |
| `diffusion_seed_base` | 20260802 | noise seed = sha256(base, eval seed, episode index, chunk index) |
| `strict_image_size` | false | true rejects frames that are not 480x640x3; false warns once and resizes/crops to the trained 256x320 bucket |
| `default_instruction` | follow the instruction | used only when the observation carries no instruction |

## Notes

- Actions: 24 absolute joint targets per inference (0.96 s at 25 Hz), all executed before the next observation. Joint targets are `anchor + delta` with the anchor being the raw joint state of the observation used for the chunk.
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
