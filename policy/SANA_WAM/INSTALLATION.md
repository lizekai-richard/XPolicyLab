# SANA_WAM Installation

Environment, model assets and a wiring check. The adapter is **eval-only** and Sana-free: it serves the
RoboDojo ARX-X5 policy through the vendored `sana_wam_min/` port, so no Sana checkout is required.
For what the knobs mean once this is running, see [README.md](README.md).

## 1. Environment

```bash
conda create -n sana_wam python=3.11 && conda activate sana_wam
cd XPolicyLab/policy/SANA_WAM && bash install.sh
```

`install.sh` installs the validated `torch 2.9.1 / torchvision 0.24.1` pair from the CUDA 12.8 wheel index
(set `SANA_WAM_SKIP_TORCH=1` to keep an existing CUDA torch), then `diffusers>=0.38`,
`transformers>=4.46,<5`, `safetensors`, `numpy`, `pillow`, `pyyaml`, `huggingface_hub`, and finally
`pip install -e` on the XPolicyLab root so `XPolicyLab.policy.SANA_WAM.model` and `client_server.ws`
resolve in this environment. It ends with a version assertion, so a silent partial install fails loudly.

If you skip the editable install, add the XPolicyLab runtime packages by hand — the websocket
`PolicyServer` will not import without them:

```text
h5py  opencv-python-headless  pyyaml  websockets>=14  msgpack  msgpack-numpy  pydantic
```

> `msgpack-numpy` is the one most often missed: it is imported by `client_server/ws/protocol/codec.py`,
> not by the policy, so the model loads fine and the server then dies at start-up.

Give the policy its own environment. Sharing one env with another policy tends to fail on transitive
pins (OpenWAM, for instance, needs `timm>=1.0.20` / `transformers>=5.5`, which this adapter's
`transformers<5` pin excludes).

## 2. Assets

Three things are needed and **none of them ship with the checkpoint**. The `text_encoder_path` /
`vae_path` defaults in `deploy.yml` are absolute cluster paths; off that cluster you must download
these and repoint the keys.

```bash
pip install -U "huggingface_hub[cli]"
mkdir -p ~/sana_wam_assets && cd ~/sana_wam_assets
```

### 2a. LTX-2.3 video VAE — 1.45 GB

The adapter only needs the `vae/` subfolder. The full repo also carries the transformer, audio VAE and
vocoder, which are many GB and are never loaded here:

```bash
hf download Efficient-Large-Model/LTX-2.3-Diffusers \
  --include "vae/*" --local-dir LTX-2.3-Diffusers
```

`load_vae` accepts a hub id, the repo root (it resolves the `vae/` subfolder itself) or the VAE folder
directly, so `vae_path: ~/sana_wam_assets/LTX-2.3-Diffusers` is the right value after this.

### 2b. Gemma-2-2b-it text encoder — ~5.3 GB

```bash
hf download Efficient-Large-Model/gemma-2-2b-it --local-dir gemma-2-2b-it \
  --exclude "gemma-2-2b-it.safetensors"
```

`Efficient-Large-Model/gemma-2-2b-it` is an **ungated mirror** — prefer it. `google/gemma-2-2b-it` is
gated and needs `hf auth login` with an account that accepted the licence.

The `--exclude` skips a 5.2 GB single-file copy of the same weights: the adapter loads through
`AutoModelForCausalLM`, which uses `model.safetensors.index.json` and its two shards. Dropping the
`--exclude` also works, it just doubles the download.

### 2c. Policy checkpoint

```text
<ckpt_dir>/
├── config.yaml                                                          # training yaml
├── model/pytorch_model_fsdp.bin                                         # fp32 FSDP state dict, 17.9 GB
└── normalization/robodojo_arx_x5_model_fps_25_f25_normalization.json    # optional, see below
```

`config.yaml` may also sit one or two levels above the checkpoint dir (the
`work_dir/checkpoints/epoch_X_step_Y/` layout of a training run); the adapter searches the dir, its
parent and its grandparent.

If the checkpoint ships **no** `normalization/`, the adapter falls back to the packaged copy at
`policy/SANA_WAM/normalization/` (sha256 `983fbd46…`). That copy encodes `joint_target_mode:
anchor_delta`, so a run trained with **absolute** joint targets must ship its own artifact — the yaml's
`normalization_sha256` pin will refuse the packaged one. Training runs that dump checkpoints per step
often keep the artifact only at the run root; copy it down next to the weights.

Then place or symlink the checkpoint under `checkpoints/` (git-ignored):

```bash
cd XPolicyLab/policy/SANA_WAM && mkdir -p checkpoints
ln -sfn /path/to/sana_wam_robodojo_320px_stepNNNNN checkpoints/sana_wam_robodojo_320px_stepNNNNN
```

## 3. Point the config at them

Either edit `deploy.yml`:

```yaml
text_encoder_path: ~/sana_wam_assets/gemma-2-2b-it
vae_path: ~/sana_wam_assets/LTX-2.3-Diffusers
```

or leave those keys `null` and export the fallbacks, which is friendlier for multi-machine runs:

```bash
export SANA_WAM_TEXT_ENCODER_PATH=~/sana_wam_assets/gemma-2-2b-it
export SANA_WAM_VAE_PATH=~/sana_wam_assets/LTX-2.3-Diffusers
```

## 4. Verify

CPU-only, seconds, no assets needed — checks the port itself:

```bash
cd XPolicyLab/policy/SANA_WAM && python -m pytest tests -q
```

Then the offline wiring check, which loads the real weights (needs a GPU, a few minutes from cold
storage) but no simulator:

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls <ckpt_name> arx_x5 joint 0 0 0 sana_wam base
```

A healthy start-up prints the normalization receipt before the model is built, then the ready line:

```text
[SANA_WAM] normalization: path=... sha256=... joint_target_mode=... action_slots_normalized=[...]
[SANA_WAM] normalization layout OK: 6 joints per arm (slots 0-5 / 29-34), grippers 16 / 45 identity
[SANA_WAM] ready: ckpt=... steps=... cfg_scale=... video_cfg_scale=... action_cfg_scale=... flow_shift=...
```

## 5. Variables

| variable | effect |
|---|---|
| `SANA_WAM_SKIP_TORCH=1` | `install.sh` keeps the environment's existing CUDA torch |
| `SANA_WAM_TEXT_ENCODER_PATH` / `SANA_WAM_VAE_PATH` | used when the matching `deploy.yml` key is `null` |
| `EVAL_ENV_TYPE=debug` | `eval.sh` runs the offline client, no simulator |
| `DEBUG_OBS_ENCODED=1` | debug client also exercises the server-side image decode path |
| `NVCC_PREPEND_FLAGS="" NVCC_APPEND_FLAGS=""` | required before `eval.sh` when the env ships the `cuda-nvcc` activation hook (see below) |

## 6. Troubleshooting

**`ModuleNotFoundError: msgpack_numpy` at server start-up.** The XPolicyLab runtime packages of
section 1 are missing. The model itself loads first, so the traceback appears *after* a successful
`[SANA_WAM] ready` line.

**Launcher aborts inside `conda activate`.** `setup_eval_policy_server.sh` activates with `set -u`, and
the `cuda-nvcc` activation hook references `NVCC_PREPEND_FLAGS` before defining it. Export both
variables (table above) before `eval.sh`.

**`utils/setup_env_client.sh` fails on a missing `yaml` module.** It reads `deploy.yml` with whatever
`python` is first on `PATH`, *before* activating the eval env. Put the policy env's `bin/` first on
`PATH`, or install `pyyaml` into the base env.

**Normalization sha mismatch.** The artifact is loaded and cross-checked against the training yaml
(`joint_target_mode`, `num_frames`) *before* the 4.47B model is built, so this fails in seconds rather
than after the 17.9 GB weight load. Check that the checkpoint's own artifact is present (section 2c)
rather than clearing the pin.

**`get_robot_action_dim_info` cannot find `env_cfg`.** The XPolicyLab checkout must sit in a parent
directory that carries `env_cfg/` (the RoboDojo workspace layout): the lookup is
`../../env_cfg/<env_cfg_type>.yml`, and the result is checked against the dual ARX-X5 contract
(`arm_dim [6, 6]`, `ee_dim [1, 1]`).
