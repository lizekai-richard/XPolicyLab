# ViGAR

**Contributor:** PKU DAGroup & AgiBot | **Paper:** Rethinking World-Action Model for Compositional and In-Context Robotic Manipulation | **arXiv:** https://arxiv.org/abs/2610.02368 | **Original code:** https://github.com/DAGroup-PKU/ViGAR

ViGAR (**Vi**sual **G**oal-conditioned **A**ction **R**easoning) is a hierarchical framework that factorizes manipulation into a visual subgoal planner and a subgoal executor. Given the current observation and global instruction, the subgoal planner predicts a visual subgoal for the next subtask. The subgoal executor then jointly generates future visual trajectories and actions conditioned on the predicted subgoal.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboTwin 2.0 Leaderboard](https://robotwin-platform.github.io/leaderboard).

## Installation

Requires Linux x86_64, an NVIDIA driver supporting CUDA 12.8, FFmpeg and [uv](https://docs.astral.sh/uv/) (see [ViGAR's installation guide](https://github.com/DAGroup-PKU/ViGAR/blob/main/docs/installation.md)). `install.sh` clones ViGAR at a pinned commit and builds its model environment (Python 3.13, PyTorch 2.10) in `source_vigar/.venv-model`, then installs XPolicyLab into it. Set `VIGAR_ROOT` to use an existing ViGAR checkout. The simulator runs in RoboTwin's own environment.

```bash
cd XPolicyLab/policy/ViGAR
bash install.sh
```

## Model Assets

```bash
bash download_checkpoint.sh
```

This downloads the policy (`weights/policy/robotwin_c2r`, with its Wan2.2 VAE and Qwen3-VL tokenizer) and the subgoal planner (`weights/subgoal_planner/robotwin_subgoal_planner`) from [DAGroup-PKU/ViGAR](https://huggingface.co/DAGroup-PKU/ViGAR) to `checkpoints/ViGAR/`. Pass a directory to download elsewhere, then use that path as `ckpt_name` or set `checkpoint_path` in `deploy.yml`.

## Data Processing

ViGAR trains on its own prepared RoboTwin data, not on the XPolicyLab LeRobot converters: 50 tasks × 50 `demo_clean` episodes with stage annotations, 49-D actions and generated goal images. `process_data.sh` downloads that data and the Cosmos3-Nano base checkpoint from Hugging Face; it does not convert raw trajectories. Rebuilding the data from raw RoboTwin trajectories is described in [ViGAR's training guide](https://github.com/DAGroup-PKU/ViGAR/blob/main/docs/training.md).

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>

# Example
bash process_data.sh RoboTwin ViGAR arx_x5 joint
```

## Training

Post-trains the policy with ViGAR's recipe: 8 GPUs, global batch 256, learning rate 1e-5, 50,000 updates, a checkpoint every 4,000 updates. Outputs go to `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/runs/`. Run `download_checkpoint.sh` first; training takes the VAE and the tokenizer from the released policy. Subgoal planner training is described in [ViGAR's training guide](https://github.com/DAGroup-PKU/ViGAR/blob/main/docs/training.md).

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example
bash train.sh RoboTwin ViGAR arx_x5 joint 0 0,1,2,3,4,5,6,7
```

## Evaluation

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env_conda_env>

# Example
bash eval.sh RoboTwin adjust_bottle ViGAR arx_x5 joint 0 0 0 uv <robotwin_env>
```

`<policy_env>` is `uv` for the environment made by `install.sh`, a ViGAR checkout containing `.venv-model`, a Python environment directory, or a conda environment. With one id in `<policy_gpu_id>`, the planner and the policy share that GPU; with two ids, such as `0,1`, the planner runs on the first and the policy on the second. The planner and the policy take about 33 GB of GPU memory each, and a simulator about 7 GB.

Server logs, including `planner.log` and `policy.log`, are written to `logs/<task_name>-<time>-<pid>/`.

## Reported Result

| Clean | Randomized | Average | Rollouts |
|---:|---:|---:|---:|
| 82.00 | 67.02 | **74.51** | 50 tasks × 2 conditions × 100 episodes = 10,000 |

## Configuration

`deploy.yml` keys to check before evaluation: `vigar_root`, `vigar_python`, `checkpoint_path`, `planner_gpu_id`, `log_dir`, `server_startup_timeout_s`, `model_request_timeout_s`, `replan_steps`, `generation_seed`, `policy_sampling`, `goal_refresh_mode`, `input_color_order`, `fast_inference`.

`input_color_order` is `bgr` because the released policy checkpoint was trained on BGR camera frames; ViGAR's client reverses the channels before the policy call, and the planner always receives RGB. Keep it at `bgr` for the released checkpoint. `fast_inference: true` enables ViGAR's CUDA Graph compilation and kernel fusion, about 3× faster per query and numerically equivalent up to BF16 rounding.

## Notes

- One policy server keeps one episode's subgoal state, so `eval_batch` must stay `false`.
- The planner samples with the episode seed that RoboTwin passes through `prepare_case`.
