# FlashVLA

Adapts [FlashVLA](https://github.com/zekail/flashvla) checkpoints to
XPolicyLab/RoboDojo. One adapter covers both policy types the repo trains, and
the type is read from the checkpoint's own `config.json`:

| `policy.type`   | Behaviour                                                            |
| --------------- | -------------------------------------------------------------------- |
| `pi05`          | Chunked baseline. Batched over all running envs.                      |
| `pi05-flashvla` | Action streaming. Single env only (the streaming buffer is per-env).  |

Unlike `Pi_05`, which serves OpenPI through a uv env, flashvla is an ordinary
conda env, so `setup_eval_policy_server.sh` activates `policy_conda_env` from
`deploy.yml` and puts `flashvla_repo` on `PYTHONPATH`.

## Observation and action mapping

RoboDojo's `arx_x5` config drives a `dual_x5`: `arm_dim: [6, 6]`,
`ee_dim: [1, 1]`. `pack_robot_state` packs that into the 14-dim
`[left_arm(6), left_ee(1), right_arm(6), right_ee(1)]` vector, which is exactly
the layout of `observation.state` / `action` in the exported LeRobot v3.0
dataset (`left_joint_0..6`, `right_joint_0..6`). Actions come back through
`unpack_robot_state`, so no per-task remapping is needed.

Cameras need one rename: RoboDojo publishes the overhead view as `cam_head`
while the dataset (and therefore the checkpoint) calls it `cam_high`. That is
what `camera_map` in `deploy.yml` expresses, and the adapter refuses to load if
the map does not cover exactly the checkpoint's `observation.images.*` features.

Images arrive as decoded RGB `H×W×3` uint8 — the policy server decodes them
before `update_obs*` is called. They are scaled to `[0, 1]` CHW floats here and
resized by the checkpoint's own preprocessor; do not swap channels.

## Chunking

`deploy.eval_one_episode_batch` executes a whole returned chunk open-loop, so
`get_action_batch` returns `n_action_steps` actions per env from one batched
forward. `predict_action_chunk` is called instead of `select_action` because the
policy's internal action queue has no env dimension and would leak actions
between RoboDojo's parallel envs. Set `n_action_steps` in `deploy.yml` to
shorten the open-loop horizon; `null` keeps the value the checkpoint was
configured with.

## Configuration

Point `model_path` at a run directory containing `config.json` and
`model.safetensors` — for a training run that is
`<output_dir>/checkpoints/<step>/pretrained_model`. Alternatively pass a path as
`ckpt_name`, or drop the run under `checkpoints/` in this directory and pass its
name; resolution follows `XPolicyLab/utils/checkpoint_resolver.py`.

## Install

```bash
conda activate flashvla     # created from flashvla/environment.yml
bash install.sh
```

`install.sh` installs the flashvla repo and XPolicyLab editable. It finds the
repo relative to this adapter, which sits at `sim_eval/robodojo/XPolicyLab` in
the flashvla tree; pass `FLASHVLA_REPO=/path/to/flashvla` if it lives elsewhere.

There is no `process_data.sh` or `train.sh`. FlashVLA trains from RoboDojo's
published LeRobot v3.0 export through its own `train/configs/pi05/robodojo/`, so
neither an in-adapter conversion nor a training wrapper would have anything to
do.

## Eval

Wiring check first — no simulator, so this catches everything except Isaac
itself:

```bash
EVAL_ENV_TYPE=debug bash eval.sh \
    RoboDojo stack_bowls <ckpt> arx_x5 joint 0 0 0 flashvla flashvla
```

Same machine, real simulator:

```bash
bash eval.sh RoboDojo <task> <ckpt> arx_x5 joint 0 0 1 flashvla <robodojo_env>
```

Split across machines, or when the simulator needs a container, start the halves
separately:

```bash
bash setup_eval_policy_server.sh \
    RoboDojo <task> <ckpt> arx_x5 joint 0 0 flashvla 6000 0.0.0.0
bash setup_eval_env_client.sh \
    RoboDojo <task> <ckpt> arx_x5 joint 0 1 <robodojo_env> \
    ckpt_name=<ckpt>,action_type=joint 6000 <policy_host>
```

On a cluster where Isaac Sim only runs containerised, flashvla's
`sim_eval/robodojo/slurm/sim_eval.sh` drives that split under Slurm and pyxis.
