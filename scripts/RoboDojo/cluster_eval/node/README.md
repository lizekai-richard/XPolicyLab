# RoboDojo simulator stack — MIT lN node installer

Per-node, node-local install of the RoboDojo evaluation stack described in
Sana `dev/rwm/simulation_evaluation/robodojo/INSTALL.md` (rwm/zekai @ bd9bf130d),
Sections 2–6, plus `launch/remote_client/README.md` (render probe, ws env).

## Layout (identical on every node)

```
/dataset2/jiamingt/robodojo/
  RoboDojo-official-clone-20260823/   pinned upstream checkout (+ Assets symlink)
  envs/robodojo-sana-eval/            Isaac Sim 5.1 + IsaacLab + curobo (python 3.11)
  envs/robodojo-ws/                   websockets>=14 responder interpreter
  .cache/                             pip / conda pkgs / HF / XDG caches
  logs/  state/                       per-step logs, markers, status.txt
  activate.sh                         exports SANA_SIM_PY, SANA_WS_PY, ROBODOJO_CHECKOUT, ...
~/miniconda3/envs/robodojo-sana-eval -> /dataset2/jiamingt/robodojo/envs/robodojo-sana-eval
```

The symlink in the shared home lets `conda run -n robodojo-sana-eval` (used by the
upstream doctor) resolve on every node to that node's local env.

## Run

From x5 (as yuyang):

```bash
bash ~/robodojo_install/launch_all.sh        # starts all four nodes in the background
bash ~/robodojo_install/status_all.sh        # one ssh to l5, shows every node
```

On a node directly:

```bash
bash ~/robodojo_install/install_node.sh              # resume (skips finished steps)
bash ~/robodojo_install/install_node.sh --from curobo
bash ~/robodojo_install/install_node.sh --only doctor
```

## Steps

preflight → conda_env → checkout → base_pip → torch → isaacsim → pins_isaacsim →
isaaclab → pins_isaaclab → curobo → pins_curobo → runtime_libs → import_probe →
assets → render_configs → ws_env → doctor → render_probe → activation → audit

`render_probe` is non-fatal (WARN) because it is the only step that depends on the
node's GPU actually rendering; `audit` diffs `pip freeze` against the working 5090
env (`reference_freeze_5090.txt`).

## Not covered

The SANA side (`sana-wam` env, policy server, checkpoints) — see INSTALL.md §6–7.

## Policy server without Slurm (added 2026-09-02)

`run_learned_server_only_local.sh` is `dev/rwm/simulation_evaluation/robodojo/launch/run_learned_server_only.sbatch`
with only the Slurm plumbing replaced (see its header). `start_server.sh` launches it detached
(`nohup setsid`) after sourcing `/dataset2/jiamingt/robodojo/activate_sana_wam.sh`; `stop_server.sh <evidence dir>`
asks for a graceful shutdown. Evidence + `client_handoff/` land under `~/robodojo_evidence/` (shared home).
Knobs: the sbatch's (`SANA_EVAL_TASK`, `SANA_TASK_SET=A|B` + `SANA_EPISODES_PER_TASK`, `SANA_TASK_LIST_FILE`,
`SANA_EVAL_SEED`, `SANA_RPC_PORT`, ...) plus `SANA_SERVER_GPU` (default 0) and `SANA_DRY_RUN=1`.
Validated on x5 by dry-running all three manifest lanes against the real RoboDojo checkout.
