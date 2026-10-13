# RoboDojo official evaluation, parallelised into two-GPU jobs

Protocol (paper arXiv:2607.04434 §3.1.3; pinned checkout `_task.yml`,
`scripts/internal/summarize_result.py`): 42 canonical tasks in 5 dimensions
(generalization 12, memory 6, long-horizon 8, precision 8, open 8), 50 episodes
per task = 2100 episodes; a generalization task's 50 = 25 standard + 25 on its
`<task>_random` variant (so 54 task configs: 30x50 + 12x25 + 12x25). Metrics:
success rate and average score; dimension = mean over tasks; overall = mean of
the five dimension means. Eval layouts exist for seeds 0/1/2; the official
number is one seed (the paper's "three seeds" are training seeds).

What this pipeline cannot serve (listed, never silently dropped): the 4 tasks
with dynamically rendered instructions (`classify_objects_by_language`,
`general_pickup`, `pour_by_language`, `stack_blocks_by_language`) and the 3 tasks
pinned to `dual_x5_and_franka_competition` (`imitate_sorting_sequence`,
`make_kong`, `play_tic_tac_toe`). Runnable: 35 standard + 12 random = 1750
episodes per seed.

## Master / workers mode (preferred): x5 schedules, A6000 slots execute

`master.py` (runs on x5, started by `start_master.sh RUN_NAME [args]`) owns the queue
of units (one task config at one seed with its full budget) and the worker slots
(node x GPU pair with private rpc/ws ports). A free slot gets the heaviest queued
unit via `ssh node nohup setsid run_unit.sh ...` (returns at once); `run_unit.sh`
runs server -> handoff -> benchmark client -> stop -> judge on the node and reports
through `RUN_ROOT/units/<unit>/{heartbeat,status.txt,result.json}` on the shared
home (one `a<attempt>/` directory per attempt, because the server and client
launchers refuse to reuse an evidence dir), which the master rsync-pulls with
`--delete` every `--poll` seconds via one node (falling back to the others).
Incomplete units are re-queued up to `--max-attempts`, preferring another slot;
a unit with no heartbeat for `--heartbeat-timeout` (judged only while pulls
succeed, so an outage cannot mass-kill) is TERMed and its slot drains until the
worker pid is gone before reuse; a slot failing twice in a row at the
infrastructure level (server_fail/server_timeout/port_busy/stale/aborted/
prologue_fail) is quarantined (lifted if work would otherwise stall). Master state lives
in `~/robodojo_master/RUN_NAME/` (`state.json`, `master.log`, `STATUS.md`,
`RESULTS.md`, `mirror/`); `--resume` continues after a master restart, and the
workers never depend on the master being alive. `--dry-run` exercises the whole
loop with fake local workers.

Checkpoint: `--checkpoint-root /homes/jiamingt/checkpoints/huggingface/logits/<repo>` selects the
policy snapshot every unit serves (default: the 320px epoch4 step25000 snapshot); it is recorded in
`state.json`, `STATUS.md`, every `result.json` and the RESULTS.md note, and a resumed run keeps its
own. The master checks on the shared home that `config.yaml`, `README.md`,
`model/pytorch_model_fsdp.bin` and `normalization/*.json` exist and no `*.incomplete` HF download is
in flight; `--wait-for-checkpoint SECONDS` lets it poll for that (every 60 s) while the download
still runs, so the run starts by itself once `download_weights.sh` finishes. Download a new snapshot
on any node (VAE/text encoder are skipped once present; the normalization input cache is copied from a
sibling snapshot when the repo does not ship it):

```bash
CKPT_REPO=logits/sana_rwm_policy_from_pretrained_robodojo_320px_epoch5_step32845 bash ~/robodojo_install/download_weights.sh
```

XPolicyLab lane (`--worker run_unit_xpl.sh`, 2026-09-08): the same master/worker files, but the policy is the
user's `XPolicyLab/policy/SANA_WAM` adapter (Sana-free `sana_wam_min`, SDPA attention) served in-process by the
official websocket PolicyServer, and the simulator is driven by the official `scripts/eval_policy.sh` +
`policy/SANA_WAM/deploy.py` loop (an observation after every action, no lazy render, no SANA RPC). Inputs on the
shared home: `~/xpl_zekai/XPolicyLab` (rsync of x5 `~/zekail/XPolicyLab`, no .git), `~/xpl_zekai/env_cfg` (copy of the
checkout's env_cfg, the adapter resolves `<repo>/utils/../../env_cfg`), `~/xpl_zekai/pydeps` (websockets/msgpack/
msgpack-numpy/h5py wheels the sana-wam env lacks; prepended to the server's PYTHONPATH only). The worker builds a
node-local workspace `/dataset2/jiamingt/robodojo/xpl_zekai_ws` from symlinks to the pinned checkout with
`XPolicyLab -> ~/xpl_zekai/XPolicyLab`. Use `--sec-per-step 4` (slower lane). Results land in the usual layout.

```bash
bash start_master.sh xpl_smoke_insert_key --worker run_unit_xpl.sh --tasks insert_key --episodes-per-task 1 --no-random \
  --nodes l5 --slots-per-node 1 --sec-per-step 4 --checkpoint-root /homes/jiamingt/checkpoints/huggingface/logits/<repo>
```

Other adapters on this lane: `--worker-env SANA_EVAL_XPL_POLICY=<POLICY>` selects `policy/<POLICY>` (default SANA_WAM).
`SANA_MOT` (the MoT dual-expert policy) runs in the same sana-wam env and takes the same Sana snapshot layout and
text-encoder / VAE keys, so it gets the same pydeps; `OpenWAM` gets its own bundle checks and no pydeps. For SANA_MOT add
`video_layout=<multiview|openwam_canvas>` to `--xpl-overrides`: the adapter asserts it against the checkpoint before any
weight is read (canvas yamls from before Sana rwm/mot 33b220373 carry no `video_layout` key). Driver for the NSC MoT
canvas checkpoint: `campaigns/mot_canvas_nsc_s36k.sh` (`smoke-gpu NODE:GPU` runs one episode with the policy server and
the Isaac client sharing ONE free GPU when no pair is free; it refuses a GPU a live master can use, a busy GPU, or a
listening port; `autogrow start` appends every USABLE pair that no live master holds to the run every 5 minutes).

Single-GPU slots (2026-09-19): a `--node-pairs` / `control/add_slots` entry `s<g>` (e.g. `l3:s1;l26:s0`) makes CUDA
device g run BOTH the policy server and the Isaac client (~17 + ~8 GB on a 48 GB A6000; slot id `<node>:g<g><g>`, ports
base + 50 + g, clear of the pair ports base + 0..30). For a lone free GPU whose partner is busy or broken; plain pair
indices behave exactly as before. Reading a unit's logs through the NFS view of another node can lag the writer by minutes (a live
server.stdout.log showed 0 bytes from l5 while l3 had written it); `../probe_unit_server.sh` shows the node's own view.

Speed knobs. `--prediction-videos off` is the default and behaviour-neutral. `--lazy-render` DEFAULTS TO OFF since 2026-09-09: the official loop calls get_obs() after every action and that call advances the simulator's render/settle cadence, so skipping mid-chunk observations changed the dynamics (insert_key ep7: 25/50 first-stage hits with the official cadence vs 11-12/50 with lazy render; the XPolicyLab lane gave 27-30/50). Runs official_seed0_ep10s65690 and official_seed0_ep7s45000_lr5e5_clip1 were taken with lazy render ON and understate the policy. Details:
`--prediction-videos off` drops the server's diagnostic prediction videos, whose VAE decode +
MP4 encode sat inside the timed inference (~2.7 s of ~15 s per chunk); `--lazy-render on` makes
the client render/capture cameras only at replan steps (every 24 actions), mid-chunk steps are
physics only, so episode videos become per-replan time lapses. `--policy-acceleration` selects the
rwm/zekai-merge server preset (fused/fused_fp8 need SM90; fused_portable is the Triton subset and
did not complete a forward on the A6000s in its first try).

Budgets: a failing episode runs to the task's step limit (`self.step_lim` in
`task/RoboDojo/tasks/<task>.py`, 200..1900) at ~0.85 s/step with four client+server
pairs per node, so the worker's hard watchdog is `1800 + EPS * max(600, step_lim*2.5+120)` s
and a stall watchdog TERMs a client that printed nothing for 45 min (`STALL_SECONDS`);
the master's max-runtime guard uses the same formula. Operator controls (files in
`~/robodojo_master/RUN_NAME/control/`): `requeue.<unit_id>` TERMs that unit's worker and
re-queues it without burning an attempt (also revives a permanently failed unit);
`pause` stops new dispatches while present. A re-queued unit is not dispatched until
its previous slot has drained (the unit-level lock would kill the new worker).

```bash
bash start_master.sh smoke_dyn --tasks stack_bowls,insert_key --episodes-per-task 1 --nodes l5 --slots-per-node 2
bash start_master.sh official_seed0_dyn                      # 47 units, 16 slots
bash start_master.sh official_seed0_dyn --resume             # after a master restart
bash start_master.sh official_seed0_ep5s32845 --wait-for-checkpoint 21600 \
  --checkpoint-root /homes/jiamingt/checkpoints/huggingface/logits/sana_rwm_policy_from_pretrained_robodojo_320px_epoch5_step32845
tail -f ~/robodojo_master/official_seed0_dyn/master.log ; cat ~/robodojo_master/official_seed0_dyn/STATUS.md
```

## Pieces (static plan mode)

| file | runs on | role |
|---|---|---|
| `plan_eval.py` | x5 (needs the RoboDojo checkout + a python with yaml) | enumerate task configs, LPT-pack (task, seed) units into N jobs, assign node / GPU pair / ports, write `plan.json`, `tasks.json`, `PLAN.md`, `lists/` |
| `run_eval_job.sh PLAN JOB` | each node (started by `launch_eval.sh`) | per session: server (task-list lane, `run_learned_server_only_local.sh`) -> wait for handoff -> benchmark client (`run_remote_benchmark_client.sh --keep-going`) -> stop server -> mark tasks with a complete `_result.json` DONE, re-queue the rest (up to `MAX_ROUNDS`=3); resumable |
| `launch_eval.sh RUN_ROOT [job..]` | x5 | start jobs on their nodes (nohup setsid), staggered per node |
| `stop_eval.sh RUN_ROOT [job..]` | x5 | TERM the runners (they stop client + server gracefully) |
| `status_eval.sh RUN_ROOT` | x5 | rsync-pull status files, one line per job |
| `summarize_eval.py RUN_ROOT` | x5 (on the rsync mirror) or a node | paper-style aggregation + coverage -> `RESULTS.md`, `results.json` |

## Flow

```bash
# 1. plan on x5, then copy the run dir to the shared home (RUN_ROOT must contain "robodojo")
python plan_eval.py --checkout ~/zekail/sana_wam_sim_benchmarks/RoboDojo-official-clone-20260823 \
    --out /tmp/robodojo_eval_<run> --run-id <run> --seeds 0
scp -r /tmp/robodojo_eval_<run> jiamingt@l5.mit.edu:robodojo_eval/<run>
# 2. launch (jobs run detached on l5/l7/l8/l25, 4 per node, GPUs (0,1) (2,3) (4,5) (6,7))
bash launch_eval.sh /homes/jiamingt/robodojo_eval/<run>
# 3. watch / stop / summarize
bash status_eval.sh /homes/jiamingt/robodojo_eval/<run>
bash stop_eval.sh   /homes/jiamingt/robodojo_eval/<run>
python summarize_eval.py ~/robodojo_eval_mirror/<run>
```

Per job on the node: `RUN_ROOT/jobs/<job>/{runner.log,status.txt,state/,sessions/<sid>_r<round>/{server,client}}`;
`JOB_COMPLETE` or `JOB_INCOMPLETE` (with the missing units) at the end. Re-running
`run_eval_job.sh` resumes. Smoke first: `--tasks stack_bowls,insert_key --episodes-per-task 1 --jobs 2 --nodes l5`.

Knobs: server GPU / client GPU / rpc / ws ports come from the plan; the client
watchdog is `900 + 480 * episodes_per_task` seconds per task; each job has its
own Isaac kit home under `/dataset2/jiamingt/robodojo/kit_home/<job>`.

### Policy-server overrides (`--xpl-overrides`)

`master.py --xpl-overrides "key=value key2=value2"` reaches the worker as `SANA_EVAL_XPL_EXTRA_OVERRIDES`
and is appended to `setup_policy_server.py --overrides` (whitespace-separated `key=value` tokens; values are
`literal_eval`'d, so `8` is an int). It is recorded in the START status line, STATUS.md and the RESULTS note.

Example -- receding-horizon replanning with the SANA_WAM adapter, which predicts 24 joint targets per
inference (25-frame tier) and by default executes all of them before the next inference:

    bash start_master.sh official_seed0_ep10clip1_xpl_n8 --worker run_unit_xpl.sh \
      --xpl-root /homes/jiamingt/xpl_zekai/XPolicyLab --xpl-overrides "n_action_steps=8" \
      --nodes l5,l7,l8,l25,l3,l26 --slots-per-node 4 --node-pairs "l3:1,2,3;l26:1,2,3" --sec-per-step 4 \
      --checkpoint-root /homes/jiamingt/checkpoints/huggingface/logits/<ckpt>

`n_action_steps=8` returns only the first 8 targets of each chunk, so the RoboDojo loop re-observes and
re-queries every 8 ticks (3x the inferences per episode -> raise `--sec-per-step` accordingly, e.g. 12).

### Layout shards (`--shards N`, XPL lane)

A task config's 25 or 50 evaluation layouts can be split into `N` contiguous ranges that run as independent
units on separate slots (`<task>__s<seed>__p<i>of<N>`), e.g. to finish a one-task control experiment in
`1/N` of the wall time. The worker pre-seeds the official client's resume manifest
(`eval_result/RoboDojo/<task>/<policy>/<env>/<seed>_<info>/_resume_<run_id>.json`) with every layout outside
the range marked `abandoned`, so `SeedManager.init_eval` evaluates exactly the range, in layout order, with the
same per-layout scenes as an unsharded run. `summarize_eval.py` merges the ranges by layout id and only counts a
task config as complete when all of its layouts are present. Example -- fold_clothes (25 + 25 layouts) on 10
workers, 5 layouts each:

    bash start_master.sh xpl_fold_clothes_ep7_s50_n12_sh --worker run_unit_xpl.sh \
      --xpl-root /homes/jiamingt/xpl_zekai/XPolicyLab --xpl-overrides "sampling_steps=50 n_action_steps=12" \
      --tasks fold_clothes,fold_clothes_random --shards 5 \
      --nodes l5,l8,l25 --slots-per-node 4 --node-pairs "l5:0,1,2,3;l8:0,1,2,3;l25:0,1" --sec-per-step 8 \
      --checkpoint-root /homes/jiamingt/checkpoints/huggingface/logits/<ckpt>

Per-layout results (`_result.json` details, videos) live under each shard's client dir; the DONE lines and
STATUS.md are per shard, RESULTS.md is per task config.

### Dynamic-instruction tasks (`--dynamic-instruction-tasks auto|on|off`)

general_pickup, stack_blocks_by_language, classify_objects_by_language and pour_by_language render their instruction per
episode (placeholders such as `<target>` are filled from the layout by the official description manager and delivered in
`obs["instruction"]` on every step). The SANA RPC lane needs one frozen prompt per task and therefore excludes them; the
XPolicyLab-lane adapter reads `obs["instruction"]` per inference, so with `--worker run_unit_xpl.sh` they are planned by
default (`auto` -> on): 51 task configs / 1950 episodes per seed instead of 47 / 1750 (still minus the 3 Franka-embodiment
tasks). Pass `--dynamic-instruction-tasks off` to reproduce the 47-config plan. None of the 8 Open-dimension tasks is in the
released training set (35 task dirs), so they are zero-shot for every checkpoint.

### Dispatch order: category by category (`--order dimension`, default since 2026-09-11)

Units are dispatched in list order as slots free up, so the order decides which comparisons become available first.
The default now sweeps the benchmark one capability dimension at a time: the `--pin-longest N` (default 2) units with
the largest work estimate (episodes x task step limit: fasten_screws 50x1900, play_stacking_toy 50x1200) start first so the
run's tail does not grow, then each category's units as a block (longest first), categories in `--dimension-order`
(comma list) or, for `auto`, by ascending total work -- for the 51-config plan: memory, open, precision, long-horizon,
generalization. `--order size` restores the old most-episodes-first rule. Combine with `--shard-plan
"fasten_screws=4,play_stacking_toy=2,fill_pen_holder=2,organize_table=2,classify_objects=2"` (per-task layout shards,
XPL lane) so no single unit delays its category's completion. Example (22 workers):

    bash start_master.sh official_seed0_<name> --worker run_unit_xpl.sh --xpl-root /homes/jiamingt/xpl_zekai/XPolicyLab \
      --xpl-overrides "sampling_steps=10" --order dimension --dimension-order precision,memory,open,long-horizon,generalization \
      --shard-plan "fasten_screws=4,play_stacking_toy=2" --nodes l5,l7,l8,l25,l3,l26 --slots-per-node 4 \
      --node-pairs "l3:1,2,3;l26:1,2,3" --sec-per-step 4 --checkpoint-root /homes/jiamingt/checkpoints/huggingface/logits/<ckpt>

The planned order is logged as `dispatch order: ...` in master.log; a running run's remaining order can be changed by
re-sorting `state.json["units"]` (queued entries only) and `--resume`.

### One category, several settings, all workers at once (`--dimensions` + `--settings`)

`--dimensions precision` plans only that category (comma list for several); `--settings 'label:key=v ...;label2:...'` runs
every planned task config once per setting from ONE queue, so all 22 slots are shared and the settings of a task finish
side by side. Unit ids get a `__label` suffix; tokens go to the policy server as deploy.yml overrides on top of the run-wide
`--xpl-overrides` (setting tokens win); `checkpoint_root=/abs/path` inside a setting evaluates another checkpoint (every
checkpoint is readiness-checked before dispatch). Override settings need `--worker run_unit_xpl.sh`. At the end the master
writes `RESULTS_<label>.md` per setting and `COMPARE_settings.md` (per-task table + dimension means, `compare_runs.py RUN@label`).
Example -- precision under three settings:

    bash start_master.sh sweep_precision_cfg --worker run_unit_xpl.sh --xpl-root /homes/jiamingt/xpl_zekai/XPolicyLab \
      --dimensions precision --settings "nfe10:sampling_steps=10;cfg3:sampling_steps=10 cfg_scale=3;cfg6:sampling_steps=10 cfg_scale=6" \
      --shard-plan "fasten_screws=4" --nodes l5,l7,l8,l25,l3,l26 --slots-per-node 4 --node-pairs "l3:1,2,3;l26:1,2,3" \
      --sec-per-step 4 --checkpoint-root /homes/jiamingt/checkpoints/huggingface/logits/<ckpt>

`summarize_eval.py <mirror> --setting <label>` and `compare_runs.py RUN@label [RUN@label2 ...] [--runs-subdir .]` read the
per-setting results; without `@label` both tools refuse to mix settings.

### Pausing one setting of a multi-setting run (`control/pause.<label>`)

`touch ~/robodojo_master/<run>/control/pause.fs3` stops NEW dispatches of the `fs3` units only: its running units finish,
its queued units are skipped, and every free slot goes to the other setting(s); remove the file to resume. `control/pause`
(no suffix) still pauses everything. STATUS.md shows `DISPATCH PAUSED for setting(s): ...`. Campaign drivers wrap it as
`pause-setting LABEL` / `resume-setting LABEL`; `resume` restarts a master in place (TERM, then `--resume` with its saved slots) --
needed once after a master.py change, workers are untouched.

### Growing a run onto GPU pairs another run has released (`control/add_slots`, `--resume --node-pairs`)

Two masters share the nodes as long as their GPU pairs are disjoint and their port bases differ (`--rpc-port-base
63000 --ws-port-base 64000` for the second run; the first holds 61000/62000). While a run is alive, write a pair spec
into its state dir and the master appends those pairs as new slots on its next poll (never removes or moves one):

    echo 'l8:0,2;l25:0' > ~/robodojo_master/<run>/control/add_slots     # master.log: "control: add_slots ... -> added [...]"

The same happens on `--resume` for every node new to the run and for pairs named explicitly in `--node-pairs` (plain
`--nodes` never grows a node the run already uses). Before handing a pair over, make sure no live master can still
dispatch onto it: a master whose queue is not empty may use ANY of its free slots. The clean way (since 2026-09-11) is
to make the other master FORGET the pairs first -- `control/drop_slots` removes the named pairs if they are idle
(a pair with a running unit or a draining worker is kept and reported), so that master can never dispatch onto them
again, and `control/add_slots` gives them back later:

    echo 'l5:1,2,3;l7:0' > ~/robodojo_master/<other run>/control/drop_slots   # master.log: "control: drop_slots ... -> dropped [...]"

Alternatives: pause it (`control/pause`, or `control/pause.<label>` for one setting) or wait for its queue to drain.
`campaigns/*.sh free-pairs` computes the pairs no live master on x5 can dispatch onto (occupied, draining, or any slot
of a master that still has a dispatchable queued unit; per-setting pauses are understood).

### Campaign drivers (`campaigns/`)

One script per evaluation campaign pins every knob (checkpoint, overrides, settings, ports) so the run can be started,
grown and re-checked by any session: `campaigns/s50k_cfg6_nfe10_fs3v8.sh {verify-norm|free-pairs|smoke PAIRS|dry|start PAIRS|grow PAIRS|status}`
(sft_p120k_1e-4cos_clip1_s50k, CFG 6 @ 10 NFE, flow_shift 3 vs 8 as two settings of one run, category by category).
`verify-norm` checks the checkpoint's normalization json against the sha256 in its config.yaml (the run also pins it
through the adapter's `normalization_sha256` override), `dry` plans the whole run locally, `start` refuses pairs a live
master holds and pauses the run whose pairs it takes over.
