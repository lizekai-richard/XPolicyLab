# RoboDojo cluster evaluation: runbook

How to run an evaluation campaign of a SANA checkpoint on the complete official RoboDojo protocol, exactly as every
published number of `policy/SANA_WAM`, `policy/SANA_MOT` and `policy/SANA_WAM_CAUSAL` was produced, and how to read the
results. Everything here runs on the **control host** from this directory (`scripts/RoboDojo/cluster_eval`); the GPU nodes
only execute what `deploy.sh` put on their shared home. Paths below use the site variables (`CE_*`, see
`site.env.example`).

- [0. What a campaign is](#0-what-a-campaign-is)
- [1. One-time setup of a site](#1-one-time-setup-of-a-site)
- [2. A new checkpoint](#2-a-new-checkpoint)
- [3. Launch and watch a campaign](#3-launch-and-watch-a-campaign)
- [4. When something goes wrong](#4-when-something-goes-wrong)
- [5. Results and reports](#5-results-and-reports)
- [6. Rules that keep runs comparable](#6-rules-that-keep-runs-comparable)

## 0. What a campaign is

| level | what | tool |
|---|---|---|
| unit | one task config (or a contiguous layout range of it) at one eval seed: a policy server + an Isaac client on ONE GPU | `eval/run_unit_xpl.sh` (on a node) |
| run | one checkpoint x one seed over the complete protocol: 42 tasks / 54 task configs / 2100 episodes, 95 units | `eval/master.py` driven by `eval/run.sh` |
| campaign | one or more checkpoints x seeds 0, 1, 2, one seed at a time on every free GPU, then the 3-seed tables | `eval/campaign.sh <def>` with a definition `eval/defs/*.env` |

The protocol (arXiv:2607.04434 s3.1.3, RoboDojo pinned at `09bd5f5` since 2026-10-18, `726e9aa` before): 42 tasks in 5 dimensions (Generalization 12,
Precision 8, Long-Horizon 8, Memory 6, Open 8); 50 episodes per task, a Generalization task being 25 standard + 25 on its
`_random` variant, so one seed = 54 task configs = 2100 episodes; seeds 0, 1 and 2. Success rate and score per task;
dimension = mean over its tasks; **overall = mean of the five dimension means** (the official `summarize_result.py`).
The tasks whose extra arms are all `type: support` (make_kong, play_tic_tac_toe, imitate_sorting_sequence) are always
included (`--support-arm-embodiments on`).

**Pin history.** `2184bf8` (2026-08-23, `node/install_node.sh`) -> `726e9aa` (2026-09-22, `node/node_pin_726e9aa.sh`) -> `09bd5f5`
(2026-10-12 upstream, `node/node_pin_09bd5f5.sh`). The last step is five upstream commits, of which only PR #48 (`487e425`)
changes the simulation: it makes `reset_drive_targets()` push the restored drive targets to PhysX (the same correction as
the `robodojo_spring_button_reset.patch` of the VPP2 repository). Before it, the rest pose of a SpringButton was the "sagged"
pose captured at scene setup: the button of `swap_blocks` rested at joint ratio 0.79 in every traced episode, so the release (> 0.9)
and press-transition (> 0.95) checks of its reward cannot hold and `swap_blocks` cannot succeed (SR and score 0.00 in every run
we have); `press_by_number` (buttons rest at 1.0) is unaffected. Measured on one checkpoint with the same layouts: swap_blocks 0/50 -> 23/50, press_by_number
20/50 -> 18/50 (noise). So every run made on the 726e9aa pin has `swap_blocks` = 0 (its Memory / overall numbers are low by that task's share);
runs on 09bd5f5 are not comparable with them on that task. Apply the pin with `bash node/node_pin_09bd5f5.sh`
and check it with `node/node_pin_verify.sh`.

## 1. One-time setup of a site

### 1.1 Control host

- bash, rsync, ssh keys that log into every node non-interactively (`ssh -o BatchMode=yes <user>@<node><domain>` must work).
- A python 3 with `pyyaml` (`CE_PY`).
- A RoboDojo checkout at `09bd5f5` (`CE_ROBODOJO_CHECKOUT`): the master plans the tasks from it, `summarize_result.py`
  writes the official tables from it, `deploy.sh` copies its `env_cfg/` to the shared home.
- This XPolicyLab checkout (the adapters are deployed from it).

### 1.2 Site configuration

```bash
cp site.env.example sites/<site>.env      # fill in every value; sites/mit_l.env is the reference site
ln -s sites/<site>.env site.env           # or: export CE_SITE=$PWD/sites/<site>.env
```

`site.env` is ignored by git. Every entry point sources it through `lib/site.sh`; python helpers read the exported `CE_*`
values (`lib/ce_site.py`). Node names in `CE_NODES` are in pull order (healthy nodes first): the master pulls the
workers' status files through its nodes in that order, 300 s per node that times out. `CE_RESERVE` lists boards nobody
else may lose (`node:cuda_index`, e.g. another user's service, or a board that looks idle but fails CUDA init).

### 1.3 GPU nodes (node-local environments)

Each node needs, under `CE_NODE_ROOT`:

| file | exports | built by |
|---|---|---|
| `activate.sh` | `SANA_SIM_PY` (the Isaac Sim / RoboDojo python), `ROBODOJO_CHECKOUT` (RoboDojo at `09bd5f5`, with its `Assets`) | `node/install_node.sh` (Isaac Sim 5.1 + IsaacLab + curobo, python 3.11; resumable steps, `--only <step>`), then `node/node_pin_09bd5f5.sh` (a fresh node fast-forwards straight from the install pin) |
| `activate_sana_wam.sh` | `SANA_PY` (the policy-server python) | `node/install_sana_wam.sh`, or by hand per `policy/SANA_WAM/INSTALLATION.md` section 1 (one env serves SANA_WAM, SANA_MOT, SANA_WAM_CAUSAL and SANA_WAM_SM) |

Never run two conda installers concurrently on one node. One single-GPU slot needs ~25 of 48 GB (policy server ~17 GB +
Isaac client ~8 GB); the reference site uses RTX A6000 48 GB boards.

### 1.4 Shared home

```bash
bash deploy.sh --dry-run      # what would change
bash deploy.sh                # eval/ + node/ + site.env -> CE_REMOTE_INSTALL, the XPolicyLab checkout -> CE_REMOTE_XPL,
                              # RoboDojo env_cfg/ -> <dirname CE_REMOTE_XPL>/env_cfg; then lists what is still missing
bash eval/run.sh check-sync   # must print "in sync"; every launch runs this check and refuses on any difference
```

`deploy.sh` does not install the following; it reports them as `MISS`:

- **Text encoder and VAE** (`CE_TEXT_ENCODER_PATH`, `CE_VAE_PATH`): the public mirrors `Efficient-Large-Model/gemma-2-2b-it` and
  `Efficient-Large-Model/LTX-2.3-Diffusers` (`vae/` only), see `policy/SANA_WAM/INSTALLATION.md` section 2.
- **pydeps** (`CE_REMOTE_PYDEPS`, optional): only if the policy-server env lacks the XPolicyLab server packages, put them in a
  directory that the worker prepends to the server's `PYTHONPATH`:
  `"$SANA_PY" -m pip install --target "$CE_REMOTE_PYDEPS" 'websockets>=14' msgpack msgpack-numpy h5py` (run on a node).
  Leave `CE_REMOTE_PYDEPS` empty when `SANA_PY` already has them.
- **The HF token** for private checkpoint repos: on the nodes (`HF_TOKEN_PATH` exported by `activate_sana_wam.sh`, or `HF_TOKEN`).

Re-run `deploy.sh` after every change to `eval/`, `node/`, the site file or an adapter. It refuses while a master of this
site is alive (that run would serve its remaining units with the new code); `--force` only when that is intended.

### 1.5 First light

```bash
bash eval/campaign.sh eval/defs/pantheon14k_pixel.env verify        # a checkpoint you have downloaded (section 2)
bash eval/campaign.sh eval/defs/pantheon14k_pixel.env run 0 a free-spec
bash eval/campaign.sh eval/defs/pantheon14k_pixel.env run 0 a smoke-gpu <node>:<gpu>   # insert_key x 1 on one free GPU
bash eval/campaign.sh eval/defs/pantheon14k_pixel.env smoke-check   # ~5-10 min later: 0 = passed
```

The worker builds its node-local workspace (`CE_NODE_XPL_WS`: symlinks to the pinned RoboDojo checkout, `XPolicyLab ->
CE_REMOTE_XPL`) on its first unit on a node. A passed smoke proves the whole chain: GPU, Isaac client, policy server,
adapter, checkpoint, ready-line contract.

## 2. A new checkpoint

1. **Read its card.** `config.yaml` (the resolved training yaml) carries what the definition needs:
   - `normalization_sha256` -> `NORM_<key>` (the adapter refuses a snapshot whose normalization artifact differs);
   - `data.extra.multiview` (sana_pixel / sana_latent / openwam) -> the layout tokens `EXPECT_<key>` (below);
   - `model.extra.sana_pixel_pad` (unmasked / masked), video fps / stride, rope -> more tokens;
   - the training commit -> the view resize: checkpoints trained BEFORE the upstream whole-frame stretch must be listed in
     `eval/pre_stretch_labels.txt` (served with the legacy crop, ready line `view_resize=crop (deploy)`); everything later is
     served stretched (`view_resize=stretch (default)`). Decide by commit ancestry, never by the description; add a label only
     with its evidence.
2. **Write the definition**: copy `eval/defs/TEMPLATE.env` (every variable is documented there) or the closest existing
   definition (`eval/defs/README.md` lists them). One key per checkpoint: `LABEL_<k>` (= the directory under
   `CE_CKPT_ROOT`), `HF_<k>`, `NORM_<k>`, `EXPECT_<k>`, `NAME_<k>`, `BASELINE_<k>`. Leave the sampling at the defaults
   unless the experiment is about it (section 6).
3. **Download and verify:**

   ```bash
   bash eval/campaign.sh <def> download        # detached on the gateway; wait for 'HF REPO READY' in its log
   bash eval/campaign.sh <def> verify          # sha256 against SHA256SUMS / MANIFEST.json + NORM_<k> == config.yaml
   ```

   `verify` writes `<chain dir>/verified.<label>`, which the chain requires. Re-verify after any storage incident.
4. **Check the definition:** `bash eval/campaign.sh <def> show` prints the run names, the overrides and every ready-line
   token the smoke will require; `bash eval/campaign.sh <def> run 0 a dry` plans the run locally (expect `units 95 task
   configs 54 episodes 2100`).

Layout tokens of the three SANA_WAM layout families (copy into `EXPECT_<k>`):

| family | tokens |
|---|---|
| sana_pixel canvas | `visual_layout=sana_pixel_canvas\|state_as_cross_attention=False\|sana_pixel_pad=unmasked\|video_fps=8\|video_frame_stride=4\|multiview=sana_pixel\|rope=aligned\|text_groups=1\|canvas_prompt=tiling\|robot_base_eef_layout=eef_only` |
| openwam canvas | `visual_layout=openwam_canvas\|state_as_cross_attention=False\|video_fps=8\|video_frame_stride=4\|multiview=openwam\|rope=aligned\|text_groups=1\|canvas_prompt=two_rows\|robot_base_eef_layout=eef_only` |
| sana latent strip | `visual_layout=three_view_strip\|state_as_cross_attention=False\|video_fps=8\|video_frame_stride=4\|multiview=sana_latent\|spatial_rope_tile=(8, 16)\|rope=aligned\|text_groups=4\|robot_base_eef_layout=eef_only` |

For another adapter, take the tokens from its `[<POLICY>] ready:` line of a smoke you have checked by hand.

## 3. Launch and watch a campaign

```bash
bash eval/campaign.sh <def> chain start      # detached daemon; log <CE_MASTER_ROOT>/.scratch/chain_<CAMPAIGN>/chain.log
bash eval/campaign.sh <def> status           # waited-for runs, per seed/key: live/ended, units, episodes, COMPLETE / HELD
bash eval/campaign.sh <def> watch 4          # blocks until an EVENT / ATTENTION / SUMMARY line (or 4 h), then prints it
bash eval/campaign.sh <def> run 0 a report   # one run: units per category, running units, last completions
```

What the daemon does (header of `eval/campaign.sh`):

1. **Waits** until every run in `WAIT_RUNS` is completely over. That means no live master, no worker of its run root on
   any node, and all units done. A waited-for run that has not started yet also counts as pending, so a campaign queued
   behind a 3-seed series never slips into the gap between two of its seeds.
2. **Requires** `verified.<label>` for every key.
3. **Smoke gate.** It runs one insert_key episode per key on a free GPU (ports `SMOKE_RPC/WS + i`). The adapter's ready
   line must carry every expected token (`show`). An existing `smoke_passed` skips the gate.
4. **Seeds one at a time.** Each key's run starts on its share of the free GPUs: up to `PER_RUN`, never fewer than
   `MIN_PER_RUN`, waiting `START_GRACE` for more. The run is topped up from GPUs freed later during `TOPUP_WINDOW`.
   - "Free" means: a healthy board with < 1 GiB in use, both slot ports closed, not held by a live master or its tail
     plan, and not in `RESERVE`.
   - Once the queue is empty, the **tail template** finishes the long unsharded units on idle GPUs. These are
     play_tic_tac_toe, imitate_sorting_sequence, fill_egg_holder and general_pickup, whose layouts the official client
     abandons. The template runs them as one-layout shards plus substitute layouts, then
     `eval/tail_overlap_check.py`, `tail assemble` and the run's official summary.
5. **Writes the 3-seed tables.** After the last seed: the combined official summary per key (`EVENT COMBINED`,
   `SUMMARY 3 seeds`), then `report` against `BASELINE_<k>` (`EVENT REPORT`), then `CHAIN DONE`.

Seed-0 timing on the reference site: 85-109 GPUs finish one seed in 2 h 15 min - 2 h 50 min (a server is ready in 60-80
s). Environment overrides apply to the daemon at `chain start`, e.g. `SEEDS=0 bash eval/campaign.sh <def> chain start`
stops after seed 0 (start it again later with the full `SEEDS`; finished seeds are recognized from their state).

Per-run operations go through `campaign.sh <def> run SEED KEY <verb>` (`eval/run.sh` with exactly the campaign's
settings): `status`, `report`, `watch`, `server-log UNIT`, `smoke-status`, `tail plan|status|... TASK`,
`resume` (restart a stopped master on its own slots), `official-summary`. Master state: `<CE_MASTER_ROOT>/<run>/` (`state.json`, `master.log`, `STATUS.md`, `mirror/`);
run root on the shared home: `<CE_EVAL_ROOT>/<run>/units/<unit>/a<attempt>/`.

## 4. When something goes wrong

**Holds.** Anything unexpected about a run makes the daemon write an `ATTENTION` line and create
`<chain dir>/hold.<run>`. Examples: failed units, a dead master with units not done, an unclean tail overlap check, a
failed assemble or summary. From then on nothing automatic happens to that run. Fix the cause, then move the hold file
away (`mv hold.<run> hold.<run>.released`); the daemon resumes on its next tick. Other `ATTENTION` notes
(`att.*` files, e.g. `smoke-failed`, `ckpts-not-verified`) are reported once.

| symptom | what to do |
|---|---|
| failed unit (3 attempts) | read `run SEED KEY server-log <unit>`; requeue by hand with `touch <master>/control/requeue.<unit>` (does not burn an attempt) after fixing the cause; then release the hold |
| a slot keeps failing (server_fail, CUDA "busy or unavailable" on an idle-looking board) | the master quarantines it after two infrastructure failures; add the board to `CE_RESERVE` (and to the def's `RESERVE` if set) so `free_spec` never offers it again; restart the daemon (`chain stop`, `chain start`) to pick up the change |
| a node becomes unreachable | its running units go stale after the heartbeat timeout (judged only while pulls succeed) and are TERMed and re-queued; but a slot cannot finish draining while its node is unreachable, and the unit lock keeps its unit from being dispatched elsewhere. If the node stays down: stop the master, back up `state.json`, clear those slots' `unit` / `draining` and set their units back to `queued`, then `run SEED KEY resume` (`control/drop_slots` keeps busy and draining slots, so it does not help here) |
| shared-home (NFS) outage | logins hang or reset, master pulls time out, workers keep running. Do not kill anything; probe one node at most every 20-30 min (hung logins pile up and make sshd reset everyone); wait for 2-3 clean logins in a row before acting; afterwards check GPU counts, reserved boards and stored checkpoints |
| tail shard failed repeatedly on one GPU | remove that GPU from the plan's `dropped_slots` (`<master>/tail_shards/<task>.json`, atomic write), reserve it, clear `relaunch.<run>.<task>` in the chain dir, release the hold |
| two shards drove one policy server | `tail_overlap_check.py <run>` names them; `run SEED KEY tail invalidate --layouts L1,L2 --reason TEXT` (decide from how a shard ran, never from its outcome), then `more-substitutes` |
| leaked Isaac clients after a crash | `node/cleanup_orphans_only_all.sh` (orphans only, safe while runs are live) |

**Operational rules.**

- **Changing code a live process runs.** Write a new file and `mv` it over the old one; never edit in place (bash reads a
  running script by offset). A daemon keeps its functions in memory: `chain stop`, then `chain start`.
- **Stopping a master.** `kill -TERM <master pid>` (`<master>/master.pid`) stops the scheduling only; workers keep running
  and a `--resume` picks them up. The master handles TERM after its current pull, which can take minutes during an outage.
  To cancel work: `touch <master>/control/pause`, requeue the running units, wait for STATUS.md to show no draining slot,
  then TERM the master.
- **Killing processes.** Kill by PID after reading `/proc/<pid>/cmdline`. Never use `pkill -f` / `pgrep -f` with a pattern
  that also appears in your own command line: it kills the calling shell.
- **Other users.** Never take a board in `CE_RESERVE`, and never disturb another user's processes or a live run of another
  campaign.

## 5. Results and reports

`RESULTS_FORMAT.md` documents every file. In short:

- per run: `<CE_RESULTS_ROOT>/runs/<run>/OFFICIAL_SUMMARY_<ts>.md` (the official `summarize_result.py` table over that seed)
  plus `RESULTS.md` (`summarize_eval.py`), `state.json`, `per_unit/<unit>/a<N>/_result.json` (per episode);
- per campaign key: `runs/<seed-0 run>/OFFICIAL_SUMMARY_combined_<ts>.md` (the three seeds laid out together: the
  leaderboard table, mean ± std over seeds) and `per_task_<campaign>_<key>_3seeds_<ts>.md` (`eval/three_seed_report.py`:
  task by task against the baseline, Generalization standard and random apart, Fisher exact test on pooled successes);
- ad hoc: `run SEED KEY compare` with `BASELINES="<run> ..."` (`eval/compare_runs.py`: per task, dimension means, the
  leaderboard view with the Generalization halves).

## 6. Rules that keep runs comparable

- **The complete protocol, always.** Every official run is 42 tasks / 54 configs / 2100 episodes per seed, seeds 0-2. Use
  subsets only for an explicitly requested A/B, and never put a partial run on a leaderboard.
- **Sampling defaults.** Change one only when the experiment is about it; the run name then records it.

  | setting | default | when it differs |
  |---|---|---|
  | sampling steps | 10 | |
  | video CFG / action CFG | 6 / 1 | guidance off (`VIDEO_CFG=1 ACTION_CFG=1`) for checkpoints trained without condition dropout |
  | flow shifts (video / action) | 5 / 1 (SANA_MOT: 8 / shared) | the checkpoint's own schedule |
  | actions | `ee`: end-effector poses through the evaluator's IK | |
  | chunk execution | the whole predicted chunk (32 targets) | `NAS=24` executes the first 24, then replans; such runs end in `_nas24` |
- **One checkpoint, one normalization.** `NORM_<k>` must equal the snapshot's own `normalization_sha256`.
- **View resize by commit ancestry** (section 2). `VIEW_RESIZE` only for a train/eval-mismatch A/B (`_evalcrop` /
  `_evalstretch` runs).
- **Single-GPU slots, one GPU per unit,** layout shards per `SHARD_PLAN`, categories dispatched one at a time
  (`DIM_ORDER`), the two longest units first (`PIN_LONGEST=2`). Changing any of these does not change scores, but keep
  them so wall times stay comparable.
- **Simulator cadence is part of the protocol.** The worker observes after every action (no lazy render). An optimization
  that changes when the simulator renders or settles has changed scores before (insert_key stage-1 hits 25 -> 11 of 50);
  A/B it on scores before calling it neutral.
- **Seeds.** Report 3-seed numbers as the combined summary's mean ± std over seeds and the pooled successes (/6300). Compare
  checkpoints with the Fisher exact test on pooled successes (overall: 6300 episodes; per task: 150, per Generalization
  half: 75).
