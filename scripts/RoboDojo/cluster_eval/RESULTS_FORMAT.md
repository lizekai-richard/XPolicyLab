# RoboDojo cluster evaluation: results format

Every table below is produced by a script in this directory, so two campaigns run from this checkout report in the same
format. The official numbers always come from RoboDojo's own `scripts/internal/summarize_result.py` (pinned checkout
`09bd5f5`; runs made on the earlier `726e9aa` pin cannot succeed on `swap_blocks`, see RUNBOOK.md "Pin history"). Nothing is hand-edited.

## Names

**Run** (one checkpoint, one eval seed), made by `eval/run.sh name`:

```
official_seed<S>_<label>_xpl_vcfg<V>_acfg<A>_nfe<N>_<vfs<v>_afs<a> | fs<v>>_<ee|joint>[_eval<crop|stretch>][_nas<K>]
```

Each field records one serving setting:

- `<label>` is the checkpoint directory under `CE_CKPT_ROOT`.
- `vcfg` / `acfg` are the video- and action-stream guidance scales.
- `nfe` is the number of sampling steps.
- `vfs` / `afs` are the video and action flow shifts. A single `fs` means a shared schedule.
- `ee` / `joint` is how actions are executed.
- `_eval<mode>` marks a view-resize A/B.
- `_nas<K>` means only the first K of the 32 predicted targets are executed before replanning.

**Unit**: `<task config>__s<seed>`. The suffixes mark how the unit was split:

- `__p<i>of<k>`: a layout shard of a long task config.
- `__sub<L>`: a tail-template shard holding exactly layout L. A layout >= the task's N is a substitute for one the
  official client abandoned.

## Where things are

| path | written by | content |
|---|---|---|
| `<CE_MASTER_ROOT>/<run>/` | `eval/master.py` | the run's live state: `state.json` (units with their attempt history, slots), `master.log`, `STATUS.md`, `RESULTS.md`, `mirror/units/<unit>/a<N>/` (the workers' small files), `tail_shards/` (tail plans, `ASSEMBLED.json`), `control/` |
| `<CE_EVAL_ROOT>/<run>/units/<unit>/a<N>/` (shared home) | `eval/run_unit_xpl.sh` | one attempt: `status.txt`, `heartbeat`, `result.json`, `resolved_inputs`, `server/` (policy-server logs; the `[<POLICY>] ready:` line), `client/<i>_<task>/` (the official client's `upstream_eval_result/_result.json`, logs, rollout videos) |
| `<CE_RESULTS_ROOT>/runs/<run>/` | `eval/consolidate_results.py` (called by `official-summary`) | the archive: `state.json`, `units.json`, `tasks.json`, `master.log`, `STATUS.md`, `RESULTS*.md`, `OFFICIAL_SUMMARY*.md`, `per_unit/<unit>/a<N>/{result.json, run_status.json, _result.json, status.txt}` |
| `<CE_RESULTS_ROOT>/INDEX.md`, `index.json` | `eval/consolidate_results.py` | one row per run: checkpoint, lane, overrides, dates, units, episodes, success / score |
| `<CE_RESULTS_ROOT>/per_task_<campaign>_<key>_3seeds_<ts>.md` | `eval/three_seed_report.py` (via `campaign.sh report`) | a key's three seeds against its baseline |
| `<CE_RESULTS_ROOT>/per_task_<run>_vs_baselines_<ts>.md` | `eval/compare_runs.py` (via `run.sh compare`) | ad-hoc run-vs-run tables |
| `<CE_MASTER_ROOT>/.scratch/chain_<campaign>/` | `eval/campaign.sh` | `chain.log` (EVENT / ATTENTION / SUMMARY lines), `verified.<label>`, `smoke_passed`, `complete.<run>` (path of the run's summary), `hold.<run>`, `progress.<run>` |

## The per-episode record

**`_result.json`** comes from the official client, one per task config and attempt.

- `details` maps an episode index (`"0"`, `"1"`, ...) to `{layout_id, success, score}` plus timings.
- `score` is the task's partial-credit stage reward in [0, 1]; `success` is the final stage.
- The client abandons a layout it cannot simulate stably and evaluates a substitute layout id >= N instead. So the
  `layout_id`s of a complete task config are not always 0..N-1.

**`result.json`** is the worker's verdict on the unit:

- `status` is one of `complete`, `incomplete`, `server_fail`, `server_timeout`, `port_busy`, `ckpt_missing`, `aborted`, or
  `prologue_fail`.
- It also records `episodes_recorded`, `successes`, `score_sum`, the node, GPU and ports, and the checkpoint.

## The official summary

`run.sh official-summary` lays the run out the way the official client would have written it, then runs RoboDojo's
`summarize_result.py`:

- It writes one `_result.json` per (task config, seed) under `eval_result/RoboDojo/<task>/<POLICY>/arx_x5/<seed>_ckpt_name=<label>,action_type=<ee|joint>/<run_id>/`.
- The layout shards of a task config are concatenated in layout order and re-keyed `0..N-1`. Substitutes stay in place of
  the layouts they replaced.
- It refuses a task config with missing episodes (`PROBLEMS`) and reports unfinished ones (`still running`). A chain holds
  the run in either case.

Its output, verbatim, is `runs/<run>/OFFICIAL_SUMMARY_<ts>.md`, with these sections:

| section | content |
|---|---|
| `## Overview (by capability dimension)` | one row per policy (`\| SANA_WAM \| ...`): Progress, then SR (%) and Score for Generalization, Precision, Long-Horizon, Memory, Open and the overall Avg = **the mean of the five dimension means**. Each cell is `mean ±std` over the seeds laid out (±0.00 for one seed). Progress reads 100% only when seeds 0, 1 and 2 are all present. |
| `## Generalization: Standard vs Random (25 episodes each)` | the 12 Generalization tasks split into their standard and `_random` halves |
| `### Per-task Standard vs Random` | per Generalization task config and seed: SR and Score on its 25 episodes |
| `## <POLICY>` | per task (42 rows; a Generalization task merged to 50 episodes) and seed: SR and Score, then Avg SR / Score with ±std |

**Combined summary.** `SUMMARY_RUNS="<seed0 run> <seed1 run> <seed2 run>" run.sh official-summary` (or
`campaign.sh <def> combined`) lays the seeds of one checkpoint into one tree. The result is
`runs/<seed-0 run>/OFFICIAL_SUMMARY_combined_<ts>.md`, whose first line names the runs. It has the same sections, with one
column pair per seed. This is the leaderboard number of a checkpoint.

**`RESULTS.md`** (`eval/summarize_eval.py`, written by the master when a run settles) is the pipeline's own per-seed table:

- coverage, per dimension and overall (mean of dimension means);
- one row per task;
- the tail units it assembled.

It must agree with the official summary of that seed.

## The 3-seed report

`eval/three_seed_report.py A B TASKS_JSON --labels a b` compares two combined summaries. `campaign.sh report` runs it with
the key's latest combined summary and `BASELINE_<key>`.

- **Header table:** for both runs, the pooled successes (/6300) and the successes per seed, the official overall SR /
  score, and every dimension's SR / score with Generalization also split into standard and random. Below it, the Fisher
  exact p on the pooled successes.
- **Tasks:** dimension by dimension, one task per row. Generalization standard and random rows are separate (25
  episodes per seed, /75 pooled); every other task is /150. Each row gives:
  - the successes per seed (`s0/s1/s2`) for both runs;
  - the pooled sums and their difference;
  - the two-sided Fisher exact p.

Successes are recovered from the per-seed SR columns of the official summaries; the dimension and overall numbers are
copied from their overview tables.

## Reporting conventions

These hold for every comparison made from these files:

- **Report the official overall** (the mean of the five dimension means), SR and score, as the 3-seed mean ± std, with the
  pooled successes out of 6300.
- **Go dimension by dimension, one task per row.** Always split Generalization into its standard and random halves, SR and
  score each. The merged 50-episode Generalization rows hide that the random half is usually near zero.
- **Test significance** with the Fisher exact test on pooled successes: overall 6300 vs 6300, per task 150, per
  Generalization half 75. Say "n.s." rather than reading direction into p > 0.05.
- **Keep the leaderboard to complete runs.** Partial runs (a category subset, a seed missing) never go on it. Seed-0-only
  numbers are labelled as such.
- **Never pool seeds with `compare_runs.py A+B`.** That form merges runs of ONE seed (a main run plus a back-fill) and
  deduplicates by unit.
