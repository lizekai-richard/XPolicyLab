# RoboDojo cluster evaluation (SANA adapters)

The orchestration behind every RoboDojo number of the SANA adapters in this fork (`policy/SANA_WAM`, `policy/SANA_MOT`,
`policy/SANA_WAM_CAUSAL`): the complete official protocol (42 tasks / 54 task configs / 2100 episodes per seed, seeds
0-2) spread over a GPU cluster, one simulator + one policy server per GPU, scheduled from a control host over ssh, with
the results written by RoboDojo's own `summarize_result.py`.

Everything site-specific lives in one file (`sites/<site>.env`). With a site configured, a campaign is one definition file
plus a few commands:

```bash
cd scripts/RoboDojo/cluster_eval
ln -s sites/mit_l.env site.env                                # once: the site (copy site.env.example for a new one)
bash deploy.sh                                                # workers, node scripts, adapters -> the cluster's shared home
D=eval/defs/pantheon33k_latent_nas24.env                      # a campaign definition (eval/defs/README.md lists them)
bash eval/campaign.sh $D download && bash eval/campaign.sh $D verify   # checkpoint -> shared home, sha256 + normalization
bash eval/campaign.sh $D show                                 # run names, overrides, smoke tokens
bash eval/campaign.sh $D chain start                          # smoke, then seeds 0, 1, 2 on every free GPU, then the 3-seed tables
bash eval/campaign.sh $D watch                                # blocks until the next EVENT / ATTENTION line
```

- **[RUNBOOK.md](RUNBOOK.md)**: site setup, a new checkpoint, launching and watching a campaign, incidents and the rules
  that keep runs comparable.
- **[RESULTS_FORMAT.md](RESULTS_FORMAT.md)**: every file a run produces, the official and combined summaries, the 3-seed
  report and the reporting conventions.
- **[eval/README.md](eval/README.md)**: the pipeline internals (master, workers, slots, shards, budgets, control files).

## Layout

| path | what |
|---|---|
| `site.env.example`, `sites/<site>.env` | the site configuration: nodes, ssh user/domain, shared-home and node-local paths, control-host paths. `site.env` (git-ignored) is the active one, or `export CE_SITE=<file>` |
| `lib/site.sh`, `lib/ce_site.py` | load the site for the shell and python entry points |
| `lib/adapters.sh` | adapter profiles: policy dir, port ranges, worker env, synced adapter dirs, default dispatch order and flow shifts |
| `lib/slots.sh` | GPU slot helpers: free single-GPU slots (healthy, empty, ports closed, held by no live master, not reserved), live-master GPUs, dealing slots between runs |
| `deploy.sh` | control host -> shared home: `eval/` + `node/` + the site file into `CE_REMOTE_INSTALL`, this XPolicyLab checkout into `CE_REMOTE_XPL`, RoboDojo's `env_cfg/` next to it; reports what the nodes still miss |
| `eval/campaign.sh` | a multi-seed campaign from a definition file: wait for queued-behind runs, checkpoint verification gate, seed-0 smoke gate (ready-line tokens), seeds one at a time on every free GPU with top-ups, the tail template, official per-seed and 3-seed summaries, the 3-seed report; holds on anything unexpected |
| `eval/run.sh` | the per-run driver (one checkpoint, one seed) for any of the three adapters: names, check-sync, smoke, dry run, start / resume, report, tail template, official summary |
| `eval/defs/*.env` | campaign definitions: every 3-seed campaign of the Pantheon line, plus `TEMPLATE.env` |
| `eval/master.py` | the scheduler: plans units (task config x seed x layout shard), dispatches them to `node:gpu` slots over ssh, mirrors the workers' status files from the shared home, re-queues stale or failed units, quarantines broken slots; `--resume`; `control/{pause,add_slots,drop_slots,requeue.<unit>}` |
| `eval/run_unit_xpl.sh` | the node-side worker: the official XPolicyLab PolicyServer with the adapter in-process + the official RoboDojo client for one unit, judged and recorded (`run_unit.sh` = the older SANA-RPC lane) |
| `eval/plan_eval.py` | the task enumeration shared by the planner and the master |
| `eval/start_master.sh` | start a master detached (refuses a second one for the same run) |
| `eval/tail_shards.py`, `eval/tail_overlap_check.py` | the tail template: long unsharded units finished as one-layout shards plus substitutes on idle GPUs, checked and assembled back into the run |
| `eval/summarize_eval.py`, `compare_runs.py`, `compare_partial_layouts.py`, `progress_xpl.py`, `consolidate_results.py`, `three_seed_report.py` | per-run tables, run-vs-run comparisons, in-progress results, the results tree, the 3-seed report |
| `eval/decode_pred_videos.sh`, `eval/grow_single_gpu_slots.sh` | real-vs-predicted videos from `prediction_latent_dir` dumps; growing live masters onto freed GPUs |
| `eval/pre_stretch_labels.txt` | checkpoints trained before the whole-frame stretch (served with the legacy crop) |
| `eval/campaigns/` | the historical per-recipe drivers and chains that produced the published runs (hard-wired to the reference site; kept as the record, superseded by `run.sh` + `campaign.sh`) |
| `eval/legacy/` | the static-plan launcher of the first runs (two-GPU jobs from a precomputed plan) |
| `node/` | node setup and housekeeping: the RoboDojo / Isaac Sim installer, the sana-wam env installer, the RoboDojo pin, HF checkpoint downloads with sha256 / MANIFEST verification, GPU / CUDA / process probes, orphan cleanup |

## Reference site

`sites/mit_l.env`: 14 nodes of 8 RTX A6000 48 GB, a shared NFS home, node-local environments, all orchestrated from one
control host. Every published SANA number was produced there.

## Not included

Private Sana repository artefacts (git bundles, patches against Sana), old deployment kits, `.bak` backups and caches.
