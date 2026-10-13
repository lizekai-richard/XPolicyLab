# RoboDojo cluster evaluation (SANA adapters)

The orchestration behind every RoboDojo number of the SANA adapters in this fork (`policy/SANA_WAM`, `policy/SANA_MOT`,
`policy/SANA_WAM_CAUSAL`, plus the `policy/OpenWAM` comparisons): the complete official protocol (42 tasks / 54 task
configs / 2100 episodes per seed, seeds 0-2) spread over a GPU cluster, one simulator + one policy server per GPU.

This directory is a **snapshot** of the deployment that actually runs (`~/zekail/robodojo_install` on the control host,
node-side scripts mirrored to the shared home). Refresh it with `bash sync_from_install.sh` (same exclusions as below),
scan for secrets, commit.

## Layout

| path | what |
|---|---|
| `eval/master.py` | the scheduler: plans units (task config x seed x layout shard), dispatches them to `node:gpu` slots over ssh, mirrors the workers' small status files from the shared home, re-queues stale/failed units, quarantines broken slots, writes `state.json` / `STATUS.md`; `--resume` continues a run; `control/{pause,add_slots,drop_slots}` steer a live run |
| `eval/run_unit_xpl.sh` | the node-side worker of the XPolicyLab lane: starts `setup_policy_server.py` for the adapter with the run's overrides, runs the official RoboDojo client for the unit, judges and records the result (`run_unit.sh` = the older Sana-RPC lane) |
| `eval/plan_eval.py` | the task enumeration shared by the planner and the master (support-arm embodiments, abandoned-layout substitutes) |
| `eval/start_master.sh`, `stop_eval.sh`, `status_eval.sh` | start (`--resume`) / stop / inspect a master |
| `eval/tail_shards.py`, `tail_overlap_check.py` | the tail template: when the queue is empty, long unsharded units are finished as one-layout shards plus substitutes on idle GPUs and assembled back into the run |
| `eval/summarize_eval.py`, `compare_runs.py`, `compare_partial_layouts.py`, `progress_xpl.py`, `consolidate_results.py` | per-run tables, run-vs-run comparisons, the results tree |
| `eval/three_seed_report.py` | two runs' 3-seed `OFFICIAL_SUMMARY_combined` files task by task (Gen standard / random apart, Fisher exact test) |
| `eval/decode_pred_videos.sh` | decodes `prediction_latent_dir` dumps into real-vs-predicted videos (`policy/SANA_WAM/tools/decode_prediction_latents.py`) |
| `eval/campaigns/*.sh` | one per-run driver per recipe (`<label> ... dry | smoke-gpu | start | status | tail ... | official-summary`) and the multi-seed **chains** (`*_allgpu_seeds.sh chain start|stop|status`: wait for queued-behind runs, checkpoint verified marker, seed-0 smoke gate with ready-line checks, then seeds one at a time on every free GPU, tail template, summaries) |
| `eval/pre_stretch_labels.txt` | checkpoints trained before the whole-frame stretch, served with the legacy crop |
| `node/` | node setup and housekeeping: environment installers, HF checkpoint downloads with sha256 / MANIFEST verification, policy-server start/stop, GPU / CUDA / process probes, orphan cleanup |

`eval/README.md` documents the pipeline in detail.

## Environment this is written for

Hard-wired to one cluster; adapt paths and hosts before reusing it elsewhere:

- control host (masters, chains, results): `~/zekail/robodojo_install/eval`, state under `~/robodojo_master/<run>/`;
- GPU nodes `lN.mit.edu` (RTX A6000, single-GPU slots), user `jiamingt`, shared NFS home `/homes/jiamingt`;
  node-local environments under `/dataset2/jiamingt/robodojo` (`activate_sana_wam.sh`, the RoboDojo checkout);
- workers execute the NFS copy `/homes/jiamingt/robodojo_install/eval/` -- sync node-side scripts after every edit;
- adapters from the NFS checkout `/homes/jiamingt/xpl_zekai/XPolicyLab` (drivers refuse to start when the x5 and NFS
  copies of `policy/SANA_WAM` differ);
- checkpoints at `/homes/jiamingt/checkpoints/huggingface/logits/<label>`; the HF token is read from `$HF_TOKEN_PATH`.

## Not included

Private Sana repository artefacts (git bundles, patches against Sana), old deployment kits, `.bak` backups and caches.
