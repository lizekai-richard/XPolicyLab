#!/usr/bin/env python3
"""plan_eval.py -- partition the official RoboDojo simulation evaluation into N
two-GPU jobs (one SANA policy server + one Isaac Sim client per job).

Protocol (RoboDojo paper arXiv:2607.04434 §3.1.3 and the pinned checkout's
task/RoboDojo/config/_task.yml + scripts/internal/summarize_result.py):
  * 42 canonical tasks in 5 dimensions (generalization 12, memory 6,
    long-horizon 8, precision 8, open 8), 50 episodes each = 2100 episodes;
  * a generalization task's 50 episodes are 25 on the standard layouts plus 25
    on its `<task>_random` variant, so the runnable population is 54 task
    configs: 30 x 50 + 12 x 25 (standard) + 12 x 25 (random);
  * success rate and average score; dimension = mean over its tasks; overall =
    mean of the five dimension means.
What THIS pipeline cannot run (the SANA server freezes instructions into a
manifest and serves the dual_x5 checkpoint): the 4 tasks whose instruction is
rendered dynamically and the 3 tasks pinned to the franka-competition
embodiment.  They are listed as excluded in tasks.json/PLAN.md, never silently
dropped.

A job runs its work as SESSIONS: one policy-server session per
(eval seed, episodes_per_task) group, because the server's task-list lane takes
a single seed and a single per-task episode budget.  Units (task, seed) are
LPT-packed by episode count so every job carries a near-equal load.

usage:
  plan_eval.py --checkout <RoboDojo clone> --out <RUN_ROOT> [--jobs 16]
               [--nodes l5,l7,l8,l25] [--gpus-per-node 8] [--seeds 0]
               [--no-random] [--tasks a,b,c] [--minutes-per-episode 4]
Outputs under RUN_ROOT: plan.json, tasks.json, PLAN.md, lists/*.txt
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

PLACEHOLDER = re.compile(r"<[^<>]+>")
DIM_ORDER = ["generalization", "memory", "long-horizon", "precision", "open"]


def single_template(source: str):
    """Mirror of the server launcher's gate: the task's one verbatim instruction literal, or None."""
    defs = [n for n in ast.walk(ast.parse(source))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "gen_instruction"]
    if len(defs) != 1:
        return None
    func = defs[0]
    returns = [n for n in ast.walk(func) if isinstance(n, ast.Return)]
    if len(returns) != 1:
        return None
    value = returns[0].value
    if isinstance(value, ast.Name):
        assigns = [n.value for n in func.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == value.id]
        if len(assigns) != 1:
            return None
        value = assigns[0]
    if not (isinstance(value, ast.List) and len(value.elts) == 1):
        return None
    element = value.elts[0]
    if not (isinstance(element, ast.Constant) and isinstance(element.value, str)):
        return None
    text = element.value
    if not text.strip() or PLACEHOLDER.search(text) or text not in source:
        return None
    return text



def _support_only_extra_arms(checkout: Path, robot: str) -> bool:
    """True when a non-dual_x5 robot_config is still drivable by a dual_x5 policy: exactly two x5 arms plus extra arms
    that are ALL `type: support` (the task replays their trajectory itself -- see make_kong.load_support_arm_traj --
    so the policy is never asked for their actions and the adapter, which reads left_/right_arm_joint_state BY NAME,
    is unaffected)."""
    cfg = checkout / "env_cfg/robot" / f"{robot}.yml"
    if not cfg.is_file():
        return False
    robots = (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("robots") or []
    x5 = [r for r in robots if r.get("robot_name") == "x5"]
    extra = [r for r in robots if r.get("robot_name") != "x5"]
    return len(x5) == 2 and bool(extra) and all(r.get("type") == "support" for r in extra)

def load_tasks(checkout: Path, seeds: list[int], allow_dynamic_instruction: bool = False,
               allow_support_arm_embodiments: bool = False) -> dict:
    """Enumerate the 54 task configs. allow_dynamic_instruction=True keeps the tasks whose instruction is rendered
    per episode (general_pickup, *_by_language): the OFFICIAL evaluator puts the rendered text into
    obs['instruction'] on every step, which the XPolicyLab-lane adapter reads, so only the SANA RPC lane (which
    needs one frozen prompt per task) has to exclude them."""
    inv = json.loads(subprocess.check_output(
        [sys.executable, str(checkout / "scripts/internal/task_inventory.py"), "--format", "json"], text=True))
    meta = yaml.safe_load((checkout / "task/RoboDojo/config/_task.yml").read_text(encoding="utf-8"))
    common, per = meta["common"], (meta.get("tasks") or {})
    layout_root = checkout / "Assets/Eval_Layout/RoboDojo/arx_x5"
    tasks = {}
    for rec in inv["tasks"]:
        name = rec["name"]
        t = per.get(name) or {}
        robot = t.get("robot_config", common["robot_config"])
        eval_nums = int(t.get("eval_nums", common["eval_nums"]))
        reasons = []
        if robot != "dual_x5" and not (allow_support_arm_embodiments and _support_only_extra_arms(checkout, robot)):
            reasons.append(f"embodiment {robot} (the served checkpoint drives dual_x5 only)")
        if not allow_dynamic_instruction and single_template((checkout / f"task/RoboDojo/tasks/{name}.py").read_text(encoding="utf-8")) is None:
            reasons.append("instruction rendered dynamically; no single placeholder-free template to freeze")
        if not rec.get("runnable", True):
            reasons.append("not runnable per task_inventory")
        layouts = {}
        pat = re.compile(rf"^{re.escape(name)}_(\d+)\.json$")
        for seed in seeds:
            n = sum(1 for p in (layout_root / str(seed)).glob(f"{name}_*.json") if pat.match(p.name))
            layouts[str(seed)] = n
            if n < eval_nums and not reasons:
                reasons.append(f"only {n} frozen layouts at eval seed {seed}, fewer than eval_nums={eval_nums}")
        tasks[name] = {
            "name": name, "dimension": rec["dimension"], "variant": rec["variant"],
            "base_task": name[:-len("_random")] if name.endswith("_random") else name,
            "eval_nums": eval_nums, "robot_config": robot, "layouts_per_seed": layouts,
            "runnable": not reasons, "excluded_reason": "; ".join(reasons),
        }
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkout", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="RUN_ROOT (created; must contain 'robodojo' in its path)")
    ap.add_argument("--jobs", type=int, default=None, help="default: nodes * gpus_per_node // 2")
    ap.add_argument("--nodes", default="l5,l7,l8,l25")
    ap.add_argument("--gpus-per-node", type=int, default=8)
    ap.add_argument("--seeds", default="0", help="eval layout seeds, e.g. 0 or 0,1,2 (the official protocol is one seed)")
    ap.add_argument("--no-random", action="store_true", help="drop the 12 *_random variants (NOT the official protocol)")
    ap.add_argument("--tasks", default=None, help="comma list restricting the task configs (debugging)")
    ap.add_argument("--episodes-per-task", type=int, default=None,
                    help="override every task's episode budget (smoke tests; NOT the official protocol)")
    ap.add_argument("--minutes-per-episode", type=float, default=4.0, help="for the wall-clock estimate only")
    ap.add_argument("--rpc-port-base", type=int, default=41000)
    ap.add_argument("--ws-port-base", type=int, default=42000)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    checkout = args.checkout.resolve()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    jobs_per_node = args.gpus_per_node // 2
    n_jobs = args.jobs or len(nodes) * jobs_per_node
    if n_jobs > len(nodes) * jobs_per_node:
        sys.exit(f"--jobs {n_jobs} exceeds {len(nodes)} nodes x {jobs_per_node} two-GPU slots")
    out = args.out.resolve()
    if "robodojo" not in str(out):
        sys.exit("RUN_ROOT must contain 'robodojo' (the server launcher requires it in evidence paths)")
    out.mkdir(parents=True, exist_ok=False)
    (out / "lists").mkdir()
    run_id = args.run_id or f"robodojo_eval_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    tasks = load_tasks(checkout, seeds)
    selected = {n: t for n, t in tasks.items() if t["runnable"]}
    if args.no_random:
        selected = {n: t for n, t in selected.items() if t["variant"] != "random"}
    if args.tasks:
        keep = {s.strip() for s in args.tasks.split(",") if s.strip()}
        unknown = keep - set(tasks)
        if unknown:
            sys.exit(f"unknown tasks: {sorted(unknown)}")
        selected = {n: t for n, t in selected.items() if n in keep}
    excluded = {n: t for n, t in tasks.items() if not t["runnable"]}
    if not selected:
        sys.exit("no runnable task configs selected")
    if args.episodes_per_task is not None:
        for t in selected.values():
            # upstream caps eval_num at the task's native count (main.py: min(EVAL_NUM, native)), so a
            # budget above eval_nums could never complete; cap it per task and record that.
            budget = min(args.episodes_per_task, t["eval_nums"])
            if min(t["layouts_per_seed"].values()) < budget:
                sys.exit(f"{t['name']} has fewer frozen layouts than {budget} episodes")
            t["episode_budget_note"] = f"--episodes-per-task {args.episodes_per_task} -> {budget} (native eval_nums {t['eval_nums']})"
            t["eval_nums"] = budget

    # units -> LPT bin packing
    units = [(t["name"], seed, t["eval_nums"]) for t in selected.values() for seed in seeds]
    units.sort(key=lambda u: (-u[2], u[1], u[0]))
    jobs = []
    for j in range(n_jobs):
        slot = j % jobs_per_node
        jobs.append({
            "job_id": f"job{j:02d}", "index": j, "node": nodes[j // jobs_per_node],
            "server_gpu": 2 * slot, "client_gpu": 2 * slot + 1,
            "rpc_port": args.rpc_port_base + 10 * j, "ws_port": args.ws_port_base + 10 * j,
            "episodes": 0, "_units": [],
        })
    for name, seed, w in units:
        job = min(jobs, key=lambda jb: (jb["episodes"], jb["index"]))
        job["_units"].append((name, seed, w))
        job["episodes"] += w
    for job in jobs:
        groups = defaultdict(list)
        for name, seed, w in job["_units"]:
            groups[(seed, w)].append(name)
        sessions = []
        for (seed, w), names in sorted(groups.items()):
            sid = f"s{seed}_e{w}"
            list_rel = f"lists/{job['job_id']}_{sid}.txt"
            (out / list_rel).write_text("".join(f"{n}\n" for n in sorted(names)), encoding="utf-8")
            sessions.append({"session_id": sid, "eval_seed": seed, "episodes_per_task": w,
                             "tasks": sorted(names), "task_list": list_rel,  # informational: the runner rewrites the list per round
                             "episodes": w * len(names)})
        job["sessions"] = sessions
        del job["_units"]

    total_eps = sum(jb["episodes"] for jb in jobs)
    official_eps = sum(t["eval_nums"] for t in tasks.values()) * len(seeds)
    plan = {
        "run_id": run_id, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "checkout": str(checkout), "seeds": seeds, "nodes": nodes, "gpus_per_node": args.gpus_per_node,
        "protocol": {
            "paper": "arXiv:2607.04434 §3.1.3: 42 tasks x 50 episodes = 2100; generalization = 25 standard + 25 random",
            "include_random": not args.no_random,
            "episodes_per_task_override": args.episodes_per_task,
            "task_configs_selected": len(selected), "task_configs_excluded": len(excluded),
            "episodes_planned": total_eps, "episodes_official_per_seed": official_eps // max(1, len(seeds)),
        },
        "jobs": jobs,
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "tasks.json").write_text(json.dumps(tasks, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # PLAN.md
    dim_count = Counter(t["dimension"] for t in tasks.values() if t["variant"] == "standard")
    lines = [f"# RoboDojo evaluation plan `{run_id}`", "",
             f"checkout: `{checkout}`  ", f"seeds: {seeds}  jobs: {n_jobs} on {nodes} ({jobs_per_node} two-GPU slots per node)", "",
             "## Protocol",
             f"- 42 canonical tasks: " + ", ".join(f"{d} {dim_count.get(d, 0)}" for d in DIM_ORDER),
             f"- official episodes per seed: {official_eps // max(1, len(seeds))} (42 x 50; generalization = 25 standard + 25 random)",
             f"- planned here: {total_eps} episodes over {len(seeds)} seed(s) from {len(selected)} task configs "
             f"({sum(1 for t in selected.values() if t['variant']=='standard')} standard + "
             f"{sum(1 for t in selected.values() if t['variant']=='random')} random)",
             f"- estimate at {args.minutes_per_episode:g} min/episode: {total_eps * args.minutes_per_episode / 60:.1f} GPU-pair-hours total, "
             f"~{max(jb['episodes'] for jb in jobs) * args.minutes_per_episode / 60:.1f} h wall-clock for the largest job", "",
             "## Excluded task configs (cannot be served by the SANA frozen-manifest lane)"]
    for n, t in sorted(excluded.items()):
        lines.append(f"- `{n}` ({t['dimension']}, eval_nums {t['eval_nums']}): {t['excluded_reason']}")
    lines += ["", "## Jobs", "", "| job | node | server GPU | client GPU | rpc | ws | episodes | sessions (seed/eps x tasks) |", "|---|---|---|---|---|---|---|---|"]
    for jb in jobs:
        sess = "; ".join(f"s{s['eval_seed']}/e{s['episodes_per_task']} x{len(s['tasks'])}" for s in jb["sessions"])
        lines.append(f"| {jb['job_id']} | {jb['node']} | {jb['server_gpu']} | {jb['client_gpu']} | {jb['rpc_port']} | {jb['ws_port']} | {jb['episodes']} | {sess} |")
    lines += ["", "## Task -> job", ""]
    for jb in jobs:
        for s in jb["sessions"]:
            lines.append(f"- {jb['job_id']} {s['session_id']}: " + ", ".join(s["tasks"]))
    (out / "PLAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"plan written to {out}: {n_jobs} jobs, {total_eps} episodes, {len(selected)} task configs, {len(excluded)} excluded")
    for jb in jobs:
        print(f"  {jb['job_id']} {jb['node']} gpus {jb['server_gpu']}/{jb['client_gpu']} ports {jb['rpc_port']}/{jb['ws_port']} episodes={jb['episodes']} sessions={len(jb['sessions'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
