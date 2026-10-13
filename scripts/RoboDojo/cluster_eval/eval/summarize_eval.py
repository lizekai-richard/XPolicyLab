#!/usr/bin/env python3
"""summarize_eval.py RUN_ROOT [--out RESULTS.md] -- aggregate a run's client
evidence into the paper's metrics.

Per task config and seed the LATEST complete upstream `_result.json`
(len(details) >= episodes_per_task, no client watchdog) is taken from
RUN_ROOT/jobs/*/sessions/*/client/<i>_<task>/upstream_eval_result/.  Then, per
seed, following scripts/internal/summarize_result.py and paper §3.1.3:
  * canonical task score = over its 50 episodes; a generalization task merges
    25 standard + 25 random-variant episodes (both must be present to count the
    task as complete);
  * success_rate = successes / episodes * 100, score = mean(score) * 100;
  * dimension = mean over its canonical tasks (only complete ones; coverage is
    reported), overall = mean of the five dimension means.
Also reports coverage: which of the 42 tasks are complete, partial, pending or
excluded by the pipeline.  Works on partial runs (progress view).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DIM_ORDER = ["generalization", "memory", "long-horizon", "precision", "open"]


def planned_units(plan: dict) -> dict:
    """{(task_config, seed): episodes_per_task} from a static plan (jobs/sessions) or a master units list."""
    eps_of = {}
    for j in plan.get("jobs", []):
        for s in j["sessions"]:
            for t in s["tasks"]:
                eps_of[(t, s["eval_seed"])] = s["episodes_per_task"]
    for u in plan.get("units", []):          # layout shards (master --shards) of one task config add up
        eps_of[(u["task"], u["eval_seed"])] = eps_of.get((u["task"], u["eval_seed"]), 0) + u["episodes_per_task"]
    return eps_of


def unit_index(plan: dict) -> dict:
    """{unit_id: unit} for master plans (empty for static launcher plans)."""
    return {u["unit_id"]: u for u in plan.get("units", []) if "unit_id" in u}


def load_plan(root: Path) -> dict:
    """plan.json (static launcher) and/or units.json (master.py) -> one dict with jobs/units/seeds/run_id/protocol."""
    plan = {"jobs": [], "units": [], "seeds": [], "run_id": root.name, "protocol": {}}
    if (root / "plan.json").exists():
        plan.update(json.loads((root / "plan.json").read_text()))
    if (root / "units.json").exists():
        plan["units"] = json.loads((root / "units.json").read_text())
    if not plan["seeds"]:
        plan["seeds"] = sorted({u["eval_seed"] for u in plan["units"]} | {s["eval_seed"] for j in plan["jobs"] for s in j["sessions"]})
    if not plan["protocol"]:
        eps = planned_units(plan)
        plan["protocol"] = {"episodes_planned": sum(eps.values()), "task_configs_selected": len({t for t, _ in eps}),
                            "task_configs_excluded": None}
    return plan


def load_results(run_root: Path, plan: dict):
    """-> {(task_config, seed): (episodes_list[(success, score)], source_dirs)}.

    Every complete client result dir (all of ITS unit's episodes recorded, no watchdog) contributes its
    episodes keyed by layout id; for a layout reported by several dirs (re-attempts) the latest run_id
    wins.  A task config counts as complete when the merged layouts cover its planned episode count --
    for an unsharded run that is exactly the old 'latest complete attempt' rule, for a --shards run the
    ranges of all shard units must be present.
    """
    eps_of = planned_units(plan)
    units = unit_index(plan)
    merged: dict = {}      # (task, seed) -> {layout_id: (order, success, score)}
    sources: dict = {}
    for res in list(run_root.glob("jobs/*/sessions/*/client/*/upstream_eval_result/_result.json")) + \
               list(run_root.glob("units/*/a*/client/*/upstream_eval_result/_result.json")) + \
               list(run_root.glob("units/*/client/*/upstream_eval_result/_result.json")):
        tdir = res.parent.parent
        task = tdir.name.split("_", 1)[1] if "_" in tdir.name else tdir.name
        try:
            details = (json.loads(res.read_text()).get("details") or {})
            rs = json.loads((tdir / "run_status.json").read_text()) if (tdir / "run_status.json").exists() else {}
        except json.JSONDecodeError:
            continue
        seed = rs.get("eval_seed")
        if seed is None:                                        # fall back to the directory names
            seed = None
            for holder in [p.name for p in tdir.parents][:4]:  # units: <task>__s<seed>[__pNofK][__label]/a<N>/client; static: s<seed>_e<eps>_r<round>/client
                try:
                    if holder in units and "eval_seed" in units[holder]:
                        seed = int(units[holder]["eval_seed"]); break
                    ms = re.search(r"__s(\d+)(?:__|$)", holder)
                    if ms:
                        seed = int(ms.group(1)); break
                    if holder.startswith("s") and "_e" in holder:
                        seed = int(holder.split("_")[0][1:]); break
                except ValueError:
                    continue
            if seed is None:
                continue
        seed = int(seed)
        unit = next((units[h.name] for h in list(tdir.parents)[:4] if h.name in units), None)
        if unit is None and (plan.get("setting_filter") or any("setting" in u for u in plan.get("units", []))):
            continue                                     # a unit of another setting (or not part of this plan)
        need = int(unit["episodes_per_task"]) if unit is not None else eps_of.get((task, seed))
        allowed = set(int(x) for x in unit["layout_ids"]) if unit is not None and unit.get("layout_ids") else None
        if need is None or rs.get("client_watchdog_fired") or len(details) < need:
            continue
        # details are keyed by the episode index; the layout evaluated is the `layout_id` field (equal to the key in an
        # unsharded run, but a layout shard's episodes 0..n-1 evaluate layouts of its range)
        rows = sorted(((int(v.get("layout_id", k)) if str(v.get("layout_id", k)).isdigit() else int(k),
                        bool(v.get("success", False)), float(v.get("score", 0.0) or 0.0))
                       for k, v in details.items() if str(k).isdigit()), key=lambda r: r[0])
        rows = [r for r in rows if r[0] in allowed] if allowed is not None else rows[:need]
        if len(rows) < need:
            continue
        order = str(rs.get("run_id") or "")                  # robodojo-remote-<ts>-<pid>: lexically sortable, survives scp
        m = merged.setdefault((task, seed), {})
        for lid, s, sc in rows:
            if lid not in m or order > m[lid][0]:
                m[lid] = (order, s, sc)
        sources.setdefault((task, seed), set()).add(str(tdir))
    out = {}
    for key, m in merged.items():
        total = eps_of.get(key)
        rows = [(m[lid][1], m[lid][2]) for lid in sorted(m)]
        if total is not None:
            if len(rows) < total:
                continue                                     # some layout shard is still missing
            rows = rows[:total]
        out[key] = (rows, "; ".join(sorted(sources[key])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="markdown output (default RUN_ROOT/RESULTS.md)")
    ap.add_argument("--run-id", default=None, help="label for the report (default: plan.json run_id, else the directory name)")
    ap.add_argument("--note", default=None, help="free-text provenance line printed under the title (checkpoint, presets, ...)")
    ap.add_argument("--setting", default=None, help="only the units of this --settings label (master multi-setting runs)")
    args = ap.parse_args()
    root = args.run_root.resolve()
    plan = load_plan(root)
    labels = sorted({u["setting"] for u in plan["units"] if u.get("setting")})
    if args.setting:
        if args.setting not in labels:
            sys.exit(f"--setting {args.setting!r} is not a label of this run; labels: {labels}")
        plan["units"] = [u for u in plan["units"] if u.get("setting") == args.setting]
        plan["setting_filter"] = args.setting
        plan["run_id"] = f"{plan['run_id']}@{args.setting}"
        plan["protocol"] = {}          # recomputed below from the filtered units
    elif labels:
        sys.exit(f"this run has several settings {labels}; pass --setting <label> (one report per setting)")
    if not plan["protocol"]:
        eps = planned_units(plan)
        plan["protocol"] = {"episodes_planned": sum(eps.values()), "task_configs_selected": len({t for t, _ in eps}),
                            "task_configs_excluded": None}
    if args.run_id:
        plan["run_id"] = args.run_id + (f"@{args.setting}" if args.setting else "")
    tasks = json.loads((root / "tasks.json").read_text())
    if plan["protocol"].get("task_configs_excluded") is None:
        plan["protocol"]["task_configs_excluded"] = sum(1 for t in tasks.values() if not t["runnable"])
    seeds = plan["seeds"]
    results = load_results(root, plan)

    canonical = sorted(n for n, t in tasks.items() if t["variant"] == "standard")
    planned = set(planned_units(plan))
    report = {"run_id": plan["run_id"], "seeds": seeds, "per_seed": {}, "coverage": {}}
    md = [f"# RoboDojo results `{plan['run_id']}`", "",
          f"episodes planned {plan['protocol']['episodes_planned']}, task configs {plan['protocol']['task_configs_selected']} "
          f"(+{plan['protocol']['task_configs_excluded']} excluded by the pipeline)", ""]
    if args.note:
        md += [args.note, ""]
    for seed in seeds:
        per_task = {}
        cov = {"complete": [], "partial": [], "pending": [], "excluded": [], "not_planned": []}
        for name in canonical:
            t = tasks[name]
            if not t["runnable"]:
                cov["excluded"].append(name); continue
            if (name, seed) not in planned:
                cov["not_planned"].append(name); continue
            parts = [name]
            if t["dimension"] == "generalization" and (f"{name}_random", seed) in planned:
                parts.append(f"{name}_random")
            rows = []
            have = [p for p in parts if (p, seed) in results]
            for p in have:
                rows.extend(results[(p, seed)][0])
            if len(have) == len(parts):
                n = len(rows); succ = sum(1 for s, _ in rows if s); sc = sum(x for _, x in rows)
                per_task[name] = {"episodes": n, "success_rate": 100.0 * succ / n, "score": 100.0 * sc / n,
                                  "dimension": t["dimension"], "parts": parts}
                cov["complete"].append(name)
            elif have:
                cov["partial"].append(name)
            else:
                cov["pending"].append(name)
        dims = {}
        for d in DIM_ORDER:
            vals = [v for v in per_task.values() if v["dimension"] == d]
            total_d = sum(1 for n in canonical if tasks[n]["dimension"] == d)
            runnable_d = sum(1 for n in canonical if tasks[n]["dimension"] == d and tasks[n]["runnable"])
            dims[d] = {"tasks_complete": len(vals), "tasks_runnable": runnable_d, "tasks_total": total_d,
                       "success_rate": statistics.mean(v["success_rate"] for v in vals) if vals else None,
                       "score": statistics.mean(v["score"] for v in vals) if vals else None}
        complete_dims = [v for v in dims.values() if v["success_rate"] is not None]
        overall = {"success_rate": statistics.mean(v["success_rate"] for v in complete_dims) if complete_dims else None,
                   "score": statistics.mean(v["score"] for v in complete_dims) if complete_dims else None,
                   "dimensions_with_data": len(complete_dims)}
        report["per_seed"][str(seed)] = {"tasks": per_task, "dimensions": dims, "overall": overall}
        report["coverage"][str(seed)] = cov

        md += [f"## eval seed {seed}", "",
               f"coverage: {len(cov['complete'])} complete, {len(cov['partial'])} partial, {len(cov['pending'])} pending, "
               f"{len(cov['excluded'])} excluded of 42 canonical tasks", "",
               "| dimension | tasks (complete/runnable/total) | success rate % | avg score % |", "|---|---|---|---|"]
        for d in DIM_ORDER:
            v = dims[d]
            sr = f"{v['success_rate']:.1f}" if v["success_rate"] is not None else "-"
            sc = f"{v['score']:.1f}" if v["score"] is not None else "-"
            md.append(f"| {d} | {v['tasks_complete']}/{v['tasks_runnable']}/{v['tasks_total']} | {sr} | {sc} |")
        osr = f"{overall['success_rate']:.1f}" if overall["success_rate"] is not None else "-"
        osc = f"{overall['score']:.1f}" if overall["score"] is not None else "-"
        md += [f"| **overall (mean of dimension means)** | {sum(v['tasks_complete'] for v in dims.values())}/"
               f"{sum(v['tasks_runnable'] for v in dims.values())}/42 | **{osr}** | **{osc}** |", "",
               "| task | dimension | episodes | success rate % | avg score % |", "|---|---|---|---|---|"]
        for name in canonical:
            if name in per_task:
                v = per_task[name]
                md.append(f"| {name} | {v['dimension']} | {v['episodes']} | {v['success_rate']:.1f} | {v['score']:.1f} |")
        if cov["partial"]:
            md += ["", "partial (one of standard/random half present): " + ", ".join(cov["partial"])]
        if cov["pending"]:
            md += ["", "pending: " + ", ".join(cov["pending"])]
        if cov["not_planned"]:
            md += ["", f"not in this plan ({len(cov['not_planned'])} runnable tasks): " + ", ".join(cov["not_planned"])]
        if cov["excluded"]:
            md += ["", "excluded by the pipeline: " + ", ".join(f"{n} ({tasks[n]['excluded_reason'].split(' (')[0].split(';')[0]})" for n in cov["excluded"])]
        md.append("")

    out = args.out or (root / "RESULTS.md")
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    (root / ("results.json" if not args.setting else f"results_{args.setting}.json")).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"[written {out} and {root / 'results.json'}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
