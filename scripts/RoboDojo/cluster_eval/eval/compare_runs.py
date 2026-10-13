#!/usr/bin/env python3
"""compare_runs.py RUN1 RUN2 ... [--results-root CE_RESULTS_ROOT] -- per-task comparison table across runs.

Reads runs/<run>/per_unit/<unit>/a<N>/_result.json (latest complete attempt per unit; generalization standard+random
merged into one 50-episode row), prints markdown with success % / avg score % per task per run, dimension means, a
leaderboard view (runs as rows, dimensions as columns, generalization split into its standard and random halves the
way the official leaderboard shows it) and the official 'mean of dimension means' overall, and marks tasks whose
reward has no partial-credit stages (binary).
"""
import argparse, glob, json, os, sys, collections
BINARY = {"align_blocks","general_pickup","make_kong","match_and_pick_from_conveyor","pick_from_conveyor_by_image","play_Xylophone","plug_in_charger",
          "pour_balls_into_vase","pour_liquid_into_cup","press_by_number","push_T","solve_equation","sort_nesting_dolls_by_size","swap_blocks","swap_T","sweep_blocks"}
GEN = "Generalization"
DIMS = {GEN: ["stack_bowls","push_T","pack_objects_into_box","fold_clothes","hang_mugs","sweep_blocks","pour_liquid_into_cup","make_toast","arrange_largest_number","sort_nesting_dolls_by_size","store_laptop_and_headphones","stack_blocks"],
        "Precision": ["fasten_screws","plug_in_charger","insert_tubes","pour_balls_into_vase","play_Xylophone","deposit_coin","insert_key","build_tower"],
        "Long-Horizon": ["put_bottles_into_dustbin","fill_pen_holder","classify_objects","play_tic_tac_toe","fill_egg_holder","organize_table","make_kong","play_stacking_toy"],
        "Memory": ["cover_blocks","match_and_pick_from_conveyor","swap_blocks","swap_T","press_by_number","imitate_sorting_sequence"],
        "Open": ["align_blocks","general_pickup","stack_blocks_by_language","solve_equation","classify_objects_by_language","pick_from_conveyor_by_image","store_tools_in_toolbox","pour_by_language"]}
ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+", help="RUN, RUN@setting_label, RUN_A+RUN_B (merge), all composable")
ap.add_argument("--results-root", default=None, help="default CE_RESULTS_ROOT (lib/site.sh)")
ap.add_argument("--runs-subdir", default="runs", help="'runs' for the consolidated tree, '.' to read master state dirs (CE_MASTER_ROOT) directly")
ap.add_argument("--out", default=None)
a = ap.parse_args()
if a.results_root is None:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
    import ce_site  # noqa: E402  (the CE_* site configuration)
    a.results_root = ce_site.get("CE_RESULTS_ROOT")

def agg(items):
    """episodes/successes/SR%/score% over a list of (success, score); None when the half was not run."""
    n = len(items)
    if not n: return None
    s = sum(1 for x in items if x[0]); sc = sum(x[1] for x in items)
    return {"episodes": n, "successes": s, "sr": 100*s/n, "score": 100*sc/n}

def load_run(run):
    """`RUN_A+RUN_B` merges several runs into one column (e.g. a full run plus the later run of the tasks it lacked)."""
    per_task = collections.defaultdict(dict)   # canonical task -> {(task_config, layout_id): (success, score)}
    st = None
    for part in run.split("+"):
        part, _, label = part.partition("@")        # RUN@label -> only that setting's units
        rd = os.path.join(a.results_root, a.runs_subdir, part); st = json.load(open(os.path.join(rd, "state.json")))
        labels_here = sorted({u.get("setting") for u in st["units"] if u.get("setting")})
        if labels_here and not label:
            raise SystemExit(f"{part} has several settings {labels_here}; use {part}@<label>")
        if label and label not in labels_here:
            raise SystemExit(f"{part}: unknown setting label {label!r}; labels: {labels_here}")
        for u in st["units"]:
            if u["state"] != "done": continue
            if label and u.get("setting") != label: continue
            att = u["result"]["attempt"]
            f = (glob.glob(f"{rd}/per_unit/{u['unit_id']}/a{att}/_result.json")
                 or glob.glob(f"{rd}/mirror/units/{u['unit_id']}/a{att}/client/*/upstream_eval_result/_result.json")
                 or glob.glob(f"{rd}/robodojo_dry_root/units/{u['unit_id']}/a{att}/client/*/upstream_eval_result/_result.json"))
            if not f: continue
            det = json.load(open(f[0])).get("details", {})
            canon = u["task"][:-len("_random")] if u["task"].endswith("_random") else u["task"]
            for k, v in det.items():
                per_task[canon][(u["task"], int(v.get("layout_id", k)))] = (bool(v.get("success")), float(v.get("score", 0.0) or 0.0))
    out = {}
    for t, d in per_task.items():
        o = agg(list(d.values()))
        o["standard"] = agg([v for k, v in d.items() if not k[0].endswith("_random")])   # the 25 standard layouts
        o["random"] = agg([v for k, v in d.items() if k[0].endswith("_random")])         # the 25 `_random` layouts
        out[t] = o
    return out, st

data = {r: load_run(r)[0] for r in a.runs}
tasks = sorted(set().union(*[set(d) for d in data.values()]))
dim_of = {t: d for d, ts in DIMS.items() for t in ts}
md = [f"# Per-task comparison: {' vs '.join(a.runs)}", "", "success % / avg score % per task (generalization tasks = 25 standard + 25 random layouts merged; `[B]` = binary reward, no partial credit)", "",
      "| task | dim | " + " | ".join(f"{r} SR% / score%" for r in a.runs) + " |", "|---|---|" + "---|" * len(a.runs)]
for t in tasks:
    cells = []
    for r in a.runs:
        v = data[r].get(t)
        cells.append(f"{v['successes']}/{v['episodes']} = {v['sr']:.1f} / {v['score']:.1f}" if v else "-")
    md.append(f"| {t}{' [B]' if t in BINARY else ''} | {dim_of.get(t,'?')} | " + " | ".join(cells) + " |")

common = [t for t in tasks if all(t in data[r] for r in a.runs)]
half_tasks = {h: [t for t in DIMS[GEN] if t in common and all(data[r][t].get(h) for r in a.runs)] for h in ("standard", "random")}
def mean(r, ts_c, half=None):
    vals = [data[r][t][half] if half else data[r][t] for t in ts_c]
    return sum(v["sr"] for v in vals)/len(vals), sum(v["score"] for v in vals)/len(vals)

gen_common = [t for t in DIMS[GEN] if t in common]     # empty when a run did not finish the generalization dimension
if gen_common:
    md += ["", "## Generalization halves (25 standard vs 25 random layouts, as the official leaderboard splits them)", "",
           "| run | standard SR% / score% | random SR% / score% | merged SR% / score% | Δ random − standard |", "|---|---|---|---|---|"]
for r in (a.runs if gen_common else []):
    ts_m = gen_common
    cells = []
    for h in ("standard", "random"):
        cells.append(f"{mean(r, half_tasks[h], h)[0]:.2f} / {mean(r, half_tasks[h], h)[1]:.2f}" if half_tasks[h] else "-")
    msr, msc = mean(r, ts_m)
    d = (f"{mean(r, half_tasks['random'], 'random')[0] - mean(r, half_tasks['standard'], 'standard')[0]:+.2f} SR / "
         f"{mean(r, half_tasks['random'], 'random')[1] - mean(r, half_tasks['standard'], 'standard')[1]:+.2f} score") if half_tasks["standard"] and half_tasks["random"] else "-"
    md.append(f"| {r} | " + " | ".join(cells) + f" | {msr:.2f} / {msc:.2f} | {d} |")

md += ["", "## Dimension means (over tasks present in ALL listed runs; official = mean of dimension means)", "", "| dimension | " + " | ".join(f"{r} SR% / score%" for r in a.runs) + " |", "|---|" + "---|" * len(a.runs)]
overall = {r: ([], []) for r in a.runs}
dimvals = collections.defaultdict(dict)   # run -> dimension -> (sr, sc); generalization halves under 'Gen standard'/'Gen random'
for d, ts in DIMS.items():
    ts_c = [t for t in ts if t in common]
    if not ts_c: continue
    cells = []
    for r in a.runs:
        sr, sc = mean(r, ts_c)
        overall[r][0].append(sr); overall[r][1].append(sc); dimvals[r][d] = (sr, sc); cells.append(f"{sr:.2f} / {sc:.2f}")
    md.append(f"| {d} ({len(ts_c)} tasks) | " + " | ".join(cells) + " |")
    if d == GEN:
        for h in ("standard", "random"):
            if not half_tasks[h]: continue
            cells = []
            for r in a.runs:
                sr, sc = mean(r, half_tasks[h], h); dimvals[r][f"Gen {h}"] = (sr, sc); cells.append(f"{sr:.2f} / {sc:.2f}")
            md.append(f"| &nbsp;&nbsp;↳ {d} {h} ({len(half_tasks[h])} tasks x 25) | " + " | ".join(cells) + " |")
md.append("| **overall** | " + " | ".join((f"**{sum(overall[r][0])/len(overall[r][0]):.2f} / {sum(overall[r][1])/len(overall[r][1]):.2f}**" if overall[r][0] else "-") for r in a.runs) + " |")

cols = [("Gen standard", "Gen standard"), ("Gen random", "Gen random"), (GEN, "Gen merged"), ("Precision", "Precision"),
        ("Long-Horizon", "Long-Horizon"), ("Memory", "Memory"), ("Open", "Open")]
cols = [(k, lbl) for k, lbl in cols if any(k in dimvals[r] for r in a.runs)]
md += ["", "## Leaderboard view (runs as rows; SR% / score%; generalization split into standard and random)", "",
       "| run | " + " | ".join(f"{lbl}" for _, lbl in cols) + " | **overall** |", "|---|" + "---|" * (len(cols) + 1)]
for r in a.runs:
    cells = [f"{dimvals[r][k][0]:.2f} / {dimvals[r][k][1]:.2f}" if k in dimvals[r] else "-" for k, _ in cols]
    ov = f"**{sum(overall[r][0])/len(overall[r][0]):.2f} / {sum(overall[r][1])/len(overall[r][1]):.2f}**" if overall[r][0] else "-"
    md.append(f"| {r} | " + " | ".join(cells) + f" | {ov} |")

md += ["", f"tasks compared: {len(common)} common of {len(tasks)} present; total successes: " + ", ".join(f"{r}: {sum(v['successes'] for v in data[r].values())}/{sum(v['episodes'] for v in data[r].values())}" for r in a.runs)]
text = "\n".join(md) + "\n"
if a.out: open(a.out, "w").write(text)
print(text)
