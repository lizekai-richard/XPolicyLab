#!/usr/bin/env python3
"""consolidate_results.py [--master-root ~/robodojo_master] [--out ~/zekail/robodojo_results] [--videos]

Collect every evaluation run's results from the master state dirs into ONE tree under ~/zekail (re-runnable):
  <out>/index.json, INDEX.md                    one row per run: checkpoint, lane/worker, overrides, dates, counts, scores
  <out>/runs/<run>/RESULTS*.md OFFICIAL_SUMMARY*.md STATUS.md state.json units.json tasks.json master.log
  <out>/runs/<run>/per_unit/<unit>/a<N>/{result.json, run_status.json, _result.json}   per-episode details
  <out>/runs/<run>/videos/...                   (with --videos) copies of the sim videos already pulled to x5
For runs that are still running, a RESULTS_partial_<ts>.md is produced with summarize_eval.py.
"""
import argparse, collections, datetime, glob, json, os, re, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--master-root", default=os.path.expanduser("~/robodojo_master"))
ap.add_argument("--out", default=os.path.expanduser("~/zekail/robodojo_results")); ap.add_argument("--videos", action="store_true")
a = ap.parse_args()
ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M")
os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)

def category(run, s):
    if run.startswith(("smoke", "xpl_smoke")) or "smoke" in run: return "smoke"
    if run.startswith("dry") or run == "robodojo_dry": return "dry-run (fake results)"
    if run.startswith("latency"): return "latency probe"
    if run.startswith("official_"): return "official protocol"
    return "control experiment"

def overrides_of(run_dir):
    log = os.path.join(run_dir, "master.log")
    if not os.path.exists(log): return ""
    m = re.search(r"overrides '([^']*)'", open(log, errors="replace").read())
    return m.group(1) if m else ""

def flags_of(run_dir):
    log = os.path.join(run_dir, "master.log")
    if not os.path.exists(log): return ""
    txt = open(log, errors="replace").read()
    m = re.search(r"policy acceleration preset '([^']*)', prediction videos (\w+), lazy render (\w+)", txt)
    return f"accel={m.group(1)} pred_videos={m.group(2)} lazy_render={m.group(3)}" if m else ""

rows = []
for run in sorted(os.listdir(a.master_root)):
    rd = os.path.join(a.master_root, run)
    if not os.path.isdir(rd): continue
    dst = os.path.join(a.out, "runs", run); os.makedirs(dst, exist_ok=True)
    for f in glob.glob(f"{rd}/RESULTS*.md") + glob.glob(f"{rd}/OFFICIAL_SUMMARY*.md") + [f"{rd}/{x}" for x in ("STATUS.md", "state.json", "units.json", "tasks.json", "excluded.json", "master.log")]:
        if os.path.exists(f): shutil.copy2(f, dst)
    sp = os.path.join(rd, "state.json")
    if not os.path.exists(sp):
        rows.append({"run": run, "category": "aborted before planning", "note": "no state.json"}); continue
    s = json.load(open(sp))
    units = s["units"]; c = collections.Counter(u["state"] for u in units)
    done = [u for u in units if u["state"] == "done"]
    eps = sum(u["result"]["episodes_recorded"] for u in done); succ = sum(u["result"]["successes"] for u in done)
    sc = sum(u["result"]["score_sum"] for u in done)
    # per-unit evidence: every attempt dir that has a result.json
    nunit = 0
    for u in units:
        for adir in sorted(glob.glob(f"{rd}/mirror/units/{u['unit_id']}/a*") + glob.glob(f"{rd}/robodojo_dry_root/units/{u['unit_id']}/a*")):
            rj = os.path.join(adir, "result.json")
            if not os.path.exists(rj): continue
            tgt = os.path.join(dst, "per_unit", u["unit_id"], os.path.basename(adir)); os.makedirs(tgt, exist_ok=True)
            shutil.copy2(rj, tgt); nunit += 1
            for f in glob.glob(f"{adir}/client/*/run_status.json") + glob.glob(f"{adir}/client/*/upstream_eval_result/_result.json") + [f"{adir}/status.txt"]:
                if os.path.exists(f): shutil.copy2(f, tgt)
    # partial RESULTS for unfinished runs (the master only writes RESULTS.md when everything settled)
    unfinished = any(u["state"] in ("running", "queued") for u in units)
    if unfinished and done and os.path.isdir(f"{rd}/mirror") and os.path.exists(f"{rd}/units.json"):
        for f in ("units.json", "tasks.json"):
            if os.path.exists(f"{rd}/{f}"): shutil.copy2(f"{rd}/{f}", f"{rd}/mirror/{f}")
        out_md = os.path.join(dst, f"RESULTS_partial_{ts}.md")
        note = (f"PARTIAL snapshot {ts} UTC: {c.get('done',0)}/{len(units)} units done; checkpoint `{s.get('checkpoint_root')}`; "
                f"worker `{s.get('worker')}`" + (f"; overrides `{overrides_of(rd)}`" if overrides_of(rd) else "") + f"; {flags_of(rd)}")
        subprocess.run([sys.executable, os.path.join(HERE, "summarize_eval.py"), f"{rd}/mirror", "--out", out_md, "--run-id", run, "--note", note],
                       capture_output=True, text=True)
    if a.videos and os.path.isdir(f"{rd}/videos"):
        shutil.copytree(f"{rd}/videos", os.path.join(dst, "videos"), dirs_exist_ok=True)
    hist = [h["dispatched"] for u in units for h in u.get("history", []) if isinstance(h.get("dispatched"), str) and h.get("dispatched")]
    fin = [h.get("finished") for u in units for h in u.get("history", []) if isinstance(h.get("finished"), str) and h.get("finished")]
    rows.append({"run": run, "category": category(run, s), "checkpoint": os.path.basename((s.get("checkpoint_root") or "epoch4_step25000 (default)").rstrip("/")),
                 "worker": s.get("worker", "run_unit.sh"), "lane": "XPolicyLab" if (s.get("worker") or "").endswith("xpl.sh") else "SANA RPC",
                 "overrides": overrides_of(rd), "flags": flags_of(rd), "first_dispatch": min(hist)[:16] if hist else "", "last_finish": max(fin)[:16] if fin else "",
                 "units": dict(c), "units_total": len(units), "episodes_done": eps, "successes": succ, "success_rate_pct": round(100*succ/eps, 2) if eps else None,
                 "avg_score_pct": round(100*sc/eps, 2) if eps else None, "results_files": sorted(os.path.basename(p) for p in glob.glob(f"{dst}/RESULTS*.md") + glob.glob(f"{dst}/OFFICIAL*.md")),
                 "per_unit_attempts_copied": nunit, "videos": os.path.isdir(os.path.join(dst, "videos"))})
json.dump({"generated": ts, "master_root": a.master_root, "runs": rows}, open(os.path.join(a.out, "index.json"), "w"), indent=1)
md = [f"# RoboDojo evaluation results index (generated {ts} UTC)", "", "Source of truth: the master state dirs under `~/robodojo_master` on x5; this tree is a re-runnable copy",
      "(`python3 ~/zekail/robodojo_install/eval/consolidate_results.py --videos`). Per-episode details: `runs/<run>/per_unit/<unit>/a<N>/_result.json`.", "",
      "| run | category | checkpoint | lane | overrides | units (done/total) | episodes | successes | success % | avg score % | files |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if "units" not in r: md.append(f"| {r['run']} | {r['category']} | | | | | | | | | {r['note']} |"); continue
    md.append(f"| {r['run']} | {r['category']} | {r['checkpoint']} | {r['lane']} | {r['overrides']} | {r['units'].get('done',0)}/{r['units_total']} | {r['episodes_done']} | {r['successes']} | "
              f"{'' if r['success_rate_pct'] is None else r['success_rate_pct']} | {'' if r['avg_score_pct'] is None else r['avg_score_pct']} | {', '.join(r['results_files'])} |")
open(os.path.join(a.out, "INDEX.md"), "w").write("\n".join(md) + "\n")
print(f"{len(rows)} runs -> {a.out}"); print("\n".join(md[6:]))
