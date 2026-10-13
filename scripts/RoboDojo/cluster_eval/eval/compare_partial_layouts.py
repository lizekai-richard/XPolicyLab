#!/usr/bin/env python3
"""compare_partial_layouts.py AB_STATE_DIR BASE_STATE_DIR BASE_LABEL [KEEP_DIR] -- layout-matched comparison of an XPL-lane run in
progress against a finished baseline run: for every unit of the A/B run (running units: the in-progress _result.json
is rsync-pulled from the node's workspace like progress_xpl.py does; done units: the mirrored copy) the episodes it has
finished are paired BY LAYOUT ID with the baseline unit of the same task (`<task>__s<seed>__<label>`; sharded A/B
units all map to the one unsharded baseline unit) and stage-1 hits (score > 0), score sums and successes are compared
on exactly those layouts. Prints one row per task plus a total; nothing else is written."""
import collections, json, os, re, subprocess, sys, tempfile
ab_dir, base_dir, label = sys.argv[1:4]
keep_dir = sys.argv[4] if len(sys.argv) > 4 else None     # optional: keep the pulled in-progress _result.json files here (<unit_id>.json)
ws = os.environ.get("XPL_WS", "/dataset2/jiamingt/robodojo/xpl_zekai_ws")
ab = json.load(open(os.path.join(ab_dir, "state.json"))); base = json.load(open(os.path.join(base_dir, "state.json")))
ab_label = os.path.basename(ab["checkpoint_root"].rstrip("/"))
ab_action_type = ab.get("xpl_action_type", "joint")
tmp = keep_dir or tempfile.mkdtemp(prefix="cmpl_")
os.makedirs(tmp, exist_ok=True)

def result_path(state_dir, u):
    att = u["result"]["attempt"] if u.get("result") else u["attempts"]
    return os.path.join(state_dir, "mirror", "units", u["unit_id"], f"a{att}", "client", f"0_{u['task']}", "upstream_eval_result", "_result.json")

def details_of(path):
    if not os.path.exists(path): return {}
    return {int(v.get("layout_id", k)): (bool(v.get("success")), float(v.get("score", 0.0) or 0.0)) for k, v in json.load(open(path)).get("details", {}).items()}

base_by_task = {}
for u in base["units"]:
    if u.get("setting") == label and u["state"] == "done":
        base_by_task.setdefault(u["task"], {}).update(details_of(result_path(base_dir, u)))

per_task = collections.defaultdict(dict)   # task -> layout -> (success, score) of the A/B run
for u in ab["units"]:
    if u["state"] == "done":
        per_task[u["task"]].update(details_of(result_path(ab_dir, u)))
    elif u["state"] == "running":
        stt = os.path.join(ab_dir, "mirror", "units", u["unit_id"], f"a{u['attempts']}", "status.txt")
        m = re.search(r"run_id=(\S+)", open(stt).read()) if os.path.exists(stt) else None
        if not m: continue
        node = u["slot"].split(":")[0]
        src = f"jiamingt@{node}.mit.edu:{ws}/eval_result/RoboDojo/{u['task']}/SANA_WAM/arx_x5/{u['eval_seed']}_ckpt_name={ab_label},action_type={ab_action_type}/{m.group(1)}/_result.json"
        dst = os.path.join(tmp, u["unit_id"] + ".json")
        p = subprocess.run(["rsync", "-az", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15", src, dst], capture_output=True, text=True, timeout=120)
        if p.returncode == 0 and os.path.exists(dst):
            per_task[u["task"]].update(details_of(dst))

print(f"{'task':22s} {'layouts':>7s} | {'A/B hits':>8s} {'score':>6s} {'succ':>4s} | {'base hits':>9s} {'score':>6s} {'succ':>4s}   (same layouts; hits = score > 0)")
T = collections.Counter(); TS = collections.Counter()
for task in sorted(per_task):
    a, b = per_task[task], base_by_task.get(task, {})
    common = sorted(set(a) & set(b))
    if not common: continue
    ah = sum(1 for l in common if a[l][1] > 0); bh = sum(1 for l in common if b[l][1] > 0)
    asc = sum(a[l][1] for l in common); bsc = sum(b[l][1] for l in common)
    asu = sum(1 for l in common if a[l][0]); bsu = sum(1 for l in common if b[l][0])
    n = len(common)
    print(f"{task:22s} {n:7d} | {ah:8d} {asc:6.2f} {asu:4d} | {bh:9d} {bsc:6.2f} {bsu:4d}")
    T.update(layouts=n, ah=ah, asu=asu, bh=bh, bsu=bsu); TS.update(); TS["asc"] += asc; TS["bsc"] += bsc
n = T["layouts"] or 1
print(f"{'TOTAL':22s} {T['layouts']:7d} | {T['ah']:8d} {TS['asc']:6.2f} {T['asu']:4d} | {T['bh']:9d} {TS['bsc']:6.2f} {T['bsu']:4d}   "
      f"avg score A/B {100*TS['asc']/n:.2f}% vs base {100*TS['bsc']/n:.2f}%")
