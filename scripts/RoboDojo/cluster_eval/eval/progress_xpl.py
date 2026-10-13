#!/usr/bin/env python3
"""progress_xpl.py STATE_DIR -- intermediate results of an XPL-lane run: pulls the in-progress _result.json of every
running unit from its node's workspace (the official client rewrites it after every episode) and prints, per unit,
episodes done / successes / score_sum / stage distribution, plus the done units and a running total."""
import collections, json, re, subprocess, sys, os, tempfile
state_dir = sys.argv[1]
st = json.load(open(os.path.join(state_dir, "state.json")))
label = os.path.basename(st["checkpoint_root"].rstrip("/"))
action_type = st.get("xpl_action_type", "joint")
ws = os.environ.get("XPL_WS", "/dataset2/jiamingt/robodojo/xpl_zekai_ws")
tmp = tempfile.mkdtemp(prefix="xplprog_")
rows = []
def dist(d): return dict(sorted(collections.Counter(round(v["score"], 3) for v in d.values()).items()))
for u in st["units"]:
    if u["state"] == "done":
        r = u["result"]; rows.append((u["task"], "done", r["episodes_recorded"], u["episodes_per_task"], r["successes"], r["score_sum"], None)); continue
    if u["state"] != "running": continue
    node = u["slot"].split(":")[0]; att = u["attempts"]
    stt = os.path.join(state_dir, "mirror", "units", u["unit_id"], f"a{att}", "status.txt")
    m = None
    if os.path.exists(stt):
        m = re.search(r"run_id=(\S+)", open(stt).read())
    if not m: rows.append((u["task"], "starting", 0, u["episodes_per_task"], 0, 0.0, None)); continue
    run_id = m.group(1)
    src = f"jiamingt@{node}.mit.edu:{ws}/eval_result/RoboDojo/{u['task']}/SANA_WAM/arx_x5/{u['eval_seed']}_ckpt_name={label},action_type={action_type}/{run_id}/_result.json"
    dst = os.path.join(tmp, u["unit_id"] + ".json")
    p = subprocess.run(["rsync", "-az", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15", src, dst], capture_output=True, text=True, timeout=120)
    if p.returncode != 0 or not os.path.exists(dst): rows.append((u["task"], "running", 0, u["episodes_per_task"], 0, 0.0, None)); continue
    d = json.load(open(dst)).get("details", {})
    rows.append((u["task"], "running", len(d), u["episodes_per_task"], sum(1 for v in d.values() if v["success"]), round(sum(v["score"] for v in d.values()), 2), dist(d)))
rows.sort(key=lambda r: (-r[4], -r[5], r[0]))
print(f"{'task':34s} {'state':8s} {'eps':>7s} {'succ':>4s} {'score':>6s}  distribution")
E = S = 0; SC = 0.0
for t, s, n, tot, su, sc, di in rows:
    E += n; S += su; SC += sc
    print(f"{t:34s} {s:8s} {n:3d}/{tot:<3d} {su:4d} {sc:6.2f}  {di if di else ''}")
print(f"\nTOTAL so far: {E} episodes, {S} successes ({100*S/max(E,1):.2f}%), avg score {100*SC/max(E,1):.2f}%")
