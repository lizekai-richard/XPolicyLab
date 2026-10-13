#!/usr/bin/env python3
"""tail_shards.py -- finish the long UNSHARDED tail units of a running RoboDojo master in parallel (user template, 2026-09-25).

Some task configs cannot be layout-sharded up front because the official client abandons layouts ("Unstable") and then
evaluates substitutes >= N instead; which layouts it abandons depends on the policy for play_tic_tac_toe and
imitate_sorting_sequence, and a pre-planned shard holding one can never complete. They run as ONE sequential unit and
become the run's tail (4-8 h). Once the queue is empty and GPUs sit idle, this tool splits the rest of such a unit:

  plan      read-only: the sequential unit's live progress (recorded / abandoned layouts, the layout in progress) and the cut
  launch    drop idle slots from the master (control/drop_slots) and start ONE-LAYOUT shard units for layouts cut..N-1, plus
            substitute shards for ids N, N+1, ... (one per layout abandoned so far + --spare), through the campaign
            driver's `smoke-gpu` (its GPU-empty / port / live-slot checks) into the run's own run root
  status    coverage: every layout 0..N-1 exactly once (the sequential part below the cut, one shard each above it),
            abandoned layouts counted, and the first U stable substitutes in id order (U = number abandoned) -- the
            episode set an unsharded run would have produced; says what is still missing or which substitute to add
  more-substitutes   start the next substitute ids when some substitute was itself abandoned (also: layouts not started yet,
            crashed shards); each GPU at most once per call
  invalidate --layouts L1,L2 --reason TEXT   discard shards that did not run under the protocol (e.g. two shards on one GPU
            drove one policy server, 2026-09-28): unit dirs -> <run_root>/invalid_tail_shards/, layouts re-queued for
            more-substitutes; decide from how a shard ran, never from its outcome. Check first: eval/tail_overlap_check.py RUN
  fill      a PRE-PLANNED layout shard that ended "incomplete" because the client abandoned layouts of its range (PhysX broken
            "[PhysX] ... abandon seeds=[13]" or unstable; a shard cannot refill from ids >= N): keep that attempt's recorded
            episodes and add substitute shards N, N+1, ... (--unit UID [--attempt 1] [--adopt 25,26,27]); fill-status checks it
  (assemble handles both. A sequential unit that completed on its own before the assembly keeps its complete, unsharded
   result -- that IS the official run -- and its tail shards are discarded.)
  assemble  once every OTHER unit of the run is done and coverage is complete: pause + stop the master, TERM only the
            sequential unit's worker (its node keeps other units), copy its partial upstream result (node-local) into
            the run root, cut it at the cut, rewrite state.json / units.json (backups *.orig_before_tail_assembly) so
            the task config = sequential part + shards + chosen substitutes, mirror the units, rewrite RESULTS.md.
            Then run the driver's `official-summary` (it concatenates the parts per task config in layout order).

The cut is fixed when the shards are launched (the layout after the one in progress), so no result decides which copy
of a layout counts: the sequential unit's layouts at or above the cut are discarded even if it finished them.

usage: tail_shards.py --run RUN --driver campaigns/<driver>.sh --task TASK [--task TASK2] <plan|launch|status|more-substitutes|invalidate|fill|fill-status|assemble>
State per task: ~/robodojo_master/<RUN>/tail_shards/<task>.json.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
EVAL = os.path.dirname(os.path.abspath(__file__))
NFS_HOST = f"jiamingt@{os.environ.get('NFS_GATEWAY', 'l7')}.mit.edu"   # l5 degraded 2026-10-12 (load ~700, NFS D-state)
NODE_WS = os.environ.get("SANA_EVAL_XPL_WS", "/dataset2/jiamingt/robodojo/xpl_zekai_ws")  # run_unit_xpl.sh XPL_WS (node-local); a driver serving another adapter copy exports its own workspace
PORT_BASE_RPC, PORT_BASE_WS = 62700, 64700                    # tail shards: 62700 + g / 64700 + g on a node (no campaign uses it)
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]


def now() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def sh(cmd: list[str], timeout: int = 300, check: bool = False) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        sys.exit(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p


def ssh(node: str, script: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return sh(SSH + [f"jiamingt@{node}.mit.edu", script], timeout=timeout)


class Run:
    def __init__(self, run: str, driver: str, seed: int | None = None):
        self.run = run
        self.state_dir = os.path.join(HOME, "robodojo_master", run)
        self.run_root = f"/homes/jiamingt/robodojo_eval/{run}"
        self.driver = os.path.abspath(driver)
        self.tail_dir = os.path.join(self.state_dir, "tail_shards")
        os.makedirs(self.tail_dir, exist_ok=True)
        self.scratch = os.path.join(self.tail_dir, "pulled")
        os.makedirs(self.scratch, exist_ok=True)
        if not os.path.isfile(os.path.join(self.state_dir, "state.json")):
            sys.exit(f"no master state for {run} at {self.state_dir}")
        if not os.path.isfile(self.driver):
            sys.exit(f"driver {self.driver} not found")
        # the run's eval seed: unit ids are <task>__s<seed>[__p..|__sub..] and node-local results sit under <seed>_<info>/
        seeds = sorted({int(u["eval_seed"]) for u in self.state()["units"]})
        if seed is None and len(seeds) != 1:
            sys.exit(f"{run} holds seeds {seeds}; pass --seed")
        self.seed = int(seed) if seed is not None else seeds[0]

    # -- master state ---------------------------------------------------------------------------------------------------
    def state(self) -> dict:
        return json.load(open(os.path.join(self.state_dir, "state.json")))

    def master_alive(self) -> bool:
        try:
            os.kill(int(open(os.path.join(self.state_dir, "master.pid")).read().strip()), 0)
            return True
        except Exception:
            return False

    def policy(self) -> str:
        m = re.search(r"^POLICY=(\w+)", open(self.driver).read(), re.M)       # the driver's adapter dir segment
        return m.group(1) if m else "SANA_WAM"

    def unit(self, uid: str) -> dict:
        for u in self.state()["units"]:
            if u["unit_id"] == uid:
                return u
        sys.exit(f"unit {uid} not in {self.run}")

    def control(self, name: str, content: str, wait: int = 120) -> None:
        cdir = os.path.join(self.state_dir, "control"); os.makedirs(cdir, exist_ok=True)
        tmp = os.path.join(cdir, name + ".tmp"); open(tmp, "w").write(content + "\n"); os.replace(tmp, os.path.join(cdir, name))
        t = 0
        while os.path.exists(os.path.join(cdir, name)) and t < wait:
            time.sleep(3); t += 3
        if os.path.exists(os.path.join(cdir, name)):
            sys.exit(f"the master did not consume control/{name} within {wait}s")

    # -- files on NFS / nodes -------------------------------------------------------------------------------------------
    def pull(self, remote: str, local: str, host: str = NFS_HOST, extra: list[str] | None = None) -> bool:
        os.makedirs(os.path.dirname(local.rstrip("/")) or ".", exist_ok=True)
        return sh(["rsync", "-a", *(extra or []), f"{host}:{remote}", local], timeout=600).returncode == 0

    def status_txt(self, uid: str, attempt: int = 1) -> str:
        local = os.path.join(self.scratch, uid, f"a{attempt}", "status.txt")
        self.pull(f"{self.run_root}/units/{uid}/a{attempt}/status.txt", local)
        return open(local).read() if os.path.exists(local) else ""


# -- the sequential unit's live progress ---------------------------------------------------------------------------------
def seq_progress(r: Run, task: str) -> dict:
    uid = f"{task}__s{r.seed}"
    u = r.unit(uid)
    att = (u.get("result") or {}).get("attempt") or len(u.get("history") or []) or 1
    st = r.status_txt(uid, att)
    m = re.search(r"run_id=(\S+)", st)
    if not m:
        sys.exit(f"{uid}: no run_id in its status.txt yet")
    rid = m.group(1)
    slot = u.get("slot") or ""
    node = slot.split(":")[0] if slot else (re.search(r"node=(\S+)", st).group(1) if re.search(r"node=(\S+)", st) else "")
    base = f"{NODE_WS}/eval_result/RoboDojo/{task}/{r.policy()}/arx_x5"
    local = os.path.join(r.scratch, f"seq_{task}")
    os.makedirs(local, exist_ok=True)
    got = sh(["rsync", "-a", f"jiamingt@{node}.mit.edu:{base}/{r.seed}_*/{rid}/_result.json", f"jiamingt@{node}.mit.edu:{base}/{r.seed}_*/_resume_{rid}.json",
              local + "/"], timeout=300)
    res = os.path.join(local, "_result.json")
    details = json.load(open(res)).get("details", {}) if os.path.exists(res) else {}
    resume = {}
    rf = os.path.join(local, f"_resume_{rid}.json")
    if os.path.exists(rf):
        resume = json.load(open(rf))
    recorded = {int(v.get("layout_id", k)): v for k, v in details.items()}
    unstable_n = int(resume.get("unstable_nums", 0) or 0)
    attempted_max = max(recorded) if recorded else -1
    # layouts below the highest recorded one that are not recorded were abandoned (the client goes in id order)
    abandoned = sorted(set(range(attempted_max + 1)) - set(recorded))
    return {"unit_id": uid, "state": u["state"], "node": node, "slot": slot, "run_id": rid, "attempt": att, "recorded": recorded,
            "abandoned": abandoned, "unstable_nums": unstable_n, "total": int(u.get("layout_total") or u["episodes_per_task"]),
            "rsync_ok": got.returncode == 0, "node_result_base": base}


def abandoned_in_log(logs: str) -> set[int]:
    """Layouts the official client abandoned mid-run because PhysX broke ("[PhysX] broken envs=[..] -> abandon seeds=[13]; refill
    from queue=[14]"): policy-dependent, NOT counted in "Unstable nums"; an unsharded client refills from ids >= N instead."""

    out: set[int] = set()
    for m in re.finditer(r"abandon seeds=\[([0-9, ]*)\]", re.sub(r"\x1b\[[0-9;]*m", "", logs)):
        out.update(int(x) for x in m.group(1).replace(" ", "").split(",") if x)
    return out


def plan_path(r: Run, task: str) -> str:
    return os.path.join(r.tail_dir, f"{task}.json")


def load_plan(r: Run, task: str) -> dict:
    p = plan_path(r, task)
    if not os.path.exists(p):
        sys.exit(f"no tail plan for {task} (run launch first): {p}")
    return json.load(open(p))


def save_plan(r: Run, task: str, plan: dict) -> None:
    json.dump(plan, open(plan_path(r, task), "w"), indent=1)


# -- idle slots and shard dispatch -----------------------------------------------------------------------------------
def idle_slots(r: Run, avoid: set[str]) -> list[tuple[str, int]]:
    s = r.state()
    busy = {sl["slot_id"] for sl in s["slots"] if sl.get("unit") or sl.get("draining")}
    out = [(sl["node"], int(sl["server_gpu"])) for sl in s["slots"]
           if sl["slot_id"] not in busy and sl["server_gpu"] == sl["client_gpu"] and sl["node"] not in avoid]
    per_node: dict[str, int] = {}
    for n, _ in out:
        per_node[n] = per_node.get(n, 0) + 1
    return sorted(out, key=lambda x: (-per_node[x[0]], x[0], x[1]))          # fill whole idle nodes first


def stale_result(res: dict, shard: dict) -> bool:
    """A result.json written by an attempt started well before this shard's (re)launch belongs to an earlier attempt."""
    try:
        started = datetime.datetime.fromisoformat(res["started"].replace("Z", "+00:00")).replace(tzinfo=None)
        launched = datetime.datetime.strptime(shard["launched"], "%Y-%m-%dT%H:%M:%SZ")
    except (KeyError, ValueError, TypeError, AttributeError):
        return False
    return started < launched - datetime.timedelta(seconds=120)


def owned_gpus(plan: dict) -> list[tuple[str, int]]:
    """Every GPU this plan dropped from the master (plan['dropped_slots'] = 'l7:s0,s3;l3:s2;...'), each once."""
    out: list[tuple[str, int]] = []
    for part in filter(None, plan.get("dropped_slots", "").split(";")):
        node, gs = part.split(":")
        for g in filter(None, gs.split(",")):
            k = (node, int(g.lstrip("s")))
            if k not in out:
                out.append(k)
    return out


def dispatch(r: Run, task: str, plan: dict, todo: list[int], free: list[tuple[str, int]], total: int) -> None:
    """Start one shard per layout of `todo` on the GPUs of `free`, each GPU at most ONCE per call: on 2026-09-28 a GPU that
    appeared twice in the list got two shards 4 s apart (the GPU-empty and port checks pass before the first worker has
    loaded), the second server failed to bind the shared port and both clients drove ONE policy server. A worker that
    exits at once or a unit with a live runner is not retried on other GPUs (each smoke-gpu attempt used to rm -rf a1/)."""
    seen: set[tuple[str, int]] = set()
    free = [x for x in free if not (x in seen or seen.add(x))]
    for lay in todo:
        ok, msg = False, "no GPU left"
        while free and not ok:
            n, g = free.pop(0)
            ok, msg = launch_shard(r, task, lay, n, int(g), total)
            print(f"   layout {lay} -> {n}:{g} | {msg}")
            if ok:
                plan["shards"][str(lay)] = {"unit_id": f"{task}__s{r.seed}__sub{lay}", "node": n, "gpu": int(g), "launched": now(),
                                            "kind": "layout" if lay < total else "substitute"}
            elif "live runner" in msg or "died" in msg:
                free.insert(0, (n, g))
                break
        if not ok:
            print(f"   layout {lay}: NOT started ({msg})")


def launch_shard(r: Run, task: str, layout: int, node: str, gpu: int, total: int) -> tuple[bool, str]:
    uid = f"{task}__s{r.seed}__sub{layout}"
    env = dict(os.environ, RUN=r.run, SMOKE_RUN_ROOT=r.run_root, SMOKE_UNIT_ID=uid, SMOKE_SEED=str(r.seed), SMOKE_LAYOUT_IDS=str(layout),
               SMOKE_LAYOUT_TOTAL=str(total), SMOKE_RPC_PORT=str(PORT_BASE_RPC + gpu), SMOKE_WS_PORT=str(PORT_BASE_WS + gpu))
    p = subprocess.run(["bash", r.driver, "smoke-gpu", f"{node}:{gpu}", task, "1"], capture_output=True, text=True, env=env, timeout=300)
    out = (p.stdout + p.stderr).strip().splitlines()
    msg = next((l for l in out if "dispatched" in l or "REFUSE" in l or "died" in l or "FAIL" in l.upper()), out[-1] if out else "")
    return ("dispatched" in msg), msg


def cmd_plan(r: Run, task: str, a) -> None:
    sp = seq_progress(r, task)
    rec = sorted(sp["recorded"])
    cut = (max(rec) + 2) if rec else 1
    print(f"== {sp['unit_id']} ({sp['state']}) on {sp['slot']} run_id {sp['run_id']}")
    print(f"   recorded {len(rec)} (succ {sum(bool(v.get('success')) for v in sp['recorded'].values())}), abandoned so far {sp['abandoned']}"
          f" (client unstable_nums {sp['unstable_nums']}), highest recorded layout {max(rec) if rec else None}")
    print(f"   proposed cut {cut}: the sequential unit keeps 0..{cut - 1} (the layout after the last recorded one is in progress);"
          f" shards {cut}..{sp['total'] - 1} ({sp['total'] - cut} one-layout units) + {len(sp['abandoned'])} substitute(s) from {sp['total']}")


def cmd_launch(r: Run, task: str, a) -> None:
    if os.path.exists(plan_path(r, task)) and not a.force:
        sys.exit(f"a tail plan for {task} exists already ({plan_path(r, task)}); use status / more-substitutes")
    if any(u["state"] == "queued" for u in r.state()["units"]) and not a.force:
        sys.exit("the run still has queued units -- tail shards are for the tail, after the queue is empty (or --force)")
    sp = seq_progress(r, task)
    if sp["state"] != "running":
        sys.exit(f"{sp['unit_id']} is {sp['state']}, not running")
    rec = sorted(sp["recorded"])
    cut = a.cut if a.cut is not None else ((max(rec) + 2) if rec else 1)
    total = sp["total"]
    layouts = list(range(cut, total))
    subs = list(range(total, total + len(sp["abandoned"]) + a.spare))
    want = layouts + subs
    avoid = set(filter(None, (a.avoid or "").split(",")))
    cand = idle_slots(r, avoid)
    if len(cand) < len(want):
        print(f"WARNING: {len(cand)} idle slots for {len(want)} shards; the rest must be started later with more-substitutes / relaunch")
    take = cand[:len(want)]
    if not take:
        sys.exit("no idle slot to use")
    spec = ";".join(f"{n}:" + ",".join(f"s{g}" for _, g in sorted(x for x in take if x[0] == n)) for n in dict.fromkeys(n for n, _ in take))
    if a.dry_run:
        print(f"[dry-run] cut {cut}, shards {layouts}, substitutes {subs}\n[dry-run] would drop slots: {spec}")
        for lay, (n, g) in zip(want, take):
            print(f"[dry-run]   layout {lay} -> {n}:{g} ports {PORT_BASE_RPC + g}/{PORT_BASE_WS + g}")
        return
    r.control("drop_slots", spec)
    plan = {"task": task, "run": r.run, "seq_unit": sp["unit_id"], "seq_node": sp["node"], "seq_slot": sp["slot"], "seq_run_id": sp["run_id"],
            "seq_attempt": sp["attempt"], "cut": cut, "total": total, "created": now(), "dropped_slots": spec,
            "seq_recorded_at_launch": rec, "seq_abandoned_at_launch": sp["abandoned"], "shards": {}}
    dispatch(r, task, plan, want, list(take), total)
    save_plan(r, task, plan)
    print(f"plan saved: {plan_path(r, task)}")


def shard_results(r: Run, plan: dict) -> dict[int, dict]:
    """layout -> {'state': running|recorded|abandoned, 'row': detail or None, 'result': result.json}"""
    out = {}
    for lay, sh_ in plan["shards"].items():
        uid = sh_["unit_id"]
        local = os.path.join(r.scratch, uid)
        # --delete: a relaunch recreates a1/ on NFS; without it the local copy kept the FAILED attempt's result.json and the
        # relaunched, running shard read as failed again -> relaunched over and over (2026-09-28)
        r.pull(f"{r.run_root}/units/{uid}/a1/", local + "/", extra=["--delete", "--exclude", "*.mp4", "--exclude", "server/", "--exclude", "*.npz", "--exclude", "*.png"])
        rj = os.path.join(local, "result.json")
        if not os.path.exists(rj):
            out[int(lay)] = {"state": "running", "row": None, "result": None}; continue
        res = json.load(open(rj))
        if stale_result(res, sh_):
            out[int(lay)] = {"state": "running", "row": None, "result": None}; continue
        det = {}
        f = glob.glob(os.path.join(local, "client", "*", "upstream_eval_result", "_result.json"))
        if f:
            det = json.load(open(f[0])).get("details", {})
        rows = [v for v in det.values() if int(v.get("layout_id", -1)) == int(lay)]
        logs = "".join(open(f, errors="replace").read() for f in glob.glob(os.path.join(local, "client", "*", "client.stdout.log")))
        unstable = re.findall(r"Unstable nums: (\d+)", logs)
        if rows and res.get("episodes_recorded", 0) >= 1:
            out[int(lay)] = {"state": "recorded", "row": rows[0], "result": res}
        elif res.get("episodes_recorded", 0) == 0 and ((unstable and int(unstable[-1]) > 0) or int(lay) in abandoned_in_log(logs)):
            out[int(lay)] = {"state": "abandoned", "row": None, "result": res}     # the client abandoned it (Unstable / PhysX broken)
        else:
            out[int(lay)] = {"state": "failed", "row": None, "result": res}         # crashed / no verdict: relaunch it
    return out


def coverage(r: Run, task: str, plan: dict, sp: dict | None = None) -> dict:
    sp = sp or seq_progress(r, task)
    cut, total = plan["cut"], plan["total"]
    seq_rec = {k: v for k, v in sp["recorded"].items() if k < cut}
    seq_hi = max(sp["recorded"]) if sp["recorded"] else -1
    seq_done = seq_hi >= cut - 1 or sp["state"] == "done"
    seq_abandoned = [k for k in range(min(cut, seq_hi + 1)) if k not in sp["recorded"]]
    shards = shard_results(r, plan)
    missing = [l for l in range(cut, total) if str(l) not in plan["shards"]]
    layout_state = {l: shards[l]["state"] for l in range(cut, total) if l in shards}
    shard_abandoned = [l for l, s_ in layout_state.items() if s_ == "abandoned"]
    pending = [l for l, s_ in layout_state.items() if s_ in ("running", "failed")]
    U = len(seq_abandoned) + len(shard_abandoned)
    subs = sorted(l for l in shards if l >= total)
    chosen, blocked = [], None
    for l in subs:                                   # the first U stable substitutes, strictly in id order
        if len(chosen) >= U:
            break
        s_ = shards[l]["state"]
        if s_ == "recorded":
            chosen.append(l)
        elif s_ == "abandoned":
            continue
        else:
            blocked = l; break
    need_more = max(0, U - len(chosen)) if blocked is None else 0
    complete = seq_done and not missing and not pending and blocked is None and len(chosen) == U
    return {"cut": cut, "total": total, "seq_recorded": sorted(seq_rec), "seq_done": seq_done, "seq_abandoned": seq_abandoned,
            "shards": shards, "missing_shards": missing, "pending": pending, "shard_abandoned": shard_abandoned, "U": U,
            "substitutes_available": subs, "substitutes_chosen": chosen, "blocked_by": blocked, "need_more_substitutes": need_more,
            "failed": [l for l, s_ in shards.items() if s_["state"] == "failed"], "complete": complete, "seq": sp}


def cmd_status(r: Run, task: str, a) -> bool:
    plan = load_plan(r, task)
    c = coverage(r, task, plan)
    succ = sum(bool(c["seq"]["recorded"][k].get("success")) for k in c["seq_recorded"])
    succ += sum(bool(c["shards"][l]["row"].get("success")) for l in c["shards"] if c["shards"][l]["state"] == "recorded"
                and (l < c["total"] or l in c["substitutes_chosen"]))
    n_eps = len(c["seq_recorded"]) + sum(1 for l in range(c["cut"], c["total"]) if c["shards"].get(l, {}).get("state") == "recorded") + len(c["substitutes_chosen"])
    print(f"== {task}: cut {c['cut']} | sequential {'passed the cut' if c['seq_done'] else 'NOT yet at the cut'}: {len(c['seq_recorded'])} recorded below it,"
          f" abandoned {c['seq_abandoned']}")
    states = {}
    for l, s_ in sorted(c["shards"].items()):
        states.setdefault(s_["state"], []).append(l)
    print(f"   shards by state: " + ", ".join(f"{k} {v}" for k, v in sorted(states.items())))
    print(f"   abandoned in shards {c['shard_abandoned']} -> U = {c['U']} substitutes needed; chosen (first stable in id order) {c['substitutes_chosen']}"
          + (f"; waiting for substitute {c['blocked_by']}" if c["blocked_by"] is not None else "")
          + (f"; NEED {c['need_more_substitutes']} MORE (more-substitutes)" if c["need_more_substitutes"] else ""))
    if c["missing_shards"] or c["failed"]:
        print(f"   NOT STARTED {c['missing_shards']}  FAILED {c['failed']} (relaunch)")
    print(f"   -> {n_eps}/{c['total']} episodes, {succ} successes; COVERAGE {'COMPLETE' if c['complete'] else 'incomplete'}")
    return c["complete"]


def cmd_more_substitutes(r: Run, task: str, a) -> None:
    plan = load_plan(r, task)
    c = coverage(r, task, plan)
    want = max(c["need_more_substitutes"], a.count or 0)
    todo = []
    for l in c["failed"]:                              # relaunch crashed shards first
        todo.append(l)
    nxt = max([c["total"] - 1] + [int(x) for x in plan["shards"]]) + 1
    todo += list(range(nxt, nxt + want))
    todo += [l for l in c["missing_shards"]]
    if not todo:
        print("nothing to start"); return
    avoid = set(filter(None, (a.avoid or "").split(",")))
    # GPUs this plan already owns (dropped from the master) are reused first -- each ONCE, and only when the last shard
    # launched on it has finished -- then more idle slots
    last_on: dict[tuple[str, int], tuple[str, int]] = {}
    for l, s_ in plan["shards"].items():
        k = (s_["node"], int(s_["gpu"]))
        if k not in last_on or s_["launched"] > last_on[k][0]:
            last_on[k] = (s_["launched"], int(l))
    busy = {k for k, (_, l) in last_on.items() if c["shards"].get(l, {}).get("state") == "running"}
    kept = {(sl["node"], int(sl["server_gpu"])) for sl in r.state()["slots"]}      # a drop the master refused (slot was busy)
    own = [k for k in owned_gpus(plan) if k not in busy and k not in kept]
    free = own + [x for x in idle_slots(r, avoid) if x not in own and x not in busy]
    extra = [x for x in free[:len(todo)] if x not in own]
    if extra and not a.dry_run:
        spec = ";".join(f"{n}:" + ",".join(f"s{g}" for _, g in sorted(x for x in extra if x[0] == n)) for n in dict.fromkeys(n for n, _ in extra))
        r.control("drop_slots", spec); plan["dropped_slots"] += ";" + spec
    if a.dry_run:
        print(f"[dry-run] todo {todo}\n[dry-run] free GPUs in order {free}\n[dry-run] busy {sorted(busy)}"); return
    dispatch(r, task, plan, todo, free, c["total"])
    save_plan(r, task, plan)


def cmd_invalidate(r: Run, task: str, a) -> None:
    """Discard shard layouts whose episodes are not valid runs of the protocol -- e.g. two shards started on one GPU drove
    ONE policy server (2026-09-28) -- decided from how they ran, never from their outcome: move units/<uid> out of units/
    (kept under invalid_tail_shards/), forget the local copy, drop the layout from the plan (-> NOT STARTED, so
    more-substitutes runs it again) and record it under plan['invalidated']. Refuses a layout whose runner is alive."""
    plan = load_plan(r, task)
    lays = [int(x) for x in (a.layouts or "").split(",") if x.strip()]
    if not lays or not a.reason:
        sys.exit("invalidate needs --layouts L1,L2,... and --reason TEXT")
    tag = re.sub(r"[^A-Za-z0-9]+", "_", a.reason)[:40].strip("_")
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    for lay in lays:
        uid = f"{task}__s{r.seed}__sub{lay}"
        src, dst = f"{r.run_root}/units/{uid}", f"{r.run_root}/invalid_tail_shards/{uid}.{tag}_{ts}"
        if a.dry_run:
            print(f"[dry-run] layout {lay}: mv {src} {dst}; drop plan entry {plan['shards'].get(str(lay))}"); continue
        p = ssh(NFS_HOST.split("@")[1].split(".")[0],
                f"set -e; [ -d {src} ] || {{ echo missing; exit 0; }}; flock -n {src}/.lock -c true || {{ echo RUNNER_ALIVE; exit 3; }}; "
                f"mkdir -p {r.run_root}/invalid_tail_shards; mv {src} {dst}; echo moved", timeout=120)
        if p.returncode != 0 or "RUNNER_ALIVE" in p.stdout:
            print(f"   layout {lay}: NOT invalidated ({(p.stdout + p.stderr).strip()[:200]})"); continue
        shutil.rmtree(os.path.join(r.scratch, uid), ignore_errors=True)
        plan.setdefault("invalidated", []).append({"layout": lay, "shard": plan["shards"].pop(str(lay), None), "moved_to": dst,
                                                   "reason": a.reason, "at": now()})
        print(f"   layout {lay}: {p.stdout.strip()} -> {dst}")
    if not a.dry_run:
        save_plan(r, task, plan)


# -- fill: pre-planned layout-shard units left incomplete by abandoned layouts ----------------------------------------------
def fill_path(r: Run, task: str) -> str:
    return os.path.join(r.tail_dir, f"{task}.fill.json")


def attempt_result(r: Run, uid: str, att: int) -> tuple[dict, dict, list[int], set[int]]:
    """(result.json, details, recorded layouts, layouts the client abandoned per its log) of one attempt of a unit."""

    local = os.path.join(r.scratch, uid, f"a{att}")
    r.pull(f"{r.run_root}/units/{uid}/a{att}/", local + "/", extra=["--exclude", "*.mp4", "--exclude", "server/", "--exclude", "*.npz", "--exclude", "*.png"])
    rj = os.path.join(local, "result.json")
    res = json.load(open(rj)) if os.path.exists(rj) else {}
    f = glob.glob(os.path.join(local, "client", "*", "upstream_eval_result", "_result.json"))
    det = json.load(open(f[0])).get("details", {}) if f else {}
    logs = "".join(open(x, errors="replace").read() for x in glob.glob(os.path.join(local, "client", "*", "client.stdout.log")))
    return res, det, sorted(int(v.get("layout_id", k)) for k, v in det.items()), abandoned_in_log(logs)


def cmd_fill(r: Run, task: str, a) -> None:
    """A unit of a layout-sharded task config ended 'incomplete' because the client abandoned layouts of its range (PhysX
    broken / unstable) and a shard cannot draw the substitutes >= N an unsharded client would: keep that attempt's recorded
    episodes and add substitute shards N, N+1, ... (first U stable in id order, U = abandoned count, over all such units
    of the task config). --adopt IDS registers substitute shards already started by hand instead of launching."""

    if os.path.exists(fill_path(r, task)) and not a.force:
        sys.exit(f"a fill plan for {task} exists already ({fill_path(r, task)})")
    if not a.unit:
        sys.exit("fill needs --unit UID (one per incomplete unit of the task config)")
    parts, total = [], None
    for uid in a.unit:
        u = r.unit(uid)
        total = int(u.get("layout_total") or u["episodes_per_task"])
        res, det, rec, ab = attempt_result(r, uid, a.attempt)
        planned = [int(x) for x in (u.get("layout_ids") or range(total))]
        missing = sorted(set(planned) - set(rec))
        unexplained = sorted(set(missing) - ab)
        print(f"== {uid} attempt {a.attempt}: {res.get('status')} recorded {len(rec)}/{len(planned)}, missing {missing}, abandoned per client log {sorted(ab)}")
        if unexplained and not a.force:
            sys.exit(f"{uid}: missing layouts {unexplained} are not abandonments in the client log (a crash?) -- rerun instead")
        parts.append({"unit_id": uid, "attempt": a.attempt, "layout_ids": planned, "recorded": rec, "abandoned": missing})
    U = sum(len(p_["abandoned"]) for p_ in parts)
    plan = {"kind": "fill", "task": task, "run": r.run, "total": total, "created": now(), "parts": parts, "U": U, "shards": {}}
    if a.adopt:
        for lay in [int(x) for x in a.adopt.split(",") if x.strip()]:
            plan["shards"][str(lay)] = {"unit_id": f"{task}__s{r.seed}__sub{lay}", "node": "?", "gpu": -1, "launched": "adopted", "kind": "substitute"}
        print(f"   adopted substitute shards {sorted(int(x) for x in plan['shards'])} (U = {U})")
    else:
        subs = list(range(total, total + U + a.spare))
        cand = idle_slots(r, set(filter(None, (a.avoid or "").split(","))))[:len(subs)]
        if len(cand) < len(subs):
            sys.exit(f"{len(cand)} idle slots for {len(subs)} substitute shards")
        spec = ";".join(f"{n}:" + ",".join(f"s{g}" for _, g in sorted(x for x in cand if x[0] == n)) for n in dict.fromkeys(n for n, _ in cand))
        if a.dry_run:
            print(f"[dry-run] substitutes {subs} on {cand}; would drop {spec}"); return
        r.control("drop_slots", spec)
        for lay, (n, g) in zip(subs, cand):
            ok, msg = launch_shard(r, task, lay, n, g, total)
            print(f"   substitute {lay} -> {n}:{g} | {msg}")
            if ok:
                plan["shards"][str(lay)] = {"unit_id": f"{task}__s{r.seed}__sub{lay}", "node": n, "gpu": g, "launched": now(), "kind": "substitute"}
    json.dump(plan, open(fill_path(r, task), "w"), indent=1)
    print(f"fill plan saved: {fill_path(r, task)}")


def fill_coverage(r: Run, task: str, plan: dict) -> dict:
    shards = shard_results(r, plan)
    chosen, blocked = [], None
    for l in sorted(shards):
        if len(chosen) >= plan["U"]:
            break
        st_ = shards[l]["state"]
        if st_ == "recorded":
            chosen.append(l)
        elif st_ == "abandoned":
            continue
        else:
            blocked = l; break
    complete = blocked is None and len(chosen) == plan["U"]
    return {"shards": shards, "chosen": chosen, "blocked_by": blocked, "complete": complete,
            "need_more": 0 if blocked is not None else max(0, plan["U"] - len(chosen))}


def cmd_fill_status(r: Run, task: str) -> bool:
    plan = json.load(open(fill_path(r, task)))
    c = fill_coverage(r, task, plan)
    states = {}
    for l, s_ in sorted(c["shards"].items()):
        states.setdefault(s_["state"], []).append(l)
    rec = sum(len(p_["recorded"]) for p_ in plan["parts"])
    print(f"== {task} (fill): parts " + ", ".join(f"{p_['unit_id']} a{p_['attempt']} {len(p_['recorded'])} rec, abandoned {p_['abandoned']}" for p_ in plan["parts"]))
    print(f"   substitutes by state: " + ", ".join(f"{k} {v}" for k, v in sorted(states.items())) + f" -> need {plan['U']}, chosen {c['chosen']}"
          + (f"; waiting for {c['blocked_by']}" if c["blocked_by"] is not None else "") + (f"; NEED {c['need_more']} MORE" if c["need_more"] else ""))
    print(f"   COVERAGE {'COMPLETE' if c['complete'] else 'incomplete'} ({rec} recorded in the parts + {len(c['chosen'])} substitutes)")
    return c["complete"]


def assemble_fill(r: Run, task: str, units: list, plan: dict, c: dict) -> str:
    for p_ in plan["parts"]:
        u = next(x for x in units if x["unit_id"] == p_["unit_id"])
        att = p_["attempt"]
        res, det, rec, _ = attempt_result(r, p_["unit_id"], att)
        assert rec == p_["recorded"], (p_["unit_id"], "recorded layouts changed")
        local = os.path.join(r.state_dir, "mirror", "units", p_["unit_id"], f"a{att}")
        r.pull(f"{r.run_root}/units/{p_['unit_id']}/a{att}/", local + "/", extra=["--exclude", "*.mp4", "--exclude", "server/"])
        succ = sum(bool(v.get("success")) for v in det.values()); score = round(sum(float(v.get("score", 0) or 0) for v in det.values()), 4)
        for h in u.get("history") or []:
            if h.get("attempt", 0) > att and h.get("status") in (None, "running"):
                h["status"] = "stopped_superseded"; h["finished"] = now()
        u.update(state="done", episodes_per_task=len(rec),
                 result={"attempt": att, "status": "complete_with_substitutes", "episodes_recorded": len(rec), "successes": succ, "score_sum": score},
                 note=f"attempt {att} kept ({len(rec)} recorded; the client abandoned {p_['abandoned']}); substitutes {c['chosen']} (tail_shards.py fill)")
        u.pop("slot", None)
    for l in c["chosen"]:
        sh_ = plan["shards"][str(l)]; res = c["shards"][l]["result"]; suid = sh_["unit_id"]
        r.pull(f"{r.run_root}/units/{suid}/a1/", os.path.join(r.state_dir, "mirror", "units", suid, "a1") + "/", extra=["--exclude", "*.mp4", "--exclude", "server/"])
        if any(x["unit_id"] == suid for x in units):
            continue
        ref = next(x for x in units if x["unit_id"] == plan["parts"][0]["unit_id"])
        units.append({"unit_id": suid, "task": task, "eval_seed": ref["eval_seed"], "dimension": ref["dimension"], "variant": ref.get("variant"),
                      "episodes_per_task": 1, "layout_ids": [l], "layout_total": plan["total"], "state": "done", "attempts": 1,
                      "history": [{"attempt": 1, "dispatched": res.get("started"), "finished": res.get("finished"), "status": "complete",
                                   "episodes": 1, "successes": res.get("successes", 0), "score_sum": res.get("score_sum", 0)}],
                      "result": {"attempt": 1, "status": "complete", "episodes_recorded": 1, "successes": res.get("successes", 0), "score_sum": res.get("score_sum", 0)},
                      "note": "substitute layout for a layout the client abandoned (tail_shards.py fill)"})
    eps = sum(x["episodes_per_task"] for x in units if x["task"] == task and x["state"] == "done")
    assert eps == plan["total"], (task, eps, plan["total"])
    return f"{task}: parts {[(p_['unit_id'], len(p_['recorded'])) for p_ in plan['parts']]} + substitutes {c['chosen']} = {eps} episodes"


# -- assembly -------------------------------------------------------------------------------------------------------
def stop_master(r: Run) -> None:
    if not r.master_alive():
        return
    cdir = os.path.join(r.state_dir, "control"); os.makedirs(cdir, exist_ok=True)
    open(os.path.join(cdir, "pause"), "w").close()
    pid = int(open(os.path.join(r.state_dir, "master.pid")).read().strip())
    os.kill(pid, 15)
    for _ in range(60):
        if not r.master_alive():
            break
        time.sleep(2)
    if r.master_alive():
        sys.exit("the master did not exit after TERM")
    print(f"master {pid} stopped (control/pause set; its workers keep running)")


def stop_seq_worker(r: Run, plan: dict) -> None:
    stop_unit_worker(r, plan["seq_unit"], plan["seq_node"])


def stop_unit_worker(r: Run, uid: str, node: str) -> None:
    script = (f"set -u; U={r.run_root}/units/{uid}; pid=$(cat $U/unit.pid 2>/dev/null); "
              f"[ -n \"$pid\" ] || {{ echo 'no unit.pid'; exit 0; }}; "
              f"if kill -0 $pid 2>/dev/null && grep -q run_unit $(echo /proc/$pid/cmdline) 2>/dev/null; then kill -TERM $pid; echo \"TERM worker $pid\"; "
              f"for i in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -0 $pid 2>/dev/null && echo STILL_ALIVE || echo worker_exited; "
              f"else echo \"worker $pid not running\"; fi")
    p = ssh(node, script, timeout=300)
    print(f"   {uid} on {node}: {p.stdout.strip()} {p.stderr.strip()[:200]}")


def cmd_assemble(r: Run, tasks: list[str], a) -> None:
    s = r.state()
    plans = {t: load_plan(r, t) for t in tasks if os.path.exists(plan_path(r, t))}
    fills = {t: json.load(open(fill_path(r, t))) for t in tasks if os.path.exists(fill_path(r, t))}
    if not plans and not fills:
        sys.exit("no tail or fill plan for the given tasks")
    finished, retried = [], []
    # a sequential unit that completed on its own before the assembly IS the unsharded result: keep it, drop its shards
    finished += [t for t in plans if next(u for u in s["units"] if u["unit_id"] == plans[t]["seq_unit"])["state"] == "done"]
    for t in finished:
        print(f"{t}: the sequential unit completed on its own -- its unsharded result stands, the tail shards are discarded")
        plans.pop(t)
    # a fill unit the master completed itself with a normal retry keeps that result (the substitutes are discarded)
    retried += [t for t in fills if all(next(u for u in s["units"] if u["unit_id"] == p_["unit_id"])["state"] == "done" for p_ in fills[t]["parts"])]
    for t in retried:
        print(f"{t}: the master completed the unit(s) with a normal retry -- that result stands, the fill substitutes are discarded")
        fills.pop(t)
    special = {plans[t]["seq_unit"] for t in plans} | {p_["unit_id"] for t in fills for p_ in fills[t]["parts"]}
    others = [u["unit_id"] for u in s["units"] if u["unit_id"] not in special and u["state"] != "done"]
    if others and not a.force:
        sys.exit(f"other units are not done yet: {others}")
    covs, fcovs = {}, {}
    for t in plans:
        covs[t] = coverage(r, t, plans[t])
        if not covs[t]["complete"]:
            cmd_status(r, t, a); sys.exit(f"{t}: coverage incomplete -- nothing changed")
    for t in fills:
        fcovs[t] = fill_coverage(r, t, fills[t])
        if not fcovs[t]["complete"]:
            cmd_fill_status(r, t); sys.exit(f"{t}: fill coverage incomplete -- nothing changed")
    stop_master(r)
    for t in plans:
        stop_seq_worker(r, plans[t])
    for t in fills:
        for p_ in fills[t]["parts"]:
            u = next(x for x in s["units"] if x["unit_id"] == p_["unit_id"])
            if u["state"] == "running" and u.get("slot"):
                stop_unit_worker(r, p_["unit_id"], u["slot"].split(":")[0])
    for name in ("state.json", "units.json"):
        src = os.path.join(r.state_dir, name)
        if os.path.exists(src) and not os.path.exists(src + ".orig_before_tail_assembly"):
            shutil.copy2(src, src + ".orig_before_tail_assembly")
    state = r.state()
    units = state["units"]
    notes = [f"{t}: sequential unit completed on its own (tail shards discarded)" for t in finished]
    notes += [f"{t}: completed by the master's normal retry (fill substitutes discarded)" for t in retried]
    for t in plans:
        plan, c = plans[t], covs[t]
        uid, att, cut, total = plan["seq_unit"], plan["seq_attempt"], c["cut"], c["total"]
        seq = next(u for u in units if u["unit_id"] == uid)
        # the sequential unit's partial upstream result (node-local, the worker only copies it on normal completion)
        base, rid = c["seq"]["node_result_base"], plan["seq_run_id"]
        cdir = f"{r.run_root}/units/{uid}/a{att}/client/0_{t}/upstream_eval_result"
        p = ssh(plan["seq_node"], f"set -e; d=$(ls -d {base}/{r.seed}_*/{rid} | head -1); mkdir -p {cdir}; cp -r \"$d\"/. {cdir}/; "
                                  f"cp -f {base}/{r.seed}_*/_resume_{rid}.json {cdir}/ 2>/dev/null || true; ls {cdir} | wc -l", timeout=900)
        if p.returncode != 0:
            sys.exit(f"{t}: copying the sequential result failed: {p.stderr[:300]}")
        local = os.path.join(r.state_dir, "mirror", "units", uid, f"a{att}")
        r.pull(f"{r.run_root}/units/{uid}/a{att}/", local + "/", extra=["--exclude", "*.mp4", "--exclude", "server/"])
        rf = glob.glob(os.path.join(local, "client", "*", "upstream_eval_result", "_result.json"))
        if not rf:
            sys.exit(f"{t}: sequential _result.json missing after the copy")
        doc = json.load(open(rf[0])); det = doc.get("details", {})
        if not os.path.exists(rf[0] + ".orig_before_tail_assembly"):
            shutil.copy2(rf[0], rf[0] + ".orig_before_tail_assembly")
        keep = {k: v for k, v in sorted(det.items(), key=lambda kv: int(kv[0])) if int(v.get("layout_id", k)) < cut}
        keep = {str(i): v for i, v in enumerate(keep.values())}
        assert sorted(int(v["layout_id"]) for v in keep.values()) == c["seq_recorded"], (t, "sequential part mismatch")
        json.dump({**doc, "details": keep}, open(rf[0], "w"), indent=1)
        seq_succ = sum(bool(v.get("success")) for v in keep.values()); seq_score = round(sum(float(v.get("score", 0) or 0) for v in keep.values()), 4)
        seq["history"][-1].update(status="stopped_after_tail_shards", episodes=len(keep), successes=seq_succ, score_sum=seq_score)
        seq.update(state="done", episodes_per_task=len(keep), layout_ids=list(range(cut)), layout_total=total,
                   result={"attempt": att, "status": "complete_first_part", "episodes_recorded": len(keep), "successes": seq_succ, "score_sum": seq_score},
                   note=f"sequential unit stopped after the tail shards (tail_shards.py): layouts 0..{cut - 1} (abandoned {c['seq_abandoned']}); "
                        f"layouts {cut}..{total - 1} from one-layout shards, substitutes {c['substitutes_chosen']}")
        used = [l for l in range(cut, total) if c["shards"][l]["state"] == "recorded"] + c["substitutes_chosen"]
        for l in used:
            sh_ = plan["shards"][str(l)]; res = c["shards"][l]["result"]; suid = sh_["unit_id"]
            r.pull(f"{r.run_root}/units/{suid}/a1/", os.path.join(r.state_dir, "mirror", "units", suid, "a1") + "/", extra=["--exclude", "*.mp4", "--exclude", "server/"])
            if any(u["unit_id"] == suid for u in units):
                continue
            units.append({"unit_id": suid, "task": t, "eval_seed": seq["eval_seed"], "dimension": seq["dimension"], "variant": seq.get("variant"),
                          "episodes_per_task": 1, "layout_ids": [l], "layout_total": total, "state": "done", "attempts": 1,
                          "history": [{"attempt": 1, "dispatched": res.get("started"), "finished": res.get("finished"), "status": "complete",
                                       "episodes": 1, "successes": res.get("successes", 0), "score_sum": res.get("score_sum", 0)}],
                          "result": {"attempt": 1, "status": "complete", "episodes_recorded": 1, "successes": res.get("successes", 0),
                                     "score_sum": res.get("score_sum", 0)},
                          "note": ("substitute layout for an abandoned one" if l >= total else "tail shard") + " (tail_shards.py)"})
        eps = len(keep) + len(used)
        assert eps == total, (t, eps, total)
        succ = seq_succ + sum(c["shards"][l]["result"].get("successes", 0) for l in used)
        notes.append(f"{t}: sequential 0..{cut - 1} ({len(keep)} eps) + {len(used) - len(c['substitutes_chosen'])} shards + substitutes "
                     f"{c['substitutes_chosen']} = {eps} episodes, {succ} successes; abandoned {c['seq_abandoned'] + c['shard_abandoned']}")
    for t in fills:
        notes.append(assemble_fill(r, t, units, fills[t], fcovs[t]))
    json.dump(state, open(os.path.join(r.state_dir, "state.json"), "w"), indent=1)
    json.dump(units, open(os.path.join(r.state_dir, "units.json"), "w"), indent=1)
    shutil.copy2(os.path.join(r.state_dir, "units.json"), os.path.join(r.state_dir, "mirror", "units.json"))
    if os.path.exists(os.path.join(r.state_dir, "tasks.json")):
        shutil.copy2(os.path.join(r.state_dir, "tasks.json"), os.path.join(r.state_dir, "mirror", "tasks.json"))
    for n in notes:
        print("ASSEMBLED " + n)
    res_md = os.path.join(r.state_dir, "RESULTS.md")
    if os.path.exists(res_md) and not os.path.exists(res_md + ".orig_before_tail_assembly"):
        shutil.copy2(res_md, res_md + ".orig_before_tail_assembly")
    py = os.environ.get("MASTER_PY", os.path.join(HOME, "miniforge3/envs/sana-wam/bin/python"))
    p = sh([py, os.path.join(EVAL, "summarize_eval.py"), os.path.join(r.state_dir, "mirror"), "--out", res_md, "--run-id", r.run,
            "--note", "tail units finished by tail_shards.py: " + " | ".join(notes)], timeout=900)
    print(f"RESULTS.md rewritten (summarize_eval rc {p.returncode}); next: bash {r.driver} official-summary")
    json.dump({"assembled": now(), "notes": notes}, open(os.path.join(r.tail_dir, "ASSEMBLED.json"), "w"), indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "launch", "status", "more-substitutes", "invalidate", "fill", "fill-status", "assemble"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--driver", required=True, help="the campaign driver whose smoke-gpu starts the shards (it must honour RUN / SMOKE_* env)")
    ap.add_argument("--task", action="append", required=True)
    ap.add_argument("--cut", type=int, default=None, help="first shard layout (default: the layout after the one in progress)")
    ap.add_argument("--spare", type=int, default=1, help="extra substitute ids beyond the abandoned count at launch")
    ap.add_argument("--count", type=int, default=0, help="more-substitutes: at least this many new substitute ids")
    ap.add_argument("--avoid", default="l2,l6", help="nodes not to use for shards (flaky boards)")
    ap.add_argument("--unit", action="append", help="fill: an incomplete unit of the task config (repeat for several)")
    ap.add_argument("--attempt", type=int, default=1, help="fill: the attempt whose recorded episodes are kept (default the first)")
    ap.add_argument("--adopt", default="", help="fill: substitute ids already started by hand (comma list) instead of launching")
    ap.add_argument("--seed", type=int, default=None, help="the run's eval seed (default: the single seed its units carry)")
    ap.add_argument("--layouts", default="", help="invalidate: shard layouts to discard and run again (comma list)")
    ap.add_argument("--reason", default="", help="invalidate: why (recorded in the plan and the moved directory name)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    r = Run(a.run, a.driver, a.seed)
    if a.cmd == "assemble":
        cmd_assemble(r, a.task, a); return
    ok = True
    for t in a.task:
        if a.cmd == "plan":
            cmd_plan(r, t, a)
        elif a.cmd == "launch":
            cmd_launch(r, t, a)
        elif a.cmd == "status":
            ok &= cmd_status(r, t, a)
        elif a.cmd == "more-substitutes":
            cmd_more_substitutes(r, t, a)
        elif a.cmd == "invalidate":
            cmd_invalidate(r, t, a)
        elif a.cmd == "fill":
            cmd_fill(r, t, a)
        elif a.cmd == "fill-status":
            ok &= cmd_fill_status(r, t)
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
