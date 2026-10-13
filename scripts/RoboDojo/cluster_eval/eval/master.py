#!/usr/bin/env python3
"""master.py -- x5-side scheduler for the RoboDojo evaluation (master/workers).

The master owns a queue of UNITS (one task config at one eval seed with its
full episode budget, per the official protocol: 50 or 25 episodes) and a set
of WORKER SLOTS (a GPU pair on an A6000 node: server GPU 2k, client GPU 2k+1,
private rpc/ws ports).  Whenever a slot is free it dispatches the heaviest
remaining unit to it (`ssh node nohup setsid run_unit.sh ... ATTEMPT`,
returning at once); the worker reports through files on the shared home
(RUN_ROOT/units/<unit>/a<attempt>/{heartbeat,status.txt,result.json}) which
the master rsync-pulls (with --delete) every --poll seconds via one node
(falling back to the others).  Incomplete units are re-queued (up to
--max-attempts, preferring a different slot).  A unit whose heartbeat is stale
is killed remotely; its slot then DRAINS until the worker pid is gone (so the
GPU/port are really free) before it is reused.  Staleness is only judged while
pulls succeed, so an rsync/network outage cannot trigger mass kills.  Slots
that fail twice in a row at the infrastructure level (server_fail,
server_timeout, port_busy, stale, aborted, prologue_fail) are quarantined
(lifted if work would otherwise stall).  State is persisted on x5 (--state-dir)
so `--resume` continues after a master restart; workers never depend on the
master being alive.

usage (on x5):
  master.py --run-root <CE_EVAL_ROOT>/<run> --checkout <RoboDojo clone>   (start it with start_master.sh, which fills these in)
            [--nodes n1,n2] [--slots-per-node 4] [--seeds 0] [--tasks a,b]
            [--episodes-per-task N] [--no-random] [--max-attempts 3] [--poll 60]
            [--heartbeat-timeout 900] [--resume] [--dry-run]
  (with --resume the slots come from the saved state; nodes new to the run, and pairs named explicitly in
   --node-pairs, are appended -- never removed by a resume.  A running master appends pairs when <state-dir>/control/add_slots
   holds a 'l8:0,2;l25:0' spec, so a run can grow onto GPU pairs another run has just released, and forgets IDLE pairs
   named in <state-dir>/control/drop_slots, so another master can take them without a collision.)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plan_eval import load_tasks  # noqa: E402  (same enumeration rules as the static planner)

SLOT_LEVEL = ("server_fail", "server_timeout", "port_busy", "stale", "aborted", "prologue_fail", "dispatch", "ckpt_missing")
# site configuration (cluster_eval/lib/ce_site.py: the CE_* values of site.env)
sys.path.insert(0, str(HERE.parent / "lib"))
import ce_site  # noqa: E402
# a checkpoint snapshot is usable once these exist (plus one normalization/*.json) and no HF download is in flight
CHECKPOINT_REQUIRED = ("config.yaml", "README.md", "model/pytorch_model_fsdp.bin")
# A non-SANA policy ships a different bundle. Keyed by the SANA_EVAL_XPL_POLICY value passed through
# --worker-env; anything not listed here is judged by CHECKPOINT_REQUIRED above (the SANA layout).
CHECKPOINT_LAYOUTS = {
    # OpenWAM: one consolidated safetensors + its own normalization .npy, no model/ dir, no normalization/ dir.
    "OpenWAM": {"files": ("config.yaml", "normalization_stats.npy"), "globs": ("checkpoint_step_*.safetensors",)},
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


DIMENSION_ORDER_AUTO = "auto"


def step_lim_of(checkout: Path, task: str) -> int:
    """The task's episode step limit from task/RoboDojo/tasks/<task>.py (the worker uses the same regex; 2000 if absent)."""
    try:
        txt = (checkout / "task/RoboDojo/tasks" / f"{task}.py").read_text(encoding="utf-8")
        m = re.search(r"self\.step_lim\s*=\s*(\d+)", txt)
        return int(m.group(1)) if m else 2000
    except OSError:
        return 2000


def work_of(unit: dict, checkout: Path) -> int:
    """Rough wall-clock proxy of a unit: episodes x step limit (a 50-episode fasten_screws unit = 95000, a 25-episode
    fold_clothes unit = 12500)."""
    return int(unit["episodes_per_task"]) * step_lim_of(checkout, unit["task"])


def order_units(units: list, checkout: Path, order: str = "dimension", dimension_order: str = DIMENSION_ORDER_AUTO,
                pin_longest: int = 2) -> list:
    """Dispatch order of the planned units.

    order=size      -- the old rule: most episodes first, then task name (long tail starts first, categories finish last).
    order=dimension -- CATEGORY BY CATEGORY (default since 2026-09-11): the `pin_longest` units with the largest work
                       estimate go first regardless of category (so the run's tail does not grow), then every category's
                       units as a block (longest first inside the block), categories in `dimension_order` (comma list) or,
                       for 'auto', by ascending total work so the first complete-category comparison arrives soonest.
    """
    if order == "size":
        return sorted(units, key=lambda u: (-u["episodes_per_task"], u["task"], u["eval_seed"]))
    work = {u["unit_id"]: work_of(u, checkout) for u in units}
    pinned = sorted(units, key=lambda u: -work[u["unit_id"]])[:max(0, int(pin_longest))]
    pinned_ids = {u["unit_id"] for u in pinned}
    rest = [u for u in units if u["unit_id"] not in pinned_ids]
    if dimension_order == DIMENSION_ORDER_AUTO:
        totals: dict = {}
        for u in rest:
            totals[u["dimension"]] = totals.get(u["dimension"], 0) + work[u["unit_id"]]
        dims = sorted(totals, key=lambda d: (totals[d], d))
    else:
        dims = [d.strip() for d in dimension_order.split(",") if d.strip()]
        dims += sorted({u["dimension"] for u in rest} - set(dims))
    rank = {d: i for i, d in enumerate(dims)}
    rest.sort(key=lambda u: (rank.get(u["dimension"], len(rank)), -work[u["unit_id"]], u["task"], u["eval_seed"]))
    return pinned + rest


def _slot_token(token: str):
    """One entry of a node-pairs list: an int k = the GPU pair 2k/2k+1, or 's<g>' = the SINGLE GPU g hosting both the
    policy server and the simulator client (for a lone free GPU whose pair partner is busy or broken)."""
    token = token.strip()
    if token[:1] in ("s", "S"):
        return f"s{int(token[1:])}"
    return int(token)


def _slot_order(k):
    return (1, int(k[1:])) if isinstance(k, str) else (0, k)


def parse_node_pairs(spec: str | None) -> dict:
    """'l3:1,2,3;l26:1,2,3' -> {node: sorted pair indices} (pair k = CUDA devices 2k/2k+1; 's<g>' = single GPU g,
    after the pairs); raises ValueError."""
    out = {}
    for item in (spec or "").split(";"):
        item = item.strip()
        if not item:
            continue
        node, sep, ks = item.partition(":")
        node = node.strip()
        if not sep or not node:
            raise ValueError(f"node-pairs entry must be NODE:k,k,... got {item!r}")
        try:
            out[node] = sorted({_slot_token(k) for k in ks.split(",") if k.strip() != ""}, key=_slot_order)
        except ValueError:
            raise ValueError(f"node-pairs entry must be NODE:k,k,... (k a pair index or s<gpu>) got {item!r}") from None
    return out


def parse_settings(spec: str | None) -> list:
    """--settings 'label:key=v key2=v;label2:...' -> [{"label", "xpl_overrides", "checkpoint_root"}] (checkpoint_root None =
    the run default). Tokens are passed to setup_policy_server.py --overrides except `checkpoint_root=<path>` which selects
    another checkpoint for that setting only. Labels must be unique and usable in file names."""
    out = []
    for item in (spec or "").split(";"):
        item = item.strip()
        if not item:
            continue
        label, sep, rest = item.partition(":")
        label = label.strip()
        if not sep or not label or not re.fullmatch(r"[A-Za-z0-9._-]+", label):
            sys.exit(f"--settings: expected 'label:key=value ...', got {item!r} (label = letters, digits, . _ -)")
        tokens, ckpt = [], None
        for tok in rest.split():
            if tok.startswith("checkpoint_root="):
                ckpt = tok.split("=", 1)[1].rstrip("/")
                if not ckpt.startswith("/"):
                    sys.exit(f"--settings {label}: checkpoint_root must be an absolute node-side path, got {ckpt!r}")
            else:
                tokens.append(tok)
        out.append({"label": label, "xpl_overrides": " ".join(tokens), "checkpoint_root": ckpt})
    if len({x["label"] for x in out}) != len(out):
        sys.exit("--settings: labels must be unique")
    return out


class Master:
    def __init__(self, a):
        self.units, self.slots = [], []            # filled by build_units()/load(); read by wait_for_checkpoint()
        self.a = a
        self.user, self.domain = a.ssh_user, a.ssh_domain
        self.run_root = a.run_root.rstrip("/")
        self.state_dir = Path(a.state_dir).expanduser().resolve()
        self.mirror = self.state_dir / "mirror"
        self.state_file = self.state_dir / "state.json"
        self.log_file = self.state_dir / "master.log"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.mirror.mkdir(exist_ok=True)
        self.stop = False
        self.last_pull_ok = 0.0
        self.pull_failures = 0            # consecutive failed pulls (a shared-home outage looks like this)
        self.liveness_grace_until = 0.0   # after an outage, heartbeats need time to refresh before anyone is judged stale
        # dry run: units execute locally (run_unit.sh EVAL_DRY_RUN=1) under this fake RUN_ROOT
        self.dry_root = (self.state_dir / "robodojo_dry_root") if a.dry_run else None

    # ---- logging / remote ------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with self.log_file.open("a") as f:
            f.write(line + "\n")

    def ssh(self, node: str, cmd: str, timeout: int = 60):
        """-> CompletedProcess, or None on timeout."""
        try:
            return subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                   f"{self.user}@{node}{self.domain}", cmd], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def pull(self) -> bool:
        """Mirror the small worker files from the shared home (--delete so re-dispatched attempts start clean)."""
        if self.dry_root:
            self.last_pull_ok = time.time()
            return True
        for node in self.a.nodes:
            cmd = ["rsync", "-az", "--delete", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15",
                   "--include=*/", "--include=result.json", "--include=heartbeat", "--include=status.txt",
                   "--include=run_status.json", "--include=_result.json", "--exclude=*",
                   f"{self.user}@{node}{self.domain}:{self.run_root}/units/", str(self.mirror / "units") + "/"]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                self.log(f"WARN rsync pull via {node} timed out"); continue
            if r.returncode in (0, 23, 24):
                if self.pull_failures:
                    # The shared home was unreachable: the workers' heartbeat writers were blocked too and may
                    # lag behind the master's view for a while, so no liveness verdicts until they had a full
                    # heartbeat window to catch up (2026-09-03: an NFS outage mass-killed 7 healthy units).
                    self.liveness_grace_until = time.time() + self.a.heartbeat_timeout
                    self.log(f"pull recovered after {self.pull_failures} failed round(s); liveness judgments "
                             f"suspended for {self.a.heartbeat_timeout}s while heartbeats catch up")
                self.pull_failures = 0
                self.last_pull_ok = time.time()
                return True
            self.log(f"WARN rsync pull via {node} rc={r.returncode}: {r.stderr.strip()[:160]}")
        self.pull_failures += 1
        return False

    def free_slot_ports(self, slot: dict, why: str) -> None:
        """Kill whatever still listens on the slot's rpc/ws ports and stop leftover servers of the slot's
        previous unit.  The ports belong to this slot alone, so any listener while the slot is free is an
        orphan (a server whose stop failed during a shared-home outage, or a unit killed during start-up)."""
        if self.dry_root:
            return
        cmd = (f"for port in {slot['rpc_port']} {slot['ws_port']}; do "
               f"pids=$(ss -ltnpH \"sport = :$port\" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u | tr '\\n' ' '); "
               f"if [ -n \"$pids\" ]; then echo \"port $port held by $pids\"; kill -TERM $pids 2>/dev/null; sleep 8; kill -KILL $pids 2>/dev/null; fi; done; "
               f"echo swept")
        r = self.ssh(slot["node"], cmd, timeout=90)
        if r is None or "swept" not in (r.stdout or ""):
            self.log(f"WARN slot {slot['slot_id']}: port sweep ({why}) inconclusive")
            return
        held = [l for l in r.stdout.splitlines() if l.startswith("port ")]
        if held:
            self.log(f"slot {slot['slot_id']}: reclaimed ports ({why}): " + "; ".join(held))

    def attempt_dir(self, uid: str, attempt: int) -> Path:
        base = (self.dry_root / "units") if self.dry_root else (self.mirror / "units")
        return base / uid / f"a{attempt}"

    # ---- planning -------------------------------------------------------------------
    def build_units(self) -> list[dict]:
        seeds = [int(s) for s in self.a.seeds.split(",") if s.strip()]
        dyn = self.a.dynamic_instruction_tasks
        if dyn == "auto":
            dyn = "on" if self.a.worker == "run_unit_xpl.sh" else "off"
        tasks = load_tasks(Path(self.a.checkout).resolve(), seeds, allow_dynamic_instruction=(dyn == "on"),
                           allow_support_arm_embodiments=(getattr(self.a, "support_arm_embodiments", "off") == "on"))
        selected = {n: t for n, t in tasks.items() if t["runnable"]}
        if self.a.no_random:
            selected = {n: t for n, t in selected.items() if t["variant"] != "random"}
        if self.a.tasks:
            keep = {s.strip() for s in self.a.tasks.split(",") if s.strip()}
            unknown = keep - set(tasks)
            if unknown:
                sys.exit(f"unknown tasks: {sorted(unknown)}")
            selected = {n: t for n, t in selected.items() if n in keep}
        if self.a.dimensions:
            dims = {d.strip() for d in self.a.dimensions.split(",") if d.strip()}
            known = {t["dimension"] for t in tasks.values()}
            if dims - known:
                sys.exit(f"unknown dimensions {sorted(dims - known)}; known: {sorted(known)}")
            selected = {n: t for n, t in selected.items() if t["dimension"] in dims}
        if not selected:
            sys.exit("no runnable task configs selected")
        settings = parse_settings(self.a.settings) or [None]
        if settings != [None]:
            if self.a.worker != "run_unit_xpl.sh" and any(x["xpl_overrides"] for x in settings):
                sys.exit("--settings override tokens are only honoured by --worker run_unit_xpl.sh (run_unit.sh ignores them; "
                         "only checkpoint_root= would differ)")
            for x in settings:      # run-wide --xpl-overrides apply to every setting; the setting's tokens come last and win
                x["xpl_overrides"] = " ".join(t for t in [(self.a.xpl_overrides or "").strip(), x["xpl_overrides"]] if t)
        units = []
        shards = max(1, int(getattr(self.a, "shards", 1) or 1))
        shard_plan = {}
        for item in (self.a.shard_plan or "").split(","):
            if item.strip():
                name, _, n = item.partition("=")
                shard_plan[name.strip()] = max(1, int(n))
        for n, t in sorted(selected.items()):
            eps = t["eval_nums"] if self.a.episodes_per_task is None else min(self.a.episodes_per_task, t["eval_nums"])
            for seed in seeds:
                common = {"task": n, "eval_seed": seed, "dimension": t["dimension"], "variant": t["variant"],
                          "state": "queued", "attempts": 0, "history": [], "slot": None}
                k = min(shard_plan.get(n, shards), eps)
                for setting in settings:
                    # --settings: the same task configs run once per setting (label suffix on the unit id, per-unit
                    # overrides/checkpoint), so one queue spreads every setting over all slots at the same time.
                    extra = {} if setting is None else {"setting": setting["label"], "xpl_overrides": setting["xpl_overrides"],
                                                        "checkpoint_root": setting["checkpoint_root"] or self.a.checkpoint_root}
                    suffix = "" if setting is None else f"__{setting['label']}"
                    if k <= 1:
                        units.append({"unit_id": f"{n}__s{seed}{suffix}", "episodes_per_task": eps, **common, **extra, "history": []})
                        continue
                    # --shards: the task's layouts 0..eps-1 are split into k contiguous, near-equal ranges that run as
                    # independent units (XPL lane only: the worker pre-seeds the official client's resume manifest so
                    # that every layout outside the range is 'abandoned' and only the range is evaluated).
                    bounds = [round(i * eps / k) for i in range(k + 1)]
                    for i in range(k):
                        ids = list(range(bounds[i], bounds[i + 1]))
                        units.append({"unit_id": f"{n}__s{seed}__p{i + 1}of{k}{suffix}", "episodes_per_task": len(ids),
                                      "layout_ids": ids, "layout_total": eps, **common, **extra, "history": []})
        units = order_units(units, Path(self.a.checkout).resolve(), self.a.order, self.a.dimension_order, self.a.pin_longest)
        seen = []
        for u in units:
            if u["dimension"] not in seen:
                seen.append(u["dimension"])
        if settings != [None]:
            self.log("settings: " + "; ".join(f"{x['label']}: overrides '{x['xpl_overrides']}'" + (f" checkpoint {x['checkpoint_root']}" if x['checkpoint_root'] else "") for x in settings))
        self.log(f"dispatch order: {self.a.order}" + (f"; pinned longest {self.a.pin_longest}: "
                 f"{[u['unit_id'] for u in units[:self.a.pin_longest]]}; categories {seen}" if self.a.order == "dimension" else ""))
        (self.state_dir / "tasks.json").write_text(json.dumps(tasks, indent=2, sort_keys=True) + "\n")
        excluded = {n: t["excluded_reason"] for n, t in tasks.items() if not t["runnable"]}
        (self.state_dir / "excluded.json").write_text(json.dumps(excluded, indent=2, sort_keys=True) + "\n")
        return units

    def make_slot(self, node: str, k) -> dict:
        """Slot record for GPU pair k (CUDA devices 2k/2k+1) on node; the ports are indexed by the pair.
        k = 's<g>' is a SINGLE-GPU slot: CUDA device g runs the policy server AND the simulator client (slot id
        '<node>:g<g><g>', ports base + 50 + g, clear of the pair ports base + 0..30)."""
        if isinstance(k, str):
            g = int(k[1:])
            if not 0 <= g <= 7:
                raise ValueError(f"single-GPU slot {k!r} on {node} is outside the 8 GPUs of a node (s0-s7)")
            return {"slot_id": f"{node}:g{g}{g}", "node": node, "server_gpu": g, "client_gpu": g,
                    "rpc_port": self.a.rpc_port_base + 50 + g, "ws_port": self.a.ws_port_base + 50 + g,
                    "unit": None, "dispatched_at": None, "consecutive_failures": 0, "quarantined": False, "draining": None}
        if not 0 <= k <= 3:
            raise ValueError(f"GPU pair index {k} on {node} is outside the 4 GPU pairs of a node (0-3)")
        return {"slot_id": f"{node}:g{2*k}{2*k+1}", "node": node, "server_gpu": 2 * k, "client_gpu": 2 * k + 1,
                "rpc_port": self.a.rpc_port_base + 10 * k, "ws_port": self.a.ws_port_base + 10 * k,
                "unit": None, "dispatched_at": None, "consecutive_failures": 0, "quarantined": False, "draining": None}

    def pairs_for(self, node: str) -> list[int]:
        """GPU pairs of a node: its --node-pairs entry, else --slot-offset .. --slot-offset+--slots-per-node-1."""
        return self.node_pairs().get(node) or list(range(self.a.slot_offset, self.a.slot_offset + self.a.slots_per_node))

    def build_slots(self) -> list[dict]:
        # GPU pairs k = slot_offset .. slot_offset+slots_per_node-1 on every node; the ports are indexed by the
        # PAIR (per node), so two masters sharing a node with disjoint --slot-offset ranges never collide on
        # GPUs or ports (the same pair number on different nodes may share a port: different hosts).
        slots = []
        for node in self.a.nodes:
            for k in self.pairs_for(node):
                try:
                    slots.append(self.make_slot(node, k))
                except ValueError as e:
                    sys.exit(f"{e} (--slot-offset/--slots-per-node/--node-pairs)")
        return slots

    def add_slots(self, spec: str, sweep: bool = True) -> list[str]:
        """Append the GPU pairs of a 'l8:0,2;l25:0' spec that the run does not have yet (same pair/port scheme; a slot
        is never removed or moved). Behind `--resume --node-pairs` and the control/add_slots file, so a run grows onto
        pairs another run has just released without a restart. Raises ValueError on a bad spec."""
        known = {s["slot_id"] for s in self.slots}
        added = []
        for node, pairs in parse_node_pairs(spec).items():
            for k in pairs:
                slot = self.make_slot(node, k)
                if slot["slot_id"] in known:
                    continue
                self.slots.append(slot); known.add(slot["slot_id"]); added.append(slot["slot_id"])
                if node not in self.a.nodes:          # ssh probes (checkpoint_ready) iterate over a.nodes
                    self.a.nodes.append(node)
                if sweep:
                    self.free_slot_ports(slot, "new slot")
        return added

    def drop_slots(self, spec: str) -> tuple[list[str], list[str]]:
        """Remove the IDLE slots of a 'l8:0,2;l25:0' spec so this run can no longer dispatch onto those GPU pairs
        (another master is about to use them). A slot with a running unit or a draining worker is kept and reported;
        pairs that are not slots of this run are ignored. Counterpart of add_slots (control/drop_slots); the pairs can
        be given back later with control/add_slots. Raises ValueError on a bad spec."""
        wanted = {self.make_slot(node, k)["slot_id"] for node, pairs in parse_node_pairs(spec).items() for k in pairs}
        dropped, kept = [], []
        for s in list(self.slots):
            if s["slot_id"] not in wanted:
                continue
            if s["unit"] is not None or s.get("draining"):
                kept.append(s["slot_id"])
                continue
            self.slots.remove(s)
            dropped.append(s["slot_id"])
        return dropped, kept

    # ---- state -----------------------------------------------------------------------
    def save(self) -> None:
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"run_root": self.run_root, "checkpoint_root": self.a.checkpoint_root, "worker": self.a.worker,
                                   "xpl_root": self.a.xpl_root, "xpl_action_type": getattr(self.a, "xpl_action_type", "joint"), "units": self.units,
                                   "slots": self.slots, "saved": now()}, indent=1) + "\n")
        os.replace(tmp, self.state_file)

    def load(self) -> None:
        st = json.loads(self.state_file.read_text())
        self.units, self.slots = st["units"], st["slots"]
        saved_action_type = st.get("xpl_action_type")
        if saved_action_type and saved_action_type != getattr(self.a, "xpl_action_type", "joint"):
            self.log(f"resume: xpl action_type {saved_action_type!r} restored from state.json (command line had {self.a.xpl_action_type!r})")
            self.a.xpl_action_type = saved_action_type
        saved_ckpt = st.get("checkpoint_root")          # states written before the knob existed served the default
        if saved_ckpt and saved_ckpt != self.a.checkpoint_root:
            if self.a.checkpoint_root_explicit:
                sys.exit(f"--checkpoint-root {self.a.checkpoint_root} differs from this run's checkpoint {saved_ckpt}; "
                         "one run evaluates one checkpoint -- use a new RUN_NAME")
            self.a.checkpoint_root = saved_ckpt
        saved_worker = st.get("worker")
        if saved_worker and saved_worker != self.a.worker:
            sys.exit(f"--worker {self.a.worker} differs from this run's worker {saved_worker}; one run uses one worker script")
        if st.get("xpl_root") and not self.a.xpl_root:
            self.a.xpl_root = st["xpl_root"]
        for s in self.slots:
            s.setdefault("draining", None)
        for u in self.units:
            # migration for states written before failures were split into real vs infrastructure:
            # a history entry without a status was a stale-kill (the STALE path did not record one)
            if "failures" not in u:
                hist = u["history"][:-1] if u["state"] == "running" else u["history"]
                u["failures"] = sum(1 for h in hist if h.get("status") in (None, "incomplete", "stale", "unknown"))
                u["infra_failures"] = sum(1 for h in hist if h.get("status") in SLOT_LEVEL)

    def node_pairs(self) -> dict:
        """--node-pairs 'l3:1,2,3;l26:1,2,3' -> {node: [pair indices]} (CUDA enumeration order: pair k = GPUs 2k/2k+1).
        Nodes not listed use --slot-offset/--slots-per-node. Lets a run skip GPUs that are broken or busy with
        somebody else's job."""
        try:
            return parse_node_pairs(self.a.node_pairs)
        except ValueError as e:
            sys.exit(f"--{e}")

    def worker_env(self) -> dict:
        """Extra KEY=VALUE pairs from --worker-env, exported into the worker's environment (and so into the
        server/client processes) -- e.g. DISABLE_XFORMERS=1 to serve the Sana model on SDPA attention."""
        out = {}
        for item in (self.a.worker_env or "").split():
            k, _, v = item.partition("=")
            if not k.isidentifier():
                sys.exit(f"--worker-env entries must be KEY=VALUE with an identifier key, got {item!r}")
            out[k] = v
        return out

    # ---- remote side ------------------------------------------------------------------
    def unit_checkpoint(self, unit: dict) -> str:
        return unit.get("checkpoint_root") or self.a.checkpoint_root

    def unit_overrides(self, unit: dict) -> str:
        return unit["xpl_overrides"] if "xpl_overrides" in unit else (self.a.xpl_overrides or "")

    def checkpoint_ready(self, checkpoint_root: str | None = None):
        """(ready, detail) for a checkpoint root (default --checkpoint-root), judged on the shared home through the first reachable node."""
        checkpoint_root = checkpoint_root or self.a.checkpoint_root
        if self.dry_root:
            return True, "dry run"
        layout = CHECKPOINT_LAYOUTS.get(self.worker_env().get("SANA_EVAL_XPL_POLICY", ""))
        files = layout["files"] if layout else CHECKPOINT_REQUIRED
        globs = layout["globs"] if layout else ("normalization/*.json",)
        cmd = ("c=" + shlex.quote(checkpoint_root) + "; ok=1; "
               "for f in " + " ".join(files) + "; do [ -f \"$c/$f\" ] || { echo \"missing $f\"; ok=0; }; done; "
               + "".join("ls \"$c\"/" + g + " >/dev/null 2>&1 || { echo 'missing " + g + "'; ok=0; }; " for g in globs)
               + "inc=$(find \"$c/.cache\" -name '*.incomplete' 2>/dev/null | head -1); "
               "[ -z \"$inc\" ] || { echo \"download still in progress: $inc\"; ok=0; }; "
               "[ $ok = 1 ] && echo READY || true")
        for node in self.a.nodes:
            r = self.ssh(node, cmd, timeout=60)
            if r is None or r.returncode != 0:
                continue
            out = r.stdout.strip()
            return "READY" in out.splitlines(), (out.replace("\n", "; ") or "ready")
        return False, "no node reachable"

    def wait_for_checkpoint(self) -> None:
        """Block until every checkpoint of the run (the default and the per-setting ones) is complete, at most --wait-for-checkpoint s."""
        units = getattr(self, "units", None) or []
        wanted = [self.a.checkpoint_root] + sorted({u["checkpoint_root"] for u in units if u.get("checkpoint_root")} - {self.a.checkpoint_root})
        for ck in wanted:
            self._wait_for_one_checkpoint(ck)

    def _wait_for_one_checkpoint(self, checkpoint_root: str) -> None:
        deadline = time.time() + self.a.wait_for_checkpoint
        polls = 0
        while True:
            ready, detail = self.checkpoint_ready(checkpoint_root)
            if ready:
                self.log(f"checkpoint {checkpoint_root}: ready")
                return
            if time.time() >= deadline:
                sys.exit(f"checkpoint {checkpoint_root} is not usable ({detail}); gave up after {self.a.wait_for_checkpoint}s")
            if polls % 10 == 0:
                self.log(f"checkpoint {checkpoint_root} not ready yet ({detail}); waiting up to {int(deadline - time.time())}s more")
            polls += 1
            for _ in range(60):
                if self.stop:
                    sys.exit("stop requested while waiting for the checkpoint")
                time.sleep(1)

    def worker_alive(self, slot: dict, uid: str):
        """True/False = pid check answered; None = node unreachable."""
        if self.dry_root:
            pidf = self.dry_root / "units" / uid / "unit.pid"
            try:
                os.kill(int(pidf.read_text().strip()), 0); return True
            except (OSError, ValueError):
                return False
        udir = f"{self.run_root}/units/{uid}"
        r = self.ssh(slot["node"], f"p=$(cat {shlex.quote(udir)}/unit.pid 2>/dev/null); [ -n \"$p\" ] && kill -0 \"$p\" 2>/dev/null && echo alive || echo dead", timeout=45)
        if r is None or r.returncode != 0:
            return None
        return "alive" in r.stdout

    def dispatch(self, unit: dict, slot: dict) -> bool:
        uid, attempt = unit["unit_id"], unit["attempts"] + 1
        adir = self.attempt_dir(uid, attempt)
        shutil.rmtree(adir, ignore_errors=True)        # never read a previous mirror of this attempt
        args = [self.run_root if not self.dry_root else str(self.dry_root), uid, unit["task"], unit["eval_seed"],
                unit["episodes_per_task"], slot["server_gpu"], slot["client_gpu"], slot["rpc_port"], slot["ws_port"], attempt]
        if self.dry_root:
            adir.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, EVAL_DRY_RUN="1", DRY_SLEEP=str(self.a.dry_sleep))
            if unit.get("layout_ids"):
                env["SANA_EVAL_LAYOUT_IDS"] = ",".join(str(x) for x in unit["layout_ids"])
                env["SANA_EVAL_LAYOUT_TOTAL"] = str(unit["layout_total"])
            # double-fork through a short-lived shell so the fake worker is reparented to init and never
            # lingers as a zombie of the master (a zombie would keep `kill -0` succeeding -> endless drain)
            cmd = "setsid bash " + " ".join(shlex.quote(str(x)) for x in [str(HERE / self.a.worker), *args]) + " >/dev/null 2>&1 </dev/null &"
            subprocess.run(["bash", "-c", cmd], env=env, check=False)
            ok = True
        else:
            udir_a = f"{self.run_root}/units/{uid}/a{attempt}"
            argv = " ".join(shlex.quote(str(x)) for x in args)
            # The attempt directory is recreated from scratch: a previous dispatch of this attempt number
            # that died at once (e.g. port_busy) left a result.json there, and reading it later would
            # wrongly settle the unit (2026-09-03: three healthy units were "failed" that way).
            # NOTE: `mkdir && nohup ... &` would background the whole AND-list as a subshell that keeps the
            # ssh session's stdout/stderr open until the worker ends (ssh then hangs to the timeout and $!
            # is the subshell, not the worker).  Only the nohup command may be backgrounded.
            remote = (f"rm -rf {shlex.quote(udir_a)}; mkdir -p {shlex.quote(udir_a)} || {{ echo mkdir_failed; exit 1; }}; "
                      f"SANA_EVAL_CHECKPOINT_ROOT={shlex.quote(self.unit_checkpoint(unit))} "
                      + (f"SANA_EVAL_XPL_ROOT={shlex.quote(self.a.xpl_root)} " if self.a.xpl_root else "")
                      + (f"SANA_EVAL_XPL_EXTRA_OVERRIDES={shlex.quote(self.unit_overrides(unit))} " if self.unit_overrides(unit) else "")
                      + f"SANA_EVAL_XPL_ACTION_TYPE={shlex.quote(getattr(self.a, 'xpl_action_type', 'joint') or 'joint')} "
                      + (f"SANA_EVAL_LAYOUT_IDS={shlex.quote(','.join(str(x) for x in unit['layout_ids']))} "
                         f"SANA_EVAL_LAYOUT_TOTAL={int(unit['layout_total'])} " if unit.get("layout_ids") else "")
                      + "".join(f"{k}={shlex.quote(v)} " for k, v in self.worker_env().items())
                      + f"RWM_EVAL_POLICY_ACCELERATION={shlex.quote(self.a.policy_acceleration)} "
                      f"SANA_PREDICTION_VIDEOS={1 if self.a.prediction_videos == 'on' else 0} "
                      f"SANA_ROBODOJO_LAZY_RENDER={1 if self.a.lazy_render == 'on' else 0} "
                      f"nohup setsid bash {self.a.remote_install}/eval/{self.a.worker} {argv} > {shlex.quote(udir_a)}/nohup.log 2>&1 < /dev/null & "
                      f"sleep 2; kill -0 $! 2>/dev/null && echo dispatched $! || echo died")
            r = self.ssh(slot["node"], remote, timeout=90)
            if r is not None and r.returncode == 0 and "dispatched" in r.stdout:
                ok = True
            elif r is not None and ("died" in r.stdout or "mkdir_failed" in r.stdout):
                ok = False
                self.log(f"dispatch FAILED {uid} -> {slot['slot_id']}: "
                         + ("could not create " + udir_a if "mkdir_failed" in r.stdout else f"worker exited immediately (see {udir_a}/nohup.log)"))
            else:
                # ssh timed out or dropped: the worker may well be running. Ask before deciding.
                time.sleep(5)
                alive = self.worker_alive(slot, uid)
                ok = bool(alive)
                self.log(f"dispatch {uid} -> {slot['slot_id']}: ssh inconclusive (rc={None if r is None else r.returncode}); "
                         f"worker alive check -> {alive}")
        if ok:
            unit.update(state="running", slot=slot["slot_id"], attempts=attempt)
            unit["history"].append({"attempt": attempt, "slot": slot["slot_id"], "dispatched": now()})
            slot.update(unit=uid, dispatched_at=time.time())
            self.log(f"dispatch {uid} (eps {unit['episodes_per_task']}) -> {slot['slot_id']} rpc {slot['rpc_port']} attempt {attempt}")
        else:
            self.slot_failure(slot, "dispatch")
            unit["infra_failures"] = unit.get("infra_failures", 0) + 1
            self.free_slot_ports(slot, "after failed dispatch")
        return ok

    def remote_kill(self, unit: dict, slot: dict) -> None:
        if self.dry_root:
            try:
                os.kill(int((self.dry_root / "units" / unit["unit_id"] / "unit.pid").read_text()), signal.SIGTERM)
            except (OSError, ValueError):
                pass
            return
        udir = f"{self.run_root}/units/{unit['unit_id']}"
        self.ssh(slot["node"], f"p=$(cat {shlex.quote(udir)}/unit.pid 2>/dev/null); [ -n \"$p\" ] && kill -TERM \"$p\" 2>/dev/null; echo sent", timeout=60)

    # ---- bookkeeping -----------------------------------------------------------------
    def slot_failure(self, slot: dict, status: str) -> None:
        if status not in SLOT_LEVEL:
            slot["consecutive_failures"] = 0
            return
        slot["consecutive_failures"] += 1
        if slot["consecutive_failures"] >= 2 and not slot["quarantined"]:
            slot["quarantined"] = True
            self.log(f"slot {slot['slot_id']} quarantined after 2 consecutive infrastructure failures ({status})")

    def fail_unit(self, u: dict, slot: dict, status: str, note: str) -> None:
        """Infrastructure outcomes (port busy, server failed to start, dispatch died, stale after an outage...)
        are the slot's fault and never spend the unit's attempts; only real evaluation failures
        (incomplete/unknown) count against --max-attempts.  A loose cap on infrastructure failures keeps a
        unit from cycling forever through broken slots."""
        self.slot_failure(slot, status)
        u.setdefault("failures", 0); u.setdefault("infra_failures", 0)
        if status in SLOT_LEVEL:
            u["infra_failures"] += 1
        else:
            u["failures"] += 1
        budget = f"real failures {u['failures']}/{self.a.max_attempts}, infrastructure {u['infra_failures']}/{3 * self.a.max_attempts}"
        if u["failures"] >= self.a.max_attempts or u["infra_failures"] >= 3 * self.a.max_attempts:
            u["state"] = "failed"
            self.log(f"FAILED {u['unit_id']} permanently after {u['attempts']} attempts ({status}: {note[:120]}; {budget})")
        else:
            u["state"] = "queued"; u["avoid_slot"] = slot["slot_id"]
            self.log(f"RE-QUEUE {u['unit_id']} ({status}: {note[:120]}); {budget}")

    def start_drain(self, slot: dict, uid: str, reason: str) -> None:
        slot["draining"] = {"unit": uid, "since": time.time(), "reason": reason}
        slot["unit"] = None; slot["dispatched_at"] = None
        self.log(f"slot {slot['slot_id']} draining ({reason}); reused once the worker pid of {uid} is gone")

    def heartbeat_age(self, adir: Path):
        hb = adir / "heartbeat"
        if not hb.exists():
            return None
        ages = [time.time() - hb.stat().st_mtime]
        try:
            ages.append(time.time() - dt.datetime.strptime(hb.read_text().strip(), "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=dt.timezone.utc).timestamp())
        except ValueError:
            pass
        return min(ages)

    def step_lim(self, task: str) -> int:
        """per-episode step limit of a task config (task/RoboDojo/tasks/<task>.py: self.step_lim = N); 2000 if unknown"""
        cache = self.__dict__.setdefault("_step_lim", {})
        if task not in cache:
            f = Path(self.a.checkout) / "task" / "RoboDojo" / "tasks" / f"{task}.py"
            m = re.search(r"self\.step_lim\s*=\s*(\d+)", f.read_text()) if f.exists() else None
            cache[task] = int(m.group(1)) if m else 2000
        return cache[task]

    def unit_budget(self, u: dict) -> float:
        """seconds a unit may legitimately take: the worker's client watchdog (same formula as run_unit.sh)
        plus server start-up and teardown"""
        per_ep = max(self.a.episode_seconds, self.step_lim(u["task"]) * self.a.sec_per_step + self.a.reset_seconds)
        return 1800 + u["episodes_per_task"] * per_ep + 7200

    def process_controls(self) -> None:
        """operator control files in <state-dir>/control/: requeue.<unit_id> (TERM the running worker and put the
        unit back in the queue; also re-queues a permanently failed unit), pause (no new dispatches while present),
        pause.<label> (no new dispatches for that --settings label only; its running units finish),
        add_slots (content 'l8:0,2;l25:0': append those GPU pairs as new slots, then the file is removed),
        drop_slots (same content: forget those pairs if idle, so another master can use them; busy/draining ones are kept)"""
        cdir = self.state_dir / "control"
        if not cdir.is_dir():
            return
        add = cdir / "add_slots"
        if add.is_file():
            spec = add.read_text().strip()
            try:
                added = self.add_slots(spec)
                self.log(f"control: add_slots {spec!r} -> " + (f"added {added}" if added else "nothing new (every pair is already a slot of this run)"))
            except ValueError as e:
                self.log(f"control: add_slots {spec!r} rejected: {e}")
            add.unlink()
        drop = cdir / "drop_slots"
        if drop.is_file():
            spec = drop.read_text().strip()
            try:
                dropped, kept = self.drop_slots(spec)
                self.log(f"control: drop_slots {spec!r} -> dropped {dropped}" + (f"; kept (running or draining) {kept}" if kept else ""))
                self.save()
            except ValueError as e:
                self.log(f"control: drop_slots {spec!r} rejected: {e}")
            drop.unlink()
        for f in sorted(cdir.iterdir()):
            if not f.name.startswith("requeue."):
                continue
            uid = f.name.split(".", 1)[1]
            u = next((x for x in self.units if x["unit_id"] == uid), None)
            if u is None:
                self.log(f"control: unknown unit {uid!r} (file removed)")
            elif u["state"] == "running":
                slot = next(x for x in self.slots if x["slot_id"] == u["slot"])
                self.log(f"control: operator requeue of {uid} on {slot['slot_id']} -> TERM + drain (attempt {u['attempts']} discarded)")
                self.remote_kill(u, slot)
                u["history"][-1].update(finished=now(), status="requeued_by_operator")
                u["state"] = "queued"; u["avoid_slot"] = None
                self.start_drain(slot, uid, "operator requeue")
            elif u["state"] == "failed":
                u["state"] = "queued"; u["avoid_slot"] = None
                u["failures"] = min(u.get("failures", 0), self.a.max_attempts - 1)
                u["infra_failures"] = 0
                self.log(f"control: operator requeue of permanently failed {uid} "
                         f"(real failures now {u['failures']}/{self.a.max_attempts}, infrastructure counter reset)")
            else:
                self.log(f"control: requeue.{uid} ignored, unit is {u['state']}")
            f.unlink()

    def reconcile(self, pull_ok: bool) -> None:
        by_id = {s["slot_id"]: s for s in self.slots}
        judge_liveness = (pull_ok and (time.time() - self.last_pull_ok) < 2 * self.a.poll + 30
                          and time.time() >= self.liveness_grace_until)
        for u in self.units:
            if u["state"] != "running":
                continue
            slot = by_id[u["slot"]]
            adir = self.attempt_dir(u["unit_id"], u["attempts"])
            res = adir / "result.json"
            if res.exists():
                try:
                    r = json.loads(res.read_text())
                except json.JSONDecodeError:
                    continue
                u["history"][-1].update(finished=now(), status=r.get("status"), episodes=r.get("episodes_recorded"),
                                        successes=r.get("successes"), score_sum=r.get("score_sum"))
                if r.get("status") == "complete":
                    u["state"] = "done"; u["result"] = r
                    slot["consecutive_failures"] = 0
                    self.log(f"DONE {u['unit_id']} on {slot['slot_id']}: {r['successes']}/{r['episodes_recorded']} successes, "
                             f"score_sum {float(r['score_sum']):.2f}")
                    slot.update(unit=None, dispatched_at=None)   # normal path: result is written after the server is stopped
                else:
                    self.fail_unit(u, slot, r.get("status", "unknown"), r.get("note", ""))
                    self.start_drain(slot, u["unit_id"], r.get("status", "unknown"))   # teardown may still be running
                continue
            if not judge_liveness:
                continue
            age = self.heartbeat_age(adir)
            since = time.time() - (slot["dispatched_at"] or time.time())
            max_runtime = self.a.heartbeat_timeout + self.unit_budget(u)
            stale = (age is not None and age > self.a.heartbeat_timeout) or (age is None and since > self.a.heartbeat_timeout)
            if stale or since > max_runtime:
                why = "no heartbeat" if stale else f"exceeded max runtime {int(max_runtime)}s"
                self.log(f"STALE {u['unit_id']} on {slot['slot_id']}: {why} (heartbeat age {None if age is None else int(age)}s, "
                         f"dispatched {int(since)}s ago) -> TERM + drain")
                self.remote_kill(u, slot)
                u["history"][-1].update(finished=now(), status="stale", note=why)
                self.fail_unit(u, slot, "stale", why)
                self.start_drain(slot, u["unit_id"], "stale")

    def drain(self) -> None:
        for slot in self.slots:
            d = slot.get("draining")
            if not d:
                continue
            alive = self.worker_alive(slot, d["unit"])
            waited = time.time() - d["since"]
            if alive is False:
                slot["draining"] = None
                self.log(f"slot {slot['slot_id']} drained ({int(waited)}s)")
                self.free_slot_ports(slot, f"after {d['reason']}")   # a server whose stop failed would block the next unit
            elif alive is None:
                self.log(f"slot {slot['slot_id']}: node unreachable while draining ({int(waited)}s)")
            elif waited > self.a.drain_timeout:
                self.log(f"slot {slot['slot_id']}: worker still alive after {int(waited)}s; SIGKILL and release")
                if not self.dry_root:
                    udir = f"{self.run_root}/units/{d['unit']}"
                    self.ssh(slot["node"], f"p=$(cat {shlex.quote(udir)}/unit.pid 2>/dev/null); [ -n \"$p\" ] && pkill -KILL -P \"$p\"; "
                                          f"kill -KILL \"$p\" 2>/dev/null; for s in {shlex.quote(udir)}/a*/server; do bash {self.a.remote_install}/stop_server.sh \"$s\" >/dev/null 2>&1; done; echo ok", timeout=120)
                slot["draining"] = None
                self.free_slot_ports(slot, "after SIGKILL")

    def schedule(self) -> None:
        if (self.state_dir / "control" / "pause").exists():
            if not getattr(self, "_paused", False):
                self.log("control: pause file present; no new dispatches until it is removed")
            self._paused = True
            return
        if getattr(self, "_paused", False):
            self._paused = False; self.log("control: pause file removed; dispatching resumes")
        # control/pause.<label>: no new dispatches for that --settings label only (its running units finish; its queued
        # units wait and are skipped over), so all slots go to the other setting(s) until the file is removed
        paused_settings = self.paused_settings()
        if paused_settings != getattr(self, "_paused_settings", set()):
            self.log("control: dispatch paused for setting(s) " + (", ".join(sorted(paused_settings)) if paused_settings else "-- none (all settings dispatch again)"))
            self._paused_settings = paused_settings
        free = [s for s in self.slots if s["unit"] is None and not s["quarantined"] and not s.get("draining")]
        # a unit whose previous worker is still draining holds its unit lock on the node: dispatching it
        # elsewhere now would just die on flock -n, so it waits until that slot reports drained
        draining_units = {s["draining"]["unit"] for s in self.slots if s.get("draining")}
        queued = [u for u in self.units if u["state"] == "queued" and u["unit_id"] not in draining_units
                  and u.get("setting") not in paused_settings]
        for slot in free:
            pick = next((u for u in queued if u.get("avoid_slot") != slot["slot_id"]), None) or (queued[0] if queued else None)
            if pick is None:
                break
            queued.remove(pick)
            self.dispatch(pick, slot)
        if queued and all(s["unit"] is None and not s.get("draining") for s in self.slots) and any(s["quarantined"] for s in self.slots):
            for s in self.slots:
                s["quarantined"] = False; s["consecutive_failures"] = 0
            self.log("all slots idle with work left: lifting quarantines")

    def paused_settings(self) -> set:
        """Labels named by <state-dir>/control/pause.<label> files (per-setting dispatch pause; `pause` alone pauses everything)."""
        cdir = self.state_dir / "control"
        return {f.name.split(".", 1)[1] for f in cdir.glob("pause.*") if f.name != "pause"} if cdir.is_dir() else set()

    # ---- reporting -------------------------------------------------------------------
    def status_table(self) -> str:
        c = {k: sum(1 for u in self.units if u["state"] == k) for k in ("queued", "running", "done", "failed")}
        eps_done = sum(u["episodes_per_task"] for u in self.units if u["state"] == "done")
        eps_total = sum(u["episodes_per_task"] for u in self.units)
        paused = self.paused_settings()
        lines = [f"units: queued {c['queued']}  running {c['running']}  done {c['done']}  failed {c['failed']}  | episodes {eps_done}/{eps_total}"
                 f"  | last successful pull {int(time.time() - self.last_pull_ok)}s ago"
                 + (f"  | DISPATCH PAUSED for setting(s): {', '.join(sorted(paused))}" if paused else ""), "slots:"]
        for s in self.slots:
            flag = "QUARANTINED " if s["quarantined"] else ("DRAINING " if s.get("draining") else "")
            age = ""
            if s["dispatched_at"]:
                u = next((x for x in self.units if x["unit_id"] == s["unit"]), None)
                age = f"{int(time.time() - s['dispatched_at'])//60} min" + (f" (budget {self.unit_budget(u)/3600:.1f} h, step_lim {self.step_lim(u['task'])})" if u else "")
            lines.append(f"  {s['slot_id']:14s} rpc {s['rpc_port']}  {flag}{s['unit'] or '-'} {age}")
        return "\n".join(lines)

    def write_status_md(self) -> None:
        settings_md = []
        seen_settings = {}
        for u in self.units:
            if u.get("setting") and u["setting"] not in seen_settings:
                seen_settings[u["setting"]] = f"- `{u['setting']}`: overrides `{self.unit_overrides(u)}`, checkpoint `{self.unit_checkpoint(u)}`"
        if seen_settings:
            settings_md = ["settings (unit id suffix `__label`):", *seen_settings.values(), ""]
        md = [f"# master status `{Path(self.run_root).name}` ({now()} UTC)", "", f"checkpoint: `{self.a.checkpoint_root}`", *settings_md,
              f"worker: `{self.a.worker}`" + (f" (XPolicyLab `{self.a.xpl_root}`, action_type `{self.a.xpl_action_type}`)" if self.a.xpl_root else ""), "",
              "```", self.status_table(), "```", "",
              "| unit | state | attempts (real/infra failures) | slot | successes/episodes | score_sum |", "|---|---|---|---|---|---|"]
        for u in sorted(self.units, key=lambda x: (x["state"], x["unit_id"])):
            r = u.get("result") or {}
            se = f"{r.get('successes','')}/{r.get('episodes_recorded','')}" if r else ""
            md.append(f"| {u['unit_id']} | {u['state']} | {u['attempts']} ({u.get('failures', 0)}/{u.get('infra_failures', 0)}) | "
                      f"{u.get('slot') or ''} | {se} | {r.get('score_sum','') if r else ''} |")
        (self.state_dir / "STATUS.md").write_text("\n".join(md) + "\n")

    # ---- main loop -----------------------------------------------------------------
    def run(self) -> int:
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop", True))
        if self.a.resume and self.state_file.exists():
            self.load(); self.log(f"resumed state from {self.state_file}; checkpoint {self.a.checkpoint_root}")
            self.wait_for_checkpoint()
            # slots the saved state does not know get appended (same GPU-pair/port scheme) so a run grows without a
            # restart: every pair of a node new to the run and, only when the pair is named explicitly in --node-pairs,
            # new pairs on a node the run already uses (--nodes alone never grows a known node, so a plain `--resume`
            # with the default node list stays a no-op). Slots are never removed. Running masters: control/add_slots.
            known_nodes = {s["node"] for s in self.slots}
            explicit = self.node_pairs()
            for node in list(self.a.nodes):
                if node in known_nodes and node not in explicit:
                    continue
                spec = f"{node}:{','.join(str(k) for k in self.pairs_for(node))}"
                try:
                    added = self.add_slots(spec, sweep=False)     # the sweep below covers every free slot
                except ValueError as e:
                    sys.exit(f"resume: {e}")
                if added:
                    self.log(f"resume: added {len(added)} slot(s) on {node}: {added}")
            # every free slot gets its ports reclaimed and its quarantine lifted: a quarantine is a symptom
            # of leftovers on the slot, and the sweep is what removes them
            for s in self.slots:
                if s["unit"] is None and not s.get("draining"):
                    self.free_slot_ports(s, "resume sweep")
                    if s["quarantined"] or s["consecutive_failures"]:
                        self.log(f"resume: quarantine/failure counter of {s['slot_id']} cleared after the port sweep")
                        s["quarantined"] = False; s["consecutive_failures"] = 0
        else:
            if self.state_file.exists():
                sys.exit(f"{self.state_file} exists: pass --resume to continue it, or use a new RUN_NAME/--state-dir")
            shutil.rmtree(self.mirror / "units", ignore_errors=True)
            if self.dry_root:
                shutil.rmtree(self.dry_root, ignore_errors=True)
            self.units, self.slots = self.build_units(), self.build_slots()
            self.wait_for_checkpoint()          # after planning: per-setting checkpoints are known only from the units
            (self.state_dir / "units.json").write_text(json.dumps(self.units, indent=1) + "\n")
            self.log(f"planned {len(self.units)} units{' (' + str(self.a.shards) + ' layout shards per task config)' if getattr(self.a, 'shards', 1) > 1 else ''}, {sum(u['episodes_per_task'] for u in self.units)} episodes, "
                     f"{len(self.slots)} slots on {self.a.nodes}, worker {self.a.worker}"
                     + (f" (XPolicyLab {self.a.xpl_root})" if self.a.xpl_root else "")
                     + (f" overrides {self.a.xpl_overrides!r}" if self.a.xpl_overrides else "")
                     + (f" action_type {self.a.xpl_action_type}" if self.a.worker == "run_unit_xpl.sh" else "")
                     + (f" worker-env {self.a.worker_env!r}" if self.a.worker_env else "") + f", checkpoint {self.a.checkpoint_root}, "
                     f"policy acceleration preset {self.a.policy_acceleration!r}, "
                     f"prediction videos {self.a.prediction_videos}, lazy render {self.a.lazy_render}")
            if not self.dry_root:
                node = self.a.nodes[0]
                r = self.ssh(node, f"mkdir -p {shlex.quote(self.run_root)}/units && echo ok", timeout=60)
                if r is None or r.returncode != 0:
                    sys.exit(f"cannot create {self.run_root} via {node}: {'timeout' if r is None else r.stderr.strip()}")
                try:
                    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(self.state_dir / "units.json"), str(self.state_dir / "tasks.json"),
                                    f"{self.user}@{node}{self.domain}:{self.run_root}/"], check=False, timeout=120)
                except subprocess.TimeoutExpired:
                    self.log("WARN: scp of units.json/tasks.json to the run root timed out (informational files only)")
        self.save()
        last_report = 0.0
        while not self.stop:
            ok = self.pull()
            self.reconcile(ok); self.process_controls(); self.drain(); self.schedule(); self.save()
            if time.time() - last_report >= self.a.report_every:
                self.log("\n" + self.status_table()); self.write_status_md(); last_report = time.time()
            if not any(u["state"] in ("queued", "running") for u in self.units) and not any(s.get("draining") for s in self.slots):
                break
            for _ in range(self.a.poll):        # interruptible sleep
                if self.stop:
                    break
                time.sleep(1)
        self.write_status_md(); self.save()
        if self.stop:
            self.log("stop requested; workers keep running (re-run with --resume to continue tracking)")
            return 130
        failed = [u["unit_id"] for u in self.units if u["state"] == "failed"]
        self.log(f"ALL UNITS SETTLED: done {sum(1 for u in self.units if u['state']=='done')}, failed {len(failed)} {failed}")
        self.summarize()
        return 0 if not failed else 2

    def summarize(self) -> None:
        root = self.dry_root if self.dry_root else self.mirror
        for f in ("units.json", "tasks.json"):
            (root / f).write_text((self.state_dir / f).read_text())
        labels = sorted({u["setting"] for u in self.units if u.get("setting")})
        for label in (labels or [None]):
            out = self.state_dir / ("RESULTS.md" if label is None else f"RESULTS_{label}.md")
            sample = next((u for u in self.units if label is None or u.get("setting") == label), {})
            note = (f"checkpoint `{self.unit_checkpoint(sample) if sample else self.a.checkpoint_root}`; worker `{self.a.worker}`"
                    + (f" (XPolicyLab `{self.a.xpl_root}`)" if self.a.xpl_root else "")
                    + (f"; setting `{label}` overrides `{self.unit_overrides(sample)}`" if label else
                       (f"; overrides `{self.a.xpl_overrides}`" if self.a.xpl_overrides else ""))
                    + f"; policy acceleration `{self.a.policy_acceleration}`, prediction videos {self.a.prediction_videos}, lazy render {self.a.lazy_render}")
            cmd = [sys.executable, str(HERE / "summarize_eval.py"), str(root), "--out", str(out), "--run-id", Path(self.run_root).name, "--note", note]
            if label:
                cmd += ["--setting", label]
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.log((f"[{label}] " if label else "") + (r.stdout[-3000:] if r.returncode == 0 else f"summarize failed: {r.stderr[-2000:]}"))
        if labels:
            run = Path(self.run_root).name
            r = subprocess.run([sys.executable, str(HERE / "compare_runs.py"), *[f"{self.state_dir.name}@{l}" for l in labels],
                                "--results-root", str(self.state_dir.parent), "--runs-subdir", ".",
                                "--out", str(self.state_dir / "COMPARE_settings.md")], capture_output=True, text=True)
            self.log(r.stdout[-2500:] if r.returncode == 0 else f"compare failed: {r.stderr[-1500:]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True, help="node-side path on the shared home, must contain 'robodojo'")
    ap.add_argument("--checkout", required=True, help="RoboDojo clone on THIS machine (for task enumeration)")
    ap.add_argument("--nodes", default=None, help="comma list of nodes, also the pull order (default CE_NODES)")
    ap.add_argument("--slots-per-node", type=int, default=4)
    ap.add_argument("--slot-offset", type=int, default=0, help="first GPU pair index to use on each node (0 = GPUs 0/1); lets several masters share a node")
    ap.add_argument("--node-pairs", default=None,
                    help="per-node GPU pair lists overriding --slot-offset/--slots-per-node, e.g. 'l3:1,2,3;l26:1,2,3' "
                         "(pair k = CUDA devices 2k/2k+1; skip pairs whose GPU is broken or occupied by another user). "
                         "'s<g>' adds a SINGLE-GPU slot (device g runs server and client), e.g. 'l3:s1;l26:s0'")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--support-arm-embodiments", choices=("on", "off"), default="on",
                    help="'on' (DEFAULT since 2026-09-24, user rule: every official run covers the COMPLETE RoboDojo "
                         "protocol, 42 tasks / 54 configs / 2100 episodes per seed) also plans the tasks whose robot_config "
                         "adds arms that are ALL `type: support` (make_kong, play_tic_tac_toe, imitate_sorting_sequence: the "
                         "task replays those arms itself, the policy is only ever asked for the two x5 arms). 'off' only "
                         "reproduces the historical 51-config population of the runs made before 2026-09-24.")
    ap.add_argument("--dimensions", default=None,
                    help="comma list of capability dimensions to plan (generalization,precision,long-horizon,memory,open); "
                         "combine with --settings to sweep one category under several settings on all workers at once")
    ap.add_argument("--settings", default=None,
                    help="'label:key=v key2=v;label2:...' -- run every planned task config once per setting (unit ids get a __label "
                         "suffix; tokens go to the policy server as deploy.yml overrides; 'checkpoint_root=/path' inside a setting "
                         "picks another checkpoint). Results: RESULTS_<label>.md per setting + COMPARE_settings.md")
    ap.add_argument("--episodes-per-task", type=int, default=None)
    ap.add_argument("--dynamic-instruction-tasks", choices=["auto", "on", "off"], default="auto",
                    help="plan general_pickup / stack_blocks_by_language / classify_objects_by_language / pour_by_language, whose "
                         "instruction is rendered per episode by the official evaluator (obs['instruction']). auto = on for the "
                         "XPolicyLab worker, off for run_unit.sh (the SANA RPC lane needs one frozen prompt per task)")
    ap.add_argument("--order", choices=["dimension", "size"], default="dimension",
                    help="dispatch order: 'dimension' (default) runs the task categories one after another so a complete "
                         "per-category comparison is available as early as possible; 'size' = most episodes first")
    ap.add_argument("--dimension-order", default=DIMENSION_ORDER_AUTO,
                    help="comma list of categories for --order dimension (e.g. precision,memory,open,long-horizon,generalization); "
                         "'auto' = ascending total work (episodes x step limit)")
    ap.add_argument("--pin-longest", type=int, default=2,
                    help="with --order dimension, dispatch this many of the longest units first regardless of category so the "
                         "run's tail (fasten_screws, play_stacking_toy) does not grow; 0 = strict category order")
    ap.add_argument("--shard-plan", default=None,
                    help="per-task layout shard counts 'fasten_screws=4,play_stacking_toy=2' (overrides --shards for those tasks; XPL lane only)")
    ap.add_argument("--shards", type=int, default=1,
                    help="split every task config's layouts into this many contiguous ranges that run as separate units "
                         "on separate slots (unit ids <task>__s<seed>__p<i>of<k>; XPL lane only -- the worker pre-seeds the "
                         "official client's resume manifest; summarize_eval.py merges the ranges by layout id)")
    ap.add_argument("--no-random", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--report-every", type=int, default=600)
    ap.add_argument("--heartbeat-timeout", type=int, default=900)
    ap.add_argument("--drain-timeout", type=int, default=1500, help="seconds to wait for a killed worker before SIGKILL + release")
    ap.add_argument("--episode-seconds", type=int, default=600, help="per-episode budget FLOOR (worker EPISODE_SECONDS); the real budget is max(this, step_lim*sec_per_step+reset)")
    ap.add_argument("--sec-per-step", type=float, default=2.5, help="per-policy-step budget (worker SEC_PER_STEP); observed ~0.85 s/step")
    ap.add_argument("--reset-seconds", type=int, default=120, help="per-episode reset budget (worker RESET_SECONDS)")
    ap.add_argument("--lazy-render", default="off", choices=("on", "off"),
                    help="client renders/captures only at replan steps. DEFAULT OFF since 2026-09-09: the official evaluator "
                         "calls get_obs() after every action and that call advances the simulator's render/settle cadence, so "
                         "skipping it changes the dynamics the policy acts in -- insert_key ep7 stage-1 hits 25/50 (off) vs 11-12/50 (on). "
                         "Runs official_seed0_ep10s65690 and official_seed0_ep7s45000_lr5e5_clip1 were taken with it ON.")
    ap.add_argument("--prediction-videos", default="off", choices=("on", "off"),
                    help="server-side diagnostic prediction videos (one VAE decode + MP4 per chunk INSIDE the timed inference, "
                         "~2.7 s of ~15 s); default off for full evaluations (no policy numerics change); on for debugging")
    ap.add_argument("--policy-acceleration", default="off", choices=("off", "fused", "fused_fp8", "fused_portable"),
                    help="server-side RWM_EVAL_POLICY_ACCELERATION preset (rwm/zekai-merge): fused/fused_fp8 need SM90 (H100); "
                         "fused_portable is the Triton/torch subset for the A6000s; success rates are not comparable across presets")
    ap.add_argument("--checkpoint-root", default=None,
                    help="node-side absolute path of the policy checkpoint snapshot every unit serves (required for a new run; "
                         "recorded in state.json -- a resumed run keeps its own checkpoint")
    ap.add_argument("--worker", default="run_unit.sh", choices=("run_unit.sh", "run_unit_xpl.sh"),
                    help="node-side worker script under <--remote-install>/eval/: run_unit.sh = the SANA RPC server lane; "
                         "run_unit_xpl.sh = the official XPolicyLab websocket lane (an in-process adapter: policy/SANA_WAM by "
                         "default, --worker-env SANA_EVAL_XPL_POLICY=SANA_MOT|OpenWAM for the others)")
    ap.add_argument("--xpl-root", default=None,
                    help="node-side path of the XPolicyLab checkout for run_unit_xpl.sh (default the worker's own: CE_REMOTE_XPL)")
    ap.add_argument("--worker-env", default=None,
                    help="extra space-separated KEY=VALUE pairs exported into every worker's environment "
                         "(reach the server and client processes), e.g. 'DISABLE_XFORMERS=1'")
    ap.add_argument("--xpl-action-type", choices=("joint", "ee"), default="joint",
                    help="run_unit_xpl.sh only: the adapter's action_type -- joint (24 joint targets per chunk, any checkpoint) or ee "
                         "(24 end-effector poses solved by the evaluator's IK; needs a robot_base_eef checkpoint). Persisted in state.json; "
                         "a resumed run keeps its own")
    ap.add_argument("--xpl-overrides", default=None,
                    help="run_unit_xpl.sh only: extra space-separated key=value deploy.yml overrides for the policy server "
                         "(e.g. 'diffusion_seed_base=20260803')")
    ap.add_argument("--wait-for-checkpoint", type=int, default=0,
                    help="seconds to wait for the checkpoint snapshot to be complete on the shared home (config.yaml, README.md, "
                         "model/pytorch_model_fsdp.bin, normalization/*.json, no *.incomplete download) before planning; 0 = must be ready now")
    # 61000+ keeps the slot ports OUT of Linux's ephemeral range (32768-60999): with 41000/42000 three of
    # sixteen first-round servers hit EADDRINUSE on 2026-09-07 because outgoing connections on the node
    # (Isaac/Kit start-up) had grabbed their port between the worker's check and the server's bind.
    # Runs started with other bases keep them across --resume (slots live in state.json).
    ap.add_argument("--rpc-port-base", type=int, default=61000)
    ap.add_argument("--ws-port-base", type=int, default=62000)
    ap.add_argument("--ssh-user", default=None, help="default CE_SSH_USER")
    ap.add_argument("--ssh-domain", default=None, help="default CE_SSH_DOMAIN")
    ap.add_argument("--remote-install", default=None,
                    help="node-side directory holding eval/<worker> and stop_server.sh (default CE_REMOTE_INSTALL; deploy.sh fills it)")
    ap.add_argument("--state-dir", default=None, help="default CE_MASTER_ROOT/<run name>")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="no ssh: units run locally via run_unit.sh EVAL_DRY_RUN=1")
    ap.add_argument("--dry-sleep", type=float, default=2.0)
    a = ap.parse_args()
    if "robodojo" not in a.run_root:
        sys.exit("--run-root must contain 'robodojo'")
    a.nodes = [n.strip() for n in (a.nodes or ",".join(ce_site.get("CE_NODES").split())).split(",") if n.strip()]
    a.checkpoint_root_explicit = a.checkpoint_root is not None
    if a.checkpoint_root is None and not a.resume and not a.dry_run:
        sys.exit("--checkpoint-root is required for a new run (<CE_CKPT_ROOT>/<label>)")
    a.checkpoint_root = (a.checkpoint_root or "/resumed-run-keeps-its-own-checkpoint").rstrip("/")
    if not a.checkpoint_root.startswith("/"):
        sys.exit("--checkpoint-root must be an absolute node-side path (no ~)")
    if a.wait_for_checkpoint < 0:
        sys.exit("--wait-for-checkpoint must be >= 0")
    a.ssh_user = a.ssh_user or ce_site.get("CE_SSH_USER")
    a.ssh_domain = a.ssh_domain if a.ssh_domain is not None else ce_site.get("CE_SSH_DOMAIN")
    a.remote_install = (a.remote_install or ce_site.get("CE_REMOTE_INSTALL")).rstrip("/")
    a.state_dir = a.state_dir or str(Path(ce_site.get("CE_MASTER_ROOT")) / Path(a.run_root).name)
    return Master(a).run()


if __name__ == "__main__":
    sys.exit(main())
