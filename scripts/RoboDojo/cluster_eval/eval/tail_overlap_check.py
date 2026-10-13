"""tail_overlap_check.py RUN -- for every tail shard (units/*__sub*) of a run: node:gpu, start/end (status.txt on NFS), server bind errors; report overlaps."""
import json, os, re, subprocess, sys, datetime as dt
RUN = sys.argv[1]
RR = f"/homes/jiamingt/robodojo_eval/{RUN}"
M = os.path.expanduser(f"~/robodojo_master/{RUN}")
OUT = os.path.join(os.path.expanduser("~/robodojo_master"), RUN, "tail_shards", "overlap_check")
os.makedirs(OUT, exist_ok=True)
subprocess.run(["rsync", "-a", "--delete", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15", "--include=*/", "--include=status.txt",
                "--include=server.stderr.log", "--include=result.json", "--exclude=*", f"jiamingt@{os.environ.get('NFS_GATEWAY', 'l7')}.mit.edu:{RR}/units/", OUT + "/"],
               check=False, timeout=600)
rows = []
for uid in sorted(os.listdir(OUT)):
    if "__sub" not in uid: continue
    st = os.path.join(OUT, uid, "a1", "status.txt")
    if not os.path.exists(st): rows.append((uid, None, None, None, "no status")); continue
    txt = open(st).read()
    m = re.search(r"^(\S+) \S+ a1 START .* node=(\S+) gpus=(\d+)/", txt, re.M)
    e = re.search(r"^(\S+) \S+ a1 RESULT (\S+)", txt, re.M)
    err = os.path.join(OUT, uid, "a1", "server", "server.stderr.log")
    bind = "BIND-ERR" if os.path.exists(err) and "Errno 98" in open(err, errors="replace").read() else ""
    t0 = dt.datetime.fromisoformat(m.group(1)) if m else None
    t1 = dt.datetime.fromisoformat(e.group(1)) if e else dt.datetime.now(dt.timezone.utc)
    rows.append((uid, f"{m.group(2)}:{m.group(3)}" if m else None, t0, t1, (e.group(2) if e else "running") + (" " + bind if bind else "")))
by = {}
for r in rows:
    if r[1]: by.setdefault(r[1], []).append(r)
bad = set()
for g, rs in by.items():
    rs.sort(key=lambda r: r[2])
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            a, b = rs[i], rs[j]
            if b[2] < a[3]:
                print(f"OVERLAP on {g}: {a[0]} [{a[2]:%H:%M:%S}-{a[3]:%H:%M:%S} {a[4]}]  vs  {b[0]} [{b[2]:%H:%M:%S}-{b[3]:%H:%M:%S} {b[4]}]")
                bad.update({a[0], b[0]})
print("shards:", len(rows), "| involved in an overlap:", sorted(bad, key=lambda u: (u.split("__")[0], int(u.split("sub")[1]))))
print("BIND-ERR shards:", [r[0] for r in rows if "BIND-ERR" in r[4]])
