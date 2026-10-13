#!/usr/bin/env python3
"""three_seed_report.py -- compare two runs' 3-seed OFFICIAL_SUMMARY_combined files task by task.

usage: three_seed_report.py A_SUMMARY B_SUMMARY TASKS_JSON [--labels A B] [--out FILE.md]

Successes per task config and seed come from the per-seed SR columns: the generalization configs from the
"Per-task Standard vs Random" table (25 episodes each, standard and random kept apart), every other task from the
all-tasks table (eval_nums from tasks.json, 50). Dimension and overall numbers are copied from each summary's own
overview tables (the official mean-of-dimension-means rule). Per-task significance: two-sided Fisher exact test on the
successes pooled over the three seeds.
"""
import argparse
import json
import re
from math import comb

DIM_ORDER = ["generalization", "precision", "long-horizon", "memory", "open"]   # tasks.json spelling
DIM_NAME = {"generalization": "Generalization", "precision": "Precision", "long-horizon": "Long-Horizon",
            "memory": "Memory", "open": "Open"}


def fisher(a, n1, b, n2):
    k, n = a + b, n1 + n2
    if k == 0 or k == n:
        return 1.0
    p = lambda x: comb(n1, x) * comb(n2, k - x) / comb(n, k)
    p0 = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= p0 * (1 + 1e-9)))


def rows_of(section):
    out = {}
    header = None
    for line in section.splitlines():
        if line.startswith("| Task"):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            continue
        m = re.match(r"\|\s*<nobr>([^<]+)</nobr>\s*\|(.*)", line)
        if not (m and header):
            continue
        cells = [c.strip() for c in m.group(2).strip().strip("|").split("|")]
        rec = dict(zip(header[1:], cells))
        sr = [rec.get(f"Seed {s} SR (%)", "") for s in range(3)]
        sc = [rec.get(f"Seed {s} Score", "") for s in range(3)]
        out.setdefault(m.group(1), (sr, sc))
    return out


def parse(path):
    text = open(path).read()
    gen_part = text.split("### Per-task Standard vs Random", 1)[1].split("\n## ", 1)[0]
    rest = text.split("### Per-task Standard vs Random", 1)[1].split("\n## ", 1)[1]
    gen = rows_of(gen_part)
    allt = rows_of(rest)
    ov = re.search(r"\n\| SANA_\w+ \| ([^|]+)\|(.*)", text)
    cells = [c.strip() for c in ov.group(2).strip().strip("|").split("|")]
    overview = dict(zip(["gen_sr", "gen_score", "prec_sr", "prec_score", "lh_sr", "lh_score", "mem_sr", "mem_score",
                         "open_sr", "open_score", "avg_sr", "avg_score"], cells))
    sv = re.search(r"Standard vs Random \(25 episodes each\).*?\n\| SANA_\w+ \|(.*)", text, re.S)
    sv_cells = [c.strip() for c in sv.group(1).strip().strip("|").split("|")]
    overview.update(dict(zip(["std_sr", "std_score", "rand_sr", "rand_score"], sv_cells)))
    return gen, allt, overview


def successes(per_seed_sr, n):
    vals = []
    for v in per_seed_sr:
        if v == "":
            return None
        vals.append(round(float(v) * n / 100))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("tasks")
    ap.add_argument("--labels", nargs=2, default=["A", "B"]); ap.add_argument("--out")
    a = ap.parse_args()
    tasks = json.load(open(a.tasks))
    A, B = parse(a.a), parse(a.b)
    la, lb = a.labels
    lines = []
    tot = {la: 0, lb: 0}
    per_seed = {la: [0, 0, 0], lb: [0, 0, 0]}
    table = []
    for name, info in tasks.items():
        dim, n = info["dimension"], int(info["eval_nums"])
        src = 0 if dim == "generalization" else 1
        ra = (A[src] if src == 0 else A[1]).get(name)
        rb = (B[src] if src == 0 else B[1]).get(name)
        sa = successes(ra[0], n) if ra else None
        sb = successes(rb[0], n) if rb else None
        if sa is None or sb is None:
            table.append((dim, name, info.get("variant"), n, None, None, None)); continue
        for s in range(3):
            per_seed[la][s] += sa[s]; per_seed[lb][s] += sb[s]
        tot[la] += sum(sa); tot[lb] += sum(sb)
        table.append((dim, name, info.get("variant"), n, sa, sb, fisher(sum(sa), 3 * n, sum(sb), 3 * n)))
    N = 3 * sum(int(i["eval_nums"]) for i in tasks.values())
    oa, ob = A[2], B[2]
    lines.append(f"# {la} vs {lb} (3 seeds)\n")
    lines.append(f"| | {la} | {lb} |\n|---|---|---|")
    lines.append(f"| successes | {tot[la]}/{N} | {tot[lb]}/{N} |")
    lines.append(f"| per seed | {' / '.join(map(str, per_seed[la]))} | {' / '.join(map(str, per_seed[lb]))} |")
    lines.append(f"| overall SR / score | {oa['avg_sr']} / {oa['avg_score']} | {ob['avg_sr']} / {ob['avg_score']} |")
    for key, nm in (("gen", "Generalization"), ("std", "  standard"), ("rand", "  random"), ("prec", "Precision"),
                    ("lh", "Long-Horizon"), ("mem", "Memory"), ("open", "Open")):
        lines.append(f"| {nm} SR / score | {oa[key + '_sr']} / {oa[key + '_score']} | {ob[key + '_sr']} / {ob[key + '_score']} |")
    lines.append(f"\nFisher (pooled successes): p = {fisher(tot[la], N, tot[lb], N):.3g}\n")
    for dim in DIM_ORDER:
        rows = [r for r in table if r[0] == dim]
        if dim == "generalization":
            rows.sort(key=lambda r: (r[1].replace("_random", ""), r[2] != "standard"))
        lines.append(f"\n## {DIM_NAME[dim]}\n")
        lines.append(f"| task | {la} (s0/s1/s2) | {lb} (s0/s1/s2) | {la} sum | {lb} sum | diff | p |\n|---|---|---|---|---|---|---|")
        for dim_, name, var, n, sa, sb, p in rows:
            if sa is None:
                lines.append(f"| {name} | missing | missing | | | | |"); continue
            mark = " **" if p < 0.05 else ""
            lines.append(f"| {name} | {'/'.join(map(str, sa))} | {'/'.join(map(str, sb))} | {sum(sa)}/{3 * n} | {sum(sb)}/{3 * n} | "
                         f"{sum(sa) - sum(sb):+d} | {p:.3g}{mark} |")
    out = "\n".join(lines) + "\n"
    if a.out:
        open(a.out, "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
