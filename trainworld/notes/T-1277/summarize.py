"""T-1277 tables from out/: python summarize.py"""
import glob
import json
import os
import re

D = os.path.dirname(os.path.abspath(__file__))
OUT = f"{D}/out"


def ceil_tables():
    print("## Ceilings and the sweep on B's train (harness)\n")
    for p in sorted(glob.glob(f"{OUT}/ceil_*.json")):
        j = json.load(open(p))
        city, var = re.match(r".*ceil_([a-z0-9]+?)_([\w.]+)\.json", p.replace("\\", "/")).groups()
        acc = j["acc"]
        sc = j["scens"]
        tot = lambda s, k: acc[2 * s][k] + acc[2 * s + 1][k]
        rc = lambda s: tot(s, 1) / (tot(s, 1) + tot(s, 3))
        a, b = sc.index("A"), sc.index("B")
        rows = []
        for i, name in enumerate(sc):
            if name.startswith("m"):
                rows.append((float(name[1:]), tot(i, 4), tot(i, 1), rc(i), tot(i, 4) / max(tot(i, 1), 1)))
        best = max(rows, key=lambda r: r[1])
        r1 = next(r for r in rows if r[0] == 1.0)
        print(f"{city} {var}: A {100*rc(a):.1f}, B {100*rc(b):.1f}; revenue-optimal x{best[0]:g} (base ${1.5*best[0]:.2f}, mean fare ${best[4]:.2f}), "
              f"revenue x{best[1]/r1[1]:.2f} of the starting fares', rail trips x{best[2]/r1[2]:.2f}")
        print("  " + " | ".join(f"x{m:g}: {rev/best[1]:.2f} ({100*sh:.0f}%)" for m, rev, _, sh, _ in rows))
    print()


def real_tables():
    print("## Real networks (calib)\n")
    runs = {}
    for p in glob.glob(f"{OUT}/real_*.json"):
        city, var, m = re.match(r".*real_(\w+?)_(\w+?)_([\d.]+)\.json", p.replace("\\", "/")).groups()
        runs.setdefault((city, var), []).append((float(m), p))
    for (city, var), lst in sorted(runs.items()):
        rows = []
        for m, p in sorted(lst):
            j = json.load(open(p))
            c, e = j["day"]["commute"], j["day"]["everyday"]
            rail = c[1] + e[1]
            pkm = sum(j["lines"]["pkm"])
            b, k = 1.5 * m, 0.10 * m
            rev = b * rail + k * pkm
            rc = rail / (c[0] + e[0] - c[2] - e[2])
            rows.append((m, rev, rail, pkm, rc, c[1], e[1]))
        best = max(rows, key=lambda r: r[1])
        r1 = next((r for r in rows if r[0] == 1.0), rows[0])
        print(f"### {city} {var}: optimum x{best[0]:g} (base ${1.5*best[0]:.2f} + ${0.1*best[0]:.2f}/km), mean fare ${best[1]/best[2]:.2f}, "
              f"revenue ${best[1]/1e6:.1f}M a real day (x{best[1]/r1[1]:.2f} of x1), rail trips x{best[2]/r1[2]:.2f}\n")
        print("| x | base $ | mean fare $ | rail trips M | commute / everyday rail M | pkm M | rail of rail + car % | revenue $M a real day | of the best |")
        print("|---|---|---|---|---|---|---|---|---|")
        for m, rev, rail, pkm, rc, cr, er in rows:
            print(f"| {m:g} | {1.5*m:.2f} | {rev/rail:.2f} | {rail/1e6:.2f} | {cr/1e6:.2f} / {er/1e6:.2f} | {pkm/1e6:.1f} | {100*rc:.1f} | {rev/1e6:.1f} | {rev/best[1]:.2f} |")
        print()


ceil_tables()
real_tables()
