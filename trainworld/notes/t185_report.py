"""T-185: print the class tables from CACHE/classes.json (t185_classes.py) as markdown.

    ..\\venv\\Scripts\\python.exe trainworld/notes/t185_report.py
"""
import json
import os

import numpy as np

import t185_common as tc

NAMES = {"nyc": "New York", "phl": "Philadelphia", "bos": "Boston", "wdc": "Washington", "bal": "Baltimore",
         "pvd": "Providence", "nhv": "New Haven", "hfd": "Hartford", "spf": "Springfield"}
AB = "CMS"


def shares(M):
    M = np.array(M)
    return M / M.sum(1, keepdims=True)


def main():
    res = json.load(open(os.path.join(tc.CACHE, "classes.json")))
    print("## Classes\n")
    print("| pack | centre zones | centre jobs/km2 from | middle from | commuters living in centre / middle / suburb | jobs |")
    print("|---|---|---|---|---|---|")
    for n in tc.CITIES:
        r = res.get(n)
        if not r:
            continue
        h = np.array(r["home"]); j = np.array(r["jobs"])
        print(f"| {NAMES[n]} | {r['zones'][0]} | {r['density_cut'][0]:,.0f} | {r['density_cut'][1]:,.0f} | "
              + " / ".join(f"{v:.0%}" for v in h / h.sum()) + " | " + " / ".join(f"{v:.0%}" for v in j / j.sum()) + " |")
    print("\n## Share of each home class's commuters by work class (LODES / now / no factor 40 km / pre-T162 29 km)\n")
    cells = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)]
    print("| pack | " + " | ".join(f"{AB[a]}->{AB[b]}" for a, b in cells) + " |")
    print("|---|" + "---|" * len(cells))
    for n in tc.CITIES:
        r = res.get(n)
        if not r:
            continue
        S = {v: shares(r[v]["cls"]) for v in ("lodes", "now", "k1", "pre")}
        row = []
        for a, b in cells:
            row.append(" / ".join(f"{S[v][a, b] * 100:.0f}" for v in ("lodes", "now", "k1", "pre")))
        print(f"| {NAMES[n]} | " + " | ".join(row) + " |")
    print("\n## Model / LODES, share of home class (now | k1 | pre)\n")
    print("| pack | " + " | ".join(f"{AB[a]}->{AB[b]}" for a, b in cells) + " |")
    print("|---|" + "---|" * len(cells))
    for n in tc.CITIES:
        r = res.get(n)
        if not r:
            continue
        L = shares(r["lodes"]["cls"])
        row = []
        for a, b in cells:
            row.append(" / ".join(f"{shares(r[v]['cls'])[a, b] / L[a, b]:.2f}" for v in ("now", "k1", "pre")))
        print(f"| {NAMES[n]} | " + " | ".join(row) + " |")
    print("\n## Mean crow-fly trip by home class, km (LODES / now / k1 / pre)\n")
    print("| pack | centre | middle | suburb | all |")
    print("|---|---|---|---|---|")
    for n in tc.CITIES:
        r = res.get(n)
        if not r:
            continue
        row = [" / ".join(f"{r[v]['cls_len'][i]:.1f}" for v in ("lodes", "now", "k1", "pre")) for i in range(3)]
        row.append(" / ".join(f"{r[v]['mean']:.1f}" for v in ("lodes", "now", "k1", "pre")))
        print(f"| {NAMES[n]} | " + " | ".join(row) + " |")
    print("\n## Cross-state share of each class pair's trips (LODES / now / k1)\n")
    print("| pack | " + " | ".join(f"{AB[a]}->{AB[b]}" for a, b in cells) + " |")
    print("|---|" + "---|" * len(cells))
    for n in ("nyc", "phl", "wdc", "bos", "pvd"):
        r = res.get(n)
        if not r:
            continue
        row = []
        for a, b in cells:
            row.append(" / ".join(f"{np.array(r[v]['cls_cross'])[a, b] / np.array(r[v]['cls'])[a, b] * 100:.0f}" for v in ("lodes", "now", "k1")))
        print(f"| {NAMES[n]} | " + " | ".join(row) + " |")
    print("\n## Scores (class / county / length misplaced, %; now | k1 | pre)\n")
    print("| pack | class | county | length | between states (LODES) |")
    print("|---|---|---|---|---|")
    for n in tc.CITIES:
        r = res.get(n)
        if not r:
            continue
        f = lambda x: " / ".join(f"{r[v][x] * 100:.1f}" for v in ("now", "k1", "pre"))
        print(f"| {NAMES[n]} | {f('class_off')} | {f('county_off')} | {f('tld_off')} | {f('cross')} ({r['lodes']['cross'] * 100:.1f}) |")


if __name__ == "__main__":
    main()
