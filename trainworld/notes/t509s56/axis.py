"""Each city's score along one parameter's axis through a point: python axis.py <param> [point.json|today]"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit

cache = fit.load_cache()
k = sys.argv[1]
base = fit.today() if len(sys.argv) < 3 or sys.argv[2] == "today" else json.load(open(sys.argv[2]))
if len(sys.argv) > 3:
    for a in sys.argv[3:]:
        kk, v = a.split("=")
        base[kk] = float(v)
rows = []
for r in cache.values():
    p = r["p"]
    if all(abs(p[x] - base[x]) < 1e-6 for x in p if x != k):
        rows.append((p[k], r["city"], r))
vals = sorted({v for v, _, _ in rows})
print(f"{k} through {base}")
print("city  " + " ".join(f"{v:>7g}" for v in vals))
for c in fit.ALL:
    d = {v: r for v, cc, r in rows if cc == c}
    print(f"{c:5s} " + " ".join(f"{d[v]['total']:7.3f}" if v in d else "      -" for v in vals))
    print(f"  lvl " + " ".join(f"{math.exp(d[v]['station_level']):7.2f}" if v in d else "      -" for v in vals))
