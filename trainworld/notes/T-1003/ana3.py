"""T-1003: every Chinese zone's baseline walk share: distribution and the zones over 50%."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ana import load, reg, tot

D = os.path.dirname(os.path.abspath(__file__))
ids = open(f"{D}/cn_all.txt").read().strip().split(",")
rows = []
fails = []
for c in ids:
    r = load(c, 'base')
    if r is None:
        fails.append(c)
        continue
    t = tot(r[1])
    if t[0] <= 0:
        fails.append(c)
        continue
    rows.append((t[1] / t[0], c, t[0], t[8] / t[0], t[7] / t[0], reg[c].get('kind')))
w = np.array([x[0] for x in rows])
trips = np.array([x[2] for x in rows])
print("zones", len(rows), "missing", fails)
print("trip-weighted walk", round(100 * (w * trips).sum() / trips.sum(), 1), "median", round(100 * np.median(w), 1))
for lo, hi in [(0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, 1)]:
    m = (w >= lo) & (w < hi)
    print(f"{lo:.0%}-{hi:.0%}: {m.sum()} zones, {100*trips[m].sum()/trips.sum():.0f}% of trips")
kinds = {}
for x in rows:
    kinds.setdefault(x[5] or 'city', []).append(x)
for k, v in kinds.items():
    ww = np.array([x[0] for x in v]); tt = np.array([x[2] for x in v])
    print(k, len(v), "walk", round(100 * (ww * tt).sum() / tt.sum(), 1))
rows.sort(reverse=True)
print("over 50%:")
for x in rows:
    if x[0] >= 0.5:
        print(f"{reg[x[1]]['name']} ({x[1]}) {100*x[0]:.0f}% home park {x[3]:.1f} paid {x[4]:.1f} kind {x[5]}")
