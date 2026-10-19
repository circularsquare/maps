"""T-700: how many people live in zones whose walk factor to themselves is inflated, per pack."""
import json
import math
import os
import sys

import numpy as np

PACKS = r"C:\Users\anita\projects\maps\trainworld\data\packs"
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
from read_pack import load_pack

cities = sys.argv[1:] or sorted(f[:-5] for f in os.listdir(PACKS) if f.endswith(".json") and len(f) == 8)
rows = []
for city in cities:
    try:
        h, a = load_pack(city)
    except Exception:
        continue
    if "walk_zone" not in a or "routes" not in h:
        continue
    gx, gy = a["zone_gx"].astype(int), a["zone_gy"].astype(int)
    n = len(gx)
    reg = np.zeros(n, int)
    zone = a["zone"]
    if "region" in a:
        # a zone's region is its cells' region
        reg[zone] = a["region"]
    W = h["routes"]["walk_zone"]["window"]
    gw, gh = h["gravity"]["grid"]
    gz = {}
    for i in range(n):
        gz.setdefault((gx[i], gy[i]), []).append(i)
    for v in gz.values():
        v.sort(key=lambda i: reg[i])
    wz = a["walk_zone"]
    selff = np.ones(n)
    worst = np.ones(n)
    k = 0
    for i in range(n):
        for dx in range(-W, W + 1):
            X = gx[i] + dx
            if X < 0 or X >= gw:
                continue
            for dy in range(-W, W + 1):
                Y = gy[i] + dy
                if Y < 0 or Y >= gh:
                    continue
                for j in gz.get((X, Y), []):
                    f = math.exp(wz[k] / 64.0)
                    if j == i:
                        selff[i] = f
                    if abs(dx) <= 1 and abs(dy) <= 1:
                        worst[i] = max(worst[i], f)
                    k += 1
    if k != len(wz):
        print(city, "layout mismatch", k, len(wz))
        continue
    w = np.bincount(zone, weights=a["pop"] + a["jobs"], minlength=n)
    tot = w.sum()
    s2 = w[selff > 2].sum() / tot
    s5 = w[selff > 5].sum() / tot
    rows.append((city, n, int((selff > 2).sum()), s2, s5))
    print(f"{city}: {n} zones; self factor over 2: {(selff > 2).sum()} zones, {s2:.1%} of people+jobs; over 5: {(selff > 5).sum()} zones, {s5:.1%}", flush=True)
rows.sort(key=lambda r: -r[3])
print("\nworst by share over 2:")
for r in rows[:15]:
    print(f"  {r[0]}: {r[2]} zones, {r[3]:.1%} (over 5: {r[4]:.1%})")
print(f"packs with routes: {len(rows)}; median share over 2: {np.median([r[3] for r in rows]):.1%}")
