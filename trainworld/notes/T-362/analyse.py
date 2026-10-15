import json, math
from pathlib import Path
import numpy as np
P = json.load(open(Path(__file__).resolve().parents[3] / "artifacts/trainworld/t362/pairs.json"))
SHOW = [("New York", "Philadelphia"), ("New York", "Boston"), ("New York", "Washington"), ("Boston", "Washington"),
        ("New York", "Toronto"), ("London", "New York"), ("London", "Paris"), ("London", "Manchester"),
        ("London", "Edinburgh"), ("London", "Dublin"), ("Toronto", "Montreal"), ("Tokyo", "Osaka"), ("Tokyo", "Fukuoka"),
        ("Tokyo", "Sapporo"), ("Beijing", "Shanghai"), ("Seoul", "Busan"), ("Sydney", "Melbourne"), ("Madrid", "Barcelona"),
        ("Delhi", "Mumbai"), ("Los Angeles", "San Francisco"), ("Los Angeles", "New York"), ("Chicago", "New York"),
        ("London", "Dubai"), ("London", "Singapore"), ("Paris", "Amsterdam"), ("Berlin", "Munich")]
idx = {(a, b): r for r in P for a, b in [(r[0], r[1]), (r[1], r[0])]}
print("pair | km | model/day | air seg/day | air/model")
for a, b in SHOW:
    r = idx[(a, b)]
    print(f"{a} - {b} | {r[2]} | {r[3]:,} | {r[4]:,} | {r[4]/max(r[3],1):.2f}")
EU = set("GB FR NL BE DE ES IT IE".split())
def kind(r):
    if r[5] == r[6]: return "domestic"
    cont = {"US": "NA", "CA": "NA", "MX": "NA", "BR": "SA", "TR": "EU", "AE": "AS", "IN": "AS", "SG": "AS", "HK": "AS", "CN": "AS",
            "KR": "AS", "JP": "AS", "AU": "OC"}
    ca, cb = cont.get(r[5], "EU"), cont.get(r[6], "EU")
    if r[5] in EU and r[6] in EU: return "intl EU"
    return "intl same continent" if ca == cb else "intercontinental"
print("\nlong pairs (>=1000 km, air>0): median and quartiles of air/model by kind and band")
for k in ["domestic", "intl EU", "intl same continent", "intercontinental"]:
    for lo, hi in [(1000, 2500), (2500, 6000), (6000, 20000)]:
        v = [r[4] / r[3] for r in P if kind(r) == k and lo <= r[2] < hi and r[3] > 0 and r[4] > 0]
        z = sum(1 for r in P if kind(r) == k and lo <= r[2] < hi and r[4] == 0)
        if v:
            q = np.percentile(v, [25, 50, 75])
            print(f"{k:22s} {lo:>5}-{hi:<5} n={len(v):3d} zero-air={z:3d}  {q[0]:.2f} {q[1]:.2f} {q[2]:.2f}")
# fit log(air/model) = a + b log d over d >= 1000 among domestic
for k in ["domestic", None]:
    rows = [r for r in P if r[2] >= 1000 and r[4] > 0 and r[3] > 0 and (k is None or kind(r) == k)]
    x = np.log([r[2] for r in rows]); y = np.log([r[4] / r[3] for r in rows])
    b, a = np.polyfit(x, y, 1)
    print(f"fit {k or 'all'}: log ratio = {a:.2f} + {b:.2f} log d, n={len(rows)}")
