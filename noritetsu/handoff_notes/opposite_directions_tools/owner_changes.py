"""Ways whose owner differs between dist and a trial, grouped by (old owner, new owner).

    python owner_changes.py <trial dir> cc [filter-substring]
"""
import json, pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"C:\Users\anita\projects\maps\noritetsu")
trial, cc = Path(sys.argv[1]), sys.argv[2]
flt = sys.argv[3] if len(sys.argv) > 3 else None


def own(d):
    w = json.loads((d / cc / "ways.json").read_text(encoding="utf-8"))
    un = set(w["unowned"])
    return {k: (None if k in un else w["lines"][v[0]]) for k, v in w["ways"].items()}


lines = {l["id"]: l for l in json.loads((trial / cc / "lines.json").read_text(encoding="utf-8"))["lines"]}
nm = lambda i: "-" if i is None else ((lines.get(i, {}).get("ref") or "") + " " + lines.get(i, {}).get("name", i))[:40]
a, b = own(ROOT / "dist" / "data"), own(trial)
with open(ROOT / "data" / "proc" / cc / "ways.pkl", "rb") as f:
    W = pickle.load(f)
c = np.load(ROOT / "data" / "proc" / cc / "coords.npz")
cid, cx, cy = c["id"], c["x"], c["y"]
grp = defaultdict(list)
for k in set(a) | set(b):
    if a.get(k) != b.get(k):
        grp[(a.get(k), b.get(k))].append(int(k))
rows = []
for (o, n), ws in grp.items():
    km, pts = 0.0, []
    for w in ws:
        nodes = np.asarray(W[w][1], dtype=np.int64)
        p = np.clip(np.searchsorted(cid, nodes), 0, cid.size - 1)
        p = p[cid[p] == nodes]
        lon, lat = cx[p] / 1e7, cy[p] / 1e7
        if len(lon) < 2:
            continue
        km += float(np.sum(np.hypot(np.diff(lon) * np.cos(np.radians(lat[:-1])), np.diff(lat)))) * 111.32
        pts.append((lat.mean(), lon.mean()))
    rows.append((km, o, n, len(ws), pts[0] if pts else None))
rows.sort(key=lambda r: -r[0])
for km, o, n, k, p in rows:
    s = f"{km:6.2f} km {k:4} ways  {nm(o):40} -> {nm(n):40} at {p[0]:.5f},{p[1]:.5f}" if p else ""
    if flt is None or flt in s:
        print(s)
