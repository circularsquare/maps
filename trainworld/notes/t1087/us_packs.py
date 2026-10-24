"""T-1087: add the 13 built Northeast packs to us_data.pkl and redo the counties they cut (read-only)."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
import json
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
from shapely import STRtree, contains_xy  # noqa: E402

import world_plan as wp  # noqa: E402

P = os.path.join(HERE, "us_data.pkl")
D = pickle.load(open(P, "rb"))
doc = json.load(open(wp.OUT, encoding="utf-8"))
with_units = {r["id"] for r in doc["metros"] if "USA" in (r.get("units") or {})}
reg = json.load(open(wp.REGISTRY, encoding="utf-8"))["cities"]
old = [c["id"] for c in reg if c.get("country") == "US" and c["id"] not in with_units]
print("old US packs:", old)
pk = {k: v for k, v in wp.packs().items() if k in old}
D["packs"] = {k: {"id": k, "name": v["name"], "geom": v["geom"]} for k, v in pk.items()}
pl = [v["geom"] for v in pk.values()]
tree = STRtree(pl)
pop = wp.Pop()
n = 0
for uid, u in D["units"].items():
    hit = tree.query(u["geom"], predicate="intersects")
    if not len(hit):
        continue
    import shapely
    x, y, p = [np.concatenate(a) for a in zip(*[pop.inside(part) for part in shapely.get_parts(u["geom"])])]
    inpk = np.zeros(len(x), bool)
    for i in hit:
        inpk |= contains_xy(pl[i], x, y)
    fp = p[~inpk]
    free = float(fp.sum())
    u["free"] = free
    u["inpack"] = (u["pop"] > 0 and free < 0.5 * u["pop"]) or free <= 0
    if free > 0:
        u["c"] = (float((x[~inpk] * fp).sum() / free), float((y[~inpk] * fp).sum() / free))
    n += 1
pickle.dump(D, open(P, "wb"))
print(n, "counties redone;", sum(u["inpack"] for u in D["units"].values()), "inside packs")
