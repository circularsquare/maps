"""T-1140: the European countries whose NUTS-3 units the plan read afresh (not in its state), from the
NUTS 2024 file, into nuts_data.pkl (data_us.py's format). Read-only.

    python data_nuts.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
import json
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
from shapely import STRtree, contains_xy, union_all  # noqa: E402
from shapely.geometry import shape  # noqa: E402

import world_plan as wp  # noqa: E402
from data_us import cells  # noqa: E402

ISOS = {"SVN": "SI", "HRV": "HR", "MKD": "MK", "XKX": "XK", "LVA": "LV", "LTU": "LT", "MLT": "MT"}
doc = json.load(open(wp.OUT, encoding="utf-8"))
gj = json.loads((wp.RAW / wp.NUTS[0]).read_text(encoding="utf-8"))
pop = wp.Pop()
allpk = wp.packs()
zone_ids = {r["id"] for r in doc["metros"] if r.get("kind") == "zone"}
nonzone = [v["geom"] for k, v in allpk.items() if k not in zone_ids]
ptree = STRtree(nonzone)
out = {}
for iso, cc in ISOS.items():
    rows = [r for r in doc["metros"] if iso in (r.get("units") or {})]
    want = {u for r in rows for u in r["units"][iso]}
    zone_u = {u for r in rows if r.get("kind") == "zone" for u in r["units"][iso]}
    U = {}
    for f in gj["features"]:
        pid = f["properties"]["NUTS_ID"]
        if pid not in want:
            continue
        g = wp.polygonal(wp.to_moll(shape(f["geometry"])))
        u = wp.Unit(pid, f["properties"].get("NAME_LATN") or f["properties"].get("NUTS_NAME"), iso, g,
                    [(pid[:3], ""), (pid[:4], "")])
        x, y, p = pop.inside(g)
        m = p > 0
        x, y, p = x[m], y[m], p[m]
        u.pop = float(p.sum())
        if pid in zone_u:
            inpk = np.zeros(len(x), bool)
            for i in ptree.query(g, predicate="intersects"):
                inpk |= contains_xy(nonzone[i], x, y)
            x, y, p = x[~inpk], y[~inpk], p[~inpk]
        u.px = (x, y, p)
        u.free = float(p.sum())
        U[pid] = u
    fixed = []
    for r in rows:
        if r.get("kind") == "zone":
            continue
        us = [i for i in r["units"][iso] if i in U]
        g = allpk[r["id"]]["geom"] if r["id"] in allpk else union_all([U[i].geom for i in us])
        x, y, p = (np.concatenate([U[i].px[k] for i in us]) for k in range(3)) if us else (np.zeros(1),) * 3
        s = float(p.sum())
        xy = (float((x * p).sum()) / s, float((y * p).sum()) / s) if s else g.representative_point().coords[0]
        fixed.append({"id": r["id"], "name": r["name"], "geom": wp.polygonal(g.simplify(500)), "xy": xy, "units": us})
    out[iso] = {"units": U, "rows": rows, "free": sorted(i for i in zone_u if i in U), "fixed": fixed, "flows": None,
                "region": {i: U[i].parents[-1] for i in U}, "cells": {i: cells(*u.px) for i, u in U.items()},
                "record": doc["zones"].get(iso), "nuts": True}
    print(iso, len(U), "of", len(want), "units", flush=True)
pickle.dump(out, open(os.path.join(HERE, "nuts_data.pkl"), "wb"))
print("written")
