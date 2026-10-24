"""T-1140: the US's and Puerto Rico's counties as the planner sees them, the plan's rows, the fixed areas
(metro rows and the 13 old Northeast packs) and the ACS county flows, into us_data.pkl beside this
file. Read-only on the plan and data.

    python data_us.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
import json
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, TW + r"\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import shapely  # noqa: E402
from shapely import STRtree, contains_xy, union_all  # noqa: E402

import world_plan as wp  # noqa: E402

t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def cells(x, y, p, km=10):
    """People summed on a km grid: (x, y, p) of the occupied cells."""
    if not len(p):
        return (np.zeros(0), np.zeros(0), np.zeros(0))
    k = km * 1000.0
    gx, gy = np.floor(x / k).astype(np.int64), np.floor(y / k).astype(np.int64)
    key = gx * 100000 + gy
    u, inv = np.unique(key, return_inverse=True)
    s = np.bincount(inv, weights=p)
    sx = np.bincount(inv, weights=x * p)
    sy = np.bincount(inv, weights=y * p)
    m = s > 0
    return (sx[m] / s[m], sy[m] / s[m], s[m])


def main():
    doc = json.load(open(wp.OUT, encoding="utf-8"))
    reg = json.load(open(wp.REGISTRY, encoding="utf-8"))["cities"]
    pop = wp.Pop()
    allpk = wp.packs()
    out = {}
    for iso in ("USA", "PRI"):
        rows = [r for r in doc["metros"] if iso in (r.get("units") or {})]
        plan_ids = {r["id"] for r in rows}
        cc = "US" if iso == "USA" else "PR"
        old = [c["id"] for c in reg if c.get("country") == cc and c["id"] not in plan_ids and c["id"] in allpk]
        log(f"{iso}: {len(rows)} plan rows; old packs {old}")
        pk = {k: allpk[k] for k in old}
        pl = [v["geom"] for v in pk.values()]
        ptree = STRtree(pl) if pl else None
        units, _ = wp.census_units(iso, pop)
        U = {}
        for u in units:
            x, y, p = u.px
            inpk = np.zeros(len(x), bool)
            if ptree is not None:
                for i in ptree.query(u.geom, predicate="intersects"):
                    inpk |= contains_xy(pl[i], x, y)
            m = (~inpk) & (p > 0)
            u.px = (x[m], y[m], p[m])
            u.free = float(p[m].sum())
            U[u.id] = u
        zrows = [r for r in rows if r.get("kind") == "zone"]
        mrows = [r for r in rows if r.get("kind") != "zone"]
        free = sorted({i for r in zrows for i in r["units"][iso]})
        fixed = []
        for r in mrows:
            g = allpk[r["id"]]["geom"] if r["id"] in allpk else union_all([U[i].geom for i in r["units"][iso]])
            px = [U[i].px for i in r["units"][iso] if i in U]
            x, y, p = (np.concatenate([q[k] for q in px]) for k in range(3))
            s = float(p.sum())
            xy = (float((x * p).sum()) / s, float((y * p).sum()) / s)
            fixed.append({"id": r["id"], "name": r["name"], "geom": wp.polygonal(g.simplify(500)), "xy": xy,
                          "units": list(r["units"][iso]), "people": r["people"]})
        for k, v in pk.items():
            x, y, p = pop.inside(v["geom"])
            s = float(p.sum())
            fixed.append({"id": k, "name": v["name"], "geom": wp.polygonal(v["geom"].simplify(500)),
                          "xy": (float((x * p).sum()) / s, float((y * p).sum()) / s), "units": [], "people": s})
        # an old pack's counties: those whose people mostly lie in it
        for f in fixed:
            if not f["units"]:
                f["units"] = [i for i, u in U.items() if u.geom.intersects(f["geom"]) and u.free < 0.5 * u.pop]
        flows = wp.commute_flows(iso) if iso == "USA" else None
        cel = {i: cells(*u.px) for i, u in U.items()}
        out[iso] = {"units": U, "rows": rows, "free": free, "fixed": fixed, "flows": flows, "cells": cel,
                    "record": doc["zones"].get(iso)}
        log(f"{iso}: {len(U)} counties, {len(free)} in zone rows, {len(fixed)} fixed areas")
    with open(os.path.join(HERE, "us_data.pkl"), "wb") as f:
        pickle.dump(out, f)
    log("written")


if __name__ == "__main__":
    main()
