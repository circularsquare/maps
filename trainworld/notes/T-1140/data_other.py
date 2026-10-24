"""T-1140: China's (geoBoundaries ADM3) or European countries' (NUTS-3 from the plan's state) units with
their people, the plan's rows and the fixed areas, into <name>_data.pkl beside this file (the format
of data_us.py). Read-only.

    python data_other.py CHN
    python data_other.py europe
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
from shapely import STRtree, contains_xy, union_all  # noqa: E402

import world_plan as wp  # noqa: E402
from data_us import cells  # noqa: E402

t0 = time.time()
GB = TW + r"\data\raw\worldplan\geoboundaries"


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def main(which):
    doc = json.load(open(wp.OUT, encoding="utf-8"))
    pop = wp.Pop()
    allpk = wp.packs()
    zone_ids = {r["id"] for r in doc["metros"] if r.get("kind") == "zone"}
    nonzone = [v["geom"] for k, v in allpk.items() if k not in zone_ids]
    ptree = STRtree(nonzone)
    if which in ("CHN", "TWN"):
        isos = [which]
    else:
        isos = [i for i in doc["zones"] if i not in ("CHN", "TWN", "USA", "PRI")]
    st = None
    out = {}
    for iso in isos:
        rows = [r for r in doc["metros"] if iso in (r.get("units") or {})]
        want = {u for r in rows for u in r["units"][iso]}
        zone_u = {u for r in rows if r.get("kind") == "zone" for u in r["units"][iso]}
        raw = []
        if iso in ("CHN", "TWN"):
            a3 = pickle.load(open(GB + (r"\CHN_ADM3.t897e.pkl" if iso == "CHN" else r"\TWN_ADM2.t897e.pkl"), "rb"))
            a1 = pickle.load(open(GB + rf"\{iso}_ADM1.t897e.pkl", "rb"))
            t1 = STRtree([g for _, _, g in a1])
            for uid, name, g in a3:
                if uid in want:
                    hit = [a1[i][1] for i in t1.query(g.representative_point(), predicate="within")]
                    raw.append(wp.Unit(uid, name, iso, g, [(hit[0] if hit else iso, hit[0] if hit else iso)]))
            nuts = False
        else:
            if st is None:
                class Here(pickle.Unpickler):
                    def find_class(self, mod, name):
                        return wp.Unit if name == "Unit" else super().find_class(mod, name)
                with open(wp.WORK / "state.pkl", "rb") as f:
                    st = Here(f).load()
            for u in st["units"]:
                if u.iso == iso and u.id in want:
                    g = wp.de_facto(iso, u.geom) or u.geom
                    v = wp.Unit(u.id, u.name, iso, g, u.parents)
                    raw.append(v)
            src = doc["unit_sources"].get(iso, {})
            nuts = src.get("kind") in ("NUTS3", "ITL3")
        U = {}
        for u in raw:
            x, y, p = pop.inside(u.geom)
            m = p > 0
            x, y, p = x[m], y[m], p[m]
            u.pop = float(p.sum())
            if u.id in zone_u:
                inpk = np.zeros(len(x), bool)
                for i in ptree.query(u.geom, predicate="intersects"):
                    inpk |= contains_xy(nonzone[i], x, y)
                x, y, p = x[~inpk], y[~inpk], p[~inpk]
            u.px = (x, y, p)
            u.free = float(p.sum())
            if not nuts and iso != "CHN":
                u.parents = list(u.parents) if u.parents else []
            U[u.id] = u
        region = {}
        for i, u in U.items():
            if iso in ("CHN", "TWN"):
                region[i] = u.parents[0][0]
            else:
                region[i] = (u.parents[-1] if nuts else u.parents[0]) if u.parents else iso
                if isinstance(region[i], list):
                    region[i] = tuple(region[i])
        free = sorted(i for i in zone_u if i in U)
        fixed = []
        for r in rows:
            if r.get("kind") == "zone":
                continue
            us = [i for i in r["units"][iso] if i in U]
            g = allpk[r["id"]]["geom"] if r["id"] in allpk else union_all([U[i].geom for i in us])
            x, y, p = (np.concatenate([U[i].px[k] for i in us]) for k in range(3)) if us else (np.zeros(1),) * 3
            s = float(p.sum())
            xy = (float((x * p).sum()) / s, float((y * p).sum()) / s) if s else g.representative_point().coords[0]
            fixed.append({"id": r["id"], "name": r["name"], "geom": wp.polygonal(g.simplify(500)), "xy": xy,
                          "units": us, "people": r.get("people", 0)})
        out[iso] = {"units": U, "rows": rows, "free": free, "fixed": fixed, "flows": None, "region": region,
                    "cells": {i: cells(*u.px) for i, u in U.items()}, "record": doc["zones"].get(iso),
                    "nuts": nuts}
        log(f"{iso}: {len(U)} of {len(want)} units, {len(free)} in zone rows, {len(fixed)} metro rows")
    with open(os.path.join(HERE, f"{which.lower()}_data.pkl"), "wb") as f:
        pickle.dump(out, f)
    log("written")


if __name__ == "__main__":
    main(sys.argv[1])
