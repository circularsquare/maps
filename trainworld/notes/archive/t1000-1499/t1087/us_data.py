"""T-1087: US counties, metro rows, built packs, seeds and ACS county flows into one cache (read-only on
the plan and data). Writes us_data.pkl beside itself."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
import collections
import json
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, TW + r"\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from shapely import STRtree, contains_xy, union_all  # noqa: E402

import world_plan as wp  # noqa: E402

t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def main():
    doc = json.load(open(wp.OUT, encoding="utf-8"))
    rows = [r for r in doc["metros"] if "USA" in (r.get("units") or {})]
    reg = json.load(open(wp.REGISTRY, encoding="utf-8"))["cities"]
    plan_ids = {r["id"] for r in rows}
    old_us = [c["id"] for c in reg if c.get("country") == "US" and c["id"] not in plan_ids]
    log(f"{len(rows)} US plan rows; built packs outside the plan: {old_us}")
    pop = wp.Pop()
    units, _ = wp.census_units("USA", pop)
    pk = wp.packs()
    pk = {k: v for k, v in pk.items() if k in old_us}
    ptree = STRtree([v["geom"] for v in pk.values()])
    pl = list(pk.values())
    out = {}
    for u in units:
        x, y, p = u.px
        inpk = np.zeros(len(x), bool)
        for i in ptree.query(u.geom, predicate="intersects"):
            inpk |= contains_xy(pl[i]["geom"], x, y)
        fp = p[~inpk]
        fx, fy = x[~inpk], y[~inpk]
        tot = float(p.sum())
        free = float(fp.sum())
        if free > 0:
            cx, cy = float((fx * fp).sum() / free), float((fy * fp).sum() / free)
        else:
            c = u.geom.representative_point()
            cx, cy = c.x, c.y
        out[u.id] = {"id": u.id, "name": u.name, "state": u.parents[0][0], "geom": u.geom,
                     "simple": wp.polygonal(u.geom.simplify(500)), "pop": tot, "free": free,
                     "inpack": (tot > 0 and free < 0.5 * tot) or free <= 0, "c": (cx, cy),
                     "area": u.geom.area / 1e6}
    log(f"{len(out)} counties")
    ne = wp.countries()
    uc, cen, iso = wp.load_ucdb(ne)
    seeds = []
    for k in range(len(uc)):
        if iso[k] != "USA":
            continue
        pt = cen[uc.ID_UC_G0[k]]
        seeds.append({"uc": int(uc.ID_UC_G0[k]), "name": uc.GC_UCN_MAI_2025[k], "pop": float(uc.GC_POP_TOT_2025[k]),
                      "xy": (pt.x, pt.y)})
    log(f"{len(seeds)} US urban centres")
    acs = pd.read_excel(TW + r"\data\raw\acsflows\table1.xlsx", header=None, skiprows=8, dtype=str).iloc[:, :10]
    acs.columns = ["hs", "hc", "hsn", "hcn", "ws", "wc", "wsn", "wcn", "n", "moe"]
    acs = acs.dropna(subset=["hs", "ws", "wc"])
    acs["h"] = acs["hs"].str.zfill(2) + acs["hc"].str.zfill(3)
    acs["w"] = acs["ws"].str[-2:] + acs["wc"].str.zfill(3)
    acs["n"] = pd.to_numeric(acs["n"], errors="coerce").fillna(0)
    flows = collections.defaultdict(dict)
    for h, w, n in zip(acs.h, acs.w, acs.n):
        flows[h][w] = flows[h].get(w, 0) + float(n)
    log(f"ACS flows: {len(flows)} home counties, {acs.n.sum() / 1e6:.1f}M workers")
    metros = [{"id": r["id"], "name": r["name"], "kind": r.get("kind", "city"), "status": r.get("status", "planned"),
               "units": r["units"]["USA"], "people": r["people"]} for r in rows]
    packs_us = {k: {"id": k, "name": v["name"], "geom": v["geom"]} for k, v in pk.items()}
    pickle.dump({"units": out, "rows": metros, "packs": packs_us, "seeds": seeds, "flows": dict(flows)},
                open(os.path.join(HERE, "us_data.pkl"), "wb"))
    log("written")


if __name__ == "__main__":
    main()
