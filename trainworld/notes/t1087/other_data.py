"""T-1087: China's (geoBoundaries ADM3) or European countries' (NUTS-3 from the plan's state) units,
plan rows and GHSL urban centres into <ISO>_data.pkl beside this file. Read-only.

    python other_data.py CHN
    python other_data.py DEU POL CZE IRL GBR
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
from shapely import STRtree, contains_xy  # noqa: E402

import world_plan as wp  # noqa: E402

t0 = time.time()
GB = TW + r"\data\raw\worldplan\geoboundaries"


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def unit_rec(uid, name, region, g, pop, pk):
    x, y, p = pop.inside(g)
    if pk is not None:
        tree, pl = pk
        inpk = np.zeros(len(x), bool)
        for i in tree.query(g, predicate="intersects"):
            inpk |= contains_xy(pl[i], x, y)
        x, y, p = x[~inpk], y[~inpk], p[~inpk]
    fr = float(p.sum())
    if fr > 0:
        c = (float((x * p).sum() / fr), float((y * p).sum() / fr))
    else:
        rp = g.representative_point()
        c = (rp.x, rp.y)
    return {"id": uid, "name": name, "state": region, "geom": g, "simple": wp.polygonal(g.simplify(500)),
            "free": fr, "c": c, "area": g.area / 1e6}


def main(isos):
    doc = json.load(open(wp.OUT, encoding="utf-8"))
    pop = wp.Pop()
    ne = wp.countries()
    uc, cen, uiso = wp.load_ucdb(ne)
    allpk = wp.packs()
    plan_ids = {r["id"] for r in doc["metros"] if r.get("units")}
    st = None
    for iso in isos:
        rows = [r for r in doc["metros"] if iso in (r.get("units") or {})]
        want = {u for r in rows for u in r["units"][iso]}
        units = {}
        if iso == "CHN":
            a3 = pickle.load(open(GB + r"\CHN_ADM3.t897e.pkl", "rb"))
            a1 = pickle.load(open(GB + r"\CHN_ADM1.t897e.pkl", "rb"))
            t1 = STRtree([g for _, _, g in a1])
            zone_u = {u for r in rows if r.get("kind") == "zone" for u in r["units"][iso]}
            for uid, name, g in a3:
                if uid not in want:
                    continue
                hit = [a1[i][1] for i in t1.query(g.representative_point(), predicate="within")]
                reg = hit[0] if hit else iso
                if uid in zone_u:
                    units[uid] = unit_rec(uid, name, reg, g, pop, None)
                else:
                    units[uid] = {"id": uid, "name": name, "state": reg, "geom": g,
                                  "simple": wp.polygonal(g.simplify(500)), "free": 0.0, "c": None, "area": g.area / 1e6}
        else:
            if st is None:
                import pickle as pk_

                class Here(pk_.Unpickler):
                    def find_class(self, mod, name):
                        return wp.Unit if name == "Unit" else super().find_class(mod, name)
                with open(wp.WORK / "state.pkl", "rb") as f:
                    st = Here(f).load()
            # packs that are not plan rows of this country (older hand-drawn ones) cut people out
            mine = {r["id"] for r in rows}
            other = [v["geom"] for k, v in allpk.items() if k not in mine]
            ptree = (STRtree(other), other)
            for u in st["units"]:
                if u.iso != iso or u.id not in want:
                    continue
                reg = u.parents[-1] if u.parents else iso
                if isinstance(reg, (list, tuple)):
                    reg = reg[0]
                g = wp.de_facto(iso, u.geom) or u.geom
                units[u.id] = unit_rec(u.id, u.name, reg, g, pop, ptree)
        log(f"{iso}: {len(units)} of {len(want)} plan units found")
        seeds = []
        for k in range(len(uc)):
            if uiso[k] == iso:
                pt = cen[uc.ID_UC_G0[k]]
                seeds.append({"uc": int(uc.ID_UC_G0[k]), "name": uc.GC_UCN_MAI_2025[k],
                              "pop": float(uc.GC_POP_TOT_2025[k]), "xy": (pt.x, pt.y)})
        out_rows = [{"id": r["id"], "name": r["name"], "kind": r.get("kind", "city"), "status": r.get("status", "planned"),
                     "units": [u for u in r["units"][iso] if u in units], "people": r["people"],
                     "solve_region": r.get("solve_region")} for r in rows]
        # packs of this country's rows: their real outlines (a metro row's pack may be cut)
        packs = {r["id"]: allpk[r["id"]]["geom"] for r in rows if r["id"] in allpk}
        pickle.dump({"units": units, "rows": out_rows, "seeds": seeds, "packs": packs},
                    open(os.path.join(HERE, f"{iso}_data.pkl"), "wb"))
        log(f"{iso}: {len(seeds)} urban centres, {len(out_rows)} rows written")


if __name__ == "__main__":
    main(sys.argv[1:])
