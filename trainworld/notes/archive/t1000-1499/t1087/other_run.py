"""T-1087 dry run for China or a European country: the plan's zones against the seeded rule, rings by
the hull and border test (no commuting data). Read-only; writes <ISO>_out.pkl here.

    python other_run.py CHN
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import pickle
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import shapely  # noqa: E402
from shapely import union_all  # noqa: E402

import rule  # noqa: E402

t0 = time.time()
RULES = {"CHN": (1_000_000, 2_500_000)}


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def run(iso):
    floor, cap = RULES.get(iso, (500_000, 1_500_000))
    D = pickle.load(open(os.path.join(HERE, f"{iso}_data.pkl"), "rb"))
    U = D["units"]
    zrows = [r for r in D["rows"] if r["kind"] == "zone"]
    mrows = [r for r in D["rows"] if r["kind"] != "zone" and r["units"]]
    free = sorted({i for r in zrows for i in r["units"]})
    region = {i: U[i]["state"] for i in U}
    fixed = {}
    for r in mrows:
        g = D["packs"].get(r["id"])
        g = shapely.simplify(g, 500) if g is not None else union_all([U[i]["simple"] for i in r["units"]])
        fixed[r["id"]] = {"id": r["id"], "name": r["name"], "geom": g, "units": r["units"], "pop": r["people"]}

    def fixed_borders(ids):
        out = []
        gs = {i: U[i]["simple"] for i in ids}
        keys = list(gs)
        tree = shapely.STRtree([gs[k] for k in keys])
        allfix = list(fixed.values())
        for m in allfix:
            near = m["geom"].buffer(rule.KM)
            ub = {}
            for b in tree.query(near, predicate="intersects"):
                ub[keys[b]] = gs[keys[b]].boundary.intersection(m["geom"].buffer(rule.KM)).length
            land = m["geom"].boundary.intersection(union_all([gs[i] for i in ub]).buffer(rule.KM)).length if ub else 0
            for o in allfix:
                if o is not m and o["geom"].intersects(near):
                    land += m["geom"].boundary.intersection(o["geom"].buffer(rule.KM)).length
            out.append((m["geom"], land, ub))
        return out

    def evaluate(name, groups):
        Z = rule.Zones(U, groups)
        ppl = sorted(Z.ppl(k) for k in Z.z)
        shapes = {k: Z.geo(k) for k in Z.z}
        kinds = {k: "zone" for k in Z.z}
        for k, m in fixed.items():
            shapes[k] = m["geom"]
            kinds[k] = "metro"
        pairs = rule.ring_pairs(shapes, kinds)
        rings = [p for p in pairs if p[2] >= 0.5 and p[3] >= 0.5]
        comp = sorted(shapes[k].area / shapes[k].convex_hull.area for k in Z.z)
        res = {"name": name, "zones": len(Z.z), "median": statistics.median(ppl),
               "under_floor_08": sum(p < 0.8 * floor for p in ppl), "over_cap": sum(p > cap for p in ppl), "max": max(ppl),
               "rings_zz": sum(kinds[p[1]] == "zone" for p in rings), "rings_zm": sum(kinds[p[1]] == "metro" for p in rings),
               "ring_zones": len({p[0] for p in rings}), "compact_median": statistics.median(comp),
               "compact_under_05": sum(c < 0.5 for c in comp)}
        log(iso, name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}))
        return res, rings

    out = {}
    before = {r["id"]: r["units"] for r in zrows}
    res, rings = evaluate("before (plan)", before)
    out["before"] = {"res": res, "groups": before, "rings": rings}
    adj = rule.adjacency(U, free)
    fb = fixed_borders(free)
    for m in fixed.values():
        c = m["geom"].representative_point()
        m.setdefault("xy", (c.x, c.y))
    for s in sorted(D["seeds"], key=lambda s: s["pop"]):  # a metro's centre: its biggest urban centre
        pt = shapely.Point(*s["xy"])
        for m in fixed.values():
            if m["geom"].contains(pt):
                m["xy"] = s["xy"]
    todo = [("guard_m", {"choose": "small"}, False, False), ("pull_cut", {"strict": True}, True, False),
            ("seeded_cut", {"strict": True}, True, True),
            ("near_cut", {"choose": "near", "strict": True}, True, False),
            ("seeded_near_cut", {"choose": "near", "strict": True}, True, True)]
    if len(sys.argv) > 2 and sys.argv[2] != "repair":
        todo = [t for t in todo if t[0] in sys.argv[2].split(",")]
    if len(sys.argv) > 2 and sys.argv[2] == "repair":
        todo = []
        Z = rule.Zones(U, {v[0]: list(v) for v in before.values()})
        lines = rule.cut_rounds(Z, fixed, adj, fb, region, floor, cap, "near")
        res, rings = evaluate("repair_only", Z.z)
        res["under_floor"] = sum(Z.ppl(k) < floor for k in Z.z)
        out["repair_only"] = {"res": res, "groups": Z.z, "rings": rings, "lines": lines}
    for tag, kw, cut, seeded in todo:
        groups = {i: [i] for i in free}
        if seeded:
            useeds = rule.unit_seeds(U, free, D["seeds"], list(fixed.values()))
            useeds = rule.peak_seeds(U, free, adj, useeds, 0.06 * floor)
            groups, owner = rule.grow(U, free, region, adj, useeds, 0.0, (floor + cap) / 2)
            for i in free:
                if i not in owner:
                    groups[i] = [i]
            log(f"  {len(useeds)} seeds")
        Z = rule.Zones(U, groups)
        lines = rule.floor_merge(Z, region, adj, floor, cap, fb, **kw)
        if cut:
            lines += rule.cut_rounds(Z, fixed, adj, fb, region, floor, cap, kw.get("choose", "pull"))
        res, rings = evaluate(f"after {tag}", Z.z)
        res["under_floor"] = sum(Z.ppl(k) < floor for k in Z.z)
        out[tag] = {"res": res, "groups": Z.z, "rings": rings, "lines": lines}
    path = os.path.join(HERE, f"{iso}_out.pkl")
    if os.path.exists(path):  # keep earlier runs' variants
        old = pickle.load(open(path, "rb"))
        old.update(out)
        out = old
    pickle.dump(out, open(path, "wb"))
    log(iso, "written")


if __name__ == "__main__":
    for iso in sys.argv[1].split(","):
        run(iso)
