"""T-1087 US dry run: today's plan against the seeded rule. Read-only; writes us_<variant>.json here.

    python us_run.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import collections
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
FLOOR, CAP = 500_000, 1_500_000


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


D = pickle.load(open(os.path.join(HERE, "us_data.pkl"), "rb"))
U = D["units"]
rows = D["rows"]
zrows = [r for r in rows if r["kind"] == "zone"]
mrows = [r for r in rows if r["kind"] != "zone"]
free = sorted({i for r in zrows for i in r["units"]})
region = {i: U[i]["state"] for i in U}
STATUS = {}
for r in zrows:
    STATUS[r["id"]] = r["status"]

# fixed areas: metro rows and the 13 built packs
fixed = {}
for r in mrows:
    fixed[r["id"]] = {"id": r["id"], "name": r["name"], "geom": union_all([U[i]["simple"] for i in r["units"]]),
                      "units": r["units"], "pop": r["people"], "status": r["status"]}
for k, p in D["packs"].items():
    fixed[k] = {"id": k, "name": p["name"], "geom": shapely.simplify(p["geom"], 500), "pop": 0, "status": "built (old)"}
# counties inside an old pack: the pack holding their people centroid
cty_area = {}
for r in mrows:
    for i in r["units"]:
        cty_area[i] = r["id"]
for i, u in U.items():
    if u["inpack"] and i not in cty_area:
        pt = shapely.Point(*u["c"])
        for k, p in D["packs"].items():
            if p["geom"].contains(pt):
                cty_area[i] = k
                break
for k in D["packs"]:
    fixed[k]["units"] = [i for i, a in cty_area.items() if a == k]
    fixed[k]["pop"] = sum(U[i]["pop"] for i in fixed[k]["units"])
# a metro's centre: its biggest urban centre
for s in sorted(D["seeds"], key=lambda s: s["pop"]):
    pt = shapely.Point(*s["xy"])
    for m in fixed.values():
        if m["geom"].contains(pt):
            m["xy"] = s["xy"]
for m in fixed.values():
    if "xy" not in m:
        c = m["geom"].representative_point()
        m["xy"] = (c.x, c.y)
FL = D["flows"]


def fixed_borders(ids, fx):
    """[(geom, land border, {unit: border along it})] of the fixed areas for floor_merge."""
    out = []
    gs = {i: U[i]["simple"] for i in ids}
    tree = shapely.STRtree(list(gs.values()))
    keys = list(gs)
    allfix = list(fx.values())
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


def belt(share=0.25):
    """Free counties sending at least `share` of their resident workers (ACS 2016-2020) to one metro
    row or built pack, chained by land to it: {metro id: [county ids]}."""
    best = {}
    for h in free:
        tot = sum(FL.get(h, {}).values())
        if not tot:
            continue
        to = {}
        for w, n in FL[h].items():
            a = cty_area.get(w)
            if a:
                to[a] = to.get(a, 0) + n
        if to:
            a, n = max(to.items(), key=lambda kv: kv[1])
            if n / tot >= share:
                best[h] = a
    adj = rule.adjacency(U, list(U))
    got = {}
    for a in fixed:
        have = set(fixed[a]["units"])
        front = [i for i in have]
        while front:
            nxt = []
            for i in front:
                for j in adj[i]:
                    if best.get(j) == a and j not in have and j not in got:
                        have.add(j)
                        got[j] = a
                        nxt.append(j)
            front = nxt
    out = {}
    for j, a in got.items():
        out.setdefault(a, []).append(j)
    return out


def grown(extra):
    """Copies of fixed and cty_area with each metro's extra counties."""
    fx = {k: dict(v) for k, v in fixed.items()}
    ca = dict(cty_area)
    for a, ids in extra.items():
        fx[a]["units"] = list(fx[a]["units"]) + ids
        fx[a]["geom"] = union_all([fx[a]["geom"]] + [U[i]["simple"] for i in ids])
        fx[a]["pop"] += sum(U[i]["free"] for i in ids)
        for i in ids:
            ca[i] = a
    return fx, ca


def evaluate(name, groups, fx=None, ca=None):
    """Sizes, rings by the hull and border test, and ACS self-containment."""
    fx = fx or fixed
    ca = ca or cty_area
    Z = rule.Zones(U, groups)
    ppl = sorted(Z.ppl(k) for k in Z.z)
    shapes = {k: Z.geo(k) for k in Z.z}
    kinds = {k: "zone" for k in Z.z}
    for k, m in fx.items():
        shapes[k] = m["geom"]
        kinds[k] = "metro"
    pairs = rule.ring_pairs(shapes, kinds)
    rings = [(a, b, ih, bd) for a, b, ih, bd in pairs if ih >= 0.5 and bd >= 0.5]
    zz = [p for p in rings if kinds[p[1]] == "zone"]
    zm = [p for p in rings if kinds[p[1]] == "metro"]
    # ACS: each zone's resident workers, those working in it, and those working in a metro it touches
    owner = {i: k for k, ids in Z.z.items() for i in ids}
    sc = {}
    tot_all = within_all = xt_all = xw_all = 0.0
    for k, ids in Z.z.items():
        tot = within = xt = xw = 0.0
        to = collections.Counter()
        for h in ids:
            for w, n in FL.get(h, {}).items():
                tot += n
                if w != h:
                    xt += n
                if owner.get(w) == k:
                    within += n
                    if w != h:
                        xw += n
                elif w in ca:
                    to[ca[w]] += n
        top = to.most_common(1)
        sc[k] = {"workers": tot, "within": within / tot if tot else 0, "x_within": xw / xt if xt else 0,
                 "x_metro": top[0][1] / xt if top and xt else 0,
                 "metro": top[0][0] if top else None, "to_metro": top[0][1] / tot if top and tot else 0}
        tot_all += tot
        within_all += within
        xt_all += xt
        xw_all += xw
    # every worker living in a zone or a metro: share working in that same area
    area_of = dict(owner)
    for k, m in fx.items():
        for i in m["units"]:
            area_of[i] = k
    t_all = w_all = 0.0
    for h, a in area_of.items():
        for w, n in FL.get(h, {}).items():
            t_all += n
            if area_of.get(w) == a:
                w_all += n
    comp = sorted(shapes[k].area / shapes[k].convex_hull.area for k in Z.z)
    res = {"name": name, "zones": len(Z.z), "median": statistics.median(ppl), "under_400k": sum(p < 400_000 for p in ppl),
           "over_cap": sum(p > CAP for p in ppl), "max": max(ppl), "rings_zz": len(zz), "rings_zm": len(zm),
           "ring_zones": len({a for a, *_ in rings}), "compact_median": statistics.median(comp),
           "compact_under_05": sum(c < 0.5 for c in comp),
           "self_contained": within_all / tot_all, "all_areas_sc": w_all / t_all, "x_within": xw_all / xt_all,
           "under_floor": sum(p < FLOOR for p in ppl),
           "sc_under_60": sum(1 for s in sc.values() if s["within"] < 0.6),
           "sc_median": statistics.median(s["within"] for s in sc.values())}
    log(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}))
    return res, Z, rings, sc


def after(choose="pull", guard=True, min_seed=0.0, fx=None, seeded=True, peak=None, cap=None, strict=False, repair=False):
    fx = fx or fixed
    taken = {i for m in fx.values() for i in m["units"]}
    ids = [i for i in free if i not in taken]
    adj = rule.adjacency(U, ids)
    useeds = rule.unit_seeds(U, ids, D["seeds"], list(fx.values())) if seeded else {}
    if peak:
        useeds = rule.peak_seeds(U, ids, adj, useeds, peak)
    groups, owner = rule.grow(U, ids, region, adj, useeds, min_seed, cap)
    left = [i for i in ids if i not in owner]
    for i in left:
        groups[i] = [i]
    Z = rule.Zones(U, groups)
    fb = fixed_borders(ids, fx)
    lines = rule.floor_merge(Z, region, adj, FLOOR, CAP, fb, choose, guard, strict=strict)
    if repair:
        lines += rule.cut_rounds(Z, fx, adj, fb, region, FLOOR, CAP, choose)
    return Z.z, useeds, left, lines


def cut_rounds(Z, fx, adj, fb, choose, rounds=3):
    """Cut each ring zone across what it wraps, let the halves regroup (strict guard), repeat."""
    lines = []
    for rnd in range(rounds):
        shapes = {k: Z.geo(k) for k in Z.z}
        kinds = {k: "zone" for k in Z.z}
        for k, m in fx.items():
            shapes[k] = m["geom"]
            kinds[k] = "metro"
        rings = [p for p in rule.ring_pairs(shapes, kinds) if p[2] >= 0.5 and p[3] >= 0.5]
        if not rings:
            break
        done = set()
        for a, b, ih, bd in sorted(rings, key=lambda p: -p[2] - p[3]):
            if a in done or a not in Z.z:
                continue
            centre = fx[b]["xy"] if kinds[b] == "metro" else Z.cen(b)
            got = rule.cut_ring(Z, a, centre, adj)
            if not got:
                lines.append(f"round {rnd}: {a} wraps {b}, no cut")
                continue
            h1, h2 = got
            Z.z.pop(a)
            Z.z[h1[0]] = h1
            Z.z[h2[0]] = h2
            done |= {h1[0], h2[0]}
            lines.append(f"round {rnd}: {a} wraps {b}, cut {len(h1)}+{len(h2)} units")
        lines += rule.floor_merge(Z, region, adj, FLOOR, CAP, fb, choose, True, strict=True)
    return lines


def main():
    before = {r["id"]: r["units"] for r in zrows}
    out = {}
    res, Zb, rb, scb = evaluate("before (plan)", before)
    out["before"] = {"res": res, "groups": before, "rings": rb, "sc": scb}
    ext = belt(0.25)
    log(f"belt at 25%: {sum(map(len, ext.values()))} counties, {sum(U[i]['free'] for v in ext.values() for i in v) / 1e6:.2f}M, "
        f"into {len(ext)} metros: " + ", ".join(f"{fixed[a]['name']} +{len(v)}" for a, v in sorted(ext.items(), key=lambda kv: -len(kv[1]))))
    fxg, cag = grown(ext)
    todo = [("guard_m", {"seeded": False, "choose": "small"}), ("pull_guard", {"seeded": False}),
            ("seeded_cap", {"cap": CAP}), ("seeded_peak", {"cap": CAP, "peak": 30_000}),
            ("seeded_peak_cap1M", {"cap": 1_000_000, "peak": 30_000}),
            ("grow25", {"fx": fxg, "seeded": False}), ("grow25_peak", {"fx": fxg, "cap": CAP, "peak": 30_000}),
            ("pull_strict", {"seeded": False, "strict": True}),
            ("grow25_near_cut", {"fx": fxg, "seeded": False, "choose": "near", "strict": True, "repair": True}),
            ("pull_cut", {"seeded": False, "strict": True, "repair": True}),
            ("peak_cut", {"cap": CAP, "peak": 30_000, "strict": True, "repair": True}),
            ("peak_cut_1M", {"cap": 1_000_000, "peak": 30_000, "strict": True, "repair": True}),
            ("near_cut", {"seeded": False, "choose": "near", "strict": True, "repair": True}),
            ("peak_near_cut_1M", {"cap": 1_000_000, "peak": 30_000, "choose": "near", "strict": True, "repair": True}),
            ("peak_cut_1M_p50", {"cap": 1_000_000, "peak": 50_000, "strict": True, "repair": True}),
            ("grow25_peak_cut_1M", {"fx": fxg, "cap": 1_000_000, "peak": 30_000, "strict": True, "repair": True}),
            ("small_cut", {"seeded": False, "choose": "small", "strict": True, "repair": True}),
            ("grow25_cut", {"fx": fxg, "seeded": False, "strict": True, "repair": True})]
    if len(sys.argv) > 1:
        todo = [t for t in todo if t[0] in sys.argv[1:]]
    if "repair_only" in sys.argv:
        # today's plan with only its rings cut and the halves regrouped (the built rows' local fix)
        ids = list(free)
        adj = rule.adjacency(U, ids)
        Z = rule.Zones(U, {v[0]: list(v) for v in before.values()})
        lines = rule.cut_rounds(Z, fixed, adj, fixed_borders(ids, fixed), region, FLOOR, CAP, "near")
        res, _, rg, sc = evaluate("repair_only", Z.z)
        out["repair_only"] = {"res": res, "groups": Z.z, "rings": rg, "sc": sc, "lines": lines}
    for tag, kw in todo:
        g, us, left, lines = after(**kw)
        res, Z, rg, sc = evaluate(f"after {tag}", g, *((fxg, cag) if tag.startswith("grow25") else ()))
        log(f"  {len(us)} seed units, {len(left)} units no seed reached")
        out[tag] = {"res": res, "groups": g, "rings": rg, "sc": sc, "lines": lines,
                    "seeds": {k: v["names"] for k, v in us.items()}}
    path = os.path.join(HERE, "us_out.pkl")
    if os.path.exists(path):  # keep the variants of earlier runs
        old = pickle.load(open(path, "rb"))
        old.update(out)
        out = old
    pickle.dump(out, open(path, "wb"))
    log("written us_out.pkl")


if __name__ == "__main__":
    main()
