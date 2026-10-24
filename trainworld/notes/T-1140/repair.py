"""T-1140 repair dry run on built rows, by the one merge rule:
1. a zone row fails when the rule, given only its own units (each alone, the rest of the country as
   it is), cannot rebuild it whole (two lobes that are not travel partners and not compact), or when
   it wraps another zone, a metro or a pack (T-1087's ring test; T-1089);
2. failed rows go back to their units; then every zone under the floor (failed units, and kept rows
   under it: T-903's small zones) joins by the rule, kept rows whole, metros and packs fixed, the
   ring cut after.
Read-only; writes <data>_repair_<variant>.pkl.

    python repair.py us USA,PRI [variant] [--aref 10000] [--keep-small]
    python repair.py chn CHN rb
    python repair.py europe ALL rb
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
import common as C  # noqa: E402
import rule1140 as R  # noqa: E402
import world_plan as wp  # noqa: E402
from run_us import FNS  # noqa: E402
from shapely import STRtree, union_all  # noqa: E402

t0 = time.time()
V = {"r": {"alpha": 0.5, "compact": ("hull", 0.75), "relative": 0.05, "small_by": "flow"},
     "rb": {"alpha": 0.5, "compact": ("hull", 0.75), "relative": 0.05, "small_by": "flow", "cross": "fallback"},
     "rb70": {"alpha": 0.5, "compact": ("hull", 0.70), "relative": 0.05, "small_by": "flow", "cross": "fallback"},
     "today": {}}


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def prep(name, D):
    path = os.path.join(C.HERE, f"{name}_prep.pkl")
    P = C.load(f"{name}_prep.pkl") if os.path.exists(path) else {}
    changed = False
    for iso, d in D.items():
        if iso in P:
            continue
        U = d["units"]
        simple = {i: wp.polygonal(u.geom.simplify(500)) for i, u in U.items()}
        adj = wp.unit_borders(simple)
        acs = C.acs_sym(d["flows"], list(U)) if d.get("flows") else None
        grav = C.gravity_flows(d["cells"], list(U)) if U else {}
        P[iso] = {"simple": simple, "adj": adj, "acs": acs, "grav": grav}
        changed = True
        log(f"{iso}: shapes, borders and flows")
    if changed:
        pickle.dump(P, open(path, "wb"))
    return P


def rings(groups, simple, fixed):
    """{zone lead: name of what it wraps}: T-1087's test (half of B's area in A's hull, A holding half
    of B's border with zones, metros and packs)."""
    shp = [("z", k, union_all([simple[i] for i in v])) for k, v in groups.items()]
    shp += [("f", f["name"], f["geom"]) for f in fixed]
    gs = [g for _, _, g in shp]
    nears = [g.buffer(1000) for g in gs]
    tree = STRtree(gs)
    nb = [{} for _ in shp]
    for a in range(len(shp)):
        for b in tree.query(nears[a], predicate="intersects"):
            if b != a:
                nb[a][int(b)] = gs[b].boundary.intersection(nears[a]).length
    out = {}
    for a, (t, k, g) in enumerate(shp):
        if t != "z":
            continue
        hull = g.convex_hull
        for b, ln in nb[a].items():
            land = sum(nb[x].get(b, 0.0) for x in nb[b])
            if not land or not gs[b].area:
                continue
            if gs[b].intersection(hull).area / gs[b].area >= wp.WRAP_HULL and ln / land >= wp.WRAP_BORDER:
                out[k] = shp[b][1] if shp[b][0] == "f" else shp[b][1]
    return out


def floor_cap(iso):
    if iso in wp.ZONE_RULES:
        r = wp.ZONE_RULES[iso]
        return r["floor"], r["cap"], r.get("contiguous", False)
    return wp.ZONE_FLOOR, wp.ZONE_CAP, True  # T-897's no-exclave rule now holds for every new plan


def main():
    args = [a for a in sys.argv[1:]]
    aref = None
    if "--aref" in args:
        k = args.index("--aref")
        aref = float(args[k + 1])
        del args[k:k + 2]
    minpart = 0.0
    if "--minpart" in args:
        k = args.index("--minpart")
        minpart = float(args[k + 1])
        del args[k:k + 2]
    keep_small = "--keep-small" in args
    args = [a for a in args if a != "--keep-small"]
    name, which = args[0], args[1]
    tag = args[2] if len(args) > 2 else "rb"
    D = C.load(f"{name}_data.pkl")
    isos = [k for k in D if D[k]["units"]] if which == "ALL" else which.split(",")
    D = {k: D[k] for k in isos}
    P = prep(name, D)
    out = {}
    for iso in isos:
        d, p = D[iso], P[iso]
        U = d["units"]
        units = list(U.values())
        free = list(d["free"])
        region = d.get("region") or {i: u.parents[0] for i, u in U.items()}
        floor, cap, contiguous = floor_cap(iso)
        kw = dict(V[tag])
        if "compact" in kw:
            kw["compact"] = (FNS[kw["compact"][0]], kw["compact"][1])
            kw["flows"] = p["acs"] or p["grav"]
        fu = {f["id"]: f["units"] for f in d["fixed"]}
        zrows = [r for r in d["rows"] if r.get("kind") == "zone"]
        plan = {r["id"]: [i for i in r["units"][iso] if i in U] for r in zrows}
        plan = {k: v for k, v in plan.items() if v}
        lead_of = {k: v[0] for k, v in plan.items()}
        rid_of = {v[0]: k for k, v in plan.items()}
        base = {v[0]: list(v) for v in plan.values()}
        fails = {}
        if kw:
            for rid, ids in plan.items():
                if len(ids) < 2:
                    continue
                init = {k: v for k, v in base.items() if k != ids[0]}
                init.update({i: [i] for i in ids})
                g, lines = R.zone_groups(units, free, region, floor, cap, contiguous, (), init=init, movable=set(ids),
                                         targets=set(ids), rings_on=False, guard=False, fixed_units=fu,
                                         simple=p["simple"], adj=p["adj"], **kw)
                parts = [v for v in g.values() if set(v) & set(ids)]
                pp = sorted((sum(U[i].free for i in v) for v in parts), reverse=True)
                if len(parts) > 1 and pp[1] >= minpart * sum(pp):
                    fails[rid] = {"why": "lobes", "parts": parts,
                                  "lines": [ln for ln in lines if any(U[i].name in ln for i in ids)]}
        rg = rings(base, p["simple"], d["fixed"])
        for lead, what in rg.items():
            rid = rid_of[lead]
            if rid in fails:
                fails[rid]["why"] += ", ring"
            else:
                fails[rid] = {"why": "ring", "parts": [plan[rid]], "lines": []}
            fails[rid]["wraps"] = what
        nl = sum(1 for f in fails.values() if "lobes" in f["why"])
        log(f"{iso}: floor {floor:,}, {len(plan)} zone rows, {sum(len(v) > 1 for v in plan.values())} of several units; "
            f"{nl} fail the partner-or-compact test, {len(rg)} wrap something")
        init = {lead_of[k]: list(v) for k, v in plan.items() if k not in fails}
        loose = [i for k in fails for i in plan[k]]
        init.update({i: [i] for i in loose})
        move = set(loose) if keep_small else set(init)
        tests = []
        g, lines = R.zone_groups(units, free, region, floor, cap, contiguous, d["fixed"], init=init, movable=move,
                                 fixed_units=fu, simple=p["simple"], adj=p["adj"], tests=tests, aref=aref, **kw)
        for lead in [k for k, v in g.items() if sum(U[i].free for i in v) < wp.ZONE_ALONE_MIN]:
            if not any(set(g[lead]) == set(v) for v in plan.values()):
                g.pop(lead)
        left = rings(g, p["simple"], d["fixed"])
        changed = [k for k, v in plan.items() if frozenset(v) not in {frozenset(x) for x in g.values()}]
        log(f"{iso}: {len(plan)} rows -> {len(g)} zones; {len(changed)} rows change; rings left {len(left)}")
        out[iso] = {"plan": plan, "fails": fails, "groups": g, "lines": lines, "floor": floor, "cap": cap,
                    "rings_before": rg, "rings_after": left, "aref": aref}
    suffix = tag + (f"_a{int(aref)}" if aref else "") + ("_ks" if keep_small else "") + (f"_m{int(minpart * 100)}" if minpart else "")
    path = os.path.join(C.HERE, f"{name}_repair_{suffix}.pkl")
    old = C.load(f"{name}_repair_{suffix}.pkl") if os.path.exists(path) else {}
    old.update(out)
    pickle.dump(old, open(path, "wb"))
    log("written", path)


if __name__ == "__main__":
    main()
