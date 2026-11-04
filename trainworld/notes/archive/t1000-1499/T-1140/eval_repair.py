"""T-1140: a repair dry run against the plan: counts, and every zone that changes (its units' names,
people, hull ratio before and after, top travel partner). Read-only.

    python eval_repair.py <data> <result.pkl> [--list] [--isos A,B]
"""
import collections
import json
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8")
import common as C  # noqa: E402

data, res = sys.argv[1], sys.argv[2]
LIST = "--list" in sys.argv
D = C.load(f"{data}_data.pkl")
P = C.load(f"{data}_prep.pkl")
Rz = C.load(res)
isos = sys.argv[sys.argv.index("--isos") + 1].split(",") if "--isos" in sys.argv else list(Rz)
tot = collections.Counter()
allchanges = []
for iso in isos:
    if iso not in Rz:
        continue
    d, r, p = D[iso], Rz[iso], P[iso]
    U = d["units"]
    S = C.Shapes(U)
    F = p["acs"] or p["grav"]
    plan, g = r["plan"], r["groups"]
    floor = r["floor"]
    region = d.get("region") or {i: u.parents[0] for i, u in U.items()}
    reg_units = collections.defaultdict(set)
    for i in U:
        reg_units[region[i]].add(i)
    fixed_owner = {i: f["id"] for f in d["fixed"] for i in f["units"]}
    fname = {f["id"]: f["name"] for f in d["fixed"]}
    pname = {k: next((x["name"] for x in d["rows"] if x["id"] == k), k) for k in plan}
    pset = {frozenset(v): k for k, v in plan.items()}
    unit_row = {i: k for k, v in plan.items() for i in v}

    def label(ids):
        lead = max(ids, key=lambda i: U[i].free)
        reg = region[lead]
        whole = set(ids) == reg_units[reg]
        nm = U[lead].name
        return nm + (f" [whole {reg[1] if isinstance(reg, tuple) else reg}]" if whole and len(ids) > 1 else "")

    def owner(gr):
        o = {i: k for k, v in gr.items() for i in v}
        for i, f in fixed_owner.items():
            o.setdefault(i, f)
        return o

    own_after = owner(g)
    gl = {k: v for k, v in g.items()}
    ppl_b = [sum(U[i].free for i in v) for v in plan.values()]
    ppl_a = [sum(U[i].free for i in v) for v in g.values()]
    changed = [k for k, v in plan.items() if frozenset(v) not in {frozenset(x) for x in g.values()}]
    new = [k for k, v in g.items() if frozenset(v) not in pset]
    fl = r["fails"]
    nl = sum("lobes" in f["why"] for f in fl.values())
    print(f"{iso}: rows {len(plan)} (under the floor {sum(x < floor for x in ppl_b)}, median {statistics.median(ppl_b) / 1e6:.2f}M); "
          f"fail the test {nl}, rings {len(r['rings_before'])}; after {len(g)} zones (under the floor {sum(x < floor for x in ppl_a)}, "
          f"median {statistics.median(ppl_a) / 1e6:.2f}M), {len(changed)} rows change into {len(new)} new zones; rings left {len(r['rings_after'])}")
    tot.update({"rows": len(plan), "after": len(g), "changed": len(changed), "new": len(new), "lobes": nl,
                "rings": len(r["rings_before"]), "rings_left": len(r["rings_after"]),
                "under_b": sum(x < floor for x in ppl_b), "under_a": sum(x < floor for x in ppl_a)})
    for k in new:
        ids = g[k]
        to = C.area_flows(F, ids, own_after)
        tt = sum(to.values())
        top = to.most_common(1)
        tn = "-"
        if top:
            a = top[0][0]
            tn = (label(g[a]) if a in g else fname.get(a, a)) + f" {top[0][1] / tt:.0%}"
        frm = collections.Counter(unit_row.get(i) for i in ids)
        src = []
        for rid, n in frm.most_common():
            hb = C.hull_ratio(S.zone(plan[rid])) if rid in plan else None
            why = fl.get(rid, {}).get("why", "")
            src.append(f"{pname.get(rid, rid)} ({rid}; {n}/{len(plan.get(rid, ()))}"
                       + (f", hull {hb:.2f}" if hb else "") + (f", {why}" if why else "") + ")")
        rec = {"iso": iso, "name": label(ids), "people": sum(U[i].free for i in ids), "n": len(ids),
               "hull": C.hull_ratio(S.zone(ids)), "top": tn, "from": src,
               "units": [U[i].name for i in sorted(ids, key=lambda i: -U[i].free)],
               "region": str(region[ids[0]][1] if isinstance(region[ids[0]], tuple) else region[ids[0]])}
        allchanges.append(rec)
    if LIST:
        for c in sorted([c for c in allchanges if c["iso"] == iso], key=lambda c: (c["region"], -c["people"])):
            print(f"  {c['region'][:16]:16} {c['name'][:40]:40} {c['people'] / 1e6:5.2f}M n{c['n']:3} hull {c['hull']:.2f} top {c['top'][:34]:34} | "
                  + "; ".join(c["from"]) + ("" if c["n"] > 6 else " | " + ", ".join(c["units"])))
print("TOTAL", dict(tot))
json.dump(allchanges, open(res.replace(".pkl", "_changes.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
