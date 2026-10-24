"""T-1140: a US dry run against the plan's zone rows: sizes, compactness, partners, ACS shares, and
every zone that changes. Read-only.

    python eval_us.py <variant> [--list]
"""
import collections
import json
import math
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8")
import common as C  # noqa: E402

D = C.load("us_data.pkl")
RUNS = C.load("us_runs.pkl")
tag = sys.argv[1]
LIST = "--list" in sys.argv
FLOOR, CAP = 500_000, 1_500_000


def disp(U, ids):
    """People dispersion: people-weighted mean distance from the people centre over the radius of a
    disc of the zone's area (a disc of even people: 0.67)."""
    xs, ys, ps = [], [], []
    for i in ids:
        x, y, p = D_iso["cells"][i]
        xs += list(x); ys += list(y); ps += list(p)
    P = sum(ps)
    if not P:
        return 0.0
    cx = sum(a * b for a, b in zip(xs, ps)) / P
    cy = sum(a * b for a, b in zip(ys, ps)) / P
    md = sum(math.hypot(x - cx, y - cy) * p for x, y, p in zip(xs, ys, ps)) / P
    area = sum(U[i].geom.area for i in ids)
    return md / math.sqrt(area / math.pi)


summary = {}
changes = []
for iso in ("USA", "PRI"):
    D_iso = d = D[iso]
    U = d["units"]
    S = C.Shapes(U)
    rows = [r for r in d["rows"] if r.get("kind") == "zone"]
    plan = {r["id"]: r["units"][iso] for r in rows}
    pname = {r["id"]: r["name"] for r in rows}
    groups = RUNS[tag][iso]["groups"]
    fixed_owner = {i: f["id"] for f in d["fixed"] for i in f["units"]}
    names = {f["id"]: f["name"] for f in d["fixed"]}
    F = C.acs_sym(d["flows"], list(U)) if d["flows"] else None
    state_units = collections.defaultdict(set)
    for i, u in U.items():
        state_units[u.parents[0][1]].add(i)

    def owner_of(gr):
        o = {i: k for k, ids in gr.items() for i in ids}
        for i, f in fixed_owner.items():
            o.setdefault(i, f)
        return o

    def label(gr, k):
        ids = gr[k]
        lead = max(ids, key=lambda i: U[i].free)
        st = U[lead].parents[0][1]
        if set(ids) == state_units[st]:
            return st + " (whole state)"
        return U[lead].name

    def stats(gr):
        own = owner_of(gr)
        tot = within = xt = xw = 0.0
        for k, ids in gr.items():
            for h in ids:
                for w, n in (d["flows"] or {}).get(h, {}).items():
                    tot += n
                    if w != h:
                        xt += n
                    if own.get(w) == k:
                        within += n
                        if w != h:
                            xw += n
        return (within / tot if tot else None), (xw / xt if xt else None)

    def zone_rec(gr, k, own):
        ids = gr[k]
        g = S.zone(ids)
        rec = {"name": label(gr, k), "people": sum(U[i].free for i in ids), "n": len(ids),
               "hull": C.hull_ratio(g), "disp": disp(U, ids)}
        if F:
            to = C.area_flows(F, ids, own)
            top = to.most_common(1)
            tot = sum(to.values())
            if top:
                a = top[0][0]
                rec["top"] = (label(gr, a) if a in gr else names.get(a, a), top[0][1] / tot)
        return rec

    for which, gr in (("plan", plan), (tag, groups)):
        ppl = sorted(sum(U[i].free for i in ids) for ids in gr.values())
        hulls = [C.hull_ratio(S.zone(ids)) for ids in gr.values()]
        a, x = stats(gr)
        summary[(iso, which)] = {"zones": len(gr), "median": statistics.median(ppl),
                                 "under_floor": sum(p < FLOOR for p in ppl), "under_400k": sum(p < 400_000 for p in ppl),
                                 "over_cap": sum(p > CAP for p in ppl), "hull_med": statistics.median(hulls),
                                 "hull_under_07": sum(h < 0.7 for h in hulls), "all_kept": a, "leavers_kept": x,
                                 "whole_states": sorted(label(gr, k) for k in gr if label(gr, k).endswith("(whole state)"))}
    # changes: new groups not identical to a plan row
    pkey = {frozenset(v): k for k, v in plan.items()}
    unit_plan = {i: k for k, ids in plan.items() for i in ids}
    own_new, own_old = owner_of(groups), owner_of(plan)
    for k, ids in sorted(groups.items(), key=lambda kv: -sum(U[i].free for i in kv[1])):
        if frozenset(ids) in pkey:
            continue
        rec = zone_rec(groups, k, own_new)
        frm = collections.Counter(unit_plan.get(i) for i in ids)
        rec["from"] = [(pname.get(p, p), n, len(plan.get(p, ())), zone_rec(plan, p, own_old)["hull"] if p in plan else None)
                       for p, n in frm.most_common()]
        rec["iso"] = iso
        rec["units"] = sorted(ids)
        rec["state"] = U[ids[0]].parents[0][1]
        changes.append(rec)
    gone = [k for k, v in plan.items() if frozenset(v) not in {frozenset(x) for x in groups.values()}]
    summary[(iso, "changed_plan_rows")] = len(gone)
    summary[(iso, "unchanged")] = len(plan) - len(gone)

for k, v in summary.items():
    print(k, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()} if isinstance(v, dict) else v, ensure_ascii=False))
print(len(changes), "new zones")
under = [c for c in changes if c["people"] < FLOOR]
print("new zones under the floor:", ", ".join(f"{c['name']} {c['people'] / 1e6:.2f}" for c in under))
if LIST:
    for c in sorted(changes, key=lambda c: (c["state"], -c["people"])):
        top = c.get("top")
        print(f"{c['state'][:14]:14} {c['name'][:30]:30} {c['people'] / 1e6:5.2f}M n{c['n']:3} hull {c['hull']:.2f} disp {c['disp']:.2f}"
              f" top {top[0] if top else '-'} {top[1] if top else 0:.0%} | from " +
              "; ".join(f"{p} ({n}/{m}{', hull %.2f' % h if h else ''})" for p, n, m, h in c["from"]))
json.dump({"summary": {f"{a}|{b}": v for (a, b), v in summary.items()}, "changes": changes},
          open(C.HERE + f"\\eval_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0, default=str)
