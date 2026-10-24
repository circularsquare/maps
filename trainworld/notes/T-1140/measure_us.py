"""T-1140: compactness and travel partners of today's US zone rows (read-only)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import collections  # noqa: E402
import json  # noqa: E402

import common as C  # noqa: E402

D = C.load("us_data.pkl")
out = []
for iso in ("USA", "PRI"):
    d = D[iso]
    U = d["units"]
    S = C.Shapes(U)
    rows = d["rows"]
    owner = {}
    for r in rows:
        for i in r["units"][iso]:
            owner[i] = r["id"]
    for f in d["fixed"]:
        for i in f["units"]:
            owner.setdefault(i, f["id"])
    names = {r["id"]: r["name"] for r in rows} | {f["id"]: f["name"] for f in d["fixed"]}
    F = C.acs_sym(d["flows"], list(U)) if d["flows"] else None
    for r in rows:
        if r.get("kind") != "zone":
            continue
        ids = r["units"][iso]
        g = S.zone(ids)
        rec = {"id": r["id"], "name": r["name"], "people": sum(U[i].free for i in ids), "n": len(ids),
               "state": U[ids[0]].parents[0][1],
               "hull": C.hull_ratio(g), "pp": C.polsby(g), "reock": C.reock(g), "span": C.span_ratio(g),
               "area": g.area / 1e6}
        if F:
            to = C.area_flows(F, ids, owner)
            tot = sum(to.values())
            top = to.most_common(3)
            rec["top"] = [(names.get(a, a), round(n / tot, 3)) for a, n in top] if tot else []
        out.append(rec)
json.dump(out, open(C.HERE + r"\us_measures.json", "w", encoding="utf-8"), indent=0)
for key in ("hull", "pp", "reock", "span"):
    v = sorted(o[key] for o in out)
    print(key, "deciles", [round(v[int(q * (len(v) - 1))], 2) for q in [0, .1, .2, .3, .5, .7, .9, 1]])
print()
for o in sorted(out, key=lambda o: o["hull"]):
    print(f"{o['id']} {o['name'][:22]:22} {o['state'][:12]:12} {o['people'] / 1e6:5.2f}M n{o['n']:3} hull {o['hull']:.2f} "
          f"pp {o['pp']:.2f} reock {o['reock']:.2f} span {o['span']:.2f}  {o.get('top', '')}")
