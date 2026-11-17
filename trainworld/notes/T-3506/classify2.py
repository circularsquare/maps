"""T-3506: enclaves by the no-ring rule's own measure: the top neighbour holds >= SHARE of the zone's border with
other packs (coast, dead space and other countries' land don't count). Each case gets a rough group from its other
borders. Writes cases2.json and prints a table. usage: classify2.py [share]"""
import json, os, sys, ast, collections
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
TW = r"C:/Users/anita/projects/maps/trainworld/"
SHARE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
B = json.load(open(os.path.join(HERE, "borders.json"), encoding="utf8"))
if os.path.exists(os.path.join(HERE, "borders_only.json")):  # a later --only run over the full one
    B.update(json.load(open(os.path.join(HERE, "borders_only.json"), encoding="utf8")))
W = json.load(open(os.path.join(HERE, "borders_only_w.json"), encoding="utf8"))  # the sea pass, cands.py ids
reg = {c["id"]: c for c in json.load(open(TW + "app/src/game/registry.json", encoding="utf8"))["cities"]}
zn = json.load(open(TW + "app/src/game/zone-names.json", encoding="utf8"))
plan = json.load(open(TW + "pipeline/world_plan.json", encoding="utf8"))
rows = {r["id"]: r for r in plan["metros"]}
nm = lambda i: (zn.get(i) or {}).get("name") or reg.get(i, {}).get("name", i)


def lit(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return None
    return v


def nunits(i):
    u = lit((rows.get(i) or {}).get("units"))
    return sum(len(v) for v in u.values()) if isinstance(u, dict) else None


def unames(i):
    s = lit((rows.get(i) or {}).get("source"))
    return (s or {}).get("unit_names") if isinstance(s, dict) else None


cases = []
for i, b in B.items():
    if i not in reg:
        continue
    nb = {k: v for k, v in b["nb"].items() if k in reg}
    if not nb:
        continue
    top, tl = max(nb.items(), key=lambda kv: kv[1])
    packs = sum(nb.values())
    if tl / packs < SHARE:
        continue
    w = W.get(i, b)
    L = b["L"]
    coast = max(L - packs - w["dead"] - w["foreign"], 0) / L  # the sea pass where run
    c = {"id": i, "name": nm(i), "kind": reg[i].get("kind", "city"), "country": reg[i]["country"], "pop": b["pop"],
         "plan_people": (rows.get(i) or {}).get("people"), "area": b["area"], "share": round(tl / packs, 3),
         "pack_border": round(packs / L, 3), "coast": round(coast, 3), "dead_km": w["dead"], "foreign_km": w["foreign"],
         "L": L, "by": top, "by_name": nm(top), "by_kind": reg[top].get("kind", "city"), "by_country": reg[top]["country"],
         "by_pop": B.get(top, {}).get("pop"), "by_plan_people": (rows.get(top) or {}).get("people"),
         "by_units": nunits(top), "units": nunits(i), "unit_names": unames(i), "status": (rows.get(i) or {}).get("status"),
         "others": {k: v for k, v in nb.items() if k != top}}
    if c["by_country"] != c["country"]:
        g = "foreign"
    elif c["pack_border"] < 0.6 and c["coast"] >= 0.25:
        g = "coast"          # most of its edge is sea: island half, peninsula tip
    elif c["pack_border"] < 0.6:
        g = "edge"           # most of its edge is dead space or another country's land
    else:
        g = "inland"
    c["group"] = g
    cases.append(c)
cases.sort(key=lambda c: -(c["pop"] or 0))
json.dump(cases, open(os.path.join(HERE, "cases2.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(len(cases), "cases at pack share", SHARE, collections.Counter(c["group"] for c in cases))
for g in ("inland", "edge", "coast", "foreign"):
    print("\n==", g)
    for c in cases:
        if c["group"] != g:
            continue
        un = c["unit_names"]
        un = (", ".join(un[:3]) + (f" +{len(un) - 3}" if len(un) > 3 else "")) if un else ""
        print(f"{c['id']:>4} {c['name'][:22]:22} {c['kind'][:4]} {c['country']} {(c['pop'] or 0):>10,.0f} sh {c['share']:.2f} packb {c['pack_border']:.2f} "
              f"coast {c['coast']:.2f} dead {c['dead_km']:.0f} for {c['foreign_km']:.0f} | in {c['by']} {c['by_name'][:16]:16} {c['by_kind'][:4]} "
              f"{(c['by_pop'] or 0):>10,.0f} u{c['by_units']} | u{c['units']} {un[:60]}")

