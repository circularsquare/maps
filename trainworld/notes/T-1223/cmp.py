# T-1223: figures of the zone in view, a LOD load against a fine one.
#   cmp.py fine.json lod.json [zone]
import json, sys
reg = json.load(open(r"C:\Users\anita\projects\maps\.tw-trees\t1223\trainworld\app\src\game\registry.json", encoding="utf-8"))
box = {c["id"]: c.get("box") for c in reg["cities"]}
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
fa = a.get("refined") or a["final"]
fb = b.get("refined") or b["final"]
zone = sys.argv[3] if len(sys.argv) > 3 else fb["active"]
print("zone", zone, "lod", json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in (fb.get("lod") or {}).items()}))
print("network rail fine %.0f lod %.0f (%+.2f%%)" % (fa["rail"], fb["rail"], 100 * (fb["rail"] / fa["rail"] - 1)))
ca, cb = fa["cities"].get(zone), fb["cities"].get(zone)
if ca and cb:
    print("zone rail fine %.0f lod %.0f (%+.2f%%); train share %.2f%% vs %.2f%%; trips %+.3f%%" % (ca["rail"], cb["rail"], 100 * (cb["rail"] / ca["rail"] - 1), ca["split"]["train"], cb["split"]["train"], 100 * (cb["trips"] / ca["trips"] - 1)))
w, s, e, n = box[zone]
where = fb.get("where") or fa.get("where") or {}
ins = [k for k, (x, y) in where.items() if w <= x <= e and s <= y <= n]
rows = []
for k in ins:
    va, vb = fa["stations"].get(k, 0), fb["stations"].get(k, 0)
    if va >= 1000:
        rows.append((vb / va - 1, k, va, vb))
rows.sort(key=lambda r: -abs(r[0]))
tot_a = sum(fa["stations"].get(k, 0) for k in ins)
tot_b = sum(fb["stations"].get(k, 0) for k in ins)
print("stations in its box", len(ins), "with 1,000+ boardings", len(rows), "; boardings there %+.2f%%" % (100 * (tot_b / tot_a - 1) if tot_a else 0))
print("worst stations:", [("%+.1f%%" % (100 * r[0]), r[1], round(r[2])) for r in rows[:6]])
big = [r for r in rows if r[2] >= 20000]
if big:
    print("worst of 20,000+:", [("%+.2f%%" % (100 * r[0]), r[1], round(r[2])) for r in big[:4]])
# cities overall: coarse zones' rail vs fine
diffs = []
for z, v in fb["cities"].items():
    u = fa["cities"].get(z)
    if u and u["rail"] > 1000:
        diffs.append((v["rail"] / u["rail"] - 1, z, u["rail"]))
diffs.sort(key=lambda r: -abs(r[0]))
print("zones' rail, worst:", [("%+.1f%%" % (100 * d), z, round(r)) for d, z, r in diffs[:6]])
print("memory lod", [(m["worker"], m["wasm"], m["heap"], m["solvePeak"], m["need"]) for m in fb["memory"]])
print("memory fine", [(m["worker"], m["wasm"], m["heap"], m["solvePeak"], m["need"]) for m in fa["memory"]])
print("secs fine", a.get("finalSecs"), "lod", b.get("finalSecs"), b.get("refineSecs"))
