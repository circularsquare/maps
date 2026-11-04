# T-1735 (from T-1733): an edit's local result against its whole solve, the new lines' longer trips too.  cmp.py out.json [edit1|edit2]
import json, sys
d = json.load(open(sys.argv[1]))
tag = sys.argv[2] if len(sys.argv) > 2 else "edit1"
e, w, b = d[tag], d["whole"], d["base"]
loc = e["local"]
print(tag, "line", e["line"], "built in %.1f s, local shown %.1f s after the edit began" % (e["builtSecs"], e["localSecs"]), "; whole shown %.1f s after edit1 began" % (d["wholeSecs"] - d["edit1"]["editSecs"]))
print("solve keys at local:", loc["solve"]["keys"][:6])
new = {str(d[t]["line"]) for t in ("edit1", "edit2") if t in d}
bs, ws = b.get("svcLines", {}), w.get("svcLines", {})
changed = {k for k, v in ws.items() if k in bs and (bs[k]["stops"] != v["stops"] or any(abs(x - y) > 1e-3 * max(1, abs(x)) for x, y in zip(bs[k]["tph"], v["tph"])))}
print("lines whose service changed:", sorted(changed)[:20], len(changed))
new |= changed
print("network rail base %.0f local %.0f whole %.0f" % (b["rail"], loc["rail"], w["rail"]))
rows = []
for z, v in w["cities"].items():
    u, o = loc["cities"].get(z), b["cities"].get(z)
    if u and v["rail"] > 1000:
        rows.append((u["rail"] / v["rail"] - 1, z, v["rail"], (u["rail"] / o["rail"] - 1) if o and o["rail"] else 0, (v["rail"] / o["rail"] - 1) if o and o["rail"] else 0))
rows.sort(key=lambda r: -abs(r[0]))
print("zones' rail local vs whole, worst (local/whole-1, zone, whole rail, local/base-1, whole/base-1):")
for r in rows[:8]:
    print("  %+.2f%% %s %.0f  (edit moved it %+.2f%% local, %+.2f%% whole)" % (100 * r[0], r[1], r[2], 100 * r[3], 100 * r[4]))
ld = []
for k, v in w["lines"].items():
    u = loc["lines"].get(k)
    if u is not None and v > 10000:
        ld.append((v / u - 1 if u else 9, k, v, u))
ld.sort(key=lambda r: -abs(r[0]))
past = [r for r in ld if abs(r[0]) > 0.05 and r[1] not in new]
print("lines of 10,000+ (whole): %d; past 5%% (not the new lines): %d; past 2%%: %d" % (len(ld), len(past), len([r for r in ld if abs(r[0]) > 0.02 and r[1] not in new])))
print("worst:", [("%+.2f%%" % (100 * r[0]), r[1], round(r[2]), round(r[3])) for r in ld[:8]])
for k in sorted(new):
    g = lambda s, f: (s.get(f) or {}).get(k)
    print("new line", k, "base", b["lines"].get(k), "local", loc["lines"].get(k), "whole", w["lines"].get(k), "; longer trips local", g(loc, "longer"), "whole", g(w, "longer"))
