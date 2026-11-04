# T-1734: the last whole solve against a fresh solve of the same network, to the bit; and the gaps
# between finals while building.  bits.py out.json
import json, sys
d = json.load(open(sys.argv[1]))
print("load %.0f s; edits %d; error %s; page errors %d" % (d.get("loadSecs", 0), len(d["edits"]), d.get("error"), len(d["errors"])))
fs = d["finals"]
print("finals while running:", [(f["version"], round(f["secs"])) for f in fs])
if d["edits"]:
    a, b = d["edits"][0]["secs"], d["edits"][-1]["secs"]
    inside = [f["secs"] for f in fs if a <= f["secs"]]
    gaps = [y - x for x, y in zip([a] + inside, inside)]
    print("edits from %.0f to %.0f s; finals after the first edit: %d; gaps (s): %s" % (a, b, len(inside), [round(g) for g in gaps]))
vs = [int(v["view"].split(":")[0]) for v in d["views"] if v["view"]]
back = [(x, y) for x, y in zip(vs, vs[1:]) if y < x]
print("views shown: %d; a view older than the one before it: %s" % (len(vs), back[:5]))
if "last" in d and "fresh" in d:
    L, F = d["last"], d["fresh"]
    for key in ("lines", "stations", "cities"):
        a, b = L[key], F[key]
        diff = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
        print(key, len(a), "differ:", len(diff), diff[:5], [(a.get(k), b.get(k)) for k in diff[:3]])
    print("rail", L["rail"], F["rail"], "equal" if L["rail"] == F["rail"] else "DIFFER")
