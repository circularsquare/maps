"""T-1150 rebuild, a dry check: each Chinese metro's 2020 township census inside its outline against
its plan row's GHSL people, the gate's pop_vs_plan (fail outside 0.7-1.5, warn outside 0.85-1.2).
Read-only.   python trainworld/notes/t1150/census_vs_plan.py [ids]"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys, json
sys.stdout.reconfigure(encoding="utf-8")
TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(TW, "pipeline"))
from shapely import union_all, make_valid
from shapely.geometry import shape
import china

with open(os.path.join(TW, "pipeline", "world_plan.json"), encoding="utf-8") as f:
    rows = [r for r in json.load(f)["metros"] if r.get("iso2") == "CN" and r.get("module") == "china"]
want = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
out = []
for r in rows:
    if want and r["id"] not in want:
        continue
    with open(os.path.join(TW, "data", "packs", f"{r['id']}.boundary.geojson")) as f:
        bnd = union_all([make_valid(shape(ft["geometry"])) for ft in json.load(f)["features"]])
    ts = china.townships(bnd)
    census = sum(t[2] * t[3] for t in ts)
    v = census / r["people"]
    flag = "FAIL" if not 0.7 <= v <= 1.5 else "warn" if not 0.85 <= v <= 1.2 else ""
    out.append((v, r["id"], census, r["people"], flag))
    print(f"{r['id']} {r['name']}: census {census:,.0f}, plan {r['people']:,.0f}, {v:.2f} {flag}", flush=True)
out.sort()
print("lowest:", ", ".join(f"{i} {v:.2f}" for v, i, *_ in out[:8]))
print("highest:", ", ".join(f"{i} {v:.2f}" for v, i, *_ in out[-5:]))
print(f"fail {sum(o[4] == 'FAIL' for o in out)}, warn {sum(o[4] == 'warn' for o in out)} of {len(out)}")
