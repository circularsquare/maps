"""T-1150: the township census inside each Chinese metro's outline against the 2020 census by county
(github.com/leiii/census, MIT, census_county_2010-2020_v1.csv with county polygons), counties
counted whole where most of their area is inside. Read-only.
    python trainworld/notes/t1150/county_check.py <county csv> [ids]"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, csv
sys.stdout.reconfigure(encoding="utf-8")
csv.field_size_limit(10 ** 9)
TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(TW, "pipeline"))
import numpy as np
import shapely
from shapely import union_all, make_valid, from_wkt
from shapely.geometry import shape
import china

with open(sys.argv[1], encoding="utf-8-sig") as f:
    cs = [r for r in csv.DictReader(f) if r["coord"] and r["popu_2020"]]
cg = np.array([make_valid(from_wkt(r["coord"])) for r in cs], dtype=object)
cp = np.array([float(r["popu_2020"]) for r in cs])
tree = shapely.STRtree(cg)
with open(os.path.join(TW, "pipeline", "world_plan.json"), encoding="utf-8") as f:
    rows = [r for r in json.load(f)["metros"] if r.get("iso2") == "CN" and r.get("module") == "china"]
want = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
for r in rows:
    if want and r["id"] not in want:
        continue
    with open(os.path.join(TW, "data", "packs", f"{r['id']}.boundary.geojson")) as f:
        bnd = union_all([make_valid(shape(ft["geometry"])) for ft in json.load(f)["features"]])
    hit = tree.query(bnd, predicate="intersects")
    share = shapely.area(shapely.intersection(cg[hit], bnd)) / np.maximum(shapely.area(cg[hit]), 1e-12)
    inside = hit[share > 0.5]
    county = cp[inside].sum()
    town = sum(t[2] * t[3] for t in china.townships(bnd))
    names = ", ".join(cs[i]["county"] for i in inside[:8])
    print(f"{r['id']} {r['name']}: townships {town:,.0f}, counties {county:,.0f} ({town / max(county, 1):.2f}), "
          f"plan GHSL {r['people']:,.0f}; {len(inside)} counties: {names}", flush=True)
