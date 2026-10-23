"""Rows whose origin lies outside their own pack boundary, or in another's, or within 1 km of the edge:
originpack.py id,id   (default: wave_ids.txt)"""
import glob, json, os, sys
import shapely
from shapely.geometry import shape, Point
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
TW = r"C:\Users\anita\projects\maps\trainworld"
plan = {r["id"]: r for r in json.load(open(TW + r"\pipeline\world_plan.json", encoding="utf-8"))["metros"] if r.get("id")}
ids = (sys.argv[1] if len(sys.argv) > 1 else open(os.path.join(HERE, "wave_ids.txt")).read().strip()).split(",")
bounds = {}
for p in glob.glob(TW + r"\data\packs\*.boundary.geojson"):
    cid = os.path.basename(p).split(".")[0]
    r = plan.get(cid)
    if not r or r.get("iso2") != "CN":
        continue
    gj = json.load(open(p, encoding="utf-8"))
    g = shapely.make_valid(shapely.union_all([shapely.make_valid(shape(f["geometry"])) for f in gj["features"]]))
    bounds[cid] = shapely.union_all([p for p in shapely.get_parts(g) if p.geom_type in ("Polygon", "MultiPolygon")])
n = 0
for cid in ids:
    o = Point(*plan[cid]["origin"])
    own = bounds.get(cid)
    if own is None:
        print(cid, "no boundary")
        continue
    n += 1
    hold = [k for k, g in bounds.items() if g.contains(o)]
    edge = shapely.boundary(own).distance(o) * 111
    if hold != [cid] or edge < 1:
        print(cid, plan[cid]["name"], plan[cid]["origin"], "in", hold, f"edge {edge:.2f} km")
print(n, "checked")
