"""For each row and each of its extracts: the share of the unit outline that only that extract's
Geofabrik polygon covers (none of the row's other extracts), and the units mostly in that part.
exonly.py id,id,..."""
import json, pickle, sys
from shapely.geometry import shape
from shapely.ops import unary_union, transform
from pyproj import Transformer
sys.stdout.reconfigure(encoding="utf-8")
TW = r"C:\Users\anita\projects\maps\trainworld"
plan = json.load(open(TW + r"\pipeline\world_plan.json", encoding="utf-8"))["metros"]
units = pickle.load(open(TW + r"\data\raw\worldplan\geoboundaries\CHN_ADM3.t897e.pkl", "rb"))
byid = {u[0]: (u[1], u[2]) for u in units}
tr = Transformer.from_crs("ESRI:54009", "EPSG:4326", always_xy=True)
gj = json.load(open(TW + r"\data\raw\worldplan\geofabrik-index-v1.json", encoding="utf-8"))
gf = {f["properties"]["urls"]["pbf"].split("download.geofabrik.de/")[1].replace("-latest.osm.pbf", ""): shape(f["geometry"]) for f in gj["features"]}
for cid in sys.argv[1].split(","):
    r = next(x for x in plan if x.get("id") == cid)
    us = [(byid[u][0], transform(tr.transform, byid[u][1])) for u in r["units"]["CHN"]]
    ll = unary_union([g for _, g in us])
    print(cid, r["name"], r["extracts"])
    for e in r["extracts"]:
        others = unary_union([gf[o] for o in r["extracts"] if o != e]) if len(r["extracts"]) > 1 else None
        only = ll.intersection(gf[e]) if others is None else ll.intersection(gf[e]).difference(others)
        ins = [f"{n} {g.intersection(gf[e]).area / g.area:.0%}" for n, g in us if g.intersection(gf[e]).area / g.area > 0.3]
        print(f"  only {e}: {only.area / ll.area * 100:.2f}%   units over 30% inside: {ins if len(ins) < 8 else len(ins)}")
    allc = unary_union([gf[e] for e in r["extracts"]])
    print(f"  outside all: {ll.difference(allc).area / ll.area * 100:.3f}%")
