"""T-3240: each market place's importance, the lowest (best) Natural Earth 10m populated-places scalerank
of the points inside its zone's outline (data/packs/<id>.boundary.geojson); writes place_rank.json
{zone: [rank, name of that point, featurecla]}.  python rank_zones.py
"""
import json, os
import shapefile
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
HERE = r"C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\2ba1fd73-4c91-45ba-b838-fe8f908d4b21\scratchpad\longdist"
TW = r"C:\Users\anita\projects\maps\trainworld"
rows = json.load(open(TW + r"\app\src\game\market-rows.json", encoding="utf-8"))
geoms, ids = [], []
for z in rows:
    p = os.path.join(TW, "data", "packs", f"{z}.boundary.geojson")
    if not os.path.exists(p):
        continue
    g = json.load(open(p, encoding="utf-8"))
    feats = g["features"] if g.get("type") == "FeatureCollection" else [g]
    for f in feats:
        geom = shape(f["geometry"] if "geometry" in f else f).buffer(0)
        geoms.append(geom); ids.append(z)
tree = STRtree(geoms)
r = shapefile.Reader(HERE + r"\dl\ne\ne_10m_populated_places_simple.shp", encoding="utf-8")
fields = [x[0] for x in r.fields[1:]]
best = {}
for rec in r.records():
    d = dict(zip(fields, rec))
    pt = Point(d["longitude"], d["latitude"])
    for k in tree.query(pt):
        if geoms[k].contains(pt):
            z = ids[k]
            if z not in best or d["scalerank"] < best[z][0]:
                best[z] = [d["scalerank"], d["name"], d["featurecla"]]
            break
json.dump(best, open(HERE + r"\place_rank.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(len(rows), "rows,", len(set(ids)), "outlines,", len(best), "with a ranked point")
import collections
print(sorted(collections.Counter(v[0] for v in best.values()).items()))
