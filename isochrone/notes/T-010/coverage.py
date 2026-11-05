"""T-010: do the extracts in data/roads/regions.json cover every land tile? And how often do two overlap?

    python isochrone/notes/T-010/coverage.py
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import rasterio
from shapely.geometry import box, shape
from shapely.strtree import STRtree

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rows = json.load(open(os.path.join(HERE, "data", "roads", "regions.json")))
keep = {r["id"] for r in rows}
feats = json.load(open(os.path.join(HERE, "data", "raw", "osm", "index-v1.json"), encoding="utf-8"))["features"]
geo = {f["properties"]["id"]: shape(f["geometry"]).buffer(0) for f in feats if f["properties"]["id"] in keep}
ids = list(geo)
tree = STRtree([geo[i] for i in ids])

# land per 1 degree tile from walk_today (85 N - 60 S), read at 1/10 resolution
with rasterio.open(os.path.join(HERE, "data", "surface", "walk_today.tif")) as src:
    a = src.read(1, out_shape=(src.height // 10, src.width // 10))
land = (a > 0).reshape(145, 12, 360, 12).mean(axis=(1, 3))
miss, multi, n = [], 0, 0
per_tile = []
for r in range(145):
    lat = 84 - r
    for c in range(360):
        if land[r, c] <= 0:
            continue
        lon = c - 180
        b = box(lon, lat, lon + 1, lat + 1)
        hit = [ids[k] for k in tree.query(b) if geo[ids[k]].intersects(b)]
        n += 1
        if not hit:
            miss.append((lat, lon, round(float(land[r, c]), 3)))
        if len(hit) > 1:
            multi += 1
        per_tile.append(len(hit))
print("land tiles %d, none %d, two or more %d, mean extracts a tile %.2f" % (n, len(miss), multi, np.mean(per_tile)))
print("uncovered:", miss[:40])
