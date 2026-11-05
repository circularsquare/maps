"""T-010: why MAP's L1 is far faster than our surface under the same any-road rule.

    python isochrone/notes/T-010/why_any.py <dir of sample.py>

Per tile, over land cells both grids have: mean time per km (h) of MAP's L1 and of our 'any'
grid; the share of cells MAP has at 5 km/h or more, and the share our 'any' grid has a road in;
the mean terrain factor on our road cells; MAP's speed on cells where we see no road.
"""
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.windows import Window

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
d0 = sys.argv[1]
print("%-14s %8s %8s %8s %8s %8s %8s %8s" % ("tile", "t map", "t any", "map>=5", "road", "tf road", "map noroad", "any noroad"))
for name in sorted(os.listdir(d0)):
    d = os.path.join(d0, name)
    meta = json.load(open(os.path.join(d, "meta.json")))
    lat, lon = meta["lat"], meta["lon"]
    g = {k: np.fromfile(os.path.join(d, "l1_%s.bin" % k), np.uint8).astype(np.float32) / 40 for k in ("map", "our", "any")}
    with rasterio.open(os.path.join(HERE, "data", "surface", "terrain_factor.tif")) as src:
        tf = src.read(1, window=Window((lon + 180) * 120, (85 - (lat + 1)) * 120, 120, 120)).astype(np.float32).ravel() / 10000
    land = (g["map"] > 0) & (g["our"] > 0)
    road = land & (g["any"] > g["our"])
    nor = land & ~road
    print("%-14s %8.3f %8.3f %8.2f %8.2f %8.2f %8.2f %8.2f" % (
        meta["label"], (1 / g["map"][land]).mean(), (1 / g["any"][land]).mean(),
        (g["map"][land] >= 4.99).mean(), road.sum() / land.sum(), tf[road].mean(),
        g["map"][nor].mean() if nor.any() else 0, g["any"][nor].mean() if nor.any() else 0))
