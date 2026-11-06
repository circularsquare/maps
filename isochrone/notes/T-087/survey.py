"""T-087: candidate test tiles: road share, land share, cropland share, terrain factor.
    python isochrone/notes/T-087/survey.py n41w094 n42w093 ...
"""
import gzip
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import rasterio
from rasterio.windows import Window

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SURF = os.path.join(HERE, "data", "surface")
BITS = os.path.join(HERE, "data", "roads", "bits")


def tile_ll(name):
    lat = int(name[1:3]) * (1 if name[0] == "n" else -1)
    lon = int(name[4:7]) * (1 if name[3] == "e" else -1)
    return lat, lon


def l1(fn, lat, lon, scale):
    with rasterio.open(os.path.join(SURF, fn)) as src:
        a = src.read(1, window=Window((lon + 180) * 120, (85 - (lat + 1)) * 120, 120, 120))
    return a.astype(np.float32) / scale


for name in sys.argv[1:]:
    lat, lon = tile_ll(name)
    p = os.path.join(BITS, name + ".bits.gz")
    if not os.path.exists(p):
        print(name, "no bits")
        continue
    bits = np.frombuffer(gzip.open(p).read(), np.uint8).reshape(1200, 1200)
    road = (bits & 7) > 0
    s = road.reshape(120, 10, 120, 10).mean(axis=(1, 3))
    walk = l1("walk_today.tif", lat, lon, 1000)
    tf = l1("terrain_factor.tif", lat, lon, 10000)
    sl = l1("slope.tif", lat, lon, 100)
    lc = l1("lc_speed.tif", lat, lon, 1000)
    land = walk > 0
    print("%s land %.2f  s_mean %.3f  cells_with_road %.2f  s_p50 %.3f  lc_mean %.2f  walk_mean %.2f  tf %.2f  slope_med %.1f" % (
        name, land.mean(), s[land].mean() if land.any() else 0, (s[land] > 0).mean() if land.any() else 0,
        np.median(s[land]) if land.any() else 0, lc[land].mean() if land.any() else 0,
        walk[land].mean() if land.any() else 0, tf[land].mean() if land.any() else 0,
        np.median(sl[land]) if land.any() else 0))
