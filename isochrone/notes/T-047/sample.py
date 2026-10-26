"""T-047: speed grids for the sample tiles, for compare.mjs. One folder per tile in OUT (argument).

    python isochrone/notes/T-047/sample.py <out dir>

Per tile (uint8 km/h x 40, 0 = water, as the page's grids):
  l1_our.bin    120 x 120   our off-road surface (data/surface/walk_today.tif), no roads
  l1_map.bin    120 x 120   the MAP L1 tiles the page shows (roads in)
  f_plain.bin   1200 x 1200 LC100 classes x the L1 terrain factor (stage 1 without roads)
  f_glo.bin     1200 x 1200 LC100 classes x MAP's elevation and slope factors from GLO-90
  f_road.bin, f_road_glo.bin   the same with the OSM roads, where data/fine/ has the tile
"""
import gzip
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import rasterio

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "tools", "fine"))
sys.path.insert(0, os.path.join(HERE, "tools", "surface"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build as B
import common as C
from tiles import TILES

N = 1200
LC = np.zeros(256, np.float32)
for k, v in C.LC_SPEED.items():
    LC[k] = v
GLO = os.path.join(HERE, "data", "raw", "glo90")


def q(kmh):
    return np.where(kmh > 0, np.clip(np.rint(kmh * 40), 1, 255), 0).astype(np.uint8)


def fine_kmh(cls, roads):
    kmh = LC[cls].copy()
    water = (cls == 80) | (cls == 200)
    if roads is not None:
        walk = (roads & 7) > 0
        kmh = np.where(~water & walk, np.maximum(kmh, 5.0), kmh)
        kmh = np.where(water & ((roads & 16) > 0), 5.0, kmh)
    else:
        kmh = np.where(water, 0, kmh)
    return kmh


def glo_factor(lat, lon, cls, roads):
    n = "Copernicus_DSM_COG_30_%s%02d_00_%s%03d_00_DEM" % ("N" if lat >= 0 else "S", abs(lat), "E" if lon >= 0 else "W", abs(lon))
    path = os.path.join(GLO, n + ".tif")
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        t = src.transform
    x = lon + (np.arange(N) + 0.5) / N
    y = lat + 1 - (np.arange(N) + 0.5) / N
    cols = np.clip(np.floor((x - t.c) / t.a).astype(int), 0, a.shape[1] - 1)
    rows = np.clip(np.floor((y - t.f) / t.e).astype(int), 0, a.shape[0] - 1)
    el = a[np.ix_(rows, cols)]
    dy = 111195.0 / N
    dx = dy * np.cos(np.radians(y))[:, None]
    gx = np.zeros_like(el); gy = np.zeros_like(el)
    gx[:, 1:-1] = (el[:, 2:] - el[:, :-2]) / (2 * dx)
    gy[1:-1, :] = (el[2:, :] - el[:-2, :]) / (2 * dy)
    grade = np.sqrt(gx * gx + gy * gy)
    flat = cls == 50
    if roads is not None:
        flat |= (roads & 7) > 0
    grade = np.where(flat, np.minimum(grade, 0.10), grade)   # streets are graded (docs/fine.md)
    slope = np.degrees(np.arctan(grade))
    return C.elev_factor(np.maximum(el, 0)) * C.slope_factor(slope)


def l1_map(lat, lon):
    r0 = (85 - (lat + 1)) * 120
    c0 = (lon + 180) * 120
    tr, tc = r0 // 600, c0 // 600
    p = os.path.join(HERE, "data", "tiles", "l1", "%d_%d.bin.gz" % (tr, tc))
    if not os.path.exists(p):
        return np.zeros((120, 120), np.uint8)
    t = np.frombuffer(gzip.open(p).read(), np.uint8)[:600 * 600].reshape(600, 600)
    return t[r0 - tr * 600:r0 - tr * 600 + 120, c0 - tc * 600:c0 - tc * 600 + 120].copy()


def l1_our(lat, lon):
    with rasterio.open(os.path.join(HERE, "data", "surface", "walk_today.tif")) as src:
        a = src.read(1, window=rasterio.windows.Window((lon + 180) * 120, (85 - (lat + 1)) * 120, 120, 120))
    return q(a.astype(np.float32) / 1000)


def main(out):
    for lat, lon, label, kind in TILES:
        d = os.path.join(out, B.tile_name(lat, lon))
        os.makedirs(d, exist_ok=True)
        box = (lon, lat, lon + 1, lat + 1)
        cls = B.land_classes(box)
        tf = np.repeat(np.repeat(B.terrain(box).astype(np.float32) / B.TF_SCALE, 10, 0), 10, 1)
        name = B.tile_name(lat, lon)
        roads = None
        rp = os.path.join(B.OUT, name + ".roads.bin.gz")
        if os.path.exists(rp):
            roads = np.frombuffer(gzip.open(rp).read(), np.uint8).reshape(N, N)
        have = ["l1_our", "l1_map", "f_plain"]
        l1_our(lat, lon).tofile(os.path.join(d, "l1_our.bin"))
        l1_map(lat, lon).tofile(os.path.join(d, "l1_map.bin"))
        q(fine_kmh(cls, None) * tf).tofile(os.path.join(d, "f_plain.bin"))
        g = glo_factor(lat, lon, cls, None)
        if g is not None:
            q(fine_kmh(cls, None) * g).tofile(os.path.join(d, "f_glo.bin"))
            have.append("f_glo")
        if roads is not None:
            q(fine_kmh(cls, roads) * tf).tofile(os.path.join(d, "f_road.bin"))
            have.append("f_road")
            g = glo_factor(lat, lon, cls, roads)
            if g is not None:
                q(fine_kmh(cls, roads) * g).tofile(os.path.join(d, "f_road_glo.bin"))
                have.append("f_road_glo")
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"lat": lat, "lon": lon, "label": label, "kind": kind, "have": have}, f)
        print(label, have, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
