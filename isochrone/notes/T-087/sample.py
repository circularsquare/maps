"""T-087: per test tile, the 100 m truth grids and the 1 km features the road rules are built from.

    python isochrone/notes/T-087/sample.py <out dir> <tile> [<tile> ...]

Per tile, in <out dir>/<tile>/ (speeds uint8 km/h x 40, 0 = water, as the page's grids):
  f_cap.bin     1200 x 1200  100 m with OSM roads, a road cell walked at the L1 road speed
                (5 km/h x height factor x slope factor of min(slope, 10% grade), as the 1 km rule)
  f_page.bin    the same with the road at the full terrain factor (what the page's L0.1 does now)
  f_plain.bin   100 m without roads
  feat.npz      120 x 120 features: walk, lc, tf, road_kmh, land, bridge, s (walkable share), s1/s2/s4
                (share of path / local / main cells), span (longest road run across the cell, 100 m
                cells 0-10), cross (cell edges a road touches, 0-4), through (1 where a road touches
                opposite edges), ncomp (road components), urban share
  meta.json
"""
import gzip
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "tools", "fine"))
sys.path.insert(0, os.path.join(HERE, "tools", "surface"))
import build as B
import common as C

N = 1200
SURF = os.path.join(HERE, "data", "surface")
BITS = os.path.join(HERE, "data", "roads", "bits")
WALK = 7
GRADE = np.degrees(np.arctan(0.10))
LC = np.zeros(256, np.float32)
for k, v in C.LC_SPEED.items():
    LC[k] = v


def q(kmh):
    return np.where(kmh > 0, np.clip(np.rint(kmh * 40), 1, 255), 0).astype(np.uint8)


def l1(name, lat, lon, scale):
    with rasterio.open(os.path.join(SURF, name)) as src:
        a = src.read(1, window=Window((lon + 180) * 120, (85 - (lat + 1)) * 120, 120, 120))
    return a.astype(np.float32) / scale


def blocks(a):
    return a.reshape(120, 10, 120, 10)


def cell_features(road):
    """span, cross, through, ncomp per 1 km cell from the walkable road mask (1200 x 1200)."""
    span = np.zeros((120, 120), np.float32)
    cross = np.zeros((120, 120), np.float32)
    through = np.zeros((120, 120), np.float32)
    ncomp = np.zeros((120, 120), np.float32)
    st = np.ones((3, 3), int)
    for i in range(120):
        for j in range(120):
            b = road[i * 10:(i + 1) * 10, j * 10:(j + 1) * 10]
            if not b.any():
                continue
            t, l, bo, r = b[0].any(), b[:, 0].any(), b[-1].any(), b[:, -1].any()
            cross[i, j] = int(t) + int(l) + int(bo) + int(r)
            through[i, j] = float((t and bo) or (l and r))
            lab, n = ndimage.label(b, structure=st)
            ncomp[i, j] = n
            best = 0
            for sl in ndimage.find_objects(lab):
                best = max(best, sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
            span[i, j] = best
    return span, cross, through, ncomp


def main(out, tiles):
    for name in tiles:
        lat = int(name[1:3]) * (1 if name[0] == "n" else -1)
        lon = int(name[4:7]) * (1 if name[3] == "e" else -1)
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        box = (lon, lat, lon + 1, lat + 1)
        cls = B.land_classes(box)
        bits = np.frombuffer(gzip.open(os.path.join(BITS, name + ".bits.gz")).read(), np.uint8).reshape(N, N)
        walk = l1("walk_today.tif", lat, lon, 1000)
        lcs = l1("lc_speed.tif", lat, lon, 1000)
        tf = l1("terrain_factor.tif", lat, lon, 10000)
        elev = l1("elev.tif", lat, lon, 1)
        slope = l1("slope.tif", lat, lon, 100)
        elev = np.where(elev < -30000, 0, elev)
        road_kmh = (5.0 * C.elev_factor(elev) * C.slope_factor(np.minimum(slope, GRADE))).astype(np.float32)
        up = lambda a: np.repeat(np.repeat(a, 10, 0), 10, 1)
        tf100, rk100 = up(tf), up(road_kmh)
        water = (cls == 80) | (cls == 200)
        road = (bits & WALK) > 0
        bridge100 = (bits & 16) > 0
        base = LC[cls]
        off = np.where(water, 0, base) * tf100
        # truth: the road cell is the better of its land cover walk and the road speed
        f_cap = np.where(road & ~water, np.maximum(off, rk100), off)
        f_cap = np.where(water & bridge100, rk100, f_cap)
        f_page = np.where(road & ~water, np.maximum(base, 5.0) * tf100, off)
        f_page = np.where(water & bridge100, 5.0 * tf100, f_page)
        q(f_cap).tofile(os.path.join(d, "f_cap.bin"))
        q(f_page).tofile(os.path.join(d, "f_page.bin"))
        q(off).tofile(os.path.join(d, "f_plain.bin"))
        s = blocks(road).mean(axis=(1, 3))
        s1 = blocks((bits & 1) > 0).mean(axis=(1, 3))
        s2 = blocks((bits & 2) > 0).mean(axis=(1, 3))
        s4 = blocks((bits & 4) > 0).mean(axis=(1, 3))
        bridge = blocks(bridge100 & road).any(axis=(1, 3))
        urban = blocks(cls == 50).mean(axis=(1, 3))
        span, cross, through, ncomp = cell_features(road)
        np.savez_compressed(os.path.join(d, "feat.npz"), walk=walk, lc=lcs, tf=tf, road_kmh=road_kmh, bridge=bridge,
                            s=s, s1=s1, s2=s2, s4=s4, span=span, cross=cross, through=through, ncomp=ncomp,
                            urban=urban)
        land = walk > 0
        meta = {"name": name, "lat": lat, "lon": lon, "s_mean": float(s[land].mean()),
                "cells_with_road": float((s[land] > 0).mean()), "land": float(land.mean()),
                "tf": float(tf[land].mean())}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
        print(name, meta, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
