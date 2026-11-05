"""T-010: 1 km speed grids with roads under each candidate rule, beside the 100 m answer.

    python isochrone/notes/T-010/sample.py <out dir>

Per built tile (the eight 100 m tiles of Cluj and New York), uint8 km/h x 40, 0 = water:
  l1_map.bin        MAP's L1 tiles (the page today)
  l1_our.bin        our surface, no roads (walk_today.tif)
  l1_<rule>.bin     our surface with the 1 km road layer (data/roads/bits) under <rule>
  f_road.bin        100 m with OSM roads (the T-047 'road' pair's 100 m side), from the same
                    bits as the 1 km layer so both read one OSM date
  f_plain.bin       100 m without roads (for the floor: our surface against it)
  urban.bin         1200 x 1200, 1 where the 100 m cell is LC100 urban (50)
Also prints the agreement of data/roads/bits with data/fine/*.roads (a later OSM date).
"""
import gzip
import importlib.util
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import rasterio
from rasterio.windows import Window

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("t047_sample", os.path.join(HERE, "notes", "T-047", "sample.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
B, LC, q = S.B, S.LC, S.q
N = 1200
SURF = os.path.join(HERE, "data", "surface")
BITS = os.path.join(HERE, "data", "roads", "bits")
WALK = 7
TILES = [(46, 23, "Cluj"), (46, 24, "Mures"), (47, 23, "Maramures"), (47, 24, "Bistrita"),
         (40, -74, "New York"), (40, -75, "Trenton"), (41, -74, "Hudson Valley"), (41, -75, "Poconos")]


def l1(name, lat, lon, scale):
    with rasterio.open(os.path.join(SURF, name)) as src:
        a = src.read(1, window=Window((lon + 180) * 120, (85 - (lat + 1)) * 120, 120, 120))
    return a.astype(np.float32) / scale


def blocks(a):
    return a.reshape(120, 10, 120, 10)


def rules(walk, lc, tf, bits, cls):
    """km/h per L1 cell under each rule; walk = our off-road speed, lc = its land cover part."""
    road = (bits & WALK) > 0
    s = blocks(road).mean(axis=(1, 3))
    bridge = blocks(((bits & 16) > 0) & road).any(axis=(1, 3))
    land = walk > 0
    lc = np.minimum(np.where(lc > 0, lc, 5.0), 5.0)
    vroad = 5.0 * tf
    out = {}

    def blend(e):
        r = e * lc / 5.0 + (1 - e)          # time with roads / time without (t_road / t_off = lc / 5)
        return np.where(land, walk / r, 0)

    def bridges(v):
        return np.where(~land & bridge, vroad, v)

    out["any"] = bridges(np.where(land & (s > 0), np.maximum(walk, vroad), walk))
    for a in (1, 1.5, 2, 3):
        out["share%g" % a] = bridges(blend(np.minimum(1.0, a * s)))
    out["sqrt"] = bridges(blend(np.sqrt(s)))
    out["share2.5"] = bridges(blend(np.minimum(1.0, 2.5 * s)))
    for k in (2, 3, 4):
        out["sat%d" % k] = bridges(blend(1 - (1 - s) ** k))    # about k s for few roads, saturating
    out["share2_nobridge"] = blend(np.minimum(1.0, 2 * s))
    # per 100 m cell: the cell's own land cover time, road cells at 5 km/h
    lcs = LC[cls]
    lnd = lcs > 0
    t0 = np.where(lnd, 1.0 / np.maximum(lcs, 1e-6), 0)
    t1 = np.where(lnd & road, 1.0 / 5.0, t0)
    T0, T1 = blocks(t0).sum(axis=(1, 3)), blocks(t1).sum(axis=(1, 3))
    r = np.where(T0 > 0, T1 / np.maximum(T0, 1e-9), 1.0)
    out["exact"] = bridges(np.where(land, walk / r, 0))
    return out, s


def main(out):
    report = {}
    for lat, lon, label in TILES:
        name = B.tile_name(lat, lon)
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        box = (lon, lat, lon + 1, lat + 1)
        cls = B.land_classes(box)
        tf100 = np.repeat(np.repeat(B.terrain(box).astype(np.float32) / B.TF_SCALE, 10, 0), 10, 1)
        old = np.frombuffer(gzip.open(os.path.join(B.OUT, name + ".roads.bin.gz")).read(), np.uint8).reshape(N, N)
        bits = np.frombuffer(gzip.open(os.path.join(BITS, name + ".bits.gz")).read(), np.uint8).reshape(N, N)
        a, b = (old & WALK) > 0, (bits & WALK) > 0
        agree = {"old": int(a.sum()), "new": int(b.sum()), "both": int((a & b).sum())}
        S.l1_map(lat, lon).tofile(os.path.join(d, "l1_map.bin"))
        walk = l1("walk_today.tif", lat, lon, 1000)
        lc = l1("lc_speed.tif", lat, lon, 1000)
        tf = l1("terrain_factor.tif", lat, lon, 10000)
        q(walk).tofile(os.path.join(d, "l1_our.bin"))
        rs, s = rules(walk, lc, tf, bits, cls)
        for k, v in rs.items():
            q(v).tofile(os.path.join(d, "l1_%s.bin" % k))
        q(S.fine_kmh(cls, bits) * tf100).tofile(os.path.join(d, "f_road.bin"))
        q(S.fine_kmh(cls, None) * tf100).tofile(os.path.join(d, "f_plain.bin"))
        (cls == 50).astype(np.uint8).tofile(os.path.join(d, "urban.bin"))
        land = walk > 0
        meta = {"lat": lat, "lon": lon, "label": label, "l1": ["map", "our"] + list(rs),
                "share": {"land_mean": float(s[land].mean()), "cells_with_road": float((s[land] > 0).mean())},
                "bits_agree": agree}
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f)
        report[label] = meta
        print(label, json.dumps(meta["share"]), agree, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
