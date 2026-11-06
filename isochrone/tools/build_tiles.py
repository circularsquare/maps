"""The page's walking levels, L1, L3 and L10, from our surface and the roads layer.

    python isochrone/tools/build_tiles.py [workers]       # into data/tiles_next/, ~6 min on 3
    python isochrone/tools/build_tiles.py swap <backup>    # serve it; what was served moves to data/<backup>/

Inputs: data/surface/walk_today.tif, lc_water.tif, elev.tif, slope.tif, terrain_factor.tif
(docs/surface.md) and data/roads/l1/<tile>.bin.gz (docs/roads.md). Rules: docs/resolution.md
(levels, water, coarse cells) and docs/roads.md ("The page part").

Outputs, under data/tiles_next/ until swapped in:
  tiles/l1/<tr>_<tc>.bin.gz   L1, 600 x 600 cells per 5 degree tile: off-road speed, uint8 km/h x 40, 0 water
  tiles/l3/<tr>_<tc>.bin.gz   L3, 200 x 200
  tiles/r1/, tiles/r3/        the road byte of each cell, same layout
  tiles/index.json            grid sizes and the tiles that exist, per level, speeds and roads
  grid.bin.gz, grid.json      L10, the whole world (4320 x 1740), off-road speed
  grid_roads.bin.gz           L10 road bytes
A speed tile is written where it has land, a road tile where it has a road byte.
"""
import gzip
import json
import os
import shutil
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import rasterio
from rasterio.windows import Window

from smallwater import SmallWater, STRIP

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "surface"))
import common as SC                 # noqa: E402  MAP's height and slope factors (docs/surface.md)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
SURF = os.path.join(DATA, "surface")
ROADS = os.path.join(DATA, "roads", "l1")
NEXT = os.path.join(DATA, "tiles_next")

W, H = 43200, 17400                 # L1: 1/120 degree, 85N to 60S
TILE_DEG = 5
TX, TY = W // STRIP, H // STRIP     # 72 x 29 tiles of 5 degrees
SIZE = {1: 600, 3: 200}             # cells per tile side
SCALE = 40                          # speed byte = km/h x 40
MIN_LAND = 0.25                     # a coarse cell is land when a quarter of its L1 cells are
SMALL_KMH = 1.0                     # water patches under 5 km2 are waded (T-013)
GAP_KMH = 3.0                       # land LC100 leaves as sea: bare or sparse, a middle guess
GAP_SEA = 0.98                      # ... a pixel LC100 has as all sea
GAP_ELEV = 2                        # ... where GMTED is above this many metres
ROAD_KMH = 5.0                      # road speed before the height and slope factors
ROAD_GRADE = 0.10                   # roads climb a steeper cell at this grade (T-084, docs/roads.md)
ROAD_CAP_DEG = float(np.degrees(np.arctan(ROAD_GRADE)))
ROAD_E = 3                          # effective road share e = min(1, 3 x share) (docs/roads.md)
RB = 250                            # road byte scale
WALK_BITS, BRIDGE = 7, 16           # road bits: path, local, main; a walkable bridge


# ---------------------------------------------------------------- rules

def coarse(hpk, n, f):
    """Each f x f block of L1 cells: summed hours per km and count of the cells n marks. The
    coarse speed is count / sum (mean time per metre), land where the count is a quarter of the
    block (docs/resolution.md). T-081 would change this."""
    rows, w = hpk.shape
    hs = np.where(n, hpk, 0).reshape(rows // f, f, w // f, f).sum(axis=(1, 3), dtype=np.float64)
    ns = n.reshape(rows // f, f, w // f, f).sum(axis=(1, 3), dtype=np.int32)
    return hs, ns, ns >= MIN_LAND * f * f


def speed_byte(kmh, land):
    v = np.clip(np.rint(kmh * SCALE), 1, 255)
    return np.where(land, v, 0).astype(np.uint8)


def road_byte(ratio, land, bridge_kmh, bridge):
    """docs/roads.md: on land, time with roads over time without x 250 (0 where that is 1); on
    water, a walkable bridge's speed over 5 km/h x 250 (1-250); 0 no road."""
    b = np.clip(np.rint(ratio * RB), 1, RB)
    b[b == RB] = 0
    w = np.clip(np.rint(bridge_kmh / ROAD_KMH * RB), 1, RB)
    return np.where(land, b, np.where(bridge, w, 0)).astype(np.uint8)


def tile_name(lat, lon):
    return "%s%02d%s%03d" % ("n" if lat >= 0 else "s", abs(lat), "e" if lon >= 0 else "w", abs(lon))


def roads_strip(i, have):
    """Road counts (0-100) and OR-ed bits of the 1 km cells of strip i, from the 1 degree tiles."""
    count = np.zeros((STRIP, W), np.uint8)
    bits = np.zeros((STRIP, W), np.uint8)
    for j in range(STRIP // 120):
        lat = 84 - TILE_DEG * i - j
        for lon in range(-180, 180):
            nm = tile_name(lat, lon)
            if nm not in have:
                continue
            b = np.frombuffer(gzip.open(os.path.join(ROADS, nm + ".bin.gz")).read(), np.uint8)
            sl = (slice(j * 120, (j + 1) * 120), slice((lon + 180) * 120, (lon + 181) * 120))
            count[sl] = b[:14400].reshape(120, 120)
            bits[sl] = b[14400:].reshape(120, 120)
    return count, bits


# ---------------------------------------------------------------- one strip of 5 degrees

def write_gz(path, a):
    data = gzip.compress(np.ascontiguousarray(a).tobytes(), compresslevel=6)
    with open(path + ".tmp", "wb") as fh:
        fh.write(data)
    os.replace(path + ".tmp", path)
    return len(data)


def do_strip(args):
    i, small_bits, have = args
    t0 = time.time()
    r0 = i * STRIP
    win = Window(0, r0, W, STRIP)
    rd = lambda name, band=1: rasterio.open(os.path.join(SURF, name + ".tif")).read(band, window=win)
    walk = rd("walk_today")
    tf = rd("terrain_factor").astype(np.float32) / 10000
    sea = rd("lc_water", 2)
    elev = rd("elev")
    slope = rd("slope").astype(np.float32) / 100
    small = np.unpackbits(small_bits, count=STRIP * W).reshape(STRIP, W).astype(bool)
    count, bits = roads_strip(i, have)

    land = walk > 0
    gap = ~land & ~small & (sea >= GAP_SEA * 250) & (elev > GAP_ELEV)
    del sea
    n0 = land | small | gap
    kmh0 = np.where(land, walk / np.float32(1000), np.float32(SMALL_KMH))
    kmh0[gap] = GAP_KMH * tf[gap]
    np.maximum(kmh0, np.float32(1 / SCALE), out=kmh0)   # the slowest a speed byte says; coarse means agree with L1
    del walk, tf
    # road speed: 5 km/h x the height factor x the slope factor of the grade a road takes, at most ROAD_GRADE
    road_kmh = (ROAD_KMH * SC.elev_factor(np.where(elev == -32768, 0, elev).astype(np.float32))
                * SC.slope_factor(np.minimum(slope, ROAD_CAP_DEG))).astype(np.float32)
    del elev, slope
    e = np.minimum(np.float32(1), ROAD_E * count.astype(np.float32) / 100)
    ratio = np.where(n0, 1 - e + e * np.minimum(np.float32(1), kmh0 / road_kmh), np.float32(1))
    del e, count
    bridge = ~n0 & (bits & BRIDGE > 0)
    kmh_b = np.maximum(road_kmh, np.float32(0.05))
    del bits, road_kmh
    hpk0 = np.where(n0, 1 / np.where(n0, kmh0, 1), 0).astype(np.float32)
    hpk1 = np.where(n0, hpk0 * ratio, np.where(bridge, 1 / kmh_b, 0)).astype(np.float32)
    n1 = n0 | bridge

    out = {1: (speed_byte(kmh0, n0), road_byte(ratio, n0, kmh_b, bridge))}
    del kmh0
    stats = {"land": int(n0.sum()), "small": int(small.sum()), "gap": int(gap.sum()), "bridge": int(bridge.sum()),
             "road": int((out[1][1] > 0).sum())}
    for f in (3, 10):
        h0, c0, l0 = coarse(hpk0, n0, f)
        h1, c1, l1 = coarse(hpk1, n1, f)
        with np.errstate(divide="ignore", invalid="ignore"):
            kmh_c0 = np.where(h0 > 0, c0 / h0, 0)
            kmh_c1 = np.where(h1 > 0, c1 / h1, 0)
            rat = np.where(l0 & (kmh_c1 > 0), kmh_c0 / np.where(kmh_c1 > 0, kmh_c1, 1), 1)
        out[f] = (speed_byte(kmh_c0, l0), road_byte(rat, l0, kmh_c1, ~l0 & l1))

    found = {k: [] for k in ("l1", "l3", "r1", "r3")}
    nbytes = {k: 0 for k in found}
    for f in (1, 3):
        size = SIZE[f]
        for kind, g in (("l", out[f][0]), ("r", out[f][1])):
            key = "%s%d" % (kind, f)
            for tc in range(TX):
                t = g[:, tc * size:(tc + 1) * size]
                if not t.any():
                    continue
                nbytes[key] += write_gz(os.path.join(NEXT, "tiles", key, "%d_%d.bin.gz" % (i, tc)), t)
                found[key].append(i * TX + tc)
    return i, out[10], found, nbytes, stats, time.time() - t0


# ---------------------------------------------------------------- build and swap

def build(workers):
    from multiprocessing import Pool
    t0 = time.time()
    for k in ("l1", "l3", "r1", "r3"):
        os.makedirs(os.path.join(NEXT, "tiles", k), exist_ok=True)
    have = set(f[:-7] for f in os.listdir(ROADS) if f.endswith(".bin.gz"))
    print("road tiles: %d" % len(have), flush=True)
    with rasterio.open(os.path.join(SURF, "walk_today.tif")) as ds:
        sw = SmallWater(ds, log=lambda *a: None, is_water=lambda a: a == 0)
        print("small water: %d patches (%.0f km2) are waded (%.0f s)"
              % (sw.small_components, sw.small_area_km2, time.time() - t0), flush=True)
        jobs = []
        for i in range(TY):
            a = ds.read(1, window=Window(0, i * STRIP, W, STRIP))
            jobs.append((i, np.packbits(sw.mask(i, a)), have))
    g10 = np.zeros((H // 10, W // 10), np.uint8)
    r10 = np.zeros((H // 10, W // 10), np.uint8)
    found = {k: [] for k in ("l1", "l3", "r1", "r3")}
    nbytes = {k: 0 for k in found}
    tot = {}
    with Pool(workers) as pool:
        for i, (g, r), fd, nb, st, sec in pool.imap_unordered(do_strip, jobs):
            g10[i * 60:(i + 1) * 60] = g
            r10[i * 60:(i + 1) * 60] = r
            for k in found:
                found[k] += fd[k]
                nbytes[k] += nb[k]
            for k, v in st.items():
                tot[k] = tot.get(k, 0) + v
            print("strip %d done in %.0f s  (%.0f s)  %s" % (i, sec, time.time() - t0, st), flush=True)

    index = {"west": -180.0, "east": 180.0, "north": 85.0, "south": -60.0, "scale": SCALE, "roadScale": RB,
             "tileDeg": TILE_DEG, "tilesX": TX, "tilesY": TY, "levels": {}}
    for f in (1, 3):
        index["levels"][str(f)] = {"factor": f, "width": W // f, "height": H // f, "tile": SIZE[f],
                                   "path": "l%d" % f, "tiles": sorted(found["l%d" % f]),
                                   "roads": {"path": "r%d" % f, "tiles": sorted(found["r%d" % f])}}
    with open(os.path.join(NEXT, "tiles", "index.json"), "w") as fh:
        json.dump(index, fh, separators=(",", ":"))
    n10 = write_gz(os.path.join(NEXT, "grid.bin.gz"), g10)
    nr10 = write_gz(os.path.join(NEXT, "grid_roads.bin.gz"), r10)
    meta = {"width": W // 10, "height": H // 10, "west": -180.0, "east": 180.0, "north": 85.0,
            "south": -60.0, "scale": SCALE, "factor": 10, "roads": "grid_roads.bin.gz", "roadScale": RB}
    with open(os.path.join(NEXT, "grid.json"), "w") as fh:
        json.dump(meta, fh)
    for k in sorted(found):
        print("%s: %d tiles, %.1f MB" % (k, len(found[k]), nbytes[k] / 1e6))
    print("L10: %.2f MB speeds, %.2f MB roads; land share %.3f" % (n10 / 1e6, nr10 / 1e6, (g10 > 0).mean()))
    print("L1 cells: %s" % tot)
    print("done in %.0f s" % (time.time() - t0))


def swap(backup):
    """Serve data/tiles_next/: what is served now moves to data/<backup>/ (kept to compare and
    roll back). Renames only, so no file is ever served half written."""
    b = os.path.join(DATA, backup)
    if os.path.exists(b):
        sys.exit("%s exists" % b)
    os.makedirs(b)
    for name in ("grid.bin.gz", "grid.json", "grid_roads.bin.gz"):
        if os.path.exists(os.path.join(DATA, name)):
            shutil.copy2(os.path.join(DATA, name), os.path.join(b, name))
    os.replace(os.path.join(NEXT, "grid_roads.bin.gz"), os.path.join(DATA, "grid_roads.bin.gz"))
    os.replace(os.path.join(DATA, "tiles"), os.path.join(b, "tiles"))
    os.replace(os.path.join(NEXT, "tiles"), os.path.join(DATA, "tiles"))
    os.replace(os.path.join(NEXT, "grid.bin.gz"), os.path.join(DATA, "grid.bin.gz"))
    os.replace(os.path.join(NEXT, "grid.json"), os.path.join(DATA, "grid.json"))
    os.rmdir(NEXT)
    print("served; the previous build is in %s" % b)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "swap":
        swap(sys.argv[2])
    else:
        build(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
