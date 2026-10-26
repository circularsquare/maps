"""Fine level (L0.1, 1/1200 degree) tiles for a region (docs/fine.md).

    python isochrone/tools/fine/build.py <region> [<region> ...]
    python isochrone/tools/fine/build.py index        (rewrite data/fine/index.json only)

Per 1 x 1 degree tile, named by its south-west corner (n40w074), into data/fine/:
  <name>.land.bin.gz   1200 x 1200 LC100 class (nearest pixel to the cell centre), then
                       120 x 120 terrain factor of the L1 cells x 250 (data/surface/terrain_factor.tif)
  <name>.roads.bin.gz  1200 x 1200 road bits (ROAD_* below) from the Geofabrik extracts in
                       data/raw/osm/; a way marks every cell it passes through
The worker turns these into speeds (js/tiles.js). One core; a state-sized extract ~2-4 min.
"""
import gzip
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import rasterio
from rasterio.windows import Window

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data", "fine")
LC100 = os.path.join(RAW, "lc100", "cgls-lc100-2019-cog.tif")
TERRAIN = os.path.join(HERE, "data", "surface", "terrain_factor.tif")
OSM = os.path.join(RAW, "osm")

N = 1200            # cells a tile side
L1N = 120           # L1 cells a tile side
TF_SCALE = 250      # terrain factor byte = factor x 250

# regions: tile box (west, south, east, north, whole degrees) and the extracts covering it
REGIONS = {
    "newyork": ((-75, 40, -73, 42), ["new-york", "new-jersey", "connecticut", "pennsylvania"]),
    "cluj": ((23, 46, 25, 48), ["romania"]),
}

# road bits (docs/fine.md "What a tile holds")
ROAD_PATH, ROAD_LOCAL, ROAD_MAIN, ROAD_MOTORWAY, ROAD_BRIDGE = 1, 2, 4, 8, 16
KIND = {}
for k in ("footway", "path", "steps", "track", "bridleway", "pedestrian", "cycleway"):
    KIND[k] = ROAD_PATH
for k in ("residential", "service", "unclassified", "living_street", "tertiary", "tertiary_link", "road"):
    KIND[k] = ROAD_LOCAL
for k in ("secondary", "secondary_link", "primary", "primary_link", "trunk", "trunk_link"):
    KIND[k] = ROAD_MAIN
for k in ("motorway", "motorway_link"):
    KIND[k] = ROAD_MOTORWAY
NO = ("no", "private", "use_sidepath", "discouraged")


def tile_name(lat, lon):
    return "%s%02d%s%03d" % ("n" if lat >= 0 else "s", abs(lat), "e" if lon >= 0 else "w", abs(lon))


def way_code(tags):
    """Road bits of a way, or 0 to leave it out."""
    kind = KIND.get(tags.get("highway"))
    if not kind:
        return 0
    tun = tags.get("tunnel")
    if tun and tun not in ("no", "building_passage"):
        return 0
    if kind != ROAD_MOTORWAY:
        foot = tags.get("foot")
        if foot in NO or (tags.get("access") in ("no",) and foot not in ("yes", "designated", "permissive")):
            kind = ROAD_MOTORWAY          # drawn, not walkable
    br = tags.get("bridge")
    if br and br != "no" and kind != ROAD_MOTORWAY:
        kind |= ROAD_BRIDGE
    return kind


def read_ways(box, extracts):
    """All kept highway ways touching box: (lons, lats, way start flags, codes per node)."""
    import osmium
    west, south, east, north = box
    pad = 0.01
    lons, lats, codes = [], [], []
    nways = 0
    for name in extracts:
        path = os.path.join(OSM, name + ".osm.pbf")
        t0 = time.time()
        fp = (osmium.FileProcessor(path)
              .with_locations()
              .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
              .with_filter(osmium.filter.KeyFilter("highway")))
        kept = 0
        for w in fp:
            code = way_code(w.tags)
            if not code:
                continue
            try:
                xs = [n.lon for n in w.nodes]
                ys = [n.lat for n in w.nodes]
            except osmium.InvalidLocationError:
                continue
            if len(xs) < 2:
                continue
            if max(xs) < west - pad or min(xs) > east + pad or max(ys) < south - pad or min(ys) > north + pad:
                continue
            lons.append(np.array(xs))
            lats.append(np.array(ys))
            codes.append(code)
            kept += 1
        nways += kept
        print("  %s: %d ways in the box, %.0f s" % (name, kept, time.time() - t0), flush=True)
    return lons, lats, codes, nways


def rasterise(box, lons, lats, codes):
    """Road bits on the region's cell grid; every cell a segment passes through."""
    west, south, east, north = box
    W, H = (east - west) * N, (north - south) * N
    out = np.zeros(H * W, np.uint8)
    if not lons:
        return out.reshape(H, W)
    sizes = np.array([len(a) for a in lons])
    x = (np.concatenate(lons) - west) * N
    y = (north - np.concatenate(lats)) * N
    code = np.repeat(np.array(codes, np.uint8), sizes)
    last = np.cumsum(sizes) - 1                    # last node of each way: no segment from it
    seg = np.ones(len(x), bool)
    seg[last] = False
    i0 = np.nonzero(seg)[0]
    x0, y0, x1, y1, c = x[i0], y[i0], x[i0 + 1], y[i0 + 1], code[i0]
    # samples a third of a cell apart along each segment
    steps = np.ceil(np.maximum(np.abs(x1 - x0), np.abs(y1 - y0)) * 3).astype(np.int64) + 1
    CH = 2_000_000
    s0 = 0
    while s0 < len(steps):
        # chunks of about CH samples
        cs = np.cumsum(steps[s0:])
        s1 = s0 + max(1, int(np.searchsorted(cs, CH)))
        st = steps[s0:s1]
        rep = np.repeat(np.arange(s0, s1), st)
        start = np.repeat(np.cumsum(st) - st, st)
        f = (np.arange(len(rep)) - start) / np.maximum(1, np.repeat(st, st) - 1)
        px = np.floor(x0[rep] + (x1[rep] - x0[rep]) * f).astype(np.int64)
        py = np.floor(y0[rep] + (y1[rep] - y0[rep]) * f).astype(np.int64)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        idx = py[ok] * W + px[ok]
        cc = c[rep][ok]
        for v in np.unique(cc):
            sel = idx[cc == v]
            out[sel] |= v
        s0 = s1
    return out.reshape(H, W)


def land_classes(box):
    """LC100 class at each cell centre of the region, (H, W) uint8."""
    west, south, east, north = box
    W, H = (east - west) * N, (north - south) * N
    with rasterio.open(LC100) as src:
        inv = ~src.transform
        lon = west + (np.arange(W) + 0.5) / N
        lat = north - (np.arange(H) + 0.5) / N
        cols = np.floor((lon - src.transform.c) / src.transform.a).astype(np.int64)
        rows = np.floor((lat - src.transform.f) / src.transform.e).astype(np.int64)
        c0, c1, r0, r1 = cols.min(), cols.max() + 1, rows.min(), rows.max() + 1
        a = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
    return a[np.ix_(rows - r0, cols - c0)]


def terrain(box):
    """L1 terrain factor x 250 for the region, (H/10, W/10) uint8."""
    west, south, east, north = box
    with rasterio.open(TERRAIN) as src:
        c0 = int(round((west - src.transform.c) / src.transform.a))
        r0 = int(round((north - src.transform.f) / src.transform.e))
        a = src.read(1, window=Window(c0, r0, (east - west) * L1N, (north - south) * L1N)).astype(np.float32)
    return np.clip(np.rint(a / 10000 * TF_SCALE), 1, 255).astype(np.uint8)


def write_gz(path, *arrays):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb", compresslevel=9) as f:
        for a in arrays:
            f.write(np.ascontiguousarray(a).tobytes())
    os.replace(tmp, path)


def build(region):
    box, extracts = REGIONS[region]
    west, south, east, north = box
    t0 = time.time()
    print("%s: land cover" % region, flush=True)
    cls = land_classes(box)
    tf = terrain(box)
    print("%s: roads from %s" % (region, ", ".join(extracts)), flush=True)
    lons, lats, codes, nways = read_ways(box, extracts)
    roads = rasterise(box, lons, lats, codes)
    del lons, lats, codes
    os.makedirs(OUT, exist_ok=True)
    for lat in range(south, north):
        for lon in range(west, east):
            r = (north - 1 - lat) * N
            c = (lon - west) * N
            name = tile_name(lat, lon)
            write_gz(os.path.join(OUT, name + ".land.bin.gz"), cls[r:r + N, c:c + N],
                     tf[r // 10:(r + N) // 10, c // 10:(c + N) // 10])
            write_gz(os.path.join(OUT, name + ".roads.bin.gz"), roads[r:r + N, c:c + N])
            sz = sum(os.path.getsize(os.path.join(OUT, name + e)) for e in (".land.bin.gz", ".roads.bin.gz"))
            print("  %s %.2f MB" % (name, sz / 1e6), flush=True)
    print("%s: %d ways, %.0f s" % (region, nways, time.time() - t0), flush=True)


def index():
    names = sorted(f[:-12] for f in os.listdir(OUT) if f.endswith(".land.bin.gz"))
    roads = [n for n in names if os.path.exists(os.path.join(OUT, n + ".roads.bin.gz"))]
    idx = {"tile": N, "terrain": L1N, "tfScale": TF_SCALE, "tiles": names, "roads": roads}
    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(idx, f)
    print("index: %d tiles" % len(names))


if __name__ == "__main__":
    args = sys.argv[1:] or ["index"]
    for a in args:
        if a != "index":
            build(a)
    index()
