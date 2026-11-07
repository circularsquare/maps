"""Fine level (L0.1, 1/1200 degree) tiles (docs/fine.md).

    python isochrone/tools/fine/build.py world [--jobs N] [--resume]
        every tile of data/fine/selection.json (tools/fine/select.py) and the test regions,
        roads from data/roads/bits/ (the T-053 world pass), then index.json
    python isochrone/tools/fine/build.py tiles <name> [<name> ...]
        whole tiles from data/roads/bits/, then index.json
    python isochrone/tools/fine/build.py region <region> [<region> ...]
        a test region with roads read from its Geofabrik extracts in data/raw/osm/, then index.json
    python isochrone/tools/fine/build.py index        (rewrite data/fine/index.json only)
Any of them with --out NAME writes into data/NAME/ instead (a build to check before swapping it in).

Per 1 x 1 degree tile, named by its south-west corner (n40w074), into data/fine/:
  <name>.land.bin.gz   1200 x 1200 LC100 class (nearest pixel to the cell centre), then
                       1200 x 1200 int16 LE GLO-90 height in metres at the cell centre as plane
                       deltas (plane_deltas), from the T-011 run's data/terrain/elev/
                       (tools/terrain/world.py)
  <name>.roads.bin.gz  1200 x 1200 road bits (ROAD_* below); a way marks every cell it passes through
A town tile holds only its selected 0.25 degree blocks: every byte of the others is 0, and
index.json gives the tile's block mask (docs/fine.md, "Town blocks"). The worker turns these into
speeds (js/tiles.js). Each file is written to a temporary name and renamed, roads before land,
and index.json last.
"""
import gzip
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import rasterio
from rasterio.windows import Window

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(HERE, "data", "raw")
# the output folder; set from --out through the environment so pool workers see it too
OUT = os.path.join(HERE, "data", os.environ.get("ISO_FINE_OUT", "fine"))
LC100 = os.path.join(RAW, "lc100", "cgls-lc100-2019-cog.tif")
ELEV = os.path.join(HERE, "data", "terrain", "elev")
OSM = os.path.join(RAW, "osm")
BITS = os.path.join(HERE, "data", "roads", "bits")
SELECTION = os.path.join(HERE, "data", "fine", "selection.json")

sys.path.insert(0, os.path.join(HERE, "tools", "terrain"))
from walk import SLOPE_RULE                # noqa: E402  the page walks the same slope rule as L1

N = 1200            # cells a tile side
LC100_NORTH = 80    # LC100's northern edge, degrees
BLOCK = 300         # cells a town block side (0.25 degree)
FULL = 0xFFFF       # block mask of a whole tile; bit 4 * i + j = block row i from the north, column j

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


def heights(name):
    """GLO-90 height of the tile's cells, metres, (N, N) int16, from the T-011 run's elev/ file
    (column deltas; ocean and the blocks outside a town tile's mask 0)."""
    p = os.path.join(ELEV, name + ".bin.gz")
    if not os.path.exists(p):
        raise SystemExit("no GLO-90 heights for %s (%s)" % (name, p))
    d = np.frombuffer(gzip.open(p).read(), "<i2").reshape(N, N)
    return np.cumsum(d, axis=1, dtype=np.int32).astype(np.int16)


def plane_deltas(z):
    """int16 heights as each value minus the plane through its west, north and north-west
    neighbours (z - w - n + nw); along the first row minus the west one, down the first column
    minus the north one, the corner as it is. 15% smaller gzipped than deltas along rows."""
    a = z.astype(np.int32)
    d = a.copy()
    d[1:, 1:] = a[1:, 1:] - a[1:, :-1] - a[:-1, 1:] + a[:-1, :-1]
    d[0, 1:] = a[0, 1:] - a[0, :-1]
    d[1:, 0] = a[1:, 0] - a[:-1, 0]
    return d.astype("<i2")


def write_gz(path, *arrays):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb", compresslevel=9) as f:
        for a in arrays:
            f.write(np.ascontiguousarray(a).tobytes())
    os.replace(tmp, path)


def tile_box(name):
    lat = int(name[1:3]) * (1 if name[0] == "n" else -1)
    lon = int(name[4:7]) * (1 if name[3] == "e" else -1)
    return (lon, lat, lon + 1, lat + 1)


def block_keep(mask):
    """(N, N) bool, true in the blocks of mask."""
    b = ((mask >> np.arange(16)) & 1).astype(bool).reshape(4, 4)
    return np.repeat(np.repeat(b, BLOCK, 0), BLOCK, 1)


def build_tile(job):
    """One tile from LC100, GLO-90 heights and data/roads/bits; job = (name, mask).
    Returns (name, mask, bytes written, has roads)."""
    name, mask = job
    box = tile_box(name)
    cls = land_classes(box)
    z = heights(name)
    bits = os.path.join(BITS, name + ".bits.gz")
    roads = np.frombuffer(gzip.open(bits).read(), np.uint8).reshape(N, N) if os.path.exists(bits) else None
    if mask != FULL:
        keep = block_keep(mask)
        cls = np.where(keep, cls, 0).astype(np.uint8)
        z = np.where(keep, z, 0).astype(np.int16)
        if roads is not None:
            roads = np.where(keep, roads, 0).astype(np.uint8)
    rp = os.path.join(OUT, name + ".roads.bin.gz")
    if roads is not None and roads.any():
        write_gz(rp, roads)
    else:
        roads = None
        if os.path.exists(rp):
            os.remove(rp)
    lp = os.path.join(OUT, name + ".land.bin.gz")
    write_gz(lp, cls, plane_deltas(z))
    size = os.path.getsize(lp) + (os.path.getsize(rp) if roads is not None else 0)
    return name, mask, size, roads is not None


def plan():
    """{tile: block mask} of every tile to build: the selection's towns, and whole its water
    country, mountains (a town tile among them whole too) and the test regions."""
    sel = json.load(open(SELECTION))
    tiles = {n: int(m) for n, m in sel["towns"].items()}
    for n in sel["water"] + sel["mountains"]:
        tiles[n] = FULL
    # LC100 stops at 80 N: the 26 mountain tiles past it (north Greenland, Ellesmere) stay at L1
    tiles = {n: m for n, m in tiles.items() if tile_box(n)[3] <= LC100_NORTH}
    for box, _ in REGIONS.values():
        west, south, east, north = box
        for lat in range(south, north):
            for lon in range(west, east):
                tiles[tile_name(lat, lon)] = FULL
    return tiles


def build_tiles(jobs, workers):
    """Build (name, mask) jobs on workers processes."""
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    total = 0
    done = 0
    if workers > 1:
        from multiprocessing import Pool
        pool = Pool(workers)
        it = pool.imap_unordered(build_tile, jobs, chunksize=4)
    else:
        pool = None
        it = map(build_tile, jobs)
    try:
        for name, mask, size, has_roads in it:
            done += 1
            total += size
            if done % 50 == 0 or done == len(jobs):
                el = time.time() - t0
                print("%d/%d tiles, %.0f MB, %.0f s, ~%.0f s left" % (
                    done, len(jobs), total / 1e6, el, el / done * (len(jobs) - done)), flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    print("built %d tiles, %.0f MB, %.0f s" % (done, total / 1e6, time.time() - t0), flush=True)


def world(args):
    workers = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 1
    tiles = plan()
    jobs = sorted(tiles.items())
    if "--resume" in args:
        # skip a tile already built with the same blocks: its land file exists and the folder's
        # index.json gives it the planned mask (a town tile that is now a mountain one is rebuilt whole)
        ip = os.path.join(OUT, "index.json")
        have = {}
        if os.path.exists(ip):
            ix = json.load(open(ip))
            have = {n: FULL for n in ix["tiles"]}
            have.update({n: int(m) for n, m in ix.get("partial", {}).items()})
        jobs = [j for j in jobs if not (os.path.exists(os.path.join(OUT, j[0] + ".land.bin.gz"))
                                        and (not have or have.get(j[0]) == j[1]))]
    print("world: %d tiles to build on %d workers" % (len(jobs), workers), flush=True)
    build_tiles(jobs, workers)


def build(region):
    box, extracts = REGIONS[region]
    west, south, east, north = box
    t0 = time.time()
    print("%s: land cover" % region, flush=True)
    cls = land_classes(box)
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
            write_gz(os.path.join(OUT, name + ".roads.bin.gz"), roads[r:r + N, c:c + N])
            write_gz(os.path.join(OUT, name + ".land.bin.gz"), cls[r:r + N, c:c + N], plane_deltas(heights(name)))
            sz = sum(os.path.getsize(os.path.join(OUT, name + e)) for e in (".land.bin.gz", ".roads.bin.gz"))
            print("  %s %.2f MB" % (name, sz / 1e6), flush=True)
    print("%s: %d ways, %.0f s" % (region, nways, time.time() - t0), flush=True)


def index():
    """index.json: tiles = whole tiles (all the page reads before it knows blocks), partial =
    {tile: block mask} for town tiles holding only some blocks, roads = tiles with a .roads file.
    A tile on disk that the plan does not hold is a whole tile (a region built by hand)."""
    masks = plan() if os.path.exists(SELECTION) else {}
    names = sorted(f[:-12] for f in os.listdir(OUT) if f.endswith(".land.bin.gz"))
    whole = [n for n in names if masks.get(n, FULL) == FULL]
    partial = {n: masks[n] for n in names if masks.get(n, FULL) != FULL}
    roads = [n for n in names if os.path.exists(os.path.join(OUT, n + ".roads.bin.gz"))]
    # elev: the land file's second part is heights (stage 2; stage 1's had "terrain" and "tfScale")
    idx = {"tile": N, "block": BLOCK, "elev": "int16 plane deltas", "slope": SLOPE_RULE,
           "tiles": whole, "partial": partial, "roads": roads}
    path = os.path.join(OUT, "index.json")
    with open(path + ".tmp", "w") as f:
        json.dump(idx, f, separators=(",", ":"))
    os.replace(path + ".tmp", path)
    nb = 16 * len(whole) + sum(bin(m).count("1") for m in partial.values())
    print("index: %d whole tiles, %d partial, %d with roads; %d blocks (%.0f tiles of area)" % (
        len(whole), len(partial), len(roads), nb, nb / 16))


if __name__ == "__main__":
    args = sys.argv[1:] or ["index"]
    if "--out" in args:
        i = args.index("--out")
        os.environ["ISO_FINE_OUT"] = args[i + 1]
        OUT = os.path.join(HERE, "data", args[i + 1])
        del args[i:i + 2]
        args = args or ["index"]
    if args[0] == "world":
        world(args[1:])
    elif args[0] == "tiles":
        build_tiles([(n, FULL) for n in args[1:]], 1)
    elif args[0] == "region":
        for a in args[1:]:
            build(a)
    elif args[0] != "index":
        raise SystemExit(__doc__)
    index()
