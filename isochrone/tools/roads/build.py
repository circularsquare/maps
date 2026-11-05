"""Roads as their own layer, from Geofabrik extracts (docs/roads.md).

    python isochrone/tools/roads/build.py extract <name> [<name> ...] [--box W,S,E,N]
    python isochrone/tools/roads/build.py reduce [<tile> ...]

extract: every highway way of data/raw/osm/<name>.osm.pbf, kinds as the fine level's
  (tools/fine/build.py way_code), drawn at 1/1200 degree and OR-ed into one file per 1 x 1 degree
  tile, data/roads/bits/<tile>.bits.gz (1200 x 1200 road bits, the fine level's .roads format).
  OR makes a second pass over the same ways harmless, so extracts that overlap at a border, or a
  rerun after a stop, never count a road twice. A finished extract is listed in data/roads/done.txt
  and skipped next time. --box limits the tiles written (tests).
reduce: each bits tile to its 1 km cells, data/roads/l1/<tile>.bin.gz: 120 x 120 counts of
  100 m cells with a walkable way (0-100), then 120 x 120 OR of the road bits.
"""
import gzip
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("fine_build", os.path.join(HERE, "tools", "fine", "build.py"))
fine = importlib.util.module_from_spec(_spec)   # way_code, tile_name, road bits
_spec.loader.exec_module(fine)

OSM = os.path.join(HERE, "data", "raw", "osm")
ROADS = os.path.join(HERE, "data", "roads")
BITS = os.path.join(ROADS, "bits")
L1 = os.path.join(ROADS, "l1")
DONE = os.path.join(ROADS, "done.txt")

N = 1200                  # 100 m cells a tile side
WALK = fine.ROAD_PATH | fine.ROAD_LOCAL | fine.ROAD_MAIN
BATCH = 400_000           # ways drawn at a time


def read_gz(path, shape):
    return np.frombuffer(gzip.open(path).read(), np.uint8).reshape(shape)


def write_gz(path, *arrays):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        for a in arrays:
            f.write(np.ascontiguousarray(a).tobytes())
    os.replace(tmp, path)


class TileLock:
    """One writer at a time per tile file, so two extracts read at once never lose each other's
    roads in a shared border tile. A lock left by a killed process is broken after STALE s."""
    STALE = 600

    def __init__(self, path):
        self.lock = path + ".lock"

    def __enter__(self):
        while True:
            try:
                os.close(os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.lock) > self.STALE:
                        os.remove(self.lock)
                        continue
                except OSError:
                    pass
                time.sleep(0.2)

    def __exit__(self, *exc):
        os.remove(self.lock)


class Tiles:
    """Road bits of the tiles one extract touches, in memory until flushed."""

    def __init__(self, box=None):
        self.t = {}
        self.box = box

    def draw(self, xs, ys, sizes, codes):
        """xs, ys: node lon/lat of every run of located nodes, end to end; sizes: nodes a run;
        codes: road bits a run."""
        sizes = np.array(sizes)
        x = (np.array(xs) + 180.0) * N
        y = (90.0 - np.array(ys)) * N
        code = np.repeat(np.array(codes, np.uint8), sizes)
        seg = np.ones(len(x), bool)
        seg[np.cumsum(sizes) - 1] = False
        i0 = np.nonzero(seg)[0]
        x0, y0, x1, y1, c = x[i0], y[i0], x[i0 + 1], y[i0 + 1], code[i0]
        keep = np.abs(x1 - x0) < 180 * N          # a segment over the antimeridian is not drawn
        x0, y0, x1, y1, c = x0[keep], y0[keep], x1[keep], y1[keep], c[keep]
        # samples a third of a cell apart, as tools/fine/build.py rasterise
        steps = np.ceil(np.maximum(np.abs(x1 - x0), np.abs(y1 - y0)) * 3).astype(np.int64) + 1
        CH = 4_000_000
        s0 = 0
        while s0 < len(steps):
            cs = np.cumsum(steps[s0:])
            s1 = s0 + max(1, int(np.searchsorted(cs, CH)))
            st = steps[s0:s1]
            rep = np.repeat(np.arange(s0, s1), st)
            start = np.repeat(np.cumsum(st) - st, st)
            f = (np.arange(len(rep)) - start) / np.maximum(1, np.repeat(st, st) - 1)
            px = np.floor(x0[rep] + (x1[rep] - x0[rep]) * f).astype(np.int64) % (360 * N)
            py = np.floor(y0[rep] + (y1[rep] - y0[rep]) * f).astype(np.int64)
            ok = (py >= 0) & (py < 180 * N)
            px, py, cc = px[ok], py[ok], c[rep][ok].astype(np.int64)
            tile = (py // N) * 360 + px // N
            key = (tile * (N * N) + (py % N) * N + px % N) * 32 + cc
            key = np.unique(key)
            cc = (key % 32).astype(np.uint8)
            cell = key // 32
            tile = cell // (N * N)
            local = cell % (N * N)
            bounds = np.flatnonzero(np.diff(tile)) + 1
            for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(tile)]):
                t = int(tile[a])
                arr = self.t.get(t)
                if arr is None:
                    arr = self.t[t] = np.zeros(N * N, np.uint8)
                lo, cv = local[a:b], cc[a:b]
                for v in np.unique(cv):
                    sel = lo[cv == v]
                    arr[sel] |= v
            s0 = s1

    def name(self, t):
        lat = 89 - t // 360
        lon = t % 360 - 180
        return lat, lon, fine.tile_name(lat, lon)

    def flush(self):
        os.makedirs(BITS, exist_ok=True)
        n = 0
        for t, arr in sorted(self.t.items()):
            lat, lon, name = self.name(t)
            if self.box:
                w, s, e, nn = self.box
                if not (w <= lon < e and s <= lat < nn):
                    continue
            path = os.path.join(BITS, name + ".bits.gz")
            with TileLock(path):
                if os.path.exists(path):
                    arr = arr | read_gz(path, (N * N,))
                write_gz(path, arr)
            n += 1
        self.t = {}
        return n


def ways(path):
    """(lon array, lat array, road bits) for each run of located nodes of each kept way."""
    import osmium
    fp = (osmium.FileProcessor(path)
          .with_locations()
          .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
          .with_filter(osmium.filter.KeyFilter("highway")))
    for w in fp:
        code = fine.way_code(w.tags)
        if not code:
            continue
        xs, ys = [], []
        for n in w.nodes:
            loc = n.location
            if loc.valid():
                xs.append(loc.lon)
                ys.append(loc.lat)
            else:
                # a node outside the extract: keep the located runs either side of it
                if len(xs) > 1:
                    yield xs, ys, code, True
                xs, ys = [], []
        if len(xs) > 1:
            yield xs, ys, code, False


def extract(name, box=None):
    path = os.path.join(OSM, name + ".osm.pbf")
    t0 = time.time()
    tiles = Tiles(box)
    xs, ys, ns, cs = [], [], [], []
    nways = nsplit = 0
    for x, y, code, split in ways(path):
        xs.extend(x)
        ys.extend(y)
        ns.append(len(x))
        cs.append(code)
        nways += 1
        nsplit += split
        if len(cs) >= BATCH:
            tiles.draw(xs, ys, ns, cs)
            xs, ys, ns, cs = [], [], [], []
            print("  %s: %d ways, %d tiles, %.0f s" % (name, nways, len(tiles.t), time.time() - t0), flush=True)
    if cs:
        tiles.draw(xs, ys, ns, cs)
    t1 = time.time()
    n = tiles.flush()
    print("%s: %d way runs (%d cut at a missing node), %d tiles written, read %.0f s, write %.0f s"
          % (name, nways, nsplit, n, t1 - t0, time.time() - t1), flush=True)
    if box is None:
        with open(DONE, "a") as f:
            f.write(name + "\n")


def reduce_tile(name):
    b = read_gz(os.path.join(BITS, name + ".bits.gz"), (N, N))
    blk = b.reshape(120, 10, 120, 10)
    count = ((blk & WALK) > 0).sum(axis=(1, 3)).astype(np.uint8)
    kinds = np.bitwise_or.reduce(np.bitwise_or.reduce(blk, axis=3), axis=1)
    os.makedirs(L1, exist_ok=True)
    write_gz(os.path.join(L1, name + ".bin.gz"), count, kinds)


def done():
    if not os.path.exists(DONE):
        return set()
    return set(open(DONE).read().split())


if __name__ == "__main__":
    args = sys.argv[1:]
    box = None
    if "--box" in args:
        i = args.index("--box")
        box = tuple(int(v) for v in args[i + 1].split(","))
        del args[i:i + 2]
    os.makedirs(ROADS, exist_ok=True)
    if args[0] == "extract":
        finished = done()
        for nm in args[1:]:
            if nm in finished and box is None:
                print("%s: done already" % nm)
                continue
            extract(nm, box)
    elif args[0] == "reduce":
        names = args[1:] or sorted(f[:-8] for f in os.listdir(BITS) if f.endswith(".bits.gz"))
        for nm in names:
            reduce_tile(nm)
        print("reduced %d tiles" % len(names))
