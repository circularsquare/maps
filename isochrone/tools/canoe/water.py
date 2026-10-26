"""Water tiles for the canoe (docs/canoe.md 4): kind, shore and links at L10, L3 and L1.

    python tools/canoe/water.py [--cache DIR]       # ~10 min, ~6 GB of memory

Reads the walking data the page uses, so canoe water is exactly walking's water at every level:
data/tiles/l1 (L1 water = speed 0, with T-013's small water already land), data/tiles/l3 and
data/grid.bin.gz (a coarse cell is water where its speed is 0). Classifies L1 water as sea
(inside the Natural Earth 10 m ocean polygon, or river-free water joined to it within GROW
pixels), lake (inside a HydroLAKES polygon; its fetch wave height coded in kind) or river water
(the rest). Writes data/canoe/: water.bin.gz (L10), l3/<r>_<c>.bin.gz, l1/<r>_<c>.bin.gz and
index.json. Byte layouts: notes/T-021.md.

--cache DIR keeps the L1 kind and shore arrays (1.5 GB) between runs, for checks.
"""
import gzip
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPS = os.path.dirname(HERE)
TILES = os.path.join(HERE, "data", "tiles")
GRID = os.path.join(HERE, "data", "grid.bin.gz")
OUT = os.path.join(HERE, "data", "canoe")
OCEAN = os.path.join(MAPS, "data", "ne_10m_lakes", "ne_10m_ocean.shp")
LAKES = os.path.join(MAPS, "data", "HydroLAKES_polys_v10_shp", "HydroLAKES_polys_v10.shp")

W1, H1 = 43200, 17400           # L1: 1/120 degree, 85N to 60S
NORTH, PIX = 85.0, 1 / 120
STRIP = 600                     # one row of 5 degree tiles at L1
TILE_DEG = 5
TILES_X, TILES_Y = 72, 29
R_KM = 6371.0088

LAND, SEA, RIVER, LAKE0 = 0, 1, 2, 3
GROW = 8                        # river-free water this many pixels from NE ocean is sea too
U_WIND, G = 7.0, 9.81           # lake waves: research/canoe.md 2
NEAR_KM = 30.0                  # exact L1 distance below this; beyond, interpolated from L10 centres

# neighbour k of a cell, as (dc, dr), in js/search.js's move order; links bit k
NEIGH = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def lake_code(area_km2):
    """kind of a lake: 3 + its fetch-limited wave height in 0.05 m steps."""
    u2g = U_WIND ** 2 / G
    fetch = np.sqrt(np.asarray(area_km2, np.float64) * 1e6)
    hs = np.minimum(0.0016 * u2g * np.sqrt(G * fetch / U_WIND ** 2), 0.21 * u2g)
    return (LAKE0 + np.rint(hs / 0.05)).astype(np.uint8)


def shore_code(km):
    """km to the nearest land -> 1..78 (1 km steps to 20 km, 10 km steps to 600 km)."""
    near = np.clip(np.rint(km), 1, 20)
    far = np.clip(20 + np.rint((km - 20) / 10), 21, 78)
    return np.where(km <= 20.5, near, far).astype(np.uint8)


# ---------------------------------------------------------------- inputs

def load_land():
    with open(os.path.join(TILES, "index.json")) as fh:
        idx = json.load(fh)
    lands = {}
    for f in ("1", "3"):
        lv = idx["levels"][f]
        a = np.zeros((lv["height"], lv["width"]), bool)
        s = lv["tile"]
        for t in lv["tiles"]:
            tr, tc = divmod(t, TILES_X)
            with gzip.open(os.path.join(TILES, lv["path"], "%d_%d.bin.gz" % (tr, tc))) as fh:
                a[tr * s:(tr + 1) * s, tc * s:(tc + 1) * s] = np.frombuffer(fh.read(), np.uint8).reshape(s, s) > 0
        lands[int(f)] = a
    with gzip.open(GRID) as fh:
        lands[10] = np.frombuffer(fh.read(), np.uint8).reshape(H1 // 10, W1 // 10) > 0
    return lands


def classify(land):
    """L1 kind: 0 land, 1 sea, 2 river water, 3+ lake."""
    import pyogrio
    import shapely
    from rasterio import features
    from rasterio.transform import from_origin

    ocean = pyogrio.read_dataframe(OCEAN).geometry.to_numpy()
    log("reading HydroLAKES")
    lakes = pyogrio.read_dataframe(LAKES, columns=["Lake_area"])
    geoms = lakes.geometry.to_numpy()
    codes = lake_code(lakes["Lake_area"].values)
    order = np.argsort(codes, kind="stable")         # larger lakes burnt last where outlines overlap
    geoms, codes = geoms[order], codes[order]
    b = shapely.bounds(geoms)
    log("%d lakes; codes %d..%d" % (len(codes), codes.min(), codes.max()))

    kind = np.zeros((H1, W1), np.uint8)
    for i in range(H1 // STRIP):
        top = NORTH - i * STRIP * PIX
        bot = top - STRIP * PIX
        tf = from_origin(-180.0, top, PIX, PIX)
        shape = (STRIP, W1)
        oc = shapely.clip_by_rect(ocean, -180, bot - 0.01, 180, top + 0.01)
        sea = features.rasterize([(g, 1) for g in oc if not g.is_empty], out_shape=shape,
                                 transform=tf, fill=0, dtype="uint8")
        sel = np.nonzero((b[:, 1] < top) & (b[:, 3] > bot))[0]
        lk = features.rasterize(zip(geoms[sel], codes[sel].tolist()), out_shape=shape, transform=tf,
                                fill=0, dtype="uint8") if len(sel) else np.zeros(shape, np.uint8)
        k = np.where(sea > 0, SEA, np.where(lk > 0, lk, RIVER)).astype(np.uint8)
        # MAP walks over some straits (the Bosporus, a gap in the Dardanelles): a land pixel that
        # NE's ocean touches keeps its land node and gets a sea canoe node too
        touched = features.rasterize([(g, 1) for g in oc if not g.is_empty], out_shape=shape,
                                     transform=tf, fill=0, dtype="uint8", all_touched=True)
        ln = land[i * STRIP:(i + 1) * STRIP]
        k[ln] = np.where(touched[ln] > 0, SEA, LAND)
        kind[i * STRIP:(i + 1) * STRIP] = k
        log("classified strip %d / %d (%d lakes)" % (i + 1, H1 // STRIP, len(sel)))
    # NE's coast is not MAP's: river-free water within GROW pixels of NE sea, through water, is sea
    log("land pixels with a sea canoe node: %d" % int((land & (kind == SEA)).sum()))
    river = kind == RIVER
    n0 = int(river.sum())
    grown = ndimage.binary_dilation(kind == SEA, structure=np.ones((3, 3), bool), iterations=GROW, mask=river | (kind == SEA))
    kind[grown & river] = SEA
    log("river water %d px before the sea grows into it, %d after" % (n0, int((kind == RIVER).sum())))
    return kind


def shore(land, kind):
    """L1 shore code from the great-circle distance to the nearest land pixel centre: 0 on land,
    1 on a land pixel with a canoe node."""
    water = ~land
    coast = land & ndimage.binary_dilation(water, structure=np.ones((3, 3), bool))
    coast[:, 0] |= land[:, 0] & water[:, -1]
    coast[:, -1] |= land[:, -1] & water[:, 0]
    r, c = np.nonzero(coast)
    log("coast pixels: %d" % len(r))
    tree = cKDTree(xyz(r, c, PIX))
    del r, c, coast
    # far field: every L10 centre, unbounded
    rr, cc = np.mgrid[0:H1 // 10, 0:W1 // 10]
    d10 = chord_km(tree.query(xyz(rr.ravel(), cc.ravel(), PIX * 10), workers=2)[0]).reshape(rr.shape)
    log("L10 centre distances done, max %.0f km" % d10.max())
    out = np.zeros((H1, W1), np.uint8)
    ub = 2 * np.sin(NEAR_KM / R_KM / 2)
    for i in range(H1 // STRIP):
        r0 = i * STRIP
        rows = (np.arange(r0, r0 + STRIP) + 0.5) / 10 - 0.5
        cols = (np.arange(W1) + 0.5) / 10 - 0.5
        rg = np.clip(rows, 0, H1 // 10 - 1)
        cg = np.clip(cols, 0, W1 // 10 - 1)
        d = ndimage.map_coordinates(d10, np.meshgrid(rg, cg, indexing="ij"), order=1, mode="nearest").astype(np.float32)
        w = water[r0:r0 + STRIP]
        near = w & (d < NEAR_KM + 10)
        nr, nc = np.nonzero(near)
        q = tree.query(xyz(nr + r0, nc, PIX), distance_upper_bound=ub, workers=2)[0]
        ok = np.isfinite(q)
        dn = d[nr, nc]
        dn[ok] = chord_km(q[ok])
        d[nr, nc] = dn
        code = shore_code(d)
        code[~w] = 0
        code[~w & (kind[r0:r0 + STRIP] != LAND)] = 1
        out[r0:r0 + STRIP] = code
        log("shore strip %d / %d (%d exact)" % (i + 1, H1 // STRIP, len(nr)))
    return out


def xyz(r, c, step):
    lat = np.radians(NORTH - (np.asarray(r, np.float64) + 0.5) * step)
    lon = np.radians(-180.0 + (np.asarray(c, np.float64) + 0.5) * step)
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=1)


def chord_km(ch):
    return 2 * R_KM * np.arcsin(np.minimum(ch, 2.0) / 2)


# ---------------------------------------------------------------- coarse levels

def coarse(kind, sh, land_c, f):
    """kind, shore and links of level f (cells of f x f L1 pixels)."""
    hc, wc = H1 // f, W1 // f
    n_sea, n_lake, n_riv = (np.zeros((hc, wc), np.int32) for _ in range(3))
    lake = np.zeros((hc, wc), np.uint8)
    smin = np.zeros((hc, wc), np.uint8)
    sc = STRIP // f
    for i in range(H1 // STRIP):            # strip by strip, to keep temporaries small
        ks = kind[i * STRIP:(i + 1) * STRIP].reshape(sc, f, wc, f)
        out = slice(i * sc, (i + 1) * sc)
        n_sea[out] = (ks == SEA).sum(axis=(1, 3), dtype=np.int32)
        n_lake[out] = (ks >= LAKE0).sum(axis=(1, 3), dtype=np.int32)
        n_riv[out] = (ks == RIVER).sum(axis=(1, 3), dtype=np.int32)
        lake[out] = ks.max(axis=(1, 3))     # lake codes are the largest kinds
        ss = sh[i * STRIP:(i + 1) * STRIP]
        smin[out] = np.where(ss > 0, ss, 255).astype(np.uint8).reshape(sc, f, wc, f).min(axis=(1, 3))
    k = np.zeros((hc, wc), np.uint8)
    open_ = n_sea + n_lake
    water_cell = ~land_c
    pick = np.where(n_sea >= n_lake, SEA, lake).astype(np.uint8)
    # a water cell is sea or lake if those pixels are at least its river pixels, else river water;
    # a land cell gets a canoe node (sea or lake) if it has any sea or lake pixel
    k[water_cell] = np.where(open_ >= n_riv, pick, RIVER)[water_cell]
    has = land_c & (open_ > 0)
    k[has] = pick[has]
    shore_c = np.where(k != LAND, smin, 0).astype(np.uint8)
    shore_c[shore_c == 255] = 0
    links = make_links(kind, f)
    # a link needs a canoe node at both ends (river-water cells have none)
    node = (k == SEA) | (k >= LAKE0)
    for b, (dc, dr) in enumerate(NEIGH):
        other = np.zeros_like(node)
        rs = slice(max(0, -dr), hc - max(0, dr))
        other[rs] = np.roll(node, -dc, axis=1)[max(0, dr):hc - max(0, -dr)]
        links &= ~((~(node & other)).astype(np.uint8) << b)
    return k, shore_c, links


def make_links(kind, f):
    """bit k set where sea or lake pixels of the cell and of neighbour k touch (8-connected)."""
    hc, wc = H1 // f, W1 // f
    cw = (kind == SEA) | (kind >= LAKE0)
    L = np.zeros((hc, wc), np.uint8)

    def touch_cols(a, b):
        """a, b: (H1, wc) columns either side of a vertical border; any 8-touch per coarse row."""
        # pixel pairs across a coarse row boundary belong to a diagonal neighbour, not this one
        inner = (np.arange(H1 - 1) % f != f - 1)[:, None]
        t = a & b
        t[:-1] |= a[:-1] & b[1:] & inner
        t[1:] |= a[1:] & b[:-1] & inner
        return t.reshape(hc, f, wc).any(axis=1)

    # east: last column of cell c against first column of cell c+1 (wrapping at 180)
    a = cw[:, f - 1::f]
    b = np.roll(cw[:, 0::f], -1, axis=1)
    e = touch_cols(a, b)
    L |= e.astype(np.uint8) << 0
    L |= np.roll(e, 1, axis=1).astype(np.uint8) << 1          # west of c+1 is east of c
    # south: last row of cell r against first row of cell r+1
    a = cw[f - 1::f, :]                     # (hc, W1)
    b = np.zeros_like(a)
    b[:-1] = cw[f::f, :]
    s = touch_rows(a, b, f)
    L |= s.astype(np.uint8) << 2
    n = np.zeros_like(s)
    n[1:] = s[:-1]
    L |= n.astype(np.uint8) << 3
    # diagonals through the shared corner
    br = cw[f - 1::f, f - 1::f]             # bottom-right pixel of each cell
    tl = cw[0::f, 0::f]
    tr = cw[0::f, f - 1::f]
    bl = cw[f - 1::f, 0::f]
    se = np.zeros((hc, wc), bool)
    se[:-1] = br[:-1] & np.roll(tl[1:], -1, axis=1)
    ne = np.zeros((hc, wc), bool)
    ne[1:] = tr[1:] & np.roll(bl[:-1], -1, axis=1)
    L |= se.astype(np.uint8) << 4
    L |= ne.astype(np.uint8) << 5
    sw = np.zeros((hc, wc), bool)           # sw of cell (r, c) is ne of (r+1, c-1)
    sw[:-1] = np.roll(ne[1:], 1, axis=1)
    nw = np.zeros((hc, wc), bool)           # nw of (r, c) is se of (r-1, c-1)
    nw[1:] = np.roll(se[:-1], 1, axis=1)
    L |= sw.astype(np.uint8) << 6
    L |= nw.astype(np.uint8) << 7
    return L


def touch_rows(a, b, f):
    """a, b: (hc, W1) rows either side of a horizontal border; any 8-touch inside one coarse column."""
    wc = a.shape[1] // f
    inner = (np.arange(a.shape[1] - 1) % f != f - 1)[None, :]
    t = a & b
    t[:, :-1] |= a[:, :-1] & b[:, 1:] & inner
    t[:, 1:] |= a[:, 1:] & b[:, :-1] & inner
    return t.reshape(a.shape[0], wc, f).any(axis=2)


# ---------------------------------------------------------------- output

def write_gz(path, planes):
    data = gzip.compress(b"".join(np.ascontiguousarray(p).tobytes() for p in planes), compresslevel=9)
    with open(path + ".tmp", "wb") as fh:
        fh.write(data)
    os.replace(path + ".tmp", path)
    return len(data)


def write_tiles(level, planes, size, land):
    d = os.path.join(OUT, "l%d" % level)
    os.makedirs(d, exist_ok=True)
    found, nbytes = [], 0
    kind = planes[0]
    for tr in range(TILES_Y):
        for tc in range(TILES_X):
            sl = (slice(tr * size, (tr + 1) * size), slice(tc * size, (tc + 1) * size))
            path = os.path.join(d, "%d_%d.bin.gz" % (tr, tc))
            if (kind[sl] == SEA).all() and not land[sl].any():
                if os.path.exists(path):
                    os.remove(path)
                continue
            nbytes += write_gz(path, [p[sl] for p in planes])
            found.append(tr * TILES_X + tc)
    return found, nbytes


def main():
    t0 = time.time()
    cache = sys.argv[sys.argv.index("--cache") + 1] if "--cache" in sys.argv else None
    lands = load_land()
    land = lands[1]
    log("land loaded: L1 %.3f, L3 %.3f, L10 %.3f land" % (land.mean(), lands[3].mean(), lands[10].mean()))
    kp = cache and os.path.join(cache, "kind1.npy")
    sp = cache and os.path.join(cache, "shore1.npy")
    if kp and os.path.exists(kp):
        kind = np.load(kp)
    else:
        kind = classify(land)
        if kp:
            os.makedirs(cache, exist_ok=True)
            np.save(kp, kind)
    if sp and os.path.exists(sp):
        sh = np.load(sp)
    else:
        sh = shore(land, kind)
        if sp:
            np.save(sp, sh)
    del land
    os.makedirs(OUT, exist_ok=True)
    index = {"west": -180.0, "east": 180.0, "north": NORTH, "south": -60.0, "tileDeg": TILE_DEG,
             "tilesX": TILES_X, "tilesY": TILES_Y, "levels": {}}
    sizes = {}
    for f in (10, 3):
        k, s, l = coarse(kind, sh, lands[f], f)
        if f == 10:
            sizes[10] = write_gz(os.path.join(OUT, "water.bin.gz"), [k, s, l])
            index["levels"]["10"] = {"factor": 10, "width": W1 // 10, "height": H1 // 10,
                                     "file": "water.bin.gz", "bands": ["kind", "shore", "links"]}
        else:
            found, sizes[3] = write_tiles(3, [k, s, l], 200, lands[3])
            index["levels"]["3"] = {"factor": 3, "width": W1 // 3, "height": H1 // 3, "tile": 200,
                                    "path": "l3", "bands": ["kind", "shore", "links"], "tiles": found}
        log("L%d: %d sea, %d lake, %d river cells; %.2f MB" % (f, (k == SEA).sum(), (k >= LAKE0).sum(), (k == RIVER).sum(), sizes[f] / 1e6))
    found, sizes[1] = write_tiles(1, [kind, sh], 600, lands[1])
    index["levels"]["1"] = {"factor": 1, "width": W1, "height": H1, "tile": 600, "path": "l1",
                            "bands": ["kind", "shore"], "tiles": found}
    log("L1: %d tiles, %.2f MB" % (len(found), sizes[1] / 1e6))
    ip = os.path.join(OUT, "index.json")
    if os.path.exists(ip):                  # keep sea.py's and rivers.py's entries
        with open(ip) as fh:
            old = json.load(fh)
        for key in ("sea", "rivers"):
            if old.get(key) is not None:
                index[key] = old[key]
    with open(ip + ".tmp", "w") as fh:
        json.dump(index, fh, separators=(",", ":"))
    os.replace(ip + ".tmp", ip)
    log("done in %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
