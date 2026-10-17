"""T-275: routed walking and driving for a finished city pack.

    python pipeline/routes.py <city> [<city> ...] [--extracts a,b] [--keep-pbf]
    python pipeline/routes.py --download <city> ...      (only fetch and read the OSM extracts)

Adds three things to data/packs/<city>.json/.bin in place (nothing else in the pack changes):

- `walk`: the walking barrier raster, `walk_row`/`walk_x` (row runs like the water mask, at
  `WALK_CELL_M`): water from the pack's own water mask, with every OSM way a person may walk
  over a bridge on (footways, paths, streets; not motorways or `foot=no`) burnt back in as land.
  The game finds walks to its stations on it (sim `demand::routes`).
- `walk_zone`: per gravity zone, the walk's detour factor to each zone within `WALK_WIN` squares
  (u8, `ln(f) * 64`; 0 = no detour), for the walk share between zones.
- `drive`: routed road distance over straight distance between zones (u8, `ln(f) * 64`), for
  zones within `DRIVE_WIN` squares (`drive_near`, each zone's window in canonical order) and
  between supercells of `SUPER` x `SUPER` squares per region beyond it (`drive_far`).

Roads and bridges come from Geofabrik's OpenStreetMap extracts, one per first-level region of the
pack (`EXTRACTS`; any country's region ids can be added there, or `--extracts` names them), read
with GDAL's OSM driver and cached in data/raw/osm/ as compact .npz files; the .osm.pbf downloads
are deleted once read unless --keep-pbf. Data (c) OpenStreetMap contributors, ODbL.

Interface (what the runtime reads, docs/demand.md 4.3 "Routed walking and driving") is in the
header's `routes` block. A zone's window is enumerated the same way here and in the sim:
squares (dx, dy) with dx and dy from -W to W, dx outer, then the zones of that square in zone id
order; zones themselves are the pack's (`zone`, `zone_gx`, `zone_gy`).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")  # before numpy; Anita is using the machine

import json
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OSM = os.path.join(RAW, "osm")
PACKS = os.environ.get("TW_PACKS", os.path.join(ROOT, "data", "packs"))  # TW_PACKS: work on copies
UA = {"User-Agent": "trainworld-pipeline/0.1 (rail game data build; python-requests)"}
GEOFABRIK = "https://download.geofabrik.de/{}-latest.osm.pbf"

# A pack's first-level region id (header `regions.ids`) -> Geofabrik extract. US: state FIPS.
EXTRACTS = {
    "09": "north-america/us/connecticut", "10": "north-america/us/delaware",
    "11": "north-america/us/district-of-columbia", "24": "north-america/us/maryland",
    "25": "north-america/us/massachusetts", "33": "north-america/us/new-hampshire",
    "34": "north-america/us/new-jersey", "36": "north-america/us/new-york",
    "42": "north-america/us/pennsylvania", "44": "north-america/us/rhode-island",
    "51": "north-america/us/virginia", "54": "north-america/us/west-virginia",
    "50": "north-america/us/vermont",
    # Canada (T-277): province codes as the Toronto pack names them, or pass --extracts
    "ON": "north-america/canada/ontario", "35": "north-america/canada/ontario",
}

# Roads a car routes over: every public street but service ways (driveways, parking aisles).
# Residential streets are in: without them rural places (whose town roads OSM often tags
# residential) joined the network kilometres away and drives came out 1.5-1.7x the straight line.
# Each end joins the nearest point of these roads in a straight line x ROAD_DETOUR.
DRIVE_CLASSES = ("motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
                 "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
                 "residential", "living_street", "road")
# Ways a person may walk over a bridge on (unless foot=no, access=no/private without foot=yes).
WALK_CLASSES = ("primary", "primary_link", "secondary", "secondary_link", "tertiary",
                "tertiary_link", "unclassified", "residential", "living_street", "service", "road",
                "pedestrian", "footway", "path", "steps", "cycleway", "track", "corridor",
                "trunk", "trunk_link")
ROAD_DETOUR = 1.3
# road pieces in a connected part of the graph smaller than this many junctions are not joined to
MIN_COMPONENT = 200


def pbf_path(extract):
    return os.path.join(OSM, extract.replace("/", "_") + ".osm.pbf")


def npz_path(extract):
    return os.path.join(OSM, extract.replace("/", "_") + ".ways2.npz")


def download(extract):
    import requests
    path = pbf_path(extract)
    if os.path.exists(path):
        return path
    url = GEOFABRIK.format(extract)
    print(f"  downloading {url}", flush=True)
    t0 = time.time()
    tmp = path + ".part"
    with requests.get(url, headers=UA, timeout=600, stream=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, path)
    print(f"  {os.path.getsize(path) / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)
    return path


def tag(other, key):
    """A value from GDAL's other_tags text ('"k"=>"v","k2"=>"v2"'), or ''."""
    if not other:
        return ""
    i = other.find('"' + key + '"=>"')
    if i < 0:
        return ""
    j = i + len(key) + 5
    return other[j:other.find('"', j)]


def read_ways(extract, keep_pbf=False):
    """The extract's drivable roads and walkable bridges, cached as an .npz:
    per way its class (index into DRIVE_CLASSES, or 255 for walk-only), flags (1 drivable,
    2 bridge a person may walk on), and its points (lon, lat) in one flat array with offsets."""
    out = npz_path(extract)
    if os.path.exists(out):
        return dict(np.load(out))
    import pyogrio
    path = download(extract)
    t0 = time.time()
    classes = sorted(set(DRIVE_CLASSES) | set(WALK_CLASSES))
    where = "highway IN (" + ",".join(f"'{c}'" for c in classes) + ")"
    df = pyogrio.read_dataframe(path, layer="lines", columns=["highway", "other_tags"], where=where)
    print(f"  {extract}: {len(df):,} ways read in {time.time() - t0:.0f} s", flush=True)
    hw = df["highway"].to_numpy()
    other = df["other_tags"].fillna("").to_numpy()
    cls = np.full(len(df), 255, np.uint8)
    flags = np.zeros(len(df), np.uint8)
    dclass = {c: i for i, c in enumerate(DRIVE_CLASSES)}
    for i in range(len(df)):
        o = other[i]
        h = hw[i]
        access = tag(o, "access")
        motor = tag(o, "motor_vehicle") or tag(o, "motorcar")
        if h in dclass and access not in ("no", "private") and motor not in ("no", "private"):
            area = tag(o, "area")
            if area != "yes":
                cls[i] = dclass[h]
                flags[i] |= 1
        if tag(o, "bridge") not in ("", "no") and h in WALK_CLASSES:
            foot = tag(o, "foot")
            walkable = foot not in ("no", "private") and (access not in ("no", "private") or foot in ("yes", "designated", "permissive"))
            if h in ("trunk", "trunk_link") and foot not in ("yes", "designated", "permissive"):
                walkable = False  # many trunk bridges carry no sidewalk; OSM says foot=yes where one does
            if walkable:
                flags[i] |= 2
    keep = flags > 0
    geoms = df.geometry.to_numpy()[keep]
    import shapely
    coords, idx = shapely.get_coordinates(geoms, return_index=True)
    off = np.zeros(len(geoms) + 1, np.int64)
    np.add.at(off, idx + 1, 1)
    off = np.cumsum(off)
    np.savez(out, cls=cls[keep], flags=flags[keep], off=off, lon=coords[:, 0].astype(np.float64), lat=coords[:, 1].astype(np.float64))
    print(f"  {extract}: {int(keep.sum()):,} kept ({int((flags[keep] & 1).sum()):,} drivable, {int((flags[keep] & 2).sum()):,} walkable bridges), {len(coords):,} points", flush=True)
    if not keep_pbf:
        os.remove(path)
    return dict(np.load(out))


# ---------------------------------------------------------------- frame and helpers

R_EARTH = 6371008.8
WALK_CELL_M = 50.0     # the walking barrier raster's pixel
WALK_WATER_FRAC = 0.75 # a 50 m pixel is a barrier when 3 of its 4 water-mask pixels are water
BRIDGE_EXTEND_M = 60.0 # a bridge's ends are carried this far on, onto the bank
MIN_LAND_PX = 40       # land enclosed by water smaller than this (0.1 km², a res-9 cell) is water
WALK_WIN = 2           # zone walk factors to zones within this many squares
DRIVE_WIN = 6          # zone drive factors within this many squares, supercells beyond
SUPER = 4              # supercell side, in squares
QSCALE = 64.0          # factor = exp(q / QSCALE), q a byte
FMAX = float(np.exp(255 / QSCALE))


def local_xy(lon, lat, lon0, lat0):
    k = np.cos(np.radians(lat0))
    return R_EARTH * np.radians(lon - lon0) * k, R_EARTH * np.radians(lat - lat0)


def quantise(f):
    return np.clip(np.rint(QSCALE * np.log(np.maximum(f, 1.0))), 0, 255).astype(np.uint8)


def merged(ways, flag):
    """The ways with `flag` set, from every extract: (offsets, lon, lat, class)."""
    offs, lons, lats, cls = [np.zeros(1, np.int64)], [], [], []
    base = 0
    for w in ways:
        keep = np.nonzero(w["flags"] & flag)[0]
        o = w["off"]
        n = o[keep + 1] - o[keep]
        idx = np.repeat(o[keep], n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        lons.append(w["lon"][idx])
        lats.append(w["lat"][idx])
        offs.append(base + np.cumsum(n))
        cls.append(w["cls"][keep])
        base += int(n.sum())
    return np.concatenate(offs), np.concatenate(lons), np.concatenate(lats), np.concatenate(cls)


def water_mask(header, a):
    import water
    wb = header["water"]
    (x0, y0), (W, H) = wb["origin_m"], wb["size"]
    return water.WaterMask(x0, y0, wb["cell_m"], W, H, np.array(a["water_row"]), np.array(a["water_x"]))


# ---------------------------------------------------------------- walking barrier raster

def burn(bar, x, y, off, x0, y0, cell):
    """Clear (make walkable) the pixels under each polyline (4-connected, so a path can follow
    it pixel to pixel without cutting a corner), its ends carried BRIDGE_EXTEND_M onto the bank."""
    H, W = bar.shape
    nway = len(off) - 1
    first, last = off[:-1], off[1:] - 1
    ok = last > first
    x = x.copy()
    y = y.copy()
    for a_, b_ in ((first[ok], first[ok] + 1), (last[ok], last[ok] - 1)):
        dx, dy = x[a_] - x[b_], y[a_] - y[b_]
        n = np.maximum(np.hypot(dx, dy), 1e-6)
        x[a_] += dx / n * BRIDGE_EXTEND_M
        y[a_] += dy / n * BRIDGE_EXTEND_M
    way = np.repeat(np.arange(nway), np.diff(off))
    s = np.nonzero(way[1:] == way[:-1])[0]
    xa, ya, xb, yb = x[s], y[s], x[s + 1], y[s + 1]
    L = np.hypot(xb - xa, yb - ya)
    k = np.ceil(L / (cell / 5)).astype(np.int64) + 1
    seg = np.repeat(np.arange(len(s)), k)
    t = (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)) / np.repeat(np.maximum(k - 1, 1), k)
    px = xa[seg] + (xb - xa)[seg] * t
    py = ya[seg] + (yb - ya)[seg] * t
    c = np.floor((px - x0) / cell).astype(np.int64)
    r = np.floor((py - y0) / cell).astype(np.int64)
    inb = (c >= 0) & (c < W) & (r >= 0) & (r < H)
    bar[r[inb], c[inb]] = False
    # a diagonal step between consecutive samples of one segment: clear the corner too
    same = seg[1:] == seg[:-1]
    diag = same & (c[1:] != c[:-1]) & (r[1:] != r[:-1])
    rc, cc = r[:-1][diag], c[1:][diag]
    inb = (cc >= 0) & (cc < W) & (rc >= 0) & (rc < H)
    bar[rc[inb], cc[inb]] = False


def walk_raster(header, a, ways):
    """The walking barrier at WALK_CELL_M: water (from the pack's own mask), less bridges."""
    import water
    m = water_mask(header, a)
    f = int(round(WALK_CELL_M / m.cell))
    assert abs(f * m.cell - WALK_CELL_M) < 1e-6, "the walk pixel must be a whole number of water pixels"
    bar = m.dense(f) >= WALK_WATER_FRAC
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    off, lon, lat, _ = merged(ways, 2)
    x, y = local_xy(lon, lat, lon0, lat0)
    before = int(bar.sum())
    burn(bar, x, y, off, m.x0, m.y0, WALK_CELL_M)
    # islets (rocks, marsh hummocks, pier ends) are water to the walk: a cell centred out on the
    # water joins the nearest real shore, not a speck of land beside it
    from scipy.ndimage import label
    lab, nlab = label(~bar)
    size = np.bincount(lab.ravel())
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    small = size < MIN_LAND_PX
    small[0] = False
    small[edge] = False
    islets = small[lab]
    bar[islets] = True
    out = water.WaterMask.from_dense(m.x0, m.y0, WALK_CELL_M, bar)
    info = {"bridges": len(off) - 1, "pixels_opened": before - int(bar.sum()) + int(islets.sum()),
            "islets_closed": int(islets.sum())}
    return out, info


# ---------------------------------------------------------------- the road graph

class RoadGraph:
    """Drivable OSM ways as an undirected graph of their junctions (points shared by ways, way
    ends), each edge the length of road between two, in the pack frame's metres. Every point of
    every way is kept for snapping: a place joins the road at its nearest point, which lies on the
    edge between junctions `pa` and `pb`, `la` and `lb` metres along the road from them."""

    def __init__(self, ways, lon0, lat0, box=None):
        from scipy.sparse import coo_matrix
        off, lon, lat, cls = merged(ways, 1)
        x, y = local_xy(lon, lat, lon0, lat0)
        nway = len(off) - 1
        way = np.repeat(np.arange(nway), np.diff(off))
        if box is not None:
            x0, y0, x1, y1 = box
            inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            keepw = np.zeros(nway, bool)
            keepw[way[inside]] = True
            kp = keepw[way]
            x, y, lon, lat, way = x[kp], y[kp], lon[kp], lat[kp], way[kp]
        key = np.rint(lon * 1e7).astype(np.int64) * (1 << 31) + np.rint(lat * 1e7).astype(np.int64)
        uniq, node = np.unique(key, return_inverse=True)
        node = node.ravel()
        s = np.nonzero(way[1:] == way[:-1])[0]
        seglen = np.hypot(x[s + 1] - x[s], y[s + 1] - y[s])
        deg = np.bincount(np.concatenate([node[s], node[s + 1]]), minlength=len(uniq))
        start = np.ones(len(way), bool)
        start[1:] = way[1:] != way[:-1]
        end = np.ones(len(way), bool)
        end[:-1] = way[1:] != way[:-1]
        kept = (deg[node] != 2) | start | end
        # length along each way to each point
        step = np.zeros(len(way))
        step[s + 1] = seglen
        cum = np.cumsum(step)
        kp = np.nonzero(kept)[0]
        same = way[kp[1:]] == way[kp[:-1]]
        ea, eb = node[kp[:-1]][same], node[kp[1:]][same]
        el = cum[kp[1:]][same] - cum[kp[:-1]][same]
        nz = ea != eb
        # compact the kept nodes
        used = np.unique(np.concatenate([ea[nz], eb[nz]]))
        remap = np.full(len(uniq), -1, np.int64)
        remap[used] = np.arange(len(used))
        lo, hi = np.minimum(remap[ea[nz]], remap[eb[nz]]), np.maximum(remap[ea[nz]], remap[eb[nz]])
        ln = el[nz]
        # parallel edges: keep the shortest
        order = np.lexsort((ln, hi, lo))
        lo, hi, ln = lo[order], hi[order], ln[order]
        firstp = np.ones(len(lo), bool)
        firstp[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
        self.lo, self.hi, self.len = lo[firstp], hi[firstp], np.maximum(ln[firstp], 0.01)
        self.n = len(used)
        pos_of = np.full(len(uniq), -1, np.int64)
        pos_of[node[::-1]] = np.arange(len(node))[::-1]
        p = pos_of[used]
        self.x, self.y = x[p], y[p]
        self.edges = len(self.lo)
        self.points = len(x)
        # every point on its edge: the kept points before and after it on its way
        k_after = np.searchsorted(kp, np.arange(len(way)), side="left")
        k_after = np.minimum(k_after, len(kp) - 1)
        nxt = kp[k_after]
        prv = kp[np.maximum(np.searchsorted(kp, np.arange(len(way)), side="right") - 1, 0)]
        pa, pb = remap[node[prv]], remap[node[nxt]]
        ok = (pa >= 0) & (pb >= 0) & (way[prv] == way[np.arange(len(way))]) & (way[nxt] == way[np.arange(len(way))])
        # places join only roads of the network proper: a piece of main road whose only links are
        # streets left out of the graph (residential, service) would leave them cut off
        from scipy.sparse.csgraph import connected_components
        nc, comp = connected_components(self.csr(), directed=False)
        size = np.bincount(comp)
        self.big = size[comp] >= MIN_COMPONENT
        ok &= np.where(pa >= 0, self.big[np.maximum(pa, 0)], False)
        self.components = int((size >= MIN_COMPONENT).sum())
        self.px, self.py = x[ok], y[ok]
        self.pa, self.pb = pa[ok], pb[ok]
        self.la, self.lb = (cum - cum[prv])[ok], (cum[nxt] - cum)[ok]
        self.pseg = prv[ok]          # the edge's first point: the same for every point of one edge
        self.pcum = cum[ok]

    def csr(self, extra=None):
        """The graph as a symmetric CSR matrix, with `extra` edges (a, b, metres) added."""
        from scipy.sparse import coo_matrix
        lo, hi, ln = self.lo, self.hi, self.len
        n = self.n
        if extra is not None:
            ea, eb, el = extra
            lo, hi, ln = np.concatenate([lo, ea]), np.concatenate([hi, eb]), np.concatenate([ln, el])
            n = max(n, int(max(ea.max(), eb.max())) + 1)
        return coo_matrix((np.concatenate([ln, ln]), (np.concatenate([lo, hi]), np.concatenate([hi, lo]))), shape=(n, n)).tocsr()

    def snap(self, px, py, water_dense=None, wm=None, k=8):
        """Each place's nearest road point whose straight line to it crosses no water (the 25 m
        mask), else the nearest. Returns the point's index and the metres to it."""
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([self.px, self.py]))
        d, i = tree.query(np.column_stack([px, py]), k=k)
        choice = np.zeros(len(px), np.int64)
        if water_dense is not None:
            done = np.zeros(len(px), bool)
            for j in range(k):
                ok = ~done & ~crosses_water(px, py, self.px[i[:, j]], self.py[i[:, j]], water_dense, wm)
                choice[ok] = j
                done |= ok
        rows = np.arange(len(px))
        return i[rows, choice], d[rows, choice]


def crosses_water(xa, ya, xb, yb, dense, wm, step=25.0):
    """Whether each segment passes over a water pixel (sampled every `step` metres)."""
    out = np.zeros(len(xa), bool)
    L = np.hypot(xb - xa, yb - ya)
    H, W = dense.shape
    for s0 in range(0, len(xa), 200_000):
        sl = slice(s0, s0 + 200_000)
        k = np.ceil(L[sl] / step).astype(np.int64) + 1
        seg = np.repeat(np.arange(len(k)), k)
        t = (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)) / np.repeat(np.maximum(k - 1, 1), k)
        px = xa[sl][seg] + (xb[sl] - xa[sl])[seg] * t
        py = ya[sl][seg] + (yb[sl] - ya[sl])[seg] * t
        c = np.floor((px - wm.x0) / wm.cell).astype(np.int64)
        r = np.floor((py - wm.y0) / wm.cell).astype(np.int64)
        inb = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        wet = np.zeros(len(px), bool)
        wet[inb] = dense[r[inb], c[inb]]
        out[sl] = np.bincount(seg, weights=wet, minlength=len(k)) > 0
    return out


# ---------------------------------------------------------------- zones

class ZoneSet:
    """The pack's zones, their cells and the window enumeration shared with the sim."""

    def __init__(self, header, a):
        self.zone = a["zone"].astype(np.int64)
        self.gx = a["zone_gx"].astype(np.int64)
        self.gy = a["zone_gy"].astype(np.int64)
        self.n = len(self.gx)
        self.reg = a["zone_region"].astype(np.int64) if "zone_region" in a else np.zeros(self.n, np.int64)
        self.nreg = int(self.reg.max()) + 1
        self.cx, self.cy = a["x_m"].astype(np.float64), a["y_m"].astype(np.float64)
        if "commute_home" in a:
            w = a["commute_home"].astype(np.float64) + a["commute_work"].astype(np.float64)
        else:
            w = a["pop"].astype(np.float64) + a["jobs"].astype(np.float64)
        self.w = w + 1e-3  # every cell counts a little
        order = np.argsort(self.zone, kind="stable")
        self.cells = order
        self.coff = np.zeros(self.n + 1, np.int64)
        self.coff[1:] = np.cumsum(np.bincount(self.zone, minlength=self.n))
        # each zone's representative cell: the one nearest its weighted centre
        wx = np.bincount(self.zone, weights=self.w * self.cx, minlength=self.n)
        wy = np.bincount(self.zone, weights=self.w * self.cy, minlength=self.n)
        ws = np.bincount(self.zone, weights=self.w, minlength=self.n)
        mx, my = wx / ws, wy / ws
        d = np.hypot(self.cx - mx[self.zone], self.cy - my[self.zone])
        rep = np.zeros(self.n, np.int64)
        best = np.full(self.n, np.inf)
        for c in range(len(d)):  # small loop body, fine for 200k
            z = self.zone[c]
            if d[c] < best[z]:
                best[z] = d[c]
                rep[z] = c
        self.rep = rep
        self.rx, self.ry = self.cx[rep], self.cy[rep]
        self.wz = ws
        self.gw, self.gh = int(self.gx.max()) + 1, int(self.gy.max()) + 1
        self.grid = np.full((self.gw, self.gh, self.nreg), -1, np.int64)
        self.grid[self.gx, self.gy, self.reg] = np.arange(self.n)

    def window(self, i, W):
        """Zone ids in zone i's window of W squares, canonical order (dx outer, dy, region)."""
        out = []
        for dx in range(-W, W + 1):
            gx = self.gx[i] + dx
            if gx < 0 or gx >= self.gw:
                continue
            for dy in range(-W, W + 1):
                gy = self.gy[i] + dy
                if gy < 0 or gy >= self.gh:
                    continue
                for r in range(self.nreg):
                    j = self.grid[gx, gy, r]
                    if j >= 0:
                        out.append(j)
        return np.array(out, np.int64)


# ---------------------------------------------------------------- drive factors

_G = {}
NEAR_LIMIT_M = 50_000.0  # a near pair's road beyond this counts as this (its factor a floor)


def _init(path):
    d = np.load(path)
    from scipy.sparse import csr_matrix
    _G["g"] = csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    for k in d.files:
        if k not in ("data", "indices", "indptr", "shape"):
            _G[k] = d[k]


def _near(zs):
    """Dijkstra (to NEAR_LIMIT_M) from the representative of each zone in zs, from its virtual
    node joined to its road point's two junctions: the factor to each zone of its window, its
    cells' road metres over their straight metres, weighted by their commuters."""
    from scipy.sparse.csgraph import dijkstra
    G = _G
    nj = int(G["njunctions"])
    D = dijkstra(G["g"], directed=False, indices=nj + np.asarray(zs), limit=NEAR_LIMIT_M)
    out = []
    for b, i in enumerate(zs):
        d = np.minimum(D[b], NEAR_LIMIT_M)
        J = G["win_ids"][G["win_off"][i]:G["win_off"][i + 1]]
        cs = np.concatenate([G["cells"][G["coff"][j]:G["coff"][j + 1]] for j in J])
        grp = np.repeat(np.arange(len(J)), G["coff"][J + 1] - G["coff"][J])
        S = np.hypot(G["cx"][cs] - G["rx"][i], G["cy"][cs] - G["ry"][i])
        road = np.minimum(d[G["ca"][cs]] + G["cla"][cs], d[G["cb"][cs]] + G["clb"][cs])
        along = np.where(G["cseg"][cs] == G["zseg"][i], np.abs(G["ccum"][cs] - G["zcum"][i]), np.inf)
        R = G["zconn"][i] + np.minimum(road, along) + G["cconn"][cs]
        w = G["w"][cs] * (S > 1.0)
        num = np.bincount(grp, weights=w * R, minlength=len(J))
        den = np.bincount(grp, weights=w * S, minlength=len(J))
        out.append((i, np.where(den > 0, num / np.maximum(den, 1e-9), ROAD_DETOUR)))
    return out


def _far(reps):
    """Full Dijkstra from each supercell's representative zone: road metres to every zone."""
    from scipy.sparse.csgraph import dijkstra
    G = _G
    nj = int(G["njunctions"])
    D = dijkstra(G["g"], directed=False, indices=nj + np.asarray(reps))
    nz = len(G["rx"])
    return [(i, (G["zconn"][i] + D[b, nj:nj + nz] + G["zconn"]).astype(np.float32)) for b, i in enumerate(reps)]


def drive_tables(zs, graph, wm, dense, workers=2, log=print):
    """Near (zone windows) and far (supercells) drive factors."""
    import multiprocessing as mp
    import tempfile
    t0 = time.time()
    zp, zd = graph.snap(zs.rx, zs.ry, dense, wm)
    cp, cd = graph.snap(zs.cx, zs.cy, dense, wm)
    log(f"  snapped {zs.n:,} zones and {len(zs.cx):,} cells to the roads in {time.time() - t0:.0f} s "
        f"(median cell {np.median(cd):.0f} m from a road, 95% {np.quantile(cd, 0.95):.0f} m)", flush=True)
    nj = graph.n
    # a virtual node per zone representative, joined to the two junctions of its road point
    vz = nj + np.arange(zs.n)
    extra = (np.concatenate([vz, vz]), np.concatenate([graph.pa[zp], graph.pb[zp]]),
             np.maximum(np.concatenate([graph.la[zp], graph.lb[zp]]), 0.01))
    g = graph.csr(extra)
    wins = [zs.window(i, DRIVE_WIN) for i in range(zs.n)]
    win_off = np.zeros(zs.n + 1, np.int64)
    win_off[1:] = np.cumsum([len(w) for w in wins])
    win_ids = np.concatenate(wins)
    # supercells per region, numbered in order of their first zone; each one's heaviest zone
    skey = (zs.gx // SUPER) * 100_000 + (zs.gy // SUPER) * 10 + zs.reg
    _, first, sup = np.unique(skey, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    sup = rank[sup.ravel()]
    nsup = len(order)
    srep = np.zeros(nsup, np.int64)
    best = np.full(nsup, -1.0)
    for i in range(zs.n):
        if zs.wz[i] > best[sup[i]]:
            best[sup[i]] = zs.wz[i]
            srep[sup[i]] = i
    tmp = os.path.join(tempfile.gettempdir(), f"t275_graph_{os.getpid()}.npz")
    np.savez(tmp, data=g.data, indices=g.indices, indptr=g.indptr, shape=np.array(g.shape), njunctions=nj,
             zconn=zd * ROAD_DETOUR, zseg=graph.pseg[zp], zcum=graph.pcum[zp],
             ca=graph.pa[cp], cb=graph.pb[cp], cla=graph.la[cp], clb=graph.lb[cp], cseg=graph.pseg[cp],
             ccum=graph.pcum[cp], cconn=cd * ROAD_DETOUR,
             cells=zs.cells, coff=zs.coff, cx=zs.cx, cy=zs.cy, rx=zs.rx, ry=zs.ry, w=zs.w,
             win_off=win_off, win_ids=win_ids)
    near = [None] * zs.n
    Rs = np.zeros((nsup, zs.n), np.float32)
    t0 = time.time()
    with mp.Pool(workers, initializer=_init, initargs=(tmp,)) as pool:
        done = 0
        for res in pool.imap_unordered(_far, [list(srep[s:s + 8]) for s in range(0, nsup, 8)]):
            for i, r in res:
                Rs[sup[i]] = r
            done += len(res)
        log(f"  routes from {nsup} supercells in {time.time() - t0:.0f} s", flush=True)
        t0 = time.time()
        done = 0
        for res in pool.imap_unordered(_near, [list(range(s, min(zs.n, s + 16))) for s in range(0, zs.n, 16)]):
            for i, f in res:
                near[i] = f
            done += len(res)
            if done % 800 < 16 or done == zs.n:
                el = time.time() - t0
                log(f"  routes from {done:,} of {zs.n:,} zones, {el:.0f} s, about {el / done * (zs.n - done):.0f} s left", flush=True)
    os.remove(tmp)
    # near: the mean of the two directions
    from scipy.sparse import coo_matrix
    rows = np.repeat(np.arange(zs.n), np.diff(win_off))
    vals = np.concatenate(near)
    M = coo_matrix((vals, (rows, win_ids)), shape=(zs.n, zs.n)).tocsr()
    fwd = np.asarray(M[rows, win_ids]).ravel()
    back = np.asarray(M[win_ids, rows]).ravel()
    f = 0.5 * (fwd + back)
    near_q = quantise(f)
    # far: from each supercell's representative to every zone, weighted by the zones' commuters
    S = np.hypot(zs.rx[srep][:, None] - zs.rx[None, :], zs.ry[srep][:, None] - zs.ry[None, :])
    good = np.isfinite(Rs) & (S > 1.0)
    w = zs.wz[None, :] * good
    num = np.zeros((nsup, nsup))
    den = np.zeros((nsup, nsup))
    for a_ in range(nsup):
        num[a_] = np.bincount(sup, weights=w[a_] * np.where(good[a_], Rs[a_], 0.0), minlength=nsup)
        den[a_] = np.bincount(sup, weights=w[a_] * S[a_], minlength=nsup)
    unreach = (den > 0) & ~np.isfinite(num)
    num, den = num + num.T, den + den.T
    ff = np.where(den > 0, num / np.maximum(den, 1e-9), ROAD_DETOUR)
    far_q = quantise(ff)
    stats = {"zones": zs.n, "near_pairs": int(len(f)), "supercells": int(nsup),
             "snap_median_m": round(float(np.median(cd)), 1), "near_limit_m": NEAR_LIMIT_M}
    return near_q, sup.astype(np.uint16), far_q.ravel(), stats, (rows, win_ids, f, None)


# ---------------------------------------------------------------- any list of places

def drive_factors(lon, lat, extracts, lon0=None, lat0=None, pad_m=20_000.0):
    """Routed road metres over straight metres between every pair of places (lon, lat arrays),
    over the drivable roads of `extracts` (Geofabrik names), the way the packs' zone tables are
    made but place to place: for zones of another kind or across pack edges (T-276). Returns an
    n x n array (1.3 on the diagonal; the game multiplies a pack's factors by
    `road_routed_scale`, 1.3 / 1.210, and should do the same with these)."""
    from scipy.sparse.csgraph import dijkstra
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    lon0 = float(np.mean(lon)) if lon0 is None else lon0
    lat0 = float(np.mean(lat)) if lat0 is None else lat0
    x, y = local_xy(lon, lat, lon0, lat0)
    ways = [read_ways(e, True) for e in extracts]
    graph = RoadGraph(ways, lon0, lat0, (x.min() - pad_m, y.min() - pad_m, x.max() + pad_m, y.max() + pad_m))
    p, d = graph.snap(x, y)
    n, nj = len(x), graph.n
    v = nj + np.arange(n)
    g = graph.csr((np.concatenate([v, v]), np.concatenate([graph.pa[p], graph.pb[p]]),
                   np.maximum(np.concatenate([graph.la[p], graph.lb[p]]), 0.01)))
    D = dijkstra(g, directed=False, indices=v)[:, nj:nj + n]
    R = d[:, None] * ROAD_DETOUR + D + d[None, :] * ROAD_DETOUR
    S = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    f = np.where(S > 1.0, R / np.maximum(S, 1.0), ROAD_DETOUR)
    return np.where(np.isfinite(f), np.maximum(f, 1.0), FMAX)


# ---------------------------------------------------------------- the pack

def zones_check(zs):
    """The zones the tables are laid out by, as the sim checks them (`routes::zones_check`)."""
    h = 0
    for i in range(zs.n):
        v = int(zs.gx[i]) * 1_000_003 + int(zs.gy[i]) * 1_009 + int(zs.reg[i])
        h = (h + (i + 1) * v) % (1 << 64)
    return h % (1 << 52)


def write_arrays(city, arrays, block_key, block, drop=()):
    """Replace or add `arrays` {name: (np array, dtype name, per)} and header[block_key] = block
    in data/packs/<city>, reading the pack fresh first (other steps may have written it). Arrays
    named in `drop` go (their bytes stay until the next rewrite of the .bin)."""
    from read_pack import load_pack
    header, _ = load_pack(city, PACKS)
    stem = os.path.join(PACKS, city)
    with open(stem + ".bin", "rb") as f:
        buf = bytearray(f.read())
    header["arrays"] = [e for e in header["arrays"] if e["name"] not in arrays and e["name"] not in drop]
    # rewrite the .bin without the bytes of replaced arrays when they are its tail (a rerun)
    gone = [e for e in load_pack(city, PACKS)[0]["arrays"] if e["name"] in arrays or e["name"] in drop]
    if gone:
        start = min(e["offset"] for e in gone)
        if all(e["offset"] < start for e in header["arrays"]):
            del buf[start:]
    for name, (v, dt, per) in arrays.items():
        buf += b"\0" * (-len(buf) % 8)
        header["arrays"].append({"name": name, "dtype": dt, "per": per, "offset": len(buf), "count": int(len(v))})
        buf += np.ascontiguousarray(v).tobytes()
    header[block_key] = block
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    with open(stem + ".json.part", "w") as f:
        json.dump(header, f, indent=2)
    os.replace(stem + ".bin.part", stem + ".bin")
    os.replace(stem + ".json.part", stem + ".json")
    return len(buf)


def cargo_run(bin_name, args):
    import shutil
    import subprocess
    env = dict(os.environ)
    cargo_bin = os.path.join(os.path.expanduser("~"), ".cargo", "bin")
    env["PATH"] = cargo_bin + os.pathsep + env.get("PATH", "")
    env.setdefault("CARGO_BUILD_JOBS", "2")
    env["CARGO_TARGET_DIR"] = os.environ.get("TW_CARGO_TARGET_DIR", os.path.join(ROOT, "data", "work", "target"))
    cargo = shutil.which("cargo", path=env["PATH"])
    if not cargo:
        sys.exit("cargo not found: install Rust (rustup) or put ~/.cargo/bin on PATH")
    # TW_SIM_DIR: a copy of sim/ to build from while the tree's own is mid-edit by another task
    sim = os.environ.get("TW_SIM_DIR", os.path.join(ROOT, "sim"))
    cmd = [cargo, "run", "--quiet", "--release", "--features", "demand", "--bin", bin_name,
           "--manifest-path", os.path.join(sim, "Cargo.toml"), "--"] + args
    subprocess.run(cmd, env=env, check=True)


def build(city, ex_arg=None, keep=False, workers=2, stage="all"):
    """Everything routes.py adds, for one finished pack (any country, given its extracts).
    stage "drive": the raster and the drive tables only; "walk": the walk factors only (needs
    the first stage's raster in the pack)."""
    from read_pack import load_pack
    t_all = time.time()
    if stage == "walk":
        return walk_stage(city, t_all)
    header, a = load_pack(city, PACKS)
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    exs = ex_arg or extracts_of(header)
    print(f"{city}: extracts {', '.join(exs)}", flush=True)
    ways = [read_ways(e, keep) for e in exs]
    wm = water_mask(header, a)
    dense = wm.dense()
    t0 = time.time()
    walk, winfo = walk_raster(header, a, ways)
    print(f"  walk raster {walk.W} x {walk.H} of {WALK_CELL_M:.0f} m: {walk.water_pixels() / (walk.W * walk.H):.1%} barrier, "
          f"{winfo['bridges']:,} walkable bridges opened {winfo['pixels_opened']:,} pixels ({time.time() - t0:.0f} s)", flush=True)
    zs = ZoneSet(header, a)
    t0 = time.time()
    pad = 20_000.0
    box = (wm.x0 - pad, wm.y0 - pad, wm.x0 + wm.W * wm.cell + pad, wm.y0 + wm.H * wm.cell + pad)
    graph = RoadGraph(ways, lon0, lat0, box)
    print(f"  road graph: {graph.n:,} junctions, {graph.edges:,} roads, from {graph.points:,} points ({time.time() - t0:.0f} s)", flush=True)
    t_drive = time.time()
    near_q, sup, far_q, stats, diag = drive_tables(zs, graph, wm, dense, workers=workers)
    stats["seconds"] = round(time.time() - t_drive)
    f = diag[2]
    print(f"  drive factors: near pairs {len(f):,}, median {np.median(f):.3f}, "
          f"share over 1.5 {np.mean(f > 1.5):.1%}, over 2 {np.mean(f > 2):.1%}; {stats['supercells']} supercells ({stats['seconds']} s)", flush=True)
    np.save(os.path.join(ROOT, "data", "work", f"{city}.drive_near_f.npy"), np.column_stack([diag[0], diag[1], f]).astype(np.float32))
    import datetime
    walk_block = walk.header({"source": "the pack's water mask at 50 m (a pixel half water or more), less OSM ways a person may walk over a bridge on",
                              **winfo})
    walk_block["encoding"] = walk_block["encoding"].replace("water_", "walk_")
    block = {
        "built": str(datetime.date.today()),
        "task": "T-275",
        "extracts": exs,
        "zones": zs.n,
        "zones_check": zones_check(zs),
        "osm": "Geofabrik extracts, (c) OpenStreetMap contributors, ODbL",
        "walk": walk_block,
        "drive": {"window": DRIVE_WIN, "super": SUPER, "supercells": stats["supercells"], "regions": zs.nreg,
                  "q": QSCALE, "factor": "routed road metres / straight metres = exp(q / q_scale); straight x factor is the road",
                  "layout": "drive_near: per zone, its window's zones (dx -W..W outer, dy -W..W, region) that exist, one byte each; drive_far: supercells x supercells; drive_super: each zone's supercell",
                  "roads": list(DRIVE_CLASSES), "graph": {"junctions": graph.n, "roads": graph.edges}, **stats},
    }
    write_arrays(city, {"walk_row": (walk.row_off.astype(np.uint32), "u32", "walk"),
                        "walk_x": (walk.xs.astype(np.uint16), "u16", "walk"),
                        "drive_near": (near_q, "u8", "routes"),
                        "drive_super": (sup, "u16", "zone"),
                        "drive_far": (far_q, "u8", "routes")}, "routes", block, drop=("walk_zone",))
    if stage == "all":
        walk_stage(city, t_all)


def walk_stage(city, t_all):
    """The walk factors between zones, by the sim's own walk search (the one the game runs)."""
    from read_pack import load_pack
    out = os.path.join(ROOT, "data", "work", f"{city}.walk_zone.u8")
    t0 = time.time()
    cargo_run("pack_walk", [os.path.join(PACKS, f"{city}.json"), out, "--window", str(WALK_WIN)])
    wz = np.fromfile(out, np.uint8)
    header, _ = load_pack(city, PACKS)
    block = header["routes"]
    block["walk_zone"] = {"window": WALK_WIN, "q": QSCALE, "seconds": round(time.time() - t0),
                          "factor": "walk metres around water / straight metres = exp(q / q_scale), from each zone's representative cell to the window zone's cells",
                          "layout": "as drive_near, with the walk's window"}
    n = write_arrays(city, {"walk_zone": (wz, "u8", "routes")}, "routes", block)
    print(f"{city}: routes written, pack {n / 1e6:.2f} MB, {time.time() - t_all:.0f} s in all", flush=True)


def us_states_on_grid(header):
    """US state FIPS whose TIGER polygon meets the pack's water grid (its extent plus 2 km): the
    bridges and roads just over a state line are part of the walks and drives inside."""
    path = os.path.join(RAW, "tl_2020_us_state.zip")
    if not os.path.exists(path) or "water" not in header:
        return []
    import water
    wb = header["water"]
    (x0, y0), (W, H), c = wb["origin_m"], wb["size"], wb["cell_m"]
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    lo, la = water.to_lonlat(np.array([x0, x0 + W * c]), np.array([y0, y0 + H * c]), lon0, lat0)
    from shapely.geometry import box
    ext = box(lo[0], la[0], lo[1], la[1])
    geoms, recs = water.read_shp(path, (lo[0], la[0], lo[1], la[1]))
    return sorted(r["STATEFP"] for g, r in zip(geoms, recs) if g.intersects(ext))


def extracts_of(header):
    ids = header.get("regions", {}).get("ids", [])
    if "US state" in header.get("regions", {}).get("source", ""):
        ids = sorted(set(ids) | {r for r in us_states_on_grid(header) if r in EXTRACTS})
    missing = [r for r in ids if r not in EXTRACTS]
    if missing:
        raise SystemExit(f"no OSM extract for regions {missing}; add them to EXTRACTS or pass --extracts")
    return sorted({EXTRACTS[r] for r in ids})


def main(argv):
    keep = "--keep-pbf" in argv
    ex_arg = None
    if "--extracts" in argv:
        ex_arg = argv[argv.index("--extracts") + 1].split(",")
    workers = 2
    stage = argv[argv.index("--stage") + 1] if "--stage" in argv else "all"
    if "--workers" in argv:
        workers = int(argv[argv.index("--workers") + 1])
    skip = {",".join(ex_arg)} if ex_arg else set()
    if "--workers" in argv:
        skip.add(str(workers))
    skip.add(stage)
    cities = [a for a in argv if not a.startswith("--") and a not in skip]
    os.makedirs(OSM, exist_ok=True)
    from read_pack import load_pack
    if "--check" in argv:
        # which packs have routes for their present zones (the game ignores stale ones)
        import glob
        names = cities or sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(PACKS, "*.json"))
                                 if "." not in os.path.basename(p)[:-5] and "intercity" not in p)
        for c in names:
            header, a = load_pack(c, PACKS)
            if "zones" not in header or "zone_gx" not in a:
                continue
            rb = header.get("routes")
            if rb is None:
                print(f"{c}: no routes (run pipeline/routes.py {c})")
                continue
            ok = rb.get("zones") == header["zones"] and rb.get("zones_check") == zones_check(ZoneSet(header, a))
            have = [n for n in ("walk_row", "walk_zone", "drive_near", "drive_far") if n in a]
            print(f"{c}: routes {rb.get('built')}, {'current' if ok else 'STALE: made for other zones, rerun pipeline/routes.py ' + c}; arrays {', '.join(have)}")
        return
    if "--download" in argv:
        exs = set(ex_arg or [])
        for c in cities:
            header, _ = load_pack(c, PACKS)
            exs |= set(extracts_of(header))
        for e in sorted(exs):
            read_ways(e, keep)
        return
    for c in cities:
        build(c, ex_arg, keep, workers, stage)


if __name__ == "__main__":
    main(sys.argv[1:])
