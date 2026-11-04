"""T-279: routed drives between the market's places (docs/regional.md 5.1, research/routed-drives.md).

A least-cost path on a world lattice of 1/60 degree cells, as the world water mask: each cell's speed
from GRIP4 road density and GEDTM30 slope, open water at a swimmer's 6 km/h (bridges the mask misses)
or by ferry where OSM maps a car ferry (T-2424; a base plus per km, the market's `FERRY` rule),
Wikidata's fixed links burnt in as land. Per pair of places on one landmass within MAX_KM, two ratios
to the market's rule (drive minutes and road km) and the ferry's km, written for the market to read;
a pair with no entry keeps the rule. T-2424: every place is searched from, and each search is read
also at the partner places' market zones, so each zone has its own detour towards each partner over
the centres' (kept where 10% or more off).

Sources (put them in data/raw/):
  grip4/{tp1,tp2,total}/grip4_*_dens_m_km2.asc  GRIP4 road density, 5 arc-minutes (Meijer et al. 2018,
      CC BY 4.0; https://dataportaal.pbl.nl/data/GRIP4/GRIP4_density_tp1.zip, _tp2, _total, unzipped)
  gedtm30/gedtm_rf_m_240m_v1.2.tif  as zone_terrain.py
  world_field/wikidata_links.json   as world_field.py (T-375)
  market_drives/car_ferries.tsv   OSM's car ferries (route=ferry with motor_vehicle or motorcar yes),
      fetched from QLever's OSM planet on the first build ((c) OpenStreetMap contributors, ODbL); the
      work folder's car_ferries.npy is made from it once
  and data/terrain/world.water      pipeline/world_water.py (T-1285); data/packs/<id>.market.json the zones

    market_drives.py slope [--work DIR] [--jobs 4]
        the world's mean slope per cell, cached in the work folder (about an hour on 4 cores; once)
    market_drives.py build [--rows FILE] [--work DIR] [--out FILE] [--jobs 4] [--grip DIR]
        the table. Each place's paths are cached in the work folder by its id and centre, so a rerun
        after new places routes from the new ones alone (minutes); `slope` runs first if not cached.

Defaults: rows app/src/game/market-rows.json, work data/work/market_drives, out
data/packs/market-drives.bin, grip data/raw/grip4.

Out (little-endian): `TWD1`, u32 n, n x (u8 len, id bytes) the places' zone ids, u16 count[n] the
pairs listed under each place (with a later place), then per pair u16 b (index of the later place),
i8 round(64 ln rt), i8 round(64 ln rk); then u32 f and f x (u32 pair index, u16 ferry km, u8 ferries);
then (T-2424, absent in older tables) u32 g and g groups of (u16 place, u16 other place, u16 k, k x
(u16 zone, i8 round(64 ln zt), i8 round(64 ln zk))): the place's zone's minutes and road km towards
the other place over the centres' ratios; with k's top bit set, a dense group: the low bits' zones in
order, each (i8, i8).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import math
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TW = HERE.parent
sys.path.insert(0, str(HERE))

PER = 60
H, W = 180 * PER, 360 * PER
MAX_KM = 1500.0
R_KM = 6371.0088
KX, KY = 111.32, 110.57  # km a degree east at the equator, north
# research/routed-drives.md v8: km/h by road density, slope factor, water
SPEED_HW, SPEED_PRI, SPEED_ROAD, SPEED_NONE, SPEED_SWIM = 110.0, 85.0, 55.0, 20.0, 6.0
# the fitted drive: T = FIX + SCALE x raster minutes, road = KM_SCALE x path km
FIX, SCALE, KM_SCALE = 24.0, 1.252, 1.107
# the market's rule (intercity.rs drive_min, ROAD_DETOUR)
RULE_FIX, RULE_DETOUR, RULE_KMH = 15.0, 1.25, 85.0
# the market's ferry (drives.rs FERRY_*; perceived minutes at 3 a dollar, the fare shared by two)
FERRY_MIN, FERRY_KMH, FERRY_USD, FERRY_USD_KM = 90.0, 30.0, 80.0, 0.5
FERRY_BASE = (FERRY_MIN + FERRY_USD * 3.0 / 2.0) / SCALE
FERRY_PER_KM = (60.0 / FERRY_KMH + FERRY_USD_KM * 3.0 / 2.0) / SCALE
# T-2424: a zone's own detour towards a partner place, kept where 64 ln of either ratio is this far
# from 0 (5% off; `--zone-q` writes fewer) and the straight line from the zone or from its place's centre to the partner's
# centre crosses ZONE_WATER_KM of water (the detour water forces; elsewhere a zone's road is its
# place's, as the size allows); zones nearer the partner's centre than ZONE_MIN_KM keep the centres'
ZONE_Q_MIN, ZONE_MIN_KM, ZONE_WATER_KM = 3, 5.0, 5.0
ZONE_MAX_KM = 400.0  # place pairs farther apart keep the centres' (most entries, little effect)
RECIPE = (f"v9 {SPEED_HW} {SPEED_PRI} {SPEED_ROAD} {SPEED_NONE} {SPEED_SWIM} osm ferry {FERRY_MIN} {FERRY_KMH} {FERRY_USD} {FERRY_USD_KM} "
          f"max {MAX_KM} zones {ZONE_Q_MIN} {ZONE_MIN_KM} {ZONE_WATER_KM} {ZONE_MAX_KM}")
TILE = 5  # source tiles, degrees
MARGIN_KM = 250.0  # a window's reach beyond its places' box
SWIM_KM = 12.0  # water farther from land: no swimmer
HOKKAIDO_LAT = 41.4

WATER_SRC = TW / "data" / "terrain" / "world.water"
LINKS_SRC = TW / "data" / "raw" / "world_field" / "wikidata_links.json"
DEM_SRC = TW / "data" / "raw" / "gedtm30" / "gedtm_rf_m_240m_v1.2.tif"
# OSM's car ferries, ways and route relations, from QLever's OSM planet (seconds; Overpass timed
# out on world and 30 degree queries, 2026-11-04)
FERRIES_SRC = TW / "data" / "raw" / "market_drives" / "car_ferries.tsv"
FERRIES_URL = "https://qlever.cs.uni-freiburg.de/api/osm-planet"
FERRIES_Q = """PREFIX osmkey: <https://www.openstreetmap.org/wiki/Key:>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
SELECT ?osm ?mv ?mc ?geometry WHERE {
  ?osm osmkey:route "ferry" .
  OPTIONAL { ?osm osmkey:motor_vehicle ?mv }
  OPTIONAL { ?osm osmkey:motorcar ?mc }
  FILTER (?mv IN ("yes", "designated", "permissive") || ?mc IN ("yes", "designated", "permissive"))
  ?osm geo:hasGeometry/geo:asWKT ?geometry .
}"""


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gc_km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[1], a[0], b[1], b[0]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_KM * math.asin(min(1.0, math.sqrt(h)))


def gc_km_v(lon, lat, lons, lats):
    la1, lo1, la2, lo2 = math.radians(lat), math.radians(lon), np.radians(lats), np.radians(lons)
    h = np.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.minimum(1.0, np.sqrt(h)))


def rule_min(km):
    return RULE_FIX + km * RULE_DETOUR / RULE_KMH * 60.0


# ---------------------------------------------------------------- layers

def read_water():
    b = WATER_SRC.read_bytes()
    assert b[:4] == b"TWW1", "world.water: not a TWW1 file"
    w, h, per = struct.unpack_from("<III", b, 4)
    assert (w, h, per) == (W, H, PER), (w, h, per)
    out = np.zeros((H, W), bool)
    p = 16

    def leb():
        nonlocal p
        n = s = 0
        while True:
            x = b[p]
            p += 1
            n |= (x & 0x7F) << s
            s += 7
            if not x & 0x80:
                return n
    for r in range(H):
        k = leb()
        last = 0
        e = []
        for _ in range(k):
            last += leb()
            e.append(last)
        for i in range(0, len(e), 2):
            out[r, e[i]:e[i + 1]] = True
    return out


def burn_links(water):
    """Every Wikidata fixed link 0.3-80 km long turns the water within half its length plus 1 km of
    its coordinate into land."""
    j = json.loads(LINKS_SRC.read_text(encoding="utf-8"))["results"]["bindings"]
    n = 0
    for bnd in j:
        c = bnd["coord"]["value"]
        if not c.startswith("POINT("):
            continue
        x, y = map(float, c[6:-1].split())
        L = float(bnd["m"]["value"]) / 1000 if "m" in bnd else 0.0
        if not (0.3 <= L <= 80) or abs(y) > 85:
            continue
        rad = L / 2 + 1.0
        ry = int(math.ceil(rad / KY * PER))
        rx = int(math.ceil(rad / (KX * max(math.cos(math.radians(y)), 0.05)) * PER))
        r0, c0 = int((y + 90) * PER), int((x + 180) * PER)
        ra0, ra1 = max(r0 - ry, 0), min(r0 + ry + 1, H)
        ca0, ca1 = max(c0 - rx, 0), min(c0 + rx + 1, W)
        sub = water[ra0:ra1, ca0:ca1]
        if sub.any():
            yy = (np.arange(ra0, ra1) - r0)[:, None]
            xx = (np.arange(ca0, ca1) - c0)[None, :]
            d = np.hypot(yy / PER * KY, xx / PER * KX * math.cos(math.radians(y)))
            sub[d <= rad] = False
            n += 1
    return n


def ferry_lines():
    """OSM's car ferries as lon/lat polylines (`route=ferry`, `motor_vehicle` or `motorcar` yes,
    designated or permissive): the ways, and the lines of route relations,
    fetched from QLever once into FERRIES_SRC (TSV: id, motor_vehicle, motorcar, WKT)."""
    if not FERRIES_SRC.exists():
        import requests
        ua = {"User-Agent": "trainworld-pipeline/0.1 (rail game data build; python-requests)", "Accept": "text/tab-separated-values"}
        r = requests.post(FERRIES_URL, data={"query": FERRIES_Q}, headers=ua, timeout=900)
        r.raise_for_status()
        FERRIES_SRC.parent.mkdir(parents=True, exist_ok=True)
        tmp = FERRIES_SRC.with_suffix(".tmp")
        tmp.write_text(r.text, encoding="utf-8")
        os.replace(tmp, FERRIES_SRC)
    from shapely import wkt
    out = []
    for ln in FERRIES_SRC.read_text(encoding="utf-8").splitlines()[1:]:
        g = ln.split("\t")[3]
        g = wkt.loads(g[1:g.rindex('"')])
        stack = [g]
        while stack:
            x = stack.pop()
            if x.geom_type == "LineString":
                out.append(list(x.coords))
            elif hasattr(x, "geoms"):
                stack += list(x.geoms)
    return out


def ferry_cells(work: Path):
    """The lattice cells (flat indices, ascending) under OSM's car ferries, cached in the work folder
    (delete car_ferries.npy, and FERRIES_SRC to fetch them again)."""
    path = work / "car_ferries.npy"
    if path.exists():
        return path
    cells = []
    lines = ferry_lines()
    for ln in lines:
        p = np.array(ln, float)
        for (x0, y0), (x1, y1) in zip(p[:-1], p[1:]):
            if abs(x1 - x0) > 180:  # across the antimeridian
                continue
            k = int(math.ceil(max(abs(x1 - x0), abs(y1 - y0)) * PER * 4)) + 1
            t = np.linspace(0, 1, k + 1)
            r = np.floor((y0 + (y1 - y0) * t + 90) * PER).astype(np.int64)
            c = np.floor((x0 + (x1 - x0) * t + 180) * PER).astype(np.int64) % W
            ok = (r >= 0) & (r < H)
            cells.append(r[ok] * W + c[ok])
    out = np.unique(np.concatenate(cells)) if cells else np.zeros(0, np.int64)
    np.save(path, out)
    log(f"car ferries: {len(lines)} lines over {len(out)} cells")
    return path


def read_grip(d: Path):
    import pandas as pd
    out = []
    for name in ("tp1", "tp2", "total"):
        a = pd.read_csv(d / name / f"grip4_{name}_dens_m_km2.asc", skiprows=6, header=None, sep=r"\s+", dtype=np.float32).to_numpy()
        out.append(np.where(a < 0, 0, a)[::-1].copy())  # rows from the south
    return out


# ---------------------------------------------------------------- slope

def slope_block(args):
    """Mean slope (degrees x 4, u8) of the cells of one block of SB degrees, rows from the south."""
    bc, br, sb = args
    from zone_terrain import Dem, M_PER_DEG_LAT, M_PER_DEG_LON
    global _DEM
    if "_DEM" not in globals():
        _DEM = Dem(DEM_SRC)
    dem = _DEM
    c0, r0, cw, ch = bc * sb * PER, br * sb * PER, sb * PER, sb * PER
    w, s = c0 / PER - 180, r0 / PER - 90
    e, n = w + sb, s + sb
    px = dem.px
    pc0 = math.floor((w - dem.lon0) / px) - 1
    pc1 = math.ceil((e - dem.lon0) / px) + 1
    pr0 = math.floor((dem.lat0 - n) / px) - 1
    pr1 = math.ceil((dem.lat0 - s) / px) + 1
    z = dem.window(pr0, pr1, pc0, pc1)
    out = np.zeros((ch, cw), np.uint8)
    if not np.isfinite(z).any():
        return bc, br, out
    lats = dem.lat0 - (np.arange(pr0, pr1) + 0.5) * px
    lons = dem.lon0 + (np.arange(pc0, pc1) + 0.5) * px
    dy = px * M_PER_DEG_LAT
    dx = px * M_PER_DEG_LON * np.cos(np.radians(lats))[:, None]
    gy, gx = np.gradient(z)
    sl = np.degrees(np.arctan(np.hypot(gx / dx, gy / dy)))
    rr = np.floor((lats + 90) * PER).astype(np.int64) - r0
    cc = np.floor((lons + 180) * PER).astype(np.int64) - c0
    ok = np.isfinite(sl) & ((rr >= 0) & (rr < ch))[:, None] & ((cc >= 0) & (cc < cw))[None, :]
    idx = (rr[:, None] * cw + cc[None, :])[ok]
    tot = np.bincount(idx, weights=sl[ok], minlength=cw * ch)
    cnt = np.bincount(idx, minlength=cw * ch)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.nan_to_num(tot / cnt)
    out[:] = np.clip(np.round(m * 4), 0, 255).reshape(ch, cw).astype(np.uint8)
    return bc, br, out


def make_slope(work: Path, jobs: int):
    path = work / "slope60.npy"
    if path.exists():
        return path
    sb = 5
    water = read_water()
    todo = []
    for br in range(180 // sb):
        lat0 = br * sb - 90
        if lat0 + sb <= -65 or lat0 >= 85:
            continue
        for bc in range(360 // sb):
            blk = water[br * sb * PER:(br + 1) * sb * PER, bc * sb * PER:(bc + 1) * sb * PER]
            if not blk.all():
                todo.append((bc, br, sb))
    del water
    log(f"slope: {len(todo)} blocks of {sb} degrees on {jobs} cores")
    tmp = work / "slope60.tmp.npy"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(H, W))
    done = 0
    with ProcessPoolExecutor(jobs) as ex:
        for f in as_completed([ex.submit(slope_block, t) for t in todo]):
            bc, br, a = f.result()
            out[br * sb * PER:(br + 1) * sb * PER, bc * sb * PER:(bc + 1) * sb * PER] = a
            done += 1
            if done % 50 == 0:
                log(f"slope: {done}/{len(todo)}")
    out.flush()
    del out
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------- routing

def speeds(hw, pri, tot, sl):
    s = np.full(hw.shape, SPEED_NONE, np.float32)
    s = np.where(tot >= 100, SPEED_ROAD, s)
    s = np.where(hw + pri >= 25, SPEED_PRI, s)
    s = s / (1 + np.maximum(sl - 3, 0) / 6)
    return np.where(hw >= 25, SPEED_HW, s)


def window_graph(r0, ch, cw, cost, water):
    """Two layers over the window's cells: every cell (water at the swimmer's cost), and a ferry
    layer over the water cells, entered and left from a land neighbour at half the ferry's base each
    way. Edge weight: km x the mean cost a km of its two ends. Both directions stored (scipy's
    undirected search transposes the graph on every call)."""
    from scipy.sparse import coo_matrix
    n = ch * cw
    lat = (r0 + np.arange(ch) + 0.5) / PER - 90
    kx = KX * np.cos(np.radians(lat)) / PER
    ky = KY / PER
    idx = np.arange(n, dtype=np.int64).reshape(ch, cw)
    wflat = water.ravel()
    nw = int(wflat.sum())
    widx = np.full(n, -1, np.int64)
    widx[wflat] = n + np.arange(nw)
    cflat = cost.ravel()
    I, J, Wt = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = idx[0:ch - dr, max(0, -dc):cw - max(0, dc)].ravel()
        b = idx[dr:ch, max(0, dc):cw - max(0, -dc)].ravel()
        ra = np.arange(ch - dr)
        kmr = np.hypot(dc * (kx[ra] + kx[ra + dr]) / 2, dr * ky)
        km = np.repeat(kmr, cw - abs(dc))
        ca, cb = cflat[a], cflat[b]
        wt = km * (ca + cb) / 2
        ok = np.isfinite(wt)
        I.append(a[ok]); J.append(b[ok]); Wt.append(wt[ok])
        wa, wb = wflat[a], wflat[b]
        m = wa & wb
        I.append(widx[a[m]]); J.append(widx[b[m]]); Wt.append(km[m] * FERRY_PER_KM)
        m = ~wa & wb
        I.append(a[m]); J.append(widx[b[m]]); Wt.append(km[m] * (ca[m] + FERRY_PER_KM) / 2 + FERRY_BASE / 2)
        m = wa & ~wb
        I.append(widx[a[m]]); J.append(b[m]); Wt.append(km[m] * (FERRY_PER_KM + cb[m]) / 2 + FERRY_BASE / 2)
    I = np.concatenate(I).astype(np.int32)
    J = np.concatenate(J).astype(np.int32)
    Wt = np.concatenate(Wt)
    g = coo_matrix((np.concatenate([Wt, Wt]), (np.concatenate([I, J]), np.concatenate([J, I]))), shape=(n + nw, n + nw)).tocsr()
    wcells = np.flatnonzero(wflat)
    return g, n, wcells


def path_sums(pred, src, n, wcells, cw, r0):
    """Along every node's path back to the source: land km, ferry km and ferries (a transition
    counts half of each, as its weight does); NaN where no path. Summed up the search's tree by
    pointer doubling."""
    N = len(pred)
    p = pred.astype(np.int64)
    v = np.flatnonzero(p >= 0)
    pv = p[v]
    cell = (lambda x: np.where(x < n, x, wcells[np.maximum(x - n, 0)])) if len(wcells) else (lambda x: x)
    rv, colv = np.divmod(cell(v), cw)
    rp, colp = np.divmod(cell(pv), cw)
    lat = (r0 + (rv + rp) / 2 + 0.5) / PER - 90
    km = np.hypot((colv - colp) * KX * np.cos(np.radians(lat)) / PER, (rv - rp) * KY / PER)
    fv, fp = v >= n, pv >= n
    both, half = fv & fp, fv != fp
    S = np.zeros((3, N))
    S[0, v] = np.where(both, 0, np.where(half, km / 2, km))
    S[1, v] = np.where(both, km, np.where(half, km / 2, 0))
    S[2, v] = np.where(half, 0.5, 0)
    del rv, colv, rp, colp, lat, km, fv, fp, both, half
    anc = np.full(N, -1, np.int64)
    anc[v] = pv
    act = v[pv != src]
    while len(act):
        a = anc[act]
        S[:, act] += S[:, a]
        anc[act] = anc[a]
        act = act[anc[act] >= 0]
    S[0, p < 0] = np.nan
    S[0, src] = 0.0
    return S


_G = {}


def _layers(work):
    if not _G:
        _G["water"] = np.load(work / "water_burnt.npy", mmap_mode="r")
        _G["slope"] = np.load(work / "slope60.npy", mmap_mode="r")
        g = np.load(work / "grip4.npz")
        _G["grip"] = (g["tp1"], g["tp2"], g["total"])
        _G["ferry"] = np.load(work / "car_ferries.npy")
    return _G


def water_km(water, r0, c0, a, lons, lats):
    """Km of the window's water on the straight line from `a` (lon, lat) to each (lons, lats), sampled
    every half cell."""
    lons, lats = np.asarray(lons, float), np.asarray(lats, float)
    if not len(lons):
        return np.zeros(0)
    d = gc_km_v(a[0], a[1], lons, lats)
    k = np.ceil(d / (KY / PER / 2)).astype(np.int64) + 1
    line = np.repeat(np.arange(len(lons)), k)
    t = (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)) / np.repeat(np.maximum(k - 1, 1), k)
    r = np.clip(np.floor((a[1] + (lats[line] - a[1]) * t + 90) * PER).astype(np.int64) - r0, 0, water.shape[0] - 1)
    c = np.clip(np.floor((a[0] + (lons[line] - a[0]) * t + 180) * PER).astype(np.int64) - c0, 0, water.shape[1] - 1)
    return np.bincount(line, water[r, c], len(lons)) * d / k


def zone_q(m, L, nf, d, c, wet):
    """Zones' (q time, q km) towards the source over the centres' `c` (m, L, nf, d): rows (zone,
    qt, qk) with either 10% or more off, where `wet(zones)` (the water on their lines, km) or the
    centres' line crosses ZONE_WATER_KM of water; zones on another ferry count, unreachable or too
    near the source keep the centres'."""
    T = FIX + SCALE * m
    rt = T / rule_min(d)
    rk = np.maximum(KM_SCALE * L, 0.01 * d) / (RULE_DETOUR * d)
    cm, cL, cnf, cd, cwet = c
    crt = (FIX + SCALE * cm) / rule_min(cd)
    crk = max(KM_SCALE * cL, 0.01 * cd) / (RULE_DETOUR * cd)
    ok = np.isfinite(T) & np.isfinite(L) & (d >= ZONE_MIN_KM) & (nf == cnf)
    with np.errstate(invalid="ignore", divide="ignore"):
        qt = np.clip(np.rint(64 * np.log(rt / crt)), -127, 127)
        qk = np.clip(np.rint(64 * np.log(rk / crk)), -127, 127)
    keep = np.flatnonzero(ok & ((np.abs(qt) >= ZONE_Q_MIN) | (np.abs(qk) >= ZONE_Q_MIN)))
    if cwet < ZONE_WATER_KM:
        keep = keep[wet(keep) >= ZONE_WATER_KM]
    return [[int(k), int(qt[k]), int(qk[k])] for k in keep]


def route_tile(args):
    """Paths from the places of one source tile to their targets: {source id: {target id: [raster
    minutes on land, land km, ferry km, ferries, the target's zone hash, its zones' detours towards
    the source (`zone_q`)]}}. `zones`: each target's zones' (lon, lat, hash)."""
    work, (tc, tr), sources, places, zones = args
    work = Path(work)
    t0 = time.time()
    G = _layers(work)
    # the window: the box of the places and their targets, widened by MARGIN_KM for detours
    lons = [p[0] for p in places.values()]
    lats = [p[1] for p in places.values()]
    s, n_ = max(min(lats) - MARGIN_KM / KY, -85), min(max(lats) + MARGIN_KM / KY, 85)
    dlon = MARGIN_KM / (KX * math.cos(math.radians(min(max(abs(s), abs(n_)), 80))))
    w, e = max(min(lons) - dlon, -180), min(max(lons) + dlon, 180)
    r0, r1 = int(math.floor((s + 90) * PER)), int(math.ceil((n_ + 90) * PER))
    c0, c1 = int(math.floor((w + 180) * PER)), int(math.ceil((e + 180) * PER))
    ch, cw = r1 - r0, c1 - c0
    water = np.array(G["water"][r0:r1, c0:c1])
    hw_, pri_, tot_ = G["grip"]
    rr = (np.arange(r0, r1) // 5)[:, None]
    cc = (np.arange(c0, c1) // 5 % 4320)[None, :]
    sl = np.asarray(G["slope"][r0:r1, c0:c1], np.float32) / 4
    sp = speeds(hw_[rr, cc], pri_[rr, cc], tot_[rr, cc], sl)
    del sl
    # water far from land: no swim; a ferry only where OSM maps a car ferry (T-2424)
    from scipy.ndimage import distance_transform_edt
    mid = math.radians((s + n_) / 2)
    shore, near = distance_transform_edt(water, sampling=(KY / PER, KX * math.cos(mid) / PER), return_indices=True)
    # a place's centre on its cell, or the nearest land to one on water (T-2424: Windsor-Essex's is
    # out in Lake St Clair, and a cell made land there was swum to)
    cellof = {}
    for pid, (lon, lat) in places.items():
        r, c = int((lat + 90) * PER) - r0, int((lon + 180) * PER) - c0
        if 0 <= r < ch and 0 <= c < cw:
            cellof[pid] = int(near[0][r, c]) * cw + int(near[1][r, c])
    cost = np.where(water, np.where(shore <= SWIM_KM, 60.0 / SPEED_SWIM, np.inf), 60.0 / sp)
    del sp, shore
    fc = G["ferry"]
    fr, fcol = np.divmod(fc[(fc >= r0 * W) & (fc < r1 * W)], W)
    ok = (fcol >= c0) & (fcol < c1)
    ferry = np.zeros_like(water)
    ferry[fr[ok] - r0, fcol[ok] - c0] = True
    ferry &= water
    g, n, wcells = window_graph(r0, ch, cw, cost, ferry)
    del cost, ferry
    # each target's zones on the lattice: their cell, or the nearest land to one on water
    zcell = {}
    for t, (zlon, zlat, _) in zones.items():
        r = np.clip(np.floor((np.asarray(zlat) + 90) * PER).astype(np.int64) - r0, 0, ch - 1)
        c = np.clip(np.floor((np.asarray(zlon) + 180) * PER).astype(np.int64) - c0, 0, cw - 1)
        zcell[t] = near[0][r, c].astype(np.int64) * cw + near[1][r, c]
    del near
    from scipy.sparse.csgraph import dijkstra
    out = {}
    for sid, tids in sources.items():
        if sid not in cellof:
            continue
        tids = [t for t in tids if t in cellof]
        if not tids:
            out[sid] = {}
            continue
        src = cellof[sid]
        slon, slat = places[sid]
        dist, pred = dijkstra(g, directed=True, indices=src, return_predecessors=True)
        S = path_sums(pred, src, n, wcells, cw, r0)
        del pred
        res = {}
        for t in tids:
            v = cellof[t]
            dt, l, f, nf = dist[v], S[0, v], S[1, v], int(round(S[2, v]))
            if not (np.isfinite(dt) and np.isfinite(l)):
                res[t] = None  # no road between: the rule's (routed, so not searched again)
                continue
            m = dt - f * FERRY_PER_KM - nf * FERRY_BASE
            zq = []
            if t in zcell and gc_km((slon, slat), places[t]) <= ZONE_MAX_KM:
                zv = zcell[t]
                zlon, zlat, zh = zones[t]
                znf = np.rint(S[2, zv]).astype(np.int64)
                zm = dist[zv] - S[1, zv] * FERRY_PER_KM - znf * FERRY_BASE
                zlon, zlat = np.asarray(zlon), np.asarray(zlat)
                zd = gc_km_v(slon, slat, zlon, zlat)
                cwet = float(water_km(water, r0, c0, (slon, slat), [places[t][0]], [places[t][1]])[0])
                wet = lambda ks: water_km(water, r0, c0, (slon, slat), zlon[ks], zlat[ks])
                zq = zone_q(zm, S[0, zv], znf, zd, (m, l, nf, gc_km((slon, slat), places[t]), cwet), wet)
            res[t] = [round(float(m), 3), round(float(l), 3), round(float(f), 3), nf, zones[t][2] if t in zones else "", zq]
        out[sid] = res
    return out, (tc, tr), len(sources), (ch, cw), round(time.time() - t0, 1)


# ---------------------------------------------------------------- build

def crossing(a, b):
    """The market's own crossings (Channel, Tsugaru) between rows a and b: those pairs keep the rule."""
    hk = lambda r: r["country"] == "JP" and r["lat"] > HOKKAIDO_LAT
    return great_britain(a) != great_britain(b) or hk(a) != hk(b)


def great_britain(r):
    """The UK less Northern Ireland, which is on Ireland's landmass (T-2423; intercity.rs great_britain)."""
    return r["country"] == "GB" and not (r["lon"] < -5.4 and r["lat"] < 55.4)


def targets_of(rows):
    """Per place, the places it is paired with (same landmass, within MAX_KM, no market crossing)."""
    ids = list(rows)
    by_road = {}
    for k in ids:
        by_road.setdefault(rows[k]["road"], []).append(k)
    tg = {k: [] for k in ids}
    for road, ks in by_road.items():
        lon = np.radians([rows[k]["lon"] for k in ks])
        lat = np.radians([rows[k]["lat"] for k in ks])
        for i, k in enumerate(ks):
            h = np.sin((lat - lat[i]) / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin((lon - lon[i]) / 2) ** 2
            d = 2 * R_KM * np.arcsin(np.minimum(1, np.sqrt(h)))
            for j in np.flatnonzero(d <= MAX_KM):
                if j != i and not crossing(rows[k], rows[ks[j]]):
                    tg[k].append(ks[j])
    return tg


def read_zones(args):
    """A market pack's zone centres as the market takes them (intercity.rs: sites weighted by people
    plus attraction, people x 2 + jobs x 2, in the pack's frame, then to lon/lat), rounded to 1e-5
    degree, and a hash of them."""
    import hashlib
    k, path, key = args
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    fr = j["frame"]
    nz = len(j["zones"])
    s = np.array([x[:5] for x in j["sites"]], np.float64).reshape(-1, 5)
    zi = s[:, 4].astype(np.int64)
    w = 2 * s[:, 2] + 2 * s[:, 3]
    tw = np.maximum(np.bincount(zi, w, nz), 1e-12)
    x = np.bincount(zi, w * s[:, 0].astype(np.float32), nz) / tw
    y = np.bincount(zi, w * s[:, 1].astype(np.float32), nz) / tw
    if fr["kind"] == "equirect":
        lat0 = math.radians(fr["lat"])
        lon = fr["lon"] + np.degrees(x / (R_KM * 1000 * math.cos(lat0)))
        lat = fr["lat"] + np.degrees(y / (R_KM * 1000))
    else:
        from pyproj import Transformer
        tr = Transformer.from_crs(f"+proj=lcc +lat_1={fr['lat']} +lat_0={fr['lat']} +lon_0={fr['lon']} +R={R_KM * 1000}", "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(x, y)
    lon, lat = [round(float(v), 5) for v in lon], [round(float(v), 5) for v in lat]
    h = hashlib.md5(json.dumps([lon, lat]).encode()).hexdigest()[:12]
    return k, {"key": key, "lon": lon, "lat": lat, "hash": h}


def zone_centres(rows, work: Path, packs: Path, jobs: int):
    """{place: (lon list, lat list, hash)} of each place's market zones, cached by file in the work
    folder (zones.json)."""
    cpath = work / "zones.json"
    old = json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else {}
    out, todo = {}, []
    for k in rows:
        f = packs / f"{k}.market.json"
        if not f.exists():
            continue
        st = f.stat()
        key = [st.st_mtime_ns, st.st_size]
        if k in old and old[k]["key"] == key:
            out[k] = old[k]
        else:
            todo.append((k, str(f), key))
    if todo:
        log(f"zones: reading {len(todo)} market packs")
        with ProcessPoolExecutor(jobs) as ex:
            for k, z in ex.map(read_zones, todo, chunksize=16):
                out[k] = z
        tmp = cpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(out), encoding="utf-8")
        os.replace(tmp, cpath)
    bad = [k for k in out if len(out[k]["lon"]) != rows[k].get("zones", len(out[k]["lon"]))]
    if bad:
        log(f"zones: {len(bad)} packs whose zones differ from their row, no zone detours: {' '.join(sorted(bad)[:20])}")
    return {k: (z["lon"], z["lat"], z["hash"]) for k, z in out.items() if k not in bad}


def build(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(a.rows).read_text(encoding="utf-8"))
    rows = {k: r for k, r in rows.items() if all(math.isfinite(r.get(x, math.nan)) for x in ("lon", "lat"))}
    log(f"{len(rows)} places from {a.rows}")
    make_slope(work, a.jobs)
    import hashlib
    recipe = RECIPE + " ferries " + hashlib.md5(ferry_cells(work).read_bytes()).hexdigest()[:10]
    zones = zone_centres(rows, work, Path(a.packs), a.jobs)
    if not (work / "water_burnt.npy").exists():
        water = read_water()
        n = burn_links(water)
        np.save(work / "water_burnt.npy", water)
        log(f"water: {n} fixed links burnt in")
    if not (work / "grip4.npz").exists():
        hw, pri, tot = read_grip(Path(a.grip))
        np.savez(work / "grip4.npz", tp1=hw, tp2=pri, total=tot)
    tg = targets_of(rows)
    npairs = sum(len(v) for v in tg.values()) // 2
    log(f"{npairs} place pairs within {MAX_KM:.0f} km on one landmass")
    # cached paths: each source's by its centre and the recipe
    cdir = work / "sources"
    cdir.mkdir(exist_ok=True)
    cache = {}
    for k, r in rows.items():
        f = cdir / f"{k}.json"
        if f.exists():
            c = json.loads(f.read_text(encoding="utf-8"))
            if c.get("recipe") == recipe and abs(c["lon"] - r["lon"]) < 1e-7 and abs(c["lat"] - r["lat"]) < 1e-7:
                cache[k] = c["res"]
    # T-2424: every place is searched from (its search gives its partners' zones' detours towards
    # it), again when a partner within ZONE_MAX_KM is new or its zones moved (a farther one's pair
    # comes from its own search)
    zh = lambda t: zones[t][2] if t in zones else ""
    near_ = lambda k, t: gc_km((rows[k]["lon"], rows[k]["lat"]), (rows[t]["lon"], rows[t]["lat"])) <= ZONE_MAX_KM

    def stale(k, t):
        if t not in cache[k]:
            return near_(k, t) or (k not in cache.get(t, {}) and t not in run_new)
        return cache[k][t] is not None and cache[k][t][4] != zh(t) and near_(k, t)
    run_new = {k for k, ts in tg.items() if ts and k not in cache}
    run = {k for k, ts in tg.items() if ts and (k not in cache or any(stale(k, t) for t in ts))}
    log(f"{len(cache)} places cached, {len(run)} to route")
    if run:
        tiles = {}
        for k in sorted(run):
            r = rows[k]
            key = (int((r["lon"] + 180) // TILE), int((r["lat"] + 90) // TILE))
            tiles.setdefault(key, []).append(k)
        allpos = {k: (r["lon"], r["lat"]) for k, r in rows.items()}
        jobs = []
        for key, ks in tiles.items():
            need = {k: tg[k] for k in ks}
            near = set(ks) | {t for k in ks for t in tg[k]}
            tz = {t for k in ks for t in tg[k] if t in zones}
            jobs.append((str(work), key, need, {k: allpos[k] for k in near}, {t: zones[t] for t in tz}))
        jobs.sort(key=lambda j: -len(j[2]))
        done = 0
        with ProcessPoolExecutor(a.jobs) as ex:
            for f in as_completed([ex.submit(route_tile, j) for j in jobs]):
                out, key, ns, shape, secs = f.result()
                for sid, res in out.items():
                    r = rows[sid]
                    (cdir / f"{sid}.json").write_text(json.dumps({"recipe": recipe, "lon": r["lon"], "lat": r["lat"], "res": res}), encoding="utf-8")
                    cache[sid] = res
                done += ns
                log(f"tile {key}: {ns} places, window {shape[0]}x{shape[1]}, {secs} s ({done}/{len(run)})")
    write_table(rows, tg, cache, Path(a.out), max(a.zone_q, ZONE_Q_MIN))


def write_table(rows, tg, cache, out: Path, zone_q=ZONE_Q_MIN):
    ids = sorted(rows, key=lambda k: rows[k]["place"])
    pos = {k: i for i, k in enumerate(ids)}
    per = [[] for _ in ids]
    ferries = []
    groups = []
    q = lambda r: int(max(-127, min(127, round(64 * math.log(r)))))
    stats = {"pairs": 0, "missing": 0, "ferry": 0, "zones": 0}
    for k in ids:
        for t in tg[k]:
            if pos[t] <= pos[k]:
                continue
            v = cache.get(k, {}).get(t) or cache.get(t, {}).get(k)
            if v is None:
                stats["missing"] += 1
                continue
            # each place's zones towards the other, from the other's search
            for src, z in ((t, k), (k, t)):
                zq = [e for e in (cache.get(src, {}).get(z) or [None] * 6)[5] or [] if max(abs(e[1]), abs(e[2])) >= zone_q]
                if zq:
                    groups.append((pos[z], pos[src], zq))
                    stats["zones"] += len(zq)
            m, L, F, nf = v[:4]
            a, b = rows[k], rows[t]
            d = gc_km((a["lon"], a["lat"]), (b["lon"], b["lat"]))
            T = FIX + SCALE * m
            K = KM_SCALE * L
            rt = T / rule_min(d)
            rk = max(K, 0.01 * d) / (RULE_DETOUR * d)
            per[pos[k]].append((pos[t], q(rt), q(rk)))
            if nf > 0:
                ferries.append((stats["pairs"], int(round(F)), nf))
                stats["ferry"] += 1
            stats["pairs"] += 1
    buf = bytearray(b"TWD1")  # the zone section trails, so a reader from before T-2424 still reads the rest
    buf += struct.pack("<I", len(ids))
    for k in ids:
        e = k.encode("utf-8")
        buf += struct.pack("<B", len(e)) + e
    buf += struct.pack(f"<{len(ids)}H", *[len(p) for p in per])
    for p in per:
        for b, lt, lk in p:
            buf += struct.pack("<Hbb", b, lt, lk)
    buf += struct.pack("<I", len(ferries))
    for i, km, nf in ferries:
        buf += struct.pack("<IHB", i, min(km, 65535), min(nf, 255))
    buf += struct.pack("<I", len(groups))
    for a, o, zq in groups:
        nz = int(rows[ids[a]].get("zones", 0))
        if 2 * nz < 4 * len(zq) and nz < 0x8000:  # most of the place's zones: all of them, in order
            q = np.zeros((nz, 2), np.int8)
            for zk, qt, qk in zq:
                q[zk] = (qt, qk)
            buf += struct.pack("<HHH", a, o, 0x8000 | nz) + q.tobytes()
            stats["dense"] = stats.get("dense", 0) + 1
            continue
        buf += struct.pack("<HHH", a, o, len(zq))
        for zk, qt, qk in zq:
            buf += struct.pack("<Hbb", zk, qt, qk)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(bytes(buf))
    os.replace(tmp, out)
    log(f"wrote {out}: {stats['pairs']} pairs ({stats['ferry']} with a ferry, {stats['missing']} unrouted), "
        f"{stats['zones']} zone detours in {len(groups)} groups ({stats.get('dense', 0)} dense), {len(buf) / 1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["slope", "build"])
    ap.add_argument("--rows", default=str(TW / "app" / "src" / "game" / "market-rows.json"))
    ap.add_argument("--work", default=str(TW / "data" / "work" / "market_drives"))
    ap.add_argument("--out", default=str(TW / "data" / "packs" / "market-drives.bin"))
    ap.add_argument("--grip", default=str(TW / "data" / "raw" / "grip4"))
    ap.add_argument("--packs", default=str(TW / "data" / "packs"))
    ap.add_argument("--zone-q", type=int, default=ZONE_Q_MIN, help="write zone detours with 64 ln this far from 0 or more")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "slope":
        Path(a.work).mkdir(parents=True, exist_ok=True)
        make_slope(Path(a.work), a.jobs)
    else:
        build(a)


if __name__ == "__main__":
    main()
