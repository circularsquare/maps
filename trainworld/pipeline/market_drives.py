"""T-279: routed drives between the market's places (docs/regional.md 5.1, research/routed-drives.md).

A least-cost path on a world lattice of 1/60 degree cells, as the world water mask: each cell's speed
from GRIP4 road density and GEDTM30 slope, open water at a swimmer's 6 km/h (bridges the mask misses)
or by ferry (a base plus per km, the market's `FERRY` rule), Wikidata's fixed links burnt in as land.
Per pair of places on one landmass within MAX_KM, two ratios to the market's rule (drive minutes and
road km) and the ferry's km, written for the market to read; a pair with no entry keeps the rule.

Sources (put them in data/raw/):
  grip4/{tp1,tp2,total}/grip4_*_dens_m_km2.asc  GRIP4 road density, 5 arc-minutes (Meijer et al. 2018,
      CC BY 4.0; https://dataportaal.pbl.nl/data/GRIP4/GRIP4_density_tp1.zip, _tp2, _total, unzipped)
  gedtm30/gedtm_rf_m_240m_v1.2.tif  as zone_terrain.py
  world_field/wikidata_links.json   as world_field.py (T-375)
  and data/terrain/world.water      pipeline/world_water.py (T-1285)

    market_drives.py slope [--work DIR] [--jobs 4]
        the world's mean slope per cell, cached in the work folder (about an hour on 4 cores; once)
    market_drives.py build [--rows FILE] [--work DIR] [--out FILE] [--jobs 4] [--grip DIR]
        the table. Each place's paths are cached in the work folder by its id and centre, so a rerun
        after new places routes from the new ones alone (minutes); `slope` runs first if not cached.

Defaults: rows app/src/game/market-rows.json, work data/work/market_drives, out
data/packs/market-drives.bin, grip data/raw/grip4.

Out (little-endian): `TWD1`, u32 n, n x (u8 len, id bytes) the places' zone ids, u16 count[n] the
pairs listed under each place (with a later place), then per pair u16 b (index of the later place),
i8 round(64 ln rt), i8 round(64 ln rk); then u32 f and f x (u32 pair index, u16 ferry km, u8 ferries).
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
# the market's ferry (intercity.rs FERRY; perceived minutes at 3 a dollar, the fare shared by two)
FERRY_MIN, FERRY_KMH, FERRY_USD, FERRY_USD_KM = 30.0, 35.0, 50.0, 0.5
FERRY_BASE = (FERRY_MIN + FERRY_USD * 3.0 / 2.0) / SCALE
FERRY_PER_KM = (60.0 / FERRY_KMH + FERRY_USD_KM * 3.0 / 2.0) / SCALE
RECIPE = f"v8 {SPEED_HW} {SPEED_PRI} {SPEED_ROAD} {SPEED_NONE} {SPEED_SWIM} ferry {FERRY_MIN} {FERRY_KMH} {FERRY_USD} {FERRY_USD_KM} max {MAX_KM}"
TILE = 5  # source tiles, degrees
MARGIN_KM = 250.0  # a window's reach beyond its places' box
SWIM_KM = 12.0  # water farther from land: no swimmer (FERRY_BASE / (60 / SPEED_SWIM - FERRY_PER_KM) is 10.4 km)
FERRY_REACH_KM = 60.0  # water farther from land: no ferry
HOKKAIDO_LAT = 41.4

WATER_SRC = TW / "data" / "terrain" / "world.water"
LINKS_SRC = TW / "data" / "raw" / "world_field" / "wikidata_links.json"
DEM_SRC = TW / "data" / "raw" / "gedtm30" / "gedtm_rf_m_240m_v1.2.tif"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gc_km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[1], a[0], b[1], b[0]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_KM * math.asin(min(1.0, math.sqrt(h)))


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


def walk(pred, src, targets, n, wcells, cw, r0):
    """Along each target's path back to the source: land km, ferry km and ferries (a transition
    counts half of each, as its weight does)."""
    cur = targets.astype(np.int64).copy()
    L = np.zeros(len(cur))
    F = np.zeros(len(cur))
    NF = np.zeros(len(cur))
    act = cur != src
    cell = (lambda v: np.where(v < n, v, wcells[np.maximum(v - n, 0)])) if len(wcells) else (lambda v: v)
    for _ in range(100000):
        if not act.any():
            break
        k = np.flatnonzero(act)
        v = cur[k]
        p = pred[v].astype(np.int64)
        bad = p < 0
        if bad.any():
            L[k[bad]] = np.nan
            act[k[bad]] = False
            k, v, p = k[~bad], v[~bad], p[~bad]
        cv, cp = cell(v), cell(p)
        rv, colv = np.divmod(cv, cw)
        rp, colp = np.divmod(cp, cw)
        lat = (r0 + (rv + rp) / 2 + 0.5) / PER - 90
        km = np.hypot((colv - colp) * KX * np.cos(np.radians(lat)) / PER, (rv - rp) * KY / PER)
        fv, fp = v >= n, p >= n
        both = fv & fp
        half = fv != fp
        L[k] += np.where(both, 0, np.where(half, km / 2, km))
        F[k] += np.where(both, km, np.where(half, km / 2, 0))
        NF[k] += np.where(half, 0.5, 0)
        cur[k] = p
        act[k] = p != src
    return L, F, NF


_G = {}


def _layers(work):
    if not _G:
        _G["water"] = np.load(work / "water_burnt.npy", mmap_mode="r")
        _G["slope"] = np.load(work / "slope60.npy", mmap_mode="r")
        g = np.load(work / "grip4.npz")
        _G["grip"] = (g["tp1"], g["tp2"], g["total"])
    return _G


def route_tile(args):
    """Paths from the places of one source tile to their targets: {source id: {target id: [raster
    minutes on land, land km, ferry km, ferries]}}."""
    work, (tc, tr), sources, places = args
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
    cellof = {}
    for pid, (lon, lat) in places.items():
        r, c = int((lat + 90) * PER) - r0, int((lon + 180) * PER) - c0
        if 0 <= r < ch and 0 <= c < cw:
            cellof[pid] = r * cw + c
            water[r, c] = False
    # water far from land: no swim (a ferry is cheaper past SWIM_KM) and no ferry past FERRY_REACH_KM
    from scipy.ndimage import distance_transform_edt
    mid = math.radians((s + n_) / 2)
    shore = distance_transform_edt(water, sampling=(KY / PER, KX * math.cos(mid) / PER))
    cost = np.where(water, np.where(shore <= SWIM_KM, 60.0 / SPEED_SWIM, np.inf), 60.0 / sp)
    del sp
    ferry = water & (shore <= FERRY_REACH_KM)
    del shore
    g, n, wcells = window_graph(r0, ch, cw, cost, ferry)
    del cost
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
        dist, pred = dijkstra(g, directed=True, indices=src, return_predecessors=True)
        tg = np.array([cellof[t] for t in tids])
        d = dist[tg]
        L, F, NF = walk(pred, src, tg, n, wcells, cw, r0)
        res = {}
        for t, dt, l, f, nf in zip(tids, d, L, F, NF):
            if not (np.isfinite(dt) and np.isfinite(l)):
                continue
            nf = int(round(nf))
            m = dt - f * FERRY_PER_KM - nf * FERRY_BASE
            res[t] = [round(float(m), 3), round(float(l), 3), round(float(f), 3), nf]
        out[sid] = res
    return out, (tc, tr), len(sources), (ch, cw), round(time.time() - t0, 1)


# ---------------------------------------------------------------- build

def crossing(a, b):
    """The market's own crossings (Channel, Tsugaru) between rows a and b: those pairs keep the rule."""
    hk = lambda r: r["country"] == "JP" and r["lat"] > HOKKAIDO_LAT
    return (a["country"] == "GB") != (b["country"] == "GB") or hk(a) != hk(b)


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


def build(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(a.rows).read_text(encoding="utf-8"))
    rows = {k: r for k, r in rows.items() if all(math.isfinite(r.get(x, math.nan)) for x in ("lon", "lat"))}
    log(f"{len(rows)} places from {a.rows}")
    make_slope(work, a.jobs)
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
            if c.get("recipe") == RECIPE and abs(c["lon"] - r["lon"]) < 1e-7 and abs(c["lat"] - r["lat"]) < 1e-7:
                cache[k] = c["res"]
    run = set()
    for k, ts in tg.items():
        for t in ts:
            if t not in cache.get(k, {}) and k not in cache.get(t, {}) and k not in run and t not in run:
                run.add(min(k, t))
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
            jobs.append((str(work), key, need, {k: allpos[k] for k in near}))
        jobs.sort(key=lambda j: -len(j[2]))
        done = 0
        with ProcessPoolExecutor(a.jobs) as ex:
            for f in as_completed([ex.submit(route_tile, j) for j in jobs]):
                out, key, ns, shape, secs = f.result()
                for sid, res in out.items():
                    r = rows[sid]
                    (cdir / f"{sid}.json").write_text(json.dumps({"recipe": RECIPE, "lon": r["lon"], "lat": r["lat"], "res": res}), encoding="utf-8")
                    cache[sid] = res
                done += ns
                log(f"tile {key}: {ns} places, window {shape[0]}x{shape[1]}, {secs} s ({done}/{len(run)})")
    write_table(rows, tg, cache, Path(a.out))


def write_table(rows, tg, cache, out: Path):
    ids = sorted(rows, key=lambda k: rows[k]["place"])
    pos = {k: i for i, k in enumerate(ids)}
    per = [[] for _ in ids]
    ferries = []
    q = lambda r: int(max(-127, min(127, round(64 * math.log(r)))))
    stats = {"pairs": 0, "missing": 0, "ferry": 0}
    for k in ids:
        for t in tg[k]:
            if pos[t] <= pos[k]:
                continue
            v = cache.get(k, {}).get(t) or cache.get(t, {}).get(k)
            if v is None:
                stats["missing"] += 1
                continue
            m, L, F, nf = v
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
    buf = bytearray(b"TWD1")
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
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(bytes(buf))
    os.replace(tmp, out)
    log(f"wrote {out}: {stats['pairs']} pairs ({stats['ferry']} with a ferry, {stats['missing']} unrouted), {len(buf) / 1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["slope", "build"])
    ap.add_argument("--rows", default=str(TW / "app" / "src" / "game" / "market-rows.json"))
    ap.add_argument("--work", default=str(TW / "data" / "work" / "market_drives"))
    ap.add_argument("--out", default=str(TW / "data" / "packs" / "market-drives.bin"))
    ap.add_argument("--grip", default=str(TW / "data" / "raw" / "grip4"))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "slope":
        Path(a.work).mkdir(parents=True, exist_ok=True)
        make_slope(Path(a.work), a.jobs)
    else:
        build(a)


if __name__ == "__main__":
    main()
