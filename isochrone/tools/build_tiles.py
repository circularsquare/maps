"""The page's walking levels, L1, L3 and L10, from our surface, the roads layer and the directed
terrain planes (T-011).

    python isochrone/tools/build_tiles.py [workers] [--out NAME] [--planes DIR[,DIR...]] [--strips A-B]
    python isochrone/tools/build_tiles.py swap <backup> [--from NAME]   # serve data/NAME/ (default tiles_next)

Inputs: data/surface/walk_today.tif, lc_water.tif, elev.tif, slope.tif, terrain_factor.tif
(docs/surface.md), data/roads/l1/<tile>.bin.gz (docs/roads.md) and the 1 degree tiles of directed
crossing speeds in data/terrain/l1/ (tools/terrain/walk.py tile_l1: no roads E W S N, then with roads
E W S N, uint8 km/h x 40, 0 not crossed), read where they exist. Rules: docs/resolution.md (levels,
water, coarse cells, directions) and docs/roads.md ("The page part").

Two passes, each over the 29 strips of 5 degrees: L1 from the inputs, then L3 and L10 by crossing the
written L1 tiles (a strip needs its neighbours' L1, so the second pass starts once the first is done).
--strips A-B builds strips A..B only (L1 one strip further each way), for tests.

Outputs, under data/NAME/ (default data/tiles_next/) until swapped in:
  tiles/l1/<tr>_<tc>.bin.gz   L1, 600 x 600 cells per 5 degree tile: speed, uint8 km/h x 40, 0 water
  tiles/l3/<tr>_<tc>.bin.gz   L3, 200 x 200
  tiles/r1/, tiles/r3/        the road byte of each cell, same layout
  tiles/d1/, tiles/d3/        four int8 planes E W S N: 32 ln(speed that way / speed), 0 within 3%
  tiles/e1/, tiles/e3/        four int8 planes: the same with roads, against the speed with roads, less d
  tiles/index.json            grid sizes and the tiles that exist, per level and kind
  grid.bin.gz, grid.json      L10, the whole world (4320 x 1740), speed
  grid_roads.bin.gz           L10 road bytes
  grid_dir.bin.gz, grid_dir_roads.bin.gz   L10's d and e planes (4 x 1740 x 4320 each)
A tile of a kind is written where it has a value other than 0.
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
PLANES = os.path.join(DATA, "terrain", "l1")

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
DSCALE = 32                         # direction byte = DSCALE ln(speed that way / speed)
DEAD = 1                            # |direction byte| up to this is written 0
KM_DEG = 111.195                    # as js/levels.js
CROSS_M = 1                         # crossing margin, in L1 cells (T-080)
CROSS_STRIP = 3                     # cells along in the strip consistency (notes/T-011.md 3)
SWEEPS = 4                          # most Gauss-Seidel sweeps a crossing block gets
HALO = 11                           # L1 cells round a 5 degree tile for L10's 3-cell strips (10 + margin)


# ---------------------------------------------------------------- rules

def coarse(hpk, n, f):
    """Each f x f block of L1 cells: summed hours per km and count of the cells n marks. The mean
    rule's speed is count / sum (mean time per metre); land where the count is a quarter of the
    block (docs/resolution.md). The fallback where a coarse cell is not crossed."""
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


def with_roads(sb, rb):
    """The speed byte with the road byte folded in, as js/tiles.js applyRoads (Math.round)."""
    sb = sb.astype(np.float64)
    rb = rb.astype(np.float64)
    land = np.clip(np.floor(sb * RB / np.where(rb > 0, rb, 1) + 0.5), 1, 255)
    water = np.clip(np.floor(rb * ROAD_KMH * SCALE / RB + 0.5), 1, 255)
    return np.where(rb > 0, np.where(sb > 0, land, water), sb).astype(np.uint8)


def directed(v4, u4):
    """Four directed speeds without roads (v4) and with (u4), (4, n) km/h, 0 where not crossed.
    Returns the speed (geometric mean of the crossed directions, the others taking it), the four
    filled, the same with roads (a direction not crossed with roads takes the one without; never
    slower than it), their geometric mean, and which cells have a crossed direction."""
    ok = v4 > 0
    cnt = ok.sum(axis=0)
    lg = np.where(ok, np.log(np.where(ok, v4, 1)), 0).sum(axis=0) / np.maximum(cnt, 1)
    g = np.exp(lg)
    vf = np.where(ok, v4, g)
    uf = np.maximum(np.where(u4 > 0, u4, vf), vf)
    gu = np.exp(np.log(uf).mean(axis=0))
    return g, vf, uf, gu, cnt > 0


def dir_bytes(vf, uf, sb, rb):
    """d and e planes (4, n) int8 of the filled directed speeds against the page's speed byte sb
    and that byte with the road byte rb folded in."""
    def q(x):
        b = np.rint(x)
        b[np.abs(b) <= DEAD] = 0
        return np.clip(b, -127, 127)
    vp = sb.astype(np.float64) / SCALE
    up = with_roads(sb, rb).astype(np.float64) / SCALE
    d = q(DSCALE * np.log(vf / vp))
    s = q(DSCALE * np.log(uf / up))
    return d.astype(np.int8), np.clip(s - d, -127, 127).astype(np.int8)


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


def planes_strip(i, planes):
    """The directed planes of strip i, (8, STRIP, W) uint8, from the 1 degree tiles that have them
    (name -> path); None when none do."""
    out = None
    for j in range(STRIP // 120):
        lat = 84 - TILE_DEG * i - j
        for lon in range(-180, 180):
            p = planes.get(tile_name(lat, lon))
            if p is None:
                continue
            if out is None:
                out = np.zeros((8, STRIP, W), np.uint8)
            b = np.frombuffer(gzip.open(p).read(), np.uint8).reshape(8, 120, 120)
            out[:, j * 120:(j + 1) * 120, (lon + 180) * 120:(lon + 181) * 120] = b
    return out


def write_gz(path, a):
    data = gzip.compress(np.ascontiguousarray(a).tobytes(), compresslevel=6)
    with open(path + ".tmp", "wb") as fh:
        fh.write(data)
    os.replace(path + ".tmp", path)
    return len(data)


def write_tiles(out_dir, i, f, kinds, found, nbytes):
    """kinds {key: array (rows, cols) or (4, rows, cols)} of strip i at level f, cut into tiles."""
    size = SIZE[f]
    for key, g in kinds.items():
        for tc in range(TX):
            t = g[..., tc * size:(tc + 1) * size]
            if not t.any():
                continue
            nbytes[key] = nbytes.get(key, 0) + write_gz(os.path.join(out_dir, "tiles", key, "%d_%d.bin.gz" % (i, tc)), t.view(np.uint8))
            found.setdefault(key, []).append(i * TX + tc)


# ---------------------------------------------------------------- pass 1: L1, one strip of 5 degrees

def l1_strip(args):
    i, small_bits, have, planes, out_dir = args
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
    np.maximum(kmh0, np.float32(1 / SCALE), out=kmh0)   # the slowest a speed byte says
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

    # the directed crossings replace a cell's speed and road ratio wherever the walker crosses it
    P = planes_strip(i, planes)
    dsel = None
    if P is not None:
        dsel = np.nonzero(n0 & (P[:4] > 0).any(axis=0))
        v4 = P[:4][:, dsel[0], dsel[1]].astype(np.float64) / SCALE
        u4 = P[4:][:, dsel[0], dsel[1]].astype(np.float64) / SCALE
        del P
        g, vf, uf, gu, _ = directed(v4, u4)
        kmh0[dsel] = g
        ratio[dsel] = g / gu
    sb = speed_byte(kmh0, n0)
    rb = road_byte(ratio, n0, kmh_b, bridge)
    d1 = np.zeros((4, STRIP, W), np.int8)
    e1 = np.zeros((4, STRIP, W), np.int8)
    if dsel is not None:
        dd, ee = dir_bytes(vf, uf, sb[dsel], rb[dsel])
        d1[:, dsel[0], dsel[1]] = dd
        e1[:, dsel[0], dsel[1]] = ee
    stats = {"land": int(n0.sum()), "small": int(small.sum()), "gap": int(gap.sum()), "bridge": int(bridge.sum()),
             "road": int((rb > 0).sum()), "directed": 0 if dsel is None else int(len(dsel[0]))}
    found, nbytes = {}, {}
    write_tiles(out_dir, i, 1, {"l1": sb, "r1": rb, "d1": d1, "e1": e1}, found, nbytes)
    return i, found, nbytes, stats, time.time() - t0


# ---------------------------------------------------------------- pass 2: L3 and L10 by crossing L1

# the page's 16 moves (dc, dr), dr down (js/search.js); knight moves with the two cells they pass
PLAIN = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
KNIGHT = []
for _sx in (1, -1):
    for _sy in (1, -1):
        KNIGHT.append(((_sx, 2 * _sy), (0, _sy), (_sx, _sy)))
        KNIGHT.append(((2 * _sx, _sy), (_sx, 0), (_sx, _sy)))
MOVES = PLAIN + [k[0] for k in KNIGHT]
BETWEEN = {k[0]: (k[1], k[2]) for k in KNIGHT}
DIRS = "EWSN"


def read_tile(out_dir, key, tr, tc, bands):
    p = os.path.join(out_dir, "tiles", key, "%d_%d.bin.gz" % (tr, tc))
    T = SIZE[1]
    if not os.path.exists(p):
        return np.zeros((bands, T, T), np.uint8)
    return np.frombuffer(gzip.open(p).read(), np.uint8).reshape(bands, T, T)


def l1_window(out_dir, i, tc, h):
    """The written L1 of tile (i, tc) with h cells of its neighbours round it (wrapping at 180
    degrees; water past 85N and 60S): speed, road byte, d, e planes, (bands, 600 + 2h, 600 + 2h)."""
    T = SIZE[1]
    P = T + 2 * h
    out = {k: np.zeros((b, P, P), np.uint8) for k, b in (("l1", 1), ("r1", 1), ("d1", 4), ("e1", 4))}
    for di in (-1, 0, 1):
        tr = i + di
        if tr < 0 or tr >= TY:
            continue
        rs, re = max(0, h + di * T), min(P, h + (di + 1) * T)
        if rs >= re:
            continue
        for dj in (-1, 0, 1):
            cs, ce = max(0, h + dj * T), min(P, h + (dj + 1) * T)
            if cs >= ce:
                continue
            tcw = (tc + dj) % TX
            for k, a in out.items():
                t = read_tile(out_dir, k, tr, tcw, a.shape[0])
                a[:, rs:re, cs:ce] = t[:, rs - h - di * T:re - h - di * T, cs - h - dj * T:ce - h - dj * T]
    return out


def slowness(sb, dirs):
    """Hours per km of each L1 cell in each direction E W S N (4, P, P) float32, from the speed byte
    and its direction bytes (int8, summed); 0 on water. And the land mask."""
    land = sb > 0
    base = np.where(land, SCALE / np.maximum(sb, 1).astype(np.float32), 0).astype(np.float32)
    return base[None] * np.exp(-dirs.astype(np.float32) / DSCALE), land


def move_costs(s4, land, north_row):
    """{move: (P, P) float32 hours} the page's directed cost of each move from each L1 cell (its
    length times the mean over the cells it touches of wx s_x + wy s_y, js/search.js), inf where
    not walked. north_row: the global L1 row of the window's first row."""
    P = land.shape[0]
    lat = 85 - (north_row + np.arange(P) + 0.5) / 120
    dx = (KM_DEG / 120 * np.cos(np.radians(lat))).astype(np.float64)
    dy = KM_DEG / 120
    out = {}
    for mv in MOVES:
        dc, dr = mv
        r0, r1 = max(0, -dr), P - max(0, dr)
        c0, c1 = max(0, -dc), P - max(0, dc)
        ddx = np.abs(dc) * 0.5 * (dx[r0:r1] + dx[r0 + dr:r1 + dr])
        ddy = np.abs(dr) * dy
        length = np.sqrt(ddx * ddx + ddy * ddy)
        wx = (ddx / (ddx + ddy)).astype(np.float32)[:, None]
        sx, sy = s4[0 if dc > 0 else 1], s4[2 if dr > 0 else 3]
        # this move's hours per km in a cell it touches, weighted by the source row's wx
        c = lambda S: wx * sx[S] + (1 - wx) * sy[S]
        U = (slice(r0, r1), slice(c0, c1))
        V = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
        ok = land[U] & land[V]
        if mv in BETWEEN:
            (a0, b0), (a1, b1) = BETWEEN[mv]
            E0 = (slice(r0 + b0, r1 + b0), slice(c0 + a0, c1 + a0))
            E1 = (slice(r0 + b1, r1 + b1), slice(c0 + a1, c1 + a1))
            mean = (c(U) + c(V) + c(E0) + c(E1)) * np.float32(0.25)
            ok &= land[E0] & land[E1]
        else:
            mean = (c(U) + c(V)) * np.float32(0.5)
        a = np.full((P, P), np.inf, np.float32)
        a[U] = np.where(ok, mean * length.astype(np.float32)[:, None], np.inf)
        out[mv] = a
    return out


def gather(a, I, J, h, f, r_off, nr, c_off, nc, rows_along):
    """The nr x nc block of `a` of each coarse cell (I, J), from L1 row h + f I + r_off and column
    h + f J + c_off: [a, b, cell] with a along rows when rows_along, else along columns."""
    rows = h + f * I[None, :] + r_off + np.arange(nr)[:, None]
    cols = h + f * J[None, :] + c_off + np.arange(nc)[:, None]
    if rows_along:
        return a[rows[:, None, :], cols[None, :, :]]
    return a[rows[None, :, :], cols[:, None, :]]


def crossings(costs, passable, I, J, h, f, along, m=CROSS_M):
    """Crossing slowness (hours per km) of the coarse cells (I, J) (f x f L1 cells each, at window
    offset h) in each direction, over `along` cells (the cell in the middle): the least time from
    the L1 cells just before the first cell, its own rows, to the far edge of the last, inside the
    block plus a margin of m L1 cells, as the page's search walks it (T-080's best crossing).
    Returns {d: (n,) float32, inf where not crossed}. The same sweeps as tools/terrain/walk.py."""
    A, Bw = along * f + 2 * m, f + 2 * m
    lead = -((along - 1) // 2) * f - m
    out = {}
    for ew in (True, False):
        args = (-m, Bw, lead, A, False) if ew else (lead, A, -m, Bw, True)
        G = {}
        for mv in MOVES:
            dc, dr = mv
            da, db = (dc, dr) if ew else (dr, dc)
            g = gather(costs[mv], I, J, h, f, *args)
            if da > 0:
                g[A - da:] = np.inf
            elif da < 0:
                g[:-da] = np.inf
            if db > 0:
                g[:, Bw - db:] = np.inf
            elif db < 0:
                g[:, :-db] = np.inf
            G[mv] = g
        okg = gather(passable, I, J, h, f, *args)
        for d in ("EW" if ew else "SN"):
            flip = d in "WN"
            cb = {}
            for (da, db) in MOVES:
                sa = -da if flip else da
                mv = (sa, db) if ew else (db, sa)
                cb[(da, db)] = G[mv][::-1] if flip else G[mv]
            ok = okg[::-1] if flip else okg
            D = solve_blocks(cb, ok, m, f)
            out[d] = D[m + along * f - 1, m:m + f].min(axis=0)
    return out


FWD = [mv for mv in MOVES if mv[0] > 0]
BWD = [mv for mv in MOVES if mv[0] < 0]


def solve_blocks(CB, ok, m, f, max_sweeps=SWEEPS, tol=1e-4):
    A, Bw, n = ok.shape
    D = np.full((A, Bw, n), np.inf, np.float32)
    D[m - 1, m:m + f] = np.where(ok[m - 1, m:m + f], 0, np.inf)
    live = np.arange(n)
    sweeps = 0
    while sweeps < max_sweeps and len(live):
        fwd = sweeps % 2 == 0
        if len(live) == n:
            d, cb = D, CB
        else:
            d, cb = D[:, :, live], {k: v[:, :, live] for k, v in CB.items()}
        before = d.copy() if sweeps else None
        _sweep(d, cb, fwd)
        if d is not D:
            D[:, :, live] = d
        sweeps += 1
        if before is not None:
            live = live[(before > d * (1 + tol)).any(axis=(0, 1))]
    return D


def _sweep(D, CB, fwd):
    A, Bw, n = D.shape
    tmp = np.empty((Bw, n), np.float32)
    down, up = CB[(0, 1)], CB[(0, -1)]
    for a in (range(A) if fwd else range(A - 1, -1, -1)):
        col = D[a]
        for (da, db) in (FWD if fwd else BWD):
            sa = a - da
            if sa < 0 or sa >= A:
                continue
            s0, s1 = max(0, -db), Bw - max(0, db)
            t = tmp[:s1 - s0]
            np.add(D[sa, s0:s1], CB[(da, db)][sa, s0:s1], out=t)
            np.minimum(col[s0 + db:s1 + db], t, out=col[s0 + db:s1 + db])
        dn, upc = down[a], up[a]
        for b in range(1, Bw):
            np.minimum(col[b], col[b - 1] + dn[b - 1], out=col[b])
        for b in range(Bw - 2, -1, -1):
            np.minimum(col[b], col[b + 1] + upc[b + 1], out=col[b])


def strip_scaled(x1, x3, d):
    """Each cell's one-cell crossing times its 3-cell strip over the strip's three one-cell
    crossings (notes/T-011.md 3; times here, not per km); its own one-cell crossing where the strip
    or a neighbour is not crossed. x1 (n + 2, n + 2) with a ring of neighbours, x3 (n, n)."""
    if d in "EW":
        prev, nxt = x1[1:-1, :-2], x1[1:-1, 2:]
    else:
        prev, nxt = x1[:-2, 1:-1], x1[2:, 1:-1]
    own = x1[1:-1, 1:-1]
    tot = prev + own + nxt
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(tot) & np.isfinite(x3), own * x3 / tot, own)


def cross_grid(costs, land, land_c, h, f, n, row0):
    """The four directed speeds (km/h, 0 where not crossed) of the n x n coarse cells of a tile
    window: one-cell crossings on them and a ring round, 3-cell strips on them, strip-scaled.
    Only cells land_c marks ((n + 2, n + 2), ring included) are crossed."""
    x1 = {d: np.full((n + 2, n + 2), np.inf, np.float32) for d in DIRS}
    x3 = {d: np.full((n, n), np.inf, np.float32) for d in DIRS}
    for along, store, off, lm in ((1, x1, -1, land_c), (CROSS_STRIP, x3, 0, land_c[1:-1, 1:-1])):
        chunk = max(1, int(1.5e6 // ((along * f + 2 * CROSS_M) * (f + 2 * CROSS_M))))
        I, J = np.nonzero(lm)
        for s in range(0, len(I), chunk):
            ii, jj = I[s:s + chunk], J[s:s + chunk]
            x = crossings(costs, land, ii + off, jj + off, h, f, along)
            for d in DIRS:
                store[d][ii, jj] = x[d]
    lat = 85 - (row0 + (np.arange(n) + 0.5) * f) / 120
    span_ew = (f * KM_DEG / 120 * np.cos(np.radians(lat)))[:, None]
    span_ns = f * KM_DEG / 120
    out = np.zeros((4, n, n))
    for k, d in enumerate(DIRS):
        x = strip_scaled(x1[d], x3[d], d) / (span_ew if d in "EW" else span_ns)
        with np.errstate(divide="ignore"):
            out[k] = np.where(np.isfinite(x) & (x > 0), 1 / x, 0)
    return out


class Win:
    """The written L1 round one 5 degree tile (l1_window) and what both coarse levels cross on:
    the page's move costs without roads and, made when first wanted, with them."""

    def __init__(self, out_dir, i, tc):
        self.i, self.h = i, HALO
        w = l1_window(out_dir, i, tc, HALO)
        self.sb, self.rb = w["l1"][0], w["r1"][0]
        self.dd, self.ee = w["d1"].view(np.int8), w["e1"].view(np.int8)
        self.sr = with_roads(self.sb, self.rb)
        s0, self.land0 = slowness(self.sb, self.dd)
        self.costs0 = move_costs(s0, self.land0, i * SIZE[1] - HALO)
        self._costs1 = None

    def costs1(self):
        if self._costs1 is None:
            s1, self.land1 = slowness(self.sr, self.dd.astype(np.int16) + self.ee)
            self._costs1 = move_costs(s1, self.land1, self.i * SIZE[1] - HALO)
        return self._costs1


def coarse_tile(wn, f):
    """L3 (f 3) or the L10 part (f 10) of a tile window: speed, road byte, d and e planes, and
    counts. Crossed where the coarse cell is land and the walk crosses it; else the mean rule."""
    from scipy.ndimage import maximum_filter
    h, T = wn.h, SIZE[1]
    n = T // f
    sb, rb, sr = wn.sb, wn.rb, wn.sr
    inner = (slice(h, h + T), slice(h, h + T))
    # the mean rule (the fallback), on the tile's own L1 cells
    n0, n1 = sb[inner] > 0, sr[inner] > 0
    h0, c0, l0 = coarse(np.where(n0, SCALE / np.maximum(sb[inner], 1), 0), n0, f)
    h1, c1, l1 = coarse(np.where(n1, SCALE / np.maximum(sr[inner], 1), 0), n1, f)
    kmh_c0 = np.where(h0 > 0, c0 / np.where(h0 > 0, h0, 1), 0)
    kmh_c1 = np.where(h1 > 0, c1 / np.where(h1 > 0, h1, 1), 0)
    rat = np.where(l0 & (kmh_c1 > 0), kmh_c0 / np.where(kmh_c1 > 0, kmh_c1, 1), 1)
    if not l0.any():
        z = np.zeros((4, n, n), np.int8)
        return (speed_byte(kmh_c0, l0), road_byte(rat, l0, kmh_c1, ~l0 & l1), z, z.copy(),
                {"cells": 0, "crossed": 0})
    # coarse land with a ring round the tile (the quarter rule), for which cells to cross
    ring = slice(h - f, h + T + f)
    q = lambda a: a[ring, ring].reshape(n + 2, f, n + 2, f).sum(axis=(1, 3))
    nl = q(sb > 0) >= MIN_LAND * f * f
    nr = q(sr > 0) >= MIN_LAND * f * f
    row0 = wn.i * T
    v4 = cross_grid(wn.costs0, wn.land0, nl, h, f, n, row0)
    # with roads: crossed again where a road byte is within two coarse cells, else the same
    near = maximum_filter(q(rb > 0) > 0, size=5, mode="constant") & nr
    u4 = v4.copy()
    if near[1:-1, 1:-1].any():
        c1s = wn.costs1()
        ur = cross_grid(c1s, wn.land1, near, h, f, n, row0)
        nn = near[1:-1, 1:-1]
        u4[:, nn] = ur[:, nn]
    g, vf, uf, gu, crossed = directed(v4.reshape(4, -1), u4.reshape(4, -1))
    ok = (crossed & l0.ravel()).reshape(n, n)
    kmh = np.where(ok, g.reshape(n, n), kmh_c0)
    ratio = np.where(ok, (g / gu).reshape(n, n), rat)
    s_b = speed_byte(kmh, l0)
    r_b = road_byte(ratio, l0, kmh_c1, ~l0 & l1)
    d_b = np.zeros((4, n, n), np.int8)
    e_b = np.zeros((4, n, n), np.int8)
    sel = np.nonzero(ok)
    if len(sel[0]):
        flat = sel[0] * n + sel[1]
        a, b = dir_bytes(vf[:, flat], uf[:, flat], s_b[sel], r_b[sel])
        d_b[:, sel[0], sel[1]] = a
        e_b[:, sel[0], sel[1]] = b
    return s_b, r_b, d_b, e_b, {"cells": int(l0.sum()), "crossed": int(ok.sum())}


def coarse_strip(args):
    i, out_dir = args
    t0 = time.time()
    res, stats = {}, {}
    for f in (3, 10):
        n = SIZE[1] // f
        res[f] = (np.zeros((n, W // f), np.uint8), np.zeros((n, W // f), np.uint8),
                  np.zeros((4, n, W // f), np.int8), np.zeros((4, n, W // f), np.int8))
    for tc in range(TX):
        has = [os.path.exists(os.path.join(out_dir, "tiles", k, "%d_%d.bin.gz" % (i, tc))) for k in ("l1", "r1")]
        if not any(has):
            continue
        wn = Win(out_dir, i, tc)
        for f in (3, 10):
            n = SIZE[1] // f
            sl = slice(tc * n, (tc + 1) * n)
            s, r, d, e = res[f]
            s[:, sl], r[:, sl], d[:, :, sl], e[:, :, sl], st = coarse_tile(wn, f)
            for k, v in st.items():
                stats["%s%d" % (k, f)] = stats.get("%s%d" % (k, f), 0) + v
        del wn
    found, nbytes = {}, {}
    s, r, d, e = res[3]
    write_tiles(out_dir, i, 3, {"l3": s, "r3": r, "d3": d, "e3": e}, found, nbytes)
    return i, res[10], found, nbytes, stats, time.time() - t0


# ---------------------------------------------------------------- build and swap

def arg(a, k, default):
    return a[a.index(k) + 1] if k in a else default


def plane_paths(dirs):
    out = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".bin.gz") and len(fn) == 14:
                out.setdefault(fn[:7], os.path.join(d, fn))
    return out


def build(workers, out_dir, plane_dirs, strips):
    from multiprocessing import Pool
    t0 = time.time()
    keys = ("l1", "l3", "r1", "r3", "d1", "d3", "e1", "e3")
    for k in keys:
        os.makedirs(os.path.join(out_dir, "tiles", k), exist_ok=True)
    have = set(f[:-7] for f in os.listdir(ROADS) if f.endswith(".bin.gz"))
    planes = plane_paths(plane_dirs)
    print("road tiles: %d; directed plane tiles: %d" % (len(have), len(planes)), flush=True)
    s_lo, s_hi = strips
    s1 = range(max(0, s_lo - 1), min(TY, s_hi + 2)) if (s_lo, s_hi) != (0, TY - 1) else range(TY)
    with rasterio.open(os.path.join(SURF, "walk_today.tif")) as ds:
        sw = SmallWater(ds, log=lambda *a: None, is_water=lambda a: a == 0)
        print("small water: %d patches (%.0f km2) are waded (%.0f s)"
              % (sw.small_components, sw.small_area_km2, time.time() - t0), flush=True)
        jobs = []
        for i in s1:
            a = ds.read(1, window=Window(0, i * STRIP, W, STRIP))
            jobs.append((i, np.packbits(sw.mask(i, a)), have, planes, out_dir))
    found = {k: [] for k in keys}
    nbytes = {k: 0 for k in keys}
    tot = {}

    def gather_out(fd, nb, st):
        for k in fd:
            found[k] += fd[k]
        for k in nb:
            nbytes[k] += nb[k]
        for k, v in st.items():
            tot[k] = tot.get(k, 0) + v

    with Pool(workers) as pool:
        for i, fd, nb, st, sec in pool.imap_unordered(l1_strip, jobs):
            if s_lo <= i <= s_hi:
                gather_out(fd, nb, st)
            print("L1 strip %d done in %.0f s  (%.0f s)  %s" % (i, sec, time.time() - t0, st), flush=True)
        g10 = np.zeros((H // 10, W // 10), np.uint8)
        r10 = np.zeros((H // 10, W // 10), np.uint8)
        d10 = np.zeros((4, H // 10, W // 10), np.int8)
        e10 = np.zeros((4, H // 10, W // 10), np.int8)
        for i, (g, r, d, e), fd, nb, st, sec in pool.imap_unordered(coarse_strip, [(i, out_dir) for i in range(s_lo, s_hi + 1)]):
            sl = slice(i * 60, (i + 1) * 60)
            g10[sl], r10[sl], d10[:, sl], e10[:, sl] = g, r, d, e
            gather_out(fd, nb, st)
            print("L3, L10 strip %d done in %.0f s  (%.0f s)  %s" % (i, sec, time.time() - t0, st), flush=True)

    index = {"west": -180.0, "east": 180.0, "north": 85.0, "south": -60.0, "scale": SCALE, "roadScale": RB,
             "dirScale": DSCALE, "tileDeg": TILE_DEG, "tilesX": TX, "tilesY": TY, "levels": {}}
    for f in (1, 3):
        index["levels"][str(f)] = {"factor": f, "width": W // f, "height": H // f, "tile": SIZE[f],
                                   "path": "l%d" % f, "tiles": sorted(found["l%d" % f]),
                                   "roads": {"path": "r%d" % f, "tiles": sorted(found["r%d" % f])},
                                   "dirs": {"path": "d%d" % f, "tiles": sorted(found["d%d" % f])},
                                   "dirsRoads": {"path": "e%d" % f, "tiles": sorted(found["e%d" % f])}}
    with open(os.path.join(out_dir, "tiles", "index.json"), "w") as fh:
        json.dump(index, fh, separators=(",", ":"))
    n10 = write_gz(os.path.join(out_dir, "grid.bin.gz"), g10)
    nr10 = write_gz(os.path.join(out_dir, "grid_roads.bin.gz"), r10)
    nd10 = write_gz(os.path.join(out_dir, "grid_dir.bin.gz"), d10.view(np.uint8))
    ne10 = write_gz(os.path.join(out_dir, "grid_dir_roads.bin.gz"), e10.view(np.uint8))
    meta = {"width": W // 10, "height": H // 10, "west": -180.0, "east": 180.0, "north": 85.0,
            "south": -60.0, "scale": SCALE, "factor": 10, "roads": "grid_roads.bin.gz", "roadScale": RB,
            "dirs": "grid_dir.bin.gz", "dirsRoads": "grid_dir_roads.bin.gz", "dirScale": DSCALE}
    with open(os.path.join(out_dir, "grid.json"), "w") as fh:
        json.dump(meta, fh)
    for k in keys:
        print("%s: %d tiles, %.1f MB" % (k, len(found[k]), nbytes[k] / 1e6))
    print("L10: %.2f MB speeds, %.2f MB roads, %.2f + %.2f MB directions; land share %.3f"
          % (n10 / 1e6, nr10 / 1e6, nd10 / 1e6, ne10 / 1e6, (g10 > 0).mean()))
    print("cells: %s" % tot)
    print("done in %.0f s" % (time.time() - t0))


def swap(backup, name):
    """Serve data/<name>/: what is served now moves to data/<backup>/ (kept to compare and roll
    back). Renames only, so no file is ever served half written."""
    src = os.path.join(DATA, name)
    b = os.path.join(DATA, backup)
    if os.path.exists(b):
        sys.exit("%s exists" % b)
    os.makedirs(b)
    files = ("grid_roads.bin.gz", "grid_dir.bin.gz", "grid_dir_roads.bin.gz", "grid.bin.gz", "grid.json")
    for fn in files:
        if os.path.exists(os.path.join(DATA, fn)):
            shutil.copy2(os.path.join(DATA, fn), os.path.join(b, fn))
    os.replace(os.path.join(DATA, "tiles"), os.path.join(b, "tiles"))
    os.replace(os.path.join(src, "tiles"), os.path.join(DATA, "tiles"))
    for fn in files:
        if os.path.exists(os.path.join(src, fn)):
            os.replace(os.path.join(src, fn), os.path.join(DATA, fn))
    os.rmdir(src)
    print("served; the previous build is in %s" % b)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "swap":
        swap(a[1], arg(a, "--from", "tiles_next"))
    else:
        out = os.path.join(DATA, arg(a, "--out", "tiles_next"))
        pdirs = arg(a, "--planes", PLANES).split(",")
        st = arg(a, "--strips", "0-%d" % (TY - 1)).split("-")
        build(int(a[0]) if a and a[0].isdigit() else 1, out, pdirs, (int(st[0]), int(st[1])))
