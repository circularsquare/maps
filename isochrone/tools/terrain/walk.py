"""Directed walking on the 100 m grid and the crossing times of L1 cells (notes/T-011.md).

The 100 m walker: 16 moves as the page's search (js/search.js), a move costing its length times
the mean hours per km of the cells it touches (two, or four for a knight move, which needs all four
passable), divided by the move's terrain factor: MAP's height factor at the mean height of its two
end cells times directed Tobler normalised to the flat on its grade (research/friction.md 2).
Between two graded cells (a walkable way or bridge on the roads grid, urban land) the grade counts
at most ROAD_GRADE either way; a move steeper than G_MAX is not walked.

The crossing of an L1 cell in a direction (T-080's rule, notes/T-087.md): the least time on the
100 m walker from the fine line just outside the cell to its far edge, inside the cell plus a
margin of m fine cells, over the cell's width. Solved for every cell at once by Gauss-Seidel
sweeps over the blocks (forward and back along the crossing until nothing changes), which reaches
the same times as a Dijkstra per cell.
"""
import gzip
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "tools", "surface"))
import common as SC                       # noqa: E402  LC100 speeds

BITS = os.path.join(HERE, "data", "roads", "bits")
N = 1200                                  # fine cells per tile side
K = 10                                    # fine cells per L1 cell side
KM_DEG = 111.195
ROAD_KMH = 5.0
ROAD_GRADE = 0.10                         # streets and paths climb at most this grade (T-084)
G_MAX = 1.0                               # steeper than 45 degrees is not walked
TOBLER_K, TOBLER_G0 = 3.5, 0.05           # directed Tobler normalised to the flat, exp(-k (|g + g0| - g0))
HEIGHT_A, HEIGHT_B = 1.016, 0.0001072     # MAP's height factor, a exp(-b h), h in metres
# the slope rule as the fine level's index.json carries it to the page (tools/fine/build.py, js/levels.js)
SLOPE_RULE = {"toblerK": TOBLER_K, "toblerG0": TOBLER_G0, "heightA": HEIGHT_A, "heightB": HEIGHT_B,
              "roadGrade": ROAD_GRADE, "gMax": G_MAX}
WALK_BITS, BRIDGE = 7, 16
URBAN, WATER = 50, (80, 200)

LC = np.zeros(256, np.float32)
for _k, _v in SC.LC_SPEED.items():
    LC[_k] = _v

# the page's 16 moves (dc, dr), dr down; knight moves with the two cells they pass between
PLAIN = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
KNIGHT = []
for _sx in (1, -1):
    for _sy in (1, -1):
        KNIGHT.append(((_sx, 2 * _sy), (0, _sy), (_sx, _sy)))
        KNIGHT.append(((2 * _sx, _sy), (_sx, 0), (_sx, _sy)))
MOVES = PLAIN + [k[0] for k in KNIGHT]
BETWEEN = {k[0]: (k[1], k[2]) for k in KNIGHT}


def tobler(g):
    """Directed Tobler normalised to the flat: 1 on the flat, 1.19 at 5% down, 0.35 at 30% up."""
    return np.exp(-TOBLER_K * (np.abs(g + TOBLER_G0) - TOBLER_G0))


def height(z):
    return HEIGHT_A * np.exp(-HEIGHT_B * np.maximum(z, 0))


# ---------------------------------------------------------------- inputs

def tile_latlon(name):
    return int(name[1:3]) * (1 if name[0] == "n" else -1), int(name[4:7]) * (1 if name[3] == "e" else -1)


def tile_name(lat, lon):
    return "%s%02d%s%03d" % ("n" if lat >= 0 else "s", abs(lat), "e" if lon >= 0 else "w", abs(lon))


def land_classes(west, north, w, h):
    """LC100 class at the centres of an h x w block of fine cells from (west, north)."""
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(SC.LC100) as src:
        t = src.transform
        lon = west + (np.arange(w) + 0.5) / N
        lat = north - (np.arange(h) + 0.5) / N
        cols = np.floor((lon - t.c) / t.a).astype(np.int64)
        rows = np.floor((lat - t.f) / t.e).astype(np.int64)
        c0, c1, r0, r1 = cols.min(), cols.max() + 1, rows.min(), rows.max() + 1
        a = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), boundless=True, fill_value=0)
    return a[np.ix_(rows - r0, cols - c0)]


def road_bits(name, halo):
    """Road bits of the tile with `halo` fine cells of its neighbours' round it (0 where none)."""
    lat, lon = tile_latlon(name)
    P = N + 2 * halo
    out = np.zeros((P, P), np.uint8)
    for di in (-1, 0, 1):          # tile rows: -1 the one to the north
        for dj in (-1, 0, 1):
            nm = tile_name(lat - di, ((lon + dj + 180) % 360) - 180)
            p = os.path.join(BITS, nm + ".bits.gz")
            if not os.path.exists(p):
                continue
            b = np.frombuffer(gzip.open(p).read(), np.uint8).reshape(N, N)
            # tile (di, dj) covers out rows halo + di N ... ; clip to out
            r0, c0 = halo + di * N, halo + dj * N
            rs, cs = max(0, r0), max(0, c0)
            re, ce = min(P, r0 + N), min(P, c0 + N)
            if rs < re and cs < ce:
                out[rs:re, cs:ce] = b[rs - r0:re - r0, cs - c0:ce - c0]
    return out


def inputs(name, halo, z_tile, z_nb=None):
    """Class, road bits and elevation on the tile plus halo, P x P with P = 1200 + 2 halo. z_tile is
    the tile's elevation at fine centres (glo90.read_fine); z_nb {(di, dj): array} its neighbours'
    where known (di -1 north, dj -1 west); elsewhere the tile's edge is repeated."""
    lat, lon = tile_latlon(name)
    P = N + 2 * halo
    cls = land_classes(lon - halo / N, lat + 1 + halo / N, P, P)
    bits = road_bits(name, halo)
    z = np.pad(z_tile, halo, mode="edge")
    for (di, dj), a in (z_nb or {}).items():
        r0, c0 = halo + di * N, halo + dj * N
        rs, cs = max(0, r0), max(0, c0)
        re, ce = min(P, r0 + N), min(P, c0 + N)
        if rs < re and cs < ce:
            z[rs:re, cs:ce] = a[rs - r0:re - r0, cs - c0:ce - c0]
    north = lat + 1 + halo / N
    return cls, bits, z.astype(np.float32), north


# ---------------------------------------------------------------- move costs

def cell_speeds(cls, bits, roads):
    """km/h of each fine cell before terrain (0 = not walked) and which cells are graded."""
    kmh = LC[cls].copy()
    water = np.isin(cls, WATER)
    graded = cls == URBAN
    if roads:
        walk = ((bits & WALK_BITS) > 0) & ~water
        bridge = water & ((bits & BRIDGE) > 0)
        kmh[walk] = np.maximum(kmh[walk], ROAD_KMH)
        kmh[bridge] = ROAD_KMH
        graded |= walk | bridge
        kmh[water & ~bridge] = 0
    else:
        kmh[water] = 0
    return kmh, graded


def move_costs(cls, bits, z, north, roads, cap=None):
    """{move: P x P float32 hours} the cost of each move from each cell, inf where not walked
    (the target off the grid, a cell not walked, a knight move past a cell not walked, too steep),
    and the cells' hours per km before terrain. A move and its opposite share their length, mean
    and height; only the sign of the grade differs. cap: Tobler's factor at most this (None: the
    full curve, 1.19 at 5% down)."""
    P = cls.shape[0]
    kmh, graded = cell_speeds(cls, bits, roads)
    s = np.where(kmh > 0, 1 / np.maximum(kmh, 1e-6), np.inf).astype(np.float32)
    lat = north - (np.arange(P) + 0.5) / N
    dx = (KM_DEG / N * np.cos(np.radians(lat))).astype(np.float32)
    dy = np.float32(KM_DEG / N)
    # -log of the height factor per cell: the factor at the mean height of two cells is the
    # geometric mean of theirs (it is exponential in height)
    nlh = (np.float32(-np.log(HEIGHT_A)) + np.float32(HEIGHT_B) * np.maximum(z, 0)).astype(np.float32)
    out = {}
    lim = None if cap is None else np.float32(-np.log(cap) / TOBLER_K)   # |g + g0| - g0 at least this
    for mv in MOVES:
        if mv in out:
            continue
        dc, dr = mv
        r0, r1 = max(0, -dr), P - max(0, dr)
        c0, c1 = max(0, -dc), P - max(0, dc)
        U = (slice(r0, r1), slice(c0, c1))
        V = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
        if mv in BETWEEN:
            (a0, b0), (a1, b1) = BETWEEN[mv]
            E0 = (slice(r0 + b0, r1 + b0), slice(c0 + a0, c1 + a0))
            E1 = (slice(r0 + b1, r1 + b1), slice(c0 + a1, c1 + a1))
            base = s[U] + s[V]
            base += s[E0]
            base += s[E1]
            base *= np.float32(0.25)
        else:
            base = s[U] + s[V]
            base *= np.float32(0.5)
        ddx = dc * 0.5 * (dx[r0:r1] + dx[r0 + dr:r1 + dr])[:, None]
        length = np.sqrt(ddx * ddx + (dr * dy) ** 2).astype(np.float32)      # km, (rows, 1)
        base *= length
        g = z[V] - z[U]
        g /= length * np.float32(1000)
        gr = graded[U] & graded[V]
        np.clip(g, -ROAD_GRADE, ROAD_GRADE, out=g, where=gr)
        steep = np.abs(g) > G_MAX
        hh = nlh[U] + nlh[V]
        hh *= np.float32(0.5)
        for sign, key in ((1, mv), (-1, (-dc, -dr))):
            e = g + np.float32(TOBLER_G0 * sign)
            np.abs(e, out=e)
            e -= np.float32(TOBLER_G0)
            if lim is not None:
                np.maximum(e, lim, out=e)
            e *= np.float32(TOBLER_K)
            e += hh
            np.exp(e, out=e)
            e *= base
            e[steep] = np.inf
            c = np.full((P, P), np.inf, np.float32)
            if sign == 1:
                c[U] = e
            else:
                c[V] = e
            out[key] = c
    return out, s


def move_costs_ref(cls, bits, z, north, roads, terrain=True):
    """The plain version of move_costs, for checking it."""
    P = cls.shape[0]
    kmh, graded = cell_speeds(cls, bits, roads)
    s = np.where(kmh > 0, 1 / np.maximum(kmh, 1e-6), np.inf).astype(np.float32)
    lat = north - (np.arange(P) + 0.5) / N
    dx = (KM_DEG / N * np.cos(np.radians(lat))).astype(np.float32)
    dy = np.float32(KM_DEG / N)
    out = {}
    for mv in MOVES:
        dc, dr = mv
        c = np.full((P, P), np.inf, np.float32)
        r0, r1 = max(0, -dr), P - max(0, dr)
        c0, c1 = max(0, -dc), P - max(0, dc)
        U = (slice(r0, r1), slice(c0, c1))
        V = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
        if mv in BETWEEN:
            (a0, b0), (a1, b1) = BETWEEN[mv]
            E0 = (slice(r0 + b0, r1 + b0), slice(c0 + a0, c1 + a0))
            E1 = (slice(r0 + b1, r1 + b1), slice(c0 + a1, c1 + a1))
            mean = (s[U] + s[V] + s[E0] + s[E1]) * np.float32(0.25)
        else:
            mean = (s[U] + s[V]) * np.float32(0.5)
        ddx = dc * 0.5 * (dx[r0:r1] + dx[r0 + dr:r1 + dr])[:, None]
        length = np.sqrt(ddx * ddx + (dr * dy) ** 2).astype(np.float32)      # km, (rows, 1)
        if terrain:
            g = (z[V] - z[U]) / (length * 1000)
            gr = graded[U] & graded[V]
            g = np.where(gr, np.clip(g, -ROAD_GRADE, ROAD_GRADE), g)
            fac = height(0.5 * (z[U] + z[V])) * tobler(g)
            fac = np.where(np.abs(g) > G_MAX, 0, fac).astype(np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                c[U] = np.where(fac > 0, length * mean / fac, np.inf)
        else:
            c[U] = length * mean
        out[mv] = c
    return out, s


# ---------------------------------------------------------------- crossings
#
# A crossing block, turned so the crossing runs along its first axis ("along", a), the second axis
# across it (b), cells last: [a, b, cell]. It covers `along` L1 cells in the direction of travel
# (the cell itself in the middle) and the cell's own K fine rows across, plus m fine cells all round.
# Sources: the fine line just before the first cell (a = m - 1), the cell's rows; target: the far
# edge of the last cell (a = m + along K - 1), the same rows.

def gather(a, cells, halo, r_off, nr, c_off, nc, rows_along):
    """a (P x P) at the fine cells of each L1 cell's block: rows K I + r_off + (0..nr-1), columns
    K J + c_off + (0..nc-1) (tile coordinates), as [a, b, cell] with the rows along when rows_along."""
    I, J = cells
    rows = halo + K * I[None, :] + r_off + np.arange(nr)[:, None]
    cols = halo + K * J[None, :] + c_off + np.arange(nc)[:, None]
    if rows_along:
        return a[rows[:, None, :], cols[None, :, :]]
    return a[rows[None, :, :], cols[:, None, :]]


def axis_blocks(costs, s, cells, halo, m, ew, along=1):
    """Block costs {grid move: [a, b, cell]} and passability for crossings east-west (ew) or
    north-south, unturned (travel to increasing a is east or south); a move leaving the block is inf."""
    A, Bw = along * K + 2 * m, K + 2 * m
    lead = -((along - 1) // 2) * K - m
    args = (-m, Bw, lead, A, False) if ew else (lead, A, -m, Bw, True)
    out = {}
    for mv in MOVES:
        dc, dr = mv
        da, db = (dc, dr) if ew else (dr, dc)
        G = gather(costs[mv], cells, halo, *args)
        if da > 0:
            G[A - da:] = np.inf
        elif da < 0:
            G[:-da] = np.inf
        if db > 0:
            G[:, Bw - db:] = np.inf
        elif db < 0:
            G[:, :-db] = np.inf
        out[mv] = G
    return out, np.isfinite(gather(s, cells, halo, *args))


def orient(G, ok, d):
    """axis_blocks' arrays as views for direction d (E, W, S, N), keyed by the move in block terms
    (da along the crossing, db across), travel to increasing a."""
    ew, flip = d in "EW", d in "WN"
    out = {}
    for (da, db) in MOVES:
        sa = -da if flip else da
        mv = (sa, db) if ew else (db, sa)
        out[(da, db)] = G[mv][::-1] if flip else G[mv]
    return out, (ok[::-1] if flip else ok)


def oriented_blocks(costs, s, cells, halo, m, d, along=1):
    G, ok = axis_blocks(costs, s, cells, halo, m, d in "EW", along)
    return orient(G, ok, d)


FWD = [mv for mv in MOVES if mv[0] > 0]
BWD = [mv for mv in MOVES if mv[0] < 0]


def solve_blocks(CB, ok, m, max_sweeps=12, tol=1e-4):
    """Least time from the source line (a = m - 1, the cell's rows) to every block node, by sweeps
    forward and back along a in turn, each with passes both ways across in every line, until a sweep
    leaves every node as it was (or max_sweeps; 1 = forward only: paths that never turn back).
    Returns (D [a, b, cell], sweeps run)."""
    A, Bw, n = ok.shape
    D = np.full((A, Bw, n), np.inf, np.float32)
    D[m - 1, m:m + K] = np.where(ok[m - 1, m:m + K], 0, np.inf)
    live = np.arange(n)            # blocks the last sweep changed
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
    return D, sweeps


def _sweep(D, CB, fwd):
    """One Gauss-Seidel sweep in place: lines in order along a, each relaxed from the lines behind
    it, then both ways across."""
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


def land_cells(s, halo):
    fin = np.isfinite(s[halo:halo + N, halo:halo + N]).reshape(N // K, K, N // K, K).any(axis=(1, 3))
    return np.nonzero(fin)


def crossings(costs, s, north, halo, m, cells=None, dirs="EWSN", max_sweeps=12, along=1):
    """Crossing slowness (hours per km) of L1 cells in each direction: {d: (ncell,) float32, inf
    where the far edge is not reached}, over `along` cells in the direction of travel (the cell in
    the middle; the time over their length); the cells ((I, J) arrays; default every cell with a
    walked fine cell) and sweeps run per direction. Needs halo >= m + (along - 1) / 2 K."""
    if cells is None:
        cells = land_cells(s, halo)
    I, J = cells
    lat = north - (halo + K * I + K / 2) / N
    span_ew = along * K * KM_DEG / N * np.cos(np.radians(lat))
    span_ns = np.full(len(I), along * K * KM_DEG / N)
    out, its = {}, {}
    for axis in ("EW", "SN"):
        ds = [d for d in dirs if d in axis]
        if not ds:
            continue
        G, okg = axis_blocks(costs, s, cells, halo, m, axis == "EW", along)
        for d in ds:
            cb, ok = orient(G, okg, d)
            D, sw = solve_blocks(cb, ok, m, max_sweeps)
            far = D[m + along * K - 1, m:m + K].min(axis=0)
            out[d] = (far / (span_ew if d in "EW" else span_ns)).astype(np.float32)
            its[d] = sw
            del cb, D
        del G
    return out, cells, its


# ---------------------------------------------------------------- the per-tile product

HALO = 16                                 # fine cells of the neighbours round a tile (3-cell strips)
STRIP = 3                                 # cells along in the consistency strip
ROAD_LANE = 3                             # its margin with roads: a road may wander off the cell's rows
SWEEPS = 2                                # forward and back: exact on 1 cell, ~0.1% mean on strips
DIRS = "EWSN"


def strip_ratio(X1, Xn, n, d):
    """Each cell's one-cell crossing times its n-cell strip over the sum of the strip's one-cell
    crossings, the strip's consistency (120 x 120 arrays, d the direction: neighbours along it);
    where the strip or a neighbour is missing, the cell's own one-cell crossing."""
    nl = X1.shape[0]
    axis = 1 if d in "EW" else 0
    h = (n - 1) // 2
    tot = np.zeros_like(X1)
    good = np.isfinite(X1)
    for k in range(-h, h + 1):
        sh = np.roll(X1, -k, axis=axis)
        sl = [slice(None)] * 2
        if k > 0:
            sl[axis] = slice(nl - k, nl)
            sh[tuple(sl)] = np.inf
        elif k < 0:
            sl[axis] = slice(0, -k)
            sh[tuple(sl)] = np.inf
        good &= np.isfinite(sh)
        tot = tot + np.where(np.isfinite(sh), sh, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(good & np.isfinite(Xn), X1 * Xn * n / tot, X1).astype(np.float32)


def road_cells(bits, cls, halo):
    """L1 cells with a walkable way or bridge in their 3 x 3 (they need the grid with roads)."""
    r = (bits[halo:halo + N, halo:halo + N] & (WALK_BITS | BRIDGE)) > 0
    c = r.reshape(N // K, K, N // K, K).any(axis=(1, 3))
    from scipy.ndimage import maximum_filter
    return maximum_filter(c, size=3, mode="constant")


def tile_l1(name, z, znb=None, engine="js", cap=None, tmp=None):
    """A tile's L1 product: per direction E, W, S, N the crossing speed without roads (one-cell
    crossing scaled by its 3-cell strip, margin 1, notes/T-011.md) and with roads (the same on the
    roads grid with a strip margin of ROAD_LANE, on cells with a road in their 3 x 3; elsewhere the
    plain value), uint8 km/h x 40 (0: not crossed).
    Returns {"bytes": 8 planes of 120 x 120 (plain E W S N, roads E W S N), "stats"}."""
    cls, bits, zz, north = inputs(name, HALO, z, znb)
    nl = N // K
    kmh, _ = cell_speeds(cls, bits, False)
    s_plain = np.where(kmh > 0, 1.0, np.inf)
    cells = land_cells(s_plain, HALO)
    kmh_r, _ = cell_speeds(cls, bits, True)
    near = road_cells(bits, cls, HALO)
    land_r = np.isfinite(np.where(kmh_r > 0, 1.0, np.inf)[HALO:HALO + N, HALO:HALO + N]).reshape(nl, K, nl, K).any(axis=(1, 3))
    rc = np.nonzero(near & land_r)
    jobs = [{"grid": "plain", "along": 1, "m": 1, "cells": cells, "max_sweeps": SWEEPS},
            {"grid": "plain", "along": STRIP, "m": 1, "cells": cells, "max_sweeps": SWEEPS}]
    if len(rc[0]):
        jobs.append({"grid": "roads", "along": 1, "m": 1, "cells": rc, "max_sweeps": SWEEPS})
        jobs.append({"grid": "roads", "along": STRIP, "m": ROAD_LANE, "cells": rc, "max_sweeps": SWEEPS})
    if engine == "js":
        outs, _, tm = crossings_js(cls, bits, zz, north, HALO, jobs, cap=cap, tmp=tmp)
    else:
        outs, tm = [], {}
        prev = None
        for j in jobs:
            if j["grid"] != prev:
                costs, s = move_costs(cls, bits, zz, north, j["grid"] == "roads", cap)
                prev = j["grid"]
            x, _, _ = crossings(costs, s, north, HALO, j["m"], cells=j["cells"], along=j["along"],
                                max_sweeps=j["max_sweeps"])
            outs.append(x)
        del costs
    full = []
    for j, x in zip(jobs, outs):
        f = {}
        for d in DIRS:
            a = np.full((nl, nl), np.inf, np.float32)
            a[j["cells"]] = x[d]
            f[d] = a
        full.append(f)
    planes = []
    plain = {d: strip_ratio(full[0][d], full[1][d], STRIP, d) for d in DIRS}
    for d in DIRS:
        planes.append(plain[d])
    for d in DIRS:
        if len(full) > 2:
            # one-cell road crossings; cells off the road list cross as without roads
            r1 = np.where(near, full[2][d], full[0][d])
            rw = strip_ratio(r1, full[3][d], STRIP, d)
            pr = plain[d]
            planes.append(np.where(near, np.where(np.isfinite(pr), np.minimum(rw, pr), rw), pr))
        else:
            planes.append(plain[d])
    out = np.stack([np.where(np.isfinite(a) & (a > 0), np.clip(np.rint(40 / np.maximum(a, 1e-9)), 1, 255), 0)
                    for a in planes]).astype(np.uint8)
    return {"bytes": out.tobytes(), "stats": {"cells": int(len(cells[0])), "roadcells": int(len(rc[0])),
                                             "t": {k: round(v, 2) for k, v in tm.items()}}}


# ---------------------------------------------------------------- whole-grid search (checks)

def csr(costs, inner=None):
    """A scipy CSR graph of the moves (16 slots a node), on the sub-grid `inner` (r0, r1, c0, c1)."""
    from scipy.sparse import csr_matrix
    P = next(iter(costs.values())).shape[0]
    r0, r1, c0, c1 = inner or (0, P, 0, P)
    h, w = r1 - r0, c1 - c0
    n = h * w
    idx = np.arange(n, dtype=np.int64).reshape(h, w)
    data = np.empty((n, 16), np.float64)
    ind = np.empty((n, 16), np.int32)
    rr, cc = np.divmod(np.arange(n), w)
    for k, (dc, dr) in enumerate(MOVES):
        c = costs[(dc, dr)][r0:r1, c0:c1].ravel().astype(np.float64)
        r2, c2 = rr + dr, cc + dc
        bad = (r2 < 0) | (r2 >= h) | (c2 < 0) | (c2 >= w) | ~np.isfinite(c)
        ind[:, k] = np.where(bad, np.arange(n), np.clip(r2, 0, h - 1) * w + np.clip(c2, 0, w - 1))
        data[:, k] = np.where(bad, 0, c)
    keep = ~((ind == np.arange(n)[:, None]) & (data == 0))
    counts = keep.sum(axis=1)
    indptr = np.concatenate([[0], np.cumsum(counts)])
    return csr_matrix((data[keep], ind[keep], indptr), shape=(n, n))


# ---------------------------------------------------------------- the same in node (cross.mjs)

def crossings_js(cls, bits, z, north, halo, jobs, cap=None, tmp=None):
    """Crossings by tools/terrain/cross.mjs (the rules of move_costs and crossings, ~3x faster).
    jobs: [{grid: 'plain'|'roads', along, m, max_sweeps, cells: (I, J)}]. Returns ([{d: (n,)
    float32}] per job, sweeps per job, node's timings)."""
    import json
    import subprocess
    import tempfile
    P = cls.shape[0]
    hdr = {"P": P, "halo": halo, "north": north, "K": K, "cap": cap, "roadKmh": ROAD_KMH, **SLOPE_RULE,
           "jobs": [{"grid": j["grid"], "along": j["along"], "m": j["m"], "maxSweeps": j.get("max_sweeps", 12),
                     "cells": int(len(j["cells"][0]))} for j in jobs]}
    fd, pin = tempfile.mkstemp(suffix=".in", dir=tmp)
    os.close(fd)
    pout = pin[:-3] + ".out"
    try:
        with open(pin, "wb") as fh:
            fh.write((json.dumps(hdr) + "\n").encode())
            fh.write(np.ascontiguousarray(cls, np.uint8).tobytes())
            fh.write(np.ascontiguousarray(bits, np.uint8).tobytes())
            fh.write(np.ascontiguousarray(z, "<f4").tobytes())
            fh.write(LC.astype("<f4").tobytes())
            for j in jobs:
                fh.write(np.stack([j["cells"][0], j["cells"][1]], 1).astype("<i4").tobytes())
        r = subprocess.run(["node", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cross.mjs"), pin, pout],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("cross.mjs: " + r.stderr[-2000:])
        b = np.fromfile(pout, "<f4")
        outs, o = [], 0
        for j in jobs:
            n = len(j["cells"][0])
            d = {}
            for dd in "EWSN":
                d[dd] = b[o:o + n].copy()
                o += n
            outs.append(d)
        sweeps = b[o:o + len(jobs)].view("<i4").tolist()
        return outs, sweeps, json.loads(r.stderr.strip().splitlines()[-1])
    finally:
        for p in (pin, pout):
            if os.path.exists(p):
                os.remove(p)
