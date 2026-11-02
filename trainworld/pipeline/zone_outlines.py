"""T-800: the coarse outline of every registered zone, `data/packs/outlines.json`.

    python pipeline/zone_outlines.py [--tol METRES] [--out FILE] [--no-water] [--box W,S,E,N]
                                     [--jobs N] [--cache DIR | --no-cache]

The map draws a zone's outline from its pack's `<id>.boundary.geojson` once the view comes near it
(map/cityBoundaries.ts); this file is what it draws until then, so every zone has an outline from the
start. It is cut from the same boundary files, for the registry's zones in registry order, and written
whole each time, so a rerun after zones are added is all the update there is; world_run.py's join and
docs/world-run.md say when.

T-1977: `--jobs` spreads the work over that many processes, and the sea's work (nearly all of the time)
is kept in `--cache` (default data/work/zone_outlines_cache, about 55 MB; see zone_water.py), so a run
redoes only the windows whose zones changed. The bytes written do not depend on either. With 4 jobs:
about 4.5 minutes with an empty cache, 20-30 s with nothing changed (about 20 minutes with 1 job and no cache).

Rings are simplified (Douglas-Peucker, TOL_M) without breaking the draw-once rule: a run of vertices
that several zones hold (the same vertices on a 1e-5 degree grid, as the app compares them) is
simplified once, in one direction, with its ends kept, so neighbours still share exactly the same
points. T-1145: a coastal zone's outline then takes the sea off its coast (zone_water.py) and the
shared runs are simplified again over the new rings; `--no-water` leaves the sea out, `--box` is a
trial over the zones whose boxes meet it. Format: {"tol": metres, "q": grid units per degree, "zones":
{id: [ring, ...]}, "water": {id: [ring, ...]}} (`water`: a coastal zone's sea alone, a little into its
land, for the shade outside the focused zone); a ring is [x0, y0, dx1, dy1, ...] in grid units,
closed (the first point is not repeated at the end).
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / "data/packs"
REG = ROOT / "app/src/game/registry.json"
OUT = PACKS / "outlines.json"
CACHE = ROOT / "data/work/zone_outlines_cache"
TOL_M = 300
Q = 10000  # the file's grid, 1/Q degree (about 11 m)
G = 100000  # the app's identity grid for a shared vertex (cityBoundaries' OutlineEdges)
M_LAT = 110574.0
M_LON = 111320.0


def pt(p):
    """A vertex on the identity grid, as integers."""
    return (round((p[0] + 180) * G), round((p[1] + 90) * G))


def dp(xy, tol):
    """Indices kept by Douglas-Peucker on an (n, 2) array in metres; both ends are kept."""
    n = len(xy)
    keep = np.zeros(n, bool)
    keep[0] = keep[n - 1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = xy[b] - xy[a]
        d = xy[a + 1:b] - xy[a]
        L = math.hypot(seg[0], seg[1])
        dist = np.hypot(d[:, 0], d[:, 1]) if L == 0 else np.abs(d[:, 0] * seg[1] - d[:, 1] * seg[0]) / L
        i = int(np.argmax(dist))
        if dist[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return np.nonzero(keep)[0]


def metres(run):
    """A run of identity-grid vertices as metres from its first, scaled by its own mean latitude, so
    a run held by two zones is simplified identically."""
    a = np.array(run, dtype=float)
    lat = a[:, 1] / G - 90
    k = math.cos(math.radians(float(lat.mean())))
    return np.column_stack([(a[:, 0] - a[0, 0]) / G * M_LON * k, (a[:, 1] - a[0, 1]) / G * M_LAT])


def simplify_run(run, tol):
    if len(run) < 3:
        return list(run)
    return [run[i] for i in dp(metres(run), tol)]


def canonical(run):
    """The run in the direction two zones holding it agree on."""
    return run if run[::-1] >= run else run[::-1]


def pmap(fn, tasks, jobs, chunk=1):
    """fn over the tasks, results in task order; over `jobs` spawned processes when more than one."""
    tasks = list(tasks)
    if jobs <= 1 or len(tasks) < 2:
        return [fn(t) for t in tasks]
    import concurrent.futures as cf
    import multiprocessing as mp
    with cf.ProcessPoolExecutor(max_workers=min(jobs, len(tasks)), mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(fn, tasks, chunksize=chunk))


def _load_task(f):
    """A boundary file's rings as (n, 2) int64 arrays of identity-grid vertices, open, no consecutive repeats."""
    rings = []
    for feat in json.loads(Path(f).read_text(encoding="utf-8"))["features"]:
        g = feat["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            for r in poly:
                a = np.asarray([p[:2] for p in r], dtype=float).reshape(-1, 2)
                q = np.column_stack([np.rint((a[:, 0] + 180) * G), np.rint((a[:, 1] + 90) * G)]).astype(np.int64)
                if len(q) > 1 and (q[0] == q[-1]).all():
                    q = q[:-1]
                # consecutive repeats on the grid carry no segment
                if len(q) > 1:
                    q = q[(q != np.roll(q, 1, axis=0)).any(axis=1)]
                if len(q) >= 3:
                    rings.append(q)
    return rings


def load(ids, jobs=1):
    zones = {}
    have = []
    for cid in ids:
        f = PACKS / f"{cid}.boundary.geojson"
        if not f.exists():
            print(f"  warning: {cid}: no boundary file, no coarse outline", flush=True)
            continue
        have.append((cid, str(f)))
    for (cid, _), rings in zip(have, pmap(_load_task, [f for _, f in have], jobs, chunk=8)):
        zones[cid] = rings
    return zones


def segkey(a, b):
    return (a, b) if a < b else (b, a)


def cut_points(zones):
    """For each zone, for each of its rings, the indices where the set of zones holding the segment
    changes (seg i runs from vertex i to i + 1). `zones`: [[(n, 2) int arrays]]. A segment's holder set
    is told apart by two 64-bit sums over its zones' random numbers (fixed seed), so equal sets are equal
    and different ones differ with all but certainty."""
    flat = [(zi, a) for zi, rs in enumerate(zones) for a in rs]
    if not flat:
        return [[] for _ in zones]
    lens = np.array([len(a) for _, a in flat])
    ends = np.cumsum(lens)
    starts = ends - lens
    V = np.concatenate([a for _, a in flat])
    N = len(V)
    code = (V[:, 0] << 26) | V[:, 1]  # one integer a vertex, in (x, y) order
    nxt = np.arange(1, N + 1)
    nxt[ends - 1] = starts
    a, b = code, code[nxt]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    zone = np.repeat(np.array([zi for zi, _ in flat]), lens)
    order = np.lexsort((zone, hi, lo))
    lo_s, hi_s, z_s = lo[order], hi[order], zone[order]
    new_seg = np.r_[True, (lo_s[1:] != lo_s[:-1]) | (hi_s[1:] != hi_s[:-1])]
    new_zone = new_seg | np.r_[True, z_s[1:] != z_s[:-1]]  # a zone holding a segment twice counts once
    gid = np.cumsum(new_seg) - 1
    first = np.nonzero(new_seg)[0]
    sig = np.empty((2, N), dtype=np.uint64)
    rng = np.random.default_rng(1977)
    for j in range(2):
        r = rng.integers(0, np.iinfo(np.uint64).max, size=len(zones), dtype=np.uint64)
        s = np.add.reduceat(np.where(new_zone, r[z_s], np.uint64(0)), first)
        sig[j][order] = s[gid]
    prv = np.arange(N) - 1
    prv[starts] = ends - 1
    cut = (sig[0] != sig[0][prv]) | (sig[1] != sig[1][prv])
    out = [[] for _ in zones]
    for (zi, _), s, e in zip(flat, starts, ends):
        out[zi].append(np.nonzero(cut[s:e])[0])
    return out


def _zone_task(a):
    """One zone's simplified rings: a = ([(ring array, cut indices)], tol)."""
    rings, tol = a
    res = []
    for arr, cuts in rings:
        r = list(map(tuple, arr.tolist()))
        n = len(r)
        cuts = cuts.tolist()
        if not cuts:
            # one holder set all round: closed, cut at its first vertex and the one farthest from it
            run = canonical_ring(r)
            far = int(np.argmax(np.hypot(*metres(run).T)))
            ring = simplify_run(run[:far + 1], tol)[:-1] + simplify_run(run[far:], tol)[:-1]
        else:
            r = r[cuts[0]:] + r[:cuts[0]]
            starts = [(c - cuts[0]) % n for c in cuts]
            ring = []
            for j, a in enumerate(starts):
                b = starts[j + 1] if j + 1 < len(starts) else n
                run = r[a:b + 1] if b < n else r[a:] + [r[0]]
                c = canonical(run)
                sim = simplify_run(c, tol)
                ring += (sim if c is run else sim[::-1])[:-1]
        if len(ring) >= 3:
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            k = math.cos(math.radians(ys[0] / G - 90))
            res.append((max((max(xs) - min(xs)) / G * M_LON * k, (max(ys) - min(ys)) / G * M_LAT), ring))
    # a ring smaller than the tolerance is a dot at any zoom this file serves; a zone keeps its largest
    keep = [ring for ext, ring in res if ext >= tol]
    if not keep and res:
        keep = [max(res, key=lambda e: e[0])[1]]
    return keep


def outline_rings(zones, tol, jobs=1):
    """{id: [ring of identity-grid vertices, open]} simplified with shared runs kept in step. A zone's
    rings are lists of vertices or (n, 2) integer arrays."""
    ids = list(zones)
    arrs = [[np.asarray(r, dtype=np.int64).reshape(-1, 2) for r in zones[c]] for c in ids]
    cuts = cut_points(arrs)
    tasks = [([(a, c) for a, c in zip(arrs[i], cuts[i])], tol) for i in range(len(ids))]
    return dict(zip(ids, pmap(_zone_task, tasks, jobs, chunk=4)))


def canonical_ring(r):
    """A ring wholly shared: opened at its least vertex, in the direction two holders agree on."""
    i = r.index(min(r))
    a = r[i:] + r[:i]
    b = [a[0]] + a[:0:-1]
    c = min(a, b)
    return c + [c[0]]


def encode(ring):
    q = [(round(p[0] / G * Q), round(p[1] / G * Q)) for p in ring]
    q = [p for i, p in enumerate(q) if p != q[i - 1]]
    if len(q) < 3:
        return None
    flat = [q[0][0] - 180 * Q, q[0][1] - 90 * Q]
    for i in range(1, len(q)):
        flat += [q[i][0] - q[i - 1][0], q[i][1] - q[i - 1][1]]
    return flat


def build(tol=TOL_M, out=OUT, root=None, water=True, within=None, jobs=1, cache=CACHE):
    reg = json.loads(REG.read_text(encoding="utf-8"))
    cities = reg["cities"]
    if within:  # a trial over the zones whose boxes meet (w, s, e, n)
        cities = [c for c in cities if c["box"][0] < within[2] and c["box"][2] > within[0] and c["box"][1] < within[3] and c["box"][3] > within[1]]
    ids = [c["id"] for c in cities]
    t0 = time.time()
    zones = load(ids, jobs)
    rings = outline_rings(zones, tol, jobs)
    print(f"  {len(rings)} zones cut and simplified, {time.time() - t0:.0f} s", flush=True)
    sea = {}
    if water:
        # T-1145: each zone's sea, then the shared runs simplified again over the new rings
        import zone_water
        grown, sea = zone_water.add_water(rings, {c["id"]: c["center"][1] for c in cities}, jobs=jobs, cache=cache,
                                          prune=within is None)
        if grown:
            rings = outline_rings({cid: grown.get(cid, rs) for cid, rs in rings.items()}, tol, jobs)
    doc = {"tol": tol, "q": Q, "zones": {}, "water": {}}
    for cid, rs in rings.items():
        enc = [e for e in map(encode, rs) if e]
        if enc:
            doc["zones"][cid] = enc
    for cid, rs in sea.items():
        enc = [e for e in map(encode, rs) if e]
        if enc:
            doc["water"][cid] = enc
    text = json.dumps(doc, separators=(",", ":"))
    tmp = Path(str(out) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, out)
    pts = sum(len(e) // 2 for z in doc["zones"].values() for e in z)
    wpts = sum(len(e) // 2 for z in doc["water"].values() for e in z)
    print(f"{out.name}: {len(doc['zones'])} of {len(ids)} zones, {pts:,} points, {len(doc['water'])} with sea "
          f"({wpts:,} points), {len(text) / 1e6:.2f} MB, {time.time() - t0:.0f} s", flush=True)
    return doc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    tol = float(a[a.index("--tol") + 1]) if "--tol" in a else TOL_M
    out = Path(a[a.index("--out") + 1]) if "--out" in a else OUT
    within = [float(v) for v in a[a.index("--box") + 1].split(",")] if "--box" in a else None
    jobs = int(a[a.index("--jobs") + 1]) if "--jobs" in a else 1
    cache = None if "--no-cache" in a else Path(a[a.index("--cache") + 1]) if "--cache" in a else CACHE
    build(tol, out, water="--no-water" not in a, within=within, jobs=jobs, cache=cache)
