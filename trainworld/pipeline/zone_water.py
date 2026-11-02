"""T-1145: the sea off each zone's coast, added to its coarse outline (zone_outlines.py).

A zone's outline takes the sea within REACH_M of its land. Sea that the zones' land and reach close in
(Tokyo Bay inside the Uraga channel, Osaka Bay) is taken whole, up to BAY_KM2, unless land no zone
holds borders most of it. Where reaches meet, each piece of sea goes to the zone with the nearest land:
a Voronoi split over points every SAMPLE_M along the outlines, with land no zone holds (another
country) as one more owner that keeps the sea nearest it. Land that is neither a zone's nor sea but
narrower than about 2 * GAP_M (where an outline's coast and OSM's disagree) counts as sea.

The sea is OSM's coastline water polygons (osm_water.sea_fgb), simplified to SEA_TOL_M. The work is in
Web Mercator metres, so a distance on the ground is scaled by 1/cos(latitude), and in windows of WIN_DEG
degrees, each seeing MARGIN_DEG round it (more than a reach, a bay or the split's nearest land need),
whose pieces are joined per zone at the end.

Out: each changed zone's new rings (its land and its sea as one), and its sea alone grown a little
into its own land (`water`): the app adds that to a focused zone's own boundary, so the shade outside
the focused zone leaves its sea clear too.

T-1977: the sea tiles, the zones' polygons, the windows and the per-zone finish run over `jobs`
processes, and their results are kept in a cache folder, so a rerun redoes only what its inputs
changed. A window's result depends on the zones near it (those whose grown box meets its extent
plus a degree, since a whole-degree sea tile exists when any zone's grown box meets it), the sea file
and this file's code, and its key is a hash of exactly those; a zone's finish adds the windows its
box meets. Every task is a function of its arguments and results are merged in a fixed order, so the
output does not depend on `jobs` or on what the cache held.
"""
import concurrent.futures as cf
import hashlib
import math
import multiprocessing as mp
import os
import pickle
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy
import shapely
from shapely import STRtree
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, unary_union

import osm_water

REACH_M = 5000
BAY_KM2 = 3000
GAP_M = 400
SAMPLE_M = 250
SEA_TOL_M = 60
WIN_DEG = 3
MARGIN_DEG = 0.6
GROW = 0.12  # a zone's box takes the sea tiles this far (degrees, longitude times the zone's scale) round it
G = 100000  # zone_outlines' identity grid
R_E = 6378137.0


def log(msg):
    print(msg, flush=True)


def merc(xy):
    """(n, 2) lon, lat to Web Mercator metres."""
    lon, lat = xy[:, 0], np.clip(xy[:, 1], -85.0511, 85.0511)
    return np.column_stack([np.radians(lon) * R_E, R_E * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))])


def unmerc(xy):
    return np.column_stack([np.degrees(xy[:, 0] / R_E), np.degrees(2 * np.arctan(np.exp(xy[:, 1] / R_E)) - np.pi / 2)])


def scale(lat):
    """Mercator metres per metre on the ground at a latitude."""
    return 1 / math.cos(math.radians(lat))


def merc_box(w, s, e, n):
    (x0, y0), (x1, y1) = merc(np.array([[w, s], [e, n]], dtype=float))
    return box(x0, y0, x1, y1)


def zone_polygon(rings):
    """The zone's area from its rings, holes by depth (a ring inside an odd number of others is a hole)."""
    polys = []
    for r in rings:
        if len(r) < 3:
            continue
        p = only_polys(shapely.make_valid(Polygon(merc(np.asarray(r, dtype=float) / G - [180, 90]))))
        if not p.is_empty and p.area > 0:
            polys.append(p)
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    tree = STRtree(polys)
    reps = [p.representative_point() for p in polys]
    depth = [sum(1 for j in tree.query(rp, predicate="within") if j != i) for i, rp in enumerate(reps)]
    out = None
    for d in sorted(set(depth)):
        level = unary_union([p for p, k in zip(polys, depth) if k == d])
        out = level if out is None else (out.union(level) if d % 2 == 0 else out.difference(level))
    return out


def polys_of(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    out = []
    for p in getattr(g, "geoms", []):
        out += polys_of(p) if p.geom_type in ("MultiPolygon", "GeometryCollection") else ([p] if p.geom_type == "Polygon" and not p.is_empty else [])
    return out


def only_polys(g):
    """The polygons of an overlay's result (whose parts never overlap), as one geometry."""
    if g is not None and g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    ps = polys_of(g)
    return shapely.multipolygons(ps) if len(ps) > 1 else (ps[0] if ps else Polygon())


# ------------------------------------------------------------------ pool and cache plumbing

_G = {}  # a worker's state: the zones, their geometries once built, the sea tiles near the last window


def _setup(state):
    _G.clear()
    _G.update(state)
    _G["geom"] = {}
    _G["tiles"] = {}


def run_tasks(fn, tasks, jobs, state=None, chunk=1):
    """fn over the tasks, results in task order, as they finish. jobs > 1: that many spawned processes,
    each set up with `state`; otherwise this process, set up the same way."""
    tasks = list(tasks)
    if not tasks:
        return
    if jobs <= 1 or len(tasks) == 1:
        _setup(state or {})
        for t in tasks:
            yield fn(t)
        return
    with cf.ProcessPoolExecutor(max_workers=min(jobs, len(tasks)), mp_context=mp.get_context("spawn"),
                                initializer=_setup, initargs=(state or {},)) as ex:
        yield from ex.map(fn, tasks, chunksize=chunk)


def code_version():
    """What a cached result must have been made by: this file's text and the libraries' versions."""
    h = hashlib.sha1(Path(__file__).read_bytes())
    h.update(f"|{shapely.__version__}|{shapely.geos_version}|{np.__version__}|{scipy.__version__}".encode())
    return h.hexdigest()[:12]


def tile_version():
    """What a cached sea tile must have been made by: the code that cuts it and the libraries'."""
    import inspect
    h = hashlib.sha1(f"{SEA_TOL_M}|{shapely.__version__}|{shapely.geos_version}|{np.__version__}".encode())
    for f in (_tile_task, merc, only_polys, polys_of):
        h.update(inspect.getsource(f).encode())
    return h.hexdigest()[:8]


def zone_hash(z, rings, lat):
    h = hashlib.sha1(f"{z}|{lat!r}|".encode())
    for r in rings:
        h.update(np.asarray(r, dtype=np.int64).tobytes())
        h.update(b"|")
    return h.hexdigest()[:20]


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, path)


def undump(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# ------------------------------------------------------------------ sea tiles

def grown_box(lonlat, k):
    w, s, e, n = lonlat
    return (w - GROW * k, s - GROW, e + GROW * k, n + GROW)


def tiles_of(box_):
    w, s, e, n = box_
    return [(x, y) for x in range(math.floor(w), math.floor(e) + 1) for y in range(max(-85, math.floor(s)), min(84, math.floor(n)) + 1)]


def tile_path(tdir, x, y):
    return Path(tdir) / f"{x}_{y}.wkb"


def _tile_task(a):
    """The sea over one whole-degree tile, simplified, in Mercator, written to the tile folder as WKB
    (an empty file when the tile has no sea)."""
    x, y, fgb, tdir = a
    import pyogrio
    p = tile_path(tdir, x, y)
    df = pyogrio.read_dataframe(fgb, bbox=(x, y, x + 1, y + 1), columns=[])
    data = b""
    if len(df):
        # simplified first, in degrees (a metre of latitude), then cut to the tile and joined
        g = shapely.simplify(df.geometry.to_numpy(), SEA_TOL_M / 111000, preserve_topology=False)
        bad = ~shapely.is_valid(g)
        g[bad] = shapely.make_valid(g[bad])
        g = only_polys(unary_union(shapely.clip_by_rect(g, x, y, x + 1, y + 1)))
        if not g.is_empty:
            g = only_polys(shapely.make_valid(shapely.transform(g, merc)))
            if not g.is_empty:
                data = shapely.to_wkb(g)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)


def read_tiles(tiles, fgb, tdir, jobs):
    """Make sure every tile has its file in tdir."""
    Path(tdir).mkdir(parents=True, exist_ok=True)
    todo = sorted(t for t in tiles if not tile_path(tdir, *t).exists())
    if not todo:
        return
    t0 = time.time()
    for i, _ in enumerate(run_tasks(_tile_task, [(x, y, fgb, str(tdir)) for x, y in todo], jobs, chunk=8)):
        if (i + 1) % 500 == 0:
            log(f"  sea: {i + 1} of {len(todo)} tiles, {time.time() - t0:.0f} s")
    log(f"  sea: {len(todo)} tiles read, {time.time() - t0:.0f} s")


def tile_geom(x, y):
    """A tile's sea from the worker's tile folder, or None; kept while windows near it run."""
    k = (x, y)
    if len(_G["tiles"]) > 120:
        _G["tiles"].clear()
    if k not in _G["tiles"]:
        data = tile_path(_G["tdir"], x, y).read_bytes()
        _G["tiles"][k] = shapely.from_wkb(data) if data else None
    return _G["tiles"][k]


def ext_of(core):
    return (core[0] - MARGIN_DEG, max(-85, core[1] - MARGIN_DEG), core[2] + MARGIN_DEG, min(85, core[3] + MARGIN_DEG))


def window_tiles(ext):
    """The tiles whose sea could meet a window's extent (a ring more than do, for the edges)."""
    return [(x, y) for x in range(math.floor(ext[0]) - 1, math.floor(ext[2]) + 2) for y in range(math.floor(ext[1]) - 1, math.floor(ext[3]) + 2)]


# ------------------------------------------------------------------ the zones

def _poly_task(a):
    """(WKB of the zone's polygon, its box in lon/lat) or None."""
    z, rs = a
    p = zone_polygon(rs)
    if p is None:
        return None
    w, s, e, n = shapely.bounds(p)
    (w, s), (e, n) = unmerc(np.array([[w, s], [e, n]]))
    return shapely.to_wkb(p), (float(w), float(s), float(e), float(n))


def _coast_task(a):
    """Does the zone lie within GAP_M of sea? (its own tiles hold all the sea that near)"""
    wkb, bx, k = a
    p = shapely.from_wkb(wkb)
    geoms = [g for g in (tile_geom(x, y) for x, y in tiles_of(bx)) if g is not None]
    if not geoms:
        return False
    return bool(len(STRtree(geoms).query(p, predicate="dwithin", distance=GAP_M * k)))


def _geom(i):
    g = _G["geom"].get(i)
    if g is None:
        g = _G["geom"][i] = shapely.from_wkb(_G["wkb"][i])
    return g


def _near(b):
    """Indices, ascending, of the zones whose polygon's box meets the box b (merc)."""
    bd = _G["bounds"]
    return np.nonzero((bd[:, 0] <= b[2]) & (bd[:, 2] >= b[0]) & (bd[:, 1] <= b[3]) & (bd[:, 3] >= b[1]))[0]


def _window_task(xy):
    """{zone id: WKB of its sea inside the window's core}."""
    x, y = xy
    core = (x * WIN_DEG, y * WIN_DEG, (x + 1) * WIN_DEG, (y + 1) * WIN_DEG)
    ext = ext_of(core)
    ext_m = merc_box(*ext)
    b = ext_m.bounds
    have = _G["tileset"]
    need = sorted(t for t in window_tiles(ext) if t in have)
    for t in [t for t in _G["tiles"] if t not in need]:
        del _G["tiles"][t]
    geoms = [g for g in (tile_geom(*t) for t in need) if g is not None]
    if not geoms:
        return {}
    stree = STRtree(geoms)
    S = only_polys(unary_union([shapely.clip_by_rect(geoms[i], *b) for i in sorted(stree.query(ext_m))]))
    if S.is_empty:
        return {}
    here = [int(i) for i in _near(b)]
    zid, k = _G["zid"], _G["k"]
    out = window(core, [zid[i] for i in here], {zid[i]: _geom(i) for i in here}, k, S, ext_m)
    return {z: shapely.to_wkb(g) for z, g in out.items()}


def _finish_task(a):
    """One zone's (grown rings or None, its sea's rings) from its window pieces (WKB)."""
    i, pieces = a
    zid, k = _G["zid"], _G["k"]
    z, Lz = zid[i], _geom(i)
    kz = k[z]
    t = 300 * kz
    w = [p for p in polys_of(unary_union([shapely.from_wkb(d) for d in pieces])) if p.distance(Lz) < 1.0 and p.area > (SAMPLE_M * kz) ** 2]
    if not w:
        return None
    w = unary_union(w)
    grown = rings_of(only_polys(unary_union([Lz, w])))
    # into its own land by 2t but no nearer than t to another zone's, so that with the zone's own
    # boundary (within t of this outline) it leaves no gap at the coast
    band = w.buffer(2 * t, quad_segs=2)
    b = band.bounds
    others = [shapely.clip_by_rect(_geom(int(j)), b[0] - t, b[1] - t, b[2] + t, b[3] + t) for j in _near(b) if int(j) != i]
    band = band.intersection(Lz)
    if others:
        band = band.difference(unary_union(others).buffer(t, quad_segs=2))
    wf = only_polys(shapely.make_valid(unary_union([w, band]).simplify(t, preserve_topology=True)))
    return grown, rings_of(wf, holes=False)


def add_water(rings, lat_of, jobs=1, cache=None, prune=False):
    """rings: {zone id: [ring of identity-grid vertices]} (the coarse outlines); lat_of: {id: latitude}.
    Out: ({id: [rings]} for the zones whose outline takes sea, {id: [rings]} their sea for the focus shade).
    cache: a folder for results kept between runs (None: none kept); prune: drop what this run did not use."""
    t0 = time.time()
    own_tmp = None
    if cache is None:
        own_tmp = cache = tempfile.mkdtemp(prefix="zone_water_")
    cache = Path(cache)
    try:
        return _add_water(rings, lat_of, jobs, cache, prune, t0)
    finally:
        if own_tmp:
            shutil.rmtree(own_tmp, ignore_errors=True)


def _add_water(rings, lat_of, jobs, cache, prune, t0):
    ver = code_version()
    fgb = osm_water.sea_fgb()
    st = os.stat(fgb)
    sea_fp = f"{st.st_size}-{st.st_mtime_ns}"
    tdir = cache / f"sea-{sea_fp}-{tile_version()}"

    # the zones' polygons, kept by the hash of their rings
    hz = {z: zone_hash(z, rs, lat_of[z]) for z, rs in rings.items()}
    store = undump(cache / "zones.pkl") if (cache / "zones.pkl").exists() else None
    store = store["zones"] if store and store.get("ver") == ver else {}
    todo = [z for z in rings if hz[z] not in store]
    for z, res in zip(todo, run_tasks(_poly_task, [(z, rings[z]) for z in todo], jobs, chunk=4)):
        store[hz[z]] = {"poly": res, "coast": {}}
    zid = [z for z in rings if store[hz[z]]["poly"] is not None]
    k = {z: scale(lat_of[z]) for z in zid}
    lonlat = {z: store[hz[z]]["poly"][1] for z in zid}
    log(f"  {len(zid)} zone polygons ({len(todo)} new), {time.time() - t0:.0f} s")

    # which zones lie on the coast, per sea file
    alltiles = set()
    for z in zid:
        alltiles.update(tiles_of(grown_box(lonlat[z], k[z])))
    need = [z for z in zid if sea_fp not in store[hz[z]]["coast"]]
    if need:
        read_tiles({t for z in need for t in tiles_of(grown_box(lonlat[z], k[z]))}, fgb, tdir, jobs)
        flags = run_tasks(_coast_task, [(store[hz[z]]["poly"][0], grown_box(lonlat[z], k[z]), k[z]) for z in need], jobs,
                          {"tdir": str(tdir)}, chunk=8)
        for z, c in zip(need, flags):
            store[hz[z]]["coast"] = {sea_fp: c}
    coastal = [z for z in zid if store[hz[z]]["coast"][sea_fp]]
    windows = set()
    for z in coastal:
        w, s, e, n = lonlat[z]
        for x in range(math.floor((w - GROW * k[z]) / WIN_DEG), math.floor((e + GROW * k[z]) / WIN_DEG) + 1):
            for y in range(math.floor((s - GROW) / WIN_DEG), math.floor((n + GROW) / WIN_DEG) + 1):
                windows.add((x, y))
    windows = sorted(windows)
    log(f"  {len(coastal)} coastal zones in {len(windows)} windows, {time.time() - t0:.0f} s")

    # each window's key: the zones whose grown box meets its extent a degree wider
    gb = np.array([grown_box(lonlat[z], k[z]) for z in zid]).reshape(-1, 4)
    wkey, wsize = {}, {}
    for x, y in windows:
        e = ext_of((x * WIN_DEG, y * WIN_DEG, (x + 1) * WIN_DEG, (y + 1) * WIN_DEG))
        m = (gb[:, 0] <= e[2] + 1) & (gb[:, 2] >= e[0] - 1) & (gb[:, 1] <= e[3] + 1) & (gb[:, 3] >= e[1] - 1)
        h = hashlib.sha1(f"{ver}|{sea_fp}|{x},{y}|".encode())
        for i in np.nonzero(m)[0]:
            h.update(hz[zid[i]].encode())
        wkey[(x, y)] = h.hexdigest()[:24]
        wsize[(x, y)] = int(m.sum())
    wdir = cache / "win"
    result = {}
    miss = []
    for w in windows:
        r = undump(wdir / f"{wkey[w]}.pkl") if (wdir / f"{wkey[w]}.pkl").exists() else None
        if r is None:
            miss.append(w)
        else:
            result[w] = r
    log(f"  {len(windows) - len(miss)} windows kept, {len(miss)} to do, {time.time() - t0:.0f} s")

    # state the pool's processes get: every zone's polygon and the sea tiles that exist
    state = {"zid": zid, "k": k, "wkb": [store[hz[z]]["poly"][0] for z in zid], "tdir": str(tdir), "tileset": alltiles,
             "bounds": np.array([shapely.bounds(shapely.from_wkb(store[hz[z]]["poly"][0])) for z in zid]).reshape(-1, 4)}
    pieces = {}
    if miss:
        read_tiles({t for w in miss for t in window_tiles(ext_of((w[0] * WIN_DEG, w[1] * WIN_DEG, (w[0] + 1) * WIN_DEG, (w[1] + 1) * WIN_DEG))) if t in alltiles},
                   fgb, tdir, jobs)
        miss.sort(key=lambda w: (-wsize[w], w))  # the busiest windows first, so none is left for last
        for i, (w, r) in enumerate(zip(miss, run_tasks(_window_task, miss, jobs, state))):
            dump(wdir / f"{wkey[w]}.pkl", r)
            result[w] = r
            if (i + 1) % 25 == 0:
                log(f"  {i + 1} of {len(miss)} windows, {time.time() - t0:.0f} s")
    pwin = {}
    for w in windows:
        for z, d in result[w].items():
            pieces.setdefault(z, []).append(d)
            pwin.setdefault(z, []).append(w)

    # a zone's finish depends on its polygon, its pieces and the zones near them (which its band avoids)
    zi = {z: i for i, z in enumerate(zid)}
    order = list(pieces)
    lb = np.array([lonlat[z] for z in zid]).reshape(-1, 4)
    fkey = {}
    for z in order:
        ws = pwin[z]
        w0, s0 = min(x for x, _ in ws) * WIN_DEG - 0.2, min(y for _, y in ws) * WIN_DEG - 0.2
        e0, n0 = (max(x for x, _ in ws) + 1) * WIN_DEG + 0.2, (max(y for _, y in ws) + 1) * WIN_DEG + 0.2
        m = (lb[:, 0] <= e0) & (lb[:, 2] >= w0) & (lb[:, 1] <= n0) & (lb[:, 3] >= s0)
        h = hashlib.sha1(f"{ver}|{hz[z]}|".encode())
        for d in pieces[z]:
            h.update(hashlib.sha1(d).digest())
        for i in np.nonzero(m)[0]:
            h.update(hz[zid[i]].encode())
        fkey[z] = h.hexdigest()[:24]
    fin = undump(cache / "finals.pkl") if (cache / "finals.pkl").exists() else None
    fin = fin["finals"] if fin and fin.get("ver") == ver else {}
    todo = [z for z in order if fkey[z] not in fin]
    for z, r in zip(todo, run_tasks(_finish_task, [(zi[z], pieces[z]) for z in todo], jobs, state, chunk=2)):
        fin[fkey[z]] = r
    grown, water = {}, {}
    for z in order:
        if fin[fkey[z]] is not None:
            grown[z], water[z] = fin[fkey[z]]
    log(f"  {len(grown)} zones take sea ({len(todo)} finished anew), {time.time() - t0:.0f} s")

    if prune:
        used = {f"{wkey[w]}.pkl" for w in windows}
        for f in wdir.glob("*.pkl") if wdir.exists() else []:
            if f.name not in used:
                f.unlink()
        for d in cache.glob("sea-*"):
            if d != tdir:
                shutil.rmtree(d, ignore_errors=True)
        keep = {hz[z] for z in rings}
        store = {h: v for h, v in store.items() if h in keep}
        used_f = set(fkey.values())
        fin = {h: v for h, v in fin.items() if h in used_f}
    dump(cache / "zones.pkl", {"ver": ver, "zones": store})
    dump(cache / "finals.pkl", {"ver": ver, "finals": fin})
    return grown, water


def window(core, here, L, k, S, ext_m):
    """{zone id: its sea inside the core box (lon/lat)}, worked out over the core and its margin; here:
    the zones whose boxes meet the extent, L their polygons, S the sea over the extent."""
    core_m = merc_box(*core)
    Lw = {z: only_polys(shapely.clip_by_rect(L[z], *ext_m.bounds)) for z in here}
    Lw = {z: g for z, g in Lw.items() if not g.is_empty}
    coastal = [z for z in Lw if shapely.dwithin(Lw[z], S, GAP_M * k[z])]
    if not coastal:
        return {}
    reach = []
    for z in coastal:
        land = Lw[z].difference(S)
        land = land if not land.is_empty else Lw[z]
        # the reach's edge is round anyway: its land a little simplified first, for speed
        reach.append(land.simplify(100 * k[z], preserve_topology=False).buffer(REACH_M * k[z], quad_segs=4))
    Rfull = unary_union(reach).intersection(ext_m)
    Lnear = unary_union(list(Lw.values()))
    Rw = Rfull.difference(Lnear)
    kk = float(np.mean([k[z] for z in coastal]))
    gap = GAP_M * kk

    # land no zone holds within reach, with its strips narrower than 2 * gap opened away
    other = only_polys(Rw.difference(S))
    other = only_polys(other.buffer(-gap, quad_segs=2).buffer(gap, quad_segs=2).intersection(Rw))

    # bays: sea the zones and their reach close in
    holes = [Polygon(r) for p in polys_of(unary_union([Rfull, Lnear])) for r in p.interiors]
    lat = float(unmerc(np.array(ext_m.centroid.coords))[0, 1])
    holes = [h for h in holes if h.area / scale(lat) ** 2 / 1e6 <= BAY_KM2]
    if holes and not other.is_empty:
        beside = other.buffer(1, quad_segs=1)
        shapely.prepare(beside)
        holes = [h for h in holes if not beside.intersects(h.exterior) or h.exterior.intersection(beside).length <= 0.5 * h.exterior.length]
    fills = polys_of(shapely.intersection(unary_union(holes), S)) if holes else []

    # the sea to share: within reach or in a bay, in no zone and not other land, in pieces that meet
    # the sea
    C = unary_union([Rw.difference(other)] + fills)
    shapely.prepare(S)
    cparts = [p for p in polys_of(C) if p.intersects(core_m) and S.intersects(p)]
    if not cparts:
        return {}
    C = unary_union(cparts)

    # owners' points: zones' outlines and other land's edges, not on the window's edge; a point two
    # owners hold (a border between zones, on the same vertices) is left out
    pts, own = [], []
    names = [""] + list(Lw)
    edge = ext_m.exterior
    for n, z in enumerate(Lw, 1):
        xy = sample(L[z], SAMPLE_M * k[z])
        if len(xy):
            pts.append(xy)
            own.append(np.full(len(xy), n))
    if not other.is_empty:
        xy = sample(other, SAMPLE_M * kk)
        pts.append(xy)
        own.append(np.zeros(len(xy), int))
    pts, own = np.concatenate(pts), np.concatenate(own)
    inside = shapely.contains_xy(ext_m, pts[:, 0], pts[:, 1]) & ~shapely.dwithin(edge, shapely.points(pts), 10.0)
    pts, own = pts[inside], own[inside]
    if not len(pts):
        return {}
    _, inv, count = np.unique(np.round(pts, 0), axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    pairs = np.unique(np.column_stack([inv, own]), axis=0)
    keep = np.bincount(pairs[:, 0], minlength=len(count))[inv] == 1
    pts, own = pts[keep], own[keep]
    pts, first = np.unique(np.round(pts, 1), axis=0, return_index=True)
    own = own[first]

    cells = owner_cells(pts, own)
    out = {}
    for n, cell in cells.items():
        if not n:
            continue
        g = only_polys(cell.intersection(C).intersection(core_m))
        if not g.is_empty:
            out[names[n]] = g
    return out


def sample(g, step):
    """Points along g's boundary no more than `step` apart, (n, 2)."""
    return shapely.get_coordinates(shapely.segmentize(g.boundary, step))


def owner_cells(pts, own):
    """{owner: the area nearer its points than any other owner's} from a Voronoi diagram of the points."""
    from scipy.spatial import Voronoi, cKDTree
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 200000.0
    frame = box(lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad)
    if len(set(own.tolist())) == 1 or len(pts) < 4:
        return {own[0]: frame}
    vor = Voronoi(pts)
    far = float(np.hypot(*(hi - lo))) + 4 * pad
    centre = pts.mean(axis=0)
    lines = [LineString(frame.exterior.coords)]
    for (p, q), vs in zip(vor.ridge_points, vor.ridge_vertices):
        if own[p] == own[q]:
            continue
        if vs[0] >= 0 and vs[1] >= 0:
            lines.append(LineString([vor.vertices[vs[0]], vor.vertices[vs[1]]]))
            continue
        v = vor.vertices[vs[1] if vs[0] < 0 else vs[0]]
        t = pts[q] - pts[p]
        t = t / np.hypot(*t)
        n = np.array([-t[1], t[0]])
        mid = (pts[p] + pts[q]) / 2
        d = n * np.sign(np.dot(mid - centre, n) or 1.0)
        lines.append(LineString([v, v + d * far]))
    faces = shapely.get_parts(shapely.polygonize(shapely.get_parts(shapely.node(shapely.multilinestrings(lines)))))
    faces = [f for f in faces if frame.contains(f.representative_point())]
    _, nearest = cKDTree(pts).query(shapely.get_coordinates([f.representative_point() for f in faces]))
    by = {}
    for f, i in zip(faces, nearest):
        by.setdefault(int(own[i]), []).append(f)
    # the faces tile the frame, so each owner's are joined as a coverage
    return {o: shapely.coverage_union_all(fs) for o, fs in by.items()}


def rings_of(g, holes=True):
    """A geometry's rings as open lists of identity-grid vertices, consecutive repeats dropped."""
    out = []
    for p in polys_of(g):
        for r in [p.exterior] + (list(p.interiors) if holes else []):
            ll = unmerc(np.asarray(r.coords))
            pts = [(int(round((x + 180) * G)), int(round((y + 90) * G))) for x, y in ll]
            if len(pts) > 1 and pts[0] == pts[-1]:
                pts.pop()
            pts = [q for i, q in enumerate(pts) if q != pts[i - 1]] if len(pts) > 1 else pts
            if len(pts) >= 3:
                out.append(pts)
    return out
