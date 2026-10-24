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
"""
import math
import time

import numpy as np
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


def read_sea(boxes):
    """{(x, y): the sea over that whole-degree tile, simplified, in Mercator} for every tile the boxes
    (lon/lat) cover that has sea."""
    tiles = set()
    for w, s, e, n in boxes:
        for x in range(math.floor(w), math.floor(e) + 1):
            for y in range(max(-85, math.floor(s)), min(84, math.floor(n)) + 1):
                tiles.add((x, y))
    import pyogrio
    path = osm_water.sea_fgb()
    out = {}
    t0 = time.time()
    for k, (x, y) in enumerate(sorted(tiles)):
        df = pyogrio.read_dataframe(path, bbox=(x, y, x + 1, y + 1), columns=[])
        if not len(df):
            continue
        # simplified first, in degrees (a metre of latitude), then cut to the tile and joined
        g = shapely.simplify(df.geometry.to_numpy(), SEA_TOL_M / 111000, preserve_topology=False)
        bad = ~shapely.is_valid(g)
        g[bad] = shapely.make_valid(g[bad])
        g = only_polys(unary_union(shapely.clip_by_rect(g, x, y, x + 1, y + 1)))
        if g.is_empty:
            continue
        g = only_polys(shapely.make_valid(shapely.transform(g, merc)))
        if not g.is_empty:
            out[(x, y)] = g
        if (k + 1) % 500 == 0:
            log(f"  sea: {k + 1} of {len(tiles)} tiles, {time.time() - t0:.0f} s")
    log(f"  sea: {len(tiles)} tiles, {len(out)} with sea, {time.time() - t0:.0f} s")
    return out


def sample(g, step):
    """Points along g's boundary no more than `step` apart, (n, 2)."""
    return shapely.get_coordinates(shapely.segmentize(g.boundary, step))


def add_water(rings, lat_of):
    """rings: {zone id: [ring of identity-grid vertices]} (the coarse outlines); lat_of: {id: latitude}.
    Out: ({id: [rings]} for the zones whose outline takes sea, {id: [rings]} their sea for the focus shade)."""
    t0 = time.time()
    L, k = {}, {}
    for z, rs in rings.items():
        p = zone_polygon(rs)
        if p is not None:
            L[z] = p
            k[z] = scale(lat_of[z])
    zid = list(L)
    lgeoms = [L[z] for z in zid]
    ltree = STRtree(lgeoms)
    log(f"  {len(zid)} zone polygons, {time.time() - t0:.0f} s")

    lonlat = {}
    for z in zid:
        w, s, e, n = shapely.bounds(L[z])
        (w, s), (e, n) = unmerc(np.array([[w, s], [e, n]]))
        lonlat[z] = (w, s, e, n)
    grow = 0.12
    sea = read_sea([(w - grow * k[z], s - grow, e + grow * k[z], n + grow) for z, (w, s, e, n) in lonlat.items()])
    if not sea:
        return {}, {}
    stiles = list(sea)
    stree = STRtree([sea[t] for t in stiles])
    coastal = [z for z in zid if len(stree.query(L[z], predicate="dwithin", distance=GAP_M * k[z]))]
    windows = set()
    for z in coastal:
        w, s, e, n = lonlat[z]
        for x in range(math.floor((w - grow * k[z]) / WIN_DEG), math.floor((e + grow * k[z]) / WIN_DEG) + 1):
            for y in range(math.floor((s - grow) / WIN_DEG), math.floor((n + grow) / WIN_DEG) + 1):
                windows.add((x, y))
    log(f"  {len(coastal)} coastal zones in {len(windows)} windows, {time.time() - t0:.0f} s")

    pieces = {}
    for i, (x, y) in enumerate(sorted(windows)):
        core = (x * WIN_DEG, y * WIN_DEG, (x + 1) * WIN_DEG, (y + 1) * WIN_DEG)
        for z, g in window(core, L, k, zid, ltree, sea, stiles, stree).items():
            pieces.setdefault(z, []).append(g)
        if (i + 1) % 25 == 0:
            log(f"  {i + 1} of {len(windows)} windows, {time.time() - t0:.0f} s")

    grown, water = {}, {}
    for z, ps in pieces.items():
        t = 300 * k[z]
        w = [p for p in polys_of(unary_union(ps)) if p.distance(L[z]) < 1.0 and p.area > (SAMPLE_M * k[z]) ** 2]
        if not w:
            continue
        w = unary_union(w)
        grown[z] = rings_of(only_polys(unary_union([L[z], w])))
        # into its own land by 2t but no nearer than t to another zone's, so that with the zone's own
        # boundary (within t of this outline) it leaves no gap at the coast
        band = w.buffer(2 * t, quad_segs=2)
        b = band.bounds
        others = [shapely.clip_by_rect(lgeoms[j], b[0] - t, b[1] - t, b[2] + t, b[3] + t) for j in ltree.query(band) if zid[j] != z]
        band = band.intersection(L[z])
        if others:
            band = band.difference(unary_union(others).buffer(t, quad_segs=2))
        wf = only_polys(shapely.make_valid(unary_union([w, band]).simplify(t, preserve_topology=True)))
        water[z] = rings_of(wf, holes=False)
    log(f"  {len(grown)} zones take sea, {time.time() - t0:.0f} s")
    return grown, water


def window(core, L, k, zid, ltree, sea, stiles, stree):
    """{zone id: its sea inside the core box (lon/lat)}, worked out over the core and its margin."""
    ext = (core[0] - MARGIN_DEG, max(-85, core[1] - MARGIN_DEG), core[2] + MARGIN_DEG, min(85, core[3] + MARGIN_DEG))
    core_m, ext_m = merc_box(*core), merc_box(*ext)
    S = only_polys(unary_union([shapely.clip_by_rect(sea[stiles[i]], *ext_m.bounds) for i in stree.query(ext_m)]))
    if S.is_empty:
        return {}
    here = [zid[j] for j in ltree.query(ext_m)]
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
