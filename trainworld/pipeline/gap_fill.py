"""T-3287: dead units that sit between built land, picked so blocs come out whole (docs/zones.md 2, "Gaps").

A unit the density or reach test leaves dead is picked when
- **between**: the straight line between two towns of TOWN_POP or more lying in built land, at most PAIR_KM apart,
  runs LINE_KM or more over it;
- **town**: it holds a town of TOWN_POP and touches built land;
- **hole**: built land holds HOLE_SHARE of its land border, and land is at least LAND_SHARE of its whole border;
then the picks count as built land and the tests run again, until a round picks nothing. Towns are Natural Earth's
populated places (pop_max). Built land is whatever the caller passes: packs, plan rows and the country's free units.

A corridor (`corridor_split`) cuts the units a route crosses into the strip within its reach and the rest, so a
thin zone can follow a road across land the reach rule leaves dead (northwestern Ontario).

Geometries are Mollweide metres (world_plan's grid); the line test measures km on the ellipsoid.
"""
import collections

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import LineString

TOWN_POP = 50_000
PAIR_KM = 300.0
LINE_KM = 15.0
HOLE_SHARE = 0.75
LAND_SHARE = 0.2
TOUCH_M = 2500.0      # a border point this near other land borders it (1:3M NUTS against OSM outlines)
STEP_M = 1000.0       # between border samples
MAX_ROUNDS = 10


def border_points(g, step=STEP_M):
    pts = []
    for ring in shapely.get_parts(shapely.boundary(g)):
        n = max(int(ring.length / step), 4)
        pts += list(shapely.line_interpolate_point(ring, np.linspace(0, ring.length, n, endpoint=False)))
    return np.array(pts, dtype=object)


def border_shares(g, built_tree, other_tree, land_tree):
    """(built share of the land border, land share of the whole border) of g: each border sample is built when built
    land lies within TOUCH_M, else land when another unit or other land does, else coast."""
    pts = border_points(g)
    if not len(pts):
        return 0.0, 0.0
    near = lambda tree: np.isin(np.arange(len(pts)), tree.query(pts, predicate="dwithin", distance=TOUCH_M)[0]) \
        if tree is not None else np.zeros(len(pts), bool)
    b = near(built_tree)
    land = b | near(other_tree) | near(land_tree)
    n_land = int(land.sum())
    return (float(b.sum()) / n_land if n_land else 0.0), n_land / len(pts)


def line_km(a, c, geod, fwd, dead, tree):
    """{index: km of the geodesic a-c (lon, lat) that runs over dead[index]}."""
    km = geod.inv(a[0], a[1], c[0], c[1])[2] / 1000.0
    pts = [a] + geod.npts(a[0], a[1], c[0], c[1], max(int(km / 2), 2)) + [c]
    xy = np.array(pts)
    x, y = fwd.transform(xy[:, 0], xy[:, 1])
    line = LineString(np.column_stack([x, y]))
    out = {}
    for k in tree.query(line, predicate="intersects"):
        f = line.intersection(dead[k]).length / line.length * km
        if f > 0.5:
            out[int(k)] = f
    return km, out


def pick(dead, built, towns, land=(), geod=None, fwd=None, rounds=MAX_ROUNDS, log=None, **kw):
    """{dead unit id: [why]} of `dead` [(id, geom)] against `built` [geom] and `towns` [(name, lon, lat, x, y, pop)]
    (lon, lat for the line, Mollweide x, y for where it lies); `land` [geom] is other land (another country not
    built), so a border with it is not coast. Keywords override the module's figures."""
    town_pop, pair_km = kw.get("town_pop", TOWN_POP), kw.get("pair_km", PAIR_KM)
    min_km, hole, land_share = kw.get("line_km", LINE_KM), kw.get("hole", HOLE_SHARE), kw.get("land_share", LAND_SHARE)
    if geod is None:
        from pyproj import Geod, Transformer
        geod = Geod(ellps="WGS84")
        fwd = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    ids = [d[0] for d in dead]
    geoms = [d[1] for d in dead]
    tree = STRtree(geoms)
    land_tree = STRtree(list(land)) if land else None
    big = [t for t in towns if t[5] >= town_pop]
    tpt = [shapely.Point(t[3], t[4]) for t in big]
    in_unit = collections.defaultdict(list)
    if tpt:
        for ti, k in zip(*tree.query(tpt, predicate="intersects")):
            in_unit[int(k)].append(big[ti])
    picks, done = {}, set()
    for r in range(rounds):
        now = list(built) + [geoms[k] for k in done]
        btree = STRtree(now)
        ends = [t for t, p in zip(big, tpt) if len(btree.query(p, predicate="intersects"))]
        new = collections.defaultdict(list)
        # between
        lon = np.array([e[1] for e in ends])
        lat = np.array([e[2] for e in ends])
        for i, e in enumerate(ends):
            if i + 1 >= len(ends):
                break
            d = geod.inv(np.full(len(ends) - i - 1, lon[i]), np.full(len(ends) - i - 1, lat[i]), lon[i + 1:], lat[i + 1:])[2]
            for jj in np.nonzero(d <= pair_km * 1000)[0]:
                f = ends[i + 1 + int(jj)]
                km, over = line_km((e[1], e[2]), (f[1], f[2]), geod, fwd, geoms, tree)
                for k, x in over.items():
                    if k not in done and x >= min_km and not new[k]:
                        new[k].append(f"between {e[0]} and {f[0]} ({km:.0f} km apart, {x:.0f} km over it)")
        for k, g in enumerate(geoms):
            if k in done:
                continue
            others = STRtree([geoms[j] for j in range(len(geoms)) if j != k and j not in done]) if len(geoms) > 1 else None
            share, land_part = border_shares(g, btree, others, land_tree)
            if in_unit[k] and share > 0:
                t = max(in_unit[k], key=lambda t: t[5])
                new[k].append(f"holds {t[0]} ({t[5] / 1e3:.0f}k)")
            if land_part >= land_share and share >= hole:
                new[k].append(f"a hole, {share:.0%} of its land border built")
        new = {k: v for k, v in new.items() if v}
        if not new:
            break
        for k, why in new.items():
            picks[ids[k]] = [f"round {r + 1}: " + w for w in why]
            done.add(k)
            if log:
                log(f"  gap pick, round {r + 1}: {ids[k]}: {'; '.join(why)}")
    return picks


def natural_earth_towns(path, fwd, min_pop=TOWN_POP):
    """[(name, lon, lat, x, y, pop)] of Natural Earth's populated places of min_pop or more."""
    import pyogrio
    df = pyogrio.read_dataframe(f"zip://{path}", columns=["name", "pop_max"])
    out = []
    for n, p, g in zip(df.name, df.pop_max, df.geometry):
        if p >= min_pop:
            x, y = fwd.transform(g.x, g.y)
            out.append((n, g.x, g.y, x, y, float(p)))
    return out


def corridor_strip(routes, km, fwd):
    """The land within km of the routes [[(lon, lat), ...]], Mollweide."""
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    lines = []
    for route in routes:
        pts = []
        for a, c in zip(route, route[1:]):
            d = geod.inv(a[0], a[1], c[0], c[1])[2] / 1000.0
            pts += [a] + geod.npts(a[0], a[1], c[0], c[1], max(int(d / 5), 1))
        pts.append(route[-1])
        xy = np.array(pts)
        x, y = fwd.transform(xy[:, 0], xy[:, 1])
        lines.append(LineString(np.column_stack([x, y])))
    return shapely.union_all([ln.buffer(km * 1000.0) for ln in lines])


def corridor_split(units, split_ids, strip, min_km2=5.0):
    """([(id, name, geom)], corridor ids): each unit of split_ids cut along the strip, the part inside as `<id>~r` and
    the rest keeping its id; a unit with under min_km2 outside the strip stays whole and is a corridor unit itself."""
    out, corridor = [], set()
    for uid, name, g in units:
        if uid not in split_ids:
            out.append((uid, name, g))
            continue
        inside = g.intersection(strip)
        rest = g.difference(strip)
        if inside.area < min_km2 * 1e6:
            out.append((uid, name, g))
        elif rest.area < min_km2 * 1e6:
            out.append((uid, name, g))
            corridor.add(uid)
        else:
            parts = [p for p in shapely.get_parts(inside) if p.geom_type in ("Polygon", "MultiPolygon")]
            out.append((f"{uid}~r", name, shapely.union_all(parts)))
            out.append((uid, name, rest))
            corridor.add(f"{uid}~r")
    return out, corridor
