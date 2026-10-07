"""
regions/*.geojson -> data/processed/regions.pmtiles, one `regions` layer, z0 to MAX_ZOOM.

Tippecanoe is what ancestrydots uses for this and it runs under WSL, which is not working on
this machine, so this is a small polygon tiler of its own. Three things it has to do that a
plain clip-and-encode would not:

1. UNITS TOO SMALL TO SEE ARE COMBINED. England's 188,880 output areas are a few hundred
   metres across; at country zoom one tile would hold tens of thousands of them, each a
   sub-pixel sliver, and dropping them instead leaves London a hole. So at every zoom, the top
   one included (see MAX_ZOOM), a unit under SMALL_PX pixels square is pooled with touching small units of
   its country, grown outward until the pool is about POOL_PX pixels square (`grow_groups`;
   a square grid was tried first and showed as rows of blocks): geometry merged, counts
   summed. The pooled shape's share is then the true share of everyone in it, which is what a
   reader at that zoom would want averaged anyway. `m` on a feature is how many units it
   pools; it is absent on a unit drawn as itself. The viewer over-zooms MAX_ZOOM from there.

2. A FEATURE IS CUT INTO TILES BY REPEATED QUARTERING, not clipped once per tile. A Siberian
   region spans thousands of z10 tiles, and clipping its full outline against each one costs
   vertices x tiles; quartering costs vertices x depth.

3. SIMPLIFICATION IS SNAPPING TO A GRID (`snap`), a quarter pixel per zoom and an eighth at
   the top, so neighbours keep one shared border instead of each straightening it their own
   way. Pools merge on the same grid, exactly. Islands and holes under a pixel are dropped
   (`drop_specks`). The viewer covers any hairline left with a seam line.

Properties per feature are the regions file's own: c, u, p and one count per node and per
ancestor (regions.py).

Usage:
    python regions_tiles.py                 # every regions_<cc>.geojson listed in index.json
"""
import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
SRC = HERE / "data" / "processed" / "regions"
OUT = HERE / "data" / "processed" / "regions.pmtiles"

# Anita, 2026-10-04: regions mode does not need the finest units ("cut off at a point and use
# the bigger regions"), and the dot mode is where the fine detail lives. So the ladder stops at
# z8 and pooling runs there too: nothing under ~SMALL_PX z8 pixels (about 2.5 km square at the
# equator) is drawn as itself, and the viewer over-zooms z8 from there on.
MAX_ZOOM = 8
EXTENT = 4096
BUFFER = 64 / EXTENT        # of a tile's width, on every side
SMALL_PX = 4                # a unit under this many pixels square is pooled below MAX_ZOOM
POOL_PX = 8                 # ...growing through its neighbours until about this many square
LAT_MAX = 85.05112878


# ------------------------------------------------------------------------------- geometry
def to_unit_mercator(coords):
    """lon/lat -> [0,1] x [0,1], y down, as tile coordinates count."""
    lon = coords[:, 0]
    lat = np.clip(coords[:, 1], -LAT_MAX, LAT_MAX)
    x = (lon + 180.0) / 360.0
    s = np.sin(np.radians(lat))
    y = 0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)
    return np.column_stack([x, y])


def load(only=None):
    """(props list, geometry array in unit mercator)."""
    idx = only or json.loads((SRC / "index.json").read_text(encoding="utf-8"))["countries"]
    props, geoms = [], []
    for cc in idx:
        text = (SRC / f"regions_{cc}.geojson").read_text(encoding="utf-8")
        fc = json.loads(text)
        # GEOS reads the whole collection at once, in feature order; a geometry at a time
        # through json.dumps took eight minutes for the world
        gc = shapely.get_parts(shapely.from_geojson(text))
        assert len(gc) == len(fc["features"]), (cc, len(gc), len(fc["features"]))
        props.extend(f["properties"] for f in fc["features"])
        geoms.extend(gc)
    geoms = shapely.transform(np.array(geoms, dtype=object), to_unit_mercator)
    # A ring that crosses the antimeridian comes out spanning the whole world as one polygon
    # and would paint a band across every tile between. Fiji's eastern islands do. Such a part
    # is moved onto one continuous x (west half shifted by +1), then cut at x = 1 and the far
    # half shifted back. A multipolygon whose PARTS merely sit either side is fine as it is.
    parts, part_of = shapely.get_parts(geoms, return_index=True)
    b = shapely.bounds(parts)
    torn = (b[:, 2] - b[:, 0]) > 0.5
    if torn.any():
        fixed, fixed_of = [], []
        for p, i in zip(parts[torn], part_of[torn]):
            un = shapely.transform(p, lambda c: np.column_stack(
                [np.where(c[:, 0] < 0.5, c[:, 0] + 1, c[:, 0]), c[:, 1]]))
            un = polygonal(un)
            east = shapely.clip_by_rect(un, 0, -1, 1, 2)
            west = shapely.transform(shapely.clip_by_rect(un, 1, -1, 2, 2),
                                     lambda c: c - np.array([1.0, 0.0]))
            for g in (east, west):
                for q in shapely.get_parts(polygonal(g)):
                    fixed.append(q); fixed_of.append(i)
        bad = sorted({props[i]["c"] + ":" + str(props[i]["u"]) for i in part_of[torn]})
        print(f"  {int(torn.sum())} polygon parts crossing 180° cut in two: {bad[:8]}")
        keep = ~torn
        parts = np.concatenate([parts[keep], np.array(fixed, dtype=object)])
        part_of = np.concatenate([part_of[keep], np.array(fixed_of, dtype=np.int64)])
        geoms = np.array(_regroup(parts, part_of, len(geoms)), dtype=object)
    return props, geoms


def _regroup(parts, idx, n):
    out = [shapely.Polygon()] * n
    groups = defaultdict(list)
    for p, i in zip(parts, idx):
        groups[i].append(p)
    for i, ps in groups.items():
        out[i] = ps[0] if len(ps) == 1 else shapely.MultiPolygon(ps)
    return out


def polygonal(g):
    """g made valid, keeping only its polygon parts (make_valid can return stray lines)."""
    if g is None or g.is_empty:
        return shapely.Polygon()
    if not g.is_valid:
        g = shapely.make_valid(g)
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    ps = [p for p in shapely.get_parts(g) if p.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.union_all(ps) if ps else shapely.Polygon()


def grow_groups(geoms, ccs, areas, px):
    """Group ids for `geoms` (all small), grown outward through touching neighbours of the same
    country until a group is about POOL_PX pixels square.

    THE FIRST VERSION POOLED BY A SQUARE GRID, and Anita saw the grid at once: over eastern
    China the pools came out in rows of blocks, because every cell boundary was a straight
    line that no pool could cross. Growing through the units' own adjacency puts every pool
    boundary on a real unit border, so there is no lattice to see. Seeds are taken along a
    Hilbert curve, so each group starts beside the last one and they grow as neighbours rather
    than as scattered splinters; a group that is left much under target (squeezed in between
    finished groups) joins a neighbouring group."""
    n = len(geoms)
    tree = shapely.STRtree(geoms)
    # "touching" within half a pixel: neighbours simplified apart no longer quite meet
    a, b = tree.query(shapely.buffer(geoms, 0.5 * px, quad_segs=1), predicate="intersects")
    keep = (a != b) & (ccs[a] == ccs[b])
    a, b = a[keep], b[keep]
    o = np.argsort(a, kind="stable")
    a, b = a[o], b[o]
    start = np.searchsorted(a, np.arange(n + 1))
    nbrs = lambda u: b[start[u]:start[u + 1]]

    import geopandas as gpd
    from collections import deque
    pts = gpd.GeoSeries(shapely.point_on_surface(geoms))
    order = np.argsort(pts.hilbert_distance().to_numpy(), kind="stable")
    target = (POOL_PX * px) ** 2
    group = np.full(n, -1, dtype=np.int64)
    total = []
    for s in order:
        if group[s] >= 0:
            continue
        g = len(total)
        group[s] = g
        t = areas[s]
        frontier = deque([s])
        while frontier and t < target:
            u = frontier.popleft()
            for v in nbrs(u):
                if group[v] < 0:
                    group[v] = g
                    t += areas[v]
                    frontier.append(v)
                    if t >= target:
                        break
        total.append(t)
    # Fold the scraps: a group under a quarter of the target joins the first full-sized group
    # any of its members touches. Only full-sized groups are targets, so one pass settles it.
    scrap = np.array(total) < 0.25 * target
    remap = np.arange(len(total))
    for u in np.flatnonzero(scrap[group]):
        gu = group[u]
        if remap[gu] != gu:
            continue                # already folded through another member
        for v in nbrs(u):
            if not scrap[group[v]]:
                remap[gu] = group[v]
                break
    return remap[group]


def drop_specks(g, min_area):
    """g without the islands and holes under min_area, keeping its largest part whatever its
    size, so a unit that is one small island is never lost.

    Simplification alone never removes an island; each keeps at least four points, so Norway's
    coast stayed at ~140,000 points at world zoom, and buffering and merging that took over ten
    minutes a zoom. Under a pixel, none of it can be seen."""
    if g.geom_type not in ("Polygon", "MultiPolygon"):
        return g
    parts = shapely.get_parts(g)
    if len(parts) == 1 and not len(parts[0].interiors):
        return g
    areas = shapely.area(parts)
    keep = areas >= min_area
    keep[np.argmax(areas)] = True
    out = []
    for p in parts[keep]:
        holes = [r for r in p.interiors if shapely.area(shapely.Polygon(r)) >= min_area]
        out.append(p if len(holes) == len(p.interiors) else shapely.Polygon(p.exterior, holes))
    return out[0] if len(out) == 1 else shapely.MultiPolygon(out)


def snap(geoms, grid):
    """Every vertex onto a grid `grid` wide, which is how this file simplifies.

    NOT DOUGLAS-PEUCKER, AND THE REASON IS NEIGHBOURS. simplify() works on one polygon at a
    time, so two units sharing a border each straighten it their own way and the land shows
    through between them: hairlines at first, then, once pooled shapes were simplified again,
    dark wedges and cut corners all over the map (Anita, 2026-10-04, in Poland, Germany and
    London). Topology-aware simplification is shapely 2.1, which needs a newer Python than the
    3.9 this project runs on. Snapping gets the same guarantee more simply: a vertex the two
    sides share lands on the same grid point from both, so the border stays one border. Grids
    at successive zooms nest (each is a whole multiple of the one above), so the zooms agree."""
    # POINTWISE: round each vertex and nothing else, which is all the shared-border guarantee
    # needs. The default mode also rebuilds each shape's topology, and on the world's largest
    # outlines (Canada's Arctic coast) that ran past fifteen minutes without finishing.
    # Rounding can leave a spike collapsed onto itself; main() repairs the invalid few, and
    # `clip` guards the rest. simplify(0) then drops the repeated and collinear points the
    # rounding made, which is where the saving is; on a shared border both sides drop the same.
    out = shapely.set_precision(geoms, grid, mode="pointwise")
    out = shapely.simplify(out, 0, preserve_topology=False)
    # A shape narrower than the grid everywhere (an islet, an exclave) snaps to nothing; it
    # keeps a plainly simplified outline instead, since it has no neighbour to disagree with.
    gone = shapely.is_empty(out) & ~shapely.is_empty(geoms)
    if gone.any():
        out[gone] = shapely.simplify(geoms[gone], grid, preserve_topology=True)
    return out


def grid_at(z, top):
    """The snapping grid at zoom z: an eighth of a pixel at the top zoom, which the viewer
    over-zooms, a quarter below it. Both are whole multiples of the next zoom's grid."""
    return (0.125 if z == top else 0.25) / (256 * (1 << z))


def pooled(props, geoms, z, grid):
    """The feature set at zoom z, pooled from the set at z+1 (note 1). Pooling the zoom
    above's pools rather than the units afresh is a fraction of the union work. `geoms` must
    hold no empty geometry: point_on_surface of one yields no coordinates."""
    areas = shapely.area(geoms)
    px = 1.0 / (256 * (1 << z))
    small = areas < (SMALL_PX * px) ** 2
    out_p, out_g = [], []
    for i in np.flatnonzero(~small):
        out_p.append(props[i]); out_g.append(geoms[i])
    idx = np.flatnonzero(small)
    groups = defaultdict(list)
    if len(idx):
        ccs = np.array([props[i]["c"] for i in idx])
        for k, g in enumerate(grow_groups(geoms[idx], ccs, areas[idx], px)):
            groups[g].append(idx[k])
    for members in groups.values():
        cc = props[members[0]]["c"]
        if len(members) == 1:
            i = members[0]
            out_p.append(props[i]); out_g.append(geoms[i])
            continue
        # Members arrive snapped to the zoom above's grid, which nests in this one, so their
        # shared borders coincide and a union on this zoom's grid is exact: no slits between
        # them and nothing to close. (An exact union of separately simplified members was slow,
        # over five minutes a zoom at z3-z1, and left wedges between them.)
        gs = geoms[members]
        try:
            g = shapely.union_all(gs, grid_size=grid)
        except shapely.errors.GEOSException:
            g = shapely.union_all(shapely.make_valid(gs), grid_size=grid)
        # `m` counts units, so a pool of pools adds up its members' counts
        sums = defaultdict(int)
        units = 0
        for i in members:
            units += props[i].get("m", 1)
            for k, v in props[i].items():
                if k not in ("c", "u", "m") and isinstance(v, (int, float)):
                    sums[k] += v
        p = {"c": cc, "u": f"{units} areas", "m": units}
        p.update(sums)
        out_p.append(p); out_g.append(polygonal(g))
    out_g = np.array(out_g, dtype=object)
    keep = ~shapely.is_empty(out_g)
    return [p for p, k in zip(out_p, keep) if k], out_g[keep]


REDONE = [0]       # clips that came back bigger than their input and were redone exactly


def clip(g, box):
    """g within box. clip_by_rect is fast and assumes a valid polygon; handed an invalid one it
    can answer with the whole box, which drew as a solid tile-sized slab of one unit's colour
    (a Norwegian unit over the Baltic, a Spanish one over Paris, a square over Germany). A
    piece larger than what it was cut from is impossible, so that is the test, and the answer
    is then recomputed by a true intersection of the repaired shape."""
    try:
        c = shapely.clip_by_rect(g, *box)
    except shapely.errors.GEOSException:
        c = None
    if c is None or c.area > g.area * 1.0001 + 1e-18:
        REDONE[0] += 1
        c = polygonal(shapely.intersection(polygonal(shapely.make_valid(g)), shapely.box(*box)))
    return c


def cut(g, z, x, y, zt, out, key):
    """Quarter g (already clipped to tile z/x/y plus buffer) down to zoom zt."""
    if z == zt:
        out[(x, y)].append((key, g))
        return
    n = 1 << (z + 1)
    w = 1.0 / n
    b = BUFFER * w
    for dx in (0, 1):
        for dy in (0, 1):
            cx, cy = 2 * x + dx, 2 * y + dy
            c = clip(g, (cx * w - b, cy * w - b, (cx + 1) * w + b, (cy + 1) * w + b))
            if not c.is_empty:
                cut(c, z + 1, cx, cy, zt, out, key)


def tiles_at(geoms, z):
    """{(x, y): [(feature index, piece)]} for zoom z."""
    out = defaultdict(list)
    b = shapely.bounds(geoms)
    for i, g in enumerate(geoms):
        if g is None or g.is_empty:
            continue
        minx, miny, maxx, maxy = b[i]
        # the deepest zoom <= z whose one tile (plus buffer) holds the whole feature
        zs = z
        while zs > 0:
            n = 1 << zs
            if (int(np.floor(minx * n)) == int(np.floor(maxx * n))
                    and int(np.floor(miny * n)) == int(np.floor(maxy * n))):
                break
            zs -= 1
        n = 1 << zs
        x0 = min(int(np.floor(minx * n)), n - 1)
        y0 = min(int(np.floor(miny * n)), n - 1)
        if zs == 0:
            x0 = y0 = 0
        cut(g, zs, x0, y0, z, out, i)
    return out


# ------------------------------------------------------------------------------- encoding
_VCACHE = {}


def _v(n):
    """One protobuf varint. Pure Python and cached: numpy costs more than it saves on one."""
    b = _VCACHE.get(n)
    if b is None:
        out, v = bytearray(), n
        while True:
            c = v & 0x7F
            v >>= 7
            if v:
                out.append(c | 0x80)
            else:
                out.append(c)
                break
        b = bytes(out)
        if n < 1 << 16:
            _VCACHE[n] = b
    return b


def varints(a):
    """Protobuf varints of a non-negative int array, as bytes."""
    if len(a) < 64:
        return b"".join(_v(int(x)) for x in a)
    a = np.asarray(a, dtype=np.uint64)
    nb = np.ones(len(a), dtype=np.int64)
    for k in (7, 14, 21, 28, 35, 42, 49, 56):
        nb += a >= (np.uint64(1) << np.uint64(k))
    pos = np.cumsum(nb) - nb
    buf = np.zeros(int(nb.sum()), dtype=np.uint8)
    for k in range(int(nb.max())):
        m = nb > k
        byte = (a[m] >> np.uint64(7 * k)) & np.uint64(0x7F)
        cont = (nb[m] > k + 1).astype(np.uint64) << np.uint64(7)
        buf[pos[m] + k] = (byte | cont).astype(np.uint8)
    return buf.tobytes()


def _ld(tag, payload):
    return tag + _v(len(payload)) + payload


def ring_ints(ring, tx, ty, n):
    c = np.asarray(ring.coords)[:-1]
    q = np.empty((len(c), 2), dtype=np.int64)
    q[:, 0] = np.round((c[:, 0] * n - tx) * EXTENT)
    q[:, 1] = np.round((c[:, 1] * n - ty) * EXTENT)
    if len(q) > 1:
        keep = np.any(q != np.roll(q, 1, axis=0), axis=1)
        q = q[keep]
    return q


def area2(q):
    x, y = q[:, 0], q[:, 1]
    return int(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def geom_ints(g, tx, ty, n):
    """MVT polygon command stream for g, or None when nothing survives quantising."""
    cmds = []
    cur = np.zeros(2, dtype=np.int64)
    polys = [p for p in shapely.get_parts(g) if p.geom_type == "Polygon"]
    for p in polys:
        rings = [p.exterior] + list(p.interiors)
        for k, ring in enumerate(rings):
            q = ring_ints(ring, tx, ty, n)
            if len(q) < 3:
                if k == 0:
                    break           # the outline collapsed, so the holes go with it
                continue
            a = area2(q)
            if a == 0:
                if k == 0:
                    break
                continue
            if (k == 0) != (a > 0):  # exterior positive, holes negative (MVT 4.3.4.4)
                q = q[::-1]
            d = np.diff(np.vstack([cur, q]), axis=0)
            zz = (d << 1) ^ (d >> 63)
            cur = q[-1]
            cmds.append(np.array([9], dtype=np.int64))
            cmds.append(zz[0])
            cmds.append(np.array([2 | ((len(q) - 1) << 3)], dtype=np.int64))
            cmds.append(zz[1:].ravel())
            cmds.append(np.array([15], dtype=np.int64))
    if not cmds:
        return None
    return np.concatenate(cmds)


def pie_points(zp, zg, z):
    """The `pies` layer at zoom z: one point per shape, inside its largest part, carrying the
    shape's top-level counts only (a pie is drawn by family; the polygon carries the rest).
    `i` is the shape's index at this zoom and `z` the zoom, so the viewer can take exactly one
    zoom's points and never count a pie twice. Returns (tile key -> [(x, y, props)])."""
    out = defaultdict(list)
    n = 1 << z
    for i, (p, g) in enumerate(zip(zp, zg)):
        parts = shapely.get_parts(g)
        big = parts[int(np.argmax(shapely.area(parts)))] if len(parts) > 1 else g
        x, y = shapely.get_coordinates(shapely.point_on_surface(big))[0]
        tx, ty = min(int(x * n), n - 1), min(int(y * n), n - 1)
        q = {k: v for k, v in p.items() if "." not in k and k != "u"}
        q["i"] = i
        q["z"] = z
        out[(tx, ty)].append((x, y, q))
    return out


def encode_points(points, tx, ty, z):
    """One `pies` layer from [(x, y, props)] in unit mercator."""
    n = 1 << z
    keys, kidx, vals, vidx = [], {}, [], {}
    feats = bytearray()
    for x, y, props in points:
        px = int(round((x * n - tx) * EXTENT))
        py = int(round((y * n - ty) * EXTENT))
        tags = []
        for k, v in props.items():
            ki = kidx.get(k)
            if ki is None:
                ki = kidx[k] = len(keys); keys.append(k)
            vk = (type(v) is str, v)
            vi = vidx.get(vk)
            if vi is None:
                vi = vidx[vk] = len(vals)
                if isinstance(v, str):
                    vals.append(_ld(b"\x22", _ld(b"\x0a", v.encode("utf-8"))))
                else:
                    vals.append(_ld(b"\x22", b"\x28" + _v(max(int(v), 0))))
            tags += (ki, vi)
        geom = varints([9, (px << 1) ^ (px >> 63), (py << 1) ^ (py >> 63)])
        feats += _ld(b"\x12", _ld(b"\x12", varints(tags)) + b"\x18\x01" + _ld(b"\x22", geom))
    if not feats:
        return b""
    layer = bytearray(_ld(b"\x0a", b"pies"))
    layer += feats
    for k in keys:
        layer += _ld(b"\x1a", k.encode("utf-8"))
    for v in vals:
        layer += v
    layer += b"\x28" + _v(EXTENT) + b"\x78\x02"
    return _ld(b"\x1a", bytes(layer))


def encode_tile(pieces, props, tx, ty, z):
    n = 1 << z
    keys, kidx, vals, vidx = [], {}, [], {}
    feats = bytearray()
    for i, g in pieces:
        gi = geom_ints(g, tx, ty, n)
        if gi is None:
            continue
        tags = []
        for k, v in props[i].items():
            if v is None:
                continue
            ki = kidx.get(k)
            if ki is None:
                ki = kidx[k] = len(keys); keys.append(k)
            vk = (type(v) is str, v)
            vi = vidx.get(vk)
            if vi is None:
                vi = vidx[vk] = len(vals)
                if isinstance(v, str):
                    vals.append(_ld(b"\x22", _ld(b"\x0a", v.encode("utf-8"))))
                else:
                    vals.append(_ld(b"\x22", b"\x28" + _v(max(int(v), 0))))
            tags += (ki, vi)
        body = (_ld(b"\x12", varints(tags)) + b"\x18\x03" + _ld(b"\x22", varints(gi)))
        feats += _ld(b"\x12", body)
    if not feats:
        return None
    layer = bytearray(_ld(b"\x0a", b"regions"))
    layer += feats
    for k in keys:
        layer += _ld(b"\x1a", k.encode("utf-8"))
    for v in vals:
        layer += v
    layer += b"\x28" + _v(EXTENT) + b"\x78\x02"
    return _ld(b"\x1a", bytes(layer))


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-zoom", type=int, default=MAX_ZOOM)
    ap.add_argument("--countries", nargs="+", help="a subset, for a quick look")
    ap.add_argument("--out", type=Path, default=OUT, help="somewhere else, e.g. for a subset")
    args = ap.parse_args()
    out_path = args.out
    t0 = time.time()
    print("reading regions…")
    props, geoms = load(args.countries)
    print(f"  {len(props):,} regions from {len({p['c'] for p in props})} countries "
          f"({time.time() - t0:.0f}s)")
    # Onto the top zoom's grid straight away, BEFORE anything else touches the shapes.
    # regions.py ships full-detail outlines (its own simplifying caused gaps), and the top zoom
    # snaps to this grid anyway, so doing it first changes nothing drawn; done after the
    # validity repair below, that repair alone ran over twenty minutes on the full detail.
    t1 = time.time()
    raw = geoms
    geoms = snap(geoms, grid_at(args.max_zoom, args.max_zoom))
    print(f"  snapped to the z{args.max_zoom} grid ({time.time() - t1:.0f}s)", flush=True)
    # A ring collapsed by rounding or snapping is one GEOS refuses to clip.
    t1 = time.time()
    bad = ~shapely.is_valid(geoms)
    if bad.any():
        geoms[bad] = [polygonal(g) for g in geoms[bad]]
    # A unit narrower than the grid can come out of that repair as nothing, and its people
    # would then be missing from every pool it should join; it keeps its own outline instead.
    lost = shapely.is_empty(geoms) & ~shapely.is_empty(raw)
    if lost.any():
        geoms[lost] = [polygonal(g) for g in raw[lost]]
    del raw
    print(f"  {int(bad.sum()):,} invalid outlines repaired, {int(lost.sum())} kept unsnapped "
          f"({time.time() - t1:.0f}s)", flush=True)
    keep = ~shapely.is_empty(geoms)
    if not keep.all():
        print(f"  !! {int((~keep).sum()):,} regions have empty geometry, dropped")
        props = [p for p, k in zip(props, keep) if k]
        geoms = geoms[keep]
    b = shapely.bounds(geoms)

    # Top down, so each zoom pools from the one above it (`pooled`).
    levels = {}
    zp, zg = props, geoms
    for z in range(args.max_zoom, -1, -1):
        tz = time.time()
        grid = grid_at(z, args.max_zoom)
        zp, zg = pooled(zp, zg, z, grid)    # at the top zoom too (see MAX_ZOOM)
        zg = snap(zg, grid)                 # the simplification (`snap`)
        px = 1.0 / (256 * (1 << z))
        zg = np.array([drop_specks(g, px * px) for g in zg], dtype=object)
        # Not every shape arrives valid (Norway's NO082 does not), and an invalid one is what
        # clip_by_rect mishandles (`clip`)
        bad = ~shapely.is_valid(zg)
        if bad.any():
            zg[bad] = [polygonal(g) for g in zg[bad]]
        # a unit narrower than the grid everywhere collapses to nothing; it is under a pixel
        keep = ~shapely.is_empty(zg)
        if not keep.all():
            lost = sum(p.get("p", 0) for p, k in zip(zp, keep) if not k)
            print(f"  !! z{z}: {int((~keep).sum())} shapes collapsed on the grid "
                  f"({lost:,} people undrawn at this zoom)")
            zp = [p for p, k in zip(zp, keep) if k]
            zg = zg[keep]
        levels[z] = (zp, zg)
        print(f"  z{z:<2} {len(zp):>8,} features, {int(bad.sum()):,} repaired "
              f"({time.time() - tz:.0f}s)", flush=True)

    tmp = out_path.with_suffix(".pmtiles.tmp")
    total = 0
    with open(tmp, "wb") as f:
        w = Writer(f)
        for z in range(0, args.max_zoom + 1):
            tz = time.time()
            zp, zg = levels.pop(z)
            tiles = tiles_at(zg, z)
            pies = pie_points(zp, zg, z)
            order = sorted(set(tiles) | set(pies), key=lambda k: zxy_to_tileid(z, k[0], k[1]))
            nbytes, biggest = 0, 0
            for (x, y) in order:
                blob = (encode_tile(tiles[(x, y)], zp, x, y, z) or b"") if (x, y) in tiles else b""
                blob += encode_points(pies.get((x, y), []), x, y, z)
                if not blob:
                    continue
                gz = gzip.compress(blob, 6)
                w.write_tile(zxy_to_tileid(z, x, y), gz)
                nbytes += len(gz); biggest = max(biggest, len(gz))
            total += nbytes
            print(f"  z{z:<2} {len(zp):>8,} features  {len(tiles):>7,} tiles  "
                  f"{nbytes / 1e6:7.1f} MB  largest {biggest / 1e3:6.0f} KB  "
                  f"{REDONE[0]:,} clips redone ({time.time() - tz:.0f}s)", flush=True)
            REDONE[0] = 0
        w.finalize(
            {
                "tile_type": TileType.MVT,
                "tile_compression": Compression.GZIP,
                "min_zoom": 0,
                "max_zoom": args.max_zoom,
                "min_lon_e7": int((b[:, 0].min() * 360 - 180) * 1e7),
                "min_lat_e7": -850000000,
                "max_lon_e7": int((b[:, 2].max() * 360 - 180) * 1e7),
                "max_lat_e7": 850000000,
                "center_zoom": 2,
                "center_lon_e7": 0,
                "center_lat_e7": 0,
            },
            {"name": "religiondots-regions",
             "vector_layers": [{"id": "regions", "fields": {"c": "String", "u": "String",
                                                            "p": "Number", "m": "Number"}},
                               {"id": "pies", "fields": {"c": "String", "p": "Number",
                                                         "m": "Number", "i": "Number",
                                                         "z": "Number"}}]},
        )
    # `npx serve` holds a file open while a page reads it, which makes the swap fail on
    # Windows; wait for it rather than lose the build.
    for attempt in range(60):
        try:
            tmp.replace(out_path)
            break
        except PermissionError:
            if attempt == 59:
                raise SystemExit(f"{out_path.name} held open for a minute; the build is in "
                                 f"{tmp.name}")
            time.sleep(1)
    print(f"{out_path.name}: {out_path.stat().st_size / 1e6:.1f} MB ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
