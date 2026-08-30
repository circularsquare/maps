"""
Turn Overpass `out geom` elements into shapely geometry.

Overpass hands back a relation as a bag of member ways, each with its own coordinate
list, in no particular order and no particular direction. Nobody hands you a ring. So
the work here is ring assembly: repeatedly take a segment, then keep gluing on whichever
remaining segment shares an endpoint (reversing it if it is the far end that matches)
until the ends meet. Only then is it a ring.

Two things bite here and both are handled:

  - **Direction is not reliable.** A member way may run either way round; matching only
    head-to-tail would silently drop half the boundaries. We check both ends.
  - **Not every ring closes.** Real city boundaries have gaps in OSM, and a bbox query
    cuts ways at the box edge by design. An unclosed chain is not an error to raise —
    for a city boundary it is usually still a usable outline — so `assemble` closes any
    chain that got most of the way there and discards the rest, reporting counts rather
    than throwing.

`inner` rings become holes, assigned to whichever outer ring contains them, because a
relation lists all its inners flat with no indication of which outer they belong to.

Coordinates are (lon, lat) — shapely's order, not Overpass's.
"""

import shapely.geometry as sg
from shapely.validation import make_valid

# A chain whose two ends are within this many degrees is treated as closed. ~1e-7 deg is
# roughly a centimetre; the gaps worth bridging are float noise and duplicated nodes,
# not real topology.
SNAP = 1e-7

# Below this, a "ring" is a degenerate sliver rather than a polygon.
MIN_RING = 4


def _pts(way):
    """Overpass geometry list -> [(lon, lat)]. `None` entries appear when a way runs
    outside the query box; they break a chain rather than shifting it."""
    g = way.get("geometry") or []
    return [(p["lng"] if "lng" in p else p["lon"], p["lat"]) for p in g if p]


def _close_enough(a, b):
    return abs(a[0] - b[0]) < SNAP and abs(a[1] - b[1]) < SNAP


def assemble(segments):
    """Glue coordinate segments into closed rings. Returns (rings, n_dropped)."""
    pool = [list(s) for s in segments if len(s) >= 2]
    rings, dropped = [], 0
    while pool:
        chain = pool.pop()
        while not _close_enough(chain[0], chain[-1]):
            for i, seg in enumerate(pool):
                if _close_enough(seg[0], chain[-1]):
                    chain.extend(seg[1:])
                    pool.pop(i)
                    break
                if _close_enough(seg[-1], chain[-1]):
                    chain.extend(reversed(seg[:-1]))
                    pool.pop(i)
                    break
            else:
                break  # nothing connects; chain is as long as it will get
        if len(chain) >= MIN_RING:
            if not _close_enough(chain[0], chain[-1]):
                chain.append(chain[0])  # force it shut, see module docstring
            rings.append(chain)
        else:
            dropped += 1
    return rings, dropped


def _build(outers, inners):
    """Outer rings + flat inner rings -> Polygon/MultiPolygon, holes matched by
    containment."""
    polys = []
    for ring in outers:
        try:
            shell = sg.Polygon(ring)
        except Exception:
            continue
        if shell.is_empty:
            continue
        holes = []
        for h in inners:
            try:
                hp = sg.Polygon(h)
            except Exception:
                continue
            if not hp.is_empty and shell.contains(hp.representative_point()):
                holes.append(h)
        try:
            polys.append(sg.Polygon(ring, holes))
        except Exception:
            polys.append(shell)
    if not polys:
        return None
    geom = polys[0] if len(polys) == 1 else sg.MultiPolygon(polys)
    if not geom.is_valid:
        # Self-touching boundaries are common in OSM and fatal to `contains`.
        geom = polygonal(make_valid(geom))
    return geom


def polygonal(geom):
    """Keep only the polygonal parts of a geometry.

    `make_valid` on a self-touching boundary routinely returns a GeometryCollection
    with the stray lines and points that the repair split off. Those are not area, and
    leaving them in makes `contains` answer a subtly different question than the caller
    asked — so they are dropped rather than carried.
    """
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        parts = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else sg.MultiPolygon(
            [p for g in parts for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])]
        )
    return None


def shape_of(el):
    """One Overpass element -> shapely geometry, or None if it has no usable shape.

    Nodes return a Point, so a caller can treat "shape or pin" uniformly and decide
    later which it got by checking `geom_type`.
    """
    t = el.get("type")
    if t == "node":
        if el.get("lat") is None:
            return None
        return sg.Point(el["lon"], el["lat"])

    if t == "way":
        pts = _pts(el)
        if len(pts) < MIN_RING:
            return sg.LineString(pts) if len(pts) >= 2 else None
        if not _close_enough(pts[0], pts[-1]):
            pts.append(pts[0])
        try:
            p = sg.Polygon(pts)
        except Exception:
            return None
        return p if p.is_valid else polygonal(make_valid(p))

    if t == "relation":
        outers, inners = [], []
        for m in el.get("members") or []:
            if m.get("type") != "way":
                continue
            pts = _pts(m)
            if len(pts) < 2:
                continue
            (inners if m.get("role") == "inner" else outers).append(pts)
        o_rings, _ = assemble(outers)
        i_rings, _ = assemble(inners)
        return _build(o_rings, i_rings)

    return None


def centroid_of(el):
    """(lon, lat) for any element, polygon or node. `representative_point` rather than
    `centroid` because a centroid can fall outside a crescent-shaped district, and this
    point is what decides which city the unit is assigned to."""
    if el.get("type") == "node" and el.get("lat") is not None:
        return (el["lon"], el["lat"])
    c = el.get("center")
    if c:
        return (c["lon"], c["lat"])
    g = shape_of(el)
    if g is None or g.is_empty:
        return None
    p = g.representative_point()
    return (p.x, p.y)
