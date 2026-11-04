"""Overpass `out geom` JSON (raw/osm_piers.json, osm_parks.json, osm_lakes.json) to shapely
geometries in lon/lat. Closed ways are polygons; multipolygon relations are assembled from their
outer and inner member ways; open ways (a linear pier, a groyne) become a strip if `line_m` is set.

A multipolygon is only trusted when its rings close: Overpass returns member geometry only inside
the box asked for, so a relation that crosses the edge of a fetch tile arrives with open ends. Those
are closed by `polygonize` (which drops open lines), never by joining the ends, so an unfinished
relation gives less area, not a wrong shape.
"""
import json

from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge, polygonize, unary_union


def _rings(members):
    lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]]) for m in members
             if m.get("geometry") and len(m["geometry"]) > 1]
    if not lines:
        return []
    return list(polygonize(unary_union(linemerge(lines) if len(lines) > 1 else lines)))


def shape_of(el, line_m=None):
    """One element as a shapely geometry, or None."""
    if el["type"] == "way":
        c = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(c) >= 4 and c[0] == c[-1]:
            g = Polygon(c)
        elif line_m and len(c) >= 2:
            g = LineString(c).buffer(line_m / 111000, cap_style=2)
        else:
            return None
    elif el["type"] == "relation":
        ways = [m for m in el.get("members", []) if m["type"] == "way"]
        outer = _rings([m for m in ways if m.get("role") in ("outer", "")])
        inner = _rings([m for m in ways if m.get("role") == "inner"])
        if not outer:
            return None
        g = unary_union(outer)
        if inner:
            g = g.difference(unary_union(inner))
    else:
        return None
    if not g.is_valid:
        g = g.buffer(0)
    return None if g.is_empty else g


def load(path, line_m=None):
    els = json.loads(open(path, encoding="utf-8").read())["elements"]
    return [g for g in (shape_of(el, line_m) for el in els) if g is not None]


def load_all(name, line_m=None):
    """raw/osm_<name>.json (ways) and raw/osm_<name>_rel.json (relations, with their members)."""
    import pathlib
    raw = pathlib.Path(__file__).parent / "raw"
    out = []
    for f in (raw / f"osm_{name}.json", raw / f"osm_{name}_rel.json"):
        if f.exists():
            out += load(f, line_m)
    return out
