"""US county shapes for drawn boundaries (T-352): TIGER/Line 2020 counties, which include the water
inside each county out to the state and county lines at sea and in the lakes (the cartographic
boundary file cb_2020_us_county_500k is clipped to the shoreline, so a city hugged its coast).

Full-resolution TIGER is too heavy to ship (every pack's boundary is fetched at startup), so the
county linework is simplified once, for the whole contiguous US, TOPOLOGICALLY: all county rings are
noded together (a shared border becomes one line), each line between junctions is simplified once
(Douglas-Peucker, TOLERANCE_DEG), and the faces are rebuilt and given back to their counties. Two
neighbouring counties, in one pack or two, therefore keep sharing exactly the same simplified border:
packs' boundaries stay disjoint. The result is cached in data/work/tiger_counties_simple.json
({GEOID: GeoJSON geometry}); delete it to rebuild (a few minutes, 3 GB).

    from tiger_counties import counties
    geoms = counties(fetch)     # {"36061": shapely geometry, ...}

Three counties (19129, 31153, 51127) lose their face to the noding (a simplified line crosses a
neighbour's) and are simplified on their own, so their borders may differ from a neighbour's by up
to the tolerance; no pack holds them. Alaska, Hawaii and Puerto Rico, which border no other state,
are simplified the same way one state at a time (`counties(fetch, state)`, T-1016), each cached
beside the main file, so the contiguous US cache is left as it was.
"""
import json
import os

import shapefile  # pyshp
import shapely
from shapely.geometry import LineString, mapping, shape
from shapely.strtree import STRtree

URL = "https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip"
TOLERANCE_DEG = 0.0002  # about 22 m north-south: under half a pixel at zoom 11
SKIP_STATES = {"02", "15"}  # Alaska, Hawaii; FIPS above 56 are territories
OWN_STATES = {"02", "15", "72"}  # each simplified on its own: Alaska, Hawaii, Puerto Rico
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "work",
                     "tiger_counties_simple.json")
_memo = {}


def _contiguous(st):
    return st not in SKIP_STATES and int(st) <= 56


def _rings(geom):
    for poly in getattr(geom, "geoms", [geom]):
        yield LineString(poly.exterior.coords)
        for hole in poly.interiors:
            yield LineString(hole.coords)


def _build(fetch, keep=_contiguous):
    r = shapefile.Reader(fetch(URL))
    fields = [f[0] for f in r.fields[1:]]
    gi, si = fields.index("GEOID"), fields.index("STATEFP")
    ids, polys, lines = [], [], []
    for i, rec in enumerate(r.iterRecords()):
        st = rec[si]
        if not keep(st):
            continue
        g = shapely.make_valid(shape(r.shape(i).__geo_interface__))
        ids.append(rec[gi])
        polys.append(g)
        lines.extend(_rings(g))
    print(f"TIGER counties: {len(ids)} counties, {sum(len(l.coords) for l in lines):,} ring vertices", flush=True)
    noded = shapely.unary_union(lines)  # one piece per segment: shared ones once
    merged = shapely.line_merge(noded)  # joined again wherever only two pieces meet
    edges = [shapely.simplify(e, TOLERANCE_DEG, preserve_topology=True) for e in merged.geoms]
    print(f"  noded into {len(edges):,} edges, "
          f"{sum(len(e.coords) for e in edges):,} vertices after simplifying", flush=True)
    faces = list(shapely.polygonize(edges).geoms)
    tree = STRtree(polys)
    by = {}
    for f in faces:
        p = f.representative_point()
        hit = [j for j in tree.query(p, predicate="intersects")]
        if len(hit) > 1:  # the county polygons overlap there (Missouri River): the one it mostly lies in
            hit.sort(key=lambda j: -f.intersection(polys[j]).area)
        if hit:
            by.setdefault(ids[hit[0]], []).append(f)
    out = {cid: shapely.union_all(fs) for cid, fs in by.items()}
    lost = sorted(set(ids) - set(out))
    print(f"  {len(faces):,} faces -> {len(out)} counties; without a face: {lost}", flush=True)
    # a county whose face did not close (simplifying a line can cross its neighbour's) is simplified
    # on its own: its border with a neighbour may then differ by up to the tolerance
    for cid in lost:
        out[cid] = shapely.simplify(polys[ids.index(cid)], TOLERANCE_DEG, preserve_topology=True)
    return out


def counties(fetch, state=None):
    """{GEOID: simplified TIGER/Line county geometry}, cached in data/work: the contiguous US, or
    one of OWN_STATES (its 2-digit FIPS)."""
    if state in _memo:
        return _memo[state]
    assert state is None or state in OWN_STATES, state
    path = CACHE if state is None else CACHE.replace(".json", f"_{state}.json")
    if not os.path.exists(path):
        out = _build(fetch) if state is None else _build(fetch, lambda st: st == state)

        def rnd(c):
            if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
                return [round(c[0], 6), round(c[1], 6)]
            return [rnd(x) for x in c]

        doc = {k: {"type": mapping(g)["type"], "coordinates": rnd(mapping(g)["coordinates"])} for k, g in out.items()}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".part", "w") as f:
            json.dump(doc, f, separators=(",", ":"))
        os.replace(path + ".part", path)
    with open(path) as f:
        doc = json.load(f)
    _memo[state] = {k: shape(v) for k, v in doc.items()}
    return _memo[state]
