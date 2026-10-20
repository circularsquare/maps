"""The point and area features of a Geofabrik extract that the tier 0 and Canada builds read,
read from the .osm.pbf once and cached beside it (T-363).

    python pipeline/osm_features.py <extract> [<extract> ...]     read ahead (Geofabrik names)

Three FlatGeobuf files per extract in data/raw/osm/, with a spatial index:

- `<extract>.points.fgb`: GDAL OSM points with any shop, amenity, office, craft, tourism,
  leisure or healthcare tag, or a place tag (columns osm_id, name, place, other_tags).
- `<extract>.stations.fgb`: the points with railway=station or halt, same columns, made in the
  same pass (the gate's station check, T-770).
- `<extract>.areas.fgb`: GDAL OSM multipolygons with any shop, amenity, office, craft, tourism or
  leisure, an airport terminal, a healthcare tag, a beach or a place tag; closed ways with
  commercial, industrial or retail land, relations with any land use; plus every building
  within 0.001 degrees of a hospital site (what enrich_medical's building floors read). Columns
  as GDAL names them (MP_COLS), and member_of (see _read_areas).

Each file is a superset of the per-city GDAL reads the scripts made of the .pbf before: `read`
applies the reader's own bbox, mask and OGR SQL filter to it and returns the rows a GDAL read
of the .pbf gives, in GDAL's order (`seq`: node id; relation id, then way id). Four GDAL passes
in four processes (points; closed ways; relations; an unfiltered one), about 6 minutes for
England. The filtering is done here in pandas, batch by batch. Water (`osm_water.py`, its own
osmconf) and roads (`routes.py`, lines) cache separately.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSM = os.path.join(ROOT, "data", "raw", "osm")

PT_COLS = ["osm_id", "name", "place", "other_tags"]
MP_COLS = ["osm_id", "osm_way_id", "name", "aeroway", "amenity", "building", "craft", "landuse", "leisure",
           "natural", "office", "place", "shop", "tourism", "other_tags"]
PLACES = ["city", "town", "borough", "suburb", "quarter", "village", "neighbourhood", "hamlet"]  # places.KINDS
PLACE_KINDS = "(" + ",".join(f"'{k}'" for k in PLACES) + ")"
PT_KEYS = ["shop", "amenity", "office", "craft", "tourism", "leisure", "healthcare"]
STATION_VALUES = ["station", "halt"]  # the gate's railway=station|halt nodes (pack_gate.stations, T-770)
MP_KEYS = ["shop", "amenity", "office", "craft", "tourism", "leisure"]
HOSPITAL_PAD = 0.001  # degrees; enrich's mask is the sites + 0.0005
WAY = 1 << 40  # GDAL emits multipolygon relations first, then closed ways


def _stem(extract):
    return os.path.join(OSM, extract.replace("/", "_"))


def points_path(extract):
    return _stem(extract) + ".points.fgb"


def stations_path(extract):
    return _stem(extract) + ".stations.fgb"


def areas_path(extract):
    return _stem(extract) + ".areas.fgb"


def _gdal():
    import pyogrio
    # the node index in memory rather than a temp file (GDAL's default is 100 MB)
    pyogrio.set_gdal_config_options({"OSM_MAX_TMPFILE_SIZE": "6000"})
    return pyogrio


def _write(df, out, what, t0, **kw):
    import pyogrio
    _in_order(df["seq"].to_numpy(), what)
    pyogrio.write_dataframe(df, out + ".part.fgb", driver="FlatGeobuf", SPATIAL_INDEX="YES", **kw)
    os.replace(out + ".part.fgb", out)
    print(f"  {os.path.basename(out)}: {len(df):,} {what}, {os.path.getsize(out) / 1e6:.0f} MB; "
          f"{time.time() - t0:.0f} s", flush=True)


def _batches(pbf, layer, cols, where=None):
    """The layer as pyarrow record batches (geometry as WKB)."""
    from pyogrio.raw import open_arrow
    _gdal()
    with open_arrow(pbf, layer=layer, columns=cols, where=where, use_pyarrow=True) as (meta, reader):
        for batch in reader:
            yield meta["geometry_name"] or "wkb_geometry", batch


# Filters in pandas, ignoring case as OGR SQL does: pyarrow 21's match_substring misses some
# matches in a 20,000-row batch that it finds in a short one (T-363).
def _lower(b, col):
    return b[col].to_pandas().str.lower()


def _has(low, text):
    return low.str.contains(text, regex=False, na=False).to_numpy(bool)


def _is(low, values):
    return low.isin(values).to_numpy(bool)


def _set(b, col):
    return b[col].to_pandas().notna().to_numpy(bool)


def _frame(tables, geom, cols):
    """pyarrow tables -> GeoDataFrame with `cols`, a shapely geometry and no index."""
    import geopandas as gpd
    import pyarrow as pa
    import shapely
    t = pa.concat_tables(tables) if tables else None
    if t is None or not len(t):
        return gpd.GeoDataFrame({c: [] for c in cols}, geometry=[], crs="EPSG:4326")
    df = t.select(cols).to_pandas()
    return gpd.GeoDataFrame(df, geometry=shapely.from_wkb(t[geom].to_numpy(zero_copy_only=False)), crs="EPSG:4326")


def _read_points(extract, pbf):
    import pyarrow as pa
    t0 = time.time()
    keep, rail, n = [], [], 0
    for geom, b in _batches(pbf, "points", PT_COLS):
        n += len(b)
        other = _lower(b, "other_tags")
        m = _is(_lower(b, "place"), PLACES)
        for k in PT_KEYS:
            m |= _has(other, f'"{k}"=>')
        keep.append(pa.Table.from_batches([b.filter(pa.array(m))]))
        st = np.zeros(len(b), bool)
        for v in STATION_VALUES:
            st |= _has(other, f'"railway"=>"{v}"')
        rail.append(pa.Table.from_batches([b.filter(pa.array(st))]))
    df = _frame(keep, geom, PT_COLS)
    df["seq"] = df["osm_id"].astype(np.int64)
    print(f"  {extract}: {n:,} points read; {time.time() - t0:.0f} s", flush=True)
    _write(df, points_path(extract), "points", t0)
    st = _frame(rail, geom, PT_COLS)
    st["seq"] = st["osm_id"].astype(np.int64)
    _write(st, stations_path(extract), "stations", t0, geometry_type="Point")  # an extract with none (Ceuta)


def _area_pass(pbf, where, landuse):
    """(kept rows, other buildings, rows read, geometry column) of the multipolygons matching
    `where`; `landuse` the land use values kept (None: any)."""
    import pyarrow as pa
    keep, bld, n, geom = [], [], 0, "wkb_geometry"
    for geom, b in _batches(pbf, "multipolygons", MP_COLS, where):
        n += len(b)
        m = ((_set(b, "landuse") if landuse is None else _is(_lower(b, "landuse"), landuse))
             | _is(_lower(b, "aeroway"), ["terminal"])
             | _has(_lower(b, "other_tags"), '"healthcare"=>') | _is(_lower(b, "natural"), ["beach"])
             | _is(_lower(b, "place"), PLACES))
        for k in MP_KEYS:
            m |= _set(b, k)
        keep.append(pa.Table.from_batches([b.filter(pa.array(m))]))
        bld.append(pa.Table.from_batches([b.filter(pa.array(_set(b, "building") & ~m))]))
    return keep, bld, n, geom


def _read_relations(extract, pbf):
    """Multipolygon relations alone (a pass of their own: see _read_areas), kept ones and other
    buildings, to a temporary file the areas pass merges. Any land use is kept: a reader filtering
    on `landuse IS NOT NULL` gets such a relation, and GDAL then leaves out its member ways."""
    t0 = time.time()
    keep, bld, n, geom = _area_pass(pbf, "osm_id IS NOT NULL", None)
    df = _frame(keep + bld, geom, MP_COLS)
    df["kept"] = np.r_[np.ones(sum(len(t) for t in keep), np.int8), np.zeros(sum(len(t) for t in bld), np.int8)]
    import pyogrio
    pyogrio.write_dataframe(df, _stem(extract) + ".relations.part.fgb", driver="FlatGeobuf")
    print(f"  {extract}: {n:,} relations read, {int(df.kept.sum()):,} kept; {time.time() - t0:.0f} s", flush=True)


def _ring_key(ring):
    """A closed ring's points, order and start free: GDAL builds a relation's ring from its member
    ways' node coordinates, so a closed member way's ring is the same point set."""
    import shapely
    c = shapely.get_coordinates(ring)[:-1]
    return c[np.lexsort((c[:, 1], c[:, 0]))].tobytes()


def _read_shown_ways(extract, pbf):
    """The closed way ids an unfiltered read reports (one with every relation in it), to a
    temporary .npy: the ways missing from it are the ones some relation hides."""
    t0 = time.time()
    ids = [b["osm_way_id"].to_pandas().dropna().astype(np.int64).to_numpy() for _, b in
           _batches(pbf, "multipolygons", ["osm_way_id"])]
    np.save(_stem(extract) + ".shown.part.npy", np.concatenate(ids))
    print(f"  {extract}: {sum(map(len, ids)):,} ways in an unfiltered read; {time.time() - t0:.0f} s", flush=True)


def _members(ways, hidden, rels):
    """Per way: ',id,...,' of the cached relations with its ring among theirs, for the ways an
    unfiltered read hides (others: None; a hidden way whose relation is not cached hides from no
    reader's result)."""
    import shapely
    rings = {}
    for rid, g in zip(rels["osm_id"], rels.geometry):
        for p in getattr(g, "geoms", [g]):
            for r in [p.exterior, *p.interiors]:
                rings.setdefault(_ring_key(r), []).append(rid)
    out = [None] * len(ways)
    geoms = ways.geometry.to_numpy()
    for i in np.flatnonzero(hidden):
        g = geoms[i]
        ids = rings.get(_ring_key(g.geoms[0].exterior if g.geom_type == "MultiPolygon" else g.exterior))
        if ids:
            out[i] = "," + ",".join(ids) + ","
    return out


def _read_areas(extract, pbf):
    """The multipolygons the readers use, plus every building near a hospital site.

    GDAL leaves out a closed way that is a member of a multipolygon relation it reports in the
    same read, even one with tags of its own (a hospital inside a campus relation), so what a read
    gives depends on its filter. Ways and relations are read in separate passes, which reports
    every way, and a third, unfiltered, says which ways a relation hides; each of those records its
    cached relations (`member_of`), and `read` leaves it out when one of them is in that read's
    result, as GDAL would (T-363). Three passes in three processes."""
    import pandas as pd
    import pyarrow as pa
    import pyogrio
    import shapely
    t0 = time.time()
    me = [sys.executable, "-u", os.path.abspath(__file__)]
    kids = [subprocess.Popen(me + ["--relations", extract]), subprocess.Popen(me + ["--shown", extract])]
    keep, bld, n, geom = _area_pass(pbf, "osm_way_id IS NOT NULL", ["industrial", "commercial", "retail"])
    ways = _frame(keep, geom, MP_COLS)
    print(f"  {extract}: {n:,} closed ways read, {len(ways):,} kept; {time.time() - t0:.0f} s", flush=True)
    if any(k.wait() for k in kids):
        raise RuntimeError(f"{extract}: the relations or unfiltered pass failed")
    tmp = _stem(extract) + ".relations.part.fgb"
    rel = pyogrio.read_dataframe(tmp)
    os.remove(tmp)
    shown = np.load(_stem(extract) + ".shown.part.npy")
    os.remove(_stem(extract) + ".shown.part.npy")
    rk, rb = rel[rel.kept == 1].drop(columns="kept"), rel[rel.kept == 0].drop(columns="kept")
    _in_order(_area_seq(ways), "closed ways")
    df = pd.concat([rk, ways], ignore_index=True)
    hosp = (df["amenity"].str.lower() == "hospital").to_numpy(bool) | \
        _has(df["other_tags"].str.lower(), '"healthcare"=>"hospital"')
    sites = df.geometry.to_numpy()[hosp]
    sites = sites[~shapely.is_missing(sites)]
    if len(sites):
        tree = shapely.STRtree(shapely.buffer(sites, HOSPITAL_PAD))
        bt = pa.concat_tables(bld)
        hit = np.zeros(len(bt), bool)
        wkb = bt[geom].to_numpy(zero_copy_only=False)
        for i in range(0, len(bt), 1_000_000):
            g = shapely.from_wkb(wkb[i:i + 1_000_000], on_invalid="ignore")  # OSM has rings of 2 points (Ukraine); a missing geometry never meets a hospital
            hit[i + np.unique(tree.query(g, predicate="intersects")[0])] = True
        near = _frame([bt.filter(pa.array(hit))], geom, MP_COLS)
        rb = rb[np.isin(np.arange(len(rb)), np.unique(tree.query(rb.geometry.to_numpy(), predicate="intersects")[0]))]
        print(f"  {extract}: {len(near) + len(rb):,} of {len(bt) + len(rel) - len(rk):,} other buildings near "
              f"{len(sites):,} hospital sites; {time.time() - t0:.0f} s", flush=True)
        del bt, wkb
        df = pd.concat([df, rb, near], ignore_index=True)
    df["seq"] = _area_seq(df)
    df = df.sort_values("seq", kind="stable").reset_index(drop=True)
    is_rel = df["osm_id"].notna().to_numpy()
    w = df[~is_rel]
    hidden = ~np.isin(w["seq"].to_numpy() - WAY, shown)
    member = _members(w, hidden, df[is_rel])
    df["member_of"] = None
    df.loc[~is_rel, "member_of"] = member
    print(f"  {extract}: {int(hidden.sum()):,} cached ways hidden by a relation, "
          f"{sum(x is not None for x in member):,} by a cached one; {time.time() - t0:.0f} s", flush=True)
    _write(df, areas_path(extract), "areas", t0)


def _area_seq(df):
    import pandas as pd
    way = pd.to_numeric(df["osm_way_id"], errors="coerce")
    rel = pd.to_numeric(df["osm_id"], errors="coerce")
    return np.where(way.notna(), WAY + way.fillna(0), rel.fillna(0)).astype(np.int64)


def _in_order(seq, what):
    """GDAL's filtered reads give nodes and ways in id order, relations first (by id in every
    read T-363 compared); `read` restores that order from `seq`."""
    if len(seq) > 1 and not (np.diff(seq) > 0).all():
        raise RuntimeError(f"GDAL's {what} are not in id order; `read` would reorder them")


def build(extract, stations=True):
    """Make the cache files for the extract if missing (downloading it if need be). `stations`
    False: the points file alone answers for the points pass (a reader that needs no stations
    does not download an extract read before T-770 for them)."""
    import routes
    need_pts = not os.path.exists(points_path(extract)) or (stations and not os.path.exists(stations_path(extract)))
    need_areas = not os.path.exists(areas_path(extract))
    if not (need_pts or need_areas):
        return
    pbf = routes.download(extract)
    t0 = time.time()
    child = None
    if need_pts and need_areas:
        child = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--points", extract])
    elif need_pts:
        _read_points(extract, pbf)
    if need_areas:
        _read_areas(extract, pbf)
    if child and child.wait():
        raise RuntimeError(f"{extract}: the points pass failed ({child.returncode})")
    print(f"  {extract}: OSM features cached in {time.time() - t0:.0f} s", flush=True)


def label(extract):
    """An extract, or a list of them, as a source line names it."""
    return extract if isinstance(extract, str) else " + ".join(extract)


def read(extract, layer, columns=None, bbox=None, mask=None, where=None):
    """Rows of the cached `layer` ("points" or "multipolygons", as GDAL names the .pbf's) as
    pyogrio.read_dataframe would give them from the .pbf: same filters, same order. `extract`
    may be a list of extracts: Geofabrik's regions overlap along their borders, so an object in
    two of them is kept once (by `seq`, its OSM id; T-431)."""
    import pandas as pd
    import pyogrio
    exs = [extract] if isinstance(extract, str) else list(extract)
    areas = layer != "points"
    helpers = ["seq"] + (["osm_id", "member_of"] if areas else [])
    cols = None if columns is None else list(dict.fromkeys(list(columns) + helpers))
    parts = []
    for e in exs:
        build(e, stations=False)
        path = areas_path(e) if areas else points_path(e)
        parts.append(pyogrio.read_dataframe(path, columns=cols, bbox=bbox, mask=mask, where=where))
    df = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True).drop_duplicates("seq")
    if areas and len(df):
        rels = set(df["osm_id"].dropna())
        member = [any(r in rels for r in s.strip(",").split(",")) if isinstance(s, str) else False
                  for s in df["member_of"]]
        df = df[~np.array(member, bool)]
    drop = [c for c in helpers if columns is None or c not in columns]
    return df.sort_values("seq", kind="stable").drop(columns=drop).reset_index(drop=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    passes = {"--points": _read_points, "--relations": _read_relations, "--shown": _read_shown_ways}
    which = [p for flag, p in passes.items() if flag in sys.argv]
    if which:
        import routes
        which[0](args[0], routes.download(args[0]))
    else:
        for e in args:
            build(e)
