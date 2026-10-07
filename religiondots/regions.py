"""
One polygon per counting unit, carrying that unit's counts, for the viewer's regions mode.

The dot map draws each unit's people as dots; regions mode shades the unit itself by the share
of the selected religion. Both read the same counts, cfg["counts"](), the (unit, node, count)
table the dots are allocated from, so the two modes cannot disagree about a number.

GEOMETRY, AND WHY IT IS NOT SIMPLY THE PLACEMENT LAYER MERGED BY UNIT

Where the placement layer is itself an admin layer (US tracts, UK output areas, India's
villages), merging it by unit is exact and that is what happens.

Most countries place on Kontur population hexes instead, and the first cut merged those too, on
the theory that a unit drawn over its populated land only would keep empty steppe from carrying
colour. Kazakhstan disproved it on sight: its hexes are so sparse that the map came out as
specks, a dot map again, and seventeen regions could not be told apart. So for a grid layer the
unit's REAL outline is found instead, from whatever admin file sits beside the grid in
data/geo/<cc>/. No country records which file its hexes were tagged from, and they are named
differently everywhere (kz_regions, np_units, my_districts), so it is found by test:

    every candidate polygon takes the unit most of the hex centres inside it belong to, and a
    file is accepted when that majority holds for 97% of the hexes AND 97% of the units get a
    polygon. The coarsest accepted file wins, being the nearest to the units themselves.

A file finer than the units passes the same test and is merged up. A polygon with no hex in it
(an empty desert district) takes the unit of the nearest hex. When nothing passes, the hexes
are merged as before and the build says so.

Counts are written for every node AND every ancestor of it, summed, so the viewer can read the
selected node's people with one `get` whatever level it sits at. `p` is the unit's total over
every node, which is the share's denominator.

Output: data/processed/regions/regions_<cc>.geojson with a .meta.json sidecar each, index.json
listing them, sources.json gathering the sidecars (which geometry, what could not be placed), and
failed.json when a country did not build. regions_tiles.py turns the lot into regions.pmtiles.

Usage:
    python regions.py --countries np my kz in us
    python regions.py --all --jobs 4                # every country with a dots file (~1 h)
    python regions.py --all --jobs 4 --skip-built   # resume
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.geometry

from countries import COUNTRIES
import kontur_cap

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
OUT = HERE / "data" / "processed" / "regions"

# Outlines are SNAPPED to a GRID-degree grid here, never simplified. Simplifying works unit by
# unit, so neighbours straighten their shared border differently and the tiles showed black
# specks between them (0.0001 simplify), and at 0.002 it shattered London's output areas.
# Not reducing at all made the files enormous (UK 360 MB, Canada 246 MB) and the tiler crawl.
# Snapping is the middle: a vertex both neighbours share lands on the same grid point from
# both, so the border stays one border, and metre-level coastline detail collapses away.
# 0.0002° (~22 m): finer than the z8 tiles can draw anywhere south of ~70°N (their finest grid
# is 0.0007° of longitude, scaled by cos(latitude) north-south); regions_tiles.py snaps again
# per zoom. 0.0001 left Canada at 223 MB, its Arctic coast still tens of millions of vertices.
SIMPLIFY = 0
GRID = 0.0002
DECIMALS = 5
PURITY = 0.97           # share of hexes whose candidate polygon's majority unit is their own
COVER = 0.97            # share of units that get at least one candidate polygon
MAX_CANDIDATE_MB = 300
SAMPLE = 300_000        # hex centres used for the test; plenty, and keeps the join quick


def unit_counts(cfg):
    """{unit: {node or ancestor: people}}, and {unit: total}."""
    df = cfg["counts"]()
    df = df[df["count"].notna() & (df["count"] > 0)]
    g = df.groupby(["unit", "node"])["count"].sum()
    by_unit = defaultdict(lambda: defaultdict(float))
    for (unit, node), n in g.items():
        parts = node.split(".")
        for i in range(1, len(parts) + 1):
            by_unit[str(unit)][".".join(parts[:i])] += float(n)
    totals = df.groupby("unit")["count"].sum()
    return by_unit, {str(u): float(v) for u, v in totals.items()}


def is_grid(src):
    name = Path(str(src)).name.lower()
    return kontur_cap.is_kontur_layer(src) or "hex" in name or "_grid" in name


def merge_by_unit(units, geoms, coverage=False):
    """{unit: one geometry}, simplified."""
    out = {}
    groups = pd.Series(np.arange(len(units))).groupby(np.asarray(units)).indices
    t0 = time.time()
    for i, (unit, idx) in enumerate(groups.items()):
        parts = geoms[idx]
        parts = parts[~shapely.is_empty(parts)]
        if not len(parts):
            continue
        g = None
        if coverage:
            # Hexes and admin units tile without overlap, which coverage_union assumes, and it
            # is several times faster than a general union. Off a true coverage it can return
            # something invalid, and then the general one runs.
            try:
                g = shapely.coverage_union_all(parts)
                if not g.is_valid:
                    g = None
            except Exception:
                g = None
        if g is None:
            g = shapely.union_all(shapely.make_valid(parts))
        g = shapely.simplify(g, SIMPLIFY, preserve_topology=True)
        if not g.is_empty:
            out[str(unit)] = g
        if (i + 1) % 1000 == 0:
            print(f"    merged {i + 1:,} / {len(groups):,} units ({time.time() - t0:.0f}s)")
    return out


def with_missing(shapes, place, cc, cfg):
    """Units the matched file has no polygon for, drawn from their own placement polygons and
    cut out of whatever unit's polygon swallowed them. Indonesia's matched file has 37 such
    units, 1.1M people, mostly cities whose kabupaten polygon still holds them; drawn as holes
    they would read as nobody living there."""
    missing = sorted(set(place["unit"].astype(str)) - set(shapes))
    if not missing:
        return shapes
    # Not water-clipped: water.clip caches by layer and expects the whole layer's rows.
    sub = place[place["unit"].astype(str).isin(missing)].reset_index(drop=True)
    own = merge_by_unit(sub["unit"].astype(str).to_numpy(), sub.geometry.to_numpy(),
                        coverage=True)
    if not own:
        return shapes
    hole = shapely.union_all(list(own.values()))
    tree_geoms = list(shapes.items())
    for u, g in tree_geoms:
        if shapely.intersects(g, hole):
            shapes[u] = shapely.difference(g, hole)
    shapes.update(own)
    print(f"    {len(own):,} units missing from the matched file drawn from their own hexes")
    return shapes


def find_outlines(cc, place, src):
    """The admin file beside a grid layer whose polygons line up with the units, as
    ({unit: geometry}, description), or (None, why) when no file passes."""
    folder = Path(str(src)).parent
    files = [p for ext in ("*.gpkg", "*.shp", "*.geojson") for p in folder.glob(ext)]
    files = [p for p in files if p.resolve() != Path(str(src)).resolve() and not is_grid(p)
             and p.stat().st_size < MAX_CANDIDATE_MB * 1e6]
    if not files:
        return None, f"no admin file beside {Path(str(src)).name}"

    pts = place[["unit"]].copy()
    pts["geometry"] = place.geometry.representative_point()
    pts = gpd.GeoDataFrame(pts, geometry="geometry", crs=place.crs)
    if len(pts) > SAMPLE:
        pts = pts.sample(SAMPLE, random_state=0)
    units = set(place["unit"])

    best, tried = None, []
    for f in files:
        try:
            cand = gpd.read_file(f)
        except Exception as e:
            tried.append(f"{f.name}: unreadable ({type(e).__name__})")
            continue
        cand = cand[cand.geometry.notna() & cand.geom_type.isin(["Polygon", "MultiPolygon"])]
        if len(cand) < 0.5 * len(units) or len(cand) > 50 * len(units):
            tried.append(f"{f.name}: {len(cand):,} polygons for {len(units):,} units")
            continue
        if cand.crs is None:
            cand = cand.set_crs(4326)
        elif cand.crs.to_epsg() != 4326:
            cand = cand.to_crs(4326)
        cand = cand[["geometry"]].reset_index(drop=True)
        j = gpd.sjoin(pts, cand, how="inner", predicate="within")
        j = j[~j.index.duplicated(keep="first")]
        if not len(j):
            tried.append(f"{f.name}: no hex falls inside it")
            continue
        tab = j.groupby(["index_right", "unit"]).size()
        top = tab.groupby(level=0).max()
        purity = top.sum() / tab.sum()
        maj = tab.groupby(level=0).idxmax().map(lambda k: k[1])
        cover = len(set(maj) & units) / len(units)
        tried.append(f"{f.name}: {len(cand):,} polygons, purity {purity:.3f}, cover {cover:.3f}")
        if purity >= PURITY and cover >= COVER and (best is None or len(cand) < len(best[1])):
            best = (f, cand, maj, purity, cover)
    for t in tried:
        print(f"    candidate {t}")
    if best is None:
        return None, "no admin file passed: " + "; ".join(tried)

    f, cand, maj, purity, cover = best
    unit_of = pd.Series(maj, index=maj.index).reindex(cand.index)
    empty = unit_of.isna()
    if empty.any():
        # A polygon nobody lives in still belongs to some unit; the nearest hex says which.
        reps = gpd.GeoDataFrame(geometry=cand.geometry[empty].representative_point(), crs=4326)
        near = gpd.sjoin_nearest(reps, pts, how="left")
        near = near[~near.index.duplicated(keep="first")]
        unit_of[empty] = near["unit"].reindex(reps.index).to_numpy()
    shapes = merge_by_unit(unit_of.to_numpy(), cand.geometry.to_numpy())
    why = (f"{f.name}, {len(cand):,} polygons matched to units by the hexes inside them "
           f"(purity {purity:.3f}, cover {cover:.3f}; {int(empty.sum()):,} empty polygons "
           f"given their nearest hex's unit)")
    return shapes, why


_SHAPES = None


def country_outline(cc):
    """The country's Natural Earth outline from country_shapes.py, or None. The UK is drawn
    there as its three censuses, so their parts are put back together."""
    global _SHAPES
    if _SHAPES is None:
        path = HERE / "data" / "processed" / "country_shapes.geojson"
        _SHAPES = defaultdict(list)
        if path.exists():
            for f in json.loads(path.read_text(encoding="utf-8"))["features"]:
                _SHAPES[f["properties"]["cc"]].append(shapely.geometry.shape(f["geometry"]))
    parts = _SHAPES.get(cc)
    return shapely.union_all(parts) if parts else None


def nearest_fill(place, outline):
    """{unit: shape} covering the whole outline: every point goes to the unit of the nearest
    placement polygon, via a Voronoi diagram of their centres. For a grid country with no admin
    file, so a unit draws as a region rather than as the specks of its populated hexes. Exact
    where people live, a guess in the empty land between, which is where it costs nothing."""
    # Natural Earth leaves out small islands people live on (39 of Tonga's villages), so the
    # outline is widened by every placement polygon before anything is clipped to it.
    try:
        populated = shapely.coverage_union_all(place.geometry.to_numpy())
    except Exception:
        populated = shapely.union_all(shapely.make_valid(place.geometry.to_numpy()))
    outline = shapely.union_all([shapely.make_valid(outline), shapely.make_valid(populated)])
    pts = shapely.point_on_surface(place.geometry.to_numpy())
    cells = shapely.get_parts(shapely.voronoi_polygons(shapely.multipoints(pts),
                                                       extend_to=shapely.envelope(outline)))
    tree = shapely.STRtree(cells)
    pi, ci = tree.query(pts, predicate="within")
    unit_of_cell = np.full(len(cells), None, dtype=object)
    unit_of_cell[ci] = place["unit"].to_numpy()[pi]
    ok = unit_of_cell != None   # noqa: E711  (elementwise)
    shapes = merge_by_unit(unit_of_cell[ok], cells[ok])
    return {u: shapely.intersection(g, outline) for u, g in shapes.items()}


def unit_shapes(cfg, cc):
    # A country that names its unit layer outright (place_unit "sjoin") has the outlines already.
    if cfg.get("units") is not None and cfg.get("unit_key"):
        units = gpd.read_file(cfg["units"])
        if units.crs is not None and units.crs.to_epsg() != 4326:
            units = units.to_crs(4326)
        return (merge_by_unit(units[cfg["unit_key"]].astype(str).to_numpy(),
                              units.geometry.to_numpy()),
                f"{Path(str(cfg['units'])).name}, the country's own unit layer")
    from scatter import read_place
    place = read_place(cfg)
    if is_grid(cfg["place"]):
        shapes, why = find_outlines(cc, place, cfg["place"])
        if shapes is not None:
            return with_missing(shapes, place, cc, cfg), why
        outline = country_outline(cc)
        units = place["unit"].unique()
        # A one-unit country is the country.
        if outline is not None and len(units) == 1:
            return ({str(units[0]): shapely.simplify(outline, SIMPLIFY, preserve_topology=True)},
                    f"the country outline (one unit; {why})")
        if outline is not None:
            # with_missing too: units whose hexes sit on another unit's (Tonga's villages)
            # get no Voronoi cell of their own
            return (with_missing(nearest_fill(place, outline), place, cc, cfg),
                    f"{Path(str(cfg['place'])).name} filled out to the country outline by "
                    f"nearest hex ({why})")
        print(f"  !! {why}, and no country outline; merging the grid, populated land only")
        import water
        place = water.clip(place, cc, cfg["place"])
        return (merge_by_unit(place["unit"].to_numpy(), place.geometry.to_numpy(), coverage=True),
                f"{Path(str(cfg['place'])).name} merged by unit (populated land only): {why}")
    import water
    place = water.clip(place, cc, cfg["place"])
    return (merge_by_unit(place["unit"].to_numpy(), place.geometry.to_numpy(), coverage=True),
            f"{Path(str(cfg['place'])).name} merged by unit")


def polygonal(g):
    """Only the polygon parts: a merge can leave stray lines where two pieces touch at a point."""
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    polys = [p for p in shapely.get_parts(g) if p.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.union_all(polys) if polys else None


def build(cc):
    cfg = COUNTRIES[cc]
    t0 = time.time()
    print(f"[{cc}] counts…")
    counts, totals = unit_counts(cfg)
    print(f"[{cc}] {len(counts):,} units with counts; shapes…")
    shapes, why = unit_shapes(cfg, cc)
    print(f"[{cc}] geometry: {why}")
    feats, no_geom = [], []
    for unit, nodes in counts.items():
        g = shapes.get(unit)
        g = polygonal(g) if g is not None else None
        if g is None or g.is_empty:
            no_geom.append(unit)
            continue
        # Snapped (see GRID). set_precision rebuilds topology and threw on an Indian
        # sub-district, which then falls back to rounding alone; a ring made slightly invalid
        # by that still fills, and regions_tiles.py repairs it. The final round only trims
        # float noise off grid values so the JSON stays short.
        # Pointwise first, which only rounds and is cheap, then the topology-rebuilding pass on
        # what is left: run on full detail, that pass took minutes a shape on Canada's coast.
        raw = g
        g = shapely.set_precision(g, GRID, mode="pointwise")
        try:
            g = polygonal(shapely.set_precision(g, GRID))
        except shapely.errors.GEOSException:
            pass
        # A unit narrower than the grid flattens to nothing (12 English and Scottish output
        # areas, ~1,000 people); it keeps its own outline rather than drop out of every pool.
        if g is None or g.is_empty or g.area == 0:
            g = raw
        if g is None or g.is_empty:
            no_geom.append(unit)
            continue
        g = shapely.transform(g, lambda c: np.round(c, DECIMALS))
        props = {"c": cc, "u": unit, "p": round(totals[unit])}
        for node, n in nodes.items():
            props[node] = round(n)
        feats.append({"type": "Feature", "properties": props,
                      "geometry": shapely.geometry.mapping(g)})
    if no_geom:
        lost = sum(totals[u] for u in no_geom)
        print(f"[{cc}] !! {len(no_geom):,} units have no geometry ({lost:,.0f} people): "
              f"{no_geom[:6]}")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"regions_{cc}.geojson"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, separators=(",", ":"))
    tmp.replace(path)
    # The sidecar is what write_index reads, so parallel builds never share a file.
    meta = {"source": why, "regions": len(feats), "units": len(counts),
            "people": round(sum(totals.values())),
            "no_geometry": len(no_geom), "no_geometry_people": round(sum(totals[u] for u in no_geom))}
    (OUT / f"regions_{cc}.meta.json").write_text(json.dumps(meta, ensure_ascii=False),
                                                 encoding="utf-8")
    print(f"[{cc}] {len(feats):,} regions -> {path.name} "
          f"({path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")
    return why


def write_index():
    """index.json, every country with a regions file; and sources.json, gathered from the
    per-country sidecars: where each one's geometry came from and what it could not place."""
    have = sorted(p.name[len("regions_"):-len(".geojson")] for p in OUT.glob("regions_*.geojson"))
    with open(OUT / "index.json", "w", encoding="utf-8") as f:
        json.dump({"countries": have}, f)
    metas = {}
    for cc in have:
        m = OUT / f"regions_{cc}.meta.json"
        if m.exists():
            metas[cc] = json.loads(m.read_text(encoding="utf-8"))
    (OUT / "sources.json").write_text(json.dumps(metas, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    print(f"index.json: {len(have)} countries")


def drawn_countries():
    """Every country the dot map draws: the ones with a dots file, in countries.py's order."""
    done = HERE / "data" / "processed"
    return [cc for cc in COUNTRIES if (done / f"dots_{cc}.geojson").exists()]


def _build_safe(cc):
    """build() in a worker, turning a failure into a line in the log rather than a dead run."""
    import traceback
    try:
        build(cc)
        return cc, None
    except BaseException as e:          # SystemExit too: a country's own checks may call it
        traceback.print_exc()
        return cc, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+")
    ap.add_argument("--all", action="store_true", help="every country with a dots file")
    ap.add_argument("--skip-built", action="store_true",
                    help="leave countries that already have a regions file alone")
    ap.add_argument("--jobs", type=int, default=1, help="countries built at once")
    args = ap.parse_args()
    ccs = drawn_countries() if args.all else (args.countries or [])
    if not ccs:
        sys.exit("give --countries or --all")
    bad = [c for c in ccs if c not in COUNTRIES]
    if bad:
        sys.exit(f"unknown countries: {bad}")
    if args.skip_built:
        ccs = [c for c in ccs if not (OUT / f"regions_{c}.geojson").exists()]
    print(f"{len(ccs)} countries to build, {args.jobs} at a time")
    failed = {}
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs, maxtasksperchild=1) as pool:
            for i, (cc, err) in enumerate(pool.imap_unordered(_build_safe, ccs), 1):
                print(f"=== {i}/{len(ccs)} {cc} {'FAILED ' + err if err else 'ok'}", flush=True)
                if err:
                    failed[cc] = err
    else:
        for cc in ccs:
            _, err = _build_safe(cc)
            if err:
                failed[cc] = err
    write_index()
    (OUT / "failed.json").unlink(missing_ok=True)     # a clean run leaves no stale list
    if failed:
        print(f"{len(failed)} failed:")
        for cc, err in failed.items():
            print(f"  {cc}: {err}")
        (OUT / "failed.json").write_text(json.dumps(failed, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
