"""T-537, pipeline stage 5: a planned city's outline from OSM admin boundaries, for any country
(nigeria.py's method made generic). The rules are docs/adding-a-city.md 3 "A planned city".

Each of the world plan row's units (geoBoundaries, NUTS-3 or ITL3, as the plan chose) takes the OSM
boundary=administrative relation, at any admin level, that overlaps it best, or the several relations
(level 8 or coarser, mostly inside it) that overlap it better together, when that overlap is
at least MATCH of the two areas' union, both taken on land less inland water; any part of a unit
the relations leave uncovered keeps the unit's own polygon when that is at least REST_MIN of it. Units are
clipped to land by the OSM coastline, holes filled, cut along every existing pack, and pieces
under MIN_PART_KM2 or with nobody in them (GHSL 100 m) dropped. OSM's names are kept per unit.

    python pipeline/osm_units.py <id> [plan.json]     write the units and outline, print the matches

The extract's admin relations are read once into data/raw/osm/<extract>.admin.fgb (beside
osm_features' caches); the units and outline go to data/work/<id>.units.geojson and .outline.geojson.
"""
import json
import os
import sys
import unicodedata

import numpy as np
import shapely
from shapely import STRtree, make_valid, union_all
from shapely.geometry import mapping, shape

import build_city as b

MATCH = 0.5            # overlap / union an OSM relation needs to stand for a plan unit
INSIDE = 0.5           # share of an OSM relation that must lie inside a plan unit to be one of its several
REST_MIN = 0.1         # share of a plan unit OSM leaves uncovered at which the unit's own polygon fills it
MULTI_LEVEL = 8        # finest admin level that may combine into one plan unit
SEA_PAD = 0.25         # degrees around the plan's box over which the OSM sea is read
# inland water smaller than this is left out of the match's water: it cannot move an overlap much,
# and Rotterdam's 66,000 ditches and canals (2.3M vertices) stalled the match for over 20 minutes
WET_MIN_KM2 = 0.1
MIN_PART_KM2 = 0.05  # outline pieces smaller than this are slivers left by the coastline clip
OSM = os.path.join(b.RAW, "osm")
PACKS = os.path.join(b.ROOT, "data", "packs")  # the real packs, which a new outline is cut along


def latin(s):
    return bool(s) and all(not c.isalpha() or "LATIN" in unicodedata.name(c, "") for c in s)


def admin_path(extract):
    return os.path.join(OSM, extract.replace("/", "_") + ".admin.fgb")


def admin_areas(extract):
    """Every boundary=administrative multipolygon of the extract (admin_level 3-11): osm_id, name,
    name_en, admin_level, geometry. Cached."""
    import pyogrio
    import routes
    out = admin_path(extract)
    if not os.path.exists(out):
        pbf = routes.download(extract)
        df = pyogrio.read_dataframe(pbf, layer="multipolygons", columns=["osm_id", "name", "admin_level", "other_tags"],
                                    where="boundary = 'administrative' AND admin_level IN ('3','4','5','6','7','8','9','10','11')")
        df["name_en"] = [routes.tag(t, "name:en") for t in df.other_tags]
        df = df.drop(columns=["other_tags"])
        df = df[df.geometry.notna()]
        pyogrio.write_dataframe(df, out + ".part.fgb", driver="FlatGeobuf")
        os.replace(out + ".part.fgb", out)
        print(f"  {extract}: {len(df):,} admin relations cached", flush=True)
    return pyogrio.read_dataframe(out)


def plan_units(row, sources):
    """[(plan unit id, name, lon/lat geometry, source kind)] of the row's units, from the files the
    plan read."""
    import world_plan as wp
    out = []
    for iso, ids in row["units"].items():
        src, want = sources[iso], set(ids)
        if src["kind"] == "geoBoundaries":
            meta = wp.europe_isos()[1]
            have = [u for adm in src["urls"] for u in wp.gb_units(meta, iso, adm)]
        elif src["kind"] == "NUTS3":
            have = wp.load_nuts({iso})[iso]
        elif src["kind"] == "ITL3":
            gj = json.loads(wp.ITL3.read_text(encoding="utf-8"))
            have = [(f["properties"]["ITL325CD"], f["properties"]["ITL325NM"], wp.to_moll(shape(f["geometry"])))
                    for f in gj["features"]]
        else:  # the territory whole (Hong Kong, Macau)
            have = [(f"{iso}-ADM0", iso, g) for i, g in wp.countries() if i == iso]
        got = [(uid, name, wp.polygonal(wp.to_ll(g)), src["kind"]) for uid, name, g in have if uid in want]
        missing = want - {u[0] for u in got}
        if missing:
            raise SystemExit(f"{row['id']}: plan units {sorted(missing)[:5]} not in {src['kind']} for {iso}")
        out += got
    return out


def _local(g, lat0):
    k = np.cos(np.radians(lat0))
    return shapely.transform(g, lambda a: a * [k, 1.0])


def match(units, admin, lat0):
    """For each plan unit, ([OSM row indexes], overlap / union), the list empty when nothing matches.
    A unit is one OSM relation, or several when together they overlap it better (a NUTS-3 unit is
    often two or three districts: T-610)."""
    loc = [_local(make_valid(g), lat0) for g in admin.geometry]
    tree = STRtree(loc)
    levels = [int(x) for x in admin.admin_level]
    out = []
    for _, _, g, _ in units:
        gl = _local(g, lat0)
        best = (None, 0.0)
        near = [int(i) for i in tree.query(gl, predicate="intersects")]
        for i in near:
            inter = gl.intersection(loc[i]).area
            iou = inter / (gl.area + loc[i].area - inter)
            if iou > best[1]:
                best = (i, iou)
        # coarsest levels first, a relation only when mostly inside the unit and not mostly under
        # one already taken
        inside = sorted((i for i in near if levels[i] <= MULTI_LEVEL and gl.intersection(loc[i]).area >= INSIDE * loc[i].area),
                        key=lambda i: (levels[i], -loc[i].area))
        taken, cover = [], shapely.Polygon()
        for i in inside:
            if loc[i].intersection(cover).area < INSIDE * loc[i].area:
                taken.append(i)
                cover = cover.union(loc[i])
        multi = 0.0
        if len(taken) > 1:
            inter = gl.intersection(cover).area
            multi = inter / (gl.area + cover.area - inter)
        if multi > best[1] and multi >= MATCH:
            out.append((taken, multi))
        elif best[1] >= MATCH:
            out.append(([best[0]], best[1]))
        else:
            out.append(([], best[1]))
    return out


def _people(part):
    """GHSL people (2020, 100 m, pixel centres) in a lon/lat polygon."""
    import tier0
    arr, X0, Y0 = tier0.ghsl_window("pop", part.bounds)
    iy, ix = np.nonzero(arr > 0)
    if not len(iy):
        return 0.0
    _, inv = tier0._moll()
    lon, lat = inv.transform(X0 + (ix + 0.5) * tier0.PIX_M, Y0 - (iy + 0.5) * tier0.PIX_M)
    shapely.prepare(part)
    return float(arr[iy, ix][shapely.contains_xy(part, np.asarray(lon), np.asarray(lat))].sum())


def km2(g, lat0):
    return _local(g, lat0).area * 111.32 * 110.57


def build(cid, row, sources):
    """Write data/work/<cid>.units.geojson and .outline.geojson; the outline's path."""
    import pandas as pd
    import osm_water
    units = plan_units(row, sources)
    w, s, e, n = row["bbox"]
    lat0 = (s + n) / 2
    pad = 0.05
    frames = []
    for ex in row["extracts"]:
        a = admin_areas(ex)
        frames.append(a.cx[w - pad:e + pad, s - pad:n + pad])
    admin = pd.concat(frames, ignore_index=True).drop_duplicates("osm_id").reset_index(drop=True)
    # units are clipped to land by the coastline; they are compared on land less inland water too,
    # since OSM relations often run out over lagoons and lakes where geoBoundaries' stop at the shore
    # (Lagos Island's relation is 161 km2, its land about 9)
    box = (w - SEA_PAD, s - SEA_PAD, e + SEA_PAD, n + SEA_PAD)
    sea = union_all(osm_water.sea(box))
    lakes = [g for g in osm_water.inland(row["extracts"], box) if km2(g, lat0) >= WET_MIN_KM2]
    wet = union_all(lakes) if lakes else shapely.Polygon()  # the sea is cut from every shape below (`landed`), so only inland water is left to dry
    shapely.prepare(sea)

    def landed(g):  # the sea within the shape's box only: a difference with a whole archipelago's coast never ends (Stockholm)
        if g.is_empty:
            return g
        c = shapely.clip_by_rect(sea, *g.bounds)
        return g.difference(c if c.is_valid else make_valid(c)).buffer(0)
    admin["geometry"] = [landed(make_valid(g)) for g in admin.geometry]
    units = [(uid, name, landed(g), kind) for uid, name, g, kind in units]

    def dried(g):  # the water within the shape's box only: a difference with all of it is slow
        if g.is_empty:
            return g
        c = shapely.clip_by_rect(wet, *g.bounds)
        return g.difference(c if c.is_valid else make_valid(c))
    dry = admin.copy()
    dry["geometry"] = [dried(g) for g in admin.geometry]
    picks = match([(u[0], u[1], dried(u[2]), u[3]) for u in units], dry, lat0)
    feats, geoms, chosen = [], [], {}

    def take(i, uid, gname, iou):
        r = admin.iloc[i]
        local = r["name"] if isinstance(r["name"], str) else ""
        shown = local if latin(local) else (r.name_en if latin(r.name_en) else gname)
        chosen[i] = len(geoms)
        geoms.append(r.geometry)
        feats.append({"name": shown, "local": local, "plan_unit": uid, "source": f"OSM r{r.osm_id}",
                      "admin_level": r.admin_level, "match": round(iou, 3)})
    for (uid, gname, _, _), (idx, iou) in zip(units, picks):
        for i in idx:
            if i not in chosen:
                take(i, uid, gname, iou)
    # whatever of a unit no OSM relation covers (a relation cut off at an extract's edge, or none
    # matching) keeps the plan's own polygon there, when it is a real part of the unit (T-610: a
    # unit left out silently halved Cardiff)
    have = union_all(geoms) if geoms else shapely.Polygon()
    for (uid, gname, g, kind), (idx, iou) in zip(units, picks):
        rest = make_valid(g.difference(have))
        share = _local(rest, lat0).area / _local(g, lat0).area
        if share >= REST_MIN:
            geoms.append(rest)
            feats.append({"name": gname, "local": "", "plan_unit": uid, "source": kind, "match": round(iou, 3)})
            print(f"  {cid}: {uid} {gname}: {share:.0%} of its land from the plan's polygon "
                  f"({len(idx)} OSM relations, overlap {iou:.2f})", flush=True)
    out_units = [{"type": "Feature", "properties": props, "geometry": mapping(g)} for props, g in zip(feats, geoms)]
    whole = union_all([shape(f["geometry"]) for f in out_units]).buffer(0)
    whole = shapely.MultiPolygon([shapely.Polygon(p.exterior) for p in shapely.get_parts(whole)
                                  if p.geom_type == "Polygon"])  # holes are water or enclaves; cut below if packs
    whole = cut_packs(whole, {cid, row.get("test_of")})
    parts = sorted([p for p in shapely.get_parts(whole) if p.geom_type == "Polygon"], key=lambda p: -p.area)
    keep, dropped = parts[:1], []
    for p in parts[1:]:
        if km2(p, lat0) >= MIN_PART_KM2 and _people(p) >= 1:
            keep.append(p)
        else:
            dropped.append(p)
    whole = shapely.MultiPolygon(keep)
    print(f"{cid}: {len(units)} plan units, {sum(1 for i, _ in picks if i)} matched to OSM; {len(chosen)} OSM units, "
          f"{len(geoms) - len(chosen)} plan polygons; outline "
          f"{len(keep)} parts ({len(dropped)} dropped, {sum(km2(p, lat0) for p in dropped):.2f} km2), "
          f"{km2(whole, lat0):,.0f} km2", flush=True)
    stem = os.path.join(b.WORK, cid)
    os.makedirs(b.WORK, exist_ok=True)
    for path, doc in ((stem + ".units.geojson", {"type": "FeatureCollection", "features": out_units}),
                      (stem + ".outline.geojson", {"type": "FeatureCollection", "features": [
                          {"type": "Feature", "properties": {"city": cid}, "geometry": mapping(whole)}]})):
        with open(path + ".part", "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        os.replace(path + ".part", path)
    return stem + ".outline.geojson"


def cut_packs(g, skip):
    """g less every existing pack's outline it meets (packs never overlap)."""
    import glob
    cut = []
    for path in glob.glob(os.path.join(PACKS, "*.boundary.geojson")):
        if os.path.basename(path).split(".")[0] in skip:
            continue
        with open(path, encoding="utf-8") as f:
            fc = json.load(f)
        q = union_all([make_valid(shape(ft["geometry"])) for ft in fc["features"]])
        if q.intersects(g):
            cut.append(q)
    return make_valid(g.difference(union_all(cut))) if cut else g


def boundary_file(cid, row, sources):
    """tier0's `boundary_file`: the outline's path, built if missing."""
    path = os.path.join(b.WORK, f"{cid}.outline.geojson")
    return path if os.path.exists(path) else build(cid, row, sources)


def latin_names(cid):
    """True when most of the units' OSM names are not in Latin script (places.py then shows
    name:en and the like)."""
    with open(os.path.join(b.WORK, f"{cid}.units.geojson"), encoding="utf-8") as f:
        names = [ft["properties"]["local"] for ft in json.load(f)["features"] if ft["properties"]["local"]]
    return bool(names) and sum(not latin(x) for x in names) > len(names) / 2


if __name__ == "__main__":
    import plan_config
    cid = sys.argv[1]
    rows, sources = plan_config.load(sys.argv[2] if len(sys.argv) > 2 else plan_config.PLAN)
    row = next(r for r in rows if r["id"] == cid)
    path = build(cid, row, sources)
    with open(os.path.join(b.WORK, f"{cid}.units.geojson"), encoding="utf-8") as f:
        for ft in json.load(f)["features"]:
            p = ft["properties"]
            print(f"  {p['name']}: {p['source']}, level {p.get('admin_level', '-')}, match {p['match']}")
