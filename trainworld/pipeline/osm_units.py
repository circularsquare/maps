"""T-537, pipeline stage 5: a planned city's outline from OSM admin boundaries, for any country
(nigeria.py's method made generic). The rules are docs/adding-a-city.md 3 "A planned city".

Each of the world plan row's units (geoBoundaries, NUTS-3 or ITL3, as the plan chose) takes the OSM
boundary=administrative relation, at any admin level, that overlaps it best, or the several relations
(level 8 or coarser, mostly inside it) that overlap it better together, when that overlap is
at least MATCH of the two areas' union, both taken on land less inland water; any part of a unit
the relations leave uncovered keeps the unit's own polygon when that is at least REST_MIN of it. Units are
clipped to land by the OSM coastline, holes filled (but for other plan rows' units in them), cut
along every existing pack, and pieces
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
# the real packs, which a new outline is cut along: from this file, not b.ROOT, which a scratch
# tree's redirect has already moved by the time this module loads (world_run's Redirect.KEEP)
PACKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "packs")
# each PACKS outline file's box, keyed by its name, size and time (T-963): cut_packs reads only the
# outlines whose box meets the new one; in data/work, not beside the packs, so a scratch build writes its own
BOXES = os.path.join(b.WORK, "pack_boxes.json")


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
        over = {}
        for i in near:
            inter = over[i] = gl.intersection(loc[i]).area
            iou = inter / (gl.area + loc[i].area - inter)
            if iou > best[1]:
                best = (i, iou)
        # coarsest levels first, a relation only when mostly inside the unit and not mostly under
        # one already taken
        inside = sorted((i for i in near if levels[i] <= MULTI_LEVEL and over[i] >= INSIDE * loc[i].area),
                        key=lambda i: (levels[i], -loc[i].area))
        taken, cover = [], shapely.Polygon()
        for i in inside:
            if loc[i].intersection(cover).area < INSIDE * loc[i].area:
                taken.append(i)
                cover = cover.union(loc[i])
        # each level alone too: a coarse relation mostly inside the unit can shadow the finer ones
        # that fit it (Region Syddanmark, 71% inside Sydjylland, hid its kommuner and took in Fyn: T-788)
        cands = [(taken, cover)] if len(taken) > 1 else []
        for lv in sorted({levels[i] for i in inside}):
            idx = [i for i in inside if levels[i] == lv]
            if len(idx) > 1:
                cands.append((idx, union_all([loc[i] for i in idx])))
        multi, multi_idx = 0.0, []
        for idx, cov in cands:
            inter = gl.intersection(cov).area
            iou = inter / (gl.area + cov.area - inter)
            if iou > multi:
                multi, multi_idx = iou, idx
        if multi > best[1] and multi >= MATCH:
            out.append((multi_idx, multi))
        elif best[1] >= MATCH:
            out.append(([best[0]], best[1]))
        else:
            out.append(([], best[1]))
    return out


def _people_many(parts):
    """GHSL people (2020, 100 m) in each lon/lat polygon: the pixels of the window ghsl_window would
    give its bounds, centres inside it. One read for all (T-888: a read per part, the whole tile
    unzipped each time, took 4-15 minutes over an archipelago's islets)."""
    import math
    import tier0
    if not parts:
        return []
    fwd, inv = tier0._moll()
    P = tier0.PIX_M
    wins = []
    for p in parts:  # ghsl_window's own window arithmetic, per part
        w, s, e, n = p.bounds
        xs, ys = fwd.transform([w, w, e, e, (w + e) / 2, (w + e) / 2], [s, n, s, n, s, n])
        X0 = tier0.MOLL_X0 + math.floor((min(xs) - tier0.MOLL_X0) / P) * P
        Y0 = tier0.MOLL_Y0 - math.floor((tier0.MOLL_Y0 - max(ys)) / P) * P
        wins.append((X0, Y0, math.ceil((max(xs) - X0) / P), math.ceil((Y0 - min(ys)) / P)))
    b4 = np.array([p.bounds for p in parts])
    arr, BX, BY = tier0.ghsl_window("pop", (b4[:, 0].min(), b4[:, 1].min(), b4[:, 2].max(), b4[:, 3].max()), pad_m=2 * P)
    iy, ix = np.nonzero(arr > 0)  # row-major, so each part's pixels sum in ghsl_window's order
    val = arr[iy, ix]
    lon, lat = inv.transform(BX + (ix + 0.5) * P, BY - (iy + 0.5) * P)
    lon, lat = np.asarray(lon), np.asarray(lat)
    out = []
    for p, (X0, Y0, W, H) in zip(parts, wins):
        r0, c0 = int(round((BY - Y0) / P)), int(round((X0 - BX) / P))
        assert r0 >= 0 and c0 >= 0 and r0 + H <= arr.shape[0] and c0 + W <= arr.shape[1], "window outside the read"
        a, z = np.searchsorted(iy, r0), np.searchsorted(iy, r0 + H)
        sel = np.arange(a, z)[(ix[a:z] >= c0) & (ix[a:z] < c0 + W)]
        if not len(sel):
            out.append(0.0)
            continue
        shapely.prepare(p)
        out.append(float(val[sel][shapely.contains_xy(p, lon[sel], lat[sel])].sum()))
    return out


def without_islets(sea, lat0):
    """The sea with its holes under MIN_PART_KM2 filled: every outline piece such an islet could
    give is that small and dropped anyway, and an archipelago's tens of thousands of them made each
    cut of the sea take minutes (T-888)."""
    polys = []
    for p in shapely.get_parts(sea):
        if p.geom_type != "Polygon":
            continue
        holes = [r for r in p.interiors if km2(shapely.Polygon(r), lat0) >= MIN_PART_KM2]
        polys.append(shapely.Polygon(p.exterior, holes) if len(holes) < len(p.interiors) else p)
    out = shapely.MultiPolygon(polys) if len(polys) != 1 else polys[0]
    return out if out.is_valid else union_all(polys)  # sea inside a filled islet's lagoon


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
    # only relations meeting a unit can match one; the rest of the box (Manner-Suomi beside Åland)
    # would only be cut from the sea for nothing (T-888)
    admin["geometry"] = [make_valid(g) for g in admin.geometry]
    meets = STRtree([u[2] for u in units]).query(admin.geometry.to_numpy(), predicate="intersects")[0]
    admin = admin.iloc[sorted(set(meets.tolist()))].reset_index(drop=True)
    # units are clipped to land by the coastline; they are compared on land less inland water too,
    # since OSM relations often run out over lagoons and lakes where geoBoundaries' stop at the shore
    # (Lagos Island's relation is 161 km2, its land about 9)
    box = (w - SEA_PAD, s - SEA_PAD, e + SEA_PAD, n + SEA_PAD)
    # the pieces first, which makes their union quick; then islets on a piece's edge, now inside
    sea = without_islets(union_all([without_islets(p, lat0) for p in osm_water.sea(box)]), lat0)
    lakes = [g for g in osm_water.inland(row["extracts"], box) if km2(g, lat0) >= WET_MIN_KM2]
    wet = union_all(lakes) if lakes else shapely.Polygon()  # the sea is cut from every shape below (`landed`), so only inland water is left to dry
    shapely.prepare(sea)

    def once(f):  # relations often repeat a shape at two levels (Åland 3 and 4): cut it once
        memo = {}

        def cached(g):
            k = g.wkb
            if k not in memo:
                memo[k] = f(g)
            return memo[k]
        return cached

    def clip(water, g):
        # polygons only: clip_by_rect leaves lines along the box's edge, and a difference with a
        # collection holding lines ran over an hour on Highland (T-903; 3.5 s without them)
        c = shapely.clip_by_rect(water, *g.bounds)
        c = shapely.MultiPolygon([q for x in shapely.get_parts(c) for q in shapely.get_parts(x) if q.geom_type == "Polygon"])
        return c if c.is_valid else make_valid(c)

    @once
    def landed(g):  # the sea within the shape's box only: a difference with a whole archipelago's coast never ends (Stockholm)
        if g.is_empty:
            return g
        return g.difference(clip(sea, g)).buffer(0)
    admin["geometry"] = [landed(g) for g in admin.geometry]
    units = [(uid, name, landed(g), kind) for uid, name, g, kind in units]

    @once
    def dried(g):  # the water within the shape's box only: a difference with all of it is slow
        if g.is_empty:
            return g
        return g.difference(clip(wet, g))
    dry = admin.copy()
    dry["geometry"] = [dried(g) for g in admin.geometry]
    picks = foreign(row, sources, units, admin, match([(u[0], u[1], dried(u[2]), u[3]) for u in units], dry, lat0))
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
    shells = [shapely.Polygon(p.exterior) for p in shapely.get_parts(whole) if p.geom_type == "Polygon"]
    filled = shapely.MultiPolygon(shells)  # holes are water or enclaves; cut below if packs
    if not filled.is_valid:  # a part inside another's hole: the shells overlap (Daegu, T-1262)
        filled = union_all(shells)
    enc = enclaves(row, sources, make_valid(filled.difference(whole)))
    whole = make_valid(filled.difference(enc)) if not enc.is_empty else filled
    whole = cut_packs(whole, {cid, row.get("test_of")})
    parts = sorted([p for p in shapely.get_parts(whole) if p.geom_type == "Polygon"], key=lambda p: -p.area)
    big = [km2(p, lat0) >= MIN_PART_KM2 for p in parts[1:]]
    people = iter(_people_many([p for p, k in zip(parts[1:], big) if k]))
    keep, dropped = parts[:1], []
    for p, k in zip(parts[1:], big):
        if k and next(people) >= 1:
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


def foreign(row, sources, units, admin, picks):
    """Drop from each several-relation pick the relations that hold half or more of another plan
    row's unit: geoBoundaries draws some city districts far smaller than OSM, so OSM's district lies
    mostly inside the neighbouring county and joined it, taking the city's centre (T-897: Yingkou's
    Zhanqian into Dashiqiao's zone, Qitaihe's Xinxing into Bin's). A unit left uncovered keeps its own
    polygon there (REST_MIN), as any unit does."""
    multi = sorted({i for idx, _ in picks if len(idx) > 1 for i in idx})
    if not multi:
        return picks
    import plan_config
    import world_plan as wp
    plan = plan_config.load()[0]
    mine = {u for ids in row["units"].values() for u in ids}
    rel = {i: make_valid(wp.to_moll(admin.geometry.iloc[i])) for i in multi}
    box = union_all(list(rel.values())).envelope
    held = {}
    for iso in row["units"]:
        if sources[iso]["kind"] != "geoBoundaries":
            continue
        others = {u for r in plan if r.get("id") != row["id"] for u in (r.get("units") or {}).get(iso, [])} - mine
        meta = wp.europe_isos()[1]
        for adm in sources[iso]["urls"]:
            for uid, name, g in wp.gb_units(meta, iso, adm):
                if uid in others and g.intersects(box):
                    for i, rg in rel.items():
                        if g.area > 0 and make_valid(g).intersection(rg).area >= 0.5 * g.area:
                            held.setdefault(i, []).append(name)
    if not held:
        return picks
    out = []
    for (uid, gname, _, _), (idx, iou) in zip(units, picks):
        keep = [i for i in idx if i not in held] if len(idx) > 1 else idx
        if keep and len(keep) < len(idx):
            for i in idx:
                if i in held:
                    print(f"  {row['id']}: {gname} leaves out OSM r{admin.osm_id.iloc[i]} {admin['name'].iloc[i]}, "
                          f"which holds most of another row's {', '.join(held[i])}", flush=True)
            out.append((keep, iou))
        else:
            out.append((idx, iou))
    return out


def enclaves(row, sources, holes):
    """Of the outline's holes, what other plan rows' units cover (a city's districts ringed by the
    counties of a zone built before them took them in: Hengyang inside Hengnan, T-897). Lon/lat."""
    if holes.is_empty:
        return shapely.Polygon()
    import plan_config
    import world_plan as wp
    plan = plan_config.load()[0]
    mine = {u for ids in row["units"].values() for u in ids}
    hm = make_valid(wp.to_moll(holes))
    out = []
    for iso in row["units"]:
        if sources[iso]["kind"] != "geoBoundaries":
            continue
        others = {u for r in plan if r.get("id") != row["id"] for u in (r.get("units") or {}).get(iso, [])} - mine
        meta = wp.europe_isos()[1]
        for adm in sources[iso]["urls"]:
            for uid, _, g in wp.gb_units(meta, iso, adm):
                if uid in others and g.intersects(hm):
                    out.append(wp.polygonal(wp.to_ll(make_valid(g).intersection(hm))))
    got = union_all(out) if out else shapely.Polygon()
    if not got.is_empty:
        print(f"  {row['id']}: {km2(got, (row['bbox'][1] + row['bbox'][3]) / 2):,.0f} km2 of other rows' units "
              "inside its outline left out", flush=True)
    return got


def pack_boxes(paths):
    """{path: the outline file's bounding box}, from BOXES where its size and time match the file's,
    else read (and BOXES rewritten)."""
    import shared_lock
    try:
        old = shared_lock.read_json(BOXES)
    except (OSError, ValueError):
        old = {}
    out, fresh = {}, 0
    for path in paths:
        name, st = os.path.basename(path), os.stat(path)
        e = old.get(name)
        if e and e[0] == st.st_size and e[1] == st.st_mtime_ns:
            out[path] = e[2]
            continue
        with open(path, encoding="utf-8") as f:
            fc = json.load(f)
        e = [st.st_size, st.st_mtime_ns, shapely.total_bounds([shape(ft["geometry"]) for ft in fc["features"]]).tolist()]
        old[name], out[path], fresh = e, e[2], fresh + 1
    gone = set(old) - {os.path.basename(p) for p in paths}
    if fresh or gone:
        for k in gone:
            del old[k]
        try:
            os.makedirs(os.path.dirname(BOXES), exist_ok=True)
            shared_lock.write_atomic(BOXES, json.dumps(old))
        except OSError:
            pass  # another build is writing it; the next build writes it again
    return out


def cut_packs(g, skip):
    """g less every existing pack's outline it meets (packs never overlap). Only outlines whose box
    meets g's are read (T-963)."""
    import glob
    gb = g.bounds
    paths = glob.glob(os.path.join(PACKS, "*.boundary.geojson"))
    boxes = pack_boxes(paths)
    cut = []
    for path in paths:
        if os.path.basename(path).split(".")[0] in skip:
            continue
        w, s, e, n = boxes[path]
        if w > gb[2] or e < gb[0] or s > gb[3] or n < gb[1]:
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
