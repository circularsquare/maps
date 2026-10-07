"""Assemble Russia's level-2 boundaries from OSM and pair them with Rosstat.

Reads the raw Overpass batches fetch_osm.py saved (relations + member ways
with geometry), builds one polygon per relation, puts each in a federal
subject, and matches it to a 1 January 2025 Rosstat municipality
(data/russia/units.csv, from fetch.py):

  1. OKTMO on the relation (oktmo:user / ref:oktmo) or on its Wikidata item
     (P764), current or 2024 code (links_2024.csv carries old to new);
  2. otherwise the name, within the subject, with type words dropped
     (crosswalk.key), then with adjectival endings cut (crosswalk.stem).
A code match that disagrees with a unique name match is reported.

Level 2 inside Moscow and St Petersburg is their admin_level=5 relations
(Moscow's 12 administrative okrugs, St Petersburg's 18 districts), matched
by name to Rosstat's okrug and district rows.

Subjects come from geoBoundaries RUS ADM1 (religiondots/data/geo/ru, read
only): a relation belongs to the subject holding its representative point.
That file has 83 subjects and no Crimea or Sevastopol, so relations there
drop out unless INCLUDE_CRIMEA (fetch.py), when OSM's Russian-tagged Crimea
and Sevastopol relations are used for them.

Level 1 is the level-2 polygons dissolved per subject, so the levels nest.

Writes helper1m/data/russia/boundaries/:
  adm1.gpkg, adm2.gpkg   code, name (English), name_cn (Russian), parent, group
  unit_map.csv           Rosstat OKTMO -> level-2 code (a polygon holding two
                         Rosstat units gets the code "A+B")
  match_report.txt       every non-trivial match decision
then runs fetch.write_population().
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import csv  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import LineString, MultiPolygon, Point, Polygon  # noqa: E402
from shapely.ops import linemerge, polygonize, unary_union  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crosswalk  # noqa: E402
import fetch  # noqa: E402

HELPER = Path(__file__).resolve().parents[2]
REPO = HELPER.parent
DATA = HELPER / "data" / "russia"
OSM = DATA / "raw" / "osm"
OUT = DATA / "boundaries"
GB_ADM1 = REPO / "religiondots" / "data" / "geo" / "ru" / "geoBoundaries-RUS-ADM1.geojson"

CITY_ISO = ("RU-MOW", "RU-SPE")
CRIMEA_REL = {3795586: "UA-43", 3788485: "UA-40"}

# OSM relation id -> Rosstat OKTMO(s), for pairs neither rule can make.
# Filled from match_report.txt.
MANUAL = {
    # OSM's Birilyussky okrug (11,744 km2) is Birilyussky district plus
    # Tyukhtetsky okrug (about 6,800 + 4,900 km2), merged in 2025 after the
    # Rosstat table; Wikidata has no coordinate for Tyukhtetsky.
    1162482: ["04606000", "04555000"],
}

# Relations too broken to assemble: their unit is drawn from the gap its
# neighbours leave instead (see the gap rule in main()).
DROP_RELATIONS = {
    # Nizhny Tagil: about a third of its border is missing from the relation
    # in 2026; joining the loose ends leaves a 21 km2 sliver.
    1317756,
}

# Town-centre coordinates for units Wikidata gives none for and no OSM
# relation covers (lon, lat; the unit's administrative centre).
MANUAL_POINTS = {
    "22537000": (44.20, 56.15),   # Kstovsky okrug: Kstovo
    "46711000": (37.85, 55.63),   # Dzerzhinsky urban okrug, Moscow Oblast
}

TRANS = dict(zip("абвгдезийклмнопрстуфхцы", "abvgdeziyklmnoprstufkhtsy"))
TRANS.update({"ё": "yo", "ж": "zh", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "",
              "ь": "", "э": "e", "ю": "yu", "я": "ya"})


def translit(s):
    out = []
    for ch in s:
        low = ch.lower()
        t = TRANS.get(low, ch)
        out.append(t.capitalize() if ch != low and t else t)
    return "".join(out).replace("iy ", "y ").replace("yy", "y")


TYPE_EN = [
    (r"муниципальн\w* район", "District"), (r"муниципальн\w* округ", "Municipal Okrug"),
    (r"городск\w* округ", "Urban Okrug"), (r"административн\w* округ", "Administrative Okrug"),
    (r"\bрайон\b", "District"), (r"\bзато\b", "ZATO"),
]


def english(osm_tags, ru_name):
    if osm_tags.get("name:en"):
        return osm_tags["name:en"]
    kind = next((en for pat, en in TYPE_EN if re.search(pat, ru_name, re.I)), "")
    base = crosswalk.key(ru_name).title()
    return (translit(base) + (" " + kind if kind else "")).strip()


def load_osm():
    rels, ways = {}, {}
    files = sorted(OSM.glob("batch_*.json")) + [OSM / "extra.json", OSM / "extra_ids.json"]
    for p in files:
        if not p.exists():
            print(f"  missing {p.name}")
            continue
        for e in json.loads(p.read_text(encoding="utf-8"))["elements"]:
            if e["type"] == "relation":
                rels[e["id"]] = e
            elif e["type"] == "way" and "geometry" in e:
                ways[e["id"]] = [(g["lon"], g["lat"]) for g in e["geometry"] if g]
    return rels, ways


GAP_LOG = []


def loose_ends(lines):
    """Line ends that meet no other line: where three ways meet at a node
    the merged pieces end there too, but that end is shared, not loose."""
    u = unary_union(lines)
    merged = linemerge(u) if u.geom_type == "MultiLineString" else u
    count = defaultdict(int)
    for p in getattr(merged, "geoms", [merged]):
        if not p.is_ring:
            for c in (p.coords[0], p.coords[-1]):
                count[(round(c[0], 7), round(c[1], 7))] += 1
    return [c for c, n in count.items() if n == 1]


def close_gaps(lines, rid, max_gap=0.15):
    """A relation whose outer ways do not close (three in Sverdlovsk, 2026:
    Pervouralsk with a 5 m gap, Verkhny and Nizhny Tagil with stretches of
    shared border missing) is closed by joining each loose end to the
    nearest other loose end with a straight line. Logged, with the longest
    join, so a bad one is visible."""
    ends = [Point(c) for c in loose_ends(lines)]
    joins, used = [], set()
    for i, a in enumerate(ends):
        if i in used:
            continue
        best, bd = None, max_gap
        for k, b in enumerate(ends):
            if k == i or k in used:
                continue
            d = a.distance(b)
            if d < bd:
                best, bd = k, d
        if best is not None:
            used |= {i, best}
            joins.append((LineString([a, ends[best]]), bd))
    GAP_LOG.append(f"relation {rid}: closed {len(joins)} gaps, longest "
                   f"{max((d for _, d in joins), default=0):.4f} deg")
    return [j for j, _ in joins]


def assemble(rel, ways):
    outer, inner = [], []
    for m in rel.get("members", []):
        if m["type"] != "way" or m["ref"] not in ways or len(ways[m["ref"]]) < 2:
            continue
        (inner if m.get("role") == "inner" else outer).append(LineString(ways[m["ref"]]))
    if not outer:
        return None
    network = unary_union(outer + inner)
    if loose_ends(outer + inner):
        joins = close_gaps(outer + inner, rel["id"])
        network = unary_union(outer + inner + joins)
    faces = list(polygonize(network))
    if not faces:
        return None
    # polygonize returns every face of the line network, holes included. A
    # district ringing its town (an urban okrug of its own) often lists the
    # town's border as plain "outer" ways, so roles cannot be trusted: keep a
    # face when a ray from inside it crosses the network an odd number of
    # times (even-odd rule).
    keep = []
    for f in faces:
        p = f.representative_point()
        ray = LineString([(p.x, p.y), (f.bounds[2] + 400, p.y)])
        hit = ray.intersection(network)
        n = len(getattr(hit, "geoms", [hit])) if not hit.is_empty else 0
        if n % 2 == 1:
            keep.append(f)
    if not keep:
        return None
    return shapely.make_valid(unary_union(keep))


OCEAN = REPO / "data" / "ne_10m_lakes" / "ne_10m_ocean.shp"


def clip_sea(geoms):
    """OSM draws coastal municipalities out to the 12-mile territorial sea.
    Cut the sea away with Natural Earth's 10 m ocean (read only), so a coast
    looks like a coast; inland water (Baikal, Ladoga, the Caspian, which NE
    files as a lake) stays inside its units."""
    ocean = gpd.read_file(OCEAN).geometry
    parts = []
    for box in ((18, 40, 180, 83), (-180, 60, -165, 73)):
        clipped = ocean.clip_by_rect(*box)
        for g in clipped:
            if not g.is_empty:
                parts += list(getattr(g, "geoms", [g]))
    parts = [p for p in parts if p.area > 0]
    tree = shapely.STRtree(parts)
    out = []
    for g in geoms:
        hits = tree.query(g, predicate="intersects")
        if len(hits):
            cut = shapely.make_valid(g.difference(unary_union([parts[k] for k in hits])))
            if cut.geom_type == "GeometryCollection":
                cut = unary_union([x for x in cut.geoms if x.geom_type in ("Polygon", "MultiPolygon")])
            # An island unit that NE's coarse coast swallows whole keeps its
            # OSM outline rather than vanishing.
            g = cut if cut.area > 0.02 * g.area else g
        out.append(g)
    return out


def unit_points(codes):
    """{OKTMO: Point} from Wikidata (P764 -> P625), cached in
    raw/osm/wikidata_points.json."""
    import urllib.parse
    import urllib.request
    cache_path = OSM / "wikidata_points.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [c for c in codes if c not in cache]
    if todo:
        query = ("SELECT ?code ?coord WHERE { VALUES ?code { "
                 + " ".join(f'"{c}"' for c in todo)
                 + " } ?item wdt:P764 ?code ; wdt:P625 ?coord . }")
        req = urllib.request.Request(
            "https://query.wikidata.org/sparql",
            data=urllib.parse.urlencode({"query": query}).encode(),
            headers={"User-Agent": "helper1m-research/1.0",
                     "Accept": "application/sparql-results+json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            res = json.load(r)
        for c in todo:
            cache[c] = None
        for b in res["results"]["bindings"]:
            m = re.match(r"Point\(([-\d.]+) ([-\d.]+)\)", b["coord"]["value"])
            if m and cache.get(b["code"]["value"]) is None:
                cache[b["code"]["value"]] = [float(m.group(1)), float(m.group(2))]
        cache_path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    return {c: Point(cache[c]) for c in codes if cache.get(c)}


def assert_no_wrap(geom, label):
    """Every polygon part must span under 180 degrees of longitude; a part
    that wraps the antimeridian is drawn across the whole world."""
    parts = geom.geoms if hasattr(geom, "geoms") else [geom]
    for p in parts:
        if p.is_empty:
            continue
        x0, _, x1, _ = p.bounds
        if x1 - x0 >= 180:
            raise SystemExit(f"{label}: a part spans {x0:.1f}..{x1:.1f} degrees of longitude")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    tags6 = {e["id"]: e["tags"] for e in json.loads((OSM / "tags.json").read_text(encoding="utf-8"))["elements"]
             if e["tags"].get("admin_level") == "6"}
    wd = json.loads((OSM / "wikidata_oktmo.json").read_text(encoding="utf-8"))
    rels, ways = load_osm()
    print(f"{len(rels)} relations and {len(ways)} ways loaded")

    with (DATA / "units.csv").open(encoding="utf-8") as f:
        units = {r["oktmo"]: r for r in csv.DictReader(f)}
    with (DATA / "links_2024.csv").open(encoding="utf-8") as f:
        old_to_new = {r["old"]: r["new"] for r in csv.DictReader(f)}
    with (DATA / "subjects.csv").open(encoding="utf-8") as f:
        subj_ru = {r["iso"]: r["name"] for r in csv.DictReader(f)}

    # Subject polygons for placing relations.
    gb = gpd.read_file(GB_ADM1)[["shapeISO", "shapeName", "geometry"]]
    subj_en = dict(zip(gb.shapeISO, gb.shapeName))
    if fetch.INCLUDE_CRIMEA:
        for rid, iso in CRIMEA_REL.items():
            g = assemble(rels[rid], ways) if rid in rels else None
            if g is None:
                raise SystemExit(f"INCLUDE_CRIMEA: relation {rid} missing; rerun fetch_osm.py")
            en = {"UA-43": "Crimea", "UA-40": "Sevastopol"}[iso]
            gb = gpd.GeoDataFrame(
                pd.concat([gb, gpd.GeoDataFrame({"shapeISO": [iso], "shapeName": [en]},
                                                geometry=[g], crs="EPSG:4326")], ignore_index=True),
                crs="EPSG:4326")
            subj_en[iso] = en

    # Candidate relations: level 6 everywhere, level 5 inside the two cities.
    cands = []
    for rid, rel in rels.items():
        t = rel.get("tags", {})
        lvl = t.get("admin_level")
        if t.get("boundary") != "administrative" or lvl not in ("5", "6"):
            continue
        if re.search(r"[іїєґ]|ськ", t.get("name", "")):  # Ukrainian-tagged Crimean raions
            continue
        if rid in DROP_RELATIONS:
            continue
        cands.append(rid)
    geoms, failed = {}, []
    for rid in cands:
        g = assemble(rels[rid], ways)
        if g is None or g.is_empty:
            failed.append((rid, rels[rid]["tags"].get("name")))
        else:
            geoms[rid] = g
    print(f"{len(geoms)} polygons assembled, {len(failed)} failed: {failed[:20]}")
    for line in GAP_LOG:
        print("  " + line)
    missing6 = sorted(set(tags6) - set(rels))
    if missing6:
        print(f"  {len(missing6)} admin_level=6 relations not downloaded yet")

    cg = gpd.GeoDataFrame({"rid": list(geoms)}, geometry=list(geoms.values()), crs="EPSG:4326")
    pts = cg.copy()
    pts["geometry"] = cg.geometry.representative_point()
    j = gpd.sjoin(pts, gb, how="left", predicate="within")
    j = j[~j.rid.duplicated()]
    rid_iso = {r: i for r, i in zip(j.rid, j.shapeISO) if isinstance(i, str)}
    # A representative point can miss every subject: a city on a lake or a
    # bay whose middle the 2017 subject file leaves out (Petrozavodsk,
    # Makhachkala), or islands (Kurils). Those go to the subject they
    # overlap most.
    gb_sidx = gb.sindex
    for rid in geoms:
        if rid in rid_iso:
            continue
        g = geoms[rid]
        best, best_a = None, 0.0
        for k in gb_sidx.query(g, predicate="intersects"):
            a = g.intersection(gb.geometry.iloc[k]).area
            if a > best_a:
                best, best_a = gb.shapeISO.iloc[k], a
        # Island and coastal units are mostly territorial sea in OSM, so a
        # small land share is enough, for relations tagged as Russian
        # municipalities (foreign ones along the border only touch).
        status = rels[rid]["tags"].get("official_status", "")
        floor = 0.01 if status.startswith("ru:") else 0.2
        if best is not None and best_a > floor * g.area:
            rid_iso[rid] = best

    report = []
    by_iso_units = defaultdict(list)
    for o, u in units.items():
        by_iso_units[u["iso"]].append(o)
    poly_units = defaultdict(set)   # rid -> {oktmo}
    for rid in geoms:
        iso = rid_iso.get(rid)
        t = rels[rid]["tags"]
        lvl = t.get("admin_level")
        if not isinstance(iso, str):
            continue                                  # outside the 83 (or 85)
        if (iso in CITY_ISO) != (lvl == "5"):
            continue      # level 6 inside the cities, level 5 outside them
        if rid in MANUAL:
            poly_units[rid] = set(MANUAL[rid])
            report.append(f"manual {rid} {t.get('name')} -> {MANUAL[rid]}")
            continue
        codes = set()
        for c in [t.get("oktmo:user"), t.get("ref:oktmo")] + wd.get(t.get("wikidata", ""), []):
            if not c:
                continue
            c8 = re.sub(r"\D", "", c)[:8]
            c8 = c8 if c8 in units else old_to_new.get(c8, "")
            if c8 in units and units[c8]["iso"] == iso:
                codes.add(c8)
        pool = by_iso_units.get(iso, [])
        name = t.get("name", "")
        by_key = [o for o in pool if crosswalk.key(units[o]["name"]) == crosswalk.key(name)]
        exact = len(by_key) == 1
        if not exact:
            by_key = [o for o in pool if crosswalk.stem(units[o]["name"]) == crosswalk.stem(name)]
        if len(by_key) > 1:
            same = [o for o in by_key
                    if crosswalk.unit_type(units[o]["name"]) == crosswalk.unit_type(name)]
            if len(same) == 1:
                by_key = same
        if len(codes) == 1:
            o = next(iter(codes))
            if len(by_key) == 1 and by_key[0] != o:
                # An exact name beats a Wikidata code (Spassk-Dalny's item
                # carries Spassky district's code); a stem-only name loses to
                # it (Maykop city vs Maykopsky district).
                pick = by_key[0] if exact else o
                report.append(f"code/name disagree {rid} {name} ({iso}): code -> {o} "
                              f"{units[o]['name']}, name -> {by_key[0]} {units[by_key[0]]['name']}; "
                              f"took {'name' if exact else 'code'}")
                o = pick
            poly_units[rid] = {o}
        elif len(by_key) == 1:
            poly_units[rid] = {by_key[0]}
            if codes:
                report.append(f"name over codes {rid} {name}: {sorted(codes)} -> {by_key[0]}")
        else:
            report.append(f"UNMATCHED OSM {rid} {name} ({iso}) codes={sorted(codes)} "
                          f"name hits={[units[o]['name'] for o in by_key]}")

    # A Rosstat unit claimed by two polygons, or by none.
    claims = defaultdict(list)
    for rid, os_ in poly_units.items():
        for o in os_:
            claims[o].append(rid)
    for o, rids in claims.items():
        if len(rids) > 1:
            # Two relations for one unit (Crimea's Bakhchisaray district is
            # mapped twice): keep the larger, drop the rest.
            rids = sorted(rids, key=lambda r: -geoms[r].area)
            for r in rids[1:]:
                poly_units[r].discard(o)
                if not poly_units[r]:
                    del poly_units[r]
            claims[o] = rids[:1]
            report.append(f"DOUBLE CLAIM {o} {units[o]['name']} by "
                          + ", ".join(f"{r} {rels[r]['tags'].get('name')}" for r in rids)
                          + f"; kept {rids[0]}")
    unclaimed = [o for o in units if o not in claims]

    # Rosstat units no OSM relation claims. OSM is a year ahead of the
    # 1 January 2025 table in places (Krasnoyarsk merged towns into the
    # districts around them in 2025: Achinsk into Achinsky okrug), and lacks
    # a few relations outright (Kstovsky okrug of Nizhny Novgorod). Each such
    # unit is placed by its Wikidata coordinate: inside a matched polygon of
    # its subject, it joins that polygon; inside a gap in the subject's
    # coverage, the gap becomes its polygon.
    if unclaimed:
        new_to_old = defaultdict(list)
        for o, n in old_to_new.items():
            new_to_old[n].append(o)
        pts = unit_points(sorted({c for o in unclaimed for c in [o] + new_to_old[o]}))
        gb_geom = dict(zip(gb.shapeISO, gb.geometry))
        still = []
        for o in sorted(unclaimed, key=lambda o: -int(units[o]["pop2025"])):
            iso = units[o]["iso"]
            p = next((pts[c] for c in [o] + new_to_old[o] if c in pts), None)
            if p is None and o in MANUAL_POINTS:
                p = Point(MANUAL_POINTS[o])
            if p is None:
                # No coordinate: a merged okrug whose name carries this
                # unit's (Kazachinsko-Pirovsky holds Pirovsky; Nazarovsky
                # holds Nazarovo town).
                s = crosswalk.stem(units[o]["name"])
                hits = [rid for rid, os_ in poly_units.items()
                        if units[next(iter(os_))]["iso"] == iso and len(s) >= 5
                        and any(part[:len(s) - 1] == s[:-1] or s.startswith(part)
                                for part in re.split(r"[- ]", crosswalk.stem(rels[rid]["tags"].get("name", "")))
                                if len(part) >= 5)]
                if len(hits) == 1:
                    poly_units[hits[0]].add(o)
                    report.append(f"absorbed {o} {units[o]['name']} {units[o]['pop2025']} into "
                                  f"{hits[0]} {rels[hits[0]]['tags'].get('name')} (by name)")
                else:
                    still.append(o)
                continue
            inside = [rid for rid, os_ in poly_units.items()
                      if units[next(iter(os_))]["iso"] == iso and geoms[rid].contains(p)]
            if inside:
                rid = inside[0]
                poly_units[rid].add(o)
                report.append(f"absorbed {o} {units[o]['name']} {units[o]['pop2025']} into "
                              f"{rid} {rels[rid]['tags'].get('name')} (Wikidata point inside it)")
                continue
            covered = unary_union([geoms[r] for r, os_ in poly_units.items()
                                   if units[next(iter(os_))]["iso"] == iso])
            gap = shapely.make_valid(gb_geom[iso].difference(covered))
            piece = next((g for g in getattr(gap, "geoms", [gap])
                          if g.geom_type in ("Polygon", "MultiPolygon") and g.buffer(0.01).contains(p)), None)
            if piece is None:
                still.append(o)
                continue
            rid = f"gap:{o}"
            geoms[rid] = piece
            rels[rid] = {"tags": {}}
            poly_units[rid] = {o}
            report.append(f"GAP POLYGON {o} {units[o]['name']} {units[o]['pop2025']}: no OSM relation; "
                          f"drawn as the part of {iso} no other unit covers "
                          f"({piece.area:.3f} sq deg)")
        unclaimed = still
    for o in unclaimed:
        report.append(f"NO POLYGON {o} {units[o]['name']} ({units[o]['iso']}) {units[o]['pop2025']}")
    (OUT / "match_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"{len(poly_units)} polygons matched, {len(unclaimed)} Rosstat units without a polygon, "
          f"{sum(1 for r in report if r.startswith('UNMATCHED'))} OSM polygons unmatched, "
          f"{sum(1 for r in report if r.startswith('DOUBLE'))} double claims, "
          f"{sum(1 for r in report if r.startswith('code/name'))} code/name disagreements "
          f"-> match_report.txt")

    # Level 2.
    recs = []
    umap = []
    for rid, os_ in poly_units.items():
        os_ = sorted(os_)
        code = "+".join(os_)
        iso = units[os_[0]]["iso"]
        ru = " + ".join(units[o]["name"] for o in os_)
        geom = geoms[rid]
        assert_no_wrap(geom, f"relation {rid}")
        recs.append({"code": code, "name": english(rels[rid]["tags"], ru), "name_cn": ru,
                     "parent": iso, "group": iso, "osm_id": str(rid), "geometry": geom})
        umap += [(o, code, iso) for o in os_]
    adm2 = gpd.GeoDataFrame(recs, crs="EPSG:4326")
    adm2["geometry"] = clip_sea(adm2.geometry)
    adm2.to_file(OUT / "adm2.gpkg", driver="GPKG")
    with (OUT / "unit_map.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["oktmo", "code", "iso"])
        w.writerows(sorted(umap))
    print(f"adm2: {len(adm2)} polygons -> {OUT / 'adm2.gpkg'}")

    # Level 1: dissolve.
    rows1 = []
    for iso, grp in adm2.groupby("parent"):
        g = shapely.make_valid(unary_union(list(grp.geometry)))
        if g.geom_type == "GeometryCollection":
            g = MultiPolygon([p for p in g.geoms if isinstance(p, Polygon)]
                             + [q for p in g.geoms if isinstance(p, MultiPolygon) for q in p.geoms])
        assert_no_wrap(g, iso)
        rows1.append({"code": iso, "name": subj_en.get(iso, iso), "name_cn": subj_ru.get(iso, ""),
                      "group": iso, "geometry": g})
    adm1 = gpd.GeoDataFrame(rows1, crs="EPSG:4326")
    adm1.to_file(OUT / "adm1.gpkg", driver="GPKG")

    # Overlap (level-2 areas summed vs their union) and coverage (union vs
    # the geoBoundaries subject), per subject, in an equal-area projection.
    a2 = adm2.to_crs("ESRI:54009")
    a2["a"] = a2.area
    sum2 = a2.groupby("parent")["a"].sum()
    a1 = adm1.to_crs("ESRI:54009").set_index("code").area
    gba = gb.to_crs("ESRI:54009").set_index("shapeISO").area
    print("subject checks (overlap = summed level-2 area / union; cover = union / geoBoundaries subject):")
    for iso in sorted(a1.index):
        ov, cov = sum2[iso] / a1[iso], a1[iso] / gba.get(iso, float("nan"))
        if ov > 1.01 or not 0.95 <= cov <= 1.10:
            print(f"  {iso}: overlap {ov:.3f}, cover {cov:.3f}")
    print(f"adm1: {len(adm1)} subjects -> {OUT / 'adm1.gpkg'}")
    chu = adm1[adm1.code == "RU-CHU"]
    if len(chu):
        print(f"  Chukotka bbox {[round(v, 2) for v in chu.total_bounds]}, "
              f"{len(chu.geometry.iloc[0].geoms) if hasattr(chu.geometry.iloc[0], 'geoms') else 1} parts")

    fetch.write_population()


if __name__ == "__main__":
    main()
