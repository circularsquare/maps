"""
Stage 5: a REAL population per unit, from Kontur, to replace the division estimate.

The division estimate (pick_levels.py, spec 3.1) is `city_pop / n_core_units`. It is an
average and it is biased upward wherever OSM's mapping of a level is incomplete, so it
can say "this level is the right SIZE" and nothing else. This script attaches a figure
that is about a particular unit.

WHAT KONTUR ACTUALLY SHIPS, having downloaded it and looked rather than trusting the
dataset description. `kontur_boundaries_<CC>_<date>.gpkg` has ONE layer, `boundaries`,
EPSG:4326, with exactly six fields:

    admin_level      int    Kontur's own adjusted level
    osm_admin_level  str    the RAW OSM admin_level  <- joins to our `admin=N` key
    name             str    OSM `name`
    name_en          str    OSM `name:en`, often null
    population       float  Kontur Population summed over the polygon
    hasc             str    only on big units

The good news is real: `osm_admin_level` is the raw tag, so `admin=8` on our side is
`osm_admin_level == "8"` on theirs, with no vocabulary in between. The bad news is the
part the description does not mention, and it decides the whole shape of this script:

  - **There is no OSM element id and no wikidata QID.** Nothing here is a foreign key.
    So this cannot be the "collapses into a join" that spec 1.2 hoped for. The join has
    to be reconstructed from `name` plus geometry, which is a heuristic, and every
    record below therefore carries a provenance marker saying which rule produced it.
  - **It only contains `boundary=administrative`.** Every `place=quarter` /
    `place=suburb` node in our survey is simply absent from it as a unit.

The second point is what spec 1.3 warned about, but it lands more kindly than expected,
because of spec 4.6: the near-duplicate levels. Seoul's *dong* are in OSM three times —
`place=quarter` (661 nodes), `admin=8` (552 polygons), `admin=10` (285). Kontur has the
polygons. So a bare `place=quarter` node named 명동 sits inside a Kontur polygon *also*
named 명동, and that polygon's population is the node's population. Name equality is
what makes that a statement about identity rather than about location; containment
alone would only say which district the pin fell in. Hence `kontur-name`.

Where a unit is a bare node with no admin twin anywhere — Amsterdam's *wijken*, most of
Sao Paulo's `place=suburb` — nothing here can reach it, and the report says so. Summing
hexes needs an extent, spec 1.3 rejected inventing one with Voronoi, and that stands.

THE THREE RULES, weakest last, first hit wins:

    kontur-admin   our key is `admin=N`, matched a row with the same `osm_admin_level`
                   and the same name. The strong case: same source, same level, same
                   name, and the population was summed over that exact polygon.
    kontur-name    a `place=*` unit matched a row at ANY level whose name equals ours
                   AND which contains our point. The near-duplicate case above; deepest
                   level wins when several nest. Restricted to `place=*` on purpose —
                   see match_boundary() for the Amstelveen case that shows why an
                   `admin=N` unit must not take a same-named row from another level.
    kontur-hex     no name match, but the unit has real geometry from the geom pass, so
                   sum the 400 m H3 hexagons over it, area-weighted (see sum_hexes:
                   assigning whole hexes by their centres undercounts badly here).

HOW GOOD EACH RULE IS, measured rather than assumed. `--check` compares every figure
against the OSM `population` tag wherever a unit carries one — 184 units over six cities
in five countries, which is 4% coverage (spec 4.2) and self-selected, but it is the only
independent number in the corpus:

    via            n   median ratio   within 2x
    kontur-admin   89      1.24           85%
    kontur-name    82      0.96           94%
    kontur-hex     13      0.79           69%

`kontur-admin`'s 1.24 is one city, not a systematic bias: Bangkok's 46 *khet* come out at
2.5-2.7x their OSM tag, and that is the tag being a registered-household figure for a
city whose actual population is roughly twice its registration. Drop Bangkok and the
median is 1.04 with every unit inside 2x. A wrong JOIN does not look like this; it looks
like a neighbourhood holding a province's population, and nothing here does.

Read `kontur-hex` with suspicion. Its worst case is Amsterdam's Westpoort, a port
district OSM tags at 664 people and Kontur's hexes give 115,203, because Kontur
Population is a MODELLED surface (GHSL plus building footprints) that distributes people
by built volume and a container terminal is a lot of built volume. The same effect the
other way flattens dense low-rise housing: Kontur puts 8,201 in the Jordaan where CBS
counts ~19,000.

The aggregation itself was checked separately, by summing hexes over Kontur's OWN
polygons and comparing with the `population` column it ships: for units wholly inside
the city bbox the ratio sits at 0.92-1.05 (Oostzaan 1.01, Zaandam 0.96, Monnickendam
0.98), the spread being the two files' different vintages. So the code agrees with
Kontur; where Kontur disagrees with reality that is Kontur.

None of these are registered-resident counts. At borough and municipality scale Kontur
is close (Amsterdam 918,353 against 921,000; Seoul's Gangseo-gu 557,530 against
~568,000); at neighbourhood scale it is a density model, and anything shown in the
browser should say so.

WHAT IT REACHES, over the six cities tested (8,370 candidate units): **52% overall, and
that average hides the only distinction that matters** —

    units OSM draws as a polygon   3,890   91% attached
    units that are a bare node     4,480   18% attached

A polygon can be measured; a node can only be recognised. So coverage is really a
question about each city's mapping, and it ranges from Taipei at 96% (every *li* is an
`admin=10` polygon) to Amsterdam at 23% and Bangkok at 32%. Bangkok is the pure form of
the node problem: 0% of its nodes attach, because Thai `place=*` names do not repeat the
*khwaeng* names that the boundaries carry.

Two things cap this, and neither is fixable here:
  - A node whose place is not also mapped as an admin boundary under the same name is
    out of reach, full stop.
  - `kontur-hex` needs `fetch_osm.py --pass geom`, which only fetches KEPT levels. Units
    in a level the rule dropped have no geometry cached and so cannot be measured even
    when OSM has a polygon for them. That is most of Amsterdam's shortfall.

WHAT WAS REJECTED
  - The global 400 m hex file: 6.6 GB. Per-country subsets are 3 MB (NL) to 150 MB (BR).
  - The global 3 km file (169 MB): a 3 km cell is bigger than most of the units here, so
    every answer would be the same answer.
  - `kontur_topology_boundaries_*`: Kontur's flattened non-overlapping layer. It keeps
    one level per area, which throws away exactly the multi-level structure we join on.
  - rasterio: not needed, none of this is raster. Not installed either.
  - Voronoi cells to give nodes an extent — spec 1.3, and it has not changed.

LICENCE. Two different ones, and spec 1.2's "CC-BY" is right for only one of them:
  - Kontur Boundaries is **ODbL 1.0** (`hdx-odc-odbl`), because it is OSM geometry.
    Share-alike; attribute OpenStreetMap contributors and Kontur.
  - Kontur Population is **CC BY 4.0**, attribute Kontur.
Anything shown in the browser that came from here is derived from both, so the
attribution string in the output covers both. It is emitted into the JSON rather than
left in this docstring so it travels with the data.

Usage:
    python fetch_kontur.py --plan                 # zero downloads, prints sizes
    python fetch_kontur.py --only Q727,Q8684      # smoke test
    python fetch_kontur.py                        # every surveyed city, resumable
    python fetch_kontur.py --no-hex               # boundaries only, skips the big files
    python fetch_kontur.py --refresh --only Q727  # redo one city's join
    python fetch_kontur.py --check                # ratios against OSM population tags

Downloading every country the 23 surveyed cities need is 2.5 GB; a single city is 8 MB
(Korea) to 380 MB (Brazil). `--only` extends data/kontur_pop.json rather than replacing
it, so this can be walked through a few countries at a time.
"""

import argparse
import datetime
import gzip
import json
import math
import pathlib
import re
import statistics
import sys
import unicodedata
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pyogrio
import shapely
import shapely.geometry as sg
from shapely import STRtree

import osmgeom
import pick_levels as pl
from fetch_osm import UA, bbox, cities

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
CACHE = HERE / "cache" / "kontur"
META = CACHE / "hdx"          # HDX package metadata, one JSON per country
JOINED = CACHE / "joined"     # per-city results, so a rerun resumes
GEOM = HERE / "cache" / "geom"
OUT = DATA / "kontur_pop.json"

HDX_SEARCH = "https://data.humdata.org/api/3/action/package_search"

# HDX groups datasets by ISO-3166 alpha-3, the seed roster carries alpha-2, and the two
# Kontur families do NOT share a slug convention ("kontur-boundaries-korea-republic-of"
# against "kontur-population-republic-of-korea"; "turkey" against "turkiye"). Searching
# by group sidesteps the slugs entirely, so this table is the only mapping needed and
# it only has to cover the roster.
ISO3 = {
    "AE": "are", "AR": "arg", "AT": "aut", "AU": "aus", "BR": "bra", "CA": "can",
    "CL": "chl", "CN": "chn", "CO": "col", "CZ": "cze", "DE": "deu", "DK": "dnk",
    "EG": "egy", "ES": "esp", "FR": "fra", "GB": "gbr", "GR": "grc", "HK": "hkg",
    "ID": "idn", "IL": "isr", "IN": "ind", "IT": "ita", "JP": "jpn", "KE": "ken",
    "KR": "kor", "MA": "mar", "MX": "mex", "MY": "mys", "NG": "nga", "NL": "nld",
    "PE": "per", "PH": "phl", "PL": "pol", "PT": "prt", "RU": "rus", "SE": "swe",
    "SG": "sgp", "TH": "tha", "TR": "tur", "TW": "twn", "US": "usa", "VN": "vnm",
    "ZA": "zaf",
}

# A same-named polygon this far from the unit's point is a different place with the same
# name, not the unit. Only used for `admin=N` matches, where level and name already
# agree and the only doubt is that the survey's "centre" is a BOUNDING-BOX centre
# (osmgeom.centroid_of falls back to Overpass `center`), which for a crescent-shaped
# district lands outside the district itself. Small on purpose.
NEAR_KM = 5.0

_ws = re.compile(r"\s+")
_paren = re.compile(r"\s*\([^)]*\)")


def norm(s):
    """A name reduced to what two OSM extracts would agree on.

    Deliberately shallow: both sides of this join are OSM `name` tags, so the same place
    is usually byte-identical already. Stripping diacritics or administrative suffixes
    would buy a few more matches and a lot of false ones — 명동 keeps its 동, Sao Paulo's
    accents stay — so the only things removed are case, NFKC differences, runs of
    whitespace and a parenthesised qualifier.
    """
    if not s:
        return None
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = _paren.sub(" ", s)
    s = _ws.sub(" ", s).strip()
    return s or None


def haversine_km(a, b):
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


# ---------------------------------------------------------------- HDX metadata


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def hdx_country(cc):
    """Both Kontur packages for one ISO-2 country, from HDX, cached on disk.

    Returns {"boundaries": resource, "population": resource} where a resource is
    {"name", "url", "size"}, or {} if the country is not in Kontur's coverage.
    """
    iso3 = ISO3.get(cc)
    if iso3 is None:
        return {}
    META.mkdir(parents=True, exist_ok=True)
    path = META / f"{cc}.json"
    if path.exists():
        return json.loads(path.read_text("utf-8"))

    q = urllib.parse.urlencode({"fq": f"organization:kontur groups:{iso3}", "rows": 20})
    payload = _get(f"{HDX_SEARCH}?{q}")
    found = {}
    for pkg in payload["result"]["results"]:
        for kind in ("boundaries", "population"):
            if not pkg["name"].startswith(f"kontur-{kind}-"):
                continue
            res = pick_resource(pkg, kind)
            if res:
                found[kind] = res
    path.write_text(json.dumps(found, ensure_ascii=False, indent=1), encoding="utf-8")
    return found


_resname = re.compile(r"^kontur_(boundaries|population)_([A-Za-z]{2,3})_(\d{8})\.gpkg$")


def pick_resource(pkg, kind):
    """The newest plain GeoPackage in a package.

    Kontur keeps every release on the same HDX package, so a dataset lists two or three
    files that differ only by date. `kontur_topology_boundaries_*` is excluded by the
    regex rather than by name: it is a different product (see the module docstring), and
    a filter that only matches the shape we understand fails loudly on a new one instead
    of quietly picking it.
    """
    best = None
    for res in pkg.get("resources", []):
        m = _resname.match(res.get("name") or "")
        if not m or m.group(1) != kind:
            continue
        if best is None or m.group(3) > best[0]:
            best = (m.group(3), {"name": res["name"], "url": res["url"], "size": res.get("size")})
    return best[1] if best else None


# ---------------------------------------------------------------- downloading


def ensure(res):
    """Download-and-decompress a Kontur GeoPackage once. Returns the local path.

    HDX serves these gzipped and GDAL wants a real file, so the gzip stream is expanded
    straight to disk as it arrives — keeping the .gz as well would double the footprint
    for nothing, and Brazil's boundaries alone are 227 MB compressed. The write goes to
    a `.part` and is renamed only on success, so an interrupted run leaves no half file
    that the next run would mistake for a cached one.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / res["name"]
    if dest.exists():
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    mb = (res["size"] or 0) / 1e6
    print(f"      downloading {res['name']}  ({mb:,.0f} MB compressed)", flush=True)
    req = urllib.request.Request(res["url"], headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as resp:
        with gzip.GzipFile(fileobj=resp) as gz, open(part, "wb") as fh:
            while True:
                chunk = gz.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    part.replace(dest)
    print(f"      -> {dest.name}  ({dest.stat().st_size / 1e6:,.0f} MB on disk)", flush=True)
    return dest


# ---------------------------------------------------------------- the units


def candidates(city):
    """Every named, level-keyed unit of this city, from the survey cache.

    Membership is the radius test alone, exactly as in pick_levels.locate() — the
    boundary is not consulted, because `in_core` is not needed here and loading and
    preparing every city's outline to discard the answer would be pure cost. Passing
    shape=None makes locate() return the radius verdict and None for core.

    All candidates, not just the units of kept levels: the keep-list will move (spec 5
    is about to de-duplicate it), and a population keyed on an OSM id stays valid
    whatever the level rule decides later.
    """
    path = pl.SURVEY / f"{city['qid']}.json"
    if not path.exists():
        return []
    out = []
    for el in json.loads(path.read_text("utf-8")).get("elements", []):
        tags = el.get("tags") or {}
        if not tags.get("name"):
            continue
        key = pl.level_key(tags, el["type"])
        if key is None:
            continue
        inside, _ = pl.locate(city, None, el)
        if not inside:
            continue
        c = osmgeom.centroid_of(el)
        if c is None:
            continue
        out.append({"el": el, "tags": tags, "key": key, "pt": c})
    return out


def unit_id(el):
    """`way/12345`. The OSM element id is the only stable, unique, always-present key.

    wikidata was the obvious alternative and is not usable as a key: it is on 39% of
    units, and spec 4.6 measured 7,839 units carrying ~6,000 distinct QIDs — one QID
    appears on 49 of them — so it is neither total nor injective. Name+centroid moves
    whenever OSM re-centres a way. The id is carried in cache/survey/*.json already;
    data/base.json happens not to copy it through, which is a one-line change in
    build.py rather than a reason to pick a worse key. `n`, `q`, `x` and `y` are written
    into every record anyway so a name+centroid join remains possible.
    """
    return f"{el['type']}/{el['id']}"


# ---------------------------------------------------------------- boundary join


def load_boundaries(path, city):
    """The country's admin polygons, cropped to the city's own bbox.

    Read through pyogrio's bbox filter rather than loaded whole: Brazil's file is 327 MB
    of polygons for one metro area's worth of question, and the GeoPackage's R-tree makes
    the crop nearly free — Sao Paulo's 30 km box comes back as 439 rows.
    """
    s, w, n, e = bbox(city)
    gdf = pyogrio.read_dataframe(path, layer="boundaries", bbox=(w, s, e, n))
    rows, geoms = [], []
    for r in gdf.itertuples(index=False):
        if r.geometry is None or r.geometry.is_empty:
            continue
        rows.append(
            {
                "lvl": str(r.osm_admin_level or "").strip(),
                "name": r.name,
                "name_en": r.name_en,
                "pop": None if r.population is None or r.population != r.population else float(r.population),
            }
        )
        geoms.append(r.geometry)

    by_name = {}
    for i, row in enumerate(rows):
        for nm in (norm(row["name"]), norm(row["name_en"])):
            if nm:
                by_name.setdefault(nm, []).append(i)
    return rows, geoms, by_name, STRtree(geoms) if geoms else None


def _nearest(idxs, geoms, pt):
    """The candidate nearest the unit's point, and how far it is in km."""
    best, best_km = None, None
    for i in idxs:
        rp = geoms[i].representative_point()
        km = haversine_km(pt, (rp.x, rp.y))
        if best_km is None or km < best_km:
            best, best_km = i, km
    return best, best_km


def match_boundary(unit, rows, geoms, by_name, tree, city_pop):
    """One unit -> (row index, provenance) or (None, None). See the module docstring.

    The population guard at the end is what stops the whole thing embarrassing itself:
    a `place=suburb` node that happens to share a name with its municipality would
    otherwise be handed the municipality's population. Nothing that outweighs the city
    it is inside is a neighbourhood of it.
    """
    if tree is None:
        return None, None
    names = {norm(unit["tags"].get("name")), norm(unit["tags"].get("name:en"))} - {None}
    cand = sorted({i for nm in names for i in by_name.get(nm, [])})
    if not cand:
        return None, None

    pt = sg.Point(unit["pt"])
    containing = set(tree.query(pt, predicate="intersects").tolist())

    pick, via = None, None

    # 1. same raw admin level, same name. Containment only breaks ties, because the
    #    survey's point is a bounding-box centre and can genuinely sit outside its own
    #    crescent-shaped district.
    if unit["key"].startswith("admin="):
        want = unit["key"].split("=", 1)[1]
        same = [i for i in cand if rows[i]["lvl"] == want]
        inside = [i for i in same if i in containing]
        if len(inside) == 1:
            pick, via = inside[0], "kontur-admin"
        elif inside:
            pick, via = _nearest(inside, geoms, unit["pt"])[0], "kontur-admin"
        elif same:
            near, km = _nearest(same, geoms, unit["pt"])
            if km is not None and km <= NEAR_KM:
                pick, via = near, "kontur-admin"

    # 2. any level, name matches AND the polygon contains the point. Deepest wins: where
    #    a dong and the gu above it were both named the same the smaller is meant.
    #
    #    Only for `place=*` units, which carry no level of their own to disagree with.
    #    An `admin=N` unit that got past rule 1 unmatched has already told us Kontur has
    #    no row at its level, and taking a same-named row from a DIFFERENT level then
    #    silently answers a different question: Kontur's Netherlands has no
    #    `osm_admin_level=8` Amstelveen, so `admin=8` Amstelveen (the municipality,
    #    ~91,000) would have been handed the level-10 *woonplaats* of the same name,
    #    78,882. Wrong unit, plausible number, no way to see it. Left unmatched instead
    #    — or picked up by the hex sum below, which at least measures the right shape.
    if pick is None and not unit["key"].startswith("admin="):
        inside = [i for i in cand if i in containing]
        if inside:
            depth = lambda i: int(rows[i]["lvl"]) if rows[i]["lvl"].isdigit() else 0
            pick, via = max(inside, key=depth), "kontur-name"

    if pick is None or rows[pick]["pop"] is None:
        return None, None
    if city_pop and rows[pick]["pop"] > city_pop:
        return None, None
    return pick, via


# ---------------------------------------------------------------- hex join


def load_hexes(path, city):
    """400 m H3 hexagons over the city bbox -> (geometries, populations, areas).

    The layer is EPSG:3857, so the bbox has to go INTO 3857 for the read and the
    hexagons come back OUT to 4326 to meet the OSM polygons. Areas are in square
    degrees, which is fine and deliberate: they are only ever used as the denominator of
    a ratio against another area from the same neighbourhood at the same latitude, and
    the distortion cancels exactly. Reprojecting to something metric would buy nothing.
    """
    from pyproj import Transformer

    s, w, n, e = bbox(city)
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x0, y0 = fwd.transform(w, s)
    x1, y1 = fwd.transform(e, n)
    gdf = pyogrio.read_dataframe(path, layer="population", bbox=(x0, y0, x1, y1))
    if len(gdf) == 0:
        return None, None, None
    gdf = gdf.to_crs("EPSG:4326")
    geoms = np.asarray(gdf.geometry.values)
    return geoms, gdf["population"].fillna(0).to_numpy(), shapely.area(geoms)


def unit_shapes(qid):
    """`type/id` -> polygon, from the geometry pass, for units that have one.

    Points are dropped rather than kept: a hex sum needs an extent and a node has none.
    Missing file is the normal case until `fetch_osm.py --pass geom` has run for a city.
    """
    path = GEOM / f"{qid}.json"
    if not path.exists():
        return {}
    out = {}
    for el in json.loads(path.read_text("utf-8")).get("elements", []):
        if el["type"] == "node":
            continue
        g = osmgeom.polygonal(osmgeom.shape_of(el))
        if g is not None and not g.is_empty:
            out[unit_id(el)] = g
    return out


def sum_hexes(shape, tree, geoms, pops, areas):
    """(population, n overlapping hexes) for one polygon, area-weighted.

    THIS HAS TO BE AREA-WEIGHTED, and the first version was not. Assigning each hex
    whole to whichever unit holds its centre is the usual shortcut and it is simply
    wrong at this scale: a hex is ~0.74 km2 (H3 resolution 8 — "400 m" is the edge, not
    the width) and an Amsterdam *wijk* is about one hex across, so nearly every hex
    straddles a border. Measured on Amsterdam, centre-assignment gave the Jordaan 5,264
    people from a single hex against a true ~19,000, and De Pijp 17,440 against ~35,000
    — a systematic undercount, because every straddling hex is lost entirely.

    Splitting each hex by the fraction of its area inside the unit assumes people are
    spread evenly within a hex, which is the assumption the hex grid already embodies.
    """
    idx = tree.query(shape, predicate="intersects")
    if len(idx) == 0:
        return None, 0
    inter = shapely.intersection(geoms[idx], shape)
    share = shapely.area(inter) / areas[idx]
    return int(round(float((pops[idx] * share).sum()))), int(len(idx))


# ---------------------------------------------------------------- per city


def do_city(city, res, use_hex):
    """Everything for one city. Returns {"units": {...}, "summary": {...}}."""
    units = candidates(city)
    result, summary = {}, {
        "city": city["name"],
        "country": city["country"],
        "candidates": len(units),
        "kontur-admin": 0,
        "kontur-name": 0,
        "kontur-hex": 0,
        "unmatched": 0,
        # Split by whether OSM has a polygon for the unit, because that is the axis the
        # whole approach turns on: a node can only ever be reached by a name match
        # against somebody else's polygon, never by measuring its own extent.
        "poly": 0,
        "node": 0,
        "hitPoly": 0,
        "hitNode": 0,
        "unmatchedPoly": 0,
        "unmatchedNode": 0,
    }
    if not units:
        return {"units": {}, "summary": summary}

    rows, geoms, by_name, tree = load_boundaries(ensure(res["boundaries"]), city)
    print(f"      {len(units):,} candidate units, {len(rows):,} Kontur polygons in bbox", flush=True)

    unmatched = []
    for u in units:
        pick, via = match_boundary(u, rows, geoms, by_name, tree, city["pop"])
        if pick is None:
            unmatched.append(u)
            continue
        result[unit_id(u["el"])] = record(city, u, int(round(rows[pick]["pop"])), via,
                                          lvl=rows[pick]["lvl"], match=rows[pick]["name"])
        summary[via] += 1

    # Hexes only for what the name join could not reach, and only where the geometry
    # pass has actually run. Everything else stays honestly unattached.
    shapes = unit_shapes(city["qid"]) if use_hex and unmatched else {}
    if shapes and res.get("population"):
        hexes, pops, areas = load_hexes(ensure(res["population"]), city)
        htree = STRtree(hexes) if hexes is not None else None
        print(f"      {len(unmatched):,} unmatched, {len(shapes):,} have geometry, "
              f"{0 if hexes is None else len(hexes):,} hexes in bbox", flush=True)
        if htree is not None:
            still = []
            for u in unmatched:
                g = shapes.get(unit_id(u["el"]))
                if g is None:
                    still.append(u)
                    continue
                pop, n_hex = sum_hexes(g, htree, hexes, pops, areas)
                if pop is None:
                    still.append(u)
                    continue
                result[unit_id(u["el"])] = record(city, u, pop, "kontur-hex", hexes=n_hex)
                summary["kontur-hex"] += 1
            unmatched = still

    for u in units:
        node = u["el"]["type"] == "node"
        summary["node" if node else "poly"] += 1
        if unit_id(u["el"]) in result:
            summary["hitNode" if node else "hitPoly"] += 1
    for u in unmatched:
        summary["unmatched"] += 1
        if u["el"]["type"] == "node":
            summary["unmatchedNode"] += 1
        else:
            summary["unmatchedPoly"] += 1
    return {"units": result, "summary": summary}


def record(city, u, pop, via, lvl=None, match=None, hexes=None):
    """One output row. `via` is never omitted — a figure without its provenance is worse
    than no figure, since the three rules are not equally trustworthy."""
    return {
        "pop": pop,
        "via": via,
        "c": city["qid"],
        "k": u["key"],
        "n": u["tags"].get("name"),
        "q": u["tags"].get("wikidata"),
        "x": round(u["pt"][0], 5),
        "y": round(u["pt"][1], 5),
        # Which Kontur polygon supplied it, so a suspicious figure can be traced without
        # rerunning the join.
        "lvl": lvl,
        "match": match if match != u["tags"].get("name") else None,
        "hexes": hexes,
    }


# ---------------------------------------------------------------- plan / main


def plan(roster):
    """What would be downloaded, and how big. Touches HDX metadata only."""
    needed = {}
    for city in roster:
        needed.setdefault(city["country"], []).append(city["name"])

    total = cached = 0
    print(f"{len(roster)} cities in {len(needed)} countries\n")
    for cc, names in sorted(needed.items()):
        res = hdx_country(cc)
        if not res:
            print(f"{cc}  NO KONTUR DATASET  ({', '.join(names)})")
            continue
        print(f"{cc}  {', '.join(names)}")
        for kind in ("boundaries", "population"):
            r = res.get(kind)
            if not r:
                print(f"    {kind:<12} MISSING")
                continue
            # HDX leaves `size` null on some resources (Egypt, Singapore), so an unknown
            # size is printed as unknown rather than as a confident 0 MB.
            mb = None if not r["size"] else r["size"] / 1e6
            have = (CACHE / r["name"]).exists()
            if mb is not None:
                cached += mb if have else 0
                total += 0 if have else mb
            size = "       ?" if mb is None else f"{mb:>8,.0f}"
            print(f"    {kind:<12} {size} MB  {r['name']}  {'(cached)' if have else ''}")
    print(f"\nwould download {total:,.0f} MB, {cached:,.0f} MB already cached")
    print("(metadata only — no dataset bytes were fetched)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated QIDs")
    ap.add_argument("--plan", "--dry-run", dest="plan", action="store_true",
                    help="report what would be downloaded, download nothing")
    ap.add_argument("--no-hex", dest="hex", action="store_false",
                    help="boundaries only; skips the large population files")
    ap.add_argument("--refresh", action="store_true", help="redo cities already joined")
    ap.add_argument("--check", action="store_true",
                    help="compare what is already joined against OSM population tags")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    # Only cities the survey has actually run for: everything here reads
    # cache/survey/<qid>.json.
    roster = [c for c in cities(only) if (pl.SURVEY / f"{c['qid']}.json").exists()]
    if not roster:
        sys.exit("no surveyed cities match — run fetch_osm.py first")

    if args.plan:
        plan(roster)
        return

    if args.check:
        check(roster)
        return

    JOINED.mkdir(parents=True, exist_ok=True)

    for i, city in enumerate(roster, 1):
        print(f"[{i}/{len(roster)}] {city['name']} ({city['qid']}, {city['country']})", flush=True)
        cached = JOINED / f"{city['qid']}.json"
        if cached.exists() and not args.refresh:
            got = json.loads(cached.read_text("utf-8"))
            print(f"      cached: {len(got['units']):,} units")
        else:
            res = hdx_country(city["country"])
            if not res.get("boundaries"):
                print("      no Kontur boundaries for this country — skipped")
                continue
            got = do_city(city, res, args.hex)
            cached.write_text(json.dumps(got, ensure_ascii=False), encoding="utf-8")
            s = got["summary"]
            print(f"      admin {s['kontur-admin']:,}  name {s['kontur-name']:,}  "
                  f"hex {s['kontur-hex']:,}  unmatched {s['unmatched']:,}", flush=True)

    # The output is assembled from EVERY city already joined, not just the ones this run
    # touched, so `--only` is a way to extend the file rather than to replace it. The
    # per-city caches are the real state; data/kontur_pop.json is a view of them.
    everything, summaries, sources = {}, {}, {}
    for city in cities():
        cached = JOINED / f"{city['qid']}.json"
        if not cached.exists():
            continue
        got = json.loads(cached.read_text("utf-8"))
        everything.update(got["units"])
        summaries[city["qid"]] = got["summary"]
        sources[city["country"]] = {
            k: v["name"] for k, v in hdx_country(city["country"]).items()
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "generated": datetime.date.today().isoformat(),
                "key": "OSM element id, `<type>/<id>`, as in cache/survey/*.json",
                "license": {
                    "boundaries": "ODbL 1.0 — Kontur Boundaries, derived from OpenStreetMap",
                    "population": "CC BY 4.0 — Kontur Population 400m H3",
                },
                "attribution": (
                    "Population data © Kontur (CC BY 4.0), built on "
                    "© OpenStreetMap contributors (ODbL 1.0)"
                ),
                "sources": sources,
                "summary": summaries,
                "units": everything,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    report(everything, summaries)
    print(f"\nwrote {OUT} — {len(everything):,} units, {OUT.stat().st_size / 1024:,.0f} KB")


def check(roster):
    """Kontur's figure against OSM's `population` tag, wherever a unit has both.

    Two different things show up in one number and they are worth separating when
    reading it:

      - **A broken join** would produce ratios that are wild and unpatterned — a
        neighbourhood handed a province's population.
      - **Kontur being Kontur** produces a ratio that is consistently off in the same
        direction at a given scale. Kontur Population is a *modelled* surface (GHSL plus
        building footprints), so it distributes people by built volume. At borough and
        municipality scale that lands within a few percent; at neighbourhood scale it
        flattens the real density gradient, over-counting commercial and industrial land
        and under-counting dense low-rise housing.

    Only ~4% of units carry an OSM `population` tag (spec 4.2) so the sample is small
    and biased toward the units somebody cared enough to tag, but it is free and it is
    the only independent number in the corpus.
    """
    pairs = []
    for city in roster:
        cached = JOINED / f"{city['qid']}.json"
        if not cached.exists():
            continue
        got = json.loads(cached.read_text("utf-8"))["units"]
        for u in candidates(city):
            rec = got.get(unit_id(u["el"]))
            osm = pl.parse_pop(u["tags"].get("population"))
            if rec and osm:
                pairs.append((rec["via"], u["key"], rec["pop"] / osm, osm, rec["pop"], u["tags"]["name"]))
    if not pairs:
        print("no unit has both an OSM population tag and a Kontur figure")
        return

    print(f"{len(pairs):,} units have both an OSM population tag and a Kontur figure\n")
    print(f"  {'via':<16}{'n':>7}{'median':>9}{'within 2x':>11}")
    for via in ("kontur-admin", "kontur-name", "kontur-hex"):
        rs = [p[2] for p in pairs if p[0] == via]
        if not rs:
            continue
        near = sum(1 for r in rs if 0.5 <= r <= 2.0) / len(rs)
        print(f"  {via:<16}{len(rs):>7,}{statistics.median(rs):>9.2f}{near:>11.0%}")

    print(f"\n  {'level':<20}{'n':>7}{'median':>9}")
    keys = {}
    for p in pairs:
        keys.setdefault(p[1], []).append(p[2])
    for k, rs in sorted(keys.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k:<20}{len(rs):>7,}{statistics.median(rs):>9.2f}")

    print("\n  worst 10 by ratio:")
    for via, k, r, osm, kon, name in sorted(pairs, key=lambda p: -max(p[2], 1 / p[2]))[:10]:
        print(f"    {name[:26]:<28}{k:<20}{via:<14} osm {osm:>9,}  kontur {kon:>9,}  {r:>6.2f}")


def report(units, summaries):
    tot = {k: 0 for k in ("candidates", "kontur-admin", "kontur-name", "kontur-hex",
                          "unmatched", "poly", "node", "hitPoly", "hitNode",
                          "unmatchedPoly", "unmatchedNode")}
    print(f"\n  {'city':<18}{'cand':>7}{'admin':>8}{'name':>8}{'hex':>7}{'none':>8}"
          f"{'cover':>8}{'poly':>8}{'node':>8}")
    for s in sorted(summaries.values(), key=lambda s: -s["candidates"]):
        for k in tot:
            tot[k] += s.get(k, 0)
        hit = s["candidates"] - s["unmatched"]
        cov = hit / s["candidates"] if s["candidates"] else 0
        pc = s["hitPoly"] / s["poly"] if s.get("poly") else 0
        nc = s["hitNode"] / s["node"] if s.get("node") else 0
        print(f"  {s['city']:<18}{s['candidates']:>7,}{s['kontur-admin']:>8,}"
              f"{s['kontur-name']:>8,}{s['kontur-hex']:>7,}{s['unmatched']:>8,}{cov:>8.0%}"
              f"{pc:>8.0%}{nc:>8.0%}")
    n = tot["candidates"] or 1
    hit = n - tot["unmatched"]
    print(f"\n  {len(units):,} of {tot['candidates']:,} candidate units carry a Kontur "
          f"population ({hit / n:.0%})")
    for k in ("kontur-admin", "kontur-name", "kontur-hex"):
        print(f"    {k:<16}{tot[k]:>8,}  ({tot[k] / n:.0%})")
    # The split that decides what is reachable at all: a unit OSM draws as a polygon can
    # fall back on its own extent; a bare node depends entirely on somebody else having
    # mapped the same place as an admin boundary under the same name.
    for kind in ("poly", "node"):
        d, h = tot[kind], tot["hit" + kind.capitalize()]
        print(f"    units with a {kind:<5}{d:>8,}  {h:,} attached ({h / (d or 1):.0%})")


if __name__ == "__main__":
    main()
