"""
Stage 2: decide, per city, WHICH LEVEL is that city's "neighbourhood".

THE RULE (Anita's, and it is the right one). Do not filter unit by unit. Filter whole
levels: if most of a city's `admin_level=9` units clear the population floor, keep ALL of
them, and if they don't, keep none. Per-unit filtering would hand back a ragged set
— Shibuya in, the ward next door out, because one of them happens to sit under 10k —
and a quiz deck built from that is incoherent. A level is the thing a local would
recognise as "the way this city is divided", so a level is the thing we keep or drop.

The size test is a FLOOR: units averaging 10,000 people or more, no upper limit. A
relative ceiling at 10% of city population was tried and dropped — see BAND_MIN for the
measurements that killed it. Large divisions are wanted.

WHY THE LEVEL KEY IS RAW, NOT NORMALISED. A level key is either `admin=9` or
`place=suburb`, straight off the OSM tags. Nothing maps a German admin_level=9 onto a
Japanese one, because they mean different things and the rule never compares them —
every threshold is evaluated inside a single city. This is the specific reason we are
not using Overture's normalised subtypes; see fetch_osm.py.

WHY A UNIT WITH BOTH TAGS COUNTS AS `admin`. Plenty of districts carry
`boundary=administrative` + `admin_level=9` + `place=suburb` together. Counting such a
unit under both keys would double it and let one physical set of polygons pass the rule
twice. `admin` wins because it is the sharper signal; the `place`-only units (typically
bare nodes with no boundary at all) still get their own bucket, which is what we want
since they are usually a different and often richer set.

POPULATION IS THE WEAK LINK RIGHT NOW. The floor test needs a population per unit and
OSM's `population` tag is patchy. So the fraction is computed over units that HAVE one,
and a level is only judged when the sample is big enough to mean anything
(MIN_POP_SAMPLE units and MIN_POP_FRAC of the level). Everything else is reported
`unknown` rather than guessed at — those are precisely the cities Kontur is for.

Usage:
    py pick_levels.py --report        # the survey table, writes nothing
    py pick_levels.py                 # write data/levels.json
    py pick_levels.py --report --only Q90
"""

import argparse
import collections
import json
import math
import pathlib
import re
import statistics
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import osmgeom
from fetch_osm import PLACE_VALUES, cities

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
SURVEY = HERE / "cache" / "survey"
BOUNDARY = HERE / "cache" / "boundary"
OUT = DATA / "levels.json"
OVERRIDE = DATA / "levels_override.json"

# A FLOOR, NOT A BAND (changed 2026-08-29). There used to be a ceiling at 10% of the city
# population. It was dropped because its width scaled with city size — the floor is
# absolute and the ceiling was relative, so the usable range was 9x for Amsterdam
# (10k..92k) and 219x for Beijing (10k..2.19M). That made the rule strict on small cities
# and nearly toothless on large ones, which is backwards. Amsterdam ended up keeping
# NOTHING: its stadsdelen average 123k against a 92k ceiling while its wijken sit just
# under the 10k floor, so it fell out of the band on both sides at once.
#
# Large divisions are now wanted, so the only size test is this floor. What bounds the
# top end instead is MIN_UNITS: a level of several named parts each averaging enough people
# is a reasonable definition of "a way this city is divided", and it needs no ceiling.
BAND_MIN = 6_000

# 10,000 -> 6,000 (2026-08-29). At 10,000 the floor was excluding real neighbourhood-scale
# divisions, and the tell was how many cities sat just underneath it: Dublin's
# `place=suburb` missed by 122 people and Tunis's by 282, and both cities kept NOTHING as
# a result — they vanished from the browser over a rounding error.
#
# THE MEASUREMENT THAT PICKED 6,000 rather than 9,000. Dropping to 9,000 rescues Dublin and
# Tunis and nothing else; no further city is rescued at any value below that. So the
# question was never about the empty cities, it was about what sits in the 6k-10k band in
# cities that already keep something. That turned out to be 33 levels, and they are not
# marginal: Budapest `admin=10` (209 units, 90% wikidata, 100% polygons), Amsterdam
# `place=neighbourhood` (172), Toronto's neighbourhoods (352), Stockholm's suburbs (270),
# Vilnius `admin=8`, Warsaw, Berlin, Belgrade. These are the divisions the project is
# actually about.
#
# Amsterdam is the case that closes it. §4.5 records it keeping nothing because "its
# stadsdelen average 123k against a 92k ceiling while its wijken sit just under the 10k
# floor" — it fell out of the band at both ends. Dropping the ceiling fixed one end; this
# fixes the other, and its buurten (6,478) finally qualify.
#
# A 6,000-person division is a neighbourhood by any ordinary reading, so this is not a
# loosening so much as a correction of where the line was drawn.

# Share of a level's population-bearing units that must clear the floor for the level
# to be kept. Anita's number. It is deliberately not 100%: every real city has a couple
# of odd units (an industrial dock, a park district, the financial core that nobody
# sleeps in) and demanding purity would reject the level over them.
KEEP_FRAC = 0.70

# How much population evidence before a verdict is allowed at all.
MIN_POP_SAMPLE = 5
MIN_POP_FRAC = 0.25

# A level with only a handful of units is a mapping artefact. Kept separate from the
# floor test so the report can show WHY something was dropped.
#
# With the ceiling gone this is also what bounds the TOP end: nothing else stops a level
# of two enormous halves from qualifying.
#
# LOWERED 8 -> 4 (2026-08-29), because at 8 it was excluding the wrong thing. The old
# comment read "not that city's neighbourhood scheme — it is its borough scheme (NYC has
# 5)", and a borough scheme is exactly the first division a browser wants; New York's five
# boroughs are in the survey three times over (`admin=7` n=5, `admin=6` n=6,
# `place=suburb` n=5, all 100% wikidata and 100% polygons) and all three were dropped
# here. Corpus-wide, 15 levels across 14 cities sit at 4-7 units with >=80% wikidata AND
# >=80% polygons: Amsterdam, Delhi, Hong Kong, Kuala Lumpur, Lisbon, Madrid, Mumbai, New
# York, Rome, San Francisco, Singapore, Sydney, Toronto, Warsaw. That is a tier the
# corpus was structurally unable to represent, not a set of artefacts.
#
# 4 rather than 2 because a two-unit "division" (an east/west half, a river split) really
# is an artefact, and because the guard still has to bound the top end on its own.
MIN_UNITS = 4

# The division estimate divides city population by unit count, so BOTH must describe the
# same area. An earlier version divided by every unit in the query radius while the
# population was the city's alone, which understated unit size by exactly the factor by
# which the radius overshot the city — 5.5x for Sao Paulo, whose 30 km box covers the
# whole metro but whose population figure is the municipio's. Paris `admin=10` came out
# at 4,393 per unit (rejected) when the honest figure over units inside Paris is ~14,700.
#
# So the estimate counts CORE units only, and the verdict then applies to the whole level
# including its peripheral units — which is what keeping neighbouring municipalities
# means. This needs enough core units for the ratio to mean anything.
MIN_CORE_UNITS = 5

# ...and enough of the level must BE core, or the level is not this city's division at
# all. Sao Paulo's `admin=10` has 417 units of which 10 are inside the municipio; that is
# not the distrito level (there are ~96 of those), it is the surrounding municipalities'
# own subdivisions, and dividing Sao Paulo's population by 10 of them yields a nonsense
# 1,150,000 per unit. Low enough to keep genuinely metro-spanning levels: Sao Paulo's
# `place=neighbourhood` at 18% and Paris's `admin=10` at 30% both survive.
MIN_CORE_FRAC = 0.15

# KNOWN BIAS, and it points one way. The estimate assumes the level is COMPLETELY mapped
# inside the city. Where OSM has only some of the units the divisor is too small and unit
# size is overstated, so `keep-est` errs toward keeping. Do not read estPop as a real
# population; it is a size class.
#
# Measured against Kontur, the overstatement reaches 77x (Seoul admin=10) and 6,370x
# (admin=11) — far past "errs toward keeping". `mark_sparse` is the defence, and it needs
# no external data. Read it before touching the estimate.

# The size ordering OSM documents for `place=*`, coarsest first. Used by `mark_sparse` the
# same way admin_level is: a finer division of the same city cannot have fewer units than
# a coarser one unless it is under-mapped.
PLACE_ORDER = ["borough", "suburb", "quarter", "neighbourhood", "city_block"]

# A boundary that contains almost none of the city's own candidate units is not a city
# boundary — it is a sub-unit the centre coordinate happened to land in. The geometric
# fallback in fetch_osm.py picks the DEEPEST enclosing relation, which is right for Cairo
# (a governorate) and wrong for Lagos (Shomolu LGA) and Nairobi ("CBD division").
#
# This matters beyond a mislabelled coreFrac: a too-small boundary drags coreFrac under
# MIN_CORE_FOR_ESTIMATE, which switches OFF the division estimate for that city and
# silently drops levels that should have been kept. So an untrustworthy boundary is
# discarded entirely and the city falls back to radius-only, which is the honest state.
MIN_BOUNDARY_TRUST = 0.10

# THE FALLBACK WHEN THERE IS NO BOUNDARY. Without one the estimate used to divide the
# city-proper population by every unit in the query box — the São Paulo error of §3.1, the
# very thing core units exist to prevent, reinstated for exactly the cities that could not
# be measured.
#
# The error is one of SCALE, not shape: the numerator describes the city and the
# denominator describes the box. Fixing scale needs the city's SIZE, and Wikidata's P2046
# supplies that for free — no shapefile, no download. So a disc of the city's own area,
# centred on the city, stands in for the boundary.
#
# MEASURED against the real boundary on the 48 cities that have one, comparing the count a
# disc yields with the true core count: median 1.01x, 40 of 48 within 1.5x, 46 within 2x.
# Worst cases São Paulo 2.27x and Dubai 0.05x — against a status quo whose error reaches
# 5.5x. It is an approximation and it is a large improvement.
#
# IT IS A DENOMINATOR ONLY. It must never populate `coreFrac`. The recall column of the
# same measurement shows a disc selects a materially different SET of units even where the
# count matches (Cairo 52%, Tel Aviv 44%), so "this unit is in the city proper" is not
# something a circle may assert — the same objection that rejected Voronoi cells in §1.3.
AREAS = DATA / "city_area.json"

# How many node-only units a rejected level must supply outlines for before it is worth
# fetching its geometry. One coincidental name match is not a donor level.
MIN_DONOR_HITS = 5

PLACE_SET = set(PLACE_VALUES.split("|"))
_num = re.compile(r"\d[\d\s,.]*")


def parse_pop(raw):
    """OSM `population` is free text: '1,234', '12 345', '~4000', '3000 (2011)'."""
    if not raw:
        return None
    m = _num.search(raw)
    if not m:
        return None
    digits = re.sub(r"[^\d]", "", m.group(0))
    if not digits:
        return None
    try:
        v = int(digits)
    except ValueError:
        return None
    return v if 0 < v < 100_000_000 else None


# A way can be an administrative boundary AND a river at the same time, because rivers are
# what borders often follow: London's `River Thames` ways carry `boundary=administrative`,
# `admin_level=6/8` AND `waterway=river`. Those are boundary LINES, not places, and they
# entered the corpus as four "neighbourhoods" called River Thames — one of which then
# matched a 21 km² Who's On First polygon of the same name and got drawn as a district.
#
# 813 elements corpus-wide are both. Excluding them is a tag-level fact, not a name
# blocklist, so it needs no per-city maintenance.
WATER_NATURAL = {"water", "coastline", "bay", "strait", "wetland"}


def is_water_feature(tags):
    return bool(
        tags.get("waterway") or tags.get("water") or tags.get("natural") in WATER_NATURAL
    )


def is_boundary_linework(tags, el_type):
    """A `boundary=administrative` WAY carrying no `place` tag is a border, not an area.

    The survey query is `nwr[boundary=administrative]`, and OSM routinely repeats the
    boundary tags on the MEMBER WAYS of a relation as well as on the relation itself.
    Such a way is named after whatever the border runs along, so it arrives as a unit
    called `River Thames`, `Pedro Gil Street`, `Bergenline Avenue` or `중랑천`.

    This generalises the water rule above, which caught only the subset of borders that
    follow rivers. Measured corpus-wide the full problem is **3,137 units, 9.5% of
    everything** — 57% of every `admin=6` unit and 40% of every `admin=8` one, against
    0.01% of `place=*`. It does not bite at the margin, it decides level picks: London
    `admin=8` is 87 units of which 53 are Thames/Wandle/Beverley Brook segments, and the
    34 that remain are London's 33 boroughs. `admin=6` is 15 units, 14 named `River
    Thames`. All 4,545 way-units carried `poly: 1` and 3,502 of them were being drawn as
    polygons, so they inflated the shape-coverage figure too.

    **The test is the `place` tag rather than the geometry** because the survey pass is
    `out tags center` and has no node list to close a ring with, and re-running 56
    Overpass surveys to learn what a tag already implies is not worth the donated CPU.
    Scored against the 4,252 ways the geometry cache CAN adjudicate: 99.9% recall
    (2,837 of 2,841) for 65 false drops, and those 65 are mostly closed street loops
    (`Pedro Gil Street`, `Elliptical Road`) that are junk anyway. Legitimate closed ways
    are untouched — 95% of them carry a `place` tag against 0.1% of the line-work.

    Relations are deliberately exempt: a `boundary=administrative` relation is an area by
    construction, and checking `cache/geom/` confirmed none of them are line-work.
    """
    return (
        el_type == "way"
        and tags.get("boundary") == "administrative"
        and "place" not in tags
    )


def level_key(tags, el_type):
    """`admin=N` beats `place=X` when a unit carries both — see the module docstring."""
    if is_water_feature(tags) or is_boundary_linework(tags, el_type):
        return None
    if tags.get("boundary") == "administrative" and tags.get("admin_level", "").isdigit():
        return f"admin={tags['admin_level']}"
    place = tags.get("place")
    if place in PLACE_SET:
        return f"place={place}"
    return None


_OVERRIDES = None


def overrides():
    """`data/levels_override.json`, parsed and cached. Two accepted shapes per city:

        "Q1353": ["admin=9"]                                    keep-list only
        "Q1353": {"keep": ["admin=9"],
                  "excludeNames": ["^Sector\\\\s+\\\\d+$"]}     keep-list + a name filter

    **Why a name filter exists at all**, since it is obviously a blunt instrument. Delhi's
    `admin=9` holds 161 units that are two different things sharing one `admin_level`: 45
    real Delhi areas (Vikaspuri, Karol Bagh, Rohini, Vasant Kunj, Saket) and 116 numbered
    planned-colony sectors, most of them Noida's and Gurgaon's. Nothing else separates
    them — `coreFrac` cannot, because Dwarka's and Rohini's sectors genuinely ARE in Delhi
    (28 of the 69 core units are "Sector N"); P31 cannot, because the level is 3%
    wikidata; and area spread cannot, because legitimate divisions vary in area by more
    than these do — Sydney's LGAs span 34,000x and are fine.

    So this is per-city knowledge, which is what the override file is for. It is scoped to
    one city on purpose: `^Sector` would be catastrophic in Singapore or Warsaw.
    """
    global _OVERRIDES
    if _OVERRIDES is None:
        raw = json.loads(OVERRIDE.read_text("utf-8")) if OVERRIDE.exists() else {}
        _OVERRIDES = {}
        for qid, v in raw.items():
            if qid.startswith("_"):
                continue  # `_doc` and friends are notes, not cities
            entry = ({"keep": v, "excludeNames": []} if isinstance(v, list)
                     else {"keep": v.get("keep"),
                           "excludeNames": v.get("excludeNames") or []})
            entry["_re"] = [re.compile(p, re.I) for p in entry["excludeNames"]]
            _OVERRIDES[qid] = entry
    return _OVERRIDES


def excluded_name(qid, name):
    """Is this unit name blocked by its city's override? Applied at INGEST, in both
    pick_levels and build, so the level statistics and the browser agree — filtering only
    at display would leave the division estimate dividing by units nobody can see."""
    entry = overrides().get(qid)
    return bool(entry) and any(r.search(name or "") for r in entry["_re"])


def haversine_km(a, b):
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def boundary_name(qid):
    """What the boundary pass actually picked for this city -> "Name (admin N)", or None.

    Kept separate from `city_shape` because it is wanted precisely when the shape is
    NOT used: a rejected boundary is only explicable if we can say what was rejected.
    """
    path = BOUNDARY / f"{qid}.json"
    if not path.exists():
        return None
    els = json.loads(path.read_text(encoding="utf-8")).get("elements", [])
    rels = [e for e in els if e.get("type") == "relation" and (e.get("tags") or {}).get("name")]
    if not rels:
        return None
    tags = rels[0]["tags"]
    lvl = tags.get("admin_level")
    return tags["name"] + (f" (admin {lvl})" if lvl else "")


_AREAS = None


def area_radius_km(qid):
    """Radius of a disc with this city's Wikidata area, or None. See AREAS.

    Absent file, absent city and absent area are all the same answer — the caller falls
    back to the old radius-only behaviour, so the pipeline runs unchanged for anyone who
    has not fetched areas. Refresh with `python roster.py --areas`.
    """
    global _AREAS
    if _AREAS is None:
        _AREAS = json.loads(AREAS.read_text("utf-8")) if AREAS.exists() else {}
    a = _AREAS.get(qid)
    return math.sqrt(a / math.pi) if a else None


def within_km(city, el, km):
    """Is this element's centroid within `km` of the city centre?

    Recomputes the centroid that `locate` already found rather than widening locate's
    return, which has five call sites across three files. Survey elements are fetched with
    `out tags center`, so centroid_of is a dictionary lookup here, not geometry.
    """
    c = osmgeom.centroid_of(el)
    return c is not None and haversine_km(c, (city["lon"], city["lat"])) <= km


def city_geom(qid):
    """The city's own outline as plain geometry, if the boundary pass has fetched it.

    A city's relation is sometimes split into several (a city and its coterminous
    county, say), so every returned relation is unioned rather than the first one taken.

    Split out of `city_shape` because the browser's zoomed-out city layer (spec 6a.6)
    wants to DRAW this outline, and a prepared geometry cannot be drawn, written or
    simplified. Every membership test still goes through `city_shape`.
    """
    path = BOUNDARY / f"{qid}.json"
    if not path.exists():
        return None
    els = json.loads(path.read_text(encoding="utf-8")).get("elements", [])
    shapes = [g for g in (osmgeom.shape_of(e) for e in els) if g is not None and not g.is_empty]
    if not shapes:
        return None
    from shapely.ops import unary_union

    geom = shapes[0] if len(shapes) == 1 else unary_union(shapes)
    return osmgeom.polygonal(geom)


def city_shape(qid):
    """`city_geom`, prepared for repeated containment tests."""
    from shapely.prepared import prep

    geom = city_geom(qid)
    if geom is None:
        return None
    # Prepared once, queried thousands of times. Without this, every `contains` rebuilds
    # the index over a relation that can carry tens of thousands of vertices (Beijing's
    # municipality, Tokyo's prefecture), which turns a fast filter into the slowest step
    # in the pipeline.
    return prep(geom)


def locate(city, shape, el):
    """-> (belongs, in_core). Membership and centrality are two questions, not one.

    **`belongs` is the radius test only.** An earlier version also required the unit to
    sit inside the city's own administrative boundary, which rejected Hoboken and Jersey
    City from New York. That was wrong for this project: a compact core surrounded by
    separately-incorporated municipalities that everyone treats as neighbourhoods is a
    common pattern — the Petite Couronne around Paris, the comuni around Milan, most of
    the US — and those municipalities are good quiz targets. They stay.

    The radius still applies, and it is doing real work: it is what keeps the Ogasawara
    Islands out of Tokyo, whose administrative boundary is a prefecture reaching 1,000 km
    into the Pacific, and most of Beijing's farmland out of Beijing.

    **`in_core` records the answer to the other question** — is this inside the city
    proper, or is it a neighbour? Nothing filters on it. It exists because "Hoboken" and
    "Park Slope" are both fair answers to "name a New York neighbourhood" but they are
    not the same *kind* of answer, and the browser and the quiz will want to tell them
    apart. `None` where no boundary was fetched.
    """
    c = osmgeom.centroid_of(el)
    if c is None:
        return False, None
    if haversine_km(c, (city["lon"], city["lat"])) > city["radiusKm"]:
        return False, None
    if shape is None:
        return True, None
    return True, shape.contains(osmgeom.sg.Point(c))


def trustworthy(city, shape, elements):
    """Does this boundary contain enough of the city's own units to be believable?

    See MIN_BOUNDARY_TRUST. Returns `(shape, reject)` — the shape, or `None` plus a
    reject record when it must be discarded and the city falls back to radius-only.

    The reject record exists because "no boundary" and "a boundary we threw away" look
    identical downstream but are not the same fact, and only the second one can be
    explained to someone reading the map. It carries what was rejected and how badly it
    missed, so the browser can say *Istanbul's boundary was one neighbourhood of Fatih*
    rather than the uninformative *no usable boundary*.
    """
    if shape is None:
        return None, None
    inside = total = 0
    for el in elements:
        tags = el.get("tags") or {}
        if not tags.get("name") or level_key(tags, el["type"]) is None:
            continue
        ok, in_core = locate(city, shape, el)
        if not ok:
            continue
        total += 1
        inside += bool(in_core)
    if total and inside / total < MIN_BOUNDARY_TRUST:
        return None, {
            "picked": boundary_name(city["qid"]),
            "covered": round(inside / total, 4),
            "floor": MIN_BOUNDARY_TRUST,
        }
    return shape, None


def survey_city(city):
    """Roll one cached survey response up into per-level statistics."""
    path = SURVEY / f"{city['qid']}.json"
    if not path.exists():
        return None
    elements = json.loads(path.read_text(encoding="utf-8")).get("elements", [])
    shape, boundary_reject = trustworthy(city, city_shape(city["qid"]), elements)
    # "No boundary" now covers three different situations, and they must not be reported
    # as one. With the roster at 301 cities the passes run at different rates — the survey
    # is far ahead of the boundary pass — so most boundary-less cities are simply NOT
    # FETCHED YET, which is a state of the pipeline, not a fact about the city. Telling a
    # reader that Accra "has no usable boundary" when nothing has looked for one yet would
    # be the same kind of confident wrongness §6a.5 already had to unwind once.
    if shape is not None:
        boundary_state = "ok"
    elif not (BOUNDARY / f"{city['qid']}.json").exists():
        boundary_state = "not-fetched"
    elif boundary_reject:
        boundary_state = "rejected"
    else:
        boundary_state = "empty"
    # Only wanted when the boundary is missing; computed once either way because it costs
    # a dictionary lookup and a square root.
    r_eq = None if shape is not None else area_radius_km(city["qid"])

    levels = {}
    n_outside = 0
    for el in elements:
        tags = el.get("tags") or {}
        if not tags.get("name"):
            # Unnamed units cannot be browsed or guessed, so they are not candidates and
            # must not dilute the level's statistics either.
            continue
        if excluded_name(city["qid"], tags["name"]):
            continue
        key = level_key(tags, el["type"])
        if key is None:
            continue
        inside, in_core = locate(city, shape, el)
        if not inside:
            n_outside += 1
            continue
        L = levels.setdefault(
            key, {"n": 0, "pops": [], "poly": 0, "wd": 0, "core": 0, "disc": 0})
        L["n"] += 1
        if in_core:
            L["core"] += 1
        if r_eq is not None and within_km(city, el, r_eq):
            L["disc"] += 1
        if el["type"] in ("way", "relation"):
            L["poly"] += 1
        if tags.get("wikidata"):
            L["wd"] += 1
        p = parse_pop(tags.get("population"))
        if p is not None:
            L["pops"].append(p)

    out = {}
    for key, L in sorted(levels.items()):
        pops = L["pops"]
        n_pop = len(pops)
        enough = n_pop >= MIN_POP_SAMPLE and n_pop >= MIN_POP_FRAC * L["n"]
        over_floor = None
        if enough:
            hits = sum(1 for p in pops if p >= BAND_MIN)
            over_floor = hits / n_pop

        # DIVISION ESTIMATE — the fallback when OSM has no populations, which is most of
        # the levels we actually want (§4.2). If a level divides the city into n parts,
        # the average part holds city_pop / n people. No per-unit data needed.
        #
        # It estimates the AVERAGE, never any particular unit, so it can only support a
        # level-wide verdict — which is exactly the shape of the rule. It answers "is this
        # the right SIZE of division for this city".
        #
        # Denominator, best available first, so it describes the same area as the city
        # population it divides; see MIN_CORE_UNITS.
        #
        #   core   units inside the real boundary
        #   disc   units inside a disc of the city's Wikidata area (AREAS)
        #   radius every unit in the query box — biased, and the old behaviour
        #
        # The disc must clear the same two gates as core, and when it cannot the level
        # falls through to `radius` rather than to no estimate at all. That keeps this
        # change a strict improvement: where the disc works it corrects the scale, and
        # where it does not, the level is exactly where it was before. Dubai is the case
        # that matters — its P2046 is the historic core, so its disc holds 5% of units,
        # fails MIN_CORE_FRAC, and is discarded rather than believed.
        def gates_pass(k):
            return L[k] >= MIN_CORE_UNITS and L[k] / L["n"] >= MIN_CORE_FRAC

        if shape is not None:
            n_denom = L["core"] if gates_pass("core") else None
            basis = "core" if n_denom else None
        elif r_eq is not None and gates_pass("disc"):
            n_denom, basis = L["disc"], "disc"
        else:
            n_denom, basis = (L["n"] or None), "radius"
        mean_pop = city["pop"] / n_denom if n_denom else None
        est_over_floor = mean_pop is not None and mean_pop >= BAND_MIN
        out[key] = {
            "n": L["n"],
            "nPop": n_pop,
            "polyFrac": round(L["poly"] / L["n"], 3),
            "wdFrac": round(L["wd"] / L["n"], 3),
            # Share inside the city proper. Not a filter — see locate(). A level at 0%
            # is entirely surrounding municipalities, which is worth seeing but is not
            # by itself a reason to drop it.
            "coreFrac": None if shape is None else round(L["core"] / L["n"], 3),
            # Which of the three denominators the estimate actually used. `keep-est` is
            # already the weaker verdict; this says how much weaker.
            "estBasis": basis,
            "medianPop": int(statistics.median(pops)) if pops else None,
            "meanPop": None if mean_pop is None else int(mean_pop),
            "overFloor": None if over_floor is None else round(over_floor, 3),
            "verdict": verdict(L["n"], over_floor, est_over_floor,
                               mean_pop is not None),
        }
    mark_sparse(out)
    return {
        "city": city["name"],
        "pop": city["pop"],
        "floor": BAND_MIN,
        "hasBoundary": shape is not None,
        "boundaryState": boundary_state,
        "boundaryReject": boundary_reject,
        # The disc's radius, present only when it was available to be used.
        "areaRadiusKm": None if r_eq is None else round(r_eq, 1),
        "nOutside": n_outside,
        "levels": out,
    }


def norm_name(s):
    """Casefold and strip accents. Deliberately NOT fuzzy: borrowing an outline asserts
    that two records are the same place, and edit distance is not evidence for that."""
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).casefold().strip()


def find_donors(city, shape, elements, keep):
    """Levels NOT kept that hold a polygon for a place a kept level only has as a node.

    Half this project's units are bare nodes, very unevenly: Copenhagen 2% polygons
    against Hamburg's 100%, because that is a national mapping convention rather than
    anything about the cities. But the same place is frequently mapped twice at different
    levels (§4.6), so a node-only `place=suburb` often has an `admin=9` twin carrying the
    outline. Those twins are worth fetching geometry for even though the level itself was
    rejected — measured, Sao Paulo has 445 such donors at `admin=10`/`admin=9` against
    109 inside its kept levels, taking it from 13% shapes to roughly half.

    Only the level NAMES are returned; the actual matching happens in build.py once the
    geometry exists, and is gated on containment there.
    """
    want = set()
    have_poly = collections.defaultdict(set)
    for el in elements:
        tags = el.get("tags") or {}
        name = tags.get("name")
        key = level_key(tags, el["type"])
        if not name or key is None or not locate(city, shape, el)[0]:
            continue
        if el["type"] == "node":
            if key in keep:
                want.add(norm_name(name))
        else:
            have_poly[norm_name(name)].add(key)

    donors = collections.Counter()
    for n in want:
        for key in have_poly.get(n, ()):
            if key not in keep:
                donors[key] += 1
    # One stray same-named polygon is coincidence, not a donor level worth a fetch.
    return sorted(k for k, c in donors.items() if c >= MIN_DONOR_HITS)


def mark_sparse(levels):
    """Flag `admin=N` levels that OSM has clearly only partly mapped.

    `admin_level` is a NESTING hierarchy: every admin=10 unit lies inside an admin=8 one,
    so a complete admin=10 layer must have at least as many units as admin=8. Fewer means
    the level is under-mapped — and that is fatal to the division estimate, whose
    denominator is a unit count. The error is not subtle. Seoul:

        admin=8   552 units   estimate 21,709   Kontur says 19,549   fine
        admin=10  285 units   estimate 34,686   Kontur says    451   77x too high
        admin=11   18 units   estimate 522,222  Kontur says     82   6,370x too high

    OSM holds 285 of Seoul's ~13,000 통 and 18 of its ~100,000 반, so dividing the city's
    population by those counts is meaningless. Both were kept as `keep-est`.

    This check finds exactly that case using nothing but the unit counts, which is why it
    is here rather than waiting on Kontur: it needs no downloads and covers every city.
    A level whose estimate cannot be trusted is not silently dropped — it becomes
    `sparse`, so the report says why.

    `place=*` gets the same treatment as its own separate family, ordered by the size
    ranking OSM documents (borough > suburb > quarter > neighbourhood > city_block). The
    two families are never compared with each other — they are independent ways of
    dividing a city, not one hierarchy.
    """
    admin = [k for _, k in sorted(
        (int(k.split("=")[1]), k) for k in levels if k.startswith("admin=")
    )]
    place = [f"place={v}" for v in PLACE_ORDER if f"place={v}" in levels]

    for family in (admin, place):
        seen_max = 0
        for key in family:
            n = levels[key]["n"]
            if n < seen_max:
                levels[key]["sparse"] = True
                # This check defends the ESTIMATE — under-mapping is what makes a divisor
                # too small — so it may only overturn a verdict the estimate produced. A
                # level whose own reported populations independently clear the floor is
                # immune, which is exactly what the old `verdict == "keep-est"` test did
                # back when reported data had its own verdict name.
                #
                # Without that exemption this fires on Paris `admin=9` (the 20
                # arrondissements, 22/22 tagged, 100% over the floor) and New York
                # `admin=7` (the boroughs). Both are flagged sparse only because the
                # `admin=8` above them is the surrounding COMMUNES and outnumbers them —
                # the two levels do not cover the same ground, so the unit-count
                # comparison is meaningless there.
                of = levels[key].get("overFloor")
                vouched = of is not None and of >= KEEP_FRAC
                if levels[key]["verdict"] == "keep" and not vouched:
                    levels[key]["verdict"] = "sparse"
            seen_max = max(seen_max, n)


def verdict(n_units, over_floor, est_over_floor, has_est):
    """THE ESTIMATE DECIDES WHERE IT CAN. Reported population decides only where it
    cannot. This reverses the old precedence (changed 2026-08-29) but does not discard
    reported data, and both halves were paid for.

    WHY REPORTED POPULATION LOST ITS PRIORITY. The old rule was "reported populations beat
    the estimate wherever they exist", which sounds obviously right — a measurement beats
    a division sum. It is wrong because OSM's `population` tags are not a SAMPLE of the
    level, they are whatever somebody happened to tag, and that skews small.

    Aleppo is the case. Its `place=suburb` has 46 units for a city of 2,003,671, an
    average of 58,931. Twelve of the 46 carry a population tag, median 4,583, and on that
    dozen the level was rejected — so Aleppo kept NOTHING and vanished from the browser.
    But if those suburbs really held 4,583 each, all 46 would house 211,000 people, a
    tenth of Aleppo, leaving 1.8 million living nowhere. The tagged twelve are the small
    ones, and a quarter of a level (MIN_POP_FRAC) was enough to outvote the arithmetic.

    WHY IT IS STILL HERE. Some levels the estimate CANNOT judge at all. The estimate
    divides the city's population by units inside the city, so a level made of the
    surrounding municipalities has no denominator — `MIN_CORE_FRAC` switches it off and
    `meanPop` is None. That is not an edge case, it is §4.4's deliberate design:

        Paris    admin=8   116 units,  3% core, 115 tagged, median 31,392
        New York admin=8    54 units,  0% core,  52 tagged, median 19,519
        Athens   admin=7    41 units,  2% core,  41 tagged, median 61,308

    These are the Petite Couronne, Hoboken and Jersey City, the demoi around Athens — kept
    on purpose, and among the best levels in the corpus. Dropping reported population
    outright deleted all three. Where the estimate is silent, near-complete tagging is the
    only evidence there is, and it is good evidence.

    The two failure modes are opposites and the rule now matches that: Aleppo's problem was
    a THIN, biased sample beating good arithmetic; Paris's is that there is no arithmetic
    to be had. So the estimate goes first, and reported data speaks when it is silent.
    """
    if n_units < MIN_UNITS:
        return "too-few"
    if has_est:
        return "keep" if est_over_floor else "unknown"
    if over_floor is not None:
        return "keep" if over_floor >= KEEP_FRAC else "below-floor"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--only")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    ov = overrides()

    result = {}
    for city in cities(only):
        s = survey_city(city)
        if s is None:
            continue
        keep = [k for k, v in s["levels"].items() if v["verdict"] in ("keep", "keep-est")]
        # A hand override replaces the whole keep-list for that city; the rule is a
        # good default, not an authority, and some cities will just need telling.
        if ov.get(city["qid"], {}).get("keep"):
            keep = ov[city["qid"]]["keep"]
            s["keepSource"] = "override"
        s["keep"] = keep
        path = SURVEY / f"{city['qid']}.json"
        elements = json.loads(path.read_text("utf-8")).get("elements", [])
        shape, _ = trustworthy(city, city_shape(city["qid"]), elements)
        s["donors"] = find_donors(city, shape, elements, set(keep)) if keep else []
        result[city["qid"]] = s

    if args.report:
        report(result)
        return

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    kept = sum(1 for v in result.values() if v["keep"])
    print(f"wrote {OUT} — {len(result)} cities surveyed, {kept} with at least one kept level")


def report(result):
    for qid, s in result.items():
        rej = s.get("boundaryReject")
        state = s.get("boundaryState")
        if s["hasBoundary"]:
            bnd = "boundary"
        elif rej:
            bnd = (f"NO BOUNDARY (rejected {rej['picked']}, held "
                   f"{rej['covered']:.0%} of units, needs {rej['floor']:.0%})")
        elif state == "not-fetched":
            bnd = "boundary NOT FETCHED YET"
        else:
            bnd = "NO BOUNDARY (none found)"
        if not s["hasBoundary"]:
            bnd += (f"; estimate over a {s['areaRadiusKm']} km area-disc"
                    if s.get("areaRadiusKm") else "; estimate over the whole radius")
        print(f"\n{s['city']}  ({qid})   city pop {s['pop']:,}   floor {s['floor']:,}+")
        print(f"  {bnd}, {s['nOutside']} named units beyond the radius")
        print(
            f"  {'level':<18}{'units':>6}{'w/pop':>7}{'poly%':>7}{'wiki%':>7}"
            f"{'core%':>7}{'medPop':>9}{'estPop':>9}{'over':>8}  verdict"
        )
        for key, v in sorted(s["levels"].items(), key=lambda kv: -kv[1]["n"]):
            est = f"{v['meanPop']:,}" if v["meanPop"] is not None else "-"
            med = f"{v['medianPop']:,}" if v["medianPop"] is not None else "-"
            ib = f"{v['overFloor']:.0%}" if v["overFloor"] is not None else "-"
            cf = f"{v['coreFrac']:.0%}" if v["coreFrac"] is not None else "-"
            mark = "***" if key in s["keep"] else ""
            print(
                f"  {key:<18}{v['n']:>6}{v['nPop']:>7}{v['polyFrac']:>7.0%}"
                f"{v['wdFrac']:>7.0%}{cf:>7}{med:>9}{est:>9}{ib:>8}  {v['verdict']} {mark}"
            )
        print(f"  KEEP: {', '.join(s['keep']) or '(nothing)'}")


if __name__ == "__main__":
    main()
