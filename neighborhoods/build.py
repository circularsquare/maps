"""
Stage 3: survey + levels -> data/base.json, the file the browser draws.

One record per named unit in a kept level. Everything the browser needs and nothing else,
with short keys because there are ~17,000 of them and this file is fetched over the wire.

WHY THIS RUNS BEFORE THE GEOMETRY PASS. The survey requested `out tags center`, so every
unit already has a point — a real centroid for ways and relations, the node itself
otherwise. That is enough to put all 17,000 on a map and see whether the level picks are
sane, which is the question actually being asked right now. Shapes are a later upgrade to
the same records (`poly` says which units will get one), not a prerequisite for looking.

Usage:
    python build.py
    python build.py --stats
"""

import argparse
import collections
import json
import math
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import osmgeom
import pick_levels as pl
from fetch_osm import cities

HERE = pathlib.Path(__file__).parent
OUT = HERE / "data" / "base.json"
UNITS_OUT = HERE / "data" / "units"
GEOM_IN = HERE / "cache" / "geom"
GEOM_OUT = HERE / "data" / "geom"
CITY_OUT = HERE / "data" / "city_shapes.json"

# Rendering simplification, ~11 m at the equator. Shapes are drawn at city zoom where
# 11 m is well under a pixel, and it takes a long tail of surveyed vertices out of the
# payload. It is a DISPLAY simplification only — anything measuring these polygons must
# read cache/geom/, not data/geom/.
SIMPLIFY_DEG = 0.0001

# The city outlines are drawn only below zoom 9 (spec 6a.6), where one pixel is about
# 0.003°, so a 0.003° tolerance is at most a pixel of error at the very top of the range
# and far less everywhere below it. It takes Tokyo's prefecture from 40,000 vertices to a
# few hundred; without it, one file for 181 cities would be larger than base.json.
CITY_SIMPLIFY_DEG = 0.003

# How many sides stand in for the circle of the search radius. 64 is smooth at every zoom
# this layer is drawn at.
DISC_SIDES = 64

WOF_IN = HERE / "data" / "wof_shapes.json"
EXT_IN = HERE / "data" / "external_shapes.json"
P31_IN = HERE / "data" / "p31.json"
POP_IN = HERE / "data" / "pop.json"
TIER_MAP = HERE / "data" / "tier_map.json"

# A type must be carried by this share of a level's units before it can name the level.
MIN_TYPE_SHARE = 0.30

# Before one `admin_level` is treated as holding two different kinds of thing, each kind
# must be this much of the level and this many units. Deliberately blunt: a level with a
# handful of oddly-typed units is a tagging wobble, not a mixed tier.
MIN_SPLIT_SHARE = 0.20
MIN_SPLIT_UNITS = 5

# A neighbourhood tier below this share of units-with-a-Wikipedia-article starts hidden,
# unless it is the only division the city has. Calibrated against Anita's own calls: it
# passes London (89%), New York (95% and 71%) and Paris (50%), and fails Beijing (0%),
# Shanghai (0%), Tokyo (22%), Seoul (13%) and Chicago (18%).
MIN_KNOWN_FRAC = 0.50

# A Who's On First polygon overlapping a corpus outline by at least this much IS that
# outline arriving from a second direction, not a new shape. 1,000 of WOF's 2,690 matches
# are duplicates at this threshold, and they cluster by provenance: where WOF's geometry
# came from a municipal open-data release (Madrid, Berlin, Barcelona, Singapore) OSM
# imported the same official file, so WOF is the same source twice — 98% duplicate. The
# genuinely new shapes are Quattroshapes/Zetashapes/Mapzen-era, 97-99% new.
WOF_DUP_IOU = 0.9


_alnum = re.compile(r"[^0-9a-z]+")


def squash(s):
    """Normalise hard, for comparing a place name to an article title: casefold, strip
    accents, drop every non-alphanumeric character. 'Tel-Aviv' and 'Tel Aviv' become one
    string."""
    return _alnum.sub("", pl.norm_name(s))


def title_differs(title, names):
    """Does the article title differ from the OSM name? INFORMATIVE ONLY — see below.

    THIS WAS BUILT AS A MIS-TAGGED-QID DETECTOR AND IT DOES NOT WORK. Recorded so nobody
    rebuilds it. Three attempts, each fixing the previous one's noise:

      1. Title vs `name` — flagged nearly every Japanese and Korean unit, because
         横浜市 vs "Yokohama" is a translation, not an error.
      2. Plus `name:en`, minus disambiguators ("Kawasaki, Kanagawa") — still 1,035 flags.
         Article titles carry descriptive suffixes the OSM name never has: Gangnam ->
         "Gangnam District", Brøndby -> "Brøndby Municipality", 1st Arrondissement ->
         "1st arrondissement of Paris".
      3. Substring containment either way — 910 flags, of which perhaps a handful are
         real. Survivors are Swedish genitives (Sundbybergs kommun / "Sundbyberg
         Municipality"), suffix swaps (강남구 / "Gangnam District"), translations (Stare
         Miasto / "Old Town, Warsaw") and transliterations (Περιστέρι / "Peristeri").

    It cannot be tightened into working. The one confirmed error, "Bronxdale" carrying
    The Bronx's QID, SHARES a substring with its title — so any rule strict enough to
    silence the noise also loses the signal. Name similarity cannot distinguish "wrong
    article" from "differently-worded article".

    So the title is carried as neutral information — useful in a card, since it says what
    Wikipedia calls this place — and is NOT rendered as a warning. A warning that fires
    910 times to catch two errors trains the reader to ignore it.
    """
    t = squash(title)
    if not t:
        return False
    return not any(n and (n in t or t in n) for n in names)


def load_wiki():
    """QID -> (sitelinks, views, article title).

    Only three fields of `wiki.json` reach the browser. The file also carries per-language
    view breakdowns and every article title, which is most of its 3 MB and none of its use
    here — `byLang` belongs to whoever is auditing the ranking, not to the map.

    `views` is 0 while the views pass is still running, so the UI must fall back to
    sitelinks rather than treating 0 as "nobody reads this".
    """
    path = pl.DATA / "wiki.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text("utf-8"))
    # SUPPLEMENT FROM THE LEDGER. `wiki.json` is a derived merge and the ~7 h views pass
    # owns it for the whole of its run, so a level-rule change that admits new units
    # leaves them with no sitelinks for hours — New York's boroughs read "20% known" when
    # Manhattan and Brooklyn simply had not been fetched. The ledger is the source of
    # truth and is append-only, so reading it back fills the gap with no coordination.
    ledger = pl.HERE / "cache" / "wiki" / "sitelinks.jsonl"
    extra = {}
    if ledger.exists():
        for line in ledger.read_text("utf-8").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("q") and rec["q"] not in raw:
                extra[rec["q"]] = len(rec.get("s") or {})
    raw = raw.get("units", raw)
    out = {}
    for qid, r in raw.items():
        if not isinstance(r, dict):
            continue
        title = r.get("title") or {}
        # Prefer the English title purely because it is the one most likely to be
        # readable to whoever is auditing; any title is better than none.
        t = title.get("en") or next(iter(title.values()), None) if title else None
        out[qid] = (r.get("sitelinks") or 0, r.get("views") or 0, t)
    for qid, n in extra.items():
        out[qid] = (n, 0, None)
    return out


def load_pop():
    """QID -> (population, year or None) from Wikidata P1082. See fetch_wiki.py.

    KEPT SEPARATE FROM `p`, which is the OSM `population` tag, rather than filling it in.
    Two figures with different provenance and different dates must not become one field
    the UI cannot tell apart — the same rule §1.6 applies to borrowed geometry, where the
    flag is `e` precisely so a borrowed shape never renders as a surveyed one.

    Where both exist they agree almost exactly: over 949 units the median P1082/OSM ratio
    is 1.000 and 99% are within 2×, which is what you would expect from two copies of the
    same census rather than two estimates of the same place. That agreement is the reason
    to trust the 3,235 units where only P1082 has a figure.
    """
    if not POP_IN.exists():
        return {}
    raw = json.loads(POP_IN.read_text("utf-8")).get("units", {})
    return {q: (r["p"], r.get("y")) for q, r in raw.items() if r.get("p") is not None}


def load_external_units():
    """-> (units by city QID, layer record by city QID). Empty if the stage has not run.

    Units that exist because a city's own publisher says so and OSM does not — Cairo's
    Abdeen and Ezbakeya, Copenhagen's Bellahøj and Humleby (§1.7). They arrive fully
    formed from `fetch_external.py`, which is also where the §3 level rule was applied to
    them, so all that happens here is admitting the layers that passed it.
    """
    if not EXT_IN.exists():
        return {}, {}
    ext = json.loads(EXT_IN.read_text("utf-8"))
    by_city = collections.defaultdict(list)
    for u in ext.get("units") or []:
        by_city[u["c"]].append(u)
    return by_city, ext.get("layers") or {}


def build():
    levels = json.loads((pl.DATA / "levels.json").read_text("utf-8"))
    wiki = load_wiki()
    wpop = load_pop()
    ext_units, ext_layers = load_external_units()
    out_cities, units = {}, []

    for city in cities():
        s = levels.get(city["qid"])
        if not s or not s["keep"]:
            continue
        path = pl.SURVEY / f"{city['qid']}.json"
        if not path.exists():
            continue
        keep = set(s["keep"])
        elements = json.loads(path.read_text("utf-8")).get("elements", [])
        # Same trust check as pick_levels, so `core` here means what `coreFrac` meant
        # there. Calling city_shape directly would reintroduce the Lagos/Nairobi
        # too-small-boundary bug in the browser only, which is the worst place for it.
        shape, _ = pl.trustworthy(city, pl.city_shape(city["qid"]), elements)

        n_before = len(units)
        for el in elements:
            tags = el.get("tags") or {}
            name = tags.get("name")
            if not name:
                continue
            if pl.excluded_name(city["qid"], name):
                continue
            key = pl.level_key(tags, el["type"])
            if key not in keep:
                continue
            inside, in_core = pl.locate(city, shape, el)
            if not inside:
                continue
            c = osmgeom.centroid_of(el)
            if c is None:
                continue
            units.append(
                {
                    "c": city["qid"],
                    # OSM type-initial + id. The join key between a dot in base.json and
                    # its outline in data/geom/<qid>.json, and stable across refetches.
                    "i": f"{el['type'][0]}{el['id']}",
                    "k": key,
                    "n": name,
                    # `name:en` is worth carrying: half these are in scripts the reader
                    # may not have, and the browser should be able to show both.
                    "en": tags.get("name:en"),
                    "x": round(c[0], 5),
                    "y": round(c[1], 5),
                    "p": pl.parse_pop(tags.get("population")),
                    "q": tags.get("wikidata"),
                    # 1 where a real polygon exists in OSM and the geometry pass will
                    # fetch one; 0 where the unit is a bare node and will stay a pin.
                    "poly": 1 if el["type"] in ("way", "relation") else 0,
                    "core": None if in_core is None else int(in_core),
                }
            )
            # SELF-REFERENCE: this unit is the city itself. Paris's `admin=8` commune,
            # Los Angeles at admin=8, Sydney's `admin=9` CBD suburb. Found because the
            # wiki ranking put them at the very top — Paris has 289 sitelinks, so it would
            # head any global quiz deck with "name the city for the neighbourhood Paris".
            #
            # MARKED, NOT DROPPED, and the distinction matters. The Paris commune is a
            # genuine member of its level alongside Boulogne-Billancourt, and removing it
            # would make the level incomplete — which §0b says the browser must not be.
            # The quiz is the layer that filters (§7). Name matching alone would also be
            # wrong to drop on: "Sydney" and "Melbourne" really are those cities' CBD
            # suburbs.
            if tags.get("wikidata") == city["qid"] or pl.norm_name(name) == pl.norm_name(
                city["name"]
            ):
                units[-1]["self"] = 1

            # Wikidata's population, on every unit that has one — including units that
            # already carry an OSM tag, so the two stay comparable and a disagreement is
            # visible rather than resolved silently in favour of whichever was written
            # last. `wy` is the P585 year and is not optional to display: 2010 and 2025
            # figures sit side by side in this corpus.
            wp = wpop.get(tags.get("wikidata"))
            if wp:
                units[-1]["wp"] = wp[0]
                if wp[1]:
                    units[-1]["wy"] = wp[1]

            w = wiki.get(tags.get("wikidata"))
            if w:
                sl, views, title = w
                units[-1]["sl"] = sl
                if views:
                    units[-1]["v"] = views
                # A title matching NEITHER the local name nor `name:en` is how a mis-tagged
                # QID shows itself — Singapore's "Peng Siang" points at the article
                # "Common year", NYC's "Bronxdale" carries The Bronx's QID.
                #
                # Comparing against `name` alone flagged almost every Japanese and Korean
                # unit, because the title we hold is English and the OSM name is not:
                # 横浜市 vs "Yokohama" is a translation, not an error. `name:en` is what
                # makes the two comparable, so a unit without one is never flagged —
                # silence beats a wall of false positives on a signal meant to be rare.
                names = {squash(name)}
                if tags.get("name:en"):
                    names.add(squash(tags["name:en"]))
                if title and title_differs(title, names):
                    units[-1]["wt"] = title

        # ------------------------------------ units from the city's own source (§1.7)
        #
        # Appended AFTER the survey loop and never mixed into it, because everything
        # above reads OSM tags these records do not have: no `wikidata`, so no notability
        # and no P31; no `population`; no `name:en` unless the source published one. They
        # carry `src`, which is what tells the rest of the pipeline — and the browser —
        # that this record is not an OSM object.
        #
        # The level rule already ran, in fetch_external, over the WHOLE source layer
        # rather than the promoted remainder. A hand override still wins: it replaces the
        # keep-list outright (§3), and an external layer is a level like any other, so it
        # is admitted only if the override names it.
        layer = ext_layers.get(city["qid"])
        if layer:
            ov = pl.overrides().get(city["qid"]) or {}
            wanted = (layer["key"] in ov["keep"]) if ov.get("keep") \
                else layer["verdict"] in ("keep", "keep-est")
            if wanted:
                keep.add(layer["key"])
                for u in ext_units.get(city["qid"], ()):
                    if pl.excluded_name(city["qid"], u["n"]):
                        continue
                    # THE SAME KEYS AS AN OSM UNIT, with `None` where the source has
                    # nothing, rather than a smaller record. Consumers index `u["q"]` and
                    # `u["p"]` directly — `--stats` does — and a record missing them is a
                    # KeyError waiting for whoever adds the next reader.
                    units.append({
                        "c": u["c"], "i": u["i"], "k": u["k"], "n": u["n"],
                        "en": u.get("en"), "x": u["x"], "y": u["y"],
                        "p": None, "q": None,
                        # `poly` means "OSM has a polygon and the geometry pass will
                        # fetch one" — false here by construction. The outline arrives
                        # instead as `e`, the external-shape flag, so nothing tells a
                        # reader an OSM survey stands behind a shape that did not come
                        # from one.
                        "poly": 0,
                        "core": u.get("core"),
                        # The record itself is external, not just its geometry. `e` says
                        # where the shape came from and is set for OSM units too; `src`
                        # says this unit exists ONLY because that source drew it, which
                        # is a different and stronger claim the browser must be able to
                        # make — and it names the source for the attribution the CC BY
                        # licences require.
                        "src": u["src"],
                    })

        out_cities[city["qid"]] = {
            "name": city["name"],
            "country": city["country"],
            "lat": city["lat"],
            "lon": city["lon"],
            "pop": city["pop"],
            "radiusKm": city["radiusKm"],
            "keep": sorted(keep),
            "hasBoundary": s["hasBoundary"],
            # ok | rejected | not-fetched | empty. Distinguishes "we looked and threw it
            # away" from "the boundary pass has not reached this city", which look
            # identical in hasBoundary and must not read the same to a person.
            "boundaryState": s.get("boundaryState"),
            # Present only when a boundary WAS found and then discarded as too small
            # (spec 4.4a). The browser needs it to explain the note it shows.
            "boundaryReject": s.get("boundaryReject"),
            # Radius of the area-disc standing in for the missing boundary (spec 3.1).
            "areaRadiusKm": s.get("areaRadiusKm"),
            "n": len(units) - n_before,
            # Per-level stats travel with the city so the UI can explain a level rather
            # than just listing it — how big its units are, how many have shapes.
            "levels": {k: v for k, v in s["levels"].items() if k in keep},
        }
        # An external layer has no row in levels.json — pick_levels only ever sees the
        # OSM survey — so its stats are filled in here in the same shape, or the level
        # panel would list a level it can say nothing about. `n` is the promoted count
        # and `nLayer` the whole source layer the verdict was actually judged on; they
        # differ by the polygons OSM already names, and showing only the first would
        # misreport what the estimate divided by.
        if layer and layer["key"] in keep:
            out_cities[city["qid"]]["levels"][layer["key"]] = {
                "n": layer["n"], "nLayer": layer["nLayer"], "nPop": 0,
                "polyFrac": 1.0, "wdFrac": 0.0,
                "coreFrac": None if shape is None else round(layer["core"] / max(layer["nLayer"], 1), 3),
                "medianPop": None, "meanPop": layer["meanPop"], "overFloor": None,
                "verdict": layer["verdict"], "external": 1,
                "source": layer["source"], "licence": layer["licence"],
                "attribution": layer["attribution"],
            }

    return {"cities": out_cities, "units": units}


def write_split(data):
    """base.json becomes an INDEX; each city's units get their own file.

    WHY. The map draws one city at a time — `visibleUnits()` filters on `u.c === city` and
    returns nothing at all before a city is chosen — but every unit of every city was
    being shipped before the first dot could appear. At 287 cities that was a **13.6 MB
    blocking download** (2.2 MB gzipped) of which one city's worth, a median 27 KB, was
    ever on screen.

    Splitting takes the upfront cost to ~350 KB and defers the rest to the click that
    needs it. This is the same shape the geometry files already had; it is that pattern
    applied to the layer that had not got it yet.

    `levelKeys` is the one thing that HAS to move into the index. Colours are assigned by
    level key in sorted order so that the same kind of division is the same colour in
    every city (see PALETTES in index.html), and that order was previously derived by
    scanning every unit. Without it here, colours would depend on which city you opened
    first — the exact bug the sorted warm-up existed to prevent.
    """
    by_city = collections.defaultdict(list)
    for u in data["units"]:
        by_city[u["c"]].append(u)

    UNITS_OUT.mkdir(parents=True, exist_ok=True)
    stale = {p.stem for p in UNITS_OUT.glob("*.json")} - set(by_city)
    for qid in stale:
        (UNITS_OUT / f"{qid}.json").unlink()

    total = 0
    for qid, us in by_city.items():
        p = UNITS_OUT / f"{qid}.json"
        p.write_text(json.dumps(us, ensure_ascii=False, separators=(",", ":")), "utf-8")
        total += p.stat().st_size

    index = {
        "cities": data["cities"],
        "levelKeys": sorted({u["k"] for u in data["units"]}),
    }
    OUT.write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), "utf-8")
    return OUT.stat().st_size / 1024, total / 1024, len(by_city)


def mark_dupes(data):
    """Link units that are the same real place recorded more than once.

    Cities divide themselves once, but OSM records the division several times: Seoul's
    *dong* exist as `place=quarter` nodes, `admin=8` polygons and `admin=10` polygons, so
    the corpus holds three records of 명동. Measured, 7,839 QID-bearing units carry only
    ~6,000 distinct QIDs.

    Two independent groupings, strongest first:

      - **Same wikidata QID in the same city.** Unambiguous — a QID names one entity.
      - **Same name, one containing the other's point.** The evidence that made borrowing
        safe (§1.3a), reused. Name alone is never enough.

    MARKED, NOT MERGED. `dupOf` points at the canonical unit; the duplicate keeps its own
    record. The browser must show whole levels (§0b), so deleting a member would leave a
    level with a hole in it — the level is the thing being judged, and a level missing the
    units that happen to be mapped twice is not that city's division any more. The quiz
    collapses on `dupOf`; the browser does not.

    Canonical = the one with its own polygon, then the one from the larger level. Both
    tie-breaks are deterministic, which matters because this runs on every build and a
    canonical that moved between builds would churn the quiz deck for no reason.
    """
    by_city = collections.defaultdict(list)
    for u in data["units"]:
        by_city[u["c"]].append(u)

    bad_qids = []
    n_groups = n_dupes = 0
    for qid, units in by_city.items():
        level_size = collections.Counter(u["k"] for u in units)
        index = {u["i"]: u for u in units}

        # UNION-FIND. The two groupings overlap — a unit can share a QID with one record
        # and a name+containment with another — and first-wins assignment would then make
        # the canonical depend on dict iteration order, so a quiz collapsing on `dupOf`
        # could see A->B and B->C at once. Merging into connected components makes the
        # answer order-independent and gives exactly one canonical per real place.
        parent = {u["i"]: u["i"] for u in units}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # A QID names ONE entity, so sharing one is normally proof of identity. But OSM
        # sometimes tags a whole district's children with the district's own QID, and
        # then the union is catastrophic rather than merely wrong: Mexico City's
        # Q6091026 sits on 49 differently-named colonias — Santa Isabel Tola, San Pedro
        # Zacatenco, Coltongo — and merged all 50 records into a single "place".
        #
        # So a QID whose bearers disagree about their NAME is not evidence of anything.
        # We cannot tell which of them is mistagged, so we decline to union at all,
        # which fails toward leaving records separate — the safe direction, since a bad
        # merge deletes distinct places from the corpus and a missed one only leaves a
        # duplicate dot.
        #
        # The cut is 3 distinct names, measured: 9,363 QIDs carry one name and 503 carry
        # two (the two-name band is legitimate — Tokyo's 大字栄和 and 栄和 are the same
        # place with and without the 大字 prefix). Only 6 QIDs corpus-wide reach three,
        # affecting 69 units, and 5 of the 6 are unambiguously bad tags. Same class of
        # error as the QID mismatches the notability pass reports (§1.4).
        MAX_QID_NAMES = 3
        by_qid = collections.defaultdict(list)
        for u in units:
            if u.get("q"):
                by_qid[u["q"]].append(u)
        for q, group in by_qid.items():
            names = {pl.norm_name(u["n"]) for u in group}
            if len(names) >= MAX_QID_NAMES:
                bad_qids.append((q, len(group), sorted(names)[:4]))
                continue
            for u in group[1:]:
                union(group[0]["i"], u["i"])

        # name+containment pairs already found while attaching geometry
        for u in units:
            if u.get("dup") and u["dup"] in parent:
                union(u["i"], u["dup"])

        comps = collections.defaultdict(list)
        for i in parent:
            comps[find(i)].append(i)

        for members in comps.values():
            if len(members) < 2:
                continue
            n_groups += 1
            best = max(
                (index[i] for i in members),
                key=lambda u: (u.get("poly", 0), level_size[u["k"]], u["i"]),
            )
            for i in members:
                if i != best["i"]:
                    index[i]["dupOf"] = best["i"]
                    n_dupes += 1
    return n_groups, n_dupes, bad_qids


def assign_tiers(data, wiki_sl):
    """Group each city's levels into the four tiers a reader actually thinks in.

    `admin=9` and `place=suburb` are OSM tagging conventions, not statements about a
    place, so neither can be read as "official" or "informal" on its own — Berlin's
    twelve Bezirke are `place=borough`, Amsterdam's stadsdelen `place=suburb`, while
    Tokyo's wards are `admin=7`. Classifying by tag family mis-tiered about a third of
    cities. Wikidata's P31 says what each unit actually IS, and `data/tier_map.json`
    maps those type QIDs to one of three classes by hand.

    The class is mapped, the tier NUMBER is derived. A flat type -> tier table cannot
    work: Q123705 "neighborhood" is Amsterdam's `place=quarter`, sitting under the
    stadsdelen, and also Los Angeles's `place=neighbourhood`, which is the only tier LA
    has. So `official` levels are ordered coarse-to-fine per city and numbered here.

    Two rules earn their place, both found by being wrong first:

    - **Specific beats popular.** The obvious rule — give a level its commonest type —
      picks the generic one. London's boroughs carry both `London borough` (31) and
      `unparished area` (33), and the plurality winner was the one that means "this part
      of England has no civil parish". Stockholm's `city district of Stockholm` (11) lost
      the same way to a generic multi-country `quarter` (18). Ranking by how few CITIES a
      type appears in fixes both, because a type that exists for one city is describing
      that city.
    - **Levels sharing a type are one tier.** New York's `admin=7` and `place=suburb` are
      both "borough of New York City"; emitting division-1 and division-2 for them would
      re-create on screen exactly the duplication the tiers exist to remove.

    A level whose units carry too few QIDs gets no tier and keeps its raw key. That is
    77 levels over 43 cities and it is honest — guessing a tier from `admin_level`
    arithmetic is the cross-country comparison the whole design refuses to make.
    """
    if not (P31_IN.exists() and TIER_MAP.exists()):
        return 0, 0, 0, 0
    p31 = json.loads(P31_IN.read_text("utf-8"))
    tmap = json.loads(TIER_MAP.read_text("utf-8"))
    types, labels, cmap = p31["types"], p31["labels"], tmap["map"]
    by_city_override = tmap.get("byCity") or {}

    # AN EXTERNAL LAYER IS TYPED BY HAND, because P31 cannot reach it: its units carry no
    # `wikidata` tag — there is no OSM object to have carried one — so every rule in this
    # function that reads a QID reads nothing, and the layer would fall through to
    # `unclassified` and be hidden as a `place=*`-shaped holding pen.
    #
    # This is the same hand mapping §9.1 already relies on, keyed on the source instead of
    # a type QID: `tier_map.json` says what Q123705 is, and `fetch_external.SOURCES` says
    # what DAWA's bydel layer is. The tier NUMBER is still derived, so an external
    # `official` layer is ordered against the city's OSM divisions coarse-to-fine like any
    # other. A synthetic type token carries it into the machinery below.
    ext_type = {}
    for qid, layer in (load_external_units()[1]).items():
        t = f"_ext:{qid}:{layer['key']}"
        ext_type[(qid, layer["key"])] = t
        cmap.setdefault(t, {"class": layer["class"]})
        labels.setdefault(t, layer["label"])

    # How many cities each type appears in at all. Low = specific = trusted.
    breadth = collections.defaultdict(set)
    for u in data["units"]:
        for t in types.get(u.get("q") or "", []):
            breadth[t].add(u["c"])

    counts = collections.defaultdict(collections.Counter)
    size = collections.Counter()
    distinct = collections.defaultdict(set)
    units_by_level = collections.defaultdict(list)
    live = {u["i"] for u in data["units"]}
    for u in data["units"]:
        k = (u["c"], u["k"])
        size[k] += 1
        distinct[k].add(u["dupOf"] if u.get("dupOf") in live else u["i"])
        units_by_level[k].append(u)
        for t in set(types.get(u.get("q") or "", [])):
            counts[k][t] += 1

    def unit_type(u, classify):
        """The most specific classifiable type this ONE unit carries."""
        cand = [t for t in types.get(u.get("q") or "", [])
                if classify(t) not in (None, "skip")]
        return min(cand, key=lambda t: len(breadth[t])) if cand else None

    def pset(us):
        """The distinct places these units resolve to, via `dupOf`."""
        return {u["dupOf"] if u.get("dupOf") in live else u["i"] for u in us}

    def places(us):
        """Distinct place count — never a raw count, which double-reports a division
        recorded twice (Sydney's 400 suburbs read as 804)."""
        return len(pset(us))

    def group_same_division(ps):
        """Group parts that are the same division recorded more than once.

        Sharing a P31 type is NOT enough on its own. New York's 19 macro-neighbourhoods
        (Harlem, Midtown) and its 325 finer ones are both Wikidata "neighborhood" but are
        nested, not duplicated, and collapsing them would lose a whole layer. Warsaw's 142
        `admin=10` and 147 `place=quarter` are the same MSI zones twice and must collapse.
        Only the dedup components (§4.6) can tell those apart, so the type must match AND
        the places must actually coincide.
        """
        out = []
        for p in ps:
            for g in out:
                if g[0]["t"] != p["t"]:
                    continue
                a = set().union(*(pset(x["u"]) for x in g))
                b = pset(p["u"])
                shared = len(a & b)
                if a and b and max(shared / len(a), shared / len(b)) >= 0.60:
                    g.append(p)
                    break
            else:
                out.append([p])
        return out

    n_typed = n_untyped = n_inferred = n_split = 0
    for qid, city in data["cities"].items():
        over = by_city_override.get(city["name"]) or {}

        def classify(t):
            return over.get(t) or (cmap.get(t) or {}).get("class")

        chosen, skipped = {}, {}
        for k in city["keep"]:
            cand = [
                (t, n) for t, n in counts[(qid, k)].items()
                if n >= MIN_TYPE_SHARE * size[(qid, k)]
                and classify(t) not in (None, "skip")
            ]
            cand.sort(key=lambda tn: (len(breadth[tn[0]]), -tn[1]))
            # The declared type wins outright for an external layer — there is nothing
            # for it to lose to, since its units carry no QIDs to be counted above.
            chosen[k] = ext_type.get((qid, k)) or (cand[0][0] if cand else None)
            # UNTYPED IS NOT THE SAME AS JUDGED-NOT-A-DIVISION. A level whose units carry
            # a type that `tier_map.json` marks `skip` has been explicitly ruled out —
            # New York's `admin=6` counties are coextensive with its boroughs and were
            # skipped on purpose — and must not later be rescued as "an admin level we
            # could not type", which is a statement about missing evidence, not a verdict.
            skipped[k] = not cand and any(
                classify(t) == "skip"
                for t, n in counts[(qid, k)].items()
                if n >= MIN_TYPE_SHARE * size[(qid, k)]
            )

        # INHERIT a type from a same-division twin. A level can be untyped only because
        # its units are thin on QIDs, not because it is a different division: Sydney's
        # `place=suburb` is 8% wikidata but is the same 400 places as its `admin=9`,
        # which is 99%. Where the dedup components say two levels are the same division,
        # the typed one names both — otherwise they land in different tiers and the
        # duplication the tiers exist to remove reappears.
        for k in city["keep"]:
            if chosen[k] or not distinct[(qid, k)]:
                continue
            best, best_frac = None, 0.0
            for other in city["keep"]:
                if not chosen[other]:
                    continue
                shared = distinct[(qid, k)] & distinct[(qid, other)]
                frac = len(shared) / len(distinct[(qid, k)])
                if frac > best_frac:
                    best, best_frac = other, frac
            if best and best_frac >= 0.60:
                chosen[k] = chosen[best]

        # POSITIONAL FALLBACK for what is still untyped. A `place=*` level with almost no
        # QIDs, sitting finer than every official division the city has, is a vernacular
        # neighbourhood layer — Sao Paulo's 1,030 bairros, Tokyo's 1,569 `place=quarter`,
        # London's 782 `place=neighbourhood`, none of which Wikidata can name. Inferred
        # from position, so it is recorded as `informal` only, never given a label.
        #
        # `place=*` ONLY, and that restriction is the whole safety of the rule: an
        # `admin_level` tag is at least a claim to administrative standing, and Paris's
        # `admin=10` quartiers administratifs really are official. A `place=*` tag makes
        # no such claim. Levels COARSER than the finest official one are left alone too —
        # Sydney's 7-unit `place=borough` is not a neighbourhood layer.
        finest = max(
            (len(distinct[(qid, k)]) for k in city["keep"]
             if chosen[k] and classify(chosen[k]) == "official"),
            default=None,
        )
        for k in city["keep"]:
            if chosen[k] or not k.startswith("place="):
                continue
            if finest is None or len(distinct[(qid, k)]) > finest:
                chosen[k] = "_positional"

        # SPLIT A LEVEL THAT HOLDS TWO KINDS OF THING. `admin_level` is a national
        # numbering, not a semantic one, and several countries put two tiers on one
        # number: Tokyo's `admin=7` is the 23 special wards AND the Tama and Saitama
        # municipalities, because a special ward ranks with a whole city; Seoul's
        # `admin=6` is its 25 gu plus Gyeonggi's si. One type per level then names the
        # level after whichever group is larger and buries the other — Tokyo's wards were
        # losing to the cities and surfacing only through a 31-unit `place=suburb`.
        #
        # NEITHER MORE SAMPLING NOR `core` CAN FIND THIS: the mixing is categorical, not
        # geographic. Komae-shi sits inside Tokyo Metropolis, so it is a core unit exactly
        # like Shibuya-ku. Only P31 separates them.
        #
        # This is presentation, not membership. Every unit still appears; it is only
        # grouped under a truer heading, so the level rule's "all of a level or none"
        # (§0b) is untouched.
        parts = []
        for k in city["keep"]:
            us = units_by_level[(qid, k)]
            groups = collections.defaultdict(list)
            for u in us:
                groups[classify(unit_type(u, classify)) or "?"].append(u)
            real = [c for c, g in groups.items()
                    if c not in ("?", "skip")
                    and len(g) >= MIN_SPLIT_UNITS
                    and len(g) >= MIN_SPLIT_SHARE * len(us)]
            if len(real) < 2:
                parts.append({"k": k, "t": chosen[k], "u": us,
                              "skipped": skipped.get(k, False)})
                continue
            # Untyped and minority units join the largest group rather than being
            # stranded. Bounded error: a level only splits when its QID coverage is good
            # enough for two groups to clear the guards, so few units land here.
            real.sort(key=lambda c: -len(groups[c]))
            spare = [u for c, g in groups.items() if c not in real for u in g]
            for c in real:
                tally = collections.Counter(
                    t for u in groups[c] if (t := unit_type(u, classify)))
                parts.append({
                    "k": k, "split": 1,
                    "t": min(tally, key=lambda t: (len(breadth[t]), -tally[t])),
                    "u": groups[c] + (spare if c == real[0] else []),
                })
            n_split += 1

        buckets = collections.defaultdict(list)
        for p in parts:
            t = p["t"]
            p["n"] = places(p["u"])
            buckets["informal" if t == "_positional"
                    else classify(t) if t else None].append(p)
            if t == "_positional":
                n_inferred += 1
            elif t:
                n_typed += 1
            else:
                n_untyped += 1

        tiers = []

        def mk(tier_id, ps):
            t = ps[0]["t"]
            pos = t == "_positional"
            return {"id": tier_id,
                    "k": sorted({p["k"] for p in ps}),
                    "t": None if pos else t,
                    "label": None if pos else labels.get(t),
                    "n": places([u for p in ps for u in p["u"]]),
                    **({"inferred": 1} if pos else {}),
                    **({"split": 1} if any(p.get("split") for p in ps) else {}),
                    **({"skipped": 1} if any(p.get("skipped") for p in ps) else {}),
                    "_u": [u for p in ps for u in p["u"]]}

        for p in sorted(buckets["municipality"], key=lambda p: p["n"]):
            tiers.append(mk("municipality", [p]))
        official = group_same_division(sorted(buckets["official"], key=lambda p: p["n"]))
        for i, ps in enumerate(official):
            tiers.append(mk(f"division-{i + 1}", ps))
        for ps in group_same_division(sorted(buckets["informal"], key=lambda p: p["n"])):
            tiers.append(mk("neighbourhood", ps))
        for p in buckets[None]:
            tiers.append(mk("unclassified", [p]))

        # WHICH TIERS START VISIBLE. Showing every layer at once is overwhelming, and the
        # right answer is per city: Chinese `place=*` layers are spotty and tiny and want
        # hiding, while London, Paris, New York and Johannesburg all want their informal
        # layer on. The criterion is **recognisability** — the share of a tier's units
        # that have a Wikipedia article at all. It separates the cases cleanly:
        #
        #   London  "area of London"      89% known -> on     Beijing place=quarter  0% -> off
        #   NY      neighborhood      95% / 71%     -> on     Shanghai               0% -> off
        #   Paris   quartiers             50%       -> on     Tokyo   place=quarter 22% -> off
        #
        # Johannesburg is the case that forces the second clause: its informal layer is
        # only 13% known, but its ONLY other tier is a set of units called "Ward 34", so
        # something beats nothing and a city with no official division shows its informal
        # one regardless.
        #
        # Third and deeper divisions start off because they are usually below
        # neighbourhood scale — Tokyo's 1,505 chōchō are 97% known and still not what
        # anyone means by a Tokyo neighbourhood. Unclassified always starts off: it is a
        # holding pen, not a claim.
        has_div = any(t["id"].startswith("division") for t in tiers)
        # The "something beats nothing" rescue applies to ONE tier, not all of them.
        # Toronto has no official division and three informal tiers; rescuing every one
        # turned on a 4%-known layer of 24 units alongside the good one.
        # `known` IS ONLY MEASURABLE OVER UNITS THAT COULD HAVE CARRIED A QID. A unit
        # promoted from a city's own source (§1.7) has no OSM object behind it and so
        # can never have a `wikidata` tag; counting it as unknown measures the absence of
        # an OSM join, not the obscurity of the place. Cairo's kism are Abdeen, Ezbakeya
        # and El-Darb El-Ahmar — every one of them has a Wikipedia article.
        #
        # This is the trap §9.5 already records, in its second form: `known` read 20% for
        # New York's boroughs because the sitelinks pass had not reached them, and
        # calibrating on that would have been calibrating on a coverage gap. A tier with
        # nothing measurable scores `None` and the test that needs the number does not
        # get to fire.
        def known_frac(tier):
            us = [u for u in tier["_u"] if not u.get("src")]
            if not us:
                return None
            return sum(1 for u in us if wiki_sl.get(u.get("q") or "", 0) > 0) / len(us)

        informal = [t for t in tiers if t["id"] == "neighbourhood"]
        rescue = None
        if informal and not has_div:
            rescue = max(informal, key=lambda t: known_frac(t) or 0.0)
        for tier in tiers:
            known = known_frac(tier)
            tier["known"] = None if known is None else round(known, 2)
            if tier["id"] == "unclassified":
                # No P31 type, so we do not know what these are — but the OSM tag is
                # itself a claim. A `boundary=administrative` level asserts administrative
                # standing (Paris's 379 `admin=10` really are quartiers administratifs);
                # a `place=*` tag asserts nothing. Same asymmetry the positional fallback
                # rests on, so it is applied the same way here.
                hide = tier.get("skipped") or not all(
                    k.startswith("admin=") for k in tier["k"]
                )
            elif tier["id"] == "neighbourhood":
                # `known is None` = nothing measurable, which is not the same as measured
                # and low. A named layer from a city's own publisher is shown; the
                # recognisability test exists to suppress the spotty `place=*` layers OSM
                # accumulates, and it has no opinion about a source it cannot see.
                hide = known is not None and known < MIN_KNOWN_FRAC and tier is not rescue
            else:
                hide = False
            if hide:
                tier["off"] = 1

        # The tier is stamped on the UNIT, not just named on the level, because a split
        # level has units in two different tiers and a level-keyed mapping cannot say so.
        for i, tier in enumerate(tiers):
            for u in tier.pop("_u"):
                u["tr"] = i
            for k in tier["k"]:
                if k in city["levels"] and not tier.get("split"):
                    city["levels"][k]["tier"] = tier["id"]
        city["tiers"] = tiers
    return n_typed, n_untyped, n_inferred, n_split


def build_geom(data):
    """cache/geom/<qid>.json -> data/geom/<qid>.json, one GeoJSON per city.

    Kept OUT of base.json deliberately. Outlines are two orders of magnitude larger than
    the points, and the browser only ever draws one city at a time — bundling them would
    turn a 3 MB startup into a ~180 MB one to show shapes nobody has asked for yet.

    BORROWING. Half the units are bare nodes, and very unevenly — Copenhagen 2% polygons
    against Hamburg's 100%, which is a national mapping convention rather than a fact
    about the cities. But the same place is often mapped twice at different levels
    (§4.6), so a node-only `place=suburb` frequently has an `admin=9` twin that does
    carry an outline. Where it does, the outline is attached to the node's own record.

    **Borrowing is gated on containment, not just on the name.** A same-named polygon
    that does not contain the node is a different place — every large country has several
    Santa Cruzes — and matching on name alone would confidently draw the wrong one. The
    node being inside the polygon is what turns a name coincidence into evidence.

    Borrowed outlines are marked `b: 1` on the feature and `b: 1` on the unit, because a
    borrowed extent is a weaker claim than a surveyed one and the UI must not draw them
    identically.
    """
    GEOM_OUT.mkdir(parents=True, exist_ok=True)
    by_city = {}
    for u in data["units"]:
        by_city.setdefault(u["c"], []).append(u)

    wof = {}
    if WOF_IN.exists():
        for uid, r in json.loads(WOF_IN.read_text("utf-8"))["shapes"].items():
            if (r.get("iou") or 0) < WOF_DUP_IOU:
                wof[uid] = r

    # Per-city official sources (fetch_external.py). Tried BEFORE Who's On First because
    # it is the stronger claim: a municipal planning authority's own layer for one named
    # city beats a world gazetteer's guess. It is what reaches the cities WOF cannot —
    # measured, WOF gave Cairo 0 shapes and Copenhagen 7.
    ext = {}
    if EXT_IN.exists():
        ext = json.loads(EXT_IN.read_text("utf-8"))["shapes"]

    written, n_borrowed, n_wof, n_ext = {}, 0, 0, 0
    for qid, units in by_city.items():
        path = GEOM_IN / f"{qid}.json"
        if not path.exists():
            continue

        corpus = {u["i"] for u in units}
        own, named = {}, collections.defaultdict(list)
        # A geometry fetch and a build can legitimately be running at once — the fetch
        # takes hours — and the fetch writes each city's file in one non-atomic
        # `write_text`. Reading one mid-write raises, and crashing the whole build over a
        # file that will be complete in a second is the wrong response: skip that city,
        # say so, and let the next build pick it up.
        try:
            elements = json.loads(path.read_text("utf-8")).get("elements", [])
        except ValueError:
            print(f"  SKIPPED {qid}: geometry cache is mid-write, rerun build after "
                  f"the fetch finishes")
            continue
        for el in elements:
            g = osmgeom.shape_of(el)
            if g is None or g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            g = g.simplify(SIMPLIFY_DEG, preserve_topology=True)
            if g.is_empty:
                continue
            uid = f"{el['type'][0]}{el['id']}"
            own[uid] = g
            name = (el.get("tags") or {}).get("name")
            if name:
                named[pl.norm_name(name)].append((uid, g))

        feats = []
        for u in units:
            g, borrowed = own.get(u["i"]), False
            # A unit that only exists because an external source drew it goes straight to
            # that source's polygon. Letting it fall into the borrowing branch would be
            # actively wrong: it would hand a Cairo kism the outline of whichever OSM way
            # shares its name, in place of the CAPMAS boundary that IS this record.
            if g is None and not u.get("src"):
                pt = osmgeom.sg.Point(u["x"], u["y"])
                for uid, cand in named.get(pl.norm_name(u["n"]), ()):
                    if not cand.contains(pt):
                        continue
                    if uid in corpus:
                        # The twin is ALREADY a unit in this city's corpus, so its outline
                        # is on screen anyway. Borrowing it would draw the same shape twice
                        # and add no information — these two records are one place counted
                        # twice (§4.6). Record the link and leave this one a pin; that is a
                        # de-duplication fact, not a shape.
                        u["dup"] = uid
                        break
                    g, borrowed = cand, True
                    break
            # Then the city's own official source, then Who's On First. Both are
            # third-party geometry for a place OSM knows only as a point, so both are
            # weaker claims than a surveyed or borrowed outline and each is flagged as its
            # own kind rather than folded into `b` — §1.3a is emphatic that an inferred
            # extent must never render identically to a surveyed one.
            ext_rec = wof_rec = None
            if g is None:
                ext_rec = ext.get(u["i"])
                if ext_rec:
                    try:
                        g = osmgeom.sg.shape(ext_rec["g"])
                    except Exception:
                        g, ext_rec = None, None
            if g is None:
                wof_rec = wof.get(u["i"])
                if wof_rec:
                    try:
                        g = osmgeom.sg.shape(wof_rec["g"])
                    except Exception:
                        g = None
            if g is None:
                continue
            if borrowed:
                u["b"] = 1
                n_borrowed += 1
            elif ext_rec:
                # `e`, NOT `x` — `x` is the unit's longitude. An earlier version of this
                # line silently rewrote the coordinate of every externally-shaped unit.
                u["e"] = 1
                n_ext += 1
            elif wof_rec:
                u["w"] = 1
                n_wof += 1
            feats.append({
                "type": "Feature",
                "geometry": json.loads(json.dumps(g.__geo_interface__)),
                "properties": {
                    "i": u["i"],
                    **({"b": 1} if borrowed else {}),
                    **({"e": 1} if ext_rec else {}),
                    **({"w": 1} if wof_rec else {}),
                },
            })

        if not feats:
            continue
        out = GEOM_OUT / f"{qid}.json"
        out.write_text(
            json.dumps({"type": "FeatureCollection", "features": feats},
                       separators=(",", ":")),
            encoding="utf-8",
        )
        written[qid] = (len(feats), out.stat().st_size)
    return written, n_borrowed, n_wof, n_ext


def _round_coords(node, dp=4):
    """Round every coordinate in a nested GeoJSON coordinate array.

    4 dp is ~11 m, well under a pixel at every zoom this layer is drawn at, and it takes
    roughly a third off the file against shapely's full float output.
    """
    if isinstance(node, (list, tuple)):
        if node and isinstance(node[0], (int, float)):
            return [round(float(v), dp) for v in node]
        return [_round_coords(x, dp) for x in node]
    return node


def _disc(lon, lat, km):
    """The search radius as a polygon. A degree of longitude shrinks with latitude, so
    this is an ellipse in lon/lat and a circle on the ground."""
    d_lat = km / 111.32
    d_lon = km / (111.32 * max(0.05, math.cos(math.radians(lat))))
    step = 2 * math.pi / DISC_SIDES
    return osmgeom.sg.Polygon([
        (lon + d_lon * math.cos(i * step), lat + d_lat * math.sin(i * step))
        for i in range(DISC_SIDES)
    ])


def build_city_shapes(data):
    """One outline per city -> data/city_shapes.json, the zoomed-out view (§6a.6).

    Zoomed out past a city, the per-unit dots are a smear and only one city's are loaded
    anyway, so below zoom 9 the browser draws this instead: every city in the corpus as a
    single shape, which is how a reader finds out which cities there are to zoom into.

    **CLIPPED TO THE SEARCH RADIUS, always.** Tokyo's `admin=4` relation is a prefecture
    reaching 1,000 km into the Pacific and Beijing's is mostly farmland — the same
    oversized boundaries §3.1 and `locate` already have to defend against. Drawn raw,
    the overview would claim a corpus that stops at 25 km covers an ocean. The disc is
    exactly the area that was surveyed, so the intersection is exactly the claim we can
    make.

    Cities with no TRUSTED boundary fall back to the convex hull of their own units,
    padded a little. That is most of the roster — the boundary pass has reached 59 of 181
    cities — and it is a blobbier shape, but it answers the same two questions: where is
    this city, and is there anything here yet. `src` records which of the two a shape is,
    so the browser is never guessing.
    """
    by_city = collections.defaultdict(list)
    for u in data["units"]:
        by_city[u["c"]].append((u["x"], u["y"]))

    feats, n_hull = [], 0
    for qid, c in data["cities"].items():
        pts = by_city.get(qid)
        if not pts:
            continue
        g, src = None, "boundary"
        if c.get("boundaryState") == "ok":
            g = pl.city_geom(qid)
            if g is not None:
                g = osmgeom.polygonal(g.intersection(_disc(c["lon"], c["lat"], c["radiusKm"])))
        if g is None or g.is_empty:
            # buffer() rather than the bare hull: a hull stops dead at the outermost
            # centroid, which at this zoom draws a city visibly smaller than it is.
            g = osmgeom.sg.MultiPoint(pts).convex_hull.buffer(0.01)
            g, src, n_hull = osmgeom.polygonal(g), "units", n_hull + 1
        if g is None or g.is_empty:
            continue
        g = g.simplify(CITY_SIMPLIFY_DEG, preserve_topology=True)
        if g.is_empty:
            continue
        gj = json.loads(json.dumps(g.__geo_interface__))
        feats.append({
            "type": "Feature",
            "geometry": {"type": gj["type"], "coordinates": _round_coords(gj["coordinates"])},
            # The whole city record is NOT copied in. The browser already holds
            # base.json's `cities` and looks the rest up by `q`; duplicating name and
            # counts here would give the overview its own copy to drift out of date.
            "properties": {"q": qid, "src": src},
        })
    CITY_OUT.write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")),
        encoding="utf-8",
    )
    return len(feats), n_hull


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    data = build()
    # Geometry FIRST: it marks units that borrowed an outline, and base.json must carry
    # that flag. It also records the name+containment `dup` pairs that mark_dupes reuses,
    # so the order here is load-bearing.
    geom, n_borrowed, n_wof, n_ext = build_geom(data)
    n_groups, n_dupes, bad_qids = mark_dupes(data)
    # AFTER mark_dupes: tier ordering is coarse-to-fine by DISTINCT places, which needs
    # `dupOf` to already be set or a level recorded twice would count as twice as fine.
    n_typed, n_untyped, n_inferred, n_split = assign_tiers(data, {q: v[0] for q, v in load_wiki().items()})

    total_u = len(data["units"])
    idx_kb, unit_kb, n_files = write_split(data)
    print(f"wrote {OUT} — {len(data['cities'])} cities, {idx_kb:,.0f} KB index")
    print(f"wrote {UNITS_OUT} — {total_u:,} units over {n_files} files, "
          f"{unit_kb:,.0f} KB total")
    print(f"  {n_dupes:,} units are duplicates of another ({n_groups:,} groups) — "
          f"{total_u - n_dupes:,} distinct places")
    print(f"  {sum(1 for u in data['units'] if u.get('self')):,} units are their own city "
          f"(marked `self`, quiz must exclude)")
    n_src = collections.Counter(u["src"] for u in data["units"] if u.get("src"))
    if n_src:
        where = ", ".join(f"{data['cities'][q]['name']} {n}" for q, n in n_src.most_common())
        print(f"  {sum(n_src.values()):,} units come from a city's own source, not OSM "
              f"({where})")
    if n_typed or n_untyped:
        print(f"  {n_typed} tiered from P31, {n_inferred} inferred informal by position, "
              f"{n_untyped} unclassified; {n_split} levels split on P31")
    for q, n, names in sorted(bad_qids, key=lambda r: -r[1]):
        print(f"  BAD WIKIDATA TAG: {q} is on {n} units with different names "
              f"({', '.join(names)}) — not merged")

    n_city, n_hull = build_city_shapes(data)
    print(f"wrote {CITY_OUT} — {n_city} city outlines "
          f"({n_city - n_hull} from a boundary, {n_hull} hulled from their units), "
          f"{CITY_OUT.stat().st_size / 1024:,.0f} KB")

    if geom:
        n = sum(v[0] for v in geom.values())
        mb = sum(v[1] for v in geom.values()) / 1024 / 1024
        total = len(data["units"])
        print(f"wrote {GEOM_OUT} — {len(geom)} cities, {n:,} outlines, {mb:,.1f} MB")
        print(f"  {n / total:.0%} of all units now have a shape "
              f"({n - n_borrowed - n_wof - n_ext:,} own, {n_borrowed:,} borrowed, "
              f"{n_ext:,} from city sources, {n_wof:,} from Who's On First)")
    else:
        print("no geometry yet — run `python fetch_osm.py --pass geom`")

    if args.stats:
        u = data["units"]
        poly = sum(x["poly"] for x in u)
        qid = sum(1 for x in u if x["q"])
        pop = sum(1 for x in u if x["p"])
        wpop = sum(1 for x in u if x.get("wp"))
        anypop = sum(1 for x in u if x["p"] or x.get("wp"))
        core = sum(1 for x in u if x["core"] == 1)
        known = sum(1 for x in u if x["core"] is not None)
        print(f"  with a polygon in OSM   {poly:>7,}  ({poly/len(u):.0%})")
        print(f"  with a wikidata QID     {qid:>7,}  ({qid/len(u):.0%})")
        print(f"  with an OSM population  {pop:>7,}  ({pop/len(u):.0%})")
        print(f"  with a P1082 population {wpop:>7,}  ({wpop/len(u):.0%})")
        print(f"  with EITHER population  {anypop:>7,}  ({anypop/len(u):.0%})")
        print(f"  inside the city proper  {core:>7,}  ({core/known:.0%} of {known:,} known)")
        print()
        for qid_, c in sorted(data["cities"].items(), key=lambda kv: -kv[1]["n"]):
            print(f"  {c['name']:<18}{c['n']:>7,}  {', '.join(c['keep'])}")


if __name__ == "__main__":
    main()
