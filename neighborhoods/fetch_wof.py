"""
Stage 6: polygons for the units OSM only has as nodes, from Who's On First.

WHY A SECOND GAZETTEER AT ALL. Half the corpus is bare `place=*` nodes and the split is
national mapping convention rather than anything about the cities (spec 1.3a): Copenhagen
2% polygons, Mumbai 8%, Sao Paulo 13%, against Hamburg and Stockholm at 100%. Borrowing
(build.py) fixes the half where OSM drew the same place twice at different levels. It
cannot touch the other half, and the measurement is unambiguous — Copenhagen has ZERO
same-named polygons anywhere in its survey, Bangkok 2 of 578 nodes, Mumbai 17 of 781,
Istanbul 24 of 737. There is no outline in OSM to borrow. Either an outside source has
one or those cities stay pins forever.

Who's On First is that outside source: an open gazetteer with a `neighbourhood` placetype
that carries real cartographic polygons, mostly inherited from Quattroshapes, Zetashapes
and a long tail of municipal open-data releases. It is a genuinely INDEPENDENT source,
which is the whole point — spec 1.1 rejected Overture precisely because it is conflated
OSM and would hand back the same polygons one release stale.

WHAT WAS REJECTED, having looked at the actual distribution rather than the docs:

  - **Per-country SQLite** (`sqlite/whosonfirst-data-admin-<cc>-latest.db.bz2`). Real and
    convenient — the `geojson` table holds whole features and sqlite3 is stdlib — but the
    roster spans 43 countries and these are the *admin* repos, so every one of them
    carries countries, regions, counties and localities we already have from OSM. India
    alone is 1.5 GB, the US 886 MB, the global file 8.6 GB. Downloading gigabytes of
    country outlines to reach a few thousand neighbourhoods is the wrong trade.
  - **The full planet bundle.** Same objection, larger.
  - So: the **per-placetype legacy bundles**, which are exactly the four placetypes in the
    neighbourhood family and nothing else — 427 MB for all four against 8.6 GB for the
    planet, and the 406 MB of it that is `neighbourhood` is the file this stage exists for.
  - **Extracting the tarballs to disk.** ~1M small GeoJSON files; the bundle is streamed
    through `tarfile` in `r|bz2` mode and only the records that fall inside a seed city's
    bbox are ever written down. That extract is the cache the matching stage re-reads, so
    the 406 MB stream happens once and `--match-only` is seconds.
  - **Fuzzy name matching.** See below. Not a judgement call.

THE FOUR PLACETYPES. `neighbourhood` is the one that matters; `macrohood`, `microhood`
and `borough` are 21 MB between them and are the same family one step coarser and finer,
so they are fetched too and reported separately — if they add nothing the report says so
and the next person can drop them. They are NOT ranked against each other: spec 1.1's
objection to Overture's normalised subtype vocabulary applies here as well, a WOF
`macrohood` in the Netherlands and one in Singapore do not denote the same thing. Where
several candidates match, the SMALLEST containing polygon wins, which is a statement about
this pair of records and needs no cross-country vocabulary.

MATCHING DISCIPLINE — the part that has burned this project twice, so it is the part with
the most code:

  1. **Containment is the evidence, not the name.** The OSM node's point must fall inside
     the WOF polygon. A same-named polygon that does not contain it is a different place;
     every large country has several Santa Cruzes. Name equality alone never produces a
     match here, and the report counts the pairs that were thrown away for exactly this.
  2. **Name equality is `pl.norm_name` — casefold + strip-accents, deliberately NOT
     fuzzy.** Attaching a shape asserts two records are the same place and edit distance
     is not evidence for that (spec 1.3a). WOF's per-language `name:<lang>_x_preferred`
     lists ARE consulted, because "Kadıköy" against WOF's `name:tur_x_preferred` is still
     exact equality on a name WOF itself asserts for that record — but those matches are
     labelled `wof-altname` and counted apart from `wof-name` so the split is auditable.
     `name:*_x_variant` is left out: "Big Apple" is a nickname, not an identity claim.
  3. **A unit that already has a shape is never given another one.** Candidates are the
     units with no feature in `data/geom/<qid>.json` at all.
  4. **The gain is fingerprinted before it is believed.** Spec 1.3a's borrowing bug —
     2,976 "new" outlines of which 2,912 were byte-identical to one already on screen —
     is the reason. Across two different gazetteers nothing is ever byte-identical, so the
     test here is geometric: every matched WOF polygon is scored by best IoU against the
     outlines the corpus already holds for that city, and the report gives the count over
     0.9 (the same polygon by another name) and over 0.5 separately from the headline.
  5. **Scale is sanity-checked.** A neighbourhood that adopts a city-sized polygon is a
     failure dressed as a success. Every match carries its area and a `sizeRatio` against
     the corpus's own outlines at the same level, the report prints both distributions, and
     a polygon larger than the city it sits in is REJECTED outright (`too-big`). That guard
     is not a tuned threshold — a part of a city cannot be bigger than the city. Measured,
     it never fires: taking the SMALLEST containing candidate already prevents the failure
     it was written for, and it is kept as the assertion that says so.

WHAT THE MEASUREMENTS SAID, so the next person does not have to rerun it (56 cities,
32,940 units, 16,541 of them with no shape):

  - **2,690 matched, but only 1,690 of them are new.** 1,000 duplicate an outline the
    corpus already has at IoU >= 0.9. The fingerprint check earned its place again.
  - **`src:geom` predicts that almost perfectly**, and is the most useful thing this stage
    found. Municipal and national open-data imports — `esp-aytomad`, `ssuberlin`,
    `esp-cartobcn`, `sg-sggov`, `arg-caba`, `pl-gugik`, `os` — come out 98-100% duplicate,
    because OSM imported the same official file; WOF is not an independent source there,
    it is the same source twice. The crowd and gazetteer sources OSM does NOT have —
    `quattroshapes` 98% new, `mz` 97%, `pedia` 99%, `zolk` 94%, `sfgov` 94%, `lacity` 100%
    — are the entire real gain. **A city's value here is predictable from provenance, not
    from coverage.**
  - **It does not rescue the cities it was fetched for.** Copenhagen 7 of 55, Bangkok 1 of
    585, Mumbai 31 of 781. What it does rescue is Istanbul (630 of 822, though 455 of those
    duplicate), London (314 of 620) and the US cities that spec 1.3 predicted — New York
    178, Chicago 128, Los Angeles 96, San Francisco 63, nearly all of it genuinely new.
  - **The binding constraint is that WOF mostly has no polygon at all.** 349,709 of its
    413,374 neighbourhood records are points, not shapes. Only ~63,000 records worldwide
    carry an outline and 7,530 of those fall inside any of the 56 city boxes.
  - **The alt-name rule pays for itself and is not a loosening.** It is what connects
    Bangkok's วังทองหลาง to "Wang Thonglang Khwaeng", Mumbai's "Malabar Hill" to a WOF
    record whose `wof:name` is the typo "Malbar Hill", and Colaba, Dharavi and Juhu to
    records filed as "Bombay Colaba", "Bombay Dharavi" and "Juhu Beach, Mumbai". Every one
    of those is still exact equality against a name WOF asserts for that record. 273 of
    2,690 matches came this way.

RESIDUAL FAILURE MODES, found by rendering the output (spec 6a.1's check, which is
stronger than reading the totals) and left in deliberately:

  - **Four London units named "River Thames"** — OSM ways tagged `boundary=administrative`
    at `admin_level` 6 and 8 — take a 21 km2 WOF "neighbourhood" also called River Thames.
    Name and containment both genuinely hold; the river is simply drawn as a district. The
    fault is upstream in both gazetteers and no rule here can see it without a
    water-feature blocklist, which would be an arbitrary special case. Reported rather
    than patched, and it is visible the moment the shapes are plotted.
  - **Copenhagen's 7 are all `quattroshapes`**, and they plot as smooth blobs rather than
    boundaries — a generalised extent around a point, not a surveyed outline. They are
    honest about what they are; the UI's "must not draw an inferred extent like a surveyed
    one" rule (spec 1.3a) is doing real work here.
  - **`sizeRatio` over-fires on node-heavy levels.** Its denominator is the units of that
    level that already have polygons, which in exactly the levels this stage serves is a
    small self-selected sample. See area_check().

WHAT IS DELIBERATELY NOT DONE. No shape is written for a unit OSM already draws, no unit
is renamed, and nothing here feeds back into level picking (spec 1.3a: shape availability
must never influence which levels are kept). This stage only ever adds a `wof` entry to a
side file; `data/base.json` is not touched and integration is somebody else's job.

VINTAGE, and spec 1.3's assumption is half wrong. The geocode.earth bundles were BUILT on
2025-10-30, and `wof:lastmodified` on the matched records runs 2019-2024 with a spike of
1,760 in 2023 — so the repository is not the 2018 fossil the spec expects. Treat the spike
with suspicion all the same: a bulk reprocessing bumps `lastmodified` without anyone having
looked at a boundary, and the `src:geom` table says most of these outlines still come from
Quattroshapes and Mapzen-era work. The report prints the distribution so the claim stays a
number rather than a rumour.

LICENCE. "Crediting Who's On First is recommended and linking back to the License is
required. For example: Data from Who's On First. License." — https://whosonfirst.org/docs/licenses/
WOF is part original work and part modification of other open data, and several of those
sources require their own attribution. Which source supplied a given outline is on the
record as `src:geom`, so the distinct values over the matched set are emitted into the
output file and printed by the report; that is the list an attribution string has to
cover. The strings travel in `data/wof_shapes.json` rather than living only here.

Usage:
    python fetch_wof.py --plan            # zero downloads, prints remote sizes
    python fetch_wof.py                   # download, scan, match, write the output
    python fetch_wof.py --only Q1748,Q1770,Q1156   # smoke test on a few cities
    python fetch_wof.py --match-only      # re-match off the extract cache, no network
    python fetch_wof.py --report          # the full quantitative report, writes nothing
    python fetch_wof.py --placetypes neighbourhood
"""

import argparse
import collections
import datetime
import json
import math
import pathlib
import statistics
import sys
import tarfile
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
import shapely.geometry as sg
from shapely import STRtree

import osmgeom
import pick_levels as pl
from fetch_osm import UA, bbox, cities

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
CACHE = HERE / "cache" / "wof"
EXTRACT = CACHE / "extract"
GEOM = HERE / "cache" / "geom"
DISPLAY_GEOM = DATA / "geom"
BASE = DATA / "base.json"
OUT = DATA / "wof_shapes.json"

DIST = "https://data.geocode.earth/wof/dist/legacy/"

# The neighbourhood family. `neighbourhood` is 95% of the bytes and ~all of the value;
# the other three are 21 MB together and are reported separately so their worth is
# visible rather than assumed. Order is not a ranking — see the module docstring.
PLACETYPES = ("neighbourhood", "macrohood", "microhood", "borough")

LICENCE = "Who's On First License — https://whosonfirst.org/docs/licenses/"
ATTRIBUTION = "Data from Who's On First. https://whosonfirst.org/docs/licenses/"

# Coordinates are rounded to this many decimals on the way out: ~11 cm, far below any
# claim these polygons make, and it roughly halves the file. This is numeric precision,
# NOT a topology simplification — unlike data/geom (spec 6a) these shapes are still
# safe to measure.
COORD_PRECISION = 6

# Best IoU against an outline the corpus already has, above which a "new" shape is really
# the same place a second time. 0.9 is the strict reading; 0.5 is printed alongside it
# because two gazetteers digitising the same district agree to about that and the reader
# should see both numbers rather than one chosen for them.
DUP_IOU = 0.9
LOOSE_IOU = 0.5


# ---------------------------------------------------------------- remote files


def url_for(pt):
    return f"{DIST}whosonfirst-data-{pt}-latest.tar.bz2"


def remote_info(pt):
    """HEAD only — size and Last-Modified. Used by --plan, which downloads nothing."""
    r = requests.head(url_for(pt), allow_redirects=True, timeout=60,
                      headers={"User-Agent": UA})
    if r.status_code != 200:
        return None
    size = r.headers.get("Content-Length")
    return {
        "url": url_for(pt),
        "size": int(size) if size else None,
        "modified": r.headers.get("Last-Modified"),
    }


def download(pt, info):
    """Fetch one bundle into cache/wof/, resuming a partial file.

    RESUMABLE because these are hundreds of MB over a link that may not hold, and a
    restart that begins again from zero is how a five-minute stage becomes a fifty-minute
    one. The bytes land in a `.part` and are renamed only once the length matches what the
    HEAD promised, so an interrupted run can never leave a truncated file that the next
    run would happily treat as cached — the failure mode that would be silent and would
    look like "WOF has no data for this city".
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"whosonfirst-data-{pt}-latest.tar.bz2"
    if dest.exists():
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    total = info["size"] or 0

    headers = {"User-Agent": UA}
    if have:
        headers["Range"] = f"bytes={have}-"
        print(f"  resuming {dest.name} at {have/1e6:,.0f} MB", flush=True)
    else:
        print(f"  downloading {dest.name}  ({total/1e6:,.0f} MB)", flush=True)

    with requests.get(url_for(pt), headers=headers, stream=True, timeout=300) as r:
        if have and r.status_code != 206:
            # Server ignored the Range. Start over rather than appending to a prefix,
            # which would corrupt the archive in a way bz2 reports 400 MB later.
            have, part_mode = 0, "wb"
        else:
            part_mode = "ab" if have else "wb"
        r.raise_for_status()
        last = time.time()
        with open(part, part_mode) as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                have += len(chunk)
                if time.time() - last > 10:
                    pct = f" ({have/total:.0%})" if total else ""
                    print(f"    {have/1e6:,.0f} MB{pct}", flush=True)
                    last = time.time()

    got = part.stat().st_size
    if total and got != total:
        raise RuntimeError(f"{dest.name}: got {got} bytes, expected {total} — rerun to resume")
    part.replace(dest)
    print(f"  -> {dest.name}  {got/1e6:,.0f} MB", flush=True)
    return dest


# ---------------------------------------------------------------- the city boxes


def city_boxes(roster):
    """(qid, south, west, north, east) per city, the same box the survey was fetched in.

    Reusing `fetch_osm.bbox` rather than inventing an extent keeps this stage's notion of
    "in London" identical to the survey's, so a WOF polygon can never be matched to a unit
    the survey itself would have placed in a different city.
    """
    return [(c["qid"], *bbox(c)) for c in roster]


def _in_any_box(minx, miny, maxx, maxy, boxes):
    """Which city boxes this WOF bbox touches. Cheap enough to run a million times."""
    hits = []
    for qid, s, w, n, e in boxes:
        if maxx >= w and minx <= e and maxy >= s and miny <= n:
            hits.append(qid)
    return hits


# ---------------------------------------------------------------- scanning bundles


def _names_of(props):
    """Every name WOF asserts for this record, normalised.

    `wof:name` is returned separately from the per-language preferred names because the
    two produce different match rules and the report has to be able to tell them apart.
    `name:*_x_variant` is excluded on purpose: a variant is a nickname or an abbreviation,
    not a claim that the record IS that.
    """
    primary = pl.norm_name(props.get("wof:name") or "")
    alt = set()
    for k, v in props.items():
        if not k.startswith("name:") or not k.endswith("_x_preferred"):
            continue
        for s in (v if isinstance(v, list) else [v]):
            n = pl.norm_name(s or "")
            if n and n != primary:
                alt.add(n)
    return primary, sorted(alt)


def _is_live(props):
    """Skip records WOF has retired.

    A superseded or deprecated neighbourhood is a shape somebody decided was wrong. It is
    still in the bundle — WOF never deletes — and taking one would be worse than taking
    nothing, because the corpus would then disagree with WOF's own current answer.
    """
    if props.get("mz:is_current") == 0:
        return False
    if props.get("wof:superseded_by"):
        return False
    dep = props.get("edtf:deprecated")
    if dep and dep != "uuuu":
        return False
    return True


def scan(pt, path, boxes, force=False):
    """Stream one bundle and write the records inside a seed city's box to JSONL.

    The bundle is ~1M small GeoJSON files. Extracting it would cost several GB of disk and
    a filesystem beating for data that is 99.9% irrelevant to a 56-city roster, so it is
    read in `r|bz2` streaming mode and thrown away as it goes.

    The extract is keyed to the ROSTER, not just to the bundle, so a sidecar records the
    boxes it was built from and the extract is rebuilt when a city is added. Without that
    a city seeded after the first run would silently have no WOF data forever — exactly
    the trap spec 1.3a describes for the geometry cache and its `_levels`.
    """
    EXTRACT.mkdir(parents=True, exist_ok=True)
    out = EXTRACT / f"{pt}.jsonl"
    meta = EXTRACT / f"{pt}.meta.json"
    # Compared as JSON text, not as Python objects: `boxes` is a list of tuples and JSON
    # brings it back as a list of lists, so an object comparison never matches and the
    # 6-minute scan would silently rerun on every invocation.
    stamp = json.dumps({"boxes": sorted(boxes), "bundle": path.stat().st_size},
                       sort_keys=True)
    if out.exists() and meta.exists() and not force:
        if json.loads(meta.read_text("utf-8")).get("stamp") == stamp:
            return out
        print(f"  {pt}: roster or bundle changed, rescanning", flush=True)

    print(f"  scanning {path.name} ...", flush=True)
    t0 = time.time()
    seen = kept = skipped_point = skipped_dead = 0
    part = out.with_suffix(".jsonl.part")
    with tarfile.open(path, mode="r|bz2") as tf, open(part, "w", encoding="utf-8") as fh:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".geojson"):
                continue
            # `<id>-alt-<label>.geojson` are ALTERNATE geometries for a record that also
            # has a canonical file. Taking both would put two shapes on one place.
            if "-alt-" in m.name:
                continue
            seen += 1
            if seen % 100_000 == 0:
                print(f"    {seen:,} records, {kept:,} kept, {time.time()-t0:,.0f}s",
                      flush=True)
            try:
                f = json.loads(tf.extractfile(m).read())
            except Exception:
                continue
            props = f.get("properties") or {}
            geom = f.get("geometry") or {}
            if geom.get("type") not in ("Polygon", "MultiPolygon"):
                # WOF keeps a record for places it has no outline for, with a Point where
                # the polygon would be. Those are pins, and this stage exists to replace
                # pins — a point would be a downgrade dressed as a match.
                skipped_point += 1
                continue
            if not _is_live(props):
                skipped_dead += 1
                continue
            bb = props.get("geom:bbox")
            if not bb:
                continue
            try:
                minx, miny, maxx, maxy = (float(x) for x in bb.split(","))
            except ValueError:
                continue
            hits = _in_any_box(minx, miny, maxx, maxy, boxes)
            if not hits:
                continue
            primary, alt = _names_of(props)
            if not primary and not alt:
                continue
            kept += 1
            fh.write(json.dumps({
                "id": props.get("wof:id"),
                "pt": props.get("wof:placetype") or pt,
                "cc": props.get("wof:country"),
                "name": props.get("wof:name"),
                "n": primary,
                "alt": alt,
                "src": props.get("src:geom"),
                "mod": props.get("wof:lastmodified"),
                "cities": hits,
                "g": geom,
            }, ensure_ascii=False) + "\n")
    part.replace(out)
    meta.write_text(json.dumps({
        "stamp": stamp,
        "scanned": seen, "kept": kept,
        "skippedNonPolygon": skipped_point, "skippedRetired": skipped_dead,
        "seconds": round(time.time() - t0, 1),
    }), "utf-8")
    print(f"  {pt}: {seen:,} records -> {kept:,} inside a city box "
          f"({skipped_point:,} had no polygon, {skipped_dead:,} retired) "
          f"in {time.time()-t0:,.0f}s", flush=True)
    return out


def split_by_city(extracts, roster_qids):
    """Fan the per-placetype extracts out into one file per city. -> {qid: path}

    Matching is a per-city operation and the placetype extracts are keyed the other way,
    so without this every record would have to be held in memory at once — the
    neighbourhood extract is every WOF polygon inside any of 56 city boxes, geometry and
    all. This is a single streaming pass and it makes the matching stage's memory a
    function of the biggest city rather than of the roster.
    """
    outdir = EXTRACT / "city"
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("*.jsonl"):
        old.unlink()
    handles, counts = {}, collections.Counter()
    try:
        for ex in extracts:
            with open(ex, encoding="utf-8") as fh:
                for line in fh:
                    for qid in json.loads(line)["cities"]:
                        if qid not in roster_qids:
                            continue
                        h = handles.get(qid)
                        if h is None:
                            h = handles[qid] = open(outdir / f"{qid}.jsonl", "w",
                                                    encoding="utf-8")
                        h.write(line)
                        counts[qid] += 1
    finally:
        for h in handles.values():
            h.close()
    print(f"  {sum(counts.values()):,} (city, WOF record) pairs across "
          f"{len(counts)} cities")
    return {qid: outdir / f"{qid}.jsonl" for qid in counts}


def read_city_extract(path):
    if path is None or not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


# ---------------------------------------------------------------- geometry helpers


def area_km2(geom, lat):
    """Planar area scaled to km2 at this latitude.

    Deliberately a formula and not `geom:area_square_m` from the WOF record, even though
    that field exists: the corpus's own outlines have no such field, and comparing WOF's
    measurement against mine would make the size sanity-check a comparison of two methods
    instead of two shapes. One function, both sides.
    """
    return geom.area * (111.32 ** 2) * max(0.01, math.cos(math.radians(lat)))


def round_geom(geom, nd=COORD_PRECISION):
    """Round a GeoJSON geometry's coordinates in place-ish (returns a new dict)."""
    def r(c):
        if isinstance(c[0], (int, float)):
            return [round(c[0], nd), round(c[1], nd)]
        return [r(x) for x in c]
    return {"type": geom["type"], "coordinates": r(geom["coordinates"])}


_corpus_cache = {}


def corpus_shapes(qid, ids):
    """The outlines the corpus already holds for this city, from cache/geom.

    `cache/geom`, not `data/geom` — spec 6a is explicit that anything MEASURING these
    shapes reads the raw cache, and this feeds both the duplicate fingerprint and the area
    distribution. Loaded lazily and only for cities that need it, because the cache is
    220 MB and reassembling a relation into rings is not free; memoised because the
    fingerprint and the area check both want the same city.
    """
    if qid in _corpus_cache:
        return {i: g for i, g in _corpus_cache[qid].items() if i in ids}
    path = GEOM / f"{qid}.json"
    if not path.exists():
        _corpus_cache[qid] = {}
        return {}
    out = {}
    for el in json.loads(path.read_text("utf-8")).get("elements", []):
        g = osmgeom.shape_of(el)
        if g is None or g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        out[f"{el['type'][0]}{el['id']}"] = g
    _corpus_cache[qid] = out
    return {i: g for i, g in out.items() if i in ids}


# ---------------------------------------------------------------- the units


def load_units(roster_qids):
    """base.json -> (all units per city, the shapeless ones per city).

    "Already has a shape" is read off `data/geom/<qid>.json` rather than off the `poly`
    flag, because `poly` means "OSM has a way or relation for this", and the geometry pass
    reaches 97% of those — the other 3% are units whose relation would not assemble into a
    ring and which are therefore still pins on screen. Those are legitimate candidates.
    Where a city has no geometry file at all the flags are the only evidence and are used.
    """
    data = json.loads(BASE.read_text("utf-8"))
    by_city, shapeless = collections.defaultdict(list), collections.defaultdict(list)
    for u in data["units"]:
        if u["c"] not in roster_qids:
            continue
        by_city[u["c"]].append(u)

    for qid, units in by_city.items():
        path = DISPLAY_GEOM / f"{qid}.json"
        drawn = set()
        if path.exists():
            for f in json.loads(path.read_text("utf-8")).get("features", []):
                drawn.add(f["properties"]["i"])
        for u in units:
            has = u["i"] in drawn if drawn else (u.get("poly") == 1 or u.get("b") == 1)
            if not has:
                shapeless[qid].append(u)
    return data, by_city, shapeless


# ---------------------------------------------------------------- matching


def city_area_km2(city, base_city):
    """The ceiling on how big a part of this city may be.

    `hasBoundary` in base.json is already POST-trust — pick_levels runs the boundary
    through `trustworthy()` before recording it, so the too-small relations of spec 4.4a
    (Istanbul's Cankurtaran Mahallesi, Sydney's City of Sydney council) are already False
    here. Recomputing the trust check would mean re-reading every survey file to get the
    elements it needs; reading the flag the pipeline already wrote is the same answer for
    none of the cost. Cities without a trusted boundary fall back to the radius disc,
    which is the same extent the survey used to decide what "in this city" means.
    """
    if base_city.get("hasBoundary"):
        shape = pl.city_shape(city["qid"])
        # city_shape returns a PREPARED geometry (it is queried thousands of times in
        # pick_levels); `.context` is the real one, and prepared geometries have no area.
        geom = getattr(shape, "context", shape)
        if geom is not None and not geom.is_empty:
            return area_km2(geom, city["lat"])
    return math.pi * city["radiusKm"] ** 2


def match_city(city, base_city, units, shapeless, recs):
    """Attach WOF polygons to this city's shapeless units. Returns (matches, tally).

    The rules, in the order the evidence is applied:

      1. build an STRtree of every WOF polygon whose bbox touched this city;
      2. for a unit, ask the tree which polygons CONTAIN its point — that query is the
         match, and everything else only narrows it;
      3. of those, keep the ones whose name set intersects the unit's name set;
      4. of those, take the SMALLEST, as the most specific claim about this place;
      5. reject it if it is larger than the city.

    Step 2 before step 3 is the whole discipline in one line. The reverse order — find
    same-named polygons, then test containment — gives the same answer but makes it far
    too easy to write the version that skips the test, which is how this project drew the
    wrong Santa Cruz the first time.
    """
    tally = collections.Counter()
    if not recs or not shapeless:
        return [], tally

    geoms, meta = [], []
    for r in recs:
        try:
            g = sg.shape(r["g"])
        except Exception:
            continue
        if g.is_empty or not g.is_valid:
            g = osmgeom.polygonal(g.buffer(0)) if not g.is_empty else None
            if g is None or g.is_empty:
                continue
        geoms.append(g)
        meta.append(r)
    if not geoms:
        return [], tally
    tree = STRtree(geoms)

    # Name -> record indices, used ONLY to count the pairs that containment threw away.
    # It never produces a match; see the docstring.
    by_name = collections.defaultdict(list)
    for i, r in enumerate(meta):
        for n in [r["n"], *r["alt"]]:
            if n:
                by_name[n].append(i)

    city_area = city_area_km2(city, base_city)

    matches = []
    for u in shapeless:
        names = {pl.norm_name(u["n"])}
        if u.get("en"):
            names.add(pl.norm_name(u["en"]))
        names.discard("")
        if not names:
            continue

        pt = sg.Point(u["x"], u["y"])
        # `within`, NOT `contains`. STRtree.query applies the predicate with the INPUT
        # geometry as the subject — `contains` would ask whether the point contains the
        # polygon, which is false for every pair on earth and silently yields a clean
        # zero-match run that looks exactly like "WOF has no data for this city". Tested
        # against a toy tree before trusting it. `within` rather than `covered_by` so
        # this means the same thing as build.py's `cand.contains(pt)`.
        inside = [int(i) for i in tree.query(pt, predicate="within")]
        named_anywhere = {i for n in names for i in by_name.get(n, ())}

        if not named_anywhere:
            tally["no-name-in-city"] += 1
            if inside:
                # Containment without a name. Counted because it is the size of the prize
                # a name-blind rule would claim, and the size of the mistake it would make.
                tally["contained-but-unnamed"] += 1
            continue

        hits = [i for i in inside if i in named_anywhere]
        if not hits:
            tally["rejected-containment"] += 1
            tally["rejected-containment-pairs"] += len(named_anywhere)
            continue

        # Smallest containing candidate wins. Not a placetype ranking — see the docstring.
        best = min(hits, key=lambda i: (geoms[i].area, meta[i]["id"]))
        g, r = geoms[best], meta[best]
        a = area_km2(g, city["lat"])
        if a > city_area:
            tally["rejected-too-big"] += 1
            continue

        rule = "wof-name" if pl.norm_name(r["name"] or "") in names else "wof-altname"
        tally[rule] += 1
        tally[f"pt:{r['pt']}"] += 1
        matches.append({
            "unit": u,
            "geom": g,
            "rec": r,
            "rule": rule,
            "areaKm2": a,
            "nCandidates": len(hits),
        })
    return matches, tally


def fingerprint(city, units, matches):
    """Best IoU of each match against an outline the corpus already has, plus its scale.

    Spec 1.3a: the first borrowing implementation reported 2,976 new outlines of which
    2,912 were the same polygon already on screen. Nothing here can be byte-identical to
    an OSM outline — different gazetteer, different digitising — so the same question has
    to be asked geometrically. A WOF polygon that overlaps an existing outline almost
    exactly is not new coverage, it is a second drawing of a place the browser already
    shows, and it must not be counted in the headline.

    `sizeRatio` is the second half of the same suspicion. It is the matched polygon's area
    over the median area of the outlines the corpus already has AT THE SAME LEVEL in the
    same city, falling back to the city-wide median where that level has no outlines at
    all — which is the common case here, since the levels needing shapes are the ones OSM
    draws as nodes. Comparing against the level rather than the city matters: London keeps
    `admin=6` boroughs and `place=neighbourhood` at once, and a single city-wide median
    would call every borough-sized shape an outlier and every neighbourhood-sized one fine.
    """
    if not matches:
        return
    have = {u["i"] for u in units}
    shapes = corpus_shapes(city["qid"], have)
    if not shapes:
        for m in matches:
            m["iou"] = 0.0
            m["sizeRatio"] = None
        return

    level_of = {u["i"]: u["k"] for u in units}
    by_level = collections.defaultdict(list)
    for uid, g in shapes.items():
        by_level[level_of.get(uid)].append(area_km2(g, city["lat"]))
    city_med = statistics.median(a for v in by_level.values() for a in v)
    level_med = {k: statistics.median(v) for k, v in by_level.items() if len(v) >= 3}
    for m in matches:
        ref = level_med.get(m["unit"]["k"], city_med)
        m["sizeRatio"] = round(m["areaKm2"] / ref, 2) if ref else None
    ids = list(shapes)
    tree = STRtree([shapes[i] for i in ids])
    for m in matches:
        g = m["geom"]
        best, best_id = 0.0, None
        for j in tree.query(g, predicate="intersects"):
            o = shapes[ids[j]]
            try:
                inter = g.intersection(o).area
                union = g.union(o).area
            except Exception:
                continue
            if union <= 0:
                continue
            iou = inter / union
            if iou > best:
                best, best_id = iou, ids[j]
        m["iou"] = round(best, 3)
        if best_id and best >= LOOSE_IOU:
            m["overlaps"] = best_id


# ---------------------------------------------------------------- reporting


def _pct(a, b):
    return f"{a/b:.0%}" if b else "  -"


def report(roster, n_recs, by_city, shapeless, per_city, tally, meta_by_pt):
    hi = ("Copenhagen", "Bangkok", "Mumbai", "Istanbul", "London", "Sao Paulo")
    print()
    print("=" * 104)
    print("WHO'S ON FIRST — coverage gained")
    print("=" * 104)
    # `wofPolys` is the count of WOF polygons of any of the fetched placetypes inside this
    # city's box, matched or not. It separates the two very different negative results —
    # "WOF has nothing here" from "WOF has plenty here and none of the names line up" —
    # which the matched column alone cannot tell apart.
    print(f"{'city':<18}{'units':>7}{'shapeless':>11}{'wofPolys':>10}{'matched':>9}"
          f"{'gain':>7}{'dup>.9':>8}{'dup>.5':>8}{'medianKm2':>11}{'corpusKm2':>11}")

    tot = collections.Counter()
    rows = []
    for city in roster:
        qid = city["qid"]
        units, sl = by_city.get(qid, []), shapeless.get(qid, [])
        ms = per_city.get(qid, [])
        dup9 = sum(1 for m in ms if m.get("iou", 0) >= DUP_IOU)
        dup5 = sum(1 for m in ms if m.get("iou", 0) >= LOOSE_IOU)
        med = statistics.median([m["areaKm2"] for m in ms]) if ms else None
        # `corpusKm2` is the scale the matched shapes are being judged against, and
        # computing it means reassembling that city's cached OSM geometry. Only the rows
        # that get printed need it, so it is deferred to the print loop rather than paid
        # for all 56 cities to show 20.
        rows.append((city["name"], len(units), len(sl), n_recs.get(qid, 0), len(ms),
                     dup9, dup5, med, city))
        tot["units"] += len(units)
        tot["shapeless"] += len(sl)
        tot["matched"] += len(ms)
        tot["dup9"] += dup9
        tot["dup5"] += dup5

    for name, n, sl, wp, m, d9, d5, med, city in sorted(rows, key=lambda r: (-r[4], r[0])):
        star = " *" if name in hi else ""
        if m == 0 and name not in hi:
            continue
        corp = _corpus_median_km2(city, by_city.get(city["qid"], []))
        print(f"{name+star:<18}{n:>7,}{sl:>11,}{wp:>10,}{m:>9,}{_pct(m, sl):>7}"
              f"{d9:>8,}{d5:>8,}"
              f"{(f'{med:,.2f}' if med else '-'):>11}"
              f"{(f'{corp:,.2f}' if corp else '-'):>11}")
    zero = [r[0] for r in rows if r[4] == 0]
    print(f"\n  {len(zero)} of {len(rows)} cities gained nothing: "
          f"{', '.join(sorted(zero)[:14])}{' ...' if len(zero) > 14 else ''}")
    print(f"\n  TOTAL  {tot['units']:,} units, {tot['shapeless']:,} shapeless, "
          f"{tot['matched']:,} matched ({_pct(tot['matched'], tot['shapeless'])} of "
          f"shapeless, {_pct(tot['matched'], tot['units'])} of all)")
    print(f"         of those matches, {tot['dup9']:,} duplicate an outline the corpus "
          f"already has at IoU>={DUP_IOU} and {tot['dup5']:,} at IoU>={LOOSE_IOU}")
    net = tot["matched"] - tot["dup9"]
    print(f"         NET new distinct places with a shape: {net:,}")

    print("\n  why candidates were dropped")
    for k in ("rejected-containment", "rejected-containment-pairs", "rejected-too-big",
              "no-name-in-city", "contained-but-unnamed"):
        print(f"    {k:<28}{tally[k]:>9,}")
    print("\n  match rule")
    for k in ("wof-name", "wof-altname"):
        print(f"    {k:<28}{tally[k]:>9,}")
    print("\n  WOF placetype supplying the shape")
    for k in sorted(k for k in tally if k.startswith("pt:")):
        print(f"    {k[3:]:<28}{tally[k]:>9,}")

    print("\n  bundle vintage (Last-Modified is the geocode.earth build, not the data)")
    for pt, m in meta_by_pt.items():
        print(f"    {pt:<16}{m.get('modified') or '?':<34}"
              f"{(m.get('size') or 0)/1e6:>8,.0f} MB")


def spot_check(per_city, roster_by_qid, n=6):
    """A handful of real matches per city, to be read rather than counted.

    Every aggregate here can look healthy while the pairs underneath are nonsense, and the
    only cheap defence against that is printing the actual name-to-name mapping for a few
    of them. Sorted by area so the sample is not all microhoods.
    """
    print("\n  spot check — matches to read, not to count")
    for qid, ms in sorted(per_city.items(), key=lambda kv: -len(kv[1]))[:8]:
        city = roster_by_qid[qid]
        print(f"\n    {city['name']}  ({len(ms)} matches)")
        step = max(1, len(ms) // n)
        for m in sorted(ms, key=lambda m: -m["areaKm2"])[::step][:n]:
            print(f"      {m['unit']['n'][:26]:<28}-> {m['rec']['name'][:26]:<28}"
                  f"{m['rec']['pt']:<14}{m['areaKm2']:>8,.2f} km2  iou={m.get('iou',0):<6}"
                  f"{m['rule']}")


def _corpus_median_km2(city, units):
    """Median area of the outlines this city already has, for the scale check."""
    have = {u["i"] for u in units if u.get("poly") == 1 or u.get("b") == 1}
    if not have:
        return None
    shapes = corpus_shapes(city["qid"], have)
    if not shapes:
        return None
    return statistics.median(area_km2(g, city["lat"]) for g in shapes.values())


def area_check(per_city, roster_by_qid):
    """Print the matched-area distribution against the corpus's own, plus the worst cases.

    A neighbourhood that adopts a city-sized polygon is a failure, so the tail is where
    the interesting information is and a median alone would hide it.
    """
    rows = []
    for qid, ms in per_city.items():
        city = roster_by_qid[qid]
        for m in ms:
            rows.append((m["areaKm2"], city["name"], m["unit"]["n"], m["rec"]["name"],
                         m["rec"]["pt"], m["rec"]["id"], m.get("iou", 0)))
    if not rows:
        print("\n  no matches, no area check")
        return
    areas = sorted(r[0] for r in rows)
    def q(vals, p):
        return vals[min(len(vals) - 1, int(p * len(vals)))]
    print(f"\n  matched polygon area km2 — "
          f"min {areas[0]:.3f}  p10 {q(areas, .10):.2f}  median {q(areas, .50):.2f}  "
          f"p90 {q(areas, .90):.2f}  max {areas[-1]:,.1f}   (n={len(areas):,})")

    # The scale check that actually answers the question. A raw area says nothing without
    # knowing what size of thing this level holds; the ratio says whether the shape a unit
    # just adopted is the size of its neighbours or the size of its city.
    ratios = sorted(m["sizeRatio"] for ms in per_city.values() for m in ms
                    if m.get("sizeRatio"))
    if ratios:
        over = sum(1 for r in ratios if r > 10)
        print(f"\n  matched area / median corpus outline at the same level — "
              f"p10 {q(ratios, .10):.2f}x  median {q(ratios, .50):.2f}x  "
              f"p90 {q(ratios, .90):.2f}x  max {ratios[-1]:,.0f}x")
        print(f"    {over:,} of {len(ratios):,} matches are more than 10x the outlines "
              f"the corpus already has at their level")
        # READ THE RATIO WITH THE DENOMINATOR IN MIND. It can only be built from the units
        # of that level that ALREADY have a polygon, and in exactly the node-heavy levels
        # this stage exists for that is a small, self-selected sample. London's
        # `place=neighbourhood` is the clean example: its 584 OSM polygons are housing
        # estates with a median of 0.02 km2, while the 620 bare NODES at the same level are
        # districts like West Kensington that WOF quite correctly draws at ~1 km2. The
        # ratio calls those 50x outliers and they are not. So this flags shapes to look at;
        # it does not condemn them, and nothing is rejected on it.
    print("  largest 12 matches (the ones to disbelieve first):")
    for a, cname, un, wn, pt, wid, iou in sorted(rows, reverse=True)[:12]:
        print(f"    {a:>10,.1f} km2  {cname:<14}{un[:22]:<24}-> {wn[:22]:<24}"
              f"{pt:<14}{wid}  iou={iou}")


# ---------------------------------------------------------------- plan


def plan(pts):
    print("Who's On First — per-placetype bundles from geocode.earth\n")
    print(f"{'placetype':<16}{'remote':>10}  {'cached':<8}{'last-modified':<32}url")
    total = todo = 0
    for pt in pts:
        info = remote_info(pt)
        if not info:
            print(f"{pt:<16}{'404':>10}")
            continue
        dest = CACHE / f"whosonfirst-data-{pt}-latest.tar.bz2"
        cached = dest.exists()
        total += info["size"] or 0
        if not cached:
            todo += info["size"] or 0
        print(f"{pt:<16}{(info['size'] or 0)/1e6:>9,.0f}M  {'yes' if cached else 'no':<8}"
              f"{info['modified'] or '?':<32}{info['url']}")
    print(f"\n  {total/1e6:,.0f} MB total, {todo/1e6:,.0f} MB still to download")
    print(f"  cache dir: {CACHE}")
    print("\n  Rejected: the per-country SQLite builds (India 1.5 GB, US 886 MB, global")
    print("  8.6 GB) — they are the ADMIN repos and carry countries, regions and")
    print("  localities this project already has from OSM. See the module docstring.")


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", "--dry-run", dest="plan", action="store_true",
                    help="zero downloads; print what would be fetched and how big")
    ap.add_argument("--only", help="comma-separated city QIDs")
    ap.add_argument("--placetypes", default=",".join(PLACETYPES))
    ap.add_argument("--match-only", action="store_true",
                    help="re-match off cache/wof/extract, no network")
    ap.add_argument("--report", action="store_true", help="report only, writes nothing")
    ap.add_argument("--rescan", action="store_true", help="rebuild the extracts")
    args = ap.parse_args()

    pts = [p.strip() for p in args.placetypes.split(",") if p.strip()]
    if args.plan:
        plan(pts)
        return

    only = set(args.only.split(",")) if args.only else None
    roster = [c for c in cities() if not only or c["qid"] in only]
    roster_by_qid = {c["qid"]: c for c in roster}
    boxes = city_boxes(roster)

    if not BASE.exists():
        sys.exit(f"{BASE} not found — run build.py first")
    data, by_city, shapeless = load_units(set(roster_by_qid))
    n_sl = sum(len(v) for v in shapeless.values())
    print(f"{sum(len(v) for v in by_city.values()):,} units in {len(by_city)} cities, "
          f"{n_sl:,} of them with no shape at all")

    # -------- bundles
    meta_by_pt, extracts = {}, []
    for pt in pts:
        path = CACHE / f"whosonfirst-data-{pt}-latest.tar.bz2"
        info = None
        if not args.match_only:
            info = remote_info(pt)
            if not info:
                print(f"  {pt}: no such bundle, skipping")
                continue
            path = download(pt, info)
        if not path.exists():
            print(f"  {pt}: not cached and --match-only given, skipping")
            continue
        meta_by_pt[pt] = info or {
            "size": path.stat().st_size,
            "modified": datetime.datetime.fromtimestamp(
                path.stat().st_mtime, datetime.timezone.utc
            ).strftime("%a, %d %b %Y %H:%M:%S GMT") + " (local cache mtime)",
        }
        extracts.append(scan(pt, path, boxes, force=args.rescan))

    by_qid_file = split_by_city(extracts, set(roster_by_qid))

    # -------- match, one city at a time
    #
    # Per-city files rather than one big in-memory index: the neighbourhood extract holds
    # every WOF polygon inside any of 56 city boxes, geometry included, and holding all of
    # it as parsed dicts at once is gigabytes for no reason. Matching only ever looks at
    # one city, so only one city is ever loaded.
    per_city, tally, n_recs = {}, collections.Counter(), {}
    for city in roster:
        qid = city["qid"]
        recs = read_city_extract(by_qid_file.get(qid))
        n_recs[qid] = len(recs)
        ms, t = match_city(city, data["cities"].get(qid, {}), by_city.get(qid, []),
                           shapeless.get(qid, []), recs)
        tally.update(t)
        if ms:
            fingerprint(city, by_city[qid], ms)
            per_city[qid] = ms

    report(roster, n_recs, by_city, shapeless, per_city, tally, meta_by_pt)
    area_check(per_city, roster_by_qid)
    spot_check(per_city, roster_by_qid)
    vintage(per_city)
    sources(per_city)

    if args.report:
        print(f"\n  --report given, {OUT.name} not written")
        return

    write(per_city, roster_by_qid, meta_by_pt, tally, shapeless)


def vintage(per_city):
    """How old the matched records actually are, as a distribution rather than a rumour."""
    stamps = []
    for ms in per_city.values():
        for m in ms:
            try:
                stamps.append(int(m["rec"].get("mod")))
            except (TypeError, ValueError):
                pass
    if not stamps:
        return
    years = collections.Counter(
        datetime.datetime.fromtimestamp(s, datetime.timezone.utc).year for s in stamps
    )
    print("\n  wof:lastmodified on the matched records — WOF has been largely")
    print("  unmaintained since Mapzen shut down in 2018, so this is the staleness:")
    for y in sorted(years):
        bar = "#" * max(1, round(60 * years[y] / max(years.values())))
        print(f"    {y}  {years[y]:>7,}  {bar}")


def sources(per_city):
    """`src:geom` over the matched set, crossed with duplication.

    Two jobs in one table. It is the list an attribution string has to cover — WOF is part
    original work and part other people's open data, and those sources have their own terms
    (https://whosonfirst.org/docs/sources/list/).

    It is also, unexpectedly, the single best predictor of whether a match is worth
    anything. The sources that are municipal or national open-data imports — Madrid's
    ayuntamiento, the Berlin Senate, CartoBCN, Singapore's government, Buenos Aires,
    Ordnance Survey, GUGiK — come out 98-100% duplicate, because OSM imported the same
    official file. The sources OSM does not have — Quattroshapes, Zetashapes, Mapzen's own
    work, sfgov, lacity — come out 0-6% duplicate and are the real gain. Provenance is
    doing the work a coverage number cannot.
    """
    agg = collections.defaultdict(lambda: [0, 0])
    for ms in per_city.values():
        for m in ms:
            agg[m["rec"].get("src")][0] += 1
            if m.get("iou", 0) >= DUP_IOU:
                agg[m["rec"].get("src")][1] += 1
    if not agg:
        return
    print("\n  src:geom of the matched outlines — attribution list, and the strongest")
    print("  predictor of whether a match is new (see the docstring):")
    print(f"    {'src:geom':<20}{'matched':>9}{'dup>=.9':>9}{'':>3}new")
    for k, (n, dup) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"    {str(k):<20}{n:>9,}{dup:>9,}{'':>3}{(n-dup)/n:.0%}")


def write(per_city, roster_by_qid, meta_by_pt, tally, shapeless):
    shapes = {}
    for qid, ms in per_city.items():
        for m in ms:
            r = m["rec"]
            shapes[m["unit"]["i"]] = {
                "c": qid,
                "g": round_geom(r["g"]),
                # PROVENANCE. Everything needed to go back to the WOF record and to
                # re-judge this match without rerunning the stage: which record, which
                # rule, what it is called there, how big it came out, and how much it
                # overlaps a shape the corpus already had.
                "wof": r["id"],
                "wofName": r["name"],
                "pt": r["pt"],
                "rule": m["rule"],
                "src": r.get("src"),
                "mod": r.get("mod"),
                "areaKm2": round(m["areaKm2"], 4),
                # How big this shape is against the outlines the corpus already has at the
                # same level. >10 means the unit adopted something of a different order
                # from its neighbours and should be looked at before it is drawn.
                "sizeRatio": m.get("sizeRatio"),
                "iou": m.get("iou", 0.0),
                **({"overlaps": m["overlaps"]} if m.get("overlaps") else {}),
                "nCandidates": m["nCandidates"],
            }
    used = collections.Counter(v["wof"] for v in shapes.values())
    out = {
        "meta": {
            "source": "Who's On First, per-placetype bundles distributed by geocode.earth",
            "urls": [url_for(pt) for pt in meta_by_pt],
            "bundles": meta_by_pt,
            "licence": LICENCE,
            # Travels with the data on purpose (same reasoning as fetch_kontur.py): a
            # licence recorded only in a docstring does not reach whoever ships the map.
            "attribution": ATTRIBUTION,
            "upstreamSources": "https://whosonfirst.org/docs/sources/list/",
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "coordPrecision": COORD_PRECISION,
            "rules": {
                "wof-name": "OSM name or name:en == WOF wof:name, and the OSM point is "
                            "inside the WOF polygon",
                "wof-altname": "OSM name or name:en == a WOF name:<lang>_x_preferred, "
                               "and the OSM point is inside the WOF polygon",
            },
            "note": "Every shape here is an INFERRED extent from a second gazetteer. "
                    "Spec 1.3a: it must not be drawn identically to a surveyed one.",
        },
        "stats": {
            "shapeless": sum(len(v) for v in shapeless.values()),
            "matched": len(shapes),
            "distinctWofIds": len(used),
            "wofIdsUsedTwice": sum(1 for v in used.values() if v > 1),
            "duplicateOfExisting": sum(1 for v in shapes.values() if v["iou"] >= DUP_IOU),
            **{k: v for k, v in tally.items()},
        },
        "shapes": shapes,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), "utf-8")
    print(f"\nwrote {OUT} — {len(shapes):,} shapes, "
          f"{len(used):,} distinct WOF records, "
          f"{OUT.stat().st_size/1e6:,.1f} MB")


if __name__ == "__main__":
    main()
