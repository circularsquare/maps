"""
Stage 1: survey what sub-city divisions OpenStreetMap actually has, city by city.

WHY A SURVEY PASS AT ALL. The level-picking rule is per city per level: if most of
a city's `admin_level=9` units land in the population band, we take ALL of them, and
if they don't we take none. That decision needs only names, levels and populations —
never geometry. Geometry is 100–1000x the bytes. So this script has two passes and
the expensive one runs second, over a shortlist:

    survey  ->  `out tags center;`   every candidate, tags + a centroid, cheap
    geom    ->  `out geom;`          only the (city, level) pairs that survived

Running `geom` over everything would work and would be wasteful in a way somebody
else pays for, since Overpass is a donated public endpoint.

WHY OVERPASS AND NOT OVERTURE. Overture's divisions theme is conflated OSM (plus
geoBoundaries), so it is not an independent source — same polygons, one month stale.
What it adds is a NORMALISED subtype vocabulary, and normalisation is exactly what we
must not have: a German admin_level=9 and a Japanese one mean different things, and
the per-city rule is only sound because it never compares them. Overture would also
mean duckdb and remote GeoParquet scans for data we can get per-city and cache.
If OSM coverage turns out too thin, Overture is the documented fallback — see spec.md.

POLITENESS. Overpass rations CPU-seconds, not requests, so a fixed sleep is not a
politeness measure. We throttle on duty cycle: after a query that took T seconds we
sleep 2*T (floor 5s), holding us near a third of one slot however slow the queries
turn out to be. 429 is honoured via Retry-After. Gateway errors get two retries;
timeouts get none, because a deterministic query that hit the server deadline will hit
it again and the retry just burns someone else's budget — the fix is a smaller bbox.
Every response is cached, so a rerun costs zero requests.

Usage:
    py fetch_osm.py --plan                 # print the plan, zero requests
    py fetch_osm.py --only Q60,Q90         # smoke test on two cities
    py fetch_osm.py                        # survey pass, resumable
    py fetch_osm.py --pass geom            # geometry pass (needs levels.json)
"""

import argparse
import csv
import json
import math
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
CACHE = HERE / "cache"
SEED = DATA / "cities_seed.csv"
LEVELS = DATA / "levels.json"

ENDPOINT = "https://overpass-api.de/api/interpreter"

# The one thing tying this traffic to a real person. Overpass operators ask for it and
# it stays exactly as written.
UA = "maps-neighborhoods/0.1 (https://anita.garden; anitaxinchen@gmail.com)"

# `place=*` values that can plausibly land in the 10k..10% band. `city_block` is
# included because a few dense Asian cities use it where others use `quarter`; it is
# far too small almost everywhere else, and the band rule is what discards it.
PLACE_VALUES = "borough|suburb|quarter|neighbourhood|city_block"

# Administrative levels worth asking about. 6 is included not because a level-6 unit
# could ever be a neighbourhood — it is the city itself in much of Europe — but so the
# survey can SEE the level above and report the hierarchy. The band rule discards it.
# Widen if a country's neighbourhoods turn out to live outside this range.
ADMIN_LEVELS = "6|7|8|9|10|11"

QUERY_TIMEOUT = 180
DUTY_CYCLE = 2.0
MIN_SLEEP = 5.0


def cities(only=None):
    """The seed roster. '#' comments and blank lines are skipped so the CSV stays
    annotatable by hand."""
    out = []
    with open(SEED, encoding="utf-8") as fh:
        rows = csv.DictReader(r for r in fh if r.strip() and not r.startswith("#"))
        for r in rows:
            if only and r["qid"] not in only:
                continue
            out.append(
                {
                    "qid": r["qid"],
                    "name": r["name"],
                    "country": r["country"],
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                    "pop": int(r["pop"]),
                    "radiusKm": float(r["radiusKm"]),
                }
            )
    return out


def bbox(city):
    """(south, west, north, east) for Overpass, from centre + radius.

    A box, not the city's admin boundary, because city boundaries are exactly the
    thing that is inconsistent between countries — Tokyo's is a prefecture, London's
    is Greater London, Paris's stops at the périphérique. Overlap between boxes is
    fine: `assign.py` gives each candidate to its nearest seed city.
    """
    dlat = city["radiusKm"] / 111.32
    dlon = city["radiusKm"] / (111.32 * max(0.05, math.cos(math.radians(city["lat"]))))
    return (
        round(city["lat"] - dlat, 5),
        round(city["lon"] - dlon, 5),
        round(city["lat"] + dlat, 5),
        round(city["lon"] + dlon, 5),
    )


def survey_query(city):
    s, w, n, e = bbox(city)
    box = f"{s},{w},{n},{e}"
    return f"""[out:json][timeout:{QUERY_TIMEOUT}];
(
  nwr["place"~"^({PLACE_VALUES})$"]({box});
  nwr["boundary"="administrative"]["admin_level"~"^({ADMIN_LEVELS})$"]({box});
);
out tags center;"""


def boundary_query(city):
    """The city's OWN administrative outline, fetched by Wikidata QID.

    This exists because of the New York problem: a bbox around New York contains
    Hoboken and Jersey City, they are admin_level=8 units in the population band, and
    the level rule happily kept them as "New York neighbourhoods". Distance cannot fix
    it — Hoboken really is nearer to Manhattan than Far Rockaway is. What separates them
    is that Hoboken lies OUTSIDE New York's boundary, so that is what we test.

    Matched on `wikidata` rather than name because names are ambiguous across a bbox
    this size and the QID is already the roster's primary key. Cities whose OSM relation
    carries no QID come back empty and are reported; the fallback is a hand entry in
    data/levels_override.json, not a fuzzy name match.
    """
    return f"""[out:json][timeout:{QUERY_TIMEOUT}];
relation["wikidata"="{city['qid']}"]["boundary"="administrative"];
out geom;"""


# Levels a "city proper" boundary plausibly sits at. 9+ is sub-city (a ward, a suburb);
# 3 and below is a country or bigger. Egypt puts governorates at 4, Australia puts
# councils at 6, Japan puts the prefecture at 4 — hence the width.
FALLBACK_LEVELS = "4|5|6|7|8"


def enclosing_query(city):
    """Tags of every administrative relation containing the city centre.

    The fallback for when the QID match finds nothing, which happens for 14 of 57 seed
    cities. The cause is systematic rather than sloppy tagging: the roster's QID names
    the city *as a place*, while OSM's boundary relation carries the QID of the
    *administrative entity*, and those are different Wikidata items. Sydney is Q3130 but
    the enclosing relations are Q1094194 (Council of the City of Sydney) and a level-9
    Q110046497; Cairo is Q85 but the governorate is Q30805.

    `is_in` + `pivot` asks the question geometrically instead, so it does not care how
    anything is tagged. Tags only — geometry comes in a second request for the one
    relation we choose, because a level-4 relation can be an entire Australian state and
    we do not want four of those to find out we wanted the level-6.
    """
    return f"""[out:json][timeout:{QUERY_TIMEOUT}];
is_in({city['lat']},{city['lon']})->.a;
relation(pivot.a)["boundary"="administrative"]["admin_level"~"^({FALLBACK_LEVELS})$"];
out tags;"""


def cached_levels(payload):
    """Which level keys a cached geometry response actually contains.

    Uses pick_levels.level_key so this cannot drift from the definition the rest of the
    pipeline uses — imported lazily because pick_levels imports THIS module, and the cycle
    is only safe once both are loaded. If it cannot be imported, return None, which the
    caller reads as "unknown" and therefore refetches: paying for a fetch is the safe
    direction, silently skipping one is not.
    """
    try:
        from pick_levels import level_key
    except ImportError:
        return None
    return {
        k
        for el in payload.get("elements") or []
        if (k := level_key(el.get("tags") or {}, el["type"]))
    }


def by_id_query(rel_id):
    return f"[out:json][timeout:{QUERY_TIMEOUT}];\nrelation({rel_id});\nout geom;"


def pick_enclosing(elements):
    """The deepest enclosing relation wins.

    Deepest = most specific = closest to "the city proper". For Cairo that is the
    governorate (4), for Sydney the council (6). Sydney's council covers only the CBD
    while the metro sprawls over thirty more councils, so most of Sydney's units will
    read as non-core — which is not a mistake but a true statement about Sydney, and
    exactly the kind of thing `coreFrac` exists to show.
    """
    best, best_level = None, -1
    for el in elements:
        raw = (el.get("tags") or {}).get("admin_level", "")
        if not raw.isdigit():
            continue
        if int(raw) > best_level:
            best, best_level = el, int(raw)
    return best


def geom_query(city, levels):
    """Geometry for the kept levels only.

    `levels` are the survey's level keys — "place=suburb" or "admin=9". They are
    turned back into the two clause shapes they came from.
    """
    s, w, n, e = bbox(city)
    box = f"{s},{w},{n},{e}"
    places = sorted(l.split("=", 1)[1] for l in levels if l.startswith("place="))
    admins = sorted(l.split("=", 1)[1] for l in levels if l.startswith("admin="))
    clauses = []
    if places:
        clauses.append(f'  nwr["place"~"^({"|".join(places)})$"]({box});')
    if admins:
        clauses.append(
            f'  nwr["boundary"="administrative"]["admin_level"~"^({"|".join(admins)})$"]({box});'
        )
    body = "\n".join(clauses)
    # `out geom` inlines member coordinates, which is what lets us rebuild multipolygons
    # without a second round trip for every way.
    return f"[out:json][timeout:{QUERY_TIMEOUT}];\n(\n{body}\n);\nout geom;"


def fetch(query):
    """POST one query. Returns (parsed_json, seconds). Raises on give-up."""
    body = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body, headers={"User-Agent": UA, "Accept": "application/json"}
    )
    gateway_tries = 0
    net_tries = 0
    while True:
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=QUERY_TIMEOUT + 60) as resp:
                raw = resp.read()
            return json.loads(raw), time.monotonic() - t0
        except urllib.error.HTTPError as err:
            elapsed = time.monotonic() - t0
            if err.code == 429:
                # Overpass says "no slot available". Retry-After is usually present and
                # honest; when it is not, back off a flat minute.
                wait = float(err.headers.get("Retry-After") or 60)
                # flush: without it these sit in the buffer and a long slot-wait is
                # indistinguishable from a hang when watching the output file.
                print(f"      429, waiting {wait:.0f}s for a slot", flush=True)
                time.sleep(wait + 2)
                continue
            if err.code in (502, 503, 504) and gateway_tries < 2:
                gateway_tries += 1
                wait = 30 * gateway_tries
                print(f"      {err.code}, retry {gateway_tries}/2 in {wait}s", flush=True)
                time.sleep(wait)
                continue
            # A 400 is our own malformed query; anything else here we have already
            # retried as far as is polite.
            raise RuntimeError(f"HTTP {err.code} after {elapsed:.0f}s: {err.read()[:300]!r}")
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            # The network itself failed — DNS, a dropped connection, a socket timeout.
            # NOT a server rejection, so it carries no politeness meaning and retrying is
            # correct. This must be caught: a single `getaddrinfo failed` once killed a
            # 35-city run at city 17, and the whole point of the duty-cycle pacing is that
            # these jobs run long enough for a transient blip to be likely.
            # HTTPError subclasses URLError, so it must be handled above this.
            if net_tries >= 3:
                raise RuntimeError(f"network failed 4x, giving up: {err}")
            net_tries += 1
            wait = 15 * net_tries
            print(f"      network error ({err}), retry {net_tries}/3 in {wait}s", flush=True)
            time.sleep(wait)
            continue


def run(which, only, refresh, limit):
    outdir = CACHE / which
    outdir.mkdir(parents=True, exist_ok=True)

    keep, keep_only = {}, {}
    if which == "geom":
        if not LEVELS.exists():
            sys.exit("geom pass needs data/levels.json — run pick_levels.py first")
        # Kept levels plus DONOR levels — rejected levels that nonetheless hold an
        # outline for a place a kept level only has as a node. See find_donors() in
        # pick_levels.py; the geometry is what makes borrowing possible in build.py.
        raw = json.loads(LEVELS.read_text("utf-8"))
        keep = {
            c: sorted(set(v["keep"]) | set(v.get("donors") or [])) for c, v in raw.items()
        }
        keep_only = {c: sorted(v["keep"]) for c, v in raw.items()}

    roster = cities(only)
    todo = []
    for city in roster:
        dest = outdir / f"{city['qid']}.json"
        want = keep.get(city["qid"]) or []
        if which == "geom" and not want:
            continue
        if dest.exists() and not refresh:
            # The geometry cache is keyed by CITY but its contents depend on which LEVELS
            # were asked for, and that set grows when donor levels are added. Without this
            # check a city fetched before donors existed would be skipped forever and
            # silently never gain the outlines borrowing depends on.
            if which != "geom":
                continue
            try:
                payload = json.loads(dest.read_text("utf-8"))
                cached = payload.get("_levels")
            except (ValueError, OSError):
                payload, cached = None, None
            if cached is None:
                # DERIVE IT FROM WHAT THE FILE ACTUALLY HOLDS, and never from the current
                # keep-list. The previous fallback assumed a pre-`_levels` cache contained
                # "the kept levels of the day" and read today's keep-list to name them,
                # which is only true while keep-lists never change. The moment the level
                # rule moved (MIN_UNITS 8 -> 4, §3.3) that assumption silently asserted
                # every old cache already held the new levels, so New York's boroughs sat
                # at 0 shapes and the pass reported "0 to fetch" forever. Reading the
                # contents cannot go stale that way.
                cached = cached_levels(payload) if payload else None
            if cached is not None and set(cached) >= set(want):
                continue
        todo.append(city)
    if limit:
        todo = todo[:limit]

    print(f"{which} pass: {len(todo)} to fetch, {len(roster) - len(todo)} cached or skipped")
    spent = 0.0
    for i, city in enumerate(todo, 1):
        if which == "survey":
            q = survey_query(city)
        elif which == "boundary":
            q = boundary_query(city)
        else:
            q = geom_query(city, keep[city["qid"]])
        print(f"  [{i}/{len(todo)}] {city['name']} ({city['qid']})", flush=True)
        try:
            payload, took = fetch(q)
        except RuntimeError as err:
            print(f"      FAILED: {err}")
            continue
        # QID match found nothing — fall back to asking geometrically. Two extra
        # requests, only for the cities that need them.
        if which == "boundary" and not payload.get("elements"):
            try:
                enclosing, t2 = fetch(enclosing_query(city))
                took += t2
                chosen = pick_enclosing(enclosing.get("elements", []))
                if chosen is not None:
                    time.sleep(MIN_SLEEP)
                    payload, t3 = fetch(by_id_query(chosen["id"]))
                    took += t3
                    tags = chosen.get("tags") or {}
                    print(
                        f"      fallback: lvl {tags.get('admin_level')} "
                        f"{tags.get('name')} (wikidata {tags.get('wikidata')})",
                        flush=True,
                    )
                else:
                    print("      fallback found no enclosing admin relation", flush=True)
            except RuntimeError as err:
                print(f"      fallback FAILED: {err}", flush=True)

        n = len(payload.get("elements", []))
        spent += took
        if which == "geom":
            payload["_levels"] = sorted(keep.get(city["qid"]) or [])
        (outdir / f"{city['qid']}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        print(f"      {n} elements in {took:.1f}s (query-seconds so far: {spent:.0f})")
        if i < len(todo):
            time.sleep(max(MIN_SLEEP, DUTY_CYCLE * took))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pass", dest="which", choices=["survey", "boundary", "geom"], default="survey"
    )
    ap.add_argument("--only", help="comma-separated QIDs")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--refresh", action="store_true", help="refetch even if cached")
    ap.add_argument("--plan", action="store_true", help="print the plan, make no requests")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    if args.plan:
        for city in cities(only):
            s, w, n, e = bbox(city)
            span = 2 * city["radiusKm"]
            print(f"{city['name']:<16} {city['qid']:<8} {span:>5.0f}km box  {s},{w},{n},{e}")
        print(f"\n{len(cities(only))} cities")
        print("\nsurvey query for the first:\n")
        print(survey_query(cities(only)[0]))
        return
    run(args.which, only, args.refresh, args.limit)


if __name__ == "__main__":
    main()
