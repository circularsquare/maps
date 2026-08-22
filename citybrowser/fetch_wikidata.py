"""
Stage 1 of the citybrowser pipeline: pull candidate settlements from Wikidata.

This is deliberately SLOW. We are an anonymous client on a free public endpoint
and there is no deadline on this project.

THE CURRENCY IS QUERY-SECONDS, NOT REQUESTS. This is the thing that is easy to
get wrong. WDQS allows roughly 60 seconds of query PROCESSING time per 60-second
window per (User-Agent + IP), with a 60s hard timeout per query. The 200 req/min
figure published for the Action/REST API does not apply here. So a fixed gap
between requests is meaningless as a politeness measure: a 55s query followed by
a 5s pause is ~92% of the allowance, not 2%.

Instead we throttle on DUTY CYCLE — after a query that took T seconds we sleep
4*T (floor 5s), holding us at <=20% of the budget however slow the queries turn
out to be. We also cap cumulative query-seconds per run, because a request count
does not bound the resource actually being rationed.

The other ban risk is retry storms. A 500 from WDQS almost always means the
query hit the server-side deadline; retrying identical deterministic SPARQL
cannot succeed, it just burns another 60 seconds of someone else's budget. So
timeouts are never retried — we go straight to splitting the country into
population bands. Only genuinely transient 502/503/504 get a retry, twice.

Other politeness: a descriptive User-Agent with contact info per the Wikimedia
UA policy (this is the one thing tying our traffic to a real person, so it stays
exactly as written); 429 honoured via Retry-After plus our own margin; every
response cached to disk so a rerun costs zero requests.

Expect several hours for ~250 countries. That is the intended speed.

What we fetch here is THE ROSTER — one row per city on the finished map, keyed
by QID. (It was originally a candidate pool to be matched against a GHS roster;
that inverted, because GHS merges Guangzhou/Shenzhen/Foshan/Dongguan into one
43M blob and we want them as four cities.) GHS now attaches to these rows as
enrichment rather than defining them.

Because these rows become map points, the P31 type is fetched and stored so
non-settlements — administrative divisions, dioceses, "Benelux" — can be
filtered downstream. See SETTLEMENT_TYPES_QUERY.

Stage 2 (GHS enrichment match) and stage 3 (aliases via wbgetentities) are
separate.

Usage:
    python fetch_wikidata.py --plan            # print the plan, zero requests
    python fetch_wikidata.py --countries 3     # smoke test
    python fetch_wikidata.py                   # the real run, resumable
"""

import argparse
import json
import pathlib
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# On Python 3.10+ socket.timeout IS TimeoutError; on 3.9 (what this box runs) it
# is a plain OSError subclass, so `except TimeoutError` does NOT catch it. That
# gap killed a 53-country run outright. Catch both, explicitly.
NET_ERRORS = (urllib.error.URLError, TimeoutError, socket.timeout, OSError)

HERE = pathlib.Path(__file__).parent
CACHE = HERE / "cache" / f"wikidata_f{10000}"   # keyed by POP_FLOOR

ENDPOINT = "https://query.wikidata.org/sparql"

# Per the Wikimedia User-Agent policy: client/version, contact, library. "bot"
# is included deliberately so their traffic classification can see what we are.
# Do NOT substitute a browser UA — the policy states bot-like traffic with a
# browser UA is assumed malicious.
UA = (
    "citybrowser/0.1 "
    "(https://anita.garden/; anitaxinchen@gmail.com) "
    "bot python-urllib/3"
)

DUTY_FACTOR = 4.0       # sleep 4x the query's own duration -> <=20% of budget
MIN_GAP = 5.0           # floor, so a burst of fast queries is still spaced
TIMEOUT = 75            # WDQS itself cuts queries off at 60s
TRANSIENT_RETRIES = 2   # 502/503/504 only; never for query timeouts

MAX_REQUESTS = 400          # bounds a runaway loop
# Bounds the resource actually being rationed. Raised from 4000 when the floor
# dropped to 10k and P31 was added: the Netherlands went from ~1s to 12.3s, and
# 323 countries at that rate would trip a 4000s ceiling mid-run. This is a
# safety stop, not a target — the duty-cycle throttle is what keeps us polite.
MAX_QUERY_SECONDS = 9000

POP_FLOOR = 10000       # low enough to catch small named towns, per Anita
POP_BANDS = [(1_000_000, None), (200_000, 1_000_000),
             (50_000, 200_000), (POP_FLOOR, 50_000)]

# The subclass closure used to tell a city from an administrative unit, fetched
# ONCE and cached. Doing the walk globally costs ~0.5s; filtering is then a local
# set-membership test.
#
# TWO ROOTS, and the second one is not optional. "Human settlement" (Q486972)
# alone looks right and silently deletes Madrid: its ONLY P31 is Q2074737
# "municipality of Spain", which Wikidata files under administrative territorial
# entity, not settlement. Same for Marseille (commune of France) and Seville.
# Sao Paulo survives that filter only by accident, because it happens to ALSO
# carry Q515 "city". Adding "municipality" (Q15284) rescues 16,240 rows —
# Italian comuni, French communes, Brazilian/Mexican/Colombian/Dutch
# municipalities — which ARE the city items in those countries.
#
# What stays excluded is still correct: French electoral cantons, Chinese
# subdistricts/townships/counties, Tanzanian wards, Catholic dioceses, Iranian
# districts, tambon, gram panchayat, UK parliamentary constituencies.
#
# NOTE: we STORE the raw types and filter downstream rather than dropping rows
# here. If the closure turns out to be wrong we must not have to re-fetch 100k
# rows — which is exactly what saved us when this bug was found.
SETTLEMENT_TYPES_QUERY = """
SELECT DISTINCT ?s WHERE {
  { ?s wdt:P279* wd:Q486972 } UNION { ?s wdt:P279* wd:Q15284 }
}
"""

# Two sources, unioned. Neither alone is sufficient.
#
# P297 (ISO 3166-1 alpha-2) gives clean labels but the ISO code does not always
# sit on the entity that cities point to. The Netherlands is the killer case:
# the code is on Q29999 "Kingdom of the Netherlands" while Dutch cities carry
# P17 = Q55 "Netherlands", so querying the ISO holder returned 7 settlements for
# the entire country against 43 urban centres. Every healthy country returns
# 6-60 settlements per urban centre, so 0.2 stood out, but only because it was
# looked for.
#
# So we also ask the cities themselves which entity they call their country.
# That is authoritative by construction. It costs one ~8s aggregate query and
# turns up ~218 entities the ISO list misses. Most are historical states
# (Soviet Union, Czechoslovakia, Austria-Hungary) whose cities also exist under
# a modern country; those dedupe by QID and are harmless.
COUNTRY_QUERY = """
SELECT ?country ?iso WHERE {
  ?country wdt:P297 ?iso .
  FILTER NOT EXISTS { ?country wdt:P576 ?dissolved . }
}
"""

# Population floor is higher here than POP_FLOOR purely to keep this aggregate
# cheap; we only need it to enumerate country entities, not to find cities.
P17_COUNTRY_QUERY = """
SELECT ?country (COUNT(DISTINCT ?c) AS ?n) WHERE {
  ?c wdt:P17 ?country ; wdt:P625 ?g ; wdt:P1082 ?p . FILTER(?p >= 50000)
} GROUP BY ?country
"""

# Below this many settlements a P17-derived entity is not worth a query of its
# own — it is nearly always a micro-territory already covered by its parent.
P17_MIN_SETTLEMENTS = 3

# Aggregated in a subquery, labelled in the outer query.
#
# NO TRANSITIVE TYPE WALK IN THIS QUERY. The obvious `?city wdt:P31/wdt:P279*
# wd:Q486972` (human settlement) is catastrophic on WDQS as of 2026 — measured
# on Gabon, a country with 26 qualifying settlements:
#
#     with the P279* walk          HTTP 502 after 43s
#     without it                   0.6s
#     explicit P31 VALUES list     3.1s, and only 9 of the 25 results
#
# So the walk is ~70x slower AND the "cheap" enumerated-type alternative
# silently loses two thirds of the data.
#
# We still need the type, because this pool is now the ROSTER, not a candidate
# list — non-settlements would become map points. So we fetch the DIRECT P31
# values here (cheap: one more OPTIONAL inside the existing aggregate) and do
# the subclass walk once, globally, via SETTLEMENT_TYPES_QUERY above. Filtering
# then happens locally against that set.
#
# The aggregation is load-bearing, not tidiness: P1082 is multi-valued (one per
# census year) and P2044/P131 are multi-valued too, so a plain SELECT returns
# their CROSS PRODUCT — dozens of near-identical rows per city.
#
# MAX(?p) rather than SAMPLE(?p) so we deterministically get the most recent
# census rather than an arbitrary one (and so band-splitting can't bias low).
# P18 is deliberately absent: not a field we display, and the worst fan-out
# multiplier of the lot.
#
# The label service is free here (0.8s with vs 0.8s without) BECAUSE it wraps an
# already-aggregated subquery. Inline it into the inner pattern and it labels
# every duplicate row before dedup.
#
# Non-Earth items (Mars craters etc. do carry P625) are excluded implicitly by
# requiring P17, which only Earth countries satisfy.
CITY_QUERY = """
SELECT ?city ?cityLabel ?pop ?coord ?elev ?admin ?adminLabel ?types WHERE {
  {
    SELECT ?city
           (MAX(?p)    AS ?pop)
           (SAMPLE(?c) AS ?coord)
           (SAMPLE(?e) AS ?elev)
           (SAMPLE(?a) AS ?admin)
           (GROUP_CONCAT(DISTINCT ?t; separator="|") AS ?types)
    WHERE {
      ?city wdt:P17  wd:%(country)s ;
            wdt:P625 ?c ;
            wdt:P1082 ?p .
      FILTER(?p >= %(lo)d)
      %(hi_filter)s
      OPTIONAL { ?city wdt:P2044 ?e . }
      OPTIONAL { ?city wdt:P131  ?a . }
      OPTIONAL { ?city wdt:P31   ?t . }
    }
    GROUP BY ?city
  }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,mul". }
}
"""


class QueryTimeout(Exception):
    """WDQS hit its server-side deadline. Retrying is pointless and rude."""


class Throttle:
    """Duty-cycle throttle. Sleeps in proportion to how much of WDQS's time the
    last query actually consumed, and caps both request count and cumulative
    query-seconds for the run."""

    def __init__(self):
        self.requests = 0
        self.seconds = 0.0
        self._pending_sleep = 0.0

    def before(self):
        if self.requests >= MAX_REQUESTS:
            raise SystemExit(f"\nrequest ceiling ({MAX_REQUESTS}) hit — stopping. "
                             "Cached work is kept; rerun to continue.")
        if self.seconds >= MAX_QUERY_SECONDS:
            raise SystemExit(f"\nquery-second ceiling ({MAX_QUERY_SECONDS}s) hit "
                             "— stopping. Cached work is kept; rerun to continue.")
        if self._pending_sleep > 0:
            time.sleep(self._pending_sleep)
            self._pending_sleep = 0.0
        self.requests += 1
        return time.monotonic()

    def after(self, started):
        elapsed = time.monotonic() - started
        self.seconds += elapsed
        self._pending_sleep = max(MIN_GAP, DUTY_FACTOR * elapsed)
        return elapsed

    def sleep_now(self, seconds):
        """For 429 backoff, which replaces rather than adds to the duty sleep."""
        self._pending_sleep = 0.0
        time.sleep(seconds)


def sparql(query, throttle):
    """Run one SPARQL query. Raises QueryTimeout if WDQS gave up on it."""
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "application/sparql-results+json",
    })

    for attempt in range(TRANSIENT_RETRIES + 1):
        started = throttle.before()
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                body = resp.read()
            throttle.after(started)
            try:
                return json.loads(body)["results"]["bindings"]
            except (ValueError, KeyError):
                # A 200 with a non-JSON body is how a timeout sometimes surfaces.
                raise QueryTimeout("non-JSON response body")

        except urllib.error.HTTPError as e:
            throttle.after(started)
            if e.code == 429:
                # They told us to slow down. Believe them, and add margin. The
                # WDQS manual is explicit that a client which keeps ignoring 429
                # gets banned for 24 hours.
                hdr = e.headers.get("Retry-After") if e.headers else None
                delay = (int(hdr) if hdr and hdr.isdigit() else 60) + 30
                print(f"    429 rate limited — sleeping {delay}s", flush=True)
                throttle.sleep_now(delay)
                continue
            if e.code in (500, 502, 503, 504):
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:2000]
                except Exception:
                    pass
                if "Timeout" in detail or e.code == 500:
                    raise QueryTimeout(f"HTTP {e.code}") from None
                delay = 30 * (attempt + 1)
                print(f"    HTTP {e.code} — sleeping {delay}s", flush=True)
                throttle.sleep_now(delay)
                continue
            raise

        except NET_ERRORS as e:
            throttle.after(started)
            delay = 30 * (attempt + 1)
            print(f"    {type(e).__name__}: {e} - sleeping {delay}s", flush=True)
            throttle.sleep_now(delay)

    raise QueryTimeout("still failing after transient retries")


def qid(uri):
    return uri.rsplit("/", 1)[-1]


def parse_point(wkt):
    """'Point(12.34 56.78)' -> (lat, lon). Wikidata gives lon first.

    Returns None for anything that is not a parseable Earth point. Not every
    P625 value is plain WKT: non-Earth globes come back prefixed with the globe
    URI, and a few values are malformed outright. An unguarded str.index() here
    raised ValueError and cost the entire India and Iran fetches — one bad row
    in 4,000 should never take a country with it.
    """
    if not wkt or "(" not in wkt or ")" not in wkt:
        return None
    try:
        inner = wkt[wkt.index("(") + 1:wkt.index(")")]
        lon, lat = (float(x) for x in inner.split()[:2])
    except (ValueError, IndexError):
        return None
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return None
    return lat, lon


def rows_from(bindings):
    out = {}
    skipped = 0
    for b in bindings:
        q = qid(b["city"]["value"])
        pt = parse_point(b.get("coord", {}).get("value"))
        if pt is None:
            skipped += 1
            continue
        lat, lon = pt
        out[q] = {
            "qid": q,
            "name": b.get("cityLabel", {}).get("value"),
            "pop": float(b["pop"]["value"]),
            "lat": lat,
            "lon": lon,
            "elev": float(b["elev"]["value"]) if "elev" in b else None,
            "admin": qid(b["admin"]["value"]) if "admin" in b else None,
            "admin_name": b.get("adminLabel", {}).get("value"),
            "types": [t.rsplit("/", 1)[-1]
                      for t in (b.get("types", {}).get("value") or "").split("|")
                      if t],
        }
    if skipped:
        print(f"    skipped {skipped} rows with unparseable coordinates", flush=True)
    return out


def fetch_country(country_qid, label, throttle):
    """One country. Returns (rows, complete). Falls back to population bands
    when the whole-country query times out, which is what happens for France,
    China, the US, Indonesia — places with tens of thousands of settlements."""
    whole = CITY_QUERY % {"country": country_qid, "lo": POP_FLOOR, "hi_filter": ""}
    try:
        return rows_from(sparql(whole, throttle)), True
    except QueryTimeout:
        print(f"  {label}: timed out, splitting by population band", flush=True)

    merged, complete = {}, True
    for lo, hi in POP_BANDS:
        hi_filter = f"FILTER(?p < {hi})" if hi else ""
        q = CITY_QUERY % {"country": country_qid, "lo": lo, "hi_filter": hi_filter}
        try:
            merged.update(rows_from(sparql(q, throttle)))
        except QueryTimeout:
            print(f"  {label}: band {lo}-{hi} timed out, left incomplete", flush=True)
            complete = False
    return merged, complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true",
                    help="print the plan and exit without making requests")
    ap.add_argument("--countries", type=int, default=None,
                    help="stop after N countries (smoke test)")
    args = ap.parse_args()

    if args.plan:
        print(f"endpoint    {ENDPOINT}")
        print(f"user-agent  {UA}")
        print(f"throttle    sleep {DUTY_FACTOR:g}x each query's own duration "
              f"(floor {MIN_GAP:g}s) -> <={100 / (1 + DUTY_FACTOR):.0f}% of the "
              f"60s-per-minute WDQS budget")
        print(f"ceilings    {MAX_REQUESTS} requests, {MAX_QUERY_SECONDS}s query time")
        print(f"cache       {CACHE}")
        print(f"plan        1 query for the country list, then 1 per country")
        print(f"            (~250); slow countries split into "
              f"{len(POP_BANDS)} population bands")
        print(f"            a 20s query costs 20s + 80s sleep, so budget hours")
        print(f"\nsample query:{CITY_QUERY % {'country': 'Q30', 'lo': POP_FLOOR, 'hi_filter': ''}}")
        return

    CACHE.mkdir(parents=True, exist_ok=True)
    throttle = Throttle()

    # One global subclass walk, cached. Cheap (0.4s) and it is what lets us
    # filter non-settlements without the per-item walk that killed the query.
    types_path = CACHE.parent / "settlement_types.json"
    if not types_path.exists():
        print("fetching settlement type closure...", flush=True)
        ids = sorted(qid(b["s"]["value"]) for b in sparql(SETTLEMENT_TYPES_QUERY, throttle))
        types_path.write_text(json.dumps(ids), encoding="utf-8")
        print(f"  {len(ids)} settlement types", flush=True)

    countries_path = CACHE / "_countries.json"
    if countries_path.exists():
        countries = json.loads(countries_path.read_text(encoding="utf-8"))
    else:
        print("fetching country list (ISO holders)...", flush=True)
        # Keyed by QID, not ISO: several ISO codes map to two undissolved
        # entities (CY -> Cyprus and Republic of Cyprus, and likewise AQ, SA).
        # Keying cache files by ISO made the second one collide with the first
        # and get silently skipped.
        seen = {}
        for b in sparql(COUNTRY_QUERY, throttle):
            seen.setdefault(qid(b["country"]["value"]), b["iso"]["value"])
        print(f"  {len(seen)} from ISO codes", flush=True)

        print("fetching country list (as cities declare it)...", flush=True)
        added = 0
        for b in sparql(P17_COUNTRY_QUERY, throttle):
            q = qid(b["country"]["value"])
            if q in seen or int(b["n"]["value"]) < P17_MIN_SETTLEMENTS:
                continue
            seen[q] = q          # no ISO code for these; label with the QID
            added += 1
        print(f"  {added} further entities cities point at", flush=True)

        countries = sorted(seen.items())
        countries_path.write_text(json.dumps(countries), encoding="utf-8")
    print(f"{len(countries)} countries", flush=True)

    if args.countries:
        countries = countries[:args.countries]

    total = 0
    for i, (cq, iso) in enumerate(countries, 1):
        label = f"{iso}/{cq}"
        done_path = CACHE / f"{cq}.json"
        part_path = CACHE / f"{cq}.partial.json"
        if done_path.exists():
            total += len(json.loads(done_path.read_text(encoding="utf-8")))
            continue

        # One bad country must never cost the whole run. Anything unexpected is
        # recorded as partial and retried on the next pass; SystemExit (the
        # ceilings) is deliberately allowed through.
        try:
            rows, complete = fetch_country(cq, label, throttle)
        except SystemExit:
            raise
        except Exception as e:
            print(f"[{i}/{len(countries)}] {label}: {type(e).__name__}: {e} "
                  f"- recorded PARTIAL, continuing", flush=True)
            rows, complete = {}, False
        # A partial result is written to a DIFFERENT filename, so the next run
        # retries it. Otherwise a country that failed every band caches as an
        # empty success and is indistinguishable from one with no settlements.
        target = done_path if complete else part_path
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
        # Clear any stale partial once the country lands complete, or the run
        # keeps reporting it as incomplete forever.
        if complete and part_path.exists():
            part_path.unlink()

        total += len(rows)
        # ASCII only in progress output: the Windows console is cp1252 and
        # mangles anything else into a replacement char.
        print(f"[{i}/{len(countries)}] {label}: {len(rows):,} settlements"
              f"{'' if complete else ' (PARTIAL)'} | "
              f"{throttle.requests} reqs, {throttle.seconds:.0f}s query time, "
              f"{total:,} total", flush=True)

    partials = list(CACHE.glob("*.partial.json"))
    print(f"\ndone: {total:,} candidate settlements in {CACHE}")
    print(f"query time used: {throttle.seconds:.0f}s over {throttle.requests} requests")
    if partials:
        print(f"{len(partials)} countries incomplete — rerun to retry them")


if __name__ == "__main__":
    sys.exit(main())
