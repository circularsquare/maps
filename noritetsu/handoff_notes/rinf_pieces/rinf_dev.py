"""Lines, stations and sections from ERA RINF, the EU Register of Infrastructure. One reader
for every RINF country; what differs between countries is a small table per country,
`rinf_countries/<cc>.py`.

    python rinf.py --fetch be                                  # SPARQL pull, a few seconds
    python extract.py --region be --pbf data/raw/belgium-260929.osm.pbf
    python build_model.py --region be --register rinf:data/raw/rinf/be

The folder name is the region code: data/raw/rinf/<cc>/ holds what --fetch wrote
(sections.json, points.json, wikidata.json) and the OSM half is read from data/proc/<cc>/.

WHAT RINF GIVES.  Every mainline of the infrastructure manager(s) the country registers, as
`SectionOfLine`s between two `OperationalPoint`s, each section with its national line id and
its length. Points carry a type (station, small station, passenger stop, junction, switch,
border point...) and a coordinate. It has NO line names and NO section geometry, and only the
registered infrastructure managers: metros, trams and light rail are never in it, so they stay
OSM lines and build_model keeps them as it keeps any OSM line.

HOW THIS READER TURNS IT INTO LINES

1. VERSIONS.  A section can appear in several versions with validity periods (Germany carries
   every section twice, "until 2026-12-31" and "from 2027-01-01", with different point URIs).
   Per section (line id, start point's uopid, end point's uopid) the version valid on the
   reference date is kept, else the latest one that has already started. The second half
   matters: Belgium's whole line 161 has a validity that ENDED 2026-09-06 and no successor
   yet, and a strict filter deleted Brussels-Namur. A version that starts in the future is
   used only when nothing else exists for that section (a line not yet open would otherwise
   appear; those are logged).

2. STATIONS.  A point typed station, small station, passenger terminal or passenger stop is a
   stop only if an OSM rail station of a matching name lies within NAME_M of it (or any OSM
   rail station within BLIND_M). RINF's typing is operational: freight yards are "stations"
   too, and Belgium lists 752 passenger-typed points for about 550 stations. Several points of
   one station ("Bruxelles-Midi" and its grids) become that one station. Everything else --
   junctions, switches, border points, unmatched stations -- is a `junction` section end,
   which build_model keeps only where OSM passenger routes run over it.

3. SECTIONS.  Within one line, a point that is not a stop and has exactly two neighbours is
   merged away, so a section runs stop to stop (or to a line end or a branch point), and its
   RINF length (`chain`) is the sum of the pieces.

4. GEOMETRY.  Each RINF section is traced over OSM rail track: a shortest path on the
   extract's track graph from where its start point lies abreast of the track to where its end
   point does (`Track.trace`; a stop is placed at its OSM station, not RINF's point). Yard and
   siding track costs more, but only for a piece with a passenger stop at an end: between two
   freight points it pushed the trace onto the main line, which then looked ridden.

   THE LENGTH CHECK IS ON THE MERGED SECTION, not on each RINF piece. A switch or yard point
   is often a few hundred metres from where its chainage says, so one piece reads short and
   the next long by as much. A merged section whose traced length is more than TOL_REL of its
   RINF length plus TOL_ABS off is traced again end to end; if that is off too it is kept only
   when OWN_SHARE of it runs on its own line's OSM relation track (routing right by
   construction, RINF's length misallocated: Torhout-Zedelgem on line 66 is 13.9 km in RINF
   for 7.8 km of track), and is otherwise left out. Every one of those is logged ("length
   off", "rejected"), and nothing is drawn as a straight-line guess.

5. NAMES.  RINF has line ids, not names. The public line number comes from:
   - `fixed` in COUNTRY, ids known outright (Belgium's line 0 is six ids, one per track);
   - the country's `ref` rule on the id (Belgium's NNN0 is line NNN), trusted only where the
     traced line lies within REL_NEAR_M of an OSM `route=railway`/`route=tracks` relation of
     that number, or near no numbered relation at all (Belgium's 0580 is a 0.7 km curve and
     line 58 is 0582, so a bare rule is wrong);
   - otherwise that relation's `ref`, excluding numbers already taken, so 36N (0366) does not
     become 36 on their shared four-track corridor. Joining a taken number needs the trace
     to run ON that relation's ways, not beside them.
   RINF ids with the same number are one line. A line whose number is known is retraced
   preferring its relations' own ways, so 36 and 36N each lie on their own pair; and a
   junction-ended section lying on top of its own line's other sections (a second track pair
   filed under another id) is dropped (`redundant`). The name is the country's template on the
   number (Infrabel writes "L.36"), else the OSM relation's name, else Wikidata's label, else
   the number; a line with no number is "first - last". Wikidata's P1671 (route number) gives
   the English name where its label is more than "line N" ("High Speed Line 1").

A NEW COUNTRY needs a file rinf_countries/<cc>.py defining COUNTRY: its ISO3 code, Wikidata item and label languages,
`fixed` and `ref` for how its RINF ids read as public numbers (both may be empty: then OSM's
relations alone number the lines; `rule_certain` when the id IS the number, as in Austria),
`ref_display` if the number is written differently from its key ("101 01"), `osm_ref` if its
OSM refs need more than dropping an "L" prefix, `osm_rel` (tags -> (ref, name) or None) where
the ref alone cannot say which relations number lines (Czechia), `no_ref` (id -> bool) for ids
that must never take a line number, such as siding leads that lie beside their line (Czechia), the name templates or `id_name` (a name read
from the id itself, as the Netherlands' "Asd-Rtd"; for a line with no number it may return
(name, name_en), as Greece's does), `generic_label` for Wikidata labels that
are not names, and its infrastructure managers' names (or `im_of(section)` returning the
manager's name from a section dict, where RINF's manager code does not tell them apart, as in
Hungary), and `skip_line(id)` for ids that are never lines (Portugal's private sidings), and
`tol_abs` where the register's section lengths leave out station track (Hungary's MÁV, 1.0 km),
`cut_at_junctions: True` to end sections where other lines meet as well as at stops
(Portugal), or a set of RINF point names or uopids to do so only there (Spain), `name_m` where RINF places stops further than NAME_M from their OSM station
(Slovenia, 1500 m), `stop_names` for points the register types as depots or yards that are
passenger stops (Slovenia's Kamnik Graben), `osm_stops` (True: OSM stations an OSM train route
stops at become stops on the stop-to-stop sections they lie on, for registers that list only
junction stations, Latvia and Estonia; "all": every OSM rail station, Bulgaria, whose RINF
lists stations but no halts), and `fix(secs, points)` to correct RINF's sections
and points in place before the build reads them (Slovakia), `highspeed_sections: False` to
leave register sections' speed unknown, so no credit is filtered by it (Greece),
`suspended(ref, rinf_ids)` for
lines that stay on the map but carry no passenger trains (Slovakia), and `stop_name(point)`
where a national station list says which points are stops and what they are called (Finland),
and `netref_wkt: True` where the points' coordinates are only on their netReference's
geometry (Luxembourg), and `direct_near_m` (metres) where an end-to-end retrace must pass
near every placed point of the section it replaces (Russia, 1500), and `light_rail_track: True`
where OSM maps some of the register's lines as railway=light_rail and their stations as
light-rail stations (Germany: the Berlin and Hamburg S-Bahn), and `station_en(point, name)`
returning an English name for a section end OSM gave none (Russia: Wikidata's label by the
point's ESR code), and `km_floor: True` where the register's section lengths leave out the
station areas and so are a floor on the track's length (Denmark: a trace is then judged against
that floor and the crow-fly distance, and no km_official is shipped), and `way_line(tags)`
naming the line a track way's own tags give (Denmark's ways carry their line's name), whose
ways pass 2 then prefers as a numbered relation's, and `osm_stop_route(tags)` for routes
besides route=train whose stops `osm_stops` takes (Denmark: the S-tog's route=light_rail), and
`osm_stop_extra(name)` for OSM stations no route lists that a national list says are served
(Denmark: the timetable's), and `no_chain: True` where the register's section lengths are too
often wrong to check a line against (Ireland: Mallow - Killarney Junction is 59 km in RINF for
1 km of track): they still guide the traces, but no km_official is shipped, and
`plain_name(tags)` giving the name OSM rail stops are read by, where OSM names them after
their platform (New Zealand: "Maungawhau 3", "Petone Station"), and `osm_stops_skip(lids)` for
lines `osm_stops` leaves alone, whose trains stop at their listed stations only (Morocco's LGV
runs beside the old line past its halts).
Then --fetch, extract,
build, and add its published lengths to check_model.REGISTER.

    python rinf.py --dry <cc>     runs build() alone and prints its log (about 25 s for Belgium)
"""
import argparse
import hashlib
import heapq
import json
import math
import os
import pickle
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"C:\Users\anita\projects\maps\noritetsu")  # DEV COPY: restore before landing
RAW = ROOT / "data" / "raw" / "rinf"

ENDPOINT = "https://graph.data.era.europa.eu/repositories/rinf-plus"
WIKIDATA = "https://query.wikidata.org/sparql"
USER_AGENT = "noritetsu-rail-map/0.1 (personal rail map research; python-urllib)"
PAGE = 10000

COUNTRY_URI = "http://publications.europa.eu/resource/authority/country/"

# RINF op-types that carry passengers: station, small station, passenger terminal, passenger stop.
PASSENGER_TYPES = {"10", "20", "30", "70"}

# OSM track a RINF section can run on. Metro, tram and light rail are never RINF lines.
TRACK = {"rail", "narrow_gauge", "preserved"}
# Track that costs more to route over, so a trace keeps to running lines where it can.
SLOW_SERVICE = {"yard", "siding", "spur"}
SLOW_COST = 1.5
OFF_LINE_COST = 1.25      # in the second pass: track not in the line's own OSM relation

NAME_M = 1000             # a passenger point is its OSM station of a matching name this close
BLIND_M = 200             # or any OSM rail station this close
NEIGHBOUR_M = 25000       # a point with no coordinate is placed this close to a placed neighbour
STOP_TO_STATION_M = 1200  # a stop position this close to a station of its name is that station
SNAP_MAX_M = 1500         # a point further than this from any track cannot be traced
SNAP_BAND_M = 150         # a point snaps to every track within its nearest distance plus this
TOL_REL, TOL_ABS = 0.15, 0.3      # a trace may be this far off the section's length (km)
REL_NEAR_M = 30           # a traced line lies on a relation's track where its ways are this close
REL_SHARE = 0.6           # ... for at least this share of its sampled length
OWN_SHARE = 0.9           # a trace this much on its own relation's ways is right however long
OWN_DIRECT = 0.5          # ... and a retrace past a failed piece must lie this much on it
AGREE = 0.03              # piecewise and end-to-end traces this close agree with each other
DETOUR = 1.3              # ... and are believed over RINF if no longer than this times crow-fly
SHORT = 0.8               # a trace shorter than this times crow-fly (less 0.1 km) is never kept
KM_FLOOR_SHARE = 0.95     # `km_floor`: a trace may be this share of the register's length, less 0.2 km,
KM_FLOOR_SLACK = 3.0      # ... and up to DETOUR x crow-fly + 0.5 km, or the length + TOL_REL + this
                          # (the two station areas left out; Vejle - Børkop winds round the fjord)
GRID_M = 500


# ================================================================ fetching

def sparql(endpoint, query, log=print, tries=3):
    body = urllib.parse.urlencode({"query": query}).encode()
    for k in range(tries):
        req = urllib.request.Request(endpoint, data=body, headers={
            "Accept": "application/sparql-results+json", "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            return [{v: b[v]["value"] for v in b} for b in d["results"]["bindings"]]
        except Exception as e:                                  # noqa: BLE001
            log(f"  SPARQL attempt {k + 1} failed: {e}")
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def paged(query, log):
    """A SELECT with ORDER BY, fetched PAGE rows at a time so no one answer is huge."""
    rows, off = [], 0
    while True:
        got = sparql(ENDPOINT, f"{query}\nLIMIT {PAGE} OFFSET {off}", log)
        rows.extend(got)
        log(f"  {len(rows)} rows")
        if len(got) < PAGE:
            return rows
        off += PAGE


Q_SECTIONS = """
PREFIX era: <http://data.europa.eu/949/>
PREFIX time: <http://www.w3.org/2006/time#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?sol ?label ?line ?a ?b ?len ?nature ?im ?from ?to WHERE {
  ?sol a era:SectionOfLine ; era:inCountry <%(country)s> ; era:opStart ?a ; era:opEnd ?b .
  OPTIONAL { ?sol rdfs:label ?label }
  OPTIONAL { ?sol era:lengthOfSectionOfLine ?len }
  OPTIONAL { ?sol era:nationalLine ?nl . ?nl era:lineId ?line }
  OPTIONAL { ?sol era:solNature ?nature }
  OPTIONAL { ?sol era:infrastructureManager ?im }
  OPTIONAL { ?sol era:validity ?v .
             OPTIONAL { ?v time:hasBeginning ?from } OPTIONAL { ?v time:hasEnd ?to } }
} ORDER BY ?sol ?line
"""

# Every point a section of this country ends at, including the far side of a border section,
# which RINF may file under the neighbour.
Q_POINTS = """
PREFIX era: <http://data.europa.eu/949/>
PREFIX time: <http://www.w3.org/2006/time#>
PREFIX wgs: <http://www.w3.org/2003/01/geo/wgs84_pos#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
SELECT DISTINCT ?op ?uopid ?name ?type ?lat ?lon ?wkt ?nwkt ?from ?to WHERE {
  { SELECT DISTINCT ?op WHERE {
      ?sol a era:SectionOfLine ; era:inCountry <%(country)s> .
      { ?sol era:opStart ?op } UNION { ?sol era:opEnd ?op } } }
  OPTIONAL { ?op era:uopid ?uopid }
  OPTIONAL { ?op era:opName ?name }
  OPTIONAL { ?op era:opType ?type }
  OPTIONAL { ?op era:netReference ?nr . ?nr wgs:lat ?lat ; wgs:long ?lon }
  OPTIONAL { ?op geo:hasGeometry ?g . ?g geo:asWKT ?wkt }
  %(netref_wkt)s
  OPTIONAL { ?op era:validity ?v .
             OPTIONAL { ?v time:hasBeginning ?from } OPTIONAL { ?v time:hasEnd ?to } }
} ORDER BY ?op
"""

# `netref_wkt` in COUNTRY: the point's geometry hangs off its netReference instead of the
# point (Luxembourg: no wgs:lat anywhere, no geo:hasGeometry on the point; all 94 points
# have it this way). Only joined for a country that sets it, so no other fetch changes.
NETREF_WKT = "OPTIONAL { ?op era:netReference ?nr2 . ?nr2 geo:hasGeometry ?g2 . ?g2 geo:asWKT ?nwkt }"

# Lines of the country that carry a route number (P1671), with labels and the OSM relation.
Q_WIKIDATA = """
SELECT ?item ?num ?osm ?lab ?lang ?len ?unit WHERE {
  ?item wdt:P17 wd:%(qid)s ; wdt:P1671 ?num .
  ?item wdt:P31/wdt:P279* wd:Q728937 .
  OPTIONAL { ?item wdt:P402 ?osm }
  OPTIONAL { ?item p:P2043/psv:P2043 ?q . ?q wikibase:quantityAmount ?len ;
                                             wikibase:quantityUnit ?unit }
  OPTIONAL { ?item rdfs:label ?lab . BIND(LANG(?lab) AS ?lang)
             FILTER(?lang IN (%(langs)s)) }
}
"""


def fetch(cc, log=print, rinf_too=True):
    conf = country(cc)
    out = RAW / cc
    out.mkdir(parents=True, exist_ok=True)
    country_uri = COUNTRY_URI + conf["iso3"]
    if rinf_too:
        log(f"RINF sections of line, {conf['iso3']}")
        secs = paged(Q_SECTIONS % {"country": country_uri}, log)
        log("RINF operational points at their ends")
        pts = paged(Q_POINTS % {"country": country_uri,
                                "netref_wkt": NETREF_WKT if conf.get("netref_wkt") else ""}, log)
        stamp = {"endpoint": ENDPOINT, "fetched": date.today().isoformat()}
        (out / "sections.json").write_text(
            json.dumps({**stamp, "rows": secs}, ensure_ascii=False), encoding="utf-8")
        (out / "points.json").write_text(
            json.dumps({**stamp, "rows": pts}, ensure_ascii=False), encoding="utf-8")
        log(f"wrote {len(secs)} section rows and {len(pts)} point rows to {out}")
    if conf.get("wikidata"):
        langs = ", ".join(f'"{x}"' for x in conf.get("langs", ["en"]))
        log("Wikidata lines with a route number (P1671)")
        wd = sparql(WIKIDATA, Q_WIKIDATA % {"qid": conf["wikidata"], "langs": langs}, log)
        (out / "wikidata.json").write_text(json.dumps(
            {"endpoint": WIKIDATA, "fetched": date.today().isoformat(), "rows": wd},
            ensure_ascii=False), encoding="utf-8")
        log(f"wrote {len(wd)} Wikidata rows")


# ================================================================ per-country table

# One file per country in rinf_countries/, so that agents can add countries in parallel
# without editing this file. `country(cc)` returns that file's COUNTRY dict.
from rinf_countries import country, osm_ref_default  # noqa: E402


# ================================================================ small helpers

def line_hash(cc, key):
    h = hashlib.blake2b(f"rinf|{cc}|{key}".encode("utf-8"), digest_size=5)
    return "r" + h.hexdigest()


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    dy = (lat2 - lat1) * 110570
    return math.hypot(dx, dy)


def path_km(pts):
    return sum(dist_m(*a, *b) for a, b in zip(pts[:-1], pts[1:])) / 1000


def iri_date(v):
    """time:hasBeginning points at an IRI like .../temporalFeature/date_2018-04-09."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", v or "")
    return m.group(1) if m else None


def tail(v):
    return (v or "").rstrip("/").rsplit("/", 1)[-1]


def refkey(r):
    return r.replace(" ", "").upper() if r else None


GERMAN_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
# Cyrillic to Latin, so a Cyrillic name has a key at all (norm() keeps only [0-9a-z]): Bulgarian
# OSM names stations in Cyrillic, RINF in Latin ("KAZANLAK" = "Казанлък"), and without this
# every Cyrillic stop position stood as a station of its own. Bulgarian streamlined system, plus
# the Russian, Ukrainian and Serbian/Macedonian letters. casefold() runs first, so lower case.
CYRILLIC_FOLD = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh", "з": "z", "и": "i",
    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sht",
    "ъ": "a", "ь": "y", "ю": "yu", "я": "ya", "ё": "e", "ы": "y", "э": "e", "і": "i", "ї": "i",
    "є": "e", "ґ": "g", "ђ": "dj", "ј": "j", "љ": "lj", "њ": "nj", "ћ": "c", "џ": "dz",
    "ѓ": "gj", "ќ": "kj", "ѕ": "dz"})
# Greek to Latin, for the same reason: Greek OSM names its stations in Greek, RINF in capitals
# of an older transliteration ("LIANOKLADION" = "Λειανοκλάδι"). Applied after accents are
# stripped, with ου as "ou" and αυ, ευ as "av", "ev" (Αυλώνας = AVLON); otherwise a plain
# letter-for-letter fold, so the older spellings (gh, ph, -on endings) still differ and are put
# right per country (rinf_countries/gr.py).
GREEK_FOLD = str.maketrans({
    "α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "i", "θ": "th",
    "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p",
    "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "y", "φ": "f", "χ": "ch", "ψ": "ps",
    "ω": "o"})
# Station words written out or abbreviated, folded to one spelling ("Linz Hbf" is ÖBB's name,
# OSM often writes "Linz Hauptbahnhof").
WORD_FOLD = [(r"hauptbahnhof", "hbf"), (r"bahnhof\b", "bf"), (r"\bbhf\b", "bf"),
             (r"haltestelle\b", "hst"), (r"\bsankt\b", "st"), (r"\bst\.", "st"),
             # ÖBB abbreviates "an der", "in der", "ob der": "Ybbs a.d.Donau"
             (r"\ba\.\s?d\.\s?", "an der "), (r"\bi\.\s?d\.\s?", "in der "),
             (r"\bo\.\s?d\.\s?", "ob der ")]
# Bookkeeping RINF appends to a name: ÖBB's "[BSTS_ID=1397220]".
NAME_TAG = re.compile(r"\s*\[[^\]]*\]")


def norm(s):
    """A name as one comparable key. Umlauts are folded to ae/oe/ue FIRST, because ÖBB's RINF
    names are written that way ("UEbelbach", "Moenchhof") and OSM's are not."""
    s = (s or "").casefold().translate(GERMAN_FOLD).translate(CYRILLIC_FOLD)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("ου", "ou").replace("αυ", "av").replace("ευ", "ev").translate(GREEK_FOLD)
    for a, b in WORD_FOLD:
        s = re.sub(a, b, s)
    return re.sub(r"[^0-9a-z]", "", s)


def name_variants(name):
    """RINF writes bilingual names "Bruxelles-Midi | Brussel-Zuid"; OSM "Bruxelles-Midi -
    Brussel-Zuid" or "Brussel-Zuid / Bruxelles-Midi". Each language part, normalised; and
    without a bracketed part, since ÖBB appends the station a point belongs to ("Wien
    Penzing (in Pz)")."""
    name = NAME_TAG.sub("", name or "").strip()
    out = set()
    for n in {name, re.sub(r"\s*\([^)]*\)", "", name).strip()}:
        parts = re.split(r"\s+[|/]\s+|\s+-\s+|;", n)
        out |= {norm(p) for p in parts if norm(p)}
        if norm(n):
            out.add(norm(n))
    return out


def display_name(name):
    parts = []
    for p in re.split(r"\s*\|\s*", NAME_TAG.sub("", name or "")):
        if p and p not in parts:
            parts.append(p)
    return " / ".join(parts)


# ================================================================ RINF

def load_rinf(path, log, ref_date=None):
    d = Path(path)
    secs = json.loads((d / "sections.json").read_text(encoding="utf-8"))
    pts = json.loads((d / "points.json").read_text(encoding="utf-8"))
    log(f"RINF: fetched {secs['fetched']}; {len(secs['rows'])} section rows, "
        f"{len(pts['rows'])} point rows")
    today = ref_date or date.today().isoformat()

    points = {}
    for r in pts["rows"]:
        p = points.setdefault(r["op"], {"op": r["op"]})
        for k in ("uopid", "name", "type"):
            if r.get(k) and not p.get(k):
                p[k] = r[k] if k != "type" else tail(r[k])
        if "lat" in r and "lon" not in p:
            try:
                p["lon"], p["lat"] = float(r["lon"]), float(r["lat"])
            except ValueError:
                pass
        if "lon" not in p and (r.get("wkt") or r.get("nwkt")):
            # LTG Infra writes a sign on positive numbers: "POINT (+22.695635 56.196012)".
            m = re.search(r"POINT\s*\(\s*([-+\d.]+)\s+([-+\d.]+)", r.get("wkt") or r["nwkt"])
            if m:
                p["lon"], p["lat"] = float(m.group(1)), float(m.group(2))

    # --- one version per section (see the docstring, 1)
    versions = defaultdict(list)
    for r in secs["rows"]:
        a, b = points.get(r["a"], {}), points.get(r["b"], {})
        key = (r.get("line") or "", a.get("uopid") or r["a"], b.get("uopid") or r["b"])
        versions[key].append(r)
    chosen, n_multi, n_expired, n_future = [], 0, 0, 0
    for key, rows in versions.items():
        n_multi += len(rows) > 1

        def span(r):
            return iri_date(r.get("from")) or "0000-00-00", iri_date(r.get("to")) or "9999-99-99"
        now = [r for r in rows if span(r)[0] <= today <= span(r)[1]]
        started = [r for r in rows if span(r)[0] <= today]
        if now:
            pick = max(now, key=lambda r: span(r)[0])
        elif started:
            pick = max(started, key=lambda r: span(r)[1])
            n_expired += 1
        else:
            pick = min(rows, key=lambda r: span(r)[0])
            n_future += 1
        chosen.append(pick)
    log(f"RINF: {len(chosen)} sections after versions ({n_multi} had several versions; "
        f"{n_expired} are past their validity with no successor and are kept; "
        f"{n_future} start after {today})")
    secs_out = []
    for r in chosen:
        try:
            km = float(r.get("len"))
        except (TypeError, ValueError):
            km = None
        secs_out.append({"sol": r["sol"], "line": r.get("line") or "", "a": r["a"], "b": r["b"],
                         "km": km, "im": tail(r.get("im")), "label": r.get("label", "")})

    # One RINF id can be several unconnected lines: Austria's "StB" is the Steiermärkische
    # Landesbahnen's three (Peggau-Übelbach, Feldbach-Bad Gleichenberg, Gleisdorf-Weiz). Each
    # connected piece is named and grouped on its own, as "<id>#<k>"; `base` keeps the id.
    by_id = defaultdict(list)
    for s in secs_out:
        s["base"] = s["line"]
        by_id[s["line"]].append(s)
    n_split = 0
    for lid, ss in by_id.items():
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for s in ss:
            parent[find(s["a"])] = find(s["b"])
        comps = defaultdict(list)
        for s in ss:
            comps[find(s["a"])].append(s)
        if len(comps) < 2:
            continue
        n_split += 1
        order = sorted(comps.values(), key=lambda c: min(
            points.get(op, {}).get("uopid") or op for s in c for op in (s["a"], s["b"])))
        for k, c in enumerate(order):
            for s in c:
                s["line"] = f"{lid}#{k + 1}"
    if n_split:
        log(f"RINF: {n_split} line ids are several unconnected pieces, each taken on its own")
    return secs_out, points


# Trolleybus words too: Hungarian Wikidata numbers Debrecen's trolleybus lines 3, 5, 9, 10,
# which otherwise named railway lines of the same number.
NOT_MAINLINE = re.compile(r"metro|tram|subway|u-bahn|stadtbahn|premetro|light rail|sneltram"
                          r"|trolleybus|trolibusz|trolejbus|troleicarro", re.I)


UNIT_KM = {"Q828224": 1.0, "Q11573": 0.001}          # kilometre, metre


def load_wikidata(path, osm_rel_ids=frozenset(), generic=None, lang="en"):
    """Route number -> the items carrying it, best first, each {"item", "osm", "km",
    <lang>: label}. Several items can share a number (in Belgium route number 2 is both HSL 2
    and Brussels metro line 2): a metro or tram never counts, then the item whose OSM relation
    (P402) is one of the extract's line relations comes first, then an item whose label in
    `lang` is a real name before one that `generic` matches (Austria's 102 01 is both
    "Ennstalbahn" and "Bahnstrecke Amstetten - Bischofshofen"), then the older item. The
    caller still checks each item's length against the line (`wikidata_item`)."""
    f = Path(path) / "wikidata.json"
    if not f.exists():
        return {}
    items = defaultdict(lambda: defaultdict(dict))
    for r in json.loads(f.read_text(encoding="utf-8"))["rows"]:
        e = items[r["num"].replace(" ", "").upper()][r["item"]]
        e["item"] = r["item"]
        if r.get("lab"):
            e.setdefault(r.get("lang", ""), r["lab"])
        if r.get("osm"):
            e["osm"] = r["osm"]
        if r.get("len") and tail(r.get("unit")) in UNIT_KM:
            e.setdefault("km", float(r["len"]) * UNIT_KM[tail(r["unit"])])
    by = {}
    for num, its in items.items():
        def score(e):
            labels = " ".join(v for k, v in e.items() if k not in ("item", "osm", "km"))
            qnum = int(re.sub(r"\D", "", tail(e["item"])) or 0)
            return ((e.get("osm") or "").isdigit() and int(e["osm"]) in osm_rel_ids,
                    not (generic and re.search(generic, e.get(lang) or "")),
                    -qnum)
        ok = [e for e in its.values() if not NOT_MAINLINE.search(
            " ".join(v for k, v in e.items() if k not in ("item", "osm", "km")))]
        if ok:
            by[num] = sorted(ok, key=score, reverse=True)
    return by


WD_KM_RANGE = (0.4, 2.5)


def wikidata_item(cands, km):
    """The first candidate item whose own length (P2043), where it has one, is within
    WD_KM_RANGE of the line's: ÖBB's 413 01 (Bruck an der Mur to Villach, 226 km) carries
    both "Bahnstrecke Bruck an der Mur-Leoben" (23 km) and "Rosentalbahn" (63 km), and is
    neither; 302 02 (37 km) is not the 275 km "Brennerbahn"."""
    for e in cands or ():
        if not e.get("km") or not km or WD_KM_RANGE[0] <= e["km"] / km <= WD_KM_RANGE[1]:
            return e
    return {}


# ================================================================ OSM track

class Track:
    """The extract's rail track as a graph of OSM nodes, with a grid of its edges for
    finding where a point lies abreast of the track."""

    def __init__(self, ways, coords, log, light_rail=False):
        self.adj = defaultdict(list)
        eu, ev, ew = [], [], []
        self.xy = {}
        self.way_tags = {}
        # `light_rail_track`: the country's register lines include track OSM maps as
        # railway=light_rail (Germany's Berlin and Hamburg S-Bahn). The second pass keeps a
        # main line off a Stadtbahn beside it, by its own relation's ways.
        kinds = TRACK | {"light_rail"} if light_rail else TRACK
        for wid, (tags, nodes) in ways.items():
            if tags.get("railway") not in kinds:
                continue
            pos, ok = coords.many(np.asarray(nodes, dtype=np.int64))
            prev = None
            for n, p, good in zip(nodes.tolist(), pos.tolist(), ok.tolist()):
                if not good:
                    prev = None
                    continue
                if n not in self.xy:
                    self.xy[n] = (coords.x[p] / 1e7, coords.y[p] / 1e7)
                if prev is not None and prev != n:
                    eu.append(prev)
                    ev.append(n)
                    ew.append(wid)
                prev = n
            self.way_tags[wid] = tags
        self.eu = np.array(eu, dtype=np.int64)
        self.ev = np.array(ev, dtype=np.int64)
        self.ew = np.array(ew, dtype=np.int64)
        lat0 = float(np.mean([y for _x, y in self.xy.values()])) if self.xy else 50.0
        self.kx, self.ky = 111320 * math.cos(math.radians(lat0)), 110570
        ax = np.array([self.xy[u] for u in eu]) if eu else np.zeros((0, 2))
        bx = np.array([self.xy[v] for v in ev]) if ev else np.zeros((0, 2))
        self.ax, self.ay = ax[:, 0] * self.kx, ax[:, 1] * self.ky
        self.bx, self.by = bx[:, 0] * self.kx, bx[:, 1] * self.ky
        self.elen = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.slow = np.array([self.way_tags[w].get("service") in SLOW_SERVICE
                              or self.way_tags[w].get("usage") in ("industrial", "military")
                              for w in ew], dtype=bool)
        self.fast = np.array([self.way_tags[w].get("highspeed") == "yes" for w in ew],
                             dtype=bool)
        self.lr = np.array([self.way_tags[w].get("railway") == "light_rail" for w in ew],
                           dtype=bool)
        for e, (u, v) in enumerate(zip(eu, ev)):
            self.adj[u].append((v, e))
            self.adj[v].append((u, e))
        self.grid = defaultdict(list)
        for e in range(len(eu)):
            x0, x1 = sorted((self.ax[e], self.bx[e]))
            y0, y1 = sorted((self.ay[e], self.by[e]))
            for i in range(int(x0 // GRID_M), int(x1 // GRID_M) + 1):
                for j in range(int(y0 // GRID_M), int(y1 // GRID_M) + 1):
                    self.grid[(i, j)].append(e)
        self.grid = {k: np.array(v, dtype=np.int64) for k, v in self.grid.items()}
        log(f"RINF: OSM track graph {len(self.xy):,} nodes, {len(eu):,} edges "
            f"over {len(self.way_tags):,} ways")

    def near(self, lon, lat, r):
        """Edges within r metres of a point: (edge ids, t along the edge, distance)."""
        x, y = lon * self.kx, lat * self.ky
        cells = []
        k = int(r // GRID_M) + 1
        ci, cj = int(x // GRID_M), int(y // GRID_M)
        for i in range(ci - k, ci + k + 1):
            for j in range(cj - k, cj + k + 1):
                g = self.grid.get((i, j))
                if g is not None:
                    cells.append(g)
        if not cells:
            return np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0)
        e = np.unique(np.concatenate(cells))
        dx, dy = self.bx[e] - self.ax[e], self.by[e] - self.ay[e]
        L2 = np.maximum(dx * dx + dy * dy, 1e-9)
        t = np.clip(((x - self.ax[e]) * dx + (y - self.ay[e]) * dy) / L2, 0, 1)
        d = np.hypot(self.ax[e] + t * dx - x, self.ay[e] + t * dy - y)
        keep = d <= r
        return e[keep], t[keep], d[keep]

    def snap(self, lon, lat):
        """Every edge the point lies abreast of, within its nearest distance plus a band.

        ABREAST means the perpendicular falls inside the edge. An edge that starts further
        down the line projects onto its own end, and taking that as a start let every trace
        begin and end up to SNAP_BAND_M past its stations: Bruxelles-Nord to Bockstael came
        out at 1.94 km, under the 2.33 km crow-fly distance. A clipped projection counts only
        as the point's nearest place on the track."""
        e, t, d = self.near(lon, lat, SNAP_MAX_M)
        if not len(e):
            return None
        d0 = float(d.min())
        inside = (t > 0) & (t < 1)
        k = (inside & (d <= d0 + SNAP_BAND_M)) | (d <= d0 + 5)
        return {"e": e[k], "t": t[k], "d0": d0}

    def lr_km(self, edges):
        """Length of these edges that is railway=light_rail track, in km."""
        e = np.asarray(edges, dtype=np.int64)
        return float(self.elen[e][self.lr[e]].sum()) / 1000 if e.size else 0.0

    def point_on(self, e, t):
        u, v = int(self.eu[e]), int(self.ev[e])
        (x1, y1), (x2, y2) = self.xy[u], self.xy[v]
        return (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)

    def trace(self, src, dst, km, own=None, prefer_main=True):
        """Shortest track from snap `src` to snap `dst`, in cost units: metres, times
        SLOW_COST on yard track (if prefer_main) and OFF_LINE_COST off the ways in `own` (if
        given). Returns (points, km, share on highspeed track, edges used, km on own) or None.

        prefer_main is for pieces with a passenger stop at an end. Between two freight
        points it sent the trace out of the yard onto the main line beside it, which then
        looked ridden by passenger trains and kept Zeebrugge's port sidings on line 51B."""
        def cf(e):
            c = SLOW_COST if prefer_main and self.slow[e] else 1.0
            if own is not None and int(self.ew[e]) not in own:
                c *= OFF_LINE_COST
            return c

        cap = ((km or 0) * 1000 * 1.6 + 1500) * SLOW_COST * OFF_LINE_COST
        dist, prev = {}, {}
        heap = []
        for e, t in zip(src["e"].tolist(), src["t"].tolist()):
            u, v, L = int(self.eu[e]), int(self.ev[e]), float(self.elen[e])
            for n, c in ((u, t * L * cf(e)), (v, (1 - t) * L * cf(e))):
                if c < dist.get(n, math.inf):
                    dist[n] = c
                    prev[n] = ("start", e, t)
                    heapq.heappush(heap, (c, n))
        goal = defaultdict(list)
        for e, t in zip(dst["e"].tolist(), dst["t"].tolist()):
            L = float(self.elen[e])
            goal[int(self.eu[e])].append((t * L * cf(e), e, t))
            goal[int(self.ev[e])].append(((1 - t) * L * cf(e), e, t))
        best, best_end = math.inf, None
        # both ends on one edge: straight along it
        s_on = {e: t for e, t in zip(src["e"].tolist(), src["t"].tolist())}
        for e, t in zip(dst["e"].tolist(), dst["t"].tolist()):
            if e in s_on:
                c = abs(t - s_on[e]) * float(self.elen[e]) * cf(e)
                if c < best:
                    best, best_end = c, ("same", e, s_on[e], t)
        seen = set()
        while heap:
            d, u = heapq.heappop(heap)
            if d >= best or d > cap:
                break
            if u in seen:
                continue
            seen.add(u)
            for c, e, t in goal.get(u, ()):
                if d + c < best:
                    best, best_end = d + c, ("node", u, e, t)
            for v, e in self.adj[u]:
                nd = d + float(self.elen[e]) * cf(e)
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = (u, e)
                    heapq.heappush(heap, (nd, v))
        if best_end is None:
            return None
        if best_end[0] == "same":
            _s, e, t0, t1 = best_end
            pts = [self.point_on(e, t0), self.point_on(e, t1)]
            edges = [e]
        else:
            _s, u, e_end, t_end = best_end
            nodes, edges = [u], [e_end]
            while True:
                p = prev[nodes[-1]]
                if p[0] == "start":
                    edges.append(p[1])
                    start_pt = self.point_on(p[1], p[2])
                    break
                nodes.append(p[0])
                edges.append(p[1])
            nodes.reverse()
            edges.reverse()
            pts = [start_pt] + [self.xy[n] for n in nodes] + [self.point_on(e_end, t_end)]
        # Measured and flagged on the path itself.
        seg = [dist_m(*a, *b) for a, b in zip(pts[:-1], pts[1:])]
        total = sum(seg)
        fast = mine = 0.0
        if len(edges) == len(seg):
            fast = sum(s for s, e in zip(seg, edges) if self.fast[e])
            if own is not None:
                mine = sum(s for s, e in zip(seg, edges) if int(self.ew[e]) in own)
        return pts, total / 1000, (fast / total if total else 0.0), edges, mine / 1000


# ================================================================ OSM stations

def osm_stations(stops, light_rail=False, plain=None):
    """OSM's rail stations a main-line train calls at: railway=station/halt, or a
    public_transport=station saying train=yes; never a metro, tram or light-rail stop, unless
    `light_rail` (a country's `light_rail_track`), when a light-rail station counts too.
    `plain(tags)`: the name to read instead of `name` (a country's `plain_name`: New Zealand
    names stop positions after their platform, "Maungawhau 3", "Petone Station")."""
    out = {}
    lr_ok = ("light_rail",) if light_rail else ()
    if plain is not None:
        stops = {nid: (dict(tags, name=plain(tags)) if tags.get("name") else tags, lon, lat)
                 for nid, (tags, lon, lat) in stops.items()}
    for nid, (tags, lon, lat) in stops.items():
        rw, pt = tags.get("railway"), tags.get("public_transport")
        if not tags.get("name"):
            continue
        if rw in ("station", "halt"):
            if tags.get("station") in ("subway", "light_rail", "monorail", "funicular") \
                    and tags.get("station") not in lr_ok and tags.get("train") != "yes":
                continue
            if any(tags.get(m) == "yes" for m in ("subway", "tram", "light_rail")
                   if m not in lr_ok) and tags.get("train") != "yes":
                continue
        elif not (pt == "station" and tags.get("train") == "yes"):
            continue
        out[nid] = {"name": tags["name"], "name_en": tags.get("name:en") or "",
                    "lon": lon, "lat": lat, "keys": name_variants(tags["name"])
                    | name_variants(tags.get("name:en"))}
    # A station mapped only as train stop positions (Buitenpost) is still a station: its first
    # stop position stands in for it, as build_model does, unless a station of that name is
    # within STOP_TO_STATION_M.
    by_key = defaultdict(list)
    for nid, s in out.items():
        for k in s["keys"]:
            by_key[k].append(nid)
    for nid, (tags, lon, lat) in sorted(stops.items()):
        if not tags.get("name") or tags.get("train") != "yes" or nid in out:
            continue
        if not (tags.get("railway") == "stop" or tags.get("public_transport") == "stop_position"):
            continue
        keys = name_variants(tags["name"])
        if any(dist_m(lon, lat, out[c]["lon"], out[c]["lat"]) <= STOP_TO_STATION_M
               for k in keys for c in by_key.get(k, ())):
            continue
        out[nid] = {"name": tags["name"], "name_en": tags.get("name:en") or "",
                    "lon": lon, "lat": lat, "keys": keys}
        for k in keys:
            by_key[k].append(nid)
    return out


class StationIndex:
    def __init__(self, st):
        self.ids = list(st)
        self.st = st
        self.lon = np.array([st[i]["lon"] for i in self.ids])
        self.lat = np.array([st[i]["lat"] for i in self.ids])

    def within(self, lon, lat, r):
        d = np.hypot((self.lon - lon) * math.cos(math.radians(lat)) * 111320,
                     (self.lat - lat) * 110570)
        k = np.nonzero(d <= r)[0]
        return [(float(d[i]), self.ids[i]) for i in k]


def names_match(rinf_keys, osm_keys):
    if rinf_keys & osm_keys:
        return True
    for a in rinf_keys:
        for b in osm_keys:
            if len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)):
                return True
    return False


# ================================================================ build

PARALLEL_M = 30           # a section this close to its line's others ...
PARALLEL_SHARE = 0.9      # ... for this share of its length is a second track pair


def redundant(out_secs, stop_nodes, kx, ky):
    """Sections of a line that end at a junction and lie on top of the line's other sections.

    Several RINF ids can be one public line, and some are its extra track pairs: Belgium's
    8275 is a second pair beside line 25 from Mechelen-Dijkstraat to Sint-Katelijne-Waver,
    3.6 km counted twice. A section between two stops is never dropped here."""
    from shapely import STRtree
    from shapely.geometry import LineString
    geo = {k: LineString([(x * kx, y * ky) for x, y in v["pts"]])
           for k, v in out_secs.items() if len(v["pts"]) >= 2}
    dropped = []
    for k in sorted(geo, key=lambda k: geo[k].length):
        if k[0] in stop_nodes and k[1] in stop_nodes:
            continue
        others = [g for kk, g in geo.items() if kk != k and kk not in dropped]
        g = geo[k]
        if not others or g.length <= 0:
            continue
        tree = STRtree(others)
        n = max(2, int(g.length // 50) + 1)
        pts = [g.interpolate(i / (n - 1), normalized=True) for i in range(n)]
        near = sum(1 for p in pts if others[int(tree.nearest(p))].distance(p) <= PARALLEL_M)
        if near >= PARALLEL_SHARE * n:
            dropped.append(k)
    return dropped


OSM_STOP_M = 120          # an OSM station a train route stops at, this close to a traced section,
OSM_STOP_BAND_M = 40      # ... and no further than this beyond its own nearest track,
OSM_STOP_END_M = 400      # ... and not this close to either end, is a stop on it (`osm_stops`)


def served_stations(rels, stops, ost, also=None, plain=None):
    """OSM stations a passenger train route stops at: the route's stop node itself, or a
    station of its name within STOP_TO_STATION_M of it. `also(tags)`: another route that
    counts (a country's `osm_stop_route`); `plain(tags)` as in osm_stations."""
    by_key = defaultdict(list)
    for sid, s in ost.items():
        for k in s["keys"]:
            by_key[k].append(sid)
    out = set()
    for tags, members in rels.values():
        if tags.get("type") != "route" or (tags.get("route") != "train"
                                           and not (also and also(tags))):
            continue
        for ty, ref, role in members:
            if ty != "n" or not role.startswith(("stop", "platform")):
                continue
            if ref in ost:
                out.add(ref)
                continue
            rec = stops.get(ref)
            if rec is None or not rec[0].get("name"):
                continue
            t, lon, lat = rec
            for k in name_variants(plain(t) if plain is not None else t.get("name")):
                for sid in by_key.get(k, ()):
                    if dist_m(lon, lat, ost[sid]["lon"], ost[sid]["lat"]) <= STOP_TO_STATION_M:
                        out.add(sid)
    return out


ROUTE_NEAR_M = 15         # track this close to a train route's ways is that route's track
ROUTE_SHARE_CUT = 0.5     # a junction-ended section this much on route track may be cut too


def route_track_share(ways, rels, coords, kx, ky, also=None):
    """A function giving the share of a section's trace (in kx/ky metres) that lies on ways an
    OSM route=train relation has as members: the question drop_unridden_sections asks before
    keeping a junction-ended section, asked here before cutting one at OSM stops."""
    from shapely import STRtree
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    wids = {r for tags, ms in rels.values()
            if tags.get("type") == "route" and (tags.get("route") == "train"
                                                or (also is not None and also(tags)))
            for ty, r, _ in ms if ty == "w"}
    geoms = []
    for w in wids:
        rec = ways.get(w)
        if rec is None:
            continue
        pts = [coords.get(int(n)) for n in rec[1]]
        pts = [(x * kx, y * ky) for x, y in (p for p in pts if p is not None)]
        if len(pts) >= 2:
            geoms.append(LineString(pts))
    if not geoms:
        return lambda g: 0.0
    tree = STRtree(geoms)

    def share(g):
        if not g.length:
            return 0.0
        idx = tree.query(g.buffer(ROUTE_NEAR_M))
        if not len(idx):
            return 0.0
        near = unary_union([geoms[i] for i in idx]).buffer(ROUTE_NEAR_M)
        return g.intersection(near).length / g.length
    return share


def split_at_osm_stops(out_secs, cands, kx, ky, stop_nodes, route_share=None):
    """Cut each traced section where a served OSM station lies on it (`osm_stops`).

    For registers that list only junction stations: Latvia's RINF has no point between Rīga
    and Aizkraukle (82 km, 15 stops), Estonia's lists about one Elron stop in three, and
    Bulgaria's lists stations but no halts (265 passenger points for about 650 places trains
    call). RINF's length is shared out over the pieces in proportion to their traced length.
    `cands` is [(node id, lon, lat, distance to its own nearest track in metres)].

    A section between two stops is always cut. A junction-ended one is cut only where
    ROUTE_SHARE_CUT of it lies on OSM train-route track (`route_share`): otherwise cutting it
    would turn part of it into a stop-to-stop piece, which build_model never questions, and
    keep freight track (Estonia's Ülemiste - Blokkpost bypass). Latvia needs the junction-ended
    case: its lines run to a border point, and Rīga - Valga's last section, with Valmiera on
    it, ends at the frontier."""
    from shapely.geometry import LineString, Point
    from shapely.ops import substring
    used = {n for key in out_secs for n in key}
    new, added = {}, set()
    for key, v in out_secs.items():
        pts = v["pts"]
        if len(pts) < 2 or not v["km"]:
            new[key] = v
            continue
        g = LineString([(x * kx, y * ky) for x, y in pts])
        if ((key[0] not in stop_nodes or key[1] not in stop_nodes)
                and (route_share is None or route_share(g) < ROUTE_SHARE_CUT)):
            new[key] = v
            continue
        L = g.length
        x0, y0, x1, y1 = g.bounds
        cuts = {}
        for n, lon, lat, d0 in cands:
            if n in used:
                continue
            px, py = lon * kx, lat * ky
            if not (x0 - OSM_STOP_M <= px <= x1 + OSM_STOP_M
                    and y0 - OSM_STOP_M <= py <= y1 + OSM_STOP_M):
                continue
            p = Point(px, py)
            d = g.distance(p)
            if d > OSM_STOP_M or d > d0 + OSM_STOP_BAND_M:
                continue
            t = g.project(p)
            if OSM_STOP_END_M <= t <= L - OSM_STOP_END_M:
                cuts[n] = t
        if not cuts:
            new[key] = v
            continue
        order = sorted(cuts, key=cuts.get)
        nodes = [key[0]] + order + [key[1]]
        ts = [0.0] + [cuts[n] for n in order] + [L]
        for i in range(len(nodes) - 1):
            piece = substring(g, ts[i], ts[i + 1])
            ppts = [(x / kx, y / ky) for x, y in piece.coords]
            na, nb = nodes[i], nodes[i + 1]
            k2 = (na, nb) if na <= nb else (nb, na)
            if k2[0] != na:
                ppts = ppts[::-1]
            new[k2] = {"km": path_km(ppts), "pts": ppts,
                       "chain": v["chain"] * (ts[i + 1] - ts[i]) / L if L else 0.0,
                       "how": v["how"], "fast": v["fast"],
                       # light-rail km shared out like the chain, or a cut S-bane line
                       # reads as rail (`light_rail_track`)
                       "lr": v.get("lr", 0.0) * (ts[i + 1] - ts[i]) / L if L else 0.0}
        added |= set(order)
    return new, added


DEBUG = []                # fill_holes' refused candidates, for RINF_FILL_DEBUG
LINK_M = 50               # a 0 km register section between a stop and a junction this close is a link
HOLE_MAX_KM = 25.0       # `fill_holes`: a gap in a line's RINF sections is filled over at most
HOLE_DETOUR = 1.5         # ... this much track, no more than this times crow-fly
HOLE_PLUS_KM = 1.0        # ... plus this,
HOLE_OWN = 0.9            # ... at least this share of it on the line's own OSM relation's ways,
HOLE_OVERLAP = 0.3        # ... and at most this share of it on the line's other sections
HOLE_END_M = 300          # (the overlap is not counted this close to either end)
HOLE_OTHER = 0.3          # ... and at most this share on track other RINF lines were traced over
FILL_SAME_NAME_M = 1000   # a fill's OSM stop this close to a register stop of its name is that stop


def _union(keys):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in keys:
        parent[find(a)] = find(b)
    return find


def fill_holes(out_secs, own, snap_node, pos_node, track, edge_lines=None, mine=frozenset()):
    """Join the pieces of a line whose RINF sections do not connect, over its own OSM track.

    RINF leaves stretches out of a line: ÖBB's 222 01, the Tauernbahn, has no section from
    Mühldorf-Möllbrücke to Pusarnitz-Süd, from Markt Paternion to Paternion-Feistritz, or from
    Gummern to Villach (none under any other id either), so the line came out in four pieces
    that trains run straight through. A loose end of one piece (a node with one section) is
    joined to a loose end of another where the track between them is the line's own: the
    shortest trace preferring the ways of the line's OSM relation (`own`, from pass 2), at most
    HOLE_MAX_KM and HOLE_DETOUR x crow-fly + HOLE_PLUS_KM, at least HOLE_OWN of it on those
    ways, and at most HOLE_OVERLAP of it on the line's other sections (so it does not run back
    over a piece between), and at most HOLE_OTHER of it on track another RINF line's sections
    were traced over (`edge_lines`: edge -> RINF ids; `mine`, this line's): there the line
    runs over another line's track, which is a bridge's job, and a fill would put a second
    line on track one already owns (Wien Penzing - Hütteldorf, 48%, is refused).
    The nearest pair is joined first, then again until nothing joins.
    A line with no numbered OSM relation is never filled: nothing else says the track between
    is the line's. Returns [(a, b, km, crow, own share)] and adds each as a section, its
    "chain" (RINF km) the traced length, "how" "hole"."""
    from shapely.geometry import LineString, Point
    filled = []
    if not own:
        return filled
    kx, ky = track.kx, track.ky
    tried = set()
    while True:
        find = _union(out_secs.keys())
        deg = Counter()
        for a, b in out_secs:
            deg[a] += 1
            deg[b] += 1
        if len({find(n) for n in deg}) < 2:
            break
        ends = sorted(n for n in deg if deg[n] == 1 and pos_node(n) is not None)
        cand = []
        for i, a in enumerate(ends):
            for b in ends[i + 1:]:
                if find(a) == find(b) or (a, b) in tried:
                    continue
                crow = dist_m(*pos_node(a), *pos_node(b)) / 1000
                if crow <= HOLE_MAX_KM:
                    cand.append((crow, a, b))
        geo = [LineString([(x * kx, y * ky) for x, y in v["pts"]])
               for v in out_secs.values() if len(v["pts"]) >= 2]
        best = None
        for crow, a, b in sorted(cand):
            if best is not None and crow >= best[0]:
                break
            tried.add((a, b))
            sa, sb = snap_node(a), snap_node(b)
            if sa is None or sb is None:
                continue
            got = track.trace(sa, sb, HOLE_DETOUR * crow + HOLE_PLUS_KM, own, prefer_main=True)
            if got is None:
                DEBUG.append((a, b, crow, "no path"))
                continue
            pts, km, fast, edges, okm = got
            if (km > HOLE_MAX_KM or km > HOLE_DETOUR * crow + HOLE_PLUS_KM
                    or km < SHORT * crow - 0.1 or okm < HOLE_OWN * km):
                DEBUG.append((a, b, crow, f"km {km:.2f}, own {okm / km if km else 0:.0%}"))
                continue
            other = 0.0
            if edge_lines:
                other = sum(float(track.elen[e]) for e in edges
                            if edge_lines.get(int(e)) and not (edge_lines[int(e)] & mine)) / 1000
                if other > HOLE_OTHER * km:
                    DEBUG.append((a, b, crow, f"other line's track {other / km:.0%}"))
                    continue
            seq = [pos_node(a)] + pts + [pos_node(b)]
            g = LineString([(x * kx, y * ky) for x, y in seq])
            if g.length > 2 * HOLE_END_M and geo:
                n = max(2, int(g.length // 50) + 1)
                ts = [g.length * i / (n - 1) for i in range(n)]
                mid = [g.interpolate(t) for t in ts if HOLE_END_M <= t <= g.length - HOLE_END_M]
                on = sum(1 for p in mid if min(h.distance(p) for h in geo) <= PARALLEL_M)
                if mid and on > HOLE_OVERLAP * len(mid):
                    DEBUG.append((a, b, crow, f"overlap {on / len(mid):.0%}"))
                    continue
            if best is None or km < best[0]:
                best = (km, a, b, seq, fast, crow, okm / km if km else 0.0, track.lr_km(edges),
                        other / km if km else 0.0,
                        sorted({x for e in edges for x in edge_lines.get(int(e), ())} - mine)
                        if edge_lines else [])
        if best is None:
            break
        km, a, b, seq, fast, crow, own_share, lr, other_share, others = best
        if os.environ.get("RINF_FILL_DEBUG"):
            DEBUG.append((a, b, crow, f"FILLED: other lines' track {other_share:.0%} "
                                      f"({', '.join(others[:6])})"))
        key = (a, b) if a <= b else (b, a)
        if key[0] != a:
            seq = seq[::-1]
        out_secs[key] = {"km": km, "pts": seq, "chain": km, "how": "hole",
                         "fast": fast >= 0.5, "lr": lr}
        filled.append((a, b, km, crow, own_share))
    return filled


def build(path, log, ref_date=None):
    cc = Path(path).name
    conf = country(cc)
    GROUPS.clear()
    LAST_CC[0] = cc
    secs, points = load_rinf(path, log, ref_date)
    # A country's own corrections to RINF's data, made in place before anything reads it
    # (Slovakia: stations typed 110, coordinates kilometres off, lengths in metres, one ŽSR
    # id spanning several timetable lines; rinf_countries/sk.py). Returns lines to log.
    fix = conf.get("fix")
    if fix:
        for msg in fix(secs, points) or ():
            log(f"RINF: fix: {msg}")
    skip = conf.get("skip_line")
    if skip:
        n0, ids0 = len(secs), {s["base"] for s in secs}
        secs = [s for s in secs if not skip(s["base"])]
        log(f"RINF: skip_line leaves out {len(ids0 - {s['base'] for s in secs})} line ids "
            f"({n0 - len(secs)} sections)")

    import build_model as bm
    ways, rels, stops, cid, cx, cy = bm.load(cc, log)
    coords = bm.Coords(cid, cx, cy)
    infra = {}
    ip = ROOT / "data" / "proc" / cc / "infra.pkl"
    if ip.exists():
        with open(ip, "rb") as f:
            infra = pickle.load(f)
    wd = load_wikidata(path, frozenset(infra), conf.get("generic_label"),
                       (conf.get("langs") or ["en"])[0])
    light_rail = bool(conf.get("light_rail_track"))
    track = Track(ways, coords, log, light_rail)

    # --- which point is which station (docstring, 2)
    # `plain_name(tags)`: a rail stop's name as the country reads it, without its platform
    # (New Zealand; unset, names as tagged).
    plain = conf.get("plain_name")
    ost = osm_stations(stops, light_rail, plain)
    sidx = StationIndex(ost)

    # A point with no coordinate at all (all 93 of the Steiermärkische Landesbahnen's in
    # Austria) is placed at the OSM station of exactly its name, if there is only one place
    # of that name. Junction points without one stay unplaced; the end-to-end retrace of the
    # merged section then runs past them.
    # Where the name is not unique in the country ("Himberg" is near Vienna and on the
    # Übelbach line), the one nearest a neighbour on the line that is placed already wins;
    # repeated, so a chain of such points fills in from its placed end.
    nbr = defaultdict(set)
    for s in secs:
        nbr[s["a"]].add(s["b"])
        nbr[s["b"]].add(s["a"])
    todo = [op for op, p in points.items() if "lon" not in p]
    cands_of = {}
    for op in todo:
        keys = name_variants(points[op].get("name"))
        # An exact name first: by prefix alone "Wulkaprodersdorf" also finds the halt
        # "Wulkaprodersdorf Rathausgasse".
        cands_of[op] = ([sid for sid, s in ost.items() if keys & s["keys"]]
                        or [sid for sid, s in ost.items() if names_match(keys, s["keys"])])
    n_placed = 0
    for _round in range(4):
        for op in todo:
            p = points[op]
            cands = cands_of[op]
            if "lon" in p or not cands:
                continue
            spots = [(ost[h]["lon"], ost[h]["lat"]) for h in cands]
            near = [(points[q]["lon"], points[q]["lat"]) for q in nbr[op] if "lon" in points[q]]
            if near:
                d, spot = min((min(dist_m(*sp, *q) for q in near), sp) for sp in spots)
                if d > NEIGHBOUR_M:
                    continue
            elif all(dist_m(*spots[0], *q) <= NAME_M for q in spots):
                spot = spots[0]
            else:
                continue
            p["lon"], p["lat"] = spot
            n_placed += 1
    if todo:
        log(f"RINF: {len(todo)} points have no coordinate; {n_placed} placed at the OSM "
            f"station of their name, {len(todo) - n_placed} left unplaced")

    stop_of = {}                                   # op uri -> OSM station node id
    unmatched, blind = [], []
    # `stop_name(point)`: a national station list's say on a RINF point (Finland: Fintraffic's
    # passenger flag and VR's name, joined on the UIC code). A name makes the point a stop
    # under that name, whatever RINF types it, at the OSM station of exactly that name if one
    # is near, else the nearest within BLIND_M, else at RINF's own coordinate; False makes it
    # never a stop; None leaves the rule below as it is. A stop placed at RINF's coordinate
    # is keyed "r<uopid>" in `ost`, so do not combine this hook with `osm_stops`.
    stop_name = conf.get("stop_name") or (lambda _p: None)
    n_listed = n_vetoed = n_at_rinf = 0
    for op, p in points.items():
        listed = stop_name(p) if "lon" in p else None
        if listed is False:
            n_vetoed += p.get("type") in PASSENGER_TYPES
            continue
        if (p.get("type") not in PASSENGER_TYPES and not listed
                and p.get("name") not in conf.get("stop_names", ())) or "lon" not in p:
            continue
        keys = name_variants(p.get("name")) | name_variants(listed or "")
        near = sorted(sidx.within(p["lon"], p["lat"], conf.get("name_m", NAME_M)))
        if listed:
            # Exact names only: by prefix Fintraffic's "Kotkan satama" (the harbour station)
            # took OSM's "Kotka" 460 m away, and Kotka lost its own name.
            n_listed += 1
            hit = next((sid for d, sid in near if keys & ost[sid]["keys"]), None)
        else:
            hit = next((sid for d, sid in near if names_match(keys, ost[sid]["keys"])), None)
        if hit is None:
            close = [(d, sid) for d, sid in near if d <= BLIND_M]
            if close:
                hit = close[0][1]
                blind.append((p.get("name"), ost[hit]["name"], round(close[0][0])))
        if hit is None and listed:
            hit = f"r{p.get('uopid') or tail(op)}"
            ost[hit] = {"name": listed, "name_en": "", "lon": p["lon"], "lat": p["lat"],
                        "keys": name_variants(listed)}
            n_at_rinf += 1
        elif listed:
            ost[hit]["name"] = listed
        if hit is None:
            unmatched.append(p.get("name"))
        else:
            stop_of[op] = hit
    if conf.get("stop_name"):
        log(f"RINF: stop_name made {n_listed} points stops ({n_at_rinf} with no OSM station, "
            f"placed at RINF's coordinate) and vetoed {n_vetoed} passenger-typed ones")
    log(f"RINF: {sum(1 for p in points.values() if p.get('type') in PASSENGER_TYPES)} "
        f"passenger-typed points; {len(stop_of)} are an OSM station "
        f"({len(set(stop_of.values()))} distinct), {len(blind)} of those by distance alone; "
        f"{len(unmatched)} have no OSM station and become junctions")
    for a, b, d in blind:
        log(f"    by distance: {a} -> {b} ({d} m)")
    if unmatched:
        log(f"    no OSM station: {', '.join(sorted(n or '?' for n in unmatched))}")

    # `osm_stops`: OSM stations a train route stops at, as candidate stops on traced sections;
    # "all" takes every OSM rail station instead, for a country with few route relations
    # (Bulgaria has 16).
    osm_stop_cands, extra_nodes = [], set()
    if conf.get("osm_stops"):
        # `osm_stop_route(tags)`: another route whose stops count (Denmark: the S-tog lines
        # are route=light_rail, and RINF lists about one S-bane halt in three).
        served = (set(ost) if conf["osm_stops"] == "all"
                  else served_stations(rels, stops, ost, conf.get("osm_stop_route"), plain))
        # `osm_stop_extra(name)`: an OSM station no route lists as a stop that counts anyway,
        # where a national list says trains call there (Denmark: the timetable's rail stations;
        # OSM's Danish routes leave out Brejning, Gelsted, Jerne...).
        if conf.get("osm_stop_extra") and conf["osm_stops"] != "all":
            n0 = len(served)
            served |= {sid for sid, s in ost.items() if conf["osm_stop_extra"](s["name"])}
            log(f"RINF: osm_stop_extra: {len(served) - n0} more OSM stations a list names")
        for sid in sorted(served):
            s = ost[sid]
            sn = track.snap(s["lon"], s["lat"])
            if sn is not None and sn["d0"] <= OSM_STOP_M:
                osm_stop_cands.append((f"e{sid}", s["lon"], s["lat"], sn["d0"]))
        log(f"RINF: osm_stops: {len(served)} OSM stations a train route stops at, "
            f"{len(osm_stop_cands)} of them within {OSM_STOP_M} m of track")
        on_route = route_track_share(ways, rels, coords, track.kx, track.ky,
                                     conf.get("osm_stop_route"))
    else:
        on_route = None

    def node_of(op):
        if op in stop_of:
            return f"e{stop_of[op]}"
        return f"e{points.get(op, {}).get('uopid') or tail(op)}"

    def pos_of(op):
        if op in stop_of:
            s = ost[stop_of[op]]
            return s["lon"], s["lat"]
        p = points.get(op, {})
        return (p["lon"], p["lat"]) if "lon" in p else None

    snaps = {}

    def snap_of(op):
        if op not in snaps:
            pos = pos_of(op)
            snaps[op] = track.snap(*pos) if pos else None
        return snaps[op]

    # --- pass 1: trace every section unbiased (docstring, 4)
    def trace_all(own_of):
        out = {}
        for s in secs:
            sa, sb = snap_of(s["a"]), snap_of(s["b"])
            if sa is None or sb is None:
                out[s["sol"]] = ("unplaced", None)
                continue
            own = own_of.get(s["line"])
            got = track.trace(sa, sb, s["km"], own,
                              prefer_main=s["a"] in stop_of or s["b"] in stop_of)
            if got is None:
                out[s["sol"]] = ("no path", None)
                continue
            # Judged on the merged section, not here: a switch or yard point is often placed
            # a few hundred metres from where its chainage is, which makes one piece read
            # short and the next long by the same amount.
            out[s["sol"]] = ("ok", got)
        return out

    tol_abs = conf.get("tol_abs", TOL_ABS)
    # `km_floor`: the register's section lengths leave out the station areas, so they are a
    # floor on the track's length, not its length (Denmark: Sorø - Slagelse is 12.3 km in RINF
    # for 13.7 km as the crow flies; København - Korsør sums to 78 km for 111). A trace is then
    # judged against that floor and the crow-fly distance between its ends, and no chainage is
    # shipped (km_official), since it would only measure the station areas.
    km_floor = bool(conf.get("km_floor"))

    def within_tol(km, chain, crow=None):
        if km_floor and crow is not None:
            return (KM_FLOOR_SHARE * chain - 0.2 <= km
                    <= max(DETOUR * crow + 0.5, (1 + TOL_REL) * chain + KM_FLOOR_SLACK))
        return abs(km - chain) <= TOL_REL * chain + tol_abs

    t0 = time.time()
    tr1 = trace_all({})
    log(f"RINF: pass 1 traced {len(secs)} sections in {time.time() - t0:.0f} s: "
        f"{Counter(v[0] for v in tr1.values())}")

    # --- names (docstring, 5)
    osm_ref = conf.get("osm_ref", osm_ref_default)
    # A country can read a relation's number and name from all its tags instead, and drop a
    # relation outright by returning None: Czech OSM carries the timetable number and SŽ's own
    # line number on the same `ref` key, told apart only by the name (rinf_countries/cz.py).
    osm_rel = conf.get("osm_rel") or (lambda t: (osm_ref(t.get("ref")), t.get("name")))
    rel_ref, rel_ways, rel_name = {}, {}, {}
    way_rels = defaultdict(set)
    for rid, (tags, members) in infra.items():
        got = osm_rel(tags)
        if not got:
            continue
        r, nm = got
        if not r and not nm:
            continue
        if r:
            rel_ref[rid] = r                  # only these number lines
        rel_name[rid] = nm or ""
        ws = {ref for ty, ref, _role in members if ty == "w"}
        rel_ways[rid] = ws
        for w in ws:
            way_rels[w].add(rid)
    by_line = defaultdict(list)
    for s in secs:
        by_line[s["line"]].append(s)
    base_of = {s["line"]: s["base"] for s in secs}
    ref_rule = conf.get("ref") or (lambda _lid: None)
    # Numbers are compared and grouped as keys (no spaces, upper case): OSM writes ÖBB's
    # "101 04" where the RINF id is 10104. `ref_display` writes a key back out at the end.
    rule = {lid: refkey(ref_rule(base_of[lid])) for lid in by_line}
    # A rule's own spelling is how the number is written back ("Hlg-Nscg", not "HLG-NSCG").
    spelled = {refkey(ref_rule(base_of[lid])): ref_rule(base_of[lid])
               for lid in by_line if ref_rule(base_of[lid])}
    rel_ref = {r: refkey(v) for r, v in rel_ref.items()}

    def on_scores(lid):
        """Share of the line's traced length that runs ON each relation's own ways."""
        on, total = Counter(), 0.0
        for s in by_line[lid]:
            st, got = tr1[s["sol"]]
            if st != "ok":
                continue
            pts, edges = got[0], got[3]
            if len(edges) != len(pts) - 1:
                continue
            for (a, b), e in zip(zip(pts[:-1], pts[1:]), edges):
                m = dist_m(*a, *b)
                total += m
                for r in way_rels.get(int(track.ew[e]), ()):
                    on[r] += m
        return {r: v / total for r, v in on.items()} if total else {}

    def rel_scores(lid):
        """Share of the line's traced length that lies within REL_NEAR_M of each relation."""
        hits, n = Counter(), 0
        for s in by_line[lid]:
            st, got = tr1[s["sol"]]
            if st != "ok":
                continue
            pts = got[0]
            for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
                seg = dist_m(x1, y1, x2, y2)
                k = max(1, int(seg // 100))
                for j in range(k):
                    f = (j + 0.5) / k
                    e, _t, _d = track.near(x1 + (x2 - x1) * f, y1 + (y2 - y1) * f, REL_NEAR_M)
                    rs = set()
                    for w in track.ew[e].tolist():
                        rs |= way_rels.get(w, set())
                    n += 1
                    for r in rs:
                        hits[r] += 1
        return {r: c / n for r, c in hits.items()} if n else {}

    # The country's rule is trusted where the traced line lies on a relation of that number,
    # or where it lies on no numbered relation at all. Where it lies on others instead, the
    # rule is wrong for that id (Belgium's 0580 is a 0.7 km curve at Ledeberg, and line 58
    # is 0582), and the relation decides. A number the rule gave to one id is taken by
    # another only when nothing else fits, and then the two are one line: that keeps 36N
    # (0366) off line 36 on a shared four-track corridor, yet puts 0582 on line 58 beside 0580.
    ref_of, rel_of, how = {}, {}, Counter()
    scores = {lid: rel_scores(lid) for lid in by_line}
    # `no_ref` keeps ids off every line number and relation name. Czech siding leads (SŽ's
    # -90..-99 ids and the 0xx private-siding ids) lie within REL_NEAR_M of their line's
    # relation; joined to it, each siding junction became a branch point and build_model cut
    # the sections either side of it out of the line wherever no OSM route runs (020 lost
    # Choceň - Újezd u Chocně, 6 km).
    no_ref = conf.get("no_ref") or (lambda _lid: False)
    for lid in by_line:
        if no_ref(base_of[lid]):
            scores[lid], rule[lid] = {}, None
    cands = {lid: sorted(((v, r) for r, v in scores[lid].items()
                          if v >= REL_SHARE and r in rel_ref), reverse=True) for lid in by_line}
    claimed = set()
    fixed = conf.get("fixed", {})
    for lid in sorted(by_line):
        f = refkey(fixed.get(base_of[lid]))
        if f:
            ref_of[lid] = f
            same = [r for r, rr in rel_ref.items() if rr == f]
            if same:
                rel_of[lid] = max(same, key=lambda r: len(rel_ways[r]))
            claimed.add(f)
            how["fixed"] += 1
    for lid in sorted(by_line):
        r0 = rule[lid]
        if not r0 or lid in ref_of:
            continue
        match = [r for _v, r in cands[lid] if rel_ref[r] == r0]
        if match or not cands[lid] or conf.get("rule_certain"):
            ref_of[lid] = r0
            same = match or [r for r, rr in rel_ref.items() if rr == r0]
            if same:
                rel_of[lid] = max(same, key=lambda r: len(rel_ways[r]))
            claimed.add(r0)
            how["rule, confirmed" if match else
                "rule, no relation near" if not cands[lid] else "rule, certain"] += 1
    for lid in sorted(by_line):
        if lid in ref_of:
            continue
        if not cands[lid]:
            how["none"] += 1
            continue
        free = [(v, r) for v, r in cands[lid] if rel_ref[r] not in claimed]
        if not free:
            # Joining a number another id already has needs the trace to run ON that
            # relation's ways, not beside them: yard and connecting tracks at Schaerbeek lie
            # within REL_NEAR_M of line 25 and were adding 8 km to it.
            on = on_scores(lid)
            free = sorted(((on.get(r, 0.0), r) for _v, r in cands[lid]
                           if on.get(r, 0.0) >= REL_SHARE), reverse=True)
            if not free:
                how["none (only near a claimed number)"] += 1
                continue
            how["osm relation, shared"] += 1
        else:
            how["osm relation"] += 1
        v, r = free[0]
        rel_of[lid], ref_of[lid] = r, rel_ref[r]
        if rule[lid]:
            log(f"    {lid}: the rule says {rule[lid]}, the track says {rel_ref[r]}")
    log(f"RINF: public line numbers for {len(by_line)} RINF line ids: {dict(how)}")
    for lid in sorted(by_line):
        if not rule[lid] and lid in ref_of:
            log(f"    {lid} -> {ref_of[lid]} (OSM relation {rel_of.get(lid)})")
    # A line with no number can still lie on a NAMED relation (Austria's Landesbahnen).
    named_rel = {}
    for lid in by_line:
        if lid in ref_of:
            continue
        best = max(((v, r) for r, v in scores[lid].items() if rel_name.get(r)), default=None)
        if best and best[0] >= REL_SHARE:
            named_rel[lid] = best[1]

    # --- pass 2: retrace lines with a relation, preferring its own track
    ways_of_ref = defaultdict(set)
    for r, rr in rel_ref.items():
        ways_of_ref[rr] |= rel_ways[r]
    own_of = {lid: ways_of_ref[ref] for lid, ref in ref_of.items() if ways_of_ref.get(ref)}
    # `way_line(tags)`: the line a track way's own tags name, where OSM names each way for its
    # line (Denmark: "Vestbanen" beside the S-bane's "Høje Taastrup-banen"). Those ways are the
    # line's own track in pass 2, as a numbered relation's are.
    way_line = conf.get("way_line")
    if way_line:
        named_ways = defaultdict(set)
        for wid, tags in track.way_tags.items():
            nm = way_line(tags)
            if nm:
                named_ways[nm].add(wid)
        n_named = 0
        for lid in by_line:
            ws = named_ways.get(base_of[lid])
            if ws:
                own_of[lid] = own_of.get(lid, set()) | ws
                n_named += 1
        log(f"RINF: way_line: {sum(len(v) for v in named_ways.values())} track ways name "
            f"{len(named_ways)} lines; {n_named} line ids prefer them")
    t0 = time.time()
    tr = trace_all(own_of)
    log(f"RINF: pass 2 traced in {time.time() - t0:.0f} s: {Counter(v[0] for v in tr.values())}")
    # A retrace that fails where pass 1 succeeded keeps pass 1's.
    n_back = 0
    for sol, (st, got) in tr.items():
        if st != "ok" and tr1[sol][0] == "ok":
            tr[sol] = tr1[sol]
            n_back += 1
    if n_back:
        log(f"RINF: {n_back} sections kept their unbiased trace")

    for s in secs:
        st = tr[s["sol"]][0]
        if st != "ok":
            a, b = points.get(s["a"], {}), points.get(s["b"], {})
            log(f"  untraceable: {s['line']:>5} {a.get('name')} -> {b.get('name')}: {st}")

    # --- group RINF ids into public lines, and cut each into sections (docstring, 3)
    groups = defaultdict(list)
    for lid in by_line:
        groups[("ref", ref_of[lid]) if lid in ref_of else ("id", lid)].append(lid)

    im_name = conf.get("im", {})
    stations, lines, geoms = {}, [], {}
    n_rejected_secs = n_direct = n_parallel = 0
    km_parallel = 0.0
    rejects, length_off = [], []
    node_name = {}
    for op, p in points.items():
        n = node_of(op)
        node_name.setdefault(n, ost[stop_of[op]]["name"] if op in stop_of
                             else display_name(p.get("name")))

    def stations_name(n):
        return node_name.get(n, n)

    by_uopid = {(p.get("uopid") or "").upper(): p for p in points.values()}

    def uop_name(uopid):
        p = by_uopid.get((uopid or "").upper())
        return display_name(p.get("name")) if p else None
    # Where another line's sections meet this one, a section ends there too, even if the point
    # has only two neighbours on this line: otherwise a ridden stretch merged with a freight
    # one on the other side of the junction is judged, and dropped, as one (Portugal's Linha
    # do Sul at Bifurcação de Águas de Moura-Sul lost 12.4 km of Intercidades track). A
    # per-country flag for now: it makes more junction-ended sections answer to OSM's routes.
    cut_at_junctions = conf.get("cut_at_junctions", False)
    # ... or a set of RINF point names / uopids: cut only there (Spain, whose branch lines end
    # at a junction merged away inside the main line, so the timetable check found no path)
    cut_pts = None if cut_at_junctions is True or not cut_at_junctions else {
        op for op, p in points.items()
        if p.get("name") in cut_at_junctions or p.get("uopid") in cut_at_junctions}
    net_nbrs = defaultdict(set)
    if cut_at_junctions:
        for s in secs:
            na, nb = node_of(s["a"]), node_of(s["b"])
            if na != nb:
                if cut_pts is None or s["a"] in cut_pts:
                    net_nbrs[na].add(nb)
                if cut_pts is None or s["b"] in cut_pts:
                    net_nbrs[nb].add(na)
    # `direct_near_m`: an end-to-end retrace (below) is believed only if every placed point
    # of the merged section lies within this many metres of it (Russia). Unset, no test.
    direct_near_m = conf.get("direct_near_m")
    n_direct_far = 0
    floor_km = [0.0]                              # `km_floor`: RINF km of lines shipped without it

    def passes_near(pts, chain_secs, r):
        from shapely.geometry import LineString, Point
        if len(pts) < 2:
            return True
        g = LineString([(x * track.kx, y * track.ky) for x, y in pts])
        for s in chain_secs:
            for op in (s["a"], s["b"]):
                p = pos_of(op)
                if p and g.distance(Point(p[0] * track.kx, p[1] * track.ky)) > r:
                    return False
        return True

    # `fill_holes: True`: join a line's pieces over its own OSM track (fill_holes). Its stops
    # there are the OSM stations a train route stops at, found once, when first needed.
    fill_on = bool(conf.get("fill_holes")) or cc in os.environ.get("RINF_FILL_HOLES", "").split(",")
    n_hole, km_hole = 0, 0.0
    lazy = {}

    def fill_cands():
        if "cands" not in lazy:
            served = served_stations(rels, stops, ost, conf.get("osm_stop_route"), plain)
            # An OSM station of a register stop's name near it is that stop again, not a new
            # one: OSM has two "Wien Hauptbahnhof" nodes 230 m apart, and a fill near the
            # car-train terminal made the second a station of its own, taking 30 lines.
            named = defaultdict(list)
            for sid0 in set(stop_of.values()):
                for k in ost[sid0]["keys"]:
                    named[k].append((ost[sid0]["lon"], ost[sid0]["lat"]))
            lazy["cands"] = []
            for sid in sorted(served):
                s = ost[sid]
                if any(dist_m(s["lon"], s["lat"], *p) <= FILL_SAME_NAME_M
                       for k in s["keys"] for p in named.get(k, ())):
                    continue
                sn = track.snap(s["lon"], s["lat"])
                if sn is not None and sn["d0"] <= OSM_STOP_M:
                    lazy["cands"].append((f"e{sid}", s["lon"], s["lat"], sn["d0"]))
        return lazy["cands"]

    def fill_route_share():
        if "share" not in lazy:
            lazy["share"] = route_track_share(ways, rels, coords, track.kx, track.ky,
                                              conf.get("osm_stop_route"))
        return lazy["share"]

    def edge_lines():
        # the RINF ids whose traced sections run over each track edge, for fill_holes' test
        # that a fill is not another line's track
        if "edges" not in lazy:
            el = defaultdict(set)
            for s in secs:
                st_, got_ = tr[s["sol"]]
                if st_ == "ok":
                    for e in got_[3]:
                        el[int(e)].add(s["line"])
            lazy["edges"] = el
        return lazy["edges"]

    for gkey, lids in sorted(groups.items()):
        pieces = []                                   # (node a, node b, sol, op a, op b)
        for lid in lids:
            for s in by_line[lid]:
                na, nb = node_of(s["a"]), node_of(s["b"])
                if na != nb:
                    pieces.append((na, nb, s))
        if not pieces:
            continue
        stop_nodes = {node_of(op) for s in (p[2] for p in pieces) for op in (s["a"], s["b"])
                      if op in stop_of}
        nbrs = defaultdict(set)
        inc = defaultdict(list)
        seen_pair = set()
        for na, nb, s in pieces:
            k = (na, nb) if na <= nb else (nb, na)
            if k in seen_pair:
                continue                              # the same pair on two ids (line 0)
            seen_pair.add(k)
            nbrs[na].add(nb)
            nbrs[nb].add(na)
            inc[na].append((nb, s))
            inc[nb].append((na, s))
        at = {n for n in nbrs if n in stop_nodes or len(nbrs[n]) != 2
              or len(net_nbrs[n]) > 2}
        if not at:
            at = {min(nbrs)}                          # a ring with no stop: cut it anywhere
        # walk from each section end along each incident piece to the next section end
        sections = {}
        for start in sorted(at):
            for nxt, s in inc[start]:
                chain_nodes, chain_secs = [start, nxt], [s]
                prev_n, cur = start, nxt
                while cur not in at:
                    step = [(n, sx) for n, sx in inc[cur] if sx is not chain_secs[-1]]
                    if not step:
                        break
                    n2, s2 = step[0]
                    chain_nodes.append(n2)
                    chain_secs.append(s2)
                    prev_n, cur = cur, n2
                    if len(chain_nodes) > 10000:
                        break
                if cur not in at or cur == start:
                    continue
                a, b = start, cur
                key = (a, b) if a <= b else (b, a)
                chain = sum(x["km"] or 0 for x in chain_secs)
                if key in sections and sections[key]["chain"] <= chain:
                    continue
                sections[key] = {"nodes": chain_nodes, "secs": chain_secs, "chain": chain}
        # geometry and length from the traced pieces
        own = set()
        for lid in lids:
            own |= own_of.get(lid, set())
        out_secs = {}
        for key, v in sections.items():
            pts_all, km, fast_km, own_km, bad = [], 0.0, 0.0, 0.0, False
            lr_km = 0.0
            for i, s in enumerate(v["secs"]):
                st, got = tr[s["sol"]]
                if st != "ok":
                    bad = True
                    break
                pts, pkm, fast, _e, okm = got
                own_km += okm
                lr_km += track.lr_km(_e) if light_rail else 0.0
                forward = node_of(s["a"]) == v["nodes"][i]
                pa, pb = pos_of(s["a"]), pos_of(s["b"])
                seq = [pa] + pts + [pb]
                if not forward:
                    seq = seq[::-1]
                pts_all.extend(seq if not pts_all else seq[1:])
                km += pkm
                fast_km += fast * pkm
            how_ok = "pieces"
            s0, s1 = v["secs"][0], v["secs"][-1]
            op0 = s0["a"] if node_of(s0["a"]) == v["nodes"][0] else s0["b"]
            op1 = s1["b"] if node_of(s1["b"]) == v["nodes"][-1] else s1["a"]
            p0, p1 = pos_of(op0), pos_of(op1)
            crow = dist_m(*p0, *p1) / 1000 if p0 and p1 else 0.0

            def not_short(x):
                # Track is never much shorter than the crow flies. Wien Quartier Belvedere to
                # Rennweg traced 0.00 km: the S-Bahn trunk is missing from OSM there, and both
                # ends snapped onto the same distant track.
                return x >= SHORT * crow - 0.1
            if (not bad and not within_tol(km, v["chain"], crow) and km
                    and own_km >= OWN_SHARE * km and not_short(km)):
                # On the line's own relation track all the way: the routing is right, and it
                # is RINF's length that is misallocated between neighbouring sections
                # (Torhout-Zedelgem is 13.94 km in RINF for 7.8 km of track, and the next
                # section is short by as much). Kept, and listed.
                how_ok = "own track"
                length_off.append(f"{'/'.join(lids)[:24]:>24} {stations_name(v['nodes'][0])} -> "
                                  f"{stations_name(v['nodes'][-1])}: RINF {v['chain']:.2f} km, "
                                  f"traced {km:.2f} km")
            elif bad or not within_tol(km, v["chain"], crow):
                # Straight from one end to the other, ignoring the points between.
                sa, sb = snap_of(op0), snap_of(op1)
                got = (track.trace(sa, sb, v["chain"], own or None,
                                   prefer_main=op0 in stop_of or op1 in stop_of)
                       if sa is not None and sb is not None else None)
                if got is not None and direct_near_m and not passes_near(
                        got[0], v["secs"], direct_near_m):
                    # The end-to-end trace misses a placed point of the section by more than
                    # `direct_near_m`: it has found some other line. Russia's Сенная - Аткарск
                    # (199 km, a piece with no path) traced 209 km round by Saratov.
                    n_direct_far += 1
                    got = None
                if (got is not None and bad and own and got[1]
                        and got[4] < OWN_DIRECT * got[1]):
                    # A piece had no path and the end-to-end trace barely touches the line's
                    # own OSM relation: it has found some other line. Poland's 223 (Czerwonka -
                    # Ełk, disused through Mrągowo) traced 135 km over Korsze and Giżycko with
                    # 1% on its own ways, inside the 15% tolerance of RINF's 121 km.
                    got = None
                if got is not None and within_tol(got[1], v["chain"], crow) and not_short(got[1]):
                    pts, km, fast, _e, _o = got
                    pts_all = [p0] + pts + [p1]
                    fast_km = fast * km
                    lr_km = track.lr_km(_e) if light_rail else 0.0
                    how_ok = "direct"
                    n_direct += 1
                elif (got is not None and not bad and km and not_short(km)
                      and abs(got[1] - km) <= AGREE * km and km <= DETOUR * crow + 0.5):
                    # Through every point on the way, and straight from end to end, the track
                    # says the same length, and it is no detour: RINF's figure is the odd one
                    # out (ÖBB's 413 01 gives Leoben-St. Michael 15.3 km for 9.1 of track).
                    how_ok = "traces agree"
                    length_off.append(f"{'/'.join(lids)[:24]:>24} {stations_name(v['nodes'][0])}"
                                      f" -> {stations_name(v['nodes'][-1])}: RINF "
                                      f"{v['chain']:.2f} km, traced {km:.2f} km (both ways)")
                else:
                    n_rejected_secs += 1
                    names = [stations_name(n) for n in (v["nodes"][0], v["nodes"][-1])]
                    what = (f"pieces {km:.2f} km" if not bad else "a piece has no path")
                    what += f", direct {got[1]:.2f} km" if got else ", no direct path"
                    rejects.append(f"{'/'.join(lids)[:24]:>24} {names[0]} -> {names[1]}: "
                                   f"RINF {v['chain']:.2f} km; {what}")
                    continue
            if v["nodes"][0] != key[0]:
                pts_all.reverse()
            out_secs[key] = {"km": km, "pts": pts_all, "chain": v["chain"], "how": how_ok,
                             "fast": fast_km >= 0.5 * km if km else False, "lr": lr_km}
        # `fill_holes`: a line whose RINF sections do not connect is joined over its own OSM
        # relation's track where RINF left a stretch out (fill_holes' docstring). The filled
        # sections then take the stops a train route makes on them, as `osm_stops` does.
        if fill_on and len(_component_km([(a, b, 0.0) for a, b in out_secs])) > 1:
            op_at = {}
            for na, nb, s in pieces:
                op_at.setdefault(na, s["a"])
                op_at.setdefault(nb, s["b"])
            got = fill_holes(out_secs, own, lambda n: snap_of(op_at[n]),
                             lambda n: pos_of(op_at[n]), track, edge_lines(), set(lids))
            if got:
                n_hole += len(got)
                km_hole += sum(g[2] for g in got)
                for a, b, km, crow, osh in got:
                    log(f"    hole filled: {'/'.join(lids)[:24]:>24} {stations_name(a)} - "
                        f"{stations_name(b)}: {km:.2f} km of track ({crow:.2f} crow-fly, "
                        f"{osh:.0%} on its own relation)")
                if not osm_stop_cands:
                    holes = {tuple(sorted((a, b))) for a, b, *_r in got}
                    cut, extra = split_at_osm_stops(
                        {k: out_secs.pop(k) for k in holes}, fill_cands(), track.kx, track.ky,
                        stop_nodes, fill_route_share())
                    out_secs.update(cut)
                    stop_nodes |= extra
                    extra_nodes |= extra
                    if extra:
                        log(f"    hole stops: {'/'.join(lids)[:24]}: "
                            + ", ".join(ost[int(n[1:])]["name"] for n in sorted(extra)))
            still = len(_component_km([(a, b, 0.0) for a, b in out_secs]))
            if still > 1:
                log(f"    hole left: {'/'.join(lids)[:24]:>24} still {still} pieces"
                    + ("" if own else " (no own OSM relation)"))
            if os.environ.get("RINF_FILL_DEBUG"):
                for a, b, crow, why in [d for d in DEBUG if d[3].startswith("FILLED")] + \
                        ([d for d in DEBUG if not d[3].startswith("FILLED")][:8] if still > 1 else []):
                    log(f"        {'' if why.startswith('FILLED') else 'refused '}"
                        f"{stations_name(a)} - {stations_name(b)} ({crow:.2f} crow-fly): {why}")
            DEBUG.clear()
        # `osm_stops_skip(lids) -> bool`: lines `osm_stops` leaves alone, whose trains stop at
        # their listed stations only (Morocco's LGV runs beside the old line past its halts).
        if osm_stop_cands and not (conf.get("osm_stops_skip")
                                   and conf["osm_stops_skip"](lids)):
            n_before = len(out_secs)
            out_secs, extra = split_at_osm_stops(out_secs, osm_stop_cands, track.kx, track.ky,
                                                 stop_nodes, on_route)
            stop_nodes |= extra
            extra_nodes |= extra
            if extra:
                log(f"    osm_stops: {'/'.join(lids)[:40]}: {len(extra)} stops added, "
                    f"{n_before} -> {len(out_secs)} sections: "
                    f"{', '.join(ost[int(n[1:])]['name'] for n in sorted(extra))}"[:400])
        # A 0 km section from a stop to a junction at the same place is a register's link,
        # not track (ru_register's clones: "<esr>@<section>" beside its stop, so a long
        # stretch ends at a junction and answers to OSM's routes). Kept here so split_pieces
        # can fold the junction back into its stop once build_model has judged the stretch.
        def same_place(a, b):
            pa, pb = pos_of(op_of_node[a]), pos_of(op_of_node[b])
            return pa is not None and pb is not None and dist_m(*pa, *pb) <= LINK_M
        op_of_node = {}
        for na, nb, s in pieces:
            op_of_node.setdefault(na, s["a"])
            op_of_node.setdefault(nb, s["b"])
        links = [(k[0], k[1]) if k[1] in stop_nodes else (k[1], k[0])
                 for k, v in sections.items()
                 if not v["chain"] and (k[0] in stop_nodes) != (k[1] in stop_nodes)
                 and same_place(*k)]
        for key in redundant(out_secs, stop_nodes, track.kx, track.ky):
            n_parallel += 1
            km_parallel += out_secs.pop(key)["km"]
        if not out_secs:
            continue

        key_ref = gkey[1] if gkey[0] == "ref" else ""
        ref = ((conf.get("ref_display") or (lambda k: spelled.get(k, k)))(key_ref)
               if key_ref else "")
        from n02 import walk_order
        order = walk_order(out_secs.keys())
        # station records
        for key in out_secs:
            for n in key:
                if n in stations:
                    continue
                if n in extra_nodes:
                    o = ost[int(n[1:])]
                    stations[n] = {"id": n, "name": o["name"], "name_en": o["name_en"],
                                   "lon": o["lon"], "lat": o["lat"], "lines": set()}
                    continue
                op = next(op for s in (p[2] for p in pieces) for op in (s["a"], s["b"])
                          if node_of(op) == n)
                if op in stop_of:
                    o = ost[stop_of[op]]
                    stations[n] = {"id": n, "name": o["name"], "name_en": o["name_en"],
                                   "lon": o["lon"], "lat": o["lat"], "lines": set()}
                else:
                    p = points[op]
                    stations[n] = {"id": n, "name": display_name(p.get("name")),
                                   "name_en": "", "lon": p["lon"], "lat": p["lat"],
                                   "lines": set(), "junction": True}
                # `station_en(point, name)`: an English name from the register's side where
                # OSM gave none (Russia: Wikidata's label by ESR code, checked against the
                # name shown, which may be OSM's).
                if conf.get("station_en") and not stations[n]["name_en"]:
                    stations[n]["name_en"] = conf["station_en"](points.get(op, {}),
                                                                stations[n]["name"]) or ""
        rel_names = [rel_name[r] for lid in lids for r in [rel_of.get(lid) or named_rel.get(lid)]
                     if r and rel_name.get(r)]
        if ref:
            w = wikidata_item(wd.get(key_ref), sum(v["chain"] for v in out_secs.values()))
            if conf.get("name"):
                name = conf["name"].format(ref=ref)
            else:
                # No national form for the number: Wikidata's label for that route number in
                # the country's first language (an exact match on the number), else the OSM
                # relation's name, else the bare number.
                langs = conf.get("langs", [])
                named = [n for n in rel_names if refkey(n) != key_ref]
                bases = sorted({base_of[lid] for lid in lids})
                id_name = conf.get("id_name")
                name = (next((w[x] for x in langs if w.get(x)), "")
                        or (named[0] if named else "")
                        or (id_name(bases[0], uop_name) if id_name else "")
                        or f"{stations[order[0]]['name']} - {stations[order[-1]]['name']}")
            tmpl_en = conf.get("name_en")
            name_en = tmpl_en.format(ref=ref) if tmpl_en else ""
            en = w.get("en") or ""
            generic = re.sub(r"\(.*?\)|belgian|railway|\bsncb\b", "", en.casefold()).strip()
            if not tmpl_en:
                name_en = en if en != name else ""
            elif en and ref.casefold() in en.casefold() and \
                    re.fullmatch(rf"line\s*{re.escape(ref.casefold())}", generic) is None:
                name_en = en
            elif en and ref.casefold() not in en.casefold():
                name_en = f"{name_en} ({en})"
        elif conf.get("id_name") and conf["id_name"](sorted({base_of[lid] for lid in lids})[0],
                                                     uop_name):
            # A line with no public number whose RINF id is its name (Lithuania's
            # "Kyviskes-Vilnius-Kaisiadorys-Kaunas-KazluRuda", Estonia's "Tapa-Tartu").
            name = conf["id_name"](sorted({base_of[lid] for lid in lids})[0], uop_name)
            # id_name may give (name, English name): Greece names lines in Greek and English.
            name, name_en = name if isinstance(name, tuple) else (name, "")
        elif rel_names:
            name, name_en = rel_names[0], ""
        else:
            first, last = stations[order[0]]["name"], stations[order[-1]]["name"]
            name, name_en = f"{first} - {last}", ""
        if conf.get("im_of"):
            # RINF's manager code does not tell them apart (Hungary files GYSEV under MÁV's)
            ims = Counter(conf["im_of"](s) for p in pieces for s in [p[2]])
            operator = ims.most_common(1)[0][0] if ims else ""
        else:
            ims = Counter(s["im"] for p in pieces for s in [p[2]])
            operator = im_name.get(ims.most_common(1)[0][0], "") if ims else ""
        lid_out = line_hash(cc, ref or "|".join(sorted(lids)))
        for key in out_secs:
            for n in key:
                stations[n]["lines"].add(lid_out)
        lines.append({
            "id": lid_out, "src": "rinf", "service": False,
            "name": name, "name_en": name_en, "ref": ref, "colour": "",
            "operator": operator, "operator_en": "", "network": "",
            # `light_rail_track`: a line traced mostly over light_rail track is light rail, so
            # build_model ties it to that track and not to the main line beside it (Hamburg's
            # S-Bahn 1244 runs within 40 m of the Berlin line 6100 to Aumühle).
            "kind": ("light_rail" if light_rail and sum(v.get("lr", 0.0) for v in out_secs.values())
                     > 0.5 * sum(v["km"] for v in out_secs.values()) else "rail"),
            "km": round(sum(v["km"] for v in out_secs.values()), 3),
            "km_official": round(sum(v["chain"] for v in out_secs.values()), 3),
            "chain": {f"{a}|{b}": round(v["chain"], 3) for (a, b), v in out_secs.items()},
            "highspeed_sections": {f"{a}|{b}": v["fast"] for (a, b), v in out_secs.items()},
            "rinf_ids": sorted(lids),
            "variants": len(lids), "straight_sections": 0,
            "display": order,
            "sections": [[a, b, round(v["km"], 3)] for (a, b), v in out_secs.items()],
        })
        # What RINF itself gave this line, for the lines-in-pieces measurement and split_pieces:
        # the RINF km of each connected piece of its section graph (stops merged), before
        # any trace was rejected.
        GROUPS[lid_out] = {"raw_pieces": _component_km(
            [(na, nb, s["km"] or 0.0) for na, nb, s in pieces]), "links": links}
        susp = conf.get("suspended")
        if susp and susp(key_ref, sorted(lids)):
            # The register's own country says no passenger train runs (Slovakia: the
            # timetable lines sk.wikipedia marks "pravidelná osobná prevádzka prerušená").
            # not_running.mark then lists every section as not running.
            lines[-1]["suspended"] = True
        if conf.get("highspeed_sections") is False:
            # Speed left unknown, so build_model filters no credit by it (Greece: one main
            # line over old and new alignment, no conventional line beside the new one; the
            # ICE's 316 km OSM section Athens - Larissa, under half on highspeed=yes track,
            # credited nothing of the new Tithorea - Lianokladi alignment).
            lines[-1].pop("highspeed_sections", None)
        if km_floor or conf.get("no_chain"):
            # A floor, not the track's length (or, `no_chain`, too often wrong): kept out of
            # check_model's chainage check.
            floor_km[0] += lines[-1].pop("km_official")
            lines[-1].pop("chain")
        geoms[lid_out] ={f"{a}|{b}": [[round(x, 5), round(y, 5)] for x, y in v["pts"]]
                          for (a, b), v in out_secs.items()}

    total = sum(l["km"] for l in lines)
    n_j = sum(1 for s in stations.values() if s.get("junction"))
    log(f"RINF: {len(lines)} lines ({sum(1 for l in lines if l['ref'])} with a public number), "
        f"{total:,.0f} km traced against "
        f"{sum(l.get('km_official', 0) for l in lines) + floor_km[0]:,.0f} km of "
        f"RINF length; {len(stations)} section ends of which {n_j} are not stops; "
        f"{n_rejected_secs} sections left out for a rejected trace, {n_direct} traced "
        f"end to end because the pieces did not add up; {n_parallel} sections "
        f"({km_parallel:.0f} km) dropped as a second track pair of their own line")
    if fill_on:
        log(f"RINF: fill_holes: {n_hole} gaps in lines' RINF sections filled over "
            f"{km_hole:.1f} km of their own OSM track")
    if direct_near_m:
        log(f"RINF: {n_direct_far} end-to-end traces passed a placed point by more than "
            f"{direct_near_m} m and were not used")
    log(f"RINF: {len(length_off)} sections kept on their own line's track although their "
        f"length disagrees with RINF's:")
    for r in length_off:
        log(f"  length off: {r}")
    for r in rejects:
        log(f"  rejected: {r}")
    if wd:
        built = {refkey(l["ref"]) for l in lines if l["ref"]}
        missing = sorted(set(wd) - built, key=lambda r: (len(r), r))
        log(f"RINF: Wikidata has {len(wd)} route numbers; {len(missing)} of them built as no "
            f"line: {' '.join(missing[:60])}")
    # `line_alias` in COUNTRY ({old RINF key: new RINF key}): line ids a country's change
    # dropped, and the line saved rides on them belong on now (Italy's split "Nodo di ..."
    # lines). Kept here as LINE_ALIAS (line id -> line id) for build_model to add to
    # aliases.json `lines`; empty unless the country sets it.
    LINE_ALIAS.clear()
    live = {l["id"] for l in lines}
    for old, new in (conf.get("line_alias") or {}).items():
        a, b = line_hash(cc, old), line_hash(cc, new)
        if a not in live and b in live:
            LINE_ALIAS[a] = b
    if LINE_ALIAS:
        log(f"RINF: {len(LINE_ALIAS)} line ids gone, aliased for saved rides: "
            + ", ".join(f"{a} -> {b}" for a, b in LINE_ALIAS.items()))
    return lines, stations, geoms


LINE_ALIAS = {}
# Per built line id: {"raw_pieces": [RINF km of each connected piece of its RINF sections],
# "links": [(junction, stop), ...] (the 0 km links build() found)}.
GROUPS = {}
LAST_CC = [None]                   # the region build() last read, for split_pieces
LINE_PIECES = {}                   # {line id: [ids split off it]}, for aliases.json `pieces`


def _rename_node(l, old, new, geoms, state):
    """Section end `old` of line l becomes `new`: sections, their geometry and per-section
    dicts, display, and ownership's state["sec_ways"]. A section left from `new` to `new` goes."""
    lid = l["id"]
    g = geoms.get(lid, {})
    sec_ways = (state or {}).get("sec_ways")
    keep = []
    for sec in l["sections"]:
        a, b = sec[0], sec[1]
        if old not in (a, b):
            keep.append(sec)
            continue
        k0 = f"{a}|{b}"
        a2, b2 = (new if a == old else a), (new if b == old else b)
        k1 = f"{a2}|{b2}"
        dicts = [d for d in (l.get("highspeed_sections"), l.get("chain")) if isinstance(d, dict)]
        if a2 == b2:
            g.pop(k0, None)
            for d in dicts:
                d.pop(k0, None)
            if sec_ways is not None:
                sec_ways.pop((lid, k0), None)
            continue
        sec[0], sec[1] = a2, b2
        if k0 in g:
            g[k1] = g.pop(k0)
        for d in dicts:
            if k0 in d:
                d[k1] = d.pop(k0)
        if sec_ways is not None and (lid, k0) in sec_ways:
            sec_ways[(lid, k1)] = sec_ways.pop((lid, k0))
        keep.append(sec)
    l["sections"] = keep
    disp = []
    for s in l.get("display", []):
        s = new if s == old else s
        if not disp or disp[-1] != s:
            disp.append(s)
    l["display"] = disp


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """build_model's hook, after drop_unridden_sections (HANDOFF, "What build() returns").

    1. LINKS. A junction a register put beside a stop with a 0 km section between them
    (ru_register's clones, so that a long stop-to-stop stretch is judged by OSM's routes) is
    folded back into its stop once the stretch has been judged: build_model drops the 0 km
    link, which no route runs over, and the line came out in pieces at every clone whose
    stretch was kept (Лена-Восточная — Хани in 14). Every RINF-read country; only a register
    with such links moves."""
    cc = LAST_CC[0]
    LINE_PIECES.clear()
    n_fold, n_lines = 0, 0
    for l in lines:
        if l.get("src") != "rinf":
            continue
        links = (GROUPS.get(l["id"]) or {}).get("links") or ()
        if not links:
            continue
        ends = {s for sec in l["sections"] for s in sec[:2]}
        did = 0
        for j, s in links:
            if j in ends and s in stations:
                _rename_node(l, j, s, geoms, state)
                stations[j]["lines"].discard(l["id"])
                stations[s]["lines"].add(l["id"])
                ends = {x for sec in l["sections"] for x in sec[:2]}
                did += 1
        if did:
            n_fold += did
            n_lines += 1
            l["km"] = round(sum(sec[2] for sec in l["sections"]), 3)
    log(f"RINF: split_pieces ({cc}): {n_fold} link junctions folded into their stops "
        f"on {n_lines} lines")
    conf = country(cc) if cc else {}
    if conf.get("bridge_pieces"):
        _bridge(conf, lines, stations, geoms, reg_ways, state, log)


def _bridge(conf, lines, stations, geoms, reg_ways, state, log):
    """2. BRIDGES (`bridge_pieces` in COUNTRY: True, or a dict of pieces.Rules settings). A
    line still in pieces is joined over the passenger track between them by
    pieces.bridge_gaps, as the UK's are: where build_model dropped a section of it no train
    runs over and trains take another line's track instead (Portugal's Linha do Sul, whose
    old line Pinheiro - Grândola Norte is dropped and whose trains run the Alcácer variant).
    A way's name for the bridge is the register line it belongs to (reg_ways); track under
    a passenger route that no register line holds is in the graph unnamed. Nothing is split:
    what no bridge joins stays one line in pieces (the rest is track missing from OSM)."""
    import pickle
    import build_model as bm
    import pieces as pc
    cc = LAST_CC[0]
    reg = {l["id"]: l["name"] for l in lines
           if l.get("src", "osm") != "osm" and not l.get("service")}
    name_of = {}
    for wid, lids in reg_ways.items():
        names = sorted(reg[x] for x in lids if x in reg)
        if names:
            name_of[wid] = names[0]
    lat = np.mean([s["lat"] for s in stations.values()]) if stations else 50.0
    opts = conf["bridge_pieces"] if isinstance(conf["bridge_pieces"], dict) else {}
    rules = pc.Rules(**{"tag": f"RINF {cc}", "id_prefix": "q", "lat": float(lat), **opts})

    def classify(wid, tags, routed):
        if tags.get("railway") not in TRACK:
            return None
        nm = name_of.get(wid)
        return nm if nm is not None else ("" if routed else None)

    def graph():
        if not state or "ways" not in state:
            return None
        f = ROOT / "data" / "proc" / cc / "rels.pkl"
        if not f.exists():
            return None
        with open(f, "rb") as fh:
            rels = pickle.load(fh)
        return pc.track_graph(state["ways"], bm.Coords(state["cid"], state["cx"], state["cy"]),
                              pc.routed_ways(rels), classify, rules, log)
    before = {l["id"]: {f"{s[0]}|{s[1]}" for s in l["sections"]} for l in lines
              if l["id"] in reg}
    pc.bridge_gaps(lines, stations, geoms, reg_ways, state, log, rules, graph)
    # A bridge has no register chainage: its traced km joins km_official, so check_model
    # still compares the register's own sections with their traces.
    for l in lines:
        if l["id"] not in before:
            continue
        new = [s for s in l["sections"] if f"{s[0]}|{s[1]}" not in before[l["id"]]]
        if new and "km_official" in l:
            l["km_official"] = round(l["km_official"] + sum(s[2] for s in new), 3)


def _component_km(edges):
    """The km of each connected piece of [(a, b, km)], biggest first."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, _km in edges:
        parent[find(a)] = find(b)
    by = defaultdict(float)
    for a, _b, km in edges:
        by[find(a)] += km
    return sorted(by.values(), reverse=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", metavar="CC", help="pull one country from RINF and Wikidata")
    ap.add_argument("--fetch-wikidata", metavar="CC", help="refresh the Wikidata half only")
    ap.add_argument("--dry", metavar="CC", help="run build() alone and print its log")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.fetch)
    if args.fetch_wikidata:
        fetch(args.fetch_wikidata, rinf_too=False)
    if args.dry:
        t = time.time()
        build(str(RAW / args.dry), lambda m: print(f"[{time.time() - t:6.1f}s] {m}", flush=True))
