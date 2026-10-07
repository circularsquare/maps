"""India: Indian Railways' lines, from Wikidata's station chains and the IR timetable's km,
written into rinf.py's input format (as ru_register.py does for Russia's tariff guide).

    python in_register.py --fetch         # Wikidata + the unofficial IR GTFS -> data/raw/in
    python extract.py --region in --pbf data/raw/india-YYMMDD.osm.pbf     (Anita's say-so: 1.7 GB)
    python in_register.py --convert       # data/raw/in/{sections,points,names}.json
    python build_model.py --region in --register in_register:data/raw/in
    python in_register.py --report        # the register without OSM: lines, km, coverage

`--register in_register:data/raw/in` converts and then runs rinf.build on the result;
`--register rinf:data/raw/in` builds from the last --convert. rinf_countries/in.py holds the
settings, in_sources.md the sources, numbers and what is off.

THE REGISTER UNIT is the section of line as Wikidata (after en.wikipedia) has it: "Mathura -
Vadodara Section", "Konkan Railway", "Jolarpettai-Shoranur line", "Sealdah-Namkhana line", and
Mumbai's suburban "Central Line", "Western Line", "Harbour line". 338 items carry a station
chain (P197 adjacency qualified by the line, P81); they barely overlap (89 km of 46,000 crow-fly
km lie on two of them), so they partition the network the way IR's own sections do. Metro, RRTS
and monorail items are left out: they stay OSM lines, as everywhere. Closed items are left out.

THE KM come from the timetable: the unofficial IR GTFS (Neo2308/indianrailways-gtfs, scraped
from NTES; stop_ids are IR station codes) gives every train's cumulative km at each call, so
consecutive calls give section km to the kilometre. Its passenger network is reduced to the
`minimal` one (an express hop that local calls already cover within 3% + 2 km goes), and each
pair of chain neighbours is measured as the shortest path over it. Calls between them that
Wikidata's chain lacks are put into the line (Samastipur - Muzaffarpur has 10 stations in
Wikidata and 82 km between two of them). A Wikidata station no train calls at is not a stop:
it stays in the line only where it is a branch point or an end, as a junction point (rinf type
80), or where an OSM train route stops at it (after the extract; Mumbai's suburban halts, which
NTES does not carry). A line no train in the feed calls at keeps every station, with km from
the crow-fly distance (x CROW_FACTOR) and flagged in names.json.

THE REST OF THE NETWORK. Wikidata's chains cover about 60% of the timetable's passenger
network (North Eastern Railway 6%), so the rest is built from the timetable too (`FILL`):
first as the Wikidata line items that have no chain but are named for their two ends
("Lucknow-Gorakhpur line", "Rajkot-Wankaner section"), each the shortest path over the
timetable network between stations of those names, then junction to junction ("Gonda Jn -
Barabanki Jn"). Anita decides whether that fill stays (in_sources.md, "Line unit").

STATIONS are placed (after the extract) at the OSM station node whose `wikidata` tag is the
station's item or whose `ref` is its IR code, or an OSM station of a matching name within
NAME_NEAR_M, and take that node's name, so rinf.py matches them exactly; else at Wikidata's
point (P625), else the feed's. A point left more than OFF_TRACK_M from any OSM rail track is
a wrong coordinate and is left unplaced, so rinf.py traces past it (Reasi on the USBRL).
OSM's track is not named for its line in India (30% of main-line km has a name) and its
passenger routes cover 44% of main-line km, so neither can be the register; rinf.py's
tracing between placed points with the timetable's km as the length check is what works.

The `path` argument is data/raw/in; the OSM half is read from data/proc/in (extract.py).
"""
import argparse
import csv
import heapq
import io
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
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import median

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "in"
PROC = ROOT / "data" / "proc" / "in"
WIKIDATA = "https://query.wikidata.org/sparql"
USER_AGENT = "noritetsu-build/1.0 (hobby rail map)"
GTFS_URL = "https://raw.githubusercontent.com/Neo2308/indianrailways-gtfs/main/gtfs/gtfs.zip"
GTFS_FILE = "ir_gtfs.zip"
MAX_FETCH_MB = 25
INF = float("inf")

FILL = True                # build the timetable network Wikidata's chains miss (see docstring)
CROW_FACTOR = 1.10         # track km per crow-fly km where no train gives a figure
REDUNDANT = (1.03, 2.0)    # an express hop locals cover within this x its km + this km goes
PATH_DETOUR = 1.5          # a chain neighbour pair's timetable path at most this x crow-fly...
PATH_SLACK_KM = 5.0        # ... plus this
COORD_DISAGREE_M = 3000    # Wikidata's and the feed's point for one code this far apart: check
CODE_NEAR_M = 300         # a station whose code the feed lacks takes a feed stop this close
OFF_TRACK_M = 1200          # a point this far from any OSM rail track has a wrong coordinate
NAME_NEAR_M = 3000         # else an OSM station of a matching name this close to the point
OSM_REF_M = 10000          # an OSM node carrying a station's code may be this far off Wikidata
OSM_SERVED_M = 300         # an OSM train route stopping this close makes an unserved station a stop
ITEM_DETOUR = 1.8          # a chainless item's path at most this x crow-fly plus 15 km (USBRL's hills)
ITEM_LEN_RANGE = (0.75, 1.33)   # ... and within this of the item's own length (P2043), if any

LANGS = '"en", "hi"'
Q_LINES = """
SELECT ?x ?cls ?lab ?lang ?osm ?num ?len ?unit ?op ?opened ?closed ?part WHERE {
  ?x wdt:P17 wd:Q668 ; wdt:P31 ?cls . ?cls wdt:P279* wd:Q728937 .
  OPTIONAL { ?x rdfs:label ?lab . BIND(LANG(?lab) AS ?lang) FILTER(?lang IN (%s)) }
  OPTIONAL { ?x wdt:P402 ?osm }
  OPTIONAL { ?x wdt:P1671 ?num }
  OPTIONAL { ?x p:P2043/psv:P2043 ?q . ?q wikibase:quantityAmount ?len ;
                                         wikibase:quantityUnit ?unit }
  OPTIONAL { ?x wdt:P137 ?op }
  OPTIONAL { ?x wdt:P1619 ?opened }
  OPTIONAL { ?x wdt:P3999 ?closed }
  OPTIONAL { ?x wdt:P361 ?part }
}
""" % LANGS

# Station to station adjacency, qualified by the line (P197 + pq:P81).
Q_ADJ = """
SELECT ?s ?a ?line WHERE {
  ?s wdt:P17 wd:Q668 ; p:P197 ?st . ?st ps:P197 ?a ; pq:P81 ?line .
}
"""

# The lines those qualifiers name, whatever their class (some are not railway-line subclasses:
# Chennai's MRTS, the RRTS), so a metro can be told from a railway.
Q_ADJ_LINES = """
SELECT DISTINCT ?line ?lab ?cls WHERE {
  ?s wdt:P17 wd:Q668 ; p:P197 ?st . ?st pq:P81 ?line .
  OPTIONAL { ?line rdfs:label ?lab . FILTER(LANG(?lab) = "en") }
  OPTIONAL { ?line wdt:P31 ?cls }
}
"""

Q_STATIONS = """
SELECT ?s ?lab ?lang ?coord ?code ?osm ?cls ?closed ?dissolved WHERE {
  { ?s wdt:P17 wd:Q668 ; wdt:P197 [] . } UNION { ?s wdt:P17 wd:Q668 ; wdt:P81 [] . }
  OPTIONAL { ?s wdt:P625 ?coord }
  OPTIONAL { ?s wdt:P5696 ?code }
  OPTIONAL { ?s wdt:P11693 ?osm }
  OPTIONAL { ?s wdt:P31 ?cls }
  OPTIONAL { ?s wdt:P3999 ?closed }
  OPTIONAL { ?s wdt:P576 ?dissolved }
  OPTIONAL { ?s rdfs:label ?lab . BIND(LANG(?lab) AS ?lang) FILTER(?lang IN (%s)) }
}
""" % LANGS

# Station -> line (P81), with the station's km on the line where given (P6710; 221 pairs).
Q_ON_LINE = """
SELECT ?s ?line ?km WHERE {
  ?s wdt:P17 wd:Q668 ; p:P81 ?st . ?st ps:P81 ?line .
  OPTIONAL { ?st pq:P6710 ?km }
}
"""

# Every item with an IR station code (P5696): label, point, division (P137) and its zone.
Q_CODES = """
SELECT ?s ?code ?lab ?coord ?divl ?zonel WHERE {
  ?s wdt:P5696 ?code .
  OPTIONAL { ?s rdfs:label ?lab . FILTER(LANG(?lab) = "en") }
  OPTIONAL { ?s wdt:P625 ?coord }
  OPTIONAL { ?s wdt:P137 ?div . ?div rdfs:label ?divl . FILTER(LANG(?divl) = "en")
    OPTIONAL { ?div wdt:P361|wdt:P749 ?zone . ?zone rdfs:label ?zonel .
               FILTER(LANG(?zonel) = "en") } }
}
"""

# Wikidata classes that make a line a metro, RRTS or monorail: OSM lines, not register lines.
METRO_CLASSES = {
    "Q15079663",    # rapid transit line
    "Q105967897",   # branched subway line
    "Q107343049",   # automated rapid transit line
    "Q1192191",     # airport rail link (every Indian one is a metro line)
    "Q187934",      # monorail
    "Q15145537",    # light metro
    "Q106793561",   # elevated metro line
    "Q124130104",   # commuter rail line (the RRTS; IR's suburban lines are Q728937)
    "Q14626453",    # regional rail (the RRTS)
    "Q5503",        # rapid transit
}
# Not register lines whatever their class: the RRTS corridor and Chennai's MRTS... are kept
# (MRTS is IR's), so this lists only what the classes miss.
NOT_REGISTER = {
    "Q30644700",    # Delhi-Meerut RRTS (NCRTC, standard gauge, its own OSM line)
    "Q138498887",   # Meerut Metro, on the RRTS corridor
    "Q3161828",     # Janakpur-Jaynagar Railway: Nepal Railways, all but Jaynagar in Nepal
}
# Line items with no chain that must never be read as an A-B path: corridors over other
# sections, high-speed and freight corridors, proposals, closed light railways.
NOT_ITEM = re.compile(r"high-speed|high speed|freight corridor|corridor|metro|monorail|"
                      r"light railway|tramway|trainways|express$|weekly|plantation|"
                      r"sugar factory|spiral|bridge|trains of|railway$|main line$", re.I)

# Wikidata's zone labels -> (code, name), the operator a line is given.
ZONES = {
    "Northern Railway zone": ("NR", "Northern Railway"),
    "North Eastern Railway zone": ("NER", "North Eastern Railway"),
    "Northeast Frontier Railway zone": ("NFR", "Northeast Frontier Railway"),
    "Eastern Railway zone": ("ER", "Eastern Railway"),
    "South Eastern Railway zone": ("SER", "South Eastern Railway"),
    "South Central Railway zone": ("SCR", "South Central Railway"),
    "Southern Railway": ("SR", "Southern Railway"),
    "Southern Railway zone": ("SR", "Southern Railway"),
    "Central Railway zone": ("CR", "Central Railway"),
    "Western Railway zone": ("WR", "Western Railway"),
    "South Western Railway zone": ("SWR", "South Western Railway"),
    "North Western Railway zone": ("NWR", "North Western Railway"),
    "West Central Railway zone": ("WCR", "West Central Railway"),
    "North Central Railway zone": ("NCR", "North Central Railway"),
    "South East Central Railway zone": ("SECR", "South East Central Railway"),
    "East Coast Railway zone": ("ECoR", "East Coast Railway"),
    "East Central Railway zone": ("ECR", "East Central Railway"),
    "South Coast Railway zone": ("SCoR", "South Coast Railway"),
    "Konkan Railway Corporation": ("KR", "Konkan Railway"),
    "Metro Railway, Kolkata": ("MR", "Metro Railway, Kolkata"),
}

# Trains that are not passenger trains, by words in the feed's route name.
NOT_PAX = re.compile(r"\b(FTR|CARGO|PARCEL|GOODS|MILK|RORO|JPP|RCS|AUTOMOBILE|TEST|TRIAL)\b")


def log_print(msg, t0=time.time()):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    dy = (lat2 - lat1) * 110570
    return math.hypot(dx, dy)


# ================================================================ fetching

def sparql(query, tries=4):
    body = urllib.parse.urlencode({"query": query}).encode()
    for k in range(tries):
        req = urllib.request.Request(WIKIDATA, data=body, headers={
            "Accept": "application/sparql-results+json", "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            return [{v: b[v]["value"].replace("http://www.wikidata.org/entity/", "")
                     for v in b} for b in d["results"]["bindings"]]
        except Exception as e:                                  # noqa: BLE001
            print(f"  SPARQL attempt {k + 1} failed: {e}", flush=True)
            if k == tries - 1:
                raise
            time.sleep(15 * (k + 1))


def fetch_small(url, out):
    """A file under MAX_FETCH_MB (HEAD first; a bigger one is reported, not fetched)."""
    head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(head, timeout=60) as r:
        n = int(r.headers.get("Content-Length") or 0)
    if n > MAX_FETCH_MB * 1e6:
        print(f"  {url} is {n / 1e6:.0f} MB: not fetched (over {MAX_FETCH_MB} MB)")
        return
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, out)
    print(f"  {out.name}: {len(data) / 1e6:.1f} MB")


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, q in (("wd_lines", Q_LINES), ("wd_adjacency", Q_ADJ),
                    ("wd_adj_lines", Q_ADJ_LINES), ("wd_stations", Q_STATIONS),
                    ("wd_on_line", Q_ON_LINE), ("wd_codes", Q_CODES)):
        t = time.time()
        rows = sparql(q)
        (RAW / f"{name}.json").write_text(json.dumps(
            {"endpoint": WIKIDATA, "fetched": date.today().isoformat(), "rows": rows},
            ensure_ascii=False), encoding="utf-8")
        print(f"{name}: {len(rows)} rows in {time.time() - t:.0f} s", flush=True)
        time.sleep(3)
    print("the IR GTFS (Neo2308/indianrailways-gtfs)")
    fetch_small(GTFS_URL, RAW / GTFS_FILE)


# ================================================================ Wikidata

def _rows(name):
    p = RAW / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: python in_register.py --fetch")
    return json.loads(p.read_text("utf-8"))["rows"]


def _point(wkt):
    m = re.match(r"Point\(([-\d.eE]+) ([-\d.eE]+)\)", wkt or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


STATION_WORDS = re.compile(r"\s+(railway station|railway halt|station|halt|metro station|"
                           r"terminus railway station|railway stn)$", re.I)


def clean_station(label):
    """"Kalyan Junction railway station" -> "Kalyan Junction"."""
    return STATION_WORDS.sub("", (label or "").strip()).strip()


def load_wikidata(log):
    lines = defaultdict(lambda: {"cls": set(), "part": set()})
    for r in _rows("wd_lines"):
        e = lines[r["x"]]
        e["cls"].add(r["cls"])
        if r.get("lab"):
            e.setdefault(r["lang"], r["lab"])
        for k in ("osm", "op", "num"):
            if r.get(k):
                e.setdefault(k, r[k])
        if r.get("closed"):
            e["closed"] = True
        if r.get("part"):
            e["part"].add(r["part"])
        if r.get("len") and r.get("unit") in ("Q828224", "Q11573"):
            e.setdefault("km", float(r["len"]) * (1.0 if r["unit"] == "Q828224" else 0.001))
    try:
        for r in _rows("wd_adj_lines"):
            e = lines[r["line"]]
            if r.get("lab"):
                e.setdefault("en", r["lab"])
            if r.get("cls"):
                e["cls"].add(r["cls"])
    except SystemExit:
        log("IN: no wd_adj_lines.json (refetch); lines outside the railway-line classes "
            "are taken by their adjacency alone")
    adj = defaultdict(set)
    for r in _rows("wd_adjacency"):
        if r["s"] != r["a"]:
            adj[r["line"]].add(frozenset((r["s"], r["a"])))
    st = defaultdict(dict)
    for r in _rows("wd_stations"):
        e = st[r["s"]]
        if r.get("lab"):
            e.setdefault(r["lang"], r["lab"])
        if r.get("coord") and "pt" not in e:
            p = _point(r["coord"])
            if p:
                e["pt"] = p
        for k in ("code", "osm"):
            if r.get(k):
                e.setdefault(k, r[k].strip().upper() if k == "code" else r[k])
        if r.get("closed") or r.get("dissolved"):
            e["closed"] = True
    codes = {}
    zone = {}
    for r in _rows("wd_codes"):
        c = r["code"].strip().upper()
        e = codes.setdefault(c, {"items": set()})
        e["items"].add(r["s"])
        if r.get("lab"):
            e.setdefault("en", r["lab"])
        if r.get("coord") and "pt" not in e:
            p = _point(r["coord"])
            if p:
                e["pt"] = p
        if r.get("zonel") in ZONES:
            zone.setdefault(c, ZONES[r["zonel"]][0])
        elif r.get("divl") and "zone" in (r.get("divl") or ""):
            z = ZONES.get(r["divl"])
            if z:
                zone.setdefault(c, z[0])
    log(f"IN: Wikidata {len(lines)} line items, {len(adj)} with adjacency, {len(st)} stations "
        f"on lines ({sum(1 for e in st.values() if e.get('code'))} with an IR code, "
        f"{sum(1 for e in st.values() if 'pt' in e)} with a point); {len(codes)} IR codes "
        f"in all, {len(zone)} with a zone")
    return lines, adj, st, codes, zone


CODE_SHARE = 0.6          # a chain is IR's when this share of its stations have an IR code


def is_register(qid, e, chain=None, wst=None):
    """An IR line: not a metro, RRTS or closed item, and (for a chain) mostly stations with
    an IR code. The code test catches metros filed as plain railway lines (Kanpur's Orange
    Line, Q110419397, ran on over IR track to Mandhana when it was let in)."""
    if qid in NOT_REGISTER or (e["cls"] & METRO_CLASSES) or e.get("closed"):
        return False
    if chain and wst is not None:
        ss = {s for p in chain for s in p}
        return sum(1 for s in ss if wst.get(s, {}).get("code")) >= CODE_SHARE * len(ss)
    return True


# ================================================================ the timetable

def load_gtfs(log):
    """stops {code: (name, lon, lat)}, calls {code: trips}, pair_km {frozenset: km}."""
    z = zipfile.ZipFile(RAW / GTFS_FILE)

    def table(n):
        return csv.DictReader(io.TextIOWrapper(z.open(n), encoding="utf-8"))
    feed = next(iter(table("feed_info.txt")), {})
    stops = {r["stop_id"].strip().upper(): (r["stop_name"], float(r["stop_lon"]),
                                            float(r["stop_lat"])) for r in table("stops.txt")}
    names = {r["route_id"]: r["route_long_name"] for r in table("routes.txt")}
    trip_route = {r["trip_id"]: r["route_id"] for r in table("trips.txt")}
    by_trip = defaultdict(list)
    for r in table("stop_times.txt"):
        by_trip[r["trip_id"]].append((int(r["stop_sequence"]), r["stop_id"].strip().upper(),
                                      r["shape_dist_traveled"]))
    calls, kms, skipped = Counter(), defaultdict(list), 0
    for tid, rows in by_trip.items():
        if NOT_PAX.search(names.get(trip_route.get(tid), "")):
            skipped += 1
            continue
        rows.sort()
        for _s, c, _d in rows:
            calls[c] += 1
        for (_s1, a, d1), (_s2, b, d2) in zip(rows[:-1], rows[1:]):
            try:
                km = float(d2) - float(d1)
            except ValueError:
                continue
            if a != b and km > 0:
                kms[frozenset((a, b))].append(km)
    pair_km = {p: median(v) for p, v in kms.items()}
    log(f"IN: IR GTFS version {feed.get('feed_version', '?')} "
        f"({feed.get('feed_start_date', '?')}-{feed.get('feed_end_date', '?')}): "
        f"{len(stops)} stops, {len(by_trip) - skipped} passenger trains ({skipped} freight and "
        f"parcel left out), {len(pair_km)} pairs of consecutive calls")
    return stops, calls, pair_km


def minimal_network(pair_km):
    """Consecutive-call pairs less the express hops that local calls already cover: shortest
    first, a pair goes when the pairs kept so far join its ends within REDUNDANT of its km.
    Returns {code: {code: km}}."""
    g = defaultdict(dict)
    for p, d in sorted(pair_km.items(), key=lambda kv: kv[1]):
        a, b = sorted(p)
        got = shortest(g, a, b, d * REDUNDANT[0] + REDUNDANT[1])
        if got is not None and got[0] >= 0.9 * d - 2:
            continue
        g[a][b] = d
        g[b][a] = d
    return g


def shortest(g, a, b, cap, banned=frozenset()):
    """(km, [a, ..., b]) over g within cap km, never through `banned` nodes; or None."""
    if a not in g or b not in g:
        return None
    dist, prev, heap = {a: 0.0}, {}, [(0.0, a)]
    while heap:
        du, u = heapq.heappop(heap)
        if du > dist.get(u, INF):
            continue
        if u == b:
            path = [b]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            return du, path[::-1]
        for v, w in g[u].items():
            if v in banned and v != b:
                continue
            nd = du + w
            if nd <= cap and nd < dist.get(v, INF):
                dist[v], prev[v] = nd, u
                heapq.heappush(heap, (nd, v))
    return None


# ================================================================ OSM (after the extract)

def load_osm(log):
    """OSM station nodes by `wikidata` tag and by `ref`, and where OSM train routes stop.
    Empty before the extract."""
    if not (PROC / "stops.pkl").exists():
        log("IN: no data/proc/in yet (no extract): stations placed at Wikidata's points, "
            "unserved stations only where they are branch points or ends")
        return {}, defaultdict(list), []
    with open(PROC / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    with open(PROC / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    by_qid, by_ref = {}, defaultdict(list)
    for nid, (tags, lon, lat) in stops.items():
        if tags.get("railway") not in ("station", "halt") and not (
                tags.get("public_transport") == "station" and tags.get("train") == "yes"):
            continue
        if tags.get("station") in ("subway", "light_rail", "monorail") and \
                tags.get("train") != "yes":
            continue
        rec = (nid, tags.get("name") or "", tags.get("name:en") or "", lon, lat)
        by_ref["*all*"].append(rec)
        if tags.get("wikidata"):
            by_qid[tags["wikidata"]] = rec
        for r in re.split(r"[;,]", tags.get("ref") or ""):
            if r.strip():
                by_ref[r.strip().upper()].append(rec)
    served = []
    for tags, members in rels.values():
        if tags.get("type") != "route" or tags.get("route") != "train":
            continue
        for ty, ref, role in members:
            if ty == "n" and role.startswith(("stop", "platform")) and ref in stops:
                served.append((stops[ref][1], stops[ref][2]))
    log(f"IN: OSM {len(by_qid)} station nodes with a wikidata tag, {len(by_ref)} codes in "
        f"`ref`, {len(set(served))} places an OSM train route stops at")
    return by_qid, by_ref, sorted(set(served))


# ================================================================ the register

def build_register(log):
    """Everything up to rinf.py's input: {"lines": {lid: {...}}, "points": {op: {...}}}."""
    lines, adj, wst, codes, zone = load_wikidata(log)
    gst, calls, pair_km = load_gtfs(log)
    net = minimal_network(pair_km)
    net_km = sum(d for a in net for b, d in net[a].items() if a < b)
    log(f"IN: the timetable's passenger network: {sum(len(v) for v in net.values()) // 2} "
        f"pairs, {net_km:,.0f} km once express hops over local calls are taken out")
    by_qid, by_ref, osm_served = load_osm(log)
    from scipy.spatial import cKDTree
    kx = 111.32 * math.cos(math.radians(22.0))
    stree = (cKDTree([(x * kx, y * 110.57) for x, y in osm_served]) if osm_served else None)

    def osm_stops_at(pt):
        return bool(stree is not None and pt and stree.query_ball_point(
            (pt[0] * kx, pt[1] * 110.57), OSM_SERVED_M / 1000))

    # --- Wikidata's point against the feed's for the same code: they agree to 46 m at the
    # median and within 2 km for 95%; where they are over COORD_DISAGREE_M apart, the one
    # nearer the station's chain neighbours wins (Khudiram Bose Pusa is 81 km off in Wikidata,
    # Hamrapur 1,307 km)
    nbrs = defaultdict(set)
    for q, pairs in adj.items():
        for p in pairs:
            a, b = tuple(p)
            nbrs[a].add(b)
            nbrs[b].add(a)
    n_fixed = 0
    for q, e in wst.items():
        c = e.get("code")
        if c not in gst or "pt" not in e:
            continue
        gp = (gst[c][1], gst[c][2])
        if dist_m(*e["pt"], *gp) <= COORD_DISAGREE_M:
            continue
        ns = [wst[n]["pt"] for n in nbrs[q] if "pt" in wst[n]]
        if not ns:
            continue
        mx = sum(x for x, _y in ns) / len(ns)
        my = sum(y for _x, y in ns) / len(ns)
        if dist_m(*gp, mx, my) < dist_m(*e["pt"], mx, my):
            e["pt_wd"] = e["pt"]
            e["pt"] = gp
            n_fixed += 1
    log(f"IN: {n_fixed} Wikidata station points over {COORD_DISAGREE_M / 1000:.0f} km from the "
        f"feed's point for the same code and further from their neighbours: the feed's taken")

    # --- each Wikidata station's feed code: its own, else the one feed stop within
    # CODE_NEAR_M that no other item claims (renamed codes: Mughalsarai MGS is DDU now)
    claimed = {e["code"] for e in wst.values() if e.get("code") in gst}
    g_ids = list(gst)
    gtree = cKDTree([(gst[c][1] * kx, gst[c][2] * 110.57) for c in g_ids])
    code_of, n_near = {}, 0
    for q, e in wst.items():
        c = e.get("code")
        if c in gst:
            code_of[q] = c
        elif "pt" in e:
            near = gtree.query_ball_point((e["pt"][0] * kx, e["pt"][1] * 110.57),
                                          CODE_NEAR_M / 1000)
            free = [g_ids[i] for i in near if g_ids[i] not in claimed]
            if len(near) == 1 and free:
                code_of[q] = free[0]
                claimed.add(free[0])
                n_near += 1
    log(f"IN: {len(code_of)} Wikidata stations found in the feed ({n_near} by a feed stop "
        f"within {CODE_NEAR_M} m, their own code being unknown to it)")

    # point ids: by IR code where there is one, so lines meet at shared stations
    def op_of_q(q):
        c = code_of.get(q) or wst[q].get("code")
        return f"in:{c}" if c else f"wd:{q}"
    points = {}

    def add_point(op, name, pt, stop, q=None, code=None):
        p = points.get(op)
        if p is None:
            p = points[op] = {"op": op, "name": name, "stop": False, "pt": pt, "q": q,
                              "code": code}
        p["stop"] = p["stop"] or stop
        if not p.get("pt") and pt:
            p["pt"] = pt
        if q and not p.get("q"):
            p["q"] = q
        return p

    def gtfs_name(c):
        e = codes.get(c)
        if e and e.get("en"):
            return clean_station(e["en"])
        return gst[c][0].title() if c in gst else c

    reg = {}                 # lid -> {"name", "pairs": [(op a, op b, km, how)], ...}
    owned = set()            # minimal-network pairs a register line has taken
    stats = Counter()
    # stations on two or more register chains are where lines meet: always kept, so the
    # register stays connected through a junction no train calls at
    n_lines_at = Counter()
    for q, e in lines.items():
        if q in adj and is_register(q, e, adj[q], wst):
            for s in {s for p in adj[q] for s in p}:
                n_lines_at[s] += 1
    for q, e in sorted(lines.items()):
        if q not in adj:
            continue
        if not is_register(q, e, adj[q], wst):
            stats["metro, RRTS, closed or non-IR items left out"] += 1
            continue
        g = defaultdict(set)
        for p in adj[q]:
            a, b = tuple(p)
            g[a].add(b)
            g[b].add(a)
        served = {s for s in g if code_of.get(s) in net}
        pairs = []
        if not served:
            # no train in the feed calls anywhere on it (Mumbai's suburban lines, the hill
            # railways): every station a stop, km from the crow-fly distance
            for p in adj[q]:
                a, b = tuple(p)
                pa, pb = wst[a].get("pt"), wst[b].get("pt")
                km = dist_m(*pa, *pb) / 1000 * CROW_FACTOR if pa and pb else None
                for s in (a, b):
                    add_point(op_of_q(s), clean_station(wst[s].get("en")), wst[s].get("pt"),
                              True, s, wst[s].get("code"))
                pairs.append((op_of_q(a), op_of_q(b), km, "crow"))
            stats["lines no train in the feed calls on"] += 1
            reg[q] = {"name": e.get("en") or q, "name_hi": e.get("hi", ""), "pairs": pairs,
                      "src": "wikidata", "unserved": True, "wd_km": e.get("km")}
            continue
        # keep: served stations, branch points and ends, and (after the extract) stations an
        # OSM train route stops at; the rest are merged away
        keep = set()
        for s in g:
            if s in served:
                keep.add(s)
            elif len(g[s]) != 2 or n_lines_at[s] > 1:
                keep.add(s)
            elif osm_stops_at(wst[s].get("pt")):
                keep.add(s)
                stats["unserved stations kept, an OSM route stops there"] += 1
            else:
                stats["unserved stations merged away"] += 1
        # walk the line's graph from each kept station to the next kept one
        done = set()
        for s in sorted(keep):
            for nb in sorted(g[s]):
                run = [s, nb]
                while run[-1] not in keep:
                    nxt = [x for x in g[run[-1]] if x != run[-2]]
                    if not nxt:
                        break
                    run.append(nxt[0])
                key = frozenset((run[0], run[-1]))
                if run[-1] not in keep or key in done or run[0] == run[-1]:
                    continue
                done.add(key)
                pairs.extend(measure_run(run, wst, code_of, net, owned, gst, add_point,
                                         op_of_q, gtfs_name, osm_stops_at, stats))
        reg[q] = {"name": e.get("en") or q, "name_hi": e.get("hi", ""), "pairs": pairs,
                  "src": "wikidata", "wd_km": e.get("km")}
    log(f"IN: {len(reg)} Wikidata lines with a chain: {dict(stats)}")

    # --- the rest of the timetable network
    if FILL:
        fill(reg, lines, adj, net, owned, gst, points, add_point, gtfs_name, log)

    # --- zones: the operator of each line is the zone most of its stations are in
    for lid, L in reg.items():
        zs = Counter(zone.get((points[a].get("code") or "")) for a, b, _k, _h in L["pairs"]
                     for a in (a, b))
        zs.pop(None, None)
        L["zone"] = zs.most_common(1)[0][0] if zs else ""
    return reg, points, net, net_km


def measure_run(run, wst, code_of, net, owned, gst, add_point, op_of_q, gtfs_name,
                osm_stops_at, stats):
    """Sections for one run of a Wikidata chain between two kept stations: [(op, op, km,
    how)]. A served pair is measured over the timetable network, and any calls the feed has
    between them that Wikidata lacks are put in; an unserved end is placed by crow-fly."""
    a, b = run[0], run[-1]
    ca, cb = code_of.get(a), code_of.get(b)
    pts = [wst[s].get("pt") for s in run]
    crow_chain = (sum(dist_m(*p, *r) for p, r in zip(pts[:-1], pts[1:])) / 1000
                  if all(pts) else None)
    for s in (a, b):
        add_point(op_of_q(s), clean_station(wst[s].get("en")), wst[s].get("pt"),
                  code_of.get(s) in net or osm_stops_at(wst[s].get("pt")), s,
                  code_of.get(s) or wst[s].get("code"))
    if ca in net and cb in net and crow_chain is not None:
        crow = dist_m(*pts[0], *pts[-1]) / 1000
        cap = max(crow_chain, crow) * PATH_DETOUR + PATH_SLACK_KM
        got = shortest(net, ca, cb, cap)
        if got is not None and got[0] >= 0.9 * crow - 0.5:
            km, path = got
            out = []
            for x, y in zip(path[:-1], path[1:]):
                owned.add(frozenset((x, y)))
                for c in (x, y):
                    if c not in (ca, cb):
                        add_point(f"in:{c}", gtfs_name(c), (gst[c][1], gst[c][2]), True,
                                  None, c)
                        stats["stations put in from the feed"] += 1
                ox = op_of_q(a) if x == ca else f"in:{x}"
                oy = op_of_q(b) if y == cb else f"in:{y}"
                out.append((ox, oy, net[x][y], "feed"))
            stats["runs measured over the feed"] += 1
            return out
        stats["runs with no feed path (crow-fly km)"] += 1
    else:
        stats["runs with an unserved end (crow-fly km)"] += 1
    km = crow_chain * CROW_FACTOR if crow_chain is not None else None
    return [(op_of_q(a), op_of_q(b), km, "crow")]


SPLIT = re.compile(r"\s*[–—-]+\s*")
LINE_WORDS = re.compile(r"\b(rail(way)? line|railway line|branch line|main line|line|section|"
                        r"route|loop|chord|railway|rail link|link|suburban railway)\b.*$", re.I)
STATION_TAIL = re.compile(r"\b(junction|jn|jct|city|cantt|cantonment|terminus|town|road|"
                          r"central|north|south|east|west|main)\b\.?", re.I)


def name_key(s):
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def fill(reg, lines, adj, net, owned, gst, points, add_point, gtfs_name, log):
    """The timetable network no Wikidata chain covers, as lines: first the chainless Wikidata
    items named for their two ends, then junction to junction. Works on point ids ("in:NDLS",
    or "wd:Q..." for a register junction the feed has no code for)."""
    on_line = defaultdict(set)                       # point -> register lines
    for lid, L in reg.items():
        for a, b, _k, _h in L["pairs"]:
            on_line[a].add(lid)
            on_line[b].add(lid)
    free = defaultdict(dict)
    for a in net:
        for b, d in net[a].items():
            oa, ob = f"in:{a}", f"in:{b}"
            if frozenset((a, b)) in owned or (on_line.get(oa, set()) & on_line.get(ob, set())):
                continue
            for c in (a, b):
                add_point(f"in:{c}", gtfs_name(c), (gst[c][1], gst[c][2]), True, None, c)
            free[oa][ob] = d
    free_km = sum(d for a in free for b, d in free[a].items() if a < b)
    log(f"IN: fill: {free_km:,.0f} km of the timetable network lies on no Wikidata chain")
    free = trim_hops(free, reg, points, log)
    free_km = sum(d for a in free for b, d in free[a].items() if a < b)
    log(f"IN: fill: {free_km:,.0f} km left once hops over register track are taken out")

    def pt(o):
        return points[o]["pt"]

    def nm(o):
        return points[o]["name"]

    # --- chainless items "A-B ..." as the shortest free path between stations of those names
    by_word = defaultdict(set)                       # first word of a station name -> points
    for o in free:
        k = name_key(nm(o))
        if k:
            by_word[k.split()[0]].add(o)

    def cands(end):
        k = name_key(STATION_TAIL.sub("", end)).strip()
        if not k:
            return set()
        first = k.split()[0]
        out = set()
        for o in by_word.get(first, ()):
            n = name_key(STATION_TAIL.sub("", nm(o))).strip()
            if n == k or n.startswith(k + " ") or k.startswith(n + " ") or n == first:
                out.add(o)
        return out

    tries = []
    for q, e in lines.items():
        if q in adj or not e.get("en") or not is_register(q, e) or NOT_ITEM.search(e["en"]):
            continue
        ends = [x for x in SPLIT.split(LINE_WORDS.sub("", e["en"]).strip()) if x]
        if len(ends) < 2:
            continue
        best = None
        for a in cands(ends[0]):
            for b in cands(ends[-1]):
                if a == b or not pt(a) or not pt(b):
                    continue
                crow = dist_m(*pt(a), *pt(b)) / 1000
                got = shortest(free, a, b, crow * ITEM_DETOUR + 15)
                if got and (best is None or got[0] < best[0]):
                    best = got
        if best is None:
            continue
        # the free path must be the way trains go, not a way round track a chain already has:
        # Lucknow - Gorakhpur found 394 km via Sitapur and Burhwal, the main line being 270
        a, b = best[1][0], best[1][-1]
        whole = shortest(net, a[3:], b[3:], best[0]) if a.startswith("in:") and \
            b.startswith("in:") else None
        if whole is not None and best[0] > whole[0] * 1.1 + 5:
            log(f"    item {e['en']}: free path {best[0]:.0f} km, the network's {whole[0]:.0f}; "
                f"not used")
            continue
        if e.get("km") and not (ITEM_LEN_RANGE[0] <= best[0] / e["km"] <= ITEM_LEN_RANGE[1]):
            log(f"    item {e['en']}: path {best[0]:.0f} km against its own {e['km']:.0f}; "
                f"not used")
            continue
        tries.append((best[0], q, best[1]))
    used = []
    for km, q, path in sorted(tries):
        steps = list(zip(path[:-1], path[1:]))
        if any(y not in free.get(x, {}) for x, y in steps):
            continue                                  # a finer item took part of it
        pairs = []
        for x, y in steps:
            pairs.append((x, y, free[x][y], "feed"))
            del free[x][y]
            del free[y][x]
        e = lines[q]
        reg[q] = {"name": e["en"], "name_hi": e.get("hi", ""), "pairs": pairs,
                  "src": "wikidata-item", "wd_km": e.get("km")}
        for x, y in steps:
            on_line[x].add(q)
            on_line[y].add(q)
        used.append((e["en"], km))
    log(f"IN: fill: {len(used)} chainless Wikidata items laid over the timetable network, "
        f"{sum(k for _n, k in used):,.0f} km: "
        + "; ".join(f"{n} {k:.0f}" for n, k in sorted(used)))

    # --- a chain that stops short of where its trains go on (Wikidata's Kalka - Shimla ends at
    # Barog) is carried on over the feed from an end no other line touches, as far as the next
    # junction
    full_deg = {f"in:{c}": len(net[c]) for c in net}
    n_ext, km_ext, ext_names = 0, 0.0, []
    for lid, L in list(reg.items()):
        deg = Counter()
        for a, b, _k, _h in L["pairs"]:
            deg[a] += 1
            deg[b] += 1
        for e0 in [o for o, n in deg.items() if n == 1]:
            if on_line.get(e0) != {lid} or len(free.get(e0, {})) != 1:
                continue
            run = [e0, next(iter(free[e0]))]
            while (len(free.get(run[-1], {})) == 2 and full_deg.get(run[-1]) == 2
                   and run[-1] not in on_line):
                nxt = [x for x in free[run[-1]] if x != run[-2]]
                if not nxt or nxt[0] in run:
                    break
                run.append(nxt[0])
            for x, y in zip(run[:-1], run[1:]):
                L["pairs"].append((x, y, free[x][y], "feed"))
                km_ext += free[x][y]
                del free[x][y]
                del free[y][x]
                on_line[y].add(lid)
            n_ext += 1
            ext_names.append(f"{L['name']} to {points[run[-1]]['name']}")
    log(f"IN: fill: {n_ext} Wikidata chains carried on from an end over the feed, "
        f"{km_ext:,.0f} km: " + "; ".join(ext_names))

    # --- the rest, junction to junction
    full_deg = {f"in:{c}": len(net[c]) for c in net}
    free = {a: v for a, v in free.items() if v}

    def terminal(o):
        return len(free.get(o, {})) != 2 or full_deg.get(o, 0) != 2 or o in on_line
    seen, n_lines, km_all = set(), 0, 0.0
    for s in sorted(free):
        if not terminal(s):
            continue
        for nb in sorted(free[s]):
            if frozenset((s, nb)) in seen:
                continue
            run = [s, nb]
            seen.add(frozenset((s, nb)))
            while not terminal(run[-1]):
                nxt = [x for x in free[run[-1]] if x != run[-2]]
                if not nxt or frozenset((run[-1], nxt[0])) in seen:
                    break
                seen.add(frozenset((run[-1], nxt[0])))
                run.append(nxt[0])
            pairs = [(x, y, free[x][y], "feed") for x, y in zip(run[:-1], run[1:])]
            a, b = sorted((run[0], run[-1]))
            lid = f"T-{a.split(':')[1]}-{b.split(':')[1]}"
            while lid in reg:
                lid += "+"
            reg[lid] = {"name": f"{nm(run[0])} - {nm(run[-1])}", "name_hi": "",
                        "pairs": pairs, "src": "timetable", "wd_km": None}
            n_lines += 1
            km_all += sum(p[2] for p in pairs)
    left = sum(d for a in free for b, d in free[a].items()
               if a < b and frozenset((a, b)) not in seen)
    log(f"IN: fill: {n_lines} junction-to-junction lines, {km_all:,.0f} km; {left:,.0f} km in "
        f"rings with no junction left out")


HOP_TOL = (1.05, 3.0)      # a feed hop the register joins within this x its km + this km


def trim_hops(free, reg, points, log):
    """Feed hops that run over register track for part or all of their length. A train that
    leaves one line for another at a junction it does not call at (no train calls at most
    junction cabins) gives a hop from a station of the first line to one of the second, which
    the minimal network cannot take apart. Where the register joins the hop's two ends within
    HOP_TOL of its km, the hop goes. Where only one end is on the register, the hop is cut at
    the register point J furthest along it from that end for which register km to J plus the
    crow-fly on to the far end is still within the hop's km: the hop becomes J - far end."""
    rg = defaultdict(dict)
    for L in reg.values():
        for a, b, km, _h in L["pairs"]:
            if km is not None and a != b:
                rg[a][b] = min(km, rg[a].get(b, INF))
                rg[b][a] = min(km, rg[b].get(a, INF))
    out = defaultdict(dict)
    n_gone = n_cut = 0
    km_gone = km_cut = 0.0
    for a in list(free):
        for b, d in free[a].items():
            if a > b:
                continue
            if a in rg and b in rg:
                got = shortest(rg, a, b, d * HOP_TOL[0] + HOP_TOL[1])
                if got is not None and got[0] >= 0.85 * d - 2:
                    n_gone += 1
                    km_gone += d
                    continue
            ends = [(x, y) for x, y in ((a, b), (b, a)) if x in rg and y not in rg]
            if len(ends) == 1 and points[ends[0][1]].get("pt"):
                x, y = ends[0]
                far = points[y]["pt"]
                # register km from x to every register point within d
                dist, heap = {x: 0.0}, [(0.0, x)]
                while heap:
                    du, u = heapq.heappop(heap)
                    if du > dist.get(u, INF):
                        continue
                    for v, w in rg[u].items():
                        nd = du + w
                        if nd <= d and nd < dist.get(v, INF):
                            dist[v] = nd
                            heapq.heappush(heap, (nd, v))
                best = None
                for j, dj in dist.items():
                    if j == x or not points[j].get("pt"):
                        continue
                    if dj + dist_m(*points[j]["pt"], *far) / 1000 <= d + 1.0 and \
                            d - dj >= 0.5 and (best is None or dj > best[1]):
                        best = (j, dj)
                if best is not None:
                    j, dj = best
                    out[j][y] = min(d - dj, out[j].get(y, INF))
                    out[y][j] = out[j][y]
                    n_cut += 1
                    km_cut += dj
                    continue
            out[a][b] = d
            out[b][a] = d
    log(f"IN: fill: {n_gone} feed hops ({km_gone:,.0f} km) lie on register track end to end "
        f"and go; {n_cut} are cut where they leave it ({km_cut:,.0f} km of them was register "
        f"track)")
    return out


# ================================================================ rinf.py's input

def convert(log=log_print, write=True):
    reg, points, net, net_km = build_register(log)
    by_qid, by_ref, _served = load_osm(log)
    # --- place points at their OSM node (by wikidata tag, then ref), named as OSM names it
    n_q = n_ref = n_far = 0
    for op, p in points.items():
        rec = None
        if p.get("q") and p["q"] in by_qid:
            rec = by_qid[p["q"]]
            n_q += 1
        elif p.get("code") and by_ref.get(p["code"]):
            cs = by_ref[p["code"]]
            if p.get("pt"):
                cs = sorted(cs, key=lambda r: dist_m(r[3], r[4], *p["pt"]))
                if dist_m(cs[0][3], cs[0][4], *p["pt"]) > OSM_REF_M:
                    n_far += 1
                    cs = []
            if cs:
                rec = cs[0]
                n_ref += 1
        if rec is not None:
            p["osm"] = rec[0]
            p["pt"] = (rec[3], rec[4])
            if rec[1]:
                p["name_wd"] = p["name"]
                p["name"] = rec[1]
    # --- the rest by name: an OSM station of a matching name (rinf.py's own name test) within
    # NAME_NEAR_M of the point. The feed's and Wikidata's points can be a few km off (Reasi on
    # the USBRL was 1.5 km from any track, so rinf could not place it and the line broke there)
    n_name = 0
    allst = by_ref.get("*all*", [])
    if allst:
        import numpy as np
        from rinf import name_variants, names_match
        ax = np.array([r[3] for r in allst])
        ay = np.array([r[4] for r in allst])
        for op, p in points.items():
            if p.get("osm") or not p.get("pt"):
                continue
            x, y = p["pt"]
            d = np.hypot((ax - x) * math.cos(math.radians(y)) * 111320, (ay - y) * 110570)
            keys = name_variants(p["name"]) | name_variants(p.get("name_feed", ""))
            best = None
            for i in np.nonzero(d <= NAME_NEAR_M)[0].tolist():
                r = allst[i]
                if names_match(keys, name_variants(r[1]) | name_variants(r[2])) and \
                        (best is None or d[i] < best[0]):
                    best = (float(d[i]), r)
            if best is not None:
                rec = best[1]
                p["osm"] = rec[0]
                p["pt"] = (rec[3], rec[4])
                if rec[1]:
                    p["name_wd"] = p["name"]
                    p["name"] = rec[1]
                n_name += 1
    # --- a point left at Wikidata's or the feed's coordinate that lies far from any OSM rail
    # track is a wrong coordinate (Reasi on the USBRL: 2.9 km off its line in a tunnel
    # section). Unplaced, rinf.py traces past it end to end instead of failing the section.
    n_off = 0
    if allst and (PROC / "ways.pkl").exists():
        import numpy as np
        from scipy.spatial import cKDTree
        with open(PROC / "ways.pkl", "rb") as f:
            ways = pickle.load(f)
        c = np.load(PROC / "coords.npz")
        want = np.unique(np.concatenate([nodes for tags, nodes in ways.values()
                                         if tags.get("railway") in
                                         ("rail", "narrow_gauge", "preserved")]))
        pos = np.searchsorted(c["id"], want)
        np.clip(pos, 0, c["id"].size - 1, out=pos)
        pos = pos[c["id"][pos] == want]
        tx, ty = c["x"][pos] / 1e7, c["y"][pos] / 1e7
        kx = 111.32 * math.cos(math.radians(22.0))
        tree = cKDTree(np.c_[tx * kx, ty * 110.57])
        for op, p in points.items():
            if p.get("osm") or not p.get("pt"):
                continue
            d, _i = tree.query((p["pt"][0] * kx, p["pt"][1] * 110.57))
            if d * 1000 > OFF_TRACK_M:
                p["pt_off"] = p.pop("pt")
                n_off += 1
    if by_qid or by_ref:
        log(f"IN: {n_off} points over {OFF_TRACK_M} m from any OSM rail track and at no OSM "
            f"node left unplaced (rinf traces past them)")
        log(f"IN: points at their OSM node: by wikidata tag {n_q}, by ref {n_ref}, by name within "
            f"{NAME_NEAR_M / 1000:.0f} km {n_name}; {n_far} refs over {OSM_REF_M / 1000:.0f} km "
            f"from Wikidata's point ignored; {len(points) - n_q - n_ref - n_name} placed by "
            f"Wikidata or the feed")
    # --- one owner per pair of points, the longer line first (as ru_register does)
    order = sorted(reg, key=lambda lid: (reg[lid]["src"] == "timetable",
                                         -len(reg[lid]["pairs"]), lid))
    owner = {}
    for lid in order:
        for a, b, _k, _h in reg[lid]["pairs"]:
            owner.setdefault(frozenset((a, b)), lid)
    rows, names, used, n_dup, n_none = [], {}, set(), 0, 0
    for lid in order:
        L = reg[lid]
        k = 0
        for a, b, km, how in L["pairs"]:
            if owner[frozenset((a, b))] != lid:
                n_dup += 1
                continue
            if km is None:
                n_none += 1
                continue
            k += 1
            rows.append({"sol": f"{lid}:{k}", "line": lid, "a": a, "b": b,
                         "len": f"{km:.3f}", "im": L.get("zone", ""),
                         "label": f"{points[a]['name']} - {points[b]['name']}", "how": how})
            used |= {a, b}
        if k:
            names[lid] = {"name": L["name"], "name_en": "", "name_hi": L.get("name_hi", ""),
                          "src": L["src"], "zone": L.get("zone", ""),
                          "km": round(sum(p[2] or 0 for p in L["pairs"]), 1),
                          "wd_km": L.get("wd_km"), "unserved": bool(L.get("unserved")),
                          "crow_sections": sum(1 for p in L["pairs"] if p[3] == "crow")}
    pts_out = []
    for op in sorted(used):
        p = points[op]
        code = op[3:] if op.startswith("in:") else ""
        r = {"op": op, "uopid": f"IN{code}" if code else f"IN{op[3:]}",
             "name": p["name"], "type": "10" if p["stop"] else "80"}
        if p.get("pt"):
            r["lon"], r["lat"] = p["pt"]
        if p.get("q"):
            r["wikidata"] = p["q"]
        if p.get("name_wd"):
            r["name_wd"] = p["name_wd"]
        pts_out.append(r)
    by_src = Counter()
    for lid, n in names.items():
        by_src[n["src"]] += n["km"]
    log(f"IN: {len(names)} lines, {sum(n['km'] for n in names.values()):,.0f} km "
        f"({', '.join(f'{k} {v:,.0f}' for k, v in by_src.items())}); {len(rows)} section "
        f"rows ({sum(1 for r in rows if r['how'] == 'crow')} with crow-fly km), "
        f"{n_dup} pairs left to the line that lists them first, {n_none} with no km at all; "
        f"{len(pts_out)} points, {sum(1 for r in pts_out if r['type'] == '10')} stops, "
        f"{sum(1 for r in pts_out if 'lon' not in r)} unplaced")
    if write:
        RAW.mkdir(parents=True, exist_ok=True)
        stamp = {"endpoint": "Wikidata + the IR GTFS (in_register.py)",
                 "fetched": date.today().isoformat()}
        for fn, obj in (("sections.json", {**stamp, "rows": rows}),
                        ("points.json", {**stamp, "rows": pts_out}),
                        ("names.json", names)):
            tmp = RAW / (fn + ".tmp")
            tmp.write_text(json.dumps(obj, ensure_ascii=False), "utf-8")
            os.replace(tmp, RAW / fn)
        log(f"IN: wrote sections.json, points.json, names.json to {RAW}")
    return rows, pts_out, names, net_km


def build(path, log):
    """build_model's register hook: convert, then rinf.py traces it over OSM track."""
    convert(log)
    import rinf
    return rinf.build(path, log)


def report(log=log_print):
    """What the register is without OSM: lines and km per source and zone, against IR's
    published route km per zone (en.wikipedia, "Indian Railways organisational structure")."""
    rows, pts, names, net_km = convert(log, write=False)
    zone_km = {"NR": 7363, "NER": 3470.5, "NFR": 4348, "ER": 2823.2, "SER": 2758.6,
               "SCR": 3572, "SR": 5093, "CR": 4203.3, "WR": 6156.6, "NWR": 5705.6,
               "SWR": 3692.34, "WCR": 3060, "NCR": 3522.6, "SECR": 2396.6, "ECoR": 2701,
               "ECR": 4238, "KR": 756.25, "SCoR": 3532.407}
    by = defaultdict(Counter)
    for lid, n in names.items():
        by[n["zone"] or "?"][n["src"]] += n["km"]
    print(f"\n{'zone':6} {'published':>9} {'wikidata':>9} {'wd item':>8} {'timetable':>9} "
          f"{'all':>8} {'share':>6}")
    for z in sorted(by, key=lambda z: -sum(by[z].values())):
        tot = sum(by[z].values())
        pub = zone_km.get(z)
        print(f"{z:6} {pub or 0:9,.0f} {by[z]['wikidata']:9,.0f} {by[z]['wikidata-item']:8,.0f} "
              f"{by[z]['timetable']:9,.0f} {tot:8,.0f} {tot / pub if pub else 0:6.2f}")
    allk = sum(n["km"] for n in names.values())
    print(f"all    {sum(zone_km.values()):9,.0f} ... {allk:,.0f} km on {len(names)} lines; the "
          f"timetable's passenger network {net_km:,.0f} km")
    chk = [(n["km"], n["wd_km"], n["name"]) for n in names.values() if n.get("wd_km")]
    print(f"\nlines whose Wikidata item has a length (P2043): {len(chk)}")
    for km, wkm, nm in sorted(chk, key=lambda x: x[0] / x[1]):
        print(f"  {km / wkm:5.2f}  {km:7.1f} of {wkm:7.1f}  {nm}")


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """build_model's hook, after drop_unridden_sections: rinf.split_pieces (folds a stop's 0 km
    link junctions back into it, and bridges where the country's settings ask)."""
    import rinf
    rinf.split_pieces(lines, stations, geoms, reg_ways, state, log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--line", metavar="NAME", help="print the sections of lines matching NAME")
    args = ap.parse_args()
    if args.line:
        reg, points, _net, _km = build_register(lambda m: None)
        for lid, L in reg.items():
            if args.line.casefold() in L["name"].casefold():
                print(f"{lid} {L['name']} [{L['src']}] {sum(p[2] or 0 for p in L['pairs']):.1f} "
                      f"km, Wikidata {L.get('wd_km')}")
                for a, b, km, how in L["pairs"]:
                    print(f"   {points[a]['name'][:28]:28} {a:12} - {points[b]['name'][:28]:28} "
                          f"{b:12} {km if km is None else round(km, 1)} {how}")
    if args.fetch:
        fetch()
    if args.convert:
        convert()
    if args.report:
        report()
