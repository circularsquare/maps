"""USA: register lines from the FRA's North American Rail Network (NARN), by subdivision.

    python us_register.py --fetch     # writes data/raw/us/narn_passenger.geojson
    python us_register.py --narn      # the register half alone, no OSM: lines, km, checks
    python us_register.py --fetch-holes   # data/raw/us/narn_holes.geojson (needs data/proc/us)
    python us_register.py --holes-dry     # the envelopes --fetch-holes would query, no fetch
    python build_model.py --region us --register us_register:data/raw/us/narn_passenger.geojson

Anita, 2026-09-30 and 2026-10-02: register lines are FRA NARN subdivisions; Amtrak routes are
services (named trains) over them, and commuter lines as OSM maps them (Metra BNSF, NJ Transit's
Northeast Corridor Line, the LIRR's branches) are operating patterns over them, as JR's operating
patterns sit over Japan's N02 lines.

SOURCE. NARN Rail Lines (FRA, distributed by BTS in the National Transportation Atlas
Database; updated 2026-07-21; a work of the US government, public domain). The ArcGIS Online
copy answers from here (geo.dot.gov and maps.bts.dot.gov do not):
    https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/
        NTAD_North_American_Rail_Network_Lines/FeatureServer/0
Only US segments with passenger service are fetched (`PASSNGR` set): A Amtrak, B Amtrak &
commuter, C commuter, I intercity high-speed (Brightline), E high-speed & commuter, R rapid
transit, T tourist or museum, D Alaska Railroad. 18,232 segments, 43,570 km on 2026-10-02.

WHAT NARN GIVES. Track segments between numbered nodes (FRFRANODE, TOFRANODE: the topology is
given, nothing is snapped), each with its owner (RROWNER1), the railroads with trackage rights,
the owner's DIVISION / SUBDIV / BRANCH names, its length (KM) and a passenger code. NO stations
and NO station names: those come from OpenStreetMap (below). NO line numbers either: the line is
a name, and the name is the owning railroad's subdivision (BNSF's La Junta, UP's Gila), or for a
commuter agency's own track its line or branch (NJ Transit's Morristown Line).

WHAT IS LEFT OUT of the register (us_sources.md has the km):
  - PASSNGR T, tourist and museum lines (3,349 km): their trains run on a few days a week in
    season at most, and Anita's rule counts service more often than about weekly. They stay
    on the map wherever OSM has them as lines. Two lines NARN codes C are museum trolleys and
    are left out the same way (TOURIST_OWNERS).
  - PASSNGR R, rapid transit (335 km): NARN has only the pieces of light-rail lines that run
    on old railroad (DART, Sacramento, TRAX, the River Line, Sprinter, the San Diego Trolley,
    PATH). OSM has those lines whole, so they stay OSM lines, as metros do in RINF countries.
  - NET other than M (yards, sidings, industry track, out of service): 218 km of the codes kept.

THE LINE UNIT is the subdivision as NARN names it, cleaned (`clean_name`): the owner's mark in
brackets dropped ("PITTSBURGH LINE (NS)"), NS's "X DISTRICT A TO B" read as X District, and a
single main track of a subdivision ("SHAFTER TRACK 1", "MCCOMB - MAIN TRACK #2", "OTTUMWA MAIN
1") read as the subdivision itself, since NARN files a stretch of separately laid track under
its own name and leaves a hole in the subdivision there otherwise. A line is one name and one
owner (Metro-North's Hudson Line and Amtrak's meet at Poughkeepsie and are two lines); another
owner's piece under SMALL_OWNER_KM joins the line of its name it touches (MBTA's 6 km of the
Northeast Corridor). Pieces of one name and owner that do not touch are one line if a gap of
at most JOIN_KM separates them, else lines of their own with one name and an English name
naming their end stops (the Northeast Corridor is Amtrak's in two pieces with Metro-North's New
Haven Line between). Segments with no name at all join the named line they touch
(`attach_unnamed`). A line still in pieces once build_model has dropped the sections no train
runs over is one line per piece (`split_pieces`, a hook build_model calls; Anita, 2026-10-04).

NAMES are written as railroads write them: "La Junta Subdivision", "Morristown Line", "Danville
District". A name from SUBDIV takes " Subdivision" unless it already ends in a kind of line
(LINE, BRANCH, DISTRICT, CORRIDOR...); one from BRANCH takes " Branch"; the MBTA's "(MBTA) X"
are its commuter lines, "X Line". NAME_FIX holds the few the register spells as no railroad
would; names that are not names at all ("VS-2 MAP 49", the LIRR's "1", "3", "4") are replaced
by the OSM route=railway relation the line's track lies on (half its km or more), else
"<owner>: <first stop> - <last stop>".

STATIONS come from OpenStreetMap, as Switzerland's and Korea's line ends do: every rail station
a passenger train route relation stops at (route=train, not service=tourism) goes on each
register line ALONG WHICH ONE OF ITS OWN ROUTES RUNS at that station: of the route's track
within ALONG_R_M of the stop, at least ALONG_M lies within NEAR_M of the line. Being near the
line is not enough: a station where the line crosses on a diamond, or beside the line on
another railroad, would otherwise go on it. A station is placed where its route's track meets
the line, projected onto the NARN geometry. OSM's route relations are not always complete (the
MBTA's Fitchburg Line lists 4 of its 18 stops), so a working train station no route lists,
within UNLISTED_M of a passenger route's track, is taken as a stop of the routes passing it
(`unlisted_stations`; never a metro or light-rail station, whatever its train tag says).
The few stations no rule places right are set by hand in NOT_ON (Long Island City is not on
Amtrak's West Subdivision; a museum depot is no stop at all), checked by name and position so
a renamed or moved station stops the build. The same unlisted stations become stops of OSM
lines of their own network (`osm_extra_stops`, rules/us.py's `extra_route_stops`, once
build_model calls that hook), with NOT_ON applied.

SECTIONS run between stations, line ends, and branch points, by an absorbing Dijkstra over the
NARN node graph (as schienennetz.py). A line end that is not a stop is a `junction: True`
station with id "uj<NARN node>", so two lines ending at one node share it, and build_model keeps
a section ending there only where OSM passenger routes run over it. A BRANCH POINT inside one
line is found from the result rather than guessed from node degrees: two sections that share
NARN track diverge somewhere, and that node becomes a junction and the sections are found again
(`branch_points`). Double track filed as two NARN segments rejoins and is not a branch; a real
branch whose two arms each end at a stop is.

GEOMETRY AND LENGTH. A section is drawn on OSM track where OSM has it (`OsmTrack.trace`: the
shortest path over OSM rail ways inside a corridor round the NARN geometry, each edge charged
more the further it lies from it), so register_way_lines and ownership find the very ways the
trains run on; NARN's own geometry where no trace agrees with it (TRACE_TOL). NARN is close to
OSM anyway: OSM passenger route track lies a median 1.4 m from it, 95% within 8 m (the build
log's "NARN vs OSM" line). The register's
own km (`chain`, `km_official`) is NARN's KM over the section, which check_model compares with
the built length line by line.

HOLES. NARN codes some of the track passenger trains run on as having none: UP's paired track
across Nevada (each direction on its own line, one coded), the second track where a double
track splits (Cajon, Donner, the New River gorge), Houston - Beaumont. `--fetch-holes` takes
the OSM passenger route track lying more than HOLE_FAR_M from every kept NARN segment, and
fetches NARN's uncoded segments in envelopes round it (data/raw/us/narn_holes.geojson; spatial,
not by name, since the track can be under any subdivision name). A fetched segment joins the
register only where an OSM passenger route runs over it (`accept_holes`): it is a missing piece
of a passenger line, never freight. Taken pieces meeting the passenger line of their own name
only at its ends fill a gap in it; pieces meeting it where it runs on are its other track,
"Elko Subdivision (second track)", a line of their own; the rest are lines as named
(`group_holes`). The build log lists each with its km.

DROPPED AFTER THE FACT: a section between two stops that no OSM passenger route runs over
(UNROUTED_SHARE) is left out, logged with its km. NARN's passenger code is the owner's word that
trains use the track, and some of it is stale or an occasional detour; two stations at its ends
that trains do stop at are not proof that trains run between them on this subdivision. A false
credit is worse than a missed one (Anita).

The `path` argument is the geojson; the OSM half is read from data/proc/us (extract.py).
"""
import hashlib
import heapq
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "us"
NARN = ("https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/"
        "NTAD_North_American_Rail_Network_Lines/FeatureServer/0/query")
WHERE = "COUNTRY='US' AND PASSNGR IS NOT NULL AND PASSNGR<>''"
PAGE = 2000
USER_AGENT = "noritetsu-rail-map/1.0"
INF = float("inf")

# ---------------------------------------------------------------- what is kept

# NARN's passenger codes kept: Amtrak, Amtrak & commuter, commuter, Brightline (I, E), Alaska
# Railroad. Left out: T tourist/museum, R rapid transit (see the docstring).
PASSENGER = {"A", "B", "C", "I", "E", "D"}
# Owners whose C-coded track is a museum line all the same: East Troy Electric Railroad (WI,
# weekend trolleys in season) and the Cuyahoga Valley Scenic Railroad (OH), whose other
# segments NARN already codes T.
TOURIST_OWNERS = {"METW", "CVSR"}

# Pieces of one name and owner closer than this are one line.
JOIN_KM = 30.0
# A second track this close to its main line (for 90% of it) is folded into it (build).
COMPANION_KM = 3.0
# Another owner's piece of a line shorter than this is part of the line it touches.
SMALL_OWNER_KM = 15.0

# ---------------------------------------------------------------- names

NOT_A_NAME = {"", "#N\\A", "#N/A", "SOLD/LEASED/ABANDONED", "NM", "NONE"}
# BRANCH values that say nothing about which line: the DIVISION names it then, if anything.
GENERIC_BRANCH = {"MAIN", "MAIN LINE", "MAINLINE", "FREIGHT MAIN LINE", "FREIGHT MAIN LI",
                  "SYSTEM"}
# Names the register spells as no railroad would. Keys are the cleaned upper-case name.
NAME_FIX = {
    "O E": "Oregon Electric", "O. E.": "Oregon Electric", "OE": "Oregon Electric",
    "FRONTRUNNER": "FrontRunner",
    # NICTD's line: BRANCH "MAIN", DIVISION "SOUTH SHORE", and SUBDIV "SOUTH SHORE" on 5 km.
    "SOUTH SHORE": "South Shore Line",
    "BLUE IS.": "Blue Island Subdivision",
    "S. CHICAGO": "South Chicago Subdivision",
    "UNIV. PARK": "University Park Subdivision",
    "ERL": "East Rail Line",
    "HELLGATE LINE": "Hell Gate Line",
}
# One line under two spellings (cleaned keys): the LIRR's own BRANCH field writes its Main
# Line "LIRR MAIN LINE" or "LIRR ML" where its SUBDIV writes "MAIN LINE (LI)".
KEY_ALIAS = {"LIRR MAIN LINE": "MAIN LINE", "LIRR ML": "MAIN LINE"}
# Segments read under another subdivision's name (FRAARCID -> SUBDIV), each with why. A row
# whose segment is not in the file stops the build, so a refetch that renumbers fails loudly.
#   387399, 389167: the North River tunnels' New York side, Amtrak's, 1.07 km (BRANCH
#   "HIGHTSTOWN IT" in both states). NARN files it under the New York Terminal Subdivision;
#   the New Jersey side, under the same BRANCH, is the Northeast Corridor. As part of the
#   New York Terminal Subdivision it was one section with the Empire Connection's last
#   0.38 km (the state line to the Empire Connection's 34th Street end, through node 489969
#   where the West Subdivision from Penn Station ends): NJ trains run over one part and
#   Empire Service trains the other, build_model found no route over half of it and dropped
#   it, and Penn Station's west end reached neither New Jersey nor the Hudson Line (2026-10-04).
#   As the Northeast Corridor's, the corridor runs into node 489969, where the West
#   Subdivision and the Empire Connection now both end: one junction for all three.
#   379801, 380038, 374683: South Station's approach to the Old Colony (1.1 km, MBTA, coded C:
#   commuter only), node 495039 by the station throat to node 495012 where the Fairmount Line
#   leaves. NARN files the first two under SUBDIV "EAST" (the third has no name and joined
#   East as the line it touches), so the East Subdivision ran a 1.1 km stub off its Back Bay
#   approach that only Old Colony and Fairmount trains use, and its strip diagram opened with
#   Newmarket and JFK/UMass ahead of South Station (Anita, 2026-10-09). Every Old Colony
#   train (Kingston, Middleborough, Greenbush, Fall River/New Bedford) runs over it, so it is
#   the Old Colony's; the Fairmount Line still ends where it leaves at 495012.
SEGMENT_NAME = {387399: "NORTHEAST CORRIDOR", 389167: "NORTHEAST CORRIDOR",
                379801: "(MBTA) OLD COLONY", 380038: "(MBTA) OLD COLONY",
                374683: "(MBTA) OLD COLONY"}
# Register names that are no name at all: a valuation map, a bare number, the agency's own
# word for a shared stretch. Named from OSM instead (see the docstring).
NOT_A_LINE_NAME = re.compile(r"^(?:\d+|VS[- ]?\d+ MAP \d+|\(MBTA\) (?:CONCURRENT|CONNECTOR))$")
# The last word(s) of a name that already say what kind of line it is.
KIND_WORDS = re.compile(
    r"\b(?:LINE|LINES|BRANCH|BR|CORRIDOR|DISTRICT|SECONDARY|DIVISION|DV|SUBDIVISION|CUTOFF|"
    r"CONNECTOR|CONNECTION|CONNECTING TRK|LEAD|LD|EXTENSION|TRACK|TRACKS|BELT|FLYOVER|ROUTE|"
    r"RT|SPUR|INDUSTRIAL TRACK|IND TRK|TRK|BUS UNIT)$")
KEEP_UPPER = {"NO&M", "RF&P", "C&M", "P&W", "NO&NE", "CT&V", "DFW", "ERL", "KCT", "KO", "UP",
              "BNSF", "CSX", "NS", "LIRR", "MBTA", "NJ", "RTD", "SEPTA", "A", "B", "N", "W",
              "E", "S", "NA", "II", "III", "IV", "MP", "AGS"}
SMALL_WORDS = {"AND": "and", "OF": "of", "THE": "the", "TO": "to"}
# Words written short in the register, written out in a name.
ABBREV = {"BR": "Branch", "DV": "Division", "LD": "Lead", "RT": "Route", "IND": "Industrial",
          "TRK": "Track", "JCT": "Junction", "ST": "St"}

# Reporting marks to the railroad or agency that owns the track (the line's `operator`, as the
# infrastructure manager is in RINF and N02). Marks not here are shown as they are.
OWNERS = {
    "AMTK": "Amtrak", "BNSF": "BNSF Railway", "UP": "Union Pacific Railroad",
    "CSXT": "CSX Transportation", "NS": "Norfolk Southern", "CN": "Canadian National",
    "CPKC": "CPKC", "ARR": "Alaska Railroad", "MNCW": "Metro-North Railroad",
    "MBTA": "MBTA", "NJT": "NJ Transit", "LI": "Long Island Rail Road", "SCAX": "Metrolink",
    "NIRC": "Metra", "FEC": "Florida East Coast Railway",
    "NECR": "New England Central Railroad", "BB": "Buckingham Branch Railroad",
    "NMRX": "New Mexico Department of Transportation", "SEPA": "SEPTA",
    "NICD": "Northern Indiana Commuter Transportation District",
    "SFRC": "South Florida Regional Transportation Authority", "VTR": "Vermont Railway",
    "CFCR": "Florida Department of Transportation (SunRail)",
    "SDNR": "North County Transit District", "MC": "Massachusetts Coastal Railroad",
    "JPBX": "Peninsula Corridor Joint Powers Board (Caltrain)", "PAS": "Pan Am Southern",
    "SMRT": "Sonoma-Marin Area Rail Transit", "UTF": "Utah Transit Authority",
    "BLF": "Brightline", "AWRR": "Austin Western Railroad (CapMetro)",
    "DRTD": "Denver RTD", "RTDC": "Denver RTD", "DART": "Dallas Area Rapid Transit",
    "NERR": "Nashville and Eastern Railroad", "CLP": "Clarendon and Pittsford Railroad",
    "VPRA": "Virginia Passenger Rail Authority", "TRE": "Trinity Railway Express",
    "CSAO": "Conrail Shared Assets", "BCLR": "Bay Colony Railroad",
    "TMBL": "Sound Transit", "PNWR": "Portland and Western Railroad",
    "TRRA": "Terminal Railroad Association of St. Louis",
    "KCTL": "Kansas City Terminal Railway", "XMDT": "MassDOT",
    "NOPB": "New Orleans Public Belt Railroad", "TEXR": "Trinity Metro (TEXRail)",
    "MNNR": "Minnesota Commercial Railway", "HRRC": "Housatonic Railroad",
    "SDRX": "Tacoma Rail", "ST": "Springfield Terminal Railway",
    "PW": "Providence and Worcester Railroad", "MARC": "MARC", "BRC": "Belt Railway of Chicago",
    "ACEX": "Altamont Corridor Express", "NYNJ": "New York New Jersey Rail",
    "CKRR": "Central Maine and Quebec", "NRTX": "Nashville Regional Transportation Authority",
    "METW": "East Troy Electric Railroad", "CVSR": "Cuyahoga Valley Scenic Railroad",
}

# The three passenger crossings into Canada (none into Mexico). A line end within BORDER_M of
# one is named for the border, as build_model names RINF border points.
BORDERS = [(-122.7564, 49.0021, "Canada – United States border"),   # Blaine - White Rock
           (-73.3420, 45.0094, "Canada – United States border"),    # Rouses Point - Lacolle
           (-79.0446, 43.1095, "Canada – United States border")]    # Niagara Falls, the bridge
BORDER_M = 3000

# ---------------------------------------------------------------- stations, sections, geometry

ALONG_R_M = 800        # a station's own route's track this close to the stop is looked at...
NEAR_M = 60            # ...and the part of it within this of a line...
ALONG_M = 300          # ...must be at least this long for the station to go on the line
STATION_M = 600        # and the stop itself this close to the line
MERGE_M = 150          # two stations of one line closer than this along it are one
NODE_SNAP_M = 25       # a station this close to a NARN node is placed at the node
UNROUTED_SHARE = 0.25  # a stop-to-stop section less covered than this by OSM routes is dropped
ROUTE_NEAR_M = 60      # ... where a route's ways lie within this of the section

CORRIDOR_M = 150       # an OSM trace keeps within this of the NARN geometry
TRACE_D0_M = 25        # an OSM edge this far from the NARN geometry costs double
TRACE_STEP_M = 50      # OSM edges are cut every this often, so a trace can end mid-edge
TRACE_TOL = (0.10, 0.4)   # a trace is kept if within 10% + 0.4 km of NARN's length
OSM_TRACK = {"rail", "narrow_gauge"}
SLOW_SERVICE = {"yard", "siding", "spur", "crossover"}


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    dy = (lat2 - lat1) * 110570
    return math.hypot(dx, dy)


def path_m(pts):
    return sum(dist_m(*a, *b) for a, b in zip(pts[:-1], pts[1:]))


R_MERC = 20037508.34 / 180.0


def merc(lon, lat):
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.clip(np.asarray(lat, dtype=np.float64), -85.05, 85.05)
    return np.column_stack([lon * R_MERC,
                            np.log(np.tan((90 + lat) * np.pi / 360)) / (np.pi / 180) * R_MERC])


def merc_scale(lat):
    """Web Mercator metres per true metre at a latitude."""
    return 1.0 / max(math.cos(math.radians(float(lat))), 0.05)


# ================================================================ fetching

def _get(params):
    url = NARN + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:          # a dropped page is retried, then the fetch fails
            if attempt == 4:
                raise
            print(f"  retry after {e}", flush=True)
            time.sleep(5 * (attempt + 1))


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    feats, offset = [], 0
    while True:
        page = _get({"where": WHERE, "outFields": "*", "f": "geojson", "outSR": 4326,
                     "orderByFields": "OBJECTID", "resultOffset": offset,
                     "resultRecordCount": PAGE})
        got = page.get("features") or []
        feats += got
        print(f"  {len(feats):,} segments", flush=True)
        if len(got) < PAGE:
            break
        offset += PAGE
    out = RAW / "narn_passenger.geojson"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "fetched": time.strftime("%Y-%m-%d"),
                   "where": WHERE, "features": feats}, f)
    tmp.replace(out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(feats):,} segments)")


# ================================================================ the register half

def _valid(v):
    v = (v or "").strip()
    return v if v.upper() not in NOT_A_NAME else ""


def raw_name(p):
    """(name, field it came from) for one segment, or ("", "")."""
    sub = _valid(p.get("SUBDIV"))
    if sub:
        return sub.upper(), "subdiv"
    br = _valid(p.get("BRANCH"))
    if br and br.upper() not in GENERIC_BRANCH:
        return br.upper(), "branch"
    div = _valid(p.get("DIVISION"))
    if br and div:                       # NICTD: BRANCH "MAIN", DIVISION "SOUTH SHORE"
        return div.upper(), "subdiv"
    return "", ""


TRACK_SUFFIX = re.compile(
    r"(?:\s*-\s*|\s+)(?:MAIN\s+)?(?:TRACK|TRK)\s*#?\s*\d+(?:\s+AND\s+\d+)?$"
    r"|\s+MAIN\s+\d+$|\s*#\s*\d+$|-B$")
OWNER_TAG = re.compile(r"\s*\([A-Z]{2,5}\)")
A_TO_B = re.compile(r"^(.*?\b(?:DISTRICT|TERMINAL|BUS UNIT))\s+\S.*\s+TO\s+\S.*$")
LETTER_AND = re.compile(r"\b([A-Z])\s*(?:&|AND)\s*([A-Z])\b")


def clean_name(name):
    """The line key for a raw register name: one spelling per line (see the docstring)."""
    n = re.sub(r"\s+", " ", name.strip().upper())
    mbta = n.startswith("(MBTA) ")
    if mbta:
        n = n[7:]
    n = OWNER_TAG.sub("", n).strip()
    n = TRACK_SUFFIX.sub("", n).strip()
    n = A_TO_B.sub(r"\1", n)
    n = LETTER_AND.sub(r"\1&\2", n)
    n = KEY_ALIAS.get(n, n)
    return ("(MBTA) " + n) if mbta else n


def title(word):
    if word in KEEP_UPPER or "&" in word:
        return word
    if word in ABBREV:
        return ABBREV[word]
    if word in SMALL_WORDS:
        return SMALL_WORDS[word]
    if re.fullmatch(r"MC[A-Z]+", word):
        return "Mc" + word[2:].capitalize()
    return "/".join("-".join(w.capitalize() for w in part.split("-"))
                    for part in word.split("/"))


def display_name(key, field):
    """How a line is written: the register's name in railroad form."""
    if key in NAME_FIX:
        return NAME_FIX[key]
    mbta = key.startswith("(MBTA) ")
    n = key[7:] if mbta else key
    words = " ".join(title(w) for w in n.split(" "))
    if mbta:
        return words if KIND_WORDS.search(n) else f"{words} Line"
    if KIND_WORDS.search(n):
        return words
    return f"{words} {'Branch' if field == 'branch' else 'Subdivision'}"


def read_narn(path, log, holes=False):
    """The kept segments, oriented from node a to node b, and what was left out by why.
    With `holes`, the file is fetch_holes' (segments NARN codes no passenger service on):
    only main-network ones are read, flagged "hole", for `accept_holes` to judge."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    segs, left = [], defaultdict(float)
    left_by = defaultdict(float)
    seen = set()
    for f in d["features"]:
        p, g = f["properties"], f.get("geometry")
        if p.get("COUNTRY") != "US" or not g or p.get("FRAARCID") in seen:
            continue
        seen.add(p.get("FRAARCID"))
        km = float(p.get("KM") or 0.0)
        code = p.get("PASSNGR")
        why = None
        if holes:
            if code:
                why = "passenger-coded (in the main file)"
            elif p.get("NET") != "M":
                why = f"not main network (NET {p.get('NET')})"
        elif code == "T":
            why = "tourist (T)"
        elif code == "R":
            why = "rapid transit (R)"
        elif code not in PASSENGER:
            why = f"passenger code {code}"
        elif p.get("RROWNER1") in TOURIST_OWNERS:
            why = "museum line coded C"
        elif p.get("NET") != "M":
            why = f"not main network (NET {p.get('NET')})"
        if why:
            left[why] += km
            left_by[(why, p.get("RROWNER1") or "", raw_name(p)[0])] += km
            continue
        if g["type"] == "LineString":
            pts = [tuple(c[:2]) for c in g["coordinates"]]
        else:                                   # never seen; joined in order if it happens
            pts = [tuple(c[:2]) for part in g["coordinates"] for c in part]
        name, field = raw_name(p)
        if int(p["FRAARCID"]) in SEGMENT_NAME:
            name, field = SEGMENT_NAME[int(p["FRAARCID"])], "subdiv"
        rights = [p.get(f"TRKRGHTS{i}") for i in range(1, 10)]
        segs.append({"id": int(p["FRAARCID"]), "a": int(p["FRFRANODE"]),
                     "b": int(p["TOFRANODE"]), "km": km, "pts": pts, "geo_km": path_m(pts) / 1000,
                     "owner": p.get("RROWNER1") or "", "code": code,
                     "state": p.get("STATEAB") or "", "raw": name, "field": field,
                     "key": clean_name(name) if name else "",
                     "rights": [r for r in rights if r], "hole": holes})
    if not holes:
        missing = set(SEGMENT_NAME) - {s["id"] for s in segs}
        if missing:
            sys.exit(f"us_register.SEGMENT_NAME: segments {sorted(missing)} are not in {path}")
    kept = sum(s["km"] for s in segs)
    log(f"NARN{' holes file' if holes else ''}: {len(segs)} segments kept, {kept:,.0f} km; "
        f"left out: "
        + ", ".join(f"{w} {km:,.0f} km" for w, km in sorted(left.items(), key=lambda x: -x[1])))
    return segs, left, left_by


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def components(segs):
    """Connected pieces of a set of segments, by shared nodes."""
    uf = UF()
    for s in segs:
        uf.union(s["a"], s["b"])
    out = defaultdict(list)
    for s in segs:
        out[uf.find(s["a"])].append(s)
    return list(out.values())


def piece_gap_km(p, q):
    """Shortest distance between the end nodes of two pieces, km."""
    def ends(segs):
        deg = Counter()
        pos = {}
        for s in segs:
            deg[s["a"]] += 1
            deg[s["b"]] += 1
            pos[s["a"]], pos[s["b"]] = s["pts"][0], s["pts"][-1]
        e = [pos[n] for n, k in deg.items() if k == 1]
        return e or list(pos.values())
    return min(dist_m(*a, *b) for a in ends(p) for b in ends(q)) / 1000


def group_lines(segs, log):
    """Register lines: [{key, field, owner, segs}], plus the km of unnamed track left out."""
    named = [s for s in segs if s["key"]]
    unnamed = [s for s in segs if not s["key"]]
    by_key = defaultdict(list)
    for s in named:
        by_key[s["key"]].append(s)
    lines = []
    for key, ss in by_key.items():
        # Pieces per owner first: Metro-North's Hudson Line and Amtrak's touch at Poughkeepsie
        # and are two railroads' lines. A piece of another owner under SMALL_OWNER_KM joins
        # the piece of the same name it touches (MBTA's 6 km of the Northeast Corridor).
        by_own = defaultdict(list)
        for s in ss:
            by_own[s["owner"]].append(s)
        pieces = []
        for owner, os_ in by_own.items():
            for comp in components(os_):
                pieces.append({"segs": comp, "owner": owner,
                               "km": sum(s["km"] for s in comp),
                               "nodes": {n for s in comp for n in (s["a"], s["b"])}})
        pieces.sort(key=lambda p: -p["km"])
        for p in [p for p in pieces if p["km"] < SMALL_OWNER_KM]:
            host = next((q for q in pieces if q is not p and q["owner"] != p["owner"]
                         and q["km"] >= SMALL_OWNER_KM and q["nodes"] & p["nodes"]), None)
            if host is not None:
                host["segs"].extend(p["segs"])
                host["nodes"] |= p["nodes"]
                pieces.remove(p)
        by_owner = defaultdict(list)
        for p in pieces:
            by_owner[p["owner"]].append(p)
        for owner, ps in by_owner.items():
            # Join pieces within JOIN_KM of each other, transitively.
            uf = UF()
            for i in range(len(ps)):
                uf.find(i)
                for j in range(i + 1, len(ps)):
                    if piece_gap_km(ps[i]["segs"], ps[j]["segs"]) <= JOIN_KM:
                        uf.union(i, j)
            groups = defaultdict(list)
            for i, p in enumerate(ps):
                groups[uf.find(i)].append(p)
            for g in groups.values():
                field = Counter(s["field"] for p in g for s in p["segs"]).most_common(1)[0][0]
                lines.append({"key": key, "field": field, "owner": owner,
                              "segs": [s for p in g for s in p["segs"]],
                              "pieces": len(g), "split": len(groups) > 1})
    left = attach_unnamed(lines, unnamed, log)
    return lines, left


def attach_unnamed(lines, unnamed, log):
    """Unnamed track joins the named line it touches at the most nodes, of its own owner
    first. A piece touching no named line is left out (it is yard and terminal track almost
    all of it; logged)."""
    at_node = defaultdict(set)
    for i, l in enumerate(lines):
        for s in l["segs"]:
            at_node[s["a"]].add(i)
            at_node[s["b"]].add(i)
    joined, left = 0.0, []
    for comp in components(unnamed):
        nodes = {n for s in comp for n in (s["a"], s["b"])}
        touch = Counter()
        for n in nodes:
            for i in at_node.get(n, ()):
                touch[i] += 1
        if not touch:
            left.append(comp)
            continue
        own = Counter(s["owner"] for s in comp).most_common(1)[0][0]
        best = max(touch, key=lambda i: (lines[i]["owner"] == own, touch[i],
                                         -len(lines[i]["segs"])))
        lines[best]["segs"].extend(comp)
        joined += sum(s["km"] for s in comp)
    left_km = sum(s["km"] for c in left for s in c)
    log(f"NARN: unnamed track: {joined:,.1f} km joined the named line it touches, "
        f"{left_km:,.1f} km in {len(left)} pieces touches none and is left out")
    return left


def line_id(owner, key, segs, split, hole=False):
    tag = f"us|{owner}|{key}"
    if hole:                      # a hole line beside a passenger line of the same name
        tag += "|hole"
    if split or (hole and not key):   # told apart by their track
        tag += f"|{min(s['id'] for s in segs)}"
    return "u" + hashlib.blake2b(tag.encode("utf-8"), digest_size=5).hexdigest()


# ---------------------------------------------------------------- holes

HOLES_FILE = "narn_holes.geojson"
HOLE_WHERE = "COUNTRY='US' AND (PASSNGR IS NULL OR PASSNGR='')"
HOLE_FAR_M = 25        # OSM passenger route track further than this from kept NARN is a hole
HOLE_CELL = 0.05       # degrees: holes are fetched by grid cell, merged along each row
HOLE_NEAR_M = 30       # a NARN segment lies under hole track where it is this close to it
HOLE_SHARE = 0.6       # ...for at least this share of its part away from passenger NARN
HOLE_MIN_M = 200       # ...which must be at least this long


def within_many(tree, pts, r_m, shapely_mod):
    """STRtree dwithin for many lon/lat points at once, scaled per 1-degree latitude band.
    Returns (point index array, geometry index array)."""
    out_p, out_g = [], []
    band = np.floor(pts[:, 1]).astype(int)
    for b in np.unique(band):
        k = np.nonzero(band == b)[0]
        sc = merc_scale(b + 0.5)
        P = shapely_mod.points(merc(pts[k, 0], pts[k, 1]))
        hit = tree.query(P, predicate="dwithin", distance=r_m * sc)
        out_p.append(k[hit[0]])
        out_g.append(hit[1])
    if not out_p:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return np.concatenate(out_p), np.concatenate(out_g)


# Networks whose route relations never vouch for a hole: excursion and heritage lines, a
# disused branch mapped as a route, Brightline West (being built), VIA Rail (Canada).
HOLE_SKIP_NETWORKS = {"Local", "local", "CNJ", "SRRNJ", "Salem County", "Steamtown",
                      "Southern Railroad of New Jersey", "BLFX", "VIA Rail"}


def hole_routes(rels, rw, n_stops, log):
    """The passenger routes whose track may bring a hole segment into the register: a route
    of a named network (not HOLE_SKIP_NETWORKS) that stops at two stations or more. OSM's US
    route=train relations with no network are excursion trains almost all (the St. Croix
    Valley, the Sacramento River Train, the Santa Cruz Beach Train, the West Virginia Line),
    and on the first trial they brought 55 km of the St. Croix Valley's track in."""
    out = {r for r in rw
           if rels[r][0].get("network") and rels[r][0].get("network") not in HOLE_SKIP_NETWORKS
           and n_stops[r] >= 2}
    log(f"holes: {len(out)} of {len(rw)} passenger routes vouch for holes (a named network, "
        f"two stops or more)")
    return out


def hole_route_points(idx, wgeo, log):
    """Points of OSM passenger route track (STEP_DENSE apart) further than HOLE_FAR_M from
    every kept NARN segment: track trains run on that the register does not have."""
    allp = np.unique(np.round(np.vstack(list(wgeo.values())), 6), axis=0)
    near_p, _g = within_many(idx.tree, allp, HOLE_FAR_M, idx.shapely)
    far = np.ones(len(allp), dtype=bool)
    far[near_p] = False
    log(f"holes: {far.sum():,} of {len(allp):,} route track points ({far.sum() * STEP_DENSE / 1000:,.0f}"
        f" km) lie more than {HOLE_FAR_M} m from kept NARN")
    return allp[far]


def hole_envelopes(log):
    """Envelopes (lon/lat) round the OSM passenger route track the register lacks, by
    HOLE_CELL grid cell, adjacent cells of one row merged. Reads data/proc/us."""
    segs, _l, _lb = read_narn(RAW / "narn_passenger.geojson", log)
    idx = SegIndex([{"segs": segs}])
    _bm, ways, rels, _stops, coords = load_osm(log)
    rw = route_ways(rels)
    wgeo = {}
    for w in {w for ws in rw.values() for w in ws}:
        if w in ways:
            _n, ll = way_lonlat(w, ways, coords)
            if ll is not None:
                wgeo[w] = densify(ll, STEP_DENSE)
    far = hole_route_points(idx, wgeo, log)
    cells = Counter(zip(np.floor(far[:, 0] / HOLE_CELL).astype(int).tolist(),
                        np.floor(far[:, 1] / HOLE_CELL).astype(int).tolist()))
    keep = sorted(c for c, n in cells.items() if n * STEP_DENSE >= HOLE_MIN_M)
    rows = defaultdict(list)
    for cx, cy in keep:
        rows[cy].append(cx)
    envs = []
    for cy, xs in sorted(rows.items()):
        xs.sort()
        start = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            envs.append((round(start * HOLE_CELL - 0.005, 4), round(cy * HOLE_CELL - 0.005, 4),
                         round((prev + 1) * HOLE_CELL + 0.005, 4),
                         round((cy + 1) * HOLE_CELL + 0.005, 4)))
            if x is not None:
                start = prev = x
    log(f"holes: {len(keep)} cells of {HOLE_CELL} degrees, {len(envs)} envelopes to fetch")
    return envs


def fetch_holes():
    """NARN segments with no passenger code inside the hole envelopes, to
    data/raw/us/narn_holes.geojson. Spatial, not by subdivision name: the track a route runs
    on where NARN codes none can be under any name (Houston - Beaumont is not on NARN's
    passenger-coded Beaumont Subdivision at all)."""
    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)
    envs = hole_envelopes(log)
    feats, ids = [], set()
    for i, (x0, y0, x1, y1) in enumerate(envs):
        offset = 0
        while True:
            page = _get({"where": HOLE_WHERE, "geometry": f"{x0},{y0},{x1},{y1}",
                         "geometryType": "esriGeometryEnvelope", "inSR": 4326,
                         "spatialRel": "esriSpatialRelIntersects", "outFields": "*",
                         "f": "geojson", "outSR": 4326, "orderByFields": "OBJECTID",
                         "resultOffset": offset, "resultRecordCount": PAGE})
            got = page.get("features") or []
            for f in got:
                k = f["properties"].get("FRAARCID")
                if k not in ids:
                    ids.add(k)
                    feats.append(f)
            if len(got) < PAGE:
                break
            offset += PAGE
        if i % 25 == 0 or i == len(envs) - 1:
            log(f"  envelope {i + 1}/{len(envs)}: {len(feats):,} segments")
    out = RAW / HOLES_FILE
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "fetched": time.strftime("%Y-%m-%d"),
                   "where": HOLE_WHERE, "envelopes": envs, "features": feats}, f)
    tmp.replace(out)
    log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(feats):,} segments)")


def accept_holes(holes, idx, far_pts, log):
    """The hole segments an OSM passenger route runs over: of each segment's part further
    than HOLE_FAR_M from passenger NARN (at least HOLE_MIN_M of it), HOLE_SHARE lies within
    HOLE_NEAR_M of route track the register lacks. A segment beside a passenger-coded one
    (a third track, a siding) is never taken: the route there is the register's already."""
    import shapely
    from shapely import STRtree
    if not holes or not len(far_pts):
        return []
    ftree = STRtree(shapely.points(merc(far_pts[:, 0], far_pts[:, 1])))
    took, part = [], []
    for s in holes:
        a = densify(np.asarray(s["pts"], dtype=np.float64), STEP_DENSE)
        near_p, _g = within_many(idx.tree, a, HOLE_FAR_M, shapely)
        away = np.ones(len(a), dtype=bool)
        away[near_p] = False
        if away.sum() * STEP_DENSE < HOLE_MIN_M:
            continue
        A = a[away]
        on_p, _g = within_many(ftree, A, HOLE_NEAR_M, shapely)
        share = len(set(on_p.tolist())) / len(A)
        if share >= HOLE_SHARE:
            took.append(s)
        elif share >= 0.2:
            part.append((s["km"], share, s["owner"], s["raw"]))
    log(f"holes: {len(took)} of {len(holes)} non-passenger segments taken "
        f"({sum(s['km'] for s in took):,.1f} km): an OSM passenger route runs over them; "
        f"{len(part)} more ({sum(p[0] for p in part):,.1f} km) only partly (20-{HOLE_SHARE:.0%})"
        f", left out")
    for km, share, owner, raw in sorted(part, reverse=True)[:15]:
        log(f"    partly: {km:6.1f} km  {share:.0%}  {owner} {raw}")
    return took


def group_holes(took, lines, log):
    """Hole lines: the taken segments grouped as passenger ones are, kept apart from the
    passenger lines (so a section never swaps the passenger-coded track for a hole), each
    flagged `hole`. Unnamed pieces join a hole line they touch, else are lines of their own.
    A hole line named as a passenger line of the same owner it touches is its other track:
    "Elko Subdivision (second track)"."""
    named = [s for s in took if s["key"]]
    hl, _left = group_lines(named, lambda m: None) if named else ([], [])
    at_node = defaultdict(set)
    for i, l in enumerate(hl):
        for s in l["segs"]:
            at_node[s["a"]].add(i)
            at_node[s["b"]].add(i)
    for comp in components([s for s in took if not s["key"]]):
        touch = Counter(i for s in comp for n in (s["a"], s["b"]) for i in at_node.get(n, ()))
        if touch:
            hl[touch.most_common(1)[0][0]]["segs"].extend(comp)
        else:
            own = Counter(s["owner"] for s in comp).most_common(1)[0][0]
            hl.append({"key": "", "field": "", "owner": own, "segs": comp, "pieces": 1,
                       "split": False})
    # A hole piece meeting the passenger line of its own name and owner ONLY at that line's
    # ends fills a gap in it (NARN coded a stretch of the subdivision without passengers) and
    # becomes part of it; one meeting it where the line runs on is its other track, and stays
    # a line of its own so no section swaps the passenger-coded track for it.
    by_name = defaultdict(list)
    for i, l in enumerate(lines):
        by_name[(l["key"], l["owner"])].append(i)
    deg_cache = {}

    def degrees(i):
        if i not in deg_cache:
            nb = defaultdict(set)
            for s in lines[i]["segs"]:
                nb[s["a"]].add(s["b"])
                nb[s["b"]].add(s["a"])
            deg_cache[i] = {n: len(v) for n, v in nb.items()}
        return deg_cache[i]
    out = []
    for l in hl:
        l["hole"] = True
        l["second"] = False
        nodes = {n for s in l["segs"] for n in (s["a"], s["b"])}
        host = None
        for i in by_name.get((l["key"], l["owner"]), ()) if l["key"] else ():
            deg = degrees(i)
            touch = nodes & set(deg)
            if not touch:
                continue
            if all(deg[n] == 1 for n in touch):
                host = i
            else:
                l["second"] = True
                l["main"] = i
            break
        if host is not None:
            lines[host]["segs"].extend(l["segs"])
            lines[host]["filled"] = lines[host].get("filled", 0.0) + sum(s["km"] for s in l["segs"])
            log(f"    hole fills a gap: {sum(s['km'] for s in l['segs']):7.1f} km  "
                f"{l['owner']:<5} {display_name(l['key'], l['field'])}")
            continue
        out.append(l)
    hl = out
    for l in sorted(hl, key=lambda l: -sum(s["km"] for s in l["segs"])):
        log(f"    hole line: {sum(s['km'] for s in l['segs']):7.1f} km  {l['owner']:<5} "
            f"{display_name(l['key'], l['field']) if l['key'] else '(unnamed)'}"
            f"{' (second track)' if l['second'] else ''}  "
            f"{''.join(sorted({s['state'] for s in l['segs']}))}")
    return hl


# ---------------------------------------------------------------- sections over the NARN graph

class LineGraph:
    """One line's track as a graph over NARN nodes, with stations inserted mid-segment.

    Edges are (node, node, km, chain, pts): `chain` is NARN's own KM for the piece (its
    share of the segment's KM by geometric length), `pts` the geometry from the first node to
    the second. Inserted nodes take negative ids."""

    def __init__(self, segs):
        self.segs = segs
        self.cuts = defaultdict(list)            # seg index -> [(t, node)]
        self.fresh = 0

    def insert(self, si, t):
        """A new node at fraction t (by length) along segment si; returns its id."""
        for tt, n in self.cuts[si]:
            if abs(tt - t) * self.segs[si]["geo_km"] * 1000 < 1.0:
                return n
        self.fresh -= 1
        self.cuts[si].append((t, self.fresh))
        return self.fresh

    def build(self):
        adj = defaultdict(list)
        pos = {}
        for si, s in enumerate(self.segs):
            pts = s["pts"]
            pos[s["a"]], pos[s["b"]] = pts[0], pts[-1]
            cuts = sorted(self.cuts.get(si, ()))
            pieces = split_line(pts, [t for t, _n in cuts])
            nodes = [s["a"]] + [n for _t, n in cuts] + [s["b"]]
            geo = s["geo_km"] or 1e-9
            for (u, v), pp in zip(zip(nodes[:-1], nodes[1:]), pieces):
                km = path_m(pp) / 1000
                chain = s["km"] * km / geo
                pos[u], pos[v] = pp[0], pp[-1]
                e = (u, v, km, chain, pp, s["id"])
                adj[u].append((v, e))
                adj[v].append((u, e))
        self.adj, self.pos = adj, pos
        return adj


def split_line(pts, ts):
    """Cut a polyline at fractions ts (sorted) of its length: len(ts)+1 pieces."""
    if not ts:
        return [list(pts)]
    seg = [dist_m(*a, *b) for a, b in zip(pts[:-1], pts[1:])]
    total = sum(seg) or 1e-9
    out, cur, acc, i = [], [pts[0]], 0.0, 0
    for t in ts:
        target = t * total
        while i < len(seg) and acc + seg[i] < target:
            acc += seg[i]
            i += 1
            cur.append(pts[i])
        if i >= len(seg):
            p = pts[-1]
        else:
            f = (target - acc) / seg[i] if seg[i] else 0.0
            a, b = pts[i], pts[i + 1]
            p = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        cur.append(p)
        out.append(cur)
        cur = [p]
    cur.extend(pts[i + 1:])
    out.append(cur)
    return out


def absorbing_sections(adj, at):
    """Sections between section ends `at` (node -> station id): a Dijkstra from each end that
    stops at the first other end on every branch (schienennetz.py). Returns {(sa, sb): {km,
    chain, path: [edge, ...] oriented sa->sb, nodes: [...]}} with sa < sb."""
    sections = {}
    for src, sid in at.items():
        dist, prev, seen = {src: 0.0}, {}, set()
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in seen:
                continue
            seen.add(u)
            other = at.get(u)
            if u != src and other is not None:
                if other != sid:
                    key = (sid, other) if sid < other else (other, sid)
                    if key not in sections or d < sections[key]["km"] - 1e-9:
                        edges, nodes, cur = [], [u], u
                        while cur != src:
                            p, e = prev[cur]
                            edges.append(e)
                            nodes.append(p)
                            cur = p
                        edges.reverse()
                        nodes.reverse()
                        if sid > other:
                            edges.reverse()
                            nodes.reverse()
                        sections[key] = {"km": d, "edges": edges, "nodes": nodes,
                                         "chain": sum(e[3] for e in edges)}
                continue                                    # absorbed
            for v, e in adj.get(u, ()):
                nd = d + e[2]
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    prev[v] = (u, e)
                    heapq.heappush(heap, (nd, v))
    return sections


SHARED_M = 30          # a node of one section this close to another lies on its track
SHARED_RUN_M = 300     # and two sections sharing at least this much have a branch point


def branch_points(sections, at, adj, pos):
    """Nodes where two sections that run over the same track part company.

    Pairwise and by position, not by shared edge ids: where NARN files the two tracks of a
    double-track line as two segments, two sections can share a stretch of route on different
    tracks. A node of P within SHARED_M of Q's geometry lies on Q; a run of such nodes at least
    SHARED_RUN_M long ends where P leaves Q, and the branch point is the nearest node of the
    run, walking back from that end, where three ways meet (the switch itself, not the first
    node of the arm that happens to lie within SHARED_M)."""
    import shapely
    from shapely.geometry import LineString
    deg = {n: len({v for v, _e in nb}) for n, nb in adj.items()}
    keys = list(sections)
    info = {}
    for k in keys:
        v = sections[k]
        g = np.asarray(section_geometry(v), dtype=np.float64)
        nodes = v["nodes"]
        npos = np.asarray([pos[n] for n in nodes], dtype=np.float64)
        cum = np.concatenate([[0.0], np.cumsum([e[2] for e in v["edges"]])]) * 1000
        info[k] = {"line": LineString(merc(g[:, 0], g[:, 1])),
                   "pts": shapely.points(merc(npos[:, 0], npos[:, 1])),
                   "scale": merc_scale(float(npos[:, 1].mean())),
                   "bbox": (g[:, 0].min(), g[:, 1].min(), g[:, 0].max(), g[:, 1].max()),
                   "nodes": nodes, "cum": cum}
    out = set()
    for p in keys:
        P = info[p]
        for q in keys:
            if p == q:
                continue
            Q = info[q]
            a, b = P["bbox"], Q["bbox"]
            if a[2] < b[0] - 0.01 or b[2] < a[0] - 0.01 or a[3] < b[1] - 0.01 or b[3] < a[1] - 0.01:
                continue
            near = shapely.distance(P["pts"], Q["line"]) / P["scale"] <= SHARED_M
            if near.sum() < 2:
                continue
            nodes, cum, n = P["nodes"], P["cum"], len(P["nodes"])
            i = 0
            while i < n:
                if not near[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < n and near[j + 1]:
                    j += 1
                if cum[j] - cum[i] >= SHARED_RUN_M:
                    if j < n - 1:                     # P leaves Q after node j
                        for t in range(j, i - 1, -1):
                            if deg.get(nodes[t], 0) >= 3:
                                if nodes[t] not in at:
                                    out.add(nodes[t])
                                break
                    if i > 0:                         # P joins Q at node i
                        for t in range(i, j + 1):
                            if deg.get(nodes[t], 0) >= 3:
                                if nodes[t] not in at:
                                    out.add(nodes[t])
                                break
                i = j + 1
    return out


def track_apart_m(geo_a, geo_b):
    """How far line a's drawn track lies from line b's: the 90th percentile, in metres, of
    points every ~200 m along a."""
    from shapely import STRtree
    from shapely.geometry import LineString

    def xy(lon, lat):
        return lon * 111320.0 * math.cos(math.radians(lat)), lat * 110570.0

    def lines(geo):
        return [LineString([xy(x, y) for x, y in pts]) for pts in geo.values() if len(pts) > 1]
    lb = lines(geo_b)
    if not lb:
        return INF
    tree = STRtree(lb)
    ds = []
    for s in lines(geo_a):
        n = max(2, int(s.length / 200))
        for i in range(n):
            p = s.interpolate(i / (n - 1), normalized=True)
            ds.append(lb[int(tree.nearest(p))].distance(p))
    return float(np.percentile(ds, 90)) if ds else INF


def section_geometry(sec):
    pts = []
    for e, u in zip(sec["edges"], sec["nodes"][:-1]):
        pp = e[4] if e[0] == u else e[4][::-1]
        pts.extend(pp if not pts else pp[1:])
    return pts


END_SPREAD_DEG = 40    # a node whose track all leaves within this many degrees is a line end
BEARING_M = 300        # ... measured over this much of each edge


def bearing_out(e, u):
    """Bearing (degrees) of edge e leaving node u, over its first BEARING_M."""
    pts = e[4] if e[0] == u else e[4][::-1]
    x0, y0 = pts[0]
    acc, (x1, y1) = 0.0, pts[-1]
    for a, b in zip(pts[:-1], pts[1:]):
        acc += dist_m(*a, *b)
        if acc >= BEARING_M:
            x1, y1 = b
            break
    dx = (x1 - x0) * math.cos(math.radians(y0))
    return math.degrees(math.atan2(y1 - y0, dx))


def line_ends(adj):
    """Nodes where a line ends: one neighbour, or every track at the node leaving the same
    way. The second is the end of a double-track line whose two tracks are two NARN segments
    meeting at the last node (New River, Alhambra): two neighbours, and no line beyond."""
    ends = set()
    for n, nb in adj.items():
        if len({v for v, _e in nb}) == 1:
            ends.add(n)
            continue
        bs = [bearing_out(e, n) for _v, e in nb]
        spread = max(abs((a - b + 180) % 360 - 180) for a in bs for b in bs)
        if spread <= END_SPREAD_DEG:
            ends.add(n)
    return ends


def narn_sections(lg, stop_at):
    """Sections of one line given its stops ({node: station id}); line ends become junctions
    "uj<node>", then branch points, until no two sections share track."""
    adj = lg.build()
    at = dict(stop_at)
    for n in line_ends(adj):
        if n not in at:
            at[n] = f"uj{n}"
    if len(set(at.values())) < 2:
        return {}, at
    for _round in range(8):
        secs = absorbing_sections(adj, at)
        extra = branch_points(secs, at, adj, lg.pos)
        if not extra:
            break
        for n in extra:
            at[n] = f"uj{n}"
    return secs, at


# ---------------------------------------------------------------- lines in pieces

# A piece with fewer than two stops and shorter than this is not made a line: a stub of a few
# metres between a station and a line end beside it (Toronto Union's crossovers, 0.04 km).
PIECE_MIN_KM = 0.1
# {line id: [the ids of the pieces split off it]}, filled by split_pieces. A ride saved on the
# line between two stations now on another piece keeps the line's id, which still exists, so no
# line alias moves it; this is the map an app migration would need (us_sources.md).
LINE_PIECES = {}
# (line id, "a|b") -> the NARN segment ids under the section, and NARN's km for it; filled by
# build for split_pieces, which runs inside build_model after it has dropped `chain`.
SECTION_SEGS = {}
SECTION_CHAIN = {}


def section_pieces(sections):
    """The connected pieces of a line's sections, biggest (km) first, each a list of indexes
    into `sections`."""
    uf = UF()
    for s in sections:
        uf.union(s[0], s[1])
    by = defaultdict(list)
    for i, s in enumerate(sections):
        by[uf.find(s[0])].append(i)
    return sorted(by.values(), key=lambda ix: (-sum(sections[i][2] for i in ix),
                                               min(min(sections[i][:2]) for i in ix)))


def piece_id(lid, seg_ids):
    """The id of a piece split off line `lid`: its id and the lowest NARN segment id the
    piece's sections run over, so it is stable while that track is in the piece."""
    tag = f"{lid}|piece|{min(seg_ids)}"
    return "u" + hashlib.blake2b(tag.encode("utf-8"), digest_size=5).hexdigest()


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """A register line whose sections do not all connect becomes one line per connected piece
    (Anita, 2026-10-04: a trip is entered station to station, so a line cannot be ridden
    across a gap in it). NARN names track by its owner's subdivision, and a subdivision can be
    in pieces for good: the New York Terminal Subdivision is the Empire Connection and a stub
    at Sunnyside; Amtrak's Michigan Line runs over CN's track through Battle Creek; a stretch
    no train runs over is left out (UNROUTED_SHARE here, build_model's junction-section drop).

    A hook build_model calls (`split_pieces` on the register module) once it has dropped the
    junction-ended sections no route runs over, so the pieces are the ones that ship. Not in
    build(): splitting before that drop gave pieces build_model then dropped whole, so a line's
    old id went to a piece that never shipped (Provo Subdivision (second track): its 11.5 km
    piece took the id and was dropped, the 7.8 km piece that ships came out under a new id),
    and a line's ways divided between pieces matched differently (the MassDOT Framingham
    Subdivision's Foxboro and Walpole stubs were both dropped). The grouping still joins
    pieces within JOIN_KM first, so the holes file can fill a gap (the Elko Subdivision's
    214 km); only what is still apart at the end is split.

    The biggest piece (km) keeps the line's id, so rides saved on it keep their line; each
    other piece takes `piece_id`. The pieces keep the name, and an English name tells them
    apart by their end stops (end junctions where a piece has under two stops), as two lines
    of one name more than JOIN_KM apart are. A piece other than the biggest with under two
    stops and under PIECE_MIN_KM is left out. reg_ways ({way id: {line id}}) and
    state["sec_ways"] ({(line id, "a|b"): [(way index, ...)]}, register_way_lines) are moved
    to the pieces, so clicks and ownership see them."""
    from n02 import walk_order
    LINE_PIECES.clear()
    sec_ways = (state or {}).get("sec_ways") or {}
    wids = (state or {}).get("wids") or []
    made, gone, pieces_of = [], [], {}
    for l in list(lines):
        if l.get("src") == "osm" or not l.get("sections"):
            continue
        ps = section_pieces(l["sections"])
        if len(ps) == 1:
            continue
        lid, secs, name = l["id"], l["sections"], l["name"]
        old_geo = geoms.get(lid, {})
        parts = []
        for k, ix in enumerate(ps):
            sub = [secs[i] for i in ix]
            keys = [f"{s[0]}|{s[1]}" for s in sub]
            ends = {s for sec in sub for s in sec[:2]}
            stops = sum(1 for s in ends if not stations[s].get("junction"))
            km = round(sum(s[2] for s in sub), 3)
            if k > 0 and stops < 2 and km < PIECE_MIN_KM:
                gone.append((km, name, l["operator"], sorted(ends)))
                parts.append((None, keys, ends))
                continue
            if k == 0:
                p = l
            else:
                ids = set()
                for key in keys:
                    ids |= set(SECTION_SEGS.get((lid, key), ()))
                p = dict(l)
                p["id"] = piece_id(lid, ids or {min(keys)})
                lines.append(p)
            p["sections"] = sub
            p["km"] = km
            p["km_official"] = round(sum(SECTION_CHAIN.get((lid, key), s[2])
                                         for key, s in zip(keys, sub)), 3)
            p["display"] = display_order(sub, walk_order)
            parts.append((p, keys, ends))
        # Ways: each way the line had goes to the pieces whose sections it lies beside; one
        # beside none of them (it was beside a dropped section) stays with the biggest piece.
        beside = []
        for p, keys, _ends in parts:
            ws = set()
            for key in keys:
                ws |= {wids[j] for j, *_r in sec_ways.get((lid, key), ())}
            beside.append((p, ws))
        for wid, ls in reg_ways.items():
            if lid in ls:
                to = {p["id"] for p, ws in beside if p is not None and wid in ws}
                ls.discard(lid)
                ls |= to or {lid}
        for p, keys, _ends in parts:
            for key in keys:
                got = sec_ways.pop((lid, key), None)
                if got is not None and p is not None:
                    sec_ways[(p["id"], key)] = got
        geoms[lid] = {}
        for sid in {s for sec in secs for s in sec[:2]}:
            stations[sid]["lines"].discard(lid)
        for p, keys, ends in parts:
            if p is None:
                continue
            geoms[p["id"]] = {key: old_geo[key] for key in keys if key in old_geo}
            for sid in ends:
                stations[sid]["lines"].add(p["id"])
        kept = [p for p, _k, _e in parts if p is not None]
        if len(kept) == 1:              # only a stub left out: one line still, its own name
            continue
        for p in kept:
            stops_ = [s for s in p["display"] if not stations[s].get("junction")]
            if len(stops_) < 2:
                stops_ = [p["display"][0], p["display"][-1]] if p["display"] else []
            if len(stops_) >= 2 and stops_[0] != stops_[-1]:
                p["name_en"] = (f"{name} ({stations[stops_[0]]['name']} – "
                                f"{stations[stops_[-1]]['name']})")
        LINE_PIECES[lid] = [p["id"] for p in kept[1:]]
        pieces_of[lid] = [p["id"] for p in kept]
        made.append((l["km"] + sum(p["km"] for p in kept[1:]), name, l["operator"],
                     [p["km"] for p in kept]))
    # A second track folded into a line now in pieces: into the nearest piece.
    for l in lines:
        main = l.get("companion_of")
        if main in pieces_of and len(pieces_of[main]) > 1:
            l["companion_of"] = min(pieces_of[main],
                                    key=lambda m: track_apart_m(geoms[l["id"]], geoms[m]))
    log(f"lines in pieces: {len(made)} register lines whose sections do not all connect made "
        f"{sum(len(m[3]) for m in made)} lines, one per piece; {len(gone)} stubs under "
        f"{PIECE_MIN_KM} km with under two stops left out ({sum(g[0] for g in gone):.2f} km)")
    for km, name, op, kms in sorted(made, reverse=True):
        log(f"    {km:7.1f} km  {name} [{op}] -> " + " + ".join(f"{k:.1f}" for k in kms))
    for km, name, op, ends in sorted(gone, reverse=True):
        log(f"    left out: {km:.3f} km  {name} [{op}]  {' - '.join(ends)}")


# ================================================================ the OSM half

def load_osm(log):
    import build_model as bm
    ways, rels, stops, cid, cx, cy = bm.load("us", log)
    return bm, ways, rels, stops, bm.Coords(cid, cx, cy)


# Excursion trains OSM maps as route=train without service=tourism: the Great Smoky Mountains
# Railroad's ("Local"), Strasburg, Steamtown, Andrews Valley Rail Tours, "Path to Boomtown".
TOURIST_ROUTE = re.compile(r"Excursion|Rail Tour|Strasburg|Steamtown|Boomtown|Scenic|Dinner",
                           re.IGNORECASE)


def is_passenger_route(tags):
    return (tags.get("type") == "route" and tags.get("route") == "train"
            and tags.get("service") not in ("tourism", "tourist", "heritage", "historic")
            and tags.get("network") not in ("Local", "local")
            and not TOURIST_ROUTE.search(f"{tags.get('name', '')} {tags.get('operator', '')}"))


def route_ways(rels):
    """{rel id: [way id]} for passenger train routes: their roleless or forward/backward way
    members, which is the track the train runs on (platforms are not)."""
    out = {}
    for rid, (tags, members) in rels.items():
        if is_passenger_route(tags):
            out[rid] = [ref for ty, ref, role in members
                        if ty == "w" and (not role or role.startswith(("forward", "backward")))]
    return out


def way_lonlat(wid, ways, coords):
    nodes = ways[wid][1]
    pos, ok = coords.many(nodes)
    if ok.sum() < 2:
        return None, None
    pos = pos[ok]
    return nodes[ok], np.column_stack([coords.x[pos] / 1e7, coords.y[pos] / 1e7])


def densify(arr, step_m):
    """Points along a lon/lat polyline at most step_m apart (vertices kept)."""
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) < 2:
        return arr
    a, b = arr[:-1], arr[1:]
    d = np.hypot((b[:, 0] - a[:, 0]) * np.cos(np.radians(a[:, 1])) * 111320,
                 (b[:, 1] - a[:, 1]) * 110570)
    k = (d // step_m).astype(np.int64) + 1          # pieces per segment
    seg = np.repeat(np.arange(len(a)), k)
    start = np.repeat(np.cumsum(k) - k, k)
    t = (np.arange(k.sum()) - start) / np.repeat(k, k)
    pts = a[seg] + (b[seg] - a[seg]) * t[:, None]
    return np.vstack([pts, arr[-1:]])


class SegIndex:
    """Every kept NARN segment in Web Mercator, for nearest-line questions."""

    def __init__(self, lines):
        import shapely
        from shapely import STRtree
        from shapely.geometry import LineString
        self.geo, self.ref = [], []
        for li, l in enumerate(lines):
            for si, s in enumerate(l["segs"]):
                a = np.asarray(s["pts"], dtype=np.float64)
                if len(a) < 2:
                    continue
                self.geo.append(LineString(merc(a[:, 0], a[:, 1])))
                self.ref.append((li, si))
        self.line_of = np.asarray([r[0] for r in self.ref])
        self.tree = STRtree(self.geo)
        self.shapely = shapely

    def within(self, lonlat, r_m):
        """Pairs (point index, segment index) of points within r_m of a segment. The scale is
        taken at the first point, which is fine within a few km."""
        sc = merc_scale(lonlat[0][1])
        pts = self.shapely.points(merc(lonlat[:, 0], lonlat[:, 1]))
        return self.tree.query(pts, predicate="dwithin", distance=r_m * sc)

    def project(self, j, lon, lat):
        """(fraction along segment j nearest a point, true metres to it)."""
        p = self.shapely.points(merc([lon], [lat]))[0]
        return (self.geo[j].project(p, normalized=True),
                self.geo[j].distance(p) / merc_scale(lat))


def place_stations(lines, idx, stations, served, rw, wgeo, log, not_on=None):
    """Which register lines each served station goes on, and where on each.

    served: {station nid: set(rel id)}; rw: {rel id: [way id]}; wgeo: {way id: densified
    lon/lat array}; not_on: resolve_not_on's {nid: {(owner, key)} or {"*"}}. Returns {line
    index: [(seg index, t, station nid, n routes)]}."""
    placed = defaultdict(list)
    n_on = n_off = 0
    off_examples = []
    not_on = not_on or {}
    took = Counter()                  # (station nid, owner, key or "*") -> placements taken off
    for nid, rids in served.items():
        s = stations[nid]
        if "*" in not_on.get(nid, ()):
            took[(nid, None, "*")] += 1
            continue
        lon, lat = s["lon"], s["lat"]
        here = np.asarray([[lon, lat]])
        pairs = idx.within(here, STATION_M)
        if pairs.shape[1] == 0:
            continue
        cand = {}
        for j in pairs[1]:
            j = int(j)
            t, d = idx.project(j, lon, lat)
            li = int(idx.line_of[j])
            if li not in cand or d < cand[li][2]:
                cand[li] = (j, t, d)
        # The station's own routes' track near it.
        pts = []
        for rid in rids:
            for w in rw.get(rid, ()):
                a = wgeo.get(w)
                if a is None:
                    continue
                m = ((np.abs(a[:, 0] - lon) * math.cos(math.radians(lat)) * 111320 <= ALONG_R_M)
                     & (np.abs(a[:, 1] - lat) * 110570 <= ALONG_R_M))
                if m.any():
                    pts.append(a[m])
        if not pts:
            continue
        P = np.unique(np.round(np.vstack(pts), 6), axis=0)
        near = idx.within(P, NEAR_M)
        along = defaultdict(set)
        for pi, j in zip(near[0].tolist(), near[1].tolist()):
            along[int(idx.line_of[j])].add(pi)
        # KNOWN FAULT (2026-10-02): a station whose route runs beside another line for ALONG_M
        # goes on that line too. Long Island City's LIRR track runs beside Amtrak's East River
        # Tunnels approach (the West Subdivision), so the West Subdivision has a section Long
        # Island City - New York Penn Station no train runs. Requiring the route's track at the
        # station (within 150 m) to be on the line took 12 real stations off their only line
        # (Tampa Union Station, Miami Central); preferring such lines took 74 line-end
        # placements at multi-line stations (Jamaica, Ogilvie, Toronto Union); requiring a stop
        # node within 25 m of a line for placements over 500 m from its ends took real ones
        # (Tucson, Sacramento, Oshawa: stop nodes on platform tracks 30-60 m off NARN's main)
        # and still left Long Island City on. What would work: which register line OWNS the
        # way a stop node lies on (ownership runs after placement here, so placement needs its
        # own way-to-line match first).
        # Looked at 2026-10-03: that cannot reach LIC. No OSM route lists LIC or Hunterspoint
        # Avenue, so both come from unlisted_stations, with no stop node, and take every route
        # passing within UNLISTED_M: 60 Amtrak (36-45 m off) and 10 LIRR Penn trains (16-24 m),
        # all on the East River tunnel approach, which is the West Subdivision. The tracks
        # nearest LIC's station node (3-10 m) are LIC Yard ways no NARN line lies within 60 m
        # of; NARN's Main Line ends 326 m from it, the West Subdivision passes 83 m off. Keeping
        # only the line nearest an unlisted station (25 m tie) takes 13 placements off, mostly
        # right (Hunterspoint Avenue off the West Subdivision, Wakefield off the New Haven Line,
        # Mansfield off CSX's Framingham Subdivision), but LIC off its own Main Line and Mineola
        # off the branch ending there; matching the station's network tag to the routes' changes
        # only Mansfield.
        # So since 2026-10-03 the few such cases are set by hand: NOT_ON (Anita's yes to a
        # short explicit list), checked by name and position so a renamed station fails.
        got = False
        for li, (j, t, d) in cand.items():
            if len(along.get(li, ())) * STEP_DENSE < ALONG_M:
                continue
            ok = (lines[li]["owner"], lines[li]["key"])
            if ok in not_on.get(nid, ()):
                took[(nid,) + ok] += 1
                continue
            placed[li].append((idx.ref[j][1], t, nid, len(rids)))
            got = True
        if got:
            n_on += 1
        else:
            n_off += 1
            if len(off_examples) < 12:
                off_examples.append(s["name"])
    log(f"stations: {n_on} served rail stations placed on a register line; {n_off} lie within "
        f"{STATION_M} m of one that none of their own routes runs along, or are overridden "
        f"(e.g. {', '.join(off_examples)})")
    for (nid, owner, key), n in sorted(took.items(), key=lambda kv: stations[kv[0][0]]["name"]):
        log(f"    override: {stations[nid]['name']} (n{nid}) "
            + ("is no stop" if key == "*" else f"taken off {owner} {key} ({n} piece(s))"))
    for nid, keys in sorted(not_on.items()):
        for k in keys:
            kk = (nid, None, "*") if k == "*" else (nid,) + k
            if kk not in took:
                log(f"    override not needed: {stations[nid]['name']} (n{nid}) "
                    f"{'*' if k == '*' else ' '.join(k)}: no placement to take off")
    return placed


STEP_DENSE = 25.0      # route track is sampled this often when measuring how far it runs along
UNLISTED_M = 150       # an unlisted station this close to a passenger route's track is on it

# ---------------------------------------------------------------- station overrides

# Stations no rule places right, set by hand (Anita, 2026-10-03: a short, explicit list).
# Each row: (the station's OSM name, (lon, lat) it lies within OVERRIDE_M of, the register
# line's owner mark and key as clean_name writes it, why). The station goes on no line of that
# owner and key, and is no unlisted stop of an OSM route running along that line there
# (osm_extra_stops). Line "*" (owner None): the station is no stop at all, on any register
# line or OSM route; for depots kept as museums, closed stations and the like that OSM still
# tags as working stations (unlisted_stations takes them otherwise).
# A row whose name no station within OVERRIDE_M of its point carries, or whose owner and key
# no register line has, stops the build: a renamed or moved station fails loudly rather than
# silently going back on the line. A row that takes nothing off is logged ("not needed").
# Found 2026-10-03 by listing register sections with an end at an unlisted station
# (us_sources.md, "Station overrides"). Canada's rows are ca_register.CA_NOT_ON.
NOT_ON = [
    # The LIRR's Long Island City branch terminal and Hunterspoint Avenue: no OSM route lists
    # them (their few weekday trains are not mapped), so unlisted_stations takes every route
    # passing within UNLISTED_M, which are Amtrak's and the LIRR's Penn Station trains on
    # the East River tunnel approach: the West Subdivision had a section Long Island City -
    # New York Penn Station no train runs. They stay on the LIRR's Main Line.
    ("Long Island City", (-73.95698, 40.74151), "AMTK", "WEST",
     "LIRR terminal beside the tunnel approach; no train through the tunnels stops"),
    ("Hunterspoint Avenue", (-73.94756, 40.74205), "AMTK", "WEST",
     "LIRR station beside the tunnel approach; no train through the tunnels stops"),
    # New Providence is a Gladstone Branch stop, 560 m from the Morristown Line where the two
    # part west of Summit; the old NJ Transit Gladstone relation lists a Morristown Line way
    # there, so it went on the Morristown Line too, between Chatham and Summit (2026-10-08).
    ("New Providence", (-74.38642, 40.71211), "NJT", "MORRISTOWN LINE",
     "Gladstone Branch stop; the Morristown Line passes 560 m off"),
    # Not stops of any train the map counts (each was a stop on a register line only because
    # a passenger route passes within UNLISTED_M):
    ("Cedar Park", (-97.82685, 30.52376), None, "*",
     "the Austin Steam Train Association's depot; MetroRail has no Cedar Park station"),
    ("Longhorn and Western Train Platform", (-98.43474, 29.54759), None, "*",
     "the Texas Transportation Museum's miniature railway"),
    ("SMART Central at Wilsonville Station", (-122.77563, 45.31086), None, "*",
     "Wilsonville's bus centre; WES's Wilsonville station is its own record, 20 m off"),
    ("Tennessee Central Railway Museum", (-86.75406, 36.15548), None, "*",
     "a museum depot beside WeGo Star's line"),
    ("Smiths Creek Depot", (-83.22844, 42.30750), None, "*",
     "Greenfield Village's depot, on its heritage railway beside the Michigan Line"),
    ("Paradise", (-76.11395, 40.00560), None, "*",
     "the Strasburg Rail Road's stop at Leaman Place; Keystone trains pass it"),
    ("Eagle", (-75.41372, 40.04886), None, "*",
     "no SEPTA or Amtrak stop: a record with no railway tag between Strafford and Devon"),
    ("Marceline", (-92.94754, 39.71588), None, "*",
     "the Santa Fe depot; the Southwest Chief stops at La Plata, not Marceline"),
    ("Clifton", (-77.38668, 38.78105), None, "*",
     "VRE stops here only for the yearly Clifton Day"),
    ("Lexington", (-80.25165, 35.82069), None, "*",
     "no Piedmont or Carolinian stop (a Lexington NC station is planned, not open); it split "
     "NS's Danville District into 28 + 26 km. Decided 2026-10-03"),
    # Northstar (Big Lake - Minneapolis) has no OSM route: its stations were on BNSF's Staples
    # Subdivision only because the Empire Builder passes them.
    ("Fridley", (-93.27102, 45.07885), None, "*", "Northstar station; no OSM route serves it"),
    ("Coon Rapids-Riverdale", (-93.35151, 45.19104), None, "*",
     "Northstar station; no OSM route serves it"),
    ("Anoka", (-93.38407, 45.20775), None, "*", "Northstar station; no OSM route serves it"),
    ("Ramsey", (-93.46179, 45.23192), None, "*", "Northstar station; no OSM route serves it"),
    ("Elk River", (-93.54229, 45.28263), None, "*", "Northstar station; no OSM route serves it"),
    ("Big Lake", (-93.72999, 45.32977), None, "*", "Northstar station; no OSM route serves it"),
]
OVERRIDE_M = 300


def resolve_not_on(st, log, rows=None):
    """{station nid: set of (owner, key), or {"*"}} for NOT_ON, by name within OVERRIDE_M of
    each row's point. Fails loudly on a row no station matches."""
    rows = NOT_ON if rows is None else rows
    by_name = defaultdict(list)
    for nid, s in st.items():
        by_name[s["name"]].append(nid)
    off, missing = defaultdict(set), []
    for name, (lon, lat), owner, key, _why in rows:
        hit = [n for n in by_name.get(name, ())
               if dist_m(lon, lat, st[n]["lon"], st[n]["lat"]) <= OVERRIDE_M]
        if not hit:
            near = sorted((dist_m(lon, lat, st[n]["lon"], st[n]["lat"]), n)
                          for n in by_name.get(name, ()))
            missing.append(f"{name!r} near {lon}, {lat}"
                           + (f" (nearest of that name {near[0][0]:,.0f} m off)" if near
                              else " (no station of that name)"))
            continue
        for n in hit:
            off[n].add("*" if key == "*" else (owner, key))
    if missing:
        raise SystemExit("station overrides (NOT_ON): no station matches "
                         + "; ".join(missing) + ". Renamed or moved in OSM? Fix the row.")
    log(f"station overrides: {len(rows)} rows, {len(off)} station records "
        f"({sum(1 for v in off.values() if '*' in v)} no stop at all)")
    return off


def check_not_on_lines(groups, rows=None):
    """Every NOT_ON row's (owner, key) is a register line's, or the build stops."""
    rows = NOT_ON if rows is None else rows
    have = {(g["owner"], g["key"]) for g in groups}
    bad = [f"{owner} {key!r} ({name})" for name, _p, owner, key, _w in rows
           if key != "*" and (owner, key) not in have]
    if bad:
        raise SystemExit("station overrides (NOT_ON): no register line " + "; ".join(bad)
                         + ". Renamed in NARN? Fix the row.")


def unlisted_stations(st, stops, served, rw, wraw, log):
    """Stations no route relation lists, beside track a passenger route runs over.

    OSM's route relations are not always complete: the MBTA's Fitchburg Line lists 4 stops
    (Wachusett, Kendal Green, Porter, North Station) of its 18, the Newburyport/Rockport Line
    one. A station tagged as a working train station (train=yes, or public_transport=station
    on a railway=station or halt) within UNLISTED_M of a passenger route's track is taken as a
    stop of the routes that pass it; place_stations then puts it on the register line those
    routes run along there. Not every railway=station: an old depot kept as a museum is often
    still one, and would put a stop where no train calls. Returns {station nid: set(rel id)}."""
    import shapely
    from shapely import STRtree
    from shapely.geometry import LineString
    users = defaultdict(set)
    for rid, ws in rw.items():
        for w in ws:
            users[w].add(rid)
    wids = [w for w in wraw if len(wraw[w]) >= 2]
    tree = STRtree([LineString(merc(wraw[w][:, 0], wraw[w][:, 1])) for w in wids])
    out, names = {}, []
    other_modes = ("subway", "light_rail", "tram", "monorail", "funicular")
    for nid, s in st.items():
        if nid in served or s.get("_tram") or not s["name"]:
            continue
        tags = stops.get(nid, ({}, 0, 0))[0]
        # A metro or light-rail station, whatever its train tag says: SEPTA's Market-Frankford
        # stations (8th & Market, 11th Street) and the MBTA's Andrew carry train=yes too, and
        # lie over the regional rail tunnel or beside the Old Colony.
        if (tags.get("station") in other_modes
                or any(tags.get(m) == "yes" for m in other_modes)):
            continue
        working = (tags.get("train") == "yes"
                   or (tags.get("public_transport") == "station"
                       and tags.get("railway") in ("station", "halt")))
        if not working:
            continue
        p = shapely.points(merc([s["lon"]], [s["lat"]]))[0]
        hit = tree.query(p, predicate="dwithin", distance=UNLISTED_M * merc_scale(s["lat"]))
        rids = {r for j in hit.tolist() for r in users[wids[j]]}
        if rids:
            out[nid] = rids
            names.append(s["name"])
    log(f"stations: {len(out)} working train stations no route lists, beside a passenger "
        f"route's track, taken as its stops (e.g. {', '.join(sorted(names)[:15])})")
    return out


# An unlisted station this close to a stop the route lists is that stop under another name
# (Harrisburg Transportation Center 48 m from the Keystone's "Harrisburg", Gilroy Transit
# Center 12 m from Caltrain's "Gilroy"), not a stop of its own.
EXTRA_TWIN_M = 300
# Networks whose OSM route relations list every stop: an unlisted station they pass is not
# one of theirs. Amtrak's: on 2026-10-03 what the rule would have added to Amtrak routes was
# the two state-fair halts, Lexington NC, the new Newport News station and Gilroy (Caltrain's,
# which the Coast Starlight passes). VIA's are not: the Corridor's lack Port Hope, Gananoque.
FULL_STOP_LISTS = {"Amtrak"}


def _net_tokens(tags):
    """A station's or route's network and operator, as lower-case tokens, each multi-word
    one also as its initials ("Virginia Railway Express" is also "vre", "Long Island Rail
    Road" "lirr", "Keewatin Railway Company" "krc")."""
    out = set()
    for k in ("network", "operator"):
        for v in (tags.get(k) or "").split(";"):
            v = v.strip().lower()
            if not v:
                continue
            out.add(v)
            words = [w for w in re.split(r"[\s\-/]+", v) if w]
            if len(words) >= 2:
                out.add("".join(w[0] for w in words))
    return out


def osm_extra_stops(ways, rels, stops, coords, stations, resolved, log):
    """Unlisted stations as stops of build_model's OSM lines (rules/us.py and rules/ca.py
    `extra_route_stops`): {route rel id: {station key: set(node id)}}, the nodes of the
    route's own track at the station, which build_model takes as its stop nodes.

    The register lines take a station no route lists from unlisted_stations; OSM lines kept
    only their relation's own stop list, which in the US is often partial ("Port Washington
    Branch (as operated)" lists 5 stops). Here each unlisted station becomes a stop of the
    passing routes, with five limits, since an OSM line is one operator's pattern, not track:
      - the route has a network (those without are excursion trains almost all: the
        Disneyland Railroad would take Tomorrowland's monorail station);
      - the route's network is not one whose relations list every stop (FULL_STOP_LISTS);
      - the station's network or operator (when it has one) must be the route's: NJ Transit's
        Edison is not a stop of the Northeast Regional, which passes it at speed;
      - NOT_ON: a "*" row is no stop of any route, and a line row is no stop of a route that
        runs along that register line at the station (place_stations' test: ALONG_M of the
        route's track within NEAR_M of the line), so Long Island City is no stop of the
        LIRR's Penn Station trains passing it on the West Subdivision;
      - a station within EXTRA_TWIN_M of a stop the route lists is that stop again.
    `stations` and `resolved` are build_model.build_stations' (the same records the register
    half makes)."""
    import shapely
    from shapely import STRtree
    from shapely.geometry import LineString
    import build_model as bm
    rw = route_ways(rels)
    served = defaultdict(set)
    for rid in rw:
        for ref in bm.stop_members(rels[rid][1]):
            s = resolved.get(ref)
            if s is not None:
                served[s].add(rid)
    wraw = {}
    for w in {w for ws in rw.values() for w in ws}:
        if w in ways:
            _n, ll = way_lonlat(w, ways, coords)
            if ll is not None:
                wraw[w] = ll
    unl = unlisted_stations(stations, stops, served, rw, wraw, log)
    not_on = resolve_not_on(stations, log)
    # The register lines NOT_ON names, from the passenger file alone (holes are not needed
    # for a station's own line).
    want = {k for ks in not_on.values() for k in ks if k != "*"}
    line_geo = {}
    if want:
        segs, _l, _lb = read_narn(RAW / "narn_passenger.geojson", lambda m: None)
        for k in want:
            geo = [LineString(merc(np.asarray(s["pts"])[:, 0], np.asarray(s["pts"])[:, 1]))
                   for s in segs if (s["owner"], s["key"]) == k and len(s["pts"]) >= 2]
            if not geo:
                raise SystemExit(f"station overrides (NOT_ON): no register line {k}")
            line_geo[k] = STRtree(geo)

    def runs_along(rid, s, k):
        lon, lat = s["lon"], s["lat"]
        pts = []
        for w in rw.get(rid, ()):
            a = wraw.get(w)
            if a is None:
                continue
            a = densify(a, STEP_DENSE)
            m = ((np.abs(a[:, 0] - lon) * math.cos(math.radians(lat)) * 111320 <= ALONG_R_M)
                 & (np.abs(a[:, 1] - lat) * 110570 <= ALONG_R_M))
            if m.any():
                pts.append(a[m])
        if not pts:
            return False
        P = np.unique(np.round(np.vstack(pts), 6), axis=0)
        hit = line_geo[k].query(shapely.points(merc(P[:, 0], P[:, 1])), predicate="dwithin",
                                distance=NEAR_M * merc_scale(lat))
        return len(set(hit[0].tolist())) * STEP_DENSE >= ALONG_M

    # route -> its own stops' lon/lat: the station records and the stop nodes themselves
    # (Brightline's "West Palm Beach" stop node is at its own platform, its record 690 m off
    # at the Amtrak station, which a name match chose).
    listed = defaultdict(list)
    for s_, rids in served.items():
        for rid in rids:
            listed[rid].append((stations[s_]["lon"], stations[s_]["lat"]))
    for rid in rw:
        for ref in bm.stop_members(rels[rid][1]):
            p = coords.get(ref)
            if p is not None:
                listed[rid].append(p)
    def stop_nodes(rid, s):
        """Nodes of the route's own track at the station, one per way within UNLISTED_M (the
        nearest of each; else the single nearest): the station is placed there exactly, on
        whichever track a variant takes, and a gap the build traces across is cut there."""
        lon, lat = s["lon"], s["lat"]
        best, near = (INF, None), {}
        for w in rw.get(rid, ()):
            a = wraw.get(w)
            if a is None or not (a[:, 0].min() - 0.01 <= lon <= a[:, 0].max() + 0.01
                                 and a[:, 1].min() - 0.01 <= lat <= a[:, 1].max() + 0.01):
                continue
            nodes, ll = way_lonlat(w, ways, coords)
            d = np.hypot((ll[:, 0] - lon) * math.cos(math.radians(lat)) * 111320,
                         (ll[:, 1] - lat) * 110570)
            j = int(np.argmin(d))
            if d[j] < best[0]:
                best = (float(d[j]), int(nodes[j]))
            if d[j] <= UNLISTED_M:
                near[w] = int(nodes[j])
        out = set(near.values())
        return out or ({best[1]} if best[1] is not None else set())

    extra = defaultdict(dict)
    n_pairs = n_net = n_star = n_line = n_nonet = n_twin = n_full = 0
    for nid, rids in unl.items():
        s = stations[nid]
        off = not_on.get(nid, set())
        mine = _net_tokens(stops.get(nid, ({}, 0, 0))[0])
        for rid in sorted(rids):
            n_pairs += 1
            if not rels[rid][0].get("network"):
                n_nonet += 1          # excursion trains, almost all (hole_routes says why)
                continue
            if rels[rid][0].get("network") in FULL_STOP_LISTS:
                n_full += 1
                continue
            if "*" in off:
                n_star += 1
                continue
            if mine and not (mine & _net_tokens(rels[rid][0])):
                n_net += 1
                continue
            if any(runs_along(rid, s, k) for k in off):
                n_line += 1
                continue
            if any(dist_m(s["lon"], s["lat"], x, y) <= EXTRA_TWIN_M for x, y in listed[rid]):
                n_twin += 1           # the route's own stop under another name
                continue
            nodes = stop_nodes(rid, s)
            if nodes:
                extra[rid][nid] = nodes
    n_st = len({n for v in extra.values() for n in v})
    log(f"OSM lines: {n_st} unlisted stations added as stops of {len(extra)} routes "
        f"({sum(len(v) for v in extra.values())} of {n_pairs} station-route pairs; "
        f"{n_nonet} left out as the route has no network, {n_full} as an Amtrak route, "
        f"{n_net} as another network's, {n_star + n_line} by NOT_ON, {n_twin} within "
        f"{EXTRA_TWIN_M} m of a stop the route lists)")
    return dict(extra)


# ---------------------------------------------------------------- broken route relations

# A route relation whose way members are in no order (2026-10-08, Anita's notes: the NJ Transit
# Morris & Essex Lines' strip diagram had Dover, Denville, Convent Station and Mount Arlington
# on side lanes and the Gladstone Branch's stops interleaved with the Morristown Line's). The
# old one-relation-both-ways NJ Transit routes ("Morristown Line: New York <=> Hackettstown")
# list their ways in no order: build_model.assemble joins them into 79 runs (the Gladstone
# Branch's into 231), place_stations reads the stops in run order, and each pair of stops
# consecutive across two runs became a section traced from one to the other over whatever track
# was shortest: Dover - Mount Tabor past Denville, Mount Arlington - Denville past Dover,
# Denville - Convent Station, East Orange - Hoboken. Such a route's runs are rebuilt here
# (`repair_route_runs`, through rules/us.py's `route_runs`): its own track as a graph, the
# stops as regions of it (each track node goes to the nearest stop along the track), stops
# joined where their regions touch, the shortest such joins that connect them all (a spanning
# tree), and that tree walked from one end of its longest path to the other as ONE run that
# goes out along each branch and back, so place_stations reads every pair of neighbouring stops
# from the same run. Track the relation lacks between two parts of it is left to build_model's
# tracing as before (a run ends there and the next starts at the stop nearest across the gap).
# Only routes whose own runs are this broken: REPAIR_MIN_RUNS runs or more holding stops.
REPAIR_MIN_RUNS = 4
REPAIR_ABSORB_M = 120   # track within this of a stop node belongs to that stop
REPAIR_SNAP_M = 400     # a stop further than ABSORB_M from the track takes its nearest node
REPAIR_GAP_M = 3000     # a loose end of the route's track this near another part is a gap
REPAIR_JOIN_M = 30      # ... and this near, track (two ways that do not quite share a node)
REPAIRED = {}           # route rel id -> (runs holding stops before, stops in the tree)


def repair_route_runs(rid, runs, members, ways, coords, station_nodes, stations):
    """New runs for route relation `rid` if its own are in pieces (see above), else None.

    runs: build_model.assemble's [(node ids, lon/lat)]; station_nodes: {station key: set(stop
    node id)}, the line's stops over all its routes (place_stations reads them all, so they are
    all placed here too where this route's track passes them)."""
    allnodes = {n for ns in station_nodes.values() for n in ns}
    if not allnodes:
        return None
    arr = np.fromiter(allnodes, dtype=np.int64)
    holding = sum(1 for ids, _xy in runs if np.isin(ids, arr).any())
    if holding < REPAIR_MIN_RUNS:
        return None

    # The route's own track as a graph.
    adj = defaultdict(dict)
    pos = {}
    for ty, ref, role in members:
        if ty != "w" or (role and not role.startswith(("forward", "backward"))):
            continue
        w = ways.get(ref)
        if w is None or len(w[1]) < 2:
            continue
        nodes = np.asarray(w[1], dtype=np.int64)
        p, ok = coords.many(nodes)
        prev = None
        for n, k, good in zip(nodes.tolist(), p.tolist(), ok.tolist()):
            if not good:
                prev = None
                continue
            pos[n] = (coords.x[k] / 1e7, coords.y[k] / 1e7)
            if prev is not None and prev != n:
                d = dist_m(*pos[prev], *pos[n])
                if d < adj[prev].get(n, INF):
                    adj[prev][n] = adj[n][prev] = d
            prev = n
    if not pos:
        return None
    gids = np.fromiter(pos.keys(), dtype=np.int64)
    gxy = np.asarray([pos[n] for n in gids.tolist()])

    # Each stop's sources: the track nodes within ABSORB_M of its stop nodes, at that distance.
    src = {}                                   # station key -> {node: metres}
    stop_at = {}                               # station key -> the stop node put in the run
    for st, ns in station_nodes.items():
        pts = [(n, pos.get(n) or coords.get(n)) for n in ns]
        pts = [(n, p) for n, p in pts if p is not None]
        if not pts and st in stations:
            pts = [(None, (stations[st]["lon"], stations[st]["lat"]))]
        got, best = {}, None
        for n, (lon, lat) in pts:
            dd = np.hypot((gxy[:, 0] - lon) * math.cos(math.radians(lat)) * 111320,
                          (gxy[:, 1] - lat) * 110570)
            k = int(np.argmin(dd))
            if best is None or dd[k] < best[0]:
                best = (float(dd[k]), int(gids[k]), n)
            for i in np.nonzero(dd <= REPAIR_ABSORB_M)[0].tolist():
                g = int(gids[i])
                got[g] = min(got.get(g, INF), float(dd[i]))
        if not got and best is not None and best[0] <= REPAIR_SNAP_M:
            got = {best[1]: best[0]}
        if got and best is not None and best[2] is not None:
            src[st] = got
            stop_at[st] = best[2]
    if len(src) < 2:
        return None

    # Parts of the route's track with no way between them (a way missing from the relation:
    # the Morristown Line's ends 450 m short of the Hoboken approach) are joined where their
    # loose ends come nearest another part holding a stop, by a link marked as a gap: a stop
    # pair joined over it is traced by build_model as any gap in a relation is.
    comp = {}
    for n0 in pos:
        if n0 in comp:
            continue
        comp[n0] = n0
        todo = [n0]
        while todo:
            u = todo.pop()
            for v in adj[u]:
                if v not in comp:
                    comp[v] = n0
                    todo.append(v)
    gap_links = set()
    holding_comps = {comp[g] for nodes in src.values() for g in nodes}
    cuf = UF()
    for c in holding_comps:
        cuf.find(c)
    gcomp = np.asarray([comp[n] for n in gids.tolist()])
    loose = [n for n in pos if len(adj[n]) <= 1 and comp[n] in holding_comps]
    links = []
    for n in loose:
        lon, lat = pos[n]
        dd = np.hypot((gxy[:, 0] - lon) * math.cos(math.radians(lat)) * 111320,
                      (gxy[:, 1] - lat) * 110570)
        other = (gcomp != comp[n]) & np.isin(gcomp, list(holding_comps))
        if not other.any():
            continue
        k = int(np.argmin(np.where(other, dd, INF)))
        if dd[k] <= REPAIR_GAP_M:
            links.append((float(dd[k]), n, int(gids[k])))
    for d, a, b in sorted(links):
        if cuf.find(comp[a]) == cuf.find(comp[b]):
            continue
        cuf.union(comp[a], comp[b])
        adj[a][b] = adj[b][a] = d
        if d > REPAIR_JOIN_M:                  # nearer, two ways mapped not quite meeting
            gap_links.add((a, b))
            gap_links.add((b, a))
        if os.environ.get("REPAIR_DEBUG"):
            print("   gap link", a, b, round(d), pos[a], pos[b])

    # Every track node to its nearest stop along the track (a node two stops both reach at
    # the same cost goes to either).
    lab, dist, pred = {}, {}, {}
    heap = []
    for st, nodes in src.items():
        for g, d in nodes.items():
            if d < dist.get(g, INF):
                dist[g], lab[g] = d, st
                pred.pop(g, None)
    heap = [(d, g) for g, d in dist.items()]
    heapq.heapify(heap)
    done = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v, w in adj[u].items():
            if d + w < dist.get(v, INF):
                dist[v], lab[v], pred[v] = d + w, lab[u], u
                heapq.heappush(heap, (d + w, v))

    # Stops whose regions touch, by the cheapest track between them.
    cand = {}
    for u, nb in adj.items():
        if u not in lab:
            continue
        for v, w in nb.items():
            if v not in lab or lab[u] == lab[v]:
                continue
            a, b = lab[u], lab[v]
            c = dist[u] + w + dist[v]
            key = (a, b) if a < b else (b, a)
            if key not in cand or c < cand[key][0]:
                cand[key] = (c, u, v) if a < b else (c, v, u)
    uf = UF()
    tree = defaultdict(dict)                   # station -> {neighbour: (km, nodes or None)}

    def chain(x):
        out = [x]
        while out[-1] in pred:
            out.append(pred[out[-1]])
        return out
    for key, (c, u, v) in sorted(cand.items(), key=lambda kv: kv[1][0]):
        a, b = key
        if uf.find(a) == uf.find(b):
            continue
        uf.union(a, b)
        nodes = chain(u)[::-1] + chain(v)      # a's source ... u, v ... b's source
        if any((x, y) in gap_links for x, y in zip(nodes[:-1], nodes[1:])):
            tree[a][b] = tree[b][a] = (c, None)
            continue
        tree[a][b] = (c, nodes)
        tree[b][a] = (c, nodes[::-1])
    # Parts of the route with no track between them: joined at their nearest stops, with no
    # nodes (build_model traces the gap as it did).
    sts = list(src)
    for st in sts:
        uf.find(st)
    while len({uf.find(s) for s in sts}) > 1:
        best = None
        for a in sts:
            for b in sts:
                if uf.find(a) == uf.find(b):
                    continue
                pa = stations[a] if a in stations else None
                pb = stations[b] if b in stations else None
                if not pa or not pb:
                    continue
                d = dist_m(pa["lon"], pa["lat"], pb["lon"], pb["lat"])
                if best is None or d < best[0]:
                    best = (d, a, b)
        if best is None:
            break
        _d, a, b = best
        uf.union(a, b)
        tree[a][b] = tree[b][a] = (best[0] * 1.5, None)

    # The walk: from one end of the tree's longest path to the other, out along each branch
    # and back, the longest path's own branch last so it is not walked back.
    def farthest(s):
        seen, todo, far = {s: 0.0}, [s], (0.0, s)
        prevs = {s: None}
        while todo:
            u = todo.pop()
            for v, (c, _n) in tree[u].items():
                if v not in seen:
                    seen[v] = seen[u] + c
                    prevs[v] = u
                    todo.append(v)
                    if seen[v] > far[0]:
                        far = (seen[v], v)
        return far[1], prevs
    start = next(iter(tree)) if tree else sts[0]
    s, _ = farthest(start)
    t, prevs = farthest(s)
    spine = set()
    x = t
    while x is not None:
        spine.add(x)
        x = prevs[x]
    out_runs, cur = [], []

    def put_stop(st, toward=None):
        n = stop_at[st]
        if toward is not None:
            # Across a gap build_model traces from this stop node: of the stop's nodes, the one
            # nearest the stop on the other side (30th Street's two, for Suburban Station).
            q = coords.get(stop_at[toward])
            got = [(dist_m(*q, *p), m) for m in station_nodes[st]
                   for p in [pos.get(m) or coords.get(m)] if p is not None] if q else []
            if got:
                n = min(got)[1]
        g = cur[-1] if cur else None
        p = pos.get(g) if g is not None else (pos.get(n) or coords.get(n))
        if p is None:
            p = (stations[st]["lon"], stations[st]["lat"])
        if not cur or cur[-1][0] != n:
            cur.append((n, p))

    def put_nodes(nodes):
        for g in nodes:
            if cur and cur[-1][0] == g:
                continue
            cur.append((g, pos[g]))

    def edge(a, b):
        nonlocal cur
        _c, nodes = tree[a][b]
        if nodes is None:                      # a gap: end this run, start the next at b
            if cur:
                out_runs.append(cur)
            cur = []
            put_stop(b, toward=a)
            return
        put_nodes(nodes)
        put_stop(b)

    stack = [(s, None, iter(sorted(tree[s], key=lambda v: (v in spine, v))))]
    put_stop(s)
    while stack:
        u, parent, it = stack[-1]
        v = next((v for v in it if v != parent), None)
        if v is None:
            stack.pop()
            if parent is not None and u not in spine:
                edge(u, parent)                # back to the branch point
            continue
        edge(u, v)
        stack.append((v, u, iter(sorted(tree[v], key=lambda x: (x in spine, x)))))
    if cur:
        out_runs.append(cur)
    if not out_runs:
        return None
    # A run of one stop between two gaps (Hoboken, past the missing way) still has to be read
    # by place_stations: the stop twice, at one point, so it is a run of two nodes.
    out_runs = [r if len(r) >= 2 else r * 2 for r in out_runs]
    # A run's first stop node, where it is no track node, sits where its track starts (as the
    # others sit where their track arrives), so no section starts with a hop off the track.
    for r in out_runs:
        if r[0][0] not in pos:
            r[0] = (r[0][0], r[1][1])
    REPAIRED[rid] = (holding, len(src))
    if os.environ.get("REPAIR_DEBUG"):
        for a in stop_at:
            print("   stop", stations[a]["name"], stop_at[a], stop_at[a] in pos,
                  sorted(station_nodes[a]))
        for a in tree:
            print("   tree", stations[a]["name"], "->",
                  [(stations[b]["name"], round(c / 1000, 2), n is not None)
                   for b, (c, n) in tree[a].items()])
    return [(np.asarray([n for n, _p in r], dtype=np.int64),
             np.asarray([p for _n, p in r], dtype=float)) for r in out_runs]


class OsmTrack:
    """OSM rail ways, for drawing a register section on the track its trains run on."""

    def __init__(self, ways, coords, log):
        import shapely
        from shapely import STRtree
        from shapely.geometry import LineString
        self.shapely = shapely
        self.nodes, self.ll, self.slow, self.wids, geo = [], [], [], [], []
        for wid, (tags, nodes) in ways.items():
            if tags.get("railway") not in OSM_TRACK:
                continue
            n, ll = way_lonlat(wid, ways, coords)
            if n is None:
                continue
            self.wids.append(wid)
            self.nodes.append(n)
            self.ll.append(ll)
            self.slow.append(tags.get("service") in SLOW_SERVICE)
            geo.append(LineString(merc(ll[:, 0], ll[:, 1])))
        self.tree = STRtree(geo)
        log(f"OSM track for tracing: {len(geo):,} rail ways")

    def trace(self, pts, km_hint):
        """Shortest OSM path from pts[0] to pts[-1] inside a corridor round the polyline pts,
        each edge costing more the further it is from it. Returns (lon/lat list, km) or None."""
        from shapely.geometry import LineString
        shp = self.shapely
        a = np.asarray(pts, dtype=np.float64)
        sc = merc_scale(float(a[:, 1].mean()))
        G = LineString(merc(a[:, 0], a[:, 1]))
        corr = G.buffer(CORRIDOR_M * sc, quad_segs=2)
        js = self.tree.query(corr, predicate="intersects")
        if len(js) == 0:
            return None
        adj = defaultdict(list)
        xy = {}
        fresh = [0]
        for j in js.tolist():
            nodes, ll = self.nodes[j], self.ll[j]
            m = merc(ll[:, 0], ll[:, 1])
            mids = shp.points((m[:-1] + m[1:]) / 2)
            d = shp.distance(mids, G) / sc
            lat = ll[:-1, 1]
            seg_m = np.hypot((ll[1:, 0] - ll[:-1, 0]) * np.cos(np.radians(lat)) * 111320,
                             (ll[1:, 1] - ll[:-1, 1]) * 110570)
            slow = 1.5 if self.slow[j] else 1.0
            for k in np.nonzero(d <= CORRIDOR_M)[0].tolist():
                u, v = int(nodes[k]), int(nodes[k + 1])
                if u == v:
                    continue
                xy[u], xy[v] = tuple(ll[k]), tuple(ll[k + 1])
                w = 1.0 + (float(d[k]) / TRACE_D0_M) ** 2
                n = int(seg_m[k] // TRACE_STEP_M)
                chain = [u]
                for i in range(1, n + 1):
                    fresh[0] -= 1
                    t = i / (n + 1)
                    xy[fresh[0]] = tuple(ll[k] + (ll[k + 1] - ll[k]) * t)
                    chain.append(fresh[0])
                chain.append(v)
                piece = seg_m[k] / (n + 1)
                for p, q in zip(chain[:-1], chain[1:]):
                    adj[p].append((q, piece * w * slow, piece))
                    adj[q].append((p, piece * w * slow, piece))
        if not xy:
            return None
        ids = np.fromiter(xy.keys(), dtype=np.int64)
        pos = np.asarray([xy[i] for i in ids.tolist()])

        def ends(lon, lat):
            dd = np.hypot((pos[:, 0] - lon) * math.cos(math.radians(lat)) * 111320,
                          (pos[:, 1] - lat) * 110570)
            r = min(float(dd.min()) + 40.0, SNAP_M)
            k = np.nonzero(dd <= r)[0]
            return {int(ids[i]): float(dd[i]) * 3.0 for i in k}    # charged for the snap
        src, dst = ends(*a[0]), ends(*a[-1])
        if not src or not dst:
            return None
        dist, prev, real = dict(src), {}, {v: 0.0 for v in src}
        heap = [(c, v) for v, c in src.items()]
        heapq.heapify(heap)
        best, best_u = INF, None
        seen = set()
        while heap:
            c, u = heapq.heappop(heap)
            if c >= best:
                break
            if u in seen:
                continue
            seen.add(u)
            if u in dst and c + dst[u] < best:
                best, best_u = c + dst[u], u
            for v, w, m in adj.get(u, ()):
                nc = c + w
                if nc < dist.get(v, INF):
                    dist[v] = nc
                    prev[v] = u
                    real[v] = real[u] + m
                    heapq.heappush(heap, (nc, v))
        if best_u is None:
            return None
        path = [best_u]
        while path[-1] in prev:
            path.append(prev[path[-1]])
        path.reverse()
        out = [xy[path[0]]] + [xy[n] for n in path[1:-1] if n > 0] + [xy[path[-1]]]
        return out, real[best_u] / 1000


SNAP_M = 150           # a trace starts and ends on OSM track at most this far from the point


class RouteCover:
    """How much of a polyline OSM passenger train routes run over."""

    def __init__(self, wgeo_raw):
        import shapely
        from shapely import STRtree
        from shapely.geometry import LineString
        self.shapely = shapely
        geo = [LineString(merc(a[:, 0], a[:, 1])) for a in wgeo_raw.values() if len(a) >= 2]
        self.tree = STRtree(geo)

    def share(self, pts):
        a = densify(np.asarray(pts, dtype=np.float64), 100.0)
        sc = merc_scale(float(a[:, 1].mean()))
        p = self.shapely.points(merc(a[:, 0], a[:, 1]))
        hit = self.tree.query(p, predicate="dwithin", distance=ROUTE_NEAR_M * sc)
        return len(set(hit[0].tolist())) / len(a)


def infra_names(path_proc, ways_near, log):
    """Names of OSM route=railway relations by way id (for lines NARN gives no name)."""
    import pickle
    f = path_proc / "infra.pkl"
    if not f.exists():
        return {}
    with open(f, "rb") as fh:
        infra = pickle.load(fh)
    out = defaultdict(list)
    for _rid, (tags, members) in infra.items():
        nm = tags.get("name")
        if not nm:
            continue
        for ty, ref, _role in members:
            if ty == "w":
                out[ref].append(nm)
    return out


def junction_name(node, lines_at, names, pos, stations_any, border=BORDERS):
    lon, lat = pos
    for blon, blat, bname in border:
        if dist_m(lon, lat, blon, blat) <= BORDER_M:
            return bname
    short = sorted({re.sub(r" (Subdivision|District|Line|Branch|Corridor)$", "", names[i])
                    for i in lines_at if names[i]})
    if len(short) >= 2:
        return " / ".join(short[:3])
    xy, nm = stations_any
    if len(nm):
        d = np.hypot((xy[:, 0] - lon) * math.cos(math.radians(lat)) * 111320,
                     (xy[:, 1] - lat) * 110570)
        j = int(np.argmin(d))
        if d[j] <= 5000:
            return f"near {nm[j]}"
    return f"end of {short[0]}" if short else "junction"


# ================================================================ build

def build(path, log):
    from n02 import walk_order
    segs, left, left_by = read_narn(path, log)
    groups, _left_unnamed = group_lines(segs, log)
    path_checks(segs, groups, log)

    bm, ways, rels, stops, coords = load_osm(log)
    st, resolved = bm.build_stations(stops, rels, coords, log, "us")
    rw = route_ways(rels)
    served = defaultdict(set)
    for rid in rw:
        for ref in bm.stop_members(rels[rid][1]):
            s = resolved.get(ref)
            if s is not None:
                served[s].add(rid)
    wraw = {}
    for w in {w for ws in rw.values() for w in ws}:
        if w in ways:
            _n, ll = way_lonlat(w, ways, coords)
            if ll is not None:
                wraw[w] = ll
    wgeo = {w: densify(a, STEP_DENSE) for w, a in wraw.items()}
    log(f"OSM: {len(rw)} passenger train routes over {len(wraw):,} ways; "
        f"{len(served)} stations they stop at")
    unlisted = unlisted_stations(st, stops, served, rw, wraw, log)
    for nid, rids in unlisted.items():
        served[nid] = rids
    not_on = resolve_not_on(st, log)

    idx = SegIndex(groups)
    offsets_report(idx, wgeo, log)
    # --- holes: track OSM's passenger routes run over that NARN codes no passenger service on
    # (fetch_holes). Each taken segment is a missing piece of a passenger line, never freight.
    hole_file = Path(path).with_name(HOLES_FILE)
    if hole_file.exists():
        n_stops = Counter()
        for nid, rids in served.items():
            for r in rids:
                n_stops[r] += 1
        hr = hole_routes(rels, rw, n_stops, log)
        far = hole_route_points(idx, {w: wgeo[w] for r in hr for w in rw[r] if w in wgeo}, log)
        holes, _hl, _hlb = read_narn(hole_file, log, holes=True)
        took = accept_holes(holes, idx, far, log)
        groups = groups + group_holes(took, groups, log)
        idx = SegIndex(groups)
    else:
        log(f"holes: no {HOLES_FILE} (python us_register.py --fetch-holes)")
    names = []
    for g in groups:
        n = display_name(g["key"], g["field"]) if g["key"] else ""
        if g.get("second"):
            n += " (second track)"
        names.append(n)
    # Junctions are named for the lines meeting there, but not by a register non-name ("3").
    jnames = [None if not_a_name(g) else n for g, n in zip(groups, names)]
    node_lines = defaultdict(set)
    for i, g in enumerate(groups):
        for s in g["segs"]:
            node_lines[s["a"]].add(i)
            node_lines[s["b"]].add(i)
    check_not_on_lines(groups)
    placed = place_stations(groups, idx, st, served, rw, wgeo, log, not_on)
    cover = RouteCover(wraw)
    track = OsmTrack(ways, coords, log) if TRACE else None
    named = [s for s in st.values() if s["name"] and not s.get("_tram")]
    stations_any = (np.asarray([[s["lon"], s["lat"]] for s in named]).reshape(-1, 2),
                    [s["name"] for s in named])
    inames = infra_names(ROOT / "data" / "proc" / "us", None, log)

    out_lines, out_st, geoms = [], {}, {}
    lid_of_group = {}
    SECTION_SEGS.clear()
    SECTION_CHAIN.clear()
    n_unrouted, km_unrouted, unrouted = 0, 0.0, []
    n_traced = n_fallback = 0
    fallback = []
    n_merged = 0
    twin_stubs = []
    for li, g in enumerate(groups):
        lg = LineGraph(g["segs"])
        # --- one station per place on this line (MERGE_M), the most-served kept
        pl = []
        for si, t, nid, nr in placed.get(li, ()):
            seg = g["segs"][si]
            p = split_line(seg["pts"], [t])[0][-1]
            pl.append((p, si, t, nid, nr))
        pl.sort(key=lambda x: (-x[4], x[3]))
        keep = []
        for x in pl:
            if any(dist_m(*x[0], *k[0]) <= MERGE_M for k in keep):
                n_merged += 1
                continue
            keep.append(x)
        # --- every track of the line through each station is cut there
        stop_at = {}
        for p, si, t, nid, _nr in keep:
            sid = f"n{nid}"
            for sj, seg in enumerate(g["segs"]):
                if sj != si:
                    a = np.asarray(seg["pts"])
                    if (p[0] < a[:, 0].min() - 0.002 or p[0] > a[:, 0].max() + 0.002
                            or p[1] < a[:, 1].min() - 0.002 or p[1] > a[:, 1].max() + 0.002):
                        continue
                    tj, dj = nearest_on(seg["pts"], p)
                    if dj > PARALLEL_M:
                        continue
                else:
                    tj = t
                node = place_node(lg, sj, tj)
                stop_at[node] = sid
        # NOT cut where another line ends partway along this one (tried 2026-10-02: on double
        # track the cut points moved sections onto other track, Austin +40 km, Coast +16,
        # and the LIRR Main Line and the NEC failed a length guard). The app reaches the stop
        # past a junction end through the operating patterns instead (pastEnds).
        secs, at = narn_sections(lg, stop_at)
        if not secs:
            continue
        lid = line_id(g["owner"], g["key"], g["segs"], g["split"], g.get("hole", False))
        name = names[li]
        sections, chain, sgeo = [], {}, {}
        for (sa, sb), v in sorted(secs.items()):
            pts = section_geometry(v)
            junc = sa.startswith("uj") or sb.startswith("uj")
            if not junc:
                share = cover.share(pts)
                if share < UNROUTED_SHARE:
                    n_unrouted += 1
                    km_unrouted += v["km"]
                    unrouted.append((v["km"], name, g["owner"], sa, sb, share))
                    continue
            km = v["km"]
            if track is not None:
                got = track.trace(pts, v["chain"])
                tol = TRACE_TOL[0] * v["chain"] + TRACE_TOL[1]
                if got is not None and abs(got[1] - v["chain"]) <= tol:
                    pts, km = got
                    n_traced += 1
                else:
                    n_fallback += 1
                    fallback.append((v["chain"], got[1] if got else None, name, sa, sb))
            sections.append([sa, sb, round(km, 3)])
            chain[f"{sa}|{sb}"] = round(v["chain"], 3)
            sgeo[f"{sa}|{sb}"] = [[round(x, 5), round(y, 5)] for x, y in pts]
            SECTION_SEGS[(lid, f"{sa}|{sb}")] = sorted({e[5] for e in v["edges"]})
            SECTION_CHAIN[(lid, f"{sa}|{sb}")] = round(v["chain"], 3)
        node_of = {sid: n for n, sid in at.items()}
        # A stop a few metres short of its line's dead end (South Station, 5 m before the end
        # of NARN's track; Rockport, Needham Heights, Downtown Carrollton): the junction there
        # is the stop itself, and its stub only gave the app a junction end to offer every
        # line near the station from (Anita, 2026-10-09: the Red Line past South Station).
        # Left out where no other line meets that node and it is no border.
        deg = Counter(s for sec in sections for s in sec[:2])
        for sec in [s for s in sections if s[2] <= TWIN_STUB_KM]:
            j = sec[0] if sec[0].startswith("uj") else sec[1] if sec[1].startswith("uj") else None
            stop = sec[1] if j == sec[0] else sec[0]
            if j is None or stop.startswith("uj") or deg[j] != 1:
                continue
            n = node_of[j]
            if node_lines.get(n, set()) - {li} or any(
                    dist_m(*lg.pos[n], bx, by) <= BORDER_M for bx, by, _bn in BORDERS):
                continue
            key = f"{sec[0]}|{sec[1]}"
            sections.remove(sec)
            chain.pop(key, None)
            sgeo.pop(key, None)
            SECTION_SEGS.pop((lid, key), None)
            SECTION_CHAIN.pop((lid, key), None)
            twin_stubs.append((sec[2], name, g["owner"], stop, j))
        if not sections:
            continue
        used = {s for sec in sections for s in sec[:2]}
        for sid in used:
            if sid not in out_st:
                if sid.startswith("uj"):
                    n = node_of[sid]
                    lon, lat = lg.pos[n]
                    out_st[sid] = {"id": sid, "lon": lon, "lat": lat, "lines": set(),
                                   "name": junction_name(n, node_lines.get(n, {li}), jnames,
                                                         (lon, lat), stations_any),
                                   "name_en": "", "junction": True}
                else:
                    s = st[int(sid[1:])]
                    out_st[sid] = {"id": sid, "name": s["name"], "name_en": s["name_en"],
                                   "lon": s["lon"], "lat": s["lat"], "lines": set()}
            out_st[sid]["lines"].add(lid)
        if not_a_name(g):
            name = osm_line_name(g, inames, track, out_st, sections) or name \
                or OWNERS.get(g["owner"], g["owner"])
        line = {
            "id": lid, "src": "narn", "service": False,
            "name": name, "name_en": "", "ref": "", "colour": "",
            "operator": OWNERS.get(g["owner"], g["owner"]), "operator_en": "",
            "network": "", "kind": "rail",
            "km": round(sum(s[2] for s in sections), 3),
            "km_official": round(sum(chain.values()), 3), "chain": chain,
            "variants": 1, "straight_sections": 0,
            "display": display_order(sections, walk_order),
            "sections": sections,
        }
        out_lines.append(line)
        geoms[lid] = sgeo
        line["_split"] = g["split"]
        line["_hole"] = g.get("hole", False)
        lid_of_group[li] = lid
        if g.get("second"):
            line["_main"] = g.get("main")
    # A second track running within COMPANION_KM of its main line (90% of it) is the same line
    # to a rider, whichever direction the train took: Anita, 2026-10-02, "both directions of a
    # line one track, unless they're super far apart". It is declared the main line's
    # companion, and ownership.py gives its track to the main line. Measured 2026-10-02: 27 of
    # 28 lie within 3 km (UP's paired track 0.5-1.5 km, Cajon 0.4-1 km); "Austin Subdivision
    # (second track)" lies 54-110 km from the Austin Subdivision and stays a line.
    for l in out_lines:
        mi = l.pop("_main", None)
        main = lid_of_group.get(mi) if mi is not None else None
        if main is None:
            continue
        p90 = track_apart_m(geoms[l["id"]], geoms[main])
        if p90 <= COMPANION_KM * 1000:
            l["companion_of"] = main
        log(f"    second track {'folded into its line' if p90 <= COMPANION_KM * 1000 else 'kept'}: "
            f"{l['name']} {l['km']:.1f} km, 90% within {p90:,.0f} m")
    fold_second_track_stubs(out_lines, out_st, geoms, log)
    # Two lines of one name and owner more than JOIN_KM apart: the same name, and an English
    # name that tells them apart by their end stops ("Northeast Corridor (Washington -
    # New Rochelle)"), which the app shows. check_model sums them under the one name. A line
    # still in pieces once build_model has dropped what no train runs over is split there
    # (split_pieces, a build_model hook) and named the same way.
    for l in out_lines:
        if l.pop("_split"):
            stops_ = [s for s in l["display"] if not out_st[s].get("junction")]
            if len(stops_) >= 2:
                l["name_en"] = (f"{l['name']} ({out_st[stops_[0]]['name']} – "
                                f"{out_st[stops_[-1]]['name']})")
    hole_lines = [l for l in out_lines if l.pop("_hole")]
    if hole_lines:
        log(f"US: {len(hole_lines)} lines from the holes file, {sum(l['km'] for l in hole_lines):,.1f}"
            f" km in sections (build_model may still drop junction-ended ones):")
        for l in sorted(hole_lines, key=lambda l: -l["km"]):
            log(f"    {l['km']:7.1f} km  {l['name']} [{l['operator']}]  "
                f"{len(l['sections'])} sections")
    log(f"US: {len(twin_stubs)} dead-end stubs of {TWIN_STUB_KM * 1000:.0f} m or less past a stop "
        f"left out (the stop ends the line)")
    for km, name, owner, stop, j in sorted(twin_stubs, reverse=True):
        log(f"    {km * 1000:5.0f} m  {name} [{owner}]  {out_name(st, stop, out_st)} - {j}")
    total = sum(l["km"] for l in out_lines)
    n_j = sum(1 for s in out_st.values() if s.get("junction"))
    log(f"US: {len(out_lines)} register lines, {total:,.0f} km, {len(out_st)} stations of "
        f"which {n_j} are junctions; {n_merged} station records merged into a neighbour on a "
        f"line (within {MERGE_M} m)")
    log(f"US: {n_unrouted} sections between two stops ({km_unrouted:,.0f} km) left out: no "
        f"OSM passenger route runs over {UNROUTED_SHARE:.0%} of them")
    for km, name, owner, sa, sb, share in sorted(unrouted, reverse=True)[:40]:
        log(f"    {km:7.1f} km  {name} [{owner}]  {out_name(st, sa)} - {out_name(st, sb)}"
            f"  ({share:.0%} on a route)")
    if track is not None:
        log(f"US: {n_traced} sections drawn on OSM track, {n_fallback} on NARN's own geometry "
            f"(no OSM path within {TRACE_TOL[0]:.0%} + {TRACE_TOL[1]} km of NARN's length)")
        for chain_km, got, name, sa, sb in sorted(fallback, key=lambda x: -x[0])[:30]:
            log(f"    {chain_km:7.1f} km NARN, traced {got if got is None else round(got, 1)}"
                f"  {name}  {out_name(st, sa, out_st)} - {out_name(st, sb, out_st)}")
    return out_lines, out_st, geoms


def fold_second_track_stubs(lines, out_st, geoms, log):
    """A line's stop-less stub from one of its branch points to a dead end where its own folded
    second track carries on is that second track's start, NARN having coded its first stretch
    as passenger and the rest not (the holes file brought the rest in as "<line> (second
    track)"). Moved into that companion, so ownership gives its ways to the line too. Left in
    the line it was a junction end leading nowhere on the line's own track, and the app offered
    whatever passed near it: the Lordsburg Subdivision's 13.1 km to the Pantano second main
    east of Tucson offered Tucson again, on the Sunset Limited (Anita, 2026-10-09). Only where
    the stub lies within COMPANION_KM of the line's other track (90% of it), as the companion
    itself must."""
    from n02 import walk_order
    byid = {l["id"]: l for l in lines}
    comps = defaultdict(list)
    for l in lines:
        if l.get("companion_of") in byid:
            comps[l["companion_of"]].append(l)
    moved = []
    for mid, cs in comps.items():
        m = byid[mid]
        while True:
            deg = Counter(s for sec in m["sections"] for s in sec[:2])
            hit = None
            for j, d in deg.items():
                if d != 1 or not out_st[j].get("junction"):
                    continue
                c = next((c for c in cs if any(j in sec[:2] for sec in c["sections"])), None)
                if c is None:
                    continue
                chain, u, prev = [], j, None
                while True:
                    sec = next(s for s in m["sections"] if u in s[:2] and s is not prev)
                    chain.append(sec)
                    u = sec[0] if sec[1] == u else sec[1]
                    prev = sec
                    if deg[u] != 2 or not out_st[u].get("junction"):
                        break
                # From a branch point of the line: not a stop, not the line's own other end.
                if not out_st[u].get("junction") or deg[u] < 3:
                    continue
                keys = {f"{s[0]}|{s[1]}" for s in chain}
                g = geoms[mid]
                rest = {k: v for k, v in g.items() if k not in keys}
                if track_apart_m({k: g[k] for k in keys}, rest) > COMPANION_KM * 1000:
                    continue
                hit = (c, chain, keys)
                break
            if hit is None:
                break
            c, chain, keys = hit
            for sec in chain:
                key = f"{sec[0]}|{sec[1]}"
                m["sections"].remove(sec)
                c["sections"].append(sec)
                geoms[c["id"]][key] = geoms[mid].pop(key)
                c["chain"][key] = m["chain"].pop(key)
                for d in (SECTION_SEGS, SECTION_CHAIN):
                    if (mid, key) in d:
                        d[(c["id"], key)] = d.pop((mid, key))
                for s in sec[:2]:
                    out_st[s]["lines"].add(c["id"])
            left = {s for sec in m["sections"] for s in sec[:2]}
            for sec in chain:
                for s in sec[:2]:
                    if s not in left:
                        out_st[s]["lines"].discard(mid)
            for l in (m, c):
                l["km"] = round(sum(s[2] for s in l["sections"]), 3)
                l["km_official"] = round(sum(l["chain"].values()), 3)
                l["display"] = display_order(l["sections"], walk_order)
            moved.append((sum(s[2] for s in chain), m["name"], m["operator"], c["id"],
                          [out_name(None, s, out_st) for s in chain[0][:2]]))
    log(f"US: {len(moved)} stubs of a line to where its own second track carries on moved into "
        f"that second track ({sum(x[0] for x in moved):.1f} km)")
    for km, name, op, cid, ends in sorted(moved, reverse=True):
        log(f"    {km:7.3f} km  {name} [{op}] -> {cid}  ({' - '.join(ends)})")


TRACE = True
PARALLEL_M = 80        # a station cuts every track of its line this close to it
TWIN_STUB_KM = 0.05    # a dead-end stub this short past a stop is the stop's own (no junction end)


def not_a_name(g):
    return not g["key"] or bool(NOT_A_LINE_NAME.match(g["key"]))


def display_order(sections, walk_order):
    """The strip diagram's order: n02.walk_order over the line's LONGEST connected piece.
    walk_order alone starts from the first terminus in sort order and lists only that piece,
    which could be a 5 km stub (La Junta's at Boise City), or a piece build_model then drops
    as unridden, leaving the line with no display at all (McComb)."""
    uf = UF()
    for a, b, _km in sections:
        uf.union(a, b)
    km = Counter()
    for a, _b, k in sections:
        km[uf.find(a)] += k
    big = max(km, key=km.get)
    return walk_order([(a, b) for a, b, _k in sections if uf.find(a) == big])


def nearest_on(pts, p):
    """(fraction along the polyline nearest point p, metres to it)."""
    a = np.asarray(pts, dtype=np.float64)
    cos = math.cos(math.radians(p[1]))
    x = (a[:, 0] - p[0]) * cos * 111320
    y = (a[:, 1] - p[1]) * 110570
    best = (INF, 0.0)
    seg = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    for i in range(len(a) - 1):
        dx, dy = x[i + 1] - x[i], y[i + 1] - y[i]
        L2 = dx * dx + dy * dy
        f = 0.0 if L2 == 0 else max(0.0, min(1.0, -(x[i] * dx + y[i] * dy) / L2))
        d = math.hypot(x[i] + dx * f, y[i] + dy * f)
        if d < best[0]:
            best = (d, (cum[i] + f * seg[i]) / (cum[-1] or 1e-9))
    return best[1], best[0]


def place_node(lg, si, t):
    """The graph node for a station at fraction t of segment si: the segment's own end node
    if within NODE_SNAP_M of it, else a node inserted there."""
    s = lg.segs[si]
    L = s["geo_km"] * 1000
    if t * L <= NODE_SNAP_M:
        return s["a"]
    if (1 - t) * L <= NODE_SNAP_M:
        return s["b"]
    return lg.insert(si, t)


def out_name(st, sid, out_st=None):
    if out_st and sid in out_st:
        return out_st[sid]["name"]
    if sid.startswith("n"):
        try:
            return st[int(sid[1:])]["name"]
        except (KeyError, ValueError):
            pass
    return sid


def osm_line_name(g, inames, track, out_st, sections):
    """A name for a line NARN does not name: the OSM route=railway relation its track lies on
    most, else '<owner>: <first stop> - <last stop>'."""
    if inames and track is not None:
        votes, n = Counter(), 0
        for s in g["segs"]:
            a = np.asarray(s["pts"])
            mid = a[len(a) // 2]
            sc = merc_scale(mid[1])
            p = track.shapely.points(merc([mid[0]], [mid[1]]))[0]
            n += s["km"]
            for nm in {nm for j in track.tree.query(p, predicate="dwithin",
                                                     distance=30 * sc).tolist()
                       for nm in inames.get(track.wids[j], ())}:
                votes[nm] += s["km"]
        if votes:
            # A tie (two relations over the same track: Melbourne's standard gauge is both the
            # Adelaide and the Sydney corridor) goes by name; most_common would follow the set
            # order above, which string hashing changes from run to run.
            nm, km = min(votes.items(), key=lambda kv: (-round(kv[1], 6), kv[0]))
            if km >= 0.5 * n:
                return nm
    stops_ = [x for sec in sections for x in sec[:2] if not out_st[x].get("junction")]
    if len(stops_) >= 2:
        from n02 import walk_order
        order = [x for x in walk_order([tuple(s[:2]) for s in sections])
                 if not out_st[x].get("junction")]
        if len(order) >= 2:
            return (f"{OWNERS.get(g['owner'], g['owner'])}: {out_st[order[0]]['name']} – "
                    f"{out_st[order[-1]]['name']}")
    return None


def offsets_report(idx, wgeo, log):
    """How far OSM passenger route track lies from NARN: the distance from each route point
    to the nearest NARN segment, for points within 200 m of one. Says whether NARN's geometry
    can be matched to OSM ways as it is."""
    allp = np.vstack(list(wgeo.values())) if wgeo else np.zeros((0, 2))
    if not len(allp):
        return
    rng = np.random.default_rng(0)
    sample = allp[rng.choice(len(allp), size=min(60000, len(allp)), replace=False)]
    ds = []
    for p in sample:
        pairs = idx.within(np.asarray([p]), 200)
        if pairs.shape[1] == 0:
            continue
        best = min(idx.project(int(j), p[0], p[1])[1] for j in pairs[1])
        ds.append(best)
    if not ds:
        return
    q = np.percentile(ds, [50, 75, 90, 95, 99])
    log(f"NARN vs OSM: of {len(sample):,} sampled route points, {len(ds):,} lie within 200 m "
        f"of NARN; distance to it: median {q[0]:.1f} m, 75% {q[1]:.1f}, 90% {q[2]:.1f}, "
        f"95% {q[3]:.1f}, 99% {q[4]:.1f}")


# ---------------------------------------------------------------- outside numbers, by path

# Shortest path over the kept NARN track between two stations, against a published length:
# the register's network as a whole, not one line. (lat, lon) pairs; Wikipedia figures.
PATH_CHECKS = [
    ("Washington - Boston", (38.8973, -77.0063), (42.3523, -71.0552), 735.0,
     "Northeast Corridor, 457 mi, Wikipedia infobox"),
    ("Philadelphia 30th St - Harrisburg", (39.9557, -75.1820), (40.2622, -76.8781), 168.3,
     "Keystone Corridor MP 104.6, Wikipedia"),
    ("Los Angeles - San Diego", (34.0562, -118.2365), (32.7165, -117.1696), 206.0,
     "Pacific Surfliner, MP 222 to 350, Wikipedia"),
    ("Chicago - St. Louis", (41.8789, -87.6403), (38.6236, -90.2045), 457.0,
     "Lincoln Service, 284 mi, Wikipedia infobox"),
    ("Porter - Dearborn", (41.6156, -87.0748), (42.3077, -83.2386), 373.0,
     "Michigan Line, 232 mi, Wikipedia"),
]


def path_checks(segs, groups, log):
    adj, pos = defaultdict(list), {}
    for s in segs:
        adj[s["a"]].append((s["b"], s["km"]))
        adj[s["b"]].append((s["a"], s["km"]))
        pos[s["a"]], pos[s["b"]] = s["pts"][0], s["pts"][-1]
    ids = list(pos)
    P = np.asarray([pos[i] for i in ids])

    def nearest(lat, lon):
        d = np.hypot((P[:, 0] - lon) * math.cos(math.radians(lat)), P[:, 1] - lat)
        return ids[int(np.argmin(d))]
    for label, a, b, km, note in PATH_CHECKS:
        src, dst = nearest(*a), nearest(*b)
        dist, h = {src: 0.0}, [(0.0, src)]
        while h:
            d, u = heapq.heappop(h)
            if u == dst or d > dist[u]:
                if u == dst:
                    break
                continue
            for v, w in adj[u]:
                if d + w < dist.get(v, INF):
                    dist[v] = d + w
                    heapq.heappush(h, (d + w, v))
        got = dist.get(dst)
        log(f"path check {label}: {'no path' if got is None else f'{got:.1f} km'} over NARN "
            f"against {km:.1f} ({note})"
            + ("" if got is None else f", ratio {got / km:.3f}"))


# ================================================================ --narn: the register alone

def narn_report(path):
    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

    segs, left, left_by = read_narn(path, log)
    groups, left_unnamed = group_lines(segs, log)
    log(f"NARN: {len(groups)} register lines")
    path_checks(segs, groups, log)
    rows = []
    n_junc_branch = 0
    for g in groups:
        lg = LineGraph(g["segs"])
        secs, at = narn_sections(lg, {})
        km = sum(v["km"] for v in secs.values())
        chain = sum(v["chain"] for v in secs.values())
        total = sum(s["km"] for s in g["segs"])
        ends = len(line_ends(lg.adj))
        n_junc_branch += len(at) - ends
        rows.append({"name": display_name(g["key"], g["field"]), "key": g["key"],
                     "owner": g["owner"], "km": km, "chain": chain, "total": total,
                     "secs": len(secs), "branch": len(at) - ends, "pieces": g["pieces"],
                     "split": g["split"],
                     "states": "".join(sorted({s["state"] for s in g["segs"]}))})
    tot = sum(r["total"] for r in rows)
    km = sum(r["km"] for r in rows)
    log(f"NARN: {len(rows)} lines, {tot:,.0f} km of track in them, {km:,.0f} km in sections "
        f"between line ends and branch points ({tot - km:,.0f} km of second tracks and loops "
        f"in no section); {n_junc_branch} branch points")
    return rows, left, left_by, left_unnamed


def main():
    if "--fetch-holes" in sys.argv:
        fetch_holes()
        return
    if "--holes-dry" in sys.argv:          # the envelopes fetch_holes would query, no fetch
        hole_envelopes(print)
        return
    if "--fetch" in sys.argv:
        fetch()
        return
    if "--narn" in sys.argv:
        path = RAW / "narn_passenger.geojson"
        rows, left, left_by, left_unnamed = narn_report(path)
        rows.sort(key=lambda r: -r["total"])
        print(f"\n{'km':>8} {'chain':>8} {'NARN':>8} sec br pc  owner  name")
        for r in rows[:40]:
            print(f"{r['km']:8.1f} {r['chain']:8.1f} {r['total']:8.1f} {r['secs']:3d} "
                  f"{r['branch']:2d} {r['pieces']:2d}  {r['owner']:<5} {r['name']}"
                  f"{'  [split]' if r['split'] else ''}  {r['states']}")
        print("\nlines with most track in no section (second tracks, loops):")
        for r in sorted(rows, key=lambda r: r["km"] - r["total"])[:15]:
            print(f"  {r['km']:8.1f} of {r['total']:8.1f}  {r['owner']:<5} {r['name']}")
        print("\nlines whose sections overcount the track (should be none):")
        for r in rows:
            if r["km"] > r["total"] + 0.05:
                print(f"  {r['km']:8.1f} of {r['total']:8.1f}  {r['owner']:<5} {r['name']}")
        print("\nsplit names (one name and owner, pieces more than JOIN_KM apart):")
        for r in sorted(rows, key=lambda r: (r["name"], r["owner"])):
            if r["split"]:
                print(f"  {r['total']:8.1f}  {r['owner']:<5} {r['name']}  {r['states']}")
        if "--all" in sys.argv:
            print("\nevery line:")
            for r in sorted(rows, key=lambda r: (r["owner"], r["name"])):
                print(f"  {r['total']:8.1f}  {r['owner']:<5} {r['name']}  [{r['key']}]")
        print("\nleft out:")
        for (why, owner, name), km in sorted(left_by.items(), key=lambda kv: -kv[1])[:40]:
            print(f"  {km:8.1f}  {why:<26} {owner:<5} {name}")
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
