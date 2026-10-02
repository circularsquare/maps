"""Which register sections passenger trains run over, read from a national timetable feed.

    (called from build_model.main around drop_unridden_sections; a region with no feed in
     data/raw/gtfs/<cc>/ builds exactly as before)

    python gtfs_served.py --fetch cz       # download FEEDS["cz"] into data/raw/gtfs/cz/

WHY.  A register lists track, not service. `build_model.drop_unridden_sections` keeps a section
that ends at a junction only where an OpenStreetMap passenger route runs over it, and in
countries where OSM's route relations are patchy that dropped real passenger track (Czechia:
238 into Havlíčkův Brod by Odb Kubešův Mlýn, 292 around Bludov), while lines closed to
passengers but still mapped as railway=rail stayed drawn as running (Hungary: 27, 37, 62).
A timetable feed (GTFS) says directly where trains call; this reads that.

HOW.
1. FEED.  Every zip in data/raw/gtfs/<cc>/ (FEEDS says where each came from). Rail trips only:
   route_type 2 and 100-117 (and 400-405, urban rail); buses and rail-replacement buses (3,
   200, 700-799 incl. 714) never count, nor the agencies in SKIP_AGENCY (Czechia's
   "nabídková trasa", offered paths that are not trains). Each trip becomes its sequence of
   stations (parent_station where the feed has one), and identical sequences are counted
   together as one pattern with its trips and trip-days. Cached in patterns.pkl beside the
   zips, keyed on the zips' sizes and times.
2. MATCHING feed stations to the build's register section ends (`match_stations`): by code
   where the stop_id carries RINF's own number (Czechia's `CZ:54943` is RINF `CZ54943`;
   CODE says how), through RINF's point to the build station of a matching name within
   1.2 km; else by normalised name within 2 km (exact, then without generic words like
   "zastávka", then abbreviations: "Praha hl.n." = "Praha hlavní nádraží", then one name
   inside the other: "Kelenföld" in "Budapest-Kelenföld"); else the nearest stop within 300 m.
   Cyrillic and Greek names are folded to Latin as rinf.norm folds them.
3. PATHS (`call_pairs`).  For every pair of consecutive matched calls, the shortest path over
   all register sections between the two, no longer than DETOUR times the crow-fly length
   along the calls plus SLACK_KM. Every section on it is served, and so is a parallel section
   between the same two ends no longer than the chosen one by PARALLEL. This is what reaches
   junction-ended sections and trains that run through stops without calling.
   - Paths weigh a section no OSM passenger route runs over OSM_TIE (5%) more, so of two
     paths about as long the one OSM's routes take wins; and such a section is not credited
     as a parallel beside one that has a route. Portugal's freight loops beside the Linha do
     Norte (Plataforma de Cacia, Bobadela, Ramal TER-TIR) were credited instead of it.
   - LINKS (`dangling_links`): a register line end at a junction (not a border point) that
     no other section touches is joined to the nearest node of another line within LINK_KM,
     unless the register already reaches that node from it within LINK_REACH_KM + 3x the
     gap (a siding's end beside a station). A link is no section and credits nothing, and
     weighs 2x its length + LINK_EXTRA_KM so it never beats real track. Portugal's Variante
     de Alcácer (line 68) ends 0.96 km from Pinheiro, so the Lisboa - Algarve path ran over
     the old Linha do Sul, Pinheiro - Grândola Norte; with the link it takes the Variante.
   - An unmatched call is stepped over (a halt the build has inside a section).
   - A matched call with no path from the previous one is stepped over too (Žatec západ is a
     RINF point of another line, not of the 160 the trains run on).
   - A call abroad (`outline`: outside this country's outline AND inside another country's,
     with a RINF border point, type 90, on the way to it) cuts the run at that border point,
     so the section up to the border is on a path. Outside this country's outline alone is
     not enough: Portugal's estuary and coast stations (Alhandra, Alverca, Bobadela, Cascais,
     Figueira da Foz) fall outside it. At the start or end of a run the outline is not
     needed: Sebnitz is 400 m beyond the Czech border.
   - Direct evidence: a section whose two ends some train calls at one after the other is
     served, whatever the path search did (Austria's Jauntalbahn, St. Michael ob Bleiburg -
     Bleiburg, after a wrong match one call earlier).
4. DECIDING, per register section (`check`), in this order:
   - weak: a junction-ended section with no OSM route, on the path only of runs longer than
     LONG_RUN_KM or winding beyond STRAIGHT x their crow-fly length + 1 km, and starting or
     ending at neither of its ends. Left as it is (so dropped, as before): Gliwice - Chałupki
     non-stop ran over Zabrze Makoszowy colliery track, and Amsterdam's register gap sent
     Bijlmer - Centraal through the Watergraafsmeer depot.
   - served: a junction-ended one is kept whatever OSM routes say (`route_share` is set to
     1.0 for it, which is all drop_unridden_sections reads), and logged as "rescued" where
     OSM alone would have dropped it.
   - ambiguous: not on any shortest path, but on a path no more than NEAR times the shortest
     between some consecutive calls (a classic line beside a new one with no stop between,
     a triangle). Left as it is.
   - seasonal: no train now, but at least SEASONAL_MIN_DAYS trip-days in a past snapshot of
     the feed (PAST, data/raw/gtfs/<cc>/past/, `--fetch-past`), and no replacement bus at a
     stop no train calls at (that would be a works closure). Drawn as running: Anita counts
     seasonal lines. Poland's 15 July snapshot has Muszyna - Leluchów's summer trains.
   - border: ends at a border point the feed calls nowhere beyond (within BORDER_SEEN_KM).
     MÁV's feed ends Budapest - Wien at Hegyeshalom, so it cannot say. Left as it is. For a
     feed with no international trains at all (NO_INTERNATIONAL: Croatia's HŽPP), also every
     section reached from such a border point through stops the feed does not know
     (Koprivnica - Peteranec - Novo Drnje - Botovo carries only the trains to Hungary).
   - unknown: no train, and either OSM routes run over it or its line's infrastructure
     manager is neither the country's main one nor named by any agency in the feed; and some
     stop at its ends has nothing in the feed calling at it. That is how an operator missing
     from the feed looks (JHMD's 228 and 229 in Czechia: OSM has no route there either, the
     manager test is what catches them). Left as it is, logged, so a feed can be added.
   - not running: every other stop-to-stop section with no train (and an "unknown" one on a
     line closed beside it, reached through stops no train calls at: replacement buses skip
     halts, Larissa - Volos), and a junction-ended one
     that is part of a closed line (shares an end, in a chain, with a not-running
     stop-to-stop section of the same line). Marked `closed` after not_running.mark (`mark`),
     so the app greys it and leaves it out of completion.
   - osm: a junction-ended section with no train that is not part of a closed line, where OSM
     routes run over it. Left as it is: with no shapes, the feed cannot say which of two
     tracks into Praha Masarykovo nádraží a train takes. Where no OSM route runs over it
     either ("dropped"), drop_unridden_sections drops it as it always did, which is right
     for freight curves.
   Works closures look the same as abandonment in a feed (Hungary's 150, the Budapest -
   Belgrade upgrade): marked not running too, which is right for as long as no train runs.

RESULTS (2026-10-01; gtfs_sources.md "Rollout" has the lists). Kept = junction-ended km OSM
routes alone would drop; closed = km marked not running.
   cz  kept 55 km (238 into Havlíčkův Brod, 292 Bludov, Kamenický Šenov...); closed 74 km on
       7 lines (317, 256, 013, 318, 245, Heřmanův Městec - Prachovice, 244)
   hu  kept 1 km; closed 726 km on 23 lines (150 works closure, 121, 84, 42, 103, 78, 152...)
   pt  kept 3.5 km (the Variante de Alcácer now carries Lisboa - Algarve); closed 8.5 km
       (Linha de Leixões, Alcântara-Mar - Alcântara-Terra)
   pl  kept 110 km (39 Suwałki - Olecko); closed 1,007 km on 30 lines: long-closed (12, 14,
       34, 144, 204, 218, 410), freight (13, 131, 171, 179, 349, 394...), works (211, 287,
       274, 90, the Bieszczady 107/108), seasonal (96 Muszyna - Leluchów, 363), unsure (281)
   be  kept 2.5 km; closed L.147 Auvelais - Fleurus (9.2 km)
   sk  kept 12.8 km; closed 398 km on 20 lines (17 already in sk.py's SUSPENDED, plus 141,
       133 Sereď - Leopoldov, Hronec - Chvatimech)
   ro  kept 47 km; closed 793 km on 20 lines (300 Cluj - Oradea works, 600, 700, 603...)
   bg  kept 19 km (8 Trakia - Plovdiv Razpredelitelna); closed 83 km (83, 61, a Sofia stub)
   fi  kept 3.5 km (Tornio - Haparanda); closed the Porvoon rata (34 km, museum trains only)
   hr  kept 22.6 km (M304 Ploče - Metković); closed 139 km (M606, L103, R104)
   gr  closed 509 km on 5 lines (25 from Serres to Alexandroupoli, 12 Palaiofarsalos -
       Kalambaka, 13 Larissa - Volos, 10 Lianokladi - Stylida, Thessaloniki - Idomeni)
   si, lt, lv, ee, lu: no change. at, nl: switched off (no feed folder).
   Seasonal (PAST, pl's 15 July snapshot): pl 96 Muszyna - Leluchów and 131 Kraski -
   Zduńska Wola Karsznice/Babiak drawn as running (pl closed 1,008 -> 910 km).

WHAT IT CANNOT SEE.  A feed that covers only part of the year misses seasonal trains (Polish
regional trains are a 5-week window, PKP Intercity from 22 September: Muszyna - Leluchów's
summer trains to Poprad read as closed) unless a past snapshot (PAST) covers the season. Works closures look like abandonment (Opole - Nysa,
Cluj - Oradea). An operator missing from the feed looks like closure where neither OSM nor the
manager's name gives it away (museum trains: Porvoo). Two equally short register paths
between consecutive calls with no shapes to choose between them are "ambiguous". A register
in many unconnected pieces leaves most runs with no path at all and the result is not to be
trusted: Austria's (93 pieces) is why at has no feed; the Netherlands' gave nothing but one
depot track, so nl has none either. The feed in data/raw/gtfs/<cc>/ is a snapshot: refetch
and rebuild to pick up a reopened line.
"""
import argparse
import csv
import heapq
import io
import math
import os
import pickle
import re
import sys
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rinf import CYRILLIC_FOLD, GREEK_FOLD  # noqa: E402  (the same folds rinf.norm uses)
RAW = ROOT / "data" / "raw" / "gtfs"
USER_AGENT = "noritetsu-rail-map/1.0"
SHAPES = ROOT.parent / "religiondots" / "data" / "processed" / "country_shapes.geojson"
NE_SHAPES = ROOT.parent / "religiondots" / "data" / "geo" / "ne_10m_admin_0_countries.geojson"
CACHE_VERSION = 2

# Where each country's feeds come from (gtfs_sources.md has the survey). The Transitous mirror
# holds keyed sources (MÁV, Mobilitätsverbünde, nap.si) openly, refreshed about daily.
# Each entry: (file name, url, slim). `slim` rewrites a multimodal feed down to its rail
# routes after downloading (`slim_feed`), so the kept zip is small.
TX = "https://api.transitous.org/gtfs/"
UDATA = "udata:"    # a udata dataset API url (data.public.lu): fetch its newest zip resource
FEEDS = {
    "cz": [("cz_CZPTT.gtfs.zip", TX + "cz_CZPTT.gtfs.zip", False)],
    "hu": [("hu_mav.gtfs.zip", TX + "hu_mav.gtfs.zip", False)],
    "pl": [("pl_Polish-Trains.gtfs.zip", TX + "pl_Polish-Trains.gtfs.zip", False),
           ("pl_PKP-Intercity.gtfs.zip", TX + "pl_PKP-Intercity.gtfs.zip", True),
           ("pl_Kolejki-Waskotorowe.gtfs.zip", TX + "pl_Kolejki-W%C4%85skotorowe.gtfs.zip",
            False)],
    "be": [("be_sncb.gtfs.zip", TX + "be_sncb.gtfs.zip", True)],
    "nl": [("nl_ovapi_rail.gtfs.zip", "https://gtfs.ovapi.nl/nl/gtfs-nl.zip", True)],
    "at": [("at_Railway-Current-Reference-Data-2026.gtfs.zip",
            TX + "at_Railway-Current-Reference-Data-2026.gtfs.zip", False)],
    "pt": [("pt_CP.gtfs.zip", TX + "pt_Comboios-de-Portugal%28CP%29.gtfs.zip", False),
           ("pt_Fertagus.gtfs.zip", TX + "pt_Fergatus.gtfs.zip", False)],
    "si": [("si_nap_rail.gtfs.zip", TX + "si_nap.gtfs.zip", True)],
    "sk": [("sk_zsr.gtfs.zip",
            "https://data.slovensko.sk/download?id=c5a63281-3c44-4dba-82c9-7b7ad603db5d", False)],
    "ro": [("ro_railway.gtfs.zip", "https://jbb.ghsq.de/gtfs/ro-railway.gtfs.zip", True)],
    "bg": [("bg_bdz.gtfs.zip", "https://gtfs.livetransport.eu/gtfs/bdz.zip", True)],
    "fi": [("fi_digitraffic.gtfs.zip",
            "https://rata.digitraffic.fi/api/v1/trains/gtfs-passenger-stops.zip", True)],
    "lt": [("lt_ltglink.gtfs.zip", "https://jbb.ghsq.de/gtfs/lt-ltglink.gtfs.zip", True)],
    "lv": [("lv_pieturas_rail.gtfs.zip", TX + "lv_pieturas.gtfs.zip", True)],
    "ee": [("ee_elron.gtfs.zip", "https://eu-gtfs.remix.com/elron.zip", True)],
    # Luxembourg's national feed (CC BY 4.0, weekly): a new file name every week, so the url
    # is the dataset and `fetch` takes its newest zip (UDATA).
    "lu": [("lu_national.gtfs.zip", UDATA + "https://data.public.lu/api/1/datasets/"
            "horaires-et-arrets-des-transport-publics-gtfs/", True)],
    "hr": [("hr_hzpp.gtfs.zip", "https://www.hzpp.hr/GTFS_files.zip", False)],
    # not slimmed: Hellenic Train's replacement buses are route_type 3, and they are what
    # tells a line closed (Larissa - Volos, Palaiofarsalos - Kalambaka) from an operator
    # missing from the feed. The file also holds four ferry companies, which never count.
    "gr": [("gr_hellenic-train.gtfs.zip", "https://jbb.ghsq.de/gtfs/gr-hellenic-train.gtfs.zip",
            False)],
}
# Past snapshots of a short-window feed, kept in data/raw/gtfs/<cc>/past/ (`--fetch-past`):
# trains in them mark a section the current feed has none on as seasonal, drawn as running.
# Poland's regional feed covers about five weeks, so summer-only trains (Muszyna - Leluchów -
# Plaveč) read as closed in autumn. Mobility Database keeps every daily download of a feed
# at files.mobilitydatabase.org/<id>/<id>-YYYYMMDDHHMM/...zip (found by date: the listing
# needs an account, the files do not); the 15 July 2026 one of Polish Trains (mdb-3191) has
# regional and PKP IC trains from 13 July to 14 August 2026.
# A past snapshot counts for a section only with this many trip-days on it: "scheduled more
# often than about once a week" (Anita), one train each way weekly over an 8-week summer.
# Leaves out one-off specials and diversions (Warszawa - Łuków over line 12, one day).
SEASONAL_MIN_DAYS = 16
MDB = "https://files.mobilitydatabase.org/"
PAST = {
    "pl": [("pl_Polish-Trains_2026-07-15.gtfs.zip",
            MDB + "mdb-3191/mdb-3191-202607150054/mdb-3191-202607150054.zip", True)],
}
# stop_id -> RINF uopid, where the feed carries RINF's own station number.
CODE = {
    "cz": (re.compile(r"(?:^|:)CZ:(\d{5})(?::|$)"), lambda m: f"CZ{m.group(1)}"),
    # CP's 94_3046 is RINF's PT03046 (Palmilheira)
    "pt": (re.compile(r"^94_(\d{1,5})$"), lambda m: f"PT{int(m.group(1)):05d}"),
    # the Romanian feed's 60309 is RINF's RO60309 (Tecuci)
    "ro": (re.compile(r"^(\d{5})$"), lambda m: f"RO{m.group(1)}"),
}
# Feed station names the register spells otherwise: another name tried as well.
NAME_ALIAS = {
    # Hellenic Train's feed mixes Latin and old Greek forms
    "gr": {"Ska": "Σιδηροδρομικό Κέντρο Αχαρνών", "Paleopharsalos": "Παλαιοφάρσαλος",
           "Aegion": "Αίγιο", "Σέρραι": "Σέρρες", "Πεδινός": "Πεδινό",
           "Πετρίτσιον": "Νέο Πετρίτσι", "Rodopolis – Λιβαδειά": "Λιβάδια Κερκίνης"},
}
# Feeds that leave out international trains, so the track between the last stop they serve
# and the border is "border", undecided (see `check`). HŽPP's feed has no SŽ, MÁV or ŽRS trains.
NO_INTERNATIONAL = {"hr"}
# Agencies whose "trips" are not trains.
SKIP_AGENCY = {
    "cz": {"nabídková trasa"},
}

RAIL_TYPES = {2, *range(100, 118), *range(400, 406)}
DETOUR, SLACK_KM = 1.6, 4.0     # a path between two calls may be this much longer than crow-fly
PARALLEL = 1.1                  # parallel sections between the same ends, this close in length
NEAR = 1.1                      # "ambiguous": on a path this close to the shortest
NAME_KM = 2.0                   # a station of a matching name this close
CODE_KM = 1.2                   # a RINF point's build station this close (rinf.NAME_M is 1 km)
BLIND_KM = 0.3                  # any stop this close
BORDER_DETOUR, BORDER_SLACK_KM = 1.15, 3.0   # a border point "on the way" to a call abroad
BORDER_SEEN_KM = 30.0           # the feed knows a border when it calls abroad this close to it
LINK_KM = 1.0                   # a register line end meeting nothing is joined to a node this close
LINK_EXTRA_KM = 0.5             # a link weighs 2x its length plus this in paths
LINK_REACH_KM = 5.0            # ... unless the register already joins them within this + 3x
LONG_RUN_KM = 40.0              # a no-OSM junction section on longer non-stop runs only: not rescued
STRAIGHT = 1.3                  # ... nor on runs whose path is longer than this x crow-fly + 1 km
OSM_TIE = 0.05                 # paths weigh a section with no OSM passenger route this much more
NEEDS_ROUTE_SHARE = 0.5         # as build_model's: an OSM route runs over it
PASSENGER_TYPES = {"10", "20", "30", "70"}
BORDER_TYPES = {"90"}


def km_between(lon1, lat1, lon2, lat2):
    r = math.pi / 180
    dx = (lon2 - lon1) * r * math.cos((lat1 + lat2) / 2 * r)
    return 6371.0 * math.hypot(dx, (lat2 - lat1) * r)


# ================================================================ the feed

def feed_zips(cc):
    d = RAW / cc
    return sorted(d.glob("*.zip")) if d.is_dir() else []


def _days(z, names):
    """Service days per service_id, from calendar.txt and calendar_dates.txt."""
    days = defaultdict(set)
    lo, hi = None, None
    if "calendar.txt" in names:
        for r in _table(z, "calendar.txt"):
            a = date(int(r["start_date"][:4]), int(r["start_date"][4:6]), int(r["start_date"][6:]))
            b = date(int(r["end_date"][:4]), int(r["end_date"][4:6]), int(r["end_date"][6:]))
            wd = [r[k] == "1" for k in ("monday", "tuesday", "wednesday", "thursday", "friday",
                                         "saturday", "sunday")]
            d = a
            while d <= b:
                if wd[d.weekday()]:
                    days[r["service_id"]].add(d.toordinal())
                d += timedelta(days=1)
            lo = a if lo is None or a < lo else lo
            hi = b if hi is None or b > hi else hi
    if "calendar_dates.txt" in names:
        for r in _table(z, "calendar_dates.txt"):
            s = r["date"]
            d = date(int(s[:4]), int(s[4:6]), int(s[6:]))
            if r["exception_type"] == "1":
                days[r["service_id"]].add(d.toordinal())
            else:
                days[r["service_id"]].discard(d.toordinal())
            lo = d if lo is None or d < lo else lo
            hi = d if hi is None or d > hi else hi
    return {k: len(v) for k, v in days.items()}, lo, hi


def _table(z, name):
    with z.open(name) as f:
        yield from csv.DictReader(io.TextIOWrapper(f, "utf-8-sig"))


def read_feed(path, prefix, skip_agency, log):
    """One zip -> (stations, patterns, info). Station ids are prefixed so several feeds of one
    country never collide."""
    z = zipfile.ZipFile(path)
    names = set(z.namelist())
    agency = {a.get("agency_id", ""): a["agency_name"] for a in _table(z, "agency.txt")}
    skip = {aid for aid, n in agency.items() if n.strip().casefold() in
            {s.casefold() for s in skip_agency}}
    rail_routes, kept_agencies = set(), set()
    for r in _table(z, "routes.txt"):
        try:
            t = int(r["route_type"])
        except ValueError:
            continue
        aid = r.get("agency_id", "") or next(iter(agency), "")
        if t in RAIL_TYPES and aid not in skip:
            rail_routes.add(r["route_id"])
            kept_agencies.add(agency.get(aid, aid))
    days, lo, hi = _days(z, names)
    trip_days = {}
    for t in _table(z, "trips.txt"):
        if t["route_id"] in rail_routes:
            trip_days[t["trip_id"]] = days.get(t["service_id"], 0)

    stops, parent = {}, {}
    for s in _table(z, "stops.txt"):
        stops[s["stop_id"]] = s
        if s.get("parent_station"):
            parent[s["stop_id"]] = s["parent_station"]

    def station(sid):
        return parent.get(sid, sid)

    seqs = defaultdict(list)
    any_call = set()            # stations anything calls at, replacement buses included
    for st in _table(z, "stop_times.txt"):
        tid = st["trip_id"]
        s = station(st["stop_id"])
        any_call.add(prefix + s)
        if tid in trip_days:
            seqs[tid].append((int(st["stop_sequence"]), s))
    pats = defaultdict(lambda: [0, 0])
    for tid, l in seqs.items():
        if trip_days[tid] == 0:
            continue
        l.sort()
        p = []
        for _, s in l:
            if not p or p[-1] != s:
                p.append(s)
        if len(p) < 2:
            continue
        rec = pats[tuple(prefix + s for s in p)]
        rec[0] += 1
        rec[1] += trip_days[tid]
    stations = {}
    for sid, s in stops.items():
        if sid in parent or not s.get("stop_lat"):
            continue
        try:
            stations[prefix + sid] = (s["stop_name"], float(s["stop_lon"]), float(s["stop_lat"]),
                                      sid)
        except ValueError:
            continue
    # a stop with no parent record of its own: use the stop itself
    for sid, pid in parent.items():
        if prefix + pid not in stations and stops[sid].get("stop_lat"):
            s = stops[sid]
            stations[prefix + pid] = (s["stop_name"], float(s["stop_lon"]), float(s["stop_lat"]),
                                      pid)
    info = {"file": path.name, "trips": len(seqs), "patterns": len(pats), "any_call": any_call,
            "from": lo.isoformat() if lo else "", "to": hi.isoformat() if hi else "",
            "agencies": sorted(kept_agencies),
            "skipped": sorted(agency[a] for a in skip)}
    return stations, [(p, n, d) for p, (n, d) in pats.items()], info


def load_feeds(cc, log):
    return _load_dir(RAW / cc, cc, log)


def past_dir(cc):
    return RAW / cc / "past"


def load_past(cc, log):
    """The country's past snapshots (PAST), or None: trains seen in them mark a section the
    current feed has no train on as seasonal, not closed."""
    return _load_dir(past_dir(cc), cc, log)


WRITE_CACHE = True      # inspection scripts set False: read the data, never write beside it


def _load_dir(d, cc, log):
    zips = sorted(d.glob("*.zip")) if d.is_dir() else []
    if not zips:
        return None
    key = (CACHE_VERSION, [(p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in zips],
           sorted(SKIP_AGENCY.get(cc, ())))
    cache = d / "patterns.pkl"
    if cache.exists():
        try:
            with open(cache, "rb") as f:
                got = pickle.load(f)
            if got[0] == key:
                return got[1]
        except Exception:                                       # noqa: BLE001
            pass
    stations, patterns, infos = {}, [], []
    for i, p in enumerate(zips):
        st, pa, info = read_feed(p, f"{i}/", SKIP_AGENCY.get(cc, ()), log)
        stations.update(st)
        patterns += pa
        infos.append(info)
    out = {"stations": stations, "patterns": patterns, "info": infos}
    if WRITE_CACHE:
        with open(cache, "wb") as f:
            pickle.dump((key, out), f)
    return out


# ================================================================ matching

GENERIC = {"zastavka", "zast", "mh", "megallohely", "megallo", "allomas", "station", "stn",
           "bahnhof", "bf", "hp", "haltestelle", "gare", "halte", "estacao", "apeadeiro",
           "stacja", "przystanek", "postajalisce", "postaja", "zel", "vlakova"}


# words in company names that say nothing about which company
ORG_WORDS = {"a", "s", "as", "r", "o", "sro", "se", "ag", "gmbh", "co", "kg", "mbh", "sa", "z",
             "sp", "spolek", "ops", "bv", "rail", "railway", "railways", "drahy", "draha",
             "vlaky", "zrt", "kft", "nv", "spa", "srl", "ltd", "the", "und", "of"}


def tokens(s):
    """Lower-case Latin words of a name. Cyrillic and Greek are folded to Latin as rinf.norm
    folds them (Bulgarian and Greek feeds and OSM names), which before left them empty."""
    s = (s or "").casefold().replace("ß", "ss").translate(CYRILLIC_FOLD)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("ου", "ou").replace("αυ", "av").replace("ευ", "ev").translate(GREEK_FOLD)
    return re.sub(r"[^a-z0-9]+", " ", s).split()


def _abbr_eq(a, b):
    return a == b or (a.startswith(b) or b.startswith(a))


def name_tier(fa, fb):
    """0 identical, 1 identical without generic words, 2 abbreviations of each other,
    3 one inside the other; None for no match."""
    a, b = tokens(fa), tokens(fb)
    if not a or not b:
        return None
    if a == b:
        return 0
    a2 = [t for t in a if t not in GENERIC] or a
    b2 = [t for t in b if t not in GENERIC] or b
    if a2 == b2:
        return 1
    if len(a2) == len(b2) and all(_abbr_eq(x, y) for x, y in zip(a2, b2)) and \
            any(x == y and len(x) >= 3 for x, y in zip(a2, b2)):
        return 2
    short, long_ = (a2, b2) if len(a2) <= len(b2) else (b2, a2)
    if sum(len(t) for t in short) >= 4:
        for i in range(len(long_) - len(short) + 1):
            if all(_abbr_eq(x, y) for x, y in zip(short, long_[i:i + len(short)])) and \
                    any(x == y and len(x) >= 3 for x, y in zip(short, long_[i:i + len(short)])):
                return 3
    return None


class Grid:
    def __init__(self, items, cell=0.05):
        self.cell = cell
        self.g = defaultdict(list)
        self.pos = {}
        for k, lon, lat in items:
            self.pos[k] = (lon, lat)
            self.g[(int(lon // cell), int(lat // cell))].append(k)

    def near(self, lon, lat, r_km):
        n = int(math.ceil(r_km / (self.cell * 60))) + 1
        gx, gy = int(lon // self.cell), int(lat // self.cell)
        out = []
        for dx in range(-n, n + 1):
            for dy in range(-n, n + 1):
                for k in self.g.get((gx + dx, gy + dy), ()):
                    x, y = self.pos[k]
                    d = km_between(lon, lat, x, y)
                    if d <= r_km:
                        out.append((d, k))
        out.sort()
        return out


def rinf_points(cc):
    p = ROOT / "data" / "raw" / "rinf" / cc / "points.json"
    if not p.exists():
        return {}
    import json
    with open(p, encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    out = {}
    for r in rows:
        u = r.get("uopid")
        if not u:
            continue
        try:
            if r.get("lon") and r.get("lat"):
                lon, lat = float(r["lon"]), float(r["lat"])
            else:
                # Portugal's points carry only a WKT, "POINT (-8.1943 39.4402)"
                lon, lat = map(float, re.findall(r"-?\d+(?:\.\d+)?", r["wkt"])[:2])
            out[u.upper()] = {"name": r.get("name", ""),
                              "type": (r.get("type") or "").rsplit("/", 1)[-1],
                              "lon": lon, "lat": lat}
        except (KeyError, ValueError, TypeError):
            continue
    return out


def outline(cc):
    """Is-abroad test, or None. A point is abroad when it is outside this country's outline
    AND inside another country's: religiondots' country outlines (about 200 m, the ones
    tools/build_regions.py simplifies), each grown by about 300 m. Outside this country's
    outline alone is not enough: the outline follows the coast, so stations on the Tagus
    estuary and the coast (Alhandra, Alverca, Bobadela, Cascais, Figueira da Foz) fall outside
    it while being in no other country either. Sebnitz is 500 m beyond the Czech border and
    inside Germany's outline. Where that file is missing, dist/regions.json's 2 km outlines
    grown by 3 km, with "outside this country" alone (Hidasnémeti is 1.5 km inside the
    Hungarian one, so the 2 km outline is a poor second)."""
    import json
    from shapely import STRtree, prepared
    from shapely.geometry import Point, Polygon, box, shape
    from shapely.ops import unary_union
    if SHAPES.exists():
        with open(SHAPES, encoding="utf-8") as f:
            raw = [(x["properties"].get("cc"), x["geometry"]) for x in json.load(f)["features"]]
        # Countries religiondots has no outline for (Luxembourg, Monaco, San Marino, Vatican)
        # from Natural Earth 1:10m, as tools/build_regions.py does: without them SNCB's calls
        # in Luxembourg would read as domestic. Its "gb" is religiondots' "uk".
        if NE_SHAPES.exists():
            have = {c for c, _ in raw} | {"gb"}
            with open(NE_SHAPES, encoding="utf-8") as f:
                for x in json.load(f)["features"]:
                    c = (x["properties"].get("ISO_A2_EH") or "").lower()
                    if len(c) == 2 and c not in have:
                        raw.append((c, x["geometry"]))
        own = [shape(g) for c, g in raw if c == cc and g]
        if own:
            home = unary_union(own)
            geom = prepared.prep(home.buffer(0.003))
            # other countries near this one (within about 5 degrees), each grown the same
            x0, y0, x1, y1 = home.bounds
            near = box(x0 - 5, y0 - 5, x1 + 5, y1 + 5)
            others = []
            for c, g in raw:
                if c == cc or not g:
                    continue
                s = shape(g)
                if s.intersects(near):
                    others.append(s.buffer(0.003))
            tree = STRtree(others)

            def abroad(lonlat):
                p = Point(lonlat[0], lonlat[1])
                if geom.contains(p):
                    return False
                return len(tree.query(p, predicate="within")) > 0
            return abroad
    p = ROOT / "dist" / "regions.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        reg = json.load(f)
    reg = reg.get("regions", reg)
    if cc not in reg or not reg[cc].get("parts"):
        return None
    geom = prepared.prep(unary_union([Polygon(q) for q in reg[cc]["parts"]]).buffer(0.03))
    return lambda lonlat: not geom.contains(Point(lonlat[0], lonlat[1]))


def match_stations(cc, feed_st, nodes, points, log):
    """Feed station id -> build node id. `nodes`: {id: (name, lon, lat, junction)}."""
    stop_ok = {}
    for n, (name, lon, lat, junc) in nodes.items():
        # a junction node is a RINF point the build found no OSM station for; it can still be
        # where trains call when RINF types it a passenger stop
        u = n[1:].upper() if n.startswith("e") else ""
        stop_ok[n] = (not junc) or points.get(u, {}).get("type") in PASSENGER_TYPES
    grid = Grid((n, v[1], v[2]) for n, v in nodes.items())
    code = CODE.get(cc)
    out, how = {}, defaultdict(int)

    def by_name(names, lon, lat, r_km):
        best = None
        for d, n in grid.near(lon, lat, r_km):
            if not stop_ok[n]:
                continue
            for nm in names:
                t = name_tier(nm, nodes[n][0])
                if t is not None and (best is None or (t, d) < best[:2]):
                    best = (t, d, n)
        return best

    for fid, (fname, flon, flat, raw_id) in feed_st.items():
        if code:
            m = code[0].search(raw_id)
            if m:
                u = code[1](m).upper()
                if "e" + u in nodes:
                    out[fid] = "e" + u
                    how["code"] += 1
                    continue
                p = points.get(u)
                if p:
                    got = by_name([fname, p["name"]], p["lon"], p["lat"], CODE_KM)
                    if got:
                        out[fid] = got[2]
                        how["code, then name"] += 1
                        continue
                    near = [n for d, n in grid.near(p["lon"], p["lat"], BLIND_KM) if stop_ok[n]]
                    if near:
                        out[fid] = near[0]
                        how["code, then nearest"] += 1
                        continue
        alias = NAME_ALIAS.get(cc, {}).get(fname)
        got = by_name([fname] + ([alias] if alias else []), flon, flat, NAME_KM)
        if got:
            out[fid] = got[2]
            how[f"name tier {got[0]}"] += 1
            continue
        near = [n for d, n in grid.near(flon, flat, BLIND_KM) if stop_ok[n]]
        if near:
            out[fid] = near[0]
            how["nearest"] += 1
    return out, dict(how)


# ================================================================ paths

class Graph:
    def __init__(self, edges, no_osm=frozenset()):
        """edges: list of (a, b, km). Parallel edges between the same two ends are grouped.
        no_osm: indices of edges no OSM passenger route runs over; such a parallel edge is not
        credited beside one that has a route (Portugal's Ramal TER-TIR beside the Norte)."""
        self.adj = defaultdict(dict)        # node -> {neighbour: (min km, [edge index])}
        self.edges = edges
        self.no_osm = no_osm
        for i, (a, b, km) in enumerate(edges):
            if a == b:
                continue
            for x, y in ((a, b), (b, a)):
                cur = self.adj[x].get(y)
                if cur is None:
                    self.adj[x][y] = (km, [i])
                else:
                    self.adj[x][y] = (min(cur[0], km), cur[1] + [i])

    def dijkstra(self, src, limit):
        dist, prev = {src: 0.0}, {}
        h = [(0.0, src)]
        while h:
            d, u = heapq.heappop(h)
            if d > dist.get(u, 1e18):
                continue
            for v, (km, _) in self.adj[u].items():
                nd = d + km
                if nd <= limit and nd < dist.get(v, 1e18):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(h, (nd, v))
        return dist, prev

    def step_edges(self, u, v):
        km, idx = self.adj[u][v]
        got = [i for i in idx if self.edges[i][2] <= PARALLEL * km + 0.2]
        if any(i not in self.no_osm for i in got):
            got = [i for i in got if i not in self.no_osm]
        return got


def dangling_links(reg, nodes, border, graph):
    """[(a, b, km)]: links for register line ends that do not meet. A junction node (not a
    stop, not a border point) that only one section of the whole register touches is a line
    end meeting nothing; it is joined to the nearest node of another line within LINK_KM.
    Portugal's Variante de Alcácer (line 68) ends 0.96 km from the Pinheiro node with no
    section between, so without the link the shortest path from Lisboa to the Algarve ran
    over the old Linha do Sul, Pinheiro - Grândola Norte. The link is no section: a path over
    it credits only the sections on either side."""
    deg = defaultdict(int)
    line_of = defaultdict(set)
    nbr = defaultdict(set)
    for l in reg:
        for s in l["sections"]:
            if s[0] == s[1]:
                continue
            nbr[s[0]].add(s[1])
            nbr[s[1]].add(s[0])
            for n in s[:2]:
                deg[n] += 1
                line_of[n].add(l["id"])
    grid = Grid((n, nodes[n][1], nodes[n][2]) for n in deg)
    links = []
    for n, k in deg.items():
        if k != 1 or not nodes[n][3] or n in border:
            continue
        own = line_of[n]
        for d, m in grid.near(nodes[n][1], nodes[n][2], LINK_KM):
            if m != n and m not in nbr[n] and not (line_of[m] <= own):
                # an end that reaches that node over the register nearby already meets it (a
                # siding's end beside a station: Chomutov prům. kolej); only a real gap is
                # joined (the Variante's start reaches Pinheiro only by Grândola, 65 km)
                lim = LINK_REACH_KM + 3 * d
                if m not in graph.dijkstra(n, lim)[0]:
                    links.append((n, m, max(d, 0.001)))
                break
    return links


def call_pairs(patterns, feed_st, fmatch, border_between, find_path):
    """Consecutive matched calls of every pattern -> {(u, v): [path km, trips, tripdays,
    [edge index, ...]]}, and the number of calls stepped over for want of a path.

    An unmatched call is stepped over (a halt the build has inside a section), unless a
    border point lies on the way to it (`border_between`): then it is a call abroad, the
    train's run is cut at that border point, and picks up again at the border point on the
    way back in. A matched call with no register path from the previous one is stepped over
    too (Žatec západ is a RINF point of another line, not of the 160 the trains use), and if
    the call after it has no path from the previous one either but has one from the skipped
    call, it was the previous one that was wrong, and the run picks up from the skipped call.
    """
    pairs = {}
    n_skip = 0

    def add(a, b, along, trips, tdays):
        if a == b:
            return True
        got = find_path(a, b, DETOUR * along + SLACK_KM)
        if got is None:
            return False
        u, v = (a, b) if a < b else (b, a)
        rec = pairs.get((u, v))
        if rec is None:
            pairs[(u, v)] = [got[0], trips, tdays, got[1], along]
        else:
            rec[1] += trips
            rec[2] += tdays
            rec[4] = min(rec[4], along)
        return True

    for seq, trips, tdays in patterns:
        # anchors: [node, km along the calls since it]; the first is the last good call, a
        # second is a matched call that had no path from it
        anchors, prev_pos, abroad, trail, first = [], None, None, None, True
        for fid in seq:
            st = feed_st.get(fid)
            if st is None:
                continue
            pos = (st[1], st[2])
            step = km_between(*prev_pos, *pos) if prev_pos else 0.0
            for a in anchors:
                a[1] += step
            node = fmatch.get(fid)
            if node is not None:
                if anchors:
                    ok = next((a for a in anchors if add(a[0], node, a[1], trips, tdays)), None)
                    if ok is None:
                        n_skip += 1
                        anchors = [anchors[0], [node, 0.0]]
                    else:
                        anchors = [[node, 0.0]]
                else:
                    if abroad is not None:
                        # calls before this one were abroad; at the start of the run the
                        # outline need not say so (Sebnitz is 400 m beyond the border)
                        b = border_between(pos, abroad, force=first)
                        if b is not None:
                            add(b[0], node, b[1], trips, tdays)
                    anchors = [[node, 0.0]]
                abroad, trail, first = None, None, False
            elif anchors:
                b = border_between(prev_pos, pos)
                if b is not None:
                    add(anchors[0][0], b[0], anchors[0][1] - step + b[1], trips, tdays)
                    anchors, abroad = [], pos
                elif trail is None:
                    trail = (prev_pos, pos, anchors[0][1] - step)
            else:
                abroad = pos
            prev_pos = pos
        if anchors and trail is not None:
            # the run ends on calls matched to nothing: abroad if a border point is on the way
            b = border_between(trail[0], trail[1], force=True)
            if b is not None:
                add(anchors[0][0], b[0], trail[2] + b[1], trips, tdays)
    return pairs, n_skip


# ================================================================ the check

def check(region, lines, stations, route_share, log):
    """Decide per register section from the region's timetable feed. Returns None when the
    region has no feed (the build then runs exactly as before); otherwise sets route_share to
    1.0 for every junction-ended section a train runs over, and returns what `mark` closes."""
    feed = load_feeds(region, log)
    if feed is None:
        return None
    for info in feed["info"]:
        log(f"timetable: {info['file']}: {info['trips']:,} rail trips in {info['patterns']:,} "
            f"stop patterns, {info['from']} to {info['to']}, {len(info['agencies'])} agencies"
            + (f" (left out: {', '.join(info['skipped'])})" if info["skipped"] else ""))
    reg = [l for l in lines if l.get("src", "osm") != "osm"]
    nodes = {}
    for l in reg:
        for s in l["sections"]:
            for n in s[:2]:
                st = stations[n]
                nodes[n] = (st["name"], st["lon"], st["lat"], bool(st.get("junction")))
    points = rinf_points(region)
    feed_st = feed["stations"]
    fmatch, how = match_stations(region, feed_st, nodes, points, log)
    called = {fid for seq, _, _ in feed["patterns"] for fid in seq}
    log(f"timetable: {len(fmatch):,} of {len(feed_st):,} feed stations matched to a register "
        f"station ({', '.join(f'{k} {v}' for k, v in sorted(how.items()))}); of the "
        f"{len(called):,} that trains call at, {len(called - set(fmatch)):,} matched nothing")

    # a call abroad: in another country's outline, with a RINF border point on the way to it
    # A border point is RINF's type 90, or any id in borders.py's table: build_model gives a
    # crossing RINF files twice one id (borders.canon), which can be the neighbour's point
    # (Romania's EU00209 becomes Bulgaria's EU00208), not in this country's points.json.
    import borders
    table = {p["id"] for p in borders.load()}
    border = [n for n in nodes if n.startswith("e") and (
              n in table or points.get(n[1:].upper(), {}).get("type") in BORDER_TYPES)]
    bgrid = Grid((n, nodes[n][1], nodes[n][2]) for n in border)
    is_abroad = outline(region)
    bcache = {}

    def border_between(p, q, force=False):
        """(border node, km from p to it) for a call at p in the country and one at q
        abroad, or None when q is not abroad or no border point lies between them. `force`:
        q is abroad if a border point lies between, whatever the outline says."""
        if (p, q, force) in bcache:
            return bcache[(p, q, force)]
        got = None
        if force or is_abroad is None or is_abroad(q):
            d = km_between(*p, *q)
            lim = BORDER_DETOUR * d + BORDER_SLACK_KM
            best = None
            for dpb, n in bgrid.near(p[0], p[1], lim):
                s = dpb + km_between(nodes[n][1], nodes[n][2], *q)
                if s <= lim and (best is None or s < best[0]):
                    best = (s, n, dpb)
            if best:
                got = (best[1], best[2])
        bcache[(p, q, force)] = got
        return got

    # A feed that stops its international trains at the last station before the border (MÁV's
    # does: Budapest - Wien ends at Hegyeshalom) cannot say whether the section from there to
    # the border is run over. A border point counts as seen only with a call abroad near it.
    abroad = [feed_st[f] for f in called if f in feed_st and is_abroad is not None
              and is_abroad(feed_st[f][1:3])]
    agrid = Grid((i, s[1], s[2]) for i, s in enumerate(abroad))
    unseen_border = {n for n in border if not agrid.near(nodes[n][1], nodes[n][2], BORDER_SEEN_KM)}
    if border:
        log(f"timetable: {len(abroad)} stations abroad called at "
            f"({', '.join(sorted(s[0] for s in abroad)[:12])}{'...' if len(abroad) > 12 else ''}); "
            f"{len(unseen_border)} of {len(border)} border points with none within "
            f"{BORDER_SEEN_KM:.0f} km, whose sections the feed cannot decide")

    unmatched = sorted({feed_st[f][0] for f in called - set(fmatch)
                        if f in feed_st and (is_abroad is None or not is_abroad(feed_st[f][1:3]))})
    if unmatched:
        log(f"timetable: called at inside the country but matched to no register station "
            f"({len(unmatched)}; stepped over): {', '.join(unmatched[:80])}")

    edges, eref = [], []
    for l in reg:
        for s in l["sections"]:
            edges.append((s[0], s[1], float(s[2])))
            eref.append((l, f"{s[0]}|{s[1]}"))
    n_reg = len(edges)
    links = dangling_links(reg, nodes, set(border), Graph(edges))
    if links:
        log(f"timetable: {len(links)} register line ends at a junction that meet no other line, "
            f"joined to the nearest station of another line within {LINK_KM:g} km by a link "
            f"that is no section: "
            + ", ".join(f"{nodes[a][0]} - {nodes[b][0]} ({km:.2f} km)" for a, b, km in links))
    edges += links
    # Paths are weighed with OSM_TIE added to sections no OSM passenger route runs over, so of
    # two paths of about the same length the one OSM's routes take wins: Portugal's freight
    # loops beside the Linha do Norte (Plataforma de Cacia, Bobadela, Ramal TER-TIR) are as long
    # as the main line between the same two junctions and were credited instead of it. A
    # section that is the only way is still found; caps are tested on the real length.
    no_osm = frozenset(i for i in range(n_reg) if route_share.get(
        (eref[i][0]["id"], eref[i][1]), 0.0) < NEEDS_ROUTE_SHARE)
    # A link weighs twice its crow-fly length plus LINK_EXTRA_KM, so it never beats real track
    # (at Olen a 0.1 km siding fragment with both ends linked came out shorter than line 15).
    wedges = [(a, b, km * (1.0 + OSM_TIE) if i in no_osm else
               2 * km + LINK_EXTRA_KM if i >= n_reg else km)
              for i, (a, b, km) in enumerate(edges)]
    g = Graph(wedges, no_osm)
    sp_cache = {}

    def find_path(u, v, cap):
        """(weighed km, [edge index, ...]) of the shortest register path from u to v, if its
        real length is within cap."""
        got = sp_cache.get(u)
        wcap = cap * (1.0 + OSM_TIE)
        if got is None or (got[0] < wcap and v not in got[1]):
            lim = max(wcap, got[0] if got else 0.0, 60.0)
            dist, prev = g.dijkstra(u, lim)
            got = sp_cache[u] = (lim, dist, prev)
        _, dist, prev = got
        if dist.get(v, 1e18) > wcap:
            return None
        path, x, real = [], v, 0.0
        while x != u:
            p = prev[x]
            step = g.step_edges(p, x)
            real += min(edges[i][2] for i in step)
            path += step
            x = p
        if real > cap:
            return None
        return dist[v], path

    pairs, n_skip = call_pairs(feed["patterns"], feed_st, fmatch, border_between, find_path)
    trips = [0] * len(edges)
    tdays = [0] * len(edges)
    # Whether some run crossing a section is good evidence for it: one that starts or ends at
    # the section's own ends (Olecko - Suwałki, 43 km, is one section between two calls), or a
    # short and direct one. Over a long non-stop run, or a path that winds well beyond the
    # crow-fly line between the calls, the shortest register path is weak evidence: Gliwice -
    # Chałupki (74 km) ran over Zabrze Makoszowy colliery track, and Amsterdam Muiderpoort -
    # Centraal (3 km apart, 7 km path, a register gap) through the Watergraafsmeer depot.
    strong = [False] * len(edges)
    for (u, v), (km, n, d, path, along) in pairs.items():
        good = km <= LONG_RUN_KM and km <= STRAIGHT * along + 1.0
        for i in path:
            trips[i] += n
            tdays[i] += d
            if good or u in edges[i][:2] or v in edges[i][:2]:
                strong[i] = True
    log(f"timetable: {len(pairs):,} pairs of consecutive calls with a register path within "
        f"{DETOUR}x crow-fly + {SLACK_KM:.0f} km; {n_skip:,} calls stepped over for want of one")
    # Direct evidence beats the path search: a section whose two ends some train calls at one
    # after the other (calls matched to nothing in between skipped) is run over, even where a
    # wrong match one call earlier sent the path search astray (Austria's Jauntalbahn: Mittlern
    # - St. Michael ob Bleiburg - Bleiburg, where St. Michael - Bleiburg came out closed).
    direct = defaultdict(lambda: [0, 0])
    for seq, n, d in feed["patterns"]:
        prev = None
        for fid in seq:
            node = fmatch.get(fid)
            if node is None:
                continue
            if prev is not None and prev != node:
                rec = direct[(prev, node) if prev < node else (node, prev)]
                rec[0] += n
                rec[1] += d
            prev = node
    n_direct = 0
    for i in range(n_reg):
        a, b, _ = edges[i]
        rec = direct.get((a, b) if a < b else (b, a))
        if rec and not trips[i]:
            trips[i], tdays[i] = rec
            strong[i] = True
            n_direct += 1
    if n_direct:
        log(f"timetable: {n_direct} sections on no path found, but with trains calling at both "
            f"ends one after the other: served")

    # Past snapshots (PAST): the same paths for their trains, kept apart. Used only to tell a
    # seasonal line from a closed one (below); they never make a section "served".
    past = load_past(region, log)
    past_trips = [0] * len(edges)
    past_days = [0] * len(edges)        # trip-days: one train a day for a day is 1
    ppairs = {}
    if past is not None:
        pmatch, _ = match_stations(region, past["stations"], nodes, points, log)
        ppairs, _ = call_pairs(past["patterns"], past["stations"], pmatch, border_between,
                               find_path)
        for (u, v), (km, n, d, path, along) in ppairs.items():
            for i in path:
                past_trips[i] += n
                past_days[i] += d
        pdirect = defaultdict(lambda: [0, 0])
        for seq, n, d in past["patterns"]:
            prev = None
            for fid in seq:
                node = pmatch.get(fid)
                if node is None:
                    continue
                if prev is not None and prev != node:
                    rec = pdirect[(prev, node) if prev < node else (node, prev)]
                    rec[0] += n
                    rec[1] += d
                prev = node
        for i in range(n_reg):
            a, b, _ = edges[i]
            rec = pdirect.get((a, b) if a < b else (b, a))
            if rec and not past_trips[i]:
                past_trips[i], past_days[i] = rec
        log(f"timetable: past snapshots: " + "; ".join(
            f"{info['file']} {info['from']} to {info['to']}, {info['trips']:,} rail trips"
            for info in past["info"]))

    # ambiguous: on a near-shortest path of some pair
    pairs_at = defaultdict(list)
    for (u, v), rec in pairs.items():
        pairs_at[u].append((v, rec[0]))
        pairs_at[v].append((u, rec[0]))

    near_cache = {}

    def reach(n, limit):
        if n not in near_cache or near_cache[n][0] < limit:
            near_cache[n] = (limit, g.dijkstra(n, limit)[0])
        return near_cache[n][1]

    def ambiguous(i):
        a, b, L = wedges[i]
        lim = 150.0
        da, db = reach(a, lim), reach(b, lim)
        for x, dx_a in da.items():
            for y, d in pairs_at.get(x, ()):
                dy_b = db.get(y)
                if dy_b is not None and dx_a + L + dy_b <= NEAR * d + 0.5:
                    return True
        for x, dx_b in db.items():
            for y, d in pairs_at.get(x, ()):
                dy_a = da.get(y)
                if dy_a is not None and dx_b + L + dy_a <= NEAR * d + 0.5:
                    return True
        return False

    # A stop the feed knows: something calls there, if only a replacement bus. A stop the
    # feed lists with nothing calling, or does not list, is where an operator it lacks runs.
    any_call = set().union(*(info["any_call"] for info in feed["info"]))
    known = {n for f, n in fmatch.items() if f in any_call}
    # A feed with no international trains (NO_INTERNATIONAL) cannot decide the track from the
    # last stop it serves to the border either: sections reached from a border point it cannot
    # see, through stops it does not know, are "border" too. Croatia: HŽPP's feed stops at
    # Koprivnica, and Koprivnica - Peteranec - Novo Drnje - Botovo carries only the trains to
    # Hungary (823 in HŽ Infrastruktura's 2025 statistics).
    border_reach = set()
    if region in NO_INTERNATIONAL:
        at = defaultdict(list)
        for i in range(n_reg):
            at[edges[i][0]].append(i)
            at[edges[i][1]].append(i)
        stack, seen = list(unseen_border), set(unseen_border)
        while stack:
            x = stack.pop()
            for i in at[x]:
                border_reach.add(i)
                y = edges[i][1] if edges[i][0] == x else edges[i][0]
                if y not in seen and y not in known:
                    seen.add(y)
                    stack.append(y)
    # A line whose infrastructure manager is neither the country's main one nor among the
    # feed's agencies by name runs its own trains, which the feed may simply not carry
    # (JHMD's 228 and 229 in Czechia). Its sections with no train are not closed unless the
    # feed knows their stops.
    ims = defaultdict(float)
    for l in reg:
        ims[l.get("operator", "")] += l["km"]
    main_im = max(ims, key=ims.get) if ims else ""
    agency_toks = [set(tokens(a)) - ORG_WORDS for info in feed["info"] for a in info["agencies"]]
    absent = {im for im in ims if im != main_im and
              not any((set(tokens(im)) - ORG_WORDS) & t for t in agency_toks)}
    if absent:
        log(f"timetable: infrastructure managers with no agency of their name in the feed: "
            f"{', '.join(sorted(repr(x) for x in absent))}")
    state = {}
    for i, (l, key) in enumerate(eref):
        a, b, L = edges[i]
        ja, jb = nodes[a][3], nodes[b][3]
        if trips[i] and (ja or jb) and i in no_osm and not strong[i]:
            # a junction-ended section no OSM route runs over is not rescued on weak evidence
            state[i] = "weak"
        elif trips[i]:
            state[i] = "served"
        elif ambiguous(i):
            state[i] = "ambiguous"
        elif a in unseen_border or b in unseen_border or i in border_reach:
            state[i] = "border"
        elif (route_share.get((l["id"], key), 0.0) >= NEEDS_ROUTE_SHARE
              or l.get("operator", "") in absent) and \
                not all(n in known for n, j in ((a, ja), (b, jb)) if not j):
            state[i] = "unknown"
        else:
            state[i] = "closed"

    # An "unknown" stop-to-stop section on a line that is closed next to it, reached from the
    # closed part through stops no train calls at, is closed too: the line has no trains, its
    # replacement buses just do not call at every halt. Greece: Larissa - Volos and
    # Palaiofarsalos - Kalambaka are bus-replaced, and Velestino - Volos closed while Velestino
    # - Stefanovikeio stayed unknown. A line whose operator is missing from the feed has no
    # closed section to grow from (JHMD's 228 and 229).
    train_nodes = {fmatch[f] for seq, _, _ in feed["patterns"] for f in seq if f in fmatch}
    by_line_i = defaultdict(list)
    for i, (l, key) in enumerate(eref):
        by_line_i[l["id"]].append(i)
    n_grown = 0
    for lid, idx in by_line_i.items():
        reach = {n for i in idx if state[i] == "closed" and not nodes[edges[i][0]][3]
                 and not nodes[edges[i][1]][3] for n in edges[i][:2]} - train_nodes
        grown = bool(reach)
        while grown:
            grown = False
            for i in idx:
                a, b, _ = edges[i]
                if state[i] == "unknown" and not nodes[a][3] and not nodes[b][3] and \
                        (a in reach or b in reach):
                    state[i] = "closed"
                    n_grown += 1
                    reach |= {a, b} - train_nodes
                    grown = True
    if n_grown:
        log(f"timetable: {n_grown} unknown sections closed with the closed line beside them")

    # Seasonal: a section with no train now that a past snapshot (PAST) has trains on is drawn
    # as running (Anita, 2026-10-01: seasonal lines count). Poland's regional feed covers five
    # autumn weeks; its July snapshot has Koleje Małopolskie's summer weekend trains Muszyna -
    # Leluchów - Plaveč. Not where replacement buses now call at a stop no train calls at:
    # that is a works closure that began after the snapshot (Opole - Nysa, closed 3 August).
    if past is not None:
        for i in range(n_reg):
            if state[i] not in ("closed", "unknown") or past_days[i] < SEASONAL_MIN_DAYS or \
                    (nodes[edges[i][0]][3] and nodes[edges[i][1]][3]):
                continue
            a, b, _ = edges[i]
            if any(n in known and n not in train_nodes for n in (a, b) if not nodes[n][3]):
                continue
            state[i] = "seasonal"

    # A junction-ended section no train's shortest path crosses is weak evidence on its own:
    # with no shapes, the feed cannot say which of two tracks into Praha Masarykovo nádraží a
    # train takes. So it is not running only as part of a closed line, sharing an end (in a
    # chain) with a closed stop-to-stop section of the same line. Otherwise, where OSM routes
    # run over it, it is left as it is ("osm"); where none do, drop_unridden_sections drops
    # it as it always did, which is right for freight curves.
    in_closed_line = set()
    by_line = defaultdict(list)
    for i, (l, key) in enumerate(eref):
        by_line[l["id"]].append(i)
    for lid, idx in by_line.items():
        ends = defaultdict(set)
        for i in idx:
            a, b, _ = edges[i]
            if state[i] == "closed" and not nodes[a][3] and not nodes[b][3]:
                in_closed_line.add(i)
                ends[a].add(i)
                ends[b].add(i)
        grown = True
        while grown:
            grown = False
            for i in idx:
                if state[i] != "closed" or i in in_closed_line:
                    continue
                a, b, _ = edges[i]
                if ends[a] or ends[b]:
                    in_closed_line.add(i)
                    ends[a].add(i)
                    ends[b].add(i)
                    grown = True

    # hand the decisions to drop_unridden_sections through route_share
    rescued, closed = defaultdict(list), defaultdict(list)
    for i, (l, key) in enumerate(eref):
        a, b, L = edges[i]
        junc = nodes[a][3] or nodes[b][3]
        rs = route_share.get((l["id"], key), 0.0)
        if state[i] == "served" and junc:
            if rs < NEEDS_ROUTE_SHARE:
                rescued[l["id"]].append(i)
            route_share[(l["id"], key)] = 1.0
        elif state[i] == "seasonal" and junc:
            route_share[(l["id"], key)] = 1.0
        elif state[i] == "closed":
            if i not in in_closed_line:
                # left to OSM's routes: kept where they run over it, else dropped by
                # drop_unridden_sections as it always was
                state[i] = "osm" if rs >= NEEDS_ROUTE_SHARE else "dropped"
                continue
            if junc:
                route_share[(l["id"], key)] = 1.0
            closed[l["id"]].append(i)

    def sec_name(i):
        a, b, L = edges[i]
        return f"{nodes[a][0]} - {nodes[b][0]} ({L:.1f} km)"

    lname = {l["id"]: f"{l.get('ref') or ''} {l['name']}".strip() for l in reg}
    km_res = sum(edges[i][2] for v in rescued.values() for i in v)
    log(f"timetable: rescued {sum(map(len, rescued.values()))} junction-ended sections "
        f"({km_res:,.0f} km) on {len(rescued)} lines that OSM routes alone would drop:")
    for lid in sorted(rescued, key=lambda k: lname[k]):
        for i in rescued[lid]:
            log(f"    rescued  {lname[lid]}: {sec_name(i)}, {trips[i]} trips")
    km_cl = sum(edges[i][2] for v in closed.values() for i in v)
    log(f"timetable: not running, no train in the feed: {sum(map(len, closed.values()))} "
        f"sections ({km_cl:,.0f} km) on {len(closed)} lines:")
    for lid in sorted(closed, key=lambda k: -sum(edges[i][2] for i in closed[k])):
        k = sum(edges[i][2] for i in closed[lid])
        log(f"    closed   {lname[lid]}: {k:.1f} km")
        for i in closed[lid]:
            log(f"        {sec_name(i)}")
    for what, label in (("unknown", "no train in the feed, but OSM routes run over it or its "
                                    "manager is not in the feed, and nothing in the feed calls "
                                    "at its stops (an operator missing from the feed?)"),
                        ("seasonal", "no train now, but trains in a past snapshot and no "
                                     "replacement bus at its stops: drawn as running"),
                        ("ambiguous", "not on a shortest path, but on one nearly as short"),
                        ("weak", f"junction-ended, no OSM route, and on the path only of runs "
                                 f"longer than {LONG_RUN_KM:.0f} km or winding beyond "
                                 f"{STRAIGHT}x crow-fly + 1 km"),
                        ("border", "to a border point the feed has no call beyond"),
                        ("osm", "junction-ended, on no train's shortest path but OSM routes "
                                "run over it, and not part of a closed line")):
        idx = [i for i in state if state[i] == what]
        if not idx:
            continue
        log(f"timetable: left as they are, {what}: {len(idx)} sections "
            f"({sum(edges[i][2] for i in idx):,.0f} km), {label}:")
        for i in sorted(idx, key=lambda i: (lname[eref[i][0]['id']], i)):
            log(f"    {what:9s} {lname[eref[i][0]['id']]}: {sec_name(i)}")
    served_km = sum(edges[i][2] for i in state if state[i] == "served")
    log(f"timetable: of {sum(e[2] for e in edges[:n_reg]):,.0f} register km, "
        f"{served_km:,.0f} served")
    LAST.clear()
    LAST.update(past_trips=past_trips, past_days=past_days, past_pairs=ppairs, past=past)
    LAST.update(nodes=nodes, fmatch=fmatch, feed=feed, pairs=pairs, graph=g, edges=edges,
                eref=eref, state=state, trips=trips, known=known)
    return {"closed": {lid: [eref[i][1] for i in v] for lid, v in closed.items()}}


LAST = {}   # the last check's internals, for looking into a decision from a script


def mark(result, lines, log):
    """After not_running.mark: add the sections the feed shows no train on to `closed`."""
    if not result:
        return
    n = 0
    for l in lines:
        keys = result["closed"].get(l["id"])
        if not keys:
            continue
        have = {f"{s[0]}|{s[1]}" for s in l["sections"]}
        cur = list(l.get("closed", []))
        add = [k for k in keys if k in have and k not in cur]
        if add:
            l["closed"] = cur + add
            n += len(add)
    log(f"timetable: {n} sections marked not running")


# ================================================================ fetching

def _download(url, path):
    """url -> path. Python's certificate check fails on some hosts on this machine
    (data.public.lu: "self signed certificate in certificate chain") where curl's passes, so a
    certificate failure is retried with curl."""
    import gzip
    import ssl
    import subprocess
    # digitraffic answers 406 without Accept-Encoding: gzip
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r, open(path, "wb") as f:
            src = gzip.GzipFile(fileobj=r) if r.headers.get("Content-Encoding") == "gzip" else r
            while True:
                b = src.read(1 << 20)
                if not b:
                    break
                f.write(b)
    except urllib.error.URLError as e:
        if not isinstance(getattr(e, "reason", None), ssl.SSLError):
            raise
        subprocess.run(["curl", "-L", "-s", "-f", "--compressed", "-A", USER_AGENT, "-o",
                        str(path), url], check=True)


def fetch(cc, past=False):
    """Download a country's feeds (or, with past, its PAST snapshots into past/). A `slim` one
    is cut down to its rail routes and the full download deleted (OVapi's Netherlands feed is
    238 MB with every bus in the country)."""
    import json
    d = past_dir(cc) if past else RAW / cc
    d.mkdir(parents=True, exist_ok=True)
    for name, url, slim in (PAST if past else FEEDS)[cc]:
        tmp = d / (name + ".download")
        if url.startswith(UDATA):
            _download(url[len(UDATA):], tmp)
            with open(tmp, encoding="utf-8") as f:
                res = [x for x in json.load(f)["resources"]
                       if (x.get("format") or "").lower() == "zip"]
            url = max(res, key=lambda x: x["created_at"])["url"]
            print(f"{cc}: newest resource {url}")
        _download(url, tmp)
        size = tmp.stat().st_size / 1e6
        if slim:
            slim_feed(tmp, d / name)
            tmp.unlink()
        else:
            os.replace(tmp, d / name)
        print(f"{d / name}: {(d / name).stat().st_size / 1e6:.1f} MB"
              + (f" (slimmed from {size:.1f} MB)" if slim else ""))


def slim_feed(src, dst):
    """Rewrite a GTFS zip keeping only rail routes (RAIL_TYPES) and rail-replacement buses
    (714), their trips, stop times, stops (with parents), services and agencies. Shapes go."""
    zin = zipfile.ZipFile(src)
    names = set(zin.namelist())

    def rows(name):
        with zin.open(name) as f:
            r = csv.reader(io.TextIOWrapper(f, "utf-8-sig"))
            head = next(r)
            yield head
            yield from r

    def write(zout, name, keep):
        if name not in names:
            return
        it = rows(name)
        head = next(it)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(head)
        test = keep(head)
        with zout.open(name, "w") as out:
            for row in it:
                if test(row):
                    w.writerow(row)
                if buf.tell() > 1 << 20:
                    out.write(buf.getvalue().encode("utf-8"))
                    buf.seek(0)
                    buf.truncate()
            out.write(buf.getvalue().encode("utf-8"))

    def col(head, c):
        return head.index(c) if c in head else None

    it = rows("routes.txt")
    head = next(it)
    ri, ti = col(head, "route_id"), col(head, "route_type")
    routes = set()
    for row in it:
        try:
            t = int(row[ti])
        except ValueError:
            continue
        if t in RAIL_TYPES or t == 714:
            routes.add(row[ri])
    it = rows("trips.txt")
    head = next(it)
    ri, tj, si = col(head, "route_id"), col(head, "trip_id"), col(head, "service_id")
    trips, services = set(), set()
    for row in it:
        if row[ri] in routes:
            trips.add(row[tj])
            services.add(row[si])
    it = rows("stop_times.txt")
    head = next(it)
    tj, sj = col(head, "trip_id"), col(head, "stop_id")
    stops = set()
    for row in it:
        if row[tj] in trips:
            stops.add(row[sj])
    it = rows("stops.txt")
    head = next(it)
    sj, pj = col(head, "stop_id"), col(head, "parent_station")
    if pj is not None:
        for row in it:
            if row[sj] in stops and row[pj]:
                stops.add(row[pj])
    tmp = Path(str(dst) + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in ("agency.txt", "feed_info.txt"):
            write(zout, name, lambda h: (lambda r: True))
        write(zout, "routes.txt", lambda h: (lambda r, i=h.index("route_id"): r[i] in routes))
        write(zout, "trips.txt", lambda h: (lambda r, i=h.index("trip_id"): r[i] in trips))
        write(zout, "stop_times.txt",
              lambda h: (lambda r, i=h.index("trip_id"): r[i] in trips))
        write(zout, "stops.txt", lambda h: (lambda r, i=h.index("stop_id"): r[i] in stops))
        for name in ("calendar.txt", "calendar_dates.txt"):
            write(zout, name, lambda h: (lambda r, i=h.index("service_id"): r[i] in services))
    os.replace(tmp, dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", metavar="CC", nargs="+",
                    help="download FEEDS[CC] into data/raw/gtfs/CC/")
    ap.add_argument("--fetch-past", metavar="CC", nargs="+",
                    help="download PAST[CC] into data/raw/gtfs/CC/past/")
    args = ap.parse_args()
    for cc in args.fetch or []:
        fetch(cc)
    for cc in args.fetch_past or []:
        fetch(cc, past=True)
