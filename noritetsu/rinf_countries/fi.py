"""Finland: RINF's line ids are Väylävirasto's track numbers (ratanumero), "001" to "751", plus
yard and depot ids like "KV225" or "003808TRE". A ratanumero is the unit of Väylä's track
kilometre system, not a line anyone rides by: 006 runs Riihimäki - Lahti - Kouvola - Imatra -
Joensuu - Nurmes - Kontiomäki (702 km), 008 Seinäjoki - Oulu - Rovaniemi - Kemijärvi. Finnish
lines are known by NAME, the names fi.wikipedia's line articles use ("Savon rata", "Karjalan
rata", "Pohjanmaan rata", "Turku–Toijala-rata"), and those cut across the track numbers.
Väylä's own open data names no lines (its "tilirataosat" are accounting sections written
"(Lahti) - (Kouvola)"), Wikidata has no route numbers for Finnish lines (P1671 holds only the
Helsinki metro and a tram line), and there is no public line number, so lines carry no `ref`.

LINES.  `fix` (rinf.py's hook for correcting RINF in place) files every section under the named
line it belongs to, and rinf.py groups by that name (`id_name` gives it back as the line's
name):

- WHOLE: a track number that is one named line, or a piece folded into one (337, Turku asema -
  Turku satama, is the harbour end of the Rantarata).
- CUT: a track number holding several named lines in a row, cut by RINF's own chainage: each
  section goes to the stretch its nearer end lies in, measured from the first station along
  the track number's own sections.
- SIDING: a section with an end at a private siding's boundary (RINF op-type 140, "Puhos Stora
  Enso Timber", "Raideinfra Oy Kouvola varikko") or a depot (50). Left in, each such point was
  a branch point of its line, the line was cut there into sections ending at junctions, and
  build_model dropped every one no OSM passenger route runs over: the Karjalan rata, which OSM
  has no route for, kept 59 of 315 km, the Orivesi–Jyväskylä-rata lost Jyväskylä - Jämsänkoski
  (53 km) at the UPM mill.
- Everything else (industrial branches, yards, depot tracks) is left out by `skip_line`, and so
  are FREIGHT: named lines or stretches with no passenger train that would survive because both
  ends are stations (checked against Fintraffic's timetable for 2026-09-30: no passenger train
  between Nurmes and Kontiomäki or Jyväskylä and Haapajärvi).

The split between named lines is fi.wikipedia's list of line articles (Suomen rautatieliikenne,
"Artikkeleita Suomen rataverkon osuuksista") with each article's own extent. Päärata's article
runs Helsinki - Oulu, but its middle and north have articles of their own (Tampere–Seinäjoki-rata,
Pohjanmaan rata), so the Päärata here is Helsinki - Tampere, the stretch the article calls the
most important one. Details and sources in fi_sources.md.

STATIONS.  `stop_name` reads Fintraffic's station list (passenger flag, VR's Finnish name),
joined on the UIC code: RINF's uopid "FI00408" is its UIC 408. See fi_stop_name.

OSM's route=railway relations are not read (`osm_rel` returns None): some carry a ratanumero as
`ref` (Uudenkaupungin rata 332, Rauman rata 342) or an unrelated one (Savon rata "231"), which
would have shown as line numbers, and every name they give is already in the tables below.
"""
import heapq
import json
import re
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_RAW = _ROOT / "data" / "raw" / "rinf" / "fi"

# Track numbers that are one named line, or a piece of one.
WHOLE = {
    "001": "Rantarata", "337": "Rantarata",
    "002": "Tampere–Pori-rata",
    "004": "Jyväskylä–Haapajärvi-rata",
    "007": "Lahden oikorata",
    "014": "Huutokoski–Parikkala-rata",
    "017": "Siilinjärvi–Viinijärvi-rata",
    "024": "Pieksämäki–Joensuu-rata",
    "087": "Iisalmi–Ylivieska-rata",
    "123": "Kehärata",                      # Vantaankosken rata redirects to it
    "125": "Vuosaaren satamarata",
    # Olli - Porvoo; 131 (Kerava - Olli - Sköldvik) is cut in CUT
    "132": "Porvoon rata",
    "141": "Hyvinkää–Karjaa-rata",
    "142": "Hangon rata",
    "213": "Luumäki–Vainikkala",            # no article; Väylä's section name
    "221": "Kotkan rata", "225": "Kotkan rata",     # 225: Kotka asema - Kotkan satama
    "222": "Haminan rata",
    "232": "Kouvola–Kuusankoski-rata",
    "251": "Lahti–Heinola-rata",
    "252": "Lahti–Loviisa-rata",
    "314": "Valkeakosken rata",
    "321": "Turku–Toijala-rata",
    "332": "Uudenkaupungin rata", "336": "Uudenkaupungin rata",
    "333": "Naantalin rata",
    "342": "Rauman rata",
    "363": "Kaipolan rata",
    "373": "Mäntän rata",
    "415": "Pietarsaaren rata", "416": "Pietarsaaren rata",
    "431": "Vaasan rata", "432": "Vaasan rata",     # the article runs on to Vaskiluoto
    "441": "Suupohjan rata",
    "513": "Tornio–Haaparanta-rata",
    "514": "Raahen rata", "515": "Raahen rata", "516": "Raahen rata",
    "527": "Laurila–Kelloselkä-rata",       # Kemijärvi - Patokangas, its far end
    "531": "Oulu–Kontiomäki-rata",
    "552": "Ämmänsaaren rata", "555": "Ämmänsaaren rata",
    "553": "Otanmäen rata",
    "554": "Vartiuksen rata",
    "558": "Talvivaaran rata",
    "722": "Ilomantsin rata",
    "751": "Niiralan rata",
    # Yard ids that carry the passenger trains into a station: Oulu_V330 - Oulu asema, where
    # the Kajaani trains leave the Oulu–Tornio-rata, and Seinäjoki asema - Seinäjoki_V603, the
    # Haapamäki line's last 2.8 km. fi.wikipedia's lengths include both (166 and 117.8 km).
    "OL202": "Oulu–Kontiomäki-rata",
    "SK804": "Haapamäki–Seinäjoki-rata",
}

# Track numbers holding several named lines in a row: (first station, its line), then each
# station where the next line begins. Station names are RINF's opName.
CUT = {
    "003": [("Helsinki asema", "Päärata"), ("Tampere asema", "Tampere–Seinäjoki-rata")],
    "005": [("Kouvola asema", "Savon rata"), ("Iisalmi", "Iisalmi–Kontiomäki-rata")],
    # Nurmes - Kontiomäki is the Joensuu–Kontiomäki-rata's far end, freight only.
    "006": [("Riihimäki asema", "Riihimäki–Lahti-rata"), ("Lahti", "Lahti–Kouvola-rata"),
            ("Kouvola asema", "Karjalan rata"), ("Joensuu asema", "Joensuu–Kontiomäki-rata"),
            ("Nurmes", "Nurmes–Kontiomäki")],
    "008": [("Seinäjoki asema", "Pohjanmaan rata"), ("Oulu asema", "Oulu–Tornio-rata"),
            ("Laurila", "Laurila–Kelloselkä-rata")],
    "009": [("Tampere asema", "Tampere–Haapamäki-rata"), ("Orivesi", "Orivesi–Jyväskylä-rata")],
    "023": [("Pieksämäki asema", "Jyväskylä–Pieksämäki-rata"),
            ("Jyväskylä", "Haapamäki–Jyväskylä-rata")],
    "066": [("Orivesi", "Tampere–Haapamäki-rata"), ("Haapamäki", "Haapamäki–Seinäjoki-rata")],
    # Laurila - Tornio is the north end of the Oulu–Tornio-rata; the Kolarin rata starts at
    # Tornio (RINF's point on this track number is Tornio-Itäinen).
    "521": [("Laurila", "Oulu–Tornio-rata"), ("Tornio-Itäinen", "Kolarin rata")],
    "731": [("Sysmäjärvi", "Siilinjärvi–Viinijärvi-rata"), ("Viinijärvi", "Pieksämäki–Joensuu-rata")],
    # Kerava (Kytömaa) - Olli is the Porvoon rata; Olli - Sköldvik, the branch to Neste's
    # refinery, is freight only (FREIGHT), and would otherwise have come back with the line.
    "131": [("Kytömaa", "Porvoon rata"), ("Olli", "Olli–Sköldvik")],
}

# No passenger train, though both ends are passenger stations (so build_model would keep them):
# fi.wikipedia ("ainoastaan tavaraliikennettä") and Fintraffic's timetable for 2026-09-30.
# Porvoon rata (Kerava - Porvoo, no VR train since 1981, museum trains on summer days) was
# here until 2026-10-01, when Anita decided to keep it. Fintraffic's timetable has no train on
# it, so gtfs_served.py draws it as not running; the refinery branch Olli - Sköldvik stays out.
FREIGHT = {"Siilinjärvi–Viinijärvi-rata", "Jyväskylä–Haapajärvi-rata", "Nurmes–Kontiomäki",
           "Olli–Sköldvik"}

NAMES = set(WHOLE.values()) | {n for v in CUT.values() for _s, n in v}

SIDING_TYPES = {"140", "50"}
SIDING = "(siding)"


def _cut(secs, points):
    """Section IRI -> named line, for the track numbers in CUT (docstring)."""
    out = {}
    by = defaultdict(list)
    for s in secs:
        if s["base"] in CUT:
            by[s["base"]].append(s)
    for lid, ss in by.items():
        adj = defaultdict(list)
        for s in ss:
            adj[s["a"]].append((s["b"], s["km"] or 0.0))
            adj[s["b"]].append((s["a"], s["km"] or 0.0))
        ops_of = defaultdict(set)
        for op in adj:
            ops_of[points.get(op, {}).get("name")].add(op)
        spec = CUT[lid]
        missing = [st for st, _n in spec if not ops_of.get(st)]
        if missing:
            raise SystemExit(f"rinf_countries/fi.py: {lid} has no point named {missing}")
        dist = {op: 0.0 for op in ops_of[spec[0][0]]}
        heap = [(0.0, op) for op in dist]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, km in adj[u]:
                if d + km < dist.get(v, float("inf")):
                    dist[v] = d + km
                    heapq.heappush(heap, (d + km, v))
        bounds = sorted((min(dist.get(op, float("inf")) for op in ops_of[st]), n)
                        for st, n in spec)
        for s in ss:
            d = min(dist.get(s["a"], float("inf")), dist.get(s["b"], float("inf")))
            pick = bounds[0][1]
            for b, n in bounds:
                if b <= d + 1e-9:
                    pick = n
            out[s["sol"]] = pick
    return out


def fi_fix(secs, points):
    """File each section under its named line (docstring, LINES). Pieces of one name that do not
    touch are numbered "<name>#k", as rinf.py does for an id in several pieces."""
    cut = _cut(secs, points)
    n_siding = 0
    for s in secs:
        if any(points.get(op, {}).get("type") in SIDING_TYPES for op in (s["a"], s["b"])):
            name = SIDING
            n_siding += 1
        else:
            name = WHOLE.get(s["base"]) or cut.get(s["sol"])
        if name:
            s["line"] = s["base"] = name
    by = defaultdict(list)
    for s in secs:
        if s["base"] in NAMES:
            by[s["base"]].append(s)
    n_split = 0
    for name, ss in by.items():
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
        order = sorted(comps.values(), key=lambda c: -sum(s["km"] or 0 for s in c))
        for k, c in enumerate(order):
            for s in c:
                s["line"] = name if k == 0 else f"{name}#{k + 1}"
    return [f"{sum(1 for s in secs if s['base'] in NAMES)} sections on "
            f"{len({s['base'] for s in secs if s['base'] in NAMES})} named lines, "
            f"{n_siding} siding leads set aside; {n_split} named lines in unconnected pieces"]


def fi_id_name(lid, _uop=None):
    return lid if lid in NAMES else None


def fi_skip(lid):
    """Everything but the named lines with passenger trains: siding leads, yard and depot
    tracks (KV225, PM805, ILR720...), industrial branches (246 Lappeenranta - Metsä Wood, 517
    Kemi - Ajos), and FREIGHT. Left in, the few that an OSM route happens to run beside came
    out as register lines named "Kouvola tavara - Kouvola_V644"."""
    return lid not in NAMES or lid in FREIGHT


# Fintraffic's station list (https://rata.digitraffic.fi/api/v1/metadata/stations, open, no
# key), saved as digitraffic_stations.json beside RINF's files. Its passenger flag says which
# points are stops: RINF types Henna, Hillosensalmi, Kempele, Nikkilä and Purola as junctions
# and Härmä as a freight point though trains call at all six, and Ilola as a stop though none
# does. Its name is VR's, in Finnish, with "asema" ("station") on the big ones: "Tammisaari" and
# "Karjaa" where OSM has the Swedish "Ekenäs" and "Karis", "Pännäinen" where OSM has
# "Jakobstad-Pedersöre", "Kotkan satama" where OSM has "Kotka satama".
_DT = None


def fi_stop_name(point):
    global _DT
    if _DT is None:
        f = _RAW / "digitraffic_stations.json"
        rows = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        _DT = {r["stationUICCode"]: r for r in rows if r.get("countryCode") == "FI"}
    m = re.fullmatch(r"FI(\d{5})", point.get("uopid") or "")
    r = _DT.get(int(m.group(1))) if m else None
    if r is None:
        return None
    if not r.get("passengerTraffic"):
        return False
    return re.sub(r"\s+asema$", "", r["stationName"])


COUNTRY = {
    "iso3": "FIN", "wikidata": None, "langs": ["fi"],
    "fix": fi_fix, "id_name": fi_id_name, "skip_line": fi_skip, "stop_name": fi_stop_name,
    "osm_rel": lambda _tags: None,
    # RINF places Uusikylä 1.2 km east of the station (Fintraffic and OSM agree on where it is).
    "name_m": 1500,
    "im": {"3109_IM": "Väylävirasto", "5590_IM": "Raideinfra"},
}
