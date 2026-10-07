"""Mexico: register lines are OpenStreetMap's named passenger track, kr_register's recipe.

    python mx_register.py --fetch      # Wikidata's Mexican railway stations -> data/raw/mx
    python mx_register.py --names      # the track names the extract gives, with km (needs data/proc/mx)
    python build_model.py --region mx --register mx_register:data/raw/mx

WHAT MEXICO HAS. Almost all of Mexico's 20,000-odd km of railway is freight (Ferromex, CPKC,
Ferrosur). Passenger trains run on a few lines only, and those are what this register holds,
as the US build keeps only NARN's passenger-coded track:

  - Tren Maya (Palenque - Escárcega - Mérida - Cancún - Tulum - Chetumal - Escárcega, 1,554 km)
  - El Insurgente, the Mexico City - Toluca interurban (Observatorio - Zinacantepec, 57.7 km)
  - Tren Suburbano (Buenavista - Cuautitlán) with its 2026 branch from Lechería to the AIFA
    airport (one line: the branch was built as the Suburbano's extension and its trains run
    from Buenavista)
  - the Chihuahua al Pacífico, El Chepe's line (Chihuahua - Creel - Los Mochis), only the part
    El Chepe runs over, not Los Mochis - Topolobampo or the freight branches of Línea Q
  - the Tren Interoceánico's Líneas Z (Coatzacoalcos - Salina Cruz), FA (Coatzacoalcos -
    Palenque) and K (Ixtepec - Tonalá only, EXTENT): passenger trains suspended since the
    derailment of 28 December 2025, so all three are `suspended` (greyed, out of
    completion), kept for riders who went before
  - the metros and light rail (urban_line: Mexico City's Metro and Tren Ligero, Monterrey's
    Metrorrey, Guadalajara's Mi Tren), by their track names within each city's box. Unlike the
    US and UK builds, which leave metros to OSM's routes: OSM's routes are broken in two of the
    three cities (Guadalajara's way members have the role "route", which build_model reads as
    no track; Metrorrey's list one direction's stops out of order).
The airport Aerotrén stays an OSM line. mx_sources.md has what runs, what does not, and why.

WHY OSM TRACK. FRA's NARN covers Mexico (COUNTRY='MX', 20,158 km) but codes no Mexican segment as
passenger and has none of the Tren Maya, El Insurgente or the Suburbano. ARTF's own Red
Ferroviaria Nacional (datos.gob.mx, CC BY 4.0) has the network, but its file host answers 403 to
a script. OSM names 99.4% of Mexico's main-line track for its line (`python probe_kr_ways.py
--region mx`): the ARTF's line letters (Línea A, B, Q, Z...) on the freight network, and the
new lines' own names ("Tren Maya", "El Insurgente", "Ferrocarril Suburbano Ramal 1",
"Ferrocarril Felipe Ángeles"). So each register line is the graph of the OSM ways carrying its
name (TRACKS), or, for El Chepe, the ways of OSM's route relation for it (CHEPE_ROUTE): Línea Q
runs on from Los Mochis to Topolobampo and branches off to Ojinaga, with no passenger train.

STATIONS. kr_register's own rule (a stop node on the line's track) plus a list per line:
the stops OSM's route relations list, on the line their own track runs on (route_lists,
gb_register's rule, reading way members whatever their role); for intercity lines also the
OSM rail station records near the track (LIST_M: the Chepe's and Línea FA's stations are
railway=station nodes beside it) and Wikidata's station items (`--fetch`,
data/raw/mx/wikidata_stations.json) where OSM has no rail record (Chihuahua, Creel and
Divisadero on the Chepe, the AIFA branch's six new stops, Línea K), each only within WD_M of
the line's track and with no OSM station near it or of its name within WD_DUP_M. Station
names lose a leading "Estación".

JUNCTIONS. kr_register's search knows no direction, so at a triangle or a branch between two
stations it pairs stations past the junction too; drop_junction_runs takes those out, then
gb_register.drop_shortcuts.

The `path` argument is data/raw/mx; the OSM half is read from data/proc/mx (extract.py).
"""
import hashlib
import json
import math
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import kr_register as kr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "mx"
REGION = "mx"
USER_AGENT = "noritetsu-build/1.0 (hobby rail map)"

# The Tren Interoceánico's three lines are the Ferrocarril del Istmo de Tehuantepec's Líneas Z,
# FA and K. Named so build_model.norm_line_name (which drops the bracket) reads all three as
# "Ferrocarril del Istmo de Tehuantepec", OSM's name for its Línea Z passenger route: that
# route is then Línea Z's twin and is dropped, rather than staying as a running line over the
# suspended one.
FIT_Z = "Ferrocarril del Istmo de Tehuantepec (Línea Z)"
FIT_FA = "Ferrocarril del Istmo de Tehuantepec (Línea FA)"
FIT_K = "Ferrocarril del Istmo de Tehuantepec (Línea K)"

# OSM track name -> register line. Exact names as OSM has them (2026-10 extract).
TRACKS = {
    "Tren Maya": "Tren Maya",
    "El Insurgente": "El Insurgente",
    "Ferrocarril Suburbano Ramal 1": "Tren Suburbano",
    "Ferrocarril Felipe Ángeles": "Tren Suburbano",
    "Línea Z": FIT_Z,
    "Línea FA": FIT_FA,
    "Línea K": FIT_K,
}
# The metros and light rail: their track names, by city (bbox lon0, lat0, lon1, lat1), read
# with accents folded ("Linea 6" is Línea 6). OSM names Mexico City's Metro track as the line
# ("Línea 1" ... "Línea 12", "Línea A", "Línea B": the A and B of the Metro, not Ferromex's
# Línea A and B, which are railway=rail; only subway and light_rail track is read here).
# Register names: Mexico City's as OSM's route_masters ("Línea 1"), Monterrey's prefixed
# "Metrorrey" (its operator, which build_model.norm_line_name strips, so OSM's "Línea 1" of
# Metrorrey still finds it), Guadalajara's as its route_masters ("Mi Tren Línea 1").
METRO_CDMX = (-99.40, 19.15, -98.85, 19.70)
METRO_MTY = (-100.60, 25.50, -100.00, 25.95)
METRO_GDL = (-103.60, 20.40, -103.20, 20.90)
GDL_TRACK = {"linea 1 del tren electrico urbano": "Mi Tren Línea 1",
             "linea 2 del tren electrico urbano": "Mi Tren Línea 2",
             "linea 3 del tren electrico urbano": "Mi Tren Línea 3",
             "linea 4 del tren electrico urbano de guadalajara": "Mi Tren Línea 4"}
MTY_TRACK = {"linea 1 metrorrey": "Metrorrey Línea 1", "linea 2 metrorrey": "Metrorrey Línea 2",
             "linea 3": "Metrorrey Línea 3"}
CDMX_LIGHT_RAIL = "Tren Ligero de la Ciudad de México"


def urban_line(tags, lon, lat):
    """The register line a metro or light-rail way belongs to, '' for none."""
    if tags.get("railway") not in ("subway", "light_rail") or tags.get("service"):
        return ""
    k = fold(tags.get("name"))

    def inside(b):
        return b[0] <= lon <= b[2] and b[1] <= lat <= b[3]
    if inside(METRO_CDMX):
        if k == fold(CDMX_LIGHT_RAIL):
            return CDMX_LIGHT_RAIL
        m = re.fullmatch(r"linea (\d{1,2}|a|b)", k)
        return f"Línea {m.group(1).upper()}" if m else ""
    if inside(METRO_MTY):
        return MTY_TRACK.get(k, "")
    if inside(METRO_GDL):
        return GDL_TRACK.get(k, "")
    return ""


# El Chepe's line is the track of OSM's route relation "Chihuahua al Pacífico" (Ferromex),
# whatever each way is called (Línea Q, its bridges and tunnels, yard track at the ends).
CHEPE_ROUTE = 5977762
CHEPE_LINE = "Chihuahua al Pacífico"

# What each register line is called in English, who runs it, and whether it runs.
LINE_INFO = {
    "Tren Maya": {"name_en": "Maya Train", "operator": "Tren Maya",
                  "network": "Tren Maya"},
    "El Insurgente": {"name_en": "El Insurgente (Mexico City - Toluca)",
                      "operator": "Tren Interurbano México-Toluca",
                      "network": "Tren Interurbano México-Toluca"},
    "Tren Suburbano": {"name_en": "Suburban Railway (Buenavista - Cuautitlán, Lechería - AIFA)",
                       "operator": "Ferrocarriles Suburbanos", "network": "Tren Suburbano"},
    CHEPE_LINE: {"name_en": "Chihuahua-Pacific Railway (El Chepe)", "operator": "Ferromex",
                 "network": "El Chepe"},
    # Tren Interoceánico: no passenger train since the derailment at Nizanda, 28 Dec 2025;
    # the government says early 2027 at the soonest (mx_sources.md).
    FIT_Z: {"name_en": "Interoceanic Train Line Z", "ref": "Z", "suspended": True},
    FIT_FA: {"name_en": "Interoceanic Train Line FA", "ref": "FA", "suspended": True},
    FIT_K: {"name_en": "Interoceanic Train Line K", "ref": "K", "suspended": True},
}
for _n in (FIT_Z, FIT_FA, FIT_K):
    # Not OSM's "Ferroistmo": build_model.same_operator would then match OSM's Línea Z route
    # to whichever of the three comes first by name; with another spelling it is matched on
    # the stations it shares, which only Línea Z's are.
    LINE_INFO[_n].update(operator="Ferrocarril del Istmo de Tehuantepec",
                         network="Tren Interoceánico")
for _x in [str(i) for i in range(1, 10)] + ["12", "A", "B"]:
    LINE_INFO[f"Línea {_x}"] = {"name_en": f"Mexico City Metro Line {_x}",
                                "operator": "Sistema de Transporte Colectivo (Metro)",
                                "network": "STC Metro", "ref": _x, "urban": True}
LINE_INFO[CDMX_LIGHT_RAIL] = {"name_en": "Mexico City Light Rail (Tasqueña - Xochimilco)",
                              "operator": "Servicio de Transportes Eléctricos",
                              "network": "STE Tren Ligero", "urban": True}
for _x in ("1", "2", "3"):
    LINE_INFO[f"Metrorrey Línea {_x}"] = {"name_en": f"Monterrey Metro Line {_x}",
                                          "operator": "Metrorrey", "network": "Metrorrey",
                                          "ref": _x, "urban": True}
# Guadalajara's colours from its OSM route relations (OSM's own lines 1-3 do not build: their
# way members carry the role "route", which build_model.assemble reads as no track).
for _x, _c in (("1", "#c5112c"), ("2", "#27be31"), ("3", "#dd0e98"), ("4", "#f28c28")):
    LINE_INFO[f"Mi Tren Línea {_x}"] = {"name_en": f"Guadalajara Light Rail Line {_x}",
                                        "operator": "SITEUR", "network": "Mi Tren",
                                        "ref": f"TL-{_x}", "colour": _c, "urban": True}

# Lines whose passenger trains ran over only part of the named track: the part between these
# two points (lon, lat), along the track. Línea K runs on to Ciudad Hidalgo on the Guatemalan
# border; passenger trains (21 Nov - 28 Dec 2025) ran Ixtepec - Tonalá only.
EXTENT = {
    FIT_K: ((-95.0942, 16.55468), (-93.758075, 16.086247)),
}
# Wikidata's "state of use" (P5817) values a station may have; none is fine too. Left out:
# disused (Q11639308), under construction (Q12377751), planned (Q811683) and the rest.
WD_IN_USE = {"Q55654238"}
# railway station, railway stop (halt)
WD_CLASSES = {"Q55488", "Q55678"}

TRACK_KIND = {"rail": "rail", "narrow_gauge": "narrow_gauge", "subway": "subway",
              "light_rail": "light_rail"}
NOT_PASSENGER = {"industrial", "military", "test", "tourism"}

# A station record this close to a line's track is one of its stations, if it is a rail record
# (intercity lines only; a metro's stations come from its routes, ALONG_M).
LIST_M = 250
ALONG_M = 250
# Stop records of other systems that lie beside the lines: the Metro's Lechería and Buenavista,
# Campeche's light rail at the Tren Maya's Campeche station, Metrobús.
OTHER_NETWORKS = re.compile(r"STC Metro|Tren ligero de Campeche|Metrob[uú]s", re.IGNORECASE)
# Wikidata stations: this close to the line's track; left out where an OSM rail record lies
# within WD_NEAR_M, or one of the same name within WD_DUP_M (Wikidata puts some Tren Maya
# stations on their town: Xpujil 5.6 km, Nicolás Bravo 12 km from the station).
WD_M = 600
WD_NEAR_M = 300
WD_DUP_M = 25000
# Stations a line's list names although their record is further than LIST_M from its track.
# Ixtepec: Línea K leaves Línea Z's track there; the station's stop node is on Z's.
EXTRA_LISTS = {FIT_K: ["Estación Ixtepec"]}
# A section whose track is, for COVER_SHARE of its length, within COVER_M of two other sections
# of the line that meet at a third station is a run past that station's junction, not a
# section (mx_sources.md: the Lechería junction, the Cancún airport triangle).
COVER_M = 30
COVER_SHARE = 0.9
# Fake node ids for Wikidata stations (never an OSM id; negative so kr's int(sid[1:]) reads them).
WD_BASE = -8_000_000_000


def fold(s):
    """Accents and case off, "Estación" off: 'Estación Tixkokob' -> 'tixkokob'."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    s = re.sub(r"^estacion\s+(?:de\s+)?", "", s.strip())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def similar(a, b):
    """Two folded names for one station: equal, or one letter or so apart ("nicolas bravo
    kohunlich" / "nicolas bravo konhunlich")."""
    if not a or not b:
        return False
    if a == b:
        return True
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio() >= 0.9


def plain_station(name):
    """A station's name as shown: without a leading 'Estación (de)'."""
    n = re.sub(r"^Estaci[oó]n\s+(?:de\s+)?", "", (name or "").strip())
    return n[:1].upper() + n[1:] if n else (name or "")


def line_id(name):
    h = hashlib.blake2b(f"mx|{name}".encode("utf-8"), digest_size=5)
    return "x" + h.hexdigest()


_S = {}


def register_name(tags):
    return tags.get("_mx_line", "")


def load_osm(log):
    import build_model as bm
    ways, rels, stops, cid, cx, cy = bm.load(REGION, log)
    coords = bm.Coords(cid, cx, cy)
    chepe = set()
    if CHEPE_ROUTE in rels:
        chepe = {r for ty, r, _ in rels[CHEPE_ROUTE][1] if ty == "w" and r in ways}
    else:
        log(f"MX: OSM route relation {CHEPE_ROUTE} (El Chepe) is not in the extract")
    km = Counter()
    for wid, (tags, nodes) in ways.items():
        if tags.get("railway") not in TRACK_KIND or tags.get("usage") in NOT_PASSENGER:
            continue
        if tags["railway"] in ("subway", "light_rail"):
            p = coords.get(int(nodes[0]))
            line = urban_line(tags, *p) if p else ""
        else:
            line = (CHEPE_LINE if wid in chepe
                    else TRACKS.get((tags.get("name") or "").strip(), ""))
        if line:
            # the way's tag dict is this process's own copy (build_model.load unpickles it)
            tags["_mx_line"] = line
            km[line] += 1
    for line, (a, b) in EXTENT.items():
        wids = [w for w, (t, _n) in ways.items() if t.get("_mx_line") == line]
        adj, xy, _fast = kr.line_graph(wids, ways, coords)
        if len(xy) < 2:
            continue
        nr = kr.Near(xy)
        va, da = nr.nearest(*a)
        vb, db = nr.nearest(*b)
        got = kr.between(adj, {va: 0.0}, {vb: 0.0}, set())
        if got is None or da > 2000 or db > 2000:
            log(f"MX: {line}: no track between its extent's ends ({da:.0f} m, {db:.0f} m off); "
                f"left out")
            for w in wids:
                ways[w][0].pop("_mx_line", None)
            continue
        on = set(got[0])
        cut = 0
        for w in wids:
            nodes = np.asarray(ways[w][1]).tolist()
            if sum(1 for n in nodes if n in on) < max(2, len(nodes) // 2):
                ways[w][0].pop("_mx_line", None)
                cut += 1
        log(f"MX: {line}: {got[1]:.1f} km between its extent's ends; {cut} of {len(wids)} "
            f"ways beyond it left out")
    log("MX: register track ways " + ", ".join(f"{k} {v}" for k, v in sorted(km.items())))
    _S.update(ways=ways, rels=rels, stops=stops, coords=coords)
    return ways, stops, coords


def load_wikidata(log):
    p = RAW / "wikidata_stations.json"
    if not p.exists():
        log(f"MX: {p} not there; no Wikidata stations (python mx_register.py --fetch)")
        return []
    rows = json.loads(p.read_text(encoding="utf-8"))["results"]["bindings"]
    out = {}
    for r in rows:
        q = r["item"]["value"].rsplit("/", 1)[-1]
        if "closed" in r or "label" not in r:
            continue
        # railway stations and halts only: no metro, light-rail or tram stops, station
        # buildings or former stations (the query fetches those too)
        if r["class"]["value"].rsplit("/", 1)[-1] not in WD_CLASSES:
            continue
        state = r.get("state", {}).get("value", "").rsplit("/", 1)[-1]
        if state and state not in WD_IN_USE:
            continue
        m = re.match(r"Point\(([-\d.]+) ([-\d.]+)\)", r["coord"]["value"])
        if not m:
            continue
        # Wikidata's disambiguation brackets are no part of the name: "Estación de Candelaria
        # (Tren Maya)", "Estación de San Rafael (Chihuahua)".
        name = re.sub(r"\s*\([^)]*\)\s*$", "", r["label"]["value"]).strip()
        out[q] = {"q": q, "name": name, "lon": float(m.group(1)), "lat": float(m.group(2))}
    return list(out.values())


def line_tracks():
    """{line: Near over its track vertices}."""
    ways, coords = _S["ways"], _S["coords"]
    by_line = defaultdict(list)
    for wid, (tags, _n) in ways.items():
        ln = register_name(tags)
        if ln:
            by_line[ln].append(wid)
    out = {}
    for ln, wids in by_line.items():
        _adj, xy, _fast = kr.line_graph(wids, ways, coords)
        if len(xy) >= 2:
            out[ln] = kr.Near(xy)
    return out


def rail_record(tags):
    """Is this stop record a train station or stop (not a bus terminal, not the Metro's)?"""
    if OTHER_NETWORKS.search(tags.get("network") or ""):
        return False
    if tags.get("subway") == "yes" or tags.get("light_rail") == "yes":
        return tags.get("train") == "yes"
    return (tags.get("railway") in ("station", "halt", "stop") or tags.get("train") == "yes")


def build_stations(stops, log):
    st, node_st, by_key, by_base = _orig["build_stations"](stops, log)
    tracks = line_tracks()
    _S["tracks"] = tracks
    # Wikidata stations OSM has no rail record of, near a register line's track
    wd = load_wikidata(log)
    # Against every OSM station record, the Metro's too: Observatorio and Lechería are one
    # complex in build_stations, under the Metro station's record.
    osm_by_fold = defaultdict(list)
    rail_ids = []
    for nid, s in st.items():
        osm_by_fold[fold(s["name"])].append(nid)
        rail_ids.append(nid)
    rx = np.array([st[n]["lon"] for n in rail_ids])
    ry = np.array([st[n]["lat"] for n in rail_ids])
    added = []
    for i, w in enumerate(sorted(wd, key=lambda w: w["q"])):
        near = None
        for ln, nr in tracks.items():
            if LINE_INFO.get(ln, {}).get("urban"):
                continue
            if not (nr.lon.min() - 0.02 <= w["lon"] <= nr.lon.max() + 0.02
                    and nr.lat.min() - 0.02 <= w["lat"] <= nr.lat.max() + 0.02):
                continue
            _v, d = nr.nearest(w["lon"], w["lat"])
            if d <= WD_M:
                near = ln
                break
        if near is None:
            continue
        fw = fold(w["name"])
        if any(similar(fw, k) and kr.dist_m(w["lon"], w["lat"], st[o]["lon"], st[o]["lat"])
               <= WD_DUP_M for k, os_ in osm_by_fold.items() for o in os_):
            continue
        d = np.hypot((rx - w["lon"]) * math.cos(math.radians(w["lat"])) * 111320,
                     (ry - w["lat"]) * 110570)
        if d.size and d.min() <= WD_NEAR_M:
            continue
        fid = WD_BASE - int(w["q"][1:])
        st[fid] = {"name": w["name"], "name_en": "", "lon": w["lon"], "lat": w["lat"], "rank": 1}
        by_key[kr.name_key(w["name"])].append(fid)
        by_base[kr.base_key(w["name"])].append(fid)
        added.append((near, w["name"], w["q"]))
    _S["wd"] = {WD_BASE - int(q[1:]): q for _ln, _n, q in added}
    _S.update(st=st, node_st=node_st)
    log(f"MX: {len(added)} Wikidata stations added where OSM has no rail record: "
        + "; ".join(f"{n} ({q}, {ln})" for ln, n, q in added))
    return st, node_st, by_key, by_base


def route_lists(log):
    """{line: {station name}}: the stations OSM's route relations stop at, on each register line
    whose track the route's own ways run on within ALONG_M of the stop (gb_register's rule).
    Way members are read whatever their role: Guadalajara's routes give theirs the role
    "route", which is why OSM's own Mi Tren lines 1-3 do not build."""
    import build_model as bm
    ways, rels, coords = _S["ways"], _S["rels"], _S["coords"]
    st, node_st = _S["st"], _S["node_st"]
    wxy = {}

    def xy(w):
        if w not in wxy:
            nodes = np.asarray(ways[w][1], dtype=np.int64)
            pos, ok = coords.many(nodes)
            pos = pos[ok]
            wxy[w] = (coords.x[pos] / 1e7, coords.y[pos] / 1e7)
        return wxy[w]

    lists = defaultdict(set)
    n_routes = 0
    for _rid, (tags, members) in rels.items():
        if tags.get("type") != "route" or tags.get("route") not in bm.ROUTE_KINDS:
            continue
        rw = [r for ty, r, _ in members if ty == "w" and r in ways
              and register_name(ways[r][0])]
        if not rw:
            continue
        n_routes += 1
        for n in bm.stop_members(members):
            s = node_st.get(n)
            if s is None:
                continue
            lon, lat = st[s]["lon"], st[s]["lat"]
            kx = math.cos(math.radians(lat)) * 111320
            for w in rw:
                x, y = xy(w)
                if x.size and np.min(np.hypot((x - lon) * kx, (y - lat) * 110570)) <= ALONG_M:
                    lists[register_name(ways[w][0])].add(st[s]["name"])
    log(f"MX: station lists from {n_routes} OSM routes on register track, "
        f"{sum(len(v) for v in lists.values())} station-line pairs")
    return lists


def load_lists(path, log):
    """{line: [[station name, ...]]}: the stations OSM's routes stop at (route_lists); for the
    intercity lines also the rail station records within LIST_M of the line's track and the
    Wikidata stations taken for it (most have no route with stops). kr_register then finds
    each listed name's record nearest the line's track (MATCH_M) as it does Korea's lists."""
    st, stops, tracks = _S["st"], _S["stops"], _S["tracks"]
    lists = route_lists(log)
    for nid, s in st.items():
        if nid in _S["wd"]:
            ok = True
        else:
            tags = stops.get(nid, ({},))[0]
            ok = rail_record(tags)
        if not ok:
            continue
        for ln, nr in tracks.items():
            if LINE_INFO.get(ln, {}).get("urban"):
                continue
            if not (nr.lon.min() - 0.01 <= s["lon"] <= nr.lon.max() + 0.01
                    and nr.lat.min() - 0.01 <= s["lat"] <= nr.lat.max() + 0.01):
                continue
            _v, d = nr.nearest(s["lon"], s["lat"])
            if d <= (WD_M if nid in _S["wd"] else LIST_M):
                lists[ln].add(s["name"])
    for ln, names in EXTRA_LISTS.items():
        lists[ln].update(names)
    for ln in sorted(lists):
        log(f"MX: {ln}: {len(lists[ln])} stations listed")
    return {k: [sorted(v)] for k, v in lists.items()}, defaultdict(list), {}


_orig = {}


def adopt():
    """Point kr_register's country-specific globals at Mexico's, for this process."""
    if _orig:
        return
    _orig["build_stations"] = kr.build_stations
    kr.load_osm = load_osm
    kr.register_name = register_name
    kr.load_lists = load_lists
    kr.build_stations = build_stations
    kr.line_id = line_id
    kr.TRACK_KIND = TRACK_KIND
    kr.NOT_PASSENGER = NOT_PASSENGER
    kr.NAME_ALIAS = {}
    kr.STATION_ALIAS = {}


def _xy_m(pts, lat0):
    a = np.asarray(pts, dtype=np.float64)
    return np.column_stack([a[:, 0] * math.cos(math.radians(lat0)) * 111320, a[:, 1] * 110570])


def _covered_share(pts, others, lat0):
    """The share of polyline `pts` (sampled every 25 m) within COVER_M of any of `others`."""
    p = _xy_m(pts, lat0)
    seg = np.hypot(*np.diff(p, axis=0).T)
    samples = [p[0]]
    for (a, b), L in zip(zip(p[:-1], p[1:]), seg):
        k = max(1, int(L // 25))
        samples.extend(a + (b - a) * (np.arange(1, k + 1) / k)[:, None])
    s = np.asarray(samples)
    best = np.full(len(s), np.inf)
    for o in others:
        q = _xy_m(o, lat0)
        A, B = q[:-1], q[1:]
        d = B - A
        L2 = np.maximum((d * d).sum(1), 1e-9)
        for i in range(0, len(s), 400):
            ss = s[i:i + 400]
            t = np.clip(((ss[:, None, :] - A[None]) * d[None]).sum(2) / L2[None], 0, 1)
            proj = A[None] + t[..., None] * d[None]
            best[i:i + 400] = np.minimum(best[i:i + 400],
                                         np.hypot(*(ss[:, None, :] - proj).transpose(2, 0, 1)).min(1))
    return float((best <= COVER_M).mean())


def drop_junction_runs(lines, geoms, log):
    """A run past a station's junction: where a branch leaves a line between two stations (the
    AIFA branch north of Lechería, the Cancún airport station on its triangle), kr_register's
    search, which knows no direction, pairs the branch's first station with the stations both
    sides of the junction, and the stations either side with each other round the triangle.
    The section whose track the two others cover (COVER_SHARE within COVER_M) goes, longest
    first; no train runs it without passing the third station's junction."""
    from n02 import walk_order
    n_drop, km_drop, what = 0, 0.0, []
    for l in lines:
        g = geoms[l["id"]]
        secs = sorted(l["sections"], key=lambda s: -s[2])
        keep = list(secs)
        for s in secs:
            a, b = s[0], s[1]
            pts = g.get(f"{a}|{b}")
            if not pts or len(pts) < 2:
                continue
            rest = [x for x in keep if x is not s]
            nb = defaultdict(dict)
            for x in rest:
                nb[x[0]][x[1]] = x
                nb[x[1]][x[0]] = x
            lat0 = pts[0][1]
            for c in set(nb[a]) & set(nb[b]):
                o = [g.get(f"{x[0]}|{x[1]}") for x in (nb[a][c], nb[b][c])]
                if not all(o):
                    continue
                if _covered_share(pts, o, lat0) >= COVER_SHARE:
                    keep = rest
                    n_drop += 1
                    km_drop += s[2]
                    what.append(f"{l['name']} {s[2]:.1f} km")
                    g.pop(f"{a}|{b}", None)
                    if isinstance(l.get("highspeed_sections"), dict):
                        l["highspeed_sections"].pop(f"{a}|{b}", None)
                    break
        if len(keep) != len(l["sections"]):
            ks = {(x[0], x[1]) for x in keep}
            l["sections"] = [x for x in l["sections"] if (x[0], x[1]) in ks]
            l["km"] = round(sum(x[2] for x in l["sections"]), 3)
            l["display"] = walk_order([(x[0], x[1]) for x in l["sections"]])
    log(f"MX: {n_drop} sections dropped as runs past a junction ({km_drop:,.1f} km): "
        + ", ".join(what))


def build(path, log):
    adopt()
    lines, stations, geoms = kr.build(path, log)
    drop_junction_runs(lines, geoms, log)
    import gb_register
    gb_register.drop_shortcuts(lines, geoms, log)   # logs as "GB:"; the rule is the same
    wd = {f"k{fid}": f"xq{q[1:]}" for fid, q in _S.get("wd", {}).items()}

    def r(sid):
        return wd.get(sid) or ("x" + sid[1:] if sid.startswith("k") else sid)

    out_st = {}
    for sid, s in stations.items():
        nid = r(sid)
        s["id"] = nid
        s["name"] = plain_station(s["name"])
        out_st[nid] = s
    out_geoms = {}
    for l in lines:
        info = LINE_INFO.get(l["name"], {})
        l["src"] = "mx"
        for k in ("name_en", "operator", "network", "ref", "colour"):
            if info.get(k):
                l[k] = info[k]
        if info.get("suspended"):
            l["suspended"] = True
        l["sections"] = [[r(a), r(b), *rest] for a, b, *rest in l["sections"]]
        l["display"] = [r(x) for x in l["display"]]
        if isinstance(l.get("highspeed_sections"), dict):
            l["highspeed_sections"] = {"|".join(r(x) for x in k.split("|")): v
                                       for k, v in l["highspeed_sections"].items()}
        out_geoms[l["id"]] = {"|".join(r(x) for x in k.split("|")): v
                              for k, v in geoms[l["id"]].items()}
        log(f"MX: {l['name']}: {l['km']:.1f} km, {len(l['sections'])} sections, "
            f"{len({s for sec in l['sections'] for s in sec[:2]})} stations"
            + (" (suspended)" if l.get("suspended") else ""))
    for s in out_st.values():
        s["lines"] = set()
    for l in lines:
        for a, b, *_ in l["sections"]:
            out_st[a]["lines"].add(l["id"])
            out_st[b]["lines"].add(l["id"])
    out_st = {k: s for k, s in out_st.items() if s["lines"]}
    missing = sorted(set(LINE_INFO) - {l["name"] for l in lines})
    if missing:
        log(f"MX: no line built for {', '.join(missing)}")
    return lines, out_st, out_geoms


# ---------------------------------------------------------------- the small sources

WD = "https://query.wikidata.org/sparql"
# Railway stations (and halts, metro/light-rail stations and the like) in Mexico, with a point.
Q_STATIONS = """SELECT ?item ?label ?coord ?line ?lineLabel ?class ?closed ?state WHERE {
  ?item wdt:P17 wd:Q96 ; wdt:P31 ?class ; wdt:P625 ?coord .
  VALUES ?class { wd:Q55488 wd:Q55678 wd:Q928830 wd:Q1793804 wd:Q4663385 wd:Q2175765
                  wd:Q12819564 wd:Q1339195 wd:Q18543139 }
  OPTIONAL { ?item rdfs:label ?label FILTER(LANG(?label) = "es") }
  OPTIONAL { ?item wdt:P81 ?line . ?line rdfs:label ?lineLabel FILTER(LANG(?lineLabel) = "es") }
  OPTIONAL { ?item wdt:P3999 ?closed }
  OPTIONAL { ?item wdt:P5817 ?state }
}"""


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    url = WD + "?" + urllib.parse.urlencode({"query": Q_STATIONS, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/sparql-results+json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            print(f"Wikidata 429, waiting (attempt {attempt + 1})", flush=True)
            time.sleep(75)
    else:
        sys.exit("Wikidata kept answering 429")
    (RAW / "wikidata_stations.json").write_bytes(data)
    n = len(json.loads(data)["results"]["bindings"])
    print(f"wrote {RAW / 'wikidata_stations.json'}: {n} rows")


def names_report():
    import build_model as bm
    ways, _rels, _stops, cid, cx, cy = bm.load(REGION, print)
    coords = bm.Coords(cid, cx, cy)
    km = Counter()
    for _wid, (tags, nodes) in ways.items():
        if tags.get("railway") not in ("rail", "narrow_gauge") or not tags.get("name"):
            continue
        n = np.asarray(nodes, dtype=np.int64)
        pos, ok = coords.many(n)
        pos = pos[ok]
        if pos.size < 2:
            continue
        x, y = coords.x[pos] / 1e7, coords.y[pos] / 1e7
        km[tags["name"]] += float(np.hypot(np.diff(x) * np.cos(np.radians(y[:-1])) * 111.32,
                                           np.diff(y) * 110.57).sum())
    for name, k in km.most_common():
        print(f"  {k:8.1f}  {name}{'  <- register' if name in TRACKS else ''}")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    elif "--names" in sys.argv:
        names_report()
    else:
        print(__doc__)
