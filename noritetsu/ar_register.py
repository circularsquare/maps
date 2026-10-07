"""Argentina: register lines are the track OSM's passenger route relations run over, grouped
into lines by hand (LINES), read by kr_register's named-track recipe.

    python ar_register.py --clip        # drop Chile, Bolivia, Paraguay, Brazil, Uruguay from data/proc/ar
    python ar_register.py --report      # each LINE's ways, km and pieces, before any build
    python build_model.py --region ar --register ar_register:data/raw/ar
    python ar_register.py --chainage    # built sections against the transport ministry's km posts

WHY NOT NAMED TRACK. OSM names 99.6% of Argentina's main-line track (probe_kr_ways), but for
its NETWORK, not its line: "FC Roca" is 5,137 km, from Constitución to Bariloche and Bahía
Blanca, nearly all of it freight or closed. The ministry's network file (ADIF's, 2022:
data/raw/ar/red_adifse_22.geojson) cuts it into ramales by code ("A", "C14", "10") with an
"Activo" flag that means open to any train, not to passengers, and no names. What OSM does
have is a route relation for every passenger service (Trenes Argentinos' seven Buenos Aires
lines, the long-distance trains, the regional ones), so a register line here is the union of
the ways of the routes listed for it in LINES, each way going to the FIRST line that lists a
route over it: the Buenos Aires lines come first, so the Mar del Plata train's line owns only
the track past the end of the Roca's commuter services.

THE LINES. Each Buenos Aires commuter line ("Línea Roca", "Línea Mitre"...) is one register
line with all its branches, as Trenes Argentinos and every rider treat it, and is named as
OSM's route_master so that the OSM line is its twin and is dropped. Track that only the
long-distance trains run over (they are named trains, rules/ar.py) is a register line per
network and stretch, named "<railway>: <end> – <end>": its track counts as the US build's
NARN passenger subdivisions do. The regional lines (Tren de las Sierras, Tren del Valle,
Salta, the Chaco lines, Mendoza's Metrotranvía...) are one line each. The Subte and Premetro
stay OSM lines (their route relations are clean, as the US and UK builds leave metros).
ar_sources.md has what runs, what does not, and each call.

STATIONS. A station is a stop of some OSM route (its stop node on the line's track, or listed
by a route whose own track runs past it, mx_register.route_lists' rule). Station records
alone never make a station: Argentina maps hundreds of closed stations as railway=station on
or beside track that a train still runs over (the Mar del Plata line passes ~40).

The `path` argument is data/raw/ar; the OSM half is read from data/proc/ar (extract.py).
cl_register.py reads Chile with this same code and its own LINES.
"""
import hashlib
import json
import math
import os
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import kr_register as kr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
SHAPES = ROOT.parent / "religiondots" / "data" / "processed" / "country_shapes.geojson"

TRACK_KIND = {"rail": "rail", "narrow_gauge": "narrow_gauge", "subway": "subway",
              "light_rail": "light_rail", "tram": "tram", "preserved": "rail",
              "monorail": "monorail", "funicular": "funicular"}
# A route's stop is put on a register line whose own track the route runs over within this.
ALONG_M = 250
# Gaps in a line's route ways (a route relation missing a way or two) are bridged over other
# rail ways by the shortest path between the pieces, if it is no longer than this.
BRIDGE_KM = 3.0
# A piece of a line's ways still apart after bridging, and shorter than this, is left out.
PIECE_KEEP_KM = 10.0
# Station records of a metro system (kept out of the register's stations).
METRO_NETWORK = re.compile(r"Subte|Premetro|Metro de Santiago|Metro S\.A\.", re.IGNORECASE)
# Fake node ids for CFG["extra_stations"] (never an OSM id).
EXTRA_BASE = -7_000_000_000
# A bracketed line or network after a station's name is dropped from the name shown:
# "Zárate (Mitre)", "Haedo (Sarmiento)", "Retiro (San Martín)".
LINE_QUALIFIER = re.compile(r"\s*\((?:Mitre|San Mart[ií]n|Sarmiento|Belgrano(?: Sur| Norte)?|"
                            r"Roca|Urquiza|FC\w*|FFCC\w*|EFE|Merval|Biotren)\)\s*$")

TA = "Trenes Argentinos"
TA_NET = "Trenes Argentinos Operaciones"


def L(name, routes=(), info=None, extent=None, lists=None):
    """A register line: `routes` are OSM route relation ids whose ways it takes (those not
    taken by an earlier line); `extent` is [(track name regex, [(lon, lat), ...])], track of
    that name along the shortest path through the points, for a stretch no OSM route maps;
    `lists` are extra station names for it (a stretch with no route has no stop lists)."""
    return {"name": name, "routes": list(routes), "info": dict(info or {}),
            "extent": extent or [], "lists": list(lists or [])}


# ---------------------------------------------------------------- Argentina's lines
# Route ids from the 2026-10-03 extract (ar_sources.md lists each with its stops).
LINES = [
    # -- Buenos Aires: Trenes Argentinos' and the private operators' seven lines
    L("Línea Mitre", [4572178, 4572179, 2790166, 129415, 2953074, 129414,
                      3747352, 3747353, 130982, 7556572],
      {"name_en": "Mitre Line", "operator": TA, "network": "Trenes Argentinos", "ref": "LM"}),
    L("Tren de la Costa", [129404, 6699648],
      {"name_en": "Tren de la Costa", "operator": TA, "network": "Trenes Argentinos",
       "ref": "TC"}),
    L("Línea Sarmiento", [1878866, 1878867, 223925, 6580287, 1289919, 223935],
      {"name_en": "Sarmiento Line", "operator": TA, "network": "Trenes Argentinos", "ref": "LS"}),
    L("Línea Roca", [3738002, 3738004, 3737975, 3737976, 3737973, 3737974, 3842635, 3739886,
                     129698, 3746860, 2953252, 129486, 3739884, 176910, 3739883, 3739885,
                     7675700, 7675703, 3846823, 3739882, 2818891, 2897277],
      {"name_en": "Roca Line", "operator": TA, "network": "Trenes Argentinos", "ref": "LR"},
      # Cañuelas - Lobos: Roca trains run it (2026 timetables), OSM has no route for it.
      extent=[(r"^FC Roca$", [(-58.7545, -35.0593), (-59.0929, -35.1847)])],
      lists=["Cañuelas", "Uribelarrea", "Lobos"]),
    L("Línea San Martín", [1894775, 1894770],
      {"name_en": "San Martín Line", "operator": TA, "network": "Trenes Argentinos",
       "ref": "LSM"}),
    L("Línea Belgrano Sur", [3443503, 233242, 2978253, 129504, 16419625, 10353270],
      {"name_en": "Belgrano Sur Line", "operator": TA, "network": "Trenes Argentinos",
       "ref": "LBS"}),
    L("Línea Belgrano Norte", [129384, 3155061],
      {"name_en": "Belgrano Norte Line", "operator": "Ferrovías", "network": "Ferrovías",
       "ref": "LBN"}),
    L("Línea Urquiza", [1889578, 1889579],
      {"name_en": "Urquiza Line", "operator": "Metrovías (Emova)", "network": "Urquiza", "ref": "LU"}),
    # -- track only the long-distance (named) trains run over, past the commuter lines' ends
    L("Ferrocarril Roca: Chascomús – Mar del Plata", [240129, 3746861, 9095328, 9095327],
      {"name_en": "Roca Railway: Chascomús – Mar del Plata", "operator": TA,
       "network": "Trenes Argentinos Larga Distancia"}),
    L("Ferrocarril Mitre: Zárate – Rosario", [7635630, 7635626],
      {"name_en": "Mitre Railway: Zárate – Rosario", "operator": TA,
       "network": "Trenes Argentinos Larga Distancia"}),
    L("Ferrocarril San Martín: Cabred – Junín", [3810993, 240065],
      {"name_en": "San Martín Railway: Cabred – Junín", "operator": TA,
       "network": "Trenes Argentinos Larga Distancia"}),
    L("Ferrocarril Sarmiento: Mercedes – Bragado", [3810995, 240311],
      {"name_en": "Sarmiento Railway: Mercedes – Bragado", "operator": TA,
       "network": "Trenes Argentinos Larga Distancia"}),
    L("Ferrocarril Roca: Viedma – Bariloche", [1607899, 13282693, 13282692, 13282691],
      {"name_en": "Roca Railway: Viedma – Bariloche (Tren Patagónico)",
       "operator": "Tren Patagónico", "network": "Tren Patagónico"}),
    # -- regional lines
    L("Tren de las Sierras", [5371511, 5371512, 13578656, 13594312, 13594311, 16699583,
                              13578652, 13578653, 13578654, 13578655],
      {"name_en": "Tren de las Sierras (Córdoba – Capilla del Monte)", "operator": TA,
       "network": "Trenes Argentinos", "ref": "TS"}),
    L("Tren Regional Salta", [2256018, 13262884, 19413651, 19413650],
      {"name_en": "Salta Regional Train", "operator": TA, "network": "Trenes Argentinos",
       "ref": "TRS"}),
    L("Tren del Valle", [5371519, 5371520],
      {"name_en": "Tren del Valle (Neuquén)", "operator": TA, "network": "Trenes Argentinos",
       "ref": "TDV"}),
    L("Metrotranvía de Mendoza", [3413331, 2119332],
      {"name_en": "Mendoza Metrotranvía", "operator": "Sociedad de Transporte de Mendoza",
       "network": "MendoTran", "ref": "MTM"}),
    L("Tren al Desarrollo", [6587619, 13308585],
      {"name_en": "Tren al Desarrollo (Santiago del Estero – La Banda)",
       "operator": "Provincia de Santiago del Estero", "network": "Tren al Desarrollo",
       "ref": "TAD"}),
    L("Tren Solar de la Quebrada", [17730839],
      {"name_en": "Quebrada de Humahuaca Solar Train", "operator": "Provincia de Jujuy",
       "network": "Tren Solar de la Quebrada", "ref": "TSQ"}),
    L("Tren Binacional Posadas Encarnación", [4281474, 13238188],
      {"name_en": "Posadas – Encarnación International Train",
       "operator": "Casimiro Zbikoski S.A.", "network": "Trenes Argentinos", "ref": "P-E"},
      lists=["Posadas", "Encarnación"]),
    # -- suspended (greyed, out of completion; ar_sources.md): the Chaco's three, stopped
    #    in September 2026, "temporarily" with no date
    L("Tren Metropolitano de Resistencia", [5460824, 1603490],
      {"name_en": "Resistencia Metropolitan Train", "operator": TA,
       "network": "Trenes Argentinos", "ref": "TMR", "suspended": True}),
    L("Tren Regional Chaco: Resistencia - Charadai", [1603491, 13328971],
      {"name_en": "Chaco Regional Train: Cacuí – Los Amores", "operator": TA,
       "network": "Trenes Argentinos", "ref": "TRR", "suspended": True}),
    L("Tren Regional Chaco: Sáenz Peña - Chorotis", [1732926, 13316075],
      {"name_en": "Chaco Regional Train: Sáenz Peña – Chorotis", "operator": TA,
       "network": "Trenes Argentinos", "ref": "TRS", "suspended": True}),
]

CFG = {
    "region": "ar",
    "prefix": "a",
    "label": "AR",
    "lines": LINES,
    # neighbours whose ground clip() takes out of the extract (religiondots' cc codes)
    "foreign": ("cl", "bo", "py", "br", "uy"),
    # route relations clip() keeps whole wherever they run (the Posadas - Encarnación train)
    "keep_routes": (4281474, 13238188),
    # Stations OSM has no record of: Encarnación (Paraguay), the binational train's far end,
    # at its track's end (the route's stop node there carries no tags).
    "extra_stations": [("Encarnación", -55.85789, -27.36775)],
    # Belgrano Norte's stop positions are named by direction: "Tortuguitas a Retiro".
    "stop_name_fix": re.compile(r"\s+a\s+(Retiro|Villa Rosa)\s*$"),
    # Stop positions named otherwise than their station record ("Manuel B. Gonnet" beside
    # the station "Manuel Bernardo Gonnet" 40 m off): read as the record's name, so they are
    # one station and OSM's Línea Roca is the register line's twin.
    "name_alias": {"Manuel B. Gonnet": "Manuel Bernardo Gonnet"},
}


def configure(cfg):
    CFG.clear()
    CFG.update(cfg)


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    s = re.sub(r"^(estacion|apeadero|parada)\s+(?:de\s+)?", "", s.strip())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def plain_station(name):
    """A station's name as shown: without a leading 'Estación'."""
    n = re.sub(r"^(Estaci[oó]n|Apeadero)\s+(?:de\s+)?", "", (name or "").strip())
    n = LINE_QUALIFIER.sub("", n)
    return n[:1].upper() + n[1:] if n else (name or "")


def line_id(name):
    h = hashlib.blake2b(f"{CFG['region']}|{name}".encode("utf-8"), digest_size=5)
    return CFG["prefix"] + h.hexdigest()


_S = {}


def register_name(tags):
    return tags.get("_reg_line", "")


def way_xy(ways, coords, w):
    nodes = np.asarray(ways[w][1], dtype=np.int64)
    pos, ok = coords.many(nodes)
    pos = pos[ok]
    return coords.x[pos] / 1e7, coords.y[pos] / 1e7


def way_km(ways, coords, w):
    x, y = way_xy(ways, coords, w)
    if x.size < 2:
        return 0.0
    return float(np.hypot(np.diff(x) * np.cos(np.radians(y[:-1])) * 111.32,
                          np.diff(y) * 110.57).sum())


def extent_ways(ways, coords, pattern, points, log, label):
    """Ways of track named like `pattern` along the shortest path through `points`."""
    rx = re.compile(pattern)
    wids = [w for w, (t, _n) in ways.items()
            if t.get("railway") in TRACK_KIND and rx.search(t.get("name") or "")
            and not t.get("service")]
    adj, xy, _f = kr.line_graph(wids, ways, coords)
    if len(xy) < 2:
        return set()
    nr = kr.Near(xy)
    on, km = set(), 0.0
    for a, b in zip(points[:-1], points[1:]):
        va, da = nr.nearest(*a)
        vb, db = nr.nearest(*b)
        got = kr.between(adj, {va: 0.0}, {vb: 0.0}, set())
        if got is None or da > 2000 or db > 2000:
            log(f"{CFG['label']}: {label}: no {pattern} track between {a} and {b} "
                f"({da:.0f} m, {db:.0f} m off)")
            continue
        on.update(got[0])
        km += got[1]
    out = set()
    for w in wids:
        nodes = np.asarray(ways[w][1]).tolist()
        if sum(1 for n in nodes if n in on) >= max(2, len(nodes) // 2):
            out.add(w)
    log(f"{CFG['label']}: {label}: {km:.1f} km of {pattern} track through its extent, "
        f"{len(out)} ways")
    return out


def components(wids, ways):
    """Connected pieces of a set of ways, by shared nodes: [set(way id)]."""
    by_node = defaultdict(list)
    for w in wids:
        for n in (ways[w][1][0], ways[w][1][-1]):
            by_node[int(n)].append(w)
        for n in ways[w][1]:
            by_node[int(n)].append(w)
    seen, out = set(), []
    for w in wids:
        if w in seen:
            continue
        comp, stack = set(), [w]
        seen.add(w)
        while stack:
            u = stack.pop()
            comp.add(u)
            for n in ways[u][1]:
                for v in by_node[int(n)]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
        out.append(comp)
    return out


def bridge(line, wids, ways, coords, owner, log):
    """Join the pieces of a line's ways where a route relation leaves a way out: the shortest
    path over any other rail way not owned by another line, from one piece to the rest, if
    it is BRIDGE_KM or less. Returns the ways added."""
    comps = components(wids, ways)
    if len(comps) < 2:
        return set()
    free = [w for w, (t, _n) in ways.items()
            if t.get("railway") in TRACK_KIND and owner.get(w) in (None, line)]
    adj, xy, _f = kr.line_graph(free, ways, coords)
    node_way = defaultdict(set)
    for w in free:
        for n in ways[w][1]:
            node_way[int(n)].add(w)
    added = set()
    comps.sort(key=len, reverse=True)
    main = set().union(*[{int(n) for w in comps[0] for n in ways[w][1]}])
    for comp in comps[1:]:
        src = {int(n): 0.0 for w in comp for n in ways[w][1] if int(n) in adj}
        dst = {n: 0.0 for n in main if n in adj}
        if not src or not dst:
            continue
        got = kr.between(adj, src, dst, set())
        if got is None or got[1] > BRIDGE_KM:
            log(f"{CFG['label']}: {line}: a piece of {len(comp)} ways stays apart "
                f"({'no path' if got is None else f'{got[1]:.1f} km away'})")
            continue
        path = [n for n in got[0] if n > 0]
        for u, v in zip(path[:-1], path[1:]):
            for w in node_way[u] & node_way[v]:
                if w not in wids:
                    added.add(w)
        main |= {int(n) for w in comp for n in ways[w][1]} | set(path)
        log(f"{CFG['label']}: {line}: bridged a gap of {got[1]:.2f} km to a piece of "
            f"{len(comp)} ways")
    return added


def load_osm(log):
    import build_model as bm
    region = CFG["region"]
    ways, rels, stops, cid, cx, cy = bm.load(region, log)
    coords = bm.Coords(cid, cx, cy)
    owner = {}
    km = Counter()
    for spec in CFG["lines"]:
        name = spec["name"]
        wids = set()
        for rid in spec["routes"]:
            if rid not in rels:
                log(f"{CFG['label']}: {name}: route relation {rid} is not in the extract")
                continue
            wids |= {r for ty, r, _ in rels[rid][1] if ty == "w" and r in ways
                     and ways[r][0].get("railway") in TRACK_KIND}
        shared = set()
        for pattern, pts in spec["extent"]:
            got = extent_ways(ways, coords, pattern, pts, log, name)
            # Track named in an extent is this line's even where an earlier line's routes
            # run over it too (Lobos: the Sarmiento's trains come in over the Roca's last
            # km): a copy of the way joins this line's graph, so both reach the station.
            # ownership.py gives the drawn way to one of them.
            shared |= {w for w in got if w in owner}
            wids |= got - shared
        mine = {w for w in wids if w not in owner}
        for w in mine:
            owner[w] = name
        extra = bridge(name, mine, ways, coords, owner, log)
        for w in extra:
            owner[w] = name
        mine |= extra
        # Pieces left apart that are short are the long-distance trains' runs over other
        # tracks of a corridor an earlier line owns (Retiro's and Once's approaches): they go
        # to no line, as the US build's track under named trains only does.
        comps = sorted(components(mine, ways), key=len, reverse=True)
        for c in comps[1:]:
            ckm = sum(way_km(ways, coords, w) for w in c)
            if ckm < PIECE_KEEP_KM:
                for w in c:
                    del owner[w]
                mine -= c
                x, y = way_xy(ways, coords, next(iter(c)))
                log(f"{CFG['label']}: {name}: a piece of {ckm:.1f} km ({len(c)} ways) apart "
                    f"from the rest, at {x[0]:.4f},{y[0]:.4f}, goes to no line")
        for w in mine:
            ways[w][0]["_reg_line"] = name
            km[name] += way_km(ways, coords, w)
        for w in shared:
            twin = -(10 ** 12) - w            # never an OSM id
            ways[twin] = (dict(ways[w][0], _reg_line=name), ways[w][1])
            km[name] += way_km(ways, coords, w)
        if shared:
            log(f"{CFG['label']}: {name}: {len(shared)} ways of its extent shared with "
                f"{', '.join(sorted({owner[w] for w in shared}))}")
        if not mine:
            log(f"{CFG['label']}: {name}: no track")
    log(f"{CFG['label']}: register track " + ", ".join(f"{k} {v:.1f} km" for k, v in km.items()))
    _S.update(ways=ways, rels=rels, stops=stops, coords=coords, owner=owner)
    return ways, stops, coords


def route_stop_nodes(members):
    """A route's stop nodes: build_model.stop_members, plus node members that are stop
    records under no role or another one than platform (the Tren Solar de la Quebrada's
    Tilcara and Posta de Hornillos have none; Llanquihue - Puerto Montt's are "station")."""
    import build_model as bm
    got = list(bm.stop_members(members))
    stops = _S["stops"]
    got += [r for ty, r, role in members if ty == "n" and r in stops and r not in got
            and not role.startswith(("platform",) + bm.STOP_ROLES)]
    return got


def routes():
    import build_model as bm
    for rid, (tags, members) in _S["rels"].items():
        if tags.get("type") == "route" and tags.get("route") in bm.ROUTE_KINDS:
            yield rid, tags, members


def route_stations(log):
    """The stations some OSM route relation stops at, by kr's station ids, and the pairs of
    stations some route calls at one after the other."""
    node_st = _S["node_st"]
    out, pairs = set(), set()
    for _rid, _tags, members in routes():
        seq = []
        for n in route_stop_nodes(members):
            s = node_st.get(n)
            if s is not None:
                out.add(s)
                if not seq or seq[-1] != s:
                    seq.append(s)
        pairs |= {frozenset(p) for p in zip(seq[:-1], seq[1:])}
    return out, pairs


def is_metro_record(tags):
    """A Subte station (or stop) record: no register line here is a metro, and the Subte's
    "Retiro (E)" would otherwise head Retiro's complex and name the Mitre's terminal."""
    if tags.get("train") == "yes":
        return False
    return (tags.get("station") == "subway" or tags.get("subway") == "yes"
            or bool(METRO_NETWORK.search(tags.get("network") or "")))


def build_stations(stops, log):
    fix = CFG.get("stop_name_fix")
    rail = {}
    for nid, (tags, lon, lat) in stops.items():
        if is_metro_record(tags):
            continue
        if fix is not None and tags.get("name") and fix.search(tags["name"]):
            tags = dict(tags, name=fix.sub("", tags["name"]).strip())
        if tags.get("name") in CFG.get("name_alias", {}):
            tags = dict(tags, name=CFG["name_alias"][tags["name"]])
        rail[nid] = (tags, lon, lat)
    _S["stops"] = rail
    st, node_st, by_key, by_base = _orig["build_stations"](rail, log)
    # kr finds a stop node's station by its exact name before its bracketless one, and gives
    # up if the exact name's only record is far away: "Mercedes", the Sarmiento's stop,
    # found a "Mercedes" in Corrientes and never "Mercedes (Sarmiento)" 20 m off.
    late = 0
    for nid, (tags, lon, lat) in rail.items():
        if nid in node_st or not tags.get("name"):
            continue
        best, bd = None, kr.STOP_TO_STATION_M
        for c in (by_key.get(kr.name_key(tags["name"]), [])
                  + by_base.get(kr.base_key(tags["name"]), [])):
            d = kr.dist_m(lon, lat, st[c]["lon"], st[c]["lat"])
            if d <= bd:
                best, bd = c, d
        if best is not None:
            node_st[nid] = best
            late += 1
    for i, (name, lon, lat) in enumerate(CFG.get("extra_stations", ())):
        fid = EXTRA_BASE - i
        st[fid] = {"name": name, "name_en": "", "lon": lon, "lat": lat, "rank": 1}
        by_key[kr.name_key(name)].append(fid)
        by_base[kr.base_key(name)].append(fid)
    _S.update(st=st, node_st=node_st)
    served, pairs = route_stations(log)
    # Only a station some route stops at may be anchored by a node on a line's track: a
    # closed station's railway=station node can sit on track trains still use.
    kept = {n: s for n, s in node_st.items() if s in served}
    log(f"{CFG['label']}: {late} stop nodes placed by their bracketless name; {len(served)} "
        f"stations some OSM route stops at; stop nodes kept {len(kept)} of {len(node_st)}")
    _S["node_st"] = kept
    _S["pairs"] = {frozenset(f"k{s}" for s in p) for p in pairs}
    return st, kept, by_key, by_base


def route_lists(log):
    """{line: {station name}}: mx_register.route_lists, over this country's ways."""
    import build_model as bm
    ways, rels, coords = _S["ways"], _S["rels"], _S["coords"]
    st, node_st = _S["st"], _S["node_st"]
    wxy = {}

    def xy(w):
        if w not in wxy:
            wxy[w] = way_xy(ways, coords, w)
        return wxy[w]

    lists = defaultdict(set)
    for _rid, _tags, members in routes():
        rw = [r for ty, r, _ in members if ty == "w" and r in ways
              and register_name(ways[r][0])]
        if not rw:
            continue
        for n in route_stop_nodes(members):
            s = node_st.get(n)
            if s is None:
                continue
            lon, lat = st[s]["lon"], st[s]["lat"]
            kx = math.cos(math.radians(lat)) * 111320
            for w in rw:
                x, y = xy(w)
                if x.size and np.min(np.hypot((x - lon) * kx, (y - lat) * 110570)) <= ALONG_M:
                    lists[register_name(ways[w][0])].add(st[s]["name"])
    return lists


def load_lists(path, log):
    lists = route_lists(log)
    for spec in CFG["lines"]:
        lists[spec["name"]].update(spec["lists"])
    for ln in sorted(lists):
        log(f"{CFG['label']}: {ln}: {len(lists[ln])} stations listed")
    return {k: [sorted(v)] for k, v in lists.items()}, defaultdict(list), {}


_orig = {}


def adopt():
    """Point kr_register's country-specific globals at this country's, for this process."""
    if _orig:
        return
    _orig["build_stations"] = kr.build_stations
    kr.load_osm = load_osm
    kr.register_name = register_name
    kr.load_lists = load_lists
    kr.build_stations = build_stations
    kr.line_id = line_id
    kr.TRACK_KIND = dict(TRACK_KIND)
    kr.NOT_PASSENGER = set()       # the routes chose the ways; yard-tagged approaches included
    kr.NAME_ALIAS = {}
    kr.STATION_ALIAS = {}
    kr.PROX_M = -1                 # no station joins a line by being near its track


def drop_runs(lines, geoms, log):
    """mx_register.drop_junction_runs, choosing better which side of a triangle goes. At a
    junction just past a station (the Mitre's branches part 300 m beyond Belgrano R: Drago
    is the José León Suárez branch's next stop, Coghlan the Mitre branch's), kr's search
    pairs all three stations, and each of the three sections is covered by the other two.
    mx drops the longest, which here was the real Belgrano R - Drago. A section whose ends
    some OSM route calls at one after the other is a train's; one no route does is the run
    past the junction, so those are tried first, then the longest."""
    from mx_register import COVER_SHARE, _covered_share
    from n02 import walk_order
    pairs = _S.get("pairs", set())
    n_drop, km_drop, what = 0, 0.0, []
    for l in lines:
        g = geoms[l["id"]]
        secs = sorted(l["sections"], key=lambda s: (frozenset(s[:2]) in pairs, -s[2]))
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
                    what.append(f"{l['name']} {s[2]:.1f} km"
                                + ("" if frozenset(s[:2]) in pairs else " (no route calls "
                                   "at both one after the other)"))
                    g.pop(f"{a}|{b}", None)
                    if isinstance(l.get("highspeed_sections"), dict):
                        l["highspeed_sections"].pop(f"{a}|{b}", None)
                    break
        if len(keep) != len(l["sections"]):
            ks = {(x[0], x[1]) for x in keep}
            l["sections"] = [x for x in l["sections"] if (x[0], x[1]) in ks]
            l["km"] = round(sum(x[2] for x in l["sections"]), 3)
            l["display"] = walk_order([(x[0], x[1]) for x in l["sections"]])
    log(f"{CFG['label']}: {n_drop} sections dropped as runs past a junction "
        f"({km_drop:,.1f} km): " + ", ".join(what))


def build(path, log):
    adopt()
    lines, stations, geoms = kr.build(path, log)
    drop_runs(lines, geoms, log)
    import gb_register
    gb_register.drop_shortcuts(lines, geoms, log)       # logs as "GB:"
    pre = CFG["prefix"]

    def r(sid):
        return pre + sid[1:] if sid.startswith("k") else sid

    out_st = {}
    for sid, s in stations.items():
        nid = r(sid)
        s["id"] = nid
        s["name"] = plain_station(s["name"])
        out_st[nid] = s
    out_geoms = {}
    info_of = {spec["name"]: spec["info"] for spec in CFG["lines"]}
    for l in lines:
        info = info_of.get(l["name"], {})
        l["src"] = CFG["region"]
        for k in ("name_en", "operator", "operator_en", "network", "ref", "colour", "kind"):
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
        log(f"{CFG['label']}: {l['name']}: {l['km']:.1f} km, {len(l['sections'])} sections, "
            f"{len({s for sec in l['sections'] for s in sec[:2]})} stations"
            + (" (suspended)" if l.get("suspended") else ""))
    for s in out_st.values():
        s["lines"] = set()
    for l in lines:
        for a, b, *_ in l["sections"]:
            out_st[a]["lines"].add(l["id"])
            out_st[b]["lines"].add(l["id"])
    out_st = {k: s for k, s in out_st.items() if s["lines"]}
    missing = sorted(set(info_of) - {l["name"] for l in lines})
    if missing:
        log(f"{CFG['label']}: no line built for {', '.join(missing)}")
    return lines, out_st, out_geoms


# ---------------------------------------------------------------- clip, report, chainage

# religiondots' outlines are generalised: at Iguazú they put the Tren Ecológico de la Selva's
# Garganta del Diablo end in Brazil. Only ground this far (degrees, ~1.5 km) beyond this
# country's own outline is clipped.
HOME_BUFFER_DEG = 0.015


def foreign_area():
    """The neighbours' ground (religiondots' outlines), less this country's own and a margin
    round it."""
    from shapely.geometry import shape
    from shapely.ops import unary_union
    feats = json.loads(SHAPES.read_text(encoding="utf-8"))["features"]
    home = unary_union([shape(f["geometry"]) for f in feats
                        if f["properties"].get("cc") == CFG["region"]])
    other = unary_union([shape(f["geometry"]) for f in feats
                         if f["properties"].get("cc") in CFG["foreign"]])
    return other.difference(home.buffer(HOME_BUFFER_DEG))


def clip(log=print):
    """Rewrite data/proc/<cc> without what lies in the neighbours: a way stays if any node
    lies outside them (a border-crossing way stays whole), a stop if it lies outside, a
    relation if a member stayed. The `keep_routes` relations keep every member. Route
    relations whose name matches `drop_routes` (lines OSM maps before they open: Santiago's
    "Línea 7 (en construcción)") go too, or build_model would build them as running."""
    from shapely import contains_xy, prepare
    area = foreign_area()
    prepare(area)
    d = ROOT / "data" / "proc" / CFG["region"]
    ways = pickle.load(open(d / "ways.pkl", "rb"))
    rels = pickle.load(open(d / "rels.pkl", "rb"))
    stops = pickle.load(open(d / "stops.pkl", "rb"))
    infra_p = d / "infra.pkl"
    infra = pickle.load(open(infra_p, "rb")) if infra_p.exists() else None
    c = np.load(d / "coords.npz")
    nid, x, y = c["id"], c["x"] / 1e7, c["y"] / 1e7
    outside = set(nid[~contains_xy(area, x, y)].tolist())
    keep_w, keep_n = set(), set()
    for rid in CFG.get("keep_routes", ()):
        if rid in rels:
            keep_w |= {r for t, r, _ in rels[rid][1] if t == "w"}
            keep_n |= {r for t, r, _ in rels[rid][1] if t == "n"}
    n0 = (len(ways), len(stops), len(rels))
    gone = Counter(t.get("name") for k, (t, n) in ways.items()
                   if k not in keep_w and not any(int(i) in outside for i in n))
    ways = {k: v for k, v in ways.items()
            if k in keep_w or any(int(i) in outside for i in v[1])}
    stops = {k: v for k, v in stops.items()
             if k in keep_n or not contains_xy(area, v[1], v[2])}
    kept = {("w", k) for k in ways} | {("n", k) for k in stops}
    unbuilt = CFG.get("drop_routes")

    def opened(tags, members):
        """Not a route over a line OSM marks unopened, by its own name or its ways' names
        (Santiago's "Línea 7: Dirección Brasil" runs over "Línea 7 (en construcción)")."""
        if unbuilt is None:
            return True
        if unbuilt.search(tags.get("name") or ""):
            return False
        names = [ways[r][0].get("name") or "" for t, r, _ in members if t == "w" and r in ways]
        return not names or sum(1 for n in names if unbuilt.search(n)) * 2 < len(names)
    routes = {k for k, (tags, members) in rels.items()
              if tags.get("type") == "route" and any((t, r) in kept for t, r, _ in members)
              and opened(tags, members)}
    gone_r = sorted(tags.get("name") or str(k) for k, (tags, _m) in rels.items()
                    if tags.get("type") == "route" and k not in routes)
    rels = {k: v for k, v in rels.items()
            if k in routes or any(t == "r" and r in routes for t, r, _ in v[1])}
    out = [("ways", ways), ("rels", rels), ("stops", stops)]
    if infra is not None:
        infra = {k: v for k, v in infra.items()
                 if any((t, r) in kept for t, r, _ in v[1])}
        out.append(("infra", infra))
    for name, obj in out:
        tmp = d / f"{name}.pkl.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, d / f"{name}.pkl")
    log(f"clipped {CFG['region']}: ways {n0[0]} -> {len(ways)}, stops {n0[1]} -> {len(stops)}, "
        f"relations {n0[2]} -> {len(rels)}; routes dropped: {', '.join(gone_r) or 'none'}; "
        f"ways dropped by name: " + ", ".join(f"{k} {v}" for k, v in gone.most_common(25)))


def report():
    """Each LINE's ways, km and connected pieces, as load_osm makes them."""
    load_osm(print)
    ways, coords, owner = _S["ways"], _S["coords"], _S["owner"]
    by = defaultdict(set)
    for w, ln in owner.items():
        by[ln].add(w)
    for spec in CFG["lines"]:
        wids = by.get(spec["name"], set())
        comps = components(wids, ways)
        kms = sorted((sum(way_km(ways, coords, w) for w in c) for c in comps), reverse=True)
        print(f"{spec['name']}: {len(wids)} ways, {sum(kms):.1f} km, {len(comps)} pieces "
              + " ".join(f"{k:.1f}" for k in kms[:8]))
        if len(comps) > 1:
            for c in sorted(comps, key=len)[:-1]:
                x, y = way_xy(ways, coords, next(iter(c)))
                names = Counter(ways[w][0].get("name") for w in c).most_common(2)
                print(f"    piece of {len(c)} ways at {x[0]:.4f},{y[0]:.4f} {names}")


MINISTRY_STATIONS = ROOT / "data" / "raw" / "ar" / "estaciones_ffcc_serv_22.json"


def chainage():
    """Built sections against the transport ministry's station km posts ("progresiva",
    Estaciones de Trenes y Servicios activos a 2022, datos.transporte.gob.ar, CC BY 4.0).

    A section is compared where both its ends are within 300 m of a ministry station with a
    km post under the same line label ("Roca", "FFCC Mitre"...): |km a - km b| against the
    built length. Posts run from each line's terminus along each branch, so a section
    between two branches' stations (Temperley - Haedo has its own origin) compares nothing
    useful; the median and the share within 5% are what count."""
    feats = json.loads(MINISTRY_STATIONS.read_text(encoding="utf-8"))["features"]
    posts = []
    for f in feats:
        p = f["properties"]
        label = next((v for k, v in p.items() if k.startswith("l") and k not in ("lat", "long")),
                     None)
        try:
            km = float(p["progresiva"])
        except (TypeError, ValueError):
            continue
        lon, lat = f["geometry"]["coordinates"][:2]
        posts.append((label, km, p["nam"], lon, lat))
    d = ROOT / "dist" / "data" / CFG["region"]
    lines = json.loads((d / "lines.json").read_text(encoding="utf-8"))["lines"]
    st = json.loads((d / "stations.json").read_text(encoding="utf-8"))["stations"]

    def near(sid):
        s = st[sid]
        out = []
        for label, km, nam, lon, lat in posts:
            if kr.dist_m(s["x"], s["y"], lon, lat) <= 300:
                out.append((label, km, nam))
        return out

    allr = []
    for l in lines:
        if l.get("src") == "osm":
            continue
        rs, bad = [], []
        for a, b, km, *_ in l["sections"]:
            pa, pb = near(a), near(b)
            got = [(abs(x[1] - y[1]), x, y) for x in pa for y in pb if x[0] == y[0]]
            if not got or km < 0.3:
                continue
            dk, x, y = min(got, key=lambda g: abs(g[0] - km))
            if dk <= 0:
                continue
            r = km / dk
            rs.append(r)
            if abs(r - 1) > 0.05:
                bad.append(f"{st[a]['n']} - {st[b]['n']} {km:.2f} against {dk:.2f} "
                           f"({x[0]} {x[1]} / {y[1]})")
        if not rs:
            continue
        rs.sort()
        allr += rs
        ok = sum(1 for r in rs if abs(r - 1) <= 0.05)
        print(f"{l['name']}: {len(rs)} of {len(l['sections'])} sections have km posts at both "
              f"ends; median {rs[len(rs) // 2]:.3f}, {ok} within 5%")
        for x in bad:
            print(f"    {x}")
    if allr:
        allr.sort()
        print(f"all: {len(allr)} sections, median {allr[len(allr) // 2]:.3f}, "
              f"{sum(1 for r in allr if abs(r - 1) <= 0.05)} within 5%")


if __name__ == "__main__":
    if "--clip" in sys.argv:
        clip()
    elif "--report" in sys.argv:
        report()
    elif "--chainage" in sys.argv:
        chainage()
    else:
        print(__doc__)
