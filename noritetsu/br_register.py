"""Brazil: register lines are the passenger track of Brazil's trains, laid on OpenStreetMap.

    python br_register.py --fetch      # Wikidata's Brazilian railway stations -> data/raw/br
    python br_register.py --names      # the track names the extract gives, with km
    python build_model.py --region br --register br_register:data/raw/br

WHAT BRAZIL HAS. About 30,000 km of railway, nearly all of it freight (Rumo, VLI/FCA, MRS,
Vale's EFVM and EFC, Transnordestina). Passenger trains run on:

  - the commuter railways: São Paulo's CPTM, ViaMobilidade and TIC Trens lines 7 to 13, Rio de
    Janeiro's SuperVia (eight lines out of Central do Brasil);
  - Vale's two long-distance trains, Vitória - Belo Horizonte on the EFVM (daily, with a
    connecting train Desembargador Drumond - Itabira) and São Luís - Parauapebas on the EFC
    (three a week each way);
  - the Serra Verde Express, Curitiba - Morretes (Friday to Sunday, daily in the summer and July
    holidays);
  - metros, light rail, monorails and trams in a dozen cities.

THE REGISTER is the passenger track of the train lines (the first three groups), the way the US
build keeps only NARN's passenger track and Mexico's only its named passenger track, plus
Teresina's metro, whose OSM route is broken (below). Metros, light rail, trams and monorails
stay OSM's route relations, as in the US and UK builds: their routes are complete, with stops,
in every city (br_sources.md has the survey). Two ways of finding a register line's track:

  - NAMED: OSM's track names, where the track is named for its line. São Paulo's CPTM track is
    ("Linha 7 - Rubi" ... "Linha 13 - Jade").
  - ROUTES: the ways of the line's own OSM route relations, where the track is named for the
    freight railway it belongs to (the EFVM's 1,100 named km include its ore branches) or for
    a track designation (SuperVia's "Via A" to "Via H"). The El Chepe recipe in mx_register.

Each way belongs to one register line: named lines claim first, then ROUTE_LINES in their
order. SuperVia's four-track trunk Central - Deodoro carries the Deodoro locals on one pair and
the Japeri and Santa Cruz trains on the other; both pairs are one line, "Linha Deodoro" (the
pairs lie side by side: Anita's rule on second tracks), so Linha Japeri begins at Deodoro and
Linha Santa Cruz where it leaves the Japeri line at Vila Militar. A SuperVia line's way
moves into a FOLD_INTO line when it lies within FOLD_M of that line's track over FOLD_SHARE of
its length. OSM's routes for the full services (Central - Japeri) stay as the lines as
operated (flagged dup: the register line is the one completion counts).

GAPS. Route relations miss ways (the EFC's at São Luís); between two consecutive stops their
own ways do not join, the shortest network track is added (fill_gaps). OSM's track itself has
breaks of a few metres to 150 m where two ways do not share a node (the EFVM near Dois
Irmãos); bridge_breaks joins them.

STATIONS. kr_register's rule (a stop node on the line's track), the stops OSM's route relations
list on the line's own track (gb_register's rule, as mx_register), EXTRA_LISTS where OSM's
routes leave stops out (Belo Horizonte on the EFVM; the EFC's route lists 8 of its 15 stops),
and Wikidata's station items for a listed name OSM has no record of (Marabá).

The `path` argument is data/raw/br; the OSM half is read from data/proc/br (extract.py).
"""
import hashlib
import json
import math
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import kr_register as kr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "br"
REGION = "br"
USER_AGENT = "noritetsu-build/1.0 (hobby rail map)"

# ---------------------------------------------------------------- the lines

# OSM track name -> register line (exact names, 2026-10 extract). Register names are OSM's
# route_master names, so the OSM line is matched to its register line (colour, twin).
NAMED = {
    "Linha 7 - Rubi": "Linha 7 - Rubi",
    "Linha 8 - Diamante": "Linha 8 - Diamante",
    "Linha 9 - Esmeralda": "Linha 9 - Esmeralda",
    "Linha 9-Esmeralda": "Linha 9 - Esmeralda",
    "Linha 10 - Turquesa": "Linha 10 - Turquesa",
    "Linha 11 - Coral": "Linha 11 - Coral",
    "Linha 12 - Safira": "Linha 12 - Safira",
    "Linha 13 - Jade": "Linha 13 - Jade",
    "Linha 13 - Jade;Expresso Aeroporto": "Linha 13 - Jade",
    "Linha 13 - Jade/Expresso Aeroporto": "Linha 13 - Jade",
}

SV_GUAPIMIRIM = "Linha Guapimirim: Saracuruna - Guapimirim"
EFVM = "Estrada de Ferro Vitória a Minas"
EFC = "Estrada de Ferro Carajás"
SERRA_VERDE = "Curitiba - Morretes"
TERESINA = "Linha 1 do Metrô de Teresina"

# Register lines made of the ways of OSM route relations, in claiming order.
ROUTE_LINES = [
    # CPTM ways OSM leaves unnamed: Line 8's Itapevi - Amador Bueno shuttle.
    ("Linha 8 - Diamante", [2108731, 2886302, 419297, 2174505]),
    # SuperVia (route relation ids: both directions of each line).
    ("Linha Deodoro", [1609111, 9963664]),
    ("Linha Japeri", [1589649, 6432247]),
    ("Linha Santa Cruz", [6432130, 1609112]),
    ("Linha Paracambi", [9963668, 1578818]),
    ("Linha Belford Roxo", [9963650, 1578053]),
    ("Linha Saracuruna", [1306884, 9963672, 6018221, 9963666]),
    ("Linha Vila Inhomirim", [1604426, 9963644]),
    (SV_GUAPIMIRIM, [9963673, 1604427]),
    # Vale's trains: Vitória (Pedro Nolasco) - Belo Horizonte, and the Itabira connection.
    (EFVM, [3331630, 7831016, 4848171, 9226080]),
    (EFC, [4845213]),
    # The Serra Verde Express's route (no stops in OSM: EXTRA_LISTS).
    (SERRA_VERDE, [2599219]),
    # Teresina: OSM's route runs on over the disused Teresina - Parnaíba railway to Luís
    # Correia (359 km; no train since the 1980s); BBOX keeps the city line.
    (TERESINA, [420628, 10570394]),
]
# Ways of a line kept only inside this box (lon0, lat0, lon1, lat1).
BBOX = {TERESINA: (-42.86, -5.13, -42.70, -5.05)}
# Lines whose route relations miss ways (the EFC's at São Luís, the EFVM's between Fundão and
# Colatina...): between two consecutive stops of a route that its own ways do not join, the
# shortest track over the whole network is taken too, if no longer than GAP_DETOUR times the
# straight distance plus GAP_MARGIN_M, as build_model traces an OSM line's gaps.
GAP_FILL = {EFVM, EFC, "Linha Deodoro", "Linha Japeri", "Linha Santa Cruz", "Linha Paracambi",
            "Linha Belford Roxo", "Linha Saracuruna", "Linha Vila Inhomirim", SV_GUAPIMIRIM}
GAP_DETOUR = 1.6
GAP_MARGIN_M = 4000
ANCHOR_M = 600
BRIDGE_M = 150
# A listed station may lie this far from its line's own track (kr_register's MATCH_M is 800):
# a SuperVia branch's track begins where it leaves the trunk, which FOLD_INTO gave the earlier
# line, 1.0-1.3 km out of Deodoro and Saracuruna, and the branch's first section still starts
# at that station.
MATCH_M = 1500
# Station records that are no stop: "CTO" beside Central do Brasil (a SuperVia operations
# point mapped as railway=station, in no route's stop list).
NOT_STOPS = {"CTO"}
# Lines whose stations are only those listed (their routes' stops and EXTRA_LISTS): a
# freight railway's track carries stop nodes of stations no passenger train calls at (Aroaba,
# Acesita on the EFVM) and passes other systems' stations (Belo Horizonte's metro Central).
STRICT = {EFVM, EFC, SERRA_VERDE}
# Lines whose ways fold into an earlier line where they run beside it (SuperVia's trunk)...
FOLD_GROUP = {"Linha Deodoro", "Linha Japeri", "Linha Santa Cruz", "Linha Paracambi",
              "Linha Belford Roxo", "Linha Saracuruna", "Linha Vila Inhomirim", SV_GUAPIMIRIM}
# ... and the lines they fold into: only the four-track trunk out of Central. Elsewhere two
# SuperVia lines side by side are two lines (the Santa Cruz branch beside the Japeri line
# west of Deodoro, the Guapimirim and Vila Inhomirim lines out of Saracuruna): folded, the
# branch lost its first station.
FOLD_INTO = {"Linha Deodoro"}
FOLD_M = 45
FOLD_SHARE = 0.9

# Stops OSM's routes leave out, by name, matched as kr_register matches Korea's lists (the OSM
# station record of that name nearest the line's track, within kr.MATCH_M).
EXTRA_LISTS = {
    # The 30 stops Vale publishes: the route lists 28 and Itabira's; Belo Horizonte is missing.
    EFVM: ["Estação Ferroviária de Belo Horizonte"],
    # Vale's 15 stops (São Luís - Arari - Vitória do Mearim - Santa Inês - Alto Alegre -
    # Mineirinho - Auzilândia - Altamira - Vila Pindaré - Nova Vida - Açailândia - São Pedro -
    # Marabá - Itainópolis - Parauapebas); OSM's route lists 8. Marabá is a Wikidata item;
    # Arari, Auzilândia, Altamira, Vila Pindaré, Nova Vida and Itainópolis have no record in
    # OSM or Wikidata (br_sources.md).
    EFC: ["Estação Ferroviária de Marabá", "Estação Ferroviária De Santa Inês"],
    SERRA_VERDE: ["Curitiba", "Morretes"],
    # Teresina's south-east branch, open from 11 May 2026 (peak hours): OSM has the two
    # stations, not yet in the line's routes.
    TERESINA: ["Colorado", "Todos os Santos"],
}
# Wikidata items taken as stations where OSM has none (by item; the station must lie within
# WD_M of the line's track).
WD_TAKE = {"Q123459428": EFC}       # Estação Ferroviária de Marabá
WD_M = 1000

# What each register line is called in English, who runs it, and whether it runs.
LINE_INFO = {
    "Linha 7 - Rubi": {"name_en": "Line 7 Ruby", "operator": "TIC Trens", "ref": "7",
                       "network": "Trem Metropolitano de São Paulo"},
    "Linha 8 - Diamante": {"name_en": "Line 8 Diamond", "operator": "ViaMobilidade Linhas 8 e 9",
                           "ref": "8", "network": "Trem Metropolitano de São Paulo"},
    "Linha 9 - Esmeralda": {"name_en": "Line 9 Emerald", "operator": "ViaMobilidade Linhas 8 e 9",
                            "ref": "9", "network": "Trem Metropolitano de São Paulo"},
    "Linha 10 - Turquesa": {"name_en": "Line 10 Turquoise",
                            "operator": "Companhia Paulista de Trens Metropolitanos", "ref": "10",
                            "network": "Companhia Paulista de Trens Metropolitanos"},
    "Linha 11 - Coral": {"name_en": "Line 11 Coral",
                         "operator": "Companhia Paulista de Trens Metropolitanos", "ref": "11",
                         "network": "Companhia Paulista de Trens Metropolitanos"},
    "Linha 12 - Safira": {"name_en": "Line 12 Sapphire",
                          "operator": "Companhia Paulista de Trens Metropolitanos", "ref": "12",
                          "network": "Companhia Paulista de Trens Metropolitanos"},
    "Linha 13 - Jade": {"name_en": "Line 13 Jade", "operator": "Trivia Trens", "ref": "13",
                        "network": "Companhia Paulista de Trens Metropolitanos"},
    EFVM: {"name_en": "Vitória-Minas Railway (Vitória - Belo Horizonte, Itabira)",
           "operator": "Vale", "network": "Trem de passageiros da EFVM", "ref": "EFVM",
           # OSM's route_master colour for the train; the EFC's (no OSM colour) picked the same
           "colour": "#00939A"},
    EFC: {"name_en": "Carajás Railway (São Luís - Parauapebas)", "operator": "Vale",
          "network": "Trem de passageiros da EFC", "ref": "EFC", "colour": "#00939A"},
    SERRA_VERDE: {"name_en": "Curitiba - Morretes (Serra Verde Express)",
                  "operator": "Serra Verde Express", "network": "Serra Verde Express"},
    TERESINA: {"name_en": "Teresina Metro Line 1", "operator": "CMTP",
               "network": "Metrô de Teresina", "ref": "1"},
}
for _n, _en, _ref in (("Linha Deodoro", "Deodoro Line", "Deodoro"),
                      ("Linha Japeri", "Japeri Line", "Japeri"),
                      ("Linha Santa Cruz", "Santa Cruz Line", "Santa Cruz"),
                      ("Linha Paracambi", "Paracambi Line", "JRI-PBI"),
                      ("Linha Belford Roxo", "Belford Roxo Line", "Belford Roxo"),
                      ("Linha Saracuruna", "Saracuruna Line", "Saracuruna"),
                      ("Linha Vila Inhomirim", "Vila Inhomirim Line", "SAR-VIN"),
                      (SV_GUAPIMIRIM, "Guapimirim Line", "SAR-GIM")):
    LINE_INFO[_n] = {"name_en": f"SuperVia {_en}", "operator": "SuperVia",
                     "network": "SuperVia", "ref": _ref}

TRACK_KIND = {"rail": "rail", "narrow_gauge": "narrow_gauge", "subway": "subway",
              "light_rail": "light_rail", "preserved": "rail"}
# Serra Verde's track is the Rumo freight line; no usage value is left out but these.
NOT_PASSENGER = {"industrial", "military", "test"}

ALONG_M = 250            # a route's stop goes on a line whose own track is this close to it
COVER_M = 30             # mx_register.drop_junction_runs' rule, re-used
WD_BASE = -9_000_000_000


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    s = re.sub(r"^esta[cç][aã]o\s+(?:ferrovi[aá]ria\s+)?(?:de\s+|do\s+|da\s+)?", "", s.strip())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def plain_station(name):
    """A station's name as shown: 'Estação Ferroviária de Marabá' -> 'Marabá'."""
    n = re.sub(r"^Esta[cç][aã]o\s+(?:Ferrovi[aá]ria\s+)?(?:[Dd]e\s+|[Dd]o\s+|[Dd]a\s+)?", "",
               (name or "").strip())
    return n[:1].upper() + n[1:] if n else (name or "")


def line_id(name):
    h = hashlib.blake2b(f"br|{name}".encode("utf-8"), digest_size=5)
    return "b" + h.hexdigest()


_S = {}


def register_name(tags):
    return tags.get("_br_line", "")


def _way_xy(ways, coords, w):
    nodes = np.asarray(ways[w][1], dtype=np.int64)
    pos, ok = coords.many(nodes)
    pos = pos[ok]
    return coords.x[pos] / 1e7, coords.y[pos] / 1e7


def _near_share(x, y, tree_xy, lat0):
    """Share of a way's points (sampled every ~20 m) within FOLD_M of `tree_xy` points."""
    kx = math.cos(math.radians(lat0)) * 111320
    pts = []
    for i in range(len(x) - 1):
        L = math.hypot((x[i + 1] - x[i]) * kx, (y[i + 1] - y[i]) * 110570)
        k = max(1, int(L // 20))
        for j in range(k):
            t = j / k
            pts.append((x[i] + (x[i + 1] - x[i]) * t, y[i] + (y[i + 1] - y[i]) * t))
    pts.append((x[-1], y[-1]))
    p = np.asarray(pts)
    tx, ty = tree_xy
    near = 0
    for px, py in p:
        d = np.hypot((tx - px) * kx, (ty - py) * 110570)
        if d.size and d.min() <= FOLD_M:
            near += 1
    return near / len(p)


def load_osm(log):
    import build_model as bm
    ways, rels, stops, cid, cx, cy = bm.load(REGION, log)
    coords = bm.Coords(cid, cx, cy)
    km = Counter()

    def ok_track(tags):
        return (tags.get("railway") in TRACK_KIND and tags.get("usage") not in NOT_PASSENGER
                and not tags.get("service"))

    for wid, (tags, _n) in ways.items():
        line = NAMED.get((tags.get("name") or "").strip(), "")
        if line and ok_track(tags):
            tags["_br_line"] = line
            km[line] += 1
    for line, rids in ROUTE_LINES:
        box = BBOX.get(line)
        for rid in rids:
            if rid not in rels:
                log(f"BR: route relation {rid} ({line}) is not in the extract")
                continue
            for ty, ref, role in rels[rid][1]:
                if ty != "w" or ref not in ways:
                    continue
                if role and not role.startswith(("forward", "backward")):
                    continue
                tags = ways[ref][0]
                if "_br_line" in tags:
                    continue
                if not (tags.get("railway") in TRACK_KIND
                        and tags.get("usage") not in NOT_PASSENGER):
                    continue
                if box is not None:
                    x, y = _way_xy(ways, coords, ref)
                    if not x.size or not (box[0] <= x.mean() <= box[2]
                                          and box[1] <= y.mean() <= box[3]):
                        continue
                tags["_br_line"] = line
                km[line] += 1
    fill_gaps(ways, rels, coords, km, log)
    # SuperVia: a later line's way that runs beside an earlier line's track is the earlier
    # line's second pair of tracks.
    order = [ln for ln, _r in ROUTE_LINES if ln in FOLD_GROUP]
    by_line = defaultdict(list)
    for wid, (tags, _n) in ways.items():
        if tags.get("_br_line") in FOLD_GROUP:
            by_line[tags["_br_line"]].append(wid)
    moved = Counter()
    for i, later in enumerate(order):
        for earlier in order[:i]:
            if earlier not in FOLD_INTO:
                continue
            ex, ey = [], []
            for w in by_line[earlier]:
                x, y = _way_xy(ways, coords, w)
                ex.append(x)
                ey.append(y)
            if not ex:
                continue
            ex, ey = np.concatenate(ex), np.concatenate(ey)
            keep = []
            for w in by_line[later]:
                x, y = _way_xy(ways, coords, w)
                if x.size < 2 or not (ex.min() - 0.01 <= x.mean() <= ex.max() + 0.01
                                      and ey.min() - 0.01 <= y.mean() <= ey.max() + 0.01):
                    keep.append(w)
                    continue
                if _near_share(x, y, (ex, ey), float(y.mean())) >= FOLD_SHARE:
                    ways[w][0]["_br_line"] = earlier
                    by_line[earlier].append(w)
                    moved[(later, earlier)] += 1
                else:
                    keep.append(w)
            by_line[later] = keep
    for (a, b), n in sorted(moved.items()):
        log(f"BR: {n} ways of {a} beside {b}'s track folded into {b}")
    bridge_breaks(ways, coords, log)
    log("BR: register track ways " + ", ".join(f"{k} {v}" for k, v in sorted(km.items())))
    _S.update(ways=ways, rels=rels, stops=stops, coords=coords)
    return ways, stops, coords


def fill_gaps(ways, rels, coords, km, log):
    """GAP_FILL: join a route's consecutive stops over the network where its own ways do not."""
    import build_model as bm
    net_ids = [w for w, (t, _n) in ways.items() if t.get("railway") in TRACK_KIND]
    net = bm.TrackGraph(net_ids, ways, coords)
    edge_way = {}
    for w in net_ids:
        ns = list(ways[w][1])
        for a, b in zip(ns[:-1], ns[1:]):
            edge_way.setdefault((a, b) if a < b else (b, a), w)
    net_near = kr.Near(net.xy)
    for line, rids in ROUTE_LINES:
        if line not in GAP_FILL:
            continue
        added, gaps, failed = 0, 0, []
        for rid in rids:
            if rid not in rels:
                continue
            own_ids = [w for w, (t, _n) in ways.items() if t.get("_br_line") == line]
            own = bm.TrackGraph(own_ids, ways, coords)
            if len(own.xy) < 2:
                continue
            own_near = kr.Near(own.xy)

            def anchor(n):
                if n in own.adj:
                    return n
                p = coords.get(n)
                if p is None:
                    return None
                v, d = own_near.nearest(*p)
                if d <= ANCHOR_M:
                    return v
                v, d = net_near.nearest(*p)
                return v if d <= ANCHOR_M else None
            stops = [a for a in (anchor(n) for n in bm.stop_members(rels[rid][1]))
                     if a is not None]
            for a, b in zip(stops[:-1], stops[1:]):
                if a == b:
                    continue
                pa, pb = net.xy.get(a) or own.xy.get(a), net.xy.get(b) or own.xy.get(b)
                if pa is None or pb is None:
                    continue
                crow = bm.dist_m(*pa, *pb)
                if own.path(a, b, max(3 * crow, crow + 2000)) is not None:
                    continue
                gaps += 1
                got = net.path(a, b, crow * GAP_DETOUR + GAP_MARGIN_M)
                if got is None:
                    failed.append(f"{a}-{b}")
                    continue
                ids = got[1].tolist()
                for u, v in zip(ids[:-1], ids[1:]):
                    w = edge_way.get((u, v) if u < v else (v, u))
                    if w is None:
                        continue
                    t = ways[w][0]
                    if t.get("_br_line") or t.get("usage") in NOT_PASSENGER:
                        continue
                    t["_br_line"] = line
                    km[line] += 1
                    added += 1
        if gaps:
            log(f"BR: {line}: {gaps} gaps between consecutive stops of its routes, {added} ways "
                f"added from the network" + (f"; no path for {', '.join(failed)}" if failed else ""))


def bridge_breaks(ways, coords, log):
    """Join a line's track where OSM's ways stop short of each other. The EFVM's track has two
    breaks of 60 and 95 m between Rio Piracicaba and Dois Irmãos, where no way joins the two
    ends (the network's own way round is 118 km for 31). A dead end of one piece of a line's
    track within BRIDGE_M of a vertex of another piece is joined to it by a two-node way of
    the line's (a negative id; it carries no OSM tags but the line's)."""
    import build_model as bm
    by_line = defaultdict(list)
    for w, (t, _n) in ways.items():
        if t.get("_br_line"):
            by_line[t["_br_line"]].append(w)
    fresh = min(0, min(ways)) - 1
    made = []
    for line, wids in sorted(by_line.items()):
        g = bm.TrackGraph(wids, ways, coords)
        comp = {}
        c = 0
        for s in g.adj:
            if s in comp:
                continue
            c += 1
            comp[s] = c
            stack = [s]
            while stack:
                u = stack.pop()
                for v, _w in g.adj[u]:
                    if v not in comp:
                        comp[v] = c
                        stack.append(v)
        if c < 2:
            continue
        ids = np.fromiter(g.xy.keys(), dtype=np.int64)
        xy = np.array([g.xy[i] for i in ids.tolist()])
        cs = np.array([comp[i] for i in ids.tolist()])
        joined = set()
        for n, nbrs in g.adj.items():
            if len(nbrs) != 1:
                continue
            x, y = g.xy[n]
            d = np.hypot((xy[:, 0] - x) * math.cos(math.radians(y)) * 111320,
                         (xy[:, 1] - y) * 110570)
            d[cs == comp[n]] = np.inf
            j = int(np.argmin(d))
            if d[j] > BRIDGE_M:
                continue
            pair = tuple(sorted((comp[n], int(cs[j]))))
            if pair in joined:
                continue
            joined.add(pair)
            ways[fresh] = ({"railway": "rail", "_br_line": line}, [n, int(ids[j])])
            made.append(f"{line} {d[j]:.0f} m at {x:.4f},{y:.4f}")
            fresh -= 1
    log(f"BR: {len(made)} breaks in a line's OSM track bridged: " + "; ".join(made))


def drop_unlisted(lines, geoms, log):
    """STRICT lines: a station their lists do not name is no stop. Its two sections become one
    (km added, geometry joined); a section to such a station at a line end goes."""
    from n02 import walk_order
    lists = _S.get("lists", {})
    st = _S["out_st"]
    gone = []
    for l in lines:
        if l["name"] not in STRICT:
            continue
        keys = set()
        for n in lists.get(l["name"], ()):
            keys.add(kr.name_key(n))
            keys.add(kr.base_key(n))
        g = geoms[l["id"]]
        while True:
            deg = defaultdict(list)
            for s in l["sections"]:
                deg[s[0]].append(s)
                deg[s[1]].append(s)
            bad = [x for x in deg if kr.name_key(st[x]["name"]) not in keys
                   and kr.base_key(st[x]["name"]) not in keys]
            if not bad:
                break
            x = bad[0]
            secs = deg[x]
            gone.append(f"{l['name']}: {st[x]['name']}")
            for s in secs:
                l["sections"].remove(s)
            if len(secs) == 2:
                (a1, b1, k1, *_), (a2, b2, k2, *_) = secs
                ea = a1 if b1 == x else b1
                eb = a2 if b2 == x else b2
                p1 = g.pop(f"{a1}|{b1}", None) or []
                p2 = g.pop(f"{a2}|{b2}", None) or []
                if b1 != x:
                    p1 = p1[::-1]          # now runs ea -> x
                if a2 != x:
                    p2 = p2[::-1]          # now runs x -> eb
                if ea != eb:
                    l["sections"].append([ea, eb, round(k1 + k2, 3)])
                    g[f"{ea}|{eb}"] = list(p1) + list(p2[1:])
            else:
                for a, b, *_ in secs:
                    g.pop(f"{a}|{b}", None)
        l["km"] = round(sum(s[2] for s in l["sections"]), 3)
        l["display"] = walk_order([(s[0], s[1]) for s in l["sections"]])
        if isinstance(l.get("highspeed_sections"), dict):
            l["highspeed_sections"] = {f"{s[0]}|{s[1]}": False for s in l["sections"]}
    log(f"BR: {len(gone)} stations no train of their line stops at taken off it: "
        + "; ".join(gone))


def load_wikidata():
    p = RAW / "wikidata_stations.json"
    if not p.exists():
        return {}
    out = {}
    for r in json.loads(p.read_text(encoding="utf-8"))["results"]["bindings"]:
        q = r["item"]["value"].rsplit("/", 1)[-1]
        if q not in WD_TAKE or "label" not in r:
            continue
        m = re.match(r"Point\(([-\d.]+) ([-\d.]+)\)", r["coord"]["value"])
        if m:
            out[q] = {"name": r["label"]["value"], "lon": float(m.group(1)),
                      "lat": float(m.group(2))}
    return out


def line_tracks():
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


def build_stations(stops, log):
    st, node_st, by_key, by_base = _orig["build_stations"](stops, log)
    for sid in [s for s, v in st.items() if v["name"] in NOT_STOPS]:
        del st[sid]
        for d in (by_key, by_base):
            for k in list(d):
                if sid in d[k]:
                    d[k].remove(sid)
        for n in [n for n, s in node_st.items() if s == sid]:
            del node_st[n]
    tracks = line_tracks()
    _S["tracks"] = tracks
    wd = load_wikidata()
    added = []
    for q, w in sorted(wd.items()):
        nr = tracks.get(WD_TAKE[q])
        if nr is None:
            continue
        _v, d = nr.nearest(w["lon"], w["lat"])
        if d > WD_M:
            log(f"BR: Wikidata {q} {w['name']} is {d:.0f} m from {WD_TAKE[q]}'s track; not taken")
            continue
        fid = WD_BASE - int(q[1:])
        st[fid] = {"name": w["name"], "name_en": "", "lon": w["lon"], "lat": w["lat"], "rank": 1}
        by_key[kr.name_key(w["name"])].append(fid)
        by_base[kr.base_key(w["name"])].append(fid)
        added.append(f"{w['name']} ({q})")
    _S["wd"] = {WD_BASE - int(q[1:]): q for q in wd}
    _S.update(st=st, node_st=node_st)
    log(f"BR: {len(added)} Wikidata stations added: {'; '.join(added)}")
    return st, node_st, by_key, by_base


def route_lists(log):
    """{line: {station name}}: the stops OSM's route relations list, on each register line
    whose own track runs within ALONG_M of the stop (mx_register's rule)."""
    import build_model as bm
    ways, rels, coords = _S["ways"], _S["rels"], _S["coords"]
    st, node_st = _S["st"], _S["node_st"]
    wxy = {}

    def xy(w):
        if w not in wxy:
            wxy[w] = _way_xy(ways, coords, w)
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
    log(f"BR: station lists from {n_routes} OSM routes on register track, "
        f"{sum(len(v) for v in lists.values())} station-line pairs")
    return lists


def load_lists(path, log):
    lists = route_lists(log)
    for ln, names in EXTRA_LISTS.items():
        lists[ln].update(names)
    _S["lists"] = {k: set(v) for k, v in lists.items()}
    for ln in sorted(lists):
        log(f"BR: {ln}: {len(lists[ln])} stations listed")
    return {k: [sorted(v)] for k, v in lists.items()}, defaultdict(list), {}


_orig = {}


def adopt():
    """Point kr_register's country-specific globals at Brazil's, for this process."""
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
    kr.MATCH_M = MATCH_M


def build(path, log):
    adopt()
    lines, stations, geoms = kr.build(path, log)
    _S["out_st"] = stations
    drop_unlisted(lines, geoms, log)
    import mx_register
    mx_register.drop_junction_runs(lines, geoms, log)   # logs as "MX:"; the rule is the same
    import gb_register
    gb_register.drop_shortcuts(lines, geoms, log)       # logs as "GB:"
    wd = {f"k{fid}": f"bq{q[1:]}" for fid, q in _S.get("wd", {}).items()}

    def r(sid):
        return wd.get(sid) or ("b" + sid[1:] if sid.startswith("k") else sid)

    out_st = {}
    for sid, s in stations.items():
        nid = r(sid)
        s["id"] = nid
        s["name"] = plain_station(s["name"])
        out_st[nid] = s
    out_geoms = {}
    for l in lines:
        info = LINE_INFO.get(l["name"], {})
        l["src"] = "br"
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
        log(f"BR: {l['name']}: {l['km']:.1f} km, {len(l['sections'])} sections, "
            f"{len({s for sec in l['sections'] for s in sec[:2]})} stations")
    for s in out_st.values():
        s["lines"] = set()
    for l in lines:
        for a, b, *_ in l["sections"]:
            out_st[a]["lines"].add(l["id"])
            out_st[b]["lines"].add(l["id"])
    out_st = {k: s for k, s in out_st.items() if s["lines"]}
    missing = sorted(set(LINE_INFO) - {l["name"] for l in lines})
    if missing:
        log(f"BR: no line built for {', '.join(missing)}")
    return lines, out_st, out_geoms


# ---------------------------------------------------------------- the small sources

WD = "https://query.wikidata.org/sparql"
# Railway, metro and light-rail stations in Brazil, with a point.
Q_STATIONS = """SELECT ?item ?label ?coord ?line ?lineLabel ?class ?closed ?state WHERE {
  ?item wdt:P17 wd:Q155 ; wdt:P31 ?class ; wdt:P625 ?coord .
  VALUES ?class { wd:Q55488 wd:Q55678 wd:Q928830 wd:Q1793804 wd:Q4663385 wd:Q2175765
                  wd:Q12819564 wd:Q1339195 wd:Q18543139 }
  OPTIONAL { ?item rdfs:label ?label FILTER(LANG(?label) = "pt") }
  OPTIONAL { ?item wdt:P81 ?line . ?line rdfs:label ?lineLabel FILTER(LANG(?lineLabel) = "pt") }
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
    for wid, (tags, _nodes) in ways.items():
        if tags.get("railway") not in TRACK_KIND or not tags.get("name"):
            continue
        x, y = _way_xy(ways, coords, wid)
        if x.size < 2:
            continue
        km[tags["name"]] += float(np.hypot(np.diff(x) * np.cos(np.radians(y[:-1])) * 111.32,
                                           np.diff(y) * 110.57).sum())
    for name, k in km.most_common():
        print(f"  {k:8.1f}  {name}{'  <- register' if name in NAMED else ''}")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    elif "--names" in sys.argv:
        names_report()
    else:
        print(__doc__)
