"""Turn route relations into LINES: the objects a rider actually talks about.

    python build_model.py --region jp

This is the half of the project that the tracker rests on.  A "line" here is one thing you
could say you have half-finished -- the Yamanote Line, the Piccadilly line, the 2 train -- and
OpenStreetMap does not store one.  What it stores is route relations, normally one per
DIRECTION plus a variant for every short turn and branch service, so the Yamanote arrives as
two relations under one route_master and a big commuter line as a dozen.

WHAT THIS BUILDS

  station   one record per real stopping place, with the per-platform stop_position nodes
            that route relations point at collapsed onto it
  line      one record per route_master (or per orphan route relation), with a colour, an
            operator, a display order of stations, and its sections
  section   the atomic ridden unit: a pair of stations that are consecutive on the line,
            with the track geometry between them and its length

SECTIONS ARE A SET, NOT A SEQUENCE, and that is the important decision here.  Taking the
longest variant as "the line" and ignoring the rest loses every branch: the Keio line's
Sagamihara branch, a metro line's depot spur service, the far end of a line where only some
trains run.  So sections are the union of consecutive-station pairs over ALL variants, which
handles branches and short turns without needing them to form a single linear order.  A
separate display order -- the longest variant -- is what the strip diagram draws, and it is
allowed to be an incomplete view of the line.

Line length is then the sum of its distinct sections, so a line that is a loop, or that has a
branch, totals correctly, and "50% of this line" means half its track, not half its stops.
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
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent

ROUTE_KINDS = {"train", "subway", "light_rail", "tram", "monorail", "funicular"}
STOP_ROLES = ("stop", "stop_entry_only", "stop_exit_only")


def stop_members(members):
    """A route's stop nodes: its stop-role members, or, where it has fewer than two, those and
    its platform-role nodes. Helsinki's tram routes list only platforms (tram 13 one stop and
    twelve platforms), and every one was dropped as having under two stops; place_stations
    snaps a node beside the track to it. Measured 2026-10-01: 26 routes gain stops in fi, a
    few heritage and city trams elsewhere (be, ch, cn, pl, ro, si), numbered single trains in
    tw (named trains anyway), 4 in kr."""
    got = [ref for ty, ref, role in members if ty == "n" and role.startswith(STOP_ROLES)]
    if len(got) >= 2:
        return got
    return got + [ref for ty, ref, role in members
                  if ty == "n" and role.startswith("platform")]

# A station node proper, as against a stop_position, which is one per platform track.
STATION_RAILWAY = {"station", "halt", "tram_stop"}

# How far a stop_position may be from the station it belongs to.  Same name: generous, because
# a long platform's stop marker can sit far from the station node.  No name to match on: tight,
# because then proximity is the only evidence and neighbouring stations can be close.
NAME_RADIUS_M = 1200
BLIND_RADIUS_M = 250

# Two station records with the same name closer than this are one station mapped twice.
DUP_RADIUS_M = 500

# Which station record to keep when several describe one station.
STATION_RANK = {"station": 0, "halt": 2, "tram_stop": 3}
# A public_transport=station with one of these set to yes is a rail station.
RAIL_MODES = ("train", "subway", "light_rail", "monorail", "tram", "funicular")

# How far a stop node may sit off the route's own path before we stop believing the match.
STOP_SNAP_M = 400


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    dy = (lat2 - lat1) * 110570
    return math.hypot(dx, dy)


def path_length_m(coords):
    if len(coords) < 2:
        return 0.0
    c = np.asarray(coords, dtype=np.float64)
    lat = np.radians((c[:-1, 1] + c[1:, 1]) / 2)
    dx = np.diff(c[:, 0]) * np.cos(lat) * 111320
    dy = np.diff(c[:, 1]) * 110570
    return float(np.hypot(dx, dy).sum())


class Pts(list):
    """A section's geometry as written to geom/, carrying the OSM node id of each point in
    `ids` (-1 for a point made up where a border cuts a segment; None for a straight line
    drawn over a gap), so ownership.py can find the ways a section runs over exactly, by node
    ids. A plain list to everything else: json writes it as one, and merge_sources moves the
    same object under its new key, so the ids follow it."""
    __slots__ = ("ids",)

    def __init__(self, pts, ids=None):
        super().__init__(pts)
        self.ids = None if ids is None else np.asarray(ids, dtype=np.int64)


def load(region, log):
    d = ROOT / "data" / "proc" / region
    with open(d / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(d / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    with open(d / "stops.pkl", "rb") as f:
        stops = pickle.load(f)
    c = np.load(d / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
    log(f"{len(ways)} ways, {len(rels)} relations, {len(stops)} stops")
    return ways, rels, stops, cid, cx, cy


class Coords:
    """Node id to lon/lat, by binary search over the sorted id array."""

    def __init__(self, cid, cx, cy):
        self.id, self.x, self.y = cid, cx, cy

    def get(self, nid):
        i = np.searchsorted(self.id, nid)
        if i >= self.id.size or self.id[i] != nid:
            return None
        return float(self.x[i]) / 1e7, float(self.y[i]) / 1e7

    def many(self, nids):
        pos = np.searchsorted(self.id, nids)
        np.clip(pos, 0, self.id.size - 1, out=pos)
        ok = self.id[pos] == nids
        return pos, ok


# A hyphen or dash (‐ ‑ ‒ – — ― −) with any spaces round it, as one "-": OSM Paris maps
# "Charles de Gaulle-Étoile" and "Charles de Gaulle — Étoile" 49 m apart. Measured
# 2026-10-02: newly merges fr 6, lu 1, pl 1 station records, all the same station.
DASHES = re.compile(r"\s*[-‐-―−]\s*")


def fold_dashes(name):
    return DASHES.sub("-", name or "")


def merge_duplicate_stations(stations):
    """Collapse station records that are the same station twice.

    OSM frequently carries BOTH a railway=station node and a public_transport=station node
    for one station, and separate nodes per operator at a shared interchange.  Mapping stop
    positions onto stations does not catch this, because both records already look like
    stations, so nothing ever merged them: the Yamanote came out with 31 stations against a
    real 30 and its strip diagram opened with Shinagawa listed twice in a row.

    Same name and within `DUP_RADIUS_M` is the test.  Different operators' platforms under
    one name -- JR and Odakyu at Shinjuku -- merge deliberately: to a rider that is one
    station.  Genuinely different stations sharing a name are far enough apart to survive.
    """
    by_name = defaultdict(list)
    for nid, s in stations.items():
        if s["name"]:
            by_name[fold_dashes(s["name"])].append(nid)

    alias = {}
    for name, ids in by_name.items():
        if len(ids) < 2:
            continue
        remaining = list(ids)
        while remaining:
            seed = remaining.pop(0)
            cluster, rest = [seed], []
            for other in remaining:
                a, b = stations[seed], stations[other]
                if dist_m(a["lon"], a["lat"], b["lon"], b["lat"]) <= DUP_RADIUS_M:
                    cluster.append(other)
                else:
                    rest.append(other)
            remaining = rest
            if len(cluster) < 2:
                continue
            # Keep the most station-like record, then one with an English name, then the
            # lowest id so the choice is the same on every build.
            rep = min(cluster, key=lambda n: (stations[n]["_rank"],
                                              0 if stations[n]["name_en"] else 1, n))
            for c in cluster:
                if c == rep:
                    continue
                if stations[c]["name_en"] and not stations[rep]["name_en"]:
                    stations[rep]["name_en"] = stations[c]["name_en"]
                alias[c] = rep
                del stations[c]
    return alias


def build_stations(stops, rels, coords, log):
    """One record per real station, and a map from every stop node to the station it serves.

    Route relations point at stop_position nodes, which exist one per platform track, so a
    six-platform station is six different node ids in six different relations.  Left alone,
    "Tokyo" becomes six stations and no line ever shares a station with another.  They are
    collapsed by name first and proximity second.
    """
    stations = {}
    for nid, (tags, lon, lat) in stops.items():
        # A public_transport=station only counts when something on it says rail: Hong Kong
        # maps bus termini that way ("Eternal East Bus", "奧運站 Olympic Station"), and with
        # the MTR station itself mapped as an area, the blind pass below took them.
        if (tags.get("railway") in STATION_RAILWAY
                or (tags.get("public_transport") == "station"
                    and (tags.get("railway") or tags.get("station")
                         or any(tags.get(m) == "yes" for m in RAIL_MODES)))):
            stations[nid] = {
                "id": f"n{nid}", "name": tags.get("name") or "",
                "name_en": tags.get("name:en") or "",
                "lon": lon, "lat": lat, "lines": set(),
                "_rank": (STATION_RANK.get(tags.get("railway"), 3)
                          if tags.get("railway") else 1),
                "_tram": tags.get("railway") == "tram_stop",
            }

    alias = merge_duplicate_stations(stations)

    by_name = defaultdict(list)
    for nid, s in stations.items():
        if s["name"]:
            by_name[fold_dashes(s["name"])].append(nid)
    ids = np.fromiter(stations.keys(), dtype=np.int64)
    pos = np.array([[stations[i]["lon"], stations[i]["lat"]] for i in ids]) if ids.size \
        else np.zeros((0, 2))
    tram_rec = np.array([bool(stations[i].get("_tram")) for i in ids], dtype=bool)

    # Only stop nodes a route relation actually uses need resolving.
    used = set()
    for tags, members in rels.values():
        if tags.get("type") != "route":
            continue
        used.update(stop_members(members))

    resolved, invented, unplaced = {}, 0, 0
    for nid in used:
        if nid in alias:
            resolved[nid] = alias[nid]
            continue
        if nid in stations:
            resolved[nid] = nid
            continue
        rec = stops.get(nid)
        if rec is None:
            unplaced += 1
            continue
        tags, lon, lat = rec
        name = tags.get("name") or ""
        best, best_d = None, None
        for cand in by_name.get(fold_dashes(name), ()) if name else ():
            d = dist_m(lon, lat, stations[cand]["lon"], stations[cand]["lat"])
            if d <= NAME_RADIUS_M and (best_d is None or d < best_d):
                best, best_d = cand, d
        if best is None and ids.size:
            dx = (pos[:, 0] - lon) * math.cos(math.radians(lat)) * 111320
            dy = (pos[:, 1] - lat) * 110570
            dd = np.hypot(dx, dy)
            # A metro or train stop never lands on a tram stop by proximity alone: at
            # Admiralty the nearest record was the tram stop "金鐘港鐵站", 183 m off.
            if tags.get("railway") != "tram_stop" and tags.get("tram") != "yes":
                dd = np.where(tram_rec, np.inf, dd)
            j = int(np.argmin(dd))
            if dd[j] <= BLIND_RADIUS_M:
                best = int(ids[j])
        if best is None:
            # No station node anywhere near: the stop node IS the station. Common for tram
            # stops and rural halts mapped only as a stop_position.
            stations[nid] = {
                "id": f"n{nid}", "name": name, "name_en": tags.get("name:en") or "",
                "lon": lon, "lat": lat, "lines": set(),
            }
            best = nid
            invented += 1
            # Its twin on the other track, of the same name, is this station too: the Kiato -
            # Aigio halts are mapped only as two stop_positions a few metres apart (Ελίκη 3 m,
            # Ακράτα 12 m), and each became a station of its own. Measured on every country
            # 2026-10-01: only true twins merge (Shanghai Metro lines were up to 27% long from
            # doubled stops, line 151 counted Pougny-Chancy - Russin twice, Görlitz was 4).
            if name:
                by_name[fold_dashes(name)].append(nid)
        resolved[nid] = best

    log(f"{len(stations)} stations ({len(alias)} duplicate records merged away, "
        f"{invented} created from a bare stop node), {len(resolved)} stop nodes resolved, "
        f"{unplaced} had no coordinate")
    return stations, resolved


def assemble(members, ways, coords):
    """Roleless way members, in relation order, joined into one or more coordinate runs.

    PTv2 says a route's ways are in travel order, so this walks them in order and only has to
    decide each way's DIRECTION, by which end touches the run so far.  Where the next way does
    not touch, the route has a gap -- a missing way, a members list never sorted -- and a new
    run is started rather than inventing a join.  Each run keeps its node ids alongside the
    coordinates, because that is how a stop node is located exactly rather than by proximity.
    """
    runs = []
    cur_n, cur_c = [], []
    for ty, ref, role in members:
        if ty != "w" or (role and not role.startswith(("forward", "backward"))):
            continue
        w = ways.get(ref)
        if w is None:
            continue
        nodes = list(w[1])
        if len(nodes) < 2:
            continue
        if not cur_n:
            cur_n, cur_c = nodes, None
            continue
        # Orient: the new way must start where the run ends.
        if cur_n[-1] == nodes[0]:
            cur_n.extend(nodes[1:])
        elif cur_n[-1] == nodes[-1]:
            cur_n.extend(reversed(nodes[:-1]))
        elif len(cur_n) == len(set(cur_n)) and cur_n[0] == nodes[0]:
            cur_n.reverse()                       # first way was laid down backwards
            cur_n.extend(nodes[1:])
        elif len(cur_n) == len(set(cur_n)) and cur_n[0] == nodes[-1]:
            cur_n.reverse()
            cur_n.extend(reversed(nodes[:-1]))
        else:
            runs.append(cur_n)
            cur_n = nodes
    if cur_n:
        runs.append(cur_n)

    out = []
    for nodes in runs:
        arr = np.asarray(nodes, dtype=np.int64)
        p, ok = coords.many(arr)
        if ok.sum() < 2:
            continue
        xy = np.column_stack([coords.x[p[ok]] / 1e7, coords.y[p[ok]] / 1e7])
        out.append((arr[ok], xy))
    return out


def place_stations(runs, station_nodes, stations):
    """EVERY station of the line, in the order this variant's path passes it.

    NOT just the stations this variant stops at, and that distinction is the whole reason
    this function exists.  A limited-stop variant lists only its calling points, so taking
    its consecutive pairs as sections gives Tokyo->Shinagawa as ONE section while the local
    variant gives the same track as three.  Union those and the shared track is counted
    twice at two different granularities: the Tokaido Main Line came out at 5,719 km against
    a published 590 before this, because every express pattern re-counted the whole corridor.

    So each variant's path is walked looking for all of the line's stations, which are its
    stop nodes pooled over every variant.  An express's path physically runs through the
    local stations, so the sections it yields are the fine ones, and the union across
    variants is then a partition of the line's track rather than a pile of overlaps.
    """
    node2st = {n: st for st, nodes in station_nodes.items() for n in nodes}
    if not node2st:
        return []
    allnodes = np.fromiter(node2st.keys(), dtype=np.int64)
    out, placed = [], set()
    for r, (ids, xy) in enumerate(runs):
        for j in np.flatnonzero(np.isin(ids, allnodes)):
            st = node2st[int(ids[j])]
            out.append((r, int(j), st))
            placed.add(st)

    # A station whose stop node is not a vertex of this path -- mapped beside the track
    # rather than on it -- is snapped by distance instead, once, to its nearest vertex.
    for st in station_nodes:
        if st in placed:
            continue
        s = stations.get(st)
        if s is None:
            continue
        best = None
        for r, (ids, xy) in enumerate(runs):
            dx = (xy[:, 0] - s["lon"]) * math.cos(math.radians(s["lat"])) * 111320
            dy = (xy[:, 1] - s["lat"]) * 110570
            dd = np.hypot(dx, dy)
            j = int(np.argmin(dd))
            if best is None or dd[j] < best[2]:
                best = (r, j, float(dd[j]))
        if best and best[2] <= STOP_SNAP_M:
            out.append((best[0], best[1], st))

    out.sort(key=lambda t: (t[0], t[1]))
    dedup = []
    for e in out:
        if dedup and dedup[-1][2] == e[2]:
            continue                      # the same station twice running is one visit
        dedup.append(e)
    return dedup


def slice_path(runs, a, b):
    """Coordinates AND node ids between two located stops, or None if not on one run.

    The node ids are what makes a section's track identifiable.  Two lines running over the
    same rails traverse the same OSM nodes, so the unordered set of adjacent node pairs is a
    fingerprint of the physical track, and it is the same fingerprint whichever direction the
    line runs in.  Parallel tracks -- the Yamanote and the Keihin-Tohoku between the same two
    stations -- are separate ways with separate nodes, so they do NOT collide, which is right:
    riding one is not riding the other.
    """
    if a is None or b is None or a[0] != b[0]:
        return None
    ids, xy = runs[a[0]]
    lo, hi = sorted((a[1], b[1]))
    if hi - lo < 1:
        return None
    return xy[lo:hi + 1], ids[lo:hi + 1]


def track_key(ids):
    """A fingerprint of the rails a section runs on, direction-independent and stable.

    blake2b over the sorted pairs rather than Python's hash(): hash randomisation differs
    per process, and two builds of identical input have to produce identical files.
    """
    pairs = sorted((int(a), int(b)) if a < b else (int(b), int(a))
                   for a, b in zip(ids[:-1], ids[1:]))
    h = hashlib.blake2b(digest_size=8)
    for a, b in pairs:
        h.update(a.to_bytes(8, "little"))
        h.update(b.to_bytes(8, "little"))
    return h.hexdigest()


def group_lines(rels, log):
    """Route relations gathered into lines: by route_master where there is one, else by name.

    route_master is the mapper's own statement that these variants are one line, so it is
    always preferred.  The orphans fall back to (operator, network, ref, name), which is
    weaker -- two genuinely different lines sharing a name and operator would merge -- but
    still better than treating every direction as its own line.
    """
    masters, routes = {}, {}
    for rid, (tags, members) in rels.items():
        if tags.get("type") == "route_master":
            masters[rid] = (tags, members)
        elif tags.get("route") in ROUTE_KINDS:
            routes[rid] = (tags, members)

    groups, claimed = [], set()
    for mid, (tags, members) in masters.items():
        kids = [r for ty, r, _ in members if ty == "r" and r in routes]
        if not kids:
            continue
        groups.append((f"m{mid}", tags, kids))
        claimed.update(kids)

    by_key = defaultdict(list)
    for rid, (tags, _m) in routes.items():
        if rid in claimed:
            continue
        key = (tags.get("operator", ""), tags.get("network", ""),
               tags.get("ref", ""), tags.get("name", ""))
        by_key[key].append(rid)
    for (op, net, ref, name), rids in by_key.items():
        tags = dict(routes[rids[0]][0])
        tags.setdefault("operator", op)
        groups.append((f"r{min(rids)}", tags, rids))

    log(f"{len(groups)} lines from {len(masters)} route_master and {len(routes)} routes "
        f"({len(routes)-len(claimed)} orphan routes grouped by name)")
    return groups, routes


def looks_like_service(tags, kind, region):
    """Is this a named TRAIN rather than a LINE?

    OpenStreetMap in Japan maps both, as route relations, with no tag that separates them:
    東海道本線 (a line) and のぞみ (a train that runs over the Tokaido Shinkansen) are the
    same kind of object.  So Japan's lines total 49,500 km against a real passenger network
    of about 27,500 -- the Shinkansen corridors are counted once as lines and again under
    every named service over them.

    The rule is deliberately crude and region-specific: a Japanese line name almost always
    ends in 線, and a train's does not.  Metro, tram, light rail and monorail are never
    services, so only route=train is tested.  It misfires on operators whose line is named
    without 線 -- 嵯峨野観光鉄道 is a railway, not a train -- so this sets a FLAG and drops
    nothing.  What the flag is for is deciding which lines a completion percentage counts,
    and that is a judgement about what the hobby means, not a data question.

    Operating patterns (京浜東北線, 中央線快速) are a third category this does not catch, as
    they are named with 線 and are lines to a rider but not to the operator's line register.
    """
    if kind != "train":
        return False
    name = tags.get("name", "")
    name_en = tags.get("name:en", "")
    if region == "jp":
        # 線 is the usual suffix, but an operating pattern introduced since the war is as
        # likely to be ライン: 上野東京ライン and 湘南新宿ライン are lines a rider rides, not
        # trains, and were being flagged as trains and sorted to the bottom of every list.
        if "線" in name or "ライン" in name:
            return False
        return "line" not in name_en.lower()
    if region == "kr":
        # Korea's lines and operating patterns are named 선 (경부선, 경의·중앙선) and so are
        # its trains ("경부선 KTX: 서울 → 부산"), so the suffix says nothing. The train brand
        # does: every named train carries one, and no line or pattern does.
        return any(b in name for b in KR_TRAIN_BRANDS)
    if region == "tw":
        # OSM in Taiwan maps every THSR and some TRA trains by train number, one relation
        # each ("台灣高鐵 821 南港->左營"). A line's name carries no train number.
        return bool(TRAIN_NUMBER.search(name))
    if region == "fr":
        # OSM France maps each long-distance train (TGV 723, Intercités 3731, Ouigo TC 4071,
        # Eurostar, Lyria) as its own relation. TER, Transilien and RER relations are lines.
        # An unnamed relation is called by its ref, and judged by it too: route 5945159
        # ("Ouigo", Marne-la-Vallée - Lyon) has only ref=Ouigo.
        return bool(FR_TRAIN_BRAND.match(name or tags.get("ref") or ""))
    if region == "cn":
        # OSM China maps single trains by number: "D2661西安北-西宁", "K27/28", "Z164/5：上海 ->
        # 拉萨", "6072：宝鸡 -> 平凉", or no name and a ref "C8600". Lines and patterns carry no
        # number up front (北京市郊铁路S2线, 金山铁路, 广清城际).
        return bool(CN_TRAIN.match(tags.get("name") or tags.get("ref") or ""))
    if region == "pt":
        # CP's long-distance products (Alfa Pendular, Intercidades) and the Celta to Vigo are
        # named trains; Regional, InterRegional and the Urbanos of Lisbon and Porto are lines.
        return bool(PT_TRAIN.match(name))
    if region == "hu":
        # OSM Hungary maps each international and InterCity train as its own relation (IC 929
        # Savaria, EC 173, EN 462, ICE 90, "Hungaria EuroCity"). S, G, Z, Sz, R/REX and the
        # InterRégió patterns (IR87 AGRIA, KISKUN IR, IR CÍVIS) run every hour or two: lines.
        return bool(HU_TRAIN.search(name))
    if region == "pl":
        # PKP Intercity maps each train as its own relation ("IC1213 Czechowicz: Warszawa
        # Wschodnia => Lublin Główny", "EIP: Kraków Główny <=> Gdynia Główna", "TLK 38190
        # Bursztyn", "EC 57 Wawel"). Polregio's IR and the regional/agglomeration lines
        # (Linia K5, S1, RE, ŁKA, SKM) carry no such brand: lines.
        return bool(PL_TRAIN.search(name))
    if region == "fi":
        # OSM Finland maps VR's interval patterns as lines ("Juna 13: Helsinki => Oulu", the
        # commuter letters R, Z, H), and single trains by number: the night trains "Juna PYO
        # 273: Helsinki => Rovaniemi" and the Parikkala - Savonlinna "Taajamajuna 751".
        return bool(FI_TRAIN.search(name))
    if region == "ro":
        # OSM Romania maps CFR Călători's, Regio's and Transferoviar's trains one relation per
        # train ("IR 1582 Constanța => București Nord", "R 3127 Arad => Brad", "Tren R9132/4:
        # Calafat - Craiova", or no name and ref "R-E 9263"); Romanian trains are known by
        # number and none runs as a branded interval line. The airport and Obor shuttles
        # (service=commuter), MÁV's "Sz: Debrecen => Valea lui Mihai" patterns and unnumbered
        # relations stay lines.
        if tags.get("service") == "commuter":
            return False
        return bool(RO_TRAIN.search(name) or RO_TRAIN.search(tags.get("ref") or ""))
    if region == "hr":
        # OSM Croatia groups HŽPP's regional trains under their timetable line number,
        # "Vlak 23" (Vlak 2300 Kloštar => Zagreb, 2301, ...): lines. Its fast and long-distance
        # trains are mapped one train or one pair per route_master: "Vlak B 182" (brzi, Split -
        # Zagreb), "IC 58 Podravka", "ICN 52", "Vlak B 188 Dalmacija", "EuroNight Lisinski".
        return bool(HR_TRAIN.search(name) or EU_TRAIN.search(name))
    if region == "ru":
        # OSM Russia maps every long-distance train one relation per train, by its number of
        # three digits and a letter: "Скорый поезд 124Ы: Красноярск → Абакан", "Высокоскоростной
        # поезд 752А «Сапсан»", "Скорый электропоезд 839В «Ласточка»" (ref "124Ы"). Suburban
        # trains ("Пригородный электропоезд: Дубна => Савёловский вокзал", МЦД-2, Novosibirsk's
        # numbered "Пригородный электропоезд 6323") are lines.
        ref = tags.get("ref") or ""
        return bool(RU_TRAIN.search(name) or RU_TRAIN.search(ref)
                    or RU_TRAIN_PAIR.search(name) or RU_TRAIN_PAIR.search(ref)
                    or RU_TRAIN_KIND.match(name))
    if region == "it":
        # OSM Italy maps Trenitalia's and Italo's long-distance brands route by route, with
        # no train number ("Frecciarossa (Milano Centrale → Napoli Centrale)", ".italo (Roma
        # Ostiense → Milano Porta Garibaldi)", "Frecciabianca (Roma Termini - Genova Piazza
        # Principe)", "InterCity Milano-Ventimiglia"), and the EuroCity, Nightjet, TGV and
        # European Sleeper trains one relation each. Regionale (R), Regionale Veloce (RV),
        # RegioExpress (RE), the Leonardo Express and the suburban S, FL, SFM and FM lines
        # are lines.
        return bool(IT_TRAIN.search(name) or EU_TRAIN.search(name))
    if region == "es":
        # OSM Spain maps Renfe's long-distance products and the open-access operators one
        # relation per train or train pair ("Alvia 00194 Madrid → Badajoz", "AVE Madrid -
        # Sevilla", "Train Iryo ...", "Intercity 00283 Irun → A Coruña", "Renfe-SNCF 9736",
        # "Train IN: Porto - Campanhã → Vigo-Guixar"). Cercanías and Rodalies (C-1, R2),
        # Media Distancia, Regional, Avant and the FGC, Euskotren, FGV and SFM lines are lines.
        # The network decides too, but only networks that hold nothing else: "Renfe Alvia" is
        # also on a Bilbao Cercanías C-3 relation, bare "TGV" on a liO TER one. 80 of 527 train
        # relations (2026-10-02 extract).
        return bool(ES_TRAIN.match(name) or EU_TRAIN.search(name)
                    or tags.get("network", "") in ES_TRAIN_NETWORKS)
    if region == "de":
        # OSM Germany maps DB Fernverkehr's ICE and IC by DB's own line number, not by train:
        # 27 "ICE 10"-style and 20 "IC 26"-style route_masters, each an hourly or two-hourly
        # interval product a rider uses as a line, as Swiss IC 1 and ÖBB's Railjet are: lines.
        # Single trains are named trains: EC, EN, NJ, European Sleeper, Eurostar, TGV
        # (EU_TRAIN), an ICE or IC by a train number of 3-5 digits, PKP Intercity's named IC
        # (IC Łużyce), Leo Express and KD Premium.
        if DE_LINE.match(name):
            return False
        return bool(EU_TRAIN.search(name) or DE_TRAIN.search(name))
    if region in EU_TRAIN_REGIONS:
        # International and long-distance trains mapped one relation per train (EC 112, EN
        # 40467, ICE 43, Eurostar, European Sleeper, Nightjet), as France, Poland, Hungary and
        # Portugal flag theirs. The interval products that are lines to a rider, IC, IR and
        # Railjet (Swiss IC 1, ÖBB's half-hourly Railjet), are left as lines.
        return bool(EU_TRAIN.search(name))
    return False


FI_TRAIN = re.compile(r"\bPYO\s?\d|^Taajamajuna\s+\d")
RU_TRAIN = re.compile(r"(?<![0-9A-Za-zА-Яа-яЁё])\d{3}\s?[А-ЯЁA-Z](?![0-9A-Za-zА-Яа-яЁё])")
# A pair numbered without its letter ("Скорый поезд 001/002 «Красная стрела»", ref 001/002),
# and long-distance trains named by their kind and no number ("Скоростной поезд «Аврора»").
RU_TRAIN_PAIR = re.compile(r"(?<![0-9A-Za-zА-Яа-яЁё])\d{3}\s?[А-ЯЁA-Z]?/\d{3}(?!\d)")
RU_TRAIN_KIND = re.compile(r"(?:Скорый|Скоростной|Высокоскоростной|Пассажирский|Фирменный)"
                           r"\s+поезд\b")
RO_TRAIN = re.compile(r"\b(?:R|R-E|RE|IR|IRN|IC|INT|EC|EN|ICN)\s?-?\s?\d{2,5}\b")
HR_TRAIN = re.compile(r"^(?:Vlak\s+)?(?:B|IC|ICN|EC|EN)\s?\d")
IT_TRAIN = re.compile(r"^(?:Treno\s+|Train\s+)?(?:Freccia(?:rossa|argento|bianca)|\.?[Ii]talo\b"
                      r"|Inter[Cc]ity\b|ICN?\s?\d)")
ES_TRAIN = re.compile(r"^(?:Train\s+|Tren\s+)?(?:AVE|AV City|Alvia|ALVIA|Avlo|AVLO|Euromed|"
                      r"Intercity|InterCity|Intercités|Iryo|IRYO|Ouigo|OUIGO|Trenhotel|Talgo|TLG|"
                      r"Renfe-SNCF|TGV|IN)(?=[\s\d:]|$)")
ES_TRAIN_NETWORKS = {"Renfe AVE", "Iryo", "Ouigo España", "OUIGO España", "Renfe InterCity",
                     "TGV Europe"}
EU_TRAIN_REGIONS = {"at", "be", "nl", "ch", "cz", "si", "bg", "sk"}
# "ICE" followed by a one- or two-digit number is a DB interval line ("ICE 43", "ICE 91"), the
# same route_master built in Germany, where it is a line (DE_LINE); not a single train.
EU_TRAIN = re.compile(r"^(?:Train\s+)?(?:EC|EN|ICE(?!\s?\d{1,2}(?:\.\d)?(?!\d))|NJ|TGV|ES|ECE"
                      r"|INT)(?:[\s\d:]|$)"
                      r"|\bEuro(?:City|Night)\b|\bNightjet\b|\bEuropean Sleeper\b"
                      r"|^(?:Eurostar|Thalys|TGV Lyria|Lyria)\b")
# Germany: "ICE 10", "IC 26.1", "ICE 42/ICE 47" are DB's interval lines; "ICE 1001",
# "IC Łużyce" single trains.
DE_LINE = re.compile(r"^(?:ICE|IC)\s?\d{1,2}(?:\.\d)?(?!\d)")
DE_TRAIN = re.compile(r"^(?:ICE|IC)\s?\d{3,5}\b|^IC\s+[A-ZÀ-ŽŁ][a-ząćęłńóśźż]"
                      r"|^Leo Express\b|^KD Premium\b")


# "EIP:" is written with a colon straight after the brand.
PL_TRAIN = re.compile(r"^(?:EIC|EIP|IC|TLK|EC|EN|ICE|RJX?|NJ)(?:[\s\d:]|$)"
                      r"|\bEuro(?:City|Night)\b|\bRailjet\b")


# "Train EC Hornád: Budapest => Košice" is mapped the Slovak way, with "Train " in front.
HU_TRAIN = re.compile(r"^(?:Train\s+)?(?:IC|EC|EN|ICE|RJX?)(?:\s|\d|$)"
                      r"|\bEuro(?:City|Night)\b|\bRailjet\b")


PT_TRAIN = re.compile(r"^(?:CP )?(?:Alfa Pendular|Intercidades)\b|^Comboio Celta|^Train IN\b")


CN_TRAIN = re.compile(r"^(?:火车|Train\s*)?[GDCZTKYLSP]?\d{1,5}(?:/[A-Z]?\d{1,5})?(?![0-9号線线])")


# Case-sensitive, so "ICE" is the German train and not a word that starts "Ice".
FR_TRAIN_BRAND = re.compile(r"^(TGV|OUIGO|Ouigo|OUIGo|Eurostar|Lyria|ICE|Intercités|"
                            r"INTERCITÉS|ICN?\s|Renfe|RENFE|Frecciarossa|Nightjet|Thalys|"
                            r"Train de nuit)")


# A free-standing train number: 821 in "台灣高鐵 821", 1 in "のぞみ1号". Not the 1 of "S1" or
# "S11", whose numbers are the line's name.
TRAIN_NUMBER = re.compile(r"(?<![A-Za-z0-9])\d{1,4}(?![A-Za-z0-9])")


KR_TRAIN_BRANDS = ("KTX", "SRT", "ITX", "새마을", "무궁화", "누리로", "직통열차", "마음")


def pretty_line_name(name):
    """What to call a register line on screen.

    N02 stores a subway as 4号線丸ノ内線 or 1号線(御堂筋線), which is its statutory name and
    not what anyone calls it. The number is dropped for display; the line is still keyed on
    the pair it came in as.
    """
    s = (name or "").strip()
    m = re.match(r"^第?[0-9０-９]+号線[（(](.+?)[）)]$", s)
    if m:
        return m.group(1)
    m = re.match(r"^第?[0-9０-９]+号線(.+)$", s)
    if m:
        return m.group(1)
    return s


CARRY_SAME_NAME_M = 500
CARRY_ANY_M = 200


def carry_aliases(out, st, station_alias, log, abroad=None):
    """Keep every station id the LAST build shipped reachable from this one.

    aliases.json used to hold only this build's own merges, so an id that simply stopped
    existing between builds was lost, and with it any saved ride naming it: a change to how
    stop nodes resolve (a bus-terminal record at 西鉄福岡 giving way to the rail station)
    does exactly that. So every id in the previous stations.json or aliases.json that this
    build neither ships nor aliases is mapped on: through its old alias if that target
    still exists, else to a station of the same name within CARRY_SAME_NAME_M, else to the
    nearest within CARRY_ANY_M. Chains are resolved to a live id.

    `abroad` ({id: the neighbour's id}, split_at_borders) are stations now left to a built
    neighbour: their targets count as live though this build does not ship them, so an older
    alias chain through one ends at the neighbour's station rather than being dropped."""
    prev_st, prev_al = {}, {}
    try:
        with open(out / "stations.json", encoding="utf-8") as f:
            prev_st = json.load(f)["stations"]
        with open(out / "aliases.json", encoding="utf-8") as f:
            prev_al = json.load(f)["stations"]
    except (OSError, ValueError, KeyError):
        pass
    abroad = {k: v for k, v in (abroad or {}).items() if k not in st}
    alias = {**station_alias, **abroad}
    ext = set(abroad.values())
    if not prev_st:
        return alias

    # Never onto a junction: nobody gets on or off at one, and a station a border cut left to
    # the neighbour can lie within reach of the border point that replaced it.
    ids = [i for i in st if not st[i].get("j")]
    pos = np.array([[st[i]["x"], st[i]["y"]] for i in ids]) if ids else np.zeros((0, 2))
    by_name = defaultdict(list)
    for i in ids:
        by_name[st[i]["n"]].append(i)

    def live(t):
        seen = set()
        while t not in st and t not in ext and t in alias and t not in seen:
            seen.add(t)
            t = alias[t]
        return t if t in st or t in ext else None

    carried, lost = 0, []
    for old in set(prev_st) | set(prev_al):
        if old in st or live(old):
            continue
        t = live(prev_al.get(old, old))
        if t is None:
            rec = prev_st.get(old) or prev_st.get(prev_al.get(old, ""))
            if rec is None or not ids:
                lost.append(old)
                continue
            x, y = rec["x"], rec["y"]
            cands = [(dist_m(x, y, st[c]["x"], st[c]["y"]), c) for c in by_name.get(rec["n"], ())]
            cands = [c for c in cands if c[0] <= CARRY_SAME_NAME_M]
            if cands:
                t = min(cands)[1]
            else:
                dd = np.hypot((pos[:, 0] - x) * math.cos(math.radians(y)) * 111320,
                              (pos[:, 1] - y) * 110570)
                j = int(np.argmin(dd))
                t = ids[j] if dd[j] <= CARRY_ANY_M else None
        if t is None:
            lost.append(old)
        else:
            alias[old] = t
            carried += 1
    # Every alias points at a live id, so the app needs one lookup, not a walk.
    for k in list(alias):
        t = live(alias[k])
        if t is None:
            del alias[k]
        else:
            alias[k] = t
    log(f"{carried} station ids from the last build carried over as aliases, "
        f"{len(lost)} with nothing within reach{': ' + ', '.join(sorted(lost)[:10]) if lost else ''}")
    return alias


BILINGUAL_NAME = re.compile("^([%s-%s%s-%s]+)\\s+[A-Za-z]" % (
    chr(0x3400), chr(0x9FFF), chr(0xF900), chr(0xFAFF)))


def norm_line_name(name, operator=""):
    """A line name in a form the two sources can be compared in.

    N02 writes 東海道線 and 山陰線 where OSM writes JR東海道本線 and JR山陰本線, so the
    operator prefix and the 本 of 本線 both have to go. Deliberately NOT a substring test:
    京浜東北線 contains 東北線, and an operating pattern must not be mistaken for the
    register line it runs over.
    """
    s = (name or "").strip()
    # A subway in the register is its statutory number and then its common name:
    # 4号線丸ノ内線, 1号線(御堂筋線), 12号線大江戸線. The common name is the one anyone
    # uses, and stripping parentheses first would have thrown exactly it away.
    m = re.match(r"^第?[0-9０-９]+号線[（(](.+?)[）)]$", s)
    if m:
        s = m.group(1)
    else:
        m = re.match(r"^第?[0-9０-９]+号線(.+)$", s)
        if m:
            s = m.group(1)
    s = re.sub(r"[（(\[].*?[）)\]]", "", s).strip()
    # Hong Kong's OSM line names are bilingual in one tag, "港鐵東鐵綫 MTR East Rail Line",
    # where the register has the Chinese half; so are a few in Shikoku ("土讃線 Dosan").
    # Only a name that is all Han up to a space and then goes on in Latin letters: Hangul,
    # kana and "台灣高鐵 821" are untouched. Not when the Han half is the operator, as in
    # "愛知高速交通株式会社 Linimo". The range is built from code points: literal range ends here
    # have been mangled in transit before, and then matched Hangul.
    m = BILINGUAL_NAME.match(s)
    if m and m.group(1) != operator:
        s = m.group(1)
    # Singapore: OSM names its routes "MRT North-South Line" and "LRT Bukit Panjang Line",
    # and its track "North South Line (NS)" and "Thomson–East Coast Line"; LTA writes
    # North-South Line, Thomson-East Coast Line and Bukit Panjang LRT.
    m = re.match(r"^LRT (.+) Line$", s)
    if m:
        s = f"{m.group(1)} LRT"
    # Taiwan's metro systems prefix their lines (台北捷運板南線) where the register writes
    # 板南線; the system names come before bare 捷運.
    for pre in ("JR", "ＪＲ", "東京メトロ", "東京地下鉄", "都営", "Osaka Metro", "大阪市営",
                "名古屋市営", "札幌市営",
                "台北捷運", "臺北捷運", "新北捷運", "桃園捷運", "臺中捷運", "高雄捷運", "捷運", "港鐵",
                "MRT ", "CP Lisboa", "CP Porto", operator):
        if pre and s.startswith(pre):
            s = s[len(pre):]
    s = s.strip(" 　:：・")
    if s.endswith("本線"):
        s = s[:-2] + "線"
    # A hyphen, an en dash and a space between words are one spelling: North-South Line
    # (LTA), North South Line (OSM track), Thomson–East Coast Line (OSM). Applied to both
    # sides, so it can only add a match; on jp, ch, kr and tw it added none.
    s = re.sub(r"\s*[-–]\s*", " ", s)
    return s


# The register and OSM do not call the same company the same thing, and neither string
# contains the other, so the match has to be told.
OP_ALIAS = {
    "東京メトロ": "東京地下鉄", "東京都交通局": "東京都", "都営地下鉄": "東京都",
    "大阪市交通局": "大阪市高速電気軌道", "Osaka Metro": "大阪市高速電気軌道",
    "大阪メトロ": "大阪市高速電気軌道",
    "札幌市交通局": "札幌市", "名古屋市交通局": "名古屋市",
    "神戸市交通局": "神戸市", "京都市交通局": "京都市", "福岡市交通局": "福岡市",
    "仙台市交通局": "仙台市", "横浜市交通局": "横浜市",
}


def same_operator(a, b):
    a, b = (a or "").strip(), (b or "").strip()
    a, b = OP_ALIAS.get(a, a), OP_ALIAS.get(b, b)
    if not a or not b:
        return False
    if a == b:
        return True
    return a in b or b in a


# An OSM line matched to a register line by name is DROPPED as a duplicate only when it also
# measures within this fraction of the register line's length ...
TWIN_KM = 0.06
# ... and shares at least this share of its stations with it (intersection over union).
TWIN_STATIONS = 0.85


# Two OSM lines are one when nearly all of the smaller one's stations are on the other.
OSM_TWIN_STATIONS = 0.9

# An OSM line and a register line with the same name but differently spelled operators are
# still one line if at least this share of the smaller one's stations are shared.
NAME_MATCH_SHARED = 0.5

# Second-pass station matching in merge_sources: a compatible name within LOOSE_NAME_M, or
# any name at all within SAME_PLACE_M.
LOOSE_NAME_M = 250
SAME_PLACE_M = 30


def station_key(name):
    """A station name in the form two sources can be compared in. NFKC folds full-width
    letters and digits; the small ke is written three ways, and 市ケ谷 against 市ヶ谷 kept the
    Toei Shinjuku Line from being recognised as its own register line."""
    s = fold_dashes(unicodedata.normalize("NFKC", name or "")).casefold()
    return s.replace("ヶ", "ケ").replace("ヵ", "ケ").replace("ｹ", "ケ").strip()


_CJK = re.compile(r"[぀-ヿ㐀-鿿豈-﫿가-힯ᄀ-ᇿ・ー々〆]+")


def affix_match(a, b):
    """Two space-free names (Japanese, Chinese, Korean) where one begins or ends with the
    other: OSM's 新線新宿 is the Toei platforms of the register's 新宿. Only used between
    stations already close together, and only for an OSM station that found no register
    station of its own name, so 西武新宿 still goes to 西武新宿."""
    a, b = station_key(a), station_key(b)
    # CJK and Hangul only. In a Latin name a prefix is just a prefix: "Wil" and "Wilen" are
    # two Swiss villages.
    if not a or not b or a == b or not (_CJK.fullmatch(a) and _CJK.fullmatch(b)):
        return False
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 2 and (long_.endswith(short) or long_.startswith(short))


def name_words(name):
    """A name's words, ignoring anything in brackets: Geneva's OSM tram stops carry a platform
    letter, "Plainpalais (C)", that the register's "Genève, Plainpalais" never will."""
    s = re.sub(r"[（(\[].*?[）)\]]", " ", station_key(name))
    return {w for w in re.split(r"[\W_]+", s) if w}


def line_stations(l):
    return {s for sec in l["sections"] for s in sec[:2]}


def is_twin(osm_line, reg_line):
    """Is this OSM line the register line again, rather than the line as operated?

    "Tokyo Metro Ginza Line" and "Tokyo Metro Ginza Line (as operated)" side by side is noise:
    same stations, same 14 km, two rows. The OSM object only earns its place where it really
    differs -- the Yamanote loop is 34.5 km over a 20.6 km register line, and a through
    service reaches stations the register line does not. So both tests have to pass: length
    within TWIN_KM and stations within TWIN_STATIONS. Called after the OSM line's stations
    have been moved onto the register's ids, or no station could ever be shared.
    """
    if not reg_line["km"]:
        return False
    if abs(osm_line["km"] / reg_line["km"] - 1) > TWIN_KM:
        return False
    a, b = line_stations(osm_line), line_stations(reg_line)
    return bool(a | b) and len(a & b) / len(a | b) >= TWIN_STATIONS


# Direction parts of an OSM route name: "(行橋 => 延岡)", "(Shibuya -> Chuo-Rinkan)",
# "（下り）", "(up)", ": 渋谷→押上", and brackets inside them: "(Nishitetsu Fukuoka (Tenjin)
# => Omuta)".
_DIR_MARK = re.compile(r"->|=>|→|⇒|↔|<=>|-->|⇄|上り|下り|^\s*(?:up|down|inbound|outbound)\s*$",
                       re.IGNORECASE)
_DIR_TAIL = re.compile(r"\s*[:：]\s*[^:：]*(?:->|=>|→|⇒|-->|↔|<=>)[^:：]*$")
_OPEN, _CLOSE = "(（", ")）"


def strip_direction(name):
    s = name or ""
    changed = True
    while changed:
        changed = False
        # Each OUTERMOST bracketed group, scanned by depth so nested brackets stay inside it.
        depth, start = 0, None
        for i, ch in enumerate(s):
            if ch in _OPEN:
                if depth == 0:
                    start = i
                depth += 1
            elif ch in _CLOSE and depth:
                depth -= 1
                if depth == 0 and _DIR_MARK.search(s[start + 1:i]):
                    s = (s[:start].rstrip() + " " + s[i + 1:].lstrip()).strip()
                    changed = True
                    break
    return _DIR_TAIL.sub("", s).strip()


def untrained(name):
    """A named train's name with its train number and direction taken out."""
    return re.sub(r"\s+", " ", TRAIN_NUMBER.sub("", strip_direction(name or ""))).strip()


def merge_osm_twins(lines, geoms, log):
    """Drop OSM lines that are another OSM line again, and return {dropped id: kept id}.

    OSM maps some services twice: the Hanzomon Line running through onto the Tokyu
    Den-en-toshi Line is one route_master (48.2 km, 29 stations) and, separately, a set of
    orphan routes under another name (48.3 km, 40 stations). And a line whose two directions
    were never put under a route_master arrives as two lines, "JR日豊本線 (行橋 => 延岡)" and
    "(延岡 => 行橋)". Either way it reads as the list repeating itself.

    Twins: length within TWIN_KM, and nearly all of the smaller one's
    stations on the larger one -- AND a shared ref or the same name once the direction is
    taken out. Without that last test S1 and S11, two Swiss S-Bahn lines over the same stops,
    would merge. The one with the most stations is kept, since its sections are the finest;
    a route_master's name is preferred for it, being the one without a direction in it.
    """
    osm = [l for l in lines if l.get("src", "osm") == "osm"]
    st = {l["id"]: line_stations(l) for l in osm}
    refs = {l["id"]: {r.strip() for r in (l["ref"] or "").split(";") if r.strip()} for l in osm}
    by_st = defaultdict(set)
    for l in osm:
        for s in st[l["id"]]:
            by_st[s].add(l["id"])
    byid = {l["id"]: l for l in osm}
    parent = {l["id"]: l["id"] for l in osm}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for l in osm:
        a = st[l["id"]]
        if not a:
            continue
        seen = set()
        for s in a:
            for o in by_st[s]:
                if o <= l["id"] or o in seen:
                    continue
                seen.add(o)
                m = byid[o]
                # Kind is NOT compared: a through service is tagged subway on its routes and
                # train on its route_master, and the two Hanzomon-Den-en-toshi copies are one
                # of each. Stations and length already say it is the same track.
                if m["service"] != l["service"]:
                    continue
                if not m["km"] or abs(l["km"] / m["km"] - 1) > TWIN_KM:
                    continue
                b = st[o]
                if len(a & b) / min(len(a), len(b)) < OSM_TWIN_STATIONS:
                    continue
                same_name = (strip_direction(l["name"]) == strip_direction(m["name"])
                             and strip_direction(l["name"]))
                # Two named trains that differ only in train number, over the same stops:
                # Taiwan maps each THSR train as its own relation, 106 of them in the list.
                if not same_name and l["service"]:
                    same_name = (untrained(l["name"]) == untrained(m["name"])
                                 and untrained(l["name"]))
                same_en = (strip_direction(l["name_en"]) == strip_direction(m["name_en"])
                           and strip_direction(l["name_en"]))
                if not (refs[l["id"]] & refs[o] or same_name or same_en):
                    continue
                parent[find(o)] = find(l["id"])

    groups = defaultdict(list)
    for l in osm:
        groups[find(l["id"])].append(l)
    alias, drop = {}, set()
    for members in groups.values():
        if len(members) < 2:
            continue
        keep = max(members, key=lambda l: (len(st[l["id"]]), l["id"].startswith("m"), l["km"]))
        master = next((l for l in members if l["id"].startswith("m")), None)
        if master is not None and master is not keep:
            for k in ("name", "name_en", "ref"):
                if master[k]:
                    keep[k] = master[k]
        keep["name"] = strip_direction(keep["name"]) or keep["name"]
        keep["name_en"] = strip_direction(keep["name_en"]) or keep["name_en"]
        # Merged across train numbers: the kept name is none of them.
        if keep["service"] and len({strip_direction(l["name"]) for l in members}) > 1:
            keep["name"] = untrained(keep["name"]) or keep["name"]
            keep["name_en"] = untrained(keep["name_en"]) or keep["name_en"]
        for l in members:
            if l is keep:
                continue
            for k in ("colour", "name_en", "operator_en", "ref"):
                if l[k] and not keep[k]:
                    keep[k] = l[k]
            if l.get("dup"):
                keep["dup"] = True
            # A twin's run to the border is the kept line's too (add_border_sections dedups).
            if l.get("_tails"):
                keep["_tails"] = (keep.get("_tails") or []) + l["_tails"]
            alias[l["id"]] = keep["id"]
            drop.add(l["id"])
            geoms.pop(l["id"], None)
    lines[:] = [l for l in lines if l["id"] not in drop]
    # A line is not a direction. An orphan route kept on its own still says "(Chuo-Rinkan ->
    # Shibuya)" though the line it stands for runs both ways.
    changed_en = set()
    # A two-way mark is not a direction: Greek route_masters are "Τρένο IC: Αθήνα ↔
    # Θεσσαλονίκη", "Τρένο IC: Αθήνα ↔ Καλαμπάκα", "Τρένο IC: Θεσσαλονίκη ↔ Σέρρες", and the
    # ": A ↔ B" is all that tells the three apart. Where taking it off would give lines with
    # different full names one name, a name with a two-way mark keeps it (English too).
    # Measured 2026-10-01: about 300 lines keep theirs, all collisions (Poland's ~80 "R",
    # five Elron, nine "liO TER Occitanie", three Narita Express), none elsewhere moved.
    two_way = re.compile(r"↔|<=>|⇄")
    full_names = defaultdict(set)
    for l in lines:
        if l.get("src", "osm") == "osm":
            full_names[strip_direction(l["name"]) or l["name"]].add(l["name"])
    keep_full = {l["id"] for l in lines if l.get("src", "osm") == "osm"
                 and two_way.search(l["name"] or "")
                 and len(full_names[strip_direction(l["name"]) or l["name"]]) > 1}
    for l in lines:
        if l["id"] in keep_full:
            continue
        if l.get("src", "osm") == "osm":
            l["name"] = strip_direction(l["name"]) or l["name"]
            en = strip_direction(l["name_en"]) or l["name_en"]
            if en != l["name_en"]:
                changed_en.add(l["id"])
            l["name_en"] = en
    # BUT two different lines must not end up with one English name. The 98 km Hanzomon
    # service through to the Tobu line was told apart from the 48 km one only by its
    # direction, "Denentoshi bypass line (Chuo-Rinkan -> Shibuya)". Where stripping makes
    # English names collide across lines whose native names differ, the one that lost a
    # direction drops its English name and shows its native one, which here is accurate:
    # ...東武スカイツリーライン直通運転.
    by_en = defaultdict(list)
    for l in lines:
        if l.get("src", "osm") == "osm" and l["name_en"]:
            by_en[l["name_en"]].append(l)
    for same in by_en.values():
        if len({l["name"] for l in same}) > 1:
            for l in same:
                if l["id"] in changed_en and l["name"]:
                    l["name_en"] = ""
    log(f"merge: {len(drop)} OSM lines dropped as the same line as another OSM line "
        f"(within {TWIN_KM:.0%} on length, {OSM_TWIN_STATIONS:.0%} of stations, same ref "
        f"or name)")
    return alias


def merge_sources(osm, n02, log):
    """Put the register's lines and OSM's on one footing, sharing one station registry.

    THE STATIONS HAVE TO BE ONE SET or the app breaks in a way that looks like missing data:
    clicking Tokyo would list either the register lines or the operating patterns, never
    both, because the two sources name stations with different ids.

    An OSM line that matches a register line by name hands over its colour and English name.
    It is then dropped if it is the same line twice (`is_twin`), and kept, flagged `dup`, if
    it is the line as operated rather than as registered. What else stays from OSM is what the
    register does not have: operating patterns, named trains, and anything N02 missed.
    Dropped ids go to `merge_sources.line_alias` so saved rides can follow them.
    """
    o_lines, o_st, o_geo = osm
    n_lines, n_st, n_geo = n02

    # --- stations: alias OSM stations onto register groups by name, then distance
    by_name = defaultdict(list)
    for sid, s in n_st.items():
        if s["name"]:
            by_name[station_key(s["name"])].append(sid)
    # KEYED BY THE STRING ID the sections use ("n123"), not by the OSM node id the dict is
    # keyed on. They are not the same thing, and getting that wrong meant every alias lookup
    # below missed: no OSM line was remapped, its stations were left out of the registry,
    # and the strip diagram showed raw ids like n3012068698 instead of station names.
    alias, matched = {}, 0
    for _nid, s in o_st.items():
        best, best_d = None, None
        for cand in by_name.get(station_key(s["name"]), ()):
            t = n_st[cand]
            d = dist_m(s["lon"], s["lat"], t["lon"], t["lat"])
            if d <= 900 and (best_d is None or d < best_d):
                best, best_d = cand, d
        if best:
            alias[s["id"]] = best
            matched += 1
            if s["name_en"] and not n_st[best]["name_en"]:
                n_st[best]["name_en"] = s["name_en"]
    # Second pass, for names that are the same station written differently. The Swiss list
    # writes "Zürich, Bellevue", "Malans GR" and "Celerina" where OSM writes "Bellevue",
    # "Malans" and "Celerina/Schlarigna": 738 OSM stations sat beside a register stop under
    # another spelling and each became a second station. Compatible means one name's words
    # are all in the other's, and it has to be close. Japanese names have no spaces, so for
    # them this is still an exact match and the pass changes nothing.
    loose = 0
    if n_st:
        rid = [sid for sid, t in n_st.items() if not t.get("junction")]
        rpos = np.array([[n_st[s]["lon"], n_st[s]["lat"]] for s in rid])
        rwords = [name_words(n_st[s]["name"]) for s in rid]
        for _nid, s in o_st.items():
            if s["id"] in alias:
                continue
            dx = (rpos[:, 0] - s["lon"]) * math.cos(math.radians(s["lat"])) * 111320
            dy = (rpos[:, 1] - s["lat"]) * 110570
            dd = np.hypot(dx, dy)
            w = name_words(s["name"])
            for j in np.argsort(dd)[:5]:
                if dd[j] > LOOSE_NAME_M:
                    break
                if (dd[j] <= SAME_PLACE_M or (w and (w <= rwords[j] or rwords[j] <= w))
                        or affix_match(s["name"], n_st[rid[j]]["name"])):
                    alias[s["id"]] = rid[j]
                    loose += 1
                    break
    log(f"merge: {matched} of {len(o_st)} OSM stations matched a register station group by "
        f"name, {loose} more by a compatible name or position")

    stations = dict(n_st)
    for _nid, s in o_st.items():
        if s["id"] not in alias:
            stations[s["id"]] = s

    # --- lines: match OSM to register by normalised name plus operator
    index = defaultdict(list)
    for l in n_lines:
        index[norm_line_name(l["name"], l["operator"])].append(l)
    keep, dropped = [], 0
    reg_st = {c["id"]: line_stations(c) for c in n_lines}
    for l in o_lines:
        key = norm_line_name(l["name"], l["operator"])
        mine = {alias.get(s, s) for sec in l["sections"] for s in sec[:2]}
        hit = None
        for cand in index.get(key, ()):
            # The operator strings are free text and need not agree even when the lines do:
            # OSM tags the Hanzomon Line "Tokyo Metro" where the register has 東京地下鉄, so
            # it was never matched and the line was listed twice. The same name plus most
            # stations shared is as good as the operator agreeing. The name alone is not:
            # 本線 is a dozen different railways' main lines.
            theirs = reg_st[cand["id"]]
            shared = (len(mine & theirs) / min(len(mine), len(theirs))
                      if mine and theirs else 0.0)
            if (same_operator(l["operator"], cand["operator"]) or not l["operator"]
                    or same_operator(l.get("operator_en"), cand["operator"])
                    or shared >= NAME_MATCH_SHARED):
                hit = cand
                break
        if hit is not None and not l["service"]:
            # Hand over what the register has not got, and KEEP it.
            if l["colour"] and not hit["colour"]:
                hit["colour"] = l["colour"]
            if l["name_en"] and not hit["name_en"]:
                # Without its direction: a register line runs both ways, and 36 in Japan took
                # a variant's "JR Yosan Line (Matsuyama => Iyo-Ōzu)" as their English name.
                hit["name_en"] = strip_direction(l["name_en"]) or l["name_en"]
            if l["operator_en"] and not hit["operator_en"]:
                hit["operator_en"] = l["operator_en"]
            if l["kind"] and l["kind"] != "train":
                hit["kind"] = l["kind"]
            if l["ref"] and not hit["ref"]:
                hit["ref"] = l["ref"]
            # NOT dropped here. The OSM object is often the line as OPERATED where the
            # register line is the line as REGISTERED, and they are different things: the
            # Yamanote runs a 34.5 km loop over a 20.6 km register line. Dropping it lost the
            # loop, which is the thing a rider looks up. It stays, marked, and the register
            # line is the one completion counts -- unless it turns out below to be the same
            # line twice, which is decided on stations and length, not on the name.
            l["dup"] = True
            l["_twin"] = hit
            dropped += 1
        l["src"] = "osm"
        keep.append(l)
    log(f"merge: {dropped} OSM lines matched a register line and handed over their colour, "
        f"{len(keep)} OSM lines kept in total")

    # --- rewrite the kept OSM lines onto the shared station ids
    geoms = dict(n_geo)
    out = list(n_lines)
    line_alias = {}
    for l in keep:
        g = o_geo.get(l["id"], {})
        secs, newg, seen = [], {}, set()
        for a, b, km in [(s[0], s[1], s[2]) for s in l["sections"]]:
            na, nb = alias.get(a, a), alias.get(b, b)
            if na == nb:
                continue                      # both ends collapsed onto one complex
            k = (na, nb) if na <= nb else (nb, na)
            if k in seen:
                continue
            seen.add(k)
            secs.append([na, nb, km])
            pts = g.get(f"{a}|{b}")
            if pts:
                newg[f"{na}|{nb}"] = pts
        if not secs:
            continue
        l["sections"] = secs
        l["display"] = [alias.get(s, s) for s in l["display"]]
        dedup = []
        for s in l["display"]:
            if not dedup or dedup[-1] != s:
                dedup.append(s)
        l["display"] = dedup
        l["km"] = round(sum(s[2] for s in secs), 3)
        hit = l.pop("_twin", None)
        if hit is not None and is_twin(l, hit):
            line_alias[l["id"]] = hit["id"]
            continue
        geoms[l["id"]] = newg
        out.append(l)
    log(f"merge: {len(line_alias)} OSM lines dropped as the same line as their register "
        f"match (within {TWIN_KM:.0%} on length, {TWIN_STATIONS:.0%} of stations shared)")
    line_alias.update(merge_osm_twins(out, geoms, log))

    for l in out:
        for a, b, km in [(s[0], s[1], s[2]) for s in l["sections"]]:
            for s in (a, b):
                if s in stations:
                    stations[s]["lines"].add(l["id"])

    for l in out:
        if l.get("src") == "n02":
            l["name"] = pretty_line_name(l["name"])
    reg = [l for l in out if l.get("src") != "osm"]
    merge_sources.alias = alias
    merge_sources.line_alias = line_alias
    log(f"merge: {len(out)} lines in total, {len(reg)} of them register lines "
        f"({sum(l['km'] for l in reg):,.0f} km), {len(stations)} stations")
    return out, stations, geoms


def track_id(reg, digest):
    """Small integer per distinct piece of physical track.

    A section with no digest is one whose geometry could not be traced along the route and
    fell back to a straight line. It gets an id of its own rather than being pooled with
    every other untraceable section, which would make unrelated track count as shared.
    """
    if digest is None:
        reg[f"straight:{len(reg)}"] = len(reg)
        return len(reg) - 1
    if digest not in reg:
        reg[digest] = len(reg)
    return reg[digest]


class TrackGraph:
    """Rail ways as a graph of OSM nodes, for tracing a section along real track where the
    route relation has a gap.

    471 of Japan's sections fell back to a straight line between two stations because the
    relation's ways did not join up: a missing member, or members in the wrong order. The
    straight line was drawn and CREDITED, so JR中央線快速 had Tokyo to Shinjuku as one 6.1 km
    chord across central Tokyo where the track is 10.3 km. The track is almost always in the
    extract; only the relation fails to say which of it the line uses.
    """

    def __init__(self, way_ids, ways, coords):
        self.adj = defaultdict(list)
        self.xy = {}
        for wid in way_ids:
            w = ways.get(wid)
            if w is None or len(w[1]) < 2:
                continue
            nodes = np.asarray(w[1], dtype=np.int64)
            pos, ok = coords.many(nodes)
            prev = None
            for n, p, good in zip(nodes.tolist(), pos.tolist(), ok.tolist()):
                if not good:
                    prev = None
                    continue
                self.xy[n] = (coords.x[p] / 1e7, coords.y[p] / 1e7)
                if prev is not None and prev != n:
                    d = dist_m(*self.xy[prev], *self.xy[n])
                    self.adj[prev].append((n, d))
                    self.adj[n].append((prev, d))
                prev = n

    def path(self, a, b, cap_m):
        """Shortest node path a -> b no longer than cap_m, as (xy array, node ids), or None."""
        if a not in self.adj or b not in self.adj:
            return None
        dist, prev, seen = {a: 0.0}, {}, set()
        heap = [(0.0, a)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in seen:
                continue
            if d > cap_m:
                return None
            if u == b:
                ids = [b]
                while ids[-1] != a:
                    ids.append(prev[ids[-1]])
                ids.reverse()
                return (np.array([self.xy[n] for n in ids]), np.array(ids, dtype=np.int64))
            seen.add(u)
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        return None


# A gap traced over the line's OWN ways may run up to this multiple of the straight distance
# (its own track is trustworthy); over the whole network, this multiple plus a margin, so a
# gap is never closed by a detour down some other line.
OWN_DETOUR = 3.0
NET_DETOUR, NET_MARGIN_M = 2.0, 2000


def pick(tags, *keys):
    for k in keys:
        v = tags.get(k)
        if v:
            return v
    return ""


# A funicular's end station this close to the end of its track is that end's station.
FUNICULAR_END_M = 150


def funicular_ends(rids, routes, ways, coords, stations, station_nodes):
    """Stations for a funicular route mapped with no stops: the two ends of its track.

    Switzerland's Niesenbahn, Stoosbahn, Gelmerbahn and the Lugano city funicular are route
    relations with ways and no stop members, so they had under two stops, never became lines,
    and once build_tiles dropped track no line runs over they vanished from the map too. A
    funicular has exactly two ends and stops at both, so the ends are its stations. Only
    funiculars: a train route with no stops is as likely a harbour freight line or a theme
    park railway. Returns 1 if it supplied the stations, else 0.
    """
    best = None
    for rid in rids:
        for nodes, xy in assemble(routes[rid][1], ways, coords):
            km = path_length_m(xy) / 1000
            if best is None or km > best[0]:
                best = (km, nodes, xy, routes[rid][0])
    if best is None or best[0] < 0.1:
        return 0
    _km, nodes, xy, tags = best
    ends = [(int(nodes[0]), xy[0], tags.get("from")), (int(nodes[-1]), xy[-1], tags.get("to"))]
    for i, (node, (lon, lat), label) in enumerate(ends):
        near = min(((dist_m(lon, lat, s["lon"], s["lat"]), sid) for sid, s in stations.items()),
                   default=(math.inf, None))
        if near[0] <= FUNICULAR_END_M:
            station_nodes[near[1]].add(node)
            continue
        # Which end is the top is not known here, so no "upper"/"lower" guess.
        name = label or f"{tags.get('name') or 'Funicular'} ({i + 1})"
        stations[node] = {"id": f"n{node}", "name": name, "name_en": "",
                          "lon": float(lon), "lat": float(lat), "lines": set()}
        station_nodes[node].add(node)
    return 1 if len(station_nodes) >= 2 else 0


# A border point is not where the tail leaves its station: the platform end is often 100 m off.
TAIL_FROM_M = 30
# A route end that finds no border point is logged only this close to one.
TAIL_MISS_LOG_M = 25000


def border_tails(placed, runs, members, resolved, bidx):
    """Where a route runs on past this country's border: the track from its first or last
    placed station to the border point, cut there (borders.py says why at a shared point).

    Only at an end of the stops placed here, and only where the route lists stops beyond that
    end which this extract does not have at all (their nodes are outside it). A stop between
    two placed stations is a gap in the relation, which build() already traces; a stop node
    the extract lacks while the route stays inside the country (an untagged node) leads to a
    tail only if that tail also passes a border point, so it never matters away from borders.
    Returns [{"st", "bp", "geom", "km", "digest"}], "bp" None where no border point lies on
    the track the extract has (logged by the caller, not drawn)."""
    seq = stop_members(members)
    first, last = placed[0][2], placed[-1][2]
    i0 = min((i for i, n in enumerate(seq) if resolved.get(n) == first), default=None)
    i1 = max((i for i, n in enumerate(seq) if resolved.get(n) == last), default=None)
    gone = [i for i, n in enumerate(seq) if n not in resolved]
    ends = []
    if i0 is not None and any(i < i0 for i in gone):
        r, j, st = placed[0]
        ends.append((st, runs[r][0][:j + 1][::-1], runs[r][1][:j + 1][::-1]))
    if i1 is not None and any(i > i1 for i in gone):
        r, j, st = placed[-1]
        ends.append((st, runs[r][0][j:], runs[r][1][j:]))
    out = []
    for st, ids, xy in ends:
        hits = [h for h in bidx.along(xy) if h[0] > TAIL_FROM_M]
        if not hits:
            out.append({"st": st, "bp": None})
            continue
        _along, p, k, fx, fy, _off = hits[0]
        g = np.vstack([xy[:k + 1], [[fx, fy]]])
        out.append({"st": st, "bp": p, "geom": g, "km": path_length_m(g) / 1000,
                    "digest": track_key(ids[:k + 2]),
                    "ids": np.concatenate([ids[:k + 1], [-1]])})
    return out


def build(region, log):
    ways, rels, stops, cid, cx, cy = load(region, log)
    coords = Coords(cid, cx, cy)
    stations, resolved = build_stations(stops, rels, coords, log)
    groups, routes = group_lines(rels, log)

    lines, geoms = [], {}
    way_lines = defaultdict(set)     # OSM way id -> the lines that run over it
    n_gap = n_sec = n_nostop = n_ends = 0
    n_own = n_net = 0
    import borders
    bidx = borders.Index(borders.load(canonical_only=True))
    build.border_only = []           # lines whose only section here runs to the border
    tail_miss = []

    # The whole passenger network as one graph, built the first time a gap needs it.
    net = {}

    def network():
        if "g" not in net:
            from build_tiles import KIND, rank_of
            on_route = {ref for t, m in rels.values() if t.get("type") == "route"
                        for ty, ref, role in m
                        if ty == "w" and (not role or role.startswith(("forward", "backward")))}
            keep = [wid for wid, (t, _n) in ways.items()
                    if KIND.get(t.get("railway")) and (rank_of(KIND[t["railway"]], t) < 2
                                                       or wid in on_route)]
            net["g"] = TrackGraph(keep, ways, coords)
            log(f"  network graph for tracing gaps: {len(keep)} ways, {len(net['g'].adj)} nodes")
        return net["g"]
    # digest -> small integer track id, shared across every line that runs over those rails;
    # used for the log line below. What a ride credits is decided by ownership.py.
    track_ids = {}
    for lid, mtags, rids in groups:
        # Pool the line's stations over every variant FIRST, so each variant's path can be
        # walked against the full set rather than only its own calling points.
        station_nodes = defaultdict(set)
        for rid in rids:
            for ref in stop_members(routes[rid][1]):
                st = resolved.get(ref)
                if st is not None:
                    station_nodes[st].add(ref)
        if len(station_nodes) < 2 and pick(mtags, "route", "route_master") == "funicular":
            n_ends += funicular_ends(rids, routes, ways, coords, stations, station_nodes)
        # One station is enough when the route runs on over the border from it (a line whose
        # only stop in this country is its last before the border); border_tails decides.
        if not station_nodes:
            n_nostop += 1
            continue

        sections = {}
        tails = {}                            # (station, border point id) -> border_tails row
        display, best_len = [], -1
        variants = 0
        own_ways = set()
        for rid in rids:
            # Which ways this line runs over, so clicking a piece of track on the map can
            # say which lines use it.
            for ty, ref, role in routes[rid][1]:
                if ty == "w" and (not role or role.startswith(("forward", "backward"))):
                    way_lines[ref].add(lid)
                    own_ways.add(ref)
        own = {}                              # the line's own track graph, built on demand
        node2st = {n: st for st, ns in station_nodes.items() for n in ns}

        def trace(na, nb):
            """A gap between two stop nodes, traced along track: the line's own ways first,
            then the whole network. Returns (xy, ids, how) or None."""
            pa, pb = coords.get(na), coords.get(nb)
            if pa is None or pb is None:
                return None
            crow = dist_m(*pa, *pb)
            if "g" not in own:
                own["g"] = TrackGraph(own_ways, ways, coords)
            got = own["g"].path(na, nb, max(crow * OWN_DETOUR, 1000))
            if got is not None:
                return (*got, "own")
            got = network().path(na, nb, crow * NET_DETOUR + NET_MARGIN_M)
            return (*got, "net") if got is not None else None
        for rid in rids:
            tags, members = routes[rid]
            runs = assemble(members, ways, coords)
            if not runs:
                continue
            placed = place_stations(runs, station_nodes, stations)
            if not placed:
                continue
            for t in border_tails(placed, runs, members, resolved, bidx):
                if t["bp"] is None:
                    tail_miss.append((lid, t["st"]))
                    continue
                tails.setdefault((t["st"], t["bp"]["id"]), t)
            if len(placed) < 2:
                continue
            variants += 1
            seq = [st for _r, _j, st in placed]

            for i in range(len(placed) - 1):
                ra, ja, a = placed[i]
                rb, jb, b = placed[i + 1]
                key = (a, b) if a <= b else (b, a)
                if key in sections:
                    continue
                cut = slice_path(runs, (ra, ja), (rb, jb))
                if cut is None:
                    got = trace(int(runs[ra][0][ja]), int(runs[rb][0][jb]))
                    if got is not None:
                        xy, ids, how = got
                        n_own += how == "own"
                        n_net += how == "net"
                        # The traced path can pass other stations of this line, because two
                        # stations "consecutive" across a gap need not be neighbours. Cut it
                        # at each one, so it adds only the sections it truly spans and never
                        # one that overlaps sections the line already has.
                        stops = [(0, a)]
                        for k in range(1, len(ids) - 1):
                            st = node2st.get(int(ids[k]))
                            if st is not None and st != stops[-1][1] and st != b:
                                stops.append((k, st))
                        stops.append((len(ids) - 1, b))
                        for (k1, s1), (k2, s2) in zip(stops[:-1], stops[1:]):
                            kk = (s1, s2) if s1 <= s2 else (s2, s1)
                            if s1 == s2 or kk in sections or k2 <= k1:
                                continue
                            g = xy[k1:k2 + 1]
                            sections[kk] = {"km": path_length_m(g) / 1000, "geom": g,
                                            "straight": False,
                                            "digest": track_key(ids[k1:k2 + 1]),
                                            "ids": ids[k1:k2 + 1]}
                        continue
                straight = cut is None
                if straight:
                    # Two stations consecutive in the order the path passes them, but not on
                    # one continuous run: the relation has a gap. Bridge it in a straight
                    # line and count it, rather than silently losing the track.
                    n_gap += 1
                    pa, pb = stations.get(a), stations.get(b)
                    if not pa or not pb:
                        continue
                    geom = np.array([[pa["lon"], pa["lat"]], [pb["lon"], pb["lat"]]])
                    digest = ids = None
                else:
                    geom, ids = cut
                    digest = track_key(ids)
                sections[key] = {"km": path_length_m(geom) / 1000, "geom": geom,
                                 "straight": straight, "digest": digest, "ids": ids}
            if len(seq) > best_len:
                display, best_len = seq, len(seq)

        if not sections and not tails:
            n_nostop += 1
            continue
        km = sum(s["km"] for s in sections.values())
        straight = sum(1 for s in sections.values() if s["straight"])
        n_sec += len(sections)
        for s in set(display):
            if s in stations:
                stations[s]["lines"].add(lid)
        for (a, b) in sections:
            for s in (a, b):
                if s in stations:
                    stations[s]["lines"].add(lid)

        kind = pick(mtags, "route", "route_master")
        if not display and tails:
            display = [next(iter(tails))[0]]
        (lines if sections else build.border_only).append({
            "id": lid,
            # In Finland a route_master whose every route is a named train is one too: "Juna 7"
            # holds the night trains PYO 273 and PYO 276 and says so nowhere else. Finland
            # only: in jp and tw it would wrongly flag JR宝塚線・福知山線 and 內灣六家線,
            # lines whose routes are all rapid or numbered services.
            "service": (looks_like_service(mtags, kind, region)
                        or (region == "fi" and bool(rids)
                            and all(looks_like_service(routes[r][0], kind, region)
                                    for r in rids))),
            # A relation with only a ref (China's "C8600") is called by it rather than nothing.
            "name": pick(mtags, "name") or pick(mtags, "ref"),
            "name_en": pick(mtags, "name:en"),
            "ref": pick(mtags, "ref"),
            "colour": pick(mtags, "colour", "color"),
            "operator": pick(mtags, "operator"),
            "operator_en": pick(mtags, "operator:en"),
            "network": pick(mtags, "network"),
            "kind": kind,
            "km": round(km, 3),
            "variants": variants,
            "straight_sections": straight,
            "display": [stations[s]["id"] for s in display if s in stations],
            "sections": [[stations[a]["id"], stations[b]["id"], round(v["km"], 3)]
                         for (a, b), v in sections.items()
                         if a in stations and b in stations],
            "_digest": [track_id(track_ids, v["digest"])
                        for (a, b), v in sections.items()
                        if a in stations and b in stations],
            # Added after the merge (add_border_sections), so nothing the merge decides --
            # twins, station matching -- sees them and the rest of the build is unchanged.
            "_tails": [{**t, "st": stations[st]["id"]} for (st, _b), t in tails.items()
                       if st in stations],
        })
        geoms[lid] = {f"{stations[a]['id']}|{stations[b]['id']}":
                      Pts([[round(float(x), 5), round(float(y), 5)] for x, y in v["geom"]],
                          v["ids"])
                      for (a, b), v in sections.items()
                      if a in stations and b in stations}

    log(f"{len(lines)} lines, {n_sec} sections, {n_gap} sections had no path along the "
        f"route and fell back to a straight line, {n_nostop} variants had under two stops; "
        f"{n_ends} funiculars mapped with no stops took the two ends of their track")
    log(f"  gaps in a route relation traced along track instead: {n_own} over the line's own "
        f"ways, {n_net} over the wider network")
    n_t = sum(len(l["_tails"]) for l in lines + build.border_only)
    km_t = sum(t["km"] for l in lines + build.border_only for t in l["_tails"])
    # Only near a border: elsewhere a stop the extract lacks is an untagged node, not abroad.
    near = lambda s: any(dist_m(s["lon"], s["lat"], p["lon"], p["lat"]) < TAIL_MISS_LOG_M
                         for p in bidx.pts)
    missed = sorted({(lid, stations[st]["name"]) for lid, st in tail_miss
                     if st in stations and near(stations[st])})
    log(f"  over a border: {n_t} sections ({km_t:,.1f} km) from a line's last station here to "
        f"the border point, {len(build.border_only)} of them on lines with no other section "
        f"here; {len(missed)} route ends run on abroad with no border point on their track")
    for lid, name in missed:
        log(f"    no border point: {lid} from {name}")

    users = defaultdict(set)
    for l in lines:
        for d in l["_digest"]:
            users[d].add(l["id"])
    multi = sum(1 for ls in users.values() if len(ls) > 1)
    log(f"{len(users)} distinct pieces of rail, {multi} of them used by more than one line "
        f"({100*multi/max(len(users),1):.0f}%) -- the same rails, not merely alongside")

    # Global section id, so one section can name another across lines.
    gid = 0
    for l in lines:
        for sec in l["sections"]:
            sec.append(gid)
            gid += 1
    return lines, stations, geoms, way_lines


def kind_family(k):
    """Kinds that can be the same physical track. A train route, a register line, a
    narrow-gauge or heritage line are all ordinary railway; metro, tram, light rail,
    monorail and funicular each stay their own."""
    return "rail" if k in ("train", "rail", "narrow_gauge", "heritage") else k


# A register line is offered for a click on an OSM way when at least this share of the way
# lies within WAY_BUFFER_M of one of the line's sections.
WAY_BUFFER_M = 40.0
WAY_MIN_FRAC = 0.6
# Of those, only the nearest line and any within this many metres of it are kept.
WAY_TIE_M = 8.0
# A register section that ends at a junction rather than a stop is kept only if at least this
# share of its own track (the drawn ways assigned to its line) is run over by an
# OpenStreetMap passenger route.
NEEDS_ROUTE_SHARE = 0.5


def drop_unridden_sections(lines, stations, geoms, route_share, log):
    """Drop register sections that end at a junction and that no passenger train runs over.

    A register that cuts its lines at junctions rather than stations (Switzerland's does)
    gives sections that end at a place nobody boards. Some are the Gotthard base tunnel, which
    every train to Ticino runs through; some are a freight curve or a yard throat. The register
    cannot tell them apart, so OpenStreetMap's passenger routes do. A section between two
    stops is never questioned: the register saying passengers board at both ends is enough.

    Returns the ids of lines left with no sections at all, which the caller drops.
    """
    junction = {sid for sid, s in stations.items() if s.get("junction")}
    if not junction:
        return set()
    n_kept = n_drop = 0
    km_kept = km_drop = 0.0
    drop_lines = set()
    for l in lines:
        if l.get("src") == "osm":
            continue
        keep = []
        for sec in l["sections"]:
            a, b, km = sec[0], sec[1], sec[2]
            if a not in junction and b not in junction:
                keep.append(sec)
                continue
            if route_share.get((l["id"], f"{a}|{b}"), 0.0) >= NEEDS_ROUTE_SHARE:
                keep.append(sec)
                n_kept += 1
                km_kept += km
            else:
                n_drop += 1
                km_drop += km
                geoms.get(l["id"], {}).pop(f"{a}|{b}", None)
        if len(keep) == len(l["sections"]):
            l.pop("chain", None)
            continue
        l["sections"] = keep
        l["km"] = round(sum(s[2] for s in keep), 3)
        if "chain" in l:
            l["km_official"] = round(sum(l["chain"].get(f"{s[0]}|{s[1]}", 0.0)
                                         for s in keep), 3)
        l.pop("chain", None)
        ends = {s for sec in keep for s in sec[:2]}
        l["display"] = [s for s in l["display"] if s in ends]
        if not keep:
            drop_lines.add(l["id"])
            geoms.pop(l["id"], None)
    for l in lines:
        if l.get("src") == "osm" or l["id"] in drop_lines:
            continue
        ends = {s for sec in l["sections"] for s in sec[:2]}
        for sid, s in stations.items():
            if l["id"] in s["lines"] and sid not in ends:
                s["lines"].discard(l["id"])
    for s in stations.values():
        s["lines"] -= drop_lines
    log(f"sections ending at a junction: kept {n_kept} ({km_kept:,.0f} km) that passenger "
        f"routes run over, dropped {n_drop} ({km_drop:,.0f} km) that none do; "
        f"{len(drop_lines)} lines left with nothing")
    return drop_lines


def register_way_lines(region, lines, geoms, log):
    """For each OSM way on the map, the REGISTER lines whose track it is; and for each
    register section, the share of the track beside it that an OSM passenger route runs over.

    The map's track is OpenStreetMap ways, and ways.json was built from OSM route relations
    only, so a click on track could only ever resolve to an OSM object. Where OSM has no line
    relation that meant the click found a named train: the San'in Line has none, so clicking
    it opened the Super Oki. Register lines have no OSM ways at all, so the link has to be
    geometric: the share of the way within WAY_BUFFER_M of the line. The same measurements
    are kept for ownership.run, which gives each way one owner from them.

    Only ways the tiles draw are tested (the passenger filter in build_tiles.geometries), and
    kinds must agree, so a subway tunnel under a JR line does not offer the JR line.

    THE REGISTER'S OWN KIND CANNOT BE TRUSTED FOR THAT. N02 has no subway code, so a metro
    line that no OSM line matched by name is still "rail": 26 of them in Japan, the Yurakucho,
    Hanzomon and Fukutoshin lines and every metro line in Nagoya, Yokohama, Kyoto, Kobe,
    Sendai and Fukuoka among them. So each register line's kind is read off the OSM track it
    actually lies on, and a line the register calls "rail" or "tram" takes the kind of that
    track. Ownership reads the kind family from the same field.
    """
    from shapely import STRtree
    from shapely.geometry import LineString
    from build_tiles import KIND, rank_of

    d = ROOT / "data" / "proc" / region
    with open(d / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(d / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    c = np.load(d / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
    on_route = {ref for tags, members in rels.values() if tags.get("type") == "route"
                for ty, ref, role in members
                if ty == "w" and (not role or role.startswith(("forward", "backward")))}

    # Web Mercator metres, as in ownership.py. Conformal, so a buffer is round everywhere;
    # it is inflated by 1/cos(lat), which is corrected per section when buffering.
    R = 20037508.34 / 180.0

    def proj(lon, lat):
        lat = np.clip(lat, -85.05, 85.05)
        return np.column_stack([lon * R,
                                np.log(np.tan((90 + lat) * np.pi / 360)) / (np.pi / 180) * R])

    wids, wkind, wgeo, whigh, wnames = [], [], [], [], []
    for wid, (tags, nodes) in ways.items():
        kind = KIND.get(tags.get("railway"))
        if kind is None:
            continue
        if rank_of(kind, tags) >= 2 and wid not in on_route:
            continue
        pos = np.searchsorted(cid, nodes)
        np.clip(pos, 0, cid.size - 1, out=pos)
        ok = cid[pos] == nodes
        if ok.sum() < 2:
            continue
        pos = pos[ok]
        g = LineString(proj(cx[pos] / 1e7, cy[pos] / 1e7))
        if g.length <= 0:
            continue
        wids.append(wid)
        wkind.append(kind)
        wgeo.append(g)
        whigh.append(tags.get("highspeed") == "yes")
        # A way's own name tag, which can list several lines: 海峡線・北海道新幹線.
        wnames.append({norm_line_name(p, tags.get("operator", ""))
                       for p in re.split(r"[・;/]", tags.get("name") or "") if p.strip()})
    tree = STRtree(wgeo)

    family = kind_family
    mids = {}

    def mid(j):
        if j not in mids:
            mids[j] = wgeo[j].interpolate(0.5, normalized=True)
        return mids[j]

    cand = defaultdict(dict)                   # way index -> {line id: metres from the way}
    route_share = {}                   # (line id, "a|b") -> share of that section OSM routes use
    sec_ways = {}                      # (line id, "a|b") -> [(way index, metres inside)]
    rekinded = []
    for l in lines:
        if l.get("src") == "osm":
            continue
        # Way index -> fraction of it inside the line's buffer. SUMMED over sections, since
        # a long way running through a station lies half in one section and half in the next.
        near, dist = defaultdict(float), {}
        for skey, pts in geoms.get(l["id"], {}).items():
            if len(pts) < 2:
                continue
            a = np.asarray(pts, dtype=np.float64)
            scale = 1.0 / max(math.cos(math.radians(float(a[:, 1].mean()))), 0.05)
            sec = LineString(proj(a[:, 0], a[:, 1]))
            buf = sec.buffer(WAY_BUFFER_M * scale, quad_segs=4)
            beside = sec_ways[(l["id"], skey)] = []
            for j in tree.query(buf):
                w = wgeo[j]
                got = w.intersection(buf).length
                near[j] += got / w.length
                dj = sec.distance(mid(j)) / scale
                beside.append((j, got, dj))
                if dj < dist.get(j, math.inf):
                    dist[j] = dj
        near = {j: min(f, 1.0) for j, f in near.items() if f >= WAY_MIN_FRAC}
        if not near:
            continue
        # What kind of track the line mostly lies on, by length.
        by_fam = defaultdict(float)
        for j, f in near.items():
            by_fam[family(wkind[j])] += wgeo[j].length * f
        fam = family(l["kind"])
        top = max(by_fam, key=by_fam.get)
        # "rail" and "tram" are the register's LEGAL categories, which say little about the
        # track: every Osaka Metro line is legally a tramway. Anything more specific than
        # that (monorail, funicular) is kept as the register says.
        if (fam in ("rail", "tram") and top != fam
                and by_fam[top] > 0.5 * sum(by_fam.values())):
            rekinded.append((l["name"], l["operator"], top))
            l["kind"] = fam = top
        # Only where the register says which lines are high-speed. One that does not say
        # still matches LGV or Shinkansen track, rather than matching nothing there.
        high = l.get("highspeed")
        key = norm_line_name(l["name"], l["operator"])
        for j in near:
            if family(wkind[j]) != fam:
                continue
            if high is not None and whigh[j] != bool(high):
                continue
            cand[j][l["id"]] = (dist[j], key in wnames[j])

    # A way lies on its OWN line's track, and a parallel line of another operator 20 m away
    # passes the buffer test too: Meitetsu beside JR out of Nagoya. Where the way's name tag
    # names one of the candidates, that settles it. Otherwise keep the nearest line and any
    # about as near, which is what genuinely shared track looks like.
    out, n_multi = {}, 0
    for j, ls in cand.items():
        named = {lid for lid, (dd, hit) in ls.items() if hit}
        if named:
            keep = named
        else:
            dmin = min(dd for dd, _hit in ls.values())
            keep = {lid for lid, (dd, _hit) in ls.items() if dd <= dmin + WAY_TIE_M}
        out[wids[j]] = keep
        n_multi += len(keep) > 1

    # How much of a section's OWN track a passenger route runs over: only ways assigned to
    # this line above count, not everything inside the buffer. Counting everything kept the
    # freight line into Basel's goods and marshalling yards, which runs beside the main line:
    # the yard track is not drawn (it is freight), so the only drawn track in its buffer was
    # the main line's, all of it on a passenger route.
    for (lid, skey), beside in sec_ways.items():
        mine = [(j, got) for j, got, _d in beside if lid in out.get(wids[j], ())]
        total = sum(got for _j, got in mine)
        on = sum(got for j, got in mine if wids[j] in on_route)
        route_share[(lid, skey)] = on / total if total else 0.0

    # Kept for ownership.run, which gives each way ONE owner once the sections are final and
    # would otherwise load the ways and measure every buffer again.
    register_way_lines.state = {"ways": ways, "cid": cid, "cx": cx, "cy": cy, "wids": wids,
                                "wkind": wkind, "wgeo": wgeo, "sec_ways": sec_ways}
    log(f"{len(out)} of {len(wgeo)} drawn ways lie on a register line's track, "
        f"{n_multi} of them on more than one; "
        f"{len(rekinded)} register lines took their kind from that track")
    for name, op, k in rekinded:
        log(f"    {name} [{op}] -> {k}")
    return out, route_share


def add_border_sections(region, lines, stations, geoms, alias, border_only, log):
    """Each line's sections from its last station here to the border point (border_tails),
    added after the merge. The border point is a junction station with the id every country
    gives it ("e" + its RINF uopid), so a RINF register's own junction there IS this station,
    and the neighbour's build of the same line ends at the same id: the app joins the two
    pieces there with nothing new to do."""
    import borders
    for l in border_only:
        l.update(src="osm", sections=[], display=[alias.get(s, s) for s in l["display"]])
        lines.append(l)
    n, km, new_st = 0, 0.0, 0
    for l in lines:
        tails = l.pop("_tails", None)
        if not tails:
            continue
        g = geoms.setdefault(l["id"], {})
        have = {frozenset(s[:2]) for s in l["sections"]}
        for t in tails:
            a, p = alias.get(t["st"], t["st"]), t["bp"]
            b = p["id"]
            if a not in stations or frozenset((a, b)) in have:
                continue
            if b not in stations:
                stations[b] = {"id": b, "name": p["name"], "name_en": "",
                               "lon": p["lon"], "lat": p["lat"], "lines": set(),
                               "junction": True}
                new_st += 1
            stations[a]["lines"].add(l["id"])
            stations[b]["lines"].add(l["id"])
            l["sections"].append([a, b, round(t["km"], 3)])
            g[f"{a}|{b}"] = Pts([[round(float(x), 5), round(float(y), 5)] for x, y in t["geom"]],
                                t.get("ids"))
            have.add(frozenset((a, b)))
            # The display order runs on to the border, so the app chains this country's piece
            # to the next one's at the point both share.
            d = l["display"]
            if d and d[-1] == a:
                d.append(b)
            elif d and d[0] == a:
                d.insert(0, b)
            elif not d:
                d.extend([a, b])
            n += 1
            km += t["km"]
        l["km"] = round(sum(s[2] for s in l["sections"]), 3)
    log(f"border: {n} sections to a border point added to {region}'s lines ({km:,.1f} km), "
        f"{new_st} border points new as stations here")


def split_at_borders(region, lines, stations, geoms, log):
    """A section built whole across a border, because this extract happens to hold the station
    on the far side as well, is cut at the border point on it: this country's part stays, and
    the far part stays too unless that country is built (dist/regions.json), whose own build
    then has it from the border point, under the same point id. Without this the two
    countries' pieces overlap and only this country's register is credited by a ride over it.
    A side is told by which end is a station of this country's register (merge has put the
    register's ids on them); a section with neither or both is left alone."""
    import borders
    bidx = borders.Index(borders.load(canonical_only=True))
    try:
        built = set(json.loads((ROOT / "dist" / "regions.json").read_text(
            encoding="utf-8"))["regions"])
    except (OSError, ValueError, KeyError):
        built = set()
    reg_st = {s for l in lines if l.get("src") != "osm" for sec in l["sections"] for s in sec[:2]}
    # A far station this build stops shipping may be named by a saved ride. The neighbour's
    # build has it under its own id (Baisieux is n663600155 here, a bare stop node, and
    # fr87286872 in France), so the old id is aliased to the neighbour's station of the same
    # name within CARRY_SAME_NAME_M, else its nearest within CARRY_ANY_M.
    abroad, nb_st = {}, {}

    def abroad_station(nb, rec):
        if rec is None:
            return None
        if nb not in nb_st:
            try:
                nb_st[nb] = json.loads((ROOT / "dist" / "data" / nb / "stations.json").read_text(
                    encoding="utf-8"))["stations"]
            except (OSError, ValueError, KeyError):
                nb_st[nb] = {}
        cands = [(dist_m(rec["lon"], rec["lat"], s["x"], s["y"]), sid)
                 for sid, s in nb_st[nb].items() if not s.get("j")]
        same = [c for c in cands if c[0] <= CARRY_SAME_NAME_M
                and nb_st[nb][c[1]]["n"] == rec["name"]]
        near = [c for c in cands if c[0] <= CARRY_ANY_M]
        best = min(same or near, default=None)
        return best[1] if best else None
    split_at_borders.abroad = abroad
    n_split = n_drop = 0
    km_drop = 0.0
    for l in lines:
        if l.get("src") != "osm":
            continue
        g = geoms.get(l["id"], {})
        out, changed = [], False
        for sec in l["sections"]:
            a, b, km = sec[0], sec[1], sec[2]
            pts = g.get(f"{a}|{b}")
            if not pts or len(pts) < 3 or stations.get(a, {}).get("junction") \
                    or stations.get(b, {}).get("junction"):
                out.append(sec)
                continue
            total = path_length_m(pts)
            hits = [h for h in bidx.along(pts) if 50 < h[0] < total - 50]
            home_a, home_b = a in reg_st, b in reg_st
            if len(hits) != 1 or home_a == home_b:
                out.append(sec)
                continue
            _along, p, k, fx, fy, _off = hits[0]
            # A section's geometry runs the way its route did, not from its first station
            # to its second.
            sa = stations.get(a)
            if sa and dist_m(sa["lon"], sa["lat"], *pts[-1]) < dist_m(sa["lon"], sa["lat"], *pts[0]):
                a, b = b, a
                home_a, home_b = home_b, home_a
            other = (p["countries"] - {region}) or {"?"}
            far_built = bool(other & built) and region in p["countries"]
            if p["id"] not in stations:
                stations[p["id"]] = {"id": p["id"], "name": p["name"],
                                     "name_en": "", "lon": p["lon"], "lat": p["lat"],
                                     "lines": set(), "junction": True}
            stations[p["id"]]["lines"].add(l["id"])
            ids = getattr(pts, "ids", None)
            if ids is not None and len(ids) != len(pts):
                ids = None
            g1 = Pts([list(c) for c in pts[:k + 1]] + [[round(fx, 5), round(fy, 5)]],
                     None if ids is None else np.concatenate([ids[:k + 1], [-1]]))
            g2 = Pts([[round(fx, 5), round(fy, 5)]] + [list(c) for c in pts[k + 1:]],
                     None if ids is None else np.concatenate([[-1], ids[k + 1:]]))
            home, far = (a, b) if home_a else (b, a)
            gh, gf = (g1, g2) if home_a else (g2, g1)
            del g[f"{sec[0]}|{sec[1]}"]
            out.append([home, p["id"], round(path_length_m(gh) / 1000, 3)])
            g[f"{home}|{p['id']}"] = gh
            n_split += 1
            changed = True
            if far_built:
                n_drop += 1
                km_drop += path_length_m(gf) / 1000
                stations.get(far, {}).get("lines", set()).discard(l["id"])
                l["display"] = [p["id"] if s == far else s for s in l["display"]]
                (nb,) = other if len(other) == 1 else ("?",)
                t = abroad_station(nb, stations.get(far))
                if t:
                    abroad[far] = t
            else:
                out.append([p["id"], far, round(path_length_m(gf) / 1000, 3)])
                g[f"{p['id']}|{far}"] = gf
        if changed:
            l["sections"] = out
            l["km"] = round(sum(s[2] for s in out), 3)
            seen, d = set(), []
            for s in l["display"]:
                if s not in seen:
                    seen.add(s)
                    d.append(s)
            l["display"] = d
    log(f"border: {n_split} sections built whole over a border cut at its border point; the far "
        f"part left to the built neighbour in {n_drop} ({km_drop:,.1f} km); "
        f"{len(abroad)} far stations aliased to the neighbour's id")


def canon_border_ids(reg, log):
    """Give a register's border junctions the one id their crossing has (borders.canon): RINF
    files some crossings twice, a point per country at one spot (Bulgaria ends at EU00208,
    Romania at EU00209), and two ids never join in the app. Renames the station in `stations`,
    `sections`, `display`, the "a|b" keys of `geoms`, `chain` and `highspeed_sections`; where
    the register has both ids, they become one station."""
    import borders
    dup = borders.canon()
    lines, stations, geoms = reg
    ren = {s: dup[s] for s in stations if s in dup}
    if not ren:
        return reg

    def r(x):
        return ren.get(x, x)

    def rekey(d):
        return {"|".join(r(x) for x in k.split("|")): v for k, v in d.items()}

    for old, new in ren.items():
        s = stations.pop(old)
        if new in stations:
            stations[new]["lines"] = set(stations[new].get("lines", ())) | set(s.get("lines", ()))
        else:
            s["id"] = new
            stations[new] = s
    for l in lines:
        secs = [[r(s[0]), r(s[1]), *s[2:]] for s in l["sections"]]
        # a few metres between the two copies of one point is no section
        gone = [s for s in secs if s[0] == s[1]]
        l["sections"] = [s for s in secs if s[0] != s[1]]
        if gone:
            l["km"] = round(l["km"] - sum(float(s[2]) for s in gone), 3)
        if l.get("display"):
            seen, disp = set(), []
            for x in map(r, l["display"]):
                if x not in seen:
                    seen.add(x)
                    disp.append(x)
            l["display"] = disp
        for k in ("chain", "highspeed_sections"):
            if isinstance(l.get(k), dict):
                l[k] = rekey(l[k])
    for lid in list(geoms):
        geoms[lid] = {k: v for k, v in rekey(geoms[lid]).items()
                      if k.split("|")[0] != k.split("|")[-1]}
    log(f"border: {len(ren)} register border points renamed to their crossing's one id: "
        + ", ".join(f"{a} -> {b}" for a, b in sorted(ren.items())))
    return reg


def name_border_points(stations, log):
    """Every border point under its one neutral name, "Belgium – France border" (borders.py),
    the same in every country: including a RINF register's own junction there, which would
    otherwise keep its register's name ("Mouscron-Frontière" in Belgium, "Frontière FR - BE
    (Tourcoing - Mouscron)" in France), so the app showed whichever country loaded first.
    Decided by Anita, 2026-10-01."""
    import borders
    by_id = {p["id"]: p for p in borders.load()}
    n = 0
    for sid, s in stations.items():
        p = by_id.get(s.get("id", sid))
        if p is not None:
            s["name"], s["name_en"] = p["name"], ""
            n += 1
    log(f"border: {n} border points named neutrally")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--n02", default=None,
                    help="shorthand for --register n02:<path>")
    ap.add_argument("--register", default=None, metavar="MODULE:PATH",
                    help="a line register to build from and merge OSM onto, as "
                         "module:path -- e.g. n02:data/raw/N02-24_GML.zip. The module needs "
                         "one function, build(path, log) -> (lines, stations, geoms). "
                         "See HANDOFF.md for the shape of each.")
    ap.add_argument("--out", default=None,
                    help="write here instead of dist/data/<region> (a trial build); the last "
                         "build's station ids are still read from dist/data/<region>")
    args = ap.parse_args()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

    lines, stations, geoms, way_lines = build(args.region, log)
    station_alias, line_alias = {}, {}
    for l in lines:
        l.setdefault("src", "osm")
    register = args.register or (f"n02:{args.n02}" if args.n02 else None)
    route_users = way_lines
    if not register:
        line_alias = merge_osm_twins(lines, geoms, log)
        route_users = {wid: {line_alias.get(x, x) for x in lids}
                       for wid, lids in way_lines.items()}
    if register:
        import importlib
        mod_name, _, path = register.partition(":")
        if not path:
            sys.exit("--register wants module:path, e.g. n02:data/raw/N02-24_GML.zip")
        z = Path(path)
        if not z.is_absolute():
            z = ROOT / z
        if not z.exists():
            sys.exit(f"register data not found: {z}")
        mod = importlib.import_module(mod_name)
        lines, stations, geoms = merge_sources(
            (lines, stations, geoms), canon_border_ids(mod.build(str(z), log), log), log)
        station_alias = getattr(merge_sources, "alias", {})
        line_alias = getattr(merge_sources, "line_alias", {})
        add_border_sections(args.region, lines, stations, geoms, station_alias,
                            getattr(build, "border_only", []), log)
        split_at_borders(args.region, lines, stations, geoms, log)
        # Register lines OSM gave no colour: Wikidata's, where it has one (line_colours.py).
        import line_colours
        line_colours.apply(args.region, lines, log)
        # A dropped twin's ways belong to the register line it duplicated.
        for wid, lids in way_lines.items():
            way_lines[wid] = {line_alias.get(x, x) for x in lids}
        # The lines whose route relations use each way, before register lines are added for
        # clicks: what ownership reads for track no register line owns.
        route_users = {wid: set(lids) for wid, lids in way_lines.items()}
        reg_ways, route_share = register_way_lines(args.region, lines, geoms, log)
        # A national timetable feed, where data/raw/gtfs/<cc>/ has one (gtfs_served.py):
        # junction-ended sections trains run over are kept whatever OSM routes say (it sets
        # their route_share to 1.0), and sections no train runs over are marked not running
        # after not_running.mark. A region with no feed gets None and builds as before.
        import gtfs_served
        timetable = gtfs_served.check(args.region, lines, stations, route_share, log)
        drop = drop_unridden_sections(lines, stations, geoms, route_share, log)
        lines = [l for l in lines if l["id"] not in drop]
        for wid, lids in reg_ways.items():
            way_lines[wid] |= lids - drop
        # Register sections with no rails on the map: listed, but no longer running.
        import not_running
        not_running.mark(args.region, lines, geoms, log)
        gtfs_served.mark(timetable, lines, log)
        # Section ids are handed out again, because merging changed which sections exist.
        gid = 0
        for l in lines:
            for sec in l["sections"]:
                if len(sec) > 3:
                    sec[3] = gid
                else:
                    sec.append(gid)
                gid += 1
        name_border_points(stations, log)

    prev = ROOT / "dist" / "data" / args.region
    out = Path(args.out) if args.out else prev
    (out / "geom").mkdir(parents=True, exist_ok=True)
    # Clear first: a line that existed in the last build and does not now would otherwise
    # leave its geometry behind to be served for an id lines.json no longer mentions.
    for stale in (out / "geom").glob("*.json"):
        stale.unlink()

    used = {s for l in lines for pair in l["sections"] for s in pair[:2]}
    used |= {s for l in lines for s in l["display"]}
    st = {s["id"]: {"n": s["name"], "e": s["name_en"],
                    "x": round(s["lon"], 5), "y": round(s["lat"], 5),
                    "l": sorted(s["lines"])}
          for s in stations.values() if s["id"] in used}
    # A section end that is not a stop: where a register line ends at a junction. Present
    # only when true, so a region without any pays nothing for it.
    for s in stations.values():
        if s.get("junction") and s["id"] in st:
            st[s["id"]]["j"] = 1
    # Stations left to a built neighbour alias to the neighbour's ids; the app looks a ride's
    # stations up in every country it runs through.
    station_alias = carry_aliases(prev, st, station_alias, log,
                                  getattr(split_at_borders, "abroad", None))

    lines.sort(key=lambda l: (-l["km"], l["id"]))
    idx = {l["id"]: i for i, l in enumerate(lines)}

    # ONE OWNER PER PIECE OF TRACK (Anita, 2026-10-01): each drawn way gets one line, and
    # each section the list of owner sections riding it credits (ownership.py). foot.json
    # replaces credits.json, the old 45 m corridor buffer that let one ride complete every
    # line beside it.
    import ownership
    try:
        built = json.loads((ROOT / "dist" / "regions.json").read_text(
            encoding="utf-8"))["regions"]
    except (OSError, ValueError, KeyError):
        built = {}
    foot, report = ownership.run(args.region, lines, geoms, route_users, stations, log,
                                 state=getattr(register_way_lines, "state", None),
                                 built_regions=built)
    ownership.write(out, args.region, foot)
    if (out / "credits.json").exists():
        (out / "credits.json").unlink()
    log(f"foot.json: {(out / 'foot.json').stat().st_size / 1e6:.2f} MB")

    # WAY TO LINE, for resolving a click on track. Fetched by the viewer only when someone
    # actually clicks a line. ways.json is
    #     {"region", "lines": [line id, ...], "ways": {way id: [line index, ...]},
    #      "unowned": [way id, ...]}
    # with indices into "lines" (the order of lines.json). A way's list is every line whose
    # route runs over it, register lines included. THE FIRST INDEX IS THE WAY'S OWNER
    # (ownership.run), the rest follow in index order: the app opens the owner and lists the
    # rest under it as also on this track. A way with no owner (only named trains run there,
    # or it lies abroad) keeps its list in index order; where that list has more than one line
    # and does not start with a named train it would read as owned, so its id goes in
    # "unowned" (named trains never own, so a list starting with one reads as unowned as it
    # is). An owner no route relation names over its way (a station throat, a single-track
    # companion's track, a register line found there by geometry alone) is added at the front.
    owner_of = {str(w): int(o) for w, o in zip(report["wids"], report["owner"]) if o >= 0}
    ways, unowned, n_added = {}, [], 0
    for wid, lids in way_lines.items():
        wid = str(wid)
        got = sorted(idx[l] for l in lids if l in idx)
        o = owner_of.get(wid)
        if o is not None:
            n_added += o not in got
            got = [o] + [i for i in got if i != o]
        elif len(got) > 1 and not lines[got[0]].get("service"):
            unowned.append(wid)
        if got:
            ways[wid] = got
    for wid, o in owner_of.items():
        if wid not in ways:
            ways[wid] = [o]
            n_added += 1
    with open(out / "ways.json", "w", encoding="utf-8") as f:
        json.dump({"region": args.region, "lines": [l["id"] for l in lines], "ways": ways,
                   "unowned": unowned}, f, ensure_ascii=False, separators=(",", ":"))
    log(f"{len(ways)} ways mapped to the lines that run over them, "
        f"{sum(1 for w in ways if w in owner_of)} with an owner first ({n_added} owners no "
        f"route there names, added), {len(unowned)} marked unowned")

    for l in lines:
        l.pop("_digest", None)
    with open(out / "lines.json", "w", encoding="utf-8") as f:
        json.dump({"region": args.region, "lines": lines}, f, ensure_ascii=False)
    with open(out / "stations.json", "w", encoding="utf-8") as f:
        json.dump({"region": args.region, "stations": st}, f, ensure_ascii=False)
    # Station ids move when a build merges OSM onto the register, and a rider's saved trips
    # name stations by id. Ship the map so the app can migrate them instead of losing them.
    # `lines` is the same for a line dropped as a register line's twin: a ride saved against
    # it names stations the register line also has, so the ride moves across whole.
    with open(out / "aliases.json", "w", encoding="utf-8") as f:
        json.dump({"region": args.region, "stations": station_alias, "lines": line_alias},
                  f, ensure_ascii=False, separators=(",", ":"))

    for lid, g in geoms.items():
        with open(out / "geom" / f"{lid}.json", "w", encoding="utf-8") as f:
            json.dump(g, f, ensure_ascii=False, separators=(",", ":"))

    total_km = sum(l["km"] for l in lines)
    svc = [l for l in lines if l["service"]]
    svc_km = sum(l["km"] for l in svc)
    with_colour = sum(1 for l in lines if l["colour"])
    log(f"{len(lines)} lines, {len(st)} stations, {total_km:,.0f} route-km")
    log(f"  of which {len(svc)} look like named trains rather than lines, {svc_km:,.0f} km; "
        f"{total_km-svc_km:,.0f} route-km without them")
    log(f"{with_colour} lines carry a colour ({100*with_colour/max(len(lines),1):.0f}%)")
    log(f"wrote {out}")
    print("\nlongest lines")
    for l in lines[:12]:
        print(f"  {l['km']:>8.1f} km  {len(l['sections']):>4} sec  {l['ref']:<6} "
              f"{l['name']}  [{l['operator']}]")


if __name__ == "__main__":
    main()
