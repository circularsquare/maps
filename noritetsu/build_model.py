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
            by_name[s["name"]].append(nid)

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
            by_name[s["name"]].append(nid)
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
        for cand in by_name.get(name, ()) if name else ():
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
                by_name[name].append(nid)
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
        return bool(FR_TRAIN_BRAND.match(name))
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
    if region in EU_TRAIN_REGIONS:
        # International and long-distance trains mapped one relation per train (EC 112, EN
        # 40467, ICE 43, Eurostar, European Sleeper, Nightjet), as France, Poland, Hungary and
        # Portugal flag theirs. The interval products that are lines to a rider, IC, IR and
        # Railjet (Swiss IC 1, ÖBB's half-hourly Railjet), are left as lines.
        return bool(EU_TRAIN.search(name))
    return False


FI_TRAIN = re.compile(r"\bPYO\s?\d|^Taajamajuna\s+\d")
RO_TRAIN = re.compile(r"\b(?:R|R-E|RE|IR|IRN|IC|INT|EC|EN|ICN)\s?-?\s?\d{2,5}\b")
HR_TRAIN = re.compile(r"^(?:Vlak\s+)?(?:B|IC|ICN|EC|EN)\s?\d")
EU_TRAIN_REGIONS = {"at", "be", "nl", "ch", "cz", "si", "bg", "sk"}
EU_TRAIN = re.compile(r"^(?:Train\s+)?(?:EC|EN|ICE|NJ|TGV|ES|ECE|INT)(?:[\s\d:]|$)"
                      r"|\bEuro(?:City|Night)\b|\bNightjet\b|\bEuropean Sleeper\b"
                      r"|^(?:Eurostar|Thalys|TGV Lyria|Lyria)\b")


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


def carry_aliases(out, st, station_alias, log):
    """Keep every station id the LAST build shipped reachable from this one.

    aliases.json used to hold only this build's own merges, so an id that simply stopped
    existing between builds was lost, and with it any saved ride naming it: a change to how
    stop nodes resolve (a bus-terminal record at 西鉄福岡 giving way to the rail station)
    does exactly that. So every id in the previous stations.json or aliases.json that this
    build neither ships nor aliases is mapped on: through its old alias if that target
    still exists, else to a station of the same name within CARRY_SAME_NAME_M, else to the
    nearest within CARRY_ANY_M. Chains are resolved to a live id."""
    prev_st, prev_al = {}, {}
    try:
        with open(out / "stations.json", encoding="utf-8") as f:
            prev_st = json.load(f)["stations"]
        with open(out / "aliases.json", encoding="utf-8") as f:
            prev_al = json.load(f)["stations"]
    except (OSError, ValueError, KeyError):
        pass
    alias = dict(station_alias)
    if not prev_st:
        return alias

    ids = list(st)
    pos = np.array([[st[i]["x"], st[i]["y"]] for i in ids]) if ids else np.zeros((0, 2))
    by_name = defaultdict(list)
    for i in ids:
        by_name[st[i]["n"]].append(i)

    def live(t):
        seen = set()
        while t not in st and t in alias and t not in seen:
            seen.add(t)
            t = alias[t]
        return t if t in st else None

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
    s = unicodedata.normalize("NFKC", name or "").casefold()
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


def build(region, log):
    ways, rels, stops, cid, cx, cy = load(region, log)
    coords = Coords(cid, cx, cy)
    stations, resolved = build_stations(stops, rels, coords, log)
    groups, routes = group_lines(rels, log)

    lines, geoms = [], {}
    way_lines = defaultdict(set)     # OSM way id -> the lines that run over it
    n_gap = n_sec = n_nostop = n_ends = 0
    n_own = n_net = 0

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
    # digest -> small integer track id, shared across every line that runs over those rails.
    # THIS IS WHAT MAKES A RIDE COUNT EVERYWHERE IT SHOULD: rides are recorded against track,
    # not against the line label they were entered under, so riding the part of the Yamanote
    # loop that is technically Tohoku Main Line credits both.
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
        if len(station_nodes) < 2:
            n_nostop += 1
            continue

        sections = {}
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
                                            "digest": track_key(ids[k1:k2 + 1])}
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
                    digest = None
                else:
                    geom, ids = cut
                    digest = track_key(ids)
                sections[key] = {"km": path_length_m(geom) / 1000, "geom": geom,
                                 "straight": straight, "digest": digest}
            if len(seq) > best_len:
                display, best_len = seq, len(seq)

        if not sections:
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
        lines.append({
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
        })
        geoms[lid] = {f"{stations[a]['id']}|{stations[b]['id']}":
                      [[round(float(x), 5), round(float(y), 5)] for x, y in v["geom"]]
                      for (a, b), v in sections.items()
                      if a in stations and b in stations}

    log(f"{len(lines)} lines, {n_sec} sections, {n_gap} sections had no path along the "
        f"route and fell back to a straight line, {n_nostop} variants had under two stops; "
        f"{n_ends} funiculars mapped with no stops took the two ends of their track")
    log(f"  gaps in a route relation traced along track instead: {n_own} over the line's own "
        f"ways, {n_net} over the wider network")

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


# Tram and light rail are one mode to a rider where they share rails, and OSM does not tell
# them apart consistently: France's T11 is route=tram on light_rail track, Hiroshima's tram
# line 2 runs on to the light_rail 宮島線, the Forchbahn runs into Zürich on tram track. But in
# Japan light_rail is mostly a rubber-tyred guideway (Astram, Nippori-Toneri Liner), which
# crosses over or under a tramway without sharing anything; at 45 m those crossings credited
# each other (about 20 pairs in jp), a false completion. So across the two kinds a credit
# needs the covered track within SAME_RAILS_M, and neither line may be a guideway: a
# register line the reader marks `"guided": True` (n02: N02's guided/AGT codes). The flag is
# needed as well as the distance because the Astram Line runs in tunnel directly beneath
# Hiroshima's tram streets, within 8 m of them. Credits only: kind_family is left alone,
# because register_way_lines' rekind test depends on it.
TRAMLIKE = {"tram", "light_rail"}
SAME_RAILS_M = 8.0


def kind_family(k):
    """Kinds that can be the same physical track. A train route, a register line, a
    narrow-gauge or heritage line are all ordinary railway; metro, tram, light rail,
    monorail and funicular each stay their own."""
    return "rail" if k in ("train", "rail", "narrow_gauge", "heritage") else k


# An OSM section is on high-speed track when this share of it lies within HS_M of a
# highspeed=yes way. Tight, because the section's geometry IS those ways; a conventional
# line beside a Shinkansen is ten metres or more away.
HS_M = 4.0
HS_SHARE = 0.5


def section_highspeed(region, lines, geoms, log):
    """Section id -> True/False where it is KNOWN whether the section is high-speed track.

    Crediting is spatial, and a Shinkansen runs within 45 m of the conventional line for long
    stretches: once train routes could credit register lines at all, a ride on the Tohoku
    Shinkansen was completing the Tohoku Line beside it, about 1,600 km of false credit over
    16 services. Kind cannot separate them (both are "rail"), so speed has to.

    OSM sections are tested against the highspeed=yes ways they lie on, which is exact
    because a section's geometry is cut from those same ways -- and it gets mini-Shinkansen
    right, since the Tsubasa's sections north of Fukushima lie on the Ou Line's ordinary
    track. A register section takes its line's `highspeed` flag where the register gives one,
    and is left unknown where it does not (Switzerland), in which case nothing is filtered.
    """
    from shapely import STRtree, union_all
    from shapely.geometry import LineString

    d = ROOT / "data" / "proc" / region
    with open(d / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    c = np.load(d / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
    R = 20037508.34 / 180.0

    def proj(lon, lat):
        lat = np.clip(lat, -85.05, 85.05)
        return np.column_stack([lon * R,
                                np.log(np.tan((90 + lat) * np.pi / 360)) / (np.pi / 180) * R])

    hs = []
    for tags, nodes in ways.values():
        if tags.get("highspeed") != "yes":
            continue
        pos = np.searchsorted(cid, nodes)
        np.clip(pos, 0, cid.size - 1, out=pos)
        pos = pos[cid[pos] == nodes]
        if pos.size >= 2:
            hs.append(LineString(proj(cx[pos] / 1e7, cy[pos] / 1e7)))
    out = {}
    tree = STRtree(hs) if hs else None
    n_hs = 0
    for l in lines:
        if l.get("src", "osm") != "osm":
            if "highspeed" in l:
                for sec in l["sections"]:
                    out[sec[3]] = bool(l["highspeed"])
            # Per section, where a register line is part high-speed: Korea's 중앙선 runs on a
            # new 250 km/h alignment for much of its length and on the old one for the rest.
            per = l.get("highspeed_sections") or {}
            for sec in l["sections"]:
                flag = per.get(f"{sec[0]}|{sec[1]}")
                if flag is not None:
                    out[sec[3]] = bool(flag)
            continue
        g = geoms.get(l["id"], {})
        for a, b, km, gid in l["sections"]:
            pts = g.get(f"{a}|{b}")
            flag = False
            if tree is not None and pts and len(pts) > 1:
                arr = np.asarray(pts, dtype=np.float64)
                sec = LineString(proj(arr[:, 0], arr[:, 1]))
                scale = 1.0 / max(math.cos(math.radians(float(arr[:, 1].mean()))), 0.05)
                near = tree.query(sec.buffer(HS_M * scale))
                if len(near) and sec.length > 0:
                    zone = union_all([hs[j].buffer(HS_M * scale, quad_segs=2) for j in near])
                    flag = sec.intersection(zone).length / sec.length >= HS_SHARE
            out[gid] = flag
            n_hs += flag
    log(f"{len(hs)} highspeed=yes ways; {n_hs} OSM sections lie on high-speed track")
    return out


def build_credits(lines, geoms, buffer_m, min_frac, log, highspeed=None):
    """Which sections a ride over one section should also credit.

    RIDES ATTACH TO TRACK, NOT TO THE LINE LABEL THEY WERE ENTERED UNDER.  Riding the part of
    the Yamanote loop north of Tabata is, in the operator's line register, riding the Tohoku
    Main Line, and it should count for both.  Identical rails cannot express that: the loop
    runs on the Yamanote's own pair of tracks and the Tohoku Main Line relation follows the
    main pair thirty metres away, so a fingerprint of the rails finds nothing in common -- it
    reported 0.0 of the Yamanote's 34.5 km as shared.

    So crediting is geometric.  Section A credits section B when B lies almost entirely
    inside a buffer around A: riding the corridor completes every line the corridor carries,
    which is how the line register works and how noritsubushi.org counts.

    What is recorded is a RANGE, not a yes/no.  Section A covers the fraction s..e of section
    B, so three local sections that each cover a third of one long express section together
    complete it, and riding one of them completes a third of it.  Whole-section crediting
    could not express that and silently dropped every express and register line whose stops
    are coarser than the service you actually rode.

    Guarded by kind, because Tokyo has metro tunnels directly beneath JR lines and they are
    not each other however close they run. By kind FAMILY, not the raw string: an OSM route
    is route=train and a register line is "rail", and comparing the two strings meant no train
    route credited any register line at all. From the N02 merge until 2026-09-30, riding the
    Yamanote as operated counted nothing towards 山手線, and no ride through the Gotthard
    base tunnel counted towards it.
    """
    from shapely import STRtree
    from shapely.geometry import LineString, Point

    R = 20037508.34 / 180.0

    def to_merc(pts):
        out = []
        for lon, lat in pts:
            lat = max(min(lat, 85.05), -85.05)
            out.append((lon * R,
                        math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180) * R))
        return out

    highspeed = highspeed or {}
    ids, kinds, geo, lats, guided = [], [], [], [], []
    for l in lines:
        for a, b, km, gid in l["sections"]:
            pts = geoms[l["id"]].get(f"{a}|{b}")
            if not pts or len(pts) < 2:
                continue
            ids.append(gid)
            kinds.append(kind_family(l["kind"]))
            guided.append(bool(l.get("guided")))
            geo.append(LineString(to_merc(pts)))
            lats.append(sum(p[1] for p in pts) / len(pts))
    if not geo:
        return {}

    tree = STRtree(geo)
    covered = defaultdict(list)
    n_pairs = 0
    for i, g in enumerate(geo):
        # Mercator metres are inflated by 1/cos(lat), so a true buffer_m is this on the plane.
        scale = 1.0 / max(math.cos(math.radians(lats[i])), 0.05)
        buf = g.buffer(buffer_m * scale, quad_segs=4)
        tight = None
        for j in tree.query(buf):
            if j == i:
                continue
            cover = buf
            if kinds[j] != kinds[i]:
                if (kinds[i] not in TRAMLIKE or kinds[j] not in TRAMLIKE
                        or guided[i] or guided[j]):
                    continue
                # Tram and light rail credit each other only on the SAME rails, within
                # SAME_RAILS_M, not merely alongside within buffer_m.
                if tight is None:
                    tight = g.buffer(SAME_RAILS_M * scale, quad_segs=4)
                cover = tight
            # High-speed and conventional track do not credit each other, where both are known.
            hi_, hj = highspeed.get(ids[i]), highspeed.get(ids[j])
            if hi_ is not None and hj is not None and hi_ != hj:
                continue
            b = geo[j]
            if b.length <= 0:
                continue
            piece = b.intersection(cover)
            if piece.is_empty or piece.length / b.length < min_frac:
                continue
            parts = (list(piece.geoms) if piece.geom_type.startswith("Multi")
                     else [piece])
            lo, hi = 1.0, 0.0
            for part in parts:
                if part.geom_type != "LineString":
                    continue
                for pt in (part.coords[0], part.coords[-1]):
                    t = b.project(Point(pt)) / b.length
                    lo, hi = min(lo, t), max(hi, t)
            if hi - lo < min_frac:
                continue
            # Keyed by the COVERING section, not the covered one: the reader always starts
            # from "here is a section that was ridden, what else does it complete", so this
            # is the direction that can be answered without walking the whole table, and the
            # direction that could be fetched lazily later.
            covered[ids[i]].append([ids[j], round(lo, 4), round(hi, 4)])
            n_pairs += 1
        if (i + 1) % 2000 == 0:
            log(f"  credits: {i+1}/{len(geo)} sections, {n_pairs} pairs so far")
    log(f"{len(covered)} sections credit another ({n_pairs} pairs) at {buffer_m} m, "
        f"ranges down to {min_frac:.0%}")
    return covered


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
    geometric -- the same test `build_credits` uses for shared corridors, with the way in the
    place of the covered section.

    Only ways the tiles draw are tested (the passenger filter in build_tiles.geometries), and
    kinds must agree, so a subway tunnel under a JR line does not offer the JR line.

    THE REGISTER'S OWN KIND CANNOT BE TRUSTED FOR THAT. N02 has no subway code, so a metro
    line that no OSM line matched by name is still "rail": 26 of them in Japan, the Yurakucho,
    Hanzomon and Fukutoshin lines and every metro line in Nagoya, Yokohama, Kyoto, Kobe,
    Sendai and Fukuoka among them. So each register line's kind is read off the OSM track it
    actually lies on, and a line the register calls "rail" or "tram" takes the kind of that
    track. This runs before `build_credits`, which is guarded on the same field.
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

    # Web Mercator metres, as in build_credits. Conformal, so a buffer is round everywhere;
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
                beside.append((j, got))
                dj = sec.distance(mid(j)) / scale
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
        mine = [(j, got) for j, got in beside if lid in out.get(wids[j], ())]
        total = sum(got for _j, got in mine)
        on = sum(got for j, got in mine if wids[j] in on_route)
        route_share[(lid, skey)] = on / total if total else 0.0

    log(f"{len(out)} of {len(wgeo)} drawn ways lie on a register line's track, "
        f"{n_multi} of them on more than one; "
        f"{len(rekinded)} register lines took their kind from that track")
    for name, op, k in rekinded:
        log(f"    {name} [{op}] -> {k}")
    return out, route_share


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
    ap.add_argument("--buffer", type=float, default=45.0,
                    help="metres; how far apart parallel track can be and still be one corridor")
    ap.add_argument("--min-frac", type=float, default=0.15,
                    help="ignore a coverage shorter than this fraction of the section")
    args = ap.parse_args()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

    lines, stations, geoms, way_lines = build(args.region, log)
    station_alias, line_alias = {}, {}
    for l in lines:
        l.setdefault("src", "osm")
    register = args.register or (f"n02:{args.n02}" if args.n02 else None)
    if not register:
        line_alias = merge_osm_twins(lines, geoms, log)
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
            (lines, stations, geoms), mod.build(str(z), log), log)
        station_alias = getattr(merge_sources, "alias", {})
        line_alias = getattr(merge_sources, "line_alias", {})
        # Register lines OSM gave no colour: Wikidata's, where it has one (line_colours.py).
        import line_colours
        line_colours.apply(args.region, lines, log)
        # A dropped twin's ways belong to the register line it duplicated.
        for wid, lids in way_lines.items():
            way_lines[wid] = {line_alias.get(x, x) for x in lids}
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

    out = ROOT / "dist" / "data" / args.region
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
    station_alias = carry_aliases(out, st, station_alias, log)

    # Way to line, as indices into the sorted line list, for resolving a click on track.
    # Fetched by the viewer only when someone actually clicks a line.
    lines.sort(key=lambda l: (-l["km"], l["id"]))
    idx = {l["id"]: i for i, l in enumerate(lines)}
    ways = {}
    for wid, lids in way_lines.items():
        got = sorted(idx[l] for l in lids if l in idx)
        if got:
            ways[str(wid)] = got
    with open(out / "ways.json", "w", encoding="utf-8") as f:
        json.dump({"region": args.region, "lines": [l["id"] for l in lines], "ways": ways},
                  f, ensure_ascii=False, separators=(",", ":"))
    log(f"{len(ways)} ways mapped to the lines that run over them")

    covered = build_credits(lines, geoms, args.buffer, args.min_frac, log,
                            section_highspeed(args.region, lines, geoms, log))
    with open(out / "credits.json", "w", encoding="utf-8") as f:
        json.dump({"region": args.region, "buffer_m": args.buffer,
                   "min_frac": args.min_frac,
                   "covers": {str(k): v for k, v in covered.items()}},
                  f, ensure_ascii=False, separators=(",", ":"))

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
