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
from collections import Counter, defaultdict
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
# ... and two with the same English name but different native ones, closer than this.
EN_DUP_RADIUS_M = 50
# ... and for metro stations where the country's rules set METRO_DUP, closer than this
# (merge_duplicate_stations).
METRO_DUP_RADIUS_M = 150
METRO_NAME_RADIUS_M = 200      # ... and a metro stop node finds its station within this


def plain_name(tags, region):
    """A rail stop's name, without its platform where the country's rules give a
    PLATFORM_SUFFIX (never a tram stop: Melbourne's are "Stop 35: ...")."""
    name = tags.get("name") or ""
    suffix = getattr(country_rules(region), "PLATFORM_SUFFIX", None)
    if (suffix is not None and tags.get("railway") != "tram_stop"
            and tags.get("tram") != "yes"):
        return suffix.sub("", name) or name
    return name

# Which station record to keep when several describe one station.
STATION_RANK = {"station": 0, "halt": 2, "tram_stop": 3}
# A public_transport=station with one of these set to yes is a rail station.
RAIL_MODES = ("train", "subway", "light_rail", "monorail", "tram", "funicular")

# How far a stop node may sit off the route's own path before we stop believing the match.
STOP_SNAP_M = 400
# ... unless the stop node lies on another variant's track (SNAP_ALONGSIDE in the country's
# rules): then only where that track runs alongside this path (place_stations, _alongside).
# Measured on fr 2026-10-08: parallel tracks keep 0-2 m of spread over the window (Lille-Europe
# 42 m off, Le Mans 40 m, Gare du Nord 79 m); a one-way loop's arms, branches and crossings
# move 20-90 m (Mirabeau 2 m off at the stop node, 38 m within 150 m; RER A's Marne-la-Vallée
# branch past Fontenay-sous-Bois 11 m, 47 m) or lie 150-400 m apart (Métro 10, 7bis's Danube).
ALONGSIDE_M = 100
ALONGSIDE_WIN_M = 150
ALONGSIDE_SPREAD_M = 15


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


def merge_duplicate_stations(stations, log=None, region=None):
    """Collapse station records that are the same station twice.

    OSM frequently carries BOTH a railway=station node and a public_transport=station node
    for one station, and separate nodes per operator at a shared interchange.  Mapping stop
    positions onto stations does not catch this, because both records already look like
    stations, so nothing ever merged them: the Yamanote came out with 31 stations against a
    real 30 and its strip diagram opened with Shinagawa listed twice in a row.

    Same name and within `DUP_RADIUS_M` is the test.  Different operators' platforms under
    one name -- JR and Odakyu at Shinjuku -- merge deliberately: to a rider that is one
    station.  Genuinely different stations sharing a name are far enough apart to survive.

    Then the same English name within `EN_DUP_RADIUS_M`, for a rename OSM caught only half
    of: Shanghai's 东昌路 became 浦东南路 in 2021, and one node 16 m from the new record still
    says 东昌路 (name:en "South Pudong Road"), so Line 14's two platforms landed on two
    records and no section joined them. Which native name survives is decided by how many
    records the first pass folded into each: the current name is mapped several times over
    (station node, stop area, each line's platforms), a stale one is a lone node.

    Where the country's rules set METRO_DUP, a metro station merges only within
    METRO_DUP_RADIUS_M (rules/us.py says why).
    """
    alias = {}
    metro_dup = getattr(country_rules(region), "METRO_DUP", False)

    def radius(a, b):
        if metro_dup and (a.get("_metro") or b.get("_metro")):
            return METRO_DUP_RADIUS_M
        return DUP_RADIUS_M
    _merge_by(stations, lambda s: fold_dashes(s["name"]), radius, alias)
    folded = Counter(alias.values())
    en = _merge_by(stations, lambda s: fold_dashes(s["name_en"]).casefold(), EN_DUP_RADIUS_M,
                   alias, prefer=lambda n: -folded[n])
    for x in alias:                    # what the first pass folded into a record now gone
        while alias[x] in alias:
            alias[x] = alias[alias[x]]
    if log:
        for c, rep, name in en:
            log(f"stations: {name!r} folded into {stations[rep]['name']!r} "
                f"(n{rep}), same English name {stations[rep]['name_en']!r}")
    return alias


def _merge_by(stations, key, radius, alias, prefer=lambda n: 0):
    """Fold records with the same key within `radius` m (a number, or a function of the two
    records) into one; return (gone, kept, name)."""
    merged = []
    far = radius if callable(radius) else (lambda a, b, r=radius: r)
    by_name = defaultdict(list)
    for nid, s in stations.items():
        k = key(s)
        if k:
            by_name[k].append(nid)

    for name, ids in by_name.items():
        if len(ids) < 2:
            continue
        remaining = list(ids)
        while remaining:
            seed = remaining.pop(0)
            cluster, rest = [seed], []
            for other in remaining:
                a, b = stations[seed], stations[other]
                if dist_m(a["lon"], a["lat"], b["lon"], b["lat"]) <= far(a, b):
                    cluster.append(other)
                else:
                    rest.append(other)
            remaining = rest
            if len(cluster) < 2:
                continue
            # Keep the most station-like record, then one with an English name, then the
            # lowest id so the choice is the same on every build.
            rep = min(cluster, key=lambda n: (prefer(n), stations[n]["_rank"],
                                              0 if stations[n]["name_en"] else 1, n))
            for c in cluster:
                if c == rep:
                    continue
                if stations[c]["name_en"] and not stations[rep]["name_en"]:
                    stations[rep]["name_en"] = stations[c]["name_en"]
                alias[c] = rep
                merged.append((c, rep, stations[c]["name"]))
                del stations[c]
    return merged


def build_stations(stops, rels, coords, log, region=None):
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
                "id": f"n{nid}", "name": plain_name(tags, region),
                "name_en": tags.get("name:en") or "",
                "lon": lon, "lat": lat, "lines": set(),
                "_rank": (STATION_RANK.get(tags.get("railway"), 3)
                          if tags.get("railway") else 1),
                "_tram": tags.get("railway") == "tram_stop",
                "_metro": (tags.get("station") in ("subway", "light_rail", "monorail")
                           or tags.get("subway") == "yes" or tags.get("light_rail") == "yes"),
            }

    alias = merge_duplicate_stations(stations, log, region)

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
    # Records made from bare stop nodes, by English name, and the native names of the stop
    # nodes each one took: see the 东昌路 note below.
    by_en, names_of = defaultdict(list), {}
    metro_dup = getattr(country_rules(region), "METRO_DUP", False)
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
        name = plain_name(tags, region)
        # Where the country's rules set METRO_DUP, a metro stop finds its station by name
        # only within METRO_NAME_RADIUS_M (rules/us.py says why).
        metro = metro_dup and (
            tags.get("subway") == "yes" or tags.get("light_rail") == "yes"
            or tags.get("station") in ("subway", "light_rail", "monorail"))
        name_r = METRO_NAME_RADIUS_M if metro else NAME_RADIUS_M
        best, best_d = None, None
        for cand in by_name.get(fold_dashes(name), ()) if name else ():
            d = dist_m(lon, lat, stations[cand]["lon"], stations[cand]["lat"])
            if d <= name_r and (best_d is None or d < best_d):
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
            if dd[j] <= (min(BLIND_RADIUS_M, METRO_NAME_RADIUS_M) if metro else BLIND_RADIUS_M):
                best = int(ids[j])
        # A stale rename on a stop node with no station node near: Shanghai's 东昌路 became
        # 浦东南路 in 2021, and one of Line 14's stop nodes still has the old name (name:en
        # "South Pudong Road" on both, 16 m apart). By native name the two never meet, and
        # Line 14 broke there. The same English name within EN_DUP_RADIUS_M is one station.
        en = fold_dashes(tags.get("name:en") or "").casefold()
        if best is None and en:
            for cand in by_en.get(en, ()):
                if dist_m(lon, lat, stations[cand]["lon"], stations[cand]["lat"]) <= EN_DUP_RADIUS_M:
                    best = cand
                    break
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
            if en:
                by_en[en].append(nid)
            names_of[nid] = Counter()
        if best in names_of and name:
            names_of[best][name] += 1
        resolved[nid] = best

    # Which native name such a record keeps: the one most of its stop nodes carry, so a lone
    # stale node does not name the station (ties: the name it was made with).
    for nid, c in names_of.items():
        if len(c) < 2:
            continue
        top = max(c.values())
        if c[stations[nid]["name"]] < top:
            new = min(n for n, k in c.items() if k == top)
            log(f"stations: n{nid} named {new!r}, not {stations[nid]['name']!r} "
                f"(stop nodes {dict(c)})")
            stations[nid]["name"] = new

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


def travel_dirs(members, ways, runs):
    """{way id: +1 or -1}: whether a route relation, as assemble() joined it into runs,
    travels each of its ways along (+1) or against (-1) the way's node order. A way whose two
    end nodes are not both found on one run in order (a way it runs over twice, a broken
    run) is left out."""
    pos = {}
    for ri, (ids, _xy) in enumerate(runs):
        for k, n in enumerate(ids.tolist()):
            pos.setdefault(n, (ri, k))
    out = {}
    for ty, ref, role in members:
        if ty != "w" or (role and not role.startswith(("forward", "backward"))):
            continue
        w = ways.get(ref)
        if w is None or len(w[1]) < 2:
            continue
        a, b = pos.get(w[1][0]), pos.get(w[1][-1])
        if a is None or b is None or a[0] != b[0] or a[1] == b[1]:
            continue
        out[ref] = 1 if b[1] > a[1] else -1
    return out


def track_at(allruns, station_nodes):
    """{stop node: [(xy, metres along, index), ...]}: where each of a line's stop nodes lies
    on its variants' assembled runs (allruns: {rel id: runs}), for place_stations' test of
    whether another variant runs alongside a station."""
    want = np.fromiter((n for ns in station_nodes.values() for n in ns), dtype=np.int64)
    out = defaultdict(list)
    if not want.size:
        return out
    for runs in allruns.values():
        for ids, xy in runs:
            hit = np.flatnonzero(np.isin(ids, want))
            if not hit.size:
                continue
            lat = math.radians(float(xy[:, 1].mean()))
            c = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xy[:, 0]) * math.cos(lat) * 111320,
                                                          np.diff(xy[:, 1]) * 110570))])
            for k in hit.tolist():
                out[int(ids[k])].append((xy, c, k))
    return out


def _alongside(xy, c, k, runs):
    """Does the track through point k of xy run alongside this variant's runs: its stop node
    within ALONGSIDE_M of them, and ALONGSIDE_WIN_M along it either way never more than
    ALONGSIDE_SPREAD_M further off than that? Parallel tracks stay at one distance; a one-way
    loop's other arm, a branch or a crossing line moves away."""
    pts = xy[(c >= c[k] - ALONGSIDE_WIN_M) & (c <= c[k] + ALONGSIDE_WIN_M)]
    lat = math.radians(float(xy[k, 1]))
    kx = math.cos(lat) * 111320
    P = np.column_stack([pts[:, 0] * kx, pts[:, 1] * 110570])
    p0 = np.array([xy[k, 0] * kx, xy[k, 1] * 110570])
    reach = ALONGSIDE_M + ALONGSIDE_SPREAD_M + ALONGSIDE_WIN_M
    best = np.full(len(P), np.inf)
    best0 = np.inf
    for _ids, rxy in runs:
        R = np.column_stack([rxy[:, 0] * kx, rxy[:, 1] * 110570])
        A, B = R[:-1], R[1:]
        near = (np.minimum(A, B) <= p0 + reach).all(1) & (np.maximum(A, B) >= p0 - reach).all(1)
        if not near.any():
            continue
        A, B = A[near], B[near]
        AB = B - A
        L2 = np.maximum((AB ** 2).sum(1), 1e-9)
        for i, q in enumerate(np.vstack([P, p0])):
            t = np.clip(((q - A) * AB).sum(1) / L2, 0, 1)
            d = float(np.hypot(*(A + AB * t[:, None] - q).T).min())
            if i == len(P):
                best0 = min(best0, d)
            else:
                best[i] = min(best[i], d)
    return best0 <= ALONGSIDE_M and float(best.max()) <= best0 + ALONGSIDE_SPREAD_M


def place_stations(runs, station_nodes, stations, track=None):
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

    With `track` (track_at's index; build() passes it where the country's rules set
    SNAP_ALONGSIDE), a station whose stop node lies on another variant's track is snapped onto
    this path only where that track runs alongside it (_alongside). Paris Métro 10's one-way
    loop at Auteuil: each direction's path came within 400 m of the other arm's stations and
    took them, so both arms were threaded into one chain of 0.06-0.2 km sections (Mirabeau -
    Église d'Auteuil, Michel-Ange-Molitor - Porte d'Auteuil) and the loop was lost. A stop
    node beside the track (on no variant's path) is snapped as before.
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
        if not best or best[2] > STOP_SNAP_M:
            continue
        if track is not None:
            on = [t for n in station_nodes[st] for t in track.get(n, ())]
            if on and not any(_alongside(xy, c, k, runs) for xy, c, k in on):
                place_stations.refused += 1
                continue
        out.append((best[0], best[1], st))

    out.sort(key=lambda t: (t[0], t[1]))
    dedup = []
    for e in out:
        if dedup and dedup[-1][2] == e[2]:
            continue                      # the same station twice running is one visit
        dedup.append(e)
    return dedup


place_stations.refused = 0


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


_RULES = {}


def country_rules(region):
    """A country's own rules for this file: `rules/<cc>.py`, owned by that country's agent so
    it never has to wait on a change here. Every name in it is optional, and a country without
    the file (or without a name) builds by the default given here. Patterns several countries
    share live in `rules/shared.py` (`from rules.shared import EU_TRAIN`). Read:

      looks_like_service(tags, name, name_en) -> bool
            Is this route=train relation (or route_master) a named train rather than a line?
            Called for route=train only; metro, tram, light rail, monorail and funicular are
            never named trains. `name`, `name_en` are its name and name:en tags ("" if
            unset). Default: False, every train route is a line.
      SERVICE_IF_ALL_ROUTES_ARE = True
            A line whose own tags do not say named train is one anyway when every one of
            its route relations is (is_service). Default: off.
      PLATFORM_SUFFIX = re.compile(...)
            Rail stop names (station records and stop nodes, never tram stops) are read with
            whatever this matches taken off (plain_name). Default: names as tagged.
      METRO_DUP = True
            Same-named metro stations merge within METRO_DUP_RADIUS_M rather than
            DUP_RADIUS_M, and a metro stop node finds its station by name only within
            METRO_NAME_RADIUS_M (merge_duplicate_stations, build_stations). Default: off.
      ROUTE_SHARE_BY_LENGTH = True
            A register section's share run over by OSM passenger routes is measured against
            the section's own length, not the length of its ways (register_way_lines).
            Default: off.
      extra_route_stops(ways, rels, stops, coords, stations, resolved, log)
            -> {route rel id: {station key: set(node id)}}: stations a route relation does
            not list that are stops of it all the same, added to its line's stops in build()
            with those nodes (of the route's own track) as their stop nodes. Called once,
            after build_stations. Default: none, a line's stops are its relations' own.
      SKIP_ROUTES = {relation id, ...}
            Route relations the OSM half leaves out: stale or broken routes (Brazil's
            Teresina route runs on over a disused railway). Default: none.
      route_runs(rid, runs, members, ways, coords, station_nodes, stations) -> runs or None
            A route relation's runs rebuilt (assemble's [(node ids, lon/lat)]) where its
            ways are in no order; None keeps assemble's. Called per variant before
            travel_dirs and place_stations. Default: none.
      REGISTER_KIND_SURE = True
            Register lines keep the register's kind: register_way_lines never re-kinds them
            from the OSM track they lie on (NARN: the Old Colony beside the Red Line came out
            "subway"). Default: off.
      TWIN_ON_STATIONS = True
            An OSM line whose stations all lie on its register match is dropped as its twin
            (merge_sources), whatever its length: Mexico's broken metro routes (Metrorrey
            1-3, Línea 4) whose stop order gives a false end-to-end section. Default: off.
      SNAP_ALONGSIDE = True
            A station whose stop node lies on another variant's track is snapped onto a
            variant's path only where that track runs alongside it (place_stations): keeps a
            one-way loop's two arms apart (Paris Métro 10 at Auteuil). Default: off, every
            station within STOP_SNAP_M is snapped.

    Returns the module, or None where the country has no file.
    """
    if not region:
        return None
    if region not in _RULES:
        import importlib.util
        p = ROOT / "rules" / f"{region}.py"
        mod = None
        if p.exists():
            # A rules file imports the shared patterns as `rules.shared`, which needs this
            # folder on the path whoever imported build_model.
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            spec = importlib.util.spec_from_file_location(f"rules_{region}", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        _RULES[region] = mod
    return _RULES[region]


def looks_like_service(tags, kind, region):
    """Is this a named TRAIN rather than a LINE?

    OpenStreetMap maps both as route relations, often with no tag that separates them: a line
    and a named train that runs over it are the same kind of object, so a corridor is counted
    once as a line and again under every named train over it. Which is which is a rule per
    country, in its `rules/<cc>.py` (`looks_like_service`; a country without one has no named
    trains). Metro, tram, light rail and monorail are never services, so only route=train is
    tested. This sets a FLAG and drops nothing: what the flag is for is deciding which lines
    a completion percentage counts, and that is a judgement about what the hobby means, not
    a data question.
    """
    if kind != "train":
        return False
    rules = country_rules(region)
    if rules is None or not hasattr(rules, "looks_like_service"):
        return False
    return bool(rules.looks_like_service(tags, tags.get("name", ""), tags.get("name:en", "")))


def is_service(mtags, kind, rids, routes, region):
    """A line's `service` flag: its own tags say named train (looks_like_service) or, where
    the country's rules set SERVICE_IF_ALL_ROUTES_ARE, every one of its routes does."""
    if looks_like_service(mtags, kind, region):
        return True
    return (bool(getattr(country_rules(region), "SERVICE_IF_ALL_ROUTES_ARE", False))
            and bool(rids)
            and all(looks_like_service(routes[r][0], kind, region) for r in rids))


# A free-standing train number: 821 in "台灣高鐵 821", 1 in "のぞみ1号". Not the 1 of "S1" or
# "S11", whose numbers are the line's name.
TRAIN_NUMBER = re.compile(r"(?<![A-Za-z0-9])\d{1,4}(?![A-Za-z0-9])")


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
    # Malaysia: the register writes the Klang Valley's lines in Malay, "Laluan Kelana Jaya",
    # "Laluan Monorel KL"; OSM's route masters are Malay or English, "Laluan Kajang",
    # "Kelana Jaya Line", "KL Monorail". Only names that start "Laluan" change.
    m = re.match(r"^Laluan Monorel (.+)$", s)
    if m:
        s = f"{m.group(1)} Monorail"
    m = re.match(r"^Laluan (.+)$", s)
    if m:
        s = f"{m.group(1)} Line"
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


def first_word(name):
    """A name's first word once the direction is out: 'Eurostar: Paris - Amsterdam' -> 'eurostar'."""
    return (re.split(r"[\s:：]+", strip_direction(name or "").strip()) or [""])[0].casefold()


_ARROW = re.compile(r"\s*(?:-->|->|=>|→|⇒|↔|<=>|⇄)\s*")


def arrow_ends(name):
    """The places either side of a name's arrows, as a set: '小倉 => 下関' -> {小倉, 下関}."""
    parts = [p.strip() for p in _ARROW.split(name or "")]
    return frozenset(parts) if len(parts) > 1 and all(parts) else None


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
                shared_ref = refs[l["id"]] & refs[o]
                if not (shared_ref or same_name or same_en):
                    continue
                # A shared ref alone does not make two named trains one: in be both European
                # Sleeper routes and the Eurostar are ref "ES". Their names must also start
                # with the same word, which keeps a train's two directions together ("NJ
                # 40235 ..." / "NJ 235 ...", "AVE Madrid - Málaga" / "AVE Málaga - Madrid"),
                # and parts はやて and はやぶさ, which OSM gives the same ref.
                # Or the same two ends either way round: OSM Japan names some trains by their
                # ends alone, "小倉 => 下関" and "下関 => 小倉".
                if l["service"] and not (same_name or same_en):
                    ok = (first_word(l["name"]) == first_word(m["name"])
                          or (arrow_ends(l["name"]) or 0) == arrow_ends(m["name"]))
                    log(f"twins on ref {'/'.join(sorted(shared_ref))} alone, "
                        f"{'merged' if ok else 'kept apart'}: {l['name']!r} / {m['name']!r}")
                    if not ok:
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


def merge_sources(osm, n02, log, region=None):
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
        if hit is not None and (is_twin(l, hit) or (
                getattr(country_rules(region), "TWIN_ON_STATIONS", False)
                and line_stations(l) and line_stations(l) <= line_stations(hit))):
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


# A transit section is at least this long between its two border points.
TRANSIT_MIN_KM = 1.0


def transit_tail(runs, members, resolved, bidx, region):
    """A route that crosses this country without a stop in it: its track from the border point
    where it comes in to the one where it leaves, as one section between the two (London -
    Amsterdam through France: OSM lists no French stop, so France built nothing of it and the
    line had a hole from the tunnel to Belgium; Anita, 2026-10-05). Only where the route lists
    stops this extract does not have (it runs on abroad), and only between two different
    points of this country's own borders at least TRANSIT_MIN_KM apart along the track, so a
    route wholly abroad that the extract's margin happens to hold never counts. The longest
    such stretch over the route's runs, as a border_tails row with "bp0" for its first end, or
    None."""
    if all(n in resolved for n in stop_members(members)):
        return None
    best = None
    for ids, xy in runs:
        hits = [h for h in bidx.along(xy) if region in h[1]["countries"]]
        if len(hits) < 2:
            continue
        h0, h1 = hits[0], hits[-1]
        if h0[1]["id"] == h1[1]["id"] or (h1[0] - h0[0]) / 1000 < TRANSIT_MIN_KM:
            continue
        _a0, p0, k0, fx0, fy0, _o0 = h0
        _a1, p1, k1, fx1, fy1, _o1 = h1
        g = np.vstack([[[fx0, fy0]], xy[k0 + 1:k1 + 1], [[fx1, fy1]]])
        km = path_length_m(g) / 1000
        if best is None or km > best["km"]:
            best = {"st": None, "bp0": p0, "bp": p1, "geom": g, "km": km,
                    "digest": track_key(ids[k0:k1 + 2]),
                    "ids": np.concatenate([[-1], ids[k0 + 1:k1 + 1], [-1]])}
    return best


def build(region, log):
    ways, rels, stops, cid, cx, cy = load(region, log)
    coords = Coords(cid, cx, cy)
    stations, resolved = build_stations(stops, rels, coords, log, region)
    groups, routes = group_lines(rels, log)
    # Stops the country's rules add to relations that do not list them (rules/<cc>.py).
    rules = country_rules(region)
    extra_stops = {}
    if rules is not None and hasattr(rules, "extra_route_stops"):
        extra_stops = rules.extra_route_stops(ways, rels, stops, coords, stations, resolved,
                                              log) or {}
    # Route relations the country's rules leave out (rules/<cc>.py SKIP_ROUTES): stale or
    # broken OSM routes (Teresina's runs on over a disused railway to the coast).
    skip = set(getattr(rules, "SKIP_ROUTES", ()) or ())
    if skip:
        n0 = len(groups)
        groups = [(lid, mtags, [r for r in rids if r not in skip])
                  for lid, mtags, rids in groups]
        groups = [g for g in groups if g[2]]
        log(f"  {len(skip)} route relations left out by the country's rules (SKIP_ROUTES); "
            f"{n0 - len(groups)} lines with no route left")
    # Snap a station onto a variant's path only where the track its stop node lies on runs
    # alongside (place_stations; rules/<cc>.py SNAP_ALONGSIDE).
    snap_alongside = bool(getattr(rules, "SNAP_ALONGSIDE", False))
    place_stations.refused = 0
    snap_refused = []

    lines, geoms = [], {}
    way_lines = defaultdict(set)     # OSM way id -> the lines that run over it
    # Line id -> one {way id: +1 / -1} per route relation: which way the relation travels each
    # of its ways, along or against the way's node order. ownership.py pairs a line's two
    # directions' tracks from it (directional_pairs).
    variant_dirs = defaultdict(list)
    build.variant_dirs = variant_dirs
    n_gap = n_sec = n_nostop = n_ends = n_transit = 0
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
            for st, nodes in extra_stops.get(rid, {}).items():
                if st in stations:
                    station_nodes[st].update(nodes)
        if len(station_nodes) < 2 and pick(mtags, "route", "route_master") == "funicular":
            n_ends += funicular_ends(rids, routes, ways, coords, stations, station_nodes)
        # One station is enough when the route runs on over the border from it (a line whose
        # only stop in this country is its last before the border); border_tails decides.
        if not station_nodes:
            # No stop here at all: a route passing through on its way between two other
            # countries keeps its track from border to border (transit_tail).
            tr = None
            for rid in rids:
                runs = assemble(routes[rid][1], ways, coords)
                t = transit_tail(runs, routes[rid][1], resolved, bidx, region) if runs else None
                if t is not None and (tr is None or t["km"] > tr["km"]):
                    tr = t
            if tr is None:
                n_nostop += 1
                continue
            n_transit += 1
            for rid in rids:
                for ty, ref, role in routes[rid][1]:
                    if ty == "w" and (not role or role.startswith(("forward", "backward"))):
                        way_lines[ref].add(lid)
            kind = pick(mtags, "route", "route_master")
            build.border_only.append({
                "id": lid, "service": is_service(mtags, kind, rids, routes, region),
                "name": pick(mtags, "name") or pick(mtags, "ref"),
                "name_en": pick(mtags, "name:en"), "ref": pick(mtags, "ref"),
                "colour": pick(mtags, "colour", "color"), "operator": pick(mtags, "operator"),
                "operator_en": pick(mtags, "operator:en"), "network": pick(mtags, "network"),
                "kind": kind, "km": 0.0, "variants": 0, "straight_sections": 0,
                "display": [], "sections": [], "_digest": [], "_tails": [tr]})
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
        allruns = {rid: assemble(routes[rid][1], ways, coords) for rid in rids}
        track = track_at(allruns, station_nodes) if snap_alongside else None
        refused0 = place_stations.refused
        for rid in rids:
            tags, members = routes[rid]
            runs = allruns[rid]
            # A country's repair of a route relation whose ways are in no order (rules/<cc>.py
            # route_runs; the US's old both-ways NJ Transit relations, 2026-10-08): its runs
            # rebuilt from its own track and stops, or None to keep assemble's.
            if runs and rules is not None and hasattr(rules, "route_runs"):
                runs = rules.route_runs(rid, runs, members, ways, coords, station_nodes,
                                        stations) or runs
            if not runs:
                continue
            variant_dirs[lid].append(travel_dirs(members, ways, runs))
            placed = place_stations(runs, station_nodes, stations, track)
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
        if place_stations.refused > refused0:
            snap_refused.append((lid, pick(mtags, "name") or pick(mtags, "ref"),
                                 place_stations.refused - refused0))

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
            "service": is_service(mtags, kind, rids, routes, region),
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
        f"{n_ends} funiculars mapped with no stops took the two ends of their track; "
        f"{n_transit} lines with no stop here cross it border to border")
    log(f"  gaps in a route relation traced along track instead: {n_own} over the line's own "
        f"ways, {n_net} over the wider network")
    if snap_alongside:
        log(f"  {place_stations.refused} times a station within {STOP_SNAP_M} m of a variant's "
            f"path was not snapped onto it: its stop node lies on another variant's track, "
            f"which does not run alongside; {len(snap_refused)} lines:")
        for lid, name, n in sorted(snap_refused, key=lambda t: -t[2]):
            log(f"    {n:3d}  {lid} {name}")
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

    A register that knows trains run over a junction-ended section no OSM route maps lists it
    in the line's `served_sections` ("a|b" keys), and it is kept: Thailand's Padang Besar -
    border (SRT 45/46, the Hat Yai shuttles), Malaysia's run-ons to Padang Besar and over the
    Johor Causeway. Taken off the line here, so never shipped.

    Returns the ids of lines left with no sections at all, which the caller drops.
    """
    served = {(l["id"], k) for l in lines for k in (l.pop("served_sections", None) or ())}
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
            if (route_share.get((l["id"], f"{a}|{b}"), 0.0) >= NEEDS_ROUTE_SHARE
                    or (l["id"], f"{a}|{b}") in served):
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


# A junction end within this of another line's track joins it. Measured on the shipped data
# (2026-10-07): of the leaf junctions not shared with another line, us 342 lie within 25 m of
# another line, 47 within 25-300 m and nearly all of those are real meetings of two FRA
# subdivisions drawn a little apart (St Paul / Staples 28 m, end of Elko 76 and 237 m: the
# Empire Builder's and the California Zephyr's track); past 300 m, yards and terminals.
CONTACT_M = 300.0
# ... unless its name says the track ends there: a depot, a yard, a siding's end. The
# registers name such points plainly (Schienennetz's "Gleisende", "Depot", "Abstellgruppe",
# "(Agl)"; RINF's "Gbf", "Rbf", "Abstellbahnhof"; "fin de voie", "cul-de-sac").
DEAD_END_NAME = re.compile(
    r"gleisende|fin de voie|fine binario|cul-de-sac|\bdepot\b|\bdép[ôo]t\b|\bdep\.|abstell|"
    r"unterhalt|rangier|\brb\b|\bgb\b|\bgbf\b|\brbf\b|\bbw\b|fracht|\(agl\)|übergabe|triage|"
    r"\byard\b|\bshops?\b|stahlwerk|\bmüll\b", re.I)


BORDER_ANCHOR_DEG = 0.03   # ~3 km: a junction end this near another country's land leads abroad
STOP_NEAR_M = 500.0        # ... and one this near a station not on its own line leads there


def prune_dead_track(region, lines, stations, geoms, log):
    """Drop track of a line that leads to no station (Anita, 2026-10-07: "if past a junction
    there are no stations, we should not be drawing this track at all ... we shouldnt consider
    it as having passenger rail").

    A section is kept when it lies on some way between two ANCHORS of its line: its stops,
    border points and junctions within BORDER_ANCHOR_DEG of another country's land (the line
    goes on abroad: Stabio to Varese has no border point), and junctions where the line meets
    other track (a section of another line ends there, or another line passes within
    CONTACT_M, unless the junction's name says the track ends: DEAD_END_NAME), from which
    trains run on to other stations, or a station not on the line lies within STOP_NEAR_M. What is left
    is track with no station beyond it: a stub siding, a yard, a register line's run to the
    "end of" a subdivision, a turning loop with no stop, a stop alone on its piece of the
    line with its stub. Leaf track off a non-anchor is pruned repeatedly; then any block of
    the line's graph (a loop, a pair of parallel tracks) reached from fewer than two anchors
    goes too, until nothing changes. A line left with nothing is dropped.

    Kept as before: a line's run to a junction where it meets other lines, however long (the
    Hastings Subdivision's 154 km to Creston: the app rides it through the junction to the
    stops beyond). Returns the ids of lines dropped whole."""
    import networkx as nx
    from shapely import LineString as LS, Point, STRtree
    import borders
    border_ids = {p["id"] for p in borders.load()}
    is_stop = lambda n: n in stations and not stations[n].get("junction")

    node_lines = defaultdict(set)
    for l in lines:
        for s in l["sections"]:
            node_lines[s[0]].add(l["id"]); node_lines[s[1]].add(l["id"])
    lats = [p[1] for g in geoms.values() for pts in g.values() for p in pts[:1]]
    lat0 = float(np.median(lats)) if lats else 0.0
    kx, ky = math.cos(math.radians(lat0)) * 111320.0, 110570.0
    tgeo, towner, tfam = [], [], []
    for l in lines:
        fam = kind_family(l.get("kind"))
        for k, pts in geoms.get(l["id"], {}).items():
            if len(pts) < 2:
                continue
            a = np.asarray(pts, dtype=np.float64)[:, :2]
            tgeo.append(LS(np.column_stack([a[:, 0] * kx, a[:, 1] * ky])))
            towner.append(l["id"]); tfam.append(fam)
    tree = STRtree(tgeo) if tgeo else None

    import ownership
    land = ownership._land()
    me = borders.ISO.get(region, region.upper()).lower()

    def near_abroad(s):
        if land.get("tree") is None:
            return False
        hit = land["tree"].query(Point(s["lon"], s["lat"]), predicate="dwithin",
                                 distance=BORDER_ANCHOR_DEG)
        return any(land["codes"][i] not in (me, "-99") for i in hit)

    stop_ids = [sid for sid, s in stations.items() if not s.get("junction") and "lon" in s]
    stop_tree = STRtree([Point(stations[sid]["lon"] * kx, stations[sid]["lat"] * ky)
                         for sid in stop_ids]) if stop_ids else None

    def contact(n, lid, fam, own=frozenset()):
        if node_lines[n] - {lid}:
            return True
        s = stations.get(n)
        if not s:
            return False
        if near_abroad(s):
            return True
        if DEAD_END_NAME.search(s.get("name") or "") or DEAD_END_NAME.search(s.get("name_en") or ""):
            return False
        # A station of another line, or of none, close by: the track leads to it (Halifax's
        # station is 400 m past the end of the Bedford Subdivision, and no line is drawn there
        # but the VIA Ocean's, which stops short of it at Truro in Canada's data).
        if stop_tree is not None:
            p = Point(s["lon"] * kx, s["lat"] * ky)
            if any(stop_ids[i] not in own
                   for i in stop_tree.query(p, predicate="dwithin", distance=STOP_NEAR_M)):
                return True
        # The FRA's (us_register's) name for a junction by a station: "near Halifax", where the
        # station is in OSM but on no line of the model. Not the line's own last stop.
        m = re.match(r"near (.+)$", s.get("name") or "")
        if m and m.group(1) not in {stations[x]["name"] for x in own
                                    if x in stations and is_stop(x)}:
            return True
        if tree is None:
            return False
        p = Point(s["lon"] * kx, s["lat"] * ky)
        # Any kind: a register's legal "rail" line can be a tram link (Zürich's Tramstrasse).
        return any(towner[i] != lid
                   for i in tree.query(p, predicate="dwithin", distance=CONTACT_M))

    drop_lines, per_line, dead_nodes, dead_keys = set(), [], {}, {}
    n_sec = 0
    km_total = 0.0
    for l in lines:
        secs = [s for s in l["sections"] if s[0] != s[1]]
        if not secs:
            continue
        fam = kind_family(l.get("kind"))
        G = nx.Graph()
        for s in secs:
            G.add_edge(s[0], s[1])
        anchor = {}

        own_nodes = frozenset(G.nodes)

        def is_anchor(n):
            if n not in anchor:
                anchor[n] = (is_stop(n) or n in border_ids or contact(n, l["id"], fam, own_nodes))
            return anchor[n]

        changed = True
        while changed:
            changed = False
            todo = [n for n in G if G.degree(n) <= 1 and not is_anchor(n)]
            while todo:
                n = todo.pop()
                if n not in G or G.degree(n) > 1 or is_anchor(n):
                    continue
                nb = list(G.neighbors(n))
                G.remove_node(n)
                changed = True
                todo += [m for m in nb if G.degree(m) <= 1 and not is_anchor(m)]
            for comp in list(nx.biconnected_component_edges(G)):
                if len(comp) < 2:
                    continue        # a bridge: leaf pruning above has settled it
                block = {x for e in comp for x in e}
                H = G.copy()
                H.remove_edges_from(comp)
                ports = 0
                for x in block:
                    if is_anchor(x) or any(is_anchor(y) for y in nx.node_connected_component(H, x)):
                        ports += 1
                        if ports >= 2:
                            break
                if ports < 2:
                    G.remove_edges_from(comp)
                    G.remove_nodes_from([x for x in block if G.degree(x) == 0])
                    changed = True
            # a piece of the line with fewer than two anchors leads nowhere either
            for comp in list(nx.connected_components(G)):
                if sum(1 for x in comp if is_anchor(x)) < 2:
                    G.remove_nodes_from(comp)
                    changed = True
        keep = [s for s in l["sections"]
                if (s[0] == s[1] and s[0] in G) or G.has_edge(s[0], s[1])]
        if len(keep) == len(l["sections"]):
            continue
        gone = [s for s in l["sections"] if s not in keep]
        km = sum(s[2] for s in gone)
        n_sec += len(gone)
        km_total += km
        per_line.append((km, l["name"], l["id"], len(gone), not keep))
        if not keep and os.environ.get("PRUNE_DEBUG"):
            for n in sorted({x for s in gone for x in s[:2]}):
                s_ = stations.get(n) or {}
                log(f"      {l['name']}: {s_.get('name')} stop={is_stop(n)} anchor={anchor.get(n)} "
                    f"lines_here={sorted(node_lines[n] - {l['id']})[:4]}")
        dead = {f"{s[0]}|{s[1]}" for s in gone}
        g = geoms.get(l["id"], {})
        ids_of = lambda ss: {int(x) for s in ss
                             for x in (getattr(g.get(f"{s[0]}|{s[1]}"), "ids", None)
                                       if getattr(g.get(f"{s[0]}|{s[1]}"), "ids", None) is not None
                                       else ()) if x > 0}
        dead_nodes[l["id"]] = ids_of(gone) - ids_of(keep)
        dead_keys[l["id"]] = (dead, {f"{s[0]}|{s[1]}" for s in keep})
        for s in gone:
            g.pop(f"{s[0]}|{s[1]}", None)
        l["sections"] = keep
        l["km"] = round(sum(s[2] for s in keep), 3)
        ends = {x for s in keep for x in s[:2]}
        l["display"] = [x for x in l["display"] if x in ends]
        for k in ("closed", "highspeed_sections", "borrowed", "suspended"):
            v = l.get(k)
            if isinstance(v, list):
                l[k] = [x for x in v if not (isinstance(x, str) and x in dead)]
            elif isinstance(v, dict):
                l[k] = {x: y for x, y in v.items() if x not in dead}
        for x in {y for s in gone for y in s[:2]} - ends:
            if x in stations:
                stations[x]["lines"].discard(l["id"])
        if not keep:
            drop_lines.add(l["id"])
            geoms.pop(l["id"], None)
    for s in stations.values():
        s["lines"] -= drop_lines
    log(f"track leading to no station: {n_sec} sections, {km_total:,.1f} km dropped from "
        f"{len(per_line)} lines, {len(drop_lines)} lines left with nothing")
    for km, name, lid, n, whole in sorted(per_line, reverse=True)[:25]:
        log(f"    {km:8.2f} km  {n:3d} sec  {name} ({lid}){'  WHOLE LINE' if whole else ''}")
    prune_dead_track.report = per_line
    prune_dead_track.dead_nodes = dead_nodes      # line id -> OSM nodes only its dropped track had
    prune_dead_track.dead_keys = dead_keys        # line id -> ("a|b" dropped, "a|b" kept)
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
    # A country whose register knows its lines' kind (rules REGISTER_KIND_SURE: NARN's are all
    # railroads) keeps it: the track test called the MBTA's Old Colony Line "subway" from the
    # Red Line running beside it.
    kind_sure = bool(getattr(country_rules(region), "REGISTER_KIND_SURE", False))
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
        if (fam in ("rail", "tram") and top != fam and not kind_sure
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
    # Where the country's rules set ROUTE_SHARE_BY_LENGTH, the share is of the section's own
    # length: the metres of its ways a route runs over, against the section, not against all
    # its ways (rules/us.py says why).
    sec_len = {}
    if getattr(country_rules(region), "ROUTE_SHARE_BY_LENGTH", False):
        for l in lines:
            for skey, pts in geoms.get(l["id"], {}).items():
                if len(pts) >= 2:
                    a = np.asarray(pts, dtype=np.float64)
                    sec_len[(l["id"], skey)] = LineString(proj(a[:, 0], a[:, 1])).length
    for (lid, skey), beside in sec_ways.items():
        mine = [(j, got) for j, got, _d in beside if lid in out.get(wids[j], ())]
        total = sum(got for _j, got in mine)
        on = sum(got for j, got in mine if wids[j] in on_route)
        if (lid, skey) in sec_len:
            route_share[(lid, skey)] = min(1.0, on / sec_len[(lid, skey)]) \
                if sec_len[(lid, skey)] else 0.0
        else:
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
            p = t["bp"]
            b = p["id"]
            # A transit section (transit_tail) starts at a border point too.
            a = t["bp0"]["id"] if t.get("bp0") else alias.get(t["st"], t["st"])
            if (not t.get("bp0") and a not in stations) or frozenset((a, b)) in have:
                continue
            for q in ([t["bp0"]] if t.get("bp0") else []) + [p]:
                if q["id"] not in stations:
                    stations[q["id"]] = {"id": q["id"], "name": q["name"], "name_en": "",
                                         "lon": q["lon"], "lat": q["lat"], "lines": set(),
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


# A line's two sections at a junction that leave it along the same rails for this long are one
# section with the spur cut out (fold_spurs); closer than SPUR_NEAR_M counts as the same rails.
SPUR_MIN_M = 150
SPUR_NEAR_M = 25
# ... and only where the line runs through the fork: its two legs, each taken this far out,
# part at more than this angle (a switchback's legs leave the fork together, in a V).
SPUR_HEAD_M = 200
SPUR_THROUGH_DEG = 100
# A detour x - spur - y beside the line's own direct section x - y goes if, without its spur,
# it is within this (or 5%) of the direct section's length.
SPUR_SAME_KM = 0.5


def _spur_len(p, q, near_m=SPUR_NEAR_M):
    """p and q both start at a junction: how many leading points of p run within near_m of
    q's first points, and the metres they cover."""
    if len(p) < 2 or len(q) < 2:
        return 0, 0.0
    k = math.cos(math.radians(p[0][1])) * 111320
    head = np.asarray(q[:400], dtype=np.float64)
    run, i = 0.0, 0
    for i in range(1, len(p)):
        x = p[i]
        d = np.hypot((head[:, 0] - x[0]) * k, (head[:, 1] - x[1]) * 110570).min()
        if d > near_m:
            return i - 1, run
        run += math.hypot((x[0] - p[i - 1][0]) * k, (x[1] - p[i - 1][1]) * 110570)
    return len(p) - 1, run


def fold_spurs(lines, stations, geoms, log, state=None):
    """A junction in a line's middle that the line reaches up a spur and leaves down it again:
    the two sections meeting there leave the junction along the same rails, so the line's
    track drawn on the map has a branch going nowhere (gb's South Wales Main Line ran up the
    Westerleigh curve to 'Junction near Yate' and back; Anita, 2026-10-05: "branches that
    don't actually lead to any stations, but they're still highlighted"). The two sections
    become one, from the fork where they part: the spur is cut out of both, its length off
    the line's km; the register's chainage and the per-section flags move to the new key.
    Only at a junction (never a stop) with exactly two of the line's sections, where they share
    at least SPUR_MIN_M, and where the line runs through the fork (through_at_fork): at a
    switchback trains do reverse up the headshunt, and its two legs leave the fork together
    (Czechia's 143 Chodov - Nová Role reverses at Chodov-úvrať). Runs after the register's
    split_pieces, whose bridges can make them, and moves state["sec_ways"] (what ownership
    reads) to the new section."""
    n, km_cut = 0, 0.0
    sec_ways = (state or {}).get("sec_ways")
    near = None

    def oriented(pts, st):
        d0 = (pts[0][0] - st["lon"]) ** 2 + (pts[0][1] - st["lat"]) ** 2
        d1 = (pts[-1][0] - st["lon"]) ** 2 + (pts[-1][1] - st["lat"]) ** 2
        return d0 <= d1

    def through_at_fork(ra, rb):
        """ra, rb: the two legs from the fork (where the spur's rails part) to the far ends.
        A route through the fork leaves it both ways (an angle over SPUR_THROUGH_DEG between
        the legs, each taken SPUR_HEAD_M out); a switchback's legs leave it together, in a V."""
        def head(r):
            k = math.cos(math.radians(r[0][1])) * 111320
            run = 0.0
            for i in range(1, len(r)):
                run += math.hypot((r[i][0] - r[i - 1][0]) * k, (r[i][1] - r[i - 1][1]) * 110570)
                if run >= SPUR_HEAD_M:
                    break
            return ((r[i][0] - r[0][0]) * k, (r[i][1] - r[0][1]) * 110570)
        (ax, ay), (bx, by) = head(ra), head(rb)
        na, nb = math.hypot(ax, ay), math.hypot(bx, by)
        if not na or not nb:
            return False
        cos = (ax * bx + ay * by) / (na * nb)
        return math.degrees(math.acos(max(-1.0, min(1.0, cos)))) > SPUR_THROUGH_DEG

    for l in lines:
        if l.get("service"):
            continue
        g = geoms.get(l["id"])
        if not g:
            continue
        changed = False
        while True:
            ends = defaultdict(list)
            for sec in l["sections"]:
                ends[sec[0]].append(sec)
                ends[sec[1]].append(sec)
            done = False
            for j, secs in ends.items():
                st = stations.get(j)
                if len(secs) != 2 or not st or not st.get("junction"):
                    continue
                (sa, sb) = secs
                ka, kb = f"{sa[0]}|{sa[1]}", f"{sb[0]}|{sb[1]}"
                pa, pb = g.get(ka), g.get(kb)
                if not pa or not pb or len(pa) < 2 or len(pb) < 2 or ka == kb:
                    continue

                fa, fb = oriented(pa, st), oriented(pb, st)
                ca = list(pa) if fa else list(pa)[::-1]
                cb = list(pb) if fb else list(pb)[::-1]
                ia, ma = _spur_len(ca, cb)
                ib, mb = _spur_len(cb, ca)
                if min(ma, mb) < SPUR_MIN_M or ia >= len(ca) - 1 or ib >= len(cb) - 1:
                    continue
                if not through_at_fork(ca[ia:], cb[ib:]):
                    continue
                x = sa[1] if sa[0] == j else sa[0]
                y = sb[1] if sb[0] == j else sb[0]
                kn = f"{x}|{y}"
                if x == y:
                    continue
                direct = next((s for s in l["sections"] if {s[0], s[1]} == {x, y}), None)
                if direct is not None:
                    # The line already runs x - y directly, and x - j - y is the same track
                    # with a spur up to j (gb's Peterborough to Lincoln, 16.8 km up and back at
                    # Lincoln): the detour goes, if without its spur it is the direct one's
                    # length. Anything else is a real second route and stays.
                    via = (path_length_m(ca[ia:]) + path_length_m(cb[ib:])) / 1000
                    if abs(via - direct[2]) > max(SPUR_SAME_KM, 0.05 * direct[2]):
                        continue
                    l["sections"] = [s for s in l["sections"] if s is not sa and s is not sb]
                    for k in (ka, kb):
                        g.pop(k, None)
                        for name in ("chain", "highspeed_sections"):
                            if isinstance(l.get(name), dict):
                                l[name].pop(k, None)
                        for name in ("served_sections", "borrowed"):
                            v = l.get(name)
                            if v and k in v:
                                rest = [q for q in v if q != k]
                                l[name] = set(rest) if isinstance(v, set) else type(v)(rest)
                        if sec_ways is not None:
                            sec_ways.pop((l["id"], k), None)
                    km_cut += sa[2] + sb[2]
                    l["display"] = [s for s in l.get("display", []) if s != j]
                    st["lines"].discard(l["id"])
                    log(f"  spur detour dropped: {l['name'][:50]} at {st.get('name', j)}: "
                        f"{min(ma, mb):.0f} m up and back beside its own direct section")
                    n += 1
                    changed = done = True
                    break
                if kn in g or f"{y}|{x}" in g:
                    continue
                ida = getattr(pa, "ids", None)
                idb = getattr(pb, "ids", None)
                if ida is not None:
                    ida = list(ida) if fa else list(ida)[::-1]
                if idb is not None:
                    idb = list(idb) if fb else list(idb)[::-1]
                # x ... fork (A from its far end back to the spur's base), then fork ... y
                coords = ca[ia:][::-1] + cb[ib:]
                ids = (ida[ia:][::-1] + idb[ib:]) if ida is not None and idb is not None else None
                km = round(path_length_m(coords) / 1000, 3)
                km_cut += sa[2] + sb[2] - km
                l["sections"] = [s for s in l["sections"] if s is not sa and s is not sb]
                l["sections"].append([x, y, km])
                del g[ka], g[kb]
                g[kn] = Pts(coords, ids)
                for name in ("chain", "highspeed_sections"):
                    d = l.get(name)
                    if isinstance(d, dict) and (ka in d or kb in d):
                        va, vb = d.pop(ka, None), d.pop(kb, None)
                        if name == "chain":
                            d[kn] = round((va or 0.0) + (vb or 0.0), 3)
                        else:
                            d[kn] = bool(va) or bool(vb)
                for name in ("served_sections", "borrowed"):
                    v = l.get(name)
                    if v and (ka in v or kb in v):
                        rest = [k for k in v if k not in (ka, kb)]
                        l[name] = type(v)(rest + [kn]) if not isinstance(v, set) else set(rest) | {kn}
                l["display"] = [s for s in l.get("display", []) if s != j]
                st["lines"].discard(l["id"])
                if sec_ways is not None:
                    sec_ways.pop((l["id"], ka), None)
                    sec_ways.pop((l["id"], kb), None)
                    if near is None:
                        import pieces
                        near = pieces.WaysNear(state)
                    sec_ways[(l["id"], kn)] = near.beside(coords, l.get("kind", "rail"))
                log(f"  spur folded: {l['name'][:50]} at {st.get('name', j)}: "
                    f"{min(ma, mb):.0f} m up and back")
                n += 1
                changed = done = True
                break
            if not done:
                break
        if changed:
            l["km"] = round(sum(s[2] for s in l["sections"]), 3)
    log(f"spurs: {n} out-and-back spurs at a line's junction folded out ({km_cut:,.1f} km)")


# A register line's piece more than this far (crow-fly, nearest stations) from its biggest piece
# is a line of its own (split_far_pieces).
FAR_PIECE_KM = 25


def split_far_pieces(lines, stations, geoms, reg_ways, state, log, keep_whole=()):
    """A register line whose passenger pieces lie far apart becomes one line per far piece
    (Anita, 2026-10-05: "ok we can split lines", for lines whose pieces are separate services
    on one register line with a closed or freight middle: France's Chartres - Bordeaux in four
    pieces up to 229 km apart). As the US and Canada have done since 2026-10-04 (pieces.py),
    but only for a piece over FAR_PIECE_KM from the line's biggest piece: a nearer one is
    more likely a hole trains run through, on handoff_notes/lines_in_pieces.md's list. A far
    piece with under two stops cannot be ridden as a trip and stays with its line. The biggest
    piece keeps the id; a split piece takes a hash of the line's id and its lowest stop, and a
    name_en from its end stops; km_official is shared out by km; reg_ways and
    state["sec_ways"] move to the new ids. A name in `keep_whole` (the register module's
    KEEP_WHOLE: cn's 青荣城际线, whose gap is track missing from OSM) is left alone. Returns
    {line id: [split ids]} for aliases.json's `pieces`, which lets a ride saved on the old id
    find its piece."""
    import pieces as pc
    from n02 import walk_order
    sec_ways = (state or {}).get("sec_ways") or {}
    wids = (state or {}).get("wids") or []
    made, out = [], {}

    def km_apart(a, b):
        return min(dist_m(stations[x]["lon"], stations[x]["lat"],
                          stations[y]["lon"], stations[y]["lat"]) for x in a for y in b) / 1000

    for l in list(lines):
        if (l.get("src", "osm") == "osm" or l.get("service") or not l.get("sections")
                or l["name"] in keep_whole):
            continue
        ps = pc.section_pieces(l["sections"])
        if len(ps) < 2:
            continue
        lid, secs = l["id"], l["sections"]
        nodes = [{s for i in ix for s in secs[i][:2]} for ix in ps]
        big = nodes[0]
        far = []
        for ix, ns in zip(ps[1:], nodes[1:]):
            stops = sorted(s for s in ns if not stations[s].get("junction"))
            if len(stops) >= 2 and km_apart(ns, big) > FAR_PIECE_KM:
                far.append((ix, ns, stops))
        if not far:
            continue
        total = sum(s[2] for s in secs) or 1.0
        old_geo = geoms.get(lid, {})
        moved = set()
        news = []
        for ix, ns, stops in far:
            sub = [secs[i] for i in ix]
            keys = {f"{s[0]}|{s[1]}" for s in sub}
            moved |= set(ix)
            p = dict(l)
            p["id"] = lid[0] + hashlib.blake2b(f"{lid}|piece|{stops[0]}".encode("utf-8"),
                                               digest_size=5).hexdigest()
            p["sections"] = sub
            p["km"] = round(sum(s[2] for s in sub), 3)
            p["display"] = walk_order([(s[0], s[1]) for s in sub])
            for k in ("borrowed", "highspeed_sections", "closed"):
                v = l.get(k)
                if isinstance(v, list):
                    p[k] = [x for x in v if x in keys]
                elif isinstance(v, dict):
                    p[k] = {x: y for x, y in v.items() if x in keys}
            if l.get("km_official"):
                p["km_official"] = round(l["km_official"] * p["km"] / total, 3)
            geoms[p["id"]] = {k: old_geo.pop(k) for k in keys if k in old_geo}
            ws = set()
            for k in keys:
                got = sec_ways.pop((lid, k), None)
                if got is not None:
                    sec_ways[(p["id"], k)] = got
                    ws |= {wids[j] for j, *_r in got}
            for wid in ws:
                if lid in reg_ways.get(wid, ()):
                    reg_ways[wid].add(p["id"])
            for sid in ns:
                stations[sid]["lines"].add(p["id"])
            news.append(p)
        rest = [s for i, s in enumerate(secs) if i not in moved]
        keep_nodes = {s for sec in rest for s in sec[:2]}
        for _ix, ns, _st in far:
            for sid in ns - keep_nodes:
                stations[sid]["lines"].discard(lid)
        # a way only the moved sections lay beside is no longer this line's
        own_ws = {wids[j] for k in (f"{s[0]}|{s[1]}" for s in rest)
                  for j, *_r in sec_ways.get((lid, k), ())}
        for wid, ls in reg_ways.items():
            if lid in ls and wid not in own_ws and any(p["id"] in ls for p in news):
                ls.discard(lid)
        km_before = l["km"]
        l["sections"] = rest
        l["km"] = round(sum(s[2] for s in rest), 3)
        l["display"] = [s for s in l.get("display", []) if s in keep_nodes] or \
            walk_order([(s[0], s[1]) for s in rest])
        if l.get("km_official"):
            l["km_official"] = round(l["km_official"] * l["km"] / total, 3)
        for k in ("borrowed", "highspeed_sections", "closed"):
            v = l.get(k)
            keys = {f"{s[0]}|{s[1]}" for s in rest}
            if isinstance(v, list):
                l[k] = [x for x in v if x in keys]
            elif isinstance(v, dict):
                l[k] = {x: y for x, y in v.items() if x in keys}
        # the name_en from before the split, less any "(first – last)" a register already gave
        # it (au names two lines of one name by their ends: no second pair of brackets)
        base = dict(l, name_en=re.sub(r" \([^()]* – [^()]*\)$", "",
                                      news[0].get("name_en") or ""))
        for p in [l] + news:
            st_ = [s for s in p["display"] if not stations[s].get("junction")] or p["display"]
            if len(st_) >= 2 and st_[0] != st_[-1]:
                p["name_en"] = pc.english_piece_name(l["name"], base, stations, st_[0], st_[-1])
        lines.extend(news)
        out[lid] = [p["id"] for p in news]
        made.append((km_before, l["name"], [l["km"]] + [p["km"] for p in news]))
    log(f"far pieces: {len(made)} register lines with pieces over {FAR_PIECE_KM} km apart made "
        f"{sum(len(m[2]) - 1 for m in made)} more lines")
    for km, name, kms in sorted(made, reverse=True):
        log(f"    {km:7.1f} km  {name} -> " + " + ".join(f"{k:.1f}" for k in kms))
    return out


# A section with neither end on the register is cut only if one end is at least this much deeper
# inside this country's outline than the other (the outline is a few hundred metres off).
SIDE_M = 300
# ... and only if one end is at most this far outside it and the other at most this far inside.
OUTLINE_SLACK_M = 1500


def split_at_borders(region, lines, stations, geoms, log):
    """A section built whole across a border, because this extract happens to hold the station
    on the far side as well, is cut at the border point on it: this country's part stays, and
    the far part stays too unless that country is built (dist/regions.json), whose own build
    then has it from the border point, under the same point id. Without this the two
    countries' pieces overlap and only this country's register is credited by a ride over it.
    A side is told by which end is a station of this country's register (merge has put the
    register's ids on them); with neither, by which end lies deeper inside this country's
    outline (borders.depth_m, at least SIDE_M apart); a section with both is left alone."""
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
    n_split = n_drop = n_osm = 0
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
            if len(hits) != 1 or (home_a and home_b):
                out.append(sec)
                continue
            if not home_a and not home_b:
                # NEITHER END ON THIS COUNTRY'S REGISTER: an OSM line over a crossing (Euskotren's
                # E2 Irun Ficoba - Hendaia, Varnsdorf's trains over the Czech border). One hit
                # puts the two ends on two sides, so the end further inside this country's
                # outline is home; the outline is coarse, so a near tie is left alone.
                sa, sb = stations.get(a), stations.get(b)
                if not sa or not sb or region not in hits[0][1]["countries"]:
                    out.append(sec)
                    continue
                da = borders.depth_m(region, sa["lon"], sa["lat"])
                db = borders.depth_m(region, sb["lon"], sb["lat"])
                # Both ends well outside this country is a section crossing twice with a point at
                # only one crossing (Ebersbach - Neugersdorf dips into Czechia, both ends in
                # Germany); both well inside, the same the other way. OUTLINE_SLACK_M allows for
                # the outline: it puts Irun Ficoba 1.1 km inside France.
                if da is None or db is None or abs(da - db) < SIDE_M \
                        or max(da, db) < -OUTLINE_SLACK_M or min(da, db) > OUTLINE_SLACK_M:
                    out.append(sec)
                    continue
                home_a, home_b = da > db, db > da
                n_osm += 1
                h, f = (sa, sb) if home_a else (sb, sa)
                log(f"border: {l['name']}: {h['name']} ({max(da, db):,.0f} m in) kept, "
                    f"{f['name']} ({min(da, db):,.0f} m) beyond {hits[0][1]['id']}")
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
    log(f"border: {n_split} sections built whole over a border cut at its border point "
        f"({n_osm} with neither end on the register, sided by the outline); the far "
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
    station_alias, line_alias, line_pieces = {}, {}, {}
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
            (lines, stations, geoms), canon_border_ids(mod.build(str(z), log), log), log,
            region=args.region)
        station_alias = getattr(merge_sources, "alias", {})
        line_alias = getattr(merge_sources, "line_alias", {})
        # A register's own line aliases (rinf.py's LINE_ALIAS, from a COUNTRY's `line_alias`:
        # line ids it no longer builds -> their successors), so rides saved on them still credit.
        line_alias.update(getattr(mod, "LINE_ALIAS", {}))
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
        timetable = gtfs_served.check(args.region, lines, stations, route_share, log, geoms)
        drop = drop_unridden_sections(lines, stations, geoms, route_share, log)
        lines = [l for l in lines if l["id"] not in drop]
        # A register line whose sections no longer all connect: one line per piece, where the
        # register module has `split_pieces` (us_register, ca_register; Anita, 2026-10-04: a
        # trip is entered station to station, so a line in pieces cannot be ridden across its
        # gap). It appends the pieces to `lines` and moves reg_ways and
        # register_way_lines.state["sec_ways"] to their ids; the biggest keeps the line's id.
        if hasattr(mod, "split_pieces"):
            mod.split_pieces(lines, stations, geoms, reg_ways, register_way_lines.state, log)
            line_pieces.update(getattr(mod, "LINE_PIECES", {}))
        # Out-and-back spurs at a line's middle junctions, after any bridges were drawn.
        fold_spurs(lines, stations, geoms, log, register_way_lines.state)
        # A register line's pieces far apart: one line each (Anita, 2026-10-05).
        for lid, ids in split_far_pieces(lines, stations, geoms, reg_ways,
                                         register_way_lines.state, log,
                                         set(getattr(mod, "KEEP_WHOLE", ()) or ())).items():
            line_pieces.setdefault(lid, []).extend(ids)
        for wid, lids in reg_ways.items():
            way_lines[wid] |= lids - drop
        # Register sections with no rails on the map: listed, but no longer running.
        import not_running
        not_running.mark(args.region, lines, geoms, log)
        gtfs_served.mark(timetable, lines, log)
        # Track that leads to no station: not passenger rail (Anita, 2026-10-07).
        dead = prune_dead_track(args.region, lines, stations, geoms, log)
        if dead:
            lines = [l for l in lines if l["id"] not in dead]
        # Its ways no longer name the line (ways.json, and so the tiles: a way no line runs
        # over is drawn as the faint rail with no passenger trains). An OSM section's ways by
        # their nodes; a register section's by the ways found beside it (sec_ways).
        st_ = getattr(register_way_lines, "state", None) or {}
        raw_ways, wids_, sec_ways_ = st_.get("ways", {}), st_.get("wids", []), st_.get("sec_ways", {})
        dn = getattr(prune_dead_track, "dead_nodes", {})
        unlink = defaultdict(set)              # way id -> line ids to take off it
        for lid, (dk, kk) in getattr(prune_dead_track, "dead_keys", {}).items():
            w_dead = {wids_[j] for k in dk for j, _got, _d in sec_ways_.get((lid, k), ())}
            w_keep = {wids_[j] for k in kk for j, _got, _d in sec_ways_.get((lid, k), ())}
            for w in w_dead - w_keep:
                unlink[w].add(lid)
        n_unlinked = 0
        for users in (way_lines, route_users):
            for wid, lids in users.items():
                lids -= dead
                if not lids:
                    continue
                gone_here = unlink.get(wid, set()) & lids
                nodes = raw_ways.get(wid, (None, ()))[1]
                for lid in [x for x in lids if dn.get(x)]:
                    if len(nodes) and sum(1 for n in nodes if int(n) in dn[lid]) >= 0.5 * len(nodes):
                        gone_here.add(lid)
                if gone_here:
                    lids -= gone_here
                    n_unlinked += len(gone_here) if users is way_lines else 0
        log(f"track leading to no station: {n_unlinked} way-line links dropped")
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
                                 built_regions=built,
                                 variants=getattr(build, "variant_dirs", None))
    ownership.write(out, args.region, foot)
    if (out / "credits.json").exists():
        (out / "credits.json").unlink()
    log(f"foot.json: {(out / 'foot.json').stat().st_size / 1e6:.2f} MB")
    # Sections running alongside one another, for the app's crediting (along.py).
    import along
    along.write(out, args.region, along.compute(args.region, lines, geoms, foot, log))
    log(f"along.json: {(out / 'along.json').stat().st_size / 1e6:.2f} MB")

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
        # `pieces`: a line split into pieces -> the ids of the pieces split off it, so a ride
        # between two stations of another piece moves there (the app's migrateRide).
        json.dump({"region": args.region, "stations": station_alias, "lines": line_alias,
                   **({"pieces": line_pieces} if line_pieces else {})},
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
