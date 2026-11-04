"""Fetch NYC subway and Staten Island Railway services from OpenStreetMap route relations.

Writes data/services.json for index.html: one entry per service (the 5, the A, the S ...) with
its bullet colour and shape, and each of its route relations (direction or branch) as an ordered
list of track ways [way id, reversed] in travel order, plus the stops in order.

Each direction is routed stop to stop through the full track network (raw/osm_tracks.json from
fetch_osm.py), keeping to the relation's own ways where they go, since OSM member order has gaps.
A run is [way id, f0, f1]: the stretch from fraction f0 to f1 of that way's length.

Overpass replies are cached in raw/osm_routes_*.json; `--refetch` downloads them again. Stop
positions come from raw/osm_stations.json where present.

    python fetch_routes.py [--refetch] [--out other.json]

Other rail (LIRR, Metro-North, NJ Transit, Amtrak, PATH ...) is routed for right-hand running: on
a line of two or more tracks each direction keeps to its own, and a relation for both ways
("<=>") gets its other direction as a variant of its own ("paired": true on both).
"""
import collections
import heapq
import json
import math
import pathlib
import re
import sys
import time

import requests

ROOT = pathlib.Path(__file__).parent
RAW = ROOT / "raw"
OUT = ROOT / "data" / "services.json"
URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
UA = "trackdepth/0.1 (personal map of NYC subway depth)"
# the map's square (fetch_osm.SQUARE): south, west, north, east
BBOX = "40.508,-74.26,40.975,-73.645"

ROUTES = f"""(rel["route"="subway"]({BBOX}); rel["route"="train"]["name"~"Staten Island"]({BBOX});
  rel["route"="train"]["network"~"Staten Island"]({BBOX});)"""

Q_RELS = f"""[out:json][timeout:300];
{ROUTES}->.r;
.r out body;
rel(br.r)["type"="route_master"];
out body;
"""
Q_WAYS = f"""[out:json][timeout:300];
{ROUTES}->.r;
way(r.r);
out skel;
"""
Q_STOPS = f"""[out:json][timeout:300];
{ROUTES}->.r;
node(r.r);
out body;
"""
# Other rail: PATH (tagged route=subway, picked up above), LIRR, Metro-North, Amtrak, NJ Transit,
# AirTrain JFK. Their relations run far outside the city, so ways and stops are cut to the bbox.
RAIL = f"""(rel["route"~"^(train|light_rail|monorail)$"]({BBOX});)"""
Q_RAIL_RELS = f"""[out:json][timeout:300];
{RAIL}->.r;
.r out body;
rel(br.r)["type"="route_master"];
out body;
"""
Q_RAIL_WAYS = f"""[out:json][timeout:300];
{RAIL}->.r;
way(r.r)({BBOX});
out skel;
"""
Q_RAIL_STOPS = f"""[out:json][timeout:300];
{RAIL}->.r;
node(r.r)({BBOX});
out body;
"""

# Official MTA trunk colours (MTA graphics standards; GTFS routes.txt route_color agrees except
# where noted). OSM colour tags are uneven, so these win.
COLOURS = {
    "1": "#EE352E", "2": "#EE352E", "3": "#EE352E",
    "4": "#00933C", "5": "#00933C", "6": "#00933C", "6X": "#00933C",
    "7": "#B933AD", "7X": "#B933AD",
    "A": "#0039A6", "C": "#0039A6", "E": "#0039A6",
    "B": "#FF6319", "D": "#FF6319", "F": "#FF6319", "FX": "#FF6319", "M": "#FF6319",
    "G": "#6CBE45", "J": "#996633", "Z": "#996633", "L": "#A7A9AC",
    "N": "#FCCC0A", "Q": "#FCCC0A", "R": "#FCCC0A", "W": "#FCCC0A",
    "GS": "#808183", "FS": "#808183", "H": "#808183", "S": "#808183",
    "SIR": "#0078C6",
}
ORDER = ["1", "2", "3", "4", "5", "6", "6X", "7", "7X", "A", "C", "E", "B", "D", "F", "FX", "M",
         "G", "J", "Z", "L", "N", "Q", "R", "W", "GS", "FS", "H", "S", "SIR"]
SHUTTLE_NAMES = {"GS": "42 St Shuttle", "FS": "Franklin Av Shuttle", "H": "Rockaway Park Shuttle"}
# Shuttles run a fixed number of trains back and forth (trains.js reads "shuttle": n off the
# service), not the headway-sized fleet of a line: one train per track on the 42 St Shuttle's two,
# one on the Franklin Av Shuttle, one per direction (per track) on the Rockaway Park Shuttle.
SHUTTLE_TRAINS = {"GS": 2, "FS": 1, "H": 2}

# Other rail: one service per PATH line, LIRR branch and Metro-North line (their official
# colours); Amtrak, NJ Transit and AirTrain JFK one service each, their routes as variants.
OTHER_COLOURS = {
    "PATH NWK–WTC": "#EF3941", "PATH JSQ–33": "#FDB827", "PATH HOB–33": "#0082C6", "PATH HOB–WTC": "#009E58",
    "PATH JSQ–33 via HOB": "#FDB827",
    "LIRR Babylon": "#00985F", "LIRR Far Rockaway": "#6E3219", "LIRR Hempstead": "#CE8E00",
    "LIRR Long Beach": "#FF6319", "LIRR Montauk": "#00B2A9", "LIRR Oyster Bay": "#00AF3F",
    "LIRR Port Jefferson": "#006EC7", "LIRR Port Washington": "#C60C30", "LIRR Ronkonkoma": "#A626AA",
    "LIRR West Hempstead": "#00A1DE",
    "MNR Harlem": "#0039A6", "MNR Hudson": "#009B3A", "MNR New Haven": "#EE0034",
    "Amtrak": "#1D5A8C", "NJT": "#A2238E", "AirTrain": "#6B7C93", "AirTrain Newark": "#6B7C93",
}
SYSTEM_ORDER = ["PATH", "LIRR", "Metro-North", "Amtrak", "NJ Transit", "AirTrain"]
# Other rail is routed for right-hand running: a segment of track on the wrong side for the
# direction of travel costs this many times its length (leaving the relation's own ways costs 6),
# so a direction keeps to its own track wherever the line has two (see Network.compute_sides).
RH_WRONG = 8.0
# Routing keeps to paths a train can drive: no turn sharper than this between one segment of track
# and the next (switches diverge by well under it; a reversal or a diamond crossing is far over).
MAX_TURN = 50.0
YARD_K = 4.0   # see Network.yard_k

# NJ Transit rail lines: key -> (words in the route's ref or name). Colours are NJ Transit's own
# line colours as OSM has them (route `colour`), these where OSM has none.
NJT_LINES = [("NEC", ("NEC", "Northeast Corridor")), ("NJCL", ("NJCL", "North Jersey Coast")),
             ("RVL", ("RVL", "Raritan Valley")), ("M&E", ("M&E", "Morris & Essex", "Morris and Essex", "Gladstone", "Morristown")),
             ("MOBO", ("MOBO", "Montclair")), ("Bergen", ("BCL", "Bergen County")),
             ("Main", ("Main Line", "MBPJ", "Port Jervis", "Main/Bergen")), ("PVL", ("PVL", "Pascack Valley")),
             ("Meadowlands", ("Meadowlands",)), ("ACRL", ("ACRL", "Atlantic City"))]
NJT_LABEL = {"NEC": "Northeast Corridor", "NJCL": "North Jersey Coast", "RVL": "Raritan Valley",
             "M&E": "Morris & Essex", "MOBO": "Montclair-Boonton", "Main": "Main", "Bergen": "Bergen County",
             "PVL": "Pascack Valley", "Meadowlands": "Meadowlands", "ACRL": "Atlantic City"}
NJT_NAME = {k: f"NJ Transit {v} Line" for k, v in NJT_LABEL.items()}
NJT_COLOURS = {"NEC": "#EF3E42", "NJCL": "#00A4E4", "RVL": "#FAA634", "M&E": "#00A94F", "MOBO": "#A3238E",
               "Main": "#FFCF01", "Bergen": "#BBBDBF", "PVL": "#8E258D", "Meadowlands": "#77787B", "ACRL": "#005DAA"}


def njt_line(tags):
    t = " ".join(tags.get(k, "") for k in ("ref", "name"))
    for key, words in NJT_LINES:
        if any(w in t for w in words):
            return key
    return None


def other_service(tags):
    """(key, system, label, name) for a PATH / LIRR / Metro-North / Amtrak / NJ Transit / AirTrain JFK
    route relation, else None (light rail, AirTrain Newark, the zoo monorail ...)."""
    netw, ref, route = tags.get("network", ""), tags.get("ref", "").strip(), tags.get("route")
    if netw == "PATH" and ref:
        ref = ref.replace(" VIA ", " via ")
        return f"PATH {ref}", "PATH", ref.replace("JSQ–33 via HOB", "JSQ–HOB–33"), f"PATH {ref}"
    if netw == "LIRR" and ref:
        return f"LIRR {ref}", "LIRR", ref, f"LIRR {ref} Branch"
    if netw.startswith("Metro-North") and ref:
        return f"MNR {ref}", "Metro-North", ref, f"Metro-North {ref} Line"
    if netw == "Amtrak":
        return "Amtrak", "Amtrak", "Amtrak", "Amtrak"
    if netw in ("NJ Transit", "NJ Transit Rail", "New Jersey Transit") and route == "train":
        # one service per line (since 2026-11-02, when the data reached the edge of the square)
        line = njt_line(tags)
        if line:
            return f"NJT {line}", "NJ Transit", NJT_LABEL.get(line, line), NJT_NAME.get(line, line)
        return None
    if netw == "AirTrain JFK":
        return "AirTrain", "AirTrain", "AirTrain", "AirTrain JFK"
    if "AirTrain Newark" in tags.get("name", "") + netw:
        return "AirTrain Newark", "AirTrain", "Newark", "AirTrain Newark"
    return None


def fetch(name, q, refetch):
    path = RAW / f"osm_routes_{name}.json"
    if path.exists() and not refetch:
        return json.loads(path.read_bytes())
    last = None
    for attempt, url in enumerate(URLS * 3):
        if attempt >= len(URLS):
            time.sleep(30 * attempt)
        try:
            r = requests.post(url, data={"data": q}, headers={"User-Agent": UA}, timeout=600)
        except requests.RequestException as e:
            print(f"{url}: {e}")
            last = e
            continue
        if r.ok:
            path.write_bytes(r.content)
            js = json.loads(r.content)
            print(f"{path.name}: {len(js['elements'])} elements, {len(r.content) / 1e6:.1f} MB")
            return js
        print(f"{url}: {r.status_code} {r.text[:200]!r}")
        last = r
    raise SystemExit(f"Overpass failed for {name}: {last}")


def service_ref(tags):
    """The service a route relation belongs to: '5', 'A', '6X', 'GS', 'SIR'."""
    ref = (tags.get("ref") or "").strip().upper()
    name = tags.get("name", "")
    if "staten island" in (name + tags.get("network", "")).lower() and ref in ("", "SIR", "SI"):
        return "SIR"
    ref = ref.replace("<", "").replace(">", "X") if ref.startswith("<") else ref
    if ref in ("S", "GS", "FS", "H", "SF", "SR"):
        low = name.lower()
        if "42" in low or "grand central" in low or "times" in low:
            return "GS"
        if "franklin" in low:
            return "FS"
        if "rockaway" in low:
            return "H"
        return {"SF": "FS", "SR": "H"}.get(ref, "S")
    return ref


class Network:
    """Every OSM rail way as a graph on its vertices (keyed by coordinate: ways that share a node
    share its coordinate exactly), for routing a service through its stops."""

    def __init__(self, elements):
        self.geom, self.cum, self.name, self.kind, self.service = {}, {}, {}, {}, {}
        self.adj = collections.defaultdict(list)  # vertex -> [(vertex, metres, way, i, j)]
        for e in elements:
            if e["type"] != "way" or e.get("tags", {}).get("railway") not in ("subway", "rail", "light_rail", "monorail"):
                continue
            g = [(round(p["lon"], 7), round(p["lat"], 7)) for p in e["geometry"]]
            cum = [0.0]
            for i in range(1, len(g)):
                cum.append(cum[-1] + metres(g[i - 1], g[i]))
            self.geom[e["id"]], self.cum[e["id"]] = g, cum
            self.name[e["id"]] = e.get("tags", {}).get("name", "")
            self.kind[e["id"]] = e["tags"]["railway"]
            self.service[e["id"]] = e["tags"].get("service", "")
            for i in range(1, len(g)):
                m = cum[i] - cum[i - 1]
                self.adj[g[i - 1]].append((g[i], m, e["id"], i - 1, i))
                self.adj[g[i]].append((g[i - 1], m, e["id"], i, i - 1))
        self.side = {}
        # yard, siding and spur tracks outside the relation cost YARD_K times more again: else a
        # yard alongside the line (Hillside, east of Jamaica) could undercut its wrong-side track
        self.yard_k = {w: YARD_K if self.service[w] in ("yard", "siding", "spur") else 1.0 for w in self.geom}
        self.fallbacks = []   # legs routed with a reversal or without the turn limit (printed per relation)

    def compute_sides(self, reach=20.0):
        """Right-hand running. For every segment of a running way (no yard, siding or spur), how
        many parallel running tracks of the same kind (rail, subway ...) lie within `reach` m on
        its right minus on its left, looking along the way: side[w][k] for the segment from vertex
        k to k+1. A train belongs on a segment whose score, seen in its direction of travel, is not
        positive: on two tracks the right-hand one (score -1 against +1), on four the two on the
        right; a middle track of three, or a single track, scores 0 both ways."""
        P = lambda p: (p[0] * 84300.0, p[1] * 111000.0)
        segs = []   # (way, k, x0, y0, x1, y1)
        for w, g in self.geom.items():
            if self.service[w] in ("yard", "siding", "spur"):
                continue
            for k in range(len(g) - 1):
                (x0, y0), (x1, y1) = P(g[k]), P(g[k + 1])
                if x0 != x1 or y0 != y1:
                    segs.append((w, k, x0, y0, x1, y1))
        C = 25.0
        grid = collections.defaultdict(list)
        for n, (w, k, x0, y0, x1, y1) in enumerate(segs):
            ln = math.hypot(x1 - x0, y1 - y0)
            for t in range(int(ln // 10) + 2):
                f = min(1.0, t * 10 / ln) if ln else 0
                cell = (int((x0 + (x1 - x0) * f) // C), int((y0 + (y1 - y0) * f) // C))
                if not grid[cell] or grid[cell][-1] != n:
                    grid[cell].append(n)
        for w in self.geom:
            self.side[w] = [0] * max(0, len(self.geom[w]) - 1)
        for n, (w, k, x0, y0, x1, y1) in enumerate(segs):
            ln = math.hypot(x1 - x0, y1 - y0)
            ux, uy = (x1 - x0) / ln, (y1 - y0) / ln
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            cx, cy = int(mx // C), int(my // C)
            left, right = set(), set()
            for gx in (cx - 1, cx, cx + 1):
                for gy in (cy - 1, cy, cy + 1):
                    for o in grid.get((gx, gy), ()):
                        w2, k2, a0, b0, a1, b1 = segs[o]
                        if w2 == w or self.kind[w2] != self.kind[w]:
                            continue
                        l2 = math.hypot(a1 - a0, b1 - b0)
                        vx, vy = (a1 - a0) / l2, (b1 - b0) / l2
                        if abs(ux * vx + uy * vy) < 0.94:
                            continue
                        # nearest point of the other segment to our midpoint
                        f = max(0.0, min(1.0, ((mx - a0) * vx + (my - b0) * vy) / l2))
                        qx, qy = a0 + (a1 - a0) * f, b0 + (b1 - b0) * f
                        if math.hypot(qx - mx, qy - my) > reach:
                            continue
                        cross = ux * (qy - my) - uy * (qx - mx)   # > 0: on the left
                        if abs(cross) < 1.0:
                            continue
                        (left if cross > 0 else right).add(w2)
            self.side[w][k] = len(right) - len(left)

    def wrong_side(self, w, i, j):
        """Travelling from vertex i to j of way w (neighbours) on the left-hand track."""
        s = self.side.get(w)
        if not s:
            return False
        sc = s[min(i, j)]
        return sc > 0 if j > i else sc < 0

    def route(self, a, b, members, off_route=6.0, wrong=1.0, cap=40000, came_from=None):
        """Cheapest path a -> b that a train can drive: no turn sharper than MAX_TURN degrees from
        one segment to the next. Without that, the cheapest path doubled back at stub ends and
        terminals (a train reversing inside Penn Station or Jamaica) and took 90-degree turns at
        diamond crossings, and the cars zigzagged across the tracks. Falls back to the unlimited
        path when no drivable one exists (a stop snapped onto a stub). Ways outside the relation
        cost `off_route` times their length, and with `wrong` > 1 a segment against right-hand
        running costs that many times more. `came_from`: the vertex the train arrived at `a` from
        (the previous leg), so it carries on the way it was going; if it can't, it may reverse at
        `a` (a terminal). Returns [(way, i, j), ...] edges, or None."""
        e = None
        if came_from is not None:
            e = self.route_drivable(a, b, members, off_route, wrong, cap, came_from)
        if e is None:
            e = self.route_drivable(a, b, members, off_route, wrong, cap, None)
            if e is not None and came_from is not None:
                self.fallbacks.append(("reversed at", a, b))
        if e is None:
            self.fallbacks.append(("not drivable", a, b))
            e = self.route_any(a, b, members, off_route, wrong, cap)
        return e

    def route_drivable(self, a, b, members, off_route, wrong, cap, came_from):
        if a == b:
            return []
        cos_max = math.cos(math.radians(MAX_TURN))
        def vec(u, v):
            return ((v[0] - u[0]) * 84300.0, (v[1] - u[1]) * 111000.0)
        # states are (vertex, the vertex before it); None before the start
        start = (a, came_from)
        dist, prev, heap = {start: 0.0}, {}, [(0.0, 0, start)]
        tick, goal = 0, None
        while heap:
            d, _, st = heapq.heappop(heap)
            u, p = st
            if u == b:
                goal = st
                break
            if d > dist.get(st, 1e18) or d > cap:
                continue
            vin = vec(p, u) if p is not None else None
            lin = math.hypot(*vin) if vin else 0
            for v, m, w, i, j in self.adj[u]:
                if v == p:
                    continue
                if vin and lin > 0:
                    vo = vec(u, v)
                    lo = math.hypot(*vo)
                    if lo > 0 and (vin[0] * vo[0] + vin[1] * vo[1]) / (lin * lo) < cos_max:
                        continue
                nd = d + m * (1.0 if w in members else off_route * self.yard_k[w]) * (wrong if wrong > 1 and self.wrong_side(w, i, j) else 1.0)
                ns = (v, u)
                if nd < dist.get(ns, 1e18):
                    dist[ns], prev[ns] = nd, (st, w, i, j)
                    tick += 1
                    heapq.heappush(heap, (nd, tick, ns))
        if goal is None:
            return None
        edges, st = [], goal
        while st != start:
            pst, w, i, j = prev[st]
            edges.append((w, i, j))
            st = pst
        return edges[::-1]

    def route_multi(self, a, came_from, targets, members, off_route, wrong, cap, slack=3.0):
        """Drivable paths (turn limit as route_drivable) from `a`, arriving the way it came from
        `came_from` (None: any way), to each vertex of `targets` it reaches: {target: (cost,
        edges, the vertex before target)}. Stops once every target is reached, or past `slack`
        times the cost of the first one found (plus 2 km), or past `cap`."""
        cos_max = math.cos(math.radians(MAX_TURN))
        def vec(u, v):
            return ((v[0] - u[0]) * 84300.0, (v[1] - u[1]) * 111000.0)
        start = (a, came_from)
        dist, prev, heap = {start: 0.0}, {}, [(0.0, 0, start)]
        tick, found, limit = 0, {}, cap
        while heap:
            d, _, st = heapq.heappop(heap)
            if d > limit:
                break
            if d > dist.get(st, 1e18):
                continue
            u, p = st
            if u in targets and u != a and u not in found:
                edges, s = [], st
                while s != start:
                    ps, w, i, j = prev[s]
                    edges.append((w, i, j))
                    s = ps
                found[u] = (d, edges[::-1], p)
                if len(found) == 1:
                    limit = min(cap, slack * d + 2000)
                if len(found) == len(targets - {a}):
                    break
            vin = vec(p, u) if p is not None else None
            lin = math.hypot(*vin) if vin else 0
            for v, m, w, i, j in self.adj[u]:
                if v == p:
                    continue
                if vin and lin > 0:
                    vo = vec(u, v)
                    lo = math.hypot(*vo)
                    if lo > 0 and (vin[0] * vo[0] + vin[1] * vo[1]) / (lin * lo) < cos_max:
                        continue
                nd = d + m * (1.0 if w in members else off_route * self.yard_k[w]) * (wrong if wrong > 1 and self.wrong_side(w, i, j) else 1.0)
                ns = (v, u)
                if nd < dist.get(ns, 1e18):
                    dist[ns], prev[ns] = nd, (st, w, i, j)
                    tick += 1
                    heapq.heappush(heap, (nd, tick, ns))
        return found

    def route_any(self, a, b, members, off_route=6.0, wrong=1.0, cap=40000):
        """route() without the turn limit."""
        dist, prev, heap = {a: 0.0}, {}, [(0.0, a)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == b:
                break
            if d > dist.get(u, 1e18) or d > cap:
                continue
            for v, m, w, i, j in self.adj[u]:
                nd = d + m * (1.0 if w in members else off_route * self.yard_k[w]) * (wrong if wrong > 1 and self.wrong_side(w, i, j) else 1.0)
                if nd < dist.get(v, 1e18):
                    dist[v], prev[v] = nd, (u, w, i, j)
                    heapq.heappush(heap, (nd, v))
        if b not in prev and a != b:
            return None
        edges, v = [], b
        while v != a:
            u, w, i, j = prev[v]
            edges.append((w, i, j))
            v = u
        return edges[::-1]

    def snap(self, stop, toward, kinds, members, reach=450.0):
        """The best vertex of snap_all, or None."""
        c = self.snap_all(stop, toward, kinds, members, reach)
        return c[0] if c else None

    def snap_all(self, stop, toward, kinds, members, reach=450.0, n=10, yards=False):
        """Candidate vertices to start or stop a right-hand run at a stop, best first, at most one
        per way and `n` in all: the vertices of running ways of `kinds` within `reach` m, a vertex
        whose track is on the wrong side for travel toward `toward` (a point further along)
        counting 400 m further away, a non-member 50 m. (OSM puts few vertices on straight track,
        so the right-hand track's nearest vertex can be a few hundred metres off; a run ending on
        the wrong track stays on it for kilometres.) Yard, siding and spur ways count only when the
        relation lists them (or `yards`, for a terminal station): a terminal's platform tracks are
        often tagged service=yard in OSM (Port Washington, Long Beach), and a stop on one used to
        snap to the running track 600 m back.)"""
        if not hasattr(self, "vgrid"):
            self.vgrid = collections.defaultdict(list)
            for w, g in self.geom.items():
                for i, p in enumerate(g):
                    self.vgrid[(int(p[0] * 84300 // 50), int(p[1] * 111000 // 50))].append((w, i))
        sx, sy = int(stop[0] * 84300 // 50), int(stop[1] * 111000 // 50)
        tx, ty = (toward[0] - stop[0]) * 84300, (toward[1] - stop[1]) * 111000
        best = {}   # way -> (score, vertex)
        nr = int(reach // 50) + 1
        for gx in range(sx - nr, sx + nr + 1):
            for gy in range(sy - nr, sy + nr + 1):
                for w, i in self.vgrid.get((gx, gy), ()):
                    if self.kind[w] not in kinds or (self.service[w] in ("yard", "siding", "spur") and w not in members and not yards):
                        continue
                    g = self.geom[w]
                    p = g[i]
                    d = metres(p, stop)
                    if d > reach:
                        continue
                    # the segment at this vertex, and which way along it the train goes
                    ka, kb = (i, i + 1) if i + 1 < len(g) else (i - 1, i)
                    dx, dy = (g[kb][0] - g[ka][0]) * 84300, (g[kb][1] - g[ka][1]) * 111000
                    fwd = (dx * tx + dy * ty) > 0
                    bad = self.wrong_side(w, ka, kb) if fwd else self.wrong_side(w, kb, ka)
                    score = d + (400 if bad else 0) + (0 if w in members else 50)
                    if w not in best or score < best[w][0]:
                        best[w] = (score, p)
        # at most 3 per line name, so a branch running alongside with many short ways (the
        # Montauk Branch beside the Atlantic Branch at Rosedale and Laurelton) doesn't crowd out
        # the line the stop is really on
        out, per = [], collections.Counter()
        for (score, p), w in sorted((v, w) for w, v in best.items()):
            if per[self.name[w]] >= 3:
                continue
            per[self.name[w]] += 1
            out.append(p)
            if len(out) >= n:
                break
        return out

    def runs(self, edges):
        """Edges -> [[way, f0, f1], ...]: stretches along one way, as fractions of its length
        (f0 > f1 means travelling against the way's direction)."""
        out = []
        for w, i, j in edges:
            if out and out[-1][0] == w and out[-1][2] == i:
                out[-1][2] = j
            else:
                out.append([w, i, j])
        res = []
        for w, i, j in out:
            tot = self.cum[w][-1] or 1.0
            res.append([w, round(self.cum[w][i] / tot, 4), round(self.cum[w][j] / tot, 4)])
        return res


def metres(a, b):
    return math.hypot((a[0] - b[0]) * 84300, (a[1] - b[1]) * 111000)


def chain(ways, nodes_of):
    """Orient member ways in relation order by shared end nodes. Returns [[id, rev], ...] and
    the number of breaks (consecutive ways that share no end node)."""
    ways = [w for w in ways if nodes_of.get(w)]
    out, breaks = [], 0
    for k, w in enumerate(ways):
        a, b = nodes_of[w][0], nodes_of[w][-1]
        if out:
            pn, prev_rev = nodes_of[out[-1][0]], out[-1][1]
            tail = pn[0] if prev_rev else pn[-1]
            if tail == a:
                out.append([w, False])
                continue
            if tail == b:
                out.append([w, True])
                continue
            breaks += 1
        # first way, or a way after a break: orient it toward the next way
        rev = False
        if k + 1 < len(ways):
            nn = nodes_of[ways[k + 1]]
            if a in (nn[0], nn[-1]) and b not in (nn[0], nn[-1]):
                rev = True
        out.append([w, rev])
    return out, breaks


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    refetch = "--refetch" in sys.argv
    out_path = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else OUT
    RAW.mkdir(exist_ok=True)
    rels = fetch("rels", Q_RELS, refetch)["elements"] + fetch("rail_rels", Q_RAIL_RELS, refetch)["elements"]
    ways = fetch("ways", Q_WAYS, refetch)["elements"] + fetch("rail_ways", Q_RAIL_WAYS, refetch)["elements"]
    nodes_of = {e["id"]: e["nodes"] for e in ways if e["type"] == "way"}
    # stop positions: fetch_osm.py's station download has them; query only what it lacks
    node = {}
    st = RAW / "osm_stations.json"
    if st.exists():
        node = {e["id"]: e for e in json.loads(st.read_bytes())["elements"] if e["type"] == "node"}
    wanted = {m["ref"] for r in rels if r["type"] == "relation" for m in r.get("members", [])
              if m["type"] == "node" and m.get("role", "").startswith("stop")}
    node.update({e["id"]: e for e in fetch("rail_stops", Q_RAIL_STOPS, refetch)["elements"] if e["type"] == "node"})
    if wanted - set(node):
        print(f"{len(wanted - set(node))} of {len(wanted)} stop nodes not in hand; querying the subway's")
        stops = fetch("stops", Q_STOPS, refetch)["elements"]
        node.update({e["id"]: e for e in stops if e["type"] == "node"})

    net = Network(json.loads((RAW / "osm_tracks.json").read_bytes())["elements"])
    t0 = time.time()
    net.compute_sides()
    sides = [v for s in net.side.values() for v in s]
    print(f"right-hand running: {len(sides)} segments, {sum(1 for v in sides if v)} beside parallel track "
          f"({time.time() - t0:.0f} s)")
    track_ids = set()
    tj = ROOT / "data" / "tracks.json"
    if tj.exists():
        tracks = json.loads(tj.read_bytes())["tracks"]
        track_ids = {t["id"] for t in tracks}
        # index.html slices tracks.json ways by fraction of length, so they must run the OSM way
        flipped = sum(1 for t in tracks if t["id"] in net.geom and
                      metres((t["lon"][0], t["lat"][0]), net.geom[t["id"]][-1]) <
                      metres((t["lon"][0], t["lat"][0]), net.geom[t["id"]][0]))
        print(f"tracks.json ways running against their OSM way: {flipped}")

    routes = [e for e in rels if e["type"] == "relation" and e["tags"].get("type") == "route"]
    masters = [e for e in rels if e["type"] == "relation" and e["tags"].get("type") == "route_master"]
    master_name = {}
    for m in masters:
        for mem in m["members"]:
            if mem["type"] == "relation":
                master_name[mem["ref"]] = m["tags"].get("name", "")

    # stations of the commuter railways, for naming and adding terminal stops (see below)
    RAILWAYS = {"LIRR": ("LIRR", "Long Island"), "Metro-North": ("Metro-North",), "NJ Transit": ("NJ Transit",), "PATH": ("PATH",)}
    stations_by = {sysname: [(n["lon"], n["lat"], n["tags"]["name"]) for n in node.values()
                             if n.get("tags", {}).get("railway") in ("station", "halt") and n["tags"].get("name")
                             and any(k in n["tags"].get("network", "") + ";" + n["tags"].get("operator", "") for k in keys)]
                   for sysname, keys in RAILWAYS.items()}
    services = {}
    for r in routes:
        tags = r["tags"]
        if tags.get("disused") == "yes" or tags.get("state") == "disused":
            continue
        other = other_service(tags)
        ref = other[0] if other else service_ref(tags)
        if not other and (ref not in ORDER or tags.get("route") != "subway" and ref != "SIR"):
            continue
        members = [m["ref"] for m in r["members"] if m["type"] == "way" and m["ref"] in net.geom
                   and not re.search(r"platform", m.get("role", ""))]
        member_set = set(members)
        stop_list = []
        for m in r["members"]:
            if m["type"] == "node" and re.match(r"stop", m.get("role", "")) and m["ref"] in node:
                n = node[m["ref"]]
                stop_list.append([round(n["lon"], 6), round(n["lat"], 6), n.get("tags", {}).get("name", "")])
        # The member order is unreliable (gaps of up to 4 km where ways are missing or out of
        # order), so route through the stops in order instead, keeping to member ways where they
        # go. Subway: each stop snaps to the nearest vertex of a member way. Other rail: right-hand
        # running (see Network.compute_sides): a stop snaps to a vertex on the track of its
        # direction, and the router keeps to the right-hand track, leaving the relation's own ways
        # for it where needed (an LIRR or NJ Transit relation often lists one track for both ways).
        verts = [v for w in members for v in net.geom[w]]
        order, how, bad = None, "stops", 0
        rh = RH_WRONG if other else 1.0
        kinds = {net.kind[w] for w in members}
        # the points to run through, in order (lon, lat)
        via = [(s[0], s[1]) for s in stop_list]
        term_pts = set()   # terminal stations the route is taken on to (see below)
        term_st = stations_by.get(other[1], []) if other else []   # stations of this service's railway
        ch = chain(members, nodes_of)[0] if other else []
        if ch:
            w0, r0 = ch[0]; w1, r1 = ch[-1]
            a_ = net.geom[w0][-1] if r0 else net.geom[w0][0]
            b_ = net.geom[w1][0] if r1 else net.geom[w1][-1]
            if len(via) >= 2:
                # A commuter route's relation can list only some stops (NJ Transit's NEC lists the
                # express stops from Secaucus on, not New York Penn): run it from the ends of its
                # member ways inside the square too, so it reaches its terminal and the square's edge.
                # The chain's first and last ways need not be its ends (members out of order,
                # gaps): take, past each end stop, the member way end farthest out on that side.
                # Only ends near the square's edge (where the line leaves it) count: past a
                # terminal inside the square there are only its yard tracks (Far Rockaway).
                SQ_ = (-74.258, 40.510, -73.647, 40.973)
                near_edge = lambda p: min((p[0] - SQ_[0]) * 84300, (SQ_[2] - p[0]) * 84300, (p[1] - SQ_[1]) * 111000, (SQ_[3] - p[1]) * 111000) < 1500
                if metres(a_, via[0]) > 800 and metres(a_, via[-1]) > metres(a_, via[0]):
                    via.insert(0, a_)
                if metres(b_, via[-1]) > 800 and metres(b_, via[0]) > metres(b_, via[-1]):
                    via.append(b_)
                ends = [p for w in members if net.service[w] not in ("yard", "siding", "spur")
                        for p in (net.geom[w][0], net.geom[w][-1]) if near_edge(p)]
                v0, v1 = via[0], via[-1]
                side0 = [p for p in ends if metres(p, v0) < metres(p, v1) and metres(p, v1) > metres(v0, v1) + 800]
                side1 = [p for p in ends if metres(p, v1) < metres(p, v0) and metres(p, v0) > metres(v0, v1) + 800]
                if side0:
                    via.insert(0, max(side0, key=lambda p: metres(p, v1)))
                if side1:
                    via.append(max(side1, key=lambda p: metres(p, v0)))
                # A route that ends 120-700 m short of a station of its railway (not at the square's
                # edge, no stop of its own near the station) goes on to the station: the Hempstead
                # Branch's ways stop at the Atlantic Terminal throat, 380 m short of the platforms.
                # The station may stand on yard-tagged track (snap_all's `yards`).
                for end in (0, -1) if other[1] in ("LIRR", "Metro-North") else ():
                    p, nxt = via[end], via[1] if end == 0 else via[-2]
                    s_ = min((s for s in term_st if 120 < metres(p, s) < 700 and metres(s, nxt) > metres(p, nxt)
                              and all(metres(s, q) > 300 for q in via)), key=lambda s: metres(p, s), default=None)
                    if s_ and not near_edge(p):
                        pt = (s_[0], s_[1])
                        term_pts.add(pt)
                        via.insert(0, pt) if end == 0 else via.append(pt)
            else:
                # no stops: from one end of the member chain to the other, or for a chain that
                # comes back to where it started (an out-and-back), out to its farthest point and back
                cv = [v for w, _ in ch for v in net.geom[w]]
                far = max(cv, key=lambda v: metres(v, a_))
                via = [a_, far, a_] if metres(a_, b_) < 500 else [a_, b_]
                if "<=>" in tags.get("name", ""):
                    # a relation for both ways lists its ways in no useful order (Port Jefferson's
                    # chain ran 2 km): offer every member way's ends, and the farthest-apart pair
                    # is picked below
                    via = [p for w in members for p in (net.geom[w][0], net.geom[w][-1])]

        both = bool(other) and "<=>" in tags.get("name", "")
        if both and len(via) >= 2:
            # A relation for both ways lists its stops in no useful order (Metro-North's Hudson
            # Line zigzags between the two tracks' stop positions; NJ Transit's go out and back):
            # run between its two farthest-apart points, and order the stops along the way.
            i0, j0 = max(((i, j) for i in range(len(via)) for j in range(i + 1, len(via))),
                         key=lambda ij: metres(via[ij[0]], via[ij[1]]))
            A, B = via[i0], via[j0]
            via = [A, B]
            seen = []
            for st in sorted(stop_list, key=lambda st: metres((st[0], st[1]), A)):
                if all(metres((st[0], st[1]), (q[0], q[1])) > 150 for q in seen):
                    seen.append(st)
            stop_list = seen

        def lay(points):
            """Runs through `points` in order, and how many legs could not be routed.

            Each point has a few candidate vertices (snap_all: the right-hand track first). A leg
            takes the first pair of candidates (start, end) with a path a train can drive from
            the way it arrived; only if none has one does it fall back to net.route's reversal
            or unlimited path. (The best-scoring vertex can sit on a track that connects to the
            line only through a crossover the train would have to reverse over: Valley Stream on
            the Far Rockaway Branch, the Main Line at Jamaica.)"""
            cands, cpts = [], []
            for k, s in enumerate(points):
                c = []
                if rh > 1:
                    if k + 1 < len(points):
                        toward = points[k + 1]
                    else:
                        toward = (2 * s[0] - points[k - 1][0], 2 * s[1] - points[k - 1][1])
                    # (a stop beyond the edge of the square has no track near it: look further, so
                    # the run still ends on the right-hand track at the edge)
                    c = net.snap_all(s, toward, kinds, member_set, yards=s in term_pts) or net.snap_all(s, toward, kinds, member_set, reach=4000)
                if not c and verts:
                    c = [min(verts, key=lambda v: metres(v, s))]
                if c:
                    cands.append(c)
                    cpts.append(s)
            # The two ends of the run keep only the candidates near the best one: a far candidate
            # makes a shorter run, and the search takes it (Port Washington's trains stopped 2 km
            # short, at a vertex of the line further back, to save the run to the terminal).
            if rh > 1:
                for k in {0, len(cands) - 1}:
                    if len(cands[k]) > 1:
                        d0 = metres(cands[k][0], cpts[k])
                        cands[k] = [c for c in cands[k] if metres(c, cpts[k]) <= d0 + 150.0]
            cap = 1e6 if rh > 1 else 40000

            # The best drivable run over every choice of candidates at once (a Viterbi search): a
            # layer per point, holding for each candidate vertex the cheapest run reaching it, its
            # cost (route cost plus 100 per rank down the candidate list) and the vertex before it.
            # Choosing stop by stop went wrong where two branches run side by side (the Montauk and
            # Atlantic Branches from Valley Stream to Laurelton): a Far Rockaway train set off on
            # the Montauk tracks and had to reverse to reach Locust Manor.
            rank = lambda k, c: 100.0 * cands[k].index(c)
            layer = {c: (rank(0, c), None, [], None) for c in cands[0]}   # c -> (cost, from, edges, came)
            back, nbad = [layer], 0
            for k in range(1, len(cands)):
                targets = set(cands[k])
                new = {}
                for keep_going in (True, False):
                    for s0, (cost, _, _, came) in layer.items():
                        if not keep_going and came is None:
                            continue   # already tried without a direction
                        legcap = min(cap, 60 * min(metres(s0, t) for t in targets) + 20000)
                        res = net.route_multi(s0, came if keep_going else None, targets, member_set, 6.0, rh, legcap)
                        for t, (c, e, before) in res.items():
                            tot = cost + c + rank(k, t) + (0 if keep_going or came is None else 5000)
                            if t not in new or tot < new[t][0]:
                                new[t] = (tot, s0, e, before)
                    if new:
                        if not keep_going:
                            net.fallbacks.append(("reversed at", cands[k - 1][0], cands[k][0]))
                        break
                if not new:
                    # nothing drivable: the old way, from the cheapest so far to the best candidate
                    s0 = min(layer, key=lambda c: layer[c][0])
                    t = cands[k][0]
                    net.fallbacks.append(("not drivable", s0, t))
                    e = net.route_any(s0, t, member_set, 6.0, rh, cap) if s0 != t else []
                    if e is None:
                        nbad += 1
                        e = []
                    new = {t: (layer[s0][0], s0, e, None)}
                layer = new
                back.append(layer)
            # follow the back-pointers from the cheapest end
            t = min(layer, key=lambda c: layer[c][0])
            chain_edges = []
            for k in range(len(back) - 1, 0, -1):
                _, s0, e, _ = back[k][t]
                chain_edges.append(e)
                t = s0
            return net.runs([e for leg in reversed(chain_edges) for e in leg]), nbad

        if len(via) >= 2 and verts:
            order, bad = lay(via)
            if len(stop_list) < 2:
                how = "chain ends"
        if not order:
            how = "member order"
            order = [[w, 1.0, 0.0] if rev else [w, 0.0, 1.0] for w, rev in ch or chain(members, nodes_of)[0]]
        # Terminal stops. A relation can leave its terminal out of its stop list (the Port Washington
        # Branch's ends at Manhasset, 2 km short of the terminal its ways run to) or list it as a
        # nameless stop node (Long Beach): a station of the railway at either end of the route that
        # no stop is near is added as the first or last stop, and a nameless stop takes the name of the
        # station beside it. Without that the trains don't stop there and the dot has no route.
        if other and order and stop_list and how != "member order":
            near_st = lambda p: min((s for s in term_st if metres(p, s) < 150), key=lambda s: metres(p, s), default=None)
            for st in stop_list:
                if not st[2]:
                    s_ = near_st((st[0], st[1]))
                    if s_:
                        st[2] = s_[2]
            for end in (0, -1):
                w, f0, f1 = order[end]
                g, cum_ = net.geom[w], net.cum[w]
                f = f0 if end == 0 else f1
                i_ = min(range(len(g)), key=lambda i: abs(cum_[i] / (cum_[-1] or 1.0) - f))
                s_ = near_st(g[i_])
                if s_ and all(metres(s_, (q[0], q[1])) > 300 for q in stop_list):
                    stop_list.insert(0, [round(s_[0], 6), round(s_[1], 6), s_[2]]) if end == 0 else stop_list.append([round(s_[0], 6), round(s_[1], 6), s_[2]])
        # A relation for both ways ("<=>": LIRR, Metro-North, NJ Transit) gets its other direction
        # as a variant of its own, routed back on the right-hand track; an out-and-back already
        # has both.
        back = None
        if both and len(via) >= 2 and metres(via[0], via[-1]) >= 500:
            border, bbad = lay(via[::-1])
            if border:
                back = {"osm": r["id"], "name": tags.get("name", ""), "from": tags.get("to", ""), "to": tags.get("from", ""),
                        "ways": border, "stops": stop_list[::-1], "paired": True}
        length = sum(abs(f1 - f0) * net.cum[w][-1] for w, f0, f1 in order)
        off = sum(abs(f1 - f0) * net.cum[w][-1] for w, f0, f1 in order if w not in member_set)
        variant = {
            "osm": r["id"],
            "name": tags.get("name", ""),
            "from": tags.get("from", ""),
            "to": tags.get("to", ""),
            "ways": order,
            "stops": stop_list,
        }
        if back:
            # A both-ways relation was run from whichever end of it came first; give each of its two
            # variants the right from and to (the city terminal at the city end).
            city = lambda n: any(k in n for k in ("New York", "Penn", "Grand Central", "Hoboken", "Atlantic Terminal", "Newark", "Secaucus"))
            inbound = metres(via[0], (-73.985, 40.752)) > metres(via[-1], (-73.985, 40.752))
            f_, t_ = variant["from"], variant["to"]
            if (inbound and city(f_) and not city(t_)) or (not inbound and city(t_) and not city(f_)):
                variant["from"], variant["to"] = t_, f_
                back["from"], back["to"] = f_, t_
        missing = sum(1 for w, *_ in order if w not in track_ids) if track_ids else 0
        print(f"{ref:4} {r['id']:>9} {len(order):4} runs {length / 1000:5.1f} km, {off / 1000:4.1f} km off members,"
              f" {bad} unrouted, {missing:2} not in tracks.json, by {how}  {tags.get('name', '')}")
        for kind, a_, b_ in net.fallbacks:
            print(f"      {kind}: ({a_[0]:.5f}, {a_[1]:.5f}) -> ({b_[0]:.5f}, {b_[1]:.5f})")
        net.fallbacks.clear()
        if other:
            # long-distance routes reach far outside the data: keep a variant only if most of it
            # is on tracks.json ways
            inside = sum(abs(f1 - f0) * net.cum[w][-1] for w, f0, f1 in order if w in track_ids)
            if length < 300 or inside < 0.6 * length:
                continue
        s = services.setdefault(ref, {"ref": ref, "variants": [], "master": master_name.get(r["id"], ""),
                                      "osm_colour": tags.get("colour", ""), "other": other})
        def same_tracks(v, ways):
            # nearly the same track, the same way round, by length (OSM has five Far Rockaway
            # relations that differ by a stop or a crossover, two Ronkonkoma ones for both ways,
            # and Amtrak's many trains through Penn)
            A = {(w, f1 > f0): abs(f1 - f0) * net.cum[w][-1] for w, f0, f1 in v["ways"]}
            B = {(w, f1 > f0): abs(f1 - f0) * net.cum[w][-1] for w, f0, f1 in ways}
            both_ = sum(min(A[k], B[k]) for k in A.keys() & B.keys())
            return both_ >= 0.9 * max(sum(A.values()), sum(B.values()), 1)
        if any(v["osm"] == r["id"] or other and same_tracks(v, order) for v in s["variants"]):
            continue  # same tracks as a variant already kept
        s["variants"].append(variant)
        if back and any(same_tracks(v, back["ways"]) for v in s["variants"]):
            back = None   # the other way is there already (from another relation)
            variant["paired"] = True
        if back:
            variant["paired"] = True   # trains.js: don't also run this one backwards on its own tracks
            s["variants"].append(back)
            print(f"{'':4} {'':>9} {len(back['ways']):4} runs, the other way")

    # LIRR to Grand Central Madison (since 2023) has no OSM route relation yet: route it from
    # Jamaica through the East Side Access tunnel ("Grand Central Branch"), preferring the Main Line
    # and Grand Central Branch ways, both directions.
    gcb = {w for w, n in net.name.items() if n in ("Grand Central Branch", "Main Line", "LIRR Main Line")}
    if gcb:
        verts = [v for w in gcb for v in net.geom[w]]
        jam = min(verts, key=lambda v: metres(v, (-73.8085, 40.6996)))
        gcm = min(verts, key=lambda v: metres(v, (-73.9768, 40.7527)))
        for a, b, frm, to in ((jam, gcm, "Jamaica", "Grand Central Madison"), (gcm, jam, "Grand Central Madison", "Jamaica")):
            # right-hand running, each way on its own track
            a = net.snap(a, b, {"rail"}, gcb) or a
            b = net.snap(b, (2 * b[0] - a[0], 2 * b[1] - a[1]), {"rail"}, gcb) or b
            e = net.route(a, b, gcb, off_route=4.0, wrong=RH_WRONG, cap=1e6)
            if not e:
                print("LIRR Grand Central: no route", frm, "->", to)
                continue
            order = net.runs(e)
            s = services.setdefault("LIRR Grand Central", {"ref": "LIRR Grand Central", "variants": [], "master": "",
                                                           "osm_colour": "", "other": ("LIRR Grand Central", "LIRR",
                                                           "Grand Central", "LIRR to Grand Central Madison")})
            ends = {"Jamaica": [round(a_[0], 6), round(a_[1], 6), "Jamaica"] for a_ in [jam]}
            ends["Grand Central Madison"] = [round(gcm[0], 6), round(gcm[1], 6), "Grand Central Madison"]
            s["variants"].append({"osm": None, "name": f"LIRR {frm} => {to}", "from": frm, "to": to, "ways": order,
                                  "stops": [ends[frm], ends[to]]})
            km = sum(abs(f1 - f0) * net.cum[w][-1] for w, f0, f1 in order) / 1000
            print(f"LIRR Grand Central {frm} -> {to}: {len(order)} runs {km:.1f} km "
                  f"{sum(1 for w, *_ in order if w not in track_ids)} not in tracks.json")
    OTHER_COLOURS.setdefault("LIRR Grand Central", "#4D5357")

    # Relations with a gap: OSM's Montauk, Oyster Bay and Ronkonkoma relations list no ways between
    # Jamaica and the square's edge (they carry on outside it), so those trains stopped at Jamaica.
    # A variant whose outer end is over 1 km inside the square while its relation has 5 or more
    # ways beyond the square is carried on to the edge along another variant of the same railway that passes
    # its end going the same way: preferably one whose onward ways are named for this line's
    # branch ("Montauk" -> the Montauk Branch tracks the Babylon trains use), else the Main Line,
    # else the longest.
    SQ = (-74.258, 40.510, -73.647, 40.973)   # index.html's LIMIT
    MID = (-73.985, 40.752)
    def pos(w, f):
        c, g = net.cum[w], net.geom[w]
        d = f * c[-1]
        i = next((k for k in range(1, len(c)) if c[k] >= d), len(c) - 1)
        return g[i]
    def edge_m(p):
        return min((p[0] - SQ[0]) * 84300, (SQ[2] - p[0]) * 84300, (p[1] - SQ[1]) * 111000, (SQ[3] - p[1]) * 111000)
    beyond = {r["id"]: sum(1 for mm in r["members"] if mm["type"] == "way" and mm["ref"] not in net.geom)
              for r in routes}
    by_system = collections.defaultdict(list)
    for s in services.values():
        if s["other"]:
            by_system[s["other"][1]].extend(s["variants"])
    for s in services.values():
        if not s["other"] or s["other"][1] not in ("LIRR", "Metro-North", "NJ Transit"):
            continue
        m_ = re.search(r"(?:LIRR|Metro-North|NJ Transit)\s+([A-Z][\w&-]+)", s["other"][3])
        word = m_.group(1) if m_ else ""
        for v in s["variants"]:
            # (a handful of ways past the edge is a terminal's yard or sidings, not a line going on:
            # Far Rockaway, Port Washington, Meadowlands)
            if not v["ways"] or beyond.get(v["osm"], 0) < 5:
                continue
            a, b = pos(v["ways"][0][0], v["ways"][0][1]), pos(v["ways"][-1][0], v["ways"][-1][2])
            outbound = metres(b, MID) > metres(a, MID)
            end = b if outbound else a
            if edge_m(end) < 1000:
                continue
            best = None
            for u in by_system[s["other"][1]]:
                if u is v or not u["ways"]:
                    continue
                pts = [pos(w, f0) for w, f0, f1 in u["ways"]] + [pos(u["ways"][-1][0], u["ways"][-1][2])]
                k = min(range(len(pts)), key=lambda i: metres(pts[i], end))
                if metres(pts[k], end) > 250:
                    continue
                part = u["ways"][k:] if outbound else u["ways"][:k]
                if not part:
                    continue
                tip = pts[-1] if outbound else pts[0]
                if edge_m(tip) > 1000 or metres(tip, MID) <= metres(end, MID):
                    continue   # doesn't reach the edge, or goes the other way
                names = [net.name.get(w, "") for w, f0, f1 in part]
                # this line's own branch; else the Main Line, counting against it any other branch
                # the continuation turns onto (the Hempstead Branch at Floral Park)
                other_branch = sum(1 for n in names if ("Branch" in n or " Line" in n) and "Main Line" not in n and not (word and word in n))
                score = (sum(1 for n in names if word and word in n),
                         sum(1 for n in names if "Main Line" in n) - 5 * other_branch,
                         sum(abs(f1 - f0) * net.cum[w][-1] for w, f0, f1 in part))
                if best is None or score > best[0]:
                    best = (score, part)
            if best:
                v["ways"] = v["ways"] + best[1] if outbound else best[1] + v["ways"]
                km = sum(abs(f1 - f0) * net.cum[w][-1] for w, f0, f1 in best[1]) / 1000
                print(f"{s['ref']}: {v['name'][:50]} carried on {km:.1f} km to the edge of the square")
            else:
                print(f"{s['ref']}: {v['name'][:50]} stops {edge_m(end) / 1000:.1f} km inside the square; no other route to carry it on")

    # Two relations of one rail service for the same trip (Ronkonkoma's two "<=>" relations, two
    # Hempstead ones eastbound) become one: variants whose ends are within 1 km of each other's
    # and which share at least half their track the same way round, or run along the same
    # corridor (70% of the vertices within 60 m: parallel tracks). The one with more track inside
    # the square stays. Real branches end apart and are kept.
    def end_points(v):
        def at(w, f):
            c, g = net.cum[w], net.geom[w]
            d = f * c[-1]
            i = next((k for k in range(1, len(c)) if c[k] >= d), len(c) - 1)
            return g[i]
        return at(v["ways"][0][0], v["ways"][0][1]), at(v["ways"][-1][0], v["ways"][-1][2])
    def shared(v1, v2):
        A = {(w, f1 > f0): abs(f1 - f0) * net.cum[w][-1] for w, f0, f1 in v1["ways"]}
        B = {(w, f1 > f0): abs(f1 - f0) * net.cum[w][-1] for w, f0, f1 in v2["ways"]}
        return sum(min(A[k], B[k]) for k in A.keys() & B.keys()) / max(1.0, min(sum(A.values()), sum(B.values())))
    def corridor(v1, v2):
        # share of v2 vertices within 60 m of v1 ones (the same corridor on a parallel track:
        # Ronkonkoma's two westbound relations take the Main Line's two westbound tracks)
        grid = collections.defaultdict(list)
        for w, f0, f1 in v1["ways"]:
            for p in net.geom[w]:
                grid[(int(p[0] * 84300 // 50), int(p[1] * 111000 // 50))].append(p)
        pts = [p for w, f0, f1 in v2["ways"] for p in net.geom[w]]
        near = 0
        for p in pts:
            gx, gy = int(p[0] * 84300 // 50), int(p[1] * 111000 // 50)
            if any(metres(p, q) < 60 for dx in (-1, 0, 1) for dy in (-1, 0, 1) for q in grid.get((gx + dx, gy + dy), ())):
                near += 1
        return near / max(1, len(pts))
    def inside_km(v):
        return sum(abs(f1 - f0) * net.cum[w][-1] for w, f0, f1 in v["ways"] if w in track_ids)
    for ref, s in services.items():
        if not s["other"]:
            continue
        vs = [v for v in s["variants"] if v["ways"]]
        keep = []
        for v in sorted(vs, key=inside_km, reverse=True):
            a, b = end_points(v)
            twin = next((u for u in keep if metres(end_points(u)[0], a) < 1000 and metres(end_points(u)[1], b) < 1000
                         and (shared(u, v) >= 0.5 or corridor(u, v) >= 0.7)), None)
            if twin is None:
                keep.append(v)
            else:
                twin["paired"] = twin.get("paired") or v.get("paired")
                print(f"{ref}: {v['name'][:60]} ({v['from']} -> {v['to']}) merged into the longer one")
        s["variants"] = [v for v in s["variants"] if v in keep]

    def sort_key(k):
        o = services[k]["other"]
        if not o:
            return (0, ORDER.index(k), k)
        return (1, SYSTEM_ORDER.index(o[1]), k)

    out = []
    for ref in sorted(services, key=sort_key):
        s = services[ref]
        if s["other"]:
            key, system, label, name = s["other"]
            vs = s["variants"]
            for v in vs:
                m = re.search(r"\(([^)]*)\)", v["name"])
                v["when"] = m.group(1) if m else ""
                v["km"] = round(sum(abs(f1 - f0) * net.cum[w][-1] for w, f0, f1 in v["ways"]) / 1000, 2)
            vs.sort(key=lambda v: (v["when"] != "", -v["km"]))
            colour = OTHER_COLOURS.get(key, "#808183")
            if key.startswith("NJT "):
                oc = s["osm_colour"] if re.match(r"^#[0-9A-Fa-f]{6}$", s["osm_colour"] or "") else ""
                if oc.upper() in ("#FFFFFF", "#000000"):   # OSM's Meadowlands Line is white
                    oc = ""
                colour = (oc or NJT_COLOURS.get(key[4:], "#A2238E")).upper()
            out.append({"ref": ref, "label": label, "name": name, "colour": colour, "shape": "tag", "system": system,
                        "text": "#000000" if colour in ("#FDB827",) else "#FFFFFF", "variants": vs})
            continue
        # The default (first) variant is the longest full-time one, then the same route the other
        # way, then the rest: part-time ones ("(late nights)" in the name) last.
        vs = s["variants"]
        for v in vs:
            m = re.search(r"\(([^)]*)\)", v["name"])
            v["when"] = m.group(1) if m else ""
            v["km"] = round(sum(abs(f1 - f0) * net.cum[w][-1] for w, f0, f1 in v["ways"]) / 1000, 2)
        vs.sort(key=lambda v: (v["when"] != "", -v["km"]))
        back = [k for k, v in enumerate(vs) if k > 0 and v["from"] == vs[0]["to"] and v["to"] == vs[0]["from"]]
        if back:
            vs.insert(1, vs.pop(back[0]))
        label = {"GS": "S", "FS": "S", "H": "S", "6X": "6", "7X": "7", "FX": "F"}.get(ref, ref)
        shape = "diamond" if ref.endswith("X") else "pill" if ref == "SIR" else "circle"
        colour = COLOURS.get(ref) or s["osm_colour"] or "#808183"
        name = SHUTTLE_NAMES.get(ref) or ("Staten Island Railway" if ref == "SIR" else s["master"] or
                                          s["variants"][0]["name"])
        out.append({"ref": ref, "label": label, "name": name, "colour": colour, "shape": shape,
                    "text": "#000000" if colour == "#FCCC0A" else "#FFFFFF", "variants": s["variants"],
                    **({"shuttle": SHUTTLE_TRAINS[ref]} if ref in SHUTTLE_TRAINS else {})})
    out_path.write_text(json.dumps({"services": out}, separators=(",", ":")), encoding="utf-8")
    print(f"{out_path}: {len(out)} services, {sum(len(s['variants']) for s in out)} variants, "
          f"{out_path.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
