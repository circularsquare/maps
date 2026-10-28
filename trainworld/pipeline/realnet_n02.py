"""T-737, T-1625: realnet's `n02` adapter, a whole country's rail network from MLIT's N02 (Japan).

N02 has every line's track (RailroadSection) and its stations (a short piece of that track each),
by operator and line name, and no services. Per (operator, line) the track is a graph; stations and
junctions are its key points; routes are the longest station paths in turn (a tree's diameter,
then what is left), a route ending at a junction carried on to the nearest station.

Services come from japanriders' passing counts (通過人員, people a day past a section, both ways),
which are per section. A route is cut where the count changes (`_split`: pieces of at least 4 km
whose counts, each a length-weighted median, differ about twofold between neighbours), and each
piece is its own line with its own count D: cars from sqrt(D / 1000), and whole trains per level
so that the cars passing carry the riders of one hour of that level at `LOAD` riders a car
(`_service`). A Shinkansen route runs high-speed trains on high-speed track. A line with no count
takes its kind's median count (one piece). Run times from track length at a speed by kind, dwell
included. Rough on purpose: a stress-test network, not a calibrated one. Each line carries its
track shape (`shapes`), which `realnet.py geom` follows. A route leaving the area kept (`inside`)
runs as its pieces inside. `src["plan"]` gives realnet the cars, whole trains, fast flag and count
of each route.
"""
import json
import math
import os
from collections import defaultdict

import numpy as np

R_EARTH = 6371008.8

# N02_002: 1 Shinkansen, 2 JR conventional, 3 public (city subways and trams), 4 private, 5 third sector.
# N02_001: 11 JR ordinary, 12 other ordinary, 13 funicular, 14-17 monorail, guideway and trolleybus,
# 21 tramway, 22-26 tramway-law monorail, guideway and maglev.
SKIP_RAIL = {"13", "17"}
# average speed between stops with the dwell, km/h: what the track model's run times come to on
# this network (realsave's "track model" against these, T-1625), so that the trains a count needs
# are worked out over the round trip the game will have
KMH = {"1": 156, "2": 65, "3": 37, "4": 57, "5": 61, "tram": 26, "mono": 36}
CARS_MAX = {"1": 16, "2": 15, "3": 10, "4": 10, "5": 6, "tram": 2, "mono": 8}
TPH_MAX = 24.0  # trains an hour each way a line's track takes at most
TURN_S = 180.0  # the game's turnaround at the ends of a line

# Share of a weekday's riders in one hour of each demand level (high 6-20, medium 20-24, low 0-6):
# the game's own split of a weekday by period (T-1616's dump of the all-Japan save: 83.0%, 15.0%
# and 2.0% of passenger-km over 14, 4 and 6 hours).
LEVEL_SHARE = (0.0593, 0.0375, 0.0033)
# Riders a car carries an hour each way at a level's mean hour, a third of the 160 a car holds at
# crush, so that peak hours (about 1.4 times the mean) and the busier direction still fit.
LOAD = 56.0
# A piece is at least this long (m); a split costs this much (km x log2(count)^2), about the gain of
# a twofold step between two halves of 12 km.
MIN_PIECE_M = 4000.0
SPLIT_COST = 6.0
SEG_MATCH_M = 400.0  # a count's section must pass this near a hop
# The least service is a train and a car all through the high level, which carries a thin line's
# count at few riders a car-km; a piece whose count comes to less than this on it joins its
# neighbour, or is left out if alone (the save must keep 10 a car-km in the game, whose riders
# differ from the counts: raise this where a run finds lines under 10).
MIN_PER_CAR_KM = 12.0
MIN_ROUTE_M = 500.0  # a route shorter than this is two points of one station
# What the game runs of this network found (notes/T-1625.md): lines whose riders came to under 15 a
# car-km have their cars and trains cut to carry them at 25 (`cars`, `trains`), or are left out
# (null) where even a train and a car leaves them under 12. Keyed by `_pieces`' line key.
TRIM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "n02_trim.json")
TRIM = json.load(open(TRIM_PATH, encoding="utf-8")) if os.path.exists(TRIM_PATH) else {}


def _class(p):
    t, o = p["N02_001"], p["N02_002"]
    if t == "21":
        return "tram"
    if t in ("14", "15", "16", "22", "23", "24", "25", "26"):
        return "mono"
    return o


def _kind(c):
    return {"1": "commuter", "2": "commuter", "3": "metro", "4": "commuter", "5": "commuter", "tram": "tram", "mono": "monorail"}[c]


def _xy(lon, lat, o):
    return (R_EARTH * np.radians(lon - o[0]) * np.cos(np.radians(o[1])), R_EARTH * np.radians(lat - o[1]))


def _counts(path, o):
    """(operator, line) -> [(people a day, the section's geometry in plane metres)], and
    (operator, line) -> the median of its sections' people a day."""
    if not path:
        return {}, {}
    from shapely import transform
    from shapely.geometry import shape
    d = json.load(open(path, encoding="utf-8"))
    segs, by = defaultdict(list), defaultdict(list)
    for f in d["features"]:
        p = f["properties"]
        v = p.get("density") or p.get("d_base")
        if not v:
            continue
        g = transform(shape(f["geometry"]), lambda c: np.stack(_xy(c[:, 0], c[:, 1], o), 1))
        segs[(p["op"], p["line"])].append((float(v), g))
        by[(p["op"], p["line"])].append(float(v))
    return segs, {k: float(np.median(v)) for k, v in by.items()}


def _hop_counts(segs, X, Y, verts, vi):
    """Each hop's (stop to next stop) count, from the section nearest most of its track, or None."""
    from shapely.geometry import Point
    out = []
    for h in range(len(vi) - 1):
        a, b = vi[h], vi[h + 1]
        if not segs:
            out.append(None)
            continue
        pts = [verts[a + int(round((b - a) * f))] for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
        votes = defaultdict(float)
        for v in pts:
            pt = Point(X[v], Y[v])
            ds = [(g.distance(pt), i) for i, (_, g) in enumerate(segs) if not g.is_empty]
            d, i = min(x for x in ds if x[0] == x[0]) if any(x[0] == x[0] for x in ds) else (math.inf, 0)
            if d <= SEG_MATCH_M:
                votes[i] += 1.0 - d / (SEG_MATCH_M * 4)
        out.append(segs[max(votes, key=votes.get)][0] if votes else None)
    return out


def _wmedian(vals, weights):
    pairs = sorted((v, w) for v, w in zip(vals, weights) if v)
    if not pairs:
        return None
    half = sum(w for _, w in pairs) / 2
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= half:
            return v
    return pairs[-1][0]


def _split(lens, dens):
    """Hop ranges (i, j) cutting a route where its count changes: the partition into pieces of at
    least MIN_PIECE_M that minimises, over the pieces, the length-weighted spread of log2(count)
    about each piece's mean plus SPLIT_COST a piece (optimal partitioning). Hops with no count take
    their nearest known neighbour's."""
    n = len(lens)
    known = [i for i, d in enumerate(dens) if d]
    if n < 2 or len(known) < 2 or sum(lens) < 2 * MIN_PIECE_M:
        return [(0, n)]
    fill = [dens[min(known, key=lambda k: abs(k - i))] for i in range(n)]
    y = [math.log2(d) for d in fill]
    w = [m / 1000.0 for m in lens]
    W, S1, S2 = [0.0], [0.0], [0.0]
    for yi, wi in zip(y, w):
        W.append(W[-1] + wi)
        S1.append(S1[-1] + wi * yi)
        S2.append(S2[-1] + wi * yi * yi)
    best, back = [0.0] + [math.inf] * n, [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j):
            span = W[j] - W[i]
            if span * 1000 < MIN_PIECE_M or best[i] == math.inf:
                continue
            c = best[i] + SPLIT_COST + (S2[j] - S2[i]) - (S1[j] - S1[i]) ** 2 / span
            if c < best[j]:
                best[j], back[j] = c, i
    if best[n] == math.inf:
        return [(0, n)]
    cuts, j = [], n
    while j > 0:
        cuts.append((back[j], j))
        j = back[j]
    return cuts[::-1]


LEVEL_HOURS = (14.0, 4.0, 6.0)


def _service(cls, D, length_m):
    """Cars and whole trains (high, medium, low) for a piece whose count is D people a day, and
    the riders a car-km that count comes to on them."""
    if cls == "1":
        cars = 16 if D >= 60000 else 12 if D >= 25000 else 8
    else:
        cars = int(np.clip(round(math.sqrt(D / 1000.0)), 1, CARS_MAX[cls]))
    rt_h = (2 * length_m / (KMH[cls] / 3.6) + 2 * TURN_S) / 3600.0
    need = [s * D / 2.0 / LOAD / cars for s in LEVEL_SHARE]  # trains an hour each way
    if need[0] > TPH_MAX:
        need = [x * TPH_MAX / need[0] for x in need]
    n_h = max(1, int(need[0] * rt_h + 0.6))
    n_m = min(n_h, int(need[1] * rt_h + 0.5))
    if n_h >= 2:
        n_m = max(n_m, 1)
    n_l = min(n_m, int(need[2] * rt_h + 0.5))
    trains = [n_h, n_m, n_l]
    # a day's passenger-km is D x km; its car-km, trains x 2 km / round trip x cars x hours
    per_km = D / (2 * cars * sum(h * n for h, n in zip(LEVEL_HOURS, trains)) / rt_h)
    return {"cars": cars, "trains": trains, "tph": [round(n / rt_h, 3) for n in trains], "fast": cls == "1", "hsr": cls == "1", "rt_h": rt_h, "per_car_km": per_km}


def _pieces(cls, lens, dens, fallback):
    """The lines a run of hops (lengths `lens`, counts `dens`) makes: [(i, j, D)] hop ranges and
    their counts. Normally the run cut where the count changes (`_split`); a piece that cannot
    carry MIN_PER_CAR_KM even at the least service (a train and a car, high level only) joins its
    neighbour with the higher count, and a run left so thin is dropped. A Shinkansen run is not cut
    (through trains matter more than counts): it is one line at its median count, and a piece of it
    whose count is 30% over that adds a line of the difference."""
    n = len(lens)

    def count(i, j):
        return _wmedian(dens[i:j], lens[i:j]) or fallback

    def thin(i, j):
        D = count(i, j)
        return D is None or _service(cls, D, sum(lens[i:j]))["per_car_km"] < MIN_PER_CAR_KM
    if cls == "1":
        D0 = count(0, n)
        out = [] if thin(0, n) else [(0, n, D0)]
        for i, j in _split(lens, dens):
            Dp = count(i, j)
            if (i, j) != (0, n) and Dp and Dp > 1.3 * D0 and not thin(i, j):
                out.append((i, j, Dp - D0))
        return out
    cuts = _split(lens, dens)
    while len(cuts) > 1:
        bad = [k for k, (i, j) in enumerate(cuts) if thin(i, j)]
        if not bad:
            break
        k = bad[0]
        nb = [m for m in (k - 1, k + 1) if 0 <= m < len(cuts)]
        m = max(nb, key=lambda m: count(*cuts[m]) or 0)
        lo, hi = sorted((k, m))
        cuts[lo:hi + 1] = [(cuts[lo][0], cuts[hi][1])]
    return [] if thin(*cuts[0]) and len(cuts) == 1 else [(i, j, count(i, j)) for i, j in cuts]


def _routes(edges, key_nodes, station_of):
    """Station paths over a key graph: `edges` {(a, b): (length, [vertex ids])}."""
    import heapq
    adj = defaultdict(list)
    for (a, b), (w, _) in edges.items():
        adj[a].append((b, w, (a, b)))
        adj[b].append((a, w, (a, b)))
    unused = set(edges)

    def sp(src, only_unused):
        dist, prev = {src: 0.0}, {}
        h = [(0.0, src)]
        while h:
            d, u = heapq.heappop(h)
            if d > dist[u]:
                continue
            for v, w, e in adj[u]:
                if only_unused and e not in unused:
                    continue
                nd = d + w
                if nd < dist.get(v, np.inf):
                    dist[v], prev[v] = nd, (u, e)
                    heapq.heappush(h, (nd, v))
        return dist, prev

    def path(prev, a, b):
        out = []
        while b != a:
            u, e = prev[b]
            out.append((u, b, e))
            b = u
        return out[::-1]
    routes = []
    while unused:
        start = next(iter(unused))[0]
        d0, _ = sp(start, True)
        a = max(d0, key=d0.get)
        d1, p1 = sp(a, True)
        b = max(d1, key=d1.get)
        if a == b:
            break
        hops = path(p1, a, b)
        for _, _, e in hops:
            unused.discard(e)
        # a route ending off a station: a junction runs on to the nearest station not on it over
        # any track (a branch meeting its trunk); a dead end (or no such station) is cut back to
        # its last station
        for end in (0, 1):
            node = hops[0][0] if end == 0 else hops[-1][1]
            if node in station_of:
                continue
            on = {u for u, _, _ in hops} | {hops[-1][1]}
            dist, prev = sp(node, False)
            st = [n for n in dist if n in station_of and n not in on]
            ext = path(prev, node, min(st, key=dist.get)) if st and len(adj[node]) > 1 else None
            if ext and not ({v for _, v, _ in ext} & on):
                hops = [(v, u, e) for u, v, e in reversed(ext)] + hops if end == 0 else hops + ext
                continue
            if end == 0:
                while hops and hops[0][0] not in station_of:
                    hops = hops[1:]
            else:
                while hops and hops[-1][1] not in station_of:
                    hops = hops[:-1]
            if not hops:
                break
        if hops:
            routes.append(hops)
    return routes


def load(src, inside, osm_interval_tph):
    """realnet adapter: (stations {key: (name, lon, lat)}, trips, route names); also fills
    src["route_kind"], src["route_operator"], src["train_m"], src["shapes"] (route -> [[lon, lat]])
    and src["plan"] (route -> cars, whole trains, trains an hour, fast, hsr, count)."""
    import re
    st_f = json.load(open(src["stations"], encoding="utf-8"))["features"]
    sec_f = json.load(open(src["sections"], encoding="utf-8"))["features"]
    o = (139.7671, 35.6812)
    segs, dens = _counts(src.get("density"), o)
    skip = re.compile(src["skip_line"]) if src.get("skip_line") else None
    groups = defaultdict(lambda: {"secs": [], "sts": []})
    for f in sec_f:
        p = f["properties"]
        if p["N02_001"] in SKIP_RAIL:
            continue
        groups[(p["N02_004"], p["N02_003"])]["secs"].append(f["geometry"])
    for i, f in enumerate(st_f):
        p = f["properties"]
        if p["N02_001"] in SKIP_RAIL:
            continue
        g = groups.get((p["N02_004"], p["N02_003"]))
        if g is not None:
            g["sts"].append((i, p, f["geometry"]))
            g["cls"] = _class(p)
    # a line with no count takes the median count of the lines of its kind
    by_cls = defaultdict(list)
    for k, g in groups.items():
        if k in dens and "cls" in g:
            by_cls[g["cls"]].append(dens[k])
    cls_count = {c: float(np.median(v)) for c, v in by_cls.items()}
    stations, trips, names = {}, [], {}
    src.setdefault("route_kind", {})
    src.setdefault("route_operator", {})
    src.setdefault("train_m", {"_": 120})
    src.setdefault("shapes", {})
    plan = src.setdefault("plan", {})
    dropped = src.setdefault("dropped", [])  # stretches too thin to run: (line, from, to, km left out, count)
    trimmed_out = []
    n_routes = 0
    for (op, line), g in sorted(groups.items()):
        if len(g["sts"]) < 2 or (skip and skip.search(op + " " + line)):
            continue
        cls = g["cls"]
        # the track graph: vertices by rounded position
        vid, pos = {}, []

        def v_of(c):
            k = (round(c[0], 6), round(c[1], 6))
            if k not in vid:
                vid[k] = len(pos)
                pos.append(k)
            return vid[k]
        adj = defaultdict(set)

        def add_line(coords):
            ids = [v_of(c) for c in coords]
            for a, b in zip(ids, ids[1:]):
                if a != b:
                    adj[a].add(b)
                    adj[b].add(a)
        for geom in g["secs"]:
            for part in ([geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]):
                add_line(part)
        station_of = {}
        for i, p, geom in g["sts"]:
            cs = geom["coordinates"] if geom["type"] == "LineString" else geom["coordinates"][0]
            add_line(cs)
            mid = np.mean(np.array(cs), axis=0)
            best = min(cs, key=lambda c: (c[0] - mid[0]) ** 2 + (c[1] - mid[1]) ** 2)
            v = v_of(best)
            if v not in station_of:
                station_of[v] = (f"n02:{i}", p["N02_005"])
        name_of = {k: nm for k, nm in station_of.values()}
        P = np.array(pos)
        X, Y = _xy(P[:, 0], P[:, 1], o)

        def w(a, b):
            return float(np.hypot(X[a] - X[b], Y[a] - Y[b]))
        key = set(station_of) | {v for v in adj if len(adj[v]) != 2}
        edges = {}
        for a in key:
            for nb in adj[a]:
                chain, prev, cur, L = [a], a, nb, w(a, nb)
                while cur not in key:
                    chain.append(cur)
                    nxt = next(iter(adj[cur] - {prev}))
                    L += w(cur, nxt)
                    prev, cur = cur, nxt
                chain.append(cur)
                if a == cur:
                    continue
                k = (min(a, cur), max(a, cur))
                if k not in edges or edges[k][0] > L:
                    edges[k] = (L, chain if a < cur else chain[::-1])
        line_D = dens.get((op, line))
        line_segs = segs.get((op, line), [])
        kmh = KMH[cls]
        served = set()
        mine = []  # this line's routes, to name the pieces apart
        for hops in _routes(edges, key, station_of):
            verts = []
            for u, v, e in hops:
                ch = edges[e][1]
                ch = ch if ch[0] == u else ch[::-1]
                verts += ch if not verts else ch[1:]
            stops, at, vi = [], [], []
            acc = 0.0
            prev = verts[0]
            for i, vv in enumerate(verts):
                acc += w(prev, vv)
                prev = vv
                if vv in station_of and (not stops or stops[-1] != station_of[vv][0]):
                    stops.append(station_of[vv][0])
                    at.append(acc)
                    vi.append(i)
            # every stop already on a longer route of the line: a parallel piece of track
            if len(stops) < 2 or set(stops) <= served:
                continue
            served |= set(stops)
            hop_d = _hop_counts(line_segs, X, Y, verts, vi)
            # a route leaving the area kept (`inside`) runs as its pieces inside
            ok = [inside(float(P[verts[i], 0]), float(P[verts[i], 1])) for i in vi]
            runs, cur = [], []
            for k, good in enumerate(ok):
                if good:
                    cur.append(k)
                elif cur:
                    runs.append(cur)
                    cur = []
            if cur:
                runs.append(cur)
            for run in runs:
                if len(run) < 2:
                    continue
                a, b = run[0], run[-1]
                if at[b] - at[a] < MIN_ROUTE_M:
                    continue
                lens = [at[k + 1] - at[k] for k in range(a, b)]
                fallback = line_D or cls_count.get(cls) or 3000.0
                made = _pieces(cls, lens, hop_d[a:b], fallback)
                if cls != "1":
                    left_out = sum(lens) - sum(sum(lens[i:j]) for i, j, _ in made)
                    if left_out > 1:
                        dropped.append((f"{op} {line}", name_of[stops[a]], name_of[stops[b]], round(left_out / 1000, 1), round(_wmedian(hop_d[a:b], lens) or fallback)))
                for i, j, D in made:
                    s0, s1 = a + i, a + j
                    sv = _service(cls, D, at[s1] - at[s0])
                    sv["d"] = round(D)
                    sv["key"] = f"{op}|{line}|{stops[s0]}|{stops[s1]}"
                    # a game run found this line thin: its service cut, or the line left out
                    if sv["key"] in TRIM:
                        t = TRIM[sv["key"]]
                        if t is None:
                            trimmed_out.append(sv["key"])
                            continue
                        sv.update(cars=t["cars"], trains=t["trains"], tph=[round(n / sv["rt_h"], 3) for n in t["trains"]])
                    n_routes += 1
                    route = f"{op}|{line}|{n_routes}"
                    mine.append((route, name_of[stops[s0]], name_of[stops[s1]]))
                    names[route] = line
                    p_verts = verts[vi[s0]:vi[s1] + 1]
                    p_stops = stops[s0:s1 + 1]
                    p_at = [x - at[s0] for x in at[s0:s1 + 1]]
                    for vv in p_verts:
                        if vv in station_of:
                            k, nm = station_of[vv]
                            stations[k] = (nm, float(P[vv, 0]), float(P[vv, 1]))
                    src["shapes"][route] = [[float(P[vv, 0]), float(P[vv, 1])] for vv in p_verts]
                    src["route_kind"][route] = _kind(cls)
                    src["route_operator"][route] = op
                    src["train_m"][route] = sv["cars"] * 20
                    plan[route] = sv
                    # one trip an hour each way: enough for realnet to build the line and its run times
                    t_at = np.array(p_at) / (kmh / 3.6)
                    for ks, tt in ((p_stops, t_at), (p_stops[::-1], (t_at[-1] - t_at)[::-1])):
                        dr = "0" if ks is p_stops else "1"
                        for h in range(24):
                            trips.append((route, dr, ks, [int(h * 3600 + 1800 + v) for v in tt]))
        # a line with several routes names each by its ends
        if len(mine) > 1:
            for route, a_nm, b_nm in mine:
                names[route] = f"{line} {a_nm}-{b_nm}"
    keep = {k for k, (_, lon, lat) in stations.items() if inside(lon, lat)}
    pl = [p for r, p in plan.items()]
    print(f"{src['name']}: {len(groups)} N02 lines, {n_routes} routes, {len(trips)} trips made up, {len(keep)} of {len(stations)} stations inside, "
          f"{sum(1 for k in groups if k in dens)} lines with a passing count, {sum(1 for p in pl if p['fast'])} high-speed routes, "
          f"{len(dropped)} stretches ({sum(d[3] for d in dropped):.0f} km) too thin to run, {len(trimmed_out)} lines left out and "
          f"{sum(1 for p in pl if p['key'] in TRIM and TRIM[p['key']])} cut by the game runs' trim")
    trips = [t for t in trips if all(k in keep for k in t[2])]
    return {k: v for k, v in stations.items() if k in keep}, trips, names

