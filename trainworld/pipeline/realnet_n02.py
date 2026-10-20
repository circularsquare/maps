"""T-737: realnet's `n02` adapter, a whole country's rail network from MLIT's N02 (Japan).

N02 has every line's track (RailroadSection) and its stations (a short piece of that track each),
by operator and line name, and no services. Per (operator, line) the track is a graph; stations and
junctions are its key points; routes are the longest station paths in turn (a tree's diameter,
then what is left), a route ending at a junction carried on to the nearest station. Trains per
hour from japanriders' passing counts (通過人員, people a day past a section, the line's median):
peak each way = 6% of the day over the train's seats, between 0.5 and 24; midday 55% of peak, the
first hour 30%. Lines without a count take a default by kind of operator. Run times from track
length at a speed by kind, dwell included. Rough on purpose: a stress-test network, not a
calibrated one. Each line carries its track shape (`shapes`), which `realnet.py geom` follows. A
route leaving the area kept (`inside`) runs as its pieces inside.
"""
import json
from collections import defaultdict

import numpy as np

R_EARTH = 6371008.8

# N02_002: 1 Shinkansen, 2 JR conventional, 3 public (city subways and trams), 4 private, 5 third sector.
# N02_001: 11 JR ordinary, 12 other ordinary, 13 funicular, 14-17 monorail, guideway and trolleybus,
# 21 tramway, 22-26 tramway-law monorail, guideway and maglev.
SKIP_RAIL = {"13", "17"}
KMH = {"1": 160, "2": 48, "3": 32, "4": 42, "5": 45, "tram": 16, "mono": 30}
DEFAULT = {"1": ((3, 2, 1), 16), "2": ((2, 1, 0.5), 4), "3": ((12, 6, 3), 6), "4": ((6, 4, 1.5), 6),
           "5": ((2, 1, 0.5), 2), "tram": ((8, 5, 2), 1), "mono": ((8, 5, 2), 4)}
SEATS_CAR = {"1": 80}  # Shinkansen seats a car; others 150 standing and seated
PEAK_SHARE = 0.06


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


def _densities(path):
    """(operator, line) -> the median of its sections' people a day."""
    if not path:
        return {}
    d = json.load(open(path, encoding="utf-8"))
    by = defaultdict(list)
    for f in d["features"]:
        p = f["properties"]
        v = p.get("density") or p.get("d_base")
        if v:
            by[(p["op"], p["line"])].append(float(v))
    return {k: float(np.median(v)) for k, v in by.items()}


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
    src["route_kind"], src["route_operator"], src["train_m"] and src["shapes"] (route -> [[lon, lat]])."""
    import re
    st_f = json.load(open(src["stations"], encoding="utf-8"))["features"]
    sec_f = json.load(open(src["sections"], encoding="utf-8"))["features"]
    dens = _densities(src.get("density"))
    skip = re.compile(src["skip_line"]) if src.get("skip_line") else None
    o = (139.7671, 35.6812)
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
    stations, trips, names = {}, [], {}
    src.setdefault("route_kind", {})
    src.setdefault("route_operator", {})
    src.setdefault("train_m", {"_": 120})
    src.setdefault("shapes", {})
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
        D = dens.get((op, line))
        if D:
            cars = 16 if cls == "1" else 1 if cls == "tram" else int(np.clip(round(D / 12000), 2, 15))
            seats = SEATS_CAR.get(cls, 150) * cars
            peak = float(np.clip(D * PEAK_SHARE / seats, 0.5, 24))
            tph = (peak, max(0.5, round(peak * 0.55, 2)), round(peak * 0.3, 2))
        else:
            tph, cars = DEFAULT[cls]
        kmh = KMH[cls]
        served = set()
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
            # a route leaving the area kept (`inside`) runs as its pieces inside
            ok = [inside(float(P[verts[i], 0]), float(P[verts[i], 1])) for i in vi]
            pieces, cur = [], []
            for k, good in enumerate(ok):
                if good:
                    cur.append(k)
                elif cur:
                    pieces.append(cur)
                    cur = []
            if cur:
                pieces.append(cur)
            all_verts, all_stops, all_at = verts, stops, at
            for piece in pieces:
                if len(piece) < 2:
                    continue
                a, b = piece[0], piece[-1]
                verts = all_verts[vi[a]:vi[b] + 1]
                stops = all_stops[a:b + 1]
                at = [x - all_at[a] for x in all_at[a:b + 1]]
                n_routes += 1
                route = f"{op}|{line}|{n_routes}"
                names[route] = line
                for vv in verts:
                    if vv in station_of:
                        k, nm = station_of[vv]
                        stations[k] = (nm, float(P[vv, 0]), float(P[vv, 1]))
                src["shapes"][route] = [[float(P[vv, 0]), float(P[vv, 1])] for vv in verts]
                src["route_kind"][route] = _kind(cls)
                src["route_operator"][route] = op
                src["train_m"][route] = cars * 20
                t_at = np.array(at) / (kmh / 3.6)
                for ks, tt in ((stops, t_at), (stops[::-1], (t_at[-1] - t_at)[::-1])):
                    dr = "0" if ks is stops else "1"
                    acc_n = 0.0
                    for h, n in enumerate(osm_interval_tph({}, "commuter", tph)):
                        acc_n += n
                        k = int(acc_n)
                        acc_n -= k
                        for j in range(k):
                            dep = h * 3600 + (j + 0.5) * 3600 / k
                            trips.append((route, dr, ks, [int(dep + v) for v in tt]))
    keep = {k for k, (_, lon, lat) in stations.items() if inside(lon, lat)}
    print(f"{src['name']}: {len(groups)} N02 lines, {n_routes} routes, {len(trips)} trips made up, {len(keep)} of {len(stations)} stations inside, "
          f"{sum(1 for k in groups if k in dens)} lines with a passing count")
    trips = [t for t in trips if all(k in keep for k in t[2])]
    return {k: v for k, v in stations.items() if k in keep}, trips, names
