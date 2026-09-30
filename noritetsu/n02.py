"""Lines, stations and sections from 国土数値情報 N02, Japan's own railway register.

    (used by build_model.py --n02 data/raw/N02-24_GML.zip)

WHY THIS EXISTS.  OpenStreetMap is not a line register and never claimed to be: it maps what
someone chose to map.  In Tokyo that is everything; along the San'in coast it is the limited
expresses and nothing else, so 220 stations and 17,847 km of track were reachable in our model
only by a named train, and named trains do not count towards completing anything.  N02 names
the line on every one of its 21,932 track sections, so the San'in Line is simply there, with
342 sections and 161 stations.

WHAT N02 GIVES AND DOES NOT GIVE.  It gives the official line name (N02_003), the operating
company (N02_004), a mode code (N02_001), and -- on stations -- a station group code
(N02_005g) which is the register's own statement of which platforms are one complex, 9,048 of
them.  It gives no colours, no English names, no operating patterns (京浜東北線 is not a line
in the register; it runs over 東北線 and 東海道線) and no named trains.  Those stay with OSM
and are matched on afterwards in build_model.

THE LINE KEY IS (name, operator), NOT NAME.  本線 alone is 639 sections, because a dozen
private railways each have a line called simply "the main line".
"""
import hashlib
import heapq
import json
import math
import zipfile
from collections import Counter, defaultdict

import numpy as np

INF = float("inf")

# N02_001, the mode. The 1x block is under the Railway Business Act and the 2x block under the
# Tramways Act, with the same modes appearing in both: Okinawa's monorail is legally a tramway.
# So is every Osaka Metro line and the Keihanna Line, which are code 21 like a street tram;
# build_model.register_way_lines corrects those from the track they lie on.
#   13 funicular   14 / 22 suspended monorail   15 / 23 straddle monorail
#   16 / 24 guided (AGT, and Nagoya's guideway bus)   25 maglev (Linimo)
# 22 was mapped to funicular until 2026-09-30, which left the Chiba Urban Monorail, the only
# line with that code, unclickable on the map.
KIND = {
    "11": "rail", "12": "rail",
    "13": "funicular",
    "14": "monorail", "15": "monorail", "22": "monorail", "23": "monorail",
    "16": "light_rail", "24": "light_rail", "25": "light_rail",
    "21": "tram",
}

# How far a station's platform geometry may sit from its line's track before the match is not
# believed. N02 draws platforms along the alignment, so this is generous on purpose.
SNAP_M = 400


def line_id(name, operator):
    h = hashlib.blake2b(f"{name}|{operator}".encode("utf-8"), digest_size=5)
    return "j" + h.hexdigest()


def centroid(coords):
    a = np.asarray(coords, dtype=np.float64)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


def length_km(coords):
    c = np.asarray(coords, dtype=np.float64)
    if len(c) < 2:
        return 0.0
    lat = np.radians((c[:-1, 1] + c[1:, 1]) / 2)
    dx = np.diff(c[:, 0]) * np.cos(lat) * 111.320
    dy = np.diff(c[:, 1]) * 110.570
    return float(np.hypot(dx, dy).sum())


def read(zip_path, log):
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        sec_name = next(n for n in names
                        if n.startswith("UTF-8/") and n.endswith("RailroadSection.geojson"))
        st_name = next(n for n in names
                       if n.startswith("UTF-8/") and n.endswith("Station.geojson"))
        with zf.open(sec_name) as f:
            sections = json.load(f)["features"]
        with zf.open(st_name) as f:
            stations = json.load(f)["features"]
    log(f"N02: {len(sections)} track sections, {len(stations)} station records")
    return sections, stations


def build_stations(station_feats, log):
    """One record per station GROUP, which is the register's own idea of a complex.

    Shinjuku is eleven records in N02 and one group; our OSM model had to guess that with
    name-and-distance matching. Here it is stated.
    """
    groups = defaultdict(list)
    for f in station_feats:
        p = f["properties"]
        g = p.get("N02_005g") or p.get("N02_005c")
        if not g:
            continue
        groups[g].append(f)

    stations, by_line = {}, defaultdict(list)
    for g, feats in groups.items():
        names = Counter(f["properties"].get("N02_005", "") for f in feats)
        xs, ys = [], []
        for f in feats:
            x, y = centroid(f["geometry"]["coordinates"])
            xs.append(x)
            ys.append(y)
        sid = f"g{g}"
        stations[sid] = {
            "id": sid, "name": names.most_common(1)[0][0], "name_en": "",
            "lon": sum(xs) / len(xs), "lat": sum(ys) / len(ys), "lines": set(),
        }
        for f in feats:
            p = f["properties"]
            key = (p.get("N02_003", ""), p.get("N02_004", ""))
            x, y = centroid(f["geometry"]["coordinates"])
            by_line[key].append((sid, x, y))
    log(f"N02: {len(stations)} station groups")
    return stations, by_line


def build_graph(coord_lists):
    """A vertex-level graph of one line's track.

    NOT a merged chain.  Merging a double-track line end to end produces a run that goes out
    along one track and back along the other, so slicing between two stations traverses both:
    the San'in Line came out at 1,330 km against a published 674, and the Ou Line at four
    times its true length.  A graph lets the shortest path between neighbouring stations use
    one track and ignore the parallel one, which is also what makes junctions and passing
    loops harmless.
    """
    g = defaultdict(list)
    for coords in coord_lists:
        prev = None
        for x, y in coords:
            k = (round(x, 5), round(y, 5))
            if prev is not None and k != prev:
                w = length_km([prev, k])
                g[prev].append((k, w))
                g[k].append((prev, w))
            prev = k
    return g


def adjacent(g, snapped):
    """Station pairs with no third station between them, and the track that joins them.

    A Dijkstra from each station that ABSORBS at any other station: the moment it reaches
    one, that pair is adjacent and the search does not continue past it. So the result is
    exactly the inter-station sections, whatever shape the line is.

    This replaced ordering the stations by distance from a terminus, which cannot work on a
    loop -- there is no terminus, so the order ran outwards in both directions at once and
    every section jumped across the loop. The Oedo Line came out at 178 km against 40.7.
    """
    at = {node: sid for sid, node in snapped.items()}
    out = {}
    for sid, src in snapped.items():
        dist, prev, seen = {src: 0.0}, {}, set()
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in seen:
                continue
            seen.add(u)
            other = at.get(u)
            if other is not None and other != sid:
                key = (sid, other) if sid <= other else (other, sid)
                if key not in out:
                    nodes, cur = [u], u
                    while cur != src:
                        cur = prev[cur]
                        nodes.append(cur)
                    nodes.reverse()
                    out[key] = (nodes, d)
                continue                     # absorbed: do not walk past a station
            for v, w in g.get(u, ()):
                nd = d + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
    return out


def walk_order(keys):
    """A reading order for the strip diagram: from a terminus if there is one, else round."""
    adj = defaultdict(list)
    for a, b in keys:
        adj[a].append(b)
        adj[b].append(a)
    if not adj:
        return []
    start = next((k for k, v in sorted(adj.items()) if len(v) == 1), None)
    if start is None:
        start = sorted(adj)[0]               # a loop has no end to start from
    order, seen, stack = [], set(), [start]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        order.append(u)
        for v in sorted(adj[u], reverse=True):
            if v not in seen:
                stack.append(v)
    return order


def snap(g, pts):
    """Each station to its nearest graph vertex, dropping any too far from the line."""
    nodes = np.array(list(g.keys()), dtype=np.float64)
    if not len(nodes):
        return {}
    out = {}
    for sid, x, y in pts:
        dx = (nodes[:, 0] - x) * math.cos(math.radians(y)) * 111320
        dy = (nodes[:, 1] - y) * 110570
        d = np.hypot(dx, dy)
        j = int(np.argmin(d))
        if d[j] <= SNAP_M:
            out[sid] = (float(nodes[j][0]), float(nodes[j][1]))
    return out


def build(zip_path, log):
    section_feats, station_feats = read(zip_path, log)
    stations, st_by_line = build_stations(station_feats, log)

    by_line = defaultdict(list)
    meta = {}
    for f in section_feats:
        p = f["properties"]
        key = (p.get("N02_003", ""), p.get("N02_004", ""))
        by_line[key].append(f["geometry"]["coordinates"])
        meta.setdefault(key, p)

    lines, geoms = [], {}
    n_unplaced = n_short = 0
    for key, coord_lists in by_line.items():
        name, operator = key
        pts = st_by_line.get(key, [])
        if len(pts) < 2:
            n_short += 1
            continue
        g = build_graph(coord_lists)
        if not g:
            continue
        snapped = snap(g, pts)
        if len(snapped) < 2:
            n_unplaced += 1
            continue
        pairs = adjacent(g, snapped)
        if not pairs:
            n_unplaced += 1
            continue
        order = walk_order(pairs.keys())

        lid = line_id(name, operator)
        sections = {}
        for k, (nodes, km) in pairs.items():
            sections[k] = {"km": km, "geom": [list(n) for n in nodes], "straight": False}

        if not sections:
            continue
        for sid in order:
            stations[sid]["lines"].add(lid)
        kind = KIND.get(meta[key].get("N02_001", ""), "rail")
        lines.append({
            "id": lid, "src": "n02", "service": False,
            "name": name, "name_en": "", "ref": "", "colour": "",
            "operator": operator, "operator_en": "", "network": "",
            "kind": kind,
            "highspeed": "新幹線" in name,
            "km": round(sum(s["km"] for s in sections.values()), 3),
            "variants": 1,
            "straight_sections": sum(1 for s in sections.values() if s["straight"]),
            "display": order,
            "sections": [[a, b, round(v["km"], 3)] for (a, b), v in sections.items()],
        })
        geoms[lid] = {f"{a}|{b}": [[round(x, 5), round(y, 5)] for x, y in v["geom"]]
                      for (a, b), v in sections.items()}

    total = sum(l["km"] for l in lines)
    log(f"N02: {len(lines)} lines, {total:,.0f} km "
        f"({n_short} line keys had under two stations, {n_unplaced} could not place any)")
    return lines, stations, geoms
