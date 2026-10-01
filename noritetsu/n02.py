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
# Guided transit, which is light_rail above but is no tram: build_model.build_credits lets tram
# and light_rail lines credit each other only where neither is `guided`. Without it the
# 広島新交通1号線, in tunnel right under the 広島電鉄 streets, credited the tram lines above it.
GUIDED = {"16", "24", "25"}

# How far a station's platform geometry may sit from its line's track before the match is not
# believed. N02 draws platforms along the alignment, so this is generous on purpose.
SNAP_M = 400

# A station cuts every track of its line within this of its platform, nearest station winning,
# and line graphs get a vertex at least every STEP_M so that there is something near enough to
# cut. Trams get less: a one-way street loop runs its other track a block over. See `footprints`.
# 80 m, not the 150 m kr_register uses: at 100 m 大阪 環状線's 福島, a 東海道線 station only for
# the うめきた branch, claimed the 神戸線 tracks passing it, and 大阪-塚本 went through 福島.
FOOT_M = 80
FOOT_TRAM_M = 50
STEP_M = 40


def line_id(name, operator):
    h = hashlib.blake2b(f"{name}|{operator}".encode("utf-8"), digest_size=5)
    return "j" + h.hexdigest()


def centroid(coords):
    a = np.asarray(coords, dtype=np.float64)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


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
            by_line[key].append((sid, x, y, f["geometry"]["coordinates"]))
    log(f"N02: {len(stations)} station groups")
    return stations, by_line


def dist_km(x1, y1, x2, y2):
    return math.hypot((x2 - x1) * math.cos(math.radians((y1 + y2) / 2)) * 111.320,
                      (y2 - y1) * 110.570)


def build_graph(coord_lists):
    """A vertex-level graph of one line's track, densified to a vertex every STEP_M.

    NOT a merged chain.  Merging a double-track line end to end produces a run that goes out
    along one track and back along the other, so slicing between two stations traverses both:
    the San'in Line came out at 1,330 km against a published 674, and the Ou Line at four
    times its true length.  A graph lets the shortest path between neighbouring stations use
    one track and ignore the parallel one, which is also what makes junctions and passing
    loops harmless.

    Densified so that a station has a vertex on every track near its platform to claim (see
    `footprints`). Track vertices are numbered by their rounded coordinates, which is what
    joins one N02 section to the next; inserted ones take negative ints. Returns (adj, xy).
    """
    g = defaultdict(list)
    xy, num = {}, {}
    fresh = [0]

    def link(a, b):
        (x1, y1), (x2, y2) = xy[a], xy[b]
        pts = [a]
        k = int(dist_km(x1, y1, x2, y2) * 1000 // STEP_M)
        for j in range(1, k + 1):
            fresh[0] -= 1
            t = j / (k + 1)
            xy[fresh[0]] = (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
            pts.append(fresh[0])
        pts.append(b)
        for u, v in zip(pts[:-1], pts[1:]):
            w = dist_km(*xy[u], *xy[v])
            g[u].append((v, w))
            g[v].append((u, w))

    for coords in coord_lists:
        prev = None
        for x, y in coords:
            r = (round(x, 5), round(y, 5))
            k = num.setdefault(r, len(num))
            xy[k] = r
            if prev is not None and k != prev:
                link(prev, k)
            prev = k
    return g, xy


def footprints(xy, pts, radius):
    """Which station each track vertex near a platform belongs to, and each station's centre.

    A STATION CUTS EVERY TRACK OF ITS LINE BESIDE ITS PLATFORM, not just the one vertex nearest
    its centre. N02 draws a platform on one track, and a multi-track line has others running
    past it: at 品川 the 東海道線's other tracks slid by, so the register gave 田町-大井町
    (4.6 km, through 品川 without stopping), 東京-品川 and 有楽町-品川 past three stations, and
    東神奈川-保土ヶ谷 past 横浜, and the strip diagram drew those as branches.

    Every vertex within `radius` of the platform geometry is the station's, the nearest station
    winning. A vertex nearest a station's centre is always its own, and a station further than
    SNAP_M from any track is not on this line. Returns ({vertex: sid}, {sid: (x, y)}, {sid:
    {vertex: km from the centre}}).
    """
    from scipy.spatial import cKDTree

    ids = list(xy)
    arr = np.array([xy[i] for i in ids], dtype=np.float64)
    lat0 = float(arr[:, 1].mean())
    kx, ky = math.cos(math.radians(lat0)) * 111320, 110570
    tree = cKDTree(np.column_stack([arr[:, 0] * kx, arr[:, 1] * ky]))

    by_st = defaultdict(list)
    for sid, x, y, coords in pts:
        by_st[sid].append((x, y, coords))
    foot, foot_d, centre, anchor = {}, {}, {}, {}
    for sid, feats in by_st.items():
        cx = sum(f[0] for f in feats) / len(feats)
        cy = sum(f[1] for f in feats) / len(feats)
        d, j = tree.query([cx * kx, cy * ky])
        if d > SNAP_M:
            continue
        centre[sid], anchor[sid] = (cx, cy), ids[j]
        # The platform, densified, so a long one claims track along all of it.
        plat = []
        for _x, _y, coords in feats:
            for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
                n = max(1, int(dist_km(x1, y1, x2, y2) * 1000 // 20))
                plat += [(x1 + (x2 - x1) * t / n, y1 + (y2 - y1) * t / n) for t in range(n)]
            plat.append(tuple(coords[-1]))
        q = np.array(plat, dtype=np.float64)
        for hits, (px, py) in zip(tree.query_ball_point(
                np.column_stack([q[:, 0] * kx, q[:, 1] * ky]), radius), plat):
            for j in hits:
                v = ids[j]
                dv = dist_km(*xy[v], px, py)
                if dv < foot_d.get(v, INF):
                    foot[v], foot_d[v] = sid, dv
    for sid, v in anchor.items():
        foot[v] = sid
    offsets = defaultdict(dict)
    for v, sid in foot.items():
        offsets[sid][v] = dist_km(*xy[v], *centre[sid])
    return foot, centre, offsets


def neighbours(g, foot):
    """Station pairs with no third station between them: a flood from each station's
    footprint that ABSORBS at the first vertex of another's, so it never walks past a station.

    Which station is met first along each branch depends only on topology, so a plain flood
    is enough to find the pairs; `between` then measures each one. This replaced ordering the
    stations by distance from a terminus, which cannot work on a loop -- there is no terminus,
    so the order ran outwards in both directions at once and every section jumped across the
    loop. The Oedo Line came out at 178 km against 40.7.
    """
    by_st = defaultdict(list)
    for v, s in foot.items():
        by_st[s].append(v)
    pairs = set()
    for sid, own in by_st.items():
        seen, frontier = set(own), list(own)
        while frontier:
            u = frontier.pop()
            for v, _w in g.get(u, ()):
                if v in seen:
                    continue
                seen.add(v)
                other = foot.get(v)
                if other is not None and other != sid:
                    pairs.add((sid, other) if sid <= other else (other, sid))
                    continue
                frontier.append(v)
    return pairs


def between(g, foot, a, b, src, dst):
    """Shortest track from station a to station b through no third station's footprint.

    Starts anywhere in a's footprint and finishes anywhere in b's, each end charged its
    distance from its station's centre (`src`, `dst`), so the section still measures centre
    to centre wherever along the platform the track it uses passes. Returns (nodes, km)."""
    dist, prev, seen = dict(src), {}, set()
    heap = [(d, v) for v, d in src.items()]
    heapq.heapify(heap)
    best, best_u = INF, None
    while heap:
        d, u = heapq.heappop(heap)
        if d >= best:
            break
        if u in seen:
            continue
        seen.add(u)
        if u in dst and d + dst[u] < best:
            best, best_u = d + dst[u], u
        for v, w in g.get(u, ()):
            s = foot.get(v)
            if s is not None and s != a and s != b:
                continue
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if best_u is None:
        return None
    path = [best_u]
    while path[-1] in prev:
        path.append(prev[path[-1]])
    path.reverse()
    return path, best


def reverses(pts, window_m=60, turn_deg=150):
    """Whether a path doubles back on itself: its heading over the `window_m` before some point
    and the `window_m` after it differ by more than `turn_deg`.

    The track graph has no idea which way a set of points faces, so the shortest track between
    two stations can run out to a junction and back down the other leg: 大井町 north towards
    北品川 and back down the 大崎 branch to 西大井, 4.2 km; 新川崎 south towards 鶴見 and back
    up to 川崎, 7.4 km. No train does that except on a switchback. The window is wide enough
    that a tram's street corner (about 90 degrees) does not count."""
    if len(pts) < 3:
        return False
    a = np.asarray(pts, dtype=np.float64)
    kx = math.cos(math.radians(float(a[:, 1].mean()))) * 111320
    p = np.column_stack([a[:, 0] * kx, a[:, 1] * 110570])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    if s[-1] < 2 * window_m:
        return False
    cos_lim = math.cos(math.radians(turn_deg))
    back = np.searchsorted(s, s - window_m, side="right") - 1
    ahead = np.searchsorted(s, s + window_m)
    for i in range(len(p)):
        j, k = back[i], ahead[i]
        if j < 0 or k >= len(p) or s[i] - s[j] < window_m / 2 or s[k] - s[i] < window_m / 2:
            continue
        u, v = p[i] - p[j], p[k] - p[i]
        nu, nv = np.hypot(*u), np.hypot(*v)
        if nu and nv and (u @ v) / (nu * nv) < cos_lim:
            return True
    return False


def drop_reversing(sections):
    """Drop sections that double back (see `reverses`) where the line's other sections still
    join their two stations. A switchback that is the only way between two stations, as on
    the 箱根登山線, stays. Longest first, one at a time, so two such sections that are each
    other's only alternative do not both go. Returns the keys dropped."""
    gone = []
    for key in sorted((k for k, v in sections.items() if v["reverses"]),
                      key=lambda k: -sections[k]["km"]):
        a, b = key
        adj = defaultdict(set)
        for x, y in sections:
            if (x, y) != key:
                adj[x].add(y)
                adj[y].add(x)
        seen, stack = {a}, [a]
        while stack:
            for v in adj[stack.pop()] - seen:
                seen.add(v)
                stack.append(v)
        if b in seen:
            del sections[key]
            gone.append(key)
    return gone


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
    reversing = []
    for key, coord_lists in by_line.items():
        name, operator = key
        pts = st_by_line.get(key, [])
        if len(pts) < 2:
            n_short += 1
            continue
        g, xy = build_graph(coord_lists)
        if not g:
            continue
        kind = KIND.get(meta[key].get("N02_001", ""), "rail")
        foot, centre, offsets = footprints(xy, pts, FOOT_TRAM_M if kind == "tram" else FOOT_M)
        if len(centre) < 2:
            n_unplaced += 1
            continue

        lid = line_id(name, operator)
        sections = {}
        for a, b in sorted(neighbours(g, foot)):
            got = between(g, foot, a, b, offsets[a], offsets[b])
            if got is None:
                continue
            nodes, km = got
            # Drawn from centre to centre, as it is measured, through the track's own vertices
            # (the inserted ones lie on its straight edges).
            sections[(a, b)] = {"km": km, "straight": False, "geom": [centre[a]] + [
                xy[n] for i, n in enumerate(nodes)
                if n >= 0 or i == 0 or i == len(nodes) - 1] + [centre[b]],
                "reverses": reverses([xy[n] for n in nodes])}
        for a, b in drop_reversing(sections):
            reversing.append(f"{name} {stations[a]['name']}-{stations[b]['name']}")
        if not sections:
            n_unplaced += 1
            continue
        order = walk_order(sections.keys())
        for sid in order:
            stations[sid]["lines"].add(lid)
        lines.append({
            "id": lid, "src": "n02", "service": False,
            "name": name, "name_en": "", "ref": "", "colour": "",
            "operator": operator, "operator_en": "", "network": "",
            "kind": kind,
            "highspeed": "新幹線" in name,
            "guided": meta[key].get("N02_001", "") in GUIDED,
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
        f"({n_short} line keys had under two stations, {n_unplaced} could not place any); "
        f"{len(reversing)} sections dropped for doubling back at a junction")
    for r in reversing:
        log(f"    {r}")
    return lines, stations, geoms
