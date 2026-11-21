"""T-3935: lines OSM maps only as track, for realauto's extract (notes/T-3935.md).

realauto builds a real network from OSM's passenger route relations. Some lines have none, only their track: Kansai's
Keihan Main Line is a `route=railway` relation and nothing else, so its Uji and Katano branches touch nothing. This
finds those lines from their track and gives each as a route relation realnet's `osm` adapter reads like any other.

The track of each line: in Japan N02 (MLIT's railway register: every line by operator and name, its track and its
stations; noritetsu's Japan extract keeps no infrastructure relations), else the region's infrastructure line
relations (noritetsu's `infra.pkl`, `route=railway` and `route=tracks`; Hong Kong, Singapore, Taiwan and Korea have
none, so none there). Left out: Shinkansen, funiculars and N02's trolleybuses; track mostly high-speed, freight,
industrial, military, tourist, preserved or service. A line's stations are its own in N02, else the stations, halts
and stops within SNAP_M of its track (of its mode, where the stop says). Its station paths are realnet_n02's (the
longest path between stations over the line's track, then the longest over what is left, a branch carried on to the
nearest station of its trunk), cut to the stations inside the pack, with the inside share and spacing rules realauto
applies to routes.

A path becomes a line when under COVERED of its length runs between neighbouring stations that one passenger route
relation both stops at (by station name; every route relation of the extract counts, the ones realauto drops too, so
track that only intercity or high-speed trains run on does not come back as a commuter line). Each N02 station is the
OSM stop of its name nearest it within NAME_M (the Uji Line's Chushojima and the main line's are then one station),
else a stop of its own named from N02 (and noritetsu's romanised name). The line's name and kind come from the OSM
track under it (`name:en`, `railway`), else noritetsu's English name for the N02 line; its colour from noritetsu's
line. Its trains are realnet's default for its kind: these relations carry no `interval`, and no realauto rule for
suburban trains applies to them."""
import hashlib
import json
import math
import os
import pickle
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from realauto import shared_path

NORI_PROC = shared_path("noritetsu", "data", "proc")
NORI_DIST = shared_path("noritetsu", "dist", "data")
N02_DIR = shared_path("riders", "tokyoriders", "data", "N02-24", "UTF-8")

COVERED = 0.5
COVER_STEP_M = 100.0
TRACK_M = 30.0
SNAP_M = 100.0
NAME_M = 600.0
SAME_NAME_M = 400.0
INSIDE_SHARE, INSIDE_ENOUGH = 0.5, 8
MAX_SPACING_M = 15000.0
MIN_KM = 1.0
STUB_M = 1500.0
BYPASS_RATIO = 1.5
RURAL_SPACING_M = 2000.0
RURAL_INTERVAL = "30"
KINDS = ("train", "subway", "light_rail", "tram", "monorail")
PASSENGER_TRACK = {"rail", "light_rail", "subway", "tram", "monorail", "narrow_gauge"}
SKIP_USAGE = {"freight", "military", "industrial", "tourism", "test"}
# a freight line's relation that does not say so in `usage` (India's Dedicated Freight Corridors)
FREIGHT = re.compile(r"freight|güter|cargo|carga|\bfret\b|грузов|货运|貨物", re.I)
ROUTE_OF_TRACK = {"rail": "train", "narrow_gauge": "train", "subway": "subway", "light_rail": "light_rail", "tram": "tram", "monorail": "monorail"}
# N02_001 funicular and trolleybus; N02_002 1 is a Shinkansen
N02_SKIP_MODE = {"13", "17"}
N02_ROUTE = {"tram": "tram", "mono": "monorail", "3": "subway"}
# synthetic ids: infrastructure relation id x 100 + path, or N02's own key hashed, above any OSM relation or node id
INFRA_ID = 10 ** 12
N02_ID = 2 * 10 ** 12
M_PER_DEG = 111195.0


def _latin(s):
    letters = [ch for ch in s if ch.isalpha()]
    return bool(letters) and all("LATIN" in unicodedata.name(ch, "") for ch in letters)


def _m(a, b, k):
    return math.hypot((a[0] - b[0]) * k, a[1] - b[1]) * M_PER_DEG


def station_paths(polys, stations, k):
    """Station paths over a line's track. `polys`: [[(lon, lat)]] lying on each other's ends where they join;
    `stations`: [(key, (lon, lat))], each a point of `polys`. Returns [([key], [arc m], [(lon, lat)] the track)]."""
    import realnet_n02
    vid, pos, adj = {}, [], defaultdict(set)

    def v_of(c):
        kk = (round(c[0], 7), round(c[1], 7))
        if kk not in vid:
            vid[kk] = len(pos)
            pos.append(kk)
        return vid[kk]
    for q in polys:
        ids = [v_of(c) for c in q]
        for a, b in zip(ids, ids[1:]):
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
    station_of = {}
    for key, c in stations:
        v = v_of(c)
        if v in adj and v not in station_of:
            station_of[v] = (key, key)

    def w(a, b):
        return _m(pos[a], pos[b], k)
    key_v = set(station_of) | {v for v in adj if len(adj[v]) != 2}
    edges = {}
    for a in key_v:
        for nb in adj[a]:
            chain, prev, cur, L = [a], a, nb, w(a, nb)
            while cur not in key_v:
                chain.append(cur)
                nxt = next(iter(adj[cur] - {prev}))
                L += w(cur, nxt)
                prev, cur = cur, nxt
            chain.append(cur)
            if a == cur:
                continue
            kk = (min(a, cur), max(a, cur))
            if kk not in edges or edges[kk][0] > L:
                edges[kk] = (L, chain if a < cur else chain[::-1])
    out, served = [], set()
    for hops in realnet_n02._routes(edges, key_v, station_of):
        verts = []
        for u, v, e in hops:
            ch = edges[e][1]
            ch = ch if ch[0] == u else ch[::-1]
            verts += ch if not verts else ch[1:]
        ks, at, vi, acc = [], [], [], 0.0
        on = set(verts)
        for i, vv in enumerate(verts):
            if i:
                acc += w(verts[i - 1], vv)
            if vv in station_of and (not ks or ks[-1] != station_of[vv][0]):
                ks.append(station_of[vv][0])
                at.append(acc)
                vi.append(i)
            elif vv in key_v and len(adj[vv]) >= 3 and ks:
                # a station at the end of a short spur off the path is where the trains turn back (the Yoro Line's
                # Ogaki): stopped at from the junction
                for nb in adj[vv] - on:
                    prev, cur, L, found = vv, nb, w(vv, nb), []
                    while L <= STUB_M:
                        if cur in station_of:
                            found.append(station_of[cur][0])
                        if len(adj[cur]) != 2:
                            break
                        nxt = next(iter(adj[cur] - {prev}))
                        L += w(cur, nxt)
                        prev, cur = cur, nxt
                    if L <= STUB_M and len(adj[cur]) == 1 and len(set(found)) == 1 and found[0] not in ks:
                        ks.append(found[0])
                        at.append(acc)
                        vi.append(i)
        if len(ks) < 2 or set(ks) <= served:
            continue
        served |= set(ks)
        out.append((ks, at, [pos[verts[i]] for i in range(vi[0], vi[-1] + 1)], vi))
    return out


class Region:
    """What every line of a pack needs: the region's stops by name, its OSM track for names and kinds, the
    stations each passenger route relation stops at."""

    def __init__(self, cc, rels, stops, ways, xy_of, bbox, in_pack, kept_nodes):
        self.cc, self.rels, self.stops, self.ways, self.xy_of = cc, rels, stops, ways, xy_of
        self.bbox, self.in_pack = bbox, in_pack
        w, s, e, n = bbox
        self.k = math.cos(math.radians((s + n) / 2))
        self._route_ways = None
        pad = 0.02
        self.by_name = defaultdict(list)  # name -> [(node id, (lon, lat), kept)]
        self.box_stops = []  # named stops in the box: (node id, (lon, lat), tags)
        for r, (t, lon, lat) in stops.items():
            if t.get("name") and w - pad <= lon <= e + pad and s - pad <= lat <= n + pad:
                self.by_name[t["name"]].append((r, (lon, lat), r in kept_nodes))
                if w <= lon <= e and s <= lat <= n:
                    self.box_stops.append((r, (lon, lat), t))
        self._tree = None
        self._ids = None

    def way_xy(self, wid):
        """A way's points (lon, lat) as an array, nodes with no coordinate left out."""
        if self._ids is None:
            z = np.load(os.path.join(NORI_PROC, self.cc, "coords.npz"))
            o = np.argsort(z["id"])
            self._ids, self._x, self._y = z["id"][o], z["x"][o] / 1e7, z["y"][o] / 1e7
        nodes = np.asarray(self.ways[wid][1], dtype=np.int64)
        j = np.clip(np.searchsorted(self._ids, nodes), 0, len(self._ids) - 1)
        ok = self._ids[j] == nodes
        return np.stack([self._x[j[ok]], self._y[j[ok]]], 1)

    def in_box(self, q, pad=0.0):
        w, s, e, n = self.bbox
        return bool(len(q)) and bool(np.any((q[:, 0] >= w - pad) & (q[:, 0] <= e + pad) & (q[:, 1] >= s - pad) & (q[:, 1] <= n + pad)))

    def track_tree(self):
        """OSM passenger track in the box, as (STRtree of LineStrings in lon/lat, [way tags])."""
        if self._tree is None:
            from shapely import STRtree
            from shapely.geometry import LineString
            geoms, tags, wids = [], [], []
            for wid, (t, nodes) in self.ways.items():
                if t.get("railway") not in PASSENGER_TRACK:
                    continue
                pts = self.way_xy(wid)
                if len(pts) < 2 or not self.in_box(pts, 0.05):
                    continue
                geoms.append(LineString(np.stack([pts[:, 0] * self.k, pts[:, 1]], 1)))
                tags.append(t)
                wids.append(wid)
            self._tree = (STRtree(geoms) if geoms else None, tags, wids)
        return self._tree

    def nearest_track(self, shape, step_m=None):
        """The OSM passenger track nearest points along `shape` within TRACK_M: [(way id or None, tags or None)], a
        point every `step_m` metres (else 24 points)."""
        import shapely
        tree, tags, wids = self.track_tree()
        if tree is None or len(shape) < 2:
            return []
        pts = _resample(shape, self.k, step_m) if step_m else shape[::max(1, len(shape) // 24)]
        P = shapely.points(np.array([(p[0] * self.k, p[1]) for p in pts], float))
        near = tree.query_nearest(P, max_distance=TRACK_M / M_PER_DEG, all_matches=False)
        within = tree.query(P, predicate="dwithin", distance=TRACK_M / M_PER_DEG)
        all_w = defaultdict(list)
        for a, b in zip(*within):
            all_w[int(a)].append((wids[int(b)], tags[int(b)]))
        first = {int(a): int(b) for a, b in zip(*near)}
        return [(wids[first[i]], tags[first[i]], all_w[i]) if i in first else (None, None, []) for i in range(len(pts))]

    def track_under(self, shape):
        """The tags of the OSM track nearest points along a line's shape."""
        return [t for _, t, _ in self.nearest_track(shape) if t is not None]

    def covered_share(self, got, own):
        """Share of `nearest_track` points where some passenger route relation (any of the extract's, dropped ones
        too) runs over the nearest track, or over any track near that is this line's own (`own(way id, tags)`: a
        line on four tracks, a station's many). A line beside another (the Keihan Main Line beside the Nara Line at
        Tofukuji) is then not taken as covered there; a point with no OSM track near is not covered."""
        if self._route_ways is None:
            self._route_ways = {r for t, ms in self.rels.values() if t.get("type") == "route" and t.get("route") in KINDS
                                for ty, r, _ in ms if ty == "w"}
        R = self._route_ways
        return sum(w0 in R or any(w in R and own(w, t) for w, t in ws) for w0, _, ws in got) / max(1, len(got))

    def take(self, shape, track):
        """A line found runs on this track now (the ways near it that `track(way id, tags)` says are its own): a
        later candidate on it (a second relation over the same track, the other track of a pair) is covered there.
        Only its own: the nearest way where two lines part (the Nakanoshima Line leaving the main line at
        Temmabashi) is the other line's."""
        got = self.nearest_track(shape, COVER_STEP_M)
        self.covered_share(got[:1], track)  # fills _route_ways
        for _, _, ws in got:
            self._route_ways.update(w for w, t in ws if track(w, t))

    def osm_stop(self, name, ll):
        """An OSM stop named `name` within NAME_M of `ll`, or None: one a kept route stops at first, then a station
        node, then the lowest id, so that two points of one station (N02 maps some twice) get the same stop."""
        c = [(not kept, self.stops[r][0].get("railway") != "station", r, p) for r, p, kept in self.by_name.get(name, [])
             if _m(p, ll, self.k) <= NAME_M]
        return min(c)[2:] if c else None


def _resample(shape, k, step):
    """Points every `step` metres along a polyline (its ends included)."""
    out = [shape[0]]
    run = 0.0
    for a, b in zip(shape, shape[1:]):
        seg = _m(a, b, k)
        while seg > 0 and run + seg >= step:
            f = (step - run) / seg
            a = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            out.append(a)
            seg = _m(a, b, k)
            run = 0.0
        run += seg
    out.append(shape[-1])
    return out


def _inside_runs(ks, ll, inside):
    runs, cur = [], []
    for i, key in enumerate(ks):
        if inside(ll[key]):
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [r for r in runs if len(r) >= 2]


def _cut(ks, at, ll, inside, k):
    """A station path's pieces inside the pack, with the inside share and spacing rules realauto gives routes:
    [(index range of ks)] or [] with the reason."""
    runs = _inside_runs(ks, ll, inside)
    n_in = sum(len(r) for r in runs)
    if not runs:
        return [], "outside"
    if n_in < INSIDE_SHARE * len(ks) and n_in < INSIDE_ENOUGH:
        return [], "mostly outside"
    out = []
    for r in runs:
        if len(runs) > 1 and len(r) < 3:
            continue  # two stations of a line that leaves the pack and comes back (the Iida Line's Ozore - Kowada)
        gap = [_m(ll[ks[a]], ll[ks[b]], k) for a, b in zip(r, r[1:])]
        if sum(gap) / len(gap) > MAX_SPACING_M:
            continue
        if at[r[-1]] - at[r[0]] < MIN_KM * 1000:
            continue
        out.append(r)
    return out, None if out else "stops far apart or too short"


_N02 = {}


def _n02():
    if not _N02:
        st = json.load(open(os.path.join(N02_DIR, "N02-24_Station.geojson"), encoding="utf-8"))["features"]
        sec = json.load(open(os.path.join(N02_DIR, "N02-24_RailroadSection.geojson"), encoding="utf-8"))["features"]
        _N02["st"], _N02["sec"] = st, sec
        p = os.path.join(NORI_DIST, "jp", "lines.json")
        _N02["lines"] = {l["id"]: l for l in json.load(open(p, encoding="utf-8"))["lines"]} if os.path.exists(p) else {}
        p = os.path.join(NORI_DIST, "jp", "names_en.json")
        _N02["names_en"] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    return _N02


def nori_line_id(name, operator):
    """noritetsu's id for an N02 line (noritetsu/n02.py line_id)."""
    return "j" + hashlib.blake2b(f"{name}|{operator}".encode("utf-8"), digest_size=5).hexdigest()


def _coords(geom):
    return [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]


def n02_candidates(reg):
    """N02's lines with 2 or more stations in the box: (op, line, mode class, polys, [(n02 index, name, (lon, lat))])."""
    import realnet_n02
    d = _n02()
    w, s, e, n = reg.bbox
    groups = defaultdict(lambda: {"polys": [], "sts": []})
    for f in d["sec"]:
        p = f["properties"]
        if p["N02_001"] in N02_SKIP_MODE or p["N02_002"] == "1":
            continue
        groups[(p["N02_004"], p["N02_003"])]["polys"] += _coords(f["geometry"])
    for i, f in enumerate(d["st"]):
        p = f["properties"]
        g = groups.get((p["N02_004"], p["N02_003"]))
        if g is None or p["N02_001"] in N02_SKIP_MODE or p["N02_002"] == "1":
            continue
        cs = _coords(f["geometry"])[0]
        mid = np.mean(np.array(cs), axis=0)
        best = min(cs, key=lambda c: (c[0] - mid[0]) ** 2 + (c[1] - mid[1]) ** 2)
        g["polys"].append(cs)
        g["sts"].append((i, p["N02_005"], (round(best[0], 7), round(best[1], 7)), p.get("N02_005g")))
        g["cls"] = realnet_n02._class(p)
    for (op, line), g in sorted(groups.items()):
        inside = [x for x in g["sts"] if w <= x[2][0] <= e and s <= x[2][1] <= n]
        if len(inside) >= 2:
            yield op, line, g["cls"], g["polys"], g["sts"]


def infra_candidates(reg):
    """The region's infrastructure line relations on passenger track in the box: (relation id, tags, polys)."""
    p = os.path.join(NORI_PROC, reg.cc, "infra.pkl")
    if not os.path.exists(p):
        return
    infra = pickle.load(open(p, "rb"))
    w, s, e, n = reg.bbox
    for rid, (t, ms) in sorted(infra.items()):
        if t.get("usage") in SKIP_USAGE or t.get("highspeed") == "yes" or t.get("state") in ("disused", "abandoned", "proposed", "construction") or \
                FREIGHT.search(" ".join(t.get(x, "") for x in ("name", "name:en", "description"))):
            continue
        wl = [r for ty, r, role in ms if ty == "w" and r in reg.ways and role not in ("platform",)]
        if not wl:
            continue
        good = [r for r in wl if reg.ways[r][0].get("railway") in PASSENGER_TRACK and not reg.ways[r][0].get("service")
                and reg.ways[r][0].get("usage") not in SKIP_USAGE and reg.ways[r][0].get("highspeed") != "yes"]
        if len(good) <= 0.5 * len(wl):
            continue
        qs = [reg.way_xy(r) for r in good]
        if not any(reg.in_box(q) for q in qs):
            continue
        polys = [[(float(a), float(b)) for a, b in q] for q in qs if len(q) >= 2]
        yield rid, t, polys, [r for ty, r, role in ms if ty == "n" and role.startswith("stop")], set(wl)


def _named(line):
    """Whether an OSM way is an N02 line's own track: no name, or a name holding the line's (N02 drops 本 from JR
    names, 関西線 for 関西本線, and puts Osaka Metro's in brackets, 1号線(御堂筋線))."""
    parts = [x for x in re.split(r"[()（）]", line) if x]
    stems = [re.sub(r"(本線|線)$", "", x) for x in parts]
    stems = [x for x in stems if x and not re.fullmatch(r"\d+号", x)] or [line]
    return lambda w, tags: not tags.get("name") or any(x in tags["name"] for x in stems)


def _snap_stations(polys, cands, k):
    """Each candidate stop (node id, (lon, lat)) within SNAP_M of the track, put on it: the track with a point added
    at each, and [(node id, the point)]."""
    segs = []
    for qi, q in enumerate(polys):
        for j in range(len(q) - 1):
            segs.append((qi, j))
    if not segs or not cands:
        return polys, []
    import shapely
    A = np.array([polys[qi][j] for qi, j in segs], float)
    B = np.array([polys[qi][j + 1] for qi, j in segs], float)
    A[:, 0] *= k
    B[:, 0] *= k
    tree = shapely.STRtree(shapely.linestrings(np.stack([A, B], 1)))
    P_all = np.array([(lon * k, lat) for _, (lon, lat) in cands], float)
    hit = np.unique(tree.query(shapely.points(P_all), predicate="dwithin", distance=SNAP_M / M_PER_DEG)[0])
    D = B - A
    LL = np.maximum((D ** 2).sum(1), 1e-18)
    inserts = defaultdict(list)
    out = []
    for ci in hit:
        nid, (lon, lat) = cands[int(ci)]
        P = np.array([lon * k, lat])
        t = np.clip(((P - A) * D).sum(1) / LL, 0.0, 1.0)
        Q = A + D * t[:, None]
        d = np.hypot(*(Q - P).T) * M_PER_DEG
        i = int(np.argmin(d))
        if d[i] > SNAP_M:
            continue
        qi, j = segs[i]
        pt = (round(float(Q[i, 0] / k), 7), round(float(Q[i, 1]), 7))
        inserts[(qi, j)].append((float(t[i]), pt))
        out.append((nid, pt))
    new = []
    for qi, q in enumerate(polys):
        nq = []
        for j, c in enumerate(q):
            nq.append((round(c[0], 7), round(c[1], 7)))
            for _, pt in sorted(inserts.get((qi, j), [])):
                nq.append(pt)
        new.append(nq)
    return new, out


def _stop_ok(tags, route):
    """A passenger stop of a line of this kind: a railway station, halt or stop (a stop position or station
    only with a rail mode: bus stops are stop positions too), not a freight one."""
    modes = {m for m in KINDS if tags.get(m) == "yes"}
    if tags.get("railway") not in ("station", "halt", "stop", "tram_stop") and not (tags.get("public_transport") in ("station", "stop_position") and modes):
        return False
    if tags.get("usage") in SKIP_USAGE or tags.get("station") in ("freight", "yard") or tags.get("bus") == "yes":
        return False
    if tags.get("railway") == "tram_stop" and route != "tram":
        return False
    return not modes or route in modes


def lines(cc, rels, stops, ways, xy_of, bbox, in_pack, kept_nodes):
    """Every line the rule finds for a pack: [line dict], and the candidates it turned down with why."""
    reg = Region(cc, rels, stops, ways, xy_of, bbox, in_pack, kept_nodes)
    k = reg.k
    found, refused = [], []
    if cc == "jp":
        for op, line, cls, polys, sts in n02_candidates(reg):
            # N02 stations as OSM stops, before the paths: a station N02 maps twice (the main line's Chushojima)
            # is then one stop, and a path over its second copy adds nothing
            # (an N02 station OSM has no stop of its name near, a closed one or a signal box, is left out)
            own, on_track = {}, []
            for i, nm, ll, gid in sts:
                hit = reg.osm_stop(nm, ll)
                if not hit:
                    continue
                r, p = hit
                own.setdefault(r, (r, nm, p, stops[r][0]))
                on_track.append((r, ll))
            paths = station_paths(polys, on_track, k)
            for p, (ks, at, shape, vi) in enumerate(paths):
                found += _make(reg, [own[r] for r in ks], at, shape, vi, f"{op} {line}", ("n02", op, line, cls), _named(line), None, refused,
                               bypass=_bypasses(paths, p))
    for rid, t, polys, stop_members, rel_ways in infra_candidates(reg):
        kinds =Counter(ROUTE_OF_TRACK[x] for x in (tg.get("railway") for tg in reg.track_under([c for q in polys for c in q[::max(1, len(q) // 4)]])) if x in ROUTE_OF_TRACK)
        route = kinds.most_common(1)[0][0] if kinds else "train"
        cands = [(r, ll) for r, ll, tg in reg.box_stops if _stop_ok(tg, route)]
        cands += [(r, (stops[r][1], stops[r][2])) for r in stop_members if r in stops and stops[r][0].get("name")]
        polys2, snapped = _snap_stations(polys, list(dict(cands).items()), k)
        if len(snapped) < 2:
            continue
        for pi, (ks, at, shape, vi) in enumerate(station_paths(polys2, [(r, ll) for r, ll in snapped], k)):
            st = [(r, stops[r][0]["name"], (stops[r][1], stops[r][2]), stops[r][0]) for r in ks]
            lab = f"r{rid} {t.get('name', '')}"
            # any route's track near counts here: an infrastructure relation beside served track (Berlin's
            # mainline ring beside the S-Bahn's) is a second line on the same corridor, not a missing one
            found += _make(reg, st, at, shape, vi, lab, ("infra", rid, pi, t, route), lambda w, tg: True,
                           lambda w, tg, mine=rel_ways: w in mine, refused)
    return found, refused


def _bypasses(paths, p):
    """Neighbouring stations of path `p` that the line's other paths also join, through stations between them, over
    at most BYPASS_RATIO times the hop's length: a bypass track (the Tokaido Line's Shin-Tarui Line from Arao to
    Sekigahara beside its line through Ogaki and Tarui), whose trains are the other paths', wherever the longest
    path happened to go."""
    import heapq
    adj = defaultdict(dict)
    for q, (ks, at, *_) in enumerate(paths):
        if q == p:
            continue
        for i in range(len(ks) - 1):
            d = abs(at[i + 1] - at[i])
            a, b = ks[i], ks[i + 1]
            adj[a][b] = min(d, adj[a].get(b, math.inf))
            adj[b][a] = adj[a][b]
    out = set()
    ks, at = paths[p][0], paths[p][1]
    for i in range(len(ks) - 1):
        a, b, hop = ks[i], ks[i + 1], at[i + 1] - at[i]
        if a not in adj or b not in adj:
            continue
        dist, h = {a: (0.0, 0)}, [(0.0, 0, a)]
        while h:
            d, n, u = heapq.heappop(h)
            if d > BYPASS_RATIO * hop:
                break
            if u == b:
                if n >= 2:
                    out.add(frozenset((a, b)))
                break
            for v, w in adj[u].items():
                if d + w < dist.get(v, (math.inf, 0))[0]:
                    dist[v] = (d + w, n + 1)
                    heapq.heappush(h, (d + w, n + 1, v))
    return out


def _make(reg, st, at, shape, vi, lab, src, mine, track, refused, bypass=frozenset()):
    """The lines one station path makes: same-name stops within SAME_NAME_M merged, cut to the pack, kept where
    under COVERED of it is covered."""
    k = reg.k
    keep_i = []
    for i, (r, nm, ll, tags) in enumerate(st):
        if keep_i:
            j = keep_i[-1]
            if st[j][0] == r or (st[j][1] == nm and _m(st[j][2], ll, k) < SAME_NAME_M):
                continue
        keep_i.append(i)
    st = [st[i] for i in keep_i]
    at = [at[i] for i in keep_i]
    vi = [vi[i] - vi[0] for i in keep_i]  # each stop's point of `shape`
    if len(st) < 2:
        return []
    ll = {i: x[2] for i, x in enumerate(st)}
    runs, why = _cut(list(range(len(st))), at, ll, reg.in_pack, k)
    if not runs:
        refused.append({"line": lab, "why": why, "stops": len(st)})
        return []
    out = []
    for run, cov, why in _stretches(reg, st, at, shape, vi, runs, mine, bypass):
        names = [st[i][1] for i in run]
        a = [at[i] for i in run]
        piece = shape[vi[run[0]]:vi[run[-1]] + 1]
        rec = {"line": lab, "from": names[0], "to": names[-1], "stops": len(run), "km": round((a[-1] - a[0]) / 1000, 1), "covered": round(cov, 2)}
        if why:
            refused.append({**rec, "why": why})
            continue
        if track:  # (N02's lines are one each already)
            reg.take(piece, track)
        under = reg.track_under(piece)
        name, route, colour = _name_kind(reg, src, under)
        if src[0] == "n02":
            sid = N02_ID + int(hashlib.blake2b(f"{src[1]}|{src[2]}|{names[0]}|{names[-1]}".encode("utf-8"), digest_size=5).hexdigest(), 16)
        else:
            sid = INFRA_ID + src[1] * 100 + src[2]
        out.append({**rec, "id": sid, "name": name, "route": route, "colour": colour, "source": src[0],
                    "stop_nodes": [(st[i][0], st[i][2], st[i][3]) for i in run], "shape": piece,
                    "names": [name_of(st[i][3], reg.cc) for i in run]})
    return out


def _stretches(reg, st, at, shape, vi, runs, mine, bypass=frozenset()):
    """Each run's stretches of hops no passenger route runs on: (stop indices, covered share, why refused or None).
    A hop is covered when COVERED of its points lie on track some route runs over, and it breaks the run when under
    half its points have OSM passenger track near (N02 track that OSM has as gone: a BRT, a closed line) or when it
    is a `bypass` (`_bypasses`)."""
    out = []
    for run in runs:
        hops = []
        for i, j in zip(run, run[1:]):
            got = reg.nearest_track(shape[vi[i]:vi[j] + 1], COVER_STEP_M)
            present = sum(w is not None for w, _, _ in got) / max(1, len(got))
            cov = reg.covered_share(got, mine)
            hops.append((present >= 0.5 and frozenset((st[i][0], st[j][0])) not in bypass, cov, at[j] - at[i]))
        rec_all = sum(c * m for p, c, m in hops) / max(1e-9, sum(m for _, _, m in hops))
        if rec_all >= COVERED:
            out.append((run, rec_all, "routes run on it"))
            continue
        # a hop is the line's to add when uncovered; a lone hop between two of the other kind takes theirs (one
        # covered hop does not cut a line in two: the Kisei Line beside Kintetsu into Tsu; one uncovered hop
        # inside covered track is a mapping gap), and a hop with no OSM track is a hard break
        free = [p and c < COVERED for p, c, _ in hops]
        free = [free[h - 1] if 0 < h < len(free) - 1 and free[h - 1] == free[h + 1] != free[h] and hops[h - 1][0]
                and hops[h + 1][0] and hops[h][0] else free[h] for h in range(len(free))]
        h = 0
        while h < len(hops):
            if not free[h]:
                h += 1
                continue
            e = h
            while e + 1 < len(hops) and free[e + 1]:
                e += 1
            nxt = e + 1
            # on to the next station where the line meets track routes run on, so that it reaches its junction
            # (the Tarumi Line's first hop into Ogaki runs beside the Tokaido Line's)
            if h > 0 and hops[h - 1][0]:
                h -= 1
            if e + 1 < len(hops) and hops[e + 1][0]:
                e += 1
            piece = run[h:e + 2]
            km = sum(m for _, _, m in hops[h:e + 1])
            cov = sum(c * m for _, c, m in hops[h:e + 1]) / max(1e-9, km)
            out.append((piece, cov, "too short" if km < MIN_KM * 1000 else None))
            h = nxt
        if not all(p for p, _, _ in hops):
            out.append((run, rec_all, "no OSM track on some hops (cut there)"))
    return out


def name_of(tags, cc):
    import realauto
    return realauto.pick_name(tags, cc)


def _name_kind(reg, src, under):
    """(Latin name, route kind, colour) of a line from the OSM track under it and its source."""
    import realauto
    en = Counter(t.get("name:en") for t in under if t.get("name:en") and _latin(t.get("name:en")))
    kinds = Counter(ROUTE_OF_TRACK[t["railway"]] for t in under if t.get("railway") in ROUTE_OF_TRACK)
    colour = None
    if src[0] == "n02":
        _, op, line, cls = src
        d = _n02()
        lid = nori_line_id(line, op)
        nl = d["lines"].get(lid, {})
        colour = nl.get("colour") or None
        # the OSM track's own English name where it is this line's (its `name` holds the N02 name's stem)
        stem = re.sub(r"(本線|線)$", "", line)
        own = Counter(t.get("name:en") for t in under if t.get("name:en") and _latin(t["name:en"]) and stem and stem in (t.get("name") or ""))
        name = (own.most_common(1)[0][0] if own else None) or nl.get("name_en") or d["names_en"].get("ln", {}).get(lid) or line
        route = kinds.most_common(1)[0][0] if kinds else N02_ROUTE.get(cls, "train")
        if cls == "tram" and route == "train":
            route = "tram"
    else:
        _, rid, pi, t, route0 = src
        name = realauto.line_name(t, reg.cc) or (en.most_common(1)[0][0] if en else f"r{rid}")
        if not _latin(name) and en:
            name = en.most_common(1)[0][0]
        route = route0
        colour = t.get("colour") or None
    return name, route, colour


def elements(found, cc):
    """The extract's relations, nodes and shapes for the lines found: (elements, {shape key: [poly]}, new node ids)."""
    el, shapes, nodes = [], {}, {}
    for L in found:
        tags = {"type": "route", "route": L["route"], "name": L["name"], "ref": "", "realtracks": L["source"]}
        # a train line with stations far apart is a rural or outer one: every half hour, not the commuter default
        if L["route"] == "train" and L["km"] * 1000 / max(1, L["stops"] - 1) > RURAL_SPACING_M:
            tags["interval"] = RURAL_INTERVAL
        if L.get("colour"):
            tags["colour"] = L["colour"]
        el.append({"type": "relation", "id": L["id"], "tags": tags,
                   "members": [{"type": "node", "ref": r, "role": "stop"} for r, _, _ in L["stop_nodes"]]})
        shapes[f"r{L['id']}"] = [[[round(a, 6), round(b, 6)] for a, b in L["shape"]]]
        for r, ll, tg in L["stop_nodes"]:
            nodes.setdefault(r, (ll, name_of(tg, cc)))
    return el, shapes, nodes
