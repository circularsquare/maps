"""Lines, stations and sections for South Korea, with OpenStreetMap's named track as the register.

    python build_model.py --region kr --register kr_register:data/raw/kr

WHY THERE IS NO GEOMETRY FILE.  Korea publishes no open line-geometry register (the government
portals want a Korean ID). What it does have is OSM track named for its LEGAL line on 98% of
main and branch kilometres (`python probe_kr_ways.py`): 경부선, 경원선, 분당선, 2호선. That is
the same shape as Japan's N02 -- track labelled with the register line -- so each register line
is the graph of the OSM ways carrying its name, and OSM route relations (1호선, 경의·중앙선, the
KTX services) stay what they are in Japan: operating patterns over it.

WHICH STATIONS ARE ON A LINE comes from the published lists, read by kr_sources.py:

  Korail's 각 선구별 거리표 (every Korail legal line, stations in order, km per section) and KRIC
  1294 (every metro and light-rail line, in order, km to the neighbour). A listed station is
  found among OSM's stations by name, and the one nearest the line's own track is taken, so
  the three stations called 교대 fall to Seoul, Busan and Daegu correctly.

plus any OSM station whose stop node is a vertex of the line's own track, which is how a
station newer than the lists (the 2025 동해선 extension, 중앙선's new alignment) gets on. What is
never used is proximity to track alone: 천안아산 (KTX) is 100 m from 아산 (장항선).

SECTIONS come from the absorbing search of n02.py, with one change that double track forces. A
station must cut EVERY track through it, or the search runs past it on the track its stop node
is not on and finds a section to the station beyond: Seoul's 2호선 came out 71 km against 60.2.
So a station claims every vertex of the line within FOOT_M of it for finding neighbours, and
each section's geometry is then traced from station to station proper, so the claimed stretch
is not lost from its length.

The register's own km, where it covers every section of a line, becomes `km_official`, which
check_model compares line by line as it does Switzerland's chainage.

The `path` argument is data/raw/kr; the OSM half is read from data/proc/kr (extract.py).
"""
import hashlib
import heapq
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
INF = float("inf")

# Track names that are the same register line under another spelling. "본선" is the trunk as
# against branches and connecting curves; the curves have names of their own (익산삼각선,
# 구로기지선) and, having no stations, drop out on their own.
NAME_ALIAS = {
    "경부본선": "경부선", "호남본선": "호남선", "영동본선": "영동선", "경의본선": "경의선",
    "태백본선": "태백선", "경원본선": "경원선",
    # 동해선's Busan-Ulsan stretch (부전, 오시리아, 기장, 일광, 남창, 덕하) is tagged 동해본선.
    "동해본선": "동해선",
    "대구 2호선": "대구 도시철도 2호선",
    "광주도시철도 1호선": "광주 도시철도 1호선", "광주 1호선": "광주 도시철도 1호선",
}

# Station renames the lists have not caught up with, list spelling -> OSM spelling.
STATION_ALIAS = {
    "신경주": "경주",           # renamed 2021; OSM's 경주 node is the KTX station
    "김천구미": "김천(구미)",
}

# Track whose usage says no scheduled passenger rides it.
NOT_PASSENGER = {"industrial", "military", "test", "tourism"}

TRACK_KIND = {"rail": "rail", "subway": "subway", "light_rail": "light_rail",
              "monorail": "monorail", "tram": "tram", "funicular": "funicular",
              "narrow_gauge": "narrow_gauge", "preserved": "rail"}

STATION_RAILWAY = {"station", "halt", "tram_stop"}
# The mode a station's tags name (train=yes, subway=yes) that each kind of track carries.
WAY_MODE = {"rail": "train", "narrow_gauge": "train", "preserved": "train", "subway": "subway",
            "light_rail": "light_rail", "monorail": "monorail", "tram": "tram",
            "funicular": "funicular"}
# public_transport=station is also every bus and ferry terminal in Korea; only these make it rail.
RAIL_MODES = ("train", "subway", "light_rail", "monorail", "tram", "funicular")

STOP_TO_STATION_M = 1200   # a stop_position belongs to the station of its name this close
DUP_M = 500                # two station records of one name this close are one station
MATCH_M = 800              # a listed station's OSM node may be this far off its line's track
FOOT_M = 150               # a station cuts every track of its line within this of its anchors
STEP_M = 40                # line graphs get a vertex at least this often, so FOOT_M can cut
FAR_FOOT_M = 600           # the most a station mapped off its track reaches from its node
MERGE_M = 100              # two stations of one line closer than this are one station
PROX_M = 80                # an unlisted station joins the nearest named track this close


def line_id(name):
    h = hashlib.blake2b(f"kr|{name}".encode("utf-8"), digest_size=5)
    return "k" + h.hexdigest()


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    dy = (lat2 - lat1) * 110570
    return math.hypot(dx, dy)


def name_key(name):
    """One spelling for a station name: NFKC, no whitespace, no trailing 역.

    OSM writes Korail's Seoul Station 서울 and the metro's 서울역; the lists write 판암역 and
    광주송정역 in places, and 망 우, 제 천 and 거제\\n해맞이 with the whitespace inside.
    """
    # 경성대·부경대, 경성대ㆍ부경대 and 시청.용인대 are one spelling apart. Before NFKC, which
    # turns the ㆍ into a jamo (U+119E) that no longer looks like a dot.
    n = re.sub(r"[·ㆍ・･.ᆞ]", "", name or "")
    n = unicodedata.normalize("NFKC", n)
    n = STATION_ALIAS.get(n.strip(), n)
    n = re.sub(r"\s+", "", n)
    if len(n) > 2 and n.endswith("역"):
        n = n[:-1]
    return n


def base_key(name):
    """The name without a bracketed sub-name: 총신대입구(이수) -> 총신대입구."""
    k = name_key(name)
    b = re.sub(r"\(.*?\)", "", k)
    return b or k


def register_name(tags):
    name = (tags.get("name") or "").strip()
    return seoul_name(NAME_ALIAS.get(name, name))


def seoul_name(name):
    """Seoul's metro track is tagged plain 2호선 ... 9호선, which every city also has (OSM's own
    route relation for Busan's line 2 is called just 2호선). Every bare one in the extract is
    Seoul's, so it takes the name OSM's route relations give it, 서울 지하철 2호선, and with it
    their colour when build_model matches the two by name. The published lists are keyed on
    the track name and go through here too."""
    return f"서울 지하철 {name}" if re.fullmatch(r"[1-9]호선", name) else name


def load_osm(log):
    import build_model as bm
    ways, _rels, stops, cid, cx, cy = bm.load("kr", log)
    return ways, stops, bm.Coords(cid, cx, cy)


def load_lists(path, log):
    """The published station lists, as {line: [[name, ...], ...]}, cumulative km per station
    per list, and English names. Empty if kr_sources or its files are not there."""
    lists, cum, en = defaultdict(list), {}, {}
    try:
        import kr_sources as ks
    except ImportError:
        log("KR: kr_sources.py not found; building from OSM stations alone")
        return lists, cum, en
    # cum[(line, list i, name key)] is a LIST of km: a loop's list names its first station again
    # at the end (Seoul's 2호선 starts and ends at 시청), and the section that closes the loop
    # needs the second figure where every other needs the first.
    cum = defaultdict(list)
    dt = ks.distance_table(path)
    dt.pop(None, None)                                  # the loader's diagnostics
    for line, chains in dt.items():
        line = seoul_name(line)
        for chain in chains:
            names = [n for n, _km, junc in chain if not junc]
            lists[line].append(names)
            for n, km, junc in chain:
                if not junc:
                    cum[(line, len(lists[line]) - 1, name_key(n))].append(km)
    us = ks.urban_stations(path)
    us.pop(None, None)
    for line, chains in us.items():
        line = seoul_name(line)
        for chain in chains:
            lists[line].append([r["name"] for r in chain])
            # Membership is the whole list; chainage restarts wherever a section's km is
            # unknown (km_prev None past the first row), since nothing spans the gap.
            part, km = len(lists[line]) - 1, 0.0
            for j, r in enumerate(chain):
                if j and r.get("km_prev") is None:
                    lists[line].append([])                # an empty list: a chainage part only
                    part, km = len(lists[line]) - 1, 0.0
                elif r.get("km_prev"):
                    km += r["km_prev"]
                cum[(line, part, name_key(r["name"]))].append(km)
                if r.get("name_en"):
                    en.setdefault(name_key(r["name"]), r["name_en"])
    for k, v in ks.names_en(path).items():
        en.setdefault(name_key(k), v)
    log(f"KR: published station lists for {len(lists)} lines, "
        f"{sum(len(c) for c in lists.values())} chains; {len(en)} English names")
    return lists, cum, en


def build_stations(stops, log):
    """OSM's rail stations, one per complex, and every stop node mapped onto one."""
    st = {}
    for nid, (tags, lon, lat) in stops.items():
        rail = (tags.get("railway") in STATION_RAILWAY
                or (tags.get("public_transport") == "station"
                    and any(tags.get(m) == "yes" for m in RAIL_MODES)))
        if rail and tags.get("name"):
            st[nid] = {"name": tags["name"], "name_en": tags.get("name:en") or "",
                       "lon": lon, "lat": lat,
                       "rank": 0 if tags.get("railway") == "station" else 1}
    # One name, close together: one complex, whatever each operator mapped. Korail's 서울 and
    # the metro's 서울역 are one station to a rider, and so are 구의 and 구의(광진구청): the
    # sub-name is on one operator's record and not the other's, and keeping both put two 구의
    # on 2호선 with a 0.1 km section between them.
    by_key = defaultdict(list)
    for nid, s in st.items():
        by_key[base_key(s["name"])].append(nid)
    alias = {}
    for ids in by_key.values():
        ids.sort(key=lambda n: (st[n]["rank"], n))
        for i, a in enumerate(ids):
            if a in alias:
                continue
            for b in ids[i + 1:]:
                if b not in alias and dist_m(st[a]["lon"], st[a]["lat"],
                                             st[b]["lon"], st[b]["lat"]) <= DUP_M:
                    alias[b] = a
                    if not st[a]["name_en"]:
                        st[a]["name_en"] = st[b]["name_en"]
    for b in alias:
        del st[b]
    # A station mapped only as stop positions (Incheon's 박촌 and 임학, 부천시청 on line 7)
    # still exists: its first rail stop position becomes the station record, as build_model
    # does for a bare stop node, and the others of its name nearby are merged onto it.
    by_key = defaultdict(list)
    for nid, s in st.items():
        by_key[base_key(s["name"])].append(nid)
    made = 0
    for nid, (tags, lon, lat) in sorted(stops.items()):
        rail_stop = (tags.get("railway") == "stop"
                     or (tags.get("public_transport") == "stop_position"
                         and any(tags.get(m) == "yes" for m in RAIL_MODES)))
        if not rail_stop or not tags.get("name") or nid in st:
            continue
        k = base_key(tags["name"])
        if any(dist_m(lon, lat, st[c]["lon"], st[c]["lat"]) <= STOP_TO_STATION_M
               for c in by_key.get(k, ())):
            continue
        st[nid] = {"name": tags["name"], "name_en": tags.get("name:en") or "",
                   "lon": lon, "lat": lat, "rank": 2}
        by_key[k].append(nid)
        made += 1
    by_key = defaultdict(list)
    by_base = defaultdict(list)
    for nid, s in st.items():
        by_key[name_key(s["name"])].append(nid)
        by_base[base_key(s["name"])].append(nid)

    node_st = {}
    for nid, (tags, lon, lat) in stops.items():
        if nid in st:
            node_st[nid] = nid
        elif nid in alias:
            node_st[nid] = alias[nid]
        elif tags.get("name"):
            best, bd = None, STOP_TO_STATION_M
            for c in (by_key.get(name_key(tags["name"]))
                      or by_base.get(base_key(tags["name"]), ())):
                d = dist_m(lon, lat, st[c]["lon"], st[c]["lat"])
                if d <= bd:
                    best, bd = c, d
            if best is not None:
                node_st[nid] = best
    log(f"KR: {len(st)} OSM rail stations ({len(alias)} records merged into a complex, "
        f"{made} made from stop positions alone), {len(node_st)} stop nodes placed on one")
    return st, node_st, by_key, by_base


def line_graph(way_ids, ways, coords):
    """One line's track as a graph over OSM node ids, densified to a vertex every STEP_M.

    DENSIFIED because a station cuts the track by claiming the vertices near it, and OSM puts
    vertices only where the track bends: a straight high-speed through track can run past a
    platform with no vertex for a kilometre either side, and then the search slides past the
    station without touching it. 경부고속선 came out at 1.73 times its length that way, with
    대전 paired to 동대구 straight past 김천(구미). Inserted vertices take negative ids.
    """
    adj = defaultdict(list)
    xy = {}
    fresh = [0]
    fast = set()                 # edges (u, v), u < v, on a highspeed=yes way

    def link(a, b, high):
        (x1, y1), (x2, y2) = xy[a], xy[b]
        pts = [a]
        k = int(dist_m(x1, y1, x2, y2) // STEP_M)
        for j in range(1, k + 1):
            fresh[0] -= 1
            t = j / (k + 1)
            xy[fresh[0]] = (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
            pts.append(fresh[0])
        pts.append(b)
        for u, v in zip(pts[:-1], pts[1:]):
            w = dist_m(*xy[u], *xy[v]) / 1000
            adj[u].append((v, w))
            adj[v].append((u, w))
            if high:
                fast.add((u, v) if u < v else (v, u))

    for wid in way_ids:
        nodes = np.asarray(ways[wid][1], dtype=np.int64)
        high = ways[wid][0].get("highspeed") == "yes"
        pos, ok = coords.many(nodes)
        prev = None
        for n, p, good in zip(nodes.tolist(), pos.tolist(), ok.tolist()):
            if not good:
                prev = None
                continue
            xy[n] = (coords.x[p] / 1e7, coords.y[p] / 1e7)
            if prev is not None and prev != n:
                link(prev, n, high)
            prev = n
    return adj, xy, fast


class Near:
    """Nearest graph vertex to a point, and all vertices within a radius."""

    def __init__(self, xy):
        self.ids = np.fromiter(xy.keys(), dtype=np.int64)
        a = np.array([xy[i] for i in self.ids.tolist()], dtype=np.float64)
        self.lon, self.lat = a[:, 0], a[:, 1]

    def d(self, lon, lat):
        return np.hypot((self.lon - lon) * math.cos(math.radians(lat)) * 111320,
                        (self.lat - lat) * 110570)

    def nearest(self, lon, lat):
        d = self.d(lon, lat)
        j = int(np.argmin(d))
        return int(self.ids[j]), float(d[j])

    def within(self, lon, lat, r):
        d = self.d(lon, lat)
        k = np.nonzero(d <= r)[0]
        return self.ids[k].tolist(), d[k].tolist()


def between(adj, src, dst, blocked):
    """Shortest track from one station to another. `src` and `dst` map each footprint vertex
    to its distance (km) from its station's anchors: the search starts at every src vertex
    already charged that much and finishes at the dst vertex minimising distance plus its own
    charge. Never enters `blocked`. Returns (node path, km) or None."""
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
        for v, w in adj.get(u, ()):
            if v in blocked:
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


def neighbours(adj, foot):
    """Station pairs with no third station between them: a search from each station's
    footprint that absorbs at the first vertex of another's."""
    by_st = defaultdict(set)
    for n, s in foot.items():
        by_st[s].add(n)
    pairs = set()
    for sid, own in by_st.items():
        seen = set(own)
        frontier = list(own)
        # Plain flood fill is enough to find neighbours: which station is met first along each
        # branch does not depend on distance, only on topology.
        while frontier:
            u = frontier.pop()
            for v, _w in adj.get(u, ()):
                if v in seen:
                    continue
                seen.add(v)
                other = foot.get(v)
                if other is not None and other != sid:
                    pairs.add((sid, other) if sid <= other else (other, sid))
                    continue
                frontier.append(v)
    return pairs


def build(path, log):
    ways, stops, coords = load_osm(log)
    st, node_st, by_key, by_base = build_stations(stops, log)
    lists, cum, en = load_lists(path, log)

    by_line = defaultdict(list)
    for wid, (tags, nodes) in ways.items():
        if tags.get("railway") not in TRACK_KIND or tags.get("usage") in NOT_PASSENGER:
            continue
        name = register_name(tags)
        if name:
            by_line[name].append(wid)
    for line in sorted(set(lists) - set(by_line)):
        log(f"  KR: published list for {line}, but no OSM track carries that name")
    cum_line = defaultdict(lambda: defaultdict(dict))  # line -> name key -> {cum key: [km]}
    for key, kms in cum.items():
        cum_line[key[0]][key[2]][key] = kms

    graphs = {}
    for name, wids in sorted(by_line.items()):
        adj, xy, fast = line_graph(wids, ways, coords)
        if len(xy) >= 2:
            graphs[name] = (adj, xy, fast, Near(xy))

    # --- stations newer than every list. 동해선's 2025 extension (영덕-삼척) is in no
    # published list and its OSM stations are nodes beside the track, not on it, so nothing
    # put them on a line and the whole stretch dropped out. A station NO list names and with
    # no stop node on any named track joins the line whose named track is nearest to it,
    # if that is within PROX_M. Only those orphans: a listed city station can sit right over
    # another line's tunnel, and nearest-track would put it on a line that does not stop there.
    # And only a station OSM says is served, by the mode the line is: closed stations are often
    # still railway=station (지천 has no mode; 제진 is train=no; 팔당's old station has no
    # railway tag at all), and the Wolmi Sea Train's station is nearest to 경인선's track.
    listed_keys = {f(n) for chains in lists.values() for ch in chains for n in ch
                   for f in (name_key, base_key)}
    on_track = {node_st[n] for g in graphs.values() for n in g[0] if n in node_st}
    line_mode = {}
    for name, wids in by_line.items():
        c = Counter(WAY_MODE.get(ways[w][0]["railway"], "train") for w in wids)
        line_mode[name] = c.most_common(1)[0][0]
    prox = {}                                   # station -> (line, vertex)
    for sid, s in st.items():
        if (sid in on_track or name_key(s["name"]) in listed_keys
                or base_key(s["name"]) in listed_keys):
            continue
        tags = stops[sid][0] if sid in stops else {}
        modes = {m for m in RAIL_MODES if tags.get(m) == "yes"}
        if tags.get("railway") not in ("station", "halt") or not modes:
            continue
        best = (PROX_M, None, None)
        for name, (_a, _x, _f, nr) in graphs.items():
            if line_mode.get(name) not in modes:
                continue
            if (s["lon"] < nr.lon.min() - 0.01 or s["lon"] > nr.lon.max() + 0.01
                    or s["lat"] < nr.lat.min() - 0.01 or s["lat"] > nr.lat.max() + 0.01):
                continue
            v, d = nr.nearest(s["lon"], s["lat"])
            if d <= best[0]:
                best = (d, name, v)
        if best[1] is not None:
            prox[sid] = (best[1], best[2])

    stations, lines, geoms = {}, [], {}
    dropped, unmatched = [], []
    n_listed = n_vertex = n_prox = 0
    prox_names = defaultdict(list)
    official_lines = 0
    for name, wids in sorted(by_line.items()):
        if name not in graphs:
            continue
        adj, xy, fast, near = graphs[name]

        # --- which stations, and where on this line's track each one is (its anchors)
        anchors = defaultdict(set)
        for n in adj:
            if n in node_st:
                anchors[node_st[n]].add(n)
        n_vertex += len(anchors)
        where = {}                                        # (list i, name key) -> station
        far = {}                                          # station -> its node, off the track
        for i, chain in enumerate(lists.get(name, ())):
            for nm in chain:
                k = name_key(nm)
                cands = by_key.get(k) or by_key.get(base_key(nm)) or by_base.get(base_key(nm), ())
                best, bd, bv = None, MATCH_M, None
                for c in cands:
                    v, d = near.nearest(st[c]["lon"], st[c]["lat"])
                    if d <= bd:
                        best, bd, bv = c, d, v
                if best is None:
                    unmatched.append((name, nm))
                    continue
                where[(i, k)] = best
                if best not in anchors:
                    anchors[best].add(bv)
                    n_listed += 1
                if bd > FOOT_M:
                    far[best] = (st[best]["lon"], st[best]["lat"], bd)
        for sid, (ln, v) in prox.items():
            if ln == name and sid not in anchors:
                anchors[sid].add(v)
                n_prox += 1
                prox_names[name].append(st[sid]["name"])

        # --- one station per place. Two records of one station can survive under names too
        # different to merge on (Incheon's 서구청 and 서해구청, 0.08 km apart), and would make a
        # section of nothing between them. Within MERGE_M on one line they are one; the one
        # the published list named is kept.
        pos = {s: tuple(np.mean([xy[a] for a in ans], axis=0)) for s, ans in anchors.items()}
        listed = set(where.values())
        order = sorted(anchors, key=lambda s: (s not in listed, st[s]["rank"], s))
        merged = {}
        for i, s in enumerate(order):
            if s in merged:
                continue
            for t in order[i + 1:]:
                if t not in merged and dist_m(*pos[s], *pos[t]) <= MERGE_M:
                    merged[t] = s
                    anchors[s] |= anchors.pop(t)
                    if t in far and s not in far:
                        far[s] = far[t]
        where = {k: merged.get(s, s) for k, s in where.items()}

        if len(anchors) < 2:
            dropped.append(name)
            continue

        # --- footprints: every vertex of this line near a station belongs to it, nearest wins.
        # A station whose node is mapped well off this line's track (a big Korail station's
        # node sits in its concourse) reaches from its node as far as the track plus FOOT_M,
        # or its anchor can land on a platform road and miss the through tracks.
        foot, foot_d = {}, {}
        for sid, ans in anchors.items():
            disks = [(*xy[a], FOOT_M) for a in ans]
            if sid in far:
                lon, lat, d = far[sid]
                disks.append((lon, lat, min(d + FOOT_M, FAR_FOOT_M)))
            for x, y, r in disks:
                ids, ds = near.within(x, y, r)
                for v, d in zip(ids, ds):
                    if d < foot_d.get(v, INF):
                        foot[v], foot_d[v] = sid, d
            for a in ans:                                 # an anchor is always its own
                foot[a], foot_d[a] = sid, 0.0
        pairs = neighbours(adj, foot)
        if not pairs:
            dropped.append(name)
            continue

        # --- each section traced from anywhere in one station's footprint to anywhere in the
        # other's, never through a third station. Each end is charged its distance from the
        # station's anchors, so the section still measures station to station. Starting from
        # the anchors alone sent a train the wrong way out of 군북 on a platform road and back
        # again, 33 km for a 10 km section.
        foot_of = defaultdict(set)
        for v, s in foot.items():
            foot_of[s].add(v)

        # Charged from the station's CENTRE, the mean of its anchors on this line, which is what
        # the register measures between. Charging from the nearest anchor instead let a
        # section start at the far end of the platform: a metro has a stop position at each
        # end of it, and 서울 지하철 1호선 came out at 0.77 of its length.
        centre = {s: tuple(np.mean([xy[a] for a in ans], axis=0)) for s, ans in anchors.items()}

        def offsets(s):
            c = centre[s]
            return {v: dist_m(*xy[v], *c) / 1000 for v in foot_of[s]}

        sections = {}
        for a, b in sorted(pairs):
            blocked = {v for v, s in foot.items() if s != a and s != b}
            got = between(adj, offsets(a), offsets(b), blocked)
            if got is None:
                continue
            nodes, km = got
            # The inserted vertices lie on straight OSM edges, so only OSM's own are drawn.
            keep = [n for i, n in enumerate(nodes) if n > 0 or i == 0 or i == len(nodes) - 1]
            # Drawn from centre to centre, as it is measured. The path itself starts and ends
            # somewhere in each station's footprint, up to FOOT_M off, and drawn as it was a
            # ridden run showed a gap at every station, like a dashed line.
            on_fast = sum(dist_m(*xy[u], *xy[v]) for u, v in zip(nodes[:-1], nodes[1:])
                          if ((u, v) if u < v else (v, u)) in fast)
            sections[(f"k{a}", f"k{b}")] = {
                "km": km, "geom": [centre[a]] + [xy[n] for n in keep] + [centre[b]],
                "fast": on_fast / 1000 >= 0.5 * km if km else False}
        if not sections:
            dropped.append(name)
            continue

        # --- the register's own km for each section, from the published cumulative km. Where
        # a station has two figures on one list (a loop's first station again at its end), the
        # pair closest together is the section; any other pairing spans the whole loop.
        at_km = defaultdict(dict)                          # station -> {part i: [km, ...]}
        for (_i, k), sid in where.items():
            for (ln, i, kk), kms in cum_line.get(name, {}).get(k, {}).items():
                at_km[sid][i] = kms
        chain = {}
        for (sa, sb) in sections:
            a, b = int(sa[1:]), int(sb[1:])
            best = None
            for i in set(at_km.get(a, {})) & set(at_km.get(b, {})):
                d = min(abs(x - y) for x in at_km[a][i] for y in at_km[b][i])
                best = d if best is None else min(best, d)
            if best is not None:
                chain[f"{sa}|{sb}"] = best
        official = len(chain) == len(sections)

        lid = line_id(name)
        _AT_KM[lid] = {f"k{s}": parts for s, parts in at_km.items()}
        kinds, ops = Counter(), Counter()
        for wid in wids:
            t = ways[wid][0]
            kinds[TRACK_KIND[t["railway"]]] += 1
            if t.get("operator"):
                ops[t["operator"]] += 1
        for sid in {s for k in sections for s in k}:
            nid = int(sid[1:])
            if sid not in stations:
                s = st[nid]
                stations[sid] = {"id": sid, "name": s["name"],
                                 "name_en": en.get(name_key(s["name"])) or s["name_en"],
                                 "lon": s["lon"], "lat": s["lat"], "lines": set()}
            stations[sid]["lines"].add(lid)
        from n02 import walk_order
        line = {
            "id": lid, "src": "kr", "service": False,
            "name": name, "name_en": "", "ref": "", "colour": "",
            "operator": ops.most_common(1)[0][0] if ops else "",
            "operator_en": "", "network": "",
            "kind": kinds.most_common(1)[0][0],
            # Per section, from the highspeed=yes ways each lies on, not per line: 중앙선,
            # 경강선 and 서해선 are partly new 250 km/h alignments, and a line-wide flag either
            # stopped KTX-이음 rides crediting them or let KTX credit 경부선 beside
            # 경부고속선. No line-wide `highspeed`, so the OSM ways match on their name tag.
            "highspeed_sections": {f"{a}|{b}": v["fast"] for (a, b), v in sections.items()},
            "km": round(sum(v["km"] for v in sections.values()), 3),
            "variants": 1, "straight_sections": 0,
            "display": walk_order(sections.keys()),
            "sections": [[a, b, round(v["km"], 3)] for (a, b), v in sections.items()],
        }
        if official:
            official_lines += 1
            line["km_official"] = round(sum(chain.values()), 3)
        lines.append(line)
        geoms[lid] = {f"{a}|{b}": [[round(x, 5), round(y, 5)] for x, y in v["geom"]]
                      for (a, b), v in sections.items()}

    total = sum(l["km"] for l in lines)
    log(f"KR: {len(lines)} register lines, {total:,.0f} km, {len(stations)} stations; "
        f"{official_lines} lines have a published km for every section")
    log(f"KR: stations placed by a stop node on the line's track {n_vertex}, "
        f"by the published list alone {n_listed}, as an unlisted station nearest a line's "
        f"track {n_prox}; {len(dropped)} named tracks had under two")
    for line, nms in sorted(prox_names.items()):
        log(f"    unlisted, by nearest track: {line}: {' '.join(nms)}")
    if unmatched:
        log(f"KR: {len(unmatched)} listed stations found no OSM station within {MATCH_M} m "
            f"of their line's track:")
        by = defaultdict(list)
        for line, nm in unmatched:
            by[line].append(nm)
        for line, nms in sorted(by.items(), key=lambda kv: -len(kv[1])):
            log(f"    {line}: {' '.join(nms[:25])}{' ...' if len(nms) > 25 else ''}")
    # where each station is on each line's track, for split_pieces after build_model's drops
    _ENDS[:] = [(sid, l["name"], *pts[0 if k == 0 else -1])
                for l in lines for key, pts in geoms[l["id"]].items()
                for k, sid in enumerate(key.split("|"))]
    return lines, stations, geoms


# ---------------------------------------------------------------- lines in pieces

# Korea's lines in pieces (kr_sources.md "Lines in pieces"), through pieces.py as gb_register:
# a gap is bridged over the track between the pieces where trains run across, the rest is one
# line per piece. Korea's own settings: the line's own named track costs half (OWN_COST) and
# counts as under a route, since OSM's KTX routes lie on the conventional line beside the
# high-speed one (호남고속선's 익산 - 정읍, where the cheapest routed track would otherwise be
# 호남선's). `dense` (pieces.Rules): a station on straight track with no OSM vertex near it
# still joins the track graph. KEEP_WHOLE: a line whose gap is track OSM does not have, on a
# line trains run through, stays one line in pieces until the track is mapped (none in Korea
# as of the 2026-10 extract; kr_sources.md has the five lines measured).
OWN_COST = 0.5
KEEP_WHOLE = set()
# {line id: [the ids of the pieces split off it]}, filled by split_pieces; build_model writes it
# into aliases.json as `pieces`.
LINE_PIECES = {}
_ENDS = []       # (station, line name, lon, lat) for every section end build() made
_AT_KM = {}      # line id -> station -> {published list part: [cumulative km]}, from build()


def official_km(lid, a, b):
    """The published km between two stations of a line (build()'s chain rule), or None."""
    at = _AT_KM.get(lid, {})
    if a not in at or b not in at:
        return None
    parts = set(at[a]) & set(at[b])
    if not parts:
        return None
    return min(abs(x - y) for i in parts for x in at[a][i] for y in at[b][i])


def rules():
    import pieces
    return pieces.Rules(tag="KR", id_prefix="k", lat=36.5, own_cost=OWN_COST,
                        keep_whole=KEEP_WHOLE, piece_name=pieces.english_piece_name,
                        dense=True)


def classify(wid, tags, routed):
    """For pieces.track_graph: every way a line name is on, and every other way under an OSM
    passenger route. The name ("" for none), or None for a way left out."""
    if tags.get("railway") not in TRACK_KIND or tags.get("usage") in NOT_PASSENGER:
        return None
    nm = register_name(tags)
    return nm if nm or routed else None


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """The build_model hook: pieces.split_pieces with Korea's rules, over the ways
    register_way_lines loaded (state) and the extract's route relations."""
    import pickle
    import build_model as bm
    import pieces
    r = rules()

    def graph():
        if not state or "ways" not in state:
            return None
        with open(ROOT / "data" / "proc" / "kr" / "rels.pkl", "rb") as f:
            rels = pickle.load(f)
        return pieces.track_graph(state["ways"], bm.Coords(state["cid"], state["cx"], state["cy"]),
                                  pieces.routed_ways(rels), classify, r, log)
    before = {l["id"]: {(s[0], s[1]) for s in l["sections"]} for l in lines
              if l.get("src", "osm") != "osm"}
    pieces.split_pieces(lines, stations, geoms, reg_ways, state, log, r, LINE_PIECES, graph,
                        _ENDS)
    # km_official where the sections changed (a bridge added some, or the line was split): the
    # published km of every section if the lists give them all, else none
    origin = {p: lid for lid, ps in LINE_PIECES.items() for p in ps}
    for l in lines:
        if l.get("src", "osm") == "osm":
            continue
        secs = {(s[0], s[1]) for s in l["sections"]}
        if before.get(l["id"]) == secs:
            continue
        got = [official_km(origin.get(l["id"], l["id"]), a, b) for a, b in secs]
        if got and all(k is not None for k in got):
            l["km_official"] = round(sum(got), 3)
            log(f"KR: {l['name']} ({l['id']}): km_official {l['km_official']} for its new "
                f"sections")
        elif l.pop("km_official", None) is not None:
            log(f"KR: {l['name']} ({l['id']}): km_official dropped, the lists give no km for "
                f"some of its new sections")
