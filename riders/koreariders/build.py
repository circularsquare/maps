# -*- coding: utf-8 -*-
"""Reconstruct per-segment passenger load for every intercity line, and write
data/segments.geojson.

The method and its validation are in README.md; prototype.py is the single-line
version with the workings printed. This is the same thing over the whole
network, driven by lines.py.

    python build.py                 # all lines, report + geojson
    python build.py --line 전라선    # one line, with its station table
"""

import argparse
import collections
import heapq
import io
import json
import math
import os
import sys

import lines as LN
import membership as M

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
RAILWAYS = os.path.join(D, "osm_railways.json")
# The junction throats, which carry no name and so are absent from the named
# pull. See `load_network` and fetch_osm.py's UNNAMED.
UNNAMED = os.path.join(D, "osm_rail_unnamed.json")
STATIONS = os.path.join(D, "osm_stations.json")
# Its own file. Both this and solve.py used to write segments.geojson, so
# whichever ran last silently decided what the map drew -- and only solve.py
# emits geometry, so a build.py run left the page with nothing to draw.
OUT = os.path.join(D, "segments_singleline.geojson")

SNAP_KM = 0.30          # how close a station must be to count as on the line
PENALTY = 40.0          # cost multiplier for track belonging to another line
# Decimal places a node is rounded to before two of them count as one place.
# Loosening this to 5 (about a metre) was tried against 영동선's broken corridor
# and changed nothing, so the break there is a real gap in the data rather than
# coordinates that miss each other.
JOIN_DP = 6
# How much further from its end stations a single run of the line's own track
# may sit before it is better to use every piece and let the penalty bridge the
# gaps. 영동선 gives up 0.47 km and wins its route; 호남고속선 would give up 90.
SPLIT_TOL_KM = 2.0


# ---------------------------------------------------------------- geometry


def haversine(a, b):
    R = 6371.0088
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def load_network(unnamed=True):
    """One graph for the whole country: {node: [(node, km, line_name), ...]}.

    Built once and reused for every line, with the per-line cost applied at
    search time -- rebuilding it per line was most of the runtime.

    The unnamed ways go in with `nm` None, which is not in any line's target
    set, so they cost the full foreign-track PENALTY and nothing routes over
    them that has an alternative. What they buy is connectivity: without them a
    junction curve is simply missing and a corridor that has to cross one goes
    the long way round or gives up. `unnamed=False` is there to measure that.
    """
    with io.open(RAILWAYS, encoding="utf-8") as f:
        ways = json.load(f)["elements"]
    if unnamed and os.path.exists(UNNAMED):
        with io.open(UNNAMED, encoding="utf-8") as f:
            ways = ways + json.load(f)["elements"]
    g = collections.defaultdict(list)
    for w in ways:
        nm = (w.get("tags") or {}).get("name")
        pts = [(p["lat"], p["lon"]) for p in w.get("geometry", [])]
        for a, b in zip(pts, pts[1:]):
            ka = (round(a[0], JOIN_DP), round(a[1], JOIN_DP))
            kb = (round(b[0], JOIN_DP), round(b[1], JOIN_DP))
            if ka == kb:
                continue
            d = haversine(a, b)
            g[ka].append((kb, d, nm))
            g[kb].append((ka, d, nm))
    return g


NODE_CELL = 0.02        # degrees, about 2 km -- grid cell for nearest_node
_node_grid = {}         # id(g) -> {cell: [node, ...]}, built once per graph


def nearest_node(g, pt):
    """The graph node closest to a point, over the whole network.

    A linear scan is 500k haversines per call and there are two per line, which
    took build.py from seconds to minutes. Bucketing the nodes into 2 km cells
    and widening the ring until something is found makes it a few dozen.
    """
    grid = _node_grid.get(id(g))
    if grid is None:
        grid = collections.defaultdict(list)
        for n in g:
            grid[(int(math.floor(n[0] / NODE_CELL)),
                  int(math.floor(n[1] / NODE_CELL)))].append(n)
        _node_grid[id(g)] = grid
    ci = int(math.floor(pt[0] / NODE_CELL))
    cj = int(math.floor(pt[1] / NODE_CELL))

    def ring(r):
        return [n for i in range(ci - r, ci + r + 1)
                for j in range(cj - r, cj + r + 1)
                for n in grid.get((i, j), ())]

    for r in range(1, 60):
        if ring(r):
            # One ring wider than the first hit: a node just outside a ring can
            # beat one sitting in its corner.
            cand = ring(r + 1)
            return min(cand, key=lambda n: haversine(n, pt))
    return min(g, key=lambda n: haversine(n, pt))


STUB_MIN_KM = 1.0       # closer than this and the corridor already reaches
STUB_MAX_KM = 30.0      # 경부고속선's 서울 stub is 22; nothing legitimate is more
STUB_MAX_RATIO = 3.0    # track km per straight km, so a stub cannot wander

# Two adjacent tracks in a depot are two nodes a few metres apart with no edge
# between them, so a route that has to change from one to the other runs to
# wherever they do join and comes back. 중부내륙선 is the case: its drawn end and
# the track that reaches 부발 are 17 m apart and only meet 1.55 km south, so the
# shortest route to a station 1.63 km away is 5.01 km, of which 3.19 is spent
# going south and returning. `reach_station` refused it at a ratio of 3.07
# against a bound of 3.00, which was the right call about the wrong quantity --
# the route is not wandering, it is doubling back.
#
# So an excursion that returns to within STUB_JOIN_KM of somewhere it has
# already been is spliced out before the route is measured. The two kinds of
# return separate cleanly and are not near each other, which is why a plain
# threshold does this safely -- measured over every line's stub:
#
#     doubling back    3.19 km over 17 m, 3.81 over 28, 20.97 over 21,
#                      103.89 over 18      -- ratios 136 to 5,772
#     double track     0.02 km over 24 m, 0.03 over 28  -- ratios about 1
#
# The second kind is two parallel tracks weaving, where the route "returns"
# having gone no further than the gap itself. Cutting those would buy nothing
# and would litter the drawn line with metre-scale jumps, so the rule takes
# both a floor and a ratio and the band between 1 and 136 is empty.
STUB_JOIN_KM = 0.030    # two tracks this close are one place on this map
STUB_LOOP_MIN_KM = 0.2  # shorter than this is weaving, not an excursion
STUB_LOOP_RATIO = 10.0  # ... and so is anything not much longer than its gap


# A degree of latitude is 111 km and a degree of longitude at least 87 across
# Korea, so nothing outside this box can be within STUB_JOIN_KM. The scan below
# is quadratic in the length of the route and a stub search over the national
# graph can return thousands of nodes, so almost every pair has to be rejected
# without a haversine: with the box it costs two subtractions instead.
STUB_JOIN_DEG = STUB_JOIN_KM / 87.0


def deloop(path):
    """Splice out excursions -- track the route covers only to come back."""
    out, i, n = [], 0, len(path)
    while i < n:
        out.append(path[i])
        cut = i
        for k in range(n - 1, i, -1):
            if (abs(path[i][0] - path[k][0]) > STUB_JOIN_DEG
                    or abs(path[i][1] - path[k][1]) > STUB_JOIN_DEG):
                continue
            gap = haversine(path[i], path[k])
            if gap > STUB_JOIN_KM:
                continue
            loop = sum(haversine(a, b)
                       for a, b in zip(path[i:k], path[i + 1:k + 1]))
            if loop >= STUB_LOOP_MIN_KM and loop >= STUB_LOOP_RATIO * gap:
                cut = k
            break
        # Resume *at* the returning node, not past it. It is the last point the
        # excursion shares with the route, so keeping it leaves one joint of at
        # most STUB_JOIN_KM and every later point exactly on the real track;
        # skipping it would add that node's own edge to the joint as well.
        i = cut if cut > i else i + 1
    return out


def reach_station(g, start, pt):
    """Track from a corridor's end to a station that sits off the line.

    Over any rails, not just the line's own -- the whole point is that the
    line's trains run on someone else's here. Bounded twice: an absolute length,
    since no Korean line reaches more than 22 km onto a neighbour, and a
    directness ratio, since a stub that triples the straight-line distance has
    gone somewhere other than the station. Either bound failing returns nothing
    and the caller leaves the end alone.

    Both bounds are applied to the de-looped route, since an excursion is track
    the trains do not cover and measuring it as if they did is what made the
    directness bound refuse routes that are perfectly direct. See `deloop`.
    """
    dst = nearest_node(g, pt)
    if dst == start:
        return None
    straight = haversine(start, pt)
    if straight > STUB_MAX_KM:
        return None
    path, _ = shortest_path(g, start, dst, set())
    if not path or len(path) < 2:
        return None
    path = deloop(path)
    if len(path) < 2:
        return None
    km = sum(haversine(a, b) for a, b in zip(path, path[1:]))
    if km > STUB_MAX_KM or km > STUB_MAX_RATIO * max(straight, 0.1):
        return None
    return path


def own_component(g, target, ends):
    """The run of the line's own track that best reaches both its end stations.

    A line's metals do not always arrive in one piece. OSM has 영동선 as a
    2,687-node run from 봉화 to 강릉 plus a stranded 0.98 km stub at 영주 that
    comes within 182 m of it and never touches -- the station throat between
    them is mapped under another name. Picking the end station's nearest node
    off *any* of the line's track lands on that stub, and the search then cannot
    reach the rest of the line along its own metals at all: 영동선 was drawn
    150 km round by 중앙선 and 태백선 through 영월, over 태백선's own route,
    which is why eight of its stations had no track near them to sit on.

    But most split lines are split for a duller reason and want the old
    behaviour: 호남고속선's track really does arrive in two halves, and letting
    the search bridge them at the foreign-track penalty draws it correctly at
    183.8 km. Restricting it to either half lost the other and drew 91 km.

    So: take the run that comes nearest both end stations, but only if it is
    barely worse than using every piece. 영동선 gives up 0.47 km by anchoring
    half a kilometre from 영주 instead of on the stub, and gets its own route in
    exchange; 호남고속선 would give up ninety, so it keeps everything and the
    penalty bridges the gap.
    """
    adj = collections.defaultdict(list)
    for u in g:
        for v, _, nm in g[u]:
            if nm in target:
                adj[u].append(v)
    seen, comps = set(), []
    for s in adj:
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    def reach(nodes):
        return sum(min(haversine(n, pt) for n in nodes for pt in pts)
                   for pts in ends)

    every = [n for c in comps for n in c]
    if not every:
        return []
    best, score = [], None
    for comp in comps:
        s = reach(comp)
        if score is None or s < score:
            best, score = comp, s
    return best if score <= reach(every) + SPLIT_TOL_KM else every


def shortest_path(g, src, dst, target):
    """Cheapest route from src to dst, preferring track named in `target`."""
    dist, prev, seen = {src: 0.0}, {}, set()
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            break
        for v, real, nm in g.get(u, ()):
            nd = d + real * (1.0 if nm in target else PENALTY)
            if nd < dist.get(v, 1e18):
                dist[v] = nd
                prev[v] = (u, nm in target)
                heapq.heappush(pq, (nd, v))
    if dst not in dist:
        return None, 0.0
    path, cur, foreign = [dst], dst, 0.0
    while cur != src:
        cur, mine = prev[cur]
        if not mine:
            foreign += 1
        path.append(cur)
    return path[::-1], foreign


def chainage(line):
    out = [0.0]
    for i in range(len(line) - 1):
        out.append(out[-1] + haversine(line[i], line[i + 1]))
    return out


def project(pt, line, cum):
    """Nearest point on the polyline: (distance_km, chainage_km)."""
    best = (1e9, 0.0)
    for i in range(len(line) - 1):
        a, b = line[i], line[i + 1]
        kx = math.cos(math.radians(a[0]))
        ax, ay, bx, by = a[1] * kx, a[0], b[1] * kx, b[0]
        px, py = pt[1] * kx, pt[0]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0,
                                             ((px - ax) * dx + (py - ay) * dy) / L2))
        d = math.hypot(px - (ax + t * dx), py - (ay + t * dy)) * 111.32
        if d < best[0]:
            best = (d, cum[i] + t * (cum[i + 1] - cum[i]))
    return best


# ---------------------------------------------------------------- per line


def point_at(poly, cum, km):
    """The (lat, lon) at a given chainage along the corridor."""
    if km <= cum[0]:
        return poly[0]
    if km >= cum[-1]:
        return poly[-1]
    lo, hi = 0, len(cum) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cum[mid] <= km:
            lo = mid
        else:
            hi = mid
    span = cum[hi] - cum[lo]
    t = 0.0 if span <= 0 else (km - cum[lo]) / span
    a, b = poly[lo], poly[hi]
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def slice_corridor(poly, cum, a, b):
    """The corridor between two chainages, ends interpolated, running a -> b."""
    lo, hi = (a, b) if a <= b else (b, a)
    pts = [point_at(poly, cum, lo)]
    pts.extend(poly[i] for i in range(len(poly)) if lo < cum[i] < hi)
    pts.append(point_at(poly, cum, hi))
    if a > b:
        pts.reverse()
    out = []
    for p in pts:                       # drop repeats, which break some renderers
        if not out or abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return out


def graft(poly, pts, at):
    """Join a run of points onto one end of a corridor.

    `pts` runs outward from the corridor's own end towards the station, which is
    the orientation both `reach_station` and a host-corridor slice produce. Onto
    the head it goes reversed; onto the tail, as it stands. The shared joint is
    dropped where the two ends coincide, which they do by construction.
    """
    if not pts or len(pts) < 2:
        return poly
    near = haversine(pts[0], poly[at]) < 0.05
    if at == 0:
        return list(pts)[::-1] + (list(poly)[1:] if near else list(poly))
    return (list(poly)[:-1] if near else list(poly)) + list(pts)


def order_stations(spec, g, named, flows, corridor=None, over=None,
                   beyond=None):
    """The line's stations, in running order, with km along the corridor.

    Pass a dict as `corridor` to get the shape back too: the polyline, its
    chainage, and where each stop sits on it before the rescale to 영업거리.
    That is what turns the segment table into something drawable.
    """
    target = set(spec["ways"])
    for end in ("first", "last"):
        if spec[end] not in named:
            return None, "no OSM station node named %s" % spec[end]

    # Anchor the search on the line's own track, so it cannot set off down a
    # crossing line at a junction station -- and on the longest connected run of
    # it, so a stranded fragment cannot strand the search with it.
    own = own_component(g, target,
                        [named[spec["first"]], named[spec["last"]]])
    if not own:
        return None, "no OSM track named %s" % "/".join(spec["ways"])

    # A name can carry several nodes -- 동대구 and 수서 have one per operator, and
    # they are not all on this line. Take whichever sits closest to this line's
    # own track.
    # One pass over own x pts gives both answers: which node of that name is
    # this line's, and which bit of the line's track it sits on. Asking for them
    # separately makes it |own| squared, which on a trunk line is minutes.
    def closest(nm):
        """(station point on this line, own-track node nearest it)."""
        _, pt, n = min((haversine(n, pt), pt, n)
                       for pt in named[nm] for n in own)
        return pt, n

    first_pt, first_n = closest(spec["first"])
    last_pt, last_n = closest(spec["last"])
    poly, _ = shortest_path(g, first_n, last_n, target)
    if poly is None:
        return None, "track not connected between %s and %s" % (spec["first"],
                                                                spec["last"])

    # The corridor now runs end to end over the line's own metals, and for five
    # lines that stops short of the end station -- because the station is on
    # someone else's rails and the line's trains reach it over them.
    #
    # 경부고속선 is the case that matters: its high-speed track begins near
    # 금천구청 and KTX reach 서울 over 경부선's metals, so the corridor ended
    # 14.29 km from 서울역 and the station was drawn there. That gave 서울-광명 a
    # length of 2.9 km where the two stations are 22 km apart, and 79,545 riders
    # a day were credited with an eighth of the distance they travel. Four more
    # are smaller: 중앙선's 경주 by 3.20 km, 영동선's 강릉 by 3.06, 광주선's
    # 광주송정 by 1.77, and 중부내륙선's 부발 by 1.63 -- which is exactly the
    # 1.9 km 중부내륙선 has been drawn short by.
    #
    # Anchoring the *search* at the station instead was tried and is much worse:
    # the nearest graph node to a station can sit on a siding that does not
    # connect locally, so the route goes the long way round. 광주선 came out at
    # 414.6 km for a 12.2 km railway, 호남선 stopped connecting at all, and
    # 중부내륙선 collected 78 stations it does not call at. The 40x penalty is
    # not a leash when the search starts outside the fence.
    #
    # So the corridor keeps its own-track spine and is *extended* at each end,
    # over any track, by a stub bounded in both length and directness. A stub
    # that cannot be found or comes out too long is skipped and the end stays
    # where it was.
    for nm, pt, at in ((spec["first"], first_pt, 0),
                       (spec["last"], last_pt, -1)):
        if haversine(poly[at], pt) < STUB_MIN_KM:
            continue
        # A named host line wins: lines.OVER says which line's metals carry the
        # trains here, and the caller has already sliced that line's corridor.
        # The search is the fallback.
        #
        # An end `over` mentions at all is the host's, even before the slice
        # exists -- a None there means "reserved, leave it short". The caller
        # builds every corridor once and only then has a host to slice, so on
        # the first pass 경부고속선's 서울 has no stub yet; letting the search
        # take it then leaves the corridor already at 서울, and the slice from
        # there to 서울 is 0.0 km. That is not hypothetical: it is what happened
        # when de-looping made the search able to reach 서울 at all, and it cost
        # the line its 18.7 km of 경부선 metals and the network eight junctions.
        if over is not None and nm in over:
            stub = over[nm]
        else:
            stub = reach_station(g, poly[at], pt)
        if not stub:
            continue
        poly = graft(poly, stub, at)

    # And where the line runs on past its last platform to a junction, draw it
    # there. 수서고속선's SRT do not stop at 평택지제 and turn round; they carry on
    # 7.5 km to 평택분기점 and onto 경부고속선. Routing between the two end
    # *stations* left that 7.5 km undrawn and the line ending in mid-air, which
    # is also the whole of its 12.9 % gap against its 영업거리. Along the line's
    # own metals, so this cannot wander: `target` still carries the 40x penalty.
    ran_on = set()
    for nm, at in ((spec["first"], 0), (spec["last"], -1)):
        jpt = (beyond or {}).get(nm)
        if jpt is None:
            continue
        dst = min(own, key=lambda n: haversine(n, jpt))
        if haversine(dst, jpt) > 1.0:
            continue
        path, _ = shortest_path(g, poly[at], dst, target)
        if not path or len(path) < 2:
            continue
        poly = graft(poly, path, at)
        ran_on.add(nm)

    cum = chainage(poly)

    snapped = {}
    for nm, pts in named.items():
        for pt in pts:
            d, km = project(pt, poly, cum)
            if d <= SNAP_KM and (nm not in snapped or d < snapped[nm][0]):
                snapped[nm] = (d, km)

    # The corridor was routed *between* the two end stations, so they sit at its
    # ends by construction even when the nearest named track stops short of the
    # platform -- 경부고속선's mapped track begins south of 광명, 광주선's at the
    # 동송정 junction. Place them rather than requiring them to snap.
    #
    # Unless the corridor was run on past one of them to a junction, in which
    # case that end is no longer the station and pinning it there would put
    # 평택지제 at 평택분기점, 7.5 km from its platform. Project instead.
    for nm, at in ((spec["first"], 0), (spec["last"], -1)):
        if nm in ran_on:
            pt = first_pt if at == 0 else last_pt
            snapped[nm] = project(pt, poly, cum)
        else:
            snapped[nm] = (0.0, cum[at])

    # Distance alone grabs whatever sits near the track -- 천안아산, a 경부고속선
    # station 100 m from 아산, would put its KTX arrivals into 장항선. A station
    # reporting traffic whose home line is another one is dropped, except at the
    # line's own two ends.
    ends = {spec["first"], spec["last"]}
    roster = spec["roster"]
    keep = [(km, nm) for nm, (d, km) in snapped.items()
            if nm in ends or not (nm in flows and roster and nm not in roster)]
    keep.sort()
    if len(keep) < 3:
        return None, "only %d stations snapped" % len(keep)

    k0 = dict((nm, km) for km, nm in keep)[spec["first"]]
    k1 = dict((nm, km) for km, nm in keep)[spec["last"]]
    if k0 > k1:
        keep = [(k0 - km, nm) for km, nm in keep][::-1]
    else:
        keep = [(km - k0, nm) for km, nm in keep]
    keep = [(km, nm) for km, nm in keep if -0.2 <= km <= abs(k1 - k0) + 0.2]

    # OSM chainage runs a few tenths of a per cent short of the published
    # 영업거리, so rescale to it -- but only where the two measure the same
    # track. `scaled` is false wherever lines.ENDS moved an end, because the
    # published figure then covers an extent this build does not draw and the
    # rescale turns a 0.3 % correction into a 24 % error. See lines.resolve().
    scale = 1.0
    if spec.get("scaled", True) and keep[-1][0] > 0:
        scale = spec["length_km"] / keep[-1][0]
    if corridor is not None:
        # Raw chainage on the polyline, which is what the slicer needs -- `keep`
        # has been re-anchored, possibly reversed, and rescaled by now.
        corridor["poly"] = poly
        corridor["cum"] = cum
        corridor["raw"] = {nm: snapped[nm][1] for _, nm in keep if nm in snapped}
        # {station: chainage of the corridor end beyond it}, so write_geojson
        # can draw the piece that runs on to the junction.
        corridor["ran_on"] = {nm: (cum[0] if nm == spec["first"] else cum[-1])
                              for nm in ran_on}
    return [(km * scale, nm) for km, nm in keep], None


def reconstruct(stops, flows, down=True, rev=False):
    """Cumulate inward from the anchor end; see README.md.

    `down` means travelling stop 0 -> stop n along the chain. That is 하행 only
    when the chain runs 기점 -> 종점; `rev` says it does not, and the 승하차
    column pair has to be swapped to match.
    """
    n = len(stops) - 1
    col = 0 if down != rev else 2
    bo = lambda nm: flows.get(nm, (0, 0, 0, 0))[col:col + 2]
    loads = [0.0] * n
    b_end, a_end = bo(stops[n][1])
    loads[n - 1] = a_end if down else b_end
    for i in range(n - 1, 0, -1):
        b, a = bo(stops[i][1])
        loads[i - 1] = (loads[i] - b + a) if down else (loads[i] + b - a)
    return loads


def run(canon, spec, g, named, flows):
    stops, err = order_stations(spec, g, named, flows)
    if err:
        return {"line": canon, "error": err}

    rev = spec.get("reversed", False)
    down = reconstruct(stops, flows, True, rev)
    up = reconstruct(stops, flows, False, rev)

    f = lambda nm: flows.get(nm, (0, 0, 0, 0))
    users = down[0] + up[-1] + sum(f(nm)[0] + f(nm)[2] for _, nm in stops[1:-1])

    # No clean terminus means the anchor read a junction's whole traffic and
    # lifted both profiles by a constant. 통과인원 shifts by 2d when they do, so
    # it pins the constant the anchor could not supply.
    shift = 0.0
    if not spec["clean_end"] and spec["passing"] > 0:
        shift = (spec["passing"] - users) / 2.0
        down = [x + shift for x in down]
        up = [x + shift for x in up]
        users = spec["passing"]

    gap = max(abs(down[i] - up[i]) / max(abs(down[i]), abs(up[i]), 1)
              for i in range(len(down)))
    pkm = sum((down[i] + up[i]) * (stops[i + 1][0] - stops[i][0])
              for i in range(len(down)))
    negative = min(min(down), min(up)) < 0

    # A verdict, so the report says which lines are usable rather than leaving
    # it to be eyeballed. A negative load is proof the cumulation is wrong --
    # trains cannot carry fewer than nobody. A wide mirror gap means the two
    # directions disagree, which the arithmetic alone cannot cause.
    #
    # The 통과인원 ratio only tests a line whose level was measured rather than
    # solved -- for a solved line it is an identity, since it is what was solved
    # for. It is also only a fair test where the line carries every train type:
    # 통과인원 counts everyone who touched the line's metals, so restricting a
    # line to conventional trains guarantees a low ratio rather than an error.
    ratio = users / spec["passing"] if spec["passing"] else 0.0
    testable = spec["clean_end"] and spec["full_types"]
    if negative:
        verdict = "broken"
    elif gap > 0.15 or (testable and not 0.6 < ratio < 1.5):
        verdict = "shaky"
    elif not spec["clean_end"]:
        verdict = "solved"
    elif not testable:
        verdict = "partial"
    else:
        verdict = "good"
    # reconstruct()'s `down` is the chain's own order, which on a reversed chain
    # is 상행 -- swap the pair back before anything is labelled 하행.
    if rev:
        down, up = up, down
    return {
        "line": canon, "stops": stops, "down": down, "up": up,
        "users": users, "passing": spec["passing"], "shift": shift,
        "mirror": gap, "length": stops[-1][0], "clean": spec["clean_end"],
        "density": pkm / stops[-1][0] / 365.0 if stops[-1][0] else 0.0,
        "negative": negative, "verdict": verdict,
        "types": "+".join(spec["types"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line")
    args = ap.parse_args()

    table, flows = LN.resolve()
    named = M.load_stations(LN)
    print("loading the national track graph ...")
    g = load_network()
    print("   %d nodes\n" % len(g))

    todo = [args.line] if args.line else list(LN.LINES)
    results = []
    for canon in todo:
        spec = table[canon]
        if "error" in spec:
            print("%-11s %s" % (canon, spec["error"]))
            continue
        if spec["passing"] <= 0:
            print("%-11s no 통과인원 -- freight only, skipped" % canon)
            continue
        r = run(canon, spec, g, named, spec["flows"])
        results.append(r)
        if "error" in r:
            print("%-11s %s" % (canon, r["error"]))

    ok = [r for r in results if "error" not in r]
    order = {"good": 0, "solved": 1, "partial": 2, "shaky": 3, "broken": 4}
    print("\n%-11s %-7s %5s %7s %11s %11s %7s %9s"
          % ("line", "verdict", "stops", "km", "통과인원", "yearbook",
             "mirror", "수송밀도"))
    print("-" * 76)
    for r in sorted(ok, key=lambda x: (order[x["verdict"]], -x["density"])):
        print("%-11s %-7s %5d %7.1f %11.0f %11.0f %6.1f%% %9.0f"
              % (r["line"], r["verdict"], len(r["stops"]), r["length"],
                 r["users"], r["passing"], 100 * r["mirror"], r["density"]))
    tally = collections.Counter(r["verdict"] for r in ok)
    print("\n%s" % ", ".join("%d %s" % (n, k) for k, n in tally.most_common()))

    if args.line and ok:
        r = ok[0]
        print("\n%-13s %8s %10s %10s %10s"
              % ("segment", "km", "하행", "상행", "명/일"))
        print("-" * 54)
        for i in range(len(r["down"])):
            a, b = r["stops"][i], r["stops"][i + 1]
            print("%-13s %8.1f %10.0f %10.0f %10.0f"
                  % ((a[1] + "-" + b[1])[:13], b[0] - a[0], r["down"][i],
                     r["up"][i], (r["down"][i] + r["up"][i]) / 365.0))

    write_geojson(ok)


def write_geojson(results):
    feats = []
    for r in results:
        for i in range(len(r["down"])):
            a, b = r["stops"][i], r["stops"][i + 1]
            feats.append({
                "type": "Feature",
                "properties": {
                    "line": r["line"], "from": a[1], "to": b[1],
                    "km": round(b[0] - a[0], 3),
                    "down": round(r["down"][i]), "up": round(r["up"][i]),
                    "daily": round((r["down"][i] + r["up"][i]) / 365.0),
                    "anchored": "clean" if r["clean"] else "solved",
                    "verdict": r["verdict"],
                },
                "geometry": None,
            })
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f,
                  ensure_ascii=False)
    print("\nwrote %s (%d segments, geometry still to come)"
          % (os.path.relpath(OUT, HERE), len(feats)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
