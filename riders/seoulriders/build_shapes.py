# -*- coding: utf-8 -*-
"""Make train dots follow the track instead of cutting straight between stops.

Post-processes data/trains.json in place, so it does not need the routing to be
re-run. For each consecutive pair of stations a train visits, find the path
along that line's OSM ways and insert the intermediate points as timed
waypoints.

The ways arrive as an unordered soup with branches, so rather than trying to
assemble one polyline per line, build a graph of the way vertices and run
Dijkstra between the two stations. Branches then take care of themselves, and
express trains that skip stations still get a real path.

**Snap both ends together, not one at a time.** A station's *nearest* vertex is
the wrong thing to route from. Seoul's trunk corridors carry four parallel
tracks a few metres apart -- local, express and Korail on the 경부선 -- and the
welded graph joins them only at the crossovers. Snap each station to its own
nearest vertex and the two ends can land on different tracks, so Dijkstra runs
to the next crossover and back: 용산 to 노량진 came out 7,962 m for a 2,704 m
chord, the train shooting past 노량진 to 신길 and reversing into it. So take
every vertex within a couple of hundred metres of each station as a *candidate*,
run one multi-source Dijkstra, and pick the pair that minimises path length plus
SNAP_PENALTY x the two offsets. The right track wins by kilometres; the penalty
stops a candidate several hundred metres down the line from being taken just to
shave the path.

Two nets underneath that, for cases the candidates cannot reach -- OSM gaps,
or a line whose relation carries only part of its track: reject a path far
longer than the chord, and reject one that doubles back past its own end.
A rejected segment falls back to the straight line, which is honest about not
knowing rather than wrong about where the track goes.

Segments whose track is effectively straight are left alone -- a waypoint that
sits on the line between its neighbours costs bytes and buys nothing.

Idempotent: the waypoints it inserts are 3-element rows, station stops are
5-element ones, so dropping every short row restores the input exactly. Run it
twice and you get the same file. build.py calls main() as its last step, which
is what stops "trains cut corners" from coming back every time the routing is
re-run; running it by hand afterwards is still fine.

    python build_shapes.py
"""

import collections
import heapq
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
TRAINS = os.path.join(D, "trains.json")
STATIONS = os.path.join(D, "stations.json")

SNAP_M = 900.0        # how far a station may sit from its line's track
SAG_M = 25.0          # keep a waypoint only if the track bows this far off
MAX_PTS = 10          # cap per segment, after simplification
QUANT = 6             # decimal places when welding way vertices together

# Snapping. CAND_SLACK_M has to clear the width of a four-track corridor plus
# the gap between platform centroid and track; 250 m does, and costs nothing
# because all the candidates share one Dijkstra run.
CAND_SLACK_M = 250.0  # candidate vertices, beyond the nearest one
CAND_MAX = 40         # ... capped, so a dense junction cannot blow up
SNAP_PENALTY = 3.0    # a metre off the platform costs this much path

# Sanity checks on the path that comes back.
DETOUR_RATIO = 2.2    # reject a path longer than this many times the chord
DETOUR_EXTRA = 700.0  # ... plus this, so short segments are not over-policed
BACK_NEAR_M = 120.0   # a path that comes this close to its own end
BACK_SLACK_M = 400.0  # ... and then runs on this much further is doubling back


def metres(a, b):
    dy = (a[0] - b[0]) * 111320.0
    dx = (a[1] - b[1]) * 111320.0 * math.cos(math.radians(a[0]))
    return math.hypot(dx, dy)


def perp_m(p, a, b):
    """Distance from p to the segment a-b, in metres."""
    ay = (a[0] - p[0]) * 111320.0
    ax = (a[1] - p[1]) * 111320.0 * math.cos(math.radians(p[0]))
    by = (b[0] - p[0]) * 111320.0
    bx = (b[1] - p[1]) * 111320.0 * math.cos(math.radians(p[0]))
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        return math.hypot(ax, ay)
    t = -(ax * vx + ay * vy) / L2
    t = max(0.0, min(1.0, t))
    return math.hypot(ax + t * vx, ay + t * vy)


def simplify(pts, eps_m):
    """Douglas-Peucker, distances in metres."""
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        worst, wi = 0.0, -1
        for k in range(i + 1, j):
            d = perp_m(pts[k], pts[i], pts[j])
            if d > worst:
                worst, wi = d, k
        if wi >= 0 and worst > eps_m:
            keep[wi] = True
            stack.append((i, wi))
            stack.append((wi, j))
    return [p for p, k in zip(pts, keep) if k]


def build_graph(ways):
    """Weld way vertices into one graph keyed by rounded coordinate."""
    adj = collections.defaultdict(list)
    nodes = {}
    for w in ways:
        prev = None
        for p in w:
            key = (round(p[0], QUANT), round(p[1], QUANT))
            nodes[key] = (p[0], p[1])
            if prev is not None and prev != key:
                d = metres(nodes[prev], nodes[key])
                adj[prev].append((key, d))
                adj[key].append((prev, d))
            prev = key
    return adj, nodes


def nearest(nodes, pt):
    best, bd = None, 1e18
    for k, v in nodes.items():
        d = metres(v, pt)
        if d < bd:
            best, bd = k, d
    return best, bd


def candidates(nodes, pt):
    """Every vertex worth routing from, nearest first, with its offset.

    One per parallel track is what we are after, but there is no cheap way to
    tell which vertices share a track, and taking a few extra costs nothing:
    they all go into the same Dijkstra.
    """
    near = []
    for k, v in nodes.items():
        d = metres(v, pt)
        if d <= SNAP_M:
            near.append((d, k))
    if not near:
        return []
    near.sort()
    cut = near[0][0] + CAND_SLACK_M
    return [(d, k) for d, k in near[:CAND_MAX] if d <= cut]


def path_between(adj, nodes, srcs, dsts, limit_m):
    """Best track path from any source vertex to any destination vertex.

    `srcs` and `dsts` are (offset_m, key) lists from candidates(). Cost is path
    length plus SNAP_PENALTY x the two offsets, so the pair that actually lies
    on one continuous track wins even when neither end is the closest vertex.
    Returns (points, path_length_m) or None.
    """
    dist = {}
    prev = {}
    pq = []
    for d0, k in srcs:
        c = d0 * SNAP_PENALTY
        if c < dist.get(k, 1e18):
            dist[k] = c
            heapq.heappush(pq, (c, k))
    want = dict((k, d0 * SNAP_PENALTY) for d0, k in dsts)
    best_end, best_cost = None, 1e18
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 1e18):
            continue
        if u in want:
            c = d + want[u]
            if c < best_cost:
                best_cost, best_end = c, u
        # Everything still queued costs at least `d`, so once the queue can no
        # longer beat the best complete pair we are done.
        if d >= best_cost:
            break
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, 1e18) and nd < limit_m:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if best_end is None:
        return None
    out = [best_end]
    while out[-1] in prev:
        out.append(prev[out[-1]])
    out.reverse()
    pts = [nodes[k] for k in out]
    plen = sum(metres(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    return pts, plen


def doubles_back(pts, p0, p1):
    """True if the path reaches one of its own ends and then runs on."""
    if len(pts) < 3:
        return False
    seg = [metres(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(seg)
    run = 0.0
    for i, p in enumerate(pts):
        ahead = total - run
        behind = run
        if ahead > BACK_SLACK_M and metres(p, p1) < BACK_NEAR_M:
            return True
        if behind > BACK_SLACK_M and metres(p, p0) < BACK_NEAR_M:
            return True
        if i < len(seg):
            run += seg[i]
    return False


def write_json(path, obj):
    """Write a whole file or none of it -- see build.py's copy for why."""
    tmp = path + ".part"
    with io.open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def main(trains_path=TRAINS):
    with io.open(STATIONS, encoding="utf-8") as f:
        net = json.load(f)
    with io.open(trains_path, encoding="utf-8") as f:
        data = json.load(f)

    print("building track graphs ...")
    graphs = {}
    for line, ways in net["geometry"].items():
        adj, nodes = build_graph(ways)
        graphs[line] = (adj, nodes)
        print("   line %-3s %6d vertices" % (line, len(nodes)))

    snap_cache = {}

    def snap(line, pt):
        key = (line, round(pt[0], 5), round(pt[1], 5))
        if key not in snap_cache:
            snap_cache[key] = candidates(graphs[line][1], pt)
        return snap_cache[key]

    seg_cache = {}
    hits = misses = straight = rejected = 0

    def segment(line, p0, p1):
        nonlocal hits, misses, straight, rejected
        key = (line, round(p0[0], 5), round(p0[1], 5),
               round(p1[0], 5), round(p1[1], 5))
        if key in seg_cache:
            return seg_cache[key]
        a, b = snap(line, p0), snap(line, p1)
        res = None
        bad = False
        if a and b:
            adj, nodes = graphs[line]
            chord = metres(p0, p1)
            got = path_between(adj, nodes, a, b,
                               max(2500.0, chord * 3.0 + 800.0))
            if got:
                pts, plen = got
                if (plen > chord * DETOUR_RATIO + DETOUR_EXTRA
                        or doubles_back(pts, p0, p1)):
                    bad = True
                elif len(pts) > 2:
                    pts = simplify(pts, SAG_M)[1:-1]
                    if len(pts) > MAX_PTS:
                        step = len(pts) / float(MAX_PTS)
                        pts = [pts[int(i * step)] for i in range(MAX_PTS)]
                    res = pts if pts else None
        if res is not None:
            hits += 1
        elif bad:
            rejected += 1
        elif a and b:
            straight += 1
        else:
            misses += 1
        seg_cache[key] = res
        return res

    # Keep every station-to-station polyline we work out, so the static
    # view can draw the same curves the trains follow instead of chords.
    link_shapes = {}

    print("shaping %d trains ..." % len(data["trains"]))
    added = 0
    for ti, tr in enumerate(data["trains"]):
        line = tr["route"]
        if line not in graphs:
            continue
        # Drop any waypoints a previous run left behind, so shaping a
        # already-shaped file is a no-op rather than a compounding one.
        tl = [r for r in tr["timeline"] if len(r) >= 4]
        out = [tl[0]]
        for i in range(len(tl) - 1):
            a, b = tl[i], tl[i + 1]
            mid = segment(line, (a[1], a[2]), (b[1], b[2]))
            key = "%s|%.5f,%.5f|%.5f,%.5f" % (line, a[1], a[2], b[1], b[2])
            if key not in link_shapes:
                pts_all = [(a[1], a[2])] + list(mid or []) + [(b[1], b[2])]
                link_shapes[key] = [[round(x, 5), round(y, 5)]
                                    for x, y in pts_all]
            if mid:
                # spread the waypoints along the segment by distance travelled
                d = [0.0]
                pts = [(a[1], a[2])] + mid + [(b[1], b[2])]
                for j in range(1, len(pts)):
                    d.append(d[-1] + metres(pts[j - 1], pts[j]))
                total = d[-1]
                if total > 1e-6:
                    for j in range(1, len(pts) - 1):
                        t = a[0] + (b[0] - a[0]) * (d[j] / total)
                        out.append([round(t), round(pts[j][0], 5),
                                    round(pts[j][1], 5)])
                        added += 1
            out.append(b)
        tr["timeline"] = out
        if ti % 500 == 0:
            print("   %d/%d  (%d waypoints so far)"
                  % (ti, len(data["trains"]), added))
            sys.stdout.flush()

    print("\nsegments shaped %d, already straight %d, unmatched %d, "
          "rejected as a detour %d" % (hits, straight, misses, rejected))
    print("waypoints inserted: %s" % format(added, ","))

    write_json(trains_path, data)
    print("wrote %s (%.1f MB)"
          % (trains_path, os.path.getsize(trains_path) / 1e6))

    shapes_path = os.path.join(D, "link_shapes.json")
    write_json(shapes_path, link_shapes)
    print("wrote %s (%d links, %.1f MB)"
          % (shapes_path, len(link_shapes),
             os.path.getsize(shapes_path) / 1e6))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1] if len(sys.argv) > 1 else TRAINS)
