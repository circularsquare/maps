"""T-1305: a real network's shared track (notes/T-1305.md).

    python pipeline/realcorridors.py <city> [--out DIR]

Every hop is routed again over the real-network graph as `realnet.py route` does (same rules and
acceptance), keeping the chains it runs on, then a second time off track whose trains are mostly
another class's (`route_hops`: London's tube lines, the sub-surface lines and main-line rail each
on their own where OSM has it), and a whole build again off pieces the first found full with
several routes, where a parallel line runs (the Brighton line's fast trains on the Quarry line).
Parallel chains (within NEAR, same heading) of one kind of railway (all main-line rail together,
else per kind), both in tunnel or both not, and joined by track within CONNECT_M, become one
corridor, laid on the first of them in order of how many lines use it. The geom file needs each line's shapes (its stops on its own track): built from
station centroids, a line at a big complex snaps to another line's track. A line's run along a
corridor is one way: where it turns back at a station (the GWR's Gatwick trains at Redhill) it is
two runs. Each corridor is cut at stations and wherever its set of lines changes; cuts that change
nothing are joined again, so a piece runs between stations or junctions.

A piece's lines are packed into double tracks (pairs) by load (docs 6.2), at most PACK_LOAD each
(PACK_LOAD_RAIL on main-line rail) and then spread evenly over them, as many as its trains need and OSM's parallel chains within WIDE allow (half of them, rounded up),
and never fewer than the sets of lines that never run on one chain there. Where some lines pass a
station at its end that others stop at, and OSM shows three or more tracks within NEAR, the
passing lines (express) and the stopping ones (local) go on pairs of their own; a pair past
WANT_LOAD is split in two the same way; light rail and trams never share with other kinds. Lines
of different routes loading one pair past APART_LOAD (more than real service ever puts on one)
get pairs of their own whatever OSM's count says: the router laid them on one chain. Most loaded
pieces first, a piece's split carries on to its neighbours where their trains fit it, so lines keep
their pair along a run; a one-pair piece between split pieces takes their split where OSM shows
three tracks. Hops off the graph (no track near a station, a detour, off the GTFS shape) are
pieces of their own with the line's shape or a straight line.

Reads `<city>_real_geom.json` (realnet.out_or_work: $REALNET_OUT first, else data/work/realnet) and
the network JSON; writes `<city>_real_corridors.json` and `<city>_corridors.png` to --out
(default realnet.OUT). Plane metres from the pack origin, as the other realnet files.

    pieces  id, line (polyline), km, pairs, tracks (OSM's parallel chains, median), lines (one list
            per pair), role per pair ("express"/"local"/""), ends (node ids), load (high-demand
            load per pair, one direction, docs 6.2), off_graph
    nodes   id, xy, kind (station, junction, end, through), stops [[piece, pair, [lines]]...]
    lines   index into the network's lines, name, path [[piece, +1/-1, pair]...], stops [node...]
"""
import argparse
import heapq
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realnet  # noqa: E402
from realroute import Graph, length, far_from  # noqa: E402
from realshared import track_model  # noqa: E402
from reallevels import assign as assign_levels  # noqa: E402

STEP = 10.0
NEAR = 15.0
# tracks a piece's trains may spread over: OSM's parallel chains this far out (a wide station's
# platforms, a trunk whose pairs lie apart)
WIDE = 30.0
PARALLEL = math.radians(25)
SNAP = 40.0
MIN_RUN = 30.0
CONNECT_M = 2000.0
TRACK_STEP = 50.0
WANT_LOAD = 0.8
MAX_LOAD = 0.9
# pairs are filled to this at most (docs 6.2 estimate), main-line rail lower: its trunks carry many
# stopping patterns, which the game queues as separate streams (T-1774)
PACK_LOAD, PACK_LOAD_RAIL = 0.9, 0.6

# no real pair is this loaded in the game's physics with more than one route on it (New York's
# busiest shared pair is 1.0): past it, the lines were laid on one chain by the router
APART_LOAD = 1.15
# a chain's cost, second routing, for a line of another class than the one with most trains on it
OTHER_CLASS = 3.0
# a chain's cost, second build, on a piece the first found over MAX_LOAD with several routes
AVOID = 3.0
# docs 6.2 (sim track/params.rs): section headway 70 s + v / 0.6; traction 1.0 m/s² to 60 km/h,
# 0.5 above, braking 0.9, top 160 km/h; platform dwell + 60 s a stopping train, 40 s a passing one
SECTION_BASE_S, SIGNAL_BRAKE = 70.0, 0.6
V_TOP, V_SWITCH, A_LOW, A_HIGH, BRAKE = 160 / 3.6, 60 / 3.6, 1.0, 0.5, 0.9
DWELL_S, PLATFORM_MARGIN_S, PASSING_S = 30.0, 60.0, 40.0
# the New York subway's real top speed (55 mph), for a second load figure beside the game's 160
V_REAL_METRO = 88 / 3.6


def wrap(a):
    a = abs(a) % (2 * math.pi)
    return 2 * math.pi - a if a > math.pi else a


def parallel(h1, h2):
    d = wrap(h1 - h2)
    return min(d, math.pi - d) < PARALLEL


def v_reach(span):
    """Top speed a start-to-stop run of `span` m reaches under the game's traction, curves aside."""
    def dist(v):
        acc = v * v / (2 * A_LOW) if v <= V_SWITCH else V_SWITCH ** 2 / (2 * A_LOW) + (v * v - V_SWITCH ** 2) / (2 * A_HIGH)
        return acc + v * v / (2 * BRAKE)
    if dist(V_TOP) <= span:
        return V_TOP
    lo, hi = 0.0, V_TOP
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if dist(mid) <= span else (lo, mid)
    return lo


class Union:
    def __init__(self):
        self.p = {}

    def find(self, a):
        p = self.p
        p.setdefault(a, a)
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def line_class(l):
    """Lines of one class may share track where the router finds them on one chain: main-line rail
    all together, else per kind and operator (London's tube lines, the sub-surface lines, the DLR)."""
    k = l.get("kind")
    return "rail" if k == "commuter" else f"{k}:{l.get('operator') or l['feed']}"


def route_hops(graph, net, geo, avoid=frozenset()):
    """Per line, per hop: {"legs", "a", "b"} (snaps as (chain, along)) on the graph, or {"poly"}.

    Routed twice: the second time each chain costs OTHER_CLASS times more for a line of another
    class than the one with most trains on it the first time, so a line the router laid on its
    neighbour's track (the Bakerloo on the Northern's tunnel to Waterloo, the Piccadilly on the
    District's at South Kensington) takes its own where OSM has it, and keeps the shared one where
    there is none (the Piccadilly on the Metropolitan to Uxbridge)."""
    lines = net["lines"]
    fam_of, why = [], Counter()

    def one(li, g, k, extra=None):
        a, b = np.array(g["nodes"][k]), np.array(g["nodes"][k + 1])
        r = graph.route(a, b, realnet.ROUTE_R, fam_of[li], extra)
        if r is None or len(r[0]) < 2:
            return None, "no route"
        if length(r[0]) > realnet.ROUTE_SLACK * np.hypot(*(b - a)) + realnet.ROUTE_SLACK_M:
            return None, "detour"
        if g["hops"][k] and far_from(np.array(g["hops"][k]), r[0]) > realnet.ROUTE_SHAPE_M:
            return None, "off its shape"
        return r, None

    first = []
    for li, (l, g) in enumerate(zip(lines, geo)):
        # a main-line line on rail where most of its stops have rail in reach: the nearest track
        # alone put the Heathrow Express on subway (the Piccadilly at Heathrow, the Circle at Paddington)
        if l.get("kind") == "commuter":
            near = [{graph.fam[c] for _, c, _, _ in graph.candidates(np.array(p), realnet.ROUTE_R)} for p in g["nodes"]]
            near = [s for s in near if s]
            fam_of.append(0 if not near or 2 * sum(0 in s for s in near) >= len(near) else 1)
        else:
            fams = [f for f in (graph.family_near(np.array(p), realnet.ROUTE_R) for p in g["nodes"]) if f is not None]
            fam_of.append(Counter(fams).most_common(1)[0][0] if fams else (1 if l.get("kind") == "metro" else 0))
        first.append([one(li, g, k) for k in range(len(g["nodes"]) - 1)])
    cls = [line_class(l) for l in lines]
    used = {c for hs in first for r, _ in hs if r is not None for c, _, _ in r[3]}
    grp = track_groups(graph, used)
    trains = defaultdict(Counter)
    for li, hs in enumerate(first):
        for g in {grp.find(c) for r, _ in hs if r is not None for c, _, _ in r[3]}:
            trains[g][cls[li]] += lines[li]["tph"][0]
    owner = {c: trains[grp.find(c)].most_common(1)[0][0] for c in used}
    out, rerouted = [], Counter()
    for li, (l, g) in enumerate(zip(lines, geo)):
        other = {c for c, o in owner.items() if o != cls[li]}
        extra = {c: OTHER_CLASS for c in other}
        for c in avoid:
            extra[c] = extra.get(c, 1.0) * AVOID
        hops = []
        for k, (r, w) in enumerate(first[li]):
            if r is not None and any(c in extra for c, _, _ in r[3]):
                r2, _ = one(li, g, k, extra)
                by_class = any(c in other for c, _, _ in r[3])
                # off a full piece, only a route away from it counts: a swap between a trunk's own
                # tracks (New York's express and local) changes nothing but which chains are shared
                if r2 is not None and [c for c, _, _ in r2[3]] != [c for c, _, _ in r[3]] and (by_class or far_from(r2[0], r[0]) > WIDE):
                    if os.environ.get("REALCORR_DEBUG"):
                        print(f"reroute {l['name']} hop {k}: {length(r[0]):.0f} m -> {length(r2[0]):.0f} m,"
                              f" families {sorted({graph.fam[c] for c, _, _ in r2[3]})}")
                    r = r2
                    rerouted["rerouted onto their own class's track" if by_class else "rerouted off a full piece"] += 1
            if r is None:
                why[w] += 1
                a, b = g["nodes"][k], g["nodes"][k + 1]
                hops.append({"poly": np.array([a] + (g["hops"][k] or []) + [b], float)})
            else:
                hops.append({"legs": r[3]})
        out.append(hops)
    why.update(rerouted)
    return out, why


def line_key(l):
    """Chains of lines of one key may be laid on one corridor (where joined by track): main-line
    rail, else per kind, so London's sub-surface lines share where their tracks are one pair."""
    return "rail" if l.get("kind") == "commuter" else l.get("kind") or l["feed"]


def chain_samples(graph, c):
    L = graph.len[c]
    n = max(1, int(math.ceil(L / STEP)))
    a = np.linspace(0.0, L, n + 1)
    p, cum = graph.pts[c], graph.cum[c]
    x, y = np.interp(a, cum, p[:, 0]), np.interp(a, cum, p[:, 1])
    a0, a1 = np.clip(a - 5, 0, L), np.clip(a + 5, 0, L)
    h = np.arctan2(np.interp(a1, cum, p[:, 1]) - np.interp(a0, cum, p[:, 1]), np.interp(a1, cum, p[:, 0]) - np.interp(a0, cum, p[:, 0]))
    return a, np.stack([x, y], 1), h


def connector(graph):
    joined = {}

    def connected(c1, c2):
        """Track joins the two chains within CONNECT_M (a crossover or a junction): tracks of one
        railway; two lines on two levels, or of systems that never meet, are not."""
        k = (min(c1, c2), max(c1, c2))
        if k not in joined:
            goal = {graph.ja[c2], graph.jb[c2]}
            best = {graph.ja[c1]: 0.0, graph.jb[c1]: 0.0}
            heap = [(0.0, j) for j in best]
            ok = False
            while heap:
                d, j = heapq.heappop(heap)
                if j in goal:
                    ok = True
                    break
                if d > best.get(j, math.inf):
                    continue
                for e in graph.adj.get(j, ()):
                    c = e >> 1
                    j2 = graph.ja[c] if e & 1 else graph.jb[c]
                    d2 = d + graph.len[c]
                    if d2 <= CONNECT_M and d2 < best.get(j2, math.inf):
                        best[j2] = d2
                        heapq.heappush(heap, (d2, j2))
            joined[k] = ok
        return joined[k]
    return connected


def track_groups(graph, chains):
    """Chains of one trunk whatever lines use them: parallel within NEAR for MIN_RUN, same family,
    both in tunnel or both not, joined by track within CONNECT_M (a pair's two tracks, mapped as
    two ways; a four-track trunk). Union over chain ids."""
    chains = sorted(chains)
    samp = [chain_samples(graph, c) for c in chains]
    xy = np.concatenate([p for _, p, _ in samp])
    hd = np.concatenate([h for _, _, h in samp])
    own = np.concatenate([np.full(len(a), i) for i, (a, _, _) in enumerate(samp)])
    tree = cKDTree(xy)
    hits = Counter()
    for i, j in tree.query_pairs(NEAR, output_type="ndarray"):
        a, b = own[i], own[j]
        if a != b and parallel(hd[i], hd[j]):
            hits[(min(a, b), max(a, b))] += 1
    connected = connector(graph)
    u = Union()
    for c in chains:
        u.find(c)
    for (a, b), n in hits.items():
        ca, cb = chains[a], chains[b]
        if n * STEP < MIN_RUN or graph.fam[ca] != graph.fam[cb] or bool(graph.flags[ca] & 4) != bool(graph.flags[cb] & 4):
            continue
        if connected(ca, cb):
            u.join(ca, cb)
    return u


def map_chains(graph, net, hops):
    """Each used chain's samples laid on a corridor master: {chain: (alongs, master[], m_along[], sign[])}."""
    users = defaultdict(set)
    for li, hs in enumerate(hops):
        for h in hs:
            for c, _, _ in h.get("legs", ()):
                users[c].add(li)
    key = {c: Counter(line_key(net["lines"][li]) for li in ls).most_common(1)[0][0] for c, ls in users.items()}
    order = sorted(users, key=lambda c: (-len(users[c]), -graph.len[c], c))
    rank = {c: i for i, c in enumerate(order)}
    samp = {c: chain_samples(graph, c) for c in order}
    base, xy, owner, heads = {}, [], [], []
    for c in order:
        base[c] = len(owner)
        a, p, h = samp[c]
        xy.append(p)
        heads.append(h)
        owner += [c] * len(a)
    xy, heads, owner = np.concatenate(xy), np.concatenate(heads), np.array(owner)
    tree = cKDTree(xy)
    tunnel = {c: bool(graph.flags[c] & 4) for c in order}
    connected = connector(graph)
    m_of = np.empty(len(owner), np.int64)
    a_of = np.empty(len(owner))
    s_of = np.empty(len(owner))
    out = {}
    for c in order:
        a, p, h = samp[c]
        b0 = base[c]
        own = np.arange(b0, b0 + len(a))
        near = tree.query_ball_point(p, NEAR)
        mm, ma, ms = np.full(len(a), c), a.copy(), np.ones(len(a))
        for i, js in enumerate(near):
            best, bd = -1, NEAR + 1
            for j in js:
                cj = owner[j]
                if rank[cj] >= rank[c] or key[cj] != key[c] or tunnel[cj] != tunnel[c] or not parallel(heads[j], h[i]):
                    continue
                d = math.hypot(*(xy[j] - p[i]))
                if d < bd and connected(c, cj):
                    best, bd = j, d
            if best >= 0:
                mm[i], ma[i] = m_of[best], a_of[best]
                ms[i] = s_of[best] * (1 if math.cos(heads[best] - h[i]) > 0 else -1)
        smooth(c, a, mm, ma, ms)
        m_of[own], a_of[own], s_of[own] = mm, ma, ms
        out[c] = (a, mm, ma, ms)
    return out, users


def smooth(c, a, mm, ma, ms):
    """Short flickers in a chain's mapping: a short stretch laid on another corridor between its own
    goes back to its own; a short own stretch between two of one other corridor joins it."""
    k = max(1, int(MIN_RUN // STEP))
    runs, i = [], 0
    while i < len(mm):
        j = i
        while j + 1 < len(mm) and mm[j + 1] == mm[i]:
            j += 1
        runs.append((i, j))
        i = j + 1
    for r, (i, j) in enumerate(runs):
        if j - i + 1 >= k or len(runs) == 1:
            continue
        prev = mm[runs[r - 1][1]] if r > 0 else None
        nxt = mm[runs[r + 1][0]] if r + 1 < len(runs) else None
        if mm[i] != c and prev in (c, None) and nxt in (c, None):
            mm[i:j + 1], ma[i:j + 1], ms[i:j + 1] = c, a[i:j + 1], 1
        elif mm[i] == c and prev is not None and prev == nxt:
            i0, j0 = runs[r - 1][1], runs[r + 1][0]
            t = (a[i:j + 1] - a[i0]) / (a[j0] - a[i0])
            mm[i:j + 1] = prev
            ma[i:j + 1] = ma[i0] + t * (ma[j0] - ma[i0])
            ms[i:j + 1] = ms[i0]


def turns(a):
    """Sample indices where a run along a master turns back, by more than MIN_RUN either side."""
    out, ext, d = [], 0, 0
    for i in range(1, len(a)):
        if d == 0:
            if abs(a[i] - a[0]) > MIN_RUN:
                d = 1 if a[i] > a[0] else -1
                ext = i
            continue
        if (a[i] - a[ext]) * d > 0:
            ext = i
        elif abs(a[i] - a[ext]) > MIN_RUN:
            out.append(ext)
            d, ext = -d, i
    return out


def line_items(hops, cmap):
    """A line as items in order: ("run", master, [alongs], {stop k: along}, [chain per along]) and
    ("free", k, poly). A run is one way along its master: a line turning back at a station (the
    GWR's Gatwick trains at Redhill) is two runs."""
    items = []
    cur = None
    for k, h in enumerate(hops):
        if "poly" in h:
            if cur:
                items.append(cur)
                cur = None
            items.append(("free", k, h["poly"]))
            continue
        for li, (c, fr, to) in enumerate(h["legs"]):
            a, mm, ma, ms = cmap[c]
            ds = np.linspace(fr, to, max(2, int(math.ceil(abs(to - fr) / STEP)) + 1))
            step = a[1] - a[0] if len(a) > 1 else 1.0
            idx = np.clip(np.rint(ds / step).astype(int), 0, len(a) - 1)
            for n, (d, i) in enumerate(zip(ds, idx)):
                m, al = int(mm[i]), float(ma[i] + ms[i] * (d - a[i]))
                stop = k if (li == 0 and n == 0) else (k + 1 if (li == len(h["legs"]) - 1 and n == len(ds) - 1) else None)
                if cur is None or cur[1] != m:
                    if cur:
                        items.append(cur)
                    cur = ("run", m, [], {}, [])
                if stop is not None:
                    cur[3].setdefault(stop, []).append(len(cur[2]))
                cur[2].append(al)
                cur[4].append(c)
        # a stop between two hops on graph is one point on each side
    if cur:
        items.append(cur)
    # short runs without a stop are crossovers between corridors: dropped
    keep = []
    for it in items:
        if it[0] == "run" and not it[3] and abs(it[2][-1] - it[2][0]) < MIN_RUN:
            continue
        keep.append(it)
    # neighbours left on one master after a drop join up
    out = []
    for it in keep:
        if out and it[0] == "run" and out[-1][0] == "run" and out[-1][1] == it[1]:
            prev = out[-1]
            st = {s: list(v) for s, v in prev[3].items()}
            for s, v in it[3].items():
                st.setdefault(s, []).extend(j + len(prev[2]) for j in v)
            out[-1] = ("run", it[1], prev[2] + it[2], st, prev[4] + it[4])
        else:
            out.append(it)
    # runs cut where they turn back; stops as alongs
    res = []
    for it in out:
        if it[0] != "run":
            res.append(it)
            continue
        _, m, al, st, ch = it
        bounds = [0] + turns(al) + [len(al) - 1]
        for s, e in zip(bounds[:-1], bounds[1:]):
            part = {k: [al[j] for j in v if s <= j <= e] for k, v in st.items()}
            res.append(("run", m, al[s:e + 1], {k: v for k, v in part.items() if v}, ch[s:e + 1]))
    return res


def cluster(values):
    """Cut points on one master: [(along, is_stop, ref)] -> along per group, group per ref."""
    values.sort(key=lambda v: v[0])
    groups, of = [], {}
    for v in values:
        if not groups or v[0] - groups[-1][0][0] > SNAP:
            groups.append([v])
        else:
            groups[-1].append(v)
    pos = []
    for g, members in enumerate(groups):
        st = [v[0] for v in members if v[1]]
        pos.append(float(np.mean(st if st else [v[0] for v in members])))
        for v in members:
            of[v[2]] = g
    return pos, of


def build(city, out_dir, avoid=None):
    """`avoid`: chains of pieces the first build found over MAX_LOAD with lines of more than one
    route on a pair; the second routes lines off them where a parallel line runs (T-1774: the
    Brighton line's fast trains on the Quarry line, not all through Redhill and Merstham)."""
    t0 = time.time()
    cfg = realnet.CITIES[city]
    o = realnet.origin(cfg["pack"])
    staged = os.path.join(realnet.OUT, os.path.basename(cfg["out"]))
    net = json.load(open(staged if os.path.exists(staged) else cfg["out"], encoding="utf-8"))
    geo = json.load(open(realnet.out_or_work(f"{city}_real_geom.json")))["lines"]
    assert len(geo) == len(net["lines"]) and all(len(g["nodes"]) == len(l["stops"]) for g, l in zip(geo, net["lines"])), \
        f"{city}_real_geom.json does not match {city}_real.json"
    xy = np.array([p for g in geo for p in g["nodes"]])
    box = (*(xy.min(0) - 5000), *(xy.max(0) + 5000))
    graph = Graph(os.path.join(realnet.REALGRAPH, f"{realnet.REGION[city]}.rgr"), o, box, realnet.R_EARTH)
    t_graph = time.time() - t0
    hops, why = route_hops(graph, net, geo, avoid or frozenset())
    t_route = time.time() - t0
    cmap, users = map_chains(graph, net, hops)
    lines = net["lines"]
    items = [line_items(h, cmap) for h in hops]

    # cut points per master: run ends and stops
    cuts = defaultdict(list)
    for li, its in enumerate(items):
        for r, it in enumerate(its):
            if it[0] != "run":
                continue
            _, m, al, st, _ = it
            cuts[m].append((al[0], False, (li, r, "a")))
            cuts[m].append((al[-1], False, (li, r, "b")))
            for s, v in st.items():
                cuts[m].append((float(np.mean(v)), True, (li, r, s)))
    cut_pos, cut_of = {}, {}
    for m, vals in cuts.items():
        cut_pos[m], cut_of[m] = cluster(vals)

    # nodes: cut points joined along each line; stops
    U = Union()
    stop_node = [dict() for _ in lines]
    runs = []  # (li, master, lo cut, hi cut)
    for li, its in enumerate(items):
        prev_end = None
        for r, it in enumerate(its):
            if it[0] == "free":
                prev_end = None
                continue
            _, m, al, st, chs = it
            ca, cb = cut_of[m][(li, r, "a")], cut_of[m][(li, r, "b")]
            if prev_end is not None:
                U.join(prev_end, ("c", m, ca))
            for s in st:
                n = ("c", m, cut_of[m][(li, r, s)])
                if s in stop_node[li]:
                    U.join(stop_node[li][s], n)
                else:
                    stop_node[li][s] = n
            runs.append((li, m, ca, cb, np.array(al), np.array(chs)))
            prev_end = ("c", m, cb)
        for k in range(len(lines[li]["stops"])):
            if k not in stop_node[li]:
                stop_node[li][k] = ("s", lines[li]["stops"][k])
    # one station's stops by different lines on one corridor are one node
    by_station = defaultdict(list)
    for li, sn in enumerate(stop_node):
        for k, n in sn.items():
            if n[0] == "c":
                by_station[lines[li]["stops"][k]].append((n, graph.point_at(n[1], cut_pos[n[1]][n[2]])))
    for ns in by_station.values():
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                if np.hypot(*(ns[i][1] - ns[j][1])) <= 1.5 * SNAP:
                    U.join(ns[i][0], ns[j][0])
    stop_node = [{k: U.find(n) for k, n in sn.items()} for sn in stop_node]

    # elementary segments per master
    # and the graph chains each line ran on there before they were laid on the master
    seg_lines = defaultdict(set)
    seg_chains = defaultdict(lambda: defaultdict(set))
    for li, m, ca, cb, al, chs in runs:
        lo, hi = min(ca, cb), max(ca, cb)
        for i in range(lo, hi):
            seg_lines[(m, i)].add(li)
            a0, a1 = cut_pos[m][i], cut_pos[m][i + 1]
            seg_chains[(m, i)][li].update(int(c) for c in chs[(al >= a0) & (al <= a1)])
    segs = {}
    for (m, i), ls in seg_lines.items():
        a0, a1 = cut_pos[m][i], cut_pos[m][i + 1]
        if a1 - a0 < 1.0:
            continue
        segs[(m, i)] = {"ends": (U.find(("c", m, i)), U.find(("c", m, i + 1))), "lines": frozenset(ls),
                        "poly": graph.line_of([(m, a0, a1)]), "kind": graph.fam[m], "tunnel": bool(graph.flags[m] & 4)}
    free = []
    for li, its in enumerate(items):
        for it in its:
            if it[0] == "free":
                k = it[1]
                free.append({"ends": (stop_node[li][k], stop_node[li][k + 1]), "lines": frozenset([li]), "poly": it[2],
                             "kind": 1 if lines[li]["kind"] == "metro" else 0, "line": li, "hop": k})

    # pieces: segments joined across nodes that are neither stations nor where lines change
    station = set(n for sn in stop_node for n in sn.values())
    at = defaultdict(list)
    for sid, s in segs.items():
        at[s["ends"][0]].append((sid, 0))
        at[s["ends"][1]].append((sid, 1))
    for fi, f in enumerate(free):
        at[f["ends"][0]].append((("f", fi), 0))
        at[f["ends"][1]].append((("f", fi), 1))

    def mergeable(n):
        e = at[n]
        if n in station or len(e) != 2 or e[0][0] == e[1][0] or "f" in (e[0][0][0], e[1][0][0]):
            return False
        return segs[e[0][0]]["lines"] == segs[e[1][0]]["lines"]

    piece_of, pieces = {}, []
    for sid in segs:
        if sid in piece_of:
            continue
        # walk back to the start of the run of mergeable segments
        cur, end = sid, 0
        seen = {sid}
        while mergeable(segs[cur]["ends"][end]):
            n = segs[cur]["ends"][end]
            nxt = [e for e in at[n] if e[0] != cur][0]
            if nxt[0] in seen:
                break
            seen.add(nxt[0])
            cur, end = nxt[0], 1 - nxt[1]
        # now walk forward from cur, leaving through 1 - end
        chain, d = [], 1 - end
        cur_seen = set()
        while True:
            chain.append((cur, 1 if d == 1 else -1))
            cur_seen.add(cur)
            n = segs[cur]["ends"][d]
            if not mergeable(n):
                break
            nxt = [e for e in at[n] if e[0] != cur][0]
            if nxt[0] in cur_seen:
                break
            cur, d = nxt[0], 1 - nxt[1]
        pid = len(pieces)
        polys = []
        chains_of = defaultdict(set)
        for s, dirn in chain:
            for li, cs in seg_chains[s].items():
                chains_of[li] |= cs
            piece_of[s] = (pid, dirn)
            p = segs[s]["poly"]
            polys.append(p if dirn == 1 else p[::-1])
        first, last = chain[0], chain[-1]
        ends = (segs[first[0]]["ends"][0 if first[1] == 1 else 1], segs[last[0]]["ends"][1 if last[1] == 1 else 0])
        pieces.append({"ends": ends, "lines": segs[sid]["lines"], "poly": np.concatenate(polys), "kind": segs[sid]["kind"],
                       "tunnel": segs[sid]["tunnel"], "off": False, "chains": chains_of})
    free_of = {(f["line"], f["hop"]): fi for fi, f in enumerate(free)}
    for fi, f in enumerate(free):
        piece_of[("f", fi)] = (len(pieces), 1)
        pieces.append({"ends": f["ends"], "lines": f["lines"], "poly": f["poly"], "kind": f["kind"], "off": True})

    # each line's path over pieces, and its stops as nodes
    paths = []
    for li, its in enumerate(items):
        path = []
        r = 0
        for it in its:
            if it[0] == "free":
                seq = [piece_of[("f", free_of[(li, it[1])])]]
            else:
                m = it[1]
                ca = cut_of[m][(li, r, "a")]
                cb = cut_of[m][(li, r, "b")]
                rng = range(ca, cb) if cb >= ca else range(ca - 1, cb - 1, -1)
                seq = []
                for i in rng:
                    if (m, i) not in piece_of:
                        continue
                    pid, dirn = piece_of[(m, i)]
                    seq.append((pid, dirn * (1 if cb >= ca else -1)))
            r += 1
            for p in seq:
                # the same piece back the other way is a line turning back at a station (the GWR's
                # Gatwick trains at Redhill)
                if not path or path[-1] != p:
                    path.append(p)
        paths.append(path)

    # OSM's track count beside each piece (all chains of the graph in the box, same family)
    allp, allh, allc, allf = [], [], [], []
    for c in range(len(graph.pts)):
        a, p, h = chain_samples(graph, c)
        allp.append(p)
        allh.append(h)
        allc.append(np.full(len(a), c))
    allp, allh, allc = np.concatenate(allp), np.concatenate(allh), np.concatenate(allc)
    alltree = cKDTree(allp)
    for P in pieces:
        if P["off"]:
            P["tracks"] = P["tracks_wide"] = 1
            continue
        line = P["poly"]
        L = length(line)
        ds = np.arange(min(30.0, L / 2), max(L - 30.0, L / 2) + 1e-6, TRACK_STEP)
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
        counts, wide = [], []
        for d in ds:
            q = np.array([np.interp(d, cum, line[:, 0]), np.interp(d, cum, line[:, 1])])
            d0, d1 = max(d - 5, 0), min(d + 5, L)
            hd = math.atan2(np.interp(d1, cum, line[:, 1]) - np.interp(d0, cum, line[:, 1]), np.interp(d1, cum, line[:, 0]) - np.interp(d0, cum, line[:, 0]))
            cs, cw = set(), set()
            for j in alltree.query_ball_point(q, WIDE):
                if graph.fam[allc[j]] == P["kind"] and bool(graph.flags[allc[j]] & 4) == P["tunnel"] and parallel(allh[j], hd):
                    cw.add(int(allc[j]))
                    if math.hypot(*(allp[j] - q)) <= NEAR:
                        cs.add(int(allc[j]))
            counts.append(len(cs))
            wide.append(len(cw))
        P["tracks"] = int(np.median(counts)) if counts else 1
        P["tracks_wide"] = int(np.median(wide)) if wide else 1
        P["km"] = L / 1000
    for P in pieces:
        P.setdefault("km", length(P["poly"]) / 1000)

    # per line: which pieces it is on, where it stops, the stop-to-stop span over each piece
    on = defaultdict(dict)  # piece -> line -> position in path
    span = {}
    hop_of = {}  # the stations either side of the line's stop-to-stop run over the piece
    for li, path in enumerate(paths):
        stops = set(stop_node[li].values())
        nodes_seq = []
        for pid, dirn in path:
            e = pieces[pid]["ends"]
            nodes_seq.append((e[0], e[1]) if dirn == 1 else (e[1], e[0]))
        cum = [0.0]
        for pid, _ in path:
            cum.append(cum[-1] + pieces[pid]["km"] * 1000)
        stop_at = [0] + [i + 1 for i in range(len(path)) if nodes_seq[i][1] in stops]
        if stop_at[-1] != len(path):
            stop_at.append(len(path))
        for i, (pid, _) in enumerate(path):
            on[pid][li] = i
            lo = max(s for s in stop_at if s <= i)
            hi = min(s for s in stop_at if s > i)
            span[(pid, li)] = cum[hi] - cum[lo]
            hop_of[(pid, li)] = frozenset((nodes_seq[lo][0], nodes_seq[hi - 1][1]))
        paths[li] = (path, nodes_seq, stops)

    def passes(pid, li, end_node):
        """The line runs on through this end of the piece without stopping."""
        path, ns, stops = paths[li]
        return end_node not in stops and end_node not in (ns[0][0], ns[-1][1])

    def stops_at(li, n):
        return n in paths[li][2]

    tph = [l["tph"][0] for l in lines]
    s_of = {k: SECTION_BASE_S + v_reach(v) / SIGNAL_BRAKE for k, v in span.items()}

    def load(pid, ls):
        return sum(tph[li] * s_of[(pid, li)] for li in ls) / 3600

    def pack_load(pid):
        ls = pieces[pid]["lines"]
        return PACK_LOAD_RAIL if ls and all(lines[li].get("kind") == "commuter" for li in ls) else PACK_LOAD

    def units(pid, ls):
        """A route's patterns that stop alike here (the A's two branches) keep to one pair."""
        by = defaultdict(list)
        for li in sorted(ls):
            by[(lines[li]["feed"], lines[li].get("route", lines[li]["name"]), hop_of[(pid, li)],
                tuple(passes(pid, li, n) for n in pieces[pid]["ends"]))].append(li)
        return sorted(by.values(), key=lambda g: (-load(pid, g), g))

    def is_light(li):
        return lines[li].get("kind") in ("light_rail", "tram")

    def apart(P, ls):
        """Sets of the piece's lines that never run on one graph chain: OSM shows them on tracks of
        their own, however close (Victoria and Piccadilly at Finsbury Park). Light rail and trams
        never share with other kinds, even where the router laid them on one chain (the DLR beside c2c)."""
        u = Union()
        by_chain = {}
        for li in ls:
            u.find(li)
            light = lines[li].get("kind") in ("light_rail", "tram")
            for c in P.get("chains", {}).get(li, ()):
                k = (c, light)
                if k in by_chain:
                    u.join(by_chain[k], li)
                else:
                    by_chain[k] = li
        return len({u.find(li) for li in ls})

    def pack(pid, ls, cap, limit=None):
        """As few pairs as first fit by load at most `limit` needs (at most `cap`), the lines then
        spread evenly over them (busiest first onto the least loaded): a first pair filled to 0.9
        of the estimate is full in the game, which brakes for curves and stations the estimate
        leaves out (SWR at Clapham Junction, T-1774)."""
        limit = pack_load(pid) if limit is None else limit
        bins = []
        us = []
        for g in units(pid, ls):
            # a unit too heavy for one pair is its patterns again (a London operator's stopping
            # patterns all count as one route: SWR's ten fast patterns out of Waterloo)
            us += [[li] for li in g] if len(g) > 1 and load(pid, g) > limit and cap > 1 else [g]
        us.sort(key=lambda g: (-load(pid, g), g))
        for g in us:
            lg = load(pid, g)
            fit = [b for b in bins if load(pid, b) + lg <= limit]
            if fit:
                fit[0].extend(g)
            elif len(bins) < cap:
                bins.append(list(g))
            else:
                min(bins, key=lambda b: load(pid, b)).extend(g)
        if len(bins) > 1:
            even = [[] for _ in bins]
            for g in us:
                min(even, key=lambda b: load(pid, b)).extend(g)
            if max(load(pid, b) for b in even) <= max(load(pid, b) for b in bins) + 1e-9:
                bins = even
        return bins or [[]]

    # pairs: lines packed into double tracks by load, as many as the trains need and OSM's tracks
    # within WIDE allow; where some lines pass a station others stop at (and OSM shows 3 or more
    # tracks within NEAR), the passing lines (express) and the stopping ones (local) apart
    def fresh_split(pid, P, ls):
        """The piece's lines packed into pairs (role, lines) within its cap."""
        if P["split"]:
            exb = pack(pid, [li for li in ls if li in P["ex"]], P["cap"] - 1)
            lob = pack(pid, [li for li in ls if li not in P["ex"]], P["cap"] - len(exb))
            return [("express", b) for b in exb] + [("local", b) for b in lob]
        else:
            lt = [li for li in ls if lines[li].get("kind") in ("light_rail", "tram")]
            if 0 < len(lt) < len(ls) and P["cap"] >= 2:
                lb = pack(pid, lt, P["cap"] - 1)
                bins = lb + pack(pid, [li for li in ls if li not in lt], P["cap"] - len(lb))
            else:
                bins = pack(pid, ls, P["cap"])
            if len(bins) == 1 and P["want"] and P["tracks"] >= 3:
                # past WANT_LOAD on one pair: two, split about evenly
                bins = pack(pid, ls, 2, P["load1"] / 2)
            return [("", b) for b in bins]

    for pid, P in enumerate(pieces):
        ls = [li for li in P["lines"] if li in on[pid]]
        P["lines"] = ls
        skip = False
        for n in P["ends"]:
            st = [li for li in ls if stops_at(li, n)]
            ps = [li for li in ls if passes(pid, li, n)]
            if st and ps:
                skip = True
        P["skip"] = skip
        P["load1"] = load(pid, ls)
        P["want"] = (skip or P["load1"] > WANT_LOAD) and len(ls) >= 2 and not P["off"]
        P["cap"] = 1 if P["off"] or len(ls) < 2 else max(1, (max(P["tracks"], P["tracks_wide"]) + 1) // 2, apart(P, ls))
        P["ex"] = {li for li in ls if any(passes(pid, li, n) for n in P["ends"])}
        P["split"] = skip and P["tracks"] >= 3 and P["cap"] >= 2 and 0 < len(P["ex"]) < len(ls) and not P["off"]
        P["fresh"] = fresh_split(pid, P, ls)
        if not P["off"] and any(load(pid, b) > APART_LOAD and len(units(pid, b)) > 1 for _, b in P["fresh"]):
            # lines of different routes, or between different stations, loading a pair past anything
            # real service does run on tracks of their own, though the router laid them on one chain
            # or OSM's count missed theirs (Victoria and Piccadilly at Finsbury Park, the Northern's
            # two branches at Euston, Tokyo station's platforms lying wider than WIDE)
            P["cap"] = max(P["cap"], len(units(pid, ls)))
            P["apart"] = True
            P["fresh"] = fresh_split(pid, P, ls)

    def neighbours(pid, li):
        path = paths[li][0]
        i = on[pid][li]
        return [path[j][0] for j in (i - 1, i + 1) if 0 <= j < len(path)]

    def inherit(src, q):
        """Piece src's split into pairs carried onto its neighbour q, if q's trains fit it: lines
        keep their pair along a run, so a plain cut needs no crossover."""
        Q = pieces[q]
        qs = set(Q["lines"])
        cand = [(r, [li for li in b if li in qs]) for r, b in bins_of[src]]
        cand = [(r, b) for r, b in cand if b]
        have = {li for _, b in cand for li in b}
        for g in units(q, [li for li in Q["lines"] if li not in have]):
            lg = load(q, g)
            fit = [b for _, b in cand if load(q, b) + lg <= pack_load(q) and is_light(b[0]) == is_light(g[0])]
            if fit:
                fit[0].extend(g)
            else:
                cand.append(("", list(g)))
        if len(cand) > Q["cap"] or len(cand) < 2:
            return None
        if any(len({is_light(li) for li in b}) > 1 for _, b in cand):
            return None
        if max(load(q, b) for _, b in cand) > max(pack_load(q), max(load(q, b) for _, b in Q["fresh"])) + 1e-9:
            return None
        if Q["split"]:
            roles = []
            for _, b in cand:
                e = sum(li in Q["ex"] for li in b)
                if 0 < e < len(b):
                    return None
                roles.append("express" if e else "local")
            cand = [(r, b) for r, (_, b) in zip(roles, cand)]
        return cand

    # express and local pieces first, then the rest, most loaded first: a piece's split goes on to
    # its neighbours where their trains fit it
    bins_of = {}
    for start in sorted(range(len(pieces)), key=lambda pid: (not pieces[pid]["split"], -pieces[pid]["load1"])):
        if start in bins_of or len(pieces[start]["fresh"]) < 2:
            continue
        bins_of[start] = pieces[start]["fresh"]
        todo = [start]
        while todo:
            p = todo.pop()
            for li in pieces[p]["lines"]:
                for q in neighbours(p, li):
                    if q in bins_of or len(pieces[q]["fresh"]) < 2:
                        continue
                    bins_of[q] = inherit(p, q) or pieces[q]["fresh"]
                    todo.append(q)
    for pid, P in enumerate(pieces):
        if pid not in bins_of and len(P["fresh"]) < 2:
            bins_of[pid] = P["fresh"]
    # a one-pair piece whose lines all run on split pieces either side takes their split, where
    # OSM shows 3 or more tracks
    for _ in range(5):
        changed = False
        for pid, P in enumerate(pieces):
            if len(bins_of[pid]) >= 2 or P["tracks"] < 3 or len(P["lines"]) < 2 or P["off"]:
                continue
            ok, sides, src = True, [0, 0], None
            for li in P["lines"]:
                path = paths[li][0]
                i = on[pid][li]
                for side, j in enumerate((i - 1, i + 1)):
                    if 0 <= j < len(path):
                        if len(bins_of[path[j][0]]) < 2:
                            ok = False
                        src = src if src is not None else path[j][0]
                        sides[side] += 1
            if ok and all(sides):
                got = inherit(src, pid)
                if got:
                    bins_of[pid] = got
                    P["smoothed"] = True
                    changed = True
        if not changed:
            break
    pair = {}
    for pid, P in enumerate(pieces):
        bins = bins_of[pid]
        P["pairs"] = len(bins)
        P["role"] = [r for r, _ in bins]
        P["pair_lines"] = [sorted(b) for _, b in bins]
        for k, (_, b) in enumerate(bins):
            for li in b:
                pair[(pid, li)] = k
        P["load"] = [round(load(pid, ls), 3) for ls in P["pair_lines"]]

    split_by_hand(city, o, pieces, paths, pair, s_of, span, lines, load)
    if avoid is None:
        full = set()
        for pid, P in enumerate(pieces):
            if P["off"]:
                continue
            for ls in P["pair_lines"]:
                if load(pid, ls) > MAX_LOAD and len({(lines[li]["feed"], lines[li].get("route", lines[li]["name"])) for li in ls}) > 1:
                    full |= {c for li in ls for c in P["chains"].get(li, ())}
        if full:
            print(f"pieces over {MAX_LOAD} with more than one route: {len(full)} chains, routed again")
            return build(city, out_dir, frozenset(full))
    trim = trim_lines(pieces, paths, pair, s_of, lines)

    # nodes
    node_ids = {}
    for P in pieces:
        for n in P["ends"]:
            node_ids.setdefault(n, len(node_ids))
    for sn in stop_node:
        for n in sn.values():
            node_ids.setdefault(n, len(node_ids))
    node_xy = defaultdict(list)
    for P in pieces:
        node_xy[P["ends"][0]].append(P["poly"][0])
        node_xy[P["ends"][1]].append(P["poly"][-1])
    deg = Counter(n for P in pieces for n in P["ends"])
    gaps = [float(np.max(np.hypot(*(np.array(v) - np.mean(v, 0)).T))) for v in node_xy.values()]
    node_stops = defaultdict(lambda: defaultdict(list))
    switches = 0
    for li, (path, ns, stops) in enumerate(paths):
        for i, (pid, dirn) in enumerate(path):
            n_in, n_out = ns[i]
            if i == 0 and n_in in stops:
                node_stops[n_in][(pid, pair[(pid, li)])].append(li)
            if n_out in stops:
                node_stops[n_out][(pid, pair[(pid, li)])].append(li)
            if i + 1 < len(path):
                q = path[i + 1][0]
                if pieces[pid]["pairs"] >= 2 and pieces[q]["pairs"] >= 2 and pair[(pid, li)] != pair[(q, li)] and deg[n_out] == 2:
                    switches += 1
    # platform loads: per node and pair of the piece a line arrives on
    plat_max = defaultdict(float)
    for li, (path, ns, stops) in enumerate(paths):
        for i, (pid, dirn) in enumerate(path):
            n = ns[i][1]
            if i + 1 >= len(path):
                continue
            s = DWELL_S + PLATFORM_MARGIN_S if n in stops else PASSING_S
            plat_max[(n, pid, pair[(pid, li)])] += tph[li] * s / 3600

    def kind_of_node(n):
        if n in station:
            return "station"
        return "junction" if deg[n] >= 3 else ("end" if deg[n] == 1 else "through")

    out_nodes = []
    for n, i in sorted(node_ids.items(), key=lambda kv: kv[1]):
        v = node_xy.get(n)
        if v is None:
            v = [realnet_st_xy(net, n[1]) if n[0] == "s" else graph.point_at(n[1], cut_pos[n[1]][n[2]])]
        out_nodes.append({"id": i, "xy": np.round(np.mean(v, 0), 1).tolist(), "kind": kind_of_node(n),
                          "stops": [[pid, p, sorted(ls)] for (pid, p), ls in sorted(node_stops[n].items())]})
    out_pieces = []
    for pid, P in enumerate(pieces):
        out_pieces.append({"id": pid, "line": realnet.simplify(P["poly"], 1.0).round(1).tolist(), "km": round(P["km"], 3),
                           "pairs": P["pairs"], "tracks": P["tracks"], "lines": P["pair_lines"], "role": P["role"],
                           "ends": [node_ids[n] for n in P["ends"]], "load": P["load"], "off_graph": P["off"]})
    out_lines = []
    for li, (path, ns, stops) in enumerate(paths):
        f = trim.get(li, [1.0, 1.0, 1.0])
        out_lines.append({"index": li, "name": lines[li]["name"], "feed": lines[li]["feed"],
                          "tph": [round(t * k, 3) for t, k in zip(lines[li]["tph"], f)], "trimmed": li in trim,
                          "path": [[pid, dirn, pair[(pid, li)]] for pid, dirn in path],
                          "occ": [round(s_of[(pid, li)], 1) for pid, _ in path],
                          "stops": [node_ids[stop_node[li][k]] for k in range(len(lines[li]["stops"]))]})

    # ---- checks ----
    sub = {li for li, l in enumerate(lines) if l["feed"] == "subway"}
    km_all = sum(P["km"] for P in pieces)
    km_sub = sum(P["km"] for P in pieces if P["lines"] and set(P["lines"]) <= sub)
    worst = sorted(((max(P["load"]), pid) for pid, P in enumerate(pieces) if P["lines"] and set(P["lines"]) <= sub), reverse=True)
    t_all = time.time() - t0
    stats = {
        "pieces": len(pieces), "two_pair_pieces": sum(P["pairs"] >= 2 for P in pieces), "three_plus_pair_pieces": sum(P["pairs"] >= 3 for P in pieces),
        "two_pair_km": round(sum(P["km"] for P in pieces if P["pairs"] >= 2), 1), "pair_km": round(sum(P["km"] * P["pairs"] for P in pieces), 1),
        "want_two_but_osm_shows_under_3": sum(1 for P in pieces if P["want"] and P["tracks"] < 3),
        "off_graph_pieces": sum(P["off"] for P in pieces), "hops_off_graph": dict(why),
        "nodes": len(node_ids), "station_nodes": sum(1 for n in node_ids if n in station),
        "junctions": sum(1 for n in node_ids if n not in station and deg[n] >= 3),
        "node_gap_m_p50_p95_max": [round(float(np.percentile(gaps, q)), 1) for q in (50, 95, 100)],
        "pair_switches_at_plain_cuts": switches,
        "corridor_km": round(km_all, 1), "corridor_km_subway": round(km_sub, 1),
        "line_km": round(sum(pieces[pid]["km"] for path, _, _ in paths for pid, _ in path), 1),
        "subway_worst_section_load": [[round(w, 3), pid, [lines[li]["name"] for ls in pieces[pid]["pair_lines"] for li in ls]] for w, pid in worst[:8]],
        "subway_pieces_over_90": sum(1 for w, _ in worst if w > MAX_LOAD),
        "subway_pieces_over_90_at_88kmh": [[pid, round(w, 3)] for pid, w in (
            (pid, max(sum(tph[li] * (SECTION_BASE_S + min(v_reach(span[(pid, li)]), V_REAL_METRO) / SIGNAL_BRAKE) for li in ls) / 3600
                      for ls in pieces[pid]["pair_lines"])) for _, pid in worst) if w > MAX_LOAD],
        "subway_worst_platform_load": round(max((v for (n, pid, p), v in plat_max.items() if set(pieces[pid]["lines"]) <= sub), default=0), 3),
        "seconds": {"graph": round(t_graph, 1), "route": round(t_route, 1), "all": round(t_all, 1)},
    }
    trunks = trunk_table(city, pieces, lines)
    stats["trunks"] = trunks
    doc = {"city": city, "stats": stats, "pieces": out_pieces, "nodes": out_nodes, "lines": out_lines}
    doc["model"] = track_model(doc, [P["poly"] for P in pieces], net)
    stats["model"] = doc["model"]["stats"]
    # T-1773: each node and stretch of edge on its real level from OSM's layer, tunnel and bridge tags
    stats["levels"] = assign_levels(doc["model"], graph)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{city}_real_corridors.json")
    with open(p, "w") as f:
        json.dump(doc, f)
    print(json.dumps({k: v for k, v in stats.items() if k != "trunks"}, indent=1))
    for t in trunks:
        print(t)
    print(f"-> {p}")
    picture(city, o, pieces, lines, os.path.join(out_dir, f"{city}_corridors.png"))


# Pieces the graph cannot tell apart, split by hand (T-1305 stage 2): stacked lines both in tunnel
# and joined by track. W 4 St: the IND's 8th Av (upper) and 6th Av (lower) levels.
SPLITS = {"nyc": [{"at": (-74.0003, 40.7322), "r": 250.0, "groups": [{"A", "C", "E"}, {"B", "D", "F", "FX", "M"}]}]}


def split_by_hand(city, o, pieces, paths, pair, s_of, span, lines, load):
    for sp in SPLITS.get(city, []):
        cx, cy = realnet.to_xy(*sp["at"], o)
        for pid in range(len(pieces)):
            P = pieces[pid]
            if P["off"] or np.hypot(*(P["poly"].mean(0) - (float(cx), float(cy)))) > sp["r"]:
                continue
            groups = [[li for li in P["lines"] if lines[li]["name"] in g] for g in sp["groups"]]
            groups = [g for g in groups if g]
            if len(groups) < 2 or sum(map(len, groups)) != len(P["lines"]):
                continue
            for gi, ls in enumerate(groups):
                q = pid if gi == 0 else len(pieces)
                Q = dict(P) if gi else P
                if gi:
                    pieces.append(Q)
                # each line's pair from the piece next to this one on its path
                got = {}
                for li in ls:
                    path = paths[li][0]
                    i = next(i for i, e in enumerate(path) if e[0] == pid)
                    nb = [path[j][0] for j in (i - 1, i + 1) if 0 <= j < len(path)]
                    ks = [pair[(r, li)] for r in nb if (r, li) in pair and pieces[r]["pairs"] >= 2]
                    got[li] = Counter(ks).most_common(1)[0][0] if ks else 0
                    if q != pid:
                        path[i] = (q, path[i][1])
                        s_of[(q, li)], span[(q, li)] = s_of[(pid, li)], span[(pid, li)]
                ks = sorted(set(got.values()))
                two = len(ks) >= 2 and P["tracks"] >= 3
                for li in ls:
                    pair[(q, li)] = ks.index(got[li]) if two else 0
                Q["lines"] = ls
                Q["pairs"] = len(ks) if two else 1
                Q["role"] = (["express", "local"] + [""] * (len(ks) - 2)) if two else [""]
                Q["pair_lines"] = [sorted(li for li in ls if pair[(q, li)] == k) for k in range(Q["pairs"])]
                Q["load"] = [round(load(q, g), 3) for g in Q["pair_lines"]]
            print(f"split by hand: piece {pid} into {len(groups)} ({', '.join('/'.join(sorted({lines[li]['name'] for li in g})) for g in groups)})")


def trim_lines(pieces, paths, pair, s_of, lines):
    """The note's trim: a pair loaded past MAX_LOAD at a demand level has its lines' trains cut in
    proportion; a line keeps the smallest factor of any pair it runs on. Returns {line: [f, f, f]}."""
    out, merged = {}, {}
    for pid, P in enumerate(pieces):
        for ls in P["pair_lines"]:
            # a pair of one route's patterns is the line's own load; of several, a merge's
            routes = len({(lines[li]["feed"], lines[li].get("route", lines[li]["name"])) for li in ls})
            for lev in range(3):
                rho = sum(lines[li]["tph"][lev] * s_of[(pid, li)] for li in ls) / 3600
                if rho > MAX_LOAD:
                    f = MAX_LOAD / rho
                    for li in ls:
                        out.setdefault(li, [1.0, 1.0, 1.0])
                        if f < out[li][lev]:
                            out[li][lev] = f
                            if lev == 0:
                                merged[li] = routes > 1
    for li, f in sorted(out.items()):
        print(f"trim {lines[li]['feed']} {lines[li]['name']} (line {li}): trains an hour {lines[li]['tph']} x {[round(x, 3) for x in f]}"
              f"{' (merge)' if merged.get(li) else ''}")
    worst = min((out[li][0] for li in out if merged.get(li)), default=1.0)
    print(f"trimmed lines {len(out)}, worst {min((f[0] for f in out.values()), default=1.0):.3f}, worst by a merge {worst:.3f}")
    return out


def realnet_st_xy(net, st):
    s = net["stations"][st]
    return [s["x"], s["y"]]


# T-1305's done-when: these trunks two pairs, these lines express and local
# (8th Av: north of Fulton St, plane y; in Brooklyn the A and C share the Cranberry St tunnel)
TRUNKS = {"nyc": [("Lexington Av", {"4", "5"}, {"6", "6X"}, None), ("7th Av", {"2", "3"}, {"1"}, None),
                  ("Queens Blvd", {"E", "F"}, {"M", "R"}, None), ("8th Av", {"A"}, {"C"}, -5000.0)]}


def trunk_table(city, pieces, lines):
    rows = []
    for name, ex, lo, ymin in TRUNKS.get(city, []):
        km = {"right": 0.0, "one pair": 0.0, "wrong": 0.0}
        for P in pieces:
            names = {lines[li]["name"] for li in P["lines"] if lines[li]["feed"] == "subway"}
            if not names >= ex | lo or (ymin is not None and P["poly"][:, 1].mean() < ymin):
                continue
            if P["pairs"] == 1:
                km["one pair"] += P["km"]
                continue
            e = {lines[li]["name"] for li in P["pair_lines"][0]}
            l = {lines[li]["name"] for li in P["pair_lines"][1]}
            km["right" if not (e & lo) and not (l & ex) else "wrong"] += P["km"]
        rows.append((name, {k: round(v, 2) for k, v in km.items()}))
    return rows


def picture(city, o, pieces, lines, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if city == "nyc":
        x0, y0 = realnet.to_xy(-74.05, 40.68, o)
        x1, y1 = realnet.to_xy(-73.85, 40.885, o)
    else:
        # the centre of the network by line km, 17 by 23 km as Manhattan's
        mid = np.array([P["poly"].mean(0) for P in pieces])
        w = np.array([P["km"] * max(1, len(P["lines"])) for P in pieces])
        c = np.array([np.average(mid[:, 0], weights=w), np.average(mid[:, 1], weights=w)])
        x0, y0, x1, y1 = c[0] - 8500, c[1] - 11500, c[0] + 8500, c[1] + 11500
    fig, ax = plt.subplots(figsize=(9, 12), dpi=130)
    ax.set_facecolor("#f4f4f2")
    for P in sorted(pieces, key=lambda P: P["pairs"]):
        q = P["poly"]
        sub = all(lines[li]["feed"] == "subway" for li in P["lines"])
        if P["off"]:
            c, w = "#bbbbbb", 0.8
        elif P["pairs"] >= 2:
            c, w = "#d9480f", 2.6
        else:
            c, w = ("#1c4f8a" if sub else "#7aa0c4"), 1.2
        ax.plot(q[:, 0], q[:, 1], color=c, lw=w, solid_capstyle="round")
    ax.set_xlim(float(x0), float(x1))
    ax.set_ylim(float(y0), float(y1))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color="#d9480f", lw=2.6, label="two pairs"),
                       Line2D([], [], color="#1c4f8a", lw=1.2, label="one pair, subway"),
                       Line2D([], [], color="#7aa0c4", lw=1.2, label="one pair, other"),
                       Line2D([], [], color="#bbbbbb", lw=0.8, label="off the graph")], loc="upper left")
    ax.set_title(f"{city}: shared-track pieces by pairs (T-1305)")
    fig.tight_layout()
    fig.savefig(path)
    print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city")
    ap.add_argument("--out", default=realnet.OUT)
    a = ap.parse_args()
    build(a.city, a.out)


if __name__ == "__main__":
    main()
