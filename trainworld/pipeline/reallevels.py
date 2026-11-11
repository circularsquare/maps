"""T-1773: a shared real network's levels from OSM's tags, for `realcorridors.py` (notes/T-1773.md).

Every edge of the track model (`realshared.track_model`) is sampled every STEP m along its line; each
sample takes the tag level of the nearest real-network chain within NEAR m running the same way:
OSM's `layer` where set (the graph's flags carry it, `realgraph.layer_of`), else -1 in a tunnel and
+1 on a bridge, else 0; a tunnel never above -1, a bridge never below +1; clamped to the game's -3..3
(layer is an order at a crossing, not a depth: clamping keeps the order except below -3 and above 3).
A node takes the commonest level of the samples within NODE_R m of it on its edges (0 where none).
Along an edge, the stretch past each end's lead and platform (half of it, all at a terminus, and 10 m)
is at that end's node level. Then, repeated until nothing changes:

- a stretch shorter than SHORT_M between two stretches of one other level takes theirs (a road bridge,
  a culvert, a short cut), unless another edge crosses it there;
- a stretch too short for its ramps (RAMP_M a level, centred where the level changes) takes the level
  of its longer neighbour; one next to an end takes the end's;
- of two nodes an edge is too short to ramp between (its flat ends and RAMP_M a level), the less sure
  moves towards the other's level until the change fits (sureness: its samples, a station's counting
  STATION_VOTE times); what that cannot settle goes on one level per group of such nodes.

Writes each node's `level`, each edge's `levels`: [[from m, to m, level], ...] along the edge's line
from node a (a, lead a, line, lead b, b; the whole edge at one level when it is one stretch), and its
`flat`: [m at a, m at b] that must stay level, which `realsave --shared` lays its corners and ramps on. Every edge stays `sep` (grade-separated from other
imported track whatever the levels, T-1751), so a crossing the levels miss is no flat crossing."""
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

from realgraph import layer_of

STEP = 10.0
NEAR = 15.0
PARALLEL_COS = math.cos(math.radians(25))
NODE_R = 60.0
SHORT_M = 400.0
RAMP_M = 200.0
# kept between a flat end and its ramp: realsave lays ramps by the alignment's chainage, a little
# shorter than the line's round curves
SLACK_M = 20.0
DENSE = 5.0
# a station's platform level is the surest: its samples count this many times a junction's
STATION_VOTE = 10
FOLLOW_ROUNDS = 20
LEVELS = (-3, 3)


def tag_level(flags):
    lay = layer_of(flags)
    tun, bri = bool(flags & 4), bool(flags & 2)
    lv = lay if lay else (-1 if tun else (1 if bri else 0))
    if tun and lv >= 0:
        lv = -1
    if bri and lv <= 0:
        lv = 1
    return max(LEVELS[0], min(LEVELS[1], lv))


def edge_line(model, e):
    """The edge's line from node a to node b, plane metres."""
    A, B = model["nodes"][e["a"]]["xy"], model["nodes"][e["b"]]["xy"]
    pts = [A]
    if e.get("lead_a") is not None:
        pts += [e["lead_a"], *e["line"], e["lead_b"]]
    pts.append(B)
    p = np.asarray(pts, float)
    keep = np.concatenate([[True], np.hypot(*np.diff(p, axis=0).T) > 1e-6])
    return p[keep] if keep.sum() >= 2 else p[[0, -1]]


def resample(p, step):
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(math.ceil(cum[-1] / step)) + 1)
    s = np.linspace(0, cum[-1], n)
    x, y = np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    d = np.diff(p, axis=0)[i]
    h = d / np.maximum(np.hypot(*d.T), 1e-9)[:, None]
    return s, np.stack([x, y], 1), h, cum[-1]


def graph_points(graph):
    """Every chain densified to DENSE m: points, unit headings, tag levels."""
    P, H, L = [], [], []
    for c, p in enumerate(graph.pts):
        s, q, h, _ = resample(p, DENSE)
        P.append(q)
        H.append(h)
        L.append(np.full(len(q), tag_level(graph.flags[c]), np.int8))
    return np.concatenate(P), np.concatenate(H), np.concatenate(L)


def runs_of(lv):
    """[(i0, i1, level)] over sample indices, i1 exclusive."""
    out, i0 = [], 0
    for i in range(1, len(lv) + 1):
        if i == len(lv) or lv[i] != lv[i0]:
            out.append([i0, i, int(lv[i0])])
            i0 = i
    return out


def assign(model, graph):
    nodes, edges = model["nodes"], model["edges"]
    gp, gh, gl = graph_points(graph)
    tree = cKDTree(gp)
    lines = [edge_line(model, e) for e in edges]
    samp = [resample(p, STEP) for p in lines]

    # tag level per sample: the nearest parallel chain point within NEAR
    raw = []
    for s, q, h, L in samp:
        d, j = tree.query(q, k=8, distance_upper_bound=NEAR)
        lv = np.full(len(q), -99, np.int16)
        for k in range(8):
            ok = (lv == -99) & np.isfinite(d[:, k])
            if not ok.any():
                continue
            jj = np.where(ok, j[:, k], 0)
            par = np.abs((gh[jj] * h).sum(1)) >= PARALLEL_COS
            take = ok & par
            lv[take] = gl[jj[take]]
        raw.append(lv)

    # node levels: the commonest sample level within NODE_R of the node on its edges
    # (where none, the nearest known sample along its edges; where its edges have none, its neighbours')
    near = defaultdict(Counter)
    nearest = {}
    deg = Counter()
    nbr = defaultdict(set)
    for ei, e in enumerate(edges):
        s, _, _, L = samp[ei]
        lv = raw[ei]
        nbr[e["a"]].add(e["b"])
        nbr[e["b"]].add(e["a"])
        for end, node in ((0, e["a"]), (1, e["b"])):
            deg[node] += 1
            m = (s <= NODE_R) if end == 0 else (s >= L - NODE_R)
            for v in lv[m & (lv != -99)]:
                near[node][int(v)] += 1
            known = np.flatnonzero(lv != -99)
            if known.size:
                k = known[0] if end == 0 else known[-1]
                d = s[k] if end == 0 else L - s[k]
                if d < nearest.get(node, (math.inf, 0))[0]:
                    nearest[node] = (d, int(lv[k]))
    level = [near[i].most_common(1)[0][0] if near[i] else nearest[i][1] if i in nearest else None for i in range(len(nodes))]
    for _ in range(len(nodes)):
        todo = [i for i, v in enumerate(level) if v is None]
        if not todo:
            break
        done = False
        for i in todo:
            vals = Counter(level[j] for j in nbr[i] if level[j] is not None)
            if vals:
                level[i] = vals.most_common(1)[0][0]
                done = True
        if not done:
            break
    level = [0 if v is None else v for v in level]

    # each end's flat stretch: the platform's reach along the edge (the sim's `platform_reach`: half
    # of it, all at a terminus) and 10 m, or the lead; a platform reaching past the short edge to the
    # node carries on into that node's other edges
    def reach(ni):
        st = nodes[ni].get("station")
        return (st["platform"] if deg[ni] <= 1 else st["platform"] / 2) + 10 if st else 0.0
    at_node = defaultdict(list)
    for ei, e in enumerate(edges):
        at_node[e["a"]].append((ei, e["b"]))
        at_node[e["b"]].append((ei, e["a"]))

    def flat_at(ei, end):
        e = edges[ei]
        ni = e["a"] if end == 0 else e["b"]
        node = nodes[ni]
        f = reach(ni)
        for ej, other in at_node[ni]:
            if ej != ei:
                f = max(f, reach(other) - samp[ej][3])
        if e.get("lead_a") is not None:
            A = np.asarray(node["xy"], float)
            lead = np.asarray(e["lead_a"] if end == 0 else e["lead_b"], float)
            f = max(f, float(np.hypot(*(lead - A))))
        return f

    # crossings between edges that share no node, as arc positions on each
    from shapely.geometry import LineString
    from shapely.strtree import STRtree
    geoms = [LineString(p) for p in lines]
    stree = STRtree(geoms)
    cross_at = defaultdict(list)
    pairs = []
    for i, g in enumerate(geoms):
        for j in stree.query(g):
            j = int(j)
            if j <= i or {edges[i]["a"], edges[i]["b"]} & {edges[j]["a"], edges[j]["b"]}:
                continue
            x = g.intersection(geoms[j])
            if x.is_empty or x.geom_type not in ("Point", "MultiPoint"):
                continue
            for pt in getattr(x, "geoms", [x]):
                si, sj = g.project(pt), geoms[j].project(pt)
                cross_at[i].append(si)
                cross_at[j].append(sj)
                pairs.append((i, si, j, sj))

    def profile(ei):
        s, _, _, L = samp[ei]
        e = edges[ei]
        lv = raw[ei].copy()
        known = np.flatnonzero(lv != -99)
        if known.size == 0:
            lv[:] = level[e["a"]]
            lv[s > L / 2] = level[e["b"]]
        elif known.size < len(lv):
            # a sample with no chain near takes the nearest known one's level, either side
            idx = np.arange(len(lv))
            pos = np.searchsorted(known, idx)
            lo = known[np.clip(pos - 1, 0, known.size - 1)]
            hi = known[np.clip(pos, 0, known.size - 1)]
            pick = np.where(np.abs(idx - lo) <= np.abs(hi - idx), lo, hi)
            lv = np.where(lv == -99, raw[ei][pick], lv)
        fa, fb = flat_at(ei, 0), flat_at(ei, 1)
        lv = lv.astype(np.int16)
        lv[s <= fa] = level[e["a"]]
        lv[s >= L - fb] = level[e["b"]]
        return s, lv, L, fa, fb

    def settle(ei):
        """The edge's stretches after the short and ramp rules; None if its two nodes cannot be
        joined at all (too short for their change)."""
        s, lv, L, fa, fb = profile(ei)
        da = abs(level[edges[ei]["a"]] - level[edges[ei]["b"]])
        if da and L < fa + fb + 2 * SLACK_M + RAMP_M * da:
            return None
        R = runs_of(lv)
        xs = sorted(cross_at.get(ei, []))
        def span(r):
            return s[min(r[1], len(s) - 1)] - s[r[0]] if r[1] < len(s) else L - s[r[0]]
        changed = True
        while changed and len(R) > 2:
            changed = False
            # short structures between two stretches of one level
            for k in range(1, len(R) - 1):
                a, r, b = R[k - 1], R[k], R[k + 1]
                if a[2] == b[2] != r[2] and span(r) < SHORT_M:
                    s0, s1 = s[r[0]], s[min(r[1], len(s) - 1)]
                    if not any(s0 <= x <= s1 for x in xs):
                        r[2] = a[2]
                        changed = True
            R = merge(R)
            # ramps: each change needs RAMP_M a level, half each side of it
            for k in range(1, len(R) - 1):
                a, r, b = R[k - 1], R[k], R[k + 1]
                need = RAMP_M / 2 * (abs(r[2] - a[2]) + abs(b[2] - r[2]))
                if span(r) < need:
                    if k == 1:
                        r[2] = a[2]
                    elif k == len(R) - 2:
                        r[2] = b[2]
                    else:
                        r[2] = a[2] if span(a) >= span(b) else b[2]
                    changed = True
                    break
            R = merge(R)
        # the end stretches hold their flat part and half a ramp: the change moves into the next
        # stretch while that stays long enough for its own ramps, else that stretch takes the end's level
        def own_need(k):
            if k == 0 or k == len(R) - 1:
                return 0.0
            return RAMP_M / 2 * (abs(R[k][2] - R[k - 1][2]) + abs(R[k + 1][2] - R[k][2]))
        while len(R) >= 2:
            need_a = fa + SLACK_M + RAMP_M / 2 * abs(R[1][2] - R[0][2])
            need_b = fb + SLACK_M + RAMP_M / 2 * abs(R[-1][2] - R[-2][2])
            bad_a, bad_b = span(R[0]) < need_a, span(R[-1]) < need_b
            if not (bad_a or bad_b):
                break
            if bad_a:
                i = int(np.searchsorted(s, need_a))
                k = 1
                if i < R[k][1] and (len(R) == 2 or (s[R[k][1]] if R[k][1] < len(s) else L) - s[i] >= own_need(k)):
                    R[0][1] = R[k][0] = i
                    if len(R) == 2 and span(R[1]) < need_b:
                        return None
                    continue
                if len(R) == 2:
                    return None
                R[1][2] = R[0][2]
            else:
                i = int(np.searchsorted(s, L - need_b, side="right")) - 1
                k = len(R) - 2
                if i > R[k][0] and (len(R) == 2 or s[i] - s[R[k][0]] >= own_need(k)):
                    R[k][1] = R[-1][0] = i
                    if len(R) == 2 and span(R[0]) < need_a:
                        return None
                    continue
                if len(R) == 2:
                    return None
                R[-2][2] = R[-1][2]
            R = merge(R)
        return [[round(float(s[r[0]]), 1), round(float(s[r[1]] if r[1] < len(s) else L), 1), r[2]] for r in R]

    # an edge too short to ramp between its nodes: the less sure node (sureness: its samples, a
    # station's counting STATION_VOTE times) moves towards the other only as far as the edge needs
    # (as many levels as fit stay), so a wrong level does not spread far down a line, and becomes a
    # little surer than the node it followed, so it is not pulled back next round; after
    # FOLLOW_ROUNDS what is left becomes one group on the group's commonest level, sureness
    # weighting (groups only grow, so it ends)
    first = list(level)
    sure = [sum(near[i].values()) * (STATION_VOTE if nodes[i].get("station") else 1) for i in range(len(nodes))]
    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    out = [None] * len(edges)
    rounds = 0
    while True:
        out = [settle(ei) for ei in range(len(edges))]
        bad = [ei for ei, r in enumerate(out) if r is None]
        if not bad:
            break
        rounds += 1
        if os.environ.get("REALLEVELS_DEBUG"):
            print(f"  levels round {rounds}: {len(bad)} edges too short for their nodes, e.g. " + "; ".join(
                f"e{ei} {edges[ei]['a']}({level[edges[ei]['a']]},{sure[edges[ei]['a']]})-{edges[ei]['b']}({level[edges[ei]['b']]},{sure[edges[ei]['b']]}) {samp[ei][3]:.0f} m"
                for ei in bad[:4]), flush=True)
        if rounds <= FOLLOW_ROUNDS:
            for ei in bad:
                a, b = edges[ei]["a"], edges[ei]["b"]
                if level[a] == level[b]:
                    continue
                lo, hi = (a, b) if (sure[a], -a) < (sure[b], -b) else (b, a)
                room = max(0, int((samp[ei][3] - flat_at(ei, 0) - flat_at(ei, 1) - 2 * SLACK_M) // RAMP_M))
                level[lo] = level[hi] + max(-room, min(room, level[lo] - level[hi]))
                sure[lo] = sure[hi] + 1
            continue
        for ei in bad:
            parent[find(edges[ei]["a"])] = find(edges[ei]["b"])
        votes = defaultdict(Counter)
        for i in range(len(nodes)):
            votes[find(i)][level[i]] += sure[i] + 1
        for i in range(len(nodes)):
            level[i] = votes[find(i)].most_common(1)[0][0]
    moved = sum(1 for i in range(len(nodes)) if level[i] != first[i])
    for i, n in enumerate(nodes):
        n["level"] = int(level[i])
    for ei, e in enumerate(edges):
        e["levels"] = out[ei]
        e["flat"] = [round(flat_at(ei, 0), 1), round(flat_at(ei, 1), 1)]

    def tag_at(ei, x):
        return int(raw[ei][min(int(round(x / STEP)), len(raw[ei]) - 1)])
    stats = measure(model, lines, pairs, tag_at)
    if os.environ.get("REALLEVELS_DEBUG"):
        dump = []
        for i, si, j, sj in pairs:
            hi, hj = height_at(edges[i]["levels"], si), height_at(edges[j]["levels"], sj)
            ri, rj = tag_at(i, si), tag_at(j, sj)
            p = np.interp(si, np.concatenate([[0], np.cumsum(np.hypot(*np.diff(lines[i], axis=0).T))]), lines[i][:, 0]), \
                np.interp(si, np.concatenate([[0], np.cumsum(np.hypot(*np.diff(lines[i], axis=0).T))]), lines[i][:, 1])
            dump.append({"i": i, "j": j, "xy": [round(float(p[0])), round(float(p[1]))], "h": [round(hi, 2), round(hj, 2)], "raw": [int(ri), int(rj)],
                         "len": [round(samp[i][3]), round(samp[j][3])], "nodes": [level[edges[i]["a"]], level[edges[i]["b"]], level[edges[j]["a"]], level[edges[j]["b"]]]})
        with open(os.environ["REALLEVELS_DEBUG"], "w") as f:
            json.dump(dump, f)
    stats["nodes moved to a neighbour's level"] = moved
    sizes = Counter(find(i) for i in range(len(nodes)))
    stats["node groups of 2 or more, largest"] = [sum(1 for v in sizes.values() if v > 1), max(sizes.values())]
    return stats


def merge(R):
    out = []
    for r in R:
        if out and out[-1][2] == r[2]:
            out[-1][1] = r[1]
        else:
            out.append(list(r))
    return out


def height_at(levels, x):
    """The level at arc x along an edge with these stretches, ramps of RAMP_M a level centred where
    the level changes (fractional on a ramp)."""
    for k in range(1, len(levels)):
        b = levels[k][0]
        a, c = levels[k - 1][2], levels[k][2]
        half = RAMP_M / 2 * abs(c - a)
        if b - half <= x <= b + half and half > 0:
            return a + (c - a) * (x - (b - half)) / (2 * half)
    for s0, s1, lv in levels:
        if s0 - 1e-6 <= x <= s1 + 1e-6:
            return lv
    return levels[-1][2]


def measure(model, lines, pairs, tag_at):
    """Crossings before (every edge on -1) and after, all of them and those where OSM's tags put the
    two at different levels (the rest are at grade in OSM, or where pairs of one piece cross)."""
    edges = model["edges"]
    def apart(i, si, j, sj):
        return abs(height_at(edges[i]["levels"], si) - height_at(edges[j]["levels"], sj)) >= 1.0
    tagged = [(i, si, j, sj) for i, si, j, sj in pairs if -99 not in (tag_at(i, si), tag_at(j, sj)) and tag_at(i, si) != tag_at(j, sj)]
    km = Counter()
    for e in edges:
        for s0, s1, lv in e["levels"]:
            km[lv] += (s1 - s0) / 1000
    return {"crossings": len(pairs), "flat before (all on -1)": len(pairs),
            "flat after": sum(1 for c in pairs if not apart(*c)),
            "crossings OSM's tags separate": len(tagged), "of those, flat after": sum(1 for c in tagged if not apart(*c)),
            "km by level": {str(k): round(v, 1) for k, v in sorted(km.items())},
            "nodes by level": dict(sorted(Counter(str(n["level"]) for n in model["nodes"]).items(), key=lambda kv: int(kv[0])))}
