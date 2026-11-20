"""T-1305 stage 2: the track model's network from a city's corridors (`realcorridors.py`), which the
sim's `realsave --shared` builds as it is: nodes, edges and each line's stops and path.

Each piece is one double-track edge per pair, the pairs 8 m apart side by side. At each corridor node the piece
ends a line runs between are one group (an express pair the locals never join is its own group,
so an express station has a platform on each pair). A group with stops is a station node (a
station takes two edge ends, one each way), or several side by side where its trains need more
platform tracks than one (PLAT_LOAD: a terminus fed by one pair, London Bridge, Clapham
Junction); where more than one end or platform meets on a side, a junction node stands past the
platforms' end, as the drawing tool puts one (nearer, where the next node that side is closer).
A group with no stops is one
plain or junction node. Every node gets a heading from its ends, and every edge leaves it along
that heading through a lead point, as `realsave` does for a station. A line changing pair at a
node is a junction there (the crossover). Every node starts on LEVEL; `reallevels.assign` then puts
nodes and stretches of edge on their real levels (T-1773). Every edge is grade separated (`sep`).

No node gets more than a game node's 8 edge ends (MAX_PORTS), the limit a player's junction has too
(T-3790), and no line turns back where a player's line couldn't:
- A group with no stops and more than 8 ends is split (`fabric`): on each side, junctions of at most
  FABRIC_G neighbouring ends (in order across the heading), each joined to the other side's junctions
  its lines run to by as many links as their trains need (PLAT_LOAD a link: one direct edge, then more
  through a plain node between, fewer where the ports run out); a lone end with one link sits on the
  junction it links to. Each move keeps to one link, binned by load.
- A station side with 8 ends or more leaves half its junction's ports for platforms and takes its ends
  on the rest, neighbouring ends sharing a port through a fan farther out (`fan_out`).
- A line turning back at a stop-less node with two or more ends (which the game refuses) turns as a
  player's line does, at a track end: a short dead end (half its longest train plus STUB_CLEAR) leaves
  the node on the other side, and the line runs into it and back.
A group that needs none of these is built exactly as before."""
import math
from collections import Counter, defaultdict

import numpy as np

PAIR_GAP = 8.0
BRANCH_CLEAR = 20.0
JUNCTION_MIN = 15.0
LEAD_TURN_COS = math.cos(math.radians(45))
NUDGE = 6.0
LEVEL = -1
LEAD_MAX = 80.0
HEAD_M = 40.0
MAX_PORTS = 8
# docs 6.2 and realsave's lines: a platform track's hold per train
DWELL_S, MARGIN_S, PASSING_S, TURN_S = 30.0, 60.0, 40.0, 180.0
SECTION_S = 144.0
PLAT_LOAD = 0.8
# T-3790: a split junction's junctions stand at most LINK_OFF either side of its centre; a fan at most
# FAN_OFF beyond its junction; a turn-back stub reaches half the longest train turning plus STUB_CLEAR
LINK_OFF = 40.0
FAN_OFF = 40.0
STUB_CLEAR = 40.0
# at most this many neighbouring ends to a junction of a split one (fewer where the links need the ports)
FABRIC_G = 3


def platform_len(cars):
    return (min(max(cars * 20, 60), 400) // 20) * 20


def along_dir(poly, at_end):
    """Unit vector into the polyline from one end, over HEAD_M or a third of it."""
    p = poly[::-1] if at_end else poly
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    d = min(HEAD_M, cum[-1] / 3) if cum[-1] > 0 else 0
    q = np.array([np.interp(d, cum, p[:, 0]), np.interp(d, cum, p[:, 1])]) if d > 0 else p[-1]
    v = q - p[0]
    n = np.hypot(*v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def track_model(doc, polys, net):
    from shapely.geometry import LineString
    pieces, clines = doc["pieces"], doc["lines"]
    lines, stations = net["lines"], net["stations"]
    st = Counter()

    # a piece shorter than a cut point's snap can have both ends on one node: lines pass it inside the node
    loop = {P["id"] for P in pieces if P["ends"][0] == P["ends"][1]}
    st["pieces inside one node"] = len(loop)
    clines = [dict(L, path=[e for e in L["path"] if e[0] not in loop],
                   occ=[s for e, s in zip(L["path"], L.get("occ", [SECTION_S] * len(L["path"]))) if e[0] not in loop]) for L in clines]
    pair_poly = {}
    for P, poly in zip(pieces, polys):
        if P["id"] in loop:
            continue
        poly = np.asarray(poly, float)
        if P["pairs"] == 1:
            pair_poly[(P["id"], 0)] = poly
            continue
        d = poly[-1] - poly[0]
        sgn = 1 if (d[0] > 0 or (d[0] == 0 and d[1] > 0)) else -1
        n = P["pairs"]
        for k in range(n):
            c = LineString(poly).offset_curve(PAIR_GAP * sgn * ((n - 1) / 2 - k))
            ok = c.geom_type == "LineString" and len(c.coords) >= 2
            pair_poly[(P["id"], k)] = np.array(c.coords) if ok else poly

    def node_of_end(e):
        return pieces[e[0]]["ends"][e[2]]

    # groups: the piece ends lines run between at each node
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (pid, k) in pair_poly:
        find((pid, k, 0)), find((pid, k, 1))
    # per line: node index j along its path -> (in end, out end); stops matched in order
    line_ends, line_stops = [], []
    for L in clines:
        path = L["path"]
        ends = []
        for i, (pid, d, k) in enumerate(path):
            ends.append(((pid, k, 0 if d == 1 else 1), (pid, k, 1 if d == 1 else 0)))
        for i in range(len(ends) - 1):
            a, b = find(ends[i][1]), find(ends[i + 1][0])
            if a != b:
                parent[b] = a
        line_ends.append(ends)
        seq = ([node_of_end(ends[0][0])] + [node_of_end(e[1]) for e in ends]) if ends else []
        matched, j = {}, 0
        for k, n in enumerate(L["stops"]):
            jj = next((x for x in range(j, len(seq)) if seq[x] == n), None)
            if jj is None:
                st["stops dropped (not on the path)"] += 1
                continue
            matched[jj] = k
            j = jj + 1
        line_stops.append(matched)

    def end_at(li, j):
        """The line's piece end at node index j of its path."""
        e = line_ends[li]
        return e[0][0] if j == 0 else e[j - 1][1]

    groups = defaultdict(list)
    for e in parent:
        groups[find(e)].append(e)
    through = defaultdict(set)
    for ends in line_ends:
        for i in range(len(ends) - 1):
            a, b = ends[i][1], ends[i + 1][0]
            through[a].add(b)
            through[b].add(a)
    gstops = defaultdict(list)  # group -> [(line, stop k)]
    for li, matched in enumerate(line_stops):
        for j, k in matched.items():
            gstops[find(end_at(li, j))].append((li, k))
    # each line's visit to each group: (line, path node index, arriving end, leaving end, stops)
    visits = defaultdict(list)
    for li, ends in enumerate(line_ends):
        for j in range(len(ends) + 1 if ends else 0):
            a = ends[j - 1][1] if j > 0 else None
            b = ends[j][0] if j < len(ends) else None
            visits[find(a if a is not None else b)].append((li, j, a, b, j in line_stops[li]))

    def platform_s(a, b, stop, side, dead):
        """A train's hold on its platform track (docs 6.2): dwell and margin, a turn on a through
        platform, half a turn at a dead end (two tracks pooled), or passing."""
        if not stop:
            return PASSING_S
        if a is None or b is None or side[a] == side[b]:
            return (TURN_S + MARGIN_S) / (2 if dead else 1)
        return DWELL_S + MARGIN_S

    nodes, edges = [], []
    placed, reoriented = [], []
    g_chain, g_side, node_of, plat_of = {}, {}, {}, {}
    fan_up, link_edges, stub_of, route_of = {}, [], {}, {}

    def across(e, base, perp):
        """Where an end lies across the heading, HEAD_M out from the node."""
        poly = pair_poly[e[:2]]
        p0 = poly[-1] if e[2] == 1 else poly[0]
        return float((p0 + HEAD_M * along_dir(poly, e[2] == 1) - base) @ perp)

    def lateral(run, off, base, perp):
        """Across the heading, where a junction `off` out from the node meets its ends: their mean
        `across`, scaled from HEAD_M to `off`."""
        return min(1.0, off / HEAD_M) * float(np.mean([across(e, base, perp) for e in run]))

    def chunks(xs, n):
        """xs in n runs as even as can be, in order."""
        return [xs[i * len(xs) // n:(i + 1) * len(xs) // n] for i in range(n)]

    def fan_out(g, chain, sids, ends, side, hv, perp, base, side_off):
        """T-3790: a station's junction whose side has more ends than it takes beside its platform edges
        takes them on every port it has left, neighbouring ends (in order across the heading) sharing a
        port through a fan farther out, a junction with one edge in to it."""
        for sd, jn in chain:
            es = [e for e in ends if side[e] == sd]
            q = MAX_PORTS - len(sids)
            if len(es) <= q:
                continue
            es.sort(key=lambda e: across(e, base, perp))
            sgn = 1 if sd == 1 else -1
            off_j = float((np.array(nodes[jn]["xy"]) - base) @ (sgn * hv))
            _, room = side_off(sd, math.inf)
            off_f = min(off_j + FAN_OFF, max(off_j + NUDGE, 0.7 * room))
            for i, run in enumerate(chunks(es, q)):
                if len(run) < 2:
                    continue
                p = base + sgn * off_f * hv + perp * lateral(run, off_f, base, perp)
                f = len(nodes)
                nodes.append({"xy": p.round(3).tolist(), "level": LEVEL, "station": None})
                fan_up[f] = jn
                link_edges.append((f, jn) if sd == 0 else (jn, f))
                link_hv[(f, jn) if sd == 0 else (jn, f)] = hv
                for e in run:
                    node_of[e] = f
                st["fans (station junctions over 8 ends)"] += 1

    link_hv = {}

    def fabric(g, ends, side, hv, perp, base, side_off):
        """T-3790: a plain junction with more ends than a game node takes (8): on each side, junctions of
        at most G neighbouring ends (in order across the heading), each joined to the other side's
        junctions its lines run to, by as many links as their trains need (one direct edge, then more
        through a plain node between); a lone end with one link sits on the other side's junction."""
        X = sorted((e for e in ends if side[e] == 0), key=lambda e: across(e, base, perp))
        Y = sorted((e for e in ends if side[e] == 1), key=lambda e: across(e, base, perp))
        mv = defaultdict(list)  # (x, y) -> [(li, j)]
        for li, j, a, b, stop in visits[g]:
            if a is not None and b is not None and side[a] != side[b]:
                mv[(a, b) if side[a] == 0 else (b, a)].append((li, j))

        def load(vs):
            return sum(max(clines[li]["tph"][:2]) * max([PASSING_S] + [clines[li]["occ"][x] for x in (j - 1, j) if 0 <= x < len(clines[li]["occ"])])
                       for li, j in vs) / 3600
        for G in range(FABRIC_G, 0, -1):
            A, B = chunks(X, math.ceil(len(X) / G)), chunks(Y, math.ceil(len(Y) / G))
            ai = {x: i for i, r in enumerate(A) for x in r}
            bj = {y: j for j, r in enumerate(B) for y in r}
            W = defaultdict(float)
            for (x, y), vs in mv.items():
                W[(ai[x], bj[y])] += load(vs)
            L = {p: max(1, math.ceil(w / PLAT_LOAD - 1e-9)) for p, w in W.items()}
            for sd, R in ((0, A), (1, B)):
                for k in range(len(R)):
                    mine = [p for p in L if p[sd] == k]
                    while len(R[k]) + sum(L[p] for p in mine) > MAX_PORTS and any(L[p] > 1 for p in mine):
                        L[max(mine, key=lambda p: L[p])] -= 1
            ports = [len(R[k]) + sum(n for p, n in L.items() if p[sd] == k) for sd, R in ((0, A), (1, B)) for k in range(len(R))]
            if max(ports) <= MAX_PORTS:
                break
        else:
            st["split junctions still over 8 ends"] += 1
        # a lone end with one link sits on the junction it links to
        on = {}
        for sd, R in ((0, A), (1, B)):
            for k, r in enumerate(R):
                mine = [p for p in L if p[sd] == k]
                if len(r) == 1 and len(mine) == 1 and L[mine[0]] == 1 and (1 - sd, mine[0][1 - sd]) not in on:
                    on[(sd, k)] = (1 - sd, mine[0][1 - sd])
        jn = {}
        for sd, R in ((0, A), (1, B)):
            sgn = 1 if sd == 1 else -1
            off, _ = side_off(sd, LINK_OFF)
            for k, r in enumerate(R):
                if (sd, k) in on:
                    continue
                jn[(sd, k)] = len(nodes)
                p = base + sgn * off * hv + perp * lateral(r, off, base, perp)
                nodes.append({"xy": p.round(3).tolist(), "level": LEVEL, "station": None})
        for (sd, k), host in on.items():
            jn[(sd, k)] = jn[host]
        for sd, R in ((0, A), (1, B)):
            for k, r in enumerate(R):
                for e in r:
                    node_of[e] = jn[(sd, k)]
        # each pair's moves binned over its links by load
        for (i, j), n in L.items():
            a, b = jn[(0, i)], jn[(1, j)]
            hops = [[a, b]] if a != b else [[a]]
            for x in range(1, n):
                m = len(nodes)
                p = (np.array(nodes[a]["xy"]) + np.array(nodes[b]["xy"])) / 2 + perp * NUDGE * x
                nodes.append({"xy": p.round(3).tolist(), "level": LEVEL, "station": None})
                hops.append([a, m, b])
            for h in hops:
                for u, v in zip(h, h[1:]):
                    link_edges.append((u, v))
                    link_hv[(u, v)] = hv
            bins = [0.0] * len(hops)
            for (x, y), vs in sorted(((k, v) for k, v in mv.items() if (ai[k[0]], bj[k[1]]) == (i, j)), key=lambda kv: -load(kv[1])):
                k = int(np.argmin(bins))
                bins[k] += load(vs)
                for li, jj in vs:
                    a_end = line_ends[li][jj - 1][1]
                    route_of[(li, jj)] = hops[k] if side[a_end] == 0 else hops[k][::-1]
            st["links at split junctions"] += len(hops) - (1 if a == b else 0)
        st["junctions split (over 8 ends)"] += 1
        st["nodes at split junctions"] += len(set(jn.values()))

    for g, ends in groups.items():
        pts = np.array([pair_poly[e[:2]][0] if e[2] == 0 else pair_poly[e[:2]][-1] for e in ends])
        us = [along_dir(pair_poly[e[:2]], e[2] == 1) for e in ends]
        h0 = math.atan2(us[0][1], us[0][0])
        v = sum((1 if math.cos(math.atan2(u[1], u[0]) - h0) >= 0 else -1) * u for u in us)
        h = math.atan2(v[1], v[0])
        hv = np.array([math.cos(h), math.sin(h)])
        geo = {e: (1 if float(u @ hv) > 0 else 0) for e, u in zip(ends, us)}
        # a line runs through: the two ends it passes between are on opposite sides, whatever the
        # end geometry says (a node joining cut points apart can point an end backwards); geometry
        # picks only which way round each connected set lies
        side = {}
        for e0 in sorted(ends, key=lambda e: -len(through[e])):
            if e0 in side:
                continue
            comp, todo = {e0: 0}, [e0]
            while todo:
                a = todo.pop()
                for b in through[a]:
                    if b not in comp:
                        comp[b] = 1 - comp[a]
                        todo.append(b)
                    elif comp[b] == comp[a]:
                        st["through moves that cannot all be on opposite sides"] += 1
            agree = sum(1 for e, c in comp.items() if geo[e] == c)
            flip = agree * 2 < len(comp)
            for e, c in comp.items():
                side[e] = (1 - c) if flip else c
        if any(side[e] != geo[e] for e in ends):
            st["groups with a side set by the lines over the geometry"] += 1
            reoriented.append(g)
            v = sum((1 if side[e] == 1 else -1) * u for e, u in zip(ends, us))
            if np.hypot(*v) > 1e-9:
                h = math.atan2(v[1], v[0])
                hv = np.array([math.cos(h), math.sin(h)])
        base = pts.mean(0)
        while any(np.hypot(*(base - q)) < NUDGE for q in placed):
            base = base + 8.0 * np.array([-hv[1], hv[0]])
        placed.append(base)
        stops = gstops.get(g, [])
        per_side = [sum(1 for e in ends if side[e] == s) for s in (0, 1)]
        # T-3790: a side with more ends than a junction takes beside one platform edge shares half the
        # junction's ports through fans (`fan_out`), leaving the other half for platforms
        per_port = [p if p < MAX_PORTS else MAX_PORTS // 2 for p in per_side]
        perp = np.array([-hv[1], hv[0]])

        def side_off(sd, most):
            """How far along the heading a junction on side sd stands: short of the next node that side
            (a junction beyond it would take its edges back in a hairpin), at most `most`."""
            sgn = 1 if sd == 1 else -1
            far = [float((pair_poly[e[:2]][-1 if e[2] == 0 else 0] - base) @ (sgn * hv)) for e in ends if side[e] == sd]
            room = min([f for f in far if f > 1.0], default=math.inf)
            return min(most, max(0.4 * room, JUNCTION_MIN)), room
        # a node with one end is a track end already, where a train may turn
        stub_sides = sorted({1 - side[a] for _, _, a, b, stop in visits[g] if a is not None and b is not None and side[a] == side[b] and not stop}) if len(ends) > 1 else []
        if stops:
            li0, k0 = stops[0]
            s = stations[lines[li0]["stops"][k0]]
            plat = platform_len(max(lines[li]["cars"] for li, _ in stops))
            # platform tracks: as many as the trains need (a terminus fed by one pair, London
            # Bridge's through lines), each a station node side by side with a junction each side;
            # a move's lines (same ends, stopping or not) keep to one
            dead = min(per_side) == 0
            moves = defaultdict(list)
            for li, j, a, b, stop in visits[g]:
                moves[(a, b, stop)].append((li, j))

            def hold_s(m, li, j):
                # the platform track, or the edge to it from the junction, which a train holds as
                # long as a section of its piece (docs 6.2), whichever is longer
                occ = clines[li]["occ"]
                sec = max([occ[x] for x in (j - 1, j) if 0 <= x < len(occ)], default=SECTION_S)
                return max(platform_s(*m, side, dead), sec)
            mload = {m: sum(max(clines[li]["tph"][:2]) * hold_s(m, li, j) for li, j in v) / 3600 for m, v in moves.items()}
            n_plat = min(max(1, math.ceil(sum(mload.values()) / PLAT_LOAD - 1e-9)), max(1, MAX_PORTS - max(per_port)), len(moves))
            bins = [0.0] * n_plat
            for m in sorted(moves, key=lambda m: -mload[m]):
                k = int(np.argmin(bins))
                bins[k] += mload[m]
                for v in moves[m]:
                    plat_of[v] = k
            if n_plat > 1:
                st["stations with more than one platform track"] += 1
                st["platform tracks added"] += n_plat - 1
            sids = []
            for k in range(n_plat):
                q = base + perp * PAIR_GAP * (k - (n_plat - 1) / 2)
                sids.append(len(nodes))
                nodes.append({"xy": q.round(3).tolist(), "level": LEVEL, "station": {"platform": plat, "name": s["name"], "key": s["key"]}})
            sid = sids[0]
            chain = []
            for sd in (0, 1):
                if per_side[sd] > 1 or (n_plat > 1 and per_side[sd] > 0):
                    # past the platform
                    off, _ = side_off(sd, plat / 2 + BRANCH_CLEAR + 10)
                    if off < plat / 2 + BRANCH_CLEAR:
                        st["junctions inside a platform's clearance (no room)"] += 1
                    q = base + (1 if sd == 1 else -1) * off * hv
                    chain.append((sd, len(nodes)))
                    nodes.append({"xy": q.round(3).tolist(), "level": LEVEL, "station": None})
            for e in ends:
                j = next((n for sd, n in chain if sd == side[e]), sid)
                node_of[e] = j
            g_chain[g] = (chain, sids)
            fan_out(g, chain, sids, ends, side, hv, perp, base, side_off)
        elif len(ends) + len(stub_sides) <= MAX_PORTS or min(per_side) == 0:
            nid = len(nodes)
            nodes.append({"xy": base.round(3).tolist(), "level": LEVEL, "station": None})
            for e in ends:
                node_of[e] = nid
            g_chain[g] = ([], [])
            if len(ends) > MAX_PORTS:
                st["nodes over 8 ends"] += 1
            elif len(ends) + len(stub_sides) <= MAX_PORTS:
                # T-3790: a line turning back here, at no stop, turns as a player's line does: at a track end,
                # a short dead end leaving on the other side
                for sd in stub_sides:
                    turners = [li for li, _, a, b, _ in visits[g] if a is not None and b is not None and side[a] == side[b] == 1 - sd]
                    q = base + (1 if sd == 1 else -1) * (max(platform_len(lines[li]["cars"]) for li in turners) / 2 + STUB_CLEAR) * hv
                    stub_of[(g, 1 - sd)] = (nid, len(nodes), sd)
                    nodes.append({"xy": q.round(3).tolist(), "level": LEVEL, "station": None})
                    st["turn-back stubs"] += 1
        else:
            fabric(g, ends, side, hv, perp, base, side_off)
            g_chain[g] = ([], [])
            if stub_sides:
                st["turns back at a split junction (no stub)"] += 1
        g_side[g] = (hv, side)

    def add_edge(a, b, ha, da, hb, db, poly):
        """An edge a to b leaving a along da * ha and b along db * hb, the polyline between."""
        A, B = np.array(nodes[a]["xy"]), np.array(nodes[b]["xy"])
        lead = min(np.hypot(*(B - A)) / 4, LEAD_MAX)
        la, lb = A + lead * da * ha, B + lead * db * hb
        seg = np.hypot(*np.diff(poly, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        lo = np.hypot(*(A - poly[0])) + lead + 10
        hi = cum[-1] - (np.hypot(*(B - poly[-1])) + lead + 10)
        inner = [p for p in poly[(cum > lo) & (cum < hi)]]
        # points that would turn the track sharply off a lead (a node away from the piece's end,
        # a short piece) go: the trace would fit them with the tightest curve and trains crawl
        def sharp(p0, p1, p2):
            u, w = p1 - p0, p2 - p1
            nu, nw = np.hypot(*u), np.hypot(*w)
            return nu < 1e-6 or nw < 1e-6 or float(u @ w) < LEAD_TURN_COS * nu * nw
        while inner and sharp(A, la, inner[0]) or (len(inner) > 1 and np.hypot(*(inner[1] - inner[0])) < 1.0):
            inner.pop(0)
        while inner and sharp(inner[-1], lb, B):
            inner.pop()
        inner = np.array(inner).reshape(-1, 2)
        edges.append({"a": a, "b": b, "lead_a": la.round(3).tolist(), "lead_b": lb.round(3).tolist(), "line": inner.round(2).tolist()})
        return len(edges) - 1

    # a station's own edges: junction to platform to junction; platforms side by side get leads
    # along the heading, so their edges leave the junction straight and swing out to the platform
    conn = {}
    for g, (chain, sids) in g_chain.items():
        hv = g_side[g][0]
        for k, sid in enumerate(sids):
            for sd, j in chain:
                a, b = (j, sid) if sd == 0 else (sid, j)
                if len(sids) == 1:
                    conn[(a, b)] = len(edges)
                    edges.append({"a": a, "b": b, "lead_a": None, "lead_b": None, "line": []})
                else:
                    A, B = np.array(nodes[a]["xy"]), np.array(nodes[b]["xy"])
                    conn[(a, b)] = add_edge(a, b, hv, 1, hv, -1, np.array([A, B]))
    # T-3790: fans and split junctions' links, each from the node behind along the heading to the one ahead
    for a, b in link_edges:
        conn[(a, b)] = add_edge(a, b, link_hv[(a, b)], 1, link_hv[(a, b)], -1, np.array([nodes[a]["xy"], nodes[b]["xy"]], float))
    for (g, _), (n, t, sd) in stub_of.items():
        hv, da = g_side[g][0], (1 if sd == 1 else -1)
        conn[(n, t)] = add_edge(n, t, hv, da, hv, -da, np.array([nodes[n]["xy"], nodes[t]["xy"]], float))

    piece_edge = {}
    for (pid, k), poly in pair_poly.items():
        ea, eb = (pid, k, 0), (pid, k, 1)
        a, b = node_of[ea], node_of[eb]
        ha, sa = g_side[find(ea)]
        hb, sb = g_side[find(eb)]
        da, db = (1 if sa[ea] == 1 else -1), (1 if sb[eb] == 1 else -1)
        if a != b:
            piece_edge[(pid, k)] = [add_edge(a, b, ha, da, hb, db, poly)]
            continue
        # both ends on one node (a loop, or a line turning back onto it): a plain node halfway
        st["pieces through a node at their middle"] += 1
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(poly, axis=0).T))])
        i = int(np.searchsorted(cum, cum[-1] / 2))
        i = min(max(i, 1), len(poly) - 2) if len(poly) >= 3 else 1
        mid = poly[i] if len(poly) >= 3 else poly.mean(0)
        u = (poly[min(i + 1, len(poly) - 1)] - poly[max(i - 1, 0)])
        hm = u / max(np.hypot(*u), 1e-9)
        m = len(nodes)
        nodes.append({"xy": np.asarray(mid).round(3).tolist(), "level": LEVEL, "station": None})
        e1 = add_edge(a, m, ha, da, hm, -1, poly[:i + 1])
        e2 = add_edge(m, b, hm, 1, hb, db, poly[i:])
        piece_edge[(pid, k)] = [e1, e2]

    def hop(u, v, out):
        """The station edge from node u to node v, if they differ."""
        if u is None or v is None or u == v:
            return
        out.append(conn[(u, v)] << 1 if (u, v) in conn else conn[(v, u)] << 1 | 1)

    out_lines, reversals = [], []
    for li, L in enumerate(clines):
        ends, matched = line_ends[li], line_stops[li]
        if not ends or len(matched) < 2 or any((pid, k) not in piece_edge for pid, _, k in L["path"]):
            st["lines left out"] += 1
            out_lines.append(None)
            continue
        n_path = len(ends)
        if 0 not in matched or n_path not in matched:
            st["lines whose path runs past an end stop"] += 1
        path, stops = [], []
        for j in range(n_path + 1):
            here = j in matched
            arrive = node_of[ends[j - 1][1]] if j > 0 else None
            leave = node_of[ends[j][0]] if j < n_path else None
            # in from a fan to its junction, and out again after (T-3790)
            fan_in, fan_out_to = arrive, leave
            arrive, leave = fan_up.get(arrive, arrive), fan_up.get(leave, leave)
            hop(fan_in, arrive, path)
            sids = g_chain[find(ends[j - 1][1] if j > 0 else ends[j][0])][1]
            sid = sids[plat_of.get((li, j), 0)] if sids else None
            if here:
                if sid is None:
                    st["stops on a node with no station"] += 1
                else:
                    if not stops or stops[-1] != sid:
                        stops.append(sid)
            if 0 < j < n_path:
                a_end, b_end = ends[j - 1][1], ends[j][0]
                sd = g_side[find(a_end)][1]
                if sd[a_end] == sd[b_end]:
                    nm = nodes[sid]["station"]["name"] if sid is not None and nodes[sid]["station"] else "a junction"
                    stub = stub_of.get((find(a_end), sd[a_end]))
                    reversals.append(f"{lines[li]['feed']} {lines[li]['name']} (line {li}) at {nm}{'' if here else ', not a stop'}{', at a stub' if stub and not here else ''}")
            if sid is not None and (here or arrive != leave):
                hop(arrive, sid, path)
                hop(sid, leave, path)
            elif (li, j) in route_of:
                r = route_of[(li, j)]
                for u, v in zip(r, r[1:]):
                    hop(u, v, path)
            elif arrive is not None and leave is not None:
                g = find(ends[j - 1][1])
                hv, side = g_side[g]
                if arrive == leave and side[ends[j - 1][1]] == side[ends[j][0]]:
                    stub = stub_of.get((g, side[ends[j - 1][1]]))
                    if stub:
                        hop(stub[0], stub[1], path)
                        hop(stub[1], stub[0], path)
                    else:
                        st["turns back at a node it does not stop at"] += 1
            hop(leave, fan_out_to, path)
            if j < n_path:
                pid, d, k = L["path"][j]
                es = piece_edge[(pid, k)]
                path += [e << 1 | (0 if d == 1 else 1) for e in (es if d == 1 else es[::-1])]
        out_lines.append({"index": li, "stops": stops, "path": path, "tph": L["tph"]})
    n_ends = Counter(n for e in edges for n in (e["a"], e["b"]))
    over = sum(1 for k in n_ends.values() if k > MAX_PORTS)
    if over:  # their edges past the 8th are refused, and lines over them don't run
        st["nodes still over 8 ends"] = over
    st.update({"nodes": len(nodes), "stations": sum(1 for n in nodes if n["station"]), "edges": len(edges),
               "lines": sum(1 for l in out_lines if l)})
    st["partway reversals"] = len(reversals)
    for g in reoriented:
        chain, sids = g_chain[g]
        n = nodes[sids[0] if sids else node_of[groups[g][0]]]
        names = sorted({lines[li]["name"] for li, _ in gstops.get(g, [])})
        print(f"  sides set by the lines: {n['station']['name'] if n['station'] else 'junction'} at {n['xy']} {names}")
    print("model:", dict(st))
    for r in reversals:
        print("  reverses:", r)
    return {"nodes": nodes, "edges": edges, "lines": out_lines, "stats": dict(st), "reversals": reversals}
