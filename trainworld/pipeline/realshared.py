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
node is a junction there (the crossover). One level for all (-1; levels per piece are T-1773),
every edge grade separated (`sep`) as before."""
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
            n_plat = min(max(1, math.ceil(sum(mload.values()) / PLAT_LOAD - 1e-9)), max(1, MAX_PORTS - max(per_side)), len(moves))
            bins = [0.0] * n_plat
            for m in sorted(moves, key=lambda m: -mload[m]):
                k = int(np.argmin(bins))
                bins[k] += mload[m]
                for v in moves[m]:
                    plat_of[v] = k
            if n_plat > 1:
                st["stations with more than one platform track"] += 1
                st["platform tracks added"] += n_plat - 1
            perp = np.array([-hv[1], hv[0]])
            sids = []
            for k in range(n_plat):
                q = base + perp * PAIR_GAP * (k - (n_plat - 1) / 2)
                sids.append(len(nodes))
                nodes.append({"xy": q.round(3).tolist(), "level": LEVEL, "station": {"platform": plat, "name": s["name"], "key": s["key"]}})
            sid = sids[0]
            chain = []
            for sd in (0, 1):
                if per_side[sd] > 1 or (n_plat > 1 and per_side[sd] > 0):
                    # past the platform, but short of the next node that side: a junction beyond
                    # it would take its edges back in a hairpin
                    sgn = 1 if sd == 1 else -1
                    far = [float((pair_poly[e[:2]][-1 if e[2] == 0 else 0] - base) @ (sgn * hv)) for e in ends if side[e] == sd]
                    room = min([f for f in far if f > 1.0], default=math.inf)
                    off = min(plat / 2 + BRANCH_CLEAR + 10, max(0.4 * room, JUNCTION_MIN))
                    if off < plat / 2 + BRANCH_CLEAR:
                        st["junctions inside a platform's clearance (no room)"] += 1
                    q = base + sgn * off * hv
                    chain.append((sd, len(nodes)))
                    nodes.append({"xy": q.round(3).tolist(), "level": LEVEL, "station": None})
            for e in ends:
                j = next((n for sd, n in chain if sd == side[e]), sid)
                node_of[e] = j
            g_chain[g] = (chain, sids)
        else:
            nid = len(nodes)
            nodes.append({"xy": base.round(3).tolist(), "level": LEVEL, "station": None})
            for e in ends:
                node_of[e] = nid
            g_chain[g] = ([], [])
            if len(ends) > MAX_PORTS:
                st["nodes over 8 ends"] += 1
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
                    nm = nodes[sid]["station"]["name"] if sid is not None else "a junction"
                    reversals.append(f"{lines[li]['feed']} {lines[li]['name']} (line {li}) at {nm}{'' if here else ', not a stop'}")
            if sid is not None and (here or arrive != leave):
                hop(arrive, sid, path)
                hop(sid, leave, path)
            elif arrive is not None and leave is not None:
                hv, side = g_side[find(ends[j - 1][1])]
                if arrive == leave and side[ends[j - 1][1]] == side[ends[j][0]]:
                    st["turns back at a node it does not stop at"] += 1
            if j < n_path:
                pid, d, k = L["path"][j]
                es = piece_edge[(pid, k)]
                path += [e << 1 | (0 if d == 1 else 1) for e in (es if d == 1 else es[::-1])]
        out_lines.append({"index": li, "stops": stops, "path": path, "tph": L["tph"]})
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
