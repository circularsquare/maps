"""T-1305 stage 2: the track model's network from a city's corridors (`realcorridors.py`), which the
sim's `realsave --shared` builds as it is: nodes, edges and each line's stops and path.

Each piece is one double-track edge per pair, the pairs 8 m apart. At each corridor node the piece
ends a line runs between are one group (an express pair the locals never join is its own group,
so an express station has a platform on each pair). A group with stops is one station node (a
station takes two edge ends, one each way); where more than one end leaves on a side, a junction
node stands past the platform's end, as the drawing tool puts one. A group with no stops is one
plain or junction node. Every node gets a heading from its ends, and every edge leaves it along
that heading through a lead point, as `realsave` does for a station. A line changing pair at a
node is a junction there (the crossover). One level for all (-1; levels per piece are stage 3),
every edge grade separated (`sep`) as before."""
import math
from collections import Counter, defaultdict

import numpy as np

PAIR_GAP = 8.0
BRANCH_CLEAR = 20.0
NUDGE = 6.0
LEVEL = -1
LEAD_MAX = 80.0
HEAD_M = 40.0


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
    clines = [dict(L, path=[e for e in L["path"] if e[0] not in loop]) for L in clines]
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
        for k in (0, 1):
            c = LineString(poly).offset_curve(PAIR_GAP / 2 * sgn * (1 if k == 0 else -1))
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

    nodes, edges = [], []
    placed, reoriented = [], []
    g_chain, g_side, node_of = {}, {}, {}
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
            sid = len(nodes)
            nodes.append({"xy": base.round(3).tolist(), "level": LEVEL, "station": {"platform": plat, "name": s["name"], "key": s["key"]}})
            chain = []
            for sd in (0, 1):
                if per_side[sd] > 1:
                    q = base + (1 if sd == 1 else -1) * (plat / 2 + BRANCH_CLEAR + 10) * hv
                    chain.append((sd, len(nodes)))
                    nodes.append({"xy": q.round(3).tolist(), "level": LEVEL, "station": None})
            order = [n for sd, n in chain if sd == 0] + [sid] + [n for sd, n in chain if sd == 1]
            for e in ends:
                j = next((n for sd, n in chain if sd == side[e]), sid)
                node_of[e] = j
            g_chain[g] = (order, sid)
        else:
            nid = len(nodes)
            nodes.append({"xy": base.round(3).tolist(), "level": LEVEL, "station": None})
            for e in ends:
                node_of[e] = nid
            g_chain[g] = ([nid], None)
            if len(ends) > 8:
                st["nodes over 8 ends"] += 1
        g_side[g] = (hv, side)
    chain_edge = {}
    for g, (order, sid) in g_chain.items():
        for i in range(len(order) - 1):
            chain_edge[(order[i], order[i + 1])] = len(edges)
            edges.append({"a": order[i], "b": order[i + 1], "lead_a": None, "lead_b": None, "line": []})

    def add_edge(a, b, ha, da, hb, db, poly):
        """An edge a to b leaving a along da * ha and b along db * hb, the polyline between."""
        A, B = np.array(nodes[a]["xy"]), np.array(nodes[b]["xy"])
        lead = min(np.hypot(*(B - A)) / 4, LEAD_MAX)
        la, lb = A + lead * da * ha, B + lead * db * hb
        seg = np.hypot(*np.diff(poly, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        lo = np.hypot(*(A - poly[0])) + lead + 10
        hi = cum[-1] - (np.hypot(*(B - poly[-1])) + lead + 10)
        inner = poly[(cum > lo) & (cum < hi)]
        edges.append({"a": a, "b": b, "lead_a": la.round(3).tolist(), "lead_b": lb.round(3).tolist(), "line": inner.round(2).tolist()})
        return len(edges) - 1

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

    def walk(i, j, out):
        """Edges from node i to node j along a group's chain."""
        if i == j:
            return
        order = g_chain_of[i][0]
        a, b = order.index(i), order.index(j)
        step = 1 if b > a else -1
        for x in range(a, b, step):
            u, v = order[x], order[x + step]
            e = chain_edge[(u, v)] if step == 1 else chain_edge[(v, u)]
            out.append(e << 1 | (0 if step == 1 else 1))

    g_chain_of = {}
    for g, c in g_chain.items():
        for n in c[0]:
            g_chain_of[n] = c

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
            sid = g_chain_of[arrive if arrive is not None else leave][1]
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
            via = sid if here and sid is not None else None
            frm = arrive if arrive is not None else via
            to = leave if leave is not None else via
            if via is not None:
                walk(frm, via, path)
                walk(via, to, path)
            elif frm is not None and to is not None:
                hv, side = g_side[find(ends[j - 1][1])]
                if frm == to and side[ends[j - 1][1]] == side[ends[j][0]]:
                    st["turns back at a node it does not stop at"] += 1
                walk(frm, to, path)
            if j < n_path:
                pid, d, k = L["path"][j]
                es = piece_edge[(pid, k)]
                path += [e << 1 | (0 if d == 1 else 1) for e in (es if d == 1 else es[::-1])]
        out_lines.append({"index": li, "stops": stops, "path": path, "tph": L["tph"]})
    st.update({"nodes": len(nodes), "stations": sum(1 for n in nodes if n["station"]), "edges": len(edges),
               "lines": sum(1 for l in out_lines if l)})
    st["partway reversals"] = len(reversals)
    for g in reoriented:
        order, sid = g_chain[g]
        n = nodes[sid if sid is not None else order[0]]
        names = sorted({lines[li]["name"] for li, _ in gstops.get(g, [])})
        print(f"  sides set by the lines: {n['station']['name'] if n['station'] else 'junction'} at {n['xy']} {names}")
    print("model:", dict(st))
    for r in reversals:
        print("  reverses:", r)
    return {"nodes": nodes, "edges": edges, "lines": out_lines, "stats": dict(st), "reversals": reversals}
