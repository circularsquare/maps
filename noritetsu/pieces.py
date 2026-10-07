"""Register lines in pieces: bridged over the track between them, or one line per piece.

Shared by the named-track registers (gb_register, kr_register, cn_register, tr_register; moved
out of gb_register 2026-10-04). Each calls `split_pieces` from the hook of the same name that
build_model calls on a register module once it has dropped the junction-ended sections no
route runs over (`drop_unridden_sections`), so it sees the pieces that ship.

Anita, 2026-10-04: a trip is entered station to station on a line's strip diagram, so a line
that cannot be ridden across a gap is broken as a line. Where a register names track one line
per way (OSM's `name`), a line that runs over another's rails for a stretch stops and starts
again: the UK's Birmingham to Peterborough Line is Nuneaton - Wigston and Syston - Oakham with
the Midland Main Line's Leicester between, Korea's 수인선 runs over 안산선 from 오이도 to
한대앞. So:

1. `bridge_gaps`: for each line in pieces, from its biggest group of pieces, the cheapest
   track to a station of another piece over the passenger track graph the country gives
   (`track_graph`), within the limits in `Rules`. The bridge is cut at the stations on it of
   the lines it runs over, and each part is a section of the line. A part mostly on another
   register line's track is listed in the line's `borrowed`, and is padded for ownership so
   that track stays its own line's (the OWNERSHIP paragraph of bridge_gaps).
2. What is still apart becomes one line per piece, as in us_register: the biggest keeps the
   id, the others take a hash of the line's id and their lowest stop's id, aliases.json
   `pieces` (from the caller's LINE_PIECES) lets the app move saved rides. A line named in
   `Rules.keep_whole` is left in pieces: its gap is track missing from OSM on a line trains
   run through, and a piece's id would vanish again once the track is mapped.

gb_sources.md "Lines in pieces" has the measurements the limits were set on; kr_, cn_ and
tr_sources.md each country's cases.
"""
import hashlib
import math
from collections import Counter, defaultdict

import numpy as np


class Rules:
    """One country's settings. The defaults are the UK's (gb_sources.md "Lines in pieces").

    A gap is bridged when the track found joining the pieces is at most `slack` times the
    crow-fly between its ends plus `plus_km`, no longer than `max_km`, at least `route_share`
    of it under an OSM passenger route (bridges under `route_free_km` exempt: platform roads
    are often left out of route relations), and its km on other lines' named track no more
    than the smaller side it joins unless under `free_km` (a 2 km stray piece of a name 40 km
    from the rest of it is that name on another line's track, not the line running on).

    The search prefers track under a passenger route: a way no route uses costs
    `unrouted_cost` times its length. Two per-line costs, off unless set: `own_cost`, the
    factor on track carrying the line's own name (which then also counts as under a route for
    `route_share`: a register line's own named track is passenger track by construction, where
    OSM's routes may lie on the old line beside it), and `slow_cost`, the factor on track not
    tagged highspeed=yes when most of the line is high-speed.

    `attach_m`: a station of a piece joins the track at its nearest vertex this close.
    `cut_m`: a station of a line the bridge runs over cuts it this close. `borrow_pad_m`: beside
    a borrowed section, a way is recorded at least this much further off than the nearest other
    register line's section, for ownership; it must stay more than ownership's EXACT_TIE_M and
    less than its TIE_M. `piece_min_km`: a split piece with under two stops shorter than this is
    no line. `keep_whole`: line names never split (the module docstring). `piece_name(name,
    line, stations, a, b)`: the name_en of a piece running from stop a to stop b.

    `dense`: OSM puts vertices only where track bends, so on straight high-speed track the
    nearest vertex to a station can be well over attach_m and cut_m away (Türkiye's Pamukova
    YHT: no vertex within 150 m, so the station never joined the graph and a bridge from the
    junction 48 km back ran past it on the old line). With `dense`, a station attach_m from
    every vertex joins the graph at the first vertex of its own section's drawn track (its own
    line's, so never a parallel line's), and a bridge is cut at a station within cut_m of its
    track between vertices, at the nearer vertex. Off for the UK, whose build it would move.

    `cut_ok(line, station)`: whether a station of a line the bridge runs over may become a stop
    of this line (None: every one may). Türkiye's high-speed lines take only YHT stations, as
    its reader places stations (`tr_register.hs_ok`).

    `no_bridge`: line names never bridged, split instead: where the track between the pieces
    is another line that replaced this one, so a bridge would send rides over the old line
    (China's 成昆线: its old-line pieces north and south, the new line 峨广线 between)."""

    def __init__(self, tag="GB", id_prefix="g", lat=54.5, slack=1.5, plus_km=5.0,
                 max_km=100.0, route_share=0.5, free_km=15.0, route_free_km=2.0,
                 unrouted_cost=4.0, attach_m=150, cut_m=150, borrow_pad_m=2.0,
                 piece_min_km=0.5, own_cost=None, slow_cost=None, keep_whole=(),
                 piece_name=None, dense=False, cut_ok=None, no_bridge=()):
        self.tag, self.id_prefix = tag, id_prefix
        self.kx = math.cos(math.radians(lat)) * 111.32     # km per degree of longitude
        self.slack, self.plus_km, self.max_km = slack, plus_km, max_km
        self.route_share, self.free_km, self.route_free_km = route_share, free_km, route_free_km
        self.unrouted_cost = unrouted_cost
        self.attach_m, self.cut_m, self.borrow_pad_m = attach_m, cut_m, borrow_pad_m
        self.piece_min_km = piece_min_km
        self.own_cost, self.slow_cost = own_cost, slow_cost
        self.keep_whole, self.no_bridge = set(keep_whole), set(no_bridge)
        self.piece_name = piece_name or native_piece_name
        self.dense, self.cut_ok = dense, cut_ok
        import ownership
        if not ownership.EXACT_TIE_M < borrow_pad_m < ownership.TIE_M:
            raise ValueError(f"borrow_pad_m {borrow_pad_m} must lie between ownership's "
                             f"EXACT_TIE_M {ownership.EXACT_TIE_M} and TIE_M {ownership.TIE_M}")


def native_piece_name(name, line, stations, a, b):
    """gb's: "Name (first stop – last stop)" from the stations' own names."""
    return f"{name} ({stations[a]['name']} – {stations[b]['name']})"


def english_piece_name(name, line, stations, a, b):
    """For a register whose names are not English: the line's English name, else its own, with
    its end stops' English names, else their own."""
    en = lambda s: stations[s].get("name_en") or stations[s]["name"]
    return f"{line.get('name_en') or name} ({en(a)} – {en(b)})"


def routed_ways(rels):
    """Ways an OSM passenger route runs over (members with no role, or forward/backward)."""
    import build_model as bm
    on_route = set()
    for tags, members in rels.values():
        if tags.get("type") == "route" and tags.get("route") in bm.ROUTE_KINDS:
            for ty, ref, role in members:
                if ty == "w" and (not role or role.startswith(("forward", "backward"))):
                    on_route.add(ref)
    return on_route


def section_pieces(sections):
    """The connected pieces of a line's sections, biggest (km) first, each a list of indexes
    into `sections`; ties by the lowest station id, so the order is stable."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s in sections:
        parent[find(s[0])] = find(s[1])
    by = defaultdict(list)
    for i, s in enumerate(sections):
        by[find(s[0])].append(i)
    return sorted(by.values(), key=lambda ix: (-sum(sections[i][2] for i in ix),
                                               min(min(sections[i][:2]) for i in ix)))


def track_graph(ways, coords, on_route, classify, rules, log):
    """The passenger track as a graph for bridging, over OSM node ids. `classify(wid, tags,
    routed)` gives a way's register line name, "" for a way in the graph under no name, None
    for a way left out. Returns a dict: G (scipy CSR; km, unrouted_cost times that off
    routes), node lon/lat, a k-d tree of nodes (km), and per edge its way and true km; per way
    its register name, route and high-speed flags; and the edges before the cheapest of each
    node pair was kept (`raw`), for per-line costs."""
    from scipy.spatial import cKDTree
    wl, names, routed, hs, chunks, pchunks = [], [], [], [], [], []
    for wid, (t, nodes) in ways.items():
        nm = classify(wid, t, wid in on_route)
        if nm is None:
            continue
        arr = np.asarray(nodes, dtype=np.int64)
        pos, ok = coords.many(arr)
        if ok.sum() < 2:
            continue
        wl.append(wid)
        names.append(nm)
        routed.append(wid in on_route)
        hs.append(t.get("highspeed") == "yes")
        chunks.append(arr[ok])
        pchunks.append(pos[ok])
    routed, hs = np.array(routed), np.array(hs)
    lens = np.array([c.size for c in chunks])
    alln, allp = np.concatenate(chunks), np.concatenate(pchunks)
    allw = np.repeat(np.arange(len(wl)), lens)
    x, y = coords.x[allp] / 1e7, coords.y[allp] / 1e7
    i = np.flatnonzero((allw[1:] == allw[:-1]) & (alln[1:] != alln[:-1]))
    uniq, inv = np.unique(alln, return_inverse=True)
    N = uniq.size
    nx, ny = np.empty(N), np.empty(N)
    nx[inv], ny[inv] = x, y
    lat = np.radians((y[i] + y[i + 1]) / 2)
    km = np.hypot((x[i + 1] - x[i]) * np.cos(lat) * 111.32, (y[i + 1] - y[i]) * 110.57)
    ew = allw[i]
    cost = np.maximum(km * np.where(routed[ew], 1.0, rules.unrouted_cost), 1e-6)
    lo, hi = np.minimum(inv[i], inv[i + 1]), np.maximum(inv[i], inv[i + 1])
    T = {"nx": nx, "ny": ny, "N": N, "wid": wl, "name": names, "routed": routed, "hs": hs,
         "raw": (lo, hi, km, ew)}
    T.update(select_edges(T, cost))
    T["tree"] = cKDTree(np.column_stack([nx * rules.kx, ny * 110.57]))
    log(f"{rules.tag}: bridging graph: {len(wl)} ways of passenger track, {N:,} nodes, "
        f"{T['key'].size:,} edges")
    return T


def select_edges(T, cost):
    """Of the edges between each pair of nodes, the cheapest by `cost` (one per raw edge): the
    graph G by cost and Gkm by km over them, and per kept edge its key, way and km."""
    from scipy.sparse import coo_matrix
    lo, hi, km, ew = T["raw"]
    N = T["N"]
    key = lo.astype(np.int64) * N + hi
    o = np.lexsort((cost, key))
    first = np.r_[True, key[o][1:] != key[o][:-1]]
    sel = o[first]
    lo, hi, km, cost, ew, key = lo[sel], hi[sel], km[sel], cost[sel], ew[sel], key[sel]
    G = coo_matrix((np.r_[cost, cost], (np.r_[lo, hi], np.r_[hi, lo])), shape=(N, N)).tocsr()
    kk = np.maximum(km, 1e-6)
    Gkm = coo_matrix((np.r_[kk, kk], (np.r_[lo, hi], np.r_[hi, lo])), shape=(N, N)).tocsr()
    return {"G": G, "Gkm": Gkm, "key": key, "ew": ew, "km": km}


def line_costs(T, l, rules):
    """T with this line's costs (Rules.own_cost, slow_cost), or T itself when neither is set.
    Also returns the per-way mask of track that counts as under a route for this line."""
    if rules.own_cost is None and rules.slow_cost is None:
        return T, T["routed"]
    own = np.array([n == l["name"] for n in T["name"]], dtype=bool)
    lo, hi, km, ew = T["raw"]
    f = np.where(T["routed"][ew], 1.0, rules.unrouted_cost)
    hss = l.get("highspeed_sections") or {}
    fast = (sum(1 for v in hss.values() if v) * 2 > len(hss)) if hss else bool(l.get("highspeed"))
    if rules.slow_cost is not None and fast:
        f = f * np.where(T["hs"][ew], 1.0, rules.slow_cost)
    if rules.own_cost is not None:
        f = np.where(own[ew], rules.own_cost, f)
        routed = T["routed"] | own
    else:
        routed = T["routed"]
    Tl = dict(T)
    Tl.update(select_edges(T, np.maximum(km * f, 1e-6)))
    return Tl, routed


def bridge_gaps(lines, stations, geoms, reg_ways, state, log, rules, graph, ends=()):
    """A register line whose sections are in pieces is joined over the track between them
    where trains can run across. OSM names track by one line only, so where a line runs over
    another's rails for a stretch its name stops and starts again: the Birmingham to
    Peterborough Line is Nuneaton - Wigston and Syston - Stamford, with the Midland Main Line's
    Leicester between.

    Runs in split_pieces, after build_model has dropped the junction-ended sections no route
    runs over, so it sees the pieces that ship. For each line in pieces, from its biggest group
    of pieces: the cheapest track (`graph()`, a track_graph, or None for no data) from any of
    its stations to a station of another piece, within the Rules' limits. The bridge is cut at
    the stations on it of the lines whose track it runs over (Leicester, Syston), and each part
    is a section of this line, drawn and measured over that track. `ends` lists (station, line
    name, lon, lat) for every section end the register built, so a station build_model dropped
    from its line still cuts a bridge where it was (Wombwell on the Hallam Line).

    OWNERSHIP. A part mostly on another register line's track is listed in the line's
    `borrowed`: that track stays its own line's, and riding it on this line credits that line.
    ownership.run gives a way to the nearest register line beside it and makes the others
    there `losers`, whose sections then credit the owner for that stretch; a tie between two
    lines on one way falls to the lowest ref and then the name, which would hand the Midland
    Main Line's unnamed ways to the Birmingham to Peterborough Line. So each way beside a
    borrowed section is recorded (state["sec_ways"], what ownership reads) borrow_pad_m further
    off than the nearest other register line that is a candidate for it: never the nearest,
    always within the losers' TIE_M. The section also takes that line's high-speed flag there,
    since ownership prefers a line whose flag agrees with the way's. Two cases still give a
    borrowed section ways, logged: no other register line's section beside the way (track an
    OSM line or a station throat held), and a way whose own OSM name is the borrowing line's
    (ownership's name rule; the Hallam Line's name on MVN2 at Wakefield, which the ELR rule
    gives the Caldervale Line). reg_ways gets the line on those ways, so a click there lists
    it."""
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree
    from n02 import walk_order
    KX = rules.kx
    reg = [l for l in lines if l.get("src", "osm") != "osm" and not l.get("service")]
    todo = [l for l in reg if len(section_pieces(l["sections"])) > 1
            and l["name"] not in rules.no_bridge]
    if not todo:
        return
    T = graph()
    if T is None:
        return
    nx, ny = T["nx"], T["ny"]
    # Where each station is on each line's track: the ends of its sections' drawn geometry.
    on_line = defaultdict(dict)              # line id -> station -> (lon, lat)
    st_lines = defaultdict(set)              # station -> names of the register lines it is on
    own_run = defaultdict(lambda: defaultdict(list))   # line id -> station -> [its tracks' points]
    for l in reg:
        for key, pts in geoms.get(l["id"], {}).items():
            a, b = key.split("|")
            on_line[l["id"]][a] = tuple(pts[0])
            on_line[l["id"]][b] = tuple(pts[-1])
            st_lines[a].add(l["name"])
            st_lines[b].add(l["name"])
            if rules.dense and len(pts) > 2:
                own_run[l["id"]][a].append(pts[1:6])
                own_run[l["id"]][b].append(pts[-2:-7:-1])
    # A station cuts a bridge where it is on one of the lines the bridge runs over: as the
    # lines are now, and as the register built them (`ends`).
    cut_ids, cut_xy = [], []
    for lid, got in on_line.items():
        for sid, p in got.items():
            if not stations[sid].get("junction"):
                cut_ids.append(sid)
                cut_xy.append((p[0] * KX, p[1] * 110.57))
    for sid, nm, lon, lat in ends:
        if sid in stations and not stations[sid].get("junction"):
            st_lines[sid].add(nm)
            cut_ids.append(sid)
            cut_xy.append((lon * KX, lat * 110.57))
    cut_tree = cKDTree(np.asarray(cut_xy))
    near_ways = WaysNear(state)
    # Each drawn way's distance to the register lines that are candidates for it in
    # ownership.run (WAY_MIN_FRAC of the way beside their sections), from what
    # register_way_lines measured, for the sections still there.
    from build_model import WAY_MIN_FRAC
    live = {(l["id"], f"{s[0]}|{s[1]}") for l in reg for s in l["sections"]}
    frac, dist = defaultdict(float), {}
    wgeo = (state or {}).get("wgeo") or []
    for k, beside in ((state or {}).get("sec_ways") or {}).items():
        if k in live:
            for j, got, dj in beside:
                frac[(j, k[0])] += got / max(wgeo[j].length, 1e-9)
                if dj < dist.get((j, k[0]), (math.inf,))[0]:
                    dist[(j, k[0])] = (dj, k[1])
    near_ways.alone = Counter()              # line id -> Mercator m of ways only it is beside
    near_ways.reg = defaultdict(dict)        # way index -> {line id: (metres, section key)}
    for (j, x), f in frac.items():
        if f >= WAY_MIN_FRAC:
            near_ways.reg[j][x] = dist[(j, x)]
    near_ways.hs = {l["id"]: (l.get("highspeed_sections") or {}, l.get("highspeed"))
                    for l in reg}

    made, refused = [], {}
    for l in todo:
        lid, name = l["id"], l["name"]
        Tl, routed_w = line_costs(T, l, rules)
        G = Tl["G"]

        def edge_ix(path):
            a, b = path[:-1], path[1:]
            k = np.minimum(a, b).astype(np.int64) * Tl["N"] + np.maximum(a, b)
            return np.searchsorted(Tl["key"], k)

        secs = l["sections"]
        ps = section_pieces(secs)
        piece_km = [sum(secs[i][2] for i in ix) for ix in ps]
        attach = {}                          # station -> the graph node it joins the track at
        for sid, p in on_line[lid].items():
            d, j = Tl["tree"].query((p[0] * KX, p[1] * 110.57))
            if d * 1000 <= rules.attach_m:
                attach[sid] = int(j)
            elif rules.dense:
                # the first vertex of the graph on the station's own section's drawn track
                got = None
                for run in own_run[lid].get(sid, ()):
                    for q in run:
                        d2, j2 = Tl["tree"].query((q[0] * KX, q[1] * 110.57))
                        if d2 * 1000 <= 2.0:
                            dd = math.hypot((q[0] - p[0]) * KX, (q[1] - p[1]) * 110.57)
                            if got is None or dd < got[0]:
                                got = (dd, int(j2))
                            break
                if got is not None:
                    attach[sid] = got[1]
        group = list(range(len(ps)))         # piece -> its group (pieces bridged together)
        st_of = [{s for i in ix for s in secs[i][:2]} for ix in ps]
        tired = set()
        while len(set(group)) > 1:
            gkm = defaultdict(float)
            for k, g in enumerate(group):
                gkm[g] += piece_km[k]
            found = None
            for g in sorted(gkm, key=lambda g: (-gkm[g], g)):
                if g in tired:
                    continue
                src = sorted({attach[s] for k in range(len(ps)) if group[k] == g
                              for s in st_of[k] if s in attach})
                if not src:
                    tired.add(g)
                    continue
                # the cheapest track (routes preferred), and the shortest as a second try
                runs = [dijkstra(M, indices=src, min_only=True, limit=rules.max_km * lim,
                                 return_predecessors=True)[:2]
                        for M, lim in ((G, max(rules.unrouted_cost,
                                               rules.slow_cost or 1.0)), (Tl["Gkm"], 1))]
                cost = runs[0][0]
                best = {}                    # other group -> (cost, station, node)
                for k in range(len(ps)):
                    h = group[k]
                    if h == g:
                        continue
                    for s in sorted(st_of[k]):
                        j = attach.get(s)
                        if j is not None and np.isfinite(cost[j]) and \
                                (h not in best or cost[j] < best[h][0]):
                            best[h] = (float(cost[j]), s, j)
                for h, (_c, sb, jb) in sorted(best.items(), key=lambda kv: kv[1][0]):
                    tried = set()
                    for _cost, pred in runs:
                        if not np.isfinite(_cost[jb]):
                            continue
                        path = [jb]
                        while pred[path[-1]] >= 0:
                            path.append(int(pred[path[-1]]))
                        path = np.asarray(path[::-1])
                        if tuple(path.tolist()) in tried:
                            continue
                        tried.add(tuple(path.tolist()))
                        ja = int(path[0])
                        sa = min(s for k in range(len(ps)) if group[k] == g for s in st_of[k]
                                 if attach.get(s) == ja)
                        crow = math.hypot(
                            (nx[ja] - nx[jb]) * math.cos(math.radians(ny[ja])) * 111.32,
                            (ny[ja] - ny[jb]) * 110.57)
                        e = edge_ix(path)
                        ekm = Tl["km"][e]
                        km = float(ekm.sum())
                        share = float(ekm[routed_w[Tl["ew"][e]]].sum() / max(km, 1e-9))
                        side = min(gkm[g], gkm[h])
                        over = Counter()
                        for ww, kk in zip(Tl["ew"][e].tolist(), ekm.tolist()):
                            over[Tl["name"][ww] or "(unnamed)"] += kk
                        # km over other lines' named track: a gap in the line's own track
                        # (sections build_model dropped) is no other line's
                        other = km - over[name] - over["(unnamed)"]
                        why = None
                        if km > rules.max_km or km > rules.slack * crow + rules.plus_km:
                            why = f"roundabout ({crow:.1f} km crow-fly)"
                        elif share < rules.route_share and km > rules.route_free_km:
                            why = f"no passenger route over {1 - share:.0%} of it"
                        elif other > rules.free_km and other > side:
                            why = f"longer than the {side:.1f} km side it would join"
                        if why:
                            refused[(name,) + tuple(sorted((sa, sb)))] = (
                                km, stations[sa]["name"], stations[sb]["name"], why, over)
                            continue
                        found = (g, h, path, e, sa, sb, km, over)
                        break
                    if found:
                        break
                if found:
                    break
                tired.add(g)
            if not found:
                break
            g, h, path, e, sa, sb, km, over = found
            refused.pop((name,) + tuple(sorted((sa, sb))), None)
            n_new = add_bridge(l, stations, geoms, Tl, path, e, sa, sb, on_line, cut_tree,
                               cut_ids, st_lines, walk_order, reg_ways, state, near_ways, rules)
            made.append((name, km, stations[sa]["name"], stations[sb]["name"], n_new, over))
            for k in range(len(ps)):
                if group[k] == h:
                    group[k] = g
    log(f"{rules.tag}: lines in pieces: {len(todo)} register lines; {len(made)} gaps bridged "
        f"over {sum(m[1] for m in made):,.1f} km of track, {len(refused)} not")
    for name, km, a, b, n, over in sorted(made, key=lambda m: -m[1]):
        log(f"    bridged {name}: {a} - {b} {km:.1f} km in {n} sections, over "
            + ", ".join(f"{k} {v:.1f}" for k, v in over.most_common(3)))
    kxm = KX / 111.32
    alone = {l["name"]: v * kxm / 1000
             for l in reg for x, v in near_ways.alone.items() if x == l["id"]}
    log(f"{rules.tag}: borrowed sections are the only register line beside "
        f"{sum(alone.values()):.1f} km of ways (no other line's section within reach), which "
        f"they then own: "
        + ", ".join(f"{k} {v:.1f}" for k, v in sorted(alone.items(), key=lambda kv: -kv[1])))
    for (name, *_ab), (km, a, b, why, over) in sorted(refused.items(), key=lambda kv: -kv[1][0]):
        log(f"    not bridged {name}: {a} - {b} {km:.1f} km, {why}; over "
            + ", ".join(f"{k} {v:.1f}" for k, v in over.most_common(3)))


class WaysNear:
    """register_way_lines' buffer test (build_model) for sections added after it ran: the
    drawn ways beside a section, as (way index, Mercator metres inside, metres from the
    section to the way's middle), which is what state["sec_ways"] holds."""

    R = 20037508.34 / 180.0

    def __init__(self, state):
        from shapely import STRtree
        self.state = state or {}
        g = self.state.get("wgeo") or []
        self.tree = STRtree(g) if g else None

    def proj(self, a):
        lat = np.clip(a[:, 1], -85.05, 85.05)
        return np.column_stack([a[:, 0] * self.R,
                                np.log(np.tan((90 + lat) * np.pi / 360)) / (np.pi / 180) * self.R])

    def beside(self, pts, kind):
        from shapely.geometry import LineString
        from build_model import WAY_BUFFER_M, kind_family
        if self.tree is None or len(pts) < 2:
            return []
        a = np.asarray(pts, dtype=np.float64)
        scale = 1.0 / max(math.cos(math.radians(float(a[:, 1].mean()))), 0.05)
        sec = LineString(self.proj(a))
        buf = sec.buffer(WAY_BUFFER_M * scale, quad_segs=4)
        wgeo, wkind = self.state["wgeo"], self.state["wkind"]
        out = []
        for j in self.tree.query(buf).tolist():
            if kind_family(wkind[j]) != kind_family(kind):
                continue
            w = wgeo[j]
            got = w.intersection(buf).length
            if got > 0:
                out.append((j, got, sec.distance(w.interpolate(0.5, normalized=True)) / scale))
        return out


def add_bridge(l, stations, geoms, T, path, e, sa, sb, on_line, cut_tree, cut_ids, st_lines,
               walk_order, reg_ways, state, near_ways, rules):
    """The bridge `path` (graph nodes from sa's to sb's) as sections of line l, cut at the
    stations on it of the lines whose track it runs over. Returns the number of sections."""
    from build_model import WAY_MIN_FRAC
    KX = rules.kx
    lid, name = l["id"], l["name"]
    nx, ny = T["nx"], T["ny"]
    ew, ekm = T["ew"][e], T["km"][e]
    cum = np.r_[0.0, np.cumsum(ekm)]
    over_names = {T["name"][w] for w in ew.tolist() if T["name"][w]} | {name}
    xy = np.column_stack([nx[path] * KX, ny[path] * 110.57])
    best = {}                                # station -> (metres, path index)
    for i, near in enumerate(cut_tree.query_ball_point(xy, rules.cut_m / 1000)):
        for c in near:
            sid = cut_ids[c]
            if sid in (sa, sb) or not (st_lines[sid] & over_names):
                continue
            d = float(np.hypot(*(cut_tree.data[c] - xy[i]))) * 1000
            if sid not in best or d < best[sid][0]:
                best[sid] = (d, i)
    if rules.dense and len(path) > 1:
        # stations beside the track between two far-apart vertices: points every 50 m along
        # each edge, a station found cutting at the edge's nearer vertex
        seg = np.hypot(*(xy[1:] - xy[:-1]).T)
        k = np.maximum(np.ceil(seg / 0.05).astype(int), 1)
        si = np.repeat(np.arange(len(seg)), k)
        t = (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)) / np.repeat(k, k)
        pts = xy[si] + (xy[si + 1] - xy[si]) * t[:, None]
        for q, near in zip(range(len(pts)), cut_tree.query_ball_point(pts, rules.cut_m / 1000)):
            for c in near:
                sid = cut_ids[c]
                if sid in (sa, sb) or not (st_lines[sid] & over_names):
                    continue
                d = float(np.hypot(*(cut_tree.data[c] - pts[q]))) * 1000
                i = int(si[q]) + (1 if t[q] > 0.5 else 0)
                if sid not in best or d < best[sid][0]:
                    best[sid] = (d, i)
    if rules.cut_ok is not None:
        best = {sid: v for sid, v in best.items() if rules.cut_ok(l, stations[sid])}
    cuts = sorted(((i, sid) for sid, (_d, i) in best.items()))
    stops = [(0, sa)]
    for i, sid in cuts:
        if 0 < i < len(path) - 1 and i > stops[-1][0] and cum[i] - cum[stops[-1][0]] > 0.05:
            stops.append((i, sid))
    if len(stops) > 1 and cum[-1] - cum[stops[-1][0]] <= 0.05:
        stops.pop()                          # a station on top of the far end is that end
    stops.append((len(path) - 1, sb))
    have = {frozenset(s[:2]) for s in l["sections"]}
    g = geoms.setdefault(lid, {})
    sec_ways = (state or {}).get("sec_ways")
    wids = (state or {}).get("wids") or []
    n = 0
    for (i0, a), (i1, b) in zip(stops[:-1], stops[1:]):
        if a == b or frozenset((a, b)) in have:
            continue
        pts = [[round(float(nx[k]), 5), round(float(ny[k]), 5)] for k in path[i0:i1 + 1].tolist()]
        if a == sa:
            pts[0] = [round(v, 5) for v in on_line[lid].get(a, pts[0])]
        if b == sb:
            pts[-1] = [round(v, 5) for v in on_line[lid].get(b, pts[-1])]
        km = round(float(cum[i1] - cum[i0]), 3)
        key = f"{a}|{b}"
        l["sections"].append([a, b, km])
        g[key] = pts
        have.add(frozenset((a, b)))
        part = Counter()
        fast = 0.0
        for w, k in zip(ew[i0:i1].tolist(), ekm[i0:i1].tolist()):
            part[T["name"][w] or ""] += k
            if T["hs"][w]:
                fast += k
        if isinstance(l.get("highspeed_sections"), dict):
            l["highspeed_sections"][key] = fast >= 0.5 * km if km else False
        borrowed = part.get(name, 0.0) + part.get("", 0.0) < 0.5 * sum(part.values())
        if borrowed:
            l.setdefault("borrowed", []).append(key)
        for s in (a, b):
            stations[s]["lines"].add(lid)
        beside = near_ways.beside(pts, l["kind"])
        if borrowed:
            # Never the nearest where another register line's section is beside the way, so
            # that line keeps it; still within ownership's TIE_M of it, so this section is
            # among the losers there and riding it credits that line. And the high-speed flag
            # of the sections it lies beside, since ownership prefers a line whose flag agrees
            # with the way's: the Trent Valley Line's Rugby - Nuneaton section is "fast", its
            # slow-line ways are not, and a bridge flagged slow took them.
            padded, flags = [], Counter()
            for j, got, dj in beside:
                others = [(d, x, k) for x, (d, k) in near_ways.reg.get(j, {}).items() if x != lid]
                if others:
                    d, x, k = min(others)
                    padded.append((j, got, max(dj, d + rules.borrow_pad_m)))
                    by_sec, whole = near_ways.hs[x]
                    flags[by_sec.get(k, whole)] += got
                    continue
                padded.append((j, got, dj))
                if got / max(state["wgeo"][j].length, 1e-9) >= WAY_MIN_FRAC:
                    near_ways.alone[lid] += state["wgeo"][j].length
            beside = padded
            if flags and isinstance(l.get("highspeed_sections"), dict):
                flag = flags.most_common(1)[0][0]
                l["highspeed_sections"][key] = None if flag is None else bool(flag)
        if sec_ways is not None:
            sec_ways[(lid, key)] = beside
        for j, got, _dj in beside:
            wl = state["wgeo"][j].length
            if wl > 0 and got / wl >= WAY_MIN_FRAC:
                reg_ways.setdefault(wids[j], set()).add(lid)
        n += 1
    l["km"] = round(sum(s[2] for s in l["sections"]), 3)
    l["display"] = walk_order([(a, b) for a, b, *_ in l["sections"]])
    return n


def split_pieces(lines, stations, geoms, reg_ways, state, log, rules, line_pieces, graph,
                 ends=()):
    """A register line in pieces once build_model has dropped the junction-ended sections no
    route runs over: bridged where it can be (bridge_gaps), otherwise one line per piece
    (us_register.split_pieces is the original, us_sources.md "Lines in pieces"). The biggest
    piece keeps the id; the others take a hash of the line's id and their lowest stop's id. A
    piece made only of `borrowed` sections is another line's track and is no line; nor is a
    piece under piece_min_km with under two stops. Pieces are told apart in name_en by their
    end stops (rules.piece_name). reg_ways and state["sec_ways"] move to the pieces' ids, so
    clicks and ownership see them. `line_pieces` ({line id: [other pieces' ids]}, the caller's
    LINE_PIECES, which build_model writes into aliases.json as `pieces`) is filled. A line
    named in rules.keep_whole stays one line in pieces."""
    from n02 import walk_order
    bridge_gaps(lines, stations, geoms, reg_ways, state, log, rules, graph, ends)
    line_pieces.clear()
    sec_ways = (state or {}).get("sec_ways") or {}
    wids = (state or {}).get("wids") or []
    made, gone, whole = [], [], []
    for l in list(lines):
        if l.get("src", "osm") == "osm" or not l.get("sections"):
            continue
        ps = section_pieces(l["sections"])
        if len(ps) == 1:
            continue
        lid, secs, name = l["id"], l["sections"], l["name"]
        if name in rules.keep_whole:
            whole.append((name, [round(sum(secs[i][2] for i in ix), 1) for ix in ps]))
            continue
        base_en = l.get("name_en")
        borrowed = set(l.get("borrowed") or ())
        orig = {k: l.get(k) for k in ("borrowed", "highspeed_sections")}
        old_geo = geoms.get(lid, {})
        parts = []
        for ix in ps:
            sub = [secs[i] for i in ix]
            keys = [f"{s[0]}|{s[1]}" for s in sub]
            ends_ = {s for sec in sub for s in sec[:2]}
            stops = sorted(s for s in ends_ if not stations[s].get("junction"))
            km = round(sum(s[2] for s in sub), 3)
            if all(k in borrowed for k in keys) or (len(stops) < 2 and km < rules.piece_min_km):
                gone.append((km, name, sorted(ends_)))
                parts.append((None, keys, ends_))
                continue
            if not any(p is not None for p, _k, _e in parts):
                p = l
            else:
                tag = f"{lid}|piece|{(stops or sorted(ends_))[0]}"
                p = dict(l)
                p["id"] = rules.id_prefix + hashlib.blake2b(tag.encode("utf-8"),
                                                            digest_size=5).hexdigest()
                lines.append(p)
            p["sections"] = sub
            p["km"] = km
            p["display"] = walk_order([(s[0], s[1]) for s in sub])
            for k, v in orig.items():
                if isinstance(v, list):
                    p[k] = [x for x in v if x in keys]
                elif isinstance(v, dict):
                    p[k] = {x: y for x, y in v.items() if x in keys}
            parts.append((p, keys, ends_))
        beside = []
        for p, keys, _ends in parts:
            ws = set()
            for key in keys:
                ws |= {wids[j] for j, *_r in sec_ways.get((lid, key), ())}
            beside.append((p, ws))
        for wid, ls in reg_ways.items():
            if lid in ls:
                to = {p["id"] for p, ws in beside if p is not None and wid in ws}
                ls.discard(lid)
                ls |= to or {lid}
        for p, keys, _ends in parts:
            for key in keys:
                got = sec_ways.pop((lid, key), None)
                if got is not None and p is not None:
                    sec_ways[(p["id"], key)] = got
        geoms[lid] = {}
        for sid in {s for sec in secs for s in sec[:2]}:
            stations[sid]["lines"].discard(lid)
        kept = [p for p, _k, _e in parts if p is not None]
        for p, keys, ends_ in parts:
            if p is None:
                continue
            geoms[p["id"]] = {key: old_geo[key] for key in keys if key in old_geo}
            for sid in ends_:
                stations[sid]["lines"].add(p["id"])
        if len(kept) <= 1:
            continue
        for p in kept:
            stops_ = [s for s in p["display"] if not stations[s].get("junction")]
            if len(stops_) < 2:
                stops_ = [p["display"][0], p["display"][-1]] if p["display"] else []
            if len(stops_) >= 2 and stops_[0] != stops_[-1]:
                p["name_en"] = rules.piece_name(name, dict(l, name_en=base_en), stations,
                                                stops_[0], stops_[-1])
        line_pieces[lid] = [p["id"] for p in kept[1:]]
        made.append((sum(p["km"] for p in kept), name, [p["km"] for p in kept]))
    log(f"{rules.tag}: lines in pieces: {len(made)} register lines whose sections do not all "
        f"connect made {sum(len(m[2]) for m in made)} lines, one per piece; {len(gone)} pieces "
        f"left out ({sum(g[0] for g in gone):.2f} km: bridges alone, or stubs)")
    for km, name, kms in sorted(made, reverse=True):
        log(f"    {km:7.1f} km  {name} -> " + " + ".join(f"{k:.1f}" for k in kms))
    for km, name, ends_ in sorted(gone, reverse=True):
        log(f"    left out: {km:.3f} km  {name}  {' - '.join(ends_[:4])}")
    for name, kms in whole:
        log(f"    kept whole in pieces (Rules.keep_whole): {name} " + " + ".join(
            f"{k:.1f}" for k in kms))
