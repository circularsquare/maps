"""
Turn stats.json into printable ribbons: one offset polyline per route per piece
of shared track.

This is the poster's copy of what riders/nycriders/index.html does in
renderStaticSegments, with one difference that matters. The web map can only
give a whole feature a single `line-offset`, so it fans a route off the track
centreline with a constant number and approximates the handoff at a junction
with a staircase of short pieces. Here there is exactly one scale — the sheet —
so the offset can be baked into the coordinates and varied per vertex, and the
splay where lines part comes out as a smooth curve instead.

The shared-track graph is the same idea as on the web map:

  * every polyline vertex is a node, every consecutive pair an edge;
  * an edge is tagged with the routes that traverse it (GTFS shapes reuse
    identical coordinates wherever trains share track, so this recovers the
    real sharing rather than guessing it from station pairs);
  * edges carrying the same route set chain into a RUN, cut at stations
    because that is where a route's stripe width changes.

A run is then fanned: its routes are laid out side by side across the
centreline in a fixed order, each as wide as its ridership. Where one run ends
and others begin, the biggest bundle at that node holds its offsets and
everything else ramps to meet it, so a pair that arrives together stays
together through the switch and only afterwards drifts onto its own centreline.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import palette

# Ramp length as a multiple of the offset change it has to absorb: a 1:3.5
# splay. Expressed as a ratio rather than a distance so a junction where the
# lines barely move gets a short ramp and a four-line merge gets a long one.
TAPER_RATIO = 3.5
TAPER_MIN_M = 30.0
TAPER_MAX_M = 400.0
TAPER_MIN_OFFSET_M = 0.5     # below this a run just butts onto its neighbour
RAMP_SAMPLES = 24            # vertices inserted along each ramp
MITER_LIMIT = 3.0


# ---------------------------------------------------------------------------
# stats.json -> one geometry and one value per (route, stop pair)
# ---------------------------------------------------------------------------

# Track that two routes physically share but that GTFS draws as two separate
# alignments a few metres apart. The graph keys on identical vertices, so it
# cannot see that sharing and centres each pair on its own line — and since a
# four-stripe bundle here is far wider than the gap between the two alignments,
# they end up drawn on top of each other. Welding the second onto the first
# inside a box makes the sharing visible and the fan then treats them as one
# corridor.
#
# The Manhattan Bridge is the case that matters: B/D use the south tracks from
# Grand St and N/Q the north tracks from Canal St, they share no station at
# either end, and the two pairs cross the river about 15 m apart.
WELDS = [
    {"name": "Manhattan Bridge",
     "donor": "N|Q01|R30",
     "recipients": ["B|D22|R30", "D|D22|R30"],
     "bbox": (-73.9950, 40.6900, -73.9800, 40.7118)},
]


def apply_welds(feats, welds=WELDS, verbose=False):
    for w in welds:
        donor = feats.get(w["donor"])
        if not donor:
            continue
        x0, y0, x1, y1 = w["bbox"]
        inside = lambda c: x0 < c[0] < x1 and y0 < c[1] < y1
        vd = [c for c in donor["coords"] if inside(c)]
        if len(vd) < 2:
            continue
        for key in w["recipients"]:
            f = feats.get(key)
            if not f:
                continue
            idx = [i for i, c in enumerate(f["coords"]) if inside(c)]
            if len(idx) < 2:
                continue
            i0, i1 = idx[0], idx[-1]
            # orient the donor slice to run the same way as the piece it replaces
            head = f["coords"][i0]
            piece = vd if (abs(head[1] - vd[0][1]) + abs(head[0] - vd[0][0])
                           <= abs(head[1] - vd[-1][1]) + abs(head[0] - vd[-1][0])) \
                else vd[::-1]
            f["coords"] = f["coords"][:i0] + [list(c) for c in piece] + f["coords"][i1 + 1:]
            if verbose:
                print(f"  weld {w['name']}: {key} {i1 - i0 + 1} pts -> {len(piece)}")


def load_features(stats_path, hour=None):
    """Collapse stats.json to {key: {route, coords, value}}.

    Directions are collapsed to the busier one and express variants folded into
    their base route, exactly as the web map does before it draws.

    hour=None totals the whole day; hour=0..23 takes that hour alone.
    """
    data = json.loads(open(stats_path, encoding="utf-8").read())

    def value_of(seg):
        bh = seg["by_hour"]
        if hour is None:
            return sum(h[0] for h in bh)
        return bh[hour][0]

    feats = {}
    for seg in data["segments"]:
        route = palette.STATIC_FOLD.get(seg["route"], seg["route"])
        a, b = sorted((seg["from"], seg["to"]))
        key = f"{route}|{a}|{b}"
        v = value_of(seg)
        f = feats.get(key)
        if f is None:
            feats[key] = {"route": route, "coords": seg["coords"], "value": v,
                          "from": a, "to": b}
        else:
            # both directions of the same route, and any folded express: the
            # stripe shows the busier direction, but a folded express is extra
            # trains on the same track, so it adds
            if seg["route"] in palette.STATIC_FOLD:
                f["value"] += v
            else:
                f["value"] = max(f["value"], v)
    apply_welds(feats)
    for f in feats.values():
        f["coords"] = densify(f["coords"])
    return data, feats


# GTFS shapes are dense round curves and bare on the straights — a run can be
# 646 m of track with two vertices in it. That is fine for drawing and useless
# for editing, because the hand-nudge tool can only grab vertices, and the two
# it would find there are the stations at either end. Splitting long segments
# gives every stretch something to take hold of.
#
# Deterministic in the two endpoints, so a segment shared by two routes
# subdivides identically in both and the graph still sees it as shared. Run
# after the welds, so a welded copy densifies the same way as its donor.
DENSIFY_M = 60.0


def densify(coords, max_m=DENSIFY_M):
    out = [coords[0]]
    for a, b in zip(coords, coords[1:]):
        dx = (b[0] - a[0]) * math.cos(math.radians(a[1]))
        d = math.hypot(dx, b[1] - a[1]) * 111320.0
        n = int(d // max_m)
        for k in range(1, n + 1):
            t = k / (n + 1)
            out.append([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
        out.append(b)
    return out


def station_totals(data, hour=None):
    """(lon, lat, boardings) per complex — entries plus transfers, as the web
    map's 'all' metric."""
    out = []
    for s in data["stations"]:
        total = 0.0
        for arr in s["by_route_hour"].values():
            if hour is None:
                total += sum(row[0] + row[1] for row in arr)
            else:
                total += arr[hour][0] + arr[hour][1]
        if total > 0:
            out.append((s["lon"], s["lat"], total, s["name"]))
    return out


# ---------------------------------------------------------------------------
# shared-track graph
# ---------------------------------------------------------------------------

def _node_key(c):
    return f"{c[0]:.5f},{c[1]:.5f}"


def build_runs(feats):
    """Chain the track graph into runs.

    Returns a list of {coords (lon/lat), members: [route...] in draw order,
    feat_of: {route: feature key}, ends: (node, node)} plus a
    node -> [run index] index.
    """
    node_coord, node_edges, edges = {}, defaultdict(list), {}
    break_nodes, feat_nodes = set(), {}

    for key, f in feats.items():
        pts = []
        for c in f["coords"]:
            nk = _node_key(c)
            if pts and pts[-1] == nk:      # stats.json repeats the first vertex
                continue
            pts.append(nk)
            node_coord.setdefault(nk, c)
        if len(pts) < 2:
            continue
        feat_nodes[key] = pts
        break_nodes.add(pts[0])
        break_nodes.add(pts[-1])
        for a, b in zip(pts, pts[1:]):
            ek = (a, b) if a < b else (b, a)
            if ek not in edges:
                edges[ek] = {"a": ek[0], "b": ek[1], "feats": set()}
                node_edges[ek[0]].append(ek)
                node_edges[ek[1]].append(ek)
            edges[ek]["feats"].add(key)

    sig = {ek: "/".join(sorted({feats[k]["route"] for k in e["feats"]}))
           for ek, e in edges.items()}

    def other(ek, n):
        return edges[ek]["b"] if edges[ek]["a"] == n else edges[ek]["a"]

    def step(node, ek, seen):
        if node in break_nodes:
            return None
        inc = node_edges[node]
        if len(inc) != 2:                  # a switch: the run stops here
            return None
        nxt = inc[0] if inc[1] == ek else inc[1] if inc[0] == ek else None
        if nxt is None or nxt in seen or sig[nxt] != sig[ek]:
            return None
        return nxt

    seen, chains = set(), []
    for ek0 in edges:
        if ek0 in seen:
            continue
        seen.add(ek0)
        chain = [ek0]
        node = edges[ek0]["b"]
        while True:
            nxt = step(node, chain[-1], seen)
            if not nxt:
                break
            seen.add(nxt)
            chain.append(nxt)
            node = other(nxt, node)
        node = edges[ek0]["a"]
        while True:
            nxt = step(node, chain[0], seen)
            if not nxt:
                break
            seen.add(nxt)
            chain.insert(0, nxt)
            node = other(nxt, node)
        chains.append(chain)

    runs, node_runs = [], defaultdict(list)
    for chain in chains:
        a, b = edges[chain[0]]["a"], edges[chain[0]]["b"]
        if len(chain) > 1 and a in (edges[chain[1]]["a"], edges[chain[1]]["b"]):
            a, b = b, a
        nodes = [a, b]
        for ek in chain[1:]:
            nodes.append(other(ek, nodes[-1]))

        by_route = defaultdict(list)
        for k in edges[chain[0]]["feats"]:
            by_route[feats[k]["route"]].append(k)
        members = sorted(by_route, key=palette.sort_key)
        # widest feature wins where a route covers the run through more than one
        # stop pair (an express overlapping its own local hop)
        feat_of = {r: max(by_route[r], key=lambda k: feats[k]["value"])
                   for r in members}

        # Provisional orientation: the lowest-sorted member's stored polyline
        # direction. stats.json normalises every sub-segment to pattern
        # direction 0, so this is at least stable. orient_runs then fixes it up
        # so that neighbouring runs agree, which the anchor alone does not
        # guarantee — see there.
        anchor = feat_nodes.get(feat_of[members[0]])
        if anchor:
            for i in range(len(anchor) - 1):
                if anchor[i] == nodes[1] and anchor[i + 1] == nodes[0]:
                    nodes.reverse()
                    break
                if anchor[i] == nodes[0] and anchor[i + 1] == nodes[1]:
                    break

        idx = len(runs)
        for n in (nodes[0], nodes[-1]):
            node_runs[n].append(idx)
        runs.append({"coords": [node_coord[n] for n in nodes],
                     "nodes": nodes, "members": members, "feat_of": feat_of,
                     "ends": (nodes[0], nodes[-1])})
    orient_runs(runs, node_runs)
    graph = {"coord": node_coord,
             "degree": {n: len(e) for n, e in node_edges.items()},
             "neighbours": {n: [other(ek, n) for ek in node_edges[n]]
                            for n in node_edges}}
    return runs, node_runs, graph


def orient_runs(runs, node_runs):
    """Make neighbouring runs agree about which way round they are.

    An offset is measured to the right of the run's coordinate direction, so
    which physical side of the corridor a route sits on depends on which way the
    run happens to be stored. Anchoring each run on its lowest-sorted member is
    stable but not consistent: the member doing the anchoring differs from run
    to run, and two runs that continue into each other can end up both pointing
    at the shared node. The same positive offset then puts the route on opposite
    sides either side of that node, and the ribbon jumps clean across the
    corridor — which is exactly the two breaks the E has where it leaves 8 Av
    for 53 St, and the G and the M have one each.

    Two runs meeting at a node continue into each other when their tangents
    there point apart; for those, one must arrive at the node and the other
    leave it. That is a 2-colouring, solved by breadth-first search over the
    runs. Tangents pointing the SAME way mean two branches leaving a switch
    together, which is not a continuation and gets no constraint — without that
    test a fork would demand that its two branches disagree.
    """
    def tangent(i, e):
        p = runs[i]["coords"]
        a, b = (p[0], p[1]) if e == 0 else (p[-1], p[-2])
        dx = (b[0] - a[0]) * math.cos(math.radians(a[1]))
        dy = b[1] - a[1]
        n = math.hypot(dx, dy)
        return (dx / n, dy / n) if n else (0.0, 0.0)

    adj = defaultdict(list)
    for node, idxs in node_runs.items():
        ends = [(i, e) for i in set(idxs) for e in (0, 1)
                if runs[i]["ends"][e] == node]
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                i, ei = ends[a]
                j, ej = ends[b]
                if i == j or not (set(runs[i]["members"]) & set(runs[j]["members"])):
                    continue
                ti, tj = tangent(i, ei), tangent(j, ej)
                if ti[0] * tj[0] + ti[1] * tj[1] >= -0.1:
                    continue                       # a fork, not a continuation
                adj[i].append((j, 1 if ei == ej else 0))
                adj[j].append((i, 1 if ei == ej else 0))

    flip = [None] * len(runs)
    conflicts = 0
    for start in range(len(runs)):
        if flip[start] is not None:
            continue
        flip[start] = 0
        stack = [start]
        while stack:
            i = stack.pop()
            for j, c in adj[i]:
                want = flip[i] ^ c
                if flip[j] is None:
                    flip[j] = want
                    stack.append(j)
                elif flip[j] != want:
                    conflicts += 1
    for i, f in enumerate(flip):
        if f:
            runs[i]["coords"].reverse()
            runs[i]["nodes"].reverse()
            runs[i]["ends"] = (runs[i]["ends"][1], runs[i]["ends"][0])
    return sum(1 for f in flip if f), conflicts // 2


# ---------------------------------------------------------------------------
# fan + handoff
# ---------------------------------------------------------------------------

def fan_relative(runs, widths, gap_m):
    """Where each route sits ACROSS its run, relative to the bundle's own
    centre: metres, positive = right of the run's direction of travel."""
    out = []
    for run in runs:
        parts = [(r, widths[run["feat_of"][r]]) for r in run["members"]]
        parts = [(r, w) for r, w in parts if w > 0]
        total = sum(w for _, w in parts) + gap_m * (len(parts) - 1)
        cum, rel = -total / 2.0, {}
        for r, w in parts:
            rel[r] = (cum + w / 2.0, w)
            cum += w + gap_m
        out.append(rel)
    return out


def solve_centres(runs, node_runs, rel, reg=0.02, sweeps=400, omega=1.7):
    """Slide each bundle sideways so that routes keep the SAME offset from run
    to run wherever they can.

    Centring every bundle on its own track is what makes lines wander. The A and
    C pick up the E at 50 St and hand it over at 42 St; centre each run
    separately and the pair swings out and back again, two big shifts that
    mostly cancel — which is what a reader sees as the line wobbling, and it is
    worse where two of those land inside one curve. But look at what actually
    changes: routes are stacked in a fixed order, so a route joining or leaving
    at the END of that order shifts everyone else by one constant. Giving each
    run a free centre lets that constant be absorbed, and the A/C run dead
    straight through while the E peels off.

    So: one unknown per run, its bundle centre. Each route shared by two runs
    that meet at a node wants their offsets equal, weighted by how wide the
    stripe is, since a fat trunk moving is far more visible than a branch
    moving. A weak spring back to zero keeps a bundle from drifting off its own
    track over a long line, and is also what makes the system non-singular.

    Solved by successive over-relaxation, which is a few milliseconds at this
    size and reads the same in Python and in the browser.
    """
    links = [[] for _ in runs]                        # (other run, route)
    for idxs in node_runs.values():
        for a_i in range(len(idxs)):
            for b_i in range(a_i + 1, len(idxs)):
                i, j = idxs[a_i], idxs[b_i]
                if i == j:
                    continue
                for route in rel[i]:
                    if route in rel[j]:
                        links[i].append((j, route))
                        links[j].append((i, route))

    wmax = max((w for r in rel for _, w in r.values()), default=1.0) or 1.0
    c = np.zeros(len(runs))
    for _ in range(sweeps):
        for i, ls in enumerate(links):
            num = 0.0
            den = reg
            for j, route in ls:
                w = rel[i][route][1] / wmax
                num += w * (c[j] + rel[j][route][0] - rel[i][route][0])
                den += w
            c[i] += omega * (num / den - c[i])
    return c


def absolute(rel, c):
    """Bundle-relative fan positions plus solved centres -> final offsets."""
    return [{r: (v[0] + c[i], v[1]) for r, v in rel[i].items()}
            for i in range(len(rel))]


def handoff(runs, node_runs, offs, run_idx, end_idx, route):
    """The offset this stripe holds exactly at one end of its run, or None if
    there is nothing to hand over to.

    Every run meeting at the node ramps to the same value — the mean of what
    they each want — so the pieces always join. After solve_centres most of
    these differ by centimetres and no ramp is drawn at all; what is left is the
    genuine changes, where the stacking order itself has to change.
    """
    node = runs[run_idx]["ends"][end_idx]
    peers = [offs[j][route][0] for j in node_runs[node] if route in offs[j]]
    if len(peers) < 2:
        return None                                   # terminus
    return sum(peers) / len(peers)


# ---------------------------------------------------------------------------
# curvature relief
# ---------------------------------------------------------------------------

def smooth_nodes(xy, graph, half_width, passes=140, lam=0.5, cap=0.75):
    """Round the corners of the shared track graph, in projected metres.

    A stripe offset by `o` onto the inside of a curve of radius R has radius
    R - o, and at R < o it turns inside out: the ribbon doubles back and ties a
    little knot. That is not a rare failure here. At about a kilometre to the
    inch a 60-mil stripe is 62 m wide on the ground, so the four-track bundles
    are 100-200 m across, while the tightest subway curves — the Canal St
    reverse curve, the South Ferry loop, the Coney Island approaches — turn
    inside 100 m. 78 of 1,169 ribbons folded before this existed.

    The cure is to give the centreline itself a minimum radius, by Laplacian
    smoothing with the displacement capped per node at a fraction of the widest
    bundle that runs through it. That makes the cap do the discriminating: a
    single thin branch line barely moves and keeps its geography, while the
    Lexington Av trunk gets its corners opened out to something its own width
    can turn through. It is also, independently, what makes an elbow read as one
    clean curve instead of a polyline of chords.

    Junctions and dead ends are pinned, so runs still meet exactly where they
    used to and no seam opens up.
    """
    keys = list(xy.keys())
    idx = {k: i for i, k in enumerate(keys)}
    p0 = np.array([xy[k] for k in keys], float)
    p = p0.copy()
    free = np.array([graph["degree"].get(k, 0) == 2 for k in keys])
    limit = np.array([cap * half_width.get(k, 0.0) for k in keys])
    nb = np.array([[idx[n] for n in graph["neighbours"][k]] if graph["degree"].get(k, 0) == 2
                   else [idx[k], idx[k]] for k in keys])

    for _ in range(passes):
        mid = 0.5 * (p[nb[:, 0]] + p[nb[:, 1]])
        p[free] += lam * (mid[free] - p[free])
        d = p - p0
        dist = np.hypot(d[:, 0], d[:, 1])
        over = dist > limit
        if over.any():
            scale = np.where(dist > 0, limit / np.maximum(dist, 1e-9), 0.0)
            p[over] = p0[over] + d[over] * scale[over, None]
    return {k: p[i] for i, k in enumerate(keys)}


def load_nudges(path):
    """The hand edits, as {"move": {...}, "add": [...], "drop": [...]}.

    Every key is "<route>|<node key>", because a control point belongs to one
    route: that is the whole point of the chain model. The node half is the
    quantised lon/lat of an original GTFS vertex, so an edit survives re-running
    everything upstream — a moved or added point is re-forced into the chain and
    a dropped one is re-dropped, rather than being at the mercy of what the
    automatic thinning decides this time.

    A bare object of key -> [dx, dy] is read as moves only, which is the format
    the first version of the editor wrote.
    """
    p = Path(path)
    if not p.exists():
        return {"move": {}, "add": set(), "drop": set()}
    raw = json.loads(p.read_text(encoding="utf-8"))
    if "move" not in raw and "add" not in raw and "drop" not in raw:
        raw = {"move": raw}
    return {
        "move": {k: np.array(v, float)
                 for k, v in (raw.get("move") or {}).items() if "|" in k},
        "add": {k for k in (raw.get("add") or []) if "|" in k},
        "drop": {k for k in (raw.get("drop") or []) if "|" in k},
        # where each moved point sat when it was moved, so drift can be spotted
        "base": {k: np.array(v, float)
                 for k, v in (raw.get("base") or {}).items() if "|" in k},
    }


def edit_report(nudges, cs, drift_m=4.0):
    """One line about the hand edits: how many applied, how many lost, and how
    many are now measured from geometry that has since moved.

    A move is a DELTA from where the automatic pass put the point, so changing
    the widths, the smoothing or the chain building shifts what it is relative
    to. The edit is not lost, but it no longer means quite what it meant — and
    silently landing somewhere else is the failure worth naming out loud.
    """
    n = len(nudges["move"]) + len(nudges["add"]) + len(nudges["drop"])
    if not n:
        return None
    here = {f"{c['route']}|{k}": c["pos"][i]
            for c in cs for i, k in enumerate(c["keys"])}
    keys = list(nudges["move"]) + list(nudges["add"]) + list(nudges["drop"])
    lost = sum(1 for k in keys if k not in here)
    drift = 0
    for k, b in nudges["base"].items():
        p = here.get(k)
        if p is not None and math.hypot(p[0] - b[0], p[1] - b[1]) > drift_m:
            drift += 1
    msg = (f"  hand edits: {len(nudges['move'])} moved, {len(nudges['add'])} added, "
           f"{len(nudges['drop'])} deleted")
    if lost:
        msg += f"; {lost} no longer match a vertex"
    if drift:
        msg += (f"; {drift} sit on geometry that has moved more than "
                f"{drift_m:.0f} m since they were placed — worth re-checking")
    if nudges["move"] and not nudges["base"]:
        msg += "; saved before positions were recorded, so drift cannot be checked"
    return msg


def nudge_sets(nudges):
    """(forced, dropped) as (route, node key) pairs, for chains.build."""
    forced = {tuple(k.split("|", 1))
              for k in list(nudges["move"]) + list(nudges["add"])}
    dropped = {tuple(k.split("|", 1)) for k in nudges["drop"]}
    return forced, dropped - forced


def node_half_widths(runs, offs):
    """Widest bundle passing through each node — what smooth_nodes caps by."""
    out = {}
    for run, o in zip(runs, offs):
        if not o:
            continue
        h = max(abs(v[0]) + v[1] / 2.0 for v in o.values())
        for n in run["nodes"]:
            if h > out.get(n, 0.0):
                out[n] = h
    return out


# ---------------------------------------------------------------------------
# baked offset geometry
# ---------------------------------------------------------------------------

def _normals(xy):
    """Unit outward normals for each vertex of a polyline, mitred at bends.

    Positive offset moves the line to the RIGHT of its direction of travel,
    matching MapLibre's line-offset so the poster puts each route on the same
    side of its trunk as the web map does.
    """
    d = np.diff(xy, axis=0)
    seg_len = np.hypot(d[:, 0], d[:, 1])
    seg_len[seg_len == 0] = 1.0
    t = d / seg_len[:, None]
    # right-hand normal in a y-up frame
    n = np.column_stack([t[:, 1], -t[:, 0]])

    out = np.empty_like(xy)
    out[0], out[-1] = n[0], n[-1]
    if len(xy) > 2:
        m = n[:-1] + n[1:]
        norm = np.hypot(m[:, 0], m[:, 1])
        flat = norm < 1e-9                    # a doubling-back spike
        norm[flat] = 1.0
        m = m / norm[:, None]
        cos_half = np.clip((m * n[1:]).sum(axis=1), 1e-3, 1.0)
        scale = np.minimum(1.0 / cos_half, MITER_LIMIT)
        out[1:-1] = m * scale[:, None]
        out[1:-1][flat] = n[:-1][flat]
    return out


def joint_normals(runs, node_runs, xy):
    """One shared normal per run end, so two runs meeting at an angle put their
    offset endpoints in the SAME place.

    An offset is taken perpendicular to the line it offsets, so where two runs
    meet at an angle their endpoints swing apart by 2*offset*sin(half the turn)
    — up to 100 m here, which is a visible break in the middle of a line.
    Mitring the joint, exactly as one would inside a polyline, closes it: both
    ends use the bisector of the two tangents, so both land on the same point.

    Only run ends with a single continuation get one. At a switch the trunk has
    two partners and can only agree with one, but the branches leave at a
    shallow angle there, so their own normals are already within a metre or two.
    """
    tang, out = {}, {}
    for i, p in enumerate(xy):
        if len(p) < 2:
            continue
        for e, (a, b) in ((0, (p[1], p[0])), (1, (p[-2], p[-1]))):
            d = b - a
            n = math.hypot(d[0], d[1])
            if n:
                tang[(i, e)] = d / n            # direction of travel at that end

    for node, idxs in node_runs.items():
        ends = [(i, e) for i in set(idxs) for e in (0, 1)
                if runs[i]["ends"][e] == node and (i, e) in tang]
        for i, e in ends:
            # travel through the node: end 1 arrives, end 0 leaves
            mine = tang[(i, e)] * (1 if e == 1 else -1)
            partners = []
            for j, e2 in ends:
                if j == i and e2 == e:
                    continue
                theirs = tang[(j, e2)] * (1 if e2 == 1 else -1)
                if float(mine @ theirs) > 0.1:   # same way through the node
                    partners.append(theirs)
            if len(partners) != 1:
                continue
            n1 = np.array([mine[1], -mine[0]])
            n2 = np.array([partners[0][1], -partners[0][0]])
            m = n1 + n2
            ln = math.hypot(m[0], m[1])
            if ln < 1e-9:
                continue
            m = m / ln
            out[(i, e)] = m * min(1.0 / max(float(m @ n2), 1e-3), MITER_LIMIT)
    return out


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ribbon(xy, off_mid, off_start, off_end, n_start=None, n_end=None):
    """One stripe as a baked offset polyline.

    xy is the run in projected metres; off_mid is the stripe's own offset and
    off_start/off_end the offsets it must hold at the run's two ends (None to
    hold off_mid). The ramps between them are smoothstepped, so the splay
    leaves and rejoins the parallel run without a corner.
    """
    seg = np.hypot(*np.diff(xy, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return None

    def ramp_len(target):
        if target is None or abs(target - off_mid) < TAPER_MIN_OFFSET_M:
            return 0.0
        want = min(TAPER_MAX_M,
                   max(TAPER_MIN_M, abs(target - off_mid) * TAPER_RATIO))
        return min(want, total * 0.45)

    l0, l1 = ramp_len(off_start), ramp_len(off_end)
    if l0 + l1 > total * 0.98:
        k = total * 0.98 / (l0 + l1)
        l0, l1 = l0 * k, l1 * k

    # sample positions: every original vertex, plus enough inside each ramp for
    # the curve to read as a curve rather than as a chord
    pos = [cum]
    if l0 > 0:
        pos.append(np.linspace(0.0, l0, RAMP_SAMPLES))
    if l1 > 0:
        pos.append(np.linspace(total - l1, total, RAMP_SAMPLES))
    s = np.unique(np.concatenate(pos))

    # points and normals at those positions
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(cum) - 2)
    span = np.where(seg[idx] > 0, seg[idx], 1.0)
    t = ((s - cum[idx]) / span)[:, None]
    pts = xy[idx] + t * (xy[idx + 1] - xy[idx])

    vert_n = _normals(xy)
    d = np.diff(xy, axis=0)
    sl = np.hypot(d[:, 0], d[:, 1])
    sl[sl == 0] = 1.0
    seg_n = np.column_stack([d[:, 1], -d[:, 0]]) / sl[:, None]
    nrm = seg_n[idx]
    on_vertex = np.isclose(t[:, 0], 0.0)      # reuse the mitred normal there
    nrm[on_vertex] = vert_n[idx[on_vertex]]
    if n_start is not None:
        nrm[0] = n_start
    if n_end is not None:
        nrm[-1] = n_end

    off = np.full(len(s), float(off_mid))
    if l0 > 0:
        m = s <= l0
        off[m] = off_start + (off_mid - off_start) * _smoothstep(s[m] / l0)
    if l1 > 0:
        m = s >= total - l1
        off[m] = off_mid + (off_end - off_mid) * _smoothstep((s[m] - (total - l1)) / l1)

    return pts + nrm * off[:, None]
