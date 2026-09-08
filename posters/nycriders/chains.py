"""
The editable geometry model: one line per route, owned by that route.

`ribbons.py` places lines automatically — it recovers shared track, orients the
runs, stacks the fan and solves the bundle centres — but what it hands back is a
corridor centreline plus a scalar offset per route. That is the wrong thing to
hand-edit, for two reasons:

  * a node belongs to the corridor, so moving it slides every route on that
    corridor in parallel. You cannot fix one line's shape, only carry the whole
    bundle around;
  * continuity between the pieces has to be *computed* — handoff offsets,
    mitred joints, tapered ramps — and every remaining tear on the sheet has
    been a case that computation did not cover.

So this module bakes that placement down one level, into per-route control
points, and everything after it works on those. A route's line is then a single
polyline through points that belong to it alone: moving one moves that route and
nothing else, and the line cannot come apart, because there is no seam left to
come apart at. Where a route genuinely splits — the 5 in the Bronx — its
branches share the junction control point and are otherwise independent, which
is a Y with a corner in it rather than a tear.

The drawn curve is a centripetal Catmull-Rom spline through the control points,
not the control polygon itself, so a shape is derived from its points rather
than carrying the GTFS micro-wiggle around. Move one point and the curve
re-derives smoothly over its neighbours; there is nothing to preserve.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

import palette
import ribbons

# Control points: every run boundary (station or switch) is kept, so a span
# never straddles a change of stripe width, and the interior is thinned to
# whatever the shape actually needs.
RDP_TOL_M = 9.0          # interior points closer than this to the chord go
MAX_SPAN_M = 260.0       # ...but never leave a stretch longer than this bare
SPLINE_SAMPLES = 10      # curve samples per span


# ---------------------------------------------------------------------------
# where each route sits at each node
# ---------------------------------------------------------------------------

def route_positions(runs, offs, xy, joints):
    """(route, node key) -> one position in metres.

    One position per pair, not one per pair per run: a node on a run boundary
    gets the mean of what the runs meeting there each want, so the two sides
    land on the same point and the line is continuous by construction rather
    than by a handoff calculation that has to get every case right.
    """
    acc = defaultdict(lambda: [np.zeros(2), 0])
    for i, run in enumerate(runs):
        if not offs[i] or len(xy[i]) < 2:
            continue
        nrm = ribbons._normals(xy[i]).copy()
        if (i, 0) in joints:
            nrm[0] = joints[(i, 0)]
        if (i, 1) in joints:
            nrm[-1] = joints[(i, 1)]
        for route, (off, _w) in offs[i].items():
            for k, key in enumerate(run["nodes"]):
                a = acc[(route, key)]
                a[0] += xy[i][k] + nrm[k] * off
                a[1] += 1
    return {k: v[0] / v[1] for k, v in acc.items()}


# ---------------------------------------------------------------------------
# per-route chains
# ---------------------------------------------------------------------------

def _rdp(pts, keep, lo, hi, tol):
    """Mark the interior points between lo and hi that the shape needs."""
    if hi <= lo + 1:
        return
    a, b = pts[lo], pts[hi]
    ab = b - a
    L = math.hypot(ab[0], ab[1])
    worst, wi = -1.0, -1
    for i in range(lo + 1, hi):
        if L < 1e-9:
            d = math.hypot(*(pts[i] - a))
        else:
            d = abs(ab[0] * (pts[i][1] - a[1]) - ab[1] * (pts[i][0] - a[0])) / L
        if d > worst:
            worst, wi = d, i
    if worst > tol:
        keep[wi] = True
        _rdp(pts, keep, lo, wi, tol)
        _rdp(pts, keep, wi, hi, tol)


def build(runs, offs, xy, joints, forced=frozenset(), dropped=frozenset(),
          rdp_tol=RDP_TOL_M, max_span=MAX_SPAN_M):
    """Per-route chains: the whole walk, plus which of its points are controls.

    A chain carries every vertex the route passes through, and `ctrl` says which
    of them currently shape the curve. Keeping the rest rather than throwing
    them away is what lets the editor ADD a control point: there is always
    something under the line to promote, and it is an existing GTFS vertex, so
    it has a stable key and lands exactly on the route. Adding and removing
    points is then the same operation as the automatic thinning, run by hand.

    `forced` is a set of (route, node key) that must be a control point — the
    ones the hand-edit file has an opinion about, so re-running the automatic
    pass cannot orphan an edit. `dropped` is the reverse, for points deleted by
    hand; a chain's two ends are never dropped, since that would shorten it.
    """
    pos = route_positions(runs, offs, xy, joints)

    # per-route adjacency over node keys, and the stripe width on each edge
    adj = defaultdict(lambda: defaultdict(set))
    width = {}
    for i, run in enumerate(runs):
        for route, (_off, w) in offs[i].items():
            for a, b in zip(run["nodes"], run["nodes"][1:]):
                if a == b:
                    continue
                adj[route][a].add(b)
                adj[route][b].add(a)
                width[(route, a, b)] = width[(route, b, a)] = w

    # run boundaries are always control points: a span then never straddles a
    # width change, and every station ends up grabbable
    boundary = set()
    for run in runs:
        boundary.add(run["nodes"][0])
        boundary.add(run["nodes"][-1])

    chains = []
    for route, nb in adj.items():
        used = set()
        # Seed from the route's own termini first, so the long strokes get built
        # before anything claims their edges; then from its branch nodes; then
        # whatever is left, which is a closed loop like the 6 in the Bronx.
        order = ([n for n in nb if len(nb[n]) == 1]
                 + [n for n in nb if len(nb[n]) >= 3] + list(nb))
        for s0 in order:
            for first in list(nb[s0]):
                if _ek(s0, first) in used:
                    continue
                used.add(_ek(s0, first))
                walk = [s0, first]
                prev, cur = s0, first
                while True:
                    nxt = _continue(route, prev, cur, nb, used, pos)
                    if nxt is None:
                        break
                    used.add(_ek(cur, nxt))
                    walk.append(nxt)
                    prev, cur = cur, nxt
                prev, cur = first, s0
                while True:
                    nxt = _continue(route, prev, cur, nb, used, pos)
                    if nxt is None:
                        break
                    used.add(_ek(cur, nxt))
                    walk.insert(0, nxt)
                    prev, cur = cur, nxt
                if len(walk) < 2:
                    continue
                chains.append(_make_chain(route, walk, pos, width, boundary,
                                          forced, dropped, rdp_tol, max_span))
    return [c for c in chains if c]


def _ek(a, b):
    return (a, b) if a < b else (b, a)


# How straight a continuation has to be to be taken THROUGH a branch node, as
# cos(turn). 0 is a right angle.
STROKE_MIN_COS = 0.0


def _continue(route, prev, cur, nb, used, pos):
    """Which way the line carries on at `cur`, having arrived from `prev`.

    At a plain through-node this is just the other edge. At a node where the
    route branches it is the STRAIGHTEST unused edge — the one a train would
    take if it were running through — so the two legs that line up become one
    chain and only the leg that genuinely turns off starts a new one.

    Without this every leg of a branch is its own chain ending at the shared
    point, each with its own free spline end, and three curves meeting a point
    from three independent directions make a cusp rather than a junction: the F
    at 6 Av reads as a southbound and an eastbound line converging head-on
    instead of running through.
    """
    cand = [c for c in nb[cur] if _ek(cur, c) not in used]
    if not cand:
        return None
    if len(nb[cur]) == 2:
        return cand[0]          # plain through-node: carry on whatever the angle
    px, py = pos[(route, prev)]
    cx, cy = pos[(route, cur)]
    ix, iy = cx - px, cy - py
    L = math.hypot(ix, iy)
    if L < 1e-9:
        return None
    ix, iy = ix / L, iy / L
    best, best_dot = None, STROKE_MIN_COS
    for c in cand:
        qx, qy = pos[(route, c)]
        ox, oy = qx - cx, qy - cy
        M = math.hypot(ox, oy)
        if M < 1e-9:
            continue
        d = (ix * ox + iy * oy) / M
        if d > best_dot:
            best_dot, best = d, c
    return best


def _make_chain(route, walk, pos, width, boundary, forced, dropped,
                rdp_tol, max_span):
    pts = np.array([pos[(route, k)] for k in walk], float)
    n = len(walk)
    keep = [False] * n
    keep[0] = keep[-1] = True
    for i, k in enumerate(walk):
        if k in boundary or (route, k) in forced:
            keep[i] = True

    # thin the interior of each kept-to-kept stretch down to the shape
    idx = [i for i in range(n) if keep[i]]
    for a, b in zip(idx, idx[1:]):
        _rdp(pts, keep, a, b, rdp_tol)

    # ...then make sure nothing is left bare over a long distance
    while True:
        idx = [i for i in range(n) if keep[i]]
        added = False
        for a, b in zip(idx, idx[1:]):
            if b <= a + 1:
                continue
            d = float(np.hypot(*(pts[b] - pts[a])))
            if d > max_span:
                keep[(a + b) // 2] = True
                added = True
        if not added:
            break

    # hand deletions win over everything except the chain's own ends
    for i in range(1, n - 1):
        if (route, walk[i]) in dropped:
            keep[i] = False

    ctrl = [i for i in range(n) if keep[i]]
    if len(ctrl) < 2:
        return None
    return {
        "route": route,
        "color": palette.color(route),
        "priority": palette.priority(route),
        "keys": walk,
        "pos": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in pts],
        "ctrl": ctrl,
        "w": [round(width.get((route, walk[i], walk[i + 1]), 0.0), 3)
              for i in range(n - 1)],
    }


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def spline(P, samples=SPLINE_SAMPLES, alpha=0.5):
    """Centripetal Catmull-Rom through the control points.

    Centripetal (alpha 0.5) rather than uniform because uniform Catmull-Rom
    loops and cusps wherever two control points are much closer together than
    their neighbours, which is exactly what a station pair inside a long block
    looks like. Returns the samples and, for each, the span it belongs to, so a
    stripe can change width at a station without a seam.
    """
    P = np.asarray(P, float)
    n = len(P)
    if n < 2:
        return P, np.zeros(len(P), int)
    if n == 2:
        t = np.linspace(0, 1, samples + 1)[:, None]
        return P[0] + t * (P[1] - P[0]), np.zeros(samples + 1, int)

    ext = np.vstack([P[0] + (P[0] - P[1]), P, P[-1] + (P[-1] - P[-2])])
    out, owner = [], []
    for i in range(n - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        t0 = 0.0
        t1 = t0 + max(float(np.hypot(*(p1 - p0))) ** alpha, 1e-6)
        t2 = t1 + max(float(np.hypot(*(p2 - p1))) ** alpha, 1e-6)
        t3 = t2 + max(float(np.hypot(*(p3 - p2))) ** alpha, 1e-6)
        last = i == n - 2
        ts = np.linspace(t1, t2, samples + 1)[:, None]
        if not last:
            ts = ts[:-1]
        a1 = (t1 - ts) / (t1 - t0) * p0 + (ts - t0) / (t1 - t0) * p1
        a2 = (t2 - ts) / (t2 - t1) * p1 + (ts - t1) / (t2 - t1) * p2
        a3 = (t3 - ts) / (t3 - t2) * p2 + (ts - t2) / (t3 - t2) * p3
        b1 = (t2 - ts) / (t2 - t0) * a1 + (ts - t0) / (t2 - t0) * a2
        b2 = (t3 - ts) / (t3 - t1) * a2 + (ts - t1) / (t3 - t1) * a3
        seg = (t2 - ts) / (t2 - t1) * b1 + (ts - t1) / (t2 - t1) * b2
        out.append(seg)
        owner.extend([i] * len(seg))
    return np.vstack(out), np.asarray(owner, int)


def control_points(chain, moves=None):
    """The chain's control points, with any hand moves applied."""
    ctrl = chain["ctrl"]
    P = np.array([chain["pos"][i] for i in ctrl], float)
    if moves:
        for j, i in enumerate(ctrl):
            d = moves.get(f"{chain['route']}|{chain['keys'][i]}")
            if d is not None:
                P[j] = P[j] + d
    return P


def draw_pieces(chain, moves=None):
    """(polyline, width) per constant-width stretch, ready to stroke."""
    P = control_points(chain, moves)
    ctrl = chain["ctrl"]
    pts, owner = spline(P)
    # a span's width is the width on its first underlying edge; run boundaries
    # are control points by default, so a span does not normally straddle a
    # change at all
    span_w = [chain["w"][ctrl[j]] for j in range(len(ctrl) - 1)]
    w_of = np.array([span_w[o] for o in owner], float)
    pieces, start = [], 0
    for i in range(1, len(w_of) + 1):
        # only cut where the width actually changes — a stripe holds one width
        # for a whole inter-station run, however many control points shape it
        if i == len(w_of) or w_of[i] != w_of[start]:
            if w_of[start] > 0 and i - start >= 2:
                # carry one sample past the cut so the pieces overlap by a hair
                # and no antialiased seam shows at a width change
                pieces.append((pts[start:min(i + 1, len(pts))], float(w_of[start])))
            start = i
    return pieces
