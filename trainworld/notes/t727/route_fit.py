"""T-727 prototype: route between two clicks on the real network, then fit it to trainworld's
track model (straights between PIs, each corner cut by a circular arc, auto radius = the largest
that fits half of each adjacent leg, the whole leg next to an end, capped at 1,796 m).

    python route_fit.py <region> <lat1> <lon1> <lat2> <lon2> [--kinds rail,narrow_gauge]
        [--weight short|main] [--tol 25] [--maxturn 60] [--name label]
"""
import os
import sys
import json
import math
import time
import heapq
import argparse
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parents[3] / "artifacts" / "trainworld" / "t727"  # outputs, not notes/
R = 6371008.8
R_CAP = 1796.0
MIN_R = 100.0
KIND_ID = {"rail": 0, "light_rail": 1, "subway": 2, "tram": 3, "monorail": 4, "funicular": 5,
           "narrow_gauge": 6, "heritage": 7}
KIND_NAME = {v: k for k, v in KIND_ID.items()}


def merc(lon, lat):
    return R * np.radians(lon), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def ground_seglen(lon, lat):
    la = np.radians(lat)
    dx = np.radians(np.diff(lon)) * np.cos((la[1:] + la[:-1]) / 2) * R
    dy = np.diff(la) * R
    return np.hypot(dx, dy)


class Graph:
    def __init__(self, region, kinds):
        g = np.load(HERE / f"graph_{region}.npz")
        self.a, self.b, self.off = g["a"], g["b"], g["off"]
        self.kind, self.rank, self.flags = g["kind"], g["rank"], g["flags"]
        self.lon, self.lat = g["lon"], g["lat"]
        self.nj = int(max(self.a.max(), self.b.max())) + 1
        C = self.a.size
        self.C = C
        allowed = np.isin(self.kind, [KIND_ID[k] for k in kinds])
        if os.environ.get("NOSERVICE"):
            allowed &= ~((self.rank == 3) & ((self.flags & 1) == 0))
        self.allowed = allowed
        seg = ground_seglen(self.lon, self.lat)
        # cumulative length per flat vertex within its chain
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        self.cumflat = cum
        self.length = cum[self.off[1:] - 1] - cum[self.off[:-1]]
        # headings at chain ends, from a point ~30 m in
        x, y = merc(self.lon, self.lat)
        self.x, self.y = x, y
        hs = np.zeros(C)
        he = np.zeros(C)
        for c in range(C):
            s, e = self.off[c], self.off[c + 1]
            cc = cum[s:e] - cum[s]
            L = cc[-1]
            i1 = min(int(np.searchsorted(cc, min(30.0, L))), e - s - 1)
            i0 = max(int(np.searchsorted(cc, max(L - 30.0, 0.0))) - 1, 0)
            i1 = max(i1, 1)
            i0 = min(i0, e - s - 2)
            hs[c] = math.atan2(y[s + i1] - y[s], x[s + i1] - x[s])
            he[c] = math.atan2(y[e - 1] - y[s + i0], x[e - 1] - x[s + i0])
        self.hs, self.he = hs, he
        # leaving lists per junction: (chain, dir)
        out = [[] for _ in range(self.nj)]
        for c in np.flatnonzero(allowed):
            out[self.a[c]].append((c, 0))
            out[self.b[c]].append((c, 1))
        self.out = out
        # snapping tree over allowed vertices
        vid = np.concatenate([np.arange(self.off[c], self.off[c + 1]) for c in np.flatnonzero(allowed)])
        self.vid = vid
        self.vchain = np.repeat(np.arange(C), np.diff(self.off))
        self.tree = cKDTree(np.column_stack([x[vid], y[vid]]))

    def heading(self, c, d, at_start):
        # direction of travel leaving (at_start) or arriving
        if d == 0:
            return self.hs[c] if at_start else self.he[c]
        return (self.he[c] + math.pi) if at_start else (self.hs[c] + math.pi)

    def snap(self, lat, lon):
        x, y = merc(np.array([lon]), np.array([lat]))
        dist, i = self.tree.query([x[0], y[0]])
        v = self.vid[i]
        k = 1 / math.cos(math.radians(lat))
        return v, dist / k


def snap_many(G, lat, lon, radius):
    """Nearest vertex per chain within radius (ground m): {chain: (vertex, distance)}."""
    x, y = merc(np.array([lon]), np.array([lat]))
    k = 1 / math.cos(math.radians(lat))
    ii = G.tree.query_ball_point([x[0], y[0]], radius * k)
    out = {}
    for i in ii:
        v = G.vid[i]
        d = math.hypot(G.x[v] - x[0], G.y[v] - y[0]) / k
        c = G.vchain[v]
        if c not in out or d < out[c][1]:
            out[c] = (v, d)
    if not out:
        v, d = G.snap(lat, lon)
        out[G.vchain[v]] = (v, d)
    return out


def route_many(G, src, dst, mult, maxturn, snap_w=2.0):
    """Turn-limited Dijkstra over directed chains from any source candidate to any target one.
    A candidate's snap distance costs snap_w times its length."""
    INF = float("inf")
    pos = lambda v: G.cumflat[v] - G.cumflat[G.off[G.vchain[v]]]
    dist, pred, pq = {}, {}, []
    best, best_end = INF, None
    for c, (v, sd) in src.items():
        p = pos(v)
        for d, cost in ((0, G.length[c] - p), (1, p)):
            s = (c, d)
            t = cost * mult[c] + snap_w * sd
            if t < dist.get(s, INF):
                dist[s] = t
                pred[s] = ("src", v)
                heapq.heappush(pq, (t, c, d))
        if c in dst:
            v1, sd1 = dst[c]
            t = abs(pos(v1) - p) * mult[c] + snap_w * (sd + sd1)
            if t < best:
                best, best_end = t, ("same", v, v1)
    maxt = math.radians(maxturn)
    pops = 0
    while pq:
        dcur, c, d = heapq.heappop(pq)
        if dcur >= best:
            break
        if dcur > dist.get((c, d), INF):
            continue
        pops += 1
        j = G.b[c] if d == 0 else G.a[c]
        hin = G.heading(c, d, False)
        for c2, d2 in G.out[j]:
            if c2 == c:
                continue
            if turn(hin, G.heading(c2, d2, True)) > maxt:
                continue
            if c2 in dst:
                v1, sd1 = dst[c2]
                p1 = pos(v1)
                part = p1 if d2 == 0 else G.length[c2] - p1
                t = dcur + part * mult[c2] + snap_w * sd1
                if t < best:
                    best, best_end = t, ("via", (c, d), c2, d2, v1)
            nd = dcur + G.length[c2] * mult[c2]
            if nd < dist.get((c2, d2), INF):
                dist[(c2, d2)] = nd
                pred[(c2, d2)] = (c, d)
                heapq.heappush(pq, (nd, c2, d2))
    if best_end is None:
        return None, pops
    if best_end[0] == "same":
        _, v0, v1 = best_end
        lo, hi = sorted((v0, v1))
        idx = np.arange(lo, hi + 1)
        return (idx if v0 < v1 else idx[::-1]), pops
    _, last, c1, d_end, v1 = best_end
    chain = []
    s = last
    while not (isinstance(pred[s], tuple) and pred[s][0] == "src"):
        chain.append(s)
        s = pred[s]
    chain.append(s)
    v0 = pred[s][1]
    chain.reverse()
    pieces = []
    c, d = chain[0]
    s_, e_ = G.off[c], G.off[c + 1]
    pieces.append(np.arange(v0, e_) if d == 0 else np.arange(v0, s_ - 1, -1))
    for c, d in chain[1:]:
        s_, e_ = G.off[c], G.off[c + 1]
        pieces.append(np.arange(s_, e_) if d == 0 else np.arange(e_ - 1, s_ - 1, -1))
    s_, e_ = G.off[c1], G.off[c1 + 1]
    pieces.append(np.arange(s_, v1 + 1) if d_end == 0 else np.arange(e_ - 1, v1 - 1, -1))
    out = [pieces[0]] + [p[1:] for p in pieces[1:]]
    return np.concatenate(out), pops


def turn(h1, h2):
    d = (h2 - h1 + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def route(G, v0, v1, mult, maxturn):
    c0, c1 = G.vchain[v0], G.vchain[v1]
    pos0 = G.cumflat[v0] - G.cumflat[G.off[c0]]
    pos1 = G.cumflat[v1] - G.cumflat[G.off[c1]]
    INF = float("inf")
    dist = {}
    pred = {}
    pq = []
    # states: (c, d) arriving at the chain's far junction
    for d, cost in ((0, G.length[c0] - pos0), (1, pos0)):
        s = (c0, d)
        dist[s] = cost * mult[c0]
        pred[s] = None
        heapq.heappush(pq, (dist[s], c0, d))
    best, best_state = INF, None
    if c0 == c1:
        best = abs(pos1 - pos0) * mult[c0]
        best_state = ("same",)
    maxt = math.radians(maxturn)
    pops = 0
    while pq:
        dcur, c, d = heapq.heappop(pq)
        if dcur > dist.get((c, d), INF) or dcur >= best:
            if dcur >= best:
                break
            continue
        pops += 1
        j = G.b[c] if d == 0 else G.a[c]
        hin = G.heading(c, d, False)
        for c2, d2 in G.out[j]:
            if c2 == c:
                continue
            if turn(hin, G.heading(c2, d2, True)) > maxt:
                continue
            if c2 == c1:
                part = pos1 if d2 == 0 else G.length[c1] - pos1
                t = dcur + part * mult[c1]
                if t < best:
                    best, best_state = t, ((c, d), d2)
            nd = dcur + G.length[c2] * mult[c2]
            if nd < dist.get((c2, d2), INF):
                dist[(c2, d2)] = nd
                pred[(c2, d2)] = (c, d)
                heapq.heappush(pq, (nd, c2, d2))
    if best_state is None:
        return None, pops
    if best_state[0] == "same":
        lo, hi = sorted((v0, v1))
        idx = np.arange(lo, hi + 1)
        return (idx if v0 < v1 else idx[::-1]), pops
    (last, d_end) = best_state
    chain = []
    s = last
    while s is not None:
        chain.append(s)
        s = pred[s]
    chain.reverse()
    pieces = []
    # first chain: from v0 to its far end
    c, d = chain[0]
    s, e = G.off[c], G.off[c + 1]
    pieces.append(np.arange(v0, e) if d == 0 else np.arange(v0, s - 1, -1))
    for c, d in chain[1:]:
        s, e = G.off[c], G.off[c + 1]
        pieces.append(np.arange(s, e) if d == 0 else np.arange(e - 1, s - 1, -1))
    s, e = G.off[c1], G.off[c1 + 1]
    pieces.append(np.arange(s, v1 + 1) if d_end == 0 else np.arange(e - 1, v1 - 1, -1))
    out = [pieces[0]]
    for p in pieces[1:]:
        out.append(p[1:])
    idx = np.concatenate(out)
    return idx, pops


# ---------------------------------------------------------------- fitting

def dp(px, py, tol_plane):
    """Douglas-Peucker; tol_plane is per vertex (tolerance times k there)."""
    n = px.size
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        x0, y0, x1, y1 = px[i], py[i], px[j], py[j]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        sx, sy = px[i + 1:j] - x0, py[i + 1:j] - y0
        if L == 0:
            dd = np.hypot(sx, sy)
        else:
            t = np.clip((sx * dx + sy * dy) / (L * L), 0, 1)
            dd = np.hypot(sx - t * dx, sy - t * dy)
        m = int(np.argmax(dd - tol_plane[i + 1:j]))
        if dd[m] > tol_plane[i + 1 + m]:
            k = i + 1 + m
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return np.flatnonzero(keep)


def radii(vx, vy, kv):
    """Auto radius per vertex (inf at the ends and at straight-through vertices), deflection."""
    n = vx.size
    lx, ly = np.diff(vx), np.diff(vy)
    L = np.hypot(lx, ly)
    hdg = np.arctan2(ly, lx)
    defl = np.zeros(n)
    defl[1:-1] = (hdg[1:] - hdg[:-1] + np.pi) % (2 * np.pi) - np.pi
    avail = L / 2
    avail[0] = L[0]
    avail[-1] = L[-1]
    if n == 2:
        avail[:] = L
    r = np.full(n, np.inf)
    i = np.arange(1, n - 1)
    a = np.minimum(avail[i - 1], avail[i])
    with np.errstate(divide="ignore"):
        rr = a / np.tan(np.abs(defl[i]) / 2)
    r[i] = np.minimum(rr, R_CAP * kv[i])
    r[i][np.abs(defl[i]) < 1e-9] = np.inf
    return r, defl, L, hdg


def fitted_samples(vx, vy, kv, step):
    """Sample the fitted alignment (plane units) every ~step ground metres."""
    r, defl, L, hdg = radii(vx, vy, kv)
    n = vx.size
    pts = []
    cur = (vx[0], vy[0])
    for i in range(1, n):
        # straight from cur to the start of the curve at i (or to the end)
        if i < n - 1 and np.isfinite(r[i]):
            T = r[i] * math.tan(abs(defl[i]) / 2)
            ux0, uy0 = math.cos(hdg[i - 1]), math.sin(hdg[i - 1])
            p1 = (vx[i] - T * ux0, vy[i] - T * uy0)
        else:
            p1 = (vx[i], vy[i])
        seg = math.hypot(p1[0] - cur[0], p1[1] - cur[1])
        m = max(int(seg / (step * kv[i])), 1)
        t = np.arange(m) / m
        pts.append(np.column_stack([cur[0] + t * (p1[0] - cur[0]), cur[1] + t * (p1[1] - cur[1])]))
        if i < n - 1 and np.isfinite(r[i]):
            s = 1 if defl[i] > 0 else -1
            nx, ny = -s * uy0, s * ux0  # towards the centre
            cx, cy = p1[0] + r[i] * nx, p1[1] + r[i] * ny
            a0 = math.atan2(p1[1] - cy, p1[0] - cx)
            arc = abs(defl[i]) * r[i]
            m = max(int(arc / (step * kv[i])), 1)
            ang = a0 + s * abs(defl[i]) * np.arange(m) / m
            pts.append(np.column_stack([cx + r[i] * np.cos(ang), cy + r[i] * np.sin(ang)]))
            ux1, uy1 = math.cos(hdg[i]), math.sin(hdg[i])
            cur = (vx[i] + T * ux1, vy[i] + T * uy1)
        else:
            cur = p1
    pts.append(np.array([[vx[-1], vy[-1]]]))
    return np.concatenate(pts), r, defl


def densify(px, py, kp, step):
    out = [np.array([[px[0], py[0]]])]
    owner = [np.array([0])]
    for i in range(px.size - 1):
        L = math.hypot(px[i + 1] - px[i], py[i + 1] - py[i])
        m = max(int(L / (step * kp[i])), 1)
        t = (np.arange(m) + 1) / m
        out.append(np.column_stack([px[i] + t * (px[i + 1] - px[i]), py[i] + t * (py[i + 1] - py[i])]))
        owner.append(np.full(m, i))
    return np.concatenate(out), np.concatenate(owner)


def evaluate(px, py, kp, keep):
    vx, vy, kv = px[keep], py[keep], kp[keep]
    fit, r, defl = fitted_samples(vx, vy, kv, 2.0)
    real, owner = densify(px, py, kp, 10.0)
    kr = kp[owner]
    d1, _ = cKDTree(fit).query(real)
    d1 = d1 / kr
    # fitted -> real
    real2, _ = densify(px, py, kp, 2.0)
    fit10 = fit[::5]
    d2, nn = cKDTree(real2).query(fit10)
    k2 = 1 / np.cos(np.arctan(np.sinh(fit10[:, 1] / R)))
    d2 = d2 / k2
    return d1, owner, d2, r, defl, kv


def fit_path(px, py, kp, tol, dp_frac=0.6, max_iter=40):
    keep = dp(px, py, tol * dp_frac * kp)
    it = 0
    while True:
        d1, owner, d2, r, defl, kv = evaluate(px, py, kp, keep)
        bad = d1 > tol
        if not bad.any() or it >= max_iter:
            break
        add = set()
        span = np.searchsorted(keep, owner[bad], side="right") - 1
        for sp in np.unique(span):
            sel = bad & (np.searchsorted(keep, owner, side="right") - 1 == sp)
            w = owner[sel][np.argmax(d1[sel])]
            lo, hi = keep[sp], keep[sp + 1] if sp + 1 < keep.size else keep[-1]
            cand = [w, w + 1]
            for cnd in cand:
                if lo < cnd < hi:
                    add.add(int(cnd))
                    break
        if not add:
            break
        keep = np.unique(np.concatenate([keep, np.array(sorted(add))]))
        it += 1
    return keep, d1, d2, r, defl, kv, it


def local_max(px, py, kp, keep, j):
    """Max offset of the real track near kept vertex j (between keep[j-2] and keep[j+2])."""
    lo, hi = max(j - 4, 0), min(j + 4, keep.size - 1)
    sub = keep[lo:hi + 1]
    s0, s1 = sub[0], sub[-1]
    qx, qy, qk = px[s0:s1 + 1], py[s0:s1 + 1], kp[s0:s1 + 1]
    fit, _, _ = fitted_samples(qx[sub - s0], qy[sub - s0], qk[sub - s0], 2.0)
    a = keep[max(j - 2, 0)] - s0
    b = keep[min(j + 2, keep.size - 1)] - s0
    real, owner = densify(qx[a:b + 1], qy[a:b + 1], qk[a:b + 1], 10.0)
    d, _ = cKDTree(fit).query(real)
    return float((d / qk[a + owner]).max())


FORCED = []


def relax(px, py, kp, keep, tol, rmin=100.5, slack=2.0):
    """Drop PIs whose auto radius is under rmin (jogs where the route changes track), keeping
    the local offset within slack * tol. Returns keep, fixed, unfixed."""
    fixed = unfixed = 0
    tried = set()
    while True:
        vx, vy, kv = px[keep], py[keep], kp[keep]
        r, defl, L, hdg = radii(vx, vy, kv)
        rg = r / kv
        cand = [j for j in np.argsort(rg) if 0 < j < keep.size - 1 and rg[j] < rmin and keep[j] not in tried]
        if not cand:
            break
        j = int(cand[0])
        tried.add(keep[j])
        ok = False
        options = []
        for drop in ([j], [j - 1, j], [j, j + 1], [j - 1, j, j + 1], [j - 2, j - 1, j, j + 1, j + 2]):
            if min(drop) < 1 or max(drop) > keep.size - 2:
                continue
            k2 = np.delete(keep, drop)
            jj = min(drop[0], k2.size - 2)
            r2, _, _, _ = radii(px[k2], py[k2], kp[k2])
            win = slice(max(jj - 3, 1), min(jj + 4, k2.size - 1))
            if np.min(r2[win] / kp[k2][win]) < rmin:
                continue
            m = local_max(px, py, kp, k2, jj)
            options.append((m, k2))
            if m <= slack * tol:
                keep = k2
                ok = True
                break
        if not ok and options:
            # forced: the least offset among the drops that clear the minimum radius
            m, keep = min(options, key=lambda o: o[0])
            FORCED.append(round(m, 1))
        fixed += ok
        unfixed += not ok
    return keep, fixed, unfixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region")
    ap.add_argument("lat1", type=float)
    ap.add_argument("lon1", type=float)
    ap.add_argument("lat2", type=float)
    ap.add_argument("lon2", type=float)
    ap.add_argument("--kinds", default="rail,narrow_gauge")
    ap.add_argument("--weight", default="short")
    ap.add_argument("--tol", default="25")
    ap.add_argument("--maxturn", type=float, default=60)
    ap.add_argument("--name", default="path")
    ap.add_argument("--snap", type=float, default=400)
    args = ap.parse_args()
    t0 = time.time()
    G = Graph(args.region, args.kinds.split(","))
    t1 = time.time()
    print(f"graph loaded in {t1-t0:.1f}s: {G.C} chains, {int(G.allowed.sum())} allowed, "
          f"{G.length[G.allowed].sum()/1000:.0f} km")
    v0, s0 = G.snap(args.lat1, args.lon1)
    v1, s1 = G.snap(args.lat2, args.lon2)
    print(f"snap: {s0:.0f} m and {s1:.0f} m")
    mult = np.ones(G.C)
    if args.weight == "pure":
        pass
    elif args.weight == "short":
        mult = np.where(G.rank == 2, 1.5, np.where(G.rank == 3, 3.0, 1.0))
    elif args.weight == "main":
        mult = np.where(G.rank == 0, 1.0, 1.3)
        mult = np.where(G.flags & 8, 0.85, mult)
        mult = np.where(np.isin(G.kind, [1, 2, 3, 4]), 3.0, mult)
        mult = np.where(G.rank == 2, 1.5, np.where(G.rank == 3, 3.0, mult))
    if args.snap > 0:
        src = snap_many(G, args.lat1, args.lon1, args.snap)
        dst = snap_many(G, args.lat2, args.lon2, args.snap)
        print(f"snap candidates: {len(src)} and {len(dst)} chains within {args.snap:.0f} m")
        idx, pops = route_many(G, src, dst, mult, args.maxturn)
    else:
        idx, pops = route(G, v0, v1, mult, args.maxturn)
    t2 = time.time()
    if idx is None:
        print(f"NO PATH ({pops} states searched, {t2-t1:.1f}s)")
        return
    lon, lat = G.lon[idx], G.lat[idx]
    seg = ground_seglen(lon, lat)
    ch = G.vchain[idx]
    segch = np.where(ch[1:] == ch[:-1], ch[1:], ch[1:])
    km = seg.sum() / 1000
    br = seg[(G.flags[segch] & 2) > 0].sum() / 1000
    tu = seg[(G.flags[segch] & 4) > 0].sum() / 1000
    hs = seg[(G.flags[segch] & 8) > 0].sum() / 1000
    r1 = seg[G.rank[segch] == 1].sum() / 1000
    r23 = [round(seg[G.rank[segch] == q].sum() / 1000, 1) for q in (2, 3)]
    print(f"freight/industrial km {r23[0]}, yard/siding/service km {r23[1]}")
    kinds = {KIND_NAME[k]: round(seg[G.kind[segch] == k].sum() / 1000, 1) for k in np.unique(G.kind[segch])}
    nchains = int(np.sum(ch[1:] != ch[:-1])) + 1
    print(f"route: {km:.1f} km, {idx.size} OSM vertices, {nchains} chains, {pops} states popped, "
          f"{t2-t1:.2f}s; bridge {br:.1f} km, tunnel {tu:.1f} km, highspeed {hs:.1f} km, "
          f"rank1 {r1:.1f} km, kinds {kinds}")
    px, py = merc(lon, lat)
    kp = 1 / np.cos(np.radians(lat))
    rows = []
    for tol in [float(t) for t in args.tol.split(",")]:
        t3 = time.time()
        keep, d1, d2, r, defl, kv, it = fit_path(px, py, kp, tol)
        before = int(np.sum((r / kv)[np.isfinite(r)] < MIN_R))
        keep, fixed, unfixed = relax(px, py, kp, keep, tol)
        d1, owner, d2, r, defl, kv = evaluate(px, py, kp, keep)
        print(f"tol {tol}: {before} curves under 100 m before relaxing, {fixed} fixed, {unfixed} not "
              f"(forced, local max offset m: {FORCED})")
        FORCED.clear()
        npi = keep.size - 2
        rg = r[np.isfinite(r)] / kv[np.isfinite(r)]
        tight = int(np.sum(rg < MIN_R))
        spd = np.sqrt(1.1 * rg) * 3.6
        under = lambda v: int(np.sum(spd < v))
        row = {"tol": tol, "pis": int(npi), "per_km": round(npi / km, 2), "iters": it,
               "max_off": round(float(d1.max()), 1), "p99_off": round(float(np.percentile(d1, 99)), 1),
               "mean_off": round(float(d1.mean()), 1),
               "max_back": round(float(d2.max()), 1),
               "r_min": round(float(rg.min()), 0) if rg.size else None,
               "under_100m": tight, "curves_under_80kmh": under(80), "under_120kmh": under(120),
               "secs": round(time.time() - t3, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if tol == float(args.tol.split(",")[0]):
            np.savez(HERE / f"fit_{args.name}.npz", lon=lon, lat=lat, keep=keep, d1=d1, r=r, kv=kv)
    with open(HERE / f"result_{args.name}.json", "w") as f:
        json.dump({"km": km, "vertices": int(idx.size), "chains": nchains, "bridge_km": br,
                   "tunnel_km": tu, "hs_km": hs, "kinds": kinds, "snap": [s0, s1], "fits": rows}, f)


if __name__ == "__main__":
    main()
