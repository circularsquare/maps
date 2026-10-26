"""Routes between stations over the real-network graph (T-1303), for `realnet.py route`: the track
tool's router (app/src/workers/realRoute.worker.ts) in the pipeline, in a city's plane metres
(realnet.to_xy). The same rules: chains between junctions, no turn sharper than 60 degrees where
two chains meet, the other family (metro or rail) at three times its length, every chain within the
snap radius a candidate at either end with its distance counted twice. One region's graph, cropped
to a box; no border joins (every real network lies in one region)."""
import heapq
import math
import os

import numpy as np

from realgraph import read

TURN_MAX = math.radians(60)
OTHER_FAMILY = 3.0
HEADING_M = 30.0
CELL = 250.0
MAX_STATES = 400_000


def wrap_turn(a, b):
    d = abs(a - b) % (2 * math.pi)
    return 2 * math.pi - d if d > math.pi else d


class Graph:
    """A region's graph (`data/packs/realgraph/<cc>.rgr`) in plane metres from `origin`, only the
    chains reaching into `box` (x0, y0, x1, y1)."""

    def __init__(self, path, origin, box, r_earth):
        g = read(path)
        lon0, lat0 = origin
        x = r_earth * np.radians(g["lon"] - lon0) * math.cos(math.radians(lat0))
        y = r_earth * np.radians(g["lat"] - lat0)
        off = g["off"].astype(np.int64)
        x0, y0, x1, y1 = box
        lo, hi = off[:-1], off[1:]
        cx0, cx1 = np.minimum.reduceat(x, lo), np.maximum.reduceat(x, lo)
        cy0, cy1 = np.minimum.reduceat(y, lo), np.maximum.reduceat(y, lo)
        keep = np.flatnonzero((cx1 >= x0) & (cx0 <= x1) & (cy1 >= y0) & (cy0 <= y1) & (hi - lo >= 2))
        self.pts, self.cum, self.ja, self.jb, self.fam, self.ha, self.hb, self.flags = [], [], [], [], [], [], [], []
        self.grid = {}
        adj = {}
        for c in keep:
            p = np.stack([x[lo[c]:hi[c]], y[lo[c]:hi[c]]], 1)
            seg = np.hypot(*np.diff(p, axis=0).T)
            k = len(self.pts)
            self.pts.append(p)
            self.cum.append(np.concatenate([[0.0], np.cumsum(seg)]))
            self.ja.append(int(g["a"][c]))
            self.jb.append(int(g["b"][c]))
            self.fam.append(1 if g["kind"][c] == 2 else 0)
            self.flags.append(int(g["flags"][c]))
            self.ha.append(self._end_heading(p, self.cum[-1], False))
            self.hb.append(self._end_heading(p, self.cum[-1], True))
            adj.setdefault(self.ja[-1], []).append(2 * k)
            adj.setdefault(self.jb[-1], []).append(2 * k + 1)
            # every cell the chain passes, a long straight cut into half cells
            cells = set()
            for i in range(len(p) - 1):
                n = max(1, int(math.ceil(seg[i] / (CELL / 2))))
                for s in range(n + 1):
                    q = p[i] + (p[i + 1] - p[i]) * (s / n)
                    cells.add((int(q[0] // CELL), int(q[1] // CELL)))
            for cell in cells:
                self.grid.setdefault(cell, []).append(k)
        self.adj = adj
        self.len = [float(c[-1]) for c in self.cum]

    @staticmethod
    def _end_heading(p, cum, at_b):
        """Heading leaving a into the chain, or arriving at b along it, over HEADING_M."""
        if not at_b:
            i = min(int(np.searchsorted(cum, HEADING_M)), len(p) - 1)
            d = p[i] - p[0]
        else:
            i = max(int(np.searchsorted(cum, cum[-1] - HEADING_M)) - 1, 0)
            d = p[-1] - p[i]
        return math.atan2(d[1], d[0])

    def candidates(self, at, r):
        """Every chain within r of the point at its nearest point: (dist, chain, along, point)."""
        cx, cy = int(at[0] // CELL), int(at[1] // CELL)
        k = int(math.ceil(r / CELL))
        seen = set()
        for i in range(cx - k, cx + k + 1):
            for j in range(cy - k, cy + k + 1):
                seen.update(self.grid.get((i, j), ()))
        out = []
        for c in seen:
            p = self.pts[c]
            a, b = p[:-1], p[1:]
            d = b - a
            l2 = (d * d).sum(1)
            t = np.clip(np.where(l2 > 0, ((at - a) * d).sum(1) / np.where(l2 > 0, l2, 1), 0), 0, 1)
            q = a + d * t[:, None]
            dist = np.hypot(*(q - at).T)
            i = int(np.argmin(dist))
            if dist[i] <= r:
                along = self.cum[c][i] + t[i] * math.sqrt(l2[i])
                out.append((float(dist[i]), c, float(along), q[i]))
        out.sort(key=lambda o: o[0])
        return out

    def family_near(self, at, r):
        c = self.candidates(at, r)
        return self.fam[c[0][1]] if c else None

    def route(self, a, b, r, fam0):
        """The cheapest route from a to b: (polyline (n, 2), start snap, end snap), or None."""
        starts, ends = self.candidates(a, r), self.candidates(b, r)
        if not starts or not ends:
            return None
        fac = lambda c: 1.0 if self.fam[c] == fam0 else OTHER_FAMILY
        ends_on = {}
        for e in ends:
            ends_on.setdefault(e[1], []).append(e)
        g, prev = {}, {}
        best = [math.inf, None, None, None, None]  # total, state, last leg, start, end
        heap = []
        h = lambda j: max(0.0, math.hypot(*(self.jxy(j) - b)) - r)
        for i, st in enumerate(starts):
            d0, c, along, _ = st
            f = fac(c)
            for e in ends_on.get(c, ()):
                total = abs(e[2] - along) * f + 2 * d0 + 2 * e[0]
                if total < best[0]:
                    best = [total, -1, (c, along, e[2]), st, e]
            for s, gc in ((2 * c, (self.len[c] - along) * f), (2 * c + 1, along * f)):
                v = gc + 2 * d0
                if v < g.get(s, math.inf):
                    g[s] = v
                    prev[s] = -1 - i
                    heapq.heappush(heap, (v + h(self.arrive(s)), s))
        states = 0
        while heap:
            f0, s = heapq.heappop(heap)
            if f0 >= best[0]:
                break
            J = self.arrive(s)
            if f0 > g[s] + h(J) + 1e-6:
                continue
            states += 1
            if states > MAX_STATES:
                break
            c, back = s >> 1, s & 1
            hin = self.ha[c] + math.pi if back else self.hb[c]
            for end in self.adj.get(J, ()):
                c2, from_b = end >> 1, end & 1
                hout = self.hb[c2] + math.pi if from_b else self.ha[c2]
                if wrap_turn(hin, hout) > TURN_MAX:
                    continue
                f2, L = fac(c2), self.len[c2]
                for e in ends_on.get(c2, ()):
                    total = g[s] + ((L - e[2]) if from_b else e[2]) * f2 + 2 * e[0]
                    if total < best[0]:
                        best = [total, s, (c2, L if from_b else 0.0, e[2]), None, e]
                s2 = c2 * 2 + from_b
                v = g[s] + L * f2
                if v < g.get(s2, math.inf):
                    g[s2] = v
                    prev[s2] = s
                    heapq.heappush(heap, (v + h(self.arrive(s2)), s2))
        if best[2] is None:
            return None
        legs = [best[2]]
        start = best[3]
        s = best[1]
        while s >= 0:
            c, back, p = s >> 1, s & 1, prev[s]
            if p < 0:
                start = starts[-1 - p]
                legs.append((c, start[2], 0.0 if back else self.len[c]))
                break
            legs.append((c, self.len[c] if back else 0.0, 0.0 if back else self.len[c]))
            s = p
        legs.reverse()
        return self.line_of(legs), start[3], best[4][3]

    def arrive(self, s):
        return self.ja[s >> 1] if s & 1 else self.jb[s >> 1]

    def jxy(self, j):
        # a junction's position: the end of any chain at it
        e = self.adj[j][0]
        p = self.pts[e >> 1]
        return p[-1] if e & 1 else p[0]

    def line_of(self, legs):
        out = []
        for c, fr, to in legs:
            p, cum = self.pts[c], self.cum[c]
            lo, hi = min(fr, to), max(fr, to)
            inner = p[(cum > lo) & (cum < hi)]
            seg = np.concatenate([[self.point_at(c, lo)], inner, [self.point_at(c, hi)]])
            out.append(seg[::-1] if to < fr else seg)
        line = np.concatenate(out)
        keep = np.concatenate([[True], np.hypot(*np.diff(line, axis=0).T) > 0.01])
        return line[keep]

    def point_at(self, c, d):
        p, cum = self.pts[c], self.cum[c]
        i = min(max(int(np.searchsorted(cum, d)), 1), len(p) - 1)
        t = (d - cum[i - 1]) / (cum[i] - cum[i - 1]) if cum[i] > cum[i - 1] else 0.0
        return p[i - 1] + (p[i] - p[i - 1]) * min(max(t, 0.0), 1.0)


def length(line):
    return float(np.hypot(*np.diff(line, axis=0).T).sum())


def far_from(points, line):
    """The largest distance from any of `points` to the polyline."""
    if len(points) == 0:
        return 0.0
    a, b = line[:-1], line[1:]
    d = b - a
    l2 = np.maximum((d * d).sum(1), 1e-9)
    worst = 0.0
    for q in points:
        t = np.clip(((q - a) * d).sum(1) / l2, 0, 1)
        worst = max(worst, float(np.hypot(*(a + d * t[:, None] - q).T).min()))
    return worst
