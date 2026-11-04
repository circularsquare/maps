"""T-1168 prototype: classify every other line against a selected one.

Meetings between the selected line L and another line O: shared track edges, and station
complexes both routes reach (a stop or a station passed). Consecutive meetings (in O's order) stay
together when the gap between them is shared track, or when O's track between them strays from L's
by at most STRAY_RATIO of O's run between them. A group of meetings that holds together over more
than one meeting, or over shared track, is a side line running alongside L; a lone meeting at a
station is a crossing.
"""
import bisect, collections, math, sys, time
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from readsave import read

STRAY_RATIO = 0.05
STRAY_FLOOR_M = 100.0
COMPLEX_M = 100.0


def complexes(nodes, links):
    st = [i for i, n in enumerate(nodes) if "platform" in n]
    par = {i: i for i in st}

    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    cell = collections.defaultdict(list)
    g = COMPLEX_M * 1.5
    for i in st:
        cell[(int(nodes[i]["x"] // g), int(nodes[i]["y"] // g))].append(i)
    for i in st:
        cx, cy = int(nodes[i]["x"] // g), int(nodes[i]["y"] // g)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cell[(cx + dx, cy + dy)]:
                    if j > i and math.dist((nodes[i]["x"], nodes[i]["y"]), (nodes[j]["x"], nodes[j]["y"])) <= COMPLEX_M:
                        par[f(i)] = f(j)
    for a, b in links:
        if a in par and b in par:
            par[f(a)] = f(b)
    return {i: f(i) for i in st}


class Route:
    """A line's path as a polyline with distance along it; its route nodes and its stops."""

    def __init__(self, d, li):
        nodes, edges = d["nodes"], d["edges"]
        L = d["lines"][li]
        self.li = li
        self.pts = []      # (x, y)
        self.s = []        # distance at each pt
        self.nodes = []    # (node, s)
        self.edges = []    # (edge, s0, s1)
        s = 0.0
        for e, dr in L["path"]:
            E = edges[e]
            pts = E["pts"] if dr == 0 else E["pts"][::-1]
            a = E["a"] if dr == 0 else E["b"]
            b = E["b"] if dr == 0 else E["a"]
            if not self.pts:
                self.pts.append(pts[0]); self.s.append(0.0); self.nodes.append((a, 0.0))
            s0 = s
            for k in range(1, len(pts)):
                s += math.dist(pts[k - 1], pts[k])
                self.pts.append(pts[k]); self.s.append(s)
            self.nodes.append((b, s))
            self.edges.append((e, s0, s, dr))
        if not self.pts and L["stops"]:
            n = nodes[L["stops"][0]]
            self.pts = [(n["x"], n["y"])]; self.s = [0.0]; self.nodes = [(L["stops"][0], 0.0)]
        self.length = s
        # stops: first route node at or after the previous stop
        self.stop_s = []
        k = 0
        for st in L["stops"]:
            while k < len(self.nodes) and self.nodes[k][0] != st:
                k += 1
            self.stop_s.append(self.nodes[k][1] if k < len(self.nodes) else None)
        self.stop_set = set(L["stops"])

    def at(self, s):
        k = bisect.bisect_left(self.s, s)
        if k <= 0: return self.pts[0]
        if k >= len(self.s): return self.pts[-1]
        f = (s - self.s[k - 1]) / max(1e-9, self.s[k] - self.s[k - 1])
        (x0, y0), (x1, y1) = self.pts[k - 1], self.pts[k]
        return (x0 + f * (x1 - x0), y0 + f * (y1 - y0))

    def nearest(self, p, s0=None, s1=None):
        """(distance, s) of the nearest point of the route (between s0 and s1 if given) to p."""
        best = (float("inf"), 0.0)
        lo = 0 if s0 is None else max(0, bisect.bisect_right(self.s, s0) - 1)
        hi = len(self.pts) - 1 if s1 is None else min(len(self.pts) - 1, bisect.bisect_left(self.s, s1))
        if hi <= lo:
            q = self.at(self.s[lo]); return (math.dist(p, q), self.s[lo])
        for k in range(lo, hi):
            (x0, y0), (x1, y1) = self.pts[k], self.pts[k + 1]
            vx, vy = x1 - x0, y1 - y0
            ll = vx * vx + vy * vy
            t = 0.0 if ll == 0 else max(0.0, min(1.0, ((p[0] - x0) * vx + (p[1] - y0) * vy) / ll))
            q = (x0 + t * vx, y0 + t * vy)
            dd = math.dist(p, q)
            if dd < best[0]:
                best = (dd, self.s[k] + t * (self.s[k + 1] - self.s[k]))
        return best


def stray(O, oa, ob, L, la, lb, step=200.0):
    """How far O's track strays from L's between two meetings: the farthest point of O between oa
    and ob from L between la and lb, sampled at O's vertices and every `step` m."""
    if ob < oa: oa, ob = ob, oa
    if lb < la: la, lb = lb, la
    ss = [oa + k * step for k in range(int((ob - oa) / step) + 1)] + [x for x in O.s if oa < x < ob] + [ob]
    return max(L.nearest(O.at(s), la, lb)[0] for s in ss)


def meetings(d, cx, L, O, edge_lines=None):
    """Meeting points of O with L, sorted along O: dicts with sL, sO, kind, and the complex."""
    out = []
    lc = collections.defaultdict(list)
    for n, s in L.nodes:
        if n in cx:
            lc[cx[n]].append((n, s))
    for n, s in O.nodes:
        if n in cx and cx[n] in lc:
            for ln, ls in lc[cx[n]]:
                out.append({"kind": "station", "sL": ls, "sO": s, "cx": cx[n], "Lstop": ln in L.stop_set, "Ostop": n in O.stop_set, "Onode": n, "Lnode": ln})
    # shared track: sL/sL2 are L's distances where O enters and leaves the edge (sL2 < sL when O
    # runs the other way), sO/sO2 O's own
    le = {e: (a, b, dr) for e, a, b, dr in L.edges}
    for e, a, b, dr in O.edges:
        if e in le:
            la, lb, ldr = le[e]
            same = dr == ldr
            out.append({"kind": "track", "sL": la if same else lb, "sL2": lb if same else la, "sO": a, "sO2": b})
    out.sort(key=lambda m: (m["sO"], m["kind"] == "track"))
    # a station seen twice on L (a loop) or matched to several L nodes: keep the nearest in order
    return out


def classify(d, cx, routes, li, verbose=False):
    L = routes[li]
    # candidates: lines through any complex on L's route, or on any edge of L
    lcx = {cx[n] for n, _ in L.nodes if n in cx}
    cands = set()
    for c in lcx:
        cands |= by_cx.get(c, set())
    for e, _, _, _ in L.edges:
        cands |= by_edge.get(e, set())
    cands.discard(li)
    rel = []
    for oi in sorted(cands):
        O = routes[oi]
        ms = meetings(d, cx, L, O)
        if not ms:
            continue
        groups = [[ms[0]]]
        for m in ms[1:]:
            p = groups[-1][-1]
            g = groups[-1]
            # direction of the group along L: the first way it moved
            dirn = None
            pts = [x for q in g for x in (q["sL"], q.get("sL2", q["sL"]))]
            for k in range(1, len(pts)):
                if abs(pts[k] - pts[0]) > 1.0:
                    dirn = 1 if pts[k] > pts[0] else -1
                    break
            step = m["sL"] - p.get("sL2", p["sL"])
            mono = dirn is None or (step * dirn >= -1.0)
            run = m["sO"] - p.get("sO2", p["sO"])
            together = False
            if mono:
                if p["kind"] == "track" and m["kind"] == "track" and abs(run) < 1.0:
                    together = True
                elif run <= 1.0:
                    together = True
                else:
                    st = stray(O, p.get("sO2", p["sO"]), m["sO"], L, p.get("sL2", p["sL"]), m["sL"])
                    m["gap"] = (st, run, abs(step))
                    together = st <= max(STRAY_FLOOR_M, STRAY_RATIO * run)
            if together:
                g.append(m)
            else:
                groups.append([m])
        for g in groups:
            kinds = {m["kind"] for m in g}
            sLs = [m["sL"] for m in g] + [m["sL2"] for m in g if "sL2" in m]
            sOs = [m["sO"] for m in g] + [m["sO2"] for m in g if "sO2" in m]
            cxs = {m["cx"] for m in g if m["kind"] == "station"}
            if "track" in kinds or len(cxs) >= 2:
                kind = "alongside"
            else:
                kind = "crossing"
            rel.append({"o": oi, "kind": kind, "track": "track" in kinds, "sL0": min(sLs), "sL1": max(sLs), "sO0": min(sOs), "sO1": max(sOs),
                        "meet": g, "stations": len(cxs), "gaps": [m["gap"] for m in g if "gap" in m]})
    return rel


def index(d, cx):
    global by_cx, by_edge
    by_cx = collections.defaultdict(set)
    by_edge = collections.defaultdict(set)
    for i, l in enumerate(d["lines"]):
        for e, _ in l["path"]:
            by_edge[e].add(i)
        for s in l["stops"]:
            if s in cx:
                by_cx[cx[s]].add(i)
    # stations passed count too: add route nodes
    return by_cx, by_edge


def load(path):
    d = read(path)
    cx = complexes(d["nodes"], d["links"])
    routes = [Route(d, i) for i in range(len(d["lines"]))]
    index(d, cx)
    for r in routes:
        for n, _ in r.nodes:
            if n in cx:
                by_cx[cx[n]].add(r.li)
    return d, cx, routes


if __name__ == "__main__":
    t0 = time.perf_counter()
    d, cx, routes = load(sys.argv[1])
    t1 = time.perf_counter()
    which = [int(a) for a in sys.argv[2:]] or range(len(d["lines"]))
    total = 0; worst = (0, None); kinds = collections.Counter(); gaps = []
    for li in which:
        a = time.perf_counter()
        rel = classify(d, cx, routes, li)
        dt = time.perf_counter() - a
        total += dt
        if dt > worst[0]: worst = (dt, li)
        for r in rel:
            kinds[r["kind"]] += 1
            gaps += [g for g in r["gaps"]]
        if len(sys.argv) > 2:
            print("==", li, d["lines"][li]["name"])
            for r in rel:
                O = d["lines"][r["o"]]
                names = [d["nodes"][m["Onode"]]["name"] for m in r["meet"] if m["kind"] == "station"]
                print(f"  {r['o']:3d} {O['name']:<26} {r['kind']:<9} stations {r['stations']:2d} sL {r['sL0']/1000:5.1f}-{r['sL1']/1000:5.1f} km  {names[0][:22] if names else ''} .. {names[-1][:22] if names else ''}",
                      " gaps(stray,run):", [(round(g[0]), round(g[1])) for g in r["gaps"]][:4])
    print(f"load {t1-t0:.2f}s, classify {len(list(which))} lines {total:.2f}s, worst {worst[0]*1000:.0f} ms (line {worst[1]})", dict(kinds))
    if gaps:
        rs = sorted(g[0] / g[1] for g in gaps)
        print("gap stray/run quantiles:", [round(rs[int(q * (len(rs) - 1))], 3) for q in (0, .1, .25, .5, .75, .9, 1)], len(rs))
