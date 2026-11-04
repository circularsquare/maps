"""T-1087: the seeded zone rule and the ring tests, on generic units (read-only; no pipeline imports).

A unit: dict with id, region, simple (Mollweide polygon), free (people), c (people centroid x, y).
A fixed area (metro row or built pack): dict with id, name, geom (Mollweide), pop, xy (its centre).
A seed: dict with name, pop, xy.
"""
import heapq
import math

import shapely
from shapely import STRtree, union_all

KM = 1000.0


def f(dm):
    """The game's commute curve (docs/zones.md 4): d^-0.8 exp(-d / 30 km), d in metres, 2 km floor."""
    d = max(dm / KM, 2.0)
    return d ** -0.8 * math.exp(-d / 30.0)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def adjacency(units, ids):
    """{id: {neighbour id: shared border m}} among `ids`, borders within 1 km."""
    ids = list(ids)
    gs = [units[i]["simple"] for i in ids]
    tree = STRtree(gs)
    adj = {i: {} for i in ids}
    for a, i in enumerate(ids):
        ga = gs[a]
        near = ga.buffer(KM)
        for b in tree.query(near, predicate="intersects"):
            j = ids[b]
            if j <= i:
                continue
            ln = ga.boundary.intersection(gs[b].buffer(KM)).length
            if ln > 0:
                adj[i][j] = ln
                adj[j][i] = ln
    return adj


def unit_seeds(units, ids, seeds, metros):
    """Seeds placed in free units: each urban centre whose point lies in a free unit (or within 2 km
    of one), several in one unit summed at the biggest's point; centres inside a metro are skipped."""
    ids = list(ids)
    gs = [units[i]["simple"] for i in ids]
    tree = STRtree(gs)
    mtree = STRtree([m["geom"] for m in metros]) if metros else None
    out = {}
    for s in sorted(seeds, key=lambda s: -s["pop"]):
        pt = shapely.Point(*s["xy"])
        if mtree is not None and len(mtree.query(pt, predicate="intersects")):
            continue
        hit = tree.query(pt, predicate="dwithin", distance=2 * KM)
        if not len(hit):
            continue
        k = min(hit, key=lambda b: gs[b].distance(pt))
        uid = ids[k]
        if uid in out:
            out[uid]["pop"] += s["pop"]
            out[uid]["names"].append(s["name"])
        else:
            out[uid] = {"unit": uid, "name": s["name"], "names": [s["name"]], "pop": s["pop"], "xy": s["xy"]}
    return out


def peak_seeds(units, ids, adj, useeds, min_pop):
    """Extra seeds where no urban centre is: a unit with at least min_pop people and more than each
    of its neighbours (the area's main town), placed at its people centroid."""
    out = dict(useeds)
    for i in ids:
        if i in out or units[i]["free"] < min_pop:
            continue
        if all(units[i]["free"] > units[j]["free"] for j in adj[i]):
            out[i] = {"unit": i, "name": units[i]["name"], "names": [units[i]["name"]], "pop": units[i]["free"],
                      "xy": units[i]["c"], "peak": True}
    return out


def grow(units, ids, region, adj, useeds, min_seed=0.0, cap=None):
    """Seeded region growing: every seed unit (centre pop >= min_seed) starts a zone; then, best pull
    first, an unclaimed unit beside a zone of its own region joins the zone whose seed pulls it hardest
    (centre people x the commute curve), while the zone stays within `cap`. Units no seed reaches stay
    unclaimed. Returns {seed unit: [ids]}."""
    owner, zones, heap, ppl = {}, {}, [], {}
    seeds = {u: s for u, s in useeds.items() if s["pop"] >= min_seed and u in adj}
    for u, s in seeds.items():
        owner[u] = u
        zones[u] = [u]
        ppl[u] = units[u]["free"]

    def push(z, u):
        for v in adj[u]:
            if v not in owner and region[v] == region[z]:
                s = seeds[z]
                heapq.heappush(heap, (-s["pop"] * f(dist(units[v]["c"], s["xy"])), v, z))
    for z in list(zones):
        push(z, z)
    while heap:
        _, v, z = heapq.heappop(heap)
        if v in owner:
            continue
        if cap and ppl[z] + units[v]["free"] > cap:
            continue
        owner[v] = z
        zones[z].append(v)
        ppl[z] += units[v]["free"]
        push(z, v)
    return zones, owner


class Zones:
    """Zones as unit lists with cached shapes, people and centroids."""

    def __init__(self, units, groups):
        self.units = units
        self.z = {k: list(v) for k, v in groups.items()}
        self._g = {}

    def ppl(self, k):
        return sum(self.units[i]["free"] for i in self.z[k])

    def cen(self, k):
        p = self.ppl(k)
        if p <= 0:
            c = self.geo(k).representative_point()
            return (c.x, c.y)
        return (sum(self.units[i]["c"][0] * self.units[i]["free"] for i in self.z[k]) / p,
                sum(self.units[i]["c"][1] * self.units[i]["free"] for i in self.z[k]) / p)

    def geo(self, k):
        key = (k, len(self.z[k]))
        if key not in self._g:
            self._g[key] = union_all([self.units[i]["simple"] for i in self.z[k]])
        return self._g[key]


def wraps_any(m, others):
    """Whether shape m wraps any of `others` [(geom, land border length)]: half of its area in m's hull,
    m holding half of its land border."""
    hull, near = m.convex_hull, m.buffer(KM)
    for gq, land in others:
        if not gq.intersects(near) or gq.intersection(hull).area < 0.5 * gq.area:
            continue
        if land and gq.boundary.intersection(near).length >= 0.5 * land:
            return True
    return False


def floor_merge(Z, region, adj, floor, cap, fixed=(), choose="pull", guard=True, water_km=10.0, strict=False, only=None):
    """Zones under `floor` join a neighbour, smallest first: of their region, else another region's
    while under 0.8 of the floor. choose: 'pull' the neighbour whose people pull its people hardest
    (people x curve between people centroids), 'small' the smallest (today's rule). Neighbours keeping
    the two within `cap` first; with `guard`, the first that wraps no zone or fixed area. Zones with
    no land neighbour join the nearest within water_km. fixed: [(geom, its land border, {unit: that
    unit's border along it})]. Returns log lines."""
    owner = {i: k for k, ids in Z.z.items() for i in ids}
    lines, done = [], set()

    def nbrs(k):
        out = {}
        for i in Z.z[k]:
            for j, ln in adj[i].items():
                o = owner.get(j)
                if o is not None and o != k:
                    out[o] = out.get(o, 0) + ln
        return out

    def land_of(q):
        return sum(nbrs(q).values())

    def wrapping(k, o):
        m = union_all([Z.geo(k), Z.geo(o)])
        hull, near = m.convex_hull, m.buffer(KM)
        b0, b1, b2, b3 = near.bounds
        for q in Z.z:
            if q in (k, o):
                continue
            gq = Z.geo(q)
            c0, c1, c2, c3 = gq.bounds
            if c0 > b2 or c2 < b0 or c1 > b3 or c3 < b1:
                continue
            if not gq.intersects(near) or gq.intersection(hull).area < 0.5 * gq.area:
                continue
            land = land_of(q) + sum(fl.get(i, 0) for _, _, fl in fixed for i in Z.z[q])
            if land and gq.boundary.intersection(near).length >= 0.5 * land:
                return True
        return wraps_any(m, [(g, ln) for g, ln, _ in fixed])

    while True:
        small = sorted((Z.ppl(k), k) for k in Z.z if k not in done and (only is None or k in only) and Z.ppl(k) < floor)
        if not small:
            break
        n, k = small[0]
        nb = nbrs(k)
        cand = [o for o in nb if region[o] == region[k]]
        if not cand and n < 0.8 * floor:
            cand = list(nb)
        into, how = None, ""
        if cand and strict and n >= 0.8 * floor and not any(n + Z.ppl(o) <= cap for o in cand):
            # near the floor with no room beside it: stays as it is rather than break the cap
            done.add(k)
            lines.append(f"{k} ({n:,.0f}) stays: no neighbour within the cap")
            continue
        if cand:
            fit = [o for o in cand if n + Z.ppl(o) <= cap]
            over = not fit
            fit = fit or cand
            if over:  # nothing fits: the smallest, as today, so no zone grows far past the cap
                fit.sort(key=lambda o: (Z.ppl(o), -nb[o]))
            elif choose == "pull":
                ck = Z.cen(k)
                fit.sort(key=lambda o: -Z.ppl(o) * f(dist(ck, Z.cen(o))))
            elif choose == "near":  # the neighbour whose people centre is nearest (Voronoi)
                ck = Z.cen(k)
                fit.sort(key=lambda o: dist(ck, Z.cen(o)))
            else:
                fit.sort(key=lambda o: (Z.ppl(o), -nb[o]))
            into = fit[0]
            how = choose
            if guard and (len(fit) > 1 or strict):
                into = next((o for o in fit if not wrapping(k, o)), None)
                if into is None and strict:
                    # any other neighbour (over the cap, or another region's) that wraps nothing
                    rest = [o for o in sorted(nb, key=lambda o: Z.ppl(o)) if o not in fit and n + Z.ppl(o) <= cap]
                    into = next((o for o in rest if not wrapping(k, o)), None)
                    how = "strict: " + ("another neighbour" if into else "stays under the floor")
                    if into is None and n < 0.8 * floor:
                        # too small to stand alone: join anyway; the ring cut takes it apart after
                        into, how = fit[0], "wraps, left to the cut"
                elif into is None:
                    into = fit[0]
        else:
            g = Z.geo(k)
            pools = [[o for o in Z.z if o != k and region[o] == region[k]]]
            if n < 0.8 * floor:
                pools.append([o for o in Z.z if o != k and region[o] != region[k]])
            for pool in pools:
                near = [(g.distance(Z.geo(o)), o) for o in pool]
                near = [x for x in near if x[0] <= water_km * KM]
                if near:
                    into = min(near)[1]
                    how = "water"
                    break
        if into is None:
            done.add(k)
            lines.append(f"{k} ({n:,.0f}) stays")
            continue
        lines.append(f"{k} ({n:,.0f}) joins {into} ({how})")
        for i in Z.z[k]:
            owner[i] = into
        Z.z[into] += Z.z.pop(k)
    return lines


def connected(ids, adj):
    ids = set(ids)
    if not ids:
        return True
    start = next(iter(ids))
    seen, front = {start}, [start]
    while front:
        nxt = []
        for i in front:
            for j in adj[i]:
                if j in ids and j not in seen:
                    seen.add(j)
                    nxt.append(j)
        front = nxt
    return len(seen) == len(ids)


def cut_ring(Z, k, centre, adj):
    """Cut ring zone k in two across the shape it wraps: its units by angle round `centre`, the arc
    opened at the widest gap, split where the two halves' people are most even, both halves connected.
    Returns the two unit lists, or None."""
    U = Z.units
    ids = Z.z[k]
    ang = sorted((math.atan2(U[i]["c"][1] - centre[1], U[i]["c"][0] - centre[0]), i) for i in ids)
    gaps = [(ang[(a + 1) % len(ang)][0] - ang[a][0]) % (2 * math.pi) for a in range(len(ang))]
    start = (max(range(len(gaps)), key=lambda a: gaps[a]) + 1) % len(ang)
    order = [ang[(start + a) % len(ang)][1] for a in range(len(ang))]
    tot = sum(U[i]["free"] for i in order)
    best = None
    run = 0.0
    for s in range(1, len(order)):
        run += U[order[s - 1]]["free"]
        a, b = order[:s], order[s:]
        if not (connected(a, adj) and connected(b, adj)):
            continue
        score = abs(run - tot / 2)
        if best is None or score < best[0]:
            best = (score, a, b)
    return (best[1], best[2]) if best else None


def cut_rounds(Z, fx, adj, fb, region, floor, cap, choose, rounds=3):
    """Cut each ring zone across what it wraps, let the halves regroup (strict guard), repeat.
    fx: {id: {geom, xy}} fixed areas; fb: their floor_merge borders."""
    lines = []
    for rnd in range(rounds):
        shapes = {k: Z.geo(k) for k in Z.z}
        kinds = {k: "zone" for k in Z.z}
        for k, m in fx.items():
            shapes[k] = m["geom"]
            kinds[k] = "metro"
        rings = [p for p in ring_pairs(shapes, kinds) if p[2] >= 0.5 and p[3] >= 0.5]
        if not rings:
            break
        done = set()
        for a, b, ih, bd in sorted(rings, key=lambda p: -p[2] - p[3]):
            if a in done or a not in Z.z:
                continue
            centre = fx[b]["xy"] if kinds[b] == "metro" else Z.cen(b)
            got = cut_ring(Z, a, centre, adj) if len(Z.z[a]) > 1 else None
            if not got and kinds[b] == "zone" and b in Z.z and b not in done and Z.ppl(a) + Z.ppl(b) <= cap:
                # a ring that cannot be cut (one unit round a city, Wrocławski) takes what it wraps
                Z.z[a] += Z.z.pop(b)
                done |= {a, b}
                lines.append(f"round {rnd}: {a} wraps {b}, merged")
                continue
            if not got:
                lines.append(f"round {rnd}: {a} wraps {b}, no cut")
                continue
            h1, h2 = got
            Z.z.pop(a)
            Z.z[h1[0]] = h1
            Z.z[h2[0]] = h2
            done |= {h1[0], h2[0]}
            lines.append(f"round {rnd}: {a} wraps {b}, cut {len(h1)}+{len(h2)} units")
        lines += floor_merge(Z, region, adj, floor, cap, fb, choose, True, strict=True, only=done)
    return lines


def ring_pairs(shapes, kinds, land_pool=None):
    """Every pair (A, B) where zone A wraps B (B a zone or a fixed area): B's area share inside A's hull
    and A's share of B's border with all shapes. shapes {id: geom}; kinds {id: 'zone' | 'metro'}.
    Returns [(A, B, in_hull, border)] for pairs touching within 1 km."""
    ids = list(shapes)
    gs = [shapes[i] for i in ids]
    tree = STRtree(gs)
    nb = {i: {} for i in ids}
    for a, i in enumerate(ids):
        near = gs[a].buffer(KM)
        for b in tree.query(near, predicate="intersects"):
            j = ids[b]
            if j == i:
                continue
            nb[i][j] = gs[b].boundary.intersection(near).length  # j's border along i
    out = []
    hulls = {}
    for a in ids:
        if kinds[a] != "zone":
            continue
        ha = hulls.setdefault(a, shapes[a].convex_hull)
        for b, ln in nb[a].items():
            land = sum(nb[x].get(b, 0) for x in nb[b])  # b's border along each neighbour
            if not land:
                continue
            gb = shapes[b]
            ih = gb.intersection(ha).area / gb.area if gb.area else 0
            out.append((a, b, ih, ln / land))
    return out
