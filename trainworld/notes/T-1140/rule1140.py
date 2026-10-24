"""T-1140: world_plan.zone_groups with the partner-or-compact join test (a copy for dry runs; read-only).

New keyword arguments (all off reproduces world_plan.zone_groups):
- flows: {unit: {unit: n}}, symmetric travel between distinct units (ACS both ways, or gravity)
- alpha: a join z -> o passes the partner test when flow(z, o) >= alpha x z's flow with its biggest
  partner among every other area (zones, metros, packs; any region)
- compact: (fn, cmin): or when fn(union) >= cmin
- small_by: "border" (today: units under ZONE_MIN_POP join the longest border) or "flow" (their
  biggest partner among the neighbours, the longest border breaking ties and where no flow)
- init: {lead: [units]} starting zones (default: each free unit alone); movable: leads that may join
  (default all); a zone that joins one of them takes its lead
- fixed_units: {fixed index: [units]} for the partner test (a metro's or pack's units)
"""
import math

import shapely
from shapely import STRtree, union_all

import world_plan as wp

ZONE_MIN_POP = wp.ZONE_MIN_POP
WATER_GAP_KM, ARCHIPELAGO_KM, ISLAND_KM = wp.WATER_GAP_KM, wp.ARCHIPELAGO_KM, wp.ISLAND_KM
WRAP_HULL, WRAP_BORDER, RING_ROUNDS = wp.WRAP_HULL, wp.WRAP_BORDER, wp.RING_ROUNDS
polygonal = wp.polygonal


def zone_groups(units, free, region, floor, cap, contiguous=False, fixed=(), *, flows=None, alpha=None,
                compact=None, small_by="border", init=None, movable=None, fixed_units=None, simple=None,
                adj=None, tests=None, targets=None, rings_on=True, relative=None, cross=False, order="near", guard=True, aref=None):
    by = {u.id: u for u in units}
    uarea = {u.id: u.geom.area / 1e6 for u in units}
    if simple is None:
        simple = {u.id: polygonal(u.geom.simplify(500)) for u in units}
    if adj is None:
        adj = wp.unit_borders(simple)
    cen = {}
    for i in free:
        x, y, p = by[i].px
        s = float(p.sum())
        c = simple[i].representative_point() if s <= 0 else None
        cen[i] = (c.x, c.y) if c is not None else (float((x * p).sum()) / s, float((y * p).sum()) / s)
    if init is None:
        zones = {i: [i] for i in free}
    else:
        zones = {k: list(v) for k, v in init.items()}
    owner = {i: z for z, ids in zones.items() for i in ids}
    move = set(zones) if movable is None else set(movable)
    lines, shapes = [], {}
    tests = tests if tests is not None else []
    farea = {}  # unit -> fixed area key, for the partner test
    for n, us in (fixed_units or {}).items():
        for i in us:
            farea[i] = ("f", n)

    def ppl(z):
        return sum(by[i].free for i in zones[z])

    def sz(z):
        """A zone's size against the floor: people over the floor, plus area over `aref` km2 (T-903's
        test) when given."""
        v = ppl(z) / floor
        if aref:
            v += sum(uarea[i] for i in zones[z]) / aref
        return v

    def centre(z):
        n = ppl(z)
        if n <= 0:
            return cen[zones[z][0]]
        return (sum(cen[i][0] * by[i].free for i in zones[z]) / n, sum(cen[i][1] * by[i].free for i in zones[z]) / n)

    def geo(z):
        k = (z, len(zones[z]))
        if k not in shapes:
            shapes[k] = union_all([simple[i] for i in zones[z]])
        return shapes[k]

    def nbrs(z):
        out = {}
        for i in zones[z]:
            for j, ln in adj[i].items():
                o = owner.get(j)
                if o is not None and o != z:
                    out[o] = out.get(o, 0.0) + ln
        return out

    def area_flows(z):
        to = {}
        mine = set(zones[z])
        for i in zones[z]:
            for j, n in flows.get(i, {}).items():
                if j in mine:
                    continue
                a = owner.get(j)
                a = ("z", a) if a is not None else farea.get(j)
                if a is not None:
                    to[a] = to.get(a, 0.0) + n
        return to

    def ratio(z, o):
        to = area_flows(z)
        top = max(to.values()) if to else 0.0
        return to.get(("z", o), 0.0) / top if top else 0.0, to.get(("z", o), 0.0)

    def accept(z, o, partner_only=False):
        """The T-1140 test: o is z's biggest travel partner or near it, or the union is compact."""
        if alpha is None and compact is None:
            return True, ""
        r = None
        if flows is not None and alpha is not None:
            r = ratio(z, o)[0]
            if r >= alpha:
                tests.append((z, o, "partner", r, None))
                return True, f"partner {r:.2f}"
        c = None
        if compact is not None and not partner_only:
            fn, cmin = compact
            c = fn(geo(z), geo(o), centre(z), centre(o))
            need = cmin
            if relative:  # no less compact than the bigger part already was (a county on a zone's edge)
                big = max((geo(z), geo(o)), key=lambda g: g.area)
                need = min(cmin, fn(big, big, (0, 0), (1, 1)) - relative)
            if c >= need:
                tests.append((z, o, "compact", r, c))
                return True, f"compact {c:.2f}"
        tests.append((z, o, "refused", r, c))
        return False, f"refused (partner {r if r is None else round(r, 2)}, compact {c if c is None else round(c, 2)})"

    fx = [dict(f) for f in fixed]
    ftree = STRtree([simple[i] for i in free]) if free else None
    for f in fx:
        f["near"] = f["geom"].buffer(1000)
        f["units"] = {}
        for b in (ftree.query(f["near"], predicate="intersects") if ftree else ()):
            i = free[b]
            ln = simple[i].boundary.intersection(f["near"]).length
            if ln > 0:
                f["units"][i] = ln
    for f in fx:
        land = 0.0
        if f["units"]:
            land = f["geom"].boundary.intersection(union_all([simple[i] for i in f["units"]]).buffer(1000)).length
        for o in fx:
            if o is not f and o["geom"].intersects(f["near"]):
                land += f["geom"].boundary.intersection(o["near"]).length
        f["land"] = land
    fx = [f for f in fx if f["land"] > 0]

    def wrapped(gq, land, hull, near, b):
        c0, c1, c2, c3 = gq.bounds
        if c0 > b[2] or c2 < b[0] or c1 > b[3] or c3 < b[1] or not land or not gq.intersects(near):
            return False
        return gq.intersection(hull).area >= WRAP_HULL * gq.area and gq.boundary.intersection(near).length >= WRAP_BORDER * land

    def wraps(z, o):
        if not guard:
            return False
        m = union_all([geo(z), geo(o)])
        hull, near = m.convex_hull, m.buffer(1000)
        b = near.bounds
        for q in zones:
            if q not in (z, o):
                gq = geo(q)
                c0, c1, c2, c3 = gq.bounds
                if c0 > b[2] or c2 < b[0] or c1 > b[3] or c3 < b[1]:
                    continue
                land = sum(nbrs(q).values()) + sum(f["units"].get(i, 0.0) for f in fx for i in zones[q])
                if wrapped(gq, land, hull, near, b):
                    return True
        return any(wrapped(f["geom"], f["land"], hull, near, b) for f in fx)

    def join(z, into, how):
        if region[into] != region[z]:
            how += ", another region"
        lines.append(f"  {by[z].name} ({ppl(z):,.0f}) joins {by[into].name} ({how})")
        for i in zones[z]:
            owner[i] = into
        zones[into] += zones.pop(z)
        if z in move:
            move.add(into)

    def merge(only=None):
        done = set()
        while True:
            small = sorted((sz(z), z) for z in zones if z not in done and z in move and (only is None or z in only) and sz(z) < 1)
            if not small:
                break
            s, z = small[0]
            n = ppl(z)
            g = geo(z)
            nb = nb_all = nbrs(z)
            if targets is not None:
                nb = {o: ln for o, ln in nb.items() if o in targets}
            same = [o for o in zones if o != z and region[o] == region[z] and (targets is None or o in targets)]
            border = {o: ln for o, ln in nb.items() if region[o] == region[z]}
            if not border and s < 0.8:
                border = dict(nb)
            island = all(j in zones[z] for i in zones[z] for j in adj[i])
            into, how, why = None, "", "no zone beside it"
            if border and n < ZONE_MIN_POP:
                if small_by == "flow" and flows is not None:
                    to = area_flows(z)
                    into = max(border, key=lambda o: (to.get(("z", o), 0.0), border[o]))
                    how = f"biggest partner, shared border {border[into] / 1000:.0f} km"
                else:
                    into = max(border, key=border.get)
                    how = f"shared border {border[into] / 1000:.0f} km"
            elif border or (cross is True and nb and n >= ZONE_MIN_POP):
                c = centre(z)
                pool = dict(border)
                if cross is True and alpha is not None and flows is not None:
                    # T-1140 variant: another region's neighbour when it is a travel partner
                    for o in nb:
                        if o not in pool and n + ppl(o) <= cap and ratio(z, o)[0] >= alpha:
                            pool[o] = nb[o]
                if order == "flow" and flows is not None:
                    fl = area_flows(z)
                    fit = sorted((o for o in pool if n + ppl(o) <= cap), key=lambda o: (-fl.get(("z", o), 0.0), math.dist(c, centre(o))))
                    how = "biggest partner"
                else:
                    fit = sorted((o for o in pool if n + ppl(o) <= cap), key=lambda o: math.dist(c, centre(o)))
                    how = "nearest centre"
                border = pool
                if not fit and s >= 0.8:
                    why = "no neighbour within the cap"
                elif not fit:
                    fit = sorted(border, key=lambda o: (ppl(o), -border[o]))
                    how = "smallest neighbour, none within the cap"
                if fit:
                    ok = {}
                    for o in fit:
                        ok[o] = accept(z, o)
                    good = [o for o in fit if ok[o][0]]
                    if not good and cross == "fallback" and s < 0.8:
                        # T-1140: as when its region has no zone left, another region's neighbour
                        other = sorted((o for o in nb if o not in border and n + ppl(o) <= cap), key=lambda o: math.dist(c, centre(o)))
                        for o in other:
                            ok[o] = accept(z, o)
                        good = [o for o in other if ok[o][0]]
                        if good:
                            how += ", none of its region accepted"
                    if good and alpha is not None and flows is not None:
                        # a travel partner first: a join by shape alone only when no neighbour is one
                        # (with `targets`, neighbours outside them count too, so a built zone fails the
                        # test when one of its parts had a partner elsewhere)
                        part = [o for o in good if ok[o][1].startswith("partner")]
                        if part:
                            good = part
                        elif targets is not None and any(
                                o not in targets and region[o] == region[z] and n + ppl(o) <= cap and ratio(z, o)[0] >= alpha
                                for o in nb_all):
                            good = []
                    if not good:
                        why = "no neighbour is a partner or makes a compact zone"
                        fit = []
                    else:
                        fit = good
                if fit:
                    into = next((o for o in fit if not wraps(z, o)), None)
                    if into is not None:
                        how += (" that wraps nothing" if into != fit[0] else "") + (f", {ok[into][1]}" if ok.get(into, (0, ""))[1] else "")
                    elif into is None and s < 0.8 and only is None:
                        into, how = fit[0], "every join wraps; left to the ring cut"
                    elif into is None:
                        why = "every join would wrap"
            elif contiguous:
                pools = [same] + ([[o for o in zones if o != z and region[o] != region[z] and (targets is None or o in targets)]]
                                  if s < 0.8 else [])
                for pool in pools:
                    near = [(d, o) for d, o in ((g.distance(geo(o)), o) for o in pool) if d <= WATER_GAP_KM * 1000]
                    if near:
                        into = min(near)[1]
                        how = f"across water, {min(near)[0] / 1000:.0f} km"
                        break
                if into is None and island:
                    def alone(o):
                        return all(j in zones[o] for i in zones[o] for j in adj[i])
                    near = sorted((g.distance(geo(o)), o) for o in same if g.distance(geo(o)) <= ARCHIPELAGO_KM * 1000)
                    near = [x for x in near if alone(x[1])]
                    if near:
                        into = near[0][1]
                        how = f"archipelago, {near[0][0] / 1000:.0f} km"
            elif island:
                reach = ISLAND_KM if n < ZONE_MIN_POP else 10.0
                near = [(ppl(o), o) for o in same if g.distance(geo(o)) <= reach * 1000]
                if near:
                    into = max(near)[1]
                    how = f"island, {g.distance(geo(into)) / 1000:.0f} km"
            if into is None:
                done.add(z)
                lines.append(f"  {by[z].name} ({n:,.0f}) stays as it is: {'an island with no zone within reach' if island else why}")
                continue
            join(z, into, how)

    def rings():
        shp = [("z", k) for k in zones] + [("f", n) for n in range(len(fx))]
        gs = [geo(k) if t == "z" else fx[k]["geom"] for t, k in shp]
        nears = [g.buffer(1000) for g in gs]
        tree = STRtree(gs)
        nb = [{} for _ in shp]
        for a in range(len(shp)):
            for b in tree.query(nears[a], predicate="intersects"):
                if b != a:
                    nb[a][int(b)] = gs[b].boundary.intersection(nears[a]).length
        out = []
        for a, (t, k) in enumerate(shp):
            if t != "z" or k not in move:
                continue
            hull = gs[a].convex_hull
            for b, ln in nb[a].items():
                land = sum(nb[x].get(b, 0.0) for x in nb[b])
                if not land or not gs[b].area:
                    continue
                ih = gs[b].intersection(hull).area / gs[b].area
                if ih >= WRAP_HULL and ln / land >= WRAP_BORDER:
                    out.append((ih + ln / land, k, shp[b]))
        return sorted(out, reverse=True)

    def cut(z, c):
        ang = sorted((math.atan2(cen[i][1] - c[1], cen[i][0] - c[0]), i) for i in zones[z])
        gaps = [(ang[(a + 1) % len(ang)][0] - ang[a][0]) % (2 * math.pi) for a in range(len(ang))]
        start = (max(range(len(gaps)), key=lambda a: gaps[a]) + 1) % len(ang)
        order = [ang[(start + a) % len(ang)][1] for a in range(len(ang))]
        links = {i: set(adj[i]) for i in order}
        if contiguous:
            t = STRtree([simple[i] for i in order])
            for a, i in enumerate(order):
                for b in t.query(simple[i], predicate="dwithin", distance=WATER_GAP_KM * 1000):
                    links[i].add(order[b])
        tot, run, best = sum(by[i].free for i in order), 0.0, None
        for s in range(1, len(order)):
            run += by[order[s - 1]].free
            if best is None or abs(run - tot / 2) < best[0]:
                a, b = order[:s], order[s:]
                if one_piece(a, links) and one_piece(b, links):
                    best = (abs(run - tot / 2), a, b)
        return best[1:] if best else None

    def one_piece(ids, links):
        ids = set(ids)
        start = next(iter(ids))
        seen, front = {start}, [start]
        while front:
            nxt = []
            for i in front:
                for j in links[i]:
                    if j in ids and j not in seen:
                        seen.add(j)
                        nxt.append(j)
            front = nxt
        return len(seen) == len(ids)

    def name(t, k):
        return by[k].name if t == "z" else fx[k]["name"]

    merge()
    for rnd in range(RING_ROUNDS if rings_on else 0):
        rs = rings()
        if not rs:
            break
        touched = set()
        for _, a, (t, b) in rs:
            if a in touched or a not in zones or (t == "z" and (b not in zones or b in touched)):
                continue
            got = cut(a, fx[b]["xy"] if t == "f" else centre(b)) if len(zones[a]) > 1 else None
            if not got and t == "z" and ppl(a) + ppl(b) <= cap:
                lines.append(f"  ring cut {rnd + 1}: {by[a].name} wraps {by[b].name}, joined")
                for i in zones[b]:
                    owner[i] = a
                zones[a] += zones.pop(b)
                touched |= {a, b}
                continue
            if not got:
                lines.append(f"  ring cut {rnd + 1}: {by[a].name} wraps {name(t, b)}, no cut keeps both halves in one piece")
                continue
            zones.pop(a)
            for h in got:
                zones[h[0]] = list(h)
                shapes.pop((h[0], len(h)), None)
                for i in h:
                    owner[i] = h[0]
                touched.add(h[0])
                move.add(h[0])
            lines.append(f"  ring cut {rnd + 1}: {by[a].name} wraps {name(t, b)}, cut into {by[got[0][0]].name} "
                         f"({len(got[0])} units) and {by[got[1][0]].name} ({len(got[1])})")
        merge(only=touched)
    for _, a, (t, b) in (rings() if rings_on else ()):
        lines.append(f"  ring left: {by[a].name} wraps {name(t, b)}")
    return zones, lines
