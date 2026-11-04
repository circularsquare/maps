"""T-1005: replay China's (or Taiwan's) zone grouping with other neighbour choices. Read-only.

    sim.py <ISO3> <variant>     variant: orig | border | compact
Writes sim_<ISO3>_<variant>.json: {lead: [unit ids]}, plus pops_<ISO3>.json (unit people, cached).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, pickle, time
sys.stdout.reconfigure(encoding='utf-8')
TW = 'C:/Users/anita/projects/maps/trainworld'
sys.path.insert(0, f'{TW}/pipeline')
import numpy as np
import shapely
from shapely import union_all, STRtree
import world_plan as wp

D = os.path.dirname(os.path.abspath(__file__))
iso, variant = sys.argv[1], sys.argv[2]
rule = wp.ZONE_RULES[iso]
floor, cap = rule['floor'], rule['cap']
GB = f'{TW}/data/raw/worldplan/geoboundaries'

plan = json.load(open(f'{TW}/pipeline/world_plan.json', encoding='utf-8'))['metros']
zone_ids = {u for r in plan if r.get('kind') == 'zone' for u in r.get('units', {}).get(iso, [])}
levels = {}
for lv in ('ADM2', 'ADM3'):
    p = f'{GB}/{iso}_{lv}.t897e.pkl'
    if os.path.exists(p):
        levels[lv] = pickle.load(open(p, 'rb'))
lv = max(levels, key=lambda k: len(zone_ids & {u[0] for u in levels[k]}))
raw = [u for u in levels[lv] if u[0] in zone_ids]
print(iso, lv, len(raw), 'of', len(zone_ids), 'zone units found', file=sys.stderr)
adm1 = pickle.load(open(f'{GB}/{iso}_ADM1.t897e.pkl', 'rb'))
t1 = STRtree([g for _, _, g in adm1])

class U:
    pass

pp = f'{D}/pops_{iso}.json'
pops = json.load(open(pp)) if os.path.exists(pp) else {}
pop = None
units = []
for uid, name, g in raw:
    u = U(); u.id, u.name, u.geom = uid, name, g
    if uid not in pops:
        pop = pop or wp.Pop()
        pops[uid] = float(pop.inside(g)[2].sum())
    u.free = pops[uid]
    rp = g.representative_point()
    hit = [adm1[i][1] for i in t1.query(rp, predicate='within')]
    u.region = hit[0] if hit else iso
    units.append(u)
json.dump(pops, open(pp, 'w'))

def zone_groups(units, free, region, floor, cap, contiguous, choose):
    """world_plan.zone_groups with the neighbour choice of a zone of ZONE_MIN_POP or more swapped."""
    by = {u.id: u for u in units}
    simple = {u.id: wp.polygonal(u.geom.simplify(500)) for u in units}
    zones, done = {i: [i] for i in free}, set()
    geoc = {}
    ppl = lambda z: sum(by[i].free for i in zones[z])
    def geo(z):
        k = (z, len(zones[z]))
        if k not in geoc:
            geoc[k] = union_all([simple[i] for i in zones[z]])
        return geoc[k]
    tree = STRtree([simple[u.id] for u in units])
    owner = {}
    for z, ids in zones.items():
        for i in ids:
            owner[i] = z
    def near_zones(g, pool):
        hits = {owner[units[j].id] for j in tree.query(g.buffer(1000), predicate='intersects') if units[j].id in owner}
        return [o for o in hits if o in pool]
    def touching(g, pool):
        pool = set(pool)
        b = [(g.boundary.intersection(geo(o).buffer(1000)).length, o) for o in near_zones(g, pool)]
        return [x for x in b if x[0] > 0]
    while True:
        small = sorted((ppl(z), z) for z in zones if z not in done and ppl(z) < floor)
        if not small:
            break
        n, z = small[0]
        g = geo(z)
        same = [o for o in zones if o != z and region[o] == region[z]]
        border = touching(g, same)
        if not border and n < 0.8 * floor:
            border = touching(g, [o for o in zones if o != z and region[o] != region[z]])
        island = not any(g.buffer(1000).intersects(simple[units[j].id]) for j in tree.query(g.buffer(1000))
                         if units[j].id not in zones[z] and units[j].id in by)
        into = None
        if border and n < wp.ZONE_MIN_POP:
            into = max(border)[1]
        elif border:
            fit = [b for b in border if n + ppl(b[1]) <= cap] or border
            if choose == 'orig':
                into = min(fit, key=lambda b: (ppl(b[1]), -b[0]))[1]
            elif choose == 'border':
                into = max(fit, key=lambda b: (b[0], -ppl(b[1])))[1]
            elif choose == 'nowrap':
                # the smallest as now, skipping a neighbour whose merge would wrap a third zone
                def wraps(o):
                    m = union_all([g, geo(o)])
                    h = m.convex_hull
                    mb = m.buffer(1000)
                    for q in near_zones(m, set(zones) - {z, o}):
                        gq = geo(q)
                        if gq.intersection(h).area < 0.5 * gq.area:
                            continue
                        land = sum(s for s, _ in touching(gq, set(zones) - {q}))
                        if land and gq.boundary.intersection(mb).length >= 0.5 * land:
                            return True
                    return False
                fit = sorted(fit, key=lambda b: (ppl(b[1]), -b[0]))
                into = next((b[1] for b in fit if not wraps(b[1])), fit[0][1])
            elif choose == 'compact':
                def sol(o):
                    m = union_all([g, geo(o)])
                    return m.area / m.convex_hull.area
                into = max(fit, key=lambda b: (sol(b[1]), b[0]))[1]
        elif contiguous:
            for pool in (same, [o for o in zones if o != z and region[o] != region[z]]):
                near = [(d, o) for d, o in ((g.distance(geo(o)), o) for o in pool) if d <= wp.WATER_GAP_KM * 1000]
                if near:
                    into = min(near)[1]
                    break
            if into is None and island:
                near = sorted((g.distance(geo(o)), o) for o in same if g.distance(geo(o)) <= wp.ARCHIPELAGO_KM * 1000)
                if near:
                    into = near[0][1]
        if into is None:
            done.add(z)
            continue
        for i in zones[z]:
            owner[i] = into
        zones[into] += zones.pop(z)
    return zones

t0 = time.time()
region = {u.id: u.region for u in units}
groups = zone_groups(units, [u.id for u in units], region, floor, cap, True, variant)
json.dump(groups, open(f'{D}/sim_{iso}_{variant}.json', 'w'))
print(iso, variant, len(groups), 'zones', round(time.time() - t0), 's', file=sys.stderr)
