"""T-1005: the retrofit for built Chinese and Taiwanese zones. Read-only.
Groups the zone-zone wrap pairs of candidates.json into components; a component under the cap merges,
else its units are regrouped by the planner with the no-wrap guard (sim.py's zone_groups), and the
result is checked for wraps against itself and the zones around it. Writes retrofit_<ISO3>.json."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, pickle, collections
sys.stdout.reconfigure(encoding='utf-8')
TW = 'C:/Users/anita/projects/maps/trainworld'
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{TW}/pipeline')
import shapely
from shapely import union_all, STRtree
import world_plan as wp

iso = sys.argv[1]
a2 = {'CHN': 'CN', 'TWN': 'TW'}[iso]
rule = wp.ZONE_RULES[iso]
floor, cap = rule['floor'], rule['cap']
GB = f'{TW}/data/raw/worldplan/geoboundaries'
pops = json.load(open(f'{D}/pops_{iso}.json'))
plan = {r['id']: r for r in json.load(open(f'{TW}/pipeline/world_plan.json', encoding='utf-8'))['metros']}
lv = 'ADM3' if iso == 'CHN' else 'ADM2'
gb = pickle.load(open(f'{GB}/{iso}_{lv}.t897e.pkl', 'rb'))
geom = {u[0]: shapely.make_valid(u[2].simplify(300)) for u in gb}
uname = {u[0]: u[1] for u in gb}
cand = [c for c in json.load(open(f'{D}/candidates.json', encoding='utf-8'))
        if c['country'] == a2 and c['a_kind'] == 'zone' and c['b_kind'] == 'zone']

# components of zones joined by wrap pairs
par = {}
def find(x):
    par.setdefault(x, x)
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x
for c in cand:
    par[find(c['a'])] = find(c['b'])
comps = collections.defaultdict(set)
for c in cand:
    comps[find(c['a'])] |= {c['a'], c['b']}

# every zone and metro row of the country, as unit unions, for the wrap check
rowgeo = {}
for rid, r in plan.items():
    us = [u for u in r.get('units', {}).get(iso, []) if u in geom]
    if us and r.get('status') not in ('dropped',):
        rowgeo[rid] = union_all([geom[u] for u in us])
rk = list(rowgeo)
rtree = STRtree([rowgeo[k] for k in rk])

def wraps(gs):
    """Wrap pairs among the groups `gs` ({name: geometry}) and the rows around them."""
    allg = dict(gs)
    hull_pool = set(gs)
    for k in gs:
        for j in rtree.query(gs[k].buffer(1000), predicate='intersects'):
            o = rk[j]
            if o in rowgeo and o not in allg:
                allg[o] = rowgeo[o]
    # rows replaced by the groups are not neighbours of their own pieces
    out = []
    keys = list(allg)
    def land(k):
        g = allg[k]
        return sum(g.boundary.intersection(allg[o].buffer(1000)).length for o in keys if o != k and allg[o].distance(g) < 1000)
    for a in hull_pool:
        A = allg[a]; H = A.convex_hull; Ab = A.buffer(1000)
        for b in keys:
            if b == a:
                continue
            B = allg[b]
            if B.distance(A) > 1000:
                continue
            if B.intersection(H).area < 0.5 * B.area:
                continue
            l = land(b)
            if l and B.boundary.intersection(Ab).length >= 0.5 * l:
                out.append((a, b))
    return out

sys.argv = [sys.argv[0], iso, 'nowrap']
import importlib.util
spec = importlib.util.spec_from_file_location('simmod', f'{D}/sim_fn.py')

class U:
    pass

result = []
for root, zs in sorted(comps.items(), key=lambda kv: -len(kv[1])):
    zs = sorted(zs)
    units = [u for z in zs for u in plan[z]['units'][iso]]
    tot = sum(pops[u] for u in units)
    before = {z: round(sum(pops[u] for u in plan[z]['units'][iso])) for z in zs}
    if tot <= cap:
        groups = {zs[0]: units}
        how = 'merge'
    else:
        from sim_fn import zone_groups
        us = []
        for u in units:
            x = U(); x.id, x.name, x.geom, x.free = u, uname[u], geom[u], pops[u]
            us.append(x)
        groups = zone_groups(us, units, {u: 'c' for u in units}, floor, cap, True, 'nowrap')
        how = 'regroup'
    gs = {f'new{i}:' + ','.join(sorted({z for z in zs if set(plan[z]['units'][iso]) & set(ids)})): union_all([geom[u] for u in ids])
          for i, ids in enumerate(groups.values())}
    # the check: rows of the component left out of the neighbour pool
    for z in zs:
        rowgeo.pop(z, None)
    w = wraps(gs)
    for z in zs:
        rowgeo[z] = union_all([geom[u] for u in plan[z]['units'][iso]])
    newz = []
    for k, ids in zip(gs, groups.values()):
        g = gs[k]
        newz.append(dict(units=[uname[u] for u in sorted(ids, key=lambda u: -pops[u])], people=round(sum(pops[u] for u in ids)),
                         km2=round(g.area / 1e6), solidity=round(g.area / g.convex_hull.area, 2), from_rows=k.split(':')[1].split(',')))
    same = sum(1 for ids in groups.values() for z in zs if sorted(ids) == sorted(plan[z]['units'][iso]))
    result.append(dict(rows=zs, names={z: plan[z]['name'] for z in zs}, before=before, how=how, new=newz,
                       unchanged=same, wraps_after=[f'{a.split(":")[0]} wraps {b}' for a, b in w]))
    print(f"{'+'.join(zs)} ({', '.join(f'{plan[z][chr(110)+chr(97)+chr(109)+chr(101)]} {before[z] / 1e6:.2f}M' for z in zs)}): {how} -> "
          + '; '.join(f"{n['people'] / 1e6:.2f}M {n['km2']:,} km2 sol {n['solidity']} [{', '.join(n['units'][:4])}{' ...' if len(n['units']) > 4 else ''}]" for n in newz)
          + (f"  STILL WRAPS: {w}" if w else ''))
json.dump(result, open(f'{D}/retrofit_{iso}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
