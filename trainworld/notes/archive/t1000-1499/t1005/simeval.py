"""T-1005: evaluate a sim_<ISO>_<variant>.json (or the plan's own groups, variant 'plan'):
zone count, people spread, wrap pairs (B half inside A's hull, A holding half B's border with zones).
    simeval.py <ISO3> <variant> [province filter]"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, pickle
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import shapely
from shapely import union_all, STRtree

TW = 'C:/Users/anita/projects/maps/trainworld'
D = os.path.dirname(os.path.abspath(__file__))
iso, variant = sys.argv[1], sys.argv[2]
GB = f'{TW}/data/raw/worldplan/geoboundaries'
pops = json.load(open(f'{D}/pops_{iso}.json'))
plan = json.load(open(f'{TW}/pipeline/world_plan.json', encoding='utf-8'))['metros']
rows = [r for r in plan if r.get('kind') == 'zone' and iso in r.get('units', {})]
metros = [r for r in plan if r.get('kind', 'city') == 'city' and iso in r.get('units', {}) and r.get('status') != 'dropped']
lv = 'ADM3' if iso == 'CHN' else 'ADM2'
gb = pickle.load(open(f'{GB}/{iso}_{lv}.t897e.pkl', 'rb'))
geom = {u[0]: u[2] for u in gb}
name = {u[0]: u[1] for u in gb}
if variant == 'plan':
    groups = {r['units'][iso][0]: r['units'][iso] for r in rows}
else:
    groups = json.load(open(f'{D}/sim_{iso}_{variant}.json'))
row_of = {tuple(sorted(r['units'][iso])): r for r in rows}
same = sum(1 for ids in groups.values() if tuple(sorted(ids)) in row_of)

G = {}
for lead, ids in groups.items():
    g = shapely.make_valid(union_all([shapely.make_valid(geom[i].simplify(300)) for i in ids]))
    G[lead] = dict(g=g, pop=sum(pops.get(i, 0) for i in ids), ids=ids, kind='zone')
for r in metros:
    ids = [i for i in r['units'][iso] if i in geom]
    if ids:
        G['M:' + r['id']] = dict(g=shapely.make_valid(union_all([shapely.make_valid(geom[i].simplify(300)) for i in ids])), pop=r['people'], ids=ids, kind='metro', mid=r['id'], mname=r['name'])
keys = list(G)
tree = STRtree([G[k]['g'] for k in keys])
border = {}
pairs = []
for a in keys:
    if G[a]['kind'] != 'zone':
        continue
    A = G[a]['g']; H = A.convex_hull; Ab = A.buffer(1000)
    for j in tree.query(Ab, predicate='intersects'):
        b = keys[j]
        if b == a:
            continue
        B = G[b]['g']
        sh = B.boundary.intersection(Ab).length
        if sh < 2000:
            continue
        pairs.append((a, b, sh, B.intersection(H).area / B.area))
for a in keys:
    pass
land = {}
for k in keys:
    B = G[k]['g']; Bb = B.boundary
    tot = 0
    for j in tree.query(B.buffer(1000), predicate='intersects'):
        o = keys[j]
        if o != k:
            tot += Bb.intersection(G[o]['g'].buffer(1000)).length
    land[k] = tot

def label(k):
    z = G[k]
    if z['kind'] == 'metro':
        return f"{z['mid']} {z['mname']} (metro)"
    r = row_of.get(tuple(sorted(z['ids'])))
    lead = max(z['ids'], key=lambda i: pops.get(i, 0))
    return (f"{r['id']} " if r else '') + name.get(lead, lead)

wr = []
for a, b, sh, inh in pairs:
    s = sh / land[b] if land[b] else 0
    if inh >= 0.5 and s >= 0.5:
        wr.append((a, b, inh, s))
pp = np.array([G[k]['pop'] for k in keys if G[k]['kind'] == 'zone'])
print(f"{iso} {variant}: {len(pp)} zones ({same} identical to plan rows); people median {np.median(pp) / 1e6:.2f}M, "
      f"under 1M {int((pp < 1e6).sum())}, over 2.5M {int((pp > 2.5e6).sum())}; wrap pairs {len(wr)}")
flt = sys.argv[3] if len(sys.argv) > 3 else None
for a, b, inh, s in sorted(wr, key=lambda x: -x[2] * x[3]):
    print(f"  {label(a)} {G[a]['pop'] / 1e6:.2f}M wraps {label(b)} {G[b]['pop'] / 1e6:.2f}M: in hull {inh:.2f}, border {s:.2f}")
