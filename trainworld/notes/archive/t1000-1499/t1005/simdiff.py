"""T-1005: which plan rows a sim variant changes, and what it makes of them.
    simdiff.py <ISO3> <variant> [row ids to show]"""
import os, sys, json, pickle, collections
sys.stdout.reconfigure(encoding='utf-8')
import shapely
TW = 'C:/Users/anita/projects/maps/trainworld'
D = os.path.dirname(os.path.abspath(__file__))
iso, variant = sys.argv[1], sys.argv[2]
show = set(sys.argv[3].split(',')) if len(sys.argv) > 3 else None
GB = f'{TW}/data/raw/worldplan/geoboundaries'
pops = json.load(open(f'{D}/pops_{iso}.json'))
plan = json.load(open(f'{TW}/pipeline/world_plan.json', encoding='utf-8'))['metros']
rows = [r for r in plan if r.get('kind') == 'zone' and iso in r.get('units', {})]
lv = 'ADM3' if iso == 'CHN' else 'ADM2'
gb = pickle.load(open(f'{GB}/{iso}_{lv}.t897e.pkl', 'rb'))
name = {u[0]: u[1] for u in gb}
adm1 = pickle.load(open(f'{GB}/{iso}_ADM1.t897e.pkl', 'rb'))
t1 = shapely.STRtree([g for _, _, g in adm1])
geom = {u[0]: u[2] for u in gb}
def prov(i):
    h = t1.query(geom[i].representative_point(), predicate='within')
    return adm1[h[0]][1] if len(h) else '?'
row_of = {u: r for r in rows for u in r['units'][iso]}
groups = json.load(open(f'{D}/sim_{iso}_{variant}.json'))
changed = collections.Counter()
same = 0
out = []
for lead, ids in groups.items():
    rs = sorted({row_of[i]['id'] for i in ids if i in row_of})
    exact = len(rs) == 1 and sorted(row_of[ids[0]]['units'][iso]) == sorted(ids)
    if exact:
        same += 1
        continue
    out.append((rs, ids))
    for r in rs:
        changed[r] += 1
print(f'{iso} {variant}: {len(groups)} zones, {same} identical to plan rows; plan rows touched: {len(changed)} of {len(rows)}')
byprov = collections.Counter(prov(row_of[r_ids[1][0]]['units'][iso][0]) if False else prov(r_ids[1][0]) for r_ids in out)
print('new zones by province:', dict(byprov.most_common()))
for rs, ids in out:
    if show and not (set(rs) & show):
        continue
    p = sum(pops[i] for i in ids)
    top = sorted(ids, key=lambda i: -pops[i])[:6]
    print(f"  {p / 1e6:.2f}M from rows {rs}: {', '.join(name[i] for i in top)}{' ...' if len(ids) > 6 else ''}")
json.dump(sorted(changed), open(f'{D}/changed_{iso}_{variant}.json', 'w'))
