"""T-1005: for each two-row regroup in retrofit_<ISO>.json, the best split of the two rows' units into
two touching groups of 1M-2.5M people, by the larger of the two groups' smaller convex-hull fill
(area / hull area). Writes split2_<ISO>.json and prints old against new."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, pickle, itertools
sys.stdout.reconfigure(encoding='utf-8')
import shapely
from shapely import union_all
TW = 'C:/Users/anita/projects/maps/trainworld'
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{TW}/pipeline')
import world_plan as wp
iso = sys.argv[1]
NMAX = 20
plan = {r['id']: r for r in json.load(open(f'{TW}/pipeline/world_plan.json', encoding='utf-8'))['metros']}
pops = json.load(open(f'{D}/pops_{iso}.json'))
lv = 'ADM3' if iso == 'CHN' else 'ADM2'
gb = pickle.load(open(f'{TW}/data/raw/worldplan/geoboundaries/{iso}_{lv}.t897e.pkl', 'rb'))
geom = {u[0]: shapely.make_valid(u[2].simplify(300)) for u in gb}
name = {u[0]: wp.zone_name(u[1], iso) for u in gb}
RF = json.load(open(f'{D}/retrofit_{iso}.json', encoding='utf-8'))
floor, cap = wp.ZONE_RULES[iso]['floor'], wp.ZONE_RULES[iso]['cap']
out = []
for r in RF:
    if len(r['rows']) != 2 or r['how'] != 'regroup':
        continue
    ids = [u for z in r['rows'] for u in plan[z]['units'][iso]]
    n = len(ids)
    if n > NMAX:
        out.append(dict(rows=r['rows'], skipped=f'{n} units'))
        print(r['rows'], 'skipped,', n, 'units')
        continue
    buf = {i: geom[i].buffer(1000) for i in ids}
    adj = {i: {j for j in ids if j != i and buf[i].intersects(geom[j])} for i in ids}
    def connected(s):
        s = set(s); start = next(iter(s)); seen = {start}; st = [start]
        while st:
            x = st.pop()
            for y in adj[x] & s:
                if y not in seen:
                    seen.add(y); st.append(y)
        return len(seen) == len(s)
    tot = sum(pops[i] for i in ids)
    best = None
    first, rest = ids[0], ids[1:]
    for k in range(0, n - 1):
        for comb in itertools.combinations(rest, k):
            a = (first,) + comb
            pa = sum(pops[i] for i in a)
            if not (floor <= pa <= cap and floor <= tot - pa <= cap):
                continue
            b = [i for i in ids if i not in a]
            if not connected(a) or not connected(b):
                continue
            ga, gbb = union_all([geom[i] for i in a]), union_all([geom[i] for i in b])
            s = min(ga.area / ga.convex_hull.area, gbb.area / gbb.convex_hull.area)
            if best is None or s > best[0]:
                best = (s, a, b, pa, tot - pa)
    old = [union_all([geom[u] for u in plan[z]['units'][iso]]) for z in r['rows']]
    olds = min(g.area / g.convex_hull.area for g in old)
    if best is None:
        out.append(dict(rows=r['rows'], none=True))
        print(r['rows'], 'no split of 1M-2.5M each')
        continue
    s, a, b, pa, pb = best
    grp = lambda xs: [name[i] for i in sorted(xs, key=lambda i: -pops[i])]
    out.append(dict(rows=r['rows'], old_min_fill=round(olds, 2), new_min_fill=round(s, 2),
                    new=[dict(units=grp(a), people=round(pa)), dict(units=grp(b), people=round(pb))]))
    print(f"{'+'.join(r['rows'])}: fill {olds:.2f} -> {s:.2f}: {pa / 1e6:.2f}M [{', '.join(grp(a)[:5])}] / {pb / 1e6:.2f}M [{', '.join(grp(b)[:5])}]")
json.dump(out, open(f'{D}/split2_{iso}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
