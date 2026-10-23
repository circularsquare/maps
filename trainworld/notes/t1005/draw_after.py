"""T-1005: before and after for a few regroups. draw_after.py out.png ISO:rowA,rowB[:title] ..."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, pickle, math
TW = 'C:/Users/anita/projects/maps/trainworld'
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{TW}/pipeline'); sys.path.insert(0, D)
import shapely
from shapely import union_all
import world_plan as wp
from sim_fn import zone_groups
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyproj import Transformer
INV = Transformer.from_crs(wp.FWD.target_crs, 'EPSG:4326', always_xy=True)
plan = {r['id']: r for r in json.load(open(f'{TW}/pipeline/world_plan.json', encoding='utf-8'))['metros']}
names = json.load(open(f'{TW}/app/src/game/zone-names.json', encoding='utf-8'))
COL = ['#f2b880', '#7fb3d5', '#a8d5a2', '#d7a9e3']

class U:
    pass

def ll(g):
    return shapely.transform(g, lambda a: __import__('numpy').column_stack(INV.transform(a[:, 0], a[:, 1])))

def fill(ax, g, c):
    for p in shapely.get_parts(g):
        if p.geom_type == 'Polygon':
            x, y = p.exterior.xy
            ax.fill(x, y, facecolor=c, edgecolor='#333333', linewidth=1)

specs = [s.split(':') for s in sys.argv[2:]]
fig, axes = plt.subplots(len(specs), 2, figsize=(10, 5 * len(specs)), squeeze=False)
cache = {}
for k, sp in enumerate(specs):
    iso, rows = sp[0], sp[1].split(',')
    title = sp[2] if len(sp) > 2 else ''
    if iso not in cache:
        lv = 'ADM3' if iso == 'CHN' else 'ADM2'
        gb = pickle.load(open(f'{TW}/data/raw/worldplan/geoboundaries/{iso}_{lv}.t897e.pkl', 'rb'))
        cache[iso] = ({u[0]: shapely.make_valid(u[2].simplify(300)) for u in gb}, {u[0]: u[1] for u in gb},
                      json.load(open(f'{D}/pops_{iso}.json')))
    geom, uname, pops = cache[iso]
    units = [u for r in rows for u in plan[r]['units'][iso]]
    us = []
    for u in units:
        x = U(); x.id, x.name, x.geom, x.free = u, uname[u], geom[u], pops[u]
        us.append(x)
    rule = wp.ZONE_RULES[iso]
    groups = zone_groups(us, units, {u: 'c' for u in units}, rule['floor'], rule['cap'], True, 'nowrap')
    before = [(r, union_all([geom[u] for u in plan[r]['units'][iso]]), sum(pops[u] for u in plan[r]['units'][iso])) for r in rows]
    after = [(None, union_all([geom[u] for u in ids]), sum(pops[u] for u in ids), ids) for ids in groups.values()]
    for col, sets, lab in ((0, before, 'Now'), (1, after, 'With the rule')):
        ax = axes[k][col]
        for i, item in enumerate(sets):
            g = ll(item[1]); fill(ax, g, COL[i % 4])
            c = g.representative_point()
            if item[0]:
                n = names.get(item[0], {}).get('name') if isinstance(names.get(item[0]), dict) else None
                txt = f"{n or plan[item[0]]['name']} ({item[0]})"
            else:
                top = [u for u in sorted(item[3], key=lambda u: -pops[u]) if wp.latin(wp.zone_name(uname[u], iso))][:3]
                txt = '\n'.join(wp.zone_name(uname[u], iso) for u in top)
            ax.text(c.x, c.y, f"{txt}\n{item[2] / 1e6:.2f}M", ha='center', va='center', fontsize=8, weight='bold')
        b = union_all([ll(s[1]) for s in sets]).bounds
        ax.set_aspect(1 / math.cos(math.radians((b[1] + b[3]) / 2)))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}: {lab.lower()}" if title else lab, fontsize=10)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=100)
print(sys.argv[1])
