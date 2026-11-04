"""T-1005: draw wrap candidates. Usage: draw.py out.png a:b[:title] a:b ..."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json, sys, math
import shapely
from shapely.geometry import shape
from shapely import STRtree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TW = 'C:/Users/anita/projects/maps/trainworld'
D = os.path.dirname(os.path.abspath(__file__))
Z = json.load(open(f'{D}/zones.json', encoding='utf-8'))
reg = {c['id']: c for c in json.load(open(f'{TW}/app/src/game/registry.json', encoding='utf-8'))['cities']}
names = {}
try:
    names = json.load(open(f'{TW}/app/src/game/zone-names.json', encoding='utf-8'))
except Exception:
    pass

def nm(i):
    v = names.get(i)
    if isinstance(v, dict) and v.get('name'):
        return v['name']
    return Z[i]['name']

_cache = {}
def geom(i):
    if i not in _cache:
        g = json.load(open(f'{TW}/data/packs/{i}.boundary.geojson', encoding='utf-8'))
        gs = [shapely.make_valid(shape(f['geometry'])) for f in g['features'] if f.get('geometry')]
        u = shapely.union_all(gs)
        parts = [p for p in shapely.get_parts(u) if p.geom_type in ('Polygon', 'MultiPolygon')]
        _cache[i] = shapely.simplify(shapely.union_all(parts), 0.002)
    return _cache[i]

def draw_poly(ax, g, **kw):
    for p in shapely.get_parts(g):
        if p.geom_type != 'Polygon':
            continue
        x, y = p.exterior.xy
        ax.fill(x, y, **kw)

out = sys.argv[1]
specs = [s.split(':') for s in sys.argv[2:]]
n = len(specs)
cols = min(3, n)
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.2 * rows), squeeze=False)
# zones near each pair, by box from the registry
boxes = {i: reg[i]['box'] for i in Z if i in reg and reg[i].get('box')}
for k, sp in enumerate(specs):
    a, b = sp[0], sp[1]
    ax = axes[k // cols][k % cols]
    ga, gb = geom(a), geom(b)
    x0, y0, x1, y1 = shapely.union_all([ga, gb]).bounds
    padx, pady = (x1 - x0) * 0.12 + 0.05, (y1 - y0) * 0.12 + 0.05
    vb = (x0 - padx, y0 - pady, x1 + padx, y1 + pady)
    for i, bx in boxes.items():
        if i in (a, b):
            continue
        if bx[0] > vb[2] or bx[2] < vb[0] or bx[1] > vb[3] or bx[3] < vb[1]:
            continue
        try:
            gi = geom(i)
        except Exception:
            continue
        draw_poly(ax, gi, facecolor='#eeeeee', edgecolor='#999999', linewidth=0.6)
        c = gi.representative_point()
        if vb[0] < c.x < vb[2] and vb[1] < c.y < vb[3]:
            ax.text(c.x, c.y, nm(i), fontsize=7, color='#777777', ha='center', va='center')
    draw_poly(ax, ga, facecolor='#f2b880', edgecolor='#a0522d', linewidth=1.2)
    draw_poly(ax, gb, facecolor='#7fb3d5', edgecolor='#1f4e79', linewidth=1.2)
    for i, g, col in ((a, ga, '#5a2a0a'), (b, gb, '#0b2a45')):
        c = g.representative_point()
        z = Z[i]
        ax.text(c.x, c.y, f"{nm(i)} ({i})\n{z['pop'] / 1e6:.2f}M, {z['area_km2']:,.0f} km²",
                fontsize=8, color=col, ha='center', va='center', weight='bold')
    ax.set_xlim(vb[0], vb[2]); ax.set_ylim(vb[1], vb[3])
    ax.set_aspect(1 / math.cos(math.radians((y0 + y1) / 2)))
    ax.set_xticks([]); ax.set_yticks([])
    title = sp[2] if len(sp) > 2 else f"{nm(a)} around {nm(b)}"
    ax.set_title(title, fontsize=10)
for k in range(n, rows * cols):
    axes[k // cols][k % cols].axis('off')
fig.suptitle('Orange: the wrapping zone. Blue: the zone it wraps. Grey: other zones.', fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=110)
print(out)
