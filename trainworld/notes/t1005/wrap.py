"""T-1005: find zones that wrap another zone. Read-only on trainworld data.

Writes zones.json (per-zone metrics) and pairs.json (per touching pair metrics) to the
scratchpad folder.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json, sys, time
import numpy as np
import shapely
from shapely.geometry import shape
from shapely import STRtree
from pyproj import Transformer

TW = 'C:/Users/anita/projects/maps/trainworld'
OUT = os.path.dirname(os.path.abspath(__file__))
TOL = 1000.0  # m: a border counts as shared within this distance

reg = json.load(open(f'{TW}/app/src/game/registry.json', encoding='utf-8'))['cities']
plan = {r['id']: r for r in json.load(open(f'{TW}/pipeline/world_plan.json', encoding='utf-8'))['metros']}

ids = [c['id'] for c in reg]
regd = {c['id']: c for c in reg}
# wave 4's gated rows have packs but are not registered yet: include them, marked
extra = [i for i, r in plan.items() if r.get('status') == 'gated' and i not in regd
         and os.path.exists(f'{TW}/data/packs/{i}.boundary.geojson')]
ids += extra

def load(i):
    p = f'{TW}/data/packs/{i}.boundary.geojson'
    if not os.path.exists(p):
        return None
    g = json.load(open(p, encoding='utf-8'))
    geoms = [shapely.make_valid(shape(f['geometry'])) for f in g['features'] if f.get('geometry')]
    u = shapely.union_all(geoms)
    return polys(u)

def polys(g):
    g = shapely.make_valid(g)
    if g.geom_type in ('Polygon', 'MultiPolygon'):
        return g
    parts = [p for p in shapely.get_parts(g) if p.geom_type in ('Polygon', 'MultiPolygon')]
    return shapely.union_all(parts)

def laea(lon, lat):
    return Transformer.from_crs('EPSG:4326', f'+proj=laea +lat_0={lat} +lon_0={lon} +units=m', always_xy=True)

def proj(g, tr):
    return shapely.transform(g, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))

t0 = time.time()
Z = {}
for i in ids:
    g = load(i)
    if g is None or g.is_empty:
        continue
    pk = {}
    pj = f'{TW}/data/packs/{i}.json'
    if os.path.exists(pj):
        try:
            pk = json.load(open(pj, encoding='utf-8')).get('totals', {})
        except Exception:
            pk = {}
    c = g.centroid
    tr = laea(c.x, c.y)
    gp = polys(shapely.simplify(proj(g, tr), 100))
    hull = gp.convex_hull
    pr = plan.get(i, {})
    rg = regd.get(i, {})
    Z[i] = dict(id=i, name=rg.get('name') or pr.get('name'), kind=rg.get('kind') or pr.get('kind') or 'city',
                country=rg.get('country') or pr.get('iso2'), registered=i in regd,
                pop=pk.get('pop') or pr.get('people'), plan_people=pr.get('people'),
                area_km2=gp.area / 1e6, perim_km=gp.length / 1e3, hull_km2=hull.area / 1e6,
                solidity=gp.area / hull.area if hull.area else 1,
                polsby=4 * np.pi * gp.area / gp.length ** 2 if gp.length else 1,
                lon=c.x, lat=c.y, units=pr.get('source', {}).get('unit_names'),
                solve_region=pr.get('solve_region'), status=pr.get('status'))
    Z[i]['_g'] = polys(shapely.simplify(g, 0.0005))
print(len(Z), 'zones loaded', round(time.time() - t0), 's', file=sys.stderr)

keys = list(Z)
tree = STRtree([Z[k]['_g'] for k in keys])
pairs = []
for a in keys:
    A = Z[a]
    hits = tree.query(A['_g'].buffer(0.02), predicate='intersects')
    tr = laea(A['lon'], A['lat'])
    Ap = polys(proj(A['_g'], tr))
    Ah = Ap.convex_hull
    Abuf = Ap.buffer(TOL)
    concav = Ah.difference(Ap)
    for h in hits:
        b = keys[h]
        if b == a:
            continue
        B = Z[b]
        Bp = polys(proj(B['_g'], tr))
        shared_b = Bp.boundary.intersection(Abuf).length  # B's perimeter lying on A
        if shared_b < 2 * TOL:
            continue
        shared_a = Ap.boundary.intersection(Bp.buffer(TOL)).length
        inhull = Bp.intersection(Ah).area
        pairs.append(dict(a=a, b=b,
                          shared_km=round(shared_b / 1e3, 2),
                          share_of_b=shared_b / Bp.boundary.length,
                          share_of_a=shared_a / Ap.boundary.length,
                          b_in_a_hull=inhull / Bp.area if Bp.area else 0,
                          b_centre_in_a_hull=bool(Ah.contains(Bp.centroid)),
                          b_fills_a_concavity=inhull / concav.area if concav.area > 1e6 else 0))
print(len(pairs), 'pairs', round(time.time() - t0), 's', file=sys.stderr)

# each zone's border shared with any registered zone (its "land" border)
landb = {}
for p in pairs:
    landb[p['b']] = landb.get(p['b'], 0) + p['shared_km']
for p in pairs:
    p['share_of_b_land'] = p['shared_km'] / landb[p['b']] if landb.get(p['b']) else 0

for z in Z.values():
    z.pop('_g')
json.dump(Z, open(f'{OUT}/zones.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
json.dump(pairs, open(f'{OUT}/pairs.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print('done', round(time.time() - t0), 's', file=sys.stderr)
