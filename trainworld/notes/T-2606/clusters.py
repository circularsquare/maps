"""T-2606: the clusters a captured network makes (demandClusters.ts `splitClusters`, approximated in
lon/lat: a city's cells' box taken as its registry box) and what joins them.

Stage 1 joins stations by lines, complexes and the 1 km rule; stage 2 adds the cities' 30 km reach;
stage 3 the market's 6 km zones' 30 km reach (zone boxes from the native dump). For each stage the
components; then, across a longitude cut, every line, city and market zone that touches a stage-1
component on each side.

python clusters.py <net.json> <dump.json> <registry.json> <cut lon, comma> <out.json>
"""
import json, math, sys, collections

net = json.load(open(sys.argv[1]))
dump = json.load(open(sys.argv[2]))
reg = json.load(open(sys.argv[3], encoding='utf8'))
cuts = [float(c) for c in sys.argv[4].split(',')]
out = sys.argv[5]
R = 6371008.8
M_PER_DEG = math.pi * R / 180
fr = net['frame']
xy = net['stXY']
S = len(xy) // 2


def lonlat(x, y):
    # the game's world frame is Mercator (t780_net reads it so)
    return math.degrees(x / R), math.degrees(math.atan(math.sinh(y / R)))


ll = [lonlat(xy[2 * s], xy[2 * s + 1]) for s in range(S)]


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def f(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def j(self, a, b):
        a, b = self.f(a), self.f(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


drop = {int(v) for v in sys.argv[6].split(',')} if len(sys.argv) > 6 else set()
off, line_stops = 0, []
for l, k in enumerate(net['lineN']):
    line_stops.append([] if l in drop else net['stops'][off:off + k])
    off += k

uf = UF(S)
for st in line_stops:
    for a, b in zip(st, st[1:]):
        uf.j(a, b)
first = {}
for s, c in enumerate(net['complex']):
    if c in first:
        uf.j(first[c], s)
    else:
        first[c] = s


def gdist(a, b):
    (x1, y1), (x2, y2) = ll[a], ll[b]
    return M_PER_DEG * math.hypot((x1 - x2) * math.cos(math.radians((y1 + y2) / 2)), y1 - y2)


grid = collections.defaultdict(list)
G = 0.02
for s in range(S):
    gx, gy = int(ll[s][0] // G), int(ll[s][1] // G)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for t in grid[(gx + dx, gy + dy)]:
                if gdist(s, t) <= 1000:
                    uf.j(s, t)
    grid[(gx, gy)].append(s)
stage1 = [uf.f(s) for s in range(S)]


def in_box(s, box, reach=30000):
    lon, lat = ll[s]
    dy = reach / M_PER_DEG
    dx = dy / math.cos(math.radians(max(abs(box[1]), abs(box[3]))))
    return box[0] - dx <= lon <= box[2] + dx and box[1] - dy <= lat <= box[3] + dy


cities = {c['id']: c for c in reg['cities']}
packs = [p for p in net.get('packs', []) if p in cities and cities[p].get('box')]
city_st = {p: [s for s in range(S) if in_box(s, cities[p]['box'])] for p in packs}
boxes = dump['boxes']
Z = len(boxes) // 5
zone_st = {}
for z in range(Z):
    b = boxes[5 * z:5 * z + 5]
    if not b[1] <= b[3]:
        continue
    zone_st[z] = [s for s in range(S) if in_box(s, b[1:5])]


def comps(uf):
    c = collections.defaultdict(list)
    for s in range(S):
        c[uf.f(s)].append(s)
    return sorted(c.values(), key=len, reverse=True)


def describe(cs):
    rows = []
    for c in cs:
        lons = [ll[s][0] for s in c]
        rows.append({'stations': len(c), 'lon': [round(min(lons), 2), round(max(lons), 2)]})
    return rows


res = {'stations': S, 'lines': len(line_stops), 'packs': len(packs), 'market zones': len(zone_st)}
res['stage1'] = describe(comps(uf))
for p, st in city_st.items():
    for a, b in zip(st, st[1:]):
        uf.j(a, b)
res['stage2'] = describe(comps(uf))
for z, st in zone_st.items():
    for a, b in zip(st, st[1:]):
        uf.j(a, b)
res['stage3'] = describe(comps(uf))
for k in ('stage1', 'stage2', 'stage3'):
    big = res[k][:8]
    print(k, len(res[k]), 'components; largest', [(r['stations'], r['lon']) for r in big])

# what crosses each cut: a stage-1 component is west or east of the cut by its stations' mean lon
c1 = collections.defaultdict(list)
for s in range(S):
    c1[stage1[s]].append(s)
side = {r: (sum(ll[s][0] for s in m) / len(m)) for r, m in c1.items()}
res['cuts'] = {}
for cut in cuts:
    w = lambda s: ll[s][0] < cut
    lines = [l for l, st in enumerate(line_stops) if st and len({ll[s][0] < cut for s in st}) > 1]
    joiners = {'lines': [], 'cities': [], 'zones': []}
    for l, st in enumerate(line_stops):
        if st and len({w(s) for s in st}) > 1:
            lons = [ll[s][0] for s in st]
            joiners['lines'].append([l, len(st), round(min(lons), 2), round(max(lons), 2)])
    for p, st in city_st.items():
        if len({w(s) for s in st}) > 1:
            joiners['cities'].append([p, cities[p]['name'], cities[p]['box'], len(st)])
    for z, st in zone_st.items():
        if len({w(s) for s in st}) > 1:
            b = boxes[5 * z:5 * z + 5]
            joiners['zones'].append([z, b[0], [round(v, 3) for v in b[1:]], len(st)])
    nw = sum(1 for s in range(S) if w(s))
    res['cuts'][str(cut)] = {'west stations': nw, 'east stations': S - nw, 'lines crossing by stops': len(lines), **joiners}
    print(f'cut {cut}: west {nw} east {S - nw} stations; joined across by {len(joiners["lines"])} lines, {len(joiners["cities"])} cities, {len(joiners["zones"])} market zones')
    for l in joiners['lines'][:10]:
        print('   line', l)
    for c in joiners['cities'][:10]:
        print('   city', c)
    for z in joiners['zones'][:10]:
        print('   zone', z)
json.dump(res, open(out, 'w'), indent=1)
