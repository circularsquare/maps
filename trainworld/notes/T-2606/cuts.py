"""T-2606: the market's rail trips and rail-km crossing longitude cuts, per period and kind.

A zone's side is its box centre's longitude; a station's its own. Rail-km is a pair's rail trips
times the great-circle km between its zones' centres. Also each line's market boardings (the
dump's free-flow `board`, summed over the line's route nodes) so lines carrying crossing trips show.

python cuts.py <net.json> <dump.json> <cut lons for a sweep: from,to,step> <cut sets, ; separated, each comma> <out.json>
"""
import json, math, sys
import numpy as np

net = json.load(open(sys.argv[1]))
dump = json.load(open(sys.argv[2]))
a0, a1, st = (float(v) for v in sys.argv[3].split(','))
sets = [[float(v) for v in s.split(',')] for s in sys.argv[4].split(';') if s]
out = sys.argv[5]
boxes = np.array(dump['boxes']).reshape(-1, 5)
Z = len(dump['zones'])
assert len(boxes) == Z, (len(boxes), Z)
clon = (boxes[:, 1] + boxes[:, 3]) / 2
clat = np.radians((boxes[:, 2] + boxes[:, 4]) / 2)
xy = net['stXY']
st_lon = np.degrees(np.array(xy[0::2]) / 6371008.8)

P = []
for per in dump['periods']:
    rows = {}
    for kind in ('long', 'com', 'ev'):
        v = np.array([r for r in per[kind] if r[1] is not None and r[0] < Z and r[1] < Z], dtype=float).reshape(-1, 4)
        a, b = v[:, 0].astype(int), v[:, 1].astype(int)
        dl = np.radians(clon[b] - clon[a])
        c = np.sin(clat[a]) * np.sin(clat[b]) + np.cos(clat[a]) * np.cos(clat[b]) * np.cos(dl)
        km = 6371.0088 * np.arccos(np.clip(c, -1, 1))
        rows[kind] = (clon[a], clon[b], v[:, 3], km)
    P.append((per['q'], rows, per))
    del per['long'], per['com'], per['ev']


def part(lon, cs):
    return np.searchsorted(np.array(cs), lon, side='right')


def tally(cs):
    res = {}
    for q, rows, _ in P:
        r = {'total': {}, 'cross': {}, 'own': [0.0] * (len(cs) + 1)}
        for kind, (la, lb, rail, km) in rows.items():
            pa, pb = part(la, cs), part(lb, cs)
            x = pa != pb
            r['total'][kind] = float(rail.sum())
            r['total'][kind + '_km'] = float((rail * km).sum())
            r['cross'][kind] = float(rail[x].sum())
            r['cross'][kind + '_km'] = float((rail * km)[x].sum())
            for i in range(len(cs) + 1):
                r['own'][i] += float(rail[(~x) & (pa == i)].sum())
        res[q] = r
    return res


def share(t, km=False):
    s = '_km' if km else ''
    T = sum(t['total'][k + s] for k in ('long', 'com', 'ev'))
    C = sum(t['cross'][k + s] for k in ('long', 'com', 'ev'))
    return C, T


result = {'sweep': [], 'sets': []}
for c in np.arange(a0, a1 + 1e-9, st):
    r = tally([c])
    stw = int((st_lon < c).sum())
    row = {'cut': float(c), 'west stations': stw, 'east stations': len(st_lon) - stw}
    line = f'cut {c:7.1f}: stations {stw:5d} | {len(st_lon) - stw:5d};'
    for q, t in r.items():
        C, T = share(t)
        Ck, Tk = share(t, km=True)
        row[f'q{q}'] = [C, T, Ck, Tk]
        line += f' q{q} {100 * C / max(T, 1):5.2f}% ({100 * Ck / max(Tk, 1):4.1f}% km)'
    result['sweep'].append(row)
    print(line)
for cs in sets:
    r = tally(cs)
    pst = part(st_lon, cs)
    row = {'cuts': cs, 'stations': [int((pst == i).sum()) for i in range(len(cs) + 1)], 'periods': r}
    result['sets'].append(row)
    print(f'\ncuts {cs}: stations by part {row["stations"]}')
    for q, t in r.items():
        C, T = share(t)
        Ck, Tk = share(t, km=True)
        kinds = {k: f"{t['cross'][k]:.0f}/{t['total'][k]:.0f}" for k in ('long', 'com', 'ev')}
        print(f'  q{q}: crossing {C:.0f} of {T:.0f} ({100 * C / max(T, 1):.2f}%), rail-km {100 * Ck / max(Tk, 1):.1f}%; by kind {kinds}; own by part {[round(v) for v in t["own"]]}')

# each line's market boardings a period (free flow), from the route nodes
rn_line = np.array(dump['rn_line'])
L = len(net['lineN'])
result['line_board'] = {}
for q, _, per in P:
    b = np.bincount(rn_line, weights=np.array(per['board']), minlength=L)
    result['line_board'][q] = b.tolist()
json.dump(result, open(out, 'w'), indent=1)
