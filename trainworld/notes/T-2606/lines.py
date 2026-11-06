"""T-2606: the captured network's stations by longitude, and the lines that span a cut.

python lines.py <net.json> <cut lons, comma>
"""
import json, math, sys, collections

net = json.load(open(sys.argv[1]))
cuts = [float(c) for c in sys.argv[2].split(',')]
R = 6371008.8
xy = net['stXY']
S = len(xy) // 2
lon = [math.degrees(xy[2 * s] / R) for s in range(S)]
lat = [math.degrees(math.atan(math.sinh(xy[2 * s + 1] / R))) for s in range(S)]
h = collections.Counter(int(math.floor(l / 5) * 5) for l in lon)
print('stations by 5 degrees of longitude:', sorted(h.items()))
off, lines = 0, []
hoff = 0
for l, k in enumerate(net['lineN']):
    st = net['stops'][off:off + k]
    km = sum(net['hopKm'][hoff:hoff + k])
    lines.append((l, st, km, net['tph'][3 * l:3 * l + 3]))
    off += k
    hoff += 2 * k
long = sorted(lines, key=lambda x: -x[2])[:25]
print('\nlongest lines: index, stops, km, lon range, lat range, trains an hour')
for l, st, km, tph in long:
    print(f'  {l:4d} {len(st):4d} {km:7.0f} km  lon {min(lon[s] for s in st):8.2f} .. {max(lon[s] for s in st):8.2f}  lat {min(lat[s] for s in st):6.2f} .. {max(lat[s] for s in st):6.2f}  {[round(t, 1) for t in tph]}')
for c in cuts:
    span = [(l, len(st), round(km)) for l, st, km, _ in lines if st and min(lon[s] for s in st) < c <= max(lon[s] for s in st)]
    w = sum(1 for v in lon if v < c)
    print(f'\ncut {c}: stations west {w}, east {S - w}; lines spanning it {span}')
