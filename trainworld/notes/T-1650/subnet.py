"""T-1650: a plain net for one part. `full` keeps every station and line and opens only the part's
market packs (its pairs alone, on the whole graph); `clip` keeps the stations its zones reach and
each line from the stop before its first such stop to the stop after its last (as reachView).

python subnet.py <plain net> <dump.json> <places, comma> <full|clip> <out.json>
"""
import json, sys

net = json.load(open(sys.argv[1]))
dump = json.load(open(sys.argv[2]))
places = set(sys.argv[3].split(','))
mode = sys.argv[4]
out = sys.argv[5]

net['markets'] = [m for m in net['markets'] if m in places]
if mode == 'full':
    json.dump(net, open(out, 'w'))
    print('markets', len(net['markets']))
    sys.exit()

st_map = dump['st_map']
merged = set()
for z, acc in enumerate(dump['access']):
    if dump['zones'][z][0] in places:
        merged.update(acc)
n_st = len(net['stXY']) // 2
inreach = [st_map[i] in merged if st_map else i in merged for i in range(n_st)]

keep_st = set()
lines = []  # (line, s0, s1)
off = 0
for l, k in enumerate(net['lineN']):
    stops = net['stops'][off:off + k]
    idx = [i for i, s in enumerate(stops) if inreach[s]]
    if idx:
        s0, s1 = max(0, idx[0] - 1), min(k - 1, idx[-1] + 1)
        if s1 > s0:
            lines.append((l, off, k, s0, s1))
            keep_st.update(stops[s0:s1 + 1])
    off += k

order = sorted(keep_st)
new_of = {s: i for i, s in enumerate(order)}
o = {'frame': net['frame'], 'fare': net['fare'], 'markets': net['markets']}
o['stXY'] = [v for s in order for v in net['stXY'][2 * s:2 * s + 2]]
o['complex'] = [net['complex'][s] for s in order]
# complexes name stations by index: keep only those whose target is kept, else alone
o['complex'] = [new_of.get(c, i) if isinstance(c, int) else c for i, c in enumerate(o['complex'])]
lineN, stops, times, tph, cars, hop = [], [], [], [], [], []
toff = 0
offs = []
o_t = 0
# times are 6 per stop, line by line: dir 0 three levels then dir 1 three levels
t_off = []
acc = 0
for k in net['lineN']:
    t_off.append(acc)
    acc += 6 * k
h_off = []
acc = 0
for k in net['lineN']:
    h_off.append(acc)
    acc += 2 * k
for l, off, k, s0, s1 in lines:
    n = s1 - s0 + 1
    lineN.append(n)
    stops.extend(new_of[s] for s in net['stops'][off + s0:off + s1 + 1])
    T = net['times'][t_off[l]:t_off[l] + 6 * k]
    for d in range(2):
        for lev in range(3):
            b = (3 * d + lev) * k
            v = T[b + s0:b + s1 + 1]
            if d == 0:
                v[-1] = 0
            else:
                v[0] = 0
            times.extend(v)
    tph.extend(net['tph'][3 * l:3 * l + 3])
    cars.append(net['cars'][l])
    H = net['hopKm'][h_off[l]:h_off[l] + 2 * k]
    v0 = H[s0:s1 + 1]
    v0[-1] = 0
    rev = H[k:2 * k]  # reversed stop order
    v1 = rev[k - 1 - s1:k - s0]
    v1[-1] = 0
    hop.extend(v0 + v1)
o.update(lineN=lineN, stops=stops, times=times, tph=tph, cars=cars, hopKm=hop)
json.dump(o, open(out, 'w'))
print('markets', len(o['markets']), 'stations', len(order), 'of', n_st, 'lines', len(lineN), 'of', len(net['lineN']))
