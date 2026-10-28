"""T-1650: a plain net plus a new express over an existing line: every `k`th stop, run times summed
times `speed`, `tph` trains an hour at each level, 12 cars.
python newline.py <plain net> <line or 'longest'> <k> <speed> <tph> <out.json>"""
import json, sys
n = json.load(open(sys.argv[1]))
k, speed, tph = int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
offs, toffs, hoffs = [], [], []
a = t = h = 0
for c in n['lineN']:
    offs.append(a); toffs.append(t); hoffs.append(h)
    a += c; t += 6 * c; h += 2 * c
def km(l):
    c = n['lineN'][l]
    return sum(n['hopKm'][hoffs[l]:hoffs[l] + c])
if sys.argv[2] == 'longest':
    l = max(range(len(n['lineN'])), key=km)
else:
    l = int(sys.argv[2])
c = n['lineN'][l]
idx = list(range(0, c, k))
if idx[-1] != c - 1:
    idx.append(c - 1)
T = n['times'][toffs[l]:toffs[l] + 6 * c]
H = n['hopKm'][hoffs[l]:hoffs[l] + 2 * c]
stops = [n['stops'][offs[l] + i] for i in idx]
times = []
for d in range(2):
    for lev in range(3):
        b = (3 * d + lev) * c
        v = T[b:b + c]
        if d == 0:
            seg = [sum(v[idx[j]:idx[j + 1]]) * speed for j in range(len(idx) - 1)] + [0]
        else:
            seg = [0] + [sum(v[idx[j] + 1:idx[j + 1] + 1]) * speed for j in range(len(idx) - 1)]
        times.extend(seg)
h0 = H[:c]
hop0 = [sum(h0[idx[j]:idx[j + 1]]) for j in range(len(idx) - 1)] + [0]
hop1 = list(reversed(hop0[:-1])) + [0]
n['lineN'].append(len(stops)); n['stops'].extend(stops); n['times'].extend(times)
n['tph'].extend([tph, tph * 0.75, tph * 0.5]); n['cars'].append(12); n['hopKm'].extend(hop0 + hop1)
json.dump(n, open(sys.argv[6], 'w'))
print(f'over line {l} ({c} stops, {km(l):.0f} km): {len(stops)} stops, {sum(times[:len(stops)]) / 60:.0f} min end to end')
