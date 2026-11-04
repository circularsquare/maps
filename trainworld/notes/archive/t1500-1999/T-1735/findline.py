"""findline.py <plain net> <km>: the lines nearest km long (by hop km), with their stop counts."""
import json, sys
n = json.load(open(sys.argv[1]))
t = float(sys.argv[2])
offs, h = [], 0
for c in n['lineN']:
    offs.append(h)
    h += 2 * c
rows = []
for l, c in enumerate(n['lineN']):
    rows.append((abs(sum(n['hopKm'][offs[l]:offs[l] + c]) - t), l, c, sum(n['hopKm'][offs[l]:offs[l] + c])))
rows.sort()
for r in rows[:8]:
    print('line', r[1], 'stops', r[2], 'km %.0f' % r[3])
