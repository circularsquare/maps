"""T-1650: the market's riders on each line (native, one period, free flow) against the line's riders
a day in the browser's full solve (jp3-on.json lineFig), on the same network.

python lines.py <dump.json> <browser measure.json> [period share]
"""
import json, sys, collections
d = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
share = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
mb = collections.Counter()
for l, x in zip(d['rn_line'], d['board']):
    mb[l] += x
lf = b['lineFig']
tot = sum(v[0] for v in lf.values())
mk = sum(mb.values()) / share
print(f'lines {len(lf)}, riders a day {tot:.0f}; market boardings a day (period / {share}) {mk:.0f} = {100 * mk / tot:.2f}%')
rows = []
for k, v in lf.items():
    l = int(k)
    m = mb[l] / share
    rows.append((v[0], m, l))
rows.sort(reverse=True)
bands = [(0.5, 1.01), (0.2, 0.5), (0.1, 0.2), (0.05, 0.1), (0.0, 0.05)]
big = [r for r in rows if r[0] >= 10000]
print('lines over 10,000 riders a day:', len(big))
for lo, hi in bands:
    n = [r for r in big if lo <= min(1, r[1] / r[0]) < hi]
    print(f'  market share {lo:.2f}-{hi:.2f}: {len(n)} lines, {sum(r[0] for r in n):.0f} riders')
print('lines with the most market riders:')
for r in sorted(rows, key=lambda r: -r[1])[:15]:
    print(f'  line {r[2]:4d}: riders {r[0]:9.0f}, market {r[1]:8.0f} ({100 * r[1] / max(1, r[0]):.0f}%)')
json.dump({'market_board_period': {str(k): v for k, v in mb.items()}}, open(sys.argv[1].replace('.dump.json', '.lines.json'), 'w'))
