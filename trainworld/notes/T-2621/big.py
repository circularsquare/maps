"""T-2621: per mode, the worst change of a line's riders (every population, a day) by line size,
the market's change taken to a day by `k` (the day's market rail over the period's).

python big.py <dump.json> <lines.json> <k>"""
import json, sys

d = json.load(open(sys.argv[1]))
tot = json.load(open(sys.argv[2]))['lineBoard']
k = float(sys.argv[3])
ex = d['modes'][0]['periods'][0]['line_all']
for m in d['modes'][1:]:
    v = m['periods'][0]['line_all']
    row = []
    for lo, hi in ((1000, 10000), (10000, 100000), (100000, 1e12)):
        e = [(abs(b - a) * k / t, abs(b - a) * k) for a, b, t in zip(ex, v, tot) if lo <= t < hi]
        w = max(e) if e else (0, 0)
        row.append(f'{lo:>6.0f}+: {len(e):3d} lines, worst {100*w[0]:.2f}% ({w[1]:.0f} riders)')
    print(f"{m['mode'][:44]:44s} " + ' | '.join(row))
