"""T-854: New York lines before (note's nyc_real) and in runs tagged argv[1..]: riders a day."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda p: {l['name']: l['riders'] for l in json.load(open(p, encoding='utf-8'))['lines']}
base = L('C:/Users/anita/projects/maps/trainworld/notes/T-854/nyc_real.json')
zero = L('C:/Users/anita/projects/maps/trainworld/notes/T-854/nyc_zero.json')
runs = {t: L(os.path.join(HERE, f'nyc_{t}.json')) for t in sys.argv[1:]}
groups = {}
for n in base:
    g = 'commuter' if n.split()[0] in ('lirr', 'mnr', 'njt') else 'subway/PATH'
    groups.setdefault(g, []).append(n)
for g, ns in groups.items():
    print(g, len(ns), 'before', round(sum(base[n] for n in ns) / 1e6, 2), 'zero', round(sum(zero.get(n, 0) for n in ns) / 1e6, 2),
          *[f'{t} {sum(r.get(n, 0) for n in ns) / 1e6:.2f}' for t, r in runs.items()])
print(sorted(base)[:60])
