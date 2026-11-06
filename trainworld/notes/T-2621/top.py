"""T-2621: the lines with the most market boardings, exact against each mode, and every line's change
as a share of its riders of every population (a t756 lines file).

python top.py <dump.json> <lines.json> [n]"""
import json, sys

d = json.load(open(sys.argv[1]))
tot = json.load(open(sys.argv[2]))['lineBoard']
n = int(sys.argv[3]) if len(sys.argv) > 3 else 12


def lines(m):
    out = None
    for p in m['periods']:
        out = list(p['line_all']) if out is None else [a + b for a, b in zip(out, p['line_all'])]
    return out


ex = lines(d['modes'][0])
ms = [(m['mode'], lines(m)) for m in d['modes'][1:]]
order = sorted(range(len(ex)), key=lambda l: -ex[l])[:n]
print('line  market  all riders  ' + '  '.join(f'[{i}]' for i in range(len(ms))))
for i, (name, _) in enumerate(ms):
    print(f'  [{i}] {name}')
for l in order:
    print(f'{l:4d} {ex[l]:8.0f} {tot[l]:10.0f}  ' + '  '.join(f'{100*(v[l]-ex[l])/ex[l]:+5.1f}% ({100*(v[l]-ex[l])/tot[l]:+.2f}%)' for _, v in ms))
