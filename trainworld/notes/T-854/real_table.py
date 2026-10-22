"""T-854: the real networks' own rail trips a day against the note's runs.  python real_table.py tag[,tag]"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NOTE = 'C:/Users/anita/projects/maps/trainworld/notes/T-854'
tags = sys.argv[1].split(',')


def load(path):
    return json.load(open(path, encoding='utf-8')) if os.path.exists(path) else None


for c in ['nyc', 'lon', 'par', 'tyo']:
    base = load(f'{NOTE}/{c}_real.json')
    zero = load(f'{NOTE}/{c}_zero.json')
    runs = [load(os.path.join(HERE, f'{c}_{t}.json')) for t in tags]
    own = lambda r: r['cities'][c]['rail'] if r and c in r['cities'] else float('nan')
    b = own(base)
    row = f'{c}: before {b / 1e6:.2f}M, nobody car-free {own(zero) / 1e6:.2f}M'
    for t, r in zip(tags, runs):
        row += f', {t} {own(r) / 1e6:.2f}M ({100 * (own(r) / b - 1):+.1f}%)'
    print(row)
    if c == 'nyc' and base:
        lines = {l['name']: l['riders'] for l in base['lines']}
        for t, r in zip(tags, runs):
            if not r:
                continue
            now = {l['name']: l['riders'] for l in r['lines']}
            # commuter rail: lines whose names are LIRR / Metro-North / NJ Transit branches
            sub = [n for n in lines if len(n) <= 2 or n in ('PATH',)]
            print(f'  {t}: lines with 1-2 letter names (subway) {sum(now.get(n, 0) for n in sub) / 1e6:.2f}M vs {sum(lines[n] for n in sub) / 1e6:.2f}M;'
                  f' the rest {sum(v for n, v in now.items() if n not in sub) / 1e6:.2f}M vs {sum(v for n, v in lines.items() if n not in sub) / 1e6:.2f}M')
