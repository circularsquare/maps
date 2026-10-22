"""T-854: tables from the ceiling harness outputs.  python table.py <tag> [vars,...]"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1]
want = sys.argv[2].split(',') if len(sys.argv) > 2 else None
packs = ['nyc', 'lon', 'par', 'tyo', 'hou']
data = {}
for p in packs:
    f = os.path.join(HERE, f'ceil_{tag}_{p}.json')
    if os.path.exists(f):
        data[p] = json.load(open(f))
d0 = next(iter(data.values()))
vars_, scens, NG = d0['vars'], d0['scens'], len(d0['groups'])


def rr(d, vi, s, g):
    a = d['acc'][(vi * len(scens) + s) * NG + g]
    return 100 * a[1] / (a[1] + a[3]) if a[1] + a[3] > 0 else float('nan')


vl = [v for v in vars_ if want is None or v in want]
print('ceilings, rail of rail + car, all homes (A / B)')
print('var'.ljust(10) + ''.join(f'{p:>14}' for p in data))
for v in vl:
    vi = vars_.index(v)
    print(v.ljust(10) + ''.join(f'{rr(d, vi, 0, 4):7.1f}{rr(d, vi, 1, 4):7.1f}' for d in data.values()))
for sname in ['T', 'S']:
    s = scens.index(sname)
    print(f'\n{sname} by home density <3k 3-10k 10-20k >20k all')
    for p, d in data.items():
        for v in vl:
            vi = vars_.index(v)
            print(f'{p} {v:10}' + ''.join(f'{rr(d, vi, s, g):6.1f}' for g in range(NG)))
print('\nfixed ratios rail / car, all homes: R1.2 R1.5 R2 R3; and the suburbs (<3k) R1.5 R2')
for p, d in data.items():
    for v in vl:
        vi = vars_.index(v)
        print(f'{p} {v:10}' + ''.join(f'{rr(d, vi, scens.index(x), 4):6.1f}' for x in ['R1.2', 'R1.5', 'R2', 'R3']) + '   ' + ''.join(f'{rr(d, vi, scens.index(x), 0):6.1f}' for x in ['R1.5', 'R2']))
