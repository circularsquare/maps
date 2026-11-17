"""T-3388 checks on probe dumps: paced line-levels per dump, the largest pace over own headway, and
B outliers against the range each line-level takes over every A dump and pace round at the same trains.

    python paced.py <dir A> <dir B>
"""
import glob
import os
import re
import sys

NUM = re.compile(r'-?\d+(?:\.\d+)?(?:e-?\d+)?')


def dump(p):
    out = {}
    for ln in open(p):
        parts = ln.rstrip('\n').split('\t')
        if len(parts) < 5:
            continue
        tph, rt, tr, pace = [[float(x) for x in NUM.findall(q)] for q in parts[1:5]]
        out[int(parts[0])] = (tph, rt, tr, pace)
    return out


def ph(p):
    out = {}
    for ln in open(p):
        parts = ln.rstrip('\n').split('\t')
        out[int(parts[0])] = [float(x) for x in NUM.findall(parts[1])]
    return out


a, b = sys.argv[1], sys.argv[2]
for d in (a, b):
    print('==', d)
    worst = (0, None)
    counts = {}
    for p in sorted(glob.glob(os.path.join(d, 'lon-*.txt'))):
        if '.ph.' in p:
            continue
        D = dump(p)
        n = sum(1 for l, (t, r, c, pc) in D.items() for lev in range(3) if pc[lev] > 0)
        counts[os.path.basename(p)] = n
        for l, (t, r, c, pc) in D.items():
            for lev in range(3):
                if pc[lev] > 0 and c[lev] > 0 and r[lev] > 0:
                    x = pc[lev] / (r[lev] / c[lev]) - 1
                    if x > worst[0]:
                        worst = (x, f'{os.path.basename(p)} line {l} lev {lev} pace {pc[lev]:.1f} own {r[lev] / c[lev]:.1f}')
    vals = sorted(set(counts.values()))
    print('paced line-levels per dump: min', vals[0], 'max', vals[-1])
    for k in sorted(counts):
        if '-seq.' in k or k.endswith('.load.txt'):
            print(f'   {k}: {counts[k]}')
    print(f'largest pace over own headway: {worst[0] * 100:.2f}% ({worst[1]})')

# outliers of B against everything A visited at the same train counts
rng = {}
for p in glob.glob(os.path.join(a, 'lon-*.txt')):
    if '.ph.' in p:
        continue
    D = dump(p)
    for l, (t, r, c, pc) in D.items():
        for lev in range(3):
            k = (l, lev, c[lev])
            lo, hi = rng.get(k, (t[lev], t[lev]))
            rng[k] = (min(lo, t[lev]), max(hi, t[lev]))
# pace rounds: trains taken from the step's dump in the same run
for p in glob.glob(os.path.join(a, 'lon-*.ph.*.txt')):
    stepdump = None
    base = os.path.basename(p).split('.ph.')[0]
    cands = [q for q in glob.glob(os.path.join(a, base + '.*.txt')) if '.ph.' not in q and not q.endswith('.load.txt')]
    if not cands:
        continue
    D = dump(cands[0])
    P = ph(p)
    for l, t in P.items():
        if l not in D:
            continue
        c = D[l][2]
        for lev in range(3):
            k = (l, lev, c[lev])
            if k in rng:
                lo, hi = rng[k]
                rng[k] = (min(lo, t[lev]), max(hi, t[lev]))
ex = []
for p in glob.glob(os.path.join(b, 'lon-*.txt')):
    if '.ph.' in p or '-seq.' in p:
        continue
    D = dump(p)
    for l, (t, r, c, pc) in D.items():
        for lev in range(3):
            k = (l, lev, c[lev])
            if k not in rng or t[lev] <= 0:
                continue
            lo, hi = rng[k]
            e = max(lo - t[lev], t[lev] - hi, 0) / t[lev]
            ex.append((e, os.path.basename(p), l, lev, t[lev], lo, hi))
ex.sort(reverse=True)
print(f'B line-levels {len(ex)}: outside every value A took at the same trains by >0.1% {sum(e[0] > 0.001 for e in ex)}, '
      f'>0.5% {sum(e[0] > 0.005 for e in ex)}, >1% {sum(e[0] > 0.01 for e in ex)}')
for e in ex[:10]:
    print(f'   {e[0] * 100:.2f}% {e[1]} line {e[2]} lev {e[3]} B {e[4]:.4f} A range {e[5]:.4f}-{e[6]:.4f}')
