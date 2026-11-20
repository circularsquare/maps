"""inrange.py <dir A> <dir B> [prefix]: T-3388's second tolerance. Each line-level's trains an hour in
every B dump against the range A took for that line, slot and trains in any dump of its runs (every
step, every edited line): B outside it by more than 0.1%, 0.5%, 1%, and the largest."""
import collections, glob, os, re, sys

NUM = re.compile(r'-?\d+(?:\.\d+)?(?:e-?\d+)?')
pre = sys.argv[3] if len(sys.argv) > 3 else 'lon'


def rows(path):
    for ln in open(path):
        parts = ln.rstrip('\n').split('\t')
        if len(parts) >= 5:
            tph, rt, tr, pace = [[float(x) for x in NUM.findall(q)] for q in parts[1:5]]
            yield int(parts[0]), tph, rt, tr


def files(d):
    return [p for p in glob.glob(os.path.join(d, f'{pre}-*.txt'))]


rng = collections.defaultdict(lambda: [float('inf'), float('-inf')])
for p in files(sys.argv[1]):
    for l, tph, rt, tr in rows(p):
        for s in range(len(tph)):
            k = (l, s, tuple(tr))
            r = rng[k]
            r[0] = min(r[0], tph[s])
            r[1] = max(r[1], tph[s])
n = 0
outs = []
missing = 0
for p in files(sys.argv[2]):
    if not os.path.exists(os.path.join(sys.argv[1], os.path.basename(p))):
        continue
    for l, tph, rt, tr in rows(p):
        for s in range(len(tph)):
            if tph[s] <= 0:
                continue
            n += 1
            k = (l, s, tuple(tr))
            if k not in rng:
                missing += 1
                continue
            lo, hi = rng[k]
            ex = max(lo - tph[s], tph[s] - hi, 0.0) / max(tph[s], 1e-9)
            if ex > 0:
                outs.append((ex, os.path.basename(p), l, s, tph[s], lo, hi))
outs.sort(reverse=True)
print(f'{n} line-levels; at trains A never had: {missing}; outside the range A took: >0.1% {sum(o[0] > 1e-3 for o in outs)}, >0.5% {sum(o[0] > 5e-3 for o in outs)}, >1% {sum(o[0] > 1e-2 for o in outs)}')
for o in outs[:12]:
    print(f'  {o[0]*100:.3f}%  {o[1]} line {o[2]} slot {o[3]}: {o[4]:.4f} against {o[5]:.4f}-{o[6]:.4f}')
