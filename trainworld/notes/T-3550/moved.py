"""moved.py <dir>...: per run, lines whose round trips changed at all (bitwise) from the load to +1 medium, and
from +1 medium to its undo: the lines whose boarding accounts an edit makes stale."""
import glob, os, sys


def dump(p):
    out = {}
    for ln in open(p):
        parts = ln.rstrip('\n').split('\t')
        if len(parts) >= 5:
            out[parts[0]] = parts[2]
    return out


for d in sys.argv[1:]:
    tot = [0, 0, 0]
    for p in glob.glob(os.path.join(d, 'lon-*.load.txt')):
        pfx = p[:-len('.load.txt')]
        try:
            a, b, c = dump(p), dump(pfx + '.count.txt'), dump(pfx + '.count_undo.txt')
        except FileNotFoundError:
            continue
        tot[0] += 1
        tot[1] += sum(a[k] != b.get(k) for k in a)
        tot[2] += sum(b[k] != c.get(k) for k in b)
    print(f'{d}: {tot[0]} runs; lines moved by +1 medium {tot[1] / max(tot[0], 1):.1f} a run, by its undo {tot[2] / max(tot[0], 1):.1f}')
