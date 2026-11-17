"""T-3388: before/after ms per step from matrix.ps1 summaries (A.txt B.txt Aseq.txt Bseq.txt)."""
import re
import sys

S = sys.argv[1]


def read(name):
    out = {}
    for ln in open(f'{S}/{name}', encoding='utf-16' if open(f'{S}/{name}', 'rb').read(2) == b'\xff\xfe' else 'utf-8'):
        m = re.match(r'(\w+) (\S+) \| (\w+)\s+([\d.]+) ms', ln.strip())
        if m:
            out[(m.group(1), m.group(2), m.group(3))] = float(m.group(4))
    return out


for a, b in (('A.txt', 'B.txt'), ('Aseq.txt', 'Bseq.txt')):
    A, B = read(a), read(b)
    print(f'== {a} vs {b}')
    for k in A:
        if k in B:
            print(f'{k[0]} {k[1]:>3} {k[2]:<14} {A[k]/1000:6.2f} s -> {B[k]/1000:6.2f} s')
