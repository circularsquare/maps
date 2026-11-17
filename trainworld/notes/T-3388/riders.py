"""T-3388: two trace.mjs runs: per edit the time to the network reaching demand and to the final, and
riders per line at each final (A before, B after).

    python riders.py <A.json> <B.json>
"""
import json
import sys

A, B = (json.load(open(p)) for p in sys.argv[1:3])
print(f"load final: A {A.get('final')} s, B {B.get('final')} s")


def cmp(fa, fb, what):
    la, lb = fa['lines'], fb['lines']
    rel = []
    for k in la:
        if k in lb and la[k]['r'] > 0:
            rel.append((abs(lb[k]['r'] - la[k]['r']) / la[k]['r'], k, la[k]['r'], lb[k]['r']))
    rel.sort(reverse=True)
    n = len(rel)
    print(f"{what}: rail trips A {fa['rail']:.0f} B {fb['rail']:.0f} ({(fb['rail'] - fa['rail']) / fa['rail'] * 100:+.3f}%), "
          f"lines {n}: riders differ >0.5% {sum(r[0] > 0.005 for r in rel)}, >1% {sum(r[0] > 0.01 for r in rel)}, >3% {sum(r[0] > 0.03 for r in rel)}")
    for r in rel[:5]:
        print(f"    line {r[1]}: {r[2]:.0f} -> {r[3]:.0f} ({r[0] * 100:.1f}%)")


cmp(A['fig0'], B['fig0'], 'load')
for ea, eb in zip(A['edits'], B['edits']):
    sa, sb = ea['solve'], eb['solve']
    print(f"{ea['kind']}: network at demand A {sa['arrivedAt'] / 1000:.2f} s B {sb['arrivedAt'] / 1000:.2f} s; "
          f"edit call answered A {ea['tAns']:.2f} B {eb['tAns']:.2f} s; final A {ea['settle']:.1f} B {eb['settle']:.1f} s")
    cmp(ea['fig'], eb['fig'], '  ' + ea['kind'])
