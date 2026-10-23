"""T-1003: commute and everyday trip lengths, and walking under the T test, per pack (base)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ana import load, CN, OTHER, reg, tot

print("| Zone | Commute km | Commute under 2 km | Everyday km | Everyday under 2 km | Walk base -> under T | Rail of all under T |")
print("|---|---|---|---|---|---|---|")
for c in CN + OTHER:
    r = load(c, 'base')
    if r is None:
        continue
    a = r[1]
    out = []
    for layer in (0, 1):
        b = a[layer][:, 5, :]
        out.append((b[:, 2].sum() / b[:, 0].sum(), b[:2, 0].sum() / b[:, 0].sum()))
    t = tot(a)
    print(f"| {reg[c]['name']} | {out[0][0]:.1f} | {100*out[0][1]:.0f} | {out[1][0]:.1f} | {100*out[1][1]:.0f} | {100*t[1]/t[0]:.0f} -> {100*t[16]/t[0]:.0f} | {100*t[15]/t[0]:.0f} |")
