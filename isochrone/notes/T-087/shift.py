"""T-087: how far a factor change moves the 1 km map: per tile, the mean time per km over land cells
(share_A) against share_3, and the share of land cells whose speed changes by over 10%.
    python isochrone/notes/T-087/shift.py <tiles dir> rule [rule ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules as R

d = sys.argv[1]
fac = sys.argv[2:]
print("%-9s %6s " % ("tile", "km/h3") + " ".join("%-18s" % (a) for a in fac))
for name in sorted(os.listdir(d)):
    p = os.path.join(d, name, "feat.npz")
    if not os.path.exists(p):
        continue
    f = dict(np.load(p))
    land = f["walk"] > 0
    v3 = R.speed("share_3", f)
    t3 = (1 / v3[land]).mean()
    row = "%-9s %6.2f " % (name, 1 / t3)
    for a in fac:
        v = R.speed(a, f)
        t = (1 / v[land]).mean()
        ch = np.abs(v[land] / v3[land] - 1) > 0.10
        row += "%-18s " % ("%+.0f%% time, %2.0f%% >10%%" % (100 * (t / t3 - 1), 100 * ch.mean()))
    print(row)
