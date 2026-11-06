"""T-087: tile-level means of the 1 km road features (over land cells with a road).
    python isochrone/notes/T-087/featstats.py <tiles dir>
"""
import os
import sys

import numpy as np

d = sys.argv[1]
print("%-9s %5s %5s %5s %5s %5s %5s %5s %5s %5s" % ("tile", "s", "s|rd", "span", "cross", "thru", "ncomp", "path", "local", "main"))
for name in sorted(os.listdir(d)):
    p = os.path.join(d, name, "feat.npz")
    if not os.path.exists(p):
        continue
    f = np.load(p)
    land = f["walk"] > 0
    m = land & (f["s"] > 0)
    print("%-9s %5.3f %5.3f %5.2f %5.2f %5.2f %5.2f %5.3f %5.3f %5.3f" % (
        name, f["s"][land].mean(), f["s"][m].mean(), f["span"][m].mean(), f["cross"][m].mean(), f["through"][m].mean(),
        f["ncomp"][m].mean(), f["s1"][land].mean(), f["s2"][land].mean(), f["s4"][land].mean()))
