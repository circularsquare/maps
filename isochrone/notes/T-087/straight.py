"""T-087: tile means of straightness = longest run across a cell / road cells in it, over cells with a road
and over cells with one component.
    python isochrone/notes/T-087/straight.py <tiles dir>
"""
import os
import sys

import numpy as np

d = sys.argv[1]
for name in sorted(os.listdir(d)):
    p = os.path.join(d, name, "feat.npz")
    if not os.path.exists(p):
        continue
    f = np.load(p)
    m = (f["walk"] > 0) & (f["s"] > 0)
    st = f["span"][m] / np.maximum(1, 100 * f["s"][m])
    one = m & (f["ncomp"] == 1)
    st1 = f["span"][one] / np.maximum(1, 100 * f["s"][one])
    print("%-9s straight %.2f (p25 %.2f)  one-component cells %.2f  straight there %.2f  cells with 1 comp & s<=.15: %.2f" % (
        name, st.mean(), np.percentile(st, 25), one.sum() / m.sum(), st1.mean(), (one & (f["s"] <= 0.15)).sum() / m.sum()))
