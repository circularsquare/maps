"""surr.py's surrogates with asc_rail held at 0: the joint fit, leave-one-out, hold-out."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surr
import fit

surr.lo[0] = surr.hi[0] = 0.0
surr.lo[0] -= 1e-9
eq = {c: 1.0 for c in fit.FIT}
print("--- asc_rail held at 0")
surr.show("joint", surr.best(eq))
for c in fit.FIT:
    surr.show(f"without {c}", surr.best({k: 1.0 for k in fit.FIT if k != c}))
for c in fit.ALL:
    surr.show(f"{c} alone (no penalty)", surr.best({c: 1.0}, pen=False))
for x in ([0, -1.25, 1.9], [0, -1.0, 1.9], [0, -1.25, 1.7]):
    surr.show(f"point {x}", np.array(x, float))
