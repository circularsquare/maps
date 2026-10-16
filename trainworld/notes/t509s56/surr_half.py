"""surr.py's surrogates with New York at half weight (Anita: a moderate New York overshoot is fine):
joint, leave-one-out, and the rounded point."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surr
import fit

w = {c: 1.0 for c in fit.FIT}
w["nyc"] = 0.5
print("--- New York at half weight")
surr.show("joint", surr.best(w))
for c in fit.FIT:
    surr.show(f"without {c}", surr.best({k: v for k, v in w.items() if k != c}))
for x in ([0.4, -1.1, 1.9], [0.3, -1.25, 1.9]):
    xx = np.array(x, float)
    surr.show(f"point {x}", xx)
    print("   objective", round(surr.obj(xx, w), 3))
