"""T-087: per tile, the factor A where share_A makes the ratio of summed times 1 (interpolated), against
the tile's mean road straightness; prints a table and writes the plot.
    python isochrone/notes/T-087/bestfactor.py <cap.json> <uni.json> <tiles dir> <plot.png>
"""
import json
import os
import sys

import numpy as np

cap = {t["name"]: t for t in json.load(open(sys.argv[1]))}
uni = {t["name"]: t for t in json.load(open(sys.argv[2]))}
tiles = sys.argv[3]
A = [2, 3, 4, 5, 6, 8]


def best(t):
    r = np.array([t["rules"]["share_%g" % a]["ratio"] for a in A])
    if r[0] >= 1:
        return 2.0     # already too slow-truth at 2 (L1 too fast): at most 2
    if r[-1] <= 1:
        return float("nan")    # not reached by 8
    i = np.nonzero(r >= 1)[0][0]
    return float(A[i - 1] + (1 - r[i - 1]) / (r[i] - r[i - 1]) * (A[i] - A[i - 1]))


rows = []
for name in sorted(cap):
    f = np.load(os.path.join(tiles, name, "feat.npz"))
    m = (f["walk"] > 0) & (f["s"] > 0)
    st = float((f["span"][m] / np.maximum(1, 100 * f["s"][m])).mean())
    land = f["walk"] > 0
    rows.append((name, float(f["s"][land].mean()), st, best(cap[name]), best(uni[name])))
print("%-9s %6s %9s %9s %9s" % ("tile", "s", "straight", "A (all)", "A (roads)"))
for r in rows:
    print("%-9s %6.3f %9.2f %9.1f %9.1f" % r)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, s, st, a1, a2 in rows:
    if name in ("n10e105",):
        continue
    ax.scatter(st, a2, c="C0")
    ax.scatter(st, a1, c="C1", marker="x")
    ax.annotate(name, (st, a2), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.axhline(3, c="grey", lw=0.8)
ax.set_xlabel("road straightness (longest run across the cell / road cells in it)")
ax.set_ylabel("factor A where share_A matches the 100 m time")
ax.scatter([], [], c="C0", label="roads only (mosaic removed from the 100 m side)")
ax.scatter([], [], c="C1", marker="x", label="against the full 100 m land cover")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(sys.argv[4], dpi=130)
