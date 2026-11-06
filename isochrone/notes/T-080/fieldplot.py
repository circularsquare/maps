"""Three panels of walking time bands: L1, L10 now, L10 from crossing times."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

SCR = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\anita\projects\maps\artifacts\isochrone"
tag, title = sys.argv[1], sys.argv[2]
m = json.load(open(os.path.join(SCR, "fields", tag + ".json")))
EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128]
cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(EDGES) - 1)))
cmap.set_over("#eeeeee")
cmap.set_bad("#9ecae1")
norm = BoundaryNorm(EDGES, cmap.N)
b = m["box"]
ext = [b["west"], b["east"], b["south"], b["north"]]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
for ax, (name, key, lab) in zip(axes, [("l1", "l1", "L1 (1 km)"), ("l10", "l10", "L10 now"), ("l10x", "l10", "L10 from crossing times")]):
    w, h = m[key]
    a = np.fromfile(os.path.join(SCR, "fields", "%s_%s.bin" % (tag, name)), dtype=np.float32).reshape(h, w)
    a = np.where(np.isinf(a), np.nan, a)
    im = ax.imshow(a, extent=ext, cmap=cmap, norm=norm, interpolation="nearest", aspect=1 / np.cos(np.radians(m["lat"])))
    ax.plot(m["lng"], m["lat"], "r*", ms=10)
    ax.set_title(lab)
fig.colorbar(im, ax=axes, shrink=0.8, label="hours walking (hazard off)", extend="max")
fig.suptitle(title)
fig.savefig(os.path.join(OUT, "t080_%s.png" % tag), dpi=100, bbox_inches="tight")
print("wrote t080_%s.png" % tag)
