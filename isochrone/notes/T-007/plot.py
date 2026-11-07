"""T-007: where L10 cells join land L1 keeps apart (count.py's l10_<tile>.npy, roads on).

    python isochrone/notes/T-007/plot.py <count dir> <out dir>

Writes t007_l10_world.png (one pixel an L10 cell) and t007_l10_crops.png (Sogndal, Coari,
Khulna, Jyvaskyla, 4 x 4 degrees, with a legend). Colour by the kind of water inside the cell:
sea inlet, strait, river, lake; cells whose only fault is a border with no land across it,
magenta. Land grey, water dark.
"""
import glob
import gzip
import os
import sys

import numpy as np

cnt, out = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
grid = np.frombuffer(gzip.open(os.path.join(HERE, "data", "grid.bin.gz")).read(), np.uint8).reshape(1740, 4320)
code = np.zeros((1740, 4320), np.uint8)
for fn in glob.glob(os.path.join(cnt, "l10_*.npy")):
    tid = int(os.path.basename(fn)[4:-4])
    tr, tc = divmod(tid, 72)
    code[tr * 60:(tr + 1) * 60, tc * 60:(tc + 1) * 60] = np.load(fn)
COL = {"water": (26, 26, 25), "land": (64, 64, 61), 1: (57, 135, 229), 2: (217, 89, 38), 3: (25, 158, 112),
       4: (201, 133, 0), "edge": (213, 81, 129)}
img = np.zeros((1740, 4320, 3), np.uint8)
img[:] = COL["water"]
img[grid > 0] = COL["land"]
k = code & 7
for v in (1, 2, 3, 4):
    img[k == v] = COL[v]
img[(k == 0) & (code & 16 > 0)] = COL["edge"]
from PIL import Image          # noqa: E402
Image.fromarray(img).save(os.path.join(out, "t007_l10_world.png"))

import matplotlib            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.patches import Patch   # noqa: E402
crops = [("Sogndal", 61.23, 7.10), ("Coari", -4.08, -63.14), ("Khulna", 22.82, 89.55), ("Jyvaskyla", 62.24, 25.75)]
fig, axs = plt.subplots(1, 4, figsize=(16, 4.8), facecolor="#1a1a19")
for ax, (name, lat, lng) in zip(axs, crops):
    c0, c1 = int((lng - 2 / np.cos(np.radians(lat)) + 180) * 12), int((lng + 2 / np.cos(np.radians(lat)) + 180) * 12)
    ax.imshow(img[int((85 - lat) * 12) - 24:int((85 - lat) * 12) + 24, c0:c1], interpolation="nearest",
              aspect=1 / np.cos(np.radians(lat)))
    ax.set_title(name, color="#f0efec", fontsize=11)
    ax.axis("off")
fig.legend(handles=[Patch(color=np.array(COL[v]) / 255, label=l) for v, l in
                    ((1, "split by a sea inlet or fjord"), (2, "split by a strait"), (3, "split by a river"), (4, "split by a lake"),
                     ("edge", "border with no land across"), ("land", "land, as L1"))],
           loc="lower center", ncol=6, frameon=False, labelcolor="#f0efec", fontsize=9)
fig.text(0.01, 0.95, "L10 cells (1/12 degree) whose land L1 keeps apart, roads on", color="#c3c2b7", fontsize=10)
plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.12, wspace=0.03)
fig.savefig(os.path.join(out, "t007_l10_crops.png"), dpi=110, facecolor=fig.get_facecolor())
print("written")
