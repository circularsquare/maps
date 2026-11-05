"""T-042 sample plots: the L10 land hazard, natural water and with OSM water points, and which month.

    python isochrone/notes/T-042/plots.py      -> maps/artifacts/isochrone/t042_*.png
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "artifacts", "isochrone")
W, H = 4320, 1740
grid = np.frombuffer(__import__("gzip").open(os.path.join(HERE, "..", "..", "data", "grid.bin.gz")).read(), np.uint8).reshape(H, W)
ld = lambda k: np.fromfile(os.path.join(HERE, "..", "..", "data", "raw", "t042", "l10_%s.f32" % k), np.float32).reshape(H, W)
ext = [-180, 180, -60, 85]


def world(a, name, title, box=None):
    a = np.where(grid > 0, a, np.nan)
    m = np.where(a > 0, a, np.nan)
    fig, ax = plt.subplots(figsize=(18, 8) if box is None else (12, 9), dpi=110)
    ax.set_facecolor("#20242a")
    land0 = np.where((grid > 0) & ~(a > 0), 1.0, np.nan)
    ax.imshow(land0, extent=ext, cmap="Greys", vmin=0, vmax=3, interpolation="nearest")
    im = ax.imshow(m, extent=ext, cmap="magma", norm=LogNorm(1e-6, 1), interpolation="nearest")
    if box:
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
    fig.colorbar(im, ax=ax, shrink=0.7, label="hazard per walking hour")
    ax.set_title(title)
    fig.savefig(os.path.join(ART, "t042_%s.png" % name), bbox_inches="tight")
    plt.close(fig)


os.makedirs(ART, exist_ok=True)
world(ld("nat"), "world_nat", "land hazard, rivers and lakes only, kinder of Jan/Jul (grey: 0)")
world(ld("osm"), "world_osm", "land hazard with OSM wells, springs, settlements, kinder of Jan/Jul (grey: 0)")
world(ld("osmjul"), "world_osmjul", "land hazard with OSM water, July everywhere (grey: 0)")
world(ld("nat"), "sahara_nat", "Sahara and Arabia, rivers and lakes only", box=(-18, 60, 8, 38))
world(ld("osm"), "sahara_osm", "Sahara and Arabia, with OSM water points", box=(-18, 60, 8, 38))
world(ld("nat"), "greenland_nat", "Greenland and the Canadian Arctic, rivers and lakes only", box=(-100, -10, 58, 84))
world(ld("nat"), "asia_nat", "Central Asia and the Himalaya, rivers and lakes only", box=(45, 120, 20, 55))
j = np.where(grid > 0, ld("july"), np.nan)
fig, ax = plt.subplots(figsize=(18, 8), dpi=90)
ax.imshow(j, extent=ext, cmap="coolwarm", vmin=0, vmax=1, interpolation="nearest")
ax.set_title("kinder month: red July, blue January (time-weighted share, natural water)")
fig.savefig(os.path.join(ART, "t042_month.png"), bbox_inches="tight")
print("written to", ART)
