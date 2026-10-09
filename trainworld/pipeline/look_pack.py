"""Quick look-image of a city pack: population and jobs per cell, side by side.

    python pipeline/look_pack.py nyc [out.png]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np

from read_pack import PACKS, R_EARTH, ROOT, load_pack


def boundary_xy(city, lon0, lat0):
    with open(os.path.join(PACKS, f"{city}.boundary.geojson")) as f:
        g = json.load(f)["features"][0]["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    k = math.cos(math.radians(lat0))
    for poly in polys:
        for ring in poly:
            r = np.array(ring)
            yield (R_EARTH * np.radians(r[:, 0] - lon0) * k / 1e3, R_EARTH * np.radians(r[:, 1] - lat0) / 1e3)


def main(city, out):
    header, a = load_pack(city)
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    x, y = a["x_m"] / 1e3, a["y_m"] / 1e3
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.6), dpi=150, facecolor="white")
    panels = (("pop", "Residents per cell (2020 Census)", "Blues"),
              ("jobs", "Jobs per cell (LODES WAC)", "Purples"))
    for ax, (k, title, cmap) in zip(axes, panels):
        v = a[k].astype(np.float64)
        m = v > 0
        order = np.argsort(v[m])  # big cells drawn last
        base = plt.get_cmap(cmap)
        ramp = LinearSegmentedColormap.from_list(cmap, base(np.linspace(0.3, 1.0, 256)))  # no white end
        sc = ax.scatter(x[m][order], y[m][order], c=v[m][order], s=0.6, marker="h", linewidths=0,
                        cmap=ramp, norm=LogNorm(vmin=10, vmax=max(10.0, float(v.max()))), rasterized=True)
        for bx, by in boundary_xy(city, lon0, lat0):
            ax.plot(bx, by, color="#999999", lw=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{title}, {int(v.sum()):,} total", fontsize=11, color="#333333")
        ax.tick_params(labelsize=8, colors="#666666")
        ax.set_xlabel("km east of Midtown", fontsize=9, color="#666666")
        for s in ax.spines.values():
            s.set_color("#cccccc")
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
        cb.ax.tick_params(labelsize=8)
    axes[0].set_ylabel("km north of Midtown", fontsize=9, color="#666666")
    fig.suptitle(f"{city} pack: {header['cells']:,} H3 res-{header['h3_res']} cells", fontsize=12, color="#333333")
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else "nyc"
    main(city, sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "T-004.png"))
