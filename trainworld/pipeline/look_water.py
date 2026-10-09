"""Look-image of a city pack's water mask (T-030): the whole extent and close-ups of the places
that must come out right, with the city boundary.

    python pipeline/look_water.py nyc [out.png]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from read_pack import PACKS, R_EARTH, ROOT, load_pack, water_mask
from look_pack import boundary_xy

# (title, lon, lat, half-width km) of close-ups
VIEWS = {
    "nyc": [
        ("Manhattan, East River, Governors Island", -73.975, 40.745, 7.5),
        ("Harlem River, Randalls Island, Rikers", -73.915, 40.815, 4.5),
        ("Jamaica Bay and JFK", -73.83, 40.62, 9.0),
        ("Newark Bay, Kill van Kull, Arthur Kill", -74.14, 40.64, 7.5),
    ],
}


def dense(m):
    """Decode row runs to a boolean raster (H, W), row 0 south."""
    off, xs = m["row"].astype(np.int64), m["x"].astype(np.int64)
    W, H = m["size"]
    d = np.zeros((H, W + 1), np.int8)
    r = np.repeat(np.arange(H), np.diff(off) // 2)
    np.add.at(d, (r, xs[0::2]), 1)
    np.add.at(d, (r, xs[1::2]), -1)
    return np.cumsum(d, axis=1, dtype=np.int8)[:, :W] > 0


def main(city, out):
    header, a = load_pack(city)
    m = water_mask(header, a)
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    k = math.cos(math.radians(lat0))
    w = dense(m)
    x0, y0 = m["origin_m"]
    cell = m["cell_m"]
    W, H = m["size"]
    views = VIEWS.get(city, [])
    fig = plt.figure(figsize=(18, 12), dpi=130, facecolor="white")
    gs = fig.add_gridspec(2, 4)
    cmap = ListedColormap(["#f3efe6", "#7fa7d9"])
    ax = fig.add_subplot(gs[:, :2])
    f = 8  # overview at 8 x the pixel size
    small = w[: H // f * f, : W // f * f].reshape(H // f, f, W // f, f).mean(axis=(1, 3))
    ext = [x0 / 1e3, (x0 + W // f * f * cell) / 1e3, y0 / 1e3, (y0 + H // f * f * cell) / 1e3]
    ax.imshow(small, origin="lower", extent=ext, cmap="Blues", vmin=-0.3, vmax=1.2, interpolation="nearest")
    for bx, by in boundary_xy(city, lon0, lat0):
        ax.plot(bx, by, color="#c0392b", lw=0.5)
    ax.set_title(f"{city} water mask: {W} x {H} pixels of {cell:.0f} m, {m['runs']:,} runs, "
                 f"{m['bytes'] / 1e6:.2f} MB; water {m['water_share']:.1%} of the extent", fontsize=10)
    ax.set_xlabel("km east of the pack origin", fontsize=8)
    ax.tick_params(labelsize=7)
    for i, (title, lo, la, hw) in enumerate(views[:4]):
        axz = fig.add_subplot(gs[i // 2, 2 + i % 2])
        cx = R_EARTH * math.radians(lo - lon0) * k
        cy = R_EARTH * math.radians(la - lat0)
        c0 = max(0, int((cx - hw * 1e3 - x0) / cell)); c1 = min(W, int((cx + hw * 1e3 - x0) / cell))
        r0 = max(0, int((cy - hw * 1e3 - y0) / cell)); r1 = min(H, int((cy + hw * 1e3 - y0) / cell))
        axz.imshow(w[r0:r1, c0:c1], origin="lower", cmap=cmap, interpolation="nearest",
                   extent=[(x0 + c0 * cell) / 1e3, (x0 + c1 * cell) / 1e3, (y0 + r0 * cell) / 1e3, (y0 + r1 * cell) / 1e3])
        axz.set_title(title, fontsize=9)
        axz.tick_params(labelsize=6)
        rect = plt.Rectangle(((x0 + c0 * cell) / 1e3, (y0 + r0 * cell) / 1e3), (c1 - c0) * cell / 1e3,
                             (r1 - r0) * cell / 1e3, fill=False, color="#333333", lw=0.6)
        ax.add_patch(rect)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else "nyc"
    main(city, sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "T-030.png"))
