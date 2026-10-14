"""T-041: the OSM water mask on New York's own grid, against the shipped TIGER mask.

    python notes/t041_compare.py [out.png]

Builds the OSM mask (pipeline/osm_water.py) on exactly the nyc pack's water grid, writes it as a
water-only pack (data/work/osm_water/nyc_osm.json/.bin), runs read_pack's New York water test
places on it, and prints the share of pixels that disagree, by place and the biggest patches.
Does not touch data/packs/nyc.*.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import math
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pipeline"))

import numpy as np

import osm_water
import read_pack
import water
from look_water import dense

EXTRACTS = ["north-america/us/new-york", "north-america/us/new-jersey", "north-america/us/connecticut",
            "north-america/us/pennsylvania", "north-america/us/massachusetts", "north-america/us/rhode-island"]

# (name, w, s, e, n) boxes for the per-place tally
PLACES = [
    ("Hudson, Battery to Tappan Zee", -74.03, 40.70, -73.86, 41.10),
    ("East River and Harlem River", -74.00, 40.70, -73.83, 40.87),
    ("Upper Bay and Kill van Kull", -74.12, 40.62, -74.01, 40.70),
    ("Lower Bay and Raritan Bay", -74.27, 40.44, -73.95, 40.60),
    ("Newark Bay and the Arthur Kill", -74.26, 40.50, -74.08, 40.72),
    ("Jamaica Bay", -73.95, 40.57, -73.74, 40.66),
    ("Long Island Sound (west)", -73.80, 40.80, -73.40, 41.05),
    ("Great South Bay", -73.40, 40.60, -72.90, 40.75),
]


def main(out):
    header, a = read_pack.load_pack("nyc")
    tm = read_pack.water_mask(header, a)
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    (x0, y0), cell, (W, H) = tm["origin_m"], tm["cell_m"], tm["size"]
    tig = dense(tm)

    gw, gs = water.to_lonlat(np.array([x0]), np.array([y0]), lon0, lat0)
    ge, gn = water.to_lonlat(np.array([x0 + W * cell]), np.array([y0 + H * cell]), lon0, lat0)
    bbox = (float(gw[0]), float(gs[0]), float(ge[0]), float(gn[0]))
    geoms, info = osm_water.source(EXTRACTS)(bbox)
    m = osm_water.mask_on_grid(geoms, lon0, lat0, x0, y0, cell, W, H).without_narrow(2)
    info["min_width_m"] = 2 * cell
    osm_water.write_pack("nyc_osm", lon0, lat0, m, info)
    oh, oa = read_pack.load_pack("nyc_osm", osm_water.OUT)
    try:
        read_pack.check_water("nyc", oh, oa)
    except AssertionError as e:
        print(f"  OSM mask: {e}")
    osm = m.dense()

    n = W * H
    both, t_only, o_only = tig & osm, tig & ~osm, ~tig & osm
    print(f"grid {W} x {H} of {cell:.0f} m; water: TIGER {tig.mean():.2%}, OSM {osm.mean():.2%}")
    print(f"disagree: {(t_only.sum() + o_only.sum()) / n:.2%} of all pixels "
          f"(TIGER water, OSM land {t_only.sum() / n:.2%}; OSM water, TIGER land {o_only.sum() / n:.2%})")
    # within 1 km of any shore (where the open sea and inland plains cannot dilute it)
    from scipy.ndimage import binary_dilation, binary_erosion, center_of_mass, label
    edge = (binary_dilation(tig, iterations=40) & ~binary_erosion(tig, iterations=40))
    dis = t_only | o_only
    print(f"within 1 km of TIGER's shoreline: {dis[edge].mean():.2%} of {edge.sum():,} pixels disagree")
    k = math.cos(math.radians(lat0))

    def rows_cols(w_, s_, e_, n_):
        c0 = int((water.R_EARTH * math.radians(w_ - lon0) * k - x0) / cell)
        c1 = int((water.R_EARTH * math.radians(e_ - lon0) * k - x0) / cell)
        r0 = int((water.R_EARTH * math.radians(s_ - lat0) - y0) / cell)
        r1 = int((water.R_EARTH * math.radians(n_ - lat0) - y0) / cell)
        return slice(max(r0, 0), min(r1, H)), slice(max(c0, 0), min(c1, W))

    print("\nby place (box): TIGER water km2, OSM water km2, pixels that disagree")
    px_km2 = cell * cell / 1e6
    for name, *bb in PLACES:
        rs, cs = rows_cols(*bb)
        t, o = tig[rs, cs], osm[rs, cs]
        print(f"  {name:32s} {t.sum() * px_km2:8.1f} {o.sum() * px_km2:8.1f}   {(t != o).mean():.2%}")

    print("\nbiggest patches that disagree (km2, which side has water, lon/lat of the centre)")
    for arr, side in ((t_only, "TIGER only"), (o_only, "OSM only")):
        lab, nl = label(arr)
        if nl == 0:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        top = np.argsort(-sizes)[:8]
        cms = center_of_mass(arr, lab, top + 1)
        for i, (rr, cc) in zip(top, cms):
            lo, la = water.to_lonlat(x0 + (cc + 0.5) * cell, y0 + (rr + 0.5) * cell, lon0, lat0)
            print(f"  {side:10s} {sizes[i] * px_km2:8.2f} km2 at {lo:.4f}, {la:.4f}")

    picture(out, tig, osm, x0, y0, cell, W, H, lon0, lat0, k)


def picture(out, tig, osm, x0, y0, cell, W, H, lon0, lat0, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    # 0 land both, 1 water both, 2 TIGER only, 3 OSM only
    code = np.zeros(tig.shape, np.uint8)
    code[tig & osm] = 1
    code[tig & ~osm] = 2
    code[osm & ~tig] = 3
    cmap = ListedColormap(["#f3efe6", "#9dbbe0", "#d7301f", "#1a9850"])
    fig = plt.figure(figsize=(18, 12), dpi=130, facecolor="white")
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[:, :2])
    f = 6
    h, w = H // f, W // f
    small = code[:h * f, :w * f].reshape(h, f, w, f)
    # overview: a block shows a disagreement colour if any of its pixels disagree
    ov = np.where((small == 2).any(axis=(1, 3)), 2, np.where((small == 3).any(axis=(1, 3)), 3,
                  (small == 1).mean(axis=(1, 3)) > 0.5)).astype(np.uint8)
    ax.imshow(ov, origin="lower", cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
              extent=[x0 / 1e3, (x0 + w * f * cell) / 1e3, y0 / 1e3, (y0 + h * f * cell) / 1e3])
    ax.set_title("New York water: OSM (T-041) against TIGER (shipped). Blue both, red TIGER only, green OSM only",
                 fontsize=10)
    ax.tick_params(labelsize=7)
    views = [("Manhattan, East River, Governors Island", -73.975, 40.745, 7.5),
             ("Harlem River, Randalls Island, Rikers", -73.915, 40.815, 4.5),
             ("Jamaica Bay and JFK", -73.83, 40.62, 9.0),
             ("Newark Bay, Kill van Kull, Arthur Kill", -74.14, 40.64, 7.5)]
    for i, (title, lo, la, hw) in enumerate(views):
        axz = fig.add_subplot(gs[i // 2, 2 + i % 2])
        cx, cy = water.R_EARTH * math.radians(lo - lon0) * k, water.R_EARTH * math.radians(la - lat0)
        c0, c1 = int((cx - hw * 1e3 - x0) / cell), int((cx + hw * 1e3 - x0) / cell)
        r0, r1 = int((cy - hw * 1e3 - y0) / cell), int((cy + hw * 1e3 - y0) / cell)
        axz.imshow(code[r0:r1, c0:c1], origin="lower", cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
                   extent=[(x0 + c0 * cell) / 1e3, (x0 + c1 * cell) / 1e3, (y0 + r0 * cell) / 1e3, (y0 + r1 * cell) / 1e3])
        axz.set_title(title, fontsize=9)
        axz.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "artifacts", "trainworld", "T-041-nyc-compare.png"))
