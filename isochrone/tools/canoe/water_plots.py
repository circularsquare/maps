"""Checks and diagnostic plots for the canoe water and sea data (T-021).

    python tools/canoe/water_plots.py [sea] [links] [kinds]

Reads data/canoe/ the way the worker will (index.json, gzip planes) and writes PNGs to
maps/artifacts/isochrone/t021_*.png. Prints the link checks for the named straits and the
link symmetry check.
"""
import glob
import gzip
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANOE = os.path.join(HERE, "data", "canoe")
ART = os.path.join(os.path.dirname(HERE), "artifacts", "isochrone")
NEIGH = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
KIND_CMAP = ListedColormap(["#d8cfb8", "#2d6fb8", "#3aa35b", "#64c8d8"])


def index():
    with open(os.path.join(CANOE, "index.json")) as fh:
        return json.load(fh)


def planes(path, n, h, w):
    with gzip.open(path) as fh:
        a = np.frombuffer(fh.read(), np.uint8)
    assert a.size == n * h * w, (path, a.size)
    return a.reshape(n, h, w)


def level10():
    lv = index()["levels"]["10"]
    return planes(os.path.join(CANOE, lv["file"]), 3, lv["height"], lv["width"])


def window(level, west, south, east, north):
    """kind and shore (and links at L3) of a box, open sea where no tile exists."""
    idx = index()
    lv = idx["levels"][str(level)]
    f = lv["factor"]
    n = len(lv["bands"])
    step = 1 / (120 / f)
    c0, c1 = int(np.floor((west + 180) / step)), int(np.ceil((east + 180) / step))
    r0, r1 = int(np.floor((85 - north) / step)), int(np.ceil((85 - south) / step))
    out = np.zeros((n, r1 - r0, c1 - c0), np.uint8)
    out[0] = 1
    out[1] = 78
    s = lv["tile"]
    have = set(lv["tiles"])
    for tr in range(r0 // s, (r1 - 1) // s + 1):
        for tc in range(c0 // s, (c1 - 1) // s + 1):
            if tr * 72 + tc not in have:
                continue
            t = planes(os.path.join(CANOE, lv["path"], "%d_%d.bin.gz" % (tr, tc)), n, s, s)
            ar0, ar1 = max(r0, tr * s), min(r1, (tr + 1) * s)
            ac0, ac1 = max(c0, tc * s), min(c1, (tc + 1) * s)
            out[:, ar0 - r0:ar1 - r0, ac0 - c0:ac1 - c0] = t[:, ar0 - tr * s:ar1 - tr * s, ac0 - tc * s:ac1 - tc * s]
    return out, (c0 * step - 180, c1 * step - 180, 85 - r1 * step, 85 - r0 * step), (r0, c0)


def kind_img(k):
    return np.select([k == 0, k == 1, k == 2], [0, 1, 2], 3)


def symmetry(L, wrap=True):
    """every link bit must equal its twin on the neighbour."""
    bad = 0
    h, w = L.shape
    for k, (dc, dr) in enumerate(NEIGH):
        twin = NEIGH.index((-dc, -dr))
        a = (L >> k) & 1
        b = (np.roll(np.roll(L, -dr, axis=0), -dc, axis=1) >> twin) & 1
        rows = slice(max(0, -dr), h - max(0, dr))
        bad += int((a[rows] != b[rows]).sum())
    return bad


def link_between(L, lon_lat_a, lon_lat_b, step):
    """the link bit from the cell holding point a to the neighbouring cell holding point b."""
    ra, ca = int((85 - lon_lat_a[1]) / step), int((lon_lat_a[0] + 180) / step)
    rb, cb = int((85 - lon_lat_b[1]) / step), int((lon_lat_b[0] + 180) / step)
    k = NEIGH.index((cb - ca, rb - ra))
    return (L[ra, ca] >> k) & 1


def plot_links(ax, k, L, ext, step, title):
    """kind image with a line for every link (east, south and the two diagonals, so each once)."""
    ax.imshow(kind_img(k), cmap=KIND_CMAP, vmin=0, vmax=3, extent=ext, interpolation="nearest")
    w, e, s, n = ext
    h, wd = k.shape
    segs = []
    for kk in (0, 2, 4, 5):
        dc, dr = NEIGH[kk]
        rr, cc = np.nonzero((L >> kk) & 1)
        for r, c in zip(rr, cc):
            x0, y0 = w + (c + 0.5) * step, n - (r + 0.5) * step
            segs.append(((x0, y0), (x0 + dc * step, y0 - dr * step)))
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors="#c0392b", linewidths=0.8))
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_title(title, fontsize=9)


def links():
    k10, s10, L10 = level10()
    print("L10 link symmetry mismatches:", symmetry(L10))
    step10 = 1 / 12
    boxes = {
        "Bosporus and Dardanelles": (25.8, 39.6, 30.2, 41.8),
        "Strait of Messina": (14.9, 37.6, 16.2, 38.6),
        "Oresund": (11.9, 55.2, 13.3, 56.4),
        "Isthmus of Corinth": (22.3, 37.6, 23.5, 38.3),
    }
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for j, (name, (w, s, e, n)) in enumerate(boxes.items()):
        for i, (level, step) in enumerate(((10, step10), (3, 1 / 40))):
            if level == 10:
                c0, c1 = int((w + 180) / step), int(np.ceil((e + 180) / step))
                r0, r1 = int((85 - n) / step), int(np.ceil((85 - s) / step))
                k, L = k10[r0:r1, c0:c1], L10[r0:r1, c0:c1]
                ext = (c0 * step - 180, c1 * step - 180, 85 - r1 * step, 85 - r0 * step)
            else:
                a, ext, _ = window(3, w, s, e, n)
                k, L = a[0], a[2]
            plot_links(axes[i, j], k, L, ext, step, "%s, L%d" % (name, level))
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "t021_links_straits.png"), dpi=110)
    plt.close(fig)
    # Bosporus close-up at L10 over the L1 kinds
    w, s, e, n = 28.4, 40.7, 29.6, 41.5
    a1, ext1, _ = window(1, w, s, e, n)
    c0, c1 = int((w + 180) / step10), int(np.ceil((e + 180) / step10))
    r0, r1 = int((85 - n) / step10), int(np.ceil((85 - s) / step10))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(kind_img(a1[0]), cmap=KIND_CMAP, vmin=0, vmax=3, extent=ext1, interpolation="nearest")
    ext = (c0 * step10 - 180, c1 * step10 - 180, 85 - r1 * step10, 85 - r0 * step10)
    k, L = k10[r0:r1, c0:c1], L10[r0:r1, c0:c1]
    for x in np.arange(ext[0], ext[1] + 1e-9, step10):
        ax.axvline(x, color="#555", lw=0.3)
    for y in np.arange(ext[2], ext[3] + 1e-9, step10):
        ax.axhline(y, color="#555", lw=0.3)
    from matplotlib.collections import LineCollection
    segs = []
    for kk in (0, 2, 4, 5):
        dc, dr = NEIGH[kk]
        rr, cc = np.nonzero((L >> kk) & 1)
        for r, c in zip(rr, cc):
            x0, y0 = ext[0] + (c + 0.5) * step10, ext[3] - (r + 0.5) * step10
            segs.append(((x0, y0), (x0 + dc * step10, y0 - dr * step10)))
    ax.add_collection(LineCollection(segs, colors="#c0392b", linewidths=1.2))
    rr, cc = np.nonzero(k > 0)
    ax.scatter(ext[0] + (cc + 0.5) * step10, ext[3] - (rr + 0.5) * step10, s=6, c="k")
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_title("Bosporus: L1 water kinds, L10 cells (dots = canoe node), L10 links (red)")
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "t021_links_bosporus.png"), dpi=120)
    plt.close(fig)


def kinds():
    boxes = {"Norway fjords": (4.5, 59.5, 8.5, 62.0), "Rio de la Plata": (-59.5, -35.5, -54.5, -33.5),
             "Great Lakes": (-93, 41, -75, 49.5), "Amazon at Manaus": (-61, -4, -58, -2.5),
             "Bangladesh delta": (88.5, 21.5, 91.5, 23.5), "Baltic, Gulf of Finland": (22, 59, 30.5, 61)}
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for ax, (name, (w, s, e, n)) in zip(axes.ravel(), boxes.items()):
        level = 3 if (e - w) > 6 else 1
        a, ext, _ = window(level, w, s, e, n)
        ax.imshow(kind_img(a[0]), cmap=KIND_CMAP, vmin=0, vmax=3, extent=ext, interpolation="nearest")
        ax.set_title("%s, L%d kind (beige land, blue sea, green river water, cyan lake)" % (name, level), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "t021_kinds.png"), dpi=100)
    plt.close(fig)
    # shore at L10, world
    k10, s10, L10 = level10()
    km = np.where(s10 <= 20, s10, 20 + (s10.astype(int) - 20) * 10).astype(float)
    km[k10 == 0] = np.nan
    fig, ax = plt.subplots(figsize=(18, 8))
    im = ax.imshow(km, extent=(-180, 180, -60, 85), cmap="viridis", vmax=600, interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.7, label="km to land (L10, least over the cell)")
    ax.set_title("L10 shore distance")
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "t021_shore_l10.png"), dpi=90)
    plt.close(fig)


def sea():
    import geopandas as gpd
    lv = index()["sea"]
    h, w = lv["height"], lv["width"]
    with gzip.open(os.path.join(CANOE, lv["file"])) as fh:
        raw = fh.read()
    n = h * w
    u = np.frombuffer(raw, np.int8, n, 0).reshape(h, w) * 0.02
    v = np.frombuffer(raw, np.int8, n, n).reshape(h, w) * 0.02
    hs = np.frombuffer(raw, np.uint8, n, 2 * n).reshape(h, w) * 0.05
    ice = np.frombuffer(raw, np.uint8, n, 3 * n).reshape(h, w).astype(float)
    sst = np.frombuffer(raw, np.int8, n, 4 * n).reshape(h, w).astype(float)
    k10 = level10()[0]
    # land mask at 0.25 degree from L10 (3 x 3 L10 cells): land where all nine are land
    land = (k10.reshape(h, 3, w, 3) == 0).all(axis=(1, 3))
    ext = (-180, 180, -60, 85)
    sp = np.hypot(u, v)
    show = lambda a: np.where(land, np.nan, a)
    fig, ax = plt.subplots(figsize=(18, 8))
    im = ax.imshow(show(sp), extent=ext, cmap="magma", vmax=1.2, interpolation="nearest")
    yy, xx = np.mgrid[0:h:12, 0:w:12]
    m = ~land[::12, ::12]
    ax.quiver((xx[m] + 0.5) * 0.25 - 180, 85 - (yy[m] + 0.5) * 0.25, u[::12, ::12][m], v[::12, ::12][m],
              color="#7fd", scale=25, width=0.0008)
    plt.colorbar(im, ax=ax, shrink=0.7, label="m/s, local summer")
    ax.set_title("Drifter currents, local summer (JJA north, DJF south)")
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "t021_currents.png"), dpi=90)
    plt.close(fig)
    # Gulf Stream close-up
    fig, ax = plt.subplots(figsize=(10, 7))
    r0, r1, c0, c1 = int((85 - 45) / 0.25), int((85 - 25) / 0.25), int((180 - 82) / 0.25), int((180 - 55) / 0.25)
    sub = (-82, -55, 25, 45)
    im = ax.imshow(show(sp)[r0:r1, c0:c1], extent=sub, cmap="magma", vmax=1.5, interpolation="nearest")
    yy, xx = np.mgrid[r0:r1:2, c0:c1:2]
    ax.quiver((xx + 0.5) * 0.25 - 180, 85 - (yy + 0.5) * 0.25, u[r0:r1:2, c0:c1:2], v[r0:r1:2, c0:c1:2],
              color="#7fd", scale=20, width=0.0015)
    ax.plot([-73], [36], "c+", ms=14)
    plt.colorbar(im, ax=ax, label="m/s")
    ax.set_title("Gulf Stream, JJA drifter mean (cross at 36N 73W)")
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "t021_gulfstream.png"), dpi=100)
    plt.close(fig)
    # waves
    fig, ax = plt.subplots(figsize=(18, 8))
    im = ax.imshow(show(hs), extent=ext, cmap="viridis", vmax=4, interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.7, label="m, mean significant wave height, local summer")
    ax.set_title("ERA5 summer wave height")
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "t021_hs.png"), dpi=90)
    plt.close(fig)
    # ice with NSIDC median edges
    dbg = np.load(os.path.join(HERE, "data", "raw", "sea_debug.npz"))
    seaice = np.where(dbg["ice_has"], ice, np.nan)
    fig, axes = plt.subplots(2, 1, figsize=(20, 14))
    raw_dir = os.path.join(HERE, "data", "raw", "nsidc_g02135")
    for ax, (name, ylim, files) in zip(axes, (("Arctic, JJA mean ice (ERA5) and NSIDC median edges: Jun yellow, Jul green, Aug magenta", (50, 85), "N"),
                                                ("Southern edge of the grid, DJF mean ice (ERA5) and NSIDC Dec/Jan/Feb median edges", (-60, -50), "S"))):
        im = ax.imshow(seaice, extent=ext, cmap="Blues_r", vmin=0, vmax=100, interpolation="nearest", aspect="auto")
        ax.set_facecolor("#999")
        ax.contour(np.linspace(-180 + 0.125, 180 - 0.125, w), np.linspace(85 - 0.125, -60 + 0.125, h), np.nan_to_num(seaice), [15], colors="r", linewidths=0.8)
        colors = {"06": "#ff0", "07": "#0f0", "08": "#f0f", "12": "#ff0", "01": "#0f0", "02": "#f0f"}
        for z in sorted(glob.glob(os.path.join(raw_dir, "median_extent_%s_*.zip" % files))):
            g = gpd.read_file("zip://" + z).to_crs(4326)
            mo = os.path.basename(z).split("_")[3]
            for geom in g.geometry:
                for line in getattr(geom, "geoms", [geom]):
                    x, y = np.asarray(line.xy)
                    brk = np.nonzero(np.abs(np.diff(x)) > 90)[0] + 1
                    for xs, ys in zip(np.split(x, brk), np.split(y, brk)):
                        ax.plot(xs, ys, color=colors[mo], lw=0.5)
        ax.set_ylim(*ylim)
        ax.set_title(name + " (red: ERA5 15%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "t021_ice.png"), dpi=100)
    plt.close(fig)
    for name, (lat, lon) in {"Hudson Bay 60N 86W": (60, -86), "Baffin Bay 72N 65W": (72, -65),
                             "Barents 75N 40E": (75, 40), "Kara 74N 70E": (74, 70), "Beaufort 72N 140W": (72, -140),
                             "Bering Strait 66N 169W": (66, -169), "Gulf of Bothnia 64N 22E": (64, 22)}.items():
        rr, cc = int((85 - lat) / 0.25), int((lon + 180) / 0.25)
        print("summer ice %-24s %3d%%" % (name, ice[rr, cc]))
    r, c = int((85 - 36) / 0.25), int((180 - 73) / 0.25)
    print("36N 73W cell: u %.2f v %.2f speed %.2f m/s; max within 1 degree %.2f" % (u[r, c], v[r, c], sp[r, c], sp[r - 4:r + 4, c - 4:c + 4].max()))
    print("summer Hs: mean over open cells %.2f m; Southern Ocean 50-60S mean %.2f m; Med mean %.2f m" % (
        hs[~land].mean(), hs[(int((85 + 50) / 0.25)):, :][~land[int((85 + 50) / 0.25):]].mean(),
        hs[int((85 - 44) / 0.25):int((85 - 31) / 0.25), int((180 - 5) / 0.25):int((180 + 30) / 0.25)][~land[int((85 - 44) / 0.25):int((85 - 31) / 0.25), int((180 - 5) / 0.25):int((180 + 30) / 0.25)]].mean()))


def straits():
    k10, s10, L10 = level10()
    st = 1 / 12
    # is A reachable from B over links inside the box?
    pairs = {
        "Bosporus (Black Sea to Marmara)": ((29.6, 41.6), (28.9, 40.8), (28.3, 40.4, 30.0, 41.8)),
        "Dardanelles (Marmara to Aegean)": ((27.5, 40.6), (25.9, 39.9), (25.6, 39.6, 28.0, 40.8)),
        "Strait of Messina": ((15.65, 38.35), (15.5, 37.85), (15.2, 37.6, 16.0, 38.6)),
        "Oresund (Kattegat to Baltic)": ((12.4, 56.3), (12.8, 55.3), (12.2, 55.2, 13.2, 56.4)),
        "Corinth isthmus (Gulf of Corinth to Saronic)": ((22.75, 38.05), (23.1, 37.85), (22.5, 37.7, 23.4, 38.2)),
    }
    for level in (10, 3):
        for name, (a, b, box) in pairs.items():
            if level == 10:
                step = st
                w, s, e, n = box
                c0, c1 = int((w + 180) / step), int(np.ceil((e + 180) / step))
                r0, r1 = int((85 - n) / step), int(np.ceil((85 - s) / step))
                k, L = k10[r0:r1, c0:c1], L10[r0:r1, c0:c1]
                origin = (85 - r0 * step, c0 * step - 180)
            else:
                step = 1 / 40
                arr, ext, (r0, c0) = window(3, *box)
                k, L = arr[0], arr[2]
                origin = (85 - r0 * step, c0 * step - 180)
            ok, n_cells = reach(k, L, a, b, origin, step)
            print("L%d %-46s %s (%d cells reached in the box)" % (level, name, "joined" if ok else "NOT joined", n_cells))


def reach(k, L, a, b, origin, step):
    top, west = origin
    cell = lambda p: (int((top - p[1]) / step), int((p[0] - west) / step))
    ra, ca = cell(a)
    rb, cb = cell(b)
    h, w = k.shape
    if k[ra, ca] in (0, 2):
        return False, 0
    seen = np.zeros((h, w), bool)
    seen[ra, ca] = True
    stack = [(ra, ca)]
    while stack:
        r, c = stack.pop()
        for kk, (dc, dr) in enumerate(NEIGH):
            if (L[r, c] >> kk) & 1:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < h and 0 <= c2 < w and not seen[r2, c2]:
                    seen[r2, c2] = True
                    stack.append((r2, c2))
    return bool(seen[rb, cb]), int(seen.sum())


if __name__ == "__main__":
    os.makedirs(ART, exist_ok=True)
    what = sys.argv[1:] or ["links", "kinds", "sea", "straits"]
    for wname in what:
        globals()[wname]()
