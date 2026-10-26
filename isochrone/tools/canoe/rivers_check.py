"""Checks on the built river steps: counts and lengths per level and region, and diagnostic PNGs.

    python tools/canoe/rivers_check.py stats
    python tools/canoe/rivers_check.py png

Reads the published files (data/canoe/rivers.bin.gz, data/canoe/rivers/l3, l1), so it also checks
the byte layout. PNGs go to maps/artifacts/isochrone/t020_*.png.
"""
import gzip
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(HERE, "data", "canoe")
ART = os.path.join(os.path.dirname(HERE), "artifacts", "isochrone")
NORTH, WEST = 85.0, -180.0
L1W, L1H = 43200, 17400
TILE = {1: 600, 3: 200}
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
CLASSES = ["flat", "rapids", "unrunnable", "fall", "dam", "weir", "reservoir"]


def read(path):
    b = gzip.open(path).read()
    n = int(np.frombuffer(b, "<u4", 1)[0])
    o = 4
    cell = np.frombuffer(b, "<u4", n, o); o += 4 * n
    length = np.frombuffer(b, "<u2", n, o); o += 2 * n
    d = np.frombuffer(b, np.uint8, n, o); o += n
    v = np.frombuffer(b, np.uint8, n, o); o += n
    c = np.frombuffer(b, np.uint8, n, o); o += n
    assert o == len(b), (path, o, len(b))
    return cell, length, d, v, c


def level_steps(level, box=None):
    """Global row, col and fields of a level's steps (optionally only tiles touching box)."""
    if level == 10:
        cell, length, d, v, c = read(os.path.join(OUT, "rivers.bin.gz"))
        W = L1W // 10
        return cell // W, cell % W, length, d, v, c
    size = TILE[level]
    idx = json.load(open(os.path.join(OUT, "rivers", "index.json")))
    parts = []
    for t in idx[str(level)]["tiles"]:
        tr, tc = t // 72, t % 72
        if box:
            w, s, e, n = box
            if tc * 5 - 180 >= e or (tc + 1) * 5 - 180 <= w or 85 - tr * 5 <= s or 85 - (tr + 1) * 5 >= n:
                continue
        cell, length, d, v, c = read(os.path.join(OUT, "rivers", "l%d" % level, "%d_%d.bin.gz" % (tr, tc)))
        parts.append((tr * size + cell // size, tc * size + cell % size, length, d, v, c))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(6))


def region(lon, lat):
    """Rough continent of a point, for the counts table."""
    r = np.full(len(lon), "other", object)
    r[(lon >= -170) & (lon < -50) & (lat >= 7)] = "N America"
    r[(lon >= -92) & (lon < -30) & (lat < 13) & ~((lon >= -170) & (lon < -77) & (lat >= 7))] = "S America"
    r[(lon >= -25) & (lon < 60) & (lat >= 36)] = "Europe"
    r[(lon >= -20) & (lon < 52) & (lat < 37.5) & (lat >= -36)] = "Africa"
    r[(lon >= 60) & (lat >= 0) | ((lon >= 40) & (lat >= 12) & (lat < 40) & (r != "Africa"))] = "Asia"
    r[(lon >= 90) & (lat < 0) & (lat >= -11) & (lon < 141)] = "Asia"
    r[(lon >= 112) & (lat < -11)] = "Australia"
    return r


def stats():
    summ = json.load(open(os.path.join(HERE, "data", "raw", "rivers_work", "summary.json")))
    print("reaches with Q >= 1:", summ["reaches"], "length %.2f M km" % (summ["km"] / 1e6))
    for level in (10, 3, 1):
        row, col, length, d, v, c = level_steps(level)
        cs = level / 120
        lon = WEST + (col + 0.5) * cs
        lat = NORTH - (row + 0.5) * cs
        km = length / 100.0
        st = summ["levels"][str(level)]
        print("\nL%d: %d steps, %.2f M km, %.1f MB in %d files; folded %d, not adjacent %d, in lakes dropped %d" %
              (level, len(km), km.sum() / 1e6, st["bytes"] / 1e6, st["files"], st["folded"], st["not_adjacent"],
               st["in_lake_dropped"]))
        reg = region(lon, lat)
        for name in ["Asia", "Europe", "Africa", "N America", "S America", "Australia", "other"]:
            m = reg == name
            print("  %-10s %9d steps %6.2f M km  mean step %.2f km" % (name, m.sum(), km[m].sum() / 1e6,
                                                                   km[m].mean() if m.any() else 0))
        print("  classes: " + ", ".join("%s %d (%.0f k km)" % (n, (c == i).sum(), km[c == i].sum() / 1e3)
                                        for i, n in enumerate(CLASSES)))
        print("  v m/s percentiles 1,50,99: %s; dirs %s" % (np.percentile(v / 100, [1, 50, 99]), np.bincount(d, minlength=8)))
        dup = len(np.unique(row.astype(np.int64) * (L1W // level) * 8 + col * 8 + d))
        print("  duplicate (cell, dir) records: %d" % (len(km) - dup))


COLORS = {  # class -> rgb
    "flat": (70, 140, 255), "rapids": (255, 210, 0), "unrunnable": (255, 120, 0), "fall": (255, 0, 0),
    "dam": (255, 0, 255), "weir": (170, 60, 255), "reservoir": (0, 230, 230)}


def water_box(level, box):
    import glob
    w, s, e, n = box
    cs = level / 120
    c0, c1 = int(np.floor((w - WEST) / cs)), int(np.ceil((e - WEST) / cs))
    r0, r1 = int(np.floor((NORTH - n) / cs)), int(np.ceil((NORTH - s) / cs))
    if level == 10:
        g = np.frombuffer(gzip.open(os.path.join(HERE, "data", "grid.bin.gz")).read(), np.uint8).reshape(L1H // 10, L1W // 10)
        return g[r0:r1, c0:c1] == 0, r0, c0
    size = TILE[level]
    out = np.ones((r1 - r0, c1 - c0), bool)
    for tr in range(r0 // size, (r1 - 1) // size + 1):
        for tc in range(c0 // size, (c1 - 1) // size + 1):
            p = os.path.join(HERE, "data", "tiles", "l%d" % level, "%d_%d.bin.gz" % (tr, tc))
            if not os.path.exists(p):
                continue
            t = np.frombuffer(gzip.open(p).read(), np.uint8).reshape(size, size) == 0
            gr0, gc0 = tr * size, tc * size
            a0, a1 = max(r0, gr0), min(r1, gr0 + size)
            b0, b1 = max(c0, gc0), min(c1, gc0 + size)
            out[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = t[a0 - gr0:a1 - gr0, b0 - gc0:b1 - gc0]
    return out, r0, c0


def png(name, level, box, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    water, r0, c0 = water_box(level, box)
    row, col, length, d, v, c = level_steps(level, box)
    H, W = water.shape
    m = (row >= r0) & (row < r0 + H) & (col >= c0) & (col < c0 + W)
    row, col, length, d, v, c = row[m] - r0, col[m] - c0, length[m], d[m], v[m], c[m]
    img = np.where(water[..., None], np.array([25, 45, 80], np.uint8), np.array([45, 45, 45], np.uint8))
    dc = np.array([mv[0] for mv in MOVES])[d]
    dr = np.array([mv[1] for mv in MOVES])[d]
    names = np.array(CLASSES, object)[c]
    scale = max(1, int(1600 / max(W, H)))
    fig_w = W * scale / 100
    fig_h = H * scale / 100 + 0.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.imshow(img, interpolation="nearest", extent=(0, W, H, 0))
    segs = np.stack([np.stack([col + 0.5, row + 0.5], 1), np.stack([col + 0.5 + dc, row + 0.5 + dr], 1)], 1)
    lw = 0.4 + 2.2 * (v / 255.0) ** 2
    cols = np.array([np.array(COLORS[n]) / 255 for n in names]) if len(names) else np.zeros((0, 3))
    order = np.argsort(np.array([list(COLORS).index(n) for n in names]))
    ax.add_collection(LineCollection(segs[order], colors=cols[order], linewidths=lw[order] * max(1, scale / 2)))
    for n in ("fall", "dam", "weir", "unrunnable"):
        k = names == n
        if k.any() and n != "unrunnable":
            ax.scatter(col[k] + 0.5, row[k] + 0.5, s=12 * scale, color=np.array(COLORS[n]) / 255, zorder=5)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    counts = ", ".join("%s %d" % (n, (names == n).sum()) for n in COLORS if (names == n).any())
    ax.set_title("%s, L%d: %s" % (title, level, counts), fontsize=8)
    fig.tight_layout()
    os.makedirs(ART, exist_ok=True)
    p = os.path.join(ART, "t020_%s.png" % name)
    fig.savefig(p)
    plt.close(fig)
    print(p, len(names), "steps")


def pngs():
    png("danube_l3", 3, (21.0, 43.0, 30.0, 48.5), "Danube and Tisza, Cluj to the Black Sea")
    png("danube_l10", 10, (13.0, 42.0, 30.0, 50.0), "Danube basin")
    png("ironGates_l1", 1, (21.5, 44.2, 23.0, 45.0), "Iron Gates")
    png("amazon_l10", 10, (-80.0, -20.0, -45.0, 6.0), "Amazon")
    png("amazon_mouth_l3", 3, (-56.0, -4.0, -48.0, 1.0), "Lower Amazon")
    png("yukon_l10", 10, (-168.0, 59.0, -128.0, 69.0), "Yukon")
    png("yukon_l3", 3, (-142.0, 63.0, -134.0, 66.0), "Yukon at Dawson")
    png("congo_l1", 1, (13.0, -6.2, 16.0, -4.0), "Congo, Kinshasa to Matadi")
    png("congo_l10", 10, (10.0, -10.0, 32.0, 6.0), "Congo basin")
    png("niagara_l1", 1, (-79.4, 42.8, -78.8, 43.35), "Niagara")


if __name__ == "__main__":
    {"stats": stats, "png": pngs}[sys.argv[1]]()
