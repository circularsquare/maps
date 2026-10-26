"""Compare walk_today.tif with MAP's 2020 walking-only surface, off-road.

MAP puts every road pixel at exactly 5 km/h; those are left out, as is water on either side.
Ratio = ours / MAP. Also works out MAP's own land cover speed at each pixel (MAP speed over
our terrain factor, snapped to its table) to show where the land cover disagrees and where
the terrain does.

Prints tables; writes data/surface/compare.json and PNG maps of a few places to
maps/artifacts/isochrone/t009_*.png.
Run: python tools/surface/compare.py   (a few minutes)
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ART = os.path.join(os.path.dirname(C.HERE), "artifacts", "isochrone")
STRIP = 600
LO, HI, NB = -5.0, 5.0, 2000          # log2 ratio histogram
MODIS = np.array([1.0, 1.62, 2.0, 2.5, 3.0, 3.24, 4.0, 4.2, 4.86, 5.0], np.float32)
REGIONS = {
    "Carpathians": (21, 45, 27, 49), "Amazon": (-70, -10, -55, 0),
    "Sahara": (-5, 18, 25, 30), "Himalaya": (78, 27, 92, 32),
    "US Midwest": (-98, 38, -86, 44), "Europe": (-10, 35, 40, 70),
    "Africa": (-18, -35, 52, 37), "South America": (-82, -56, -34, 13),
    "North America": (-170, 15, -50, 72), "Asia": (40, -10, 180, 80),
    "Australia": (112, -45, 155, -10), "Siberia": (60, 55, 180, 75),
}
PLACES = {
    "cluj": (21.5, 45.5, 25.5, 47.8), "amazon": (-64, -6, -59, -2),
    "sahara": (6, 20, 12, 24), "himalaya": (84, 26.5, 90, 30),
    "midwest": (-95, 40, -90, 43),
}


def stats(h):
    n = h.sum()
    if n == 0:
        return None
    cdf = np.cumsum(h) / n
    edges = np.linspace(LO, HI, NB + 1)
    q = lambda p: float(2 ** edges[min(np.searchsorted(cdf, p) + 1, NB)])
    w = int(np.ceil(np.log2(1.1) / ((HI - LO) / NB)))
    mid = NB // 2
    within = h[mid - w:mid + w].sum() / n
    return {"n": int(n), "median": round(q(0.5), 3), "p10": round(q(0.1), 3),
            "p90": round(q(0.9), 3), "within10": round(float(within), 3)}


def main():
    t0 = time.time()
    names = ("walk_today", "lc_class", "lc_speed", "terrain_factor")
    src = {n: rasterio.open(os.path.join(C.OUT, n + ".tif")) for n in names}
    mp = rasterio.open(C.MAP_WALK)
    table = np.zeros(256, np.float32)
    for k, v in C.LC_SPEED.items():
        table[k] = v
    H = {}
    conf = np.zeros((256, len(MODIS) + 1), np.int64)
    counts = {"map_water_ours_land": 0, "map_land_ours_water": 0, "both_land": 0,
              "map_road": 0, "offroad": 0}

    def add(key, idx):
        h = H.setdefault(key, np.zeros(NB, np.int64))
        h += np.bincount(idx, minlength=NB)[:NB]

    for r0 in range(0, C.H, STRIP):
        win = Window(0, r0, C.W, min(STRIP, C.H - r0))
        ours = src["walk_today"].read(1, window=win).astype(np.float32) / C.SPEED_SCALE
        cls = src["lc_class"].read(1, window=win)
        lcs = src["lc_speed"].read(1, window=win).astype(np.float32) / C.SPEED_SCALE
        tf = src["terrain_factor"].read(1, window=win).astype(np.float32) / 10000
        f = mp.read(1, window=win)
        mvalid = f > 0
        mwater = f == np.float32(0.06)
        with np.errstate(divide="ignore", invalid="ignore"):
            mk = np.where(mvalid, 0.06 / f, 0)
        mland = mvalid & ~mwater
        road = mland & (np.abs(mk - 5.0) < 0.005)
        oland = ours > 0
        counts["map_water_ours_land"] += int((mwater & oland).sum())
        counts["map_land_ours_water"] += int((mland & ~oland).sum())
        counts["both_land"] += int((mland & oland).sum())
        counts["map_road"] += int((road & oland).sum())
        sel = mland & oland & ~road
        counts["offroad"] += int(sel.sum())
        if not sel.any():
            continue
        r = np.log2(ours[sel] / mk[sel])
        idx = np.clip(((r - LO) / (HI - LO) * NB).astype(np.int64), 0, NB - 1)
        add("all", idx)
        c = cls[sel]
        pure = np.abs(lcs[sel] - table[c]) < 0.01 * np.maximum(table[c], 1e-3)
        # MAP's own land cover speed, snapped to its table
        imp = mk[sel] / np.maximum(tf[sel], 1e-4)
        d = np.abs(imp[:, None] / MODIS[None, :] - 1)
        mc = np.where(d.min(1) < 0.03, d.argmin(1), len(MODIS))
        np.add.at(conf, (c[pure], mc[pure]), 1)
        same_lc = pure & (mc < len(MODIS)) & (np.abs(MODIS[np.minimum(mc, len(MODIS) - 1)]
                                                      - table[c]) < 0.01)
        add("same land cover speed (terrain only)", idx[same_lc])
        add("different land cover speed", idx[pure & ~same_lc])
        for k in np.unique(c):
            m = c == k
            add("class %d all" % k, idx[m])
            add("class %d pure" % k, idx[m & pure])
        rows, cols = np.nonzero(sel)
        lat = C.NORTH - (r0 + rows + 0.5) * C.RES
        lon = C.WEST + (cols + 0.5) * C.RES
        for name, (x0, y0, x1, y1) in REGIONS.items():
            m = (lon >= x0) & (lon < x1) & (lat >= y0) & (lat < y1)
            if m.any():
                add("region " + name, idx[m])
        print("rows", r0 + win.height, "%.0f s" % (time.time() - t0), flush=True)

    res = {"counts": counts, "groups": {k: stats(h) for k, h in H.items()},
           "confusion_pure": {str(k): conf[k].tolist() for k in range(256) if conf[k].sum()},
           "confusion_cols": [str(v) for v in MODIS] + ["other"]}
    with open(os.path.join(C.OUT, "compare.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(counts))
    for k, s in res["groups"].items():
        print("%-45s %s" % (k, s))
    print("confusion (pure pixels, LC100 centre class x MAP implied km/h):")
    print("      " + " ".join("%7s" % c for c in res["confusion_cols"]))
    for k, row in res["confusion_pure"].items():
        print("%5s " % k + " ".join("%7d" % v for v in row))
    for s in src.values():
        s.close()
    places(mp)
    print("done %.0f s" % (time.time() - t0), flush=True)


def places(mp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(ART, exist_ok=True)
    walk = rasterio.open(os.path.join(C.OUT, "walk_today.tif"))
    for name, (x0, y0, x1, y1) in PLACES.items():
        win = from_bounds(x0, y0, x1, y1, mp.transform).round_offsets().round_lengths()
        f = mp.read(1, window=win)
        ours = walk.read(1, window=win).astype(np.float32) / C.SPEED_SCALE
        mwater = (f == np.float32(0.06)) | (f <= 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mk = np.where(mwater, np.nan, 0.06 / f)
            road = np.abs(mk - 5.0) < 0.005
            ok = ~mwater & (ours > 0)
            ratio = np.where(ok & ~road, ours / mk, np.nan)
        o = np.where(ours > 0, ours, np.nan)
        fig, ax = plt.subplots(1, 3, figsize=(21, 6.5), constrained_layout=True)
        ext = (x0, x1, y0, y1)
        kw = dict(extent=ext, interpolation="nearest")
        for a, img, t in ((ax[0], mk, "MAP 2020 walking (roads at 5 km/h)"),
                          (ax[1], o, "ours, off-road")):
            im = a.imshow(img, cmap="viridis", vmin=0, vmax=5.2, **kw)
            a.set_title(t + ", km/h (black: water)")
            a.set_facecolor("black")
        fig.colorbar(im, ax=ax[:2], shrink=0.8)
        ax[2].set_facecolor("black")
        ax[2].imshow(np.where(road & ok, 1.0, np.nan), cmap="Greys", vmin=0, vmax=3, **kw)
        im2 = ax[2].imshow(np.log2(ratio), cmap="RdBu", vmin=-1.5, vmax=1.5, **kw)
        ax[2].set_title("ours / MAP, blue = ours faster (grey: MAP road, left out)")
        cb = fig.colorbar(im2, ax=ax[2], shrink=0.8, ticks=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        cb.ax.set_yticklabels(["0.35", "0.5", "0.71", "1", "1.41", "2", "2.83"])
        fig.suptitle("T-009 %s" % name)
        path = os.path.join(ART, "t009_%s.png" % name)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "places":
        places(rasterio.open(C.MAP_WALK))
    else:
        main()
