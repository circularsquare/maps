"""T-047: what each 1 x 1 degree land tile holds, from the 1 km layers, for the selection rule.

    python isochrone/notes/T-047/world.py <out.csv>      (~5 min, one core)

Per tile: land share; urban share (LC100 class 50 at the pixel centre); mixed water (pixels
with 2-98% water: shores, rivers and ponds narrower than 1 km); rugged (pixels sloping more
than 5 degrees in GMTED at 1 km); road share (MAP pixels at road speed: MAP km/h over our
terrain factor within 3% of 5).
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import rasterio
from rasterio.windows import Window

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = os.path.join(HERE, "data", "surface")
MAP = os.path.join(HERE, "data", "raw", "unzipped", "2020_walking_only_friction_surface.geotiff")


def blocks(a):
    return a.reshape(120, 360, 120).transpose(1, 0, 2).reshape(360, -1)


def main(out):
    src = {n: rasterio.open(os.path.join(S, n + ".tif")) for n in ("lc_class", "lc_water", "slope", "terrain_factor", "walk_today")}
    mp = rasterio.open(MAP)
    rows = []
    for t in range(145):
        win = Window(0, t * 120, 43200, 120)
        cls = src["lc_class"].read(1, window=win)
        wat = src["lc_water"].read(window=win).astype(np.float32).sum(0) / 250
        sl = src["slope"].read(1, window=win).astype(np.float32) / 100
        tf = src["terrain_factor"].read(1, window=win).astype(np.float32) / 10000
        walk = src["walk_today"].read(1, window=win)
        fr = mp.read(1, window=win).astype(np.float32)
        land = walk > 0
        mkmh = np.where((fr > 0) & np.isfinite(fr), 0.06 / np.maximum(fr, 1e-9), 0)
        road = land & (np.abs(mkmh / np.maximum(tf, 1e-6) - 5.0) < 0.15)
        B = {k: blocks(v) for k, v in (("land", land), ("urban", (cls == 50) & land),
                                       ("mixed", (wat > 0.02) & (wat < 0.98)), ("rugged", land & (sl > 5)),
                                       ("road", road))}
        nl = B["land"].sum(1)
        lat = 84 - t
        for c in np.nonzero(nl > 0)[0]:
            n = nl[c]
            rows.append((lat, c - 180, n / 14400, B["urban"][c].sum() / n, B["mixed"][c].sum() / 14400,
                         B["rugged"][c].sum() / n, B["road"][c].sum() / n))
        print("row", t, len(rows), flush=True)
    with open(out, "w") as f:
        f.write("lat,lon,land,urban,mixed,rugged,road\n")
        for r in rows:
            f.write("%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n" % r)


if __name__ == "__main__":
    main(sys.argv[1])
