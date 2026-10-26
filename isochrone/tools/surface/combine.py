"""Combine the layers into today's off-road walking speed (MAP's recipe, no roads).

    speed = landcover_kmh * elev_factor(elevation) * slope_factor(slope)

A pixel is water (0) when inland water + sea cover half of it or more; otherwise the land
part's speed stands. Where LC100 has no class (north of 80 N; class 0, as on Svalbard), land
is where GMTED is above 0 m, walked as snow and ice.

Outputs in data/surface/:
  terrain_factor.tif  uint16 elev_factor * slope_factor * 10000
  walk_today.tif      uint16 km/h * 1000, 0 = water or no land
Run: python tools/surface/combine.py   (about a minute)
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

STRIP = 1200
WATER_MIN = 0.5


def main():
    t0 = time.time()
    names = ("lc_speed", "lc_water", "elev", "slope")
    src = {n: rasterio.open(os.path.join(C.OUT, n + ".tif")) for n in names}
    pt = C.profile("uint16")
    with rasterio.open(os.path.join(C.OUT, "terrain_factor.tif"), "w", **pt) as ft, \
            rasterio.open(os.path.join(C.OUT, "walk_today.tif"), "w", **pt) as fw:
        for r0 in range(0, C.H, STRIP):
            win = Window(0, r0, C.W, min(STRIP, C.H - r0))
            lc = src["lc_speed"].read(1, window=win).astype(np.float32) / C.SPEED_SCALE
            wat = src["lc_water"].read(window=win).astype(np.float32).sum(0) / C.SHARE_SCALE
            el = src["elev"].read(1, window=win)
            sl = src["slope"].read(1, window=win).astype(np.float32) / 100
            has_el = el != -32768
            f = C.elev_factor(np.where(has_el, el, 0).astype(np.float32)) * C.slope_factor(sl)
            ft.write(np.rint(f * 10000).astype(np.uint16), 1, window=win)
            # no LC100 class (north of 80 N, or class 0 as on Svalbard): GMTED sea is 0 m
            nocls = (lc == 0) & (wat < WATER_MIN) & has_el & (el > 0)
            lc = np.where(nocls, C.SNOW_KMH, lc)
            land = (wat < WATER_MIN) & (lc > 0)
            spd = np.where(land, np.clip(np.rint(lc * f * C.SPEED_SCALE), 1, 65535), 0)
            fw.write(spd.astype(np.uint16), 1, window=win)
            print("rows", r0 + win.height, flush=True)
    for s in src.values():
        s.close()
    print("done %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
