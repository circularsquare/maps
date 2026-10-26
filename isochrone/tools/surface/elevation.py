"""GMTED2010 30" mean elevation -> elevation and slope on MAP's 30" grid.

Outputs in data/surface/:
  elev.tif   int16 metres; GMTED gives sea as 0 m and has nothing north of 84 N (-32768).
             Bilinear from GMTED's grid, which sits half a pixel off MAP's.
  slope.tif  uint16 slope angle in degrees * 100, from the 4 neighbours of each pixel
             (as Earth Engine's ee.Terrain.slope, which MAP used), sea taken as 0 m.
Run: python tools/surface/elevation.py   (a minute or two)
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

STRIP = 600
R_EARTH = 6371008.8
NODATA = -32768


def main():
    t0 = time.time()
    p = C.profile("int16", nodata=NODATA)
    elev = np.full((C.H, C.W), NODATA, np.int16)
    with rasterio.open(C.GMTED) as src:
        print(src.profile, src.bounds, flush=True)
        with WarpedVRT(src, crs="EPSG:4326", transform=p["transform"], width=C.W,
                       height=C.H, resampling=Resampling.bilinear, src_nodata=src.nodata,
                       nodata=NODATA, dtype="int16") as vrt:
            for r0 in range(0, C.H, STRIP):
                elev[r0:r0 + STRIP] = vrt.read(1, window=Window(0, r0, C.W, STRIP))
                print("elev rows", r0 + STRIP, flush=True)
    os.makedirs(C.OUT, exist_ok=True)
    with rasterio.open(os.path.join(C.OUT, "elev.tif"), "w", **p) as out:
        out.write(elev, 1)

    slope = np.zeros((C.H, C.W), np.uint16)
    dy = R_EARTH * np.radians(C.RES)
    for r0 in range(0, C.H, STRIP):
        r1 = min(r0 + STRIP, C.H)
        z = elev[np.clip(np.arange(r0 - 1, r1 + 1), 0, C.H - 1)].astype(np.float32)
        z[z == NODATA] = 0
        lat = C.NORTH - (np.arange(r0, r1) + 0.5) * C.RES
        dx = (dy * np.cos(np.radians(lat)))[:, None]
        zc = z[1:-1]
        gx = (np.roll(zc, -1, axis=1) - np.roll(zc, 1, axis=1)) / (2 * dx)
        gy = (z[2:] - z[:-2]) / (2 * dy)
        s = np.degrees(np.arctan(np.hypot(gx, gy)))
        slope[r0:r0 + zc.shape[0]] = np.rint(s * 100).astype(np.uint16)
    with rasterio.open(os.path.join(C.OUT, "slope.tif"), "w", **C.profile("uint16")) as out:
        out.write(slope, 1)
    print("done %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
