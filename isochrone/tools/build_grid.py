"""Shrink the walking-only friction surface to a coarse grid for the page.

Input: MAP 2020 walking-only friction surface, 43200 x 17400 pixels of ~1 km,
EPSG:4326, 85N to 60S, minutes per metre. Exactly 0.06 (1 km/h) is water.

Output: data/grid.bin.gz, uint8, row-major, FACTOR-times coarser:
  0       water or no land worth walking on
  1..255  walking speed in km/h times SCALE

A coarse cell's speed comes from the mean of the land pixels' friction (time per
distance averages correctly across a stretch of ground), turned back into km/h.
A cell counts as land when at least MIN_LAND of its pixels are land. Water patches
under 5 km2 count as land, at the 0.06 they have (smallwater.py).
"""
import gzip
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import rasterio

from smallwater import SmallWater, STRIP

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "data", "raw", "unzipped", "2020_walking_only_friction_surface.geotiff")
OUT = os.path.join(HERE, "data", "grid.bin.gz")
META = os.path.join(HERE, "data", "grid.json")

FACTOR = 10
SCALE = 40          # stored value = km/h * 40, so 5 km/h -> 200
MIN_LAND = 0.25
WATER = np.float32(0.06)


def main():
    with rasterio.open(SRC) as ds:
        w, h = ds.width, ds.height
        assert w % FACTOR == 0 and h % FACTOR == 0 and STRIP % FACTOR == 0
        gw, gh = w // FACTOR, h // FACTOR
        out = np.zeros((gh, gw), dtype=np.uint8)
        sw = SmallWater(ds, log=lambda *a: None)
        print("small water: %d patches (%.0f km2) become land" % (sw.small_components, sw.small_area_km2), flush=True)
        for r0 in range(0, h, STRIP):
            rows = min(STRIP, h - r0)
            a = ds.read(1, window=((r0, r0 + rows), (0, w)))
            land = ((a != WATER) & (a > 0)) | sw.mask(r0 // STRIP, a)
            f = np.where(land, a, 0).astype(np.float64)
            # block sums
            fs = f.reshape(rows // FACTOR, FACTOR, gw, FACTOR).sum(axis=(1, 3))
            ls = land.reshape(rows // FACTOR, FACTOR, gw, FACTOR).sum(axis=(1, 3))
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_f = fs / ls
                kmh = 0.06 / mean_f
            ok = ls >= MIN_LAND * FACTOR * FACTOR
            val = np.clip(np.rint(kmh * SCALE), 1, 255)
            val[~ok] = 0
            out[r0 // FACTOR:(r0 + rows) // FACTOR] = np.nan_to_num(val).astype(np.uint8)
            print("rows", r0 + rows, "/", h, flush=True)

    with gzip.open(OUT, "wb", compresslevel=9) as f:
        f.write(out.tobytes())
    meta = {"width": gw, "height": gh, "west": -180.0, "east": 180.0, "north": 85.0,
            "south": -60.0, "scale": SCALE, "factor": FACTOR}
    with open(META, "w") as f:
        json.dump(meta, f)
    land = out > 0
    print("grid", gw, "x", gh, "land share", land.mean(), "gz bytes", os.path.getsize(OUT))
    print("land km/h percentiles 1,10,50,90,99:",
          np.percentile(out[land] / SCALE, [1, 10, 50, 90, 99]))


if __name__ == "__main__":
    main()
