"""Cut the reprojected population grid into XYZ raster tiles.

512 px tiles on the standard XYZ grid, so the MapLibre source wants
tileSize: 512.  prep.py aligned the GeoTIFF to the z10 tile grid and gave it an
averaged overview pyramid, so every tile here is a plain window read at an
exact overview level -- no resampling decisions are made in this file.

Tiles whose population is entirely below the ramp floor are not written at all;
MapLibre treats a missing tile as empty and the dark base shows through.
"""

import os

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

import time

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

import ramp

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "tur_pop_3857.tif")
OUT = os.path.join(HERE, "data", "tiles")

MERC_MAX = 20037508.342789244
TILE_PX = 512
MAX_Z = 10
MIN_Z = 2  # below this, scale stops dividing TILE_PX evenly


def main():
    t0 = time.time()
    written = empty = 0
    total_bytes = 0

    with rasterio.open(SRC) as d:
        res = d.transform.a
        tile_m = TILE_PX * res
        tx0 = int(round((d.transform.c + MERC_MAX) / tile_m))
        ty0 = int(round((MERC_MAX - d.transform.f) / tile_m))
        n_tx, n_ty = d.width // TILE_PX, d.height // TILE_PX
        print(f"grid at z{MAX_Z}: x {tx0}..{tx0 + n_tx - 1}, y {ty0}..{ty0 + n_ty - 1}")

        for z in range(MIN_Z, MAX_Z + 1):
            scale = 1 << (MAX_Z - z)
            zw = zh = 0
            for x in range(tx0 // scale, (tx0 + n_tx - 1) // scale + 1):
                col_off = (x * scale - tx0) * TILE_PX
                c0, c1 = max(col_off, 0), min(col_off + TILE_PX * scale, d.width)
                if c1 <= c0:
                    continue
                for y in range(ty0 // scale, (ty0 + n_ty - 1) // scale + 1):
                    row_off = (y * scale - ty0) * TILE_PX
                    r0, r1 = max(row_off, 0), min(row_off + TILE_PX * scale, d.height)
                    if r1 <= r0:
                        continue

                    sub = d.read(
                        1,
                        window=Window(c0, r0, c1 - c0, r1 - r0),
                        out_shape=((r1 - r0) // scale, (c1 - c0) // scale),
                    )
                    if sub.max() <= ramp.LO:
                        empty += 1
                        continue

                    grid = np.zeros((TILE_PX, TILE_PX), dtype="float32")
                    oc, orow = (c0 - col_off) // scale, (r0 - row_off) // scale
                    grid[orow : orow + sub.shape[0], oc : oc + sub.shape[1]] = sub

                    path = os.path.join(OUT, str(z), str(x))
                    os.makedirs(path, exist_ok=True)
                    path = os.path.join(path, f"{y}.png")
                    Image.fromarray(ramp.colorize(grid), "RGBA").save(path)
                    total_bytes += os.path.getsize(path)
                    written += 1
                    zw += 1
                zh += 1
            print(f"  z{z:<2} {zw:5d} tiles  {time.time() - t0:6.1f}s", flush=True)

    print(f"wrote {written} tiles ({total_bytes / 1e6:.1f} MB), skipped {empty} empty")
    print(f"tiles at {OUT}\\{{z}}\\{{x}}\\{{y}}.png   zoom {MIN_Z}-{MAX_Z}, {TILE_PX} px")


if __name__ == "__main__":
    main()
