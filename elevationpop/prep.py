"""Reproject the WorldPop grid to Web Mercator, aligned to the XYZ tile grid.

Output is a float32 GeoTIFF whose pixel edges fall exactly on z10 tile
boundaries (512 px tiles), with an average-resampled overview pyramid down to
z0.  tiles.py then reads a window per tile and never has to resample itself.

Cell values stay "people per source cell" all the way down the pyramid: the
warp is nearest (the destination is finer than the source, so it just
replicates) and the overviews are averaged.  A pixel at any zoom therefore
means "mean people per 100 m cell over the area this pixel covers", which is a
density, and one legend stays honest at every zoom.
"""

import os

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

import math
import time

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "tur_pop.tif")
DST = os.path.join(HERE, "data", "tur_pop_3857.tif")

MERC_MAX = 20037508.342789244
TILE_PX = 512
MAX_Z = 10

RES = 2 * MERC_MAX / (2**MAX_Z * TILE_PX)  # metres per pixel at max zoom
TILE_M = TILE_PX * RES  # tile edge in metres at max zoom


def snap(lo, hi):
    """Widen [lo, hi] out to whole z10 tile boundaries."""
    i0 = math.floor((lo + MERC_MAX) / TILE_M)
    i1 = math.ceil((hi + MERC_MAX) / TILE_M)
    return i0 * TILE_M - MERC_MAX, i1 * TILE_M - MERC_MAX


def main():
    t0 = time.time()
    with rasterio.open(SRC) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:3857", *src.bounds)
        x0, x1 = snap(west, east)
        y0, y1 = snap(south, north)

        width = int(round((x1 - x0) / RES))
        height = int(round((y1 - y0) / RES))
        transform = Affine(RES, 0.0, x0, 0.0, -RES, y1)

        print(f"source      {src.width} x {src.height}  {src.crs}")
        print(f"destination {width} x {height}  EPSG:3857 @ {RES:.4f} m/px")
        print(f"tile grid   z{MAX_Z}, {TILE_PX} px, {width // TILE_PX} x {height // TILE_PX} tiles")

        vrt_opts = dict(
            crs="EPSG:3857",
            transform=transform,
            width=width,
            height=height,
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            nodata=0.0,
        )

        profile = dict(
            driver="GTiff",
            dtype="float32",
            count=1,
            width=width,
            height=height,
            crs="EPSG:3857",
            transform=transform,
            nodata=0.0,
            tiled=True,
            blockxsize=TILE_PX,
            blockysize=TILE_PX,
            compress="deflate",
            predictor=2,
            BIGTIFF="YES",
        )

        with WarpedVRT(src, **vrt_opts) as vrt, rasterio.open(DST, "w", **profile) as dst:
            blocks = list(dst.block_windows(1))
            for n, (_, window) in enumerate(blocks, 1):
                dst.write(vrt.read(1, window=window), 1, window=window)
                if n % 100 == 0 or n == len(blocks):
                    print(f"  warped {n}/{len(blocks)} blocks  {time.time() - t0:6.1f}s", flush=True)

    factors = [2**k for k in range(1, MAX_Z + 1)]
    print(f"building overviews {factors}")
    with rasterio.open(DST, "r+") as dst:
        dst.build_overviews(factors, Resampling.average)

    size = os.path.getsize(DST) / 1e6
    print(f"done  {DST}  {size:.1f} MB  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
