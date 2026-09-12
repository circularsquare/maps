"""Recover 2020 census population per township by zonal-summing the ASPECT grid.

ASPECT (Ju et al. 2025, Sci Data, CC BY 4.0) takes the township counts printed in
"Tabulation on the 2020 China Population Census by Township" and spreads each
one across 100 m cells by dasymetric mapping. The spreading is mass-preserving,
so summing the grid back over a township returns that township's census count.
Doing it this way avoids joining 43,655 Chinese name tuples to a book we cannot
download, and it degrades gracefully where our 2018-vintage township boundaries
disagree with the 2020 ones: people are counted where the grid says they live.

Grid values are documented as persons per hectare, but the grid is EPSG:4326
with square 0.00089832 degree cells, whose true ground area shrinks with
cos(latitude). Weighting each cell by its true area totals 1,187 M against a
census total of 1,411 M — short by exactly the population-weighted mean of
1/cos(latitude). So the values are persons per nominal hectare, one per cell,
and the recovery is a plain sum. TRUE_CELL_AREA keeps the other reading
available for anyone re-checking that.

Usage:
    python zonal_pop.py                      # all of mainland China
    python zonal_pop.py --provinces 01 02    # by adm1 code, for a quick test

Writes helper1m/data/china/township_pop2020.csv (code, pop_2020).
"""
import os

# Cap BLAS threads before numpy loads — she is using the box.
os.environ.setdefault("OMP_NUM_THREADS", "6")

import argparse
import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

REPO_ROOT = Path(__file__).resolve().parents[3]
RASTER = REPO_ROOT / "data/asia1m/china/aspect_population_total_pop.tif"
ADM4 = REPO_ROOT / "helper1m/data/china/boundaries/adm4.gpkg"
OUT = REPO_ROOT / "helper1m/data/china/township_pop2020.csv"

BLOCK_ROWS = 512

# Authalic radius of WGS84 — cell areas from it are within ~0.1%.
R_AUTHALIC = 6371007.181

# One cell is one hectare by definition of the grid's units (see the note above).
# Flip this to True to weight by true ground area instead, which undercounts.
TRUE_CELL_AREA = False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def row_areas_ha(transform, row0, nrows):
    """True area of one cell in hectares, per raster row.

    On a sphere the area between two parallels over a longitude span dlon is
    R^2 * dlon * (sin(lat_top) - sin(lat_bottom)), which is exact rather than
    the usual cos(lat) approximation.
    """
    dlon = math.radians(abs(transform.a))
    top = transform.f + row0 * transform.e
    dlat = transform.e  # negative: latitude decreases with row
    lat_top = np.radians(top + np.arange(nrows) * dlat)
    lat_bot = np.radians(top + (np.arange(nrows) + 1) * dlat)
    area_m2 = R_AUTHALIC ** 2 * dlon * (np.sin(lat_top) - np.sin(lat_bot))
    return area_m2 / 1e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provinces", nargs="*", default=None,
                    help="adm1 codes to restrict to (default: all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    log(f"reading {ADM4.name}")
    gdf = gpd.read_file(ADM4)
    if args.provinces:
        gdf = gdf[gdf["group"].isin(args.provinces)].reset_index(drop=True)
        log(f"  restricted to provinces {args.provinces}: {len(gdf)} townships")
    else:
        log(f"  {len(gdf)} townships")

    # Zone ids are 1..N; 0 means "no township", which is how unassigned cells
    # (coast, borders, outside mainland China) are counted and reported.
    gdf = gdf.reset_index(drop=True)
    gdf["zone"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
    bounds = gdf.geometry.bounds.to_numpy()  # minx, miny, maxx, maxy

    with rasterio.open(RASTER) as src:
        log(f"  raster {src.width} x {src.height}, dtype={src.dtypes[0]}, "
            f"nodata={src.nodata}")
        log(f"  pixel {src.transform.a:.8f} x {src.transform.e:.8f} deg")
        transform = src.transform
        nodata = src.nodata

        # Rows each polygon can touch, so a block only rasterizes what overlaps it.
        top, dlat = transform.f, transform.e
        poly_row_max = np.ceil((bounds[:, 1] - top) / dlat).astype(np.int64)   # miny
        poly_row_min = np.floor((bounds[:, 3] - top) / dlat).astype(np.int64)  # maxy
        poly_row_min = np.clip(poly_row_min, 0, src.height)
        poly_row_max = np.clip(poly_row_max + 1, 0, src.height)

        totals = np.zeros(len(gdf) + 1, dtype=np.float64)
        raster_mass = 0.0
        t0 = time.time()
        nblocks = math.ceil(src.height / BLOCK_ROWS)

        for bi, row0 in enumerate(range(0, src.height, BLOCK_ROWS)):
            nrows = min(BLOCK_ROWS, src.height - row0)
            hit = np.where((poly_row_min < row0 + nrows) & (poly_row_max > row0))[0]
            if len(hit) == 0:
                continue

            window = Window(0, row0, src.width, nrows)
            data = src.read(1, window=window)
            if nodata is not None:
                data = np.where(data == nodata, 0.0, data)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            if not data.any():
                continue

            win_transform = rasterio.windows.transform(window, transform)
            zones = rasterize(
                ((geom, z) for geom, z in
                 zip(gdf.geometry.values[hit], gdf["zone"].values[hit])),
                out_shape=(nrows, src.width),
                transform=win_transform,
                fill=0,
                dtype=np.int32,
            )

            if TRUE_CELL_AREA:
                areas = row_areas_ha(transform, row0, nrows).astype(np.float64)
                counts = data.astype(np.float64) * areas[:, None]
            else:
                counts = data.astype(np.float64)
            raster_mass += float(counts.sum())
            totals += np.bincount(zones.ravel(), weights=counts.ravel(),
                                  minlength=len(gdf) + 1)

            if bi % 10 == 0 or bi == nblocks - 1:
                el = time.time() - t0
                log(f"  block {bi + 1}/{nblocks}  rows {row0}-{row0 + nrows}  "
                    f"{len(hit)} polys  {el:.0f}s elapsed, "
                    f"~{el / (bi + 1) * nblocks:.0f}s total")

    assigned = totals[1:].sum()
    log(f"  raster mass in scanned blocks: {raster_mass / 1e6:.2f} M")
    log(f"  assigned to townships:         {assigned / 1e6:.2f} M "
        f"({100 * assigned / raster_mass:.2f}%)")
    log(f"  unassigned (no township under the cell centre): "
        f"{totals[0] / 1e6:.2f} M")

    out = pd.DataFrame({
        "code": gdf["code"].values,
        "name": gdf["name"].values,
        "name_cn": gdf["name_cn"].values,
        "pop_2020": np.round(totals[1:]).astype(np.int64),
    })
    path = Path(args.out) if args.out else OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")
    zero = int((out["pop_2020"] == 0).sum())
    log(f"  wrote {path} — {len(out)} townships, {zero} with zero population")


if __name__ == "__main__":
    main()
