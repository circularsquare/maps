"""A ~500 m population grid over mainland China, with a county on every cell, for placing dots.

ASPECT (Ju et al. 2025, Scientific Data, CC BY 4.0, figshare doi:10.6084/m9.figshare.27323106)
takes the 2020 census township counts and spreads each over 100 m cells by dasymetric
mapping, so it is the same census as the tables this map draws, placed at township grain.
That makes it a much better placement weight than a modelled surface: inside a county, dots
go where the 2020 census's own townships put people.

The 100 m grid is 68,573 x 39,410 cells (10.8 GB as float32), far finer than a dot needs.
This sums it into FACTOR x FACTOR blocks, reading straight out of the zip (GDAL's /vsizip/,
sequential rows, so it never needs unpacking), then burns the county polygons onto the same
grid so every populated cell knows its county.

Units: the grid's values are persons per NOMINAL hectare, one per cell, so a plain sum is a
count. helper1m/scripts/china/zonal_pop.py explains why reading them as true-area densities
comes out 19% short.

Writes data/geo/cells.npz:
    row, col      int16   cell position on the coarse grid
    pop           float32 people in the cell (ASPECT, 2020)
    county        int16   index into `adcodes` (the order of data/geo/counties.gpkg)
    adcodes       str     county adcodes
    transform     float64 the coarse grid's affine (a, b, c, d, e, f)
    fb_row, fb_col, fb_county   every cell of any county that has NO populated cell, so
                  its people can still be spread uniformly (scatter.py's fallback)

Usage:
    python aspect.py              # ~2-4 min
    python aspect.py --factor 10  # ~1 km cells
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")

import argparse
import math
import sys
import time

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

from common import ASPECT_ZIP, GEO

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MEMBER = "population_total_pop.tif"
COUNTIES = os.path.join(GEO, "counties.gpkg")
OUT = os.path.join(GEO, "cells.npz")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=5)
    args = ap.parse_args()
    F = args.factor

    path = "/vsizip/" + ASPECT_ZIP.replace("\\", "/") + "/" + MEMBER
    with rasterio.open(path) as src:
        W, H = src.width, src.height
        t = src.transform
        log(f"ASPECT {W} x {H}, pixel {t.a:.8f} deg; summing {F}x{F} blocks")
        w, h = math.ceil(W / F), math.ceil(H / F)
        out_t = t * rasterio.Affine.scale(F)
        pop = np.zeros((h, w), dtype=np.float32)
        step = 100 * F
        mass = 0.0
        t0 = time.time()
        nwin = math.ceil(H / step)
        for wi, row0 in enumerate(range(0, H, step)):
            nrows = min(step, H - row0)
            a = src.read(1, window=Window(0, row0, W, nrows))
            a[~np.isfinite(a)] = 0
            np.maximum(a, 0, out=a)
            a = a.astype(np.float64)
            mass += float(a.sum())
            ph, pw = math.ceil(nrows / F) * F, w * F
            if a.shape != (ph, pw):
                a = np.pad(a, ((0, ph - a.shape[0]), (0, pw - a.shape[1])))
            blk = a.reshape(ph // F, F, w, F).sum(axis=(1, 3))
            pop[row0 // F: row0 // F + blk.shape[0]] = blk
            if wi % 10 == 0 or wi == nwin - 1:
                el = time.time() - t0
                log(f"  window {wi + 1}/{nwin}  {el:.0f}s elapsed, "
                    f"~{el / (wi + 1) * nwin:.0f}s total")
    log(f"grid mass {mass / 1e6:,.2f} M people; coarse grid {w} x {h}")

    log("burning county polygons onto the grid…")
    counties = gpd.read_file(COUNTIES).sort_values("adcode").reset_index(drop=True)
    if len(counties) > 32000:
        raise SystemExit("too many counties for int16 indices")
    cid = rasterize(((geom, i + 1) for i, geom in enumerate(counties.geometry)),
                    out_shape=(h, w), transform=out_t, fill=0, dtype="int16")

    rows, cols = np.nonzero(pop > 0)
    county = cid[rows, cols]
    cpop = pop[rows, cols]
    outside = county == 0
    log(f"populated cells {len(rows):,}; outside every county {int(outside.sum()):,} "
        f"holding {cpop[outside].sum() / 1e6:.2f} M people "
        f"({cpop[outside].sum() / cpop.sum():.2%}) — coast, borders and water")
    keep = ~outside
    rows, cols, county, cpop = rows[keep], cols[keep], county[keep] - 1, cpop[keep]

    have = np.zeros(len(counties), dtype=bool)
    have[np.unique(county)] = True
    fb_r, fb_c, fb_k = [], [], []
    for i in np.nonzero(~have)[0]:
        r, c = np.nonzero(cid == i + 1)
        if not len(r):
            # smaller than one cell: use the cell under its representative point
            pt = counties.geometry.iloc[i].representative_point()
            cc, rr = ~out_t * (pt.x, pt.y)
            r, c = np.array([int(rr)]), np.array([int(cc)])
        fb_r.append(r)
        fb_c.append(c)
        fb_k.append(np.full(len(r), i))
    if fb_r:
        names = [f"{counties.adcode.iloc[i]} {counties.name.iloc[i]}" for i in np.nonzero(~have)[0]]
        log(f"{len(names)} counties have no populated cell; uniform fallback: {names[:12]}")

    os.makedirs(GEO, exist_ok=True)
    np.savez_compressed(
        OUT, row=rows.astype(np.int16), col=cols.astype(np.int16), pop=cpop.astype(np.float32),
        county=county.astype(np.int16), adcodes=counties["adcode"].to_numpy().astype(str),
        transform=np.array(out_t[:6], dtype=np.float64),
        fb_row=np.concatenate(fb_r).astype(np.int16) if fb_r else np.zeros(0, np.int16),
        fb_col=np.concatenate(fb_c).astype(np.int16) if fb_c else np.zeros(0, np.int16),
        fb_county=np.concatenate(fb_k).astype(np.int16) if fb_k else np.zeros(0, np.int16))
    log(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
