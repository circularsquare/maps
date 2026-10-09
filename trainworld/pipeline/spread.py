"""T-013: spread each census block's people and jobs over the H3 cells its polygon covers, by land
area, instead of putting all of it in the cell holding the block's internal point.

Each block is rasterized on a 50 m grid aligned with the water mask (pixel centre inside the
polygon); a pixel weighs its land share from the 25 m water mask, and each pixel goes to the res-9
cell holding its centre. A block that catches no pixel centre (smaller than ~50 m across) keeps
its internal point; one whose pixels are all water is spread over them unweighted.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import time

import h3
import numpy as np
import shapefile
import shapely
from shapely.geometry import shape

import water

# Per-county block polygons from the 2020 redistricting (PL 94-171) TIGER release; the same 2020
# blocks as TABBLOCK20, but one small zip per county instead of one per state (326 MB for NY).
TIGER_PL_BLOCKS = "https://www2.census.gov/geo/tiger/TIGER2020PL/LAYER/TABBLOCK/2020/tl_2020_{fips}_tabblock20.zip"


def block_polygons(fips, fetch, want):
    """{GEOID20: shapely polygon (lon/lat)} for the blocks of one county that are in `want`."""
    r = shapefile.Reader(fetch(TIGER_PL_BLOCKS.format(fips=fips)))
    fields = [f[0] for f in r.fields[1:]]
    gi = fields.index("GEOID20")
    out = {}
    for sr in r.iterShapeRecords():
        g = sr.record[gi]
        if g in want and sr.shape.points:
            out[g] = shapely.make_valid(shape(sr.shape.__geo_interface__))
    return out


def spread(df, counties, fetch, mask, lon0, lat0, cell=50.0, cols=("pop", "jobs")):
    """df: one row per block with geoid, lat, lon (internal point) and the value columns `cols`.
    Returns (h3 ids uint64, {col: per-cell values}, stats)."""
    t0 = time.time()
    f = int(round(cell / mask.cell))
    assert abs(f * mask.cell - cell) < 1e-6, "block grid must be a multiple of the water pixel"
    land = 1.0 - mask.dense(f)  # (H // f, W // f), share of land per block pixel
    H, W = land.shape
    x0, y0 = mask.x0, mask.y0
    want = set(df["geoid"])
    geoms, gids = [], []
    for fips in sorted(counties):
        polys = block_polygons(fips, fetch, want)
        for g, p in polys.items():
            geoms.append(water.to_local(p, lon0, lat0))
            gids.append(g)
    print(f"  block polygons: {len(geoms):,} of {len(want):,} blocks, {time.time() - t0:.0f} s", flush=True)
    idx = {g: i for i, g in enumerate(df["geoid"])}
    bidx = np.array([idx[g] for g in gids], np.int64)
    x1, y1, x2, y2, gid = water.polygon_edges(geoms)
    gid = bidx[gid]
    b, row, c0, c1 = water.scanline_runs(x1, y1, x2, y2, gid, x0, y0, cell, W, H)
    n = c1 - c0
    pb = np.repeat(b, n)
    pr = np.repeat(row, n)
    start = np.cumsum(n) - n
    pc = np.repeat(c0, n) + (np.arange(int(n.sum())) - np.repeat(start, n))
    wgt = land[pr, pc].astype(np.float64)
    nb = len(df)
    wsum = np.bincount(pb, weights=wgt, minlength=nb)
    npix = np.bincount(pb, minlength=nb)
    all_water = (npix > 0) & (wsum == 0)
    wgt[all_water[pb]] = 1.0
    wsum = np.bincount(pb, weights=wgt, minlength=nb)
    print(f"  {len(pb):,} pixels of {cell:.0f} m; blocks with no pixel centre {int((npix == 0).sum()):,}, "
          f"all water {int(all_water.sum()):,}; {time.time() - t0:.0f} s", flush=True)
    # cell of each pixel centre (pixels of one block never repeat, but blocks can share edges)
    lon, lat = water.to_lonlat(x0 + (pc + 0.5) * cell, y0 + (pr + 0.5) * cell, lon0, lat0)
    key = pr * W + pc
    ukey, inv = np.unique(key, return_inverse=True)
    first = np.zeros(len(ukey), np.int64)
    first[inv] = np.arange(len(key))
    ulat, ulon = lat[first], lon[first]
    ucell = np.fromiter((h3.str_to_int(h3.latlng_to_cell(a, o, 9)) for a, o in zip(ulat, ulon)),
                        dtype=np.uint64, count=len(ukey))
    pcell = ucell[inv]
    print(f"  h3 for {len(ukey):,} pixel centres, {time.time() - t0:.0f} s", flush=True)
    share = wgt / np.where(wsum[pb] > 0, wsum[pb], 1.0)
    rest = np.nonzero(npix == 0)[0]
    rcell = np.fromiter((h3.str_to_int(h3.latlng_to_cell(a, o, 9))
                         for a, o in zip(df["lat"].to_numpy()[rest], df["lon"].to_numpy()[rest])),
                        dtype=np.uint64, count=len(rest))
    ids = np.concatenate([pcell, rcell])
    uniq, inv2 = np.unique(ids, return_inverse=True)
    out = {}
    for c in cols:
        v = df[c].to_numpy(np.float64)
        vals = np.concatenate([v[pb] * share, v[rest]])
        out[c] = np.bincount(inv2, weights=vals, minlength=len(uniq))
        assert abs(out[c].sum() - v.sum()) < 1e-3 * max(1, v.sum()), c
    stats = {"blocks": nb, "blocks_spread": int((npix > 0).sum()), "blocks_point": int(len(rest)),
             "blocks_all_water": int(all_water.sum()), "pixels": int(len(pb)), "seconds": round(time.time() - t0, 1)}
    return uniq, out, stats
