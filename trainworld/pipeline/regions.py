"""T-162: the first-level region of every pack cell, for the region-aware work gravity
(build_city.py and canada.py write it as they build). `pack_gravity` splits zones where a region
border cuts a 2 km square and multiplies the decay between zones of different regions by the
border factor.

A cell's region is the one holding most of what its census blocks put into it (people, jobs and
resident job holders, plus a little for land so every cell has one), spread over the block
polygons exactly as the pack's own figures were (spread.py), so the border follows the blocks.
In the US the region is the state: the first two digits of the block GEOID. Outside the US any
admin-1 layer (geoBoundaries ADM1, OSM admin_level 4) gives the same per-cell id.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SOURCE = "US state: the first two digits of each 2020 census block GEOID; a cell takes the state holding most of its blocks' people, jobs and land"


def add_columns(df):
    """Per region, a column weighting the blocks in it. Returns (df, codes, column names)."""
    codes = sorted(df["geoid"].str[:2].unique())
    assert len(codes) < 256, "region ids are one byte"
    w = df["pop"].to_numpy(np.float64) + df["jobs"].to_numpy(np.float64) + 1e-3
    if "rac" in df:
        w = w + df["rac"].to_numpy(np.float64)
    st = df["geoid"].str[:2].to_numpy()
    cols = [f"region_{c}" for c in codes]
    df = df.assign(**{col: np.where(st == c, w, 0.0) for c, col in zip(codes, cols)})
    return df, codes, cols


def of_cells(vals, cols):
    """Region index per spread cell: the column with the most weight."""
    return np.argmax(np.stack([vals[c] for c in cols], axis=1), axis=1).astype(np.uint8)


def header_block(codes):
    import build_city as bc
    return {"level": "admin-1", "ids": codes, "names": [bc.STATES.get(c, c).upper() for c in codes], "source": SOURCE}
