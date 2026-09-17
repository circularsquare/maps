"""Nationality breakdown for every Chinese admin unit, from the chinaethnicity build.

helper1m draws the shapes and knows the populations; `maps/chinaethnicity/` has the 2020
census county tables by nationality. This carries one onto the other and writes
`countries/china/composition.json`, which the viewer draws as a pie over each shape.

The two use different boundaries: chinaethnicity is on DataV's 2025 county polygons,
helper1m on a 2018 township shapefile dissolved upward. A county whose name matches on both
sides is not the same ground -- urban districts annexed populated fringe in between -- so
this does not join by name. Both sit on the same ASPECT 100 m population grid, so instead
every populated cell is labelled with the helper1m unit it falls in, and each census
county's nationality counts are split between helper1m units in proportion to the people
the grid puts in each piece. A district that took half a neighbouring county's population
takes half its nationalities with it.

Levels 2 and 1 are summed from level 3 through helper1m's own `parent_code`, so they agree
with the level below them whatever the boundaries did. Level 4, township, gets nothing: no
source publishes nationality below the county, and a township pie would only be repeating
its county's.

Each unit also gets the mean position of its people, so the viewer can put the pie where
the population is rather than in the middle of the shape.

Reads:
    ../../../chinaethnicity/data/processed/units.json   county breakdowns (chinaethnicity/units.py)
    ../../../chinaethnicity/data/geo/cells.npz          the ASPECT cells, with their county
    ../../../chinaethnicity/colors.csv                  the hand-edited palette
    ../../countries/china/adm3.geojson                  helper1m's counties

Writes:
    ../../countries/china/composition.json

Usage:
    python scripts/china/ethnicity.py        # from helper1m/
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")

import csv
import json
import sys
import time

import numpy as np
import rasterio.features
from affine import Affine

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.dirname(os.path.dirname(HERE))
MAPS = os.path.dirname(HELPER)
CE = os.path.join(MAPS, "chinaethnicity")

UNITS = os.path.join(CE, "data", "processed", "units.json")
CELLS = os.path.join(CE, "data", "geo", "cells.npz")
COLORS = os.path.join(CE, "colors.csv")
COUNTRY = os.path.join(HELPER, "countries", "china")
ADM3 = os.path.join(COUNTRY, "adm3.geojson")
OUT = os.path.join(COUNTRY, "composition.json")

STRIP = 1024            # rasterise this many grid rows at a time, to keep memory small
LEVELS = [3, 2, 1]      # county, prefecture, province; township has no source


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_groups():
    """The census's column order and the palette, taken from chinaethnicity."""
    sys.path.insert(0, CE)
    from common import GROUPS
    colors = {}
    if os.path.exists(COLORS):
        with open(COLORS, encoding="utf-8", newline="") as fh:
            colors = {r["key"]: r["color"] for r in csv.DictReader(fh)}
    else:
        log("!! no chinaethnicity/colors.csv: every nationality will be grey")
    return [{"key": k, "en": en, "cn": cn, "color": colors.get(k, "#888888")}
            for k, en, cn in GROUPS]


def label_cells(geo, row, col, transform):
    """Index into geo['features'] for every cell, or -1 where no unit covers it."""
    shapes = [(f["geometry"], i) for i, f in enumerate(geo["features"])]
    out = np.full(len(row), -1, dtype=np.int32)
    width = int(col.max()) + 1
    order = np.argsort(row, kind="stable")
    r_sorted = row[order]
    for r0 in range(0, int(row.max()) + 1, STRIP):
        a = np.searchsorted(r_sorted, r0, "left")
        b = np.searchsorted(r_sorted, r0 + STRIP, "left")
        if a == b:
            continue
        band = rasterio.features.rasterize(
            shapes, out_shape=(STRIP, width), fill=-1, dtype=np.int32,
            transform=transform * Affine.translation(0, r0))
        idx = order[a:b]
        out[idx] = band[row[idx] - r0, col[idx]]
    return out


def main():
    groups = load_groups()
    n_groups = len(groups)

    with open(UNITS, encoding="utf-8") as fh:
        counties = json.load(fh)["levels"]["county"]
    census = np.zeros((len(counties), n_groups))
    for i, u in enumerate(counties):
        census[i, u["g"]] = u["k"]
    census_of = {u["c"]: i for i, u in enumerate(counties)}

    C = np.load(CELLS)
    row = C["row"].astype(np.int64)
    col = C["col"].astype(np.int64)
    pop = C["pop"].astype(np.float64)
    adcodes = C["adcodes"].astype(str)
    # cells.npz numbers counties by position in `adcodes`; map that to the units.json row,
    # with -1 for a county units.json has no people for (Nansha, Kinmen).
    to_census = np.array([census_of.get(a, -1) for a in adcodes], dtype=np.int64)
    cell_census = to_census[C["county"].astype(np.int64)]
    transform = Affine(*C["transform"])
    log(f"{len(row):,} populated cells, {len(counties):,} census counties")

    with open(ADM3, encoding="utf-8") as fh:
        adm3 = json.load(fh)
    log(f"labelling cells with {len(adm3['features']):,} helper1m counties…")
    cell_unit = label_cells(adm3, row, col, transform)
    lost = pop[cell_unit < 0].sum()
    log(f"  {lost:,.0f} people ({lost / pop.sum() * 100:.2f}%) fall outside helper1m's "
        f"counties, which are mainland only")

    # Each census county's counts split between helper1m units by the grid population of
    # each piece. Both labels are needed, so drop cells missing either.
    keep = (cell_unit >= 0) & (cell_census >= 0)
    u_idx, c_idx, w = cell_unit[keep], cell_census[keep], pop[keep]
    n_units = len(adm3["features"])
    pair = c_idx * n_units + u_idx
    uniq, inv = np.unique(pair, return_inverse=True)
    part = np.bincount(inv, weights=w, minlength=len(uniq))
    whole = np.bincount(c_idx, weights=w, minlength=len(counties))
    pc, pu = np.divmod(uniq, n_units)
    share = np.where(whole[pc] > 0, part / np.maximum(whole[pc], 1e-9), 0.0)

    level3 = np.zeros((n_units, n_groups))
    np.add.at(level3, pu, census[pc] * share[:, None])
    log(f"{len(uniq):,} (census county, helper1m county) pieces, "
        f"{level3.sum():,.0f} of {census.sum():,.0f} people placed")

    # Mean position of each unit's people, on the same grid.
    lon = transform.c + (col + 0.5) * transform.a
    lat = transform.f + (row + 0.5) * transform.e
    ok = cell_unit >= 0
    wsum = np.bincount(cell_unit[ok], weights=pop[ok], minlength=n_units)
    cx = np.bincount(cell_unit[ok], weights=pop[ok] * lon[ok], minlength=n_units)
    cy = np.bincount(cell_unit[ok], weights=pop[ok] * lat[ok], minlength=n_units)

    codes = [f["properties"]["code"] for f in adm3["features"]]
    parents = [f["properties"].get("parent_code") for f in adm3["features"]]
    tables = {3: (codes, level3, cx, cy, wsum)}
    for lvl in (2, 1):
        below_codes, below, bcx, bcy, bw = tables[lvl + 1]
        # helper1m's codes nest as prefixes, so a level's parent is its code truncated;
        # parent_code is only carried on the level below, so walk it once and reuse.
        up = parents if lvl == 2 else [c[:2] for c in below_codes]
        keys = sorted(set(up))
        at = {k: i for i, k in enumerate(keys)}
        j = np.array([at[k] for k in up])
        agg = np.zeros((len(keys), n_groups))
        np.add.at(agg, j, below)
        tables[lvl] = (keys, agg,
                       np.bincount(j, weights=bcx, minlength=len(keys)),
                       np.bincount(j, weights=bcy, minlength=len(keys)),
                       np.bincount(j, weights=bw, minlength=len(keys)))

    levels = {}
    for lvl in LEVELS:
        keys, table, ax, ay, aw = tables[lvl]
        out = {}
        for i, code in enumerate(keys):
            v = np.rint(table[i]).astype(np.int64)
            g = np.flatnonzero(v > 0)
            if not len(g) or aw[i] <= 0:
                continue
            g = g[np.argsort(-v[g], kind="stable")]
            out[code] = {"t": int(v.sum()),
                         "x": round(float(ax[i] / aw[i]), 4),
                         "y": round(float(ay[i] / aw[i]), 4),
                         "g": [int(k) for k in g], "k": [int(v[k]) for k in g]}
        levels[str(lvl)] = out
        log(f"  level {lvl}: {len(out):,} of {len(keys):,} units, "
            f"{sum(u['t'] for u in out.values()):,} people")

    doc = {
        "label": "Nationality",
        "year": 2020,
        "levels": levels,
        "groups": groups,
        "source": "2020 census, population by region, sex and nationality, from each "
                  "province's own census yearbook where it is published (16 provinces) and "
                  "estimated from the 2000 county pattern elsewhere. Carried onto these "
                  "boundaries through the ASPECT 100 m population grid. See "
                  "maps/chinaethnicity.",
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, separators=(",", ":"))
    log(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
