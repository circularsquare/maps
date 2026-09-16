"""Turn county counts into dots: one dot per DOT_VALUE people, placed on the ASPECT grid.

For each county row the census gives a count per nationality, and the row covers one or
more county polygons (join.py). The estimated provinces' rows come from a second file
(fallback.py) in the same shape, with counts that are not whole numbers. Inside those
polygons the people are spread over the ~500 m
ASPECT cells in proportion to each cell's 2020 population (aspect.py). Every nationality is
spread by the same weights: the table says how many of each group live in the county, not
where in the county, so nothing finer is claimed.

Whole dots come from a carry along a Hilbert curve, never from ranking (Anita's rule from
religiondots): for each nationality, walk every (row, cell) pair in Hilbert order,
accumulate expected people, and drop a dot in the cell where the running total passes the
next multiple of DOT_VALUE. So a cell holding a third of a dot's worth gets a dot about a
third of the time, next to the people who earned it, and the national count of every group
is exact to under one dot.

A dot lands at a uniformly random point inside its cell.

Writes:
    data/processed/dots_2020.npz   lon, lat (float64), g (group index), p (province index)
    data/processed/legend.json     groups with colours and totals, provinces with status

Usage:
    python scatter.py
    python scatter.py --dot-value 100
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")

import argparse
import csv
import json
import sys
import time

import numpy as np
import pandas as pd

from common import GEO, GROUPS, HERE, KEYS, PROC, PROVINCES, WORK

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LEAVES = os.path.join(WORK, "leaves_2020.csv")
FALLBACK = os.path.join(WORK, "leaves_fallback.csv")    # fallback.py's estimated provinces
CELLS = os.path.join(GEO, "cells.npz")
COLORS = os.path.join(HERE, "colors.csv")
DOT_VALUE = 200
SEED = 20260914
PCODES = sorted(PROVINCES)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hilbert_d(x, y, order=16):
    """Distance along a Hilbert curve for integer grid positions (vectorised xy2d)."""
    n = 1 << order
    x = x.astype(np.int64).copy()
    y = y.astype(np.int64).copy()
    d = np.zeros(len(x), dtype=np.int64)
    s = n >> 1
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx.astype(np.int64)) ^ ry.astype(np.int64))
        flip = ~ry & rx
        x = np.where(flip, n - 1 - x, x)
        y = np.where(flip, n - 1 - y, y)
        swap = ~ry
        x, y = np.where(swap, y, x), np.where(swap, x, y)
        s >>= 1
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dot-value", type=int, default=DOT_VALUE)
    args = ap.parse_args()
    DV = args.dot_value
    rng = np.random.default_rng(SEED)

    leaves = pd.read_csv(LEAVES, dtype={"prov": str, "adcodes": str})
    measured = set(leaves["prov"])
    if os.path.exists(FALLBACK):
        fb = pd.read_csv(FALLBACK, dtype={"prov": str, "adcodes": str})
        leaves = pd.concat([leaves, fb[~fb["prov"].isin(measured)]], ignore_index=True)
    else:
        log("no leaves_fallback.csv: only the measured provinces are drawn")
    C = np.load(CELLS)
    n_pop = len(C["row"])
    row = np.concatenate([C["row"], C["fb_row"]]).astype(np.int64)
    col = np.concatenate([C["col"], C["fb_col"]]).astype(np.int64)
    wgt = np.concatenate([C["pop"].astype(np.float64), np.zeros(len(C["fb_row"]))])
    cty = np.concatenate([C["county"], C["fb_county"]]).astype(np.int64)
    is_fb = np.arange(len(row)) >= n_pop
    code_idx = {a: i for i, a in enumerate(C["adcodes"].astype(str))}
    log(f"{len(leaves):,} county rows, {n_pop:,} populated cells")

    order = np.argsort(cty, kind="stable")
    bounds = np.searchsorted(cty[order], np.arange(len(code_idx) + 1))

    def cells_of(code):
        ci = code_idx[code]
        return order[bounds[ci]:bounds[ci + 1]]

    def spread(idx, name):
        """ASPECT-weighted cells of a set of counties, or uniform if it has no population."""
        real = idx[~is_fb[idx]]
        if len(real) and wgt[real].sum() > 0:
            return real, wgt[real]
        idx = idx[is_fb[idx]]
        if not len(idx):
            raise SystemExit(f"row {name} has no cells at all")
        log(f"  uniform fallback for {name}")
        return idx, np.ones(len(idx))

    # A row's `adcodes` is one county, several ("a;b", spread over all of them by population),
    # or several with explicit shares ("a:0.6;b:0.4", join.py's development-zone folds, where
    # each county's share is set first and then spread inside it by population).
    E_leaf, E_cell, E_w = [], [], []
    for li, (spec, name) in enumerate(zip(leaves["adcodes"], leaves["name"])):
        parts = [p.split(":") for p in spec.split(";")]
        if all(len(p) == 2 for p in parts):
            idx_l, w_l = [], []
            for code, share in parts:
                i, w = spread(cells_of(code), f"{name} in {code}")
                idx_l.append(i)
                w_l.append(w / w.sum() * float(share))
            idx, w = np.concatenate(idx_l), np.concatenate(w_l)
        else:
            idx, w = spread(np.concatenate([cells_of(p[0]) for p in parts]), name)
        E_leaf.append(np.full(len(idx), li))
        E_cell.append(idx)
        E_w.append(w / w.sum())
    E_leaf = np.concatenate(E_leaf)
    E_cell = np.concatenate(E_cell)
    E_w = np.concatenate(E_w)
    h = hilbert_d(col[E_cell], row[E_cell])
    o = np.lexsort((E_leaf, h))
    E_leaf, E_cell, E_w = E_leaf[o], E_cell[o], E_w[o]
    log(f"{len(E_leaf):,} (row, cell) pairs in Hilbert order")

    counts = leaves[KEYS].to_numpy(dtype=np.float64)
    leaf_prov = np.array([PCODES.index(p) for p in leaves["prov"]])
    d_cell, d_grp, d_prov = [], [], []
    report = []
    for g, key in enumerate(KEYS):
        c = counts[:, g]
        people = c.sum()
        if people == 0:
            report.append((key, 0, 0))
            continue
        exp = c[E_leaf] * E_w
        nz = np.nonzero(exp > 0)[0]
        cum = np.cumsum(exp[nz]) + rng.uniform(0, DV)
        alloc = np.diff(np.concatenate([[0], np.floor(cum / DV).astype(np.int64)]))
        sel = nz[alloc > 0]
        k = alloc[alloc > 0]
        d_cell.append(np.repeat(E_cell[sel], k))
        d_grp.append(np.full(int(k.sum()), g, dtype=np.uint8))
        d_prov.append(np.repeat(leaf_prov[E_leaf[sel]], k).astype(np.uint8))
        report.append((key, people, int(k.sum())))
    cell = np.concatenate(d_cell)
    grp = np.concatenate(d_grp)
    prov = np.concatenate(d_prov)

    a, b, c0, d, e, f = C["transform"]
    lon = c0 + (col[cell] + rng.random(len(cell))) * a
    lat = f + (row[cell] + rng.random(len(cell))) * e
    log(f"{len(cell):,} dots at 1:{DV:,}")
    for key, people, dots in sorted(report, key=lambda r: -r[1])[:12]:
        print(f"    {key:14s} {people:>13,.0f} people  {dots:>9,} dots")

    os.makedirs(PROC, exist_ok=True)
    np.savez(os.path.join(PROC, "dots_2020.npz"), lon=lon, lat=lat, g=grp, p=prov,
             dot_value=np.array(DV))

    colors = {}
    if os.path.exists(COLORS):
        with open(COLORS, encoding="utf-8", newline="") as fh:
            colors = {r["key"]: r["color"] for r in csv.DictReader(fh)}
    else:
        log("!! no colors.csv yet: run palette.py; every group is grey in legend.json")
    people = counts.sum(axis=0)
    drawn = sorted(leaves["prov"].unique())

    def status(code):
        # measured: a published 2020 county table. estimated: fallback.py's 2000 county
        # pattern scaled to 2020 prefecture totals or to the province total.
        if code in measured:
            return {"status": "measured"}
        if code in drawn:
            how = leaves.loc[leaves["prov"] == code, "how"].iloc[0]
            return {"status": "estimated", "scaled_to": how.split("-", 1)[1]}
        return {"status": "queued"}

    legend = {
        "year": 2020,
        "dot_value": DV,
        "groups": [{"key": k, "en": en, "cn": cn, "color": colors.get(k, "#888888"),
                    "people": int(people[i]), "dots": int((grp == i).sum())}
                   for i, (k, en, cn) in enumerate(GROUPS)],
        "provinces": {
            code: {"key": PROVINCES[code][0], "name": PROVINCES[code][1],
                   **status(code),
                   **({"people": int(leaves.loc[leaves["prov"] == code, "total"].sum()),
                       "groups": {k: int(v) for k, v in zip(
                           KEYS, leaves.loc[leaves["prov"] == code, KEYS].sum()) if v}}
                      if code in drawn else {})}
            for code in PCODES},
    }
    with open(os.path.join(PROC, "legend.json"), "w", encoding="utf-8") as fh:
        json.dump(legend, fh, ensure_ascii=False)
    log(f"wrote dots_2020.npz and legend.json")


if __name__ == "__main__":
    main()
