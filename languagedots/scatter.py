"""
Scatter language dots. Country-agnostic; per-country wiring lives in countries.py.

This is religiondots' scatter.py with the religion-only parts taken out (congregations, the
US demographic weights). The allocation is unchanged and its long comments are not repeated:
read religiondots/scatter.py for why.

  * LEFTOVERS ARE SWITCHED OFF (LEFTOVERS below; Anita, 2026-10-08, too heavy for now), so
    every row goes through the carry in the next bullet but one. When on:
  * MEASURED rows (2026-10-08, Anita): a unit's count of a language is drawn as its whole dots,
    floor(count / DOT_VALUE), plus ONE LEFTOVER MARK carrying the remainder at its true weight,
    so 400 speakers draw as 400, never as 0 or 1,000. Until then every row went through the
    carry below, which drew a full dot in a unit holding a few hundred speakers (the "1,000
    Vietnamese on a Minnesota island" complaint). The leftover sits exactly ON one of the unit's
    own dots, picked at random per leftover (any language's), so in pies mode it becomes a
    slice of a pie that is already there rather than a speck too small to hover; random rather
    than one dot per unit, so no pie collects a unit's every small language. A unit with no
    whole dot puts its leftovers on one shared point, placed like a dot. No leftover leaves its
    unit. The trial (us, za, zw) put the archive at 2.2x.
  * DERIVED and MODELLED rows keep the old allocation: each language's dots are carried ALONG A
    HILBERT CURVE through the units, dropping a dot wherever the running total passes the dot
    value. Their per-unit counts are a split of a larger total, and drawing every unit's
    remainder would assert speakers in every unit the split touched.
  * Inside a unit, dots are split across the placement polygons by the same carry, weighted by
    the polygon's population.
  * A language that draws nothing at all (no dot, no leftover: derived or modelled rows under
    one dot) gets ONE ring of its national total, at its largest concentration (see the comment
    at the rings).

Outputs:
    data/processed/dots_<cc>.geojson    one feature per DOT_VALUE people, property n = node
    data/processed/rings_<cc>.geojson   marks of their own weight (`count`): every leftover
                                        (why=leftover) and every national ring (why=under_dot)

Usage:
    python scatter.py --country np
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

import rdlink
from countries import COUNTRIES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
OUT = HERE / "data" / "processed"
DOT_VALUE = 1000
SEED = 20261004
LEFT_MIN = 0.5      # a leftover under half a person is arithmetic (a fractional row), not a speaker
# OFF (Anita, 2026-10-08, after the full build): leftovers made the archive 710 MB, 2.2x, and a
# view at country zoom (z3-6) about twice the download and the marks to draw, which is not worth
# it yet. With this False every row goes through the carry, exactly as before 2026-10-08. The
# docstring's MEASURED bullet describes the switched-on behaviour; one idea for turning it back
# on cheaply is to fold leftovers into their cell's slices in the tiles below about z7.
LEFTOVERS = False


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    for attempt in range(60):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 59:
                raise SystemExit(f"could not replace {path.name}; new output is in {tmp.name}")
            time.sleep(1)


def random_points_in_polygon(geom, n: int, rng) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 2))
    shapely.prepare(geom)
    minx, miny, maxx, maxy = geom.bounds
    out, got = [], 0
    while got < n:
        batch = max(64, int((n - got) * 2.5))
        xs = rng.uniform(minx, maxx, batch)
        ys = rng.uniform(miny, maxy, batch)
        keep = shapely.contains(geom, shapely.points(xs, ys))
        if keep.any():
            out.append(np.column_stack([xs[keep], ys[keep]]))
            got += int(keep.sum())
    return np.vstack(out)[:n]


def read_place(cfg):
    print("reading placement polygons…")
    place = gpd.read_file(cfg["place"])
    print(f"  {len(place):,} placement polygons, crs={place.crs}")
    place["unit"] = cfg["place_unit"](place)
    if place.crs is not None and place.crs.to_epsg() != 4326:
        place = place.to_crs(4326)
    empty = place.geometry.isna() | place.geometry.is_empty
    if empty.any():
        print(f"  !! {int(empty.sum()):,} placement polygons have empty geometry — dropped")
        place = place[~empty]
    return place.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True, choices=sorted(COUNTRIES))
    ap.add_argument("--dot-value", type=int, default=DOT_VALUE)
    ap.add_argument("--no-water", action="store_true")
    args = ap.parse_args()
    cc = args.country
    cfg = COUNTRIES[cc]
    dot_value = args.dot_value
    rng = np.random.default_rng(SEED)

    geo = rdlink.module("geo_checks")
    print(f"country: {cc}")
    df = cfg["counts"]()
    print(f"  {len(df):,} (unit, node) rows, {df['node'].nunique()} nodes, "
          f"{df['unit'].nunique():,} units, {df['count'].sum():,.0f} people")

    place = read_place(cfg)
    geo.check_torn(cc, geo.torn_parts(place))
    kc = rdlink.module("kontur_cap")
    if cfg.get("place_weight") is not None:
        place = kc.apply(place, cc, cfg["place"])
    if not args.no_water:
        place = rdlink.water_clip(place, cc, cfg["place"])

    have, want = set(place["unit"]), set(df["unit"])
    missing = sorted(want - have)
    if missing:
        lost = df[df["unit"].isin(missing)]["count"].sum()
        raise SystemExit(f"{len(missing):,} units in the data have no polygons "
                         f"({lost:,.0f} people): {missing[:6]}")
    extra = len(have - want)
    if extra:
        print(f"  {extra:,} units have polygons but no language rows")

    by_unit = {u: g.index.to_numpy() for u, g in place.groupby("unit")}
    geoms = place.geometry.to_numpy()
    poly_hilbert = place.geometry.hilbert_distance().to_numpy()
    weighter = cfg["place_weight"](place) if cfg.get("place_weight") else None
    if weighter is not None and kc.is_kontur_layer(cfg["place"]):
        geo.check_grid_floor(cc, geo.grid_floor(place))

    if "tier" not in df.columns:
        df["tier"] = "measured"
    rank = {"measured": 0, "derived": 1, "modelled": 2}
    df["tier"] = df["tier"].fillna("measured").map(rank)
    agg = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    agg = agg[agg["count"] > 0].reset_index(drop=True)

    hb = pd.Series(poly_hilbert, index=place["unit"].to_numpy())
    agg["_h"] = agg["unit"].map(hb.groupby(level=0).min())
    agg = agg.sort_values(["node", "_h", "unit", "tier"], kind="mergesort").reset_index(drop=True)
    counts_arr = agg["count"].to_numpy(dtype=float)
    meas = (agg["tier"].to_numpy() == 0) & LEFTOVERS
    alloc = np.zeros(len(agg), dtype=np.int64)
    # measured: the unit's own whole dots, the remainder a leftover mark (docstring)
    alloc[meas] = np.floor(counts_arr[meas] / dot_value).astype(np.int64)
    left = np.where(meas, counts_arr - alloc * dot_value, 0.0)
    agg["left"] = np.where(left >= LEFT_MIN, np.maximum(1, np.rint(left)), 0).astype(np.int64)
    # derived / modelled: the carry, among those rows only
    nm = np.flatnonzero(~meas)
    for node, idx in agg.iloc[nm].groupby("node", sort=False).indices.items():
        ii = nm[idx]
        crossed = np.floor(np.cumsum(counts_arr[ii]) / dot_value).astype(np.int64)
        alloc[ii] = np.diff(np.concatenate([[0], crossed]))
    agg["dots"] = alloc
    total = float(counts_arr.sum())
    undrawn = total - alloc.sum() * dot_value - agg["left"].sum()
    print(f"  {int((agg['left'] > 0).sum()):,} leftover marks ({agg['left'].sum():,} people); "
          f"{undrawn:,.0f} of {total:,.0f} people ({undrawn / total:.2%}) are carried rows "
          "under one dot per language nationally and draw no dot")

    # ---- rings: ONE mark per language that reaches no dot anywhere in the country.
    #
    # Its weight is the language's whole national total, which is under one dot by definition
    # (spec §4: "ONE dot of its own true weight ... at its largest concentration"). Until
    # 2026-10-05 it carried only the count of the single row it was placed on, which understated
    # every language spread over several units, and for a derived split spread over hundreds of
    # units would have drawn a speck.
    #
    # Derived-only and modelled-only languages ring too (Anita, 2026-10-05, "small languages:
    # sure"; it covers modelled rows as well, e.g. Azerbaijan's nationality model, which lost 4
    # small-language rings under the old rule). Inherited from religiondots §3.10, a language
    # whose every row was derived (or modelled) drew nothing, on the reasoning
    # that a proportional split spreads a total and cannot establish anyone is present in a given
    # unit. Here that cost 313 of Poland's 344 languages, Cornish and Manx, and more: languages
    # the census did count nationally, only not by place. The national count is what the ring
    # asserts, and it is not an artifact of the split; only its position is an estimate.
    #
    # Placement is unchanged for anything with a measured row: the largest measured row. A
    # language with no measured row falls back to its largest derived (then modelled) row, which
    # for a population-proportional split is the most populous unit.
    #
    # Guard: a language whose national total rounds to no one (under half a person) draws
    # nothing, and every ring carries at least 1. Derived rows are fractional, so a split can
    # leave a node with a fraction of a person that is arithmetic rather than a speaker
    # (Australia's shared remainders: 0.07; Poland's multi-answer sharing: 28 languages at 0.41
    # each). Measured rows can be fractional too: Brazil had 9 rings carrying 0 people, int() of
    # a placed row under 1, which tiles.py dropped; they now carry their total, at least 1.
    #
    # Since 2026-10-08 measured rows draw every remainder as a leftover mark, so a ring is only
    # for a language that draws neither: in practice one whose rows are all derived or modelled
    # and under one dot after the carry. A language drawn by leftovers alone gets no ring.
    RING_MIN = 0.5
    drawn = set(agg.loc[(agg["dots"] > 0) | (agg["left"] > 0), "node"])
    sub = agg[~agg["node"].isin(drawn)]
    node_total = sub.groupby("node")["count"].sum()
    keep = set(node_total.index[node_total >= RING_MIN])
    cand = sub[sub["node"].isin(keep)]
    cand = cand[cand["tier"] == cand.groupby("node")["tier"].transform("min")]
    ring_rows = (agg.loc[cand.groupby("node")["count"].idxmax()] if len(cand) else cand).copy()
    ring_rows["count"] = np.maximum(1, ring_rows["node"].map(node_total).round()).astype(np.int64)
    assert (ring_rows["count"] >= 1).all(), ring_rows[ring_rows["count"] < 1]
    n_too_small = node_total.size - len(keep)
    n_by_derived = int((ring_rows["tier"] > 0).sum())

    print(f"allocating dots at 1:{dot_value}…")
    per_poly, n_dots = {}, 0
    for row in agg.itertuples(index=False):
        if row.dots == 0:
            continue
        idx = by_unit[row.unit]
        dots = int(row.dots)
        w = weighter.weights(row.node, idx, float(row.count), plain=bool(row.tier)) if weighter else None
        if w is None:
            base, rem = divmod(dots, len(idx))
            alloc_p = np.full(len(idx), base)
            if rem:
                alloc_p[rng.choice(len(idx), rem, replace=False)] += 1
        else:
            order = np.argsort(poly_hilbert[idx], kind="mergesort")
            share = np.asarray(w, dtype=float)[order]
            share = share / share.sum() * dots
            crossed = np.floor(np.cumsum(share) + rng.random()).astype(np.int64)
            crossed[-1] = dots
            alloc_p = np.zeros(len(idx), dtype=np.int64)
            alloc_p[order] = np.diff(np.concatenate([[0], crossed]))
        assert int(alloc_p.sum()) == dots and (alloc_p >= 0).all(), (row.unit, row.node, dots)
        for t, k in zip(idx, alloc_p):
            if k:
                per_poly.setdefault(t, []).append((row.node, int(k), int(row.tier)))
        n_dots += dots
    assert n_dots == int(alloc.sum())
    if weighter is not None:
        print("  " + weighter.summary())
    print(f"  {n_dots:,} dots across {len(per_poly):,} polygons; {len(ring_rows)} rings "
          f"({n_by_derived} placed on derived or modelled rows, no measured row to place them on)")
    if n_too_small:
        print(f"  {n_too_small} sub-dot languages total under {RING_MIN:g} of a person and draw nothing")

    print("placing…")
    feats = []
    unit_of = place["unit"].to_numpy()
    unit_dots = {}          # unit -> its dots' coordinates, for the leftovers to sit on
    for t, items in per_poly.items():
        pts = random_points_in_polygon(geoms[t], sum(k for _, k, _ in items), rng)
        i = 0
        for node, k, tier in items:
            props = {"n": node} if not tier else {"n": node, "t": tier}
            for x, y in pts[i:i + k]:
                xy = [round(float(x), 4), round(float(y), 4)]
                feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": xy},
                              "properties": dict(props)})
                unit_dots.setdefault(unit_of[t], []).append(xy)
            i += k

    stem = cc if dot_value == DOT_VALUE else f"{cc}_{dot_value // 1000}k"
    OUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT / f"dots_{stem}.geojson", {"type": "FeatureCollection", "features": feats})
    print(f"wrote {len(feats):,} dots -> dots_{stem}.geojson")

    # A ring's polygon is drawn by the weights its language's dots would get in that unit: the
    # country's per-language weighter where it has one, else the unit's population, and only
    # where a unit holds no population at all, polygon area (2026-10-05). Until then it was a
    # uniform random polygon of the unit, so in a coarse unit (Cambodia is one national unit
    # placed by province per language) a ring could land anywhere in the country, ignoring the
    # weights its dots would follow. Seeded rng as before; rings are drawn after every dot, so
    # this moves rings only.
    pop_all = place["pop"].to_numpy(dtype=float) if "pop" in place.columns else None
    n_ring_area = 0

    def place_one(row, count):
        """One point in row's unit, on the weights row's language's dots would get there."""
        nonlocal n_ring_area
        idx = by_unit[row.unit]
        w = (weighter.weights(row.node, idx, float(count), plain=bool(row.tier))
             if weighter else None)
        if w is None and pop_all is not None and np.nansum(pop_all[idx]) > 0:
            w = pop_all[idx]
        w = None if w is None else np.nan_to_num(np.asarray(w, dtype=float)).clip(min=0)
        if w is None or w.sum() <= 0:
            w = shapely.area(geoms[idx]).astype(float)
            n_ring_area += 1
        t = int(rng.choice(idx, p=w / w.sum()))
        (x, y), = random_points_in_polygon(geoms[t], 1, rng)
        return [round(float(x), 4), round(float(y), 4)]

    rfeats = []
    # LEFTOVERS (docstring): each on a random one of its unit's own dots, so it joins a pie that
    # is already drawn; a unit with no dot puts all its leftovers on one shared point
    n_on_dot = n_shared = 0
    for unit, g in agg[agg["left"] > 0].groupby("unit", sort=False):
        mine = unit_dots.get(unit)
        if mine:
            spots = [mine[int(k)] for k in rng.integers(len(mine), size=len(g))]
            n_on_dot += 1
        else:
            big = g.loc[g["left"].idxmax()]
            spots = [place_one(big, float(g["left"].sum()))] * len(g)
            n_shared += 1
        for row, xy in zip(g.itertuples(index=False), spots):
            rfeats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": xy},
                           "properties": {"n": row.node, "why": "leftover", "count": int(row.left)}})
    n_left = len(rfeats)
    for row in ring_rows.itertuples(index=False):
        rfeats.append({"type": "Feature",
                       "geometry": {"type": "Point", "coordinates": place_one(row, row.count)},
                       "properties": {"n": row.node, "why": "under_dot", "count": int(row.count)}})
    write_json_atomic(OUT / f"rings_{stem}.geojson", {"type": "FeatureCollection", "features": rfeats})
    print(f"wrote {n_left:,} leftovers ({n_on_dot:,} units on their own dots, {n_shared:,} on one "
          f"shared point) and {len(rfeats) - n_left:,} rings -> rings_{stem}.geojson"
          + (f" ({n_ring_area} placed by area, their unit has no population)" if n_ring_area else ""))


if __name__ == "__main__":
    main()
