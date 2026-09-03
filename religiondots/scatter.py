"""
Scatter religion dots. Country-agnostic; per-country wiring lives in countries.py.

Placement uses NO population data. The placement layer is whatever fine unit the country's
statistical agency designs to a population target — US census tracts (~3,400 people), Canadian
dissemination areas, Australian SA1s (~406) — so an equal share of a unit's dots per placement
polygon is already a population weighting (spec §8.2).

Two outputs, matching the two symbols of spec §4.3:
    data/processed/dots_<cc>.geojson    one feature per DOT_VALUE people
    data/processed/rings_<cc>.geojson   one feature per (body, unit) with no dots — either no
                                        count at all, or fewer people than one dot

Usage:
    python scatter.py --country us
    python scatter.py --country ca --dot-value 1000
    python scatter.py --country us --state 36        # one state, for quick iteration
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from countries import COUNTRIES

HERE = Path(__file__).parent
OUT = HERE / "data" / "processed"

# People per dot. 1:1000 is what a global build can afford and what the tiles ship at
# (spec §4.2); 1:100 is the nicer map and four times the features.
DOT_VALUE = 1000
SEED = 20260827


def random_points_in_polygon(geom, n: int, rng) -> np.ndarray:
    """Rejection-sample n points inside geom. Vectorised contains, as ancestrydots does."""
    if n <= 0:
        return np.empty((0, 2))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True, choices=sorted(COUNTRIES))
    ap.add_argument("--dot-value", type=int, default=DOT_VALUE)
    ap.add_argument("--state", help="us only: two-digit FIPS, for quick iteration")
    args = ap.parse_args()
    cfg = COUNTRIES[args.country]
    dot_value = args.dot_value
    rng = np.random.default_rng(SEED)

    print(f"country: {args.country}  ({cfg['note']})")
    print("reading counts…")
    df = cfg["counts"]()
    print(f"  {len(df):,} (unit, node) rows, {df['node'].nunique()} nodes, "
          f"{df['unit'].nunique():,} units")

    print("reading placement polygons…")
    place = gpd.read_file(cfg["place"])
    print(f"  {len(place):,} placement polygons, crs={place.crs}")

    if cfg["place_unit"] == "sjoin":
        # Derive each placement polygon's unit by point-in-polygon. Done in the source CRS,
        # before reprojection: both layers are already in it, and a planar join is faster and
        # avoids any question about geometry validity after a transform.
        units = gpd.read_file(cfg["units"])
        if units.crs != place.crs:
            units = units.to_crs(place.crs)
        print(f"  spatial join onto {len(units):,} units on {cfg['unit_key']}…")
        pts = place.copy()
        pts["geometry"] = place.geometry.representative_point()
        joined = gpd.sjoin(pts[["geometry"]], units[[cfg["unit_key"], "geometry"]],
                           how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]
        place["unit"] = joined[cfg["unit_key"]].reindex(place.index)
        orphan = place["unit"].isna().sum()
        if orphan:
            print(f"  !! {orphan:,} placement polygons fell outside every unit — dropped")
            place = place[place["unit"].notna()]
    else:
        place["unit"] = cfg["place_unit"](place)

    if place.crs is not None and place.crs.to_epsg() != 4326:
        print(f"  reprojecting {place.crs.to_string()} -> EPSG:4326")
        place = place.to_crs(4326)

    # spec §8.1, three directions: unmatched data, unmatched polygons, and — the one that
    # two-way matching misses — codes that match but carry no geometry.
    empty = place.geometry.isna() | place.geometry.is_empty
    if empty.any():
        print(f"  !! {int(empty.sum()):,} placement polygons have empty geometry — dropped")
        place = place[~empty]
    place = place.reset_index(drop=True)

    if args.state:
        place = place[place["unit"].str[:2] == args.state].reset_index(drop=True)
        df = df[df["unit"].str[:2] == args.state]

    have, want = set(place["unit"]), set(df["unit"])
    missing = sorted(want - have)
    if missing:
        lost = df[df["unit"].isin(missing)]["count"].sum()
        print(f"  !! {len(missing):,} units in the data have no polygons "
              f"({lost:,.0f} people): {missing[:6]}")
    extra = len(have - want)
    if extra:
        print(f"  {extra:,} units have polygons but no religion rows")

    by_unit = {u: g.index.to_numpy() for u, g in place.groupby("unit")}
    geoms = place.geometry.to_numpy()

    # ---- roll sub-floor fragments up the taxonomy rather than losing them.
    #
    # Splitting a country into 147 categories across 5,161 units shatters its population into
    # fragments that are individually under one dot: Canada lost 5.5M people, 15% of the
    # country, to pairs that each rounded to zero. Dropping them is honest but wasteful, and
    # inventing a dot for them is dishonest. Rolling them into their parent node is neither —
    # 400 Old Order Mennonites in a small subdivision become 400 Anabaptists, which is true,
    # just less specific. Mass is preserved, no presence is invented, and granularity degrades
    # exactly where the dot value cannot carry it.
    nodes = json.loads((HERE / "taxonomy" / "religions.json").read_text(encoding="utf-8"))
    known = {n["id"] for n in nodes["nodes"]}

    def parent(nid):
        while "." in nid:
            nid = nid.rsplit(".", 1)[0]
            if nid in known:
                return nid
        return None

    agg = df.groupby(["unit", "node"], as_index=False).agg(
        count=("count", lambda x: x.sum(min_count=1)),
        congregations=("congregations", "sum"),
        may_ring=("may_ring", "all") if "may_ring" in df else ("count", "size"))
    if "may_ring" not in df.columns:
        agg["may_ring"] = True

    ring_rows = agg[(agg["count"].isna()) |
                    ((agg["count"] < dot_value) & agg["may_ring"])].copy()

    def drawable(frame):
        c = frame["count"].fillna(0)
        return int((c // dot_value).sum())

    dots_before = drawable(agg)
    moved_people = moved_pairs = 0
    for _ in range(6):
        small = agg["count"].notna() & (agg["count"] < dot_value)
        if not small.any():
            break
        up = agg[small].copy()
        up["node"] = up["node"].map(parent)
        stuck = up["node"].isna()
        moved_people += up.loc[~stuck, "count"].sum()
        moved_pairs += int((~stuck).sum())
        agg = pd.concat([agg[~small], up[~stuck]], ignore_index=True)
        agg = agg.groupby(["unit", "node"], as_index=False).agg(
            count=("count", lambda x: x.sum(min_count=1)),
        congregations=("congregations", "sum"),
            may_ring=("may_ring", "all"))
    if moved_pairs:
        # Report the NET effect, not the traffic. `moved_people` counts a fragment again on
        # every hop, so a value rolled leaf -> branch -> family is counted three times and the
        # figure reads several times larger than the map actually gains. What the reader wants
        # is how many dots exist that would not have — and many rolled fragments still land
        # under the floor at their parent and are never drawn at all.
        gained = drawable(agg) - dots_before
        print(f"  rolled {moved_pairs:,} sub-floor fragments up the tree over "
              f"{moved_pairs and 'several'} passes")
        print(f"  dots {dots_before:,} -> {drawable(agg):,} "
              f"(+{gained:,} = {gained * dot_value:,} people who would have rounded to zero)")

    df = agg
    print(f"allocating dots at 1:{dot_value}…")
    per_poly, rings = {}, []
    n_dots = no_count = suppressed = 0

    for row in df.itertuples(index=False):
        idx = by_unit.get(row.unit)
        if idx is None or len(idx) == 0 or pd.isna(row.count):
            continue
        dots = int(row.count // dot_value)
        if dots == 0:
            continue
        base, rem = divmod(dots, len(idx))
        alloc = np.full(len(idx), base)
        if rem:
            alloc[rng.choice(len(idx), rem, replace=False)] += 1
        for t, k in zip(idx, alloc):
            if k:
                per_poly.setdefault(t, []).append((row.node, int(k)))
        n_dots += dots

    # Rings come from the PRE-rollup snapshot: a ring is a claim about a specific group being
    # present, so it has to be made at the granularity the source actually reported, not at
    # whatever ancestor the fragment was rolled into. spec §3.10 keeps derived rows out of it —
    # allocation spreads a total and cannot establish that anyone is here at all.
    for row in ring_rows.itertuples(index=False):
        idx = by_unit.get(row.unit)
        if idx is None or len(idx) == 0:
            continue
        if not row.may_ring:
            suppressed += 1
            continue
        why = "no_count" if pd.isna(row.count) else "below_floor"
        no_count += pd.isna(row.count)
        rings.append((int(rng.choice(idx)), row.node, why,
                      0 if pd.isna(row.congregations) else row.congregations))

    print(f"  {n_dots:,} dots across {len(per_poly):,} polygons")
    print(f"  {len(rings):,} rings ({no_count:,} with no count, "
          f"{len(rings) - no_count:,} below the {dot_value}-person floor)")
    if suppressed:
        print(f"  {suppressed:,} derived (allocated) pairs fell below the floor and were "
              f"dropped rather than ringed — spec §3.10")

    print("placing…")
    feats = []
    for n, (t, items) in enumerate(per_poly.items()):
        pts = random_points_in_polygon(geoms[t], sum(k for _, k in items), rng)
        i = 0
        for node, k in items:
            for x, y in pts[i:i + k]:
                feats.append({"type": "Feature",
                              "geometry": {"type": "Point",
                                           "coordinates": [round(float(x), 4),
                                                           round(float(y), 4)]},
                              "properties": {"n": node}})
            i += k
        if n and n % 20000 == 0:
            print(f"  {n:,}/{len(per_poly):,}")

    # A partial run must not overwrite the national output. Testing with --state and then
    # tiling is a mistake with no symptom: the archive builds fine and contains one state.
    stem = f"{args.country}_{args.state}" if args.state else args.country
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"dots_{stem}.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    print(f"wrote {len(feats):,} dots -> dots_{stem}.geojson")

    rfeats = []
    by_poly = {}
    for t, node, why, congs in rings:
        by_poly.setdefault(t, []).append((node, why, congs))
    for t, items in by_poly.items():
        pts = random_points_in_polygon(geoms[t], len(items), rng)
        for (node, why, congs), (x, y) in zip(items, pts):
            rfeats.append({"type": "Feature",
                           "geometry": {"type": "Point",
                                        "coordinates": [round(float(x), 4),
                                                        round(float(y), 4)]},
                           "properties": {"n": node, "why": why,
                                          "congregations": int(congs)}})
    with open(OUT / f"rings_{stem}.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": rfeats}, f)
    print(f"wrote {len(rfeats):,} rings -> rings_{stem}.geojson")


if __name__ == "__main__":
    main()
