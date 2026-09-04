"""
Scatter religion dots. Country-agnostic; per-country wiring lives in countries.py.

Placement uses NO population data. The placement layer is whatever fine unit the country's
statistical agency designs to a population target — US census tracts (~3,400 people), Canadian
dissemination areas, Australian SA1s (~406) — so an equal share of a unit's dots per placement
polygon is already a population weighting (spec §8.2).

Fractions are carried ALONG A HILBERT CURVE through the units and a dot is dropped wherever
the running total passes dot_value. Nothing is ranked and nothing is handed to the top n —
see the long note beside the allocation, and the Whitechapel case that produced this rule.

Two outputs, matching the two symbols of spec §4.3:
    data/processed/dots_<cc>.geojson    one feature per DOT_VALUE people
    data/processed/rings_<cc>.geojson   one feature per RELIGION that draws no dot anywhere in
                                        the country, placed at its largest concentration —
                                        so a handful per country, not one per unit

Usage:
    python scatter.py --country us
    python scatter.py --country ca --dot-value 1000
    python scatter.py --country us --state 36        # one state, for quick iteration
"""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from countries import COUNTRIES

# Windows consoles here are cp1252 and the data is not: source names, categories and country
# notes carry Č, š, ú, ł and much else. Without this the script dies inside a print() after the
# real work has succeeded, which reads like a pipeline failure and is not one.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
    ap.add_argument("--no-weights", action="store_true",
                    help="ignore the country's place_weight hook and split each unit's dots "
                         "equally across its placement polygons (spec §8.2). The before "
                         "picture, and the way to check that weighting moved dots INSIDE "
                         "units without changing any unit's total.")
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
    poly_hilbert = place.geometry.hilbert_distance().to_numpy()

    # Optional: something better than an equal share per placement polygon (spec §8.2, §8.4).
    weighter = None
    if cfg.get("place_weight") is not None and not args.no_weights:
        weighter = cfg["place_weight"](place)

    # ---- carry the fractions ALONG A SPATIAL ORDER, dropping each dot where the carry
    # crosses. Anita's rule, 2026-09-03: "never hand to the top n — accumulate as we go in
    # geographic proximity and once we get over 1000, drop it wherever we are."
    #
    # Two bugs came before this one and both are worth keeping in view.
    #
    # 1. `count // dot_value` per (unit, node) threw away every remainder independently.
    #    Canada lost 5.5M people, 15% of the country, to pairs that each rounded to zero.
    # 2. The fix for that spread each node's national total by LARGEST REMAINDER — floor
    #    everywhere, then hand the dots still owed to the units with the biggest leftovers.
    #    That preserved the mass exactly and destroyed the geography, because the leftover
    #    IS the local count when no unit can reach a whole dot on its own. In England and
    #    Wales the median Output Area holds 306 people against a 1,000-person dot, so 2 of
    #    188,880 units earned a dot from the floor and every other dot went by rank. Rank on
    #    absolute count means the densest areas take everything:
    #
    #        Christians in OAs that are 10-20% Christian    drawn at   10% of actual
    #        Christians in OAs that are 70%+ Christian      drawn at  282%
    #        Muslims    in OAs that are 5-10% Muslim        drawn at    1%
    #        Muslims    in OAs that are 50-70% Muslim       drawn at  332%
    #
    #    Whitechapel is 22% Christian and drew no Christian dot at all, because an OA needed
    #    193 Christians to win one and it had about 49. The map said 99% Muslim about a
    #    place that is 40% Muslim. Every national total was exactly right the whole time,
    #    which is why nothing caught it: the error was purely spatial.
    #
    # So the carry runs along a HILBERT CURVE through the units. Walk them in that order,
    # add up the people, and every time the running total passes another dot_value, put a
    # dot in the unit you are standing in. A unit holding a third of a dot's worth of people
    # gets a dot about a third of the time, and — the point — the dot lands among the people
    # who contributed it rather than in whichever unit ranked highest nationally.
    #
    # The earlier code rejected a sequential carry as "an arbitrary spatial bias". It is,
    # if the sequence is arbitrary. A space-filling curve is not arbitrary: it is chosen so
    # that consecutive units are neighbours on the ground, which is exactly the property
    # that makes the carry land in the right place.
    #
    # Still exact: each node emits floor(national total / dot_value) dots, so what no dot
    # represents is under one dot per NODE, once, for the whole country (§4.1).
    # spec §7: THE TIER IS PART OF THE ROW'S IDENTITY, not something aggregated over it.
    #
    # The first cut took the weakest tier on each (unit, node) pair, reasoning that a pair
    # which is part census and part spread-out total is not a measurement. That is true of
    # the pair and false of the people in it, and the effect was the opposite of the
    # intention: Ireland's 4.64M measured people and 508k derived ones came out as 764
    # measured dots against 4,030 derived, because most Catholic pairs carry one large
    # measured row and one tiny allocated one, and `max` let the tiny row relabel the lot.
    #
    # So the tier joins the group key. A (unit, node) pair holding both becomes two rows,
    # each with its own people, and the dots divide between them in proportion by
    # construction — no rule needed, and nothing can launder anything in either direction.
    # `measured` is the default for an adapter that says nothing, which is right for a census
    # read at its own geography and is why an adapter that spreads must say so.
    if "tier" not in df.columns:
        df["tier"] = "measured"
    df["tier"] = df["tier"].fillna("measured")
    rank = {"measured": 0, "derived": 1, "modelled": 2}
    unknown = sorted(set(df["tier"]) - set(rank))
    if unknown:
        raise SystemExit(f"unknown confidence tier(s) {unknown}; spec §7 has three")
    df["tier"] = df["tier"].map(rank)

    agg = df.groupby(["unit", "node", "tier"], as_index=False).agg(
        count=("count", lambda x: x.sum(min_count=1)),
        congregations=("congregations", "sum"),
        may_ring=("may_ring", "all") if "may_ring" in df else ("count", "size"))
    if "may_ring" not in df.columns:
        agg["may_ring"] = True

    # rows with no count at all are a different thing from rows that are merely small: they are
    # a body that reported congregations and no membership, and only §4.3 has anything to say
    # about them. They never enter the allocation.
    uncounted = agg[agg["count"].isna()].copy()
    agg = agg[agg["count"].notna()].reset_index(drop=True)

    # One spatial key per unit: the Hilbert index of its placement polygons, smallest first.
    # Units are contiguous, so any of their polygons puts them in the right stretch of curve.
    hb = pd.Series(poly_hilbert, index=place["unit"].to_numpy())
    unit_key = hb.groupby(level=0).min()
    agg["_h"] = agg["unit"].map(unit_key)
    if agg["_h"].isna().any():
        # units with counts but no polygon; already reported above, and they cannot be walked
        agg["_h"] = agg["_h"].fillna(agg["_h"].max() + 1)
    agg = agg.sort_values(["node", "_h", "unit", "tier"],
                          kind="mergesort").reset_index(drop=True)

    counts_arr = agg["count"].to_numpy(dtype=float)
    floored_only = int((counts_arr // dot_value).sum())
    alloc = np.zeros(len(agg), dtype=np.int64)
    for node, idx in agg.groupby("node", sort=False).indices.items():
        cum = np.cumsum(counts_arr[idx])
        crossed = np.floor(cum / dot_value).astype(np.int64)
        alloc[idx] = np.diff(np.concatenate([[0], crossed]))
    agg["dots"] = alloc
    agg = agg.drop(columns="_h")
    carried = int(alloc.sum()) - floored_only
    # The real loss is what no dot anywhere represents: sum(count) - dots*dot_value, which is
    # under one dot per NODE. It is not the population of the pairs that drew nothing — those
    # people are represented, by dots the carry placed in other units of the same node, and
    # reporting their headcount here would overstate the loss by an order of magnitude.
    total_people = float(counts_arr.sum())
    unrepresented = total_people - int(alloc.sum()) * dot_value
    below = agg[agg["dots"] == 0]
    print(f"  carried fractions into {carried:,} dots that per-unit flooring would have lost "
          f"({carried * dot_value:,} people)")
    print(f"  {unrepresented:,.0f} of {total_people:,.0f} people ({unrepresented / max(total_people, 1):.2%}) "
          f"are under one dot at national level and draw nothing")
    if len(below):
        # not a loss and not a ring candidate: these people are drawn, as dots the carry placed
        # in other units of the same node
        print(f"  {len(below):,} (unit, node, tier) rows draw no dot of their own; their people are "
              f"carried into other units of the same node")

    # ---- rings: at most ONE per (country, node), and only for a node that draws no dot here.
    #
    # This is what the carry leaves behind. Once a node's national total is spread by largest
    # remainder, the only people no dot represents are the final `total % dot_value` — under one
    # dot, once, for the whole country. A per-area ring was therefore saying "this group is also
    # here" about people who were already drawn a unit or two away: 152,396 rings across three
    # countries, nearly all of them redundant with a dot next door.
    #
    # So a ring now means one thing only: **this religion is in this country and is too small to
    # reach a single dot anywhere in it.** A node that draws even one dot gets no ring, because
    # the dots already say it is here. That also bounds the layer at roughly one ring per node
    # per country instead of one per node per census tract.
    #
    # Rows with no count at all get nothing (Anita, 2026-09-03). A body that reported
    # congregations and never reported membership is not a small group, it is an unmeasured one —
    # the US has 155 of them holding 27,433 congregations, which is on the order of 13M people —
    # and drawing an "it is tiny" symbol for them is the opposite of true. §4.4's
    # congregation-to-adherent conversion is the right answer and it is not built yet; until it
    # is, they are absent rather than misdrawn.
    drawn_nodes = set(agg.loc[agg["dots"] > 0, "node"])
    cand = agg[(~agg["node"].isin(drawn_nodes)) & agg["may_ring"]
               & agg["count"].notna() & (agg["count"] > 0)]
    # §3.10: a node whose every row here is derived cannot ring — allocation spreads a total and
    # cannot establish that anyone is present at all.
    suppressed_nodes = (set(agg.loc[~agg["node"].isin(drawn_nodes), "node"])
                        - set(cand["node"]))
    # place the one ring at the group's largest concentration, which is the most defensible
    # single location for a claim that is really about the whole country
    ring_rows = (agg.loc[cand.groupby("node")["count"].idxmax()]
                 if len(cand) else cand).copy()
    ring_rows["why"] = "under_dot"

    uncounted_dropped = len(uncounted)

    df = agg
    print(f"allocating dots at 1:{dot_value}…")
    per_poly, rings = {}, []
    n_dots = 0

    # ---- inside a unit. §8.2's default is an equal share of the unit's dots per placement
    # polygon, which is a population weighting only because agencies design their smallest unit
    # to a population target. Where a country supplies something better, `place_weight` returns
    # a weighter and this becomes a weighted split — for the US that is spec §8.4: real tract
    # populations for every node, and a demographic redistribution on top for the nodes whose
    # held-out-metro correlation earned one.
    #
    # The split itself carries fractions the same way the national allocation does (§4.1a): walk
    # the unit's polygons in Hilbert order and drop a dot each time the running weight crosses.
    # A random offset per row keeps the first polygon on the curve from always losing its
    # fraction. Largest-remainder here would be the Whitechapel bug again, one unit down.
    for row in df.itertuples(index=False):
        idx = by_unit.get(row.unit)
        if idx is None or len(idx) == 0 or row.dots == 0:
            continue
        dots = int(row.dots)
        # `plain` says "this row is not a measurement" (§7). A weighter may hold a per-node
        # model fitted on measured data — the US does, §8.4 — and applying it to a derived
        # row would place the residual exactly where the measured people already are, which
        # for §3.5a's "on nobody's roll" residual is the one place they are not. Position by
        # population, yes; by a model of the rolls, no.
        w = (weighter.weights(row.node, idx, float(row.count), plain=bool(row.tier))
             if weighter else None)
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
            crossed[-1] = dots                          # float drift on the last step only
            alloc_p = np.zeros(len(idx), dtype=np.int64)
            alloc_p[order] = np.diff(np.concatenate([[0], crossed]))
        for t, k in zip(idx, alloc_p):
            if k:
                per_poly.setdefault(t, []).append((row.node, int(k), int(row.tier)))
        n_dots += dots

    # A ring asserts "this group is HERE", so a row has to establish presence before it can get
    # one, and a missing number does not. Found 2026-09-03: Canada has 644 census subdivisions
    # that publish no religion data at all, which arrives as NaN in all 147 categories, and every
    # one of those was becoming a ring — 5,152 of them, each asserting a religion was present in
    # a place whose source said nothing whatsoever. That is §3.5 backwards: an absence of data
    # drawn as a presence of people.
    #
    # The distinction is exactly whether the source counted SOMETHING. ASARB's uncounted rows are
    # a body reporting congregations and no membership — 14,877 rings over 27,433 congregations,
    # and the congregation count is the evidence of presence. Canada reports no congregations at
    # all, so a NaN there can only mean "not reported" and can never earn a ring.
    for row in ring_rows.itertuples(index=False):
        idx = by_unit.get(row.unit)
        if idx is None or len(idx) == 0:
            continue
        rings.append((int(rng.choice(idx)), row.node, row.why,
                      0 if pd.isna(row.congregations) else row.congregations))

    print(f"  {n_dots:,} dots across {len(per_poly):,} polygons")
    if weighter is not None:
        print(f"  {weighter.n_weighted:,} (unit, node) rows placed on fitted demographic "
              f"weights, {weighter.n_authored:,} on an authored ethnic tie, "
              f"{weighter.n_residual:,} on the CES-fitted residual model, "
              f"{weighter.n_uniform:,} on population alone (§8.4, §8.4a)")
    print(f"  {len(rings):,} rings — one per religion that draws no dot anywhere in the country")
    if uncounted_dropped:
        print(f"  {uncounted_dropped:,} (unit, node) pairs had no count at all and are not drawn "
              f"(§4.3; §4.4 is the fix for the bodies behind them)")
    if suppressed_nodes:
        print(f"  {len(suppressed_nodes):,} sub-dot nodes are derived-only and get no ring "
              f"(§3.10)")


    print("placing…")
    feats = []
    for n, (t, items) in enumerate(per_poly.items()):
        pts = random_points_in_polygon(geoms[t], sum(k for _, k, _ in items), rng)
        i = 0
        for node, k, tier in items:
            # `t` is omitted for `measured`, which is most of the map: the viewer reads a
            # missing one as full saturation, and a property written on every dot of every
            # country would cost tile size for the common case.
            props = {"n": node} if not tier else {"n": node, "t": tier}
            for x, y in pts[i:i + k]:
                feats.append({"type": "Feature",
                              "geometry": {"type": "Point",
                                           "coordinates": [round(float(x), 4),
                                                           round(float(y), 4)]},
                              "properties": dict(props)})
            i += k
        if n and n % 20000 == 0:
            print(f"  {n:,}/{len(per_poly):,}")

    # A partial run must not overwrite the national output. Testing with --state and then
    # tiling is a mistake with no symptom: the archive builds fine and contains one state.
    stem = f"{args.country}_{args.state}" if args.state else args.country
    if args.no_weights:
        stem += "_unweighted"
    # A non-default dot value is a SECOND build of the same country, not a replacement: the
    # viewer offers 1:1,000 and 1:10,000 side by side (§4.1b), so both files have to survive
    # in data/processed/ at once. Without the suffix the coarse run silently overwrites the
    # fine one and the next tile build is 1:10,000 everywhere with nothing to say so.
    if dot_value != DOT_VALUE:
        stem += f"_{dot_value // 1000}k" if dot_value % 1000 == 0 else f"_dv{dot_value}"
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
