"""
India's placement layer: 649,618 villages and towns, each carrying its own 2011 population.

WHAT THIS IS FOR.  spec.md §8.2 places a unit's dots by giving each polygon of a finer layer
an equal share, which is a population weighting only because statistical agencies design
their smallest unit to a population target.  §8.2a is the section where that stops being
true: India has no engineered layer between the sub-district and the settlement, so India
placed on its count layer — one polygon per unit, median 204,000 people, 551 km², the
coarsest count unit on the map — and at city zoom looked blockier than anywhere else.

This builds the layer §8.2a says is missing.  The counts do not move and cannot: this file
writes no religion figure and scatter.py still takes every number from C-01.  All it changes
is WHERE inside a sub-district a dot is allowed to land, which is exactly the licence §8.2
grants placement and §14.4 bounds it to.

THE TWO SOURCES, AND WHY THE JOIN IS THE HARD PART.

  shrug-village-pc11.parquet   649,618 village and town POLYGONS, already downloaded for
                               in_geo.py's town unions.  Carries pc11_s_id/d_id/sd_id, so
                               state+district+sub-district NESTS EXACTLY in the count layer
                               — the thing §8.2d says to prefer over a geometric join.
  Census_Villages.parquet      645,828 village POINTS with `t_pop2011`, summing to
                               828,886,066: India's entire rural population.  Points only,
                               so it is the weight and never the shape.

**A six-digit code does not identify a settlement.  3,892 of them name both a village and a
town.**  Joining population on the code alone would hand a town's polygon a village's
population 3,892 times over, and nothing downstream could see it — the dot counts would all
still be right.  What is unique is (unit, code): the polygon file has no duplicate pair, and
C-01's 8,067 sub-district-level town rows have no duplicate pair either.  So the classifier
is (unit, code) against C-01's own town list, and a polygon is a village exactly when no town
row claims it.

**But the points file's unit codes cannot be used for that join**, and this is the same trap
[[reference_india_census_geo]] records against `SubDistricts_2011`: it splits units into
`(Pt)` parts, so it carries 6,200 unit codes where SHRUG has 5,967, and unit+code matches
only 82.4% of the polygons.  Village population therefore joins on the **village code alone**,
which reaches 99.92% of village polygons, and the unit assignment comes from SHRUG's file —
the same file the count layer is built from, so the two cannot drift apart.

WHERE THE POPULATION IS MISSING, AND WHAT HAPPENS THERE.

**Assam's rural population is simply absent from the points file.**  Its 26,599 villages carry
353 non-zero populations between them and sum to 449,486 people against a rural Assam of about
26.8 million.  The names match exactly — Mankachar, Kuchnimara, Jhawdanga Pt.III — so this is
a hole in the file and not a failed join, and no amount of key work will fill it.  Nationally
105,584 of the 645,828 points report zero and a quarter of those are Assam alone.

So a zero cannot be read as "nobody lives here" without checking what else the unit knows:

  * A unit whose villages carry a real population uses it, and a village reporting zero
    there is taken at its word — India has genuinely uninhabited revenue villages.
  * A unit whose villages sum to zero has told us nothing about its villages, so the rural
    side is treated as UNKNOWN rather than empty.  Reading those zeros literally would cram
    a whole sub-district's rural population into whichever town has a C-01 row.

**AND WHATEVER IS LEFT OVER GOES BACK TO THE SUB-DISTRICT.**  Every unit gets one extra
placement polygon — its own outline, straight from the count layer — carrying however many
of its people the settlements do not account for.  That is the whole fallback, and it is
worth being clear about why it is the right one.  The alternative, spreading a unit's
shortfall over whichever of its settlements happen to lack a population row, asserts that
the missing people live in those particular villages.  They generally do not: a shortfall
is mostly people whose settlement has no polygon at all, and putting them anywhere specific
invents a location the source never gave.  Sending them to the unit outline says only
"somewhere in this sub-district", which is exactly and precisely what §8.2a's map already
said about all of them.

**So no unit is ever drawn worse than it is today, and the fallback is proportionate.**  A
unit that accounts for all of itself never uses it; Assam's sub-districts put their towns in
the right place and let the rural remainder spread as before; a unit that knows nothing is
back to uniform sampling over its own polygon, which is where it started.  Every unit's
weights sum to its census total by construction, so the ratio below is 1.000 everywhere and
a drift away from it is a bug rather than a judgement.

Settlements the layer knows nothing about are then inert — weight zero — and are dropped
rather than shipped, because the unit outline already covers their ground.

The urban half comes from C-01 itself — 8,067 town rows at sub-district level, 377,106,125
people — and not from the points file, which is rural by construction.  Rows above
sub-district level are dropped: a town also appears under its district and its state, and
summing those is the double-count [[reference_india_census_geo]] warns about.

THE 23 `99999` UNITS.  `Area not under any Sub-district` has no polygon of its own and
in_geo.py gives it the union of its towns.  The same towns have to move here, or their area
would sit in the placement layer twice — once under the real sub-district SHRUG files them in,
once under the residual unit that actually counts their people.  149 of the 153 move on an
unambiguous code; the four that do not stay where SHRUG put them and are reported.

Usage:
    python sources/in_place.py --fetch    download Census_Villages.parquet (29MB) if missing
    python sources/in_place.py            build data/geo/in/in_places.gpkg
    python sources/in_place.py --towns    re-read the C-01 town rows, ignoring the cache
"""

import argparse
import os
import re
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "in")
RAW = os.path.join(ROOT, "data", "raw", "in")
NORM = os.path.join(ROOT, "data", "normalized", "in.csv")

POLY = os.path.join(GEO, "shrug-village-pc11.parquet")
POINTS = os.path.join(GEO, "Census_Villages.parquet")
SUBDIST = os.path.join(GEO, "in_subdistricts.gpkg")     # the count layer, from in_geo.py
TOWNS = os.path.join(GEO, "in_towns.csv")
OUT = os.path.join(GEO, "in_places.gpkg")

BASE = "https://github.com/yashveeeeeeer/india-geodata/releases/download/census%2F2011/"
POINTS_MIN = 25_000_000

RESIDUAL_SD = "99999"

# C-01's layout, as in.py reads it. `FIRST_COUNT_COL` is the `Total` universe column, which
# is the town's whole population and not a religion.
STATE_RE = re.compile(r"DDW(\d\d)C-01 MDDS\.XLS$", re.I)
FIRST_COUNT_COL = 7


def fetch():
    """Only Census_Villages.parquet. The polygons are in_geo.py's dependency, not ours."""
    os.makedirs(GEO, exist_ok=True)
    if os.path.exists(POINTS) and os.path.getsize(POINTS) >= POINTS_MIN:
        print(f"  have Census_Villages.parquet ({os.path.getsize(POINTS):,} bytes)")
        return
    import requests
    print("  getting Census_Villages.parquet ...")
    r = requests.get(BASE + "Census_Villages.parquet", timeout=1800, stream=True)
    if r.status_code != 200:
        raise SystemExit(f"Census_Villages.parquet: HTTP {r.status_code}")
    with open(POINTS, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    n = os.path.getsize(POINTS)
    if n < POINTS_MIN:
        raise SystemExit(f"Census_Villages.parquet: {n:,} bytes, expected >= {POINTS_MIN:,}")
    print(f"  got  Census_Villages.parquet ({n:,} bytes)")


def _code(v, width):
    """in.py's zero-padder. Excel stores the same code as text in one file and a number in
    another, and the difference is invisible until a join silently halves."""
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if not s.isdigit():
        raise SystemExit(f"non-numeric code {v!r}")
    if len(s) > width:
        raise SystemExit(f"code {s!r} wider than the expected {width}")
    return s.zfill(width)


def read_towns(force=False):
    """C-01's Urban town rows: unit, town code, name, population.

    Cached, because it is 35 workbooks for a table that never changes and re-reading them is
    minutes. Only sub-district-level rows are kept: the same town is also printed under its
    district and its state, and those are aggregates of these.
    """
    if os.path.exists(TOWNS) and not force:
        t = pd.read_csv(TOWNS, dtype={"unit": str, "town": str})
        print(f"  {len(t):,} town rows from {os.path.basename(TOWNS)} (cached)")
        return t
    rows = []
    files = [f for f in sorted(os.listdir(RAW))
             if STATE_RE.search(f) and STATE_RE.search(f).group(1) != "00"]
    if not files:
        raise SystemExit(f"no C-01 state workbooks in {RAW} — run sources/in.py --fetch")
    print(f"  reading town rows from {len(files)} C-01 workbooks…")
    for f in files:
        df = pd.ExcelFile(os.path.join(RAW, f)).parse("C01", header=None, dtype=object)
        df = df[df[0].astype(str).str.strip() == "C0101"]
        for _, r in df.iterrows():
            town, sd = _code(r.iloc[4], 6), _code(r.iloc[3], 5)
            # town rows are Urban-only (in.py asserts it); `sd == 00000` is the same town
            # repeated as a district or state aggregate and would double count
            if town == "000000" or sd == "00000":
                continue
            if str(r.iloc[6]).strip() != "Urban":
                continue
            rows.append({"unit": _code(r.iloc[1], 2) + _code(r.iloc[2], 3) + sd,
                         "town": town, "name": str(r.iloc[5]).strip(),
                         "pop": int(r.iloc[FIRST_COUNT_COL])})
    t = pd.DataFrame(rows)
    dup = t.duplicated(["unit", "town"]).sum()
    if dup:
        raise SystemExit(f"{dup} town rows share a (unit, code) — the classifier below "
                         f"assumes that pair is unique")
    t.to_csv(TOWNS, index=False)
    print(f"  {len(t):,} town rows, {t['pop'].sum():,} people -> "
          f"{os.path.basename(TOWNS)}")
    return t


def read_polygons():
    """The settlement polygons, with the count layer's own key on every row.

    Read through pyarrow rather than gpd.read_parquet: only four columns of 309MB are
    wanted, and pushing that into the reader is the difference between seconds and a
    multi-gigabyte frame. Bypassing geopandas means the geometry arrives as raw WKB.
    """
    tbl = pq.read_table(POLY, columns=["pc11_s_id", "pc11_d_id", "pc11_sd_id",
                                       "pc11_tv_id", "tv_name", "geometry"])
    g = tbl.to_pandas()
    g["geometry"] = shapely.from_wkb(g["geometry"])
    g = gpd.GeoDataFrame(g, geometry="geometry", crs="OGC:CRS84")
    g["unit"] = g["pc11_s_id"] + g["pc11_d_id"] + g["pc11_sd_id"]
    g["key"] = g["unit"] + g["pc11_tv_id"]
    if g["key"].duplicated().any():
        raise SystemExit("(unit, code) is not unique in the polygon file; the town "
                         "classifier depends on it")
    print(f"  {len(g):,} settlement polygons, {g['unit'].nunique():,} sub-districts")
    return g


def build():
    print("reading placement polygons…")
    g = read_polygons()

    print("reading town populations…")
    towns = read_towns()
    res = towns[towns["unit"].str.endswith(RESIDUAL_SD)]
    real = towns[~towns["unit"].str.endswith(RESIDUAL_SD)]

    # ---- towns in real sub-districts: (unit, code) straight onto the polygon
    tkey = dict(zip(real["unit"] + real["town"], real["pop"]))
    hit = g["key"].isin(tkey)
    miss = real[~(real["unit"] + real["town"]).isin(set(g["key"]))]
    print(f"  {int(hit.sum()):,} of {len(real):,} town rows matched a polygon; "
          f"{len(miss):,} did not ({miss['pop'].sum():,} people, "
          f"{miss['pop'].sum() / towns['pop'].sum():.2%} of urban India)")

    # ---- towns of a `99999` unit: C-01's unit is a placeholder, so they can only be found
    # by code, and they MOVE to the residual unit so their area is not claimed twice.
    by_code = g.groupby("pc11_tv_id").indices
    moved, ambiguous = {}, []
    for r in res.itertuples():
        idx = by_code.get(r.town, [])
        if len(idx) == 1:
            moved[int(idx[0])] = (r.unit, r.pop)
        else:
            ambiguous.append(r.town)
    print(f"  {len(moved):,} of {len(res):,} `99999` towns moved to their residual unit"
          + (f"; {len(ambiguous)} ambiguous and left in place: {sorted(set(ambiguous))}"
             if ambiguous else ""))

    g["is_town"] = hit
    g["tpop"] = g["key"].map(tkey)
    if moved:
        rows = np.fromiter(moved, dtype=int)
        g.loc[g.index[rows], "is_town"] = True
        g.loc[g.index[rows], "unit"] = [moved[i][0] for i in rows]
        g.loc[g.index[rows], "tpop"] = [moved[i][1] for i in rows]

    # ---- villages: population on the CODE alone (see the docstring on the `(Pt)` trap).
    print("reading village populations…")
    pts = pq.read_table(POINTS, columns=["vilcode11", "t_pop2011"]).to_pandas()
    if pts["vilcode11"].duplicated().any():
        raise SystemExit("village codes are not unique in the points file")
    vpop = dict(zip(pts["vilcode11"], pts["t_pop2011"]))
    print(f"  {len(pts):,} village points, {pts['t_pop2011'].sum():,} rural people, "
          f"{int((pts['t_pop2011'] == 0).sum()):,} of them reporting zero")

    vil = ~g["is_town"]
    g["vpop"] = np.where(vil, g["pc11_tv_id"].map(vpop), np.nan)
    matched = vil & g["vpop"].notna()
    print(f"  {int(matched.sum()):,} of {int(vil.sum()):,} village polygons matched "
          f"({matched.sum() / max(int(vil.sum()), 1):.2%})")

    # A code shared by two village polygons is one village in two pieces. Splitting its
    # population between them by area keeps the total right; taking it twice would not.
    # Planar degrees^2, and geopandas is right to warn about that in general. Here it is
    # only ever used to divide ONE village's population between ITS OWN parts, which are
    # adjacent, so the latitude scaling cancels exactly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g["area"] = g.geometry.area
    share = g["pc11_tv_id"].where(matched)
    n_parts = share.map(share.value_counts())
    split = matched & (n_parts > 1)
    if split.any():
        frac = g.loc[split, "area"] / g.loc[split].groupby(
            g.loc[split, "pc11_tv_id"])["area"].transform("sum")
        g.loc[split, "vpop"] = g.loc[split, "vpop"] * frac
        print(f"  {int(split.sum()):,} village polygons share a code with another and "
              f"split their population by area")

    # ---- keep only the count layer's own units
    df = pd.read_csv(NORM, dtype={"geo_id": str})
    sub = df[(df["geo_level"] == "subdistrict") & (df["source_category"] == "Total")]
    census = dict(zip(sub["geo_id"], sub["count"]))
    before = len(g)
    g = g[g["unit"].isin(census)].reset_index(drop=True)
    print(f"  dropped {before - len(g):,} polygons whose unit is not a census unit")
    orphan = set(census) - set(g["unit"])
    if orphan:
        lost = sum(census[u] for u in orphan)
        print(f"  {len(orphan)} census units have no settlement polygon at all "
              f"({lost:,} people, {lost / sub['count'].sum():.3%}) — they are drawn "
              f"entirely on the sub-district fallback below, i.e. exactly as today")

    # ---- weights. Everything above is a join; this is the only judgement in the file.
    print("weighting…")

    # Does this unit know anything about its villages? Assam does not (see docstring).
    rural_known = g.groupby("unit")["vpop"].transform("sum") > 0
    town_known = g["is_town"] & g["tpop"].notna()
    village_known = rural_known & ~g["is_town"]

    g["pop"] = 0.0
    g.loc[village_known, "pop"] = g.loc[village_known, "vpop"]
    g.loc[town_known, "pop"] = g.loc[town_known, "tpop"]
    # A village in an otherwise-known unit that has no population row of its own, and a town
    # C-01 does not list, both land here as NaN. They MUST become zero rather than stay NaN:
    # the weighter sums a unit's column, and one NaN makes the whole sum NaN, which reads as
    # "no population data" and drops that unit onto an equal share per settlement — §8.2a's
    # error, silently, in the one country the section is about. Caught in Kerala, 2026-09-07.
    g["pop"] = g["pop"].fillna(0.0)
    g["src"] = np.where(town_known & (g["pop"] > 0), "town",
                        np.where(village_known & (g["pop"] > 0), "village", "unknown"))

    # A settlement nothing is known about takes no dots. The unit outline added below covers
    # the same ground, so shipping it would cost a sixth of the file for no information.
    inert = g["pop"] <= 0
    no_row = inert & g["vpop"].isna() & g["tpop"].isna()
    print(f"  {int(inert.sum()):,} settlements draw nothing and are dropped: "
          f"{int((inert & ~no_row).sum()):,} published as empty or in a unit whose village "
          f"figures are missing, {int(no_row.sum()):,} with no row of any kind")
    g = g[~inert].reset_index(drop=True)

    # ---- the fallback: one row per unit, its own outline, carrying whatever the
    # settlements did not account for. This is §8.2a's placement, kept for exactly the
    # people §8.2a is still the best available answer for.
    known = g.groupby("unit")["pop"].sum()
    sd = gpd.read_file(SUBDIST)
    # Concatenated below without reprojecting, so a mismatch would move every fallback
    # polygon and nothing downstream would object — the ratio check cannot see geometry.
    if sd.crs != g.crs:
        raise SystemExit(f"{os.path.basename(SUBDIST)} is {sd.crs}, the settlements are "
                         f"{g.crs}; reproject before concatenating")
    if sd["kod"].duplicated().any():
        raise SystemExit("in_subdistricts.gpkg has duplicate units; each would take the "
                         "unit's whole residual again")
    sd = sd[sd["kod"].isin(census)].rename(columns={"kod": "unit", "name": "tv_name"})
    sd["pop"] = (sd["unit"].map(census) - sd["unit"].map(known).fillna(0.0)).clip(lower=0)
    sd["pc11_tv_id"] = "000000"
    sd["src"] = "subdistrict"
    used = sd[sd["pop"] > 0]
    print(f"  {len(used):,} of {len(sd):,} units need the sub-district fallback, "
          f"carrying {used['pop'].sum():,.0f} people "
          f"({used['pop'].sum() / sum(census.values()):.2%} of India)")
    g = gpd.GeoDataFrame(
        pd.concat([g[["unit", "pc11_tv_id", "tv_name", "src", "pop", "geometry"]],
                   used[["unit", "pc11_tv_id", "tv_name", "src", "pop", "geometry"]]],
                  ignore_index=True),
        geometry="geometry", crs=g.crs)

    # ---- what the layer is made of, by weight and not by row count
    mass = g.groupby("src")["pop"].sum()
    tot = mass.sum()
    print("\n  weight by source:")
    for k in ("village", "town", "subdistrict"):
        if k in mass:
            print(f"    {k:<9} {mass[k]:>15,.0f}  {mass[k] / tot:6.2%}")

    # ---- validation: the layer should track the census unit by unit, and this is the only
    # check available on the whole join. Nothing here can move a count, so a bad ratio is a
    # placement that will look wrong, not a number that is wrong.
    u = pd.DataFrame({"place": g.groupby("unit")["pop"].sum()})
    u["census"] = u.index.map(census)
    u["ratio"] = u["place"] / u["census"]
    print(f"\n  per-unit placement/census ratio over {len(u):,} units:")
    for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
        print(f"    p{int(q * 100):<3} {u['ratio'].quantile(q):.3f}")
    print(f"    units below 0.8: {int((u['ratio'] < 0.8).sum()):,}   "
          f"above 1.2: {int((u['ratio'] > 1.2).sum()):,}")
    print(f"    national: {u['place'].sum():,.0f} weighted against "
          f"{u['census'].sum():,.0f} counted ({u['place'].sum() / u['census'].sum():.3f})")

    if g["pop"].isna().any():
        raise SystemExit(f"{int(g['pop'].isna().sum()):,} polygons have a NaN weight; the "
                         f"weighter sums per unit and one NaN silently drops that unit onto "
                         f"an equal share per settlement")
    if (g["pop"] < 0).any():
        raise SystemExit("negative weight — a residual was spread the wrong way")
    zero_units = int((g.groupby("unit")["pop"].sum() <= 0).sum())
    if zero_units:
        raise SystemExit(f"{zero_units} units have zero total weight; scatter.py would "
                         f"fall back to an equal share per settlement, which is the one "
                         f"thing §8.2a says India must never do")
    # The fallback can only ever ADD, so a unit below 1.0 means it did not fire and the
    # shortfall would concentrate onto whichever settlements survived — the one failure this
    # design exists to prevent. Above 1.0 is a different thing and is not an error: it is a
    # unit whose settlements out-count C-01's own total for it, reported rather than clipped.
    short = u[u["ratio"] < 1 - 1e-9]
    if len(short):
        raise SystemExit(f"{len(short)} units are short of their census total despite the "
                         f"fallback, worst {short['ratio'].min():.3f} at "
                         f"{short['ratio'].idxmin()}")
    over = u[u["ratio"] > 1 + 1e-9]
    if len(over):
        print(f"    {len(over)} units whose settlements out-count C-01's own total for "
              f"them, worst {over['ratio'].max():.3f} — left as published, not clipped")

    bad = g.geometry.is_empty | g.geometry.isna()
    if bad.any():
        print(f"  dropping {int(bad.sum()):,} polygons with empty geometry")
        g = g[~bad].reset_index(drop=True)

    out = g[["unit", "pc11_tv_id", "tv_name", "src", "pop", "geometry"]].rename(
        columns={"pc11_tv_id": "tv", "tv_name": "name"})
    out.to_file(OUT, driver="GPKG", layer="places")
    print(f"\nwrote {OUT}")
    print(f"  {len(out):,} polygons over {out['unit'].nunique():,} sub-districts "
          f"({len(out) / out['unit'].nunique():.0f} per unit), "
          f"{os.path.getsize(OUT) / 1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--fetch", action="store_true",
                    help="download Census_Villages.parquet first")
    ap.add_argument("--towns", action="store_true",
                    help="re-read the C-01 town rows instead of using the cache")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    if args.towns and os.path.exists(TOWNS):
        os.remove(TOWNS)
    build()


if __name__ == "__main__":
    main()
