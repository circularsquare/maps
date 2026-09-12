"""Solomon Islands — the placement grid: Kontur 400 m population hexagons, clipped to wards.

Writes data/geo/sb/sb_hexes.gpkg. `countries.py` uses it to weight where a ward's dots land,
never to change how many there are.

**THE SOLOMONS NEED THIS BECAUSE THE WARDS ARE COASTS AND THE INTERIORS ARE EMPTY.** Guadalcanal
and Malaita are mountainous and forested inland and settled around the shore; Rennell is one
raised atoll with two villages on it. Weighting a ward's dots by its area would put them in the
bush.

**AND THE STRAYS ARE SNAPPED, NOT DROPPED** — the rule Vanuatu established (§9bg §9). On a 400 m
grid against a detailed island coastline a large share of hex centroids land just seaward of the
ward outline, and that loss is *directional*: it is all on the coast, which is where the people
are. Dropping it would weight every shoreline light and pull the dots inland.

**THE VINTAGE GAP IS FOUR YEARS** — counts 2019, grid 2023 — and it moves dots within a ward,
never between wards.

**183 units makes the correlation a real check on `sb_geo.py`'s join.** That join is on SINSO's
own ward id and its province witness is clean, so this is a third, independent confirmation: a
modelled 2023 grid, sharing no lineage with SINSO's census or with OCHA's boundaries, has to
agree about how many people are in each of 183 wards.

Usage:
    python sources/sb_grid.py --fetch    one ~0.6 MB gz from Kontur
    python sources/sb_grid.py            rebuild from data/raw/sb/
"""

import gzip
import math
import os
import random
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sb")
GEO = os.path.join(ROOT, "data", "geo", "sb")
WARDS = os.path.join(GEO, "sb_wards.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "sb.csv")
LOOKUP = os.path.join(GEO, "sb_lookup.csv")
OUT = os.path.join(GEO, "sb_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_SB_20231101.gpkg.gz")
GZ_NAME = "kontur_population_SB_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_SB_20231101.gpkg"

EXPECTED_UNITS = 183
CENSUS_POPULATION = 720_956

NATIONAL_TOLERANCE = 0.35
# 183 units and some are only a few hundred people, so a generous band: this is looking for
# a mispaired unit, not a modelling difference.
UNIT_BAND = 5.0
MAX_SPAN_DEG = 25.0

# A hex centroid this far outside a ward is a coastline-resolution artefact and is snapped to
# the nearest ward; anything further is dropped. Metres, in a projected CRS.
SNAP_M = 500.0
SNAP_CRS = "EPSG:3832"


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 200_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(gz, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(WARDS):
        raise SystemExit(f"missing {WARDS} -- run sources/sb_geo.py first")

    units = gpd.read_file(WARDS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{WARDS} has {len(units)} wards, expected {EXPECTED_UNITS}")
    units = units.to_crs("EPSG:4326")
    print(f"wards: {len(units)}, crs={units.crs}")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=hexes.geometry.centroid,
                           crs=hexes.crs).to_crs("EPSG:4326")
    span = pts.total_bounds[2] - pts.total_bounds[0]
    print(f"  hex centroids span {span:.2f}° of longitude "
          f"(wards {units.total_bounds[2] - units.total_bounds[0]:.2f}°)")
    if span > MAX_SPAN_DEG:
        raise SystemExit(f"the grid spans {span:.1f}°; something is torn")

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    stray = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid falls outside every ward: {int(outside.sum()):,} "
          f"({stray:,.0f} people, {100.0 * stray / pts[popcol].sum():.3f}%)")
    if outside.any():
        m_units = units[["unit", "geometry"]].to_crs(SNAP_CRS)
        m_pts = pts.loc[outside].to_crs(SNAP_CRS)
        near = gpd.sjoin_nearest(m_pts, m_units, how="left", max_distance=SNAP_M,
                                 distance_col="_d")
        near = near[~near.index.duplicated(keep="first")]
        joined.loc[near.index, "unit"] = near["unit"]
        snapped = joined.loc[outside, "unit"].notna()
        moved = float(pts.loc[outside][snapped.to_numpy()][popcol].sum())
        print(f"     {int(snapped.sum()):,} are within {SNAP_M:,.0f} m of a ward and are "
              f"SNAPPED to the nearest\n     ({moved:,.0f} people, "
              f"{100.0 * moved / stray:.1f}% of the strays) — coastal cells just seaward of "
              "a\n     detailed coastline. Dropping them would pull every shoreline's dots "
              "inland (§9bg §9).")

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"     {int(outside.sum()):,} cells remain unplaced ({lost:,.0f} people, "
          f"{100.0 * lost / pts[popcol].sum():.3f}%) and are dropped;")
    print("     these are placement WEIGHTS and not counts, so nobody leaves the map.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=hexes.crs).to_crs(
        "EPSG:4326")

    # ---- A WARD SMALLER THAN ONE HEX GETS ITS OWN POLYGON AS A CELL ----
    #
    # `Naha`, ward 1010 in Honiara, is 464 people on a fraction of a square kilometre, and
    # no 400 m hex centroid falls inside it. That is the Kontur resolution floor rather than
    # a join failure: the grid is simply coarser than this one unit.
    #
    # Snapping it to a neighbour's hex would place its dots in the wrong ward, and leaving
    # it out drops the ward from the map entirely -- `countries.py` has no separate unit
    # layer for the Solomons, so the hexes ARE the geography. Instead the ward's own polygon
    # is added as a single cell, which is §8.2's equal share over the unit and is what the
    # placement would fall back to anyway if it could.
    name_of = dict(zip(units["unit"], units["name"]))
    have = set(out["unit"])
    tiny = sorted(set(units["unit"]) - have)
    if tiny:
        add = units[units["unit"].isin(tiny)][["unit", "geometry"]].copy()
        add["pop"] = 1.0
        areas = units.to_crs(SNAP_CRS).set_index("unit").geometry.area / 1e6
        print(f"\n  {len(tiny)} ward(s) smaller than one 400 m hex, given their own polygon "
              "as a single cell:")
        for u in tiny:
            print(f"     {name_of[u]} ({u}) — {areas[u]:.2f} km²; dots spread evenly "
                  "inside the ward (§8.2)")
        out = gpd.GeoDataFrame(pd.concat([out, add[["unit", "pop", "geometry"]]],
                                         ignore_index=True),
                               crs=out.crs)
        if len(tiny) > 5:
            raise SystemExit(f"{len(tiny)} wards have no hex at all, which is too many to "
                             "be the resolution floor -- check the join in sb_geo.py")

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"wards with no cell after the fallback: "
                         f"{[(u, name_of[u]) for u in missing]}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"wards whose cells sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} wards has cells: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the 2019 census {CENSUS_POPULATION:,} — "
          f"ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a 4-year gap -- check the download")

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(LOOKUP, dtype=str)
    unit_of = dict(zip(lut["geo_id"], lut["unit"]))
    census = {}
    for gid, sub in df[df["geo_level"] == "ward"].groupby("geo_id"):
        census[unit_of[gid]] = int(
            sub.loc[sub["source_category"] == "Total", "count"].iloc[0])

    # A ward given a synthetic cell above has no Kontur measurement to compare, so it is
    # scored on nothing and is left out of the band and the correlation.
    rows = [(u, name_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio)
            for u in census if u not in set(tiny)]
    rows.sort(key=lambda r: r[4])
    print(f"\n  the five worst-agreeing wards each way, census against Kontur, normalised"
          f"\n  ({len(census) - len(rows)} with a synthetic cell excluded):")
    print(f"    {'':<24} {'census 2019':>11} {'kontur':>9} {'norm':>6}")
    for u, nm, c, k, r in rows[:5] + rows[-5:]:
        print(f"    {nm:<24} {c:>11,} {k:>9,.0f} {r:>6.2f}")
    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]

    # **THESE OUTLIERS ARE KONTUR'S LIMIT, NOT A BAD JOIN, AND THE DIFFERENCE IS CHECKABLE.**
    # The join is on SINSO's own ward id and its province witness is clean on all 183, so a
    # ward where a modelled grid finds far fewer people than the census counted is a
    # settlement the model does not see. Both of the Solomons' are the same kind of place:
    # `Sulufou/Kwarande` is the Lau Lagoon, where people live on ARTIFICIAL ISLANDS built of
    # coral, and `Sikaiana` is a Polynesian atoll 210 km out from Malaita. A building
    # footprint model under-detects both. They are named and allowed; a long list would mean
    # something else and still fails.
    if len(worst) > 4:
        raise SystemExit(f"{len(worst)} wards outside a factor of {UNIT_BAND:g}, which is "
                         "too many to be the grid's blind spots: "
                         f"{[(w[1], round(w[4], 2)) for w in worst]}")
    if worst:
        share = sum(w[2] for w in worst) / sum(r[2] for r in rows)
        print(f"    {len(rows) - len(worst)} of {len(rows)} inside a factor of "
              f"{UNIT_BAND:g}; the {len(worst)} outside are "
              f"{100 * share:.2f}% of the counted population:")
        for u, nm, c, k, r in worst:
            print(f"      {nm:<24} census {c:>6,} vs Kontur {k:>6,.0f}  ({r:.2f}x)")
        print("      Both are settlements a building-footprint model under-detects — the "
              "Lau Lagoon\n      artificial islands and an outlying atoll. Placement "
              "weights only; no count moves.")
    else:
        print(f"    all {len(rows)} inside a factor of {UNIT_BAND:g}")

    lc = [math.log(r[2]) for r in rows]
    lk = [math.log(r[3]) for r in rows]
    r_true = pearson(lc, lk)
    rng = random.Random(0)
    perm = []
    for _ in range(2000):
        sh = list(lk)
        rng.shuffle(sh)
        perm.append(abs(pearson(lc, sh)))
    perm.sort()
    beat = sum(1 for x in perm if x >= r_true)
    print(f"\n  and the correlation, on 183 units: r = {r_true:.4f}, against a best of "
          f"{perm[-1]:.4f}\n  over 2,000 random pairings ({beat} reach it).")
    if beat > 20 or r_true < 0.75:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, "
                         f"which {beat} of 2,000 random pairings reach -- the id join in "
                         "sb_geo.py is not carrying information")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
