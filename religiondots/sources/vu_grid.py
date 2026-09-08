"""Vanuatu — the placement grid: Kontur 400 m population hexagons, clipped to area councils.

Writes data/geo/vu/vu_hexes.gpkg. `countries.py` uses it to weight where a unit's dots land,
never to change how many there are.

**VANUATU NEEDS THIS BECAUSE ITS UNITS ARE ISLANDS AND ITS PEOPLE ARE ON THE COASTS.** 83
islands, 66 units; Torres is six islands, Shepherds is a scatter of small ones, and the
interior of Santo and Malekula is close to empty while the shore is not. An equal share per
polygon would put dots inland and offshore in roughly the proportion the polygon's area
suggests, which is not where anyone lives.

**NO ANTIMERIDIAN MACHINERY HERE, UNLIKE FIJI (§9bd §7).** Vanuatu is 166.5°E to 170.2°E, so
Kontur's EPSG:3857 tiling and the boundaries' EPSG:4326 are both continuous across the country
and the join is an ordinary planar point-in-polygon. The span is asserted anyway.

**THE VINTAGE GAP IS THREE YEARS** — counts 2020, grid 2023 — which is the smallest on this
map after the countries whose grid and census share a year. It moves dots *within* a unit and
never between units.

**AND WITH 66 UNITS THE CORRELATION IS A REAL CHECK ON THE JOIN**, which is the thing Fiji's
fifteen provinces could not support. `vu_geo.py` pairs the census and the boundaries on NAME;
a modelled 2023 population grid, sharing no lineage with either VNSO's census or OCHA's
boundaries, then has to agree about how many people are in each of 66 area councils. A
mispaired unit shows up here as an outlier even when every total still reconciles.

Usage:
    python sources/vu_grid.py --fetch    one ~0.3 MB gz from Kontur
    python sources/vu_grid.py            rebuild from data/raw/vu/
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
RAW = os.path.join(ROOT, "data", "raw", "vu")
GEO = os.path.join(ROOT, "data", "geo", "vu")
COUNCILS = os.path.join(GEO, "vu_councils.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "vu.csv")
LOOKUP = os.path.join(GEO, "vu_lookup.csv")
OUT = os.path.join(GEO, "vu_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_VU_20231101.gpkg.gz")
GZ_NAME = "kontur_population_VU_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_VU_20231101.gpkg"

EXPECTED_UNITS = 66
# Table 3.5's universe: population in PRIVATE HOUSEHOLDS, which is what this map draws.
CENSUS_POPULATION = 293_963

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Three years is a short gap, so this is
# tighter than Fiji's 0.45.
NATIONAL_TOLERANCE = 0.35
# Per unit, after normalising by the national ratio. 66 units, the smallest a few hundred
# people, so a generous band: this catches a mispaired unit, not a modelling difference.
UNIT_BAND = 4.0
MAX_SPAN_DEG = 20.0

# A hex centroid this far outside a council is a coastline-resolution artefact and is snapped
# to the nearest one; anything further is dropped. 99.9% of Vanuatu's strays are inside 500 m
# and the largest are 15-150 m out. Metres, in a projected CRS -- EPSG:3832 is PDC Mercator,
# used here only for distance and not for any Fiji-style antimeridian handling.
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
    if not os.path.exists(COUNCILS):
        raise SystemExit(f"missing {COUNCILS} -- run sources/vu_geo.py first")

    units = gpd.read_file(COUNCILS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{COUNCILS} has {len(units)} units, expected {EXPECTED_UNITS}")
    units = units.to_crs("EPSG:4326")
    print(f"area councils: {len(units)}, crs={units.crs}")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    # Centroid in the CRS the hexes were TILED in, then reproject the POINTS. Reprojecting
    # first and taking the centroid after moves it.
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(
        "EPSG:4326")
    span = pts.total_bounds[2] - pts.total_bounds[0]
    print(f"  hex centroids span {span:.2f}° of longitude "
          f"(units {units.total_bounds[2] - units.total_bounds[0]:.2f}°)")
    if span > MAX_SPAN_DEG:
        raise SystemExit(f"the grid spans {span:.1f}°; Vanuatu is 3.7° wide, so something "
                         "is torn -- see sources/fj_grid.py for the antimeridian case")

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    # ---- 13% OF KONTUR'S PEOPLE FALL OUTSIDE EVERY COUNCIL, AND THEY MUST BE SNAPPED
    #      RATHER THAN DROPPED ----
    #
    # 645 cells, 44,347 people. Measured against the council outlines, **99.9% of them are
    # within 500 m of a boundary** and the largest are 15-150 m out: they are coastal cells
    # whose centroid lands just seaward of a detailed island coastline on a 400 m grid, and
    # they cluster around Port Vila and Luganville because that is where the people are.
    #
    # Fiji (§9bd §8) dropped its equivalent 6% and could afford to. Dropping 13% here would
    # be a directional error, not a rounding one: the discarded weight is almost entirely
    # SEAWARD, so every coastal unit would be weighted light exactly along the shore where
    # its population actually lives, and the dots would drift inland. So each stray cell is
    # assigned to the NEAREST council within SNAP_M, which is what a centroid a few dozen
    # metres offshore obviously means.
    outside = joined["unit"].isna()
    stray = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid falls outside every area council: "
          f"{int(outside.sum()):,} ({stray:,.0f} people, "
          f"{100.0 * stray / pts[popcol].sum():.3f}%)")
    if outside.any():
        m_units = units[["unit", "geometry"]].to_crs(SNAP_CRS)
        m_pts = pts.loc[outside].to_crs(SNAP_CRS)
        near = gpd.sjoin_nearest(m_pts, m_units, how="left", max_distance=SNAP_M,
                                 distance_col="_d")
        near = near[~near.index.duplicated(keep="first")]
        joined.loc[near.index, "unit"] = near["unit"]
        snapped = joined.loc[outside, "unit"].notna()
        moved = float(pts.loc[outside][snapped.to_numpy()][popcol].sum())
        print(f"     {int(snapped.sum()):,} of them are within {SNAP_M:,.0f} m of a "
              f"council and are SNAPPED to the nearest\n     ({moved:,.0f} people, "
              f"{100.0 * moved / stray:.1f}% of the strays) — coastal cells just seaward of "
              "a\n     detailed coastline. Dropping them would weight every shoreline light "
              "and pull the\n     dots inland, which is a direction, not a rounding.")

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

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        name_of = dict(zip(units["unit"], units["name"]))
        raise SystemExit(f"area councils with no populated hex: "
                         f"{[(u, name_of[u]) for u in missing]}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"area councils whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} units has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the 2020 census {CENSUS_POPULATION:,} "
          f"(private households) — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a 3-year gap -- check the download")

    # ---- per unit, the band, and the correlation ----
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(LOOKUP, dtype=str)
    unit_of = dict(zip(lut["geo_id"], lut["unit"]))
    name_of = dict(zip(units["unit"], units["name"]))
    census = {}
    for gid, sub in df[df["geo_level"] == "area_council"].groupby("geo_id"):
        census[unit_of[gid]] = int(
            sub.loc[sub["source_category"] == "Total", "count"].iloc[0])

    rows = [(u, name_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio) for u in census]
    rows.sort(key=lambda r: r[4])
    print("\n  the ten worst-agreeing units, census against Kontur, normalised:")
    print(f"    {'':<22} {'census 2020':>11} {'kontur':>9} {'norm':>6}")
    for u, nm, c, k, r in rows[:5] + rows[-5:]:
        print(f"    {nm:<22} {c:>11,} {k:>9,.0f} {r:>6.2f}")
    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    if worst:
        raise SystemExit(f"{len(worst)} units outside a factor of {UNIT_BAND:g}: "
                         f"{[(w[1], round(w[4], 2)) for w in worst]} -- a unit this far out "
                         "is usually a mispaired name, not a modelling difference")
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
    print(f"\n  and the correlation, which on 66 units is the check with real power: "
          f"r = {r_true:.4f},\n  against a best of {perm[-1]:.4f} over 2,000 random pairings "
          f"({beat} reach it).")
    if beat > 20 or r_true < 0.80:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, which "
                         f"{beat} of 2,000 random pairings reach -- the name join in "
                         "vu_geo.py is not carrying information")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
