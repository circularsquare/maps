"""Uganda — the placement layer: Kontur 400 m population hexagons, keyed to the 2002 district.

Writes data/geo/ug/ug_hexes.gpkg.

**56 DISTRICTS OVER ABOUT 200,500 km² OF LAND IS ROUGHLY 3,600 km² PER UNIT**, and
Uganda is not uniformly habitable at that scale. Karamoja is a fifth of the country's
area and a twentieth of its people; Lake Kyoga's swamp sprawls across the middle of
Apac, Soroti, Kamuli and Kaberamaido; Murchison Falls sits inside Masindi and Gulu, and
Queen Elizabeth inside Kasese and Bushenyi. Spread a district's dots evenly over its
polygon and Uganda's dots land in national parks and papyrus.

Kontur removes that without a rule: an empty hex has no population and takes no dots.

**THE VINTAGE IS THE CAVEAT AND IT IS STATED RATHER THAN HIDDEN.** The counts are the
2002 census; Kontur's grid is 2023-11, by which time Uganda held nearly twice as many
people. It is used ONLY as a within-district weight, so the level does not matter, but
the shape has moved in twenty years: a 2023 surface puts a district's dots where its
people live now rather than where they lived in 2002. Inside a 3,600 km² district that
is mostly the difference between a trading centre that has grown and one that has not,
and the largest real shift — Kampala's overspill — happened into Wakiso, which is a
district of its own here and so absorbs it rather than smearing it.

**RE-LEVELLING ONTO THE CENSUS'S OWN FINER GEOGRAPHY WAS CONSIDERED AND REFUSED.** Table
C1 of the 2002 annex series carries the population of 940 sub-counties, which is what
`sources/rw_grid.py` uses for Rwanda. It is not used here because the join cannot be
proved: Uganda's sub-counties have multiplied from about 960 to 1,520 since 2002, the
names split rather than persist, and unlike the district concordance there is no second
publication giving sub-county population on both vintages to check an assignment
against. An unproved join at 940 units would move dots on a surface nobody could audit
([[reference_check_needs_power]]); Kontur at least does not pretend to be a census.

THE JOIN IS SPATIAL, on hex CENTROIDS, so a hex on a district line belongs wholly to one
side and no hex is split or double-counted.

Usage:
    python sources/ug_grid.py --fetch    one ~13 MB gzipped gpkg from Kontur
    python sources/ug_grid.py            rebuild from data/raw/ug/
"""

import gzip
import os
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")     # [[reference_scipy_eats_all_cores]]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ug")
GEO = os.path.join(ROOT, "data", "geo", "ug")
DISTRICTS = os.path.join(GEO, "ug_districts.gpkg")
OUT = os.path.join(GEO, "ug_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_UG_20231101.gpkg.gz")
GZ_NAME = "kontur_population_UG_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_UG_20231101.gpkg"

EXPECTED_DISTRICTS = 56

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must
# not be asserted equal to one. The anchor here is Uganda's OWN 2024 census count rather
# than a projection series, because Uganda has one and it is recent: NPHC 2024 counted
# 45,905,417 people. Kontur's vintage is 2023-11, four months earlier, so the grid should
# read a little low. The band is wide because a settlement model's level is not the thing
# being used.
CENSUS_2024_POPULATION = 45_905_417
KONTUR_TOLERANCE = 0.35

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 500_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(gz + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(gz + ".part", gz)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    # A 200 is not a download and a gunzip that runs is not a GeoPackage.
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage — starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run with --fetch first")
    if not os.path.exists(DISTRICTS):
        raise SystemExit(f"missing {DISTRICTS} — run sources/ug_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    dist = gpd.read_file(DISTRICTS)
    if len(dist) != EXPECTED_DISTRICTS:
        raise SystemExit(f"{DISTRICTS} has {len(dist)} districts, "
                         f"expected {EXPECTED_DISTRICTS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in,
    # then reproject the POINTS — reprojecting first and taking the centroid after moves
    # it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(dist.crs)
    hexes = hexes.to_crs(dist.crs)

    joined = gpd.sjoin(pts, dist[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every district: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's UG extract overruns the COD-AB coastline of Lake Victoria and "
          "the\n     Congo, Sudan and Kenya borders; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=dist.crs)

    # Every district must get hexes, or its dots fall back to an equal share over the
    # whole polygon and the district silently un-weights itself.
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(dist["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"districts with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"districts whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_DISTRICTS} districts has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_2024_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the 2024 census's {CENSUS_2024_POPULATION:,} — "
          f"ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight — check the download")
    print("     used only as a WITHIN-district weight, so the level does not matter and "
          "the\n     shape does.")

    # The per-district ratio is what would bias a placement, and it is printed rather than
    # asserted: a district Kontur models badly gets its dots on a worse surface, it does
    # not get the wrong number of them. Compared against the district's 2002 census
    # population, so a ratio near 1.9 is just twenty years of growth.
    import pandas as pd

    norm = pd.read_csv(os.path.join(ROOT, "data", "normalized", "ug.csv"))
    pop02 = (norm[norm["source_category"] == "Total"]
             .set_index("geo_id")[["geo_name", "count"]])
    print("\n  Kontur-2023 over census-2002, by district (Uganda's population roughly "
          "doubled\n  over that span, so about 1.9 is the expected middle):")
    rows = []
    for unit, row in per.iterrows():
        rows.append((pop02.loc[unit, "geo_name"], int(row["size"]),
                     row["sum"] / pop02.loc[unit, "count"]))
    rows.sort(key=lambda t: t[2])
    for nm, n, r in rows[:4] + rows[-4:]:
        print(f"      {nm:<16} {n:>7,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
