"""Belize — the placement layer: Kontur 400 m population hexagons, keyed to district.

Writes data/geo/bz/bz_hexes.gpkg.

**BELIZE IS THE EMPTY-INTERIOR CASE, AND ITS RATIO OF AREA TO UNITS IS THE WORST HERE AFTER
ZIMBABWE'S.** Six districts over 22,966 km² is **3,828 km² per unit** — five times Jamaica's
785 — and the country is nothing like uniformly habitable. Cayo contains the Chiquibul
Forest Reserve and most of the Maya Mountains; Toledo is largely rainforest with its people
strung along the Southern Highway and the coast; Orange Walk runs north into the Rio Bravo
conservation area. Spread dots evenly over those polygons and a large share of Belize is
drawn into forest nobody lives in, while Belize City — 64,000 people in a district that is
also mangrove and lagoon — is smeared across the whole of it.

Kontur removes the problem rather than patching it: an empty hex has no population and takes
no dots, so the reserves simply have no weight. `water.py` is not involved.

**AND IT MATTERS MORE HERE THAN THE UNIT COUNT SUGGESTS**, because Belize's religions are
sorted by settlement type rather than by district: the Mennonite colonies are specific
places (Blue Creek, Shipyard, Spanish Lookout, Little Belize) inside districts whose other
90% is not Mennonite. Six districts cannot show that, and nothing here pretends to — but
placing the district's dots on where people actually are is the difference between a
Mennonite share drawn across settled northern farmland and one drawn over empty bush.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, and that is deliberate — a hex on a district
line belongs wholly to one side, so no hex is split and no population is double-counted.
Hexes whose centroid falls outside every district (the Mexican and Guatemalan border
overrun, and the cayes beyond the boundary file's coastline) are dropped and reported.

Usage:
    python sources/bz_grid.py --fetch    one ~420 KB gzipped gpkg from Kontur
    python sources/bz_grid.py            rebuild from data/raw/bz/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bz")
GEO = os.path.join(ROOT, "data", "geo", "bz")
DISTRICTS = os.path.join(GEO, "bz_districts.gpkg")
OUT = os.path.join(GEO, "bz_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_BZ_20231101.gpkg.gz")
GZ_NAME = "kontur_population_BZ_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_BZ_20231101.gpkg"

EXPECTED_DISTRICTS = 6

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census and must
# not be asserted equal to it (§12, North Macedonia). Used ONLY as a within-district weight,
# so all that matters is that it is not wildly out — assert the RELATIONSHIP.
#
# THE BAND IS MEASURED HERE AND NOT COPIED (§9u's rule). Kontur's vintage is 2023-11 and the
# census is May 2022, eighteen months apart, and Belize grows ~1.9%/yr — so the grid should
# read a little HIGH, by a couple of per cent. The band is wide because SIB's published
# figure is the UNDERCOUNT-ADJUSTED one (see sources/bz.py): the raw enumerated count is
# lower, and which of the two a model like Kontur is closer to is not knowable in advance.
CENSUS_POPULATION = 397_483
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
    # §5a: a 200 is not a download, and a gunzip that runs is not a gpkg.
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage -- starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(DISTRICTS):
        raise SystemExit(f"missing {DISTRICTS} -- run sources/bz_geo.py first")

    hexes = gpd.read_file(gpkg)
    # §12: assert the feature count, not the absence of an exception.
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    districts = gpd.read_file(DISTRICTS)
    if len(districts) != EXPECTED_DISTRICTS:
        raise SystemExit(f"{DISTRICTS} has {len(districts)} districts, "
                         f"expected {EXPECTED_DISTRICTS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS -- reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(districts.crs)
    hexes = hexes.to_crs(districts.crs)

    joined = gpd.sjoin(pts, districts[["unit", "geometry"]], how="left",
                       predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every district: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's BZ extract overruns the Mexican and Guatemalan borders, and the "
          "outer\n     cayes sit beyond COD's coastline; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=districts.crs)

    # Every district must get some hexes, or its dots fall back to an equal share over the
    # whole polygon and the district silently un-weights itself.
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(districts["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"districts with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"districts whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_DISTRICTS} districts has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")
    for unit, row in per.sort_values("size").iterrows():
        nm = districts.loc[districts["unit"] == unit, "name"].iloc[0]
        print(f"      {nm:<12} ({unit}) {int(row['size']):>6,} hexes, "
              f"{row['sum']:>9,.0f} people")

    # Kontur is a MODEL, not the census. Assert the relationship, never equality (§12).
    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census {CENSUS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio-1)*100:.0f}%, "
                         "which is too much for a weight -- check the download")
    print("     used only as a WITHIN-district weight, so the level does not matter and "
          "the\n     shape does.")

    # The per-district ratio is the one that would bias a share, and it is printed rather
    # than asserted: a district Kontur models badly gets its dots on a worse surface, it
    # does not get the wrong number of them (§9t).
    print("\n  per-district Kontur/census ratio (the shape check):")
    import pandas as pd
    cen = pd.read_csv(os.path.join(ROOT, "data", "normalized", "bz.csv"),
                      dtype={"geo_id": str})
    cen = cen[(cen["geo_level"] == "district") & (cen["source_category"] == "Total")]
    cen = dict(zip(cen["geo_id"], cen["count"]))
    for unit, row in per.iterrows():
        nm = districts.loc[districts["unit"] == unit, "name"].iloc[0]
        r = row["sum"] / cen[unit]
        print(f"      {nm:<12} {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
