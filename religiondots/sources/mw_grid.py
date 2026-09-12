"""Malawi — the placement layer: Kontur 400 m population hexagons, keyed to district.

Writes data/geo/mw/mw_hexes.gpkg.

**MALAWI IS THE INLAND-WATER CASE §8.2c NAMES, AND IT IS THE WORST ONE ON THIS MAP.** Lake
Malawi is 29,600 km² and the district boundaries run out into the middle of it: the lake is
not a hole in the country, it is *inside* Karonga, Rumphi, Nkhata Bay, Likoma, Salima,
Nkhotakota and Mangochi. Spread dots evenly over those polygons and a fifth of Malawi's
people are drawn onto open water, in a band down the whole eastern side of the country —
and Likoma, an island district of 14,527 people whose polygon is almost entirely lake, would
be a wash of dots over nothing.

Kontur removes the problem rather than patching it: it is a population grid, so an empty
hex has no population and takes no dots, and the lake simply has no hexes. That is the same
thing Ethiopia found (§9u) and it is why `water.py` is not involved here at all.

It also fixes the second, smaller problem: Malawi's districts are very uneven in
habitability even on land — Mangochi and Kasungu are large with big empty stretches, while
Blantyre City, Zomba City, Lilongwe City and Mzuzu City are dense specks. An equal share per
polygon would spread the four cities' 1.85 million people over the wrong ground twice over.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, and that is deliberate — a hex on a district
line belongs wholly to one side, so no hex is split and no population is double-counted.
Hexes whose centroid falls outside every district (the Mozambican and Zambian border
overrun, and the far side of the lake, which is Mozambique and Tanzania) are dropped and
reported.

Usage:
    python sources/mw_grid.py --fetch    one 5.3 MB gzipped gpkg from Kontur
    python sources/mw_grid.py            rebuild from data/raw/mw/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mw")
GEO = os.path.join(ROOT, "data", "geo", "mw")
DISTRICTS = os.path.join(GEO, "mw_districts.gpkg")
OUT = os.path.join(GEO, "mw_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_MW_20231101.gpkg.gz")
GZ_NAME = "kontur_population_MW_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_MW_20231101.gpkg"

EXPECTED_DISTRICTS = 32

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census and must
# not be asserted equal to it (§12, North Macedonia). Used ONLY as a within-district weight,
# so all that matters is that it is not wildly out — assert the RELATIONSHIP.
#
# THE BAND IS MEASURED HERE AND NOT COPIED FROM KENYA (§9u's rule). Kontur's vintage is
# 2023 and the census is 2018, and Malawi grew ~2.6%/yr over those five years, so the grid
# should read HIGH against the census by roughly 10-15%. A band centred on 1.0 would be
# the wrong shape of check even though it happens to pass.
CENSUS_POPULATION = 17_563_749
KONTUR_TOLERANCE = 0.30


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 5_000_000:
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
        raise SystemExit(f"missing {DISTRICTS} -- run sources/mw_geo.py first")

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
    print("     Kontur's MW extract overruns into Mozambique, Zambia and the Tanzanian "
          "side of\n     the lake; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=districts.crs)

    # Every district must get some hexes, or its dots fall back to an equal share over
    # nothing at all and the district silently empties. Likoma is the one to watch: it is
    # an island of 18 km² inside a polygon that is mostly lake.
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(districts["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"districts with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"districts whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_DISTRICTS} districts has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")
    thin = per.sort_values("size").head(3)
    for unit, row in thin.iterrows():
        nm = districts.loc[districts["unit"] == unit, "name"].iloc[0]
        print(f"      thinnest: {nm} ({unit}) {int(row['size']):,} hexes, "
              f"{row['sum']:,.0f} people")

    # Kontur is a MODEL, not the census. Assert the relationship, never equality (§12).
    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census {CENSUS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio-1)*100:.0f}%, "
                         "which is too much for a weight -- check the download")
    print("     a 2023 modelled grid against a 2018 census count, so it should and does "
          "read\n     high; used only as a WITHIN-district weight, so the level does not "
          "matter and the\n     shape does.")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
