"""Kenya — the placement layer: Kontur 400 m population hexagons, keyed to county.

Writes data/geo/ke/ke_hexes.gpkg.

**KENYA IS WHY §8.2's PLACEMENT PROBLEM IS NOT COSMETIC.** The counts are 47 counties for
47.2M people, the coarsest counting geography on this map, and the counties are wildly
uneven in habitability: Turkana is 68,680 km² of mostly desert with 926,976 people, Marsabit
70,961 km² with 459,785, Wajir 56,686 km² with 781,263. Spread those uniformly and the
northern half of Kenya fills with an even wash of dots over ground where nobody lives — and
because Wajir, Mandera and Garissa are each 97-99% Muslim, that wash is one colour and it is
the most visually dominant thing on the map. It would not be a rounding error; it would be
the wrong picture.

**THIS IS THE FIRST COUNTRY TO USE KONTUR, and sources.md §5 chose it long ago** — "400m H3,
on HDX — vector hexagons out of the box" — without any country having needed it. Kenya needs
it. The two alternatives were both worse:

  * **COD ADM2** (290 constituencies) nests cleanly by pcode and needs no spatial join, but
    it is still 6 blobs of even wash across Turkana rather than 1. Better, not right.
  * **Weighting those by sub-county population** is what the Philippines does and it does not
    work here: HDX's `ken_admpop_2019.xlsx` has 345 ADM2 rows against COD's 290 polygons,
    because Kenya's administrative SUB-COUNTIES and its CONSTITUENCIES are different tiers
    with different names. Matched within county by name it is 182 of 345 rows and **40.5% of
    the population unplaced**. Two files at "ADM2" are not two files at the same level.

Kontur sidesteps both: it is a population grid, so it needs no administrative alignment at
all, and every hex carries its own population.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, and that is deliberate — a hex on a county line
belongs wholly to one side, so no hex is split and no population is double-counted. Hexes
whose centroid falls outside every county (the Indian Ocean edge, the Somali and Ethiopian
border strips where Kontur's coverage overruns) are dropped and reported.

Usage:
    python sources/ke_grid.py --fetch    one 16.5 MB gzipped gpkg from Kontur via HDX
    python sources/ke_grid.py            rebuild from data/raw/ke/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ke")
GEO = os.path.join(ROOT, "data", "geo", "ke")
COUNTIES = os.path.join(GEO, "ke_counties.gpkg")
OUT = os.path.join(GEO, "ke_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_KE_20231101.gpkg.gz")
GZ_NAME = "kontur_population_KE_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_KE_20231101.gpkg"

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census and must
# not be asserted equal to it (§12, North Macedonia). It is used ONLY as a within-county
# weight, so all that matters is that it is not wildly out — assert the RELATIONSHIP.
CENSUS_POPULATION = 47_564_296
KONTUR_TOLERANCE = 0.25        # national totals within 25% of each other


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 10_000_000:
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
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(COUNTIES):
        raise SystemExit(f"missing {COUNTIES} -- run sources/ke_geo.py first")

    hexes = gpd.read_file(gpkg)
    # §12: assert the feature count, not the absence of an exception.
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    counties = gpd.read_file(COUNTIES)
    if len(counties) != 47:
        raise SystemExit(f"{COUNTIES} has {len(counties)} counties, expected 47")

    # Kontur ships in EPSG:3857. Reproject the hexes, not the counties: the centroid has to
    # be taken in the CRS the hexes were tiled in, or a hex near the poles shifts.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(counties.crs)
    hexes = hexes.to_crs(counties.crs)

    joined = gpd.sjoin(pts, counties[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every county: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     these are the ocean edge and the Somali/Ethiopian/Ugandan border overrun; "
          "dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=counties.crs)

    # Every county must get some hexes, or its dots fall back to an equal share over
    # nothing at all and the county silently empties.
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(counties["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"counties with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"counties whose hexes sum to zero population: {zero}")
    print(f"  every one of the 47 counties has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    # Kontur is a MODEL, not the census. Assert the relationship, never equality (§12).
    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census {CENSUS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio-1)*100:.0f}%, "
                         "which is too much for a weight -- check the download")
    print("     a modelled grid against a census count; used only as a WITHIN-county "
          "weight,\n     so the level does not matter and the shape does.")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
