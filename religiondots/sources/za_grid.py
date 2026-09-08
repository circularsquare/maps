"""South Africa — the placement layer: Kontur 400 m population hexagons, keyed to province.

Writes data/geo/za/za_hexes.gpkg.

**NINE UNITS OVER 1.22 MILLION KM², AND THE POPULATION IS NOWHERE NEAR EVENLY SPREAD**, so
an equal share of dots per polygon would draw a different country. Northern Cape is 30.5% of
South Africa's land and 2.2% of its people; Gauteng is 1.5% of the land and 23.9% of the
people, a density ratio of about 470 to 1. Spread dots evenly over the province outlines and
the Karoo and the Kalahari come out as populous as the Witwatersrand, which is the single
most misleading thing this map could do with this country.

It matters more here than in most places because the units are so few. With nine polygons
there is no cancelling out: every dot in Northern Cape is placed by this grid and by
nothing else, and the province is the size of Germany.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, so a hex on a provincial line belongs wholly to
one side and no population is split or double-counted.

**LESOTHO IS THE ONE TO WATCH, AND ESWATINI BESIDE IT.** Lesotho is entirely surrounded by
South Africa and Kontur's ZA extract does not respect that as reliably as the boundary file
does; any hex whose centroid lands inside the Lesotho hole falls outside all nine provinces
and is dropped, which is correct, but it means the dropped-population line is expected to be
non-zero and must not be read as a fault. It is reported rather than silently discarded.
The same goes for the Mozambique and Zimbabwe borders in the north east.

**PRINCE EDWARD AND MARION ISLANDS** are South African territory 1,700 km south east of
Cape Town, in Western Cape. They have no permanent population, only a rotating research
station, so Kontur has nothing there and nothing is drawn; they are mentioned because a
bounding box computed from this layer will not include them and that is the right answer.

Usage:
    python sources/za_grid.py --fetch    one ~50 MB gzipped gpkg from Kontur
    python sources/za_grid.py            rebuild from data/raw/za/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "za")
GEO = os.path.join(ROOT, "data", "geo", "za")
UNITS = os.path.join(GEO, "za_provinces.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "za.csv")
OUT = os.path.join(GEO, "za_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_ZA_20231101.gpkg.gz")
GZ_NAME = "kontur_population_ZA_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_ZA_20231101.gpkg"

EXPECTED_UNITS = 9

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a count and must not
# be asserted equal to one (§12, North Macedonia). It is used ONLY as a within-province
# weight, so what matters is that it is not wildly out, and the band is deliberately wide.
#
# The comparison is against the people actually drawn from data/normalized/za.csv, which is
# the CS 2016 answer universe. Kontur's grid is dated November 2023 against a 2016 survey on
# a country growing about 1.3%/yr, and the CSV also drops the `Do not know` and `Unspecified`
# answers, so the ratio SHOULD read high and a value near 1.10 is the expected result rather
# than a problem.
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
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS} -- run sources/za_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} units, expected {EXPECTED_UNITS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS -- reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every province: "
          f"{int(outside.sum()):,} ({lost:,.0f} people, "
          f"{100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's ZA extract overruns the border and, more to the point, covers the\n"
          "     Lesotho enclave and the Eswatini salient, neither of which is a South\n"
          "     African province; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    names = {u: n for u, n in zip(units["unit"], units["name"])}
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit("provinces with no populated hex: "
                         + ", ".join(f"{names.get(u, u)} ({u})" for u in missing))
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"provinces whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} provinces has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")

    print("\n  population per province, Kontur's own view:")
    for unit, row in per.sort_values("sum", ascending=False).iterrows():
        print(f"    {names.get(unit, unit):16s} {int(row['size']):>9,} hexes  "
              f"{row['sum']:>12,.0f}  {100.0 * row['sum'] / per['sum'].sum():5.1f}%")

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/za.py first")
    drawn = pd.read_csv(NORM, keep_default_na=False, na_values=[""])["count"].sum()
    tot = float(out["pop"].sum())
    ratio = tot / drawn
    print(f"\n  Kontur {tot:,.0f} vs the drawn CS 2016 answers {drawn:,} - "
          f"ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and CS 2016 disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much even for a weight -- check the download")
    print("     a November 2023 model against a 2016 survey whose non-answers are not "
          "drawn, so\n     it should and does read high; used only as a WITHIN-province "
          "weight, so the level\n     does not matter and the shape does.")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
