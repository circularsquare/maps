"""Liberia — the placement layer: Kontur 400 m population hexagons, keyed to county.

Writes data/geo/lr/lr_hexes.gpkg.

**THIS IS THE CASE THE GRID EXISTS FOR.** 15 counties over 95,845 km² of land is **6,390 km²
per unit**, coarser than any LAPOP country, and Liberia's people are nothing like evenly
spread inside them:

  * **Montserrado is 36.6% of the country on 1.9% of its land**, and 92% of it is urban.
    Spread its 1.92 million dots over the polygon and Greater Monrovia's colour lands on the
    Bong Range foothills.
  * **Gbarpolu, Grand Gedeh and Sinoe are about 10,000 km² each with under 220,000 people**,
    most of them strung along a road or a river with rainforest between.
  * Liberia is 45% rural on the census's own definition, so this is not a country where the
    error washes out.

[[reference_kontur_resolution_floor]]'s test is whether the grid is finer than the counting
tier: at 6,390 km² per county against 400 m hexes it is finer by four orders of magnitude, so
Kontur is doing almost all of the placement work here and almost none of the counting.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS — a hex on a county line belongs wholly to one
side, so no hex is split and no population is double-counted. Hexes whose centroid falls
outside every county (Kontur's LR extract overruns into Sierra Leone, Guinea and Côte
d'Ivoire) are dropped and reported.

Usage:
    python sources/lr_grid.py --fetch    one gzipped gpkg from Kontur
    python sources/lr_grid.py            rebuild from data/raw/lr/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "lr")
GEO = os.path.join(ROOT, "data", "geo", "lr")
COUNTIES = os.path.join(GEO, "lr_counties.gpkg")
OUT = os.path.join(GEO, "lr_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_LR_20231101.gpkg.gz")
GZ_NAME = "kontur_population_LR_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_LR_20231101.gpkg"

EXPECTED_COUNTIES = 15

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Used ONLY as a within-county weight, so
# what matters is that it is not wildly out — assert the RELATIONSHIP, not the level.
#
# THE BAND IS MEASURED HERE AND NOT COPIED (§9u's rule). The comparison is against the 2022
# census night count, which is what data/geo/lr/lr_lookup.csv carries. Kontur's 2023-11
# vintage is a year later than the census, and the two disagree in a direction nobody can
# predict for Liberia: the 2022 census came in 51% above 2008, far above the projections that
# preceded it, so the modelled surfaces were built against a denominator the census then
# raised. The band is wide for that reason.
CENSUS_2022 = 5_250_187
KONTUR_TOLERANCE = 0.40

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
        raise SystemExit(f"{gpkg} is not a GeoPackage — starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run with --fetch first")
    if not os.path.exists(COUNTIES):
        raise SystemExit(f"missing {COUNTIES} — run sources/lr_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    counties = gpd.read_file(COUNTIES)
    if len(counties) != EXPECTED_COUNTIES:
        raise SystemExit(f"{COUNTIES} has {len(counties)} counties, "
                         f"expected {EXPECTED_COUNTIES}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS — reprojecting first and taking the centroid after moves it.
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
    print("     Kontur's LR extract overruns into Sierra Leone, Guinea and Côte d'Ivoire; "
          "dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=counties.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(counties["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"counties with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"counties whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_COUNTIES} counties has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_2022
    print(f"\n  Kontur {tot:,.0f} vs the 2022 census {CENSUS_2022:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight — check the download")
    print("     used only as a WITHIN-county weight, so the level does not matter and the "
          "shape does.")

    print("\n  per-county Kontur/census ratio (the shape check):")
    cod = dict(zip(counties["unit"], counties["pop"]))
    nm = dict(zip(counties["unit"], counties["name"]))
    rows = [(nm[u], int(r["size"]), r["sum"] / cod[u]) for u, r in per.iterrows()]
    for name, n, r in sorted(rows, key=lambda t: t[2]):
        print(f"      {name:<18} {n:>7,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
