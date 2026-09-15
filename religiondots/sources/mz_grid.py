"""Mozambique — the placement layer: Kontur 400 m population hexagons, keyed to province.

Writes data/geo/mz/mz_hexes.gpkg.

Kenya's module (`sources/ke_grid.py`) with Mozambique's files. Eleven provinces for 26.9
million people is coarse, and Niassa alone is 129,000 km² with its people along the lake
shore and a few roads; an equal share of dots per polygon would put Niassa's dots across the
Niassa Special Reserve, where almost nobody lives. So a province's dots are spread over
Kontur hexagons by hex population. It is a POPULATION weight and not a religion one.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, so a hex on a provincial line belongs wholly to
one side. Hexes whose centroid falls outside every province (the ocean edge, the lake, the
border overrun into Malawi, Zambia, Zimbabwe, South Africa, Eswatini and Tanzania) are dropped
and reported.

Usage:
    python sources/mz_grid.py --fetch    one gzipped gpkg from Kontur's public bucket
    python sources/mz_grid.py            rebuild from data/raw/mz/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mz")
GEO = os.path.join(ROOT, "data", "geo", "mz")
PROVINCES = os.path.join(GEO, "mz_provinces.gpkg")
OUT = os.path.join(GEO, "mz_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_MZ_20231101.gpkg.gz")
GZ_NAME = "kontur_population_MZ_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_MZ_20231101.gpkg"

# Kontur is a 2023 model and the census counted 2017, six years of roughly 2.8% annual growth
# earlier, so the two should differ by about a fifth. Used ONLY as a within-province weight,
# so the level does not matter; the tolerance is wide enough for the growth and no wider.
CENSUS_POPULATION = 26_899_105
KONTUR_TOLERANCE = 0.40
EXPECTED = 11


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
    if not os.path.exists(PROVINCES):
        raise SystemExit(f"missing {PROVINCES} -- run sources/mz_geo.py first")

    hexes = gpd.read_file(gpkg)
    # §12: assert the feature count, not the absence of an exception.
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    provs = gpd.read_file(PROVINCES)
    if len(provs) != EXPECTED:
        raise SystemExit(f"{PROVINCES} has {len(provs)} provinces, expected {EXPECTED}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(provs.crs)
    hexes = hexes.to_crs(provs.crs)

    joined = gpd.sjoin(pts, provs[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every province: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%); dropped")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=provs.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(provs["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"provinces with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"provinces whose hexes sum to zero population: {zero}")
    for unit, r in per.iterrows():
        print(f"    {unit}  {int(r['size']):>8,} hexes  {r['sum']:>12,.0f} people")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs 2017 census {CENSUS_POPULATION:,}, ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is more than six years of growth -- check the download")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
