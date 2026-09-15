"""Uzbekistan — the placement layer: Kontur 400 m population hexagons, keyed to region.

Writes data/geo/uz/uz_hexes.gpkg.

Fourteen units over 448,978 km2 is 32,000 km2 apiece, and most of that is desert.
Karakalpakstan is 166,590 km2, much of it the Kyzylkum and the dry bed of the Aral Sea, with
its people along the Amu Darya delta; Navoi is 111,000 km2 of the same desert around a string
of mining towns. Spread either region's dots over its polygon and most of the colour lands on
sand. The Fergana Valley regions are the opposite case: dense oases walled by mountains.

THE JOIN IS SPATIAL, on hex CENTROIDS, so a hex on a regional line belongs wholly to one side.
Hexes whose centroid falls outside every region are dropped and reported. This is
`sources/kg_grid.py` with the paths changed.

Usage:
    python sources/uz_grid.py --fetch    one ~7.7 MB gzipped gpkg from Kontur
    python sources/uz_grid.py            rebuild
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "geo", "kontur")
GEO = os.path.join(ROOT, "data", "geo", "uz")
REGIONS = os.path.join(GEO, "uz_regions.gpkg")
LOOKUP = os.path.join(GEO, "uz_lookup.csv")
OUT = os.path.join(GEO, "uz_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_UZ_20231101.gpkg.gz")
GZ_NAME = "kontur_population_UZ_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_UZ_20231101.gpkg"

EXPECTED_UNITS = 14

# Kontur is modelled (GHSL, HRSL, building footprints), not a census, and is used only as a
# within-region weight. Its vintage is 2023-11 against the office's 1 January 2026, about two
# years of growth near 2% a year, so it should read somewhat low.
OFFICE_POPULATION = 38_236_700
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
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage — starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run with --fetch first")
    if not os.path.exists(REGIONS):
        raise SystemExit(f"missing {REGIONS} — run sources/uz_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    reg = gpd.read_file(REGIONS)
    if len(reg) != EXPECTED_UNITS:
        raise SystemExit(f"{REGIONS} has {len(reg)} units, expected {EXPECTED_UNITS}")

    # Centroid in the CRS the hexes were tiled in, then reproject the points.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(reg.crs)
    hexes = hexes.to_crs(reg.crs)
    joined = gpd.sjoin(pts, reg[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every region: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%); dropped")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=reg.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(reg["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"units with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"units whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} units has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / OFFICE_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the office's {OFFICE_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the office disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight — check the download")

    print("\n  per-region Kontur/office ratio (the shape check):")
    lut = pd.read_csv(LOOKUP)
    office = dict(zip(lut["geo_id"], lut["pop"]))
    name = dict(zip(lut["geo_id"], lut["name"]))
    for u, r in sorted(per.iterrows(), key=lambda t: t[1]["sum"] / office[t[0]]):
        print(f"      {name[u]:<16} {int(r['size']):>7,} hexes   {r['sum'] / office[u]:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
