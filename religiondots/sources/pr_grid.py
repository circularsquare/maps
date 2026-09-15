"""Puerto Rico — the placement layer: Kontur 400 m population hexagons, keyed to region.

Writes data/geo/pr/pr_hexes.gpkg. `sources/pr.md` is this country's record.

Six regions over 8,900 km2 is about 1,500 km2 a unit, and every region holds both a dense
coastal strip and empty uplands or forest: Este runs from Carolina's edge out to El Yunque and
the Sierra de Luquillo, Centro is the Cordillera Central, and Oeste's Mayagüez polygon includes
Mona Island, 57 km offshore with nobody on it. Kontur puts each region's dots on the people.

THE JOIN IS A SPATIAL ONE, on hex centroids, as in `sources/bz_grid.py`: a hex on a region line
belongs wholly to one side. Hexes whose centroid falls outside every region (sea, and the edges
of the cartographic boundary file's coastline) are dropped and reported.

Usage:
    python sources/pr_grid.py --fetch    one ~1 MB gzipped gpkg from Kontur
    python sources/pr_grid.py            rebuild from data/raw/pr/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pr")
GEO = os.path.join(ROOT, "data", "geo", "pr")
REGIONS = os.path.join(GEO, "pr_regions.gpkg")
LOOKUP = os.path.join(GEO, "pr_lookup.csv")
OUT = os.path.join(GEO, "pr_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_PR_20231101.gpkg.gz")
GZ_NAME = "kontur_population_PR_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_PR_20231101.gpkg"

EXPECTED_REGIONS = 6

# Kontur is a model, used only as a within-region weight (§12, North Macedonia): assert the
# relationship, never equality. Its vintage is November 2023; the Census Bureau's July 2024
# estimate is the comparison.
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
    if not (os.path.exists(gz) and os.path.getsize(gz) > 100_000):
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
        raise SystemExit(f"{gpkg} is not a GeoPackage -- starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd
    import pandas as pd

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or (not os.path.exists(gpkg)
                                 and os.path.exists(os.path.join(RAW, GZ_NAME))):
        fetch()
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(REGIONS):
        raise SystemExit(f"missing {REGIONS} -- run sources/pr_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    regions = gpd.read_file(REGIONS)
    if len(regions) != EXPECTED_REGIONS:
        raise SystemExit(f"{REGIONS} has {len(regions)} regions, expected {EXPECTED_REGIONS}")

    # Centroid in the CRS the hexes were tiled in, then reproject the points.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(regions.crs)
    hexes = hexes.to_crs(regions.crs)
    minx, miny, maxx, maxy = pts.total_bounds
    if not (-68.2 < minx and maxx < -65.0 and 17.5 < miny and maxy < 18.8):
        raise SystemExit(f"Kontur's PR extract spans {pts.total_bounds}, which is not Puerto Rico")

    joined = gpd.sjoin(pts, regions[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every region: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%); dropped")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=regions.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(regions["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"regions with no populated hex: {missing}")
    if (per["sum"] <= 0).any():
        raise SystemExit(f"regions whose hexes sum to zero: {sorted(per.index[per['sum'] <= 0])}")

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str, "unit": str})
    cen = dict(zip(lut["unit"], lut["pop_2024"]))
    name = dict(zip(lut["unit"], lut["name"]))
    tot = float(out["pop"].sum())
    ratio = tot / sum(cen.values())
    print(f"  Kontur {tot:,.0f} against the Census Bureau's July 2024 {sum(cen.values()):,}, "
          f"ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the estimates disagree by {abs(ratio - 1) * 100:.0f}%, too "
                         "much for a weight; check the download")
    print("  per-region Kontur/estimate ratio (the shape check, printed not asserted):")
    for unit, row in per.iterrows():
        print(f"      {name[unit]:<14} {int(row['size']):>6,} hexes  {row['sum']:>10,.0f}  "
              f"{row['sum'] / cen[unit]:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
