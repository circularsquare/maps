"""Tanzania — the placement layer: Kontur 400 m population hexagons, keyed to the 30 units.

Writes data/geo/tz/tz_hexes.gpkg.

30 units over 947,000 km2 is about 31,600 km2 a unit, coarser even than Nigeria's states, and
Tanzania's people are concentrated in a few belts inside units that are mostly thinly settled:
the lake shore around Mwanza, the Kilimanjaro and Meru slopes, the southern highlands, the coast.
Tabora is 76,000 km2 and Dar es Salaam 1,400 km2. [[reference_kontur_resolution_floor]]'s test
is whether the grid is finer than the counting tier, and it is by five orders of magnitude, so
Kontur does all of the placement and none of the counting.

THE JOIN IS SPATIAL, on hex CENTROIDS, so a hex on a region line belongs wholly to one side and
no population is counted twice. Hexes whose centroid is outside every unit (Kontur's TZ extract
overruns into Kenya, Uganda, Rwanda, Burundi, the DR Congo, Zambia, Malawi and Mozambique, and
onto the lakes) are dropped and reported.

Usage:
    python sources/tz_grid.py --fetch    one gzipped gpkg from Kontur
    python sources/tz_grid.py            rebuild from data/raw/tz/
"""

import gzip
import os
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tz")
GEO = os.path.join(ROOT, "data", "geo", "tz")
UNITS = os.path.join(GEO, "tz_regions.gpkg")
OUT = os.path.join(GEO, "tz_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_TZ_20231101.gpkg.gz")
GZ_NAME = "kontur_population_TZ_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_TZ_20231101.gpkg"

EXPECTED_UNITS = 30

# Kontur is modelled (GHSL, HRSL, building footprints), not a census, and is used only as a
# within-unit weight (§12, North Macedonia). Assert that it is not wildly out, not that it agrees.
CENSUS_2022 = 61_741_120
KONTUR_TOLERANCE = 0.25

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
        r = requests.get(GZ_URL, timeout=3600, stream=True, headers={"User-Agent": UA})
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
    if magic[:4] != b"SQLi":                             # §5a: a gunzip that runs is not a gpkg
        raise SystemExit(f"{gpkg} is not a GeoPackage, starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg}; run with --fetch first")
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS}; run sources/tz_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} units, expected {EXPECTED_UNITS}")

    # Centroid in the CRS the hexes were tiled in, then reproject the POINTS.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every unit: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%); dropped")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"units with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"units whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} units has hexes: "
          f"{per['size'].min():,} to {per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_2022
    print(f"\n  Kontur {tot:,.0f} vs NBS 2022 census {CENSUS_2022:,}: ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, which "
                         "is too much for a weight; check the download")

    print("\n  per-unit Kontur/census ratio (the shape check):")
    cen = dict(zip(units["unit"], units["pop"]))
    nm = dict(zip(units["unit"], units["name"]))
    rows = [(nm[u], int(r["size"]), r["sum"] / cen[u]) for u, r in per.iterrows()]
    for name, n, r in sorted(rows, key=lambda t: t[2]):
        print(f"      {name:<24} {n:>9,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
