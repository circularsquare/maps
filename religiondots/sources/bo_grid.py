"""Bolivia — the placement layer: Kontur 400 m population hexagons, keyed to department.

Writes data/geo/bo/bo_hexes.gpkg.

§8.2's two reasons, both at full strength. **Emptiness**: Beni, Pando and the Santa Cruz
lowlands (the Chiquitanía and the Chaco) are about two thirds of the land and under a fifth of
the people, and the Altiplano's salt flats and lakes are empty. **Concentration**: the La
Paz-El Alto, Santa Cruz de la Sierra and Cochabamba metropolitan areas hold close to half the
country. An equal share per polygon would paint Santa Cruz's evangelicals across the Pantanal.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, `sources/co_grid.py`'s construction. Hexes whose
centroid is outside every department (the BO extract overruns into Peru, Brazil, Paraguay,
Argentina and Chile) are dropped and reported. COD-AB's ADM1 encloses Lake Titicaca and the
salt flats, which ADM2 and ADM3 leave as holes, so every department polygon is solid.

Usage:
    python sources/bo_grid.py --fetch    one ~12 MB gzipped gpkg from Kontur
    python sources/bo_grid.py            rebuild from data/raw/bo/ (a minute or two)
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bo")
GEO = os.path.join(ROOT, "data", "geo", "bo")
UNITS = os.path.join(GEO, "bo_departamentos.gpkg")
POP = os.path.join(GEO, "bo_pop_2024.csv")
OUT = os.path.join(GEO, "bo_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_BO_20231101.gpkg.gz")
GZ_NAME = "kontur_population_BO_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_BO_20231101.gpkg"

EXPECTED = 9

# Kontur is modelled, not a census (§12, North Macedonia); only the relationship is asserted.
# Measured here, not copied (§9u): a 2023-11 grid against the 2024 census.
CENSUS_POPULATION = 11_365_333
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


def unpack():
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 500_000:
        return gpkg
    if not os.path.exists(gz):
        raise SystemExit(f"missing {gz} — run with --fetch first")
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage — starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")
    return gpkg


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    gpkg = unpack()
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS} — run sources/bo_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    deps = gpd.read_file(UNITS)
    if len(deps) != EXPECTED:
        raise SystemExit(f"{UNITS} has {len(deps)} departments, expected {EXPECTED}")

    # centroid in the tiling CRS, then reproject the points (reprojecting first moves it)
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(deps.crs)
    hexes = hexes.to_crs(deps.crs)

    joined = gpd.sjoin(pts, deps[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every department: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%); dropped")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=deps.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(deps["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"departments with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"departments whose hexes sum to zero population: {zero}")
    print(f"  every department has hexes: {per['size'].min():,}-{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the 2024 census {CENSUS_POPULATION:,}, ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, which "
                         "is too much for a weight — check the download")

    print("\n  per-department Kontur/census ratio (the shape check):")
    cen = pd.read_csv(POP, encoding="utf-8-sig")
    cen = dict(zip(cen["geo_id"].astype(str).str.strip(), cen["pop"]))
    rows = [(deps.loc[deps["unit"] == u, "name"].iloc[0], int(r["size"]), r["sum"] / cen[u])
            for u, r in per.iterrows()]
    for nm, n, r in sorted(rows, key=lambda t: t[2]):
        print(f"      {nm[:34]:<34} {n:>8,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
