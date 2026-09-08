"""El Salvador — the placement layer: Kontur 400 m population hexagons, keyed to department.

Writes data/geo/sv/sv_hexes.gpkg.

**EL SALVADOR IS THE OPPOSITE CASE TO GUATEMALA AND STILL NEEDS THIS.** 14 departments over
21,041 km² is **1,503 km² per unit**, the tightest ratio of any LAPOP country here and less
than a third of Guatemala's 4,950. There is no Petén: nothing in the country is empty at that
scale, and the smallest department, Cuscatlán, is 756 km².

What it needs the grid for is the other half of §8.2, and it is severe here: **San Salvador is
1.7 million people in 886 km², a quarter of the country on 4% of its land**, and the
metropolitan area spills across the La Libertad line rather than stopping at it. Spread San
Salvador's dots evenly over its polygon and the Volcán de San Salvador and the Cordillera del
Bálsamo get a quarter of the country's colour. The northern departments have the reverse
problem at a smaller scale: Chalatenango and Morazán are mountain border country whose people
are strung along the valleys.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, and that is deliberate — a hex on a department
line belongs wholly to one side, so no hex is split and no population is double-counted.
Hexes whose centroid falls outside every department (Kontur's SV extract overruns into
Guatemala and Honduras) are dropped and reported.

Usage:
    python sources/sv_grid.py --fetch    one ~2 MB gzipped gpkg from Kontur
    python sources/sv_grid.py            rebuild from data/raw/sv/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sv")
GEO = os.path.join(ROOT, "data", "geo", "sv")
DEPARTMENTS = os.path.join(GEO, "sv_departamentos.gpkg")
OUT = os.path.join(GEO, "sv_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_SV_20231101.gpkg.gz")
GZ_NAME = "kontur_population_SV_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_SV_20231101.gpkg"

EXPECTED_DEPARTMENTS = 14

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Used ONLY as a within-department weight, so
# all that matters is that it is not wildly out — assert the RELATIONSHIP.
#
# THE BAND IS MEASURED HERE AND NOT COPIED (§9u's rule). Kontur's vintage is 2023-11 against a
# 2024 COD-PS projection, so it should read slightly low. The band is wide because El
# Salvador's population is unusually hard to model: the 2007 census came in about a million
# under the projections that preceded it, and emigration keeps the resident count moving.
CODPS_POPULATION = 6_350_969
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
    if not os.path.exists(DEPARTMENTS):
        raise SystemExit(f"missing {DEPARTMENTS} — run sources/sv_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    deps = gpd.read_file(DEPARTMENTS)
    if len(deps) != EXPECTED_DEPARTMENTS:
        raise SystemExit(f"{DEPARTMENTS} has {len(deps)} departments, "
                         f"expected {EXPECTED_DEPARTMENTS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS — reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(deps.crs)
    hexes = hexes.to_crs(deps.crs)

    joined = gpd.sjoin(pts, deps[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every department: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's SV extract overruns into Guatemala and Honduras; dropped.")

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
    print(f"  every one of the {EXPECTED_DEPARTMENTS} departments has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CODPS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs COD-PS {CODPS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and COD-PS disagree by {abs(ratio - 1) * 100:.0f}%, which "
                         "is too much for a weight — check the download")
    print("     used only as a WITHIN-department weight, so the level does not matter and "
          "the\n     shape does.")

    print("\n  per-department Kontur/COD-PS ratio (the shape check):")
    cod = pd.read_csv(os.path.join(RAW, "slv_admpop_adm1_2024.csv"), encoding="utf-8-sig")
    cod = dict(zip(cod["ADM1_PCODE"].astype(str).str.strip(), cod["T_TL"]))
    rows = [(deps.loc[deps["unit"] == u, "name"].iloc[0], int(r["size"]), r["sum"] / cod[u])
            for u, r in per.iterrows()]
    for nm, n, r in sorted(rows, key=lambda t: t[2]):
        print(f"      {nm:<16} {n:>7,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
