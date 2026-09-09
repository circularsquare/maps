"""Uruguay — the placement layer: Kontur 400 m population hexagons, keyed to department.

Writes data/geo/uy/uy_hexes.gpkg.

**URUGUAY IS THE MOST LOPSIDED COUNTRY THIS MAP HAS DRAWN AT ADM1 IN THE AMERICAS.**
Montevideo is 0.3% of the land and 37% of the people; the seventeen departments of the
interior are between 20 and 40 people per km² and their population sits almost entirely in
one departmental capital each, with grazing country in between. Spread a department's dots
evenly over its polygon and Uruguay's religion is painted across the estancias.

The second reason is the coast. Maldonado and Rocha are long thin departments whose people
are on a 200 km strip of Atlantic shoreline — Punta del Este, La Paloma, Chuy — and whose
interiors are almost empty; those two are also the departments with the highest unaffiliated
shares in the country, so getting their dots onto the coast is getting the map's most
striking claim into the right place.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, so a hex on a department line belongs wholly to
one side and no population is double-counted. Hexes whose centroid falls outside every
department (Kontur's UY extract overruns into Brazil and Argentina, and the Río de la Plata
and Uruguay river boundaries are wide) are dropped and reported.

Usage:
    python sources/uy_grid.py --fetch    one ~2 MB gzipped gpkg from Kontur
    python sources/uy_grid.py            rebuild from data/raw/uy/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "uy")
GEO = os.path.join(ROOT, "data", "geo", "uy")
DEPARTMENTS = os.path.join(GEO, "uy_departamentos.gpkg")
POP = os.path.join(GEO, "uy_pop_2023.csv")
OUT = os.path.join(GEO, "uy_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_UY_20231101.gpkg.gz")
GZ_NAME = "kontur_population_UY_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_UY_20231101.gpkg"

EXPECTED_DEPARTMENTS = 19
CAPITAL = "Montevideo"

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Used ONLY as a within-department weight, so
# what matters is that it is not wildly out — assert the RELATIONSHIP.
INE_POPULATION = 3_496_400          # INE's estimate at 30 June 2023
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
        raise SystemExit(f"missing {DEPARTMENTS} — run sources/uy_geo.py first")

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
    print("     Kontur's UY extract overruns into Brazil and Argentina; dropped.")

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

    cap = deps.loc[deps["name"] == CAPITAL, "unit"]
    if len(cap) != 1:
        raise SystemExit(f"{CAPITAL} is not one department in {DEPARTMENTS}")
    cap = cap.iloc[0]
    cap_share = per.loc[cap, "sum"] / out["pop"].sum()
    print(f"    {CAPITAL} ({cap}) holds {cap_share:.1%} of Kontur's Uruguayan population "
          f"in {int(per.loc[cap, 'size']):,} hexes")
    if not 0.25 < cap_share < 0.50:
        raise SystemExit(f"{CAPITAL} holds {cap_share:.1%} of the grid, which is not the "
                         "37% INE counts there — the spatial join is wrong")

    tot = float(out["pop"].sum())
    ratio = tot / INE_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs INE 2023 {INE_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and INE disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight — check the download")
    print("     used only as a WITHIN-department weight, so the level does not matter and "
          "the\n     shape does.")

    print("\n  per-department Kontur/INE ratio (the shape check):")
    ine = pd.read_csv(POP, encoding="utf-8-sig")
    ine = dict(zip(ine["geo_id"].astype(str).str.strip(), ine["pop_2023"]))
    rows = [(deps.loc[deps["unit"] == u, "name"].iloc[0], int(r["size"]), r["sum"] / ine[u])
            for u, r in per.iterrows()]
    for nm, n, r in sorted(rows, key=lambda t: t[2]):
        print(f"      {nm:<20} {n:>8,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
