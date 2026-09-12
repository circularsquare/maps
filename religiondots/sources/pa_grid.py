"""Panama — the placement layer: Kontur 400 m population hexagons, keyed to province.

Writes data/geo/pa/pa_hexes.gpkg.

**PANAMA IS THE GUATEMALA CASE, HARDER.** Twelve units over 75,000 km² is 6,285 km² apiece,
and the country is mostly empty in a way none of the other LAPOP countries are: Darien is
12,030 km² of forest holding 54,235 people, and the single unit that dominates the map,
Panama with Panama Oeste, is 11,690 km² of which the metropolitan area is a strip along the
canal and the Pacific coast. Spread that unit's 2.09 million dots evenly over its polygon and
half the country's colour lands in the Chagres watershed and the Darien approaches.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, and that is deliberate — a hex on a province line
belongs wholly to one side, so no hex is split and no population is double-counted. Hexes
whose centroid falls outside every unit (Kontur's PA extract overruns into Costa Rica and
Colombia) are dropped and reported.

**Kuna Yala and Embera keep their hexes.** They draw no religion, because LAPOP never sampled
them, but they are part of the country and the grid should say so rather than leave a hole
that later reads as an error.

Usage:
    python sources/pa_grid.py --fetch    one gzipped gpkg from Kontur
    python sources/pa_grid.py            rebuild from data/raw/pa/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pa")
GEO = os.path.join(ROOT, "data", "geo", "pa")
PROVINCES = os.path.join(GEO, "pa_provincias.gpkg")
OUT = os.path.join(GEO, "pa_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_PA_20231101.gpkg.gz")
GZ_NAME = "kontur_population_PA_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_PA_20231101.gpkg"

EXPECTED_UNITS = 12

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Used ONLY as a within-unit weight, so all
# that matters is that it is not wildly out — assert the RELATIONSHIP.
#
# THE BAND IS MEASURED HERE AND NOT COPIED (§9u's rule). Kontur's vintage is 2023-11 and
# COD-PS's is the 2023 census year itself, so they should be close.
CODPS_POPULATION = 4_064_445
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
    if not os.path.exists(PROVINCES):
        raise SystemExit(f"missing {PROVINCES} — run sources/pa_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    provs = gpd.read_file(PROVINCES)
    if len(provs) != EXPECTED_UNITS:
        raise SystemExit(f"{PROVINCES} has {len(provs)} units, expected {EXPECTED_UNITS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS — reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(provs.crs)
    hexes = hexes.to_crs(provs.crs)

    joined = gpd.sjoin(pts, provs[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every unit: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's PA extract overruns into Costa Rica and Colombia; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=provs.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(provs["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"units with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"units whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} units has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CODPS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs COD-PS {CODPS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and COD-PS disagree by {abs(ratio - 1) * 100:.0f}%, which "
                         "is too much for a weight — check the download")
    print("     used only as a WITHIN-unit weight, so the level does not matter and the\n"
          "     shape does.")

    print("\n  per-unit Kontur/COD-PS ratio (the shape check):")
    cod = pd.read_csv(os.path.join(RAW, "pan_admpop_adm1_2023.csv"), encoding="utf-8-sig")
    cod = dict(zip(cod["ADM1_PCODE"].astype(str).str.strip(), cod["T_TL"]))
    cod["PA12"] = cod["PA12"] + cod.pop("PA11")      # the dissolve, mirrored here
    rows = [(provs.loc[provs["unit"] == u, "name"].iloc[0], int(r["size"]), r["sum"] / cod[u])
            for u, r in per.iterrows()]
    for nm, n, r in sorted(rows, key=lambda t: t[2]):
        print(f"      {nm:<22} {n:>7,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
