"""Costa Rica — the placement layer: Kontur 400 m population hexagons, keyed to province.

Writes data/geo/cr/cr_hexes.gpkg.

**SEVEN UNITS OVER 51,169 km² IS 7,310 km² APIECE**, the loosest ratio of any country drawn
from this survey — a third again as coarse as Guatemala's 4,950 and nearly five times El
Salvador's 1,503. So §8.2 binds harder here than anywhere else in the set, and it binds in
both of its directions at once.

The empty half: **Guanacaste, Puntarenas and Limón are 30,668 km² between them and hold 27% of
the country.** Puntarenas alone runs the whole Pacific coast from Nicaragua to Panama and
includes the Osa peninsula and Corcovado; Limón is Talamanca and the Caribbean lowland. Spread
their dots evenly and the map paints rainforest, the Cordillera de Talamanca and the Área de
Conservación Guanacaste in whatever colour those provinces happen to be.

The crowded half: **the Valle Central is most of Costa Rica's people in a small share of its
land**, and it crosses four of the seven provincial lines rather than sitting inside one. San
José, Alajuela, Cartago and Heredia between them are 20,501 km² and 73% of the population, but
the actual settlement is the ring from Alajuela through Heredia and San José to Cartago, a
plateau maybe 60 km across, with the rest of those four provinces reaching north into the San
Carlos plain and south over Talamanca. Heredia is the sharpest case: the province is a narrow
strip running from the Central Valley to the Nicaraguan border, and its population sits
entirely at the southern end.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, so a hex on a provincial line belongs wholly to
one side and no population is split or double-counted. Hexes whose centroid falls outside every
province are dropped and reported; Kontur's CR extract overruns into Nicaragua and Panama.

Usage:
    python sources/cr_grid.py --fetch    one ~2.4 MB gzipped gpkg from Kontur
    python sources/cr_grid.py            rebuild from data/raw/cr/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "cr")
GEO = os.path.join(ROOT, "data", "geo", "cr")
PROVINCES = os.path.join(GEO, "cr_provincias.gpkg")
OUT = os.path.join(GEO, "cr_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_CR_20231101.gpkg.gz")
GZ_NAME = "kontur_population_CR_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_CR_20231101.gpkg"

EXPECTED_PROVINCES = 7

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Used ONLY as a within-province weight, so
# what matters is the SHAPE and not the level — assert the relationship, loosely.
#
# THE BAND IS MEASURED HERE AND NOT COPIED (§9u's rule). Kontur's vintage is 2023-11 against
# INEC's 2022 estimate, one year apart in Kontur's favour, so it should read very slightly high
# if it reads true at all.
INEC_POPULATION_2022 = 5_044_197
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
        raise SystemExit(f"missing {PROVINCES} — run sources/cr_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    provs = gpd.read_file(PROVINCES)
    if len(provs) != EXPECTED_PROVINCES:
        raise SystemExit(f"{PROVINCES} has {len(provs)} provinces, "
                         f"expected {EXPECTED_PROVINCES}")

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
    print(f"\n  hexes whose centroid is outside every province: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's CR extract overruns into Nicaragua and Panama; dropped.")

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
    print(f"  every one of the {EXPECTED_PROVINCES} provinces has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / INEC_POPULATION_2022
    print(f"\n  Kontur {tot:,.0f} vs INEC 2022 {INEC_POPULATION_2022:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and INEC disagree by {abs(ratio - 1) * 100:.0f}%, which is "
                         "too much for a weight — check the download")
    print("     used only as a WITHIN-province weight, so the level does not matter and the"
          "\n     shape does.")

    print("\n  per-province Kontur/INEC ratio (the shape check):")
    lut = pd.read_csv(os.path.join(GEO, "cr_lookup.csv"))
    inec = dict(zip(lut["unit"], lut["pop"]))
    rows = [(provs.loc[provs["unit"] == u, "name"].iloc[0], int(r["size"]),
             r["sum"] / inec[u]) for u, r in per.iterrows()]
    for nm, n, r in sorted(rows, key=lambda t: t[2]):
        print(f"      {nm:<14} {n:>7,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
