"""Ecuador — the placement layer: Kontur 400 m population hexagons, keyed to province.

Writes data/geo/ec/ec_hexes.gpkg.

**ECUADOR NEEDS THIS MORE THAN EITHER COUNTRY BEFORE IT, AND FOR BOTH OF §8.2'S REASONS AT
ONCE.** Guatemala's problem is emptiness and El Salvador's is concentration; Ecuador has each
of them worse than the country that taught it.

**EMPTINESS.** The six Amazonian provinces — Morona Santiago, Napo, Pastaza, Orellana,
Sucumbíos, Zamora Chinchipe — are **44% of the country's land and 5.4% of its people**.
Pastaza alone is 29,000 km², bigger than Belgium, with 111,915 people almost all of them on
the Puyo road; the rest is forest with nobody in it. Spread Pastaza's dots over Pastaza's
polygon and the map paints a religion across a roadless basin.

**CONCENTRATION.** Guayas and Pichincha are **44% of the people on 5% of the land**, and
inside them it is worse still: Guayaquil and Quito are each a couple of hundred km² holding
two to three million people. An equal share per polygon puts Quito's Catholics on the slopes
of Pichincha volcano and Guayaquil's evangelicals in the rice fields of the Guayas basin.

**AND THE ANDES DO IT AT A THIRD SCALE.** The sierra provinces are ridge-and-valley country
where the people are in the valleys, so even a small province like Cañar or Bolívar is mostly
uninhabited ground at 400 m resolution.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, and that is deliberate — a hex on a province
line belongs wholly to one side, so no hex is split and no population is double-counted.
Hexes whose centroid falls outside every province (Kontur's EC extract overruns into
Colombia and Peru) are dropped and reported.

**GALÁPAGOS MUST COME OUT WITH HEXES.** It is 1,000 km offshore and the one province LAPOP
never sampled, so it is also the one whose hexes nothing else would notice were missing.
Asserted by name below rather than left to the generic "every province has hexes" check.

Usage:
    python sources/ec_grid.py --fetch    one ~8 MB gzipped gpkg from Kontur
    python sources/ec_grid.py            rebuild from data/raw/ec/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ec")
GEO = os.path.join(ROOT, "data", "geo", "ec")
PROVINCES = os.path.join(GEO, "ec_provincias.gpkg")
POP = os.path.join(GEO, "ec_pop_2022.csv")
OUT = os.path.join(GEO, "ec_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_EC_20231101.gpkg.gz")
GZ_NAME = "kontur_population_EC_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_EC_20231101.gpkg"

EXPECTED_PROVINCES = 24
ISLAND_PROVINCE = "Galápagos"

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Used ONLY as a within-province weight, so
# what matters is that it is not wildly out — assert the RELATIONSHIP.
#
# THE BAND IS MEASURED HERE AND NOT COPIED (§9u's rule). Kontur's vintage is 2023-11 against
# a November 2022 census, so it should read slightly HIGH — the opposite sign to El Salvador,
# where a 2023 grid met a 2024 projection.
CENSUS_POPULATION = 16_938_986
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
        raise SystemExit(f"missing {PROVINCES} — run sources/ec_geo.py first")

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
    print("     Kontur's EC extract overruns into Colombia and Peru; dropped.")

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

    # The islands, explicitly. Galápagos is 1,000 km offshore and is the province nothing
    # else in this build would miss — sources/ec.py never sees a LAPOP row for it.
    isl = provs.loc[provs["name"] == ISLAND_PROVINCE, "unit"]
    if len(isl) != 1:
        raise SystemExit(f"{ISLAND_PROVINCE} is not one province in {PROVINCES}")
    isl = isl.iloc[0]
    if isl not in per.index or per.loc[isl, "sum"] <= 0:
        raise SystemExit(f"{ISLAND_PROVINCE} has no populated hexes — Kontur's EC extract "
                         "does not reach the archipelago and the province cannot be placed")
    print(f"    {ISLAND_PROVINCE} ({isl}): {int(per.loc[isl, 'size']):,} hexes, "
          f"{per.loc[isl, 'sum']:,.0f} people")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs INEC 2022 census {CENSUS_POPULATION:,} — "
          f"ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight — check the download")
    print("     used only as a WITHIN-province weight, so the level does not matter and "
          "the\n     shape does.")

    print("\n  per-province Kontur/census ratio (the shape check):")
    cod = pd.read_csv(POP, encoding="utf-8-sig")
    cod = dict(zip(cod["geo_id"].astype(str).str.strip(), cod["pop"]))
    rows = [(provs.loc[provs["unit"] == u, "name"].iloc[0], int(r["size"]), r["sum"] / cod[u])
            for u, r in per.iterrows()]
    for nm, n, r in sorted(rows, key=lambda t: t[2]):
        print(f"      {nm:<34} {n:>8,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
