"""Dominican Republic — the placement layer: Kontur 400 m population hexagons, keyed to province.

Writes data/geo/do/do_hexes.gpkg.

32 provinces over 48,671 km² is **1,521 km² per unit**, almost exactly El Salvador's ratio and
a third of Guatemala's, so this country is not drawn for §8.2's emptiness reason. It is drawn
for the concentration one, and the concentration is extreme in a way that is specific to
Hispaniola.

**Santo Domingo province is 2.77 million people ringing a capital it does not contain.** The
Distrito Nacional is 1.03 million in 91 km², so the two together are a third of the country
inside a 40 km circle, and the province's own polygon runs from that ring north-east into the
empty Sierra de Yamasá. Spread its dots evenly and a third of the Dominican Republic's colour
lands on a mountain range.

The reverse problem is on the other side of the island. **Pedernales, Independencia and Elías
Piña are the Haitian border**, and their people are strung along a handful of roads with the
Sierra de Bahoruco, the Hoya de Enriquillo and Lago Enriquillo itself in between — the lake is
a 350 km² inland sea below sea level and nobody lives on it. Those three provinces are also
where the map's no-religion share peaks, so getting their dots onto the settlements rather
than into the salt pans is not cosmetic.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, and that is deliberate — a hex on a province line
belongs wholly to one side, so no hex is split and no population is double-counted. Hexes
whose centroid falls outside every province are dropped and reported; Kontur's DO extract is
cut at the international border, so what is lost here is coastline rather than Haiti.

Usage:
    python sources/do_grid.py --fetch    one ~4 MB gzipped gpkg from Kontur
    python sources/do_grid.py            rebuild from data/raw/do/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "do")
GEO = os.path.join(ROOT, "data", "geo", "do")
PROVINCES = os.path.join(GEO, "do_provincias.gpkg")
OUT = os.path.join(GEO, "do_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_DO_20231101.gpkg.gz")
GZ_NAME = "kontur_population_DO_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_DO_20231101.gpkg"

EXPECTED_PROVINCES = 32

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Used ONLY as a within-province weight, so
# all that matters is that it is not wildly out — assert the RELATIONSHIP.
#
# THE BAND IS MEASURED HERE AND NOT COPIED (§9u's rule). Kontur's vintage is 2023-11 against a
# November 2022 census, so it should read close. The band is wide because Hispaniola is one of
# the harder islands to model from buildings: the Haitian-origin population is undercounted in
# every Dominican source and the batey settlements are small and dispersed.
CENSUS_POPULATION = 10_771_504
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

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} — run with --fetch first")
    if not os.path.exists(PROVINCES):
        raise SystemExit(f"missing {PROVINCES} — run sources/do_geo.py first")

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
    print("     the DO extract is cut at the Haitian border, so this is coastline; dropped.")

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
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the 2022 census {CENSUS_POPULATION:,} — "
          f"ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight — check the download")
    print("     used only as a WITHIN-province weight, so the level does not matter and "
          "the\n     shape does.")

    print("\n  per-province Kontur/census ratio (the shape check):")
    census = dict(zip(provs["unit"], provs["pop"]))
    rows = [(provs.loc[provs["unit"] == u, "name"].iloc[0], int(r["size"]),
             r["sum"] / census[u]) for u, r in per.iterrows()]
    for nm, n, r in sorted(rows, key=lambda t: t[2]):
        print(f"      {nm:<24} {n:>7,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
