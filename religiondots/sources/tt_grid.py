"""Trinidad and Tobago — the placement layer: Kontur 400 m population hexagons, keyed to
municipality.

Writes data/geo/tt/tt_hexes.gpkg.

**THE PROBLEM HERE IS NOT EMPTY LAND, IT IS THE SIZE RANGE OF THE UNITS.** Measured on the
built layer, Trinidad's 15 municipalities run from the **Borough of Arima at 13.1 km²** to
**Sangre Grande at 931.0 km²**, a **71-fold spread**, and the small ones are the dense ones:
the City of Port of Spain holds 35,914 people on 13.8 km² while Sangre Grande holds 75,605 on
931. Uniform scatter within a polygon would be defensible in the boroughs and badly wrong in
the big rural corporations — Sangre Grande's people are on the coastal road and the Eastern
Main Road, not spread over the Northern Range, and Mayaro/Rio Claro's are along the coast
rather than in the interior forest. Couva/Tabaquite/Talparo has the same shape.

Kontur handles both ends without a special case: an empty hex has no population and takes
no dots, and a dense borough is simply a small polygon full of heavy hexes. `water.py` is
not involved.

**AND TOBAGO IS ITS OWN UNIT, WHICH MAKES THE ISLAND SPLIT FREE.** Tobago is one of the 15,
so no hex crosses between the islands and the sea between them has no weight.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, and that is deliberate — a hex on a boundary
belongs wholly to one side, so no hex is split and no population is double-counted. Hexes
whose centroid falls outside every municipality are dropped and reported; for an island
country that is the coastline, not a border overrun.

Usage:
    python sources/tt_grid.py --fetch    one ~335 KB gzipped gpkg from Kontur
    python sources/tt_grid.py            rebuild from data/raw/tt/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tt")
GEO = os.path.join(ROOT, "data", "geo", "tt")
MUNIS = os.path.join(GEO, "tt_municipalities.gpkg")
OUT = os.path.join(GEO, "tt_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_TT_20231101.gpkg.gz")
GZ_NAME = "kontur_population_TT_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_TT_20231101.gpkg"

EXPECTED_MUNIS = 15

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census and must
# not be asserted equal to it (§12, North Macedonia). Used ONLY as a within-municipality
# weight, so all that matters is that it is not wildly out — assert the RELATIONSHIP.
#
# THE BAND IS MEASURED HERE AND NOT COPIED (§9u's rule), and this is the widest vintage gap
# on the map after Benin's: Kontur is 2023-11 and the census is 2011, TWELVE years apart.
# Trinidad's population barely moved over those years (~1.35M then and now, near-zero
# natural increase with emigration), so the grid should read close to level — but the census
# figure here is the NON-INSTITUTIONAL universe, which is 5,473 people short of the count.
CENSUS_POPULATION = 1_322_546
KONTUR_TOLERANCE = 0.30

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
        raise SystemExit(f"{gpkg} is not a GeoPackage -- starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(MUNIS):
        raise SystemExit(f"missing {MUNIS} -- run sources/tt_geo.py first")

    hexes = gpd.read_file(gpkg)
    # §12: assert the feature count, not the absence of an exception.
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    munis = gpd.read_file(MUNIS)
    if len(munis) != EXPECTED_MUNIS:
        raise SystemExit(f"{MUNIS} has {len(munis)} municipalities, "
                         f"expected {EXPECTED_MUNIS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS -- reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(munis.crs)
    hexes = hexes.to_crs(munis.crs)

    joined = gpd.sjoin(pts, munis[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every municipality: "
          f"{int(outside.sum()):,} ({lost:,.0f} people, "
          f"{100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     an island country, so this is coastline disagreement between Kontur and "
          "COD,\n     not a land border; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=munis.crs)

    # Every municipality must get some hexes, or its dots fall back to an equal share over
    # the whole polygon. The three smallest are the ones to watch: a Kontur hex here
    # measures 0.670 km² (H3 r8), so Arima and Port of Spain hold only 18 apiece. That is
    # thin, and it is also the case where it matters least — a 13 km² borough is close to
    # uniform anyway, which is the opposite of Saint Vincent's problem (§9ac), where the
    # grid was coarser than the counting units and was removed.
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(munis["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"municipalities with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"municipalities whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_MUNIS} municipalities has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    # Kontur is a MODEL, not the census. Assert the relationship, never equality (§12).
    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census {CENSUS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio-1)*100:.0f}%, "
                         "which is too much for a weight -- check the download")
    print("     a 2023 modelled grid against a 2011 census, and Trinidad's population "
          "barely\n     moved between them; used only as a WITHIN-municipality weight.")

    # The per-unit ratio is the one that would bias a share, and it is printed rather than
    # asserted: a unit Kontur models badly gets its dots on a worse surface, it does not get
    # the wrong number of them (§9t).
    cen = pd.read_csv(os.path.join(ROOT, "data", "normalized", "tt.csv"),
                      dtype={"geo_id": str}, keep_default_na=False, na_values=[])
    cen = cen[(cen["geo_level"] == "municipality") & (cen["source_category"] == "Total")]
    cen = dict(zip(cen["geo_id"], pd.to_numeric(cen["count"])))
    print("\n  per-municipality Kontur/census ratio (the shape check):")
    rows = []
    for unit, row in per.iterrows():
        nm = munis.loc[munis["unit"] == unit, "name"].iloc[0]
        rows.append((row["sum"] / cen[unit], nm, int(row["size"])))
    for r, nm, n in sorted(rows):
        print(f"      {nm:<28} {r:5.2f}x   {n:>5,} hexes")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
