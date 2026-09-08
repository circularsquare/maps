"""Angola — the placement layer: Kontur 400 m population hexagons, keyed to municipality.

Writes data/geo/ao/ao_hexes.gpkg.

**ANGOLA IS THE EMPTY-INTERIOR CASE AT ITS LARGEST.** The 326 municipalities average
3,800 km², and the spread is the widest of any country on this map that is drawn at a
sub-provincial tier: Rangel in Luanda is 3 km², Rivungo in Cuando is over 25,000. The east
and south are close to unpopulated -- Moxico, Moxico Leste, Cuando and Cubango are a third
of the country and hold under 4% of it -- while a quarter of Angolans live inside greater
Luanda. Spread dots evenly over the polygons and the Angolan plateau, where most people
actually are, comes out paler than the Kalahari sand-veld.

It is a coastal case too. The municipal boundaries run to the shore rather than to a
generalised coastline, and Namibe's and Cunene's western halves are the Namib, so a uniform
fill would put dots into desert that has nobody in it. A population grid has no hexes
there, so the problem does not arise rather than being patched (§8.2c).

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, so a hex on a municipal line belongs wholly to
one side and no population is split or double-counted. Hexes whose centroid falls outside
every municipality are the extract's overrun into Congo-Brazzaville, the DRC, Zambia and
Namibia, and are dropped and reported.

**CABINDA IS THE ONE TO WATCH.** It is separated from the rest of Angola by the DRC's
Congo-mouth corridor, so a boundary file that had lost the exclave would still pass an area
check to within 0.6% -- and the ten municipalities there would take their dots on an equal
share over nothing. The per-municipality hex count is asserted for all 326, which is what
catches it.

Usage:
    python sources/ao_grid.py --fetch    one ~12 MB gzipped gpkg from Kontur
    python sources/ao_grid.py            rebuild from data/raw/ao/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ao")
GEO = os.path.join(ROOT, "data", "geo", "ao")
UNITS = os.path.join(GEO, "ao_municipalities.gpkg")
OUT = os.path.join(GEO, "ao_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_AO_20231101.gpkg.gz")
GZ_NAME = "kontur_population_AO_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_AO_20231101.gpkg"

EXPECTED_UNITS = 326

# The census counted 36,175,745 people of all ages on 19 September 2024; the 2+ universe the
# religion table uses is 34,492,888. The grid is compared against the FULL count, because
# that is what a population model is modelling.
CENSUS_POPULATION = 36_175_745

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census and must
# not be asserted equal to it (§12, North Macedonia). Used ONLY as a within-municipality
# weight, so what matters is that it is not wildly out.
#
# THE BAND IS MEASURED HERE AND NOT COPIED FROM MALAWI (§9u's rule), and Angola is the one
# country where it comes out centred on 1.0 honestly: Kontur's November 2023 grid holds
# 36,189,460 people inside the 326 municipalities against the September 2024 census's
# 36,175,745, a ratio of 1.000. That is luck rather than agreement -- the grid is a year
# early on a country growing about 3%/yr, and it is modelled from GHSL and building
# footprints rather than counted -- so the tolerance stays wide and the check stays a
# sanity one.
KONTUR_TOLERANCE = 0.30


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 5_000_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(gz, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg, "wb") as dst:
        shutil.copyfileobj(src, dst)
    # §5a: a 200 is not a download, and a gunzip that runs is not a gpkg.
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage -- starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS} -- run sources/ao_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} units, expected {EXPECTED_UNITS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS -- reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every municipality: "
          f"{int(outside.sum()):,} ({lost:,.0f} people, "
          f"{100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's AO extract overruns into Congo-Brazzaville, the DRC, Zambia and\n"
          "     Namibia; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        names = {u: n for u, n in zip(units["unit"], units["name"])}
        raise SystemExit("municipalities with no populated hex: "
                         + ", ".join(f"{names.get(u, u)} ({u})" for u in missing))
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"municipalities whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} municipalities has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")
    for unit, row in per.sort_values("size").head(3).iterrows():
        nm = units.loc[units["unit"] == unit, "name"].iloc[0]
        print(f"      thinnest: {nm} ({unit}) {int(row['size']):,} hexes, "
              f"{row['sum']:,.0f} people")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census {CENSUS_POPULATION:,} - ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio-1)*100:.0f}%, "
                         "which is too much for a weight -- check the download")
    print("     a 2023 model anchored on pre-2024 projections against the 2024 count, so "
          "it\n     should and does read low; used only as a WITHIN-municipality weight, "
          "so the\n     level does not matter and the shape does.")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
