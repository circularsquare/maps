"""Timor-Leste — the placement layer: Kontur 400 m population hexagons, keyed to municipality.

Writes data/geo/tl/tl_hexes.gpkg.

**THE COUNTRY IS MOUNTAIN AND THE PEOPLE ARE NOT EVENLY ON IT.** The fourteen units average
1,068 km² and every one of them except Dili and Atauro runs from a coastal plain up to a
central ridge; Ramelau is 2,986 m and the interior of Lautém, Manatuto and Viqueque is close
to empty. Spread dots evenly inside a municipality and the uplands come out as populated as
the valleys they drain.

**THE TWO SPLIT UNITS ARE WHAT MAKES THE GRID NECESSARY RATHER THAN NICE.** Dili is a
227 km² polygon holding 324,738 people and Atauro is a 140 km² island holding 10,295, so the
pair differ by a factor of 22 in density and share a p-code prefix; a hex that landed on the
wrong side of that line would move a Protestant-majority island's dots into the capital. The
join is spatial, on hex CENTROIDS, so a hex on the strait belongs wholly to one side.

Kontur's TL extract also overruns into Indonesian West Timor, which is a land border rather
than a sea one, and those hexes are dropped by the same test.

Usage:
    python sources/tl_grid.py --fetch    one gzipped gpkg from Kontur
    python sources/tl_grid.py            rebuild from data/raw/tl/
"""

import gzip
import os
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")     # [[reference_scipy_eats_all_cores]]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tl")
GEO = os.path.join(ROOT, "data", "geo", "tl")
UNITS = os.path.join(GEO, "tl_municipalities.gpkg")
OUT = os.path.join(GEO, "tl_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_TL_20231101.gpkg.gz")
GZ_NAME = "kontur_population_TL_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_TL_20231101.gpkg"

EXPECTED_UNITS = 14
CENSUS_POPULATION = 1_341_737     # table 4.03, everybody, which is what a grid models

# Kontur is modelled from GHSL, HRSL and building footprints and is not the census, so this
# is a sanity band and not an equality (§12, North Macedonia). Its grid is November 2023
# against a September 2022 count of a country growing ~2%/yr, so it should read a little
# high; it is used only as a WITHIN-municipality weight, where the level cancels.
KONTUR_TOLERANCE = 0.30


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 1_000_000:
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
        raise SystemExit(f"missing {UNITS} -- run sources/tl_geo.py first")

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
    hx = hexes.to_crs(units.crs)

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every municipality: "
          f"{int(outside.sum()):,} ({lost:,.0f} people, "
          f"{100.0 * lost / pts[popcol].sum():.3f}%) -- the extract's overrun into "
          "Indonesian West Timor; dropped")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hx.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        names = dict(zip(units["unit"], units["name"]))
        raise SystemExit("municipalities with no populated hex: "
                         + ", ".join(f"{names.get(u, u)} ({u})" for u in missing))
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"municipalities whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} municipalities has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")
    names = dict(zip(units["unit"], units["name"]))
    for unit, row in per.sort_values("size").head(3).iterrows():
        print(f"      thinnest: {names[unit]} ({unit}) {int(row['size']):,} hexes, "
              f"{row['sum']:,.0f} people")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census {CENSUS_POPULATION:,} - ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight -- check the download")

    # THE SHAPE IS WHAT IS USED, so report it: how far each municipality's Kontur share is
    # from its census share. This is not asserted, because Kontur is a 2023 model and the
    # census is a 2022 count and the difference between them is real information.
    import pandas as pd
    cen = pd.read_csv(os.path.join(ROOT, "data", "normalized", "tl.csv"),
                      dtype={"geo_id": str}).groupby("geo_id")["count"].sum()
    print("\n  unit          Kontur share   census share")
    for unit in units["unit"]:
        ks = per["sum"][unit] / tot
        cs = cen.get(unit, 0) / cen.sum()
        print(f"  {unit:<6} {names[unit]:<10} {100 * ks:>9.2f}%    {100 * cs:>9.2f}%")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
