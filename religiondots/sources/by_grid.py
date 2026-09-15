"""Belarus: the placement layer, Kontur 400 m population hexagons keyed to oblast.

Writes data/geo/by/by_hexes.gpkg.

Seven units over 207,600 km² is about 29,700 km² apiece, and an oblast's people are its cities
(Gomel, Mogilev, Vitebsk, Grodno, Brest) plus a thinning countryside and the Polesian marshes, so
an equal share of dots per polygon would put most of each oblast's colour on forest and bog.

The join is spatial, on hex centroids, so a hex on a line belongs wholly to one side. Hexes whose
centroid falls outside every unit are dropped and reported (the extract overruns into Poland,
Lithuania, Latvia, Russia and Ukraine). The unit polygons are `sources/by_geo.py`'s, with Minsk
City already replaced by OSM's 353 km² boundary; on COD's own 87 km² city most of the capital's
hexes would have been keyed to Minsk oblast.

Kontur is modelled (GHSL, building footprints), vintage 2023-11, against Belstat's 1 January 2026
figures, so it is used only as a weight inside each unit. The per-unit ratio is still banded,
because a swap of Minsk city's and Minsk oblast's populations would pass every other check here:
city 1,995,091 against oblast 1,454,737 swapped puts both about 35% off.

Usage:
    python sources/by_grid.py --fetch    one ~8.6 MB gzipped gpkg from Kontur
    python sources/by_grid.py            rebuild from data/raw/by/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
RAW = os.path.join(ROOT, "data", "raw", "by")
GEO = os.path.join(ROOT, "data", "geo", "by")
UNITS = os.path.join(GEO, "by_oblasts.gpkg")
LOOKUP = os.path.join(GEO, "by_lookup.csv")
OUT = os.path.join(GEO, "by_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_BY_20231101.gpkg.gz")
GZ_NAME = "kontur_population_BY_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_BY_20231101.gpkg"

EXPECTED_UNITS = 7
BELSTAT_TOTAL = 9_056_080
KONTUR_TOLERANCE = 0.15          # national level
UNIT_BAND = (0.80, 1.25)         # per unit, Kontur / Belstat; a Minsk swap lands near 0.73 and 1.37

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"


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
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd
    import pandas as pd
    import geo_checks

    if "--fetch" in sys.argv:
        fetch()
    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg}; run with --fetch")
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS}; run sources/by_geo.py first")

    hexes = geo_checks.read_layer(gpkg, "Kontur BY 2023-11")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} units, expected {EXPECTED_UNITS}")

    # Centroids in the CRS the hexes were tiled in, then reproject the points.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=cent,
                           crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"  hexes whose centroid is outside every unit: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%); dropped")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing or (per["sum"] <= 0).any():
        raise SystemExit(f"units with no populated hex: {missing or list(per.index[per['sum'] <= 0])}")

    tot = float(out["pop"].sum())
    ratio = tot / BELSTAT_TOTAL
    print(f"  Kontur {tot:,.0f} against Belstat's {BELSTAT_TOTAL:,}, ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and Belstat disagree by {abs(ratio - 1) * 100:.0f}%")

    lut = pd.read_csv(LOOKUP).set_index("geo_id")
    print("  per-unit Kontur / Belstat (a weight only, but a Minsk swap would show here):")
    band = geo_checks.ratio_band(lut["pop"], per["sum"], *UNIT_BAND, what="unit")
    for u, r in band.iterrows():
        print(f"      {lut.loc[u, 'name']:<14} {int(per.loc[u, 'size']):>7,} hexes   {r.iloc[-1]:.3f}")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
