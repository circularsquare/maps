"""Zambia — the placement layer: Kontur 400 m population hexagons, keyed to constituency.

Writes data/geo/zm/zm_hexes.gpkg.

156 constituencies over 752,000 km2 is about 4,800 km2 a unit, and the spread inside them is
the whole problem: Kabwata or Mandevu in Lusaka is a few dozen km2 of city, while Zambezi West,
Shangombo or Kaputa are tens of thousands of km2 with their people along a river, a road or a
lake shore. [[reference_kontur_resolution_floor]]'s test is whether the grid is finer than the
counting tier, and at 400 m hexes against thousands of km2 it is, by four orders of magnitude.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS -- a hex on a constituency line belongs wholly to
one side, so no hex is split and no population is double-counted. Hexes whose centroid falls
outside every constituency (Kontur's ZM extract overruns into its eight neighbours) are
dropped and reported.

Usage:
    python sources/zm_grid.py --fetch    one gzipped gpkg from Kontur (~18 MB)
    python sources/zm_grid.py            rebuild from data/raw/zm/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "zm")
GEO = os.path.join(ROOT, "data", "geo", "zm")
UNITS = os.path.join(GEO, "zm_constituencies.gpkg")
OUT = os.path.join(GEO, "zm_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_ZM_20231101.gpkg.gz")
GZ_NAME = "kontur_population_ZM_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_ZM_20231101.gpkg"

EXPECTED_UNITS = 156

# Kontur is modelled, not a census, and must not be asserted equal to one (§12, North
# Macedonia). Used ONLY as a within-constituency weight, so assert the RELATIONSHIP and not
# the level. The comparison is against the census's de jure total of 19,693,423 (Summary
# Report Part 2, 2.1); Kontur's 2023-11 vintage is a year later.
CENSUS_2022 = 19_693_423
KONTUR_TOLERANCE = 0.40

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
    if not os.path.exists(gz) or os.path.getsize(gz) < 1_000_000:
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

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS} -- run sources/zm_geo.py first")

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
        raise SystemExit(f"{UNITS} has {len(units)} constituencies, expected {EXPECTED_UNITS}")

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
    print(f"\n  hexes whose centroid is outside every constituency: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's ZM extract overruns into the neighbouring countries; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"constituencies with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"constituencies whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} constituencies has hexes: "
          f"{per['size'].min():,} to {per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_2022
    print(f"\n  Kontur {tot:,.0f} vs the 2022 census (de jure) {CENSUS_2022:,}: ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight -- check the download")

    # The shape check against the religion tables' own constituency totals, when zm.py has
    # run. Printed, not asserted: Kontur is a weight here, and the tails are the thing to read.
    norm = os.path.join(ROOT, "data", "normalized", "zm.csv")
    if os.path.exists(norm):
        import pandas as pd
        n = pd.read_csv(norm, dtype={"geo_id": str})
        cnt = n.groupby("geo_id")["count"].sum()
        nm = dict(zip(units["unit"], units["name"]))
        rows = sorted(((nm[u], int(r["size"]), r["sum"] / cnt[u]) for u, r in per.iterrows()),
                      key=lambda t: t[2])
        print("\n  per-constituency Kontur / religion-table ratio, the ten lowest and highest:")
        for name, k, r in rows[:10] + [("...", 0, float("nan"))] + rows[-10:]:
            print(f"      {name:<20} {k:>7,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
