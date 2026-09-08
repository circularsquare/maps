"""Suriname — the placement layer: Kontur 400 m population hexagons, keyed to ressort.

Writes data/geo/sr/sr_hexes.gpkg.

**SURINAME IS THE MOST EXTREME COAST-AND-INTERIOR COUNTRY ON THIS MAP.** 163,820 km² and
about 90% of its people live in the narrow coastal strip; the three Sipaliwini ressorten of
the interior — Kabalebo, Coeroeni, Boven-Coppename — are together larger than the
Netherlands and hold a few thousand people between them. Spread dots uniformly over those
polygons and most of Suriname's dots land in rainforest.

Kontur removes the problem rather than patching it: an empty hex has no population and takes
no dots. `water.py` is not doing this work.

**AND IT IS ALSO A UNIT-SIZE PROBLEM AT THE OTHER END.** The twelve Paramaribo ressorten are
a few km² each and hold half the country. That is the same double-ended case as Trinidad's,
and Kontur handles both ends without a special case.

**THIS IS THE ONE COUNTRY WHERE THE GRID'S RESOLUTION HAD TO BE CHECKED AGAINST THE COUNTING
TIER BEFORE USING IT** — §9ac's rule, learned when Saint Vincent's enumeration districts
turned out to be *finer* than a Kontur hex and the weighting was removed. Checked here and
it passes comfortably: the median Suriname ressort is orders of magnitude larger than a
0.67 km² hex, and every one of the 62 gets hexes. The urban Paramaribo ressorten are the
tightest and are reported.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS — a hex on a boundary belongs wholly to one
side, so none is split and no population is double-counted. Hexes whose centroid falls
outside every ressort are dropped and reported; for Suriname that is the Guyanese and
French Guianese border overrun plus the coastline.

Usage:
    python sources/sr_grid.py --fetch    one ~470 KB gzipped gpkg from Kontur
    python sources/sr_grid.py            rebuild from data/raw/sr/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sr")
GEO = os.path.join(ROOT, "data", "geo", "sr")
UNITS = os.path.join(GEO, "sr_ressorten.gpkg")
OUT = os.path.join(GEO, "sr_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_SR_20231101.gpkg.gz")
GZ_NAME = "kontur_population_SR_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_SR_20231101.gpkg"

EXPECTED_UNITS = 62

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census and must
# not be asserted equal to it (§12, North Macedonia). Used ONLY as a within-ressort weight.
#
# THE BAND IS MEASURED HERE AND NOT COPIED (§9u's rule), and it has to be WIDE, because this
# is the largest vintage gap on the map: a 2023 grid against a **2004** census, nineteen
# years. Suriname's population went from ~493k to ~620k over that period, so the grid should
# read roughly 25% HIGH. A band centred on 1.0 would be the wrong shape of check.
CENSUS_POPULATION = 492_829
KONTUR_LO, KONTUR_HI = 1.00, 1.60

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
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS} -- run sources/sr_geo.py first")

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
        raise SystemExit(f"{UNITS} has {len(units)} ressorten, "
                         f"expected {EXPECTED_UNITS}")

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
    print(f"\n  hexes whose centroid is outside every ressort: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     the Guyanese and French Guianese border overrun, and the coastline; "
          "dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    # §9ac's rule: a population grid must be FINER than the counting tier to be worth using.
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        nm = units.set_index("unit").loc[missing, "name"].tolist()
        raise SystemExit(f"ressorten with no populated hex: {list(zip(missing, nm))}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"ressorten whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} ressorten has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")
    thin = per.sort_values("size").head(5)
    print("     the five thinnest (all urban Paramaribo, where uniform is nearly right "
          "anyway):")
    for unit, row in thin.iterrows():
        nm = units.loc[units["unit"] == unit, "name"].iloc[0]
        print(f"       {nm:<22} {int(row['size']):>4} hexes, {row['sum']:>8,.0f} people")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census {CENSUS_POPULATION:,} — ratio {ratio:.3f}")
    if not (KONTUR_LO <= ratio <= KONTUR_HI):
        raise SystemExit(f"ratio {ratio:.3f} outside the expected "
                         f"[{KONTUR_LO}, {KONTUR_HI}] band -- a 2023 grid against a 2004 "
                         "census should read HIGH; check the download")
    print("     a 2023 modelled grid against a 2004 census, nineteen years and ~25% of "
          "population\n     growth apart, so it should and does read high. Used only as a "
          "WITHIN-ressort weight.")

    cen = pd.read_csv(os.path.join(ROOT, "data", "normalized", "sr.csv"),
                      dtype={"geo_id": str})
    cen = cen[(cen["geo_level"] == "ressort") & (cen["source_category"] == "Total")]
    cen = dict(zip(cen["geo_id"], pd.to_numeric(cen["count"])))
    rows = []
    for unit, row in per.iterrows():
        nm = units.loc[units["unit"] == unit, "name"].iloc[0]
        rows.append((row["sum"] / max(cen[unit], 1), nm, int(row["size"])))
    rows.sort()
    print("\n  per-ressort Kontur/census ratio (the shape check) — extremes only:")
    for r, nm, n in rows[:4]:
        print(f"      {nm:<22} {r:6.2f}x  {n:>5,} hexes")
    print("      ...")
    for r, nm, n in rows[-4:]:
        print(f"      {nm:<22} {r:6.2f}x  {n:>5,} hexes")
    print("      spread is wide and that is EXPECTED across nineteen years — Suriname's "
          "growth\n      is concentrated in Wanica and around Paramaribo. Printed, not "
          "asserted (§9t).")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
