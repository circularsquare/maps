"""Guatemala — the placement layer: Kontur 400 m population hexagons, keyed to department.

Writes data/geo/gt/gt_hexes.gpkg.

**22 DEPARTMENTS OVER 108,889 km² IS 4,950 km² PER UNIT**, which is worse than Belize's 3,828
and second only to Zimbabwe's among the countries drawn here. Guatemala is also nothing like
uniformly habitable: **Petén alone is 35,903 km², a third of the country's land and 3.6% of
its people**, most of it the Maya Biosphere Reserve and the Petén lowland forest. Spread
Petén's dots evenly over its polygon and the largest single block of colour on the Guatemalan
map lands in uninhabited jungle, while the Guatemala City metropolitan area — a fifth of the
country inside 2,206 km² — is smeared across a department that also contains volcanoes.

Kontur removes that rather than patching it: an empty hex has no population and takes no
dots, so the reserve simply has no weight. `water.py` is not involved.

**AND IT MATTERS MORE HERE THAN THE UNIT COUNT SUGGESTS**, because the two categories this
country draws with their own geography are sorted by altitude and settlement history, not by
department. Every department except Guatemala contains both a highland Maya part and a
lowland or ladino part, and 22 polygons cannot show the difference. Nothing here pretends to.
What the grid buys is that a department's dots land where its people are, which is the
difference between an evangelical share drawn over the Izabal banana coast and one drawn over
the Sierra de las Minas.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, and that is deliberate — a hex on a department
line belongs wholly to one side, so no hex is split and no population is double-counted.
Hexes whose centroid falls outside every department (Kontur's GT extract overruns into
Mexico, Belize and Honduras) are dropped and reported.

Usage:
    python sources/gt_grid.py --fetch    one ~6 MB gzipped gpkg from Kontur
    python sources/gt_grid.py            rebuild from data/raw/gt/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "gt")
GEO = os.path.join(ROOT, "data", "geo", "gt")
DEPARTMENTS = os.path.join(GEO, "gt_departamentos.gpkg")
OUT = os.path.join(GEO, "gt_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_GT_20231101.gpkg.gz")
GZ_NAME = "kontur_population_GT_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_GT_20231101.gpkg"

EXPECTED_DEPARTMENTS = 22

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Used ONLY as a within-department weight,
# so all that matters is that it is not wildly out — assert the RELATIONSHIP.
#
# THE BAND IS MEASURED HERE AND NOT COPIED (§9u's rule). Kontur's vintage is 2023-11 and the
# denominator this country is drawn on is COD-PS's 2024 projection, so the grid should read a
# little LOW, by roughly a year of growth (~1.4%). The band is wide because Guatemala's own
# 2018 census came in about 15% under the projection series that preceded it, and which of
# those two a model like Kontur is anchored to is not knowable in advance.
CODPS_POPULATION = 17_843_132
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
    if not os.path.exists(DEPARTMENTS):
        raise SystemExit(f"missing {DEPARTMENTS} — run sources/gt_geo.py first")

    hexes = gpd.read_file(gpkg)
    # §12: assert the feature count, not the absence of an exception.
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    deps = gpd.read_file(DEPARTMENTS)
    if len(deps) != EXPECTED_DEPARTMENTS:
        raise SystemExit(f"{DEPARTMENTS} has {len(deps)} departments, "
                         f"expected {EXPECTED_DEPARTMENTS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS — reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(deps.crs)
    hexes = hexes.to_crs(deps.crs)

    joined = gpd.sjoin(pts, deps[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every department: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's GT extract overruns into Mexico, Belize and Honduras; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=deps.crs)

    # Every department must get some hexes, or its dots fall back to an equal share over the
    # whole polygon and the department silently un-weights itself.
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(deps["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"departments with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"departments whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_DEPARTMENTS} departments has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CODPS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs COD-PS {CODPS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and COD-PS disagree by {abs(ratio - 1) * 100:.0f}%, which "
                         "is too much for a weight — check the download")
    print("     used only as a WITHIN-department weight, so the level does not matter and "
          "the\n     shape does.")

    # The per-department ratio is the one that would bias a share, and it is printed rather
    # than asserted: a department Kontur models badly gets its dots on a worse surface, it
    # does not get the wrong number of them (§9t).
    print("\n  per-department Kontur/COD-PS ratio (the shape check):")
    cod = pd.read_csv(os.path.join(RAW, "gtm_admpop_adm1_2024.csv"), encoding="utf-8-sig")
    cod = dict(zip(cod["ADM1_PCODE"].astype(str).str.strip(), cod["T_TL"]))
    rows = []
    for unit, row in per.iterrows():
        nm = deps.loc[deps["unit"] == unit, "name"].iloc[0]
        rows.append((nm, unit, int(row["size"]), row["sum"] / cod[unit]))
    for nm, unit, n, r in sorted(rows, key=lambda t: t[3]):
        print(f"      {nm:<16} {n:>7,} hexes   {r:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
