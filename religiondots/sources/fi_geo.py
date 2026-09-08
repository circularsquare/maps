"""Finland — the placement layer, and the NUTS 3 counting units it rolls up to.

Writes data/geo/fi/fi_lau.gpkg, one row per municipality (kunta):

    lau      the Statistics Finland municipality number, zero-padded to 3
    unit     its NUTS 3 maakunta, NUTS 2021 codes — THE COUNTING UNIT, 19 of them
    name     the municipality's Latin name
    pop      its 2021 population, which is the dot-placement weight

NO DOWNLOAD AT ALL. Both inputs are the GISCO LAU 2021 bundle that has been on disk since
Poland (§9e) — the shapefile for the polygons and the correspondence workbook for the NUTS 3
code and the population. Portugal (§9v) is the other country that paid nothing for its
boundaries; Finland is the second, and for the same reason.

THE COUNTING GEOGRAPHY IS NUTS 3, WHICH IS THE 19 MAAKUNNAT at about 291,000 people each.
That is finer per person than Georgia's 12 regions (§9x, 310,000) and than Greece's NUTS 2
(§9z, 750,000), and it is what the European Social Survey gives Finland in every round it
has sampled since 2010. The municipalities are 310 units and are the placement layer only:
nothing counts religion at that level in Finland, and sources/fi.md §2 is the record of how
thoroughly that was established.

Usage:
    python sources/fi_geo.py
"""

import argparse
import os
import sys

import geopandas as gpd
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAU_SHP = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326",
                       "LAU_RG_01M_2021_4326.shp")
LAU_XLSX = os.path.join(ROOT, "data", "geo", "lau2021",
                        "EU-27-LAU-2021-NUTS-2021.xlsx")
OUT_DIR = os.path.join(ROOT, "data", "geo", "fi")
OUT = os.path.join(OUT_DIR, "fi_lau.gpkg")

N_LAU = 310
N_NUTS3 = 19
POP_2021 = 5_525_292        # the workbook's own Finnish total, asserted so a new vintage says so


def main():
    for p in (LAU_SHP, LAU_XLSX):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — the GISCO LAU 2021 bundle is a shared asset")

    print("reading the LAU shapefile (FI only)…")
    g = gpd.read_file(LAU_SHP, where="CNTR_CODE='FI'")
    g["lau"] = g["LAU_ID"].astype(str).str.strip().str.zfill(3)
    print(f"  {len(g):,} Finnish municipalities, crs={g.crs}")

    print("reading the NUTS correspondence workbook…")
    x = pd.read_excel(LAU_XLSX, sheet_name="FI")
    # The workbook stores the kunta number as an INTEGER, so Alajärvi's 005 arrives as 5 and
    # a naive string cast gives "5". Both sides are zero-padded to three before the join.
    x["lau"] = (x["LAU CODE"].astype(str).str.strip()
                .str.replace(r"\.0$", "", regex=True).str.zfill(3))
    x["unit"] = x["NUTS 3 CODE"].astype(str).str.strip()
    x["pop"] = pd.to_numeric(x["POPULATION"], errors="coerce").fillna(0.0)
    x["name"] = x["LAU NAME LATIN"].astype(str)
    print(f"  {len(x):,} rows, {x['unit'].nunique()} NUTS 3, "
          f"population {x['pop'].sum():,.0f}")

    # spec §8.1, BOTH directions. A one-sided check passes on a file that has silently lost
    # half its rows on the other side.
    only_shp = sorted(set(g["lau"]) - set(x["lau"]))
    only_xls = sorted(set(x["lau"]) - set(g["lau"]))
    print(f"  join: {len(set(g['lau']) & set(x['lau'])):,} matched, "
          f"{len(only_shp)} shapefile-only, {len(only_xls)} workbook-only")
    if only_shp or only_xls:
        sys.exit(f"!! kunta codes do not agree: {only_shp[:8]} / {only_xls[:8]}")

    g = g.merge(x[["lau", "unit", "pop", "name"]], on="lau", how="left")

    if len(g) != N_LAU:
        sys.exit(f"!! expected {N_LAU} municipalities, got {len(g)}")
    if g["unit"].nunique() != N_NUTS3:
        sys.exit(f"!! expected {N_NUTS3} NUTS 3 units, got {g['unit'].nunique()}")
    if abs(g["pop"].sum() - POP_2021) > 0:
        sys.exit(f"!! population {g['pop'].sum():,.0f} is not the expected {POP_2021:,} — "
                 f"the workbook vintage has changed and fi.py's own totals need re-reading")
    if g["lau"].duplicated().any():
        sys.exit("!! duplicate kunta codes")

    empty = g.geometry.isna() | g.geometry.is_empty
    if empty.any():
        print(f"  !! {int(empty.sum())} empty geometries dropped")
        g = g[~empty]
    if (g["pop"] <= 0).any():
        n = int((g["pop"] <= 0).sum())
        print(f"  !! {n} municipalities have no population and will attract no dots")

    # ANTIMERIDIAN AND POLAR SANITY. Finland crosses neither, but Lappi reaches 70°N and a
    # broken projection shows up here rather than three steps later ([[reference_antimeridian]]).
    minx, miny, maxx, maxy = g.total_bounds
    print(f"  bbox {minx:.2f},{miny:.2f} .. {maxx:.2f},{maxy:.2f}")
    if not (18 < minx < 22 and 59 < miny < 61 and 30 < maxx < 32.5 and 69 < maxy < 71):
        sys.exit("!! bbox is not Finland's — check the CRS and the country filter")

    per = g.groupby("unit").agg(kuntia=("lau", "size"), pop=("pop", "sum"))
    print(f"  population per maakunta (mean {per['pop'].mean():,.0f} people per unit):")
    for u in sorted(per.index):
        print(f"    {u}  {per.loc[u, 'pop']:>10,.0f}   {per.loc[u, 'kuntia']:>3} kuntia")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["lau", "unit", "pop", "name", "geometry"]].reset_index(drop=True)
    out.to_file(OUT, driver="GPKG", layer="lau")
    print(f"wrote {OUT}  ({len(out):,} municipalities)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="accepted for symmetry; both inputs are already on disk")
    ap.parse_args()
    main()
