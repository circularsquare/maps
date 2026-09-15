"""Latvia — the placement layer, and the counting units it rolls up to.

Writes data/geo/lv/lv_lau.gpkg, one row per LAU:

    lau      the LAU code, zero-padded to 7 ("0010000" Riga), GISCO's own spelling
    unit     its NUTS 3 statistical region, NUTS 2021 codes, THE COUNTING UNIT, 6 of them
    name     the LAU's name as the workbook spells it
    pop      its 2021 population, which is the dot-placement weight

NO DOWNLOAD. Both inputs are the GISCO LAU 2021 bundle on disk since Poland (§9e): the shapefile
for the polygons, and the EU-27 correspondence workbook for the NUTS 3 code and the population.

THE LAUs ARE THE 119 MUNICIPALITIES OF BEFORE 1 JULY 2021 (9 republican cities and 110 novadi).
Latvia's administrative reform merged them into 43 on that date; GISCO's 2021 vintage is dated 1
January and still has the old ones, which is finer for placement and changes nothing about the
counting units.

THE COUNTING UNIT IS NUTS 3, THE 6 STATISTICAL REGIONS, about 315,000 people each, because that is
both where the 2021 census counts citizenship (Eurostat `cens_21ctz_r3`) and where ESS places its
respondents. Latvia's NUTS 1 and NUTS 2 are the whole country. sources/lv.py has the rest.

ONE JOIN TRAP. The shapefile spells the code with seven digits and the workbook stores it as an
integer, so "0010000" meets 10000 and nothing matches until both sides are padded.

Usage:
    python sources/lv_geo.py
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
OUT_DIR = os.path.join(ROOT, "data", "geo", "lv")
OUT = os.path.join(OUT_DIR, "lv_lau.gpkg")

N_LAU = 119
N_NUTS3 = 6
POP_2021 = 1_892_623        # the workbook's own Latvian total, asserted so a new vintage says so

NUTS3 = {
    "LV003": "Kurzeme", "LV005": "Latgale", "LV006": "Rīga", "LV007": "Pierīga",
    "LV008": "Vidzeme", "LV009": "Zemgale",
}


def main():
    for p in (LAU_SHP, LAU_XLSX):
        if not os.path.exists(p):
            sys.exit(f"missing {p}, the GISCO LAU 2021 bundle is a shared asset")

    print("reading the LAU shapefile (LV only)…")
    g = gpd.read_file(LAU_SHP, where="CNTR_CODE='LV'")
    g["lau"] = g["LAU_ID"].astype(str).str.strip().str.zfill(7)
    print(f"  {len(g):,} Latvian LAUs, crs={g.crs}, GISCO POP_2021 "
          f"{g['POP_2021'].astype(float).sum():,.0f}")

    print("reading the NUTS correspondence workbook…")
    x = pd.read_excel(LAU_XLSX, sheet_name="LV")
    x["lau"] = (x["LAU CODE"].astype(str).str.strip()
                .str.replace(r"\.0$", "", regex=True).str.zfill(7))
    x["unit"] = x["NUTS 3 CODE"].astype(str).str.strip()
    x["pop"] = pd.to_numeric(x["POPULATION"], errors="coerce").fillna(0.0)
    x["name"] = x["LAU NAME NATIONAL"].astype(str)
    print(f"  {len(x):,} rows, {x['unit'].nunique()} NUTS 3, population {x['pop'].sum():,.0f}")

    # spec §8.1, both directions.
    only_shp = sorted(set(g["lau"]) - set(x["lau"]))
    only_xls = sorted(set(x["lau"]) - set(g["lau"]))
    print(f"  join: {len(set(g['lau']) & set(x['lau'])):,} matched, "
          f"{len(only_shp)} shapefile-only, {len(only_xls)} workbook-only")
    if only_shp or only_xls:
        sys.exit(f"!! LAU codes do not agree: {only_shp[:8]} / {only_xls[:8]}")

    g = g.merge(x[["lau", "unit", "pop", "name"]], on="lau", how="left")

    # The join above matched on code; check it also matched on name, since a padding slip that
    # happened to line up two different LAUs would pass the set comparison.
    shp_name = g["LAU_NAME"].astype(str).str.strip().str.lower()
    xls_name = g["name"].astype(str).str.strip().str.lower()
    differ = g[shp_name != xls_name]
    print(f"  names differing between shapefile and workbook: {len(differ)}")
    for _, r in differ.head(10).iterrows():
        print(f"    {r['lau']}  {r['LAU_NAME']!r} / {r['name']!r}")
    if len(differ) > 5:
        sys.exit("!! more than 5 LAU names disagree; check the code padding")

    stray = sorted(set(g["unit"]) - set(NUTS3))
    if stray:
        sys.exit(f"!! NUTS 3 codes outside the 6 statistical regions: {stray}")
    if len(g) != N_LAU:
        sys.exit(f"!! expected {N_LAU} LAUs, got {len(g)}")
    if g["unit"].nunique() != N_NUTS3:
        sys.exit(f"!! expected {N_NUTS3} NUTS 3 units, got {g['unit'].nunique()}")
    if POP_2021 is None:
        print(f"  !! POP_2021 is unset; this workbook has {g['pop'].sum():,.0f}")
    elif int(g["pop"].sum()) != POP_2021:
        sys.exit(f"!! population {g['pop'].sum():,.0f} is not the expected {POP_2021:,}")
    if g["lau"].duplicated().any():
        sys.exit("!! duplicate LAU codes")

    empty = g.geometry.isna() | g.geometry.is_empty
    if empty.any():
        print(f"  !! {int(empty.sum())} empty geometries dropped")
        g = g[~empty]
    if (g["pop"] <= 0).any():
        print(f"  !! {int((g['pop'] <= 0).sum())} LAUs have no population")

    # Nida (Rucava) is Latvia's west and south at about 20.97E, 55.67N; Ainaži its north near
    # 58.09N; Zilupe and Pasiene its east near 28.24E.
    minx, miny, maxx, maxy = g.total_bounds
    print(f"  bbox {minx:.2f},{miny:.2f} .. {maxx:.2f},{maxy:.2f}")
    if not (20.7 < minx < 21.2 and 55.5 < miny < 55.8 and 28.0 < maxx < 28.4
            and 57.9 < maxy < 58.2):
        sys.exit("!! bbox is not Latvia's; check the CRS and the country filter")

    per = g.groupby("unit").agg(laus=("lau", "size"), pop=("pop", "sum"))
    print(f"  population per region (mean {per['pop'].mean():,.0f}):")
    for u in sorted(per.index):
        print(f"    {u}  {NUTS3[u]:<10}{per.loc[u, 'pop']:>10,.0f}   {per.loc[u, 'laus']:>3} LAUs")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["lau", "unit", "pop", "name", "geometry"]].reset_index(drop=True)
    out.to_file(OUT, driver="GPKG", layer="lau")
    print(f"wrote {OUT}  ({len(out):,} LAUs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="accepted for symmetry; both inputs are already on disk")
    ap.parse_args()
    main()
