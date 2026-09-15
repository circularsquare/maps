"""Denmark — the placement layer, and the counting units it rolls up to.

Writes data/geo/dk/dk_lau.gpkg, one row per municipality (kommune):

    lau      the kommune code, zero-padded to 3 ("101" Copenhagen)
    unit     its NUTS 3 landsdel, NUTS 2021 codes, THE COUNTING UNIT, 11 of them
    nuts2    the NUTS 2 region the landsdel sits in, 5 of them
    name     the kommune's name as the workbook spells it
    pop      its 2021 population, which is the dot-placement weight

NO DOWNLOAD. Both inputs are the GISCO LAU 2021 bundle on disk since Poland (§9e): the shapefile
for the polygons, and the EU-27 correspondence workbook for the NUTS 3 code and the population.
Denmark is a member state, so the workbook has its sheet (Norway's did not, no_geo.py).

THE COUNTING UNIT IS NUTS 3, THE 11 LANDSDELE, about 530,000 people each, because that is where
the 2021 census counts citizenship (Eurostat `cens_21ctz_r3`) and so where the foreign half is
measured. ESS gives Denmark `region` at NUTS 2 only, the 5 regioner, and the landsdele nest in
them exactly (a landsdel's NUTS 2 is its code's first four characters), so the citizen half of
each landsdel takes its region's composition with no blending. sources/dk.py has that half.

Usage:
    python sources/dk_geo.py
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
OUT_DIR = os.path.join(ROOT, "data", "geo", "dk")
OUT = os.path.join(OUT_DIR, "dk_lau.gpkg")

# 98 kommuner and Christianso, which is not a kommune (the state runs it) and is still its own LAU.
N_LAU = 99
N_NUTS3 = 11
N_NUTS2 = 5
POP_2021 = 5_840_045        # the workbook's own Danish total, asserted so a new vintage says so

NUTS3 = {
    "DK011": "Byen København", "DK012": "Københavns omegn", "DK013": "Nordsjælland",
    "DK014": "Bornholm", "DK021": "Østsjælland", "DK022": "Vest- og Sydsjælland",
    "DK031": "Fyn", "DK032": "Sydjylland", "DK041": "Vestjylland", "DK042": "Østjylland",
    "DK050": "Nordjylland",
}


def main():
    for p in (LAU_SHP, LAU_XLSX):
        if not os.path.exists(p):
            sys.exit(f"missing {p}, the GISCO LAU 2021 bundle is a shared asset")

    print("reading the LAU shapefile (DK only)…")
    g = gpd.read_file(LAU_SHP, where="CNTR_CODE='DK'")
    g["lau"] = g["LAU_ID"].astype(str).str.strip().str.zfill(3)
    print(f"  {len(g):,} Danish kommuner, crs={g.crs}, GISCO POP_2021 "
          f"{g['POP_2021'].astype(float).sum():,.0f}")

    print("reading the NUTS correspondence workbook…")
    x = pd.read_excel(LAU_XLSX, sheet_name="DK")
    # The workbook stores codes as integers (se_geo.py's trap), so pad both sides.
    x["lau"] = (x["LAU CODE"].astype(str).str.strip()
                .str.replace(r"\.0$", "", regex=True).str.zfill(3))
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
        sys.exit(f"!! kommune codes do not agree: {only_shp[:8]} / {only_xls[:8]}")

    g = g.merge(x[["lau", "unit", "pop", "name"]], on="lau", how="left")
    g["nuts2"] = g["unit"].str[:4]

    stray = sorted(set(g["unit"]) - set(NUTS3))
    if stray:
        sys.exit(f"!! NUTS 3 codes outside the 11 landsdele: {stray}")
    if len(g) != N_LAU:
        sys.exit(f"!! expected {N_LAU} kommuner, got {len(g)}")
    if g["unit"].nunique() != N_NUTS3:
        sys.exit(f"!! expected {N_NUTS3} NUTS 3 units, got {g['unit'].nunique()}")
    if g["nuts2"].nunique() != N_NUTS2:
        sys.exit(f"!! expected {N_NUTS2} NUTS 2 units, got {g['nuts2'].nunique()}")
    if POP_2021 is None:
        print(f"  !! POP_2021 is unset; this workbook has {g['pop'].sum():,.0f}")
    elif int(g["pop"].sum()) != POP_2021:
        sys.exit(f"!! population {g['pop'].sum():,.0f} is not the expected {POP_2021:,}")
    if g["lau"].duplicated().any():
        sys.exit("!! duplicate kommune codes")

    empty = g.geometry.isna() | g.geometry.is_empty
    if empty.any():
        print(f"  !! {int(empty.sum())} empty geometries dropped")
        g = g[~empty]
    if (g["pop"] <= 0).any():
        print(f"  !! {int((g['pop'] <= 0).sum())} kommuner have no population")

    # Blavandshuk is Denmark's west point at 8.07E, Gedser its south at 54.56N, Skagen its north
    # at 57.75N, and Bornholm (Christianso is not a kommune) its east near 15.2E.
    minx, miny, maxx, maxy = g.total_bounds
    print(f"  bbox {minx:.2f},{miny:.2f} .. {maxx:.2f},{maxy:.2f}")
    if not (7.5 < minx < 8.5 and 54.4 < miny < 54.8 and 14.5 < maxx < 15.5
            and 57.6 < maxy < 57.9):
        sys.exit("!! bbox is not Denmark's; check the CRS and the country filter")

    per = g.groupby("unit").agg(kommuner=("lau", "size"), pop=("pop", "sum"))
    print(f"  population per landsdel (mean {per['pop'].mean():,.0f}):")
    for u in sorted(per.index):
        print(f"    {u}  {NUTS3[u]:<22}{per.loc[u, 'pop']:>10,.0f}   "
              f"{per.loc[u, 'kommuner']:>3} kommuner")
    per2 = g.groupby("nuts2")["pop"].sum()
    print(f"  and per region (mean {per2.mean():,.0f}):")
    for u in sorted(per2.index):
        print(f"    {u}  {per2.loc[u]:>10,.0f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["lau", "unit", "nuts2", "pop", "name", "geometry"]].reset_index(drop=True)
    out.to_file(OUT, driver="GPKG", layer="lau")
    print(f"wrote {OUT}  ({len(out):,} kommuner)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="accepted for symmetry; both inputs are already on disk")
    ap.parse_args()
    main()
