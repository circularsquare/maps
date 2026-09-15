"""Sweden — the placement layer, and the counting units it rolls up to.

Writes data/geo/se/se_lau.gpkg, one row per municipality (kommun):

    lau      the SCB kommun code, zero-padded to 4 ("0114" Upplands Vasby)
    unit     its NUTS 3 lan, NUTS 2021 codes — THE COUNTING UNIT, 21 of them
    nuts2    the NUTS 2 riksomrade the lan sits in, 8 of them
    name     the kommun's Latin name
    pop      its 2021 population, which is the dot-placement weight

NO DOWNLOAD AT ALL. Both inputs are the GISCO LAU 2021 bundle that has been on disk since
Poland (§9e) — the shapefile for the polygons and the correspondence workbook for the NUTS
code and the population. Finland (§9by) and Portugal (§9v) are the other two countries whose
boundaries cost nothing.

THE COUNTING UNIT IS NUTS 3, THE 21 LAN, at 494,000 people each — finer per person than
Greece's NUTS 2 (§9z, 750,000) and than Georgia's 12 regions (§9x, 310,000) is coarse. Both
halves are counted there: the census counts citizenship at NUTS 3 exactly, and ESS gives
Sweden `region` at NUTS 3 in rounds 5 to 8. `nuts2` is carried too, because ESS drops to
NUTS 2 in rounds 9 and 11 and Catholics, Orthodox and Jews are drawn at their riksomrade's
rate (sources/se.py takes the riksomrade from the NUTS 3 code's first four characters).

The 290 kommuner are the placement layer only. Nothing counts religion at that level in
Sweden except the Church of Sweden's own membership, which is one denomination and is the
witness rather than the source; sources/se.md §4 is the record.

Usage:
    python sources/se_geo.py
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
OUT_DIR = os.path.join(ROOT, "data", "geo", "se")
OUT = os.path.join(OUT_DIR, "se_lau.gpkg")

N_LAU = 290
N_NUTS3 = 21
N_NUTS2 = 8
POP_2021 = 10_379_295       # the workbook's own Swedish total, asserted so a new vintage says so


def main():
    for p in (LAU_SHP, LAU_XLSX):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — the GISCO LAU 2021 bundle is a shared asset")

    print("reading the LAU shapefile (SE only)…")
    g = gpd.read_file(LAU_SHP, where="CNTR_CODE='SE'")
    g["lau"] = g["LAU_ID"].astype(str).str.strip().str.zfill(4)
    print(f"  {len(g):,} Swedish kommuner, crs={g.crs}")

    print("reading the NUTS correspondence workbook…")
    x = pd.read_excel(LAU_XLSX, sheet_name="SE")
    # The workbook stores the kommun code as an INTEGER, so Upplands Vasby's 0114 arrives as
    # 114. Both sides are zero-padded to four before the join — Finland's fi_geo.py hit the
    # same thing at three digits and the trap is the workbook's, not the country's.
    x["lau"] = (x["LAU CODE"].astype(str).str.strip()
                .str.replace(r"\.0$", "", regex=True).str.zfill(4))
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
        sys.exit(f"!! kommun codes do not agree: {only_shp[:8]} / {only_xls[:8]}")

    g = g.merge(x[["lau", "unit", "pop", "name"]], on="lau", how="left")
    g["nuts2"] = g["unit"].str[:4]

    if len(g) != N_LAU:
        sys.exit(f"!! expected {N_LAU} kommuner, got {len(g)}")
    if g["unit"].nunique() != N_NUTS3:
        sys.exit(f"!! expected {N_NUTS3} NUTS 3 units, got {g['unit'].nunique()}")
    if g["nuts2"].nunique() != N_NUTS2:
        sys.exit(f"!! expected {N_NUTS2} NUTS 2 units, got {g['nuts2'].nunique()}")
    if abs(g["pop"].sum() - POP_2021) > 0:
        sys.exit(f"!! population {g['pop'].sum():,.0f} is not the expected {POP_2021:,} — "
                 f"the workbook vintage has changed and se.py's own totals need re-reading")
    if g["lau"].duplicated().any():
        sys.exit("!! duplicate kommun codes")

    empty = g.geometry.isna() | g.geometry.is_empty
    if empty.any():
        print(f"  !! {int(empty.sum())} empty geometries dropped")
        g = g[~empty]
    if (g["pop"] <= 0).any():
        n = int((g["pop"] <= 0).sum())
        print(f"  !! {n} kommuner have no population and will attract no dots")

    # ANTIMERIDIAN AND POLAR SANITY. Sweden crosses neither, but Kiruna reaches 69°N and a
    # broken projection shows up here rather than three steps later ([[reference_antimeridian]]).
    minx, miny, maxx, maxy = g.total_bounds
    print(f"  bbox {minx:.2f},{miny:.2f} .. {maxx:.2f},{maxy:.2f}")
    if not (10 < minx < 12 and 55 < miny < 56 and 23 < maxx < 25 and 68 < maxy < 70):
        sys.exit("!! bbox is not Sweden's — check the CRS and the country filter")

    per = g.groupby("unit").agg(kommuner=("lau", "size"), pop=("pop", "sum"))
    print(f"  population per lan (mean {per['pop'].mean():,.0f} people per unit):")
    for u in sorted(per.index):
        print(f"    {u}  {per.loc[u, 'pop']:>10,.0f}   {per.loc[u, 'kommuner']:>3} kommuner")
    per2 = g.groupby("nuts2")["pop"].sum()
    print(f"  and per riksomrade (mean {per2.mean():,.0f}):")
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
