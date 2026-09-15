"""Belgium — the placement layer, and the NUTS 2 counting units it rolls up to.

Writes data/geo/be/be_lau.gpkg, one row per commune / gemeente:

    lau      the INS/NIS code, five digits as a string
    unit     its NUTS 2 province, NUTS 2021 codes — THE COUNTING UNIT, 11 of them
    name     the commune's name in its own official language
    pop      its 2021 population, which is the dot-placement weight

NO DOWNLOAD AT ALL. Both inputs are the GISCO LAU 2021 bundle that has been on disk since
Poland (§9e) — the shapefile for the polygons and the correspondence workbook for the NUTS
code and the population. Finland (§9by) and Portugal (§9v) paid nothing for their boundaries
for the same reason; Belgium is the third.

THE COUNTING GEOGRAPHY IS NUTS 2, which for Belgium is the ten provinces plus the
Brussels-Capital Region, about 1.05 million people each. That is what the European Social
Survey gives Belgium, unchanged in every round it has sampled since 2010, and nothing
Belgian measures religion below it: `sources/be.md` §1 is the record of how that was
established. The 581 communes are the placement layer only.

NUTS 3, the 44 arrondissements, sits between the two and is NOT used. It is available for
the foreign half (Eurostat's census citizenship table publishes it), and using it there
while the survey half stays at NUTS 2 would put the sharper geography on one half of the
country and not the other. Greece made the same call for the same reason.

Usage:
    python sources/be_geo.py
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
OUT_DIR = os.path.join(ROOT, "data", "geo", "be")
OUT = os.path.join(OUT_DIR, "be_lau.gpkg")

N_LAU = 581
N_NUTS2 = 11
POP_2021 = 11_566_041       # the workbook's own Belgian total, asserted so a new vintage says so

# The eleven counting units. Names are each province's own, which in a bilingual country is
# the only choice that is not a decision about the country: Dutch in Flanders, French in
# Wallonia, both in Brussels. Nothing here needs romanising.
NUTS2 = {
    "BE10": "Brussels / Bruxelles",
    "BE21": "Antwerpen",
    "BE22": "Limburg",
    "BE23": "Oost-Vlaanderen",
    "BE24": "Vlaams-Brabant",
    "BE25": "West-Vlaanderen",
    "BE31": "Brabant wallon",
    "BE32": "Hainaut",
    "BE33": "Liege",
    "BE34": "Luxembourg",
    "BE35": "Namur",
}


def main():
    for p in (LAU_SHP, LAU_XLSX):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — the GISCO LAU 2021 bundle is a shared asset")

    print("reading the LAU shapefile (BE only)…")
    g = gpd.read_file(LAU_SHP, where="CNTR_CODE='BE'")
    g["lau"] = g["LAU_ID"].astype(str).str.strip().str.zfill(5)
    print(f"  {len(g):,} Belgian communes, crs={g.crs}")

    print("reading the NUTS correspondence workbook…")
    x = pd.read_excel(LAU_XLSX, sheet_name="BE")
    # The INS code is stored as an INTEGER, so Anderlecht's 21001 survives a naive cast and
    # a four-digit code would not. Both sides are zero-padded to five before the join, the
    # same trap fi_geo.py records for the three-digit Finnish kunta number.
    x["lau"] = (x["LAU CODE"].astype(str).str.strip()
                .str.replace(r"\.0$", "", regex=True).str.zfill(5))
    x["unit"] = x["NUTS 3 CODE"].astype(str).str.strip().str[:4]
    x["pop"] = pd.to_numeric(x["POPULATION"], errors="coerce").fillna(0.0)
    # LAU NAME NATIONAL is the commune's name in its own language; LAU NAME LATIN repeats it.
    x["name"] = x["LAU NAME NATIONAL"].astype(str)
    print(f"  {len(x):,} rows, {x['unit'].nunique()} NUTS 2, "
          f"population {x['pop'].sum():,.0f}")

    # spec §8.1, BOTH directions. A one-sided check passes on a file that has silently lost
    # half its rows on the other side.
    only_shp = sorted(set(g["lau"]) - set(x["lau"]))
    only_xls = sorted(set(x["lau"]) - set(g["lau"]))
    print(f"  join: {len(set(g['lau']) & set(x['lau'])):,} matched, "
          f"{len(only_shp)} shapefile-only, {len(only_xls)} workbook-only")
    if only_shp or only_xls:
        sys.exit(f"!! INS codes do not agree: {only_shp[:8]} / {only_xls[:8]}")

    g = g.merge(x[["lau", "unit", "pop", "name"]], on="lau", how="left")

    if len(g) != N_LAU:
        sys.exit(f"!! expected {N_LAU} communes, got {len(g)}")
    if sorted(g["unit"].unique()) != sorted(NUTS2):
        sys.exit(f"!! NUTS 2 codes are not the eleven expected: "
                 f"{sorted(g['unit'].unique())}")
    if abs(g["pop"].sum() - POP_2021) > 0:
        sys.exit(f"!! population {g['pop'].sum():,.0f} is not the expected {POP_2021:,} — "
                 f"the workbook vintage has changed and be.py's own totals need re-reading")
    if g["lau"].duplicated().any():
        sys.exit("!! duplicate INS codes")

    empty = g.geometry.isna() | g.geometry.is_empty
    if empty.any():
        print(f"  !! {int(empty.sum())} empty geometries dropped")
        g = g[~empty]
    if (g["pop"] <= 0).any():
        n = int((g["pop"] <= 0).sum())
        print(f"  !! {n} communes have no population and will attract no dots")

    # Belgium crosses no antimeridian and no pole, but a broken CRS or a country filter that
    # picked up a neighbour shows up here rather than three steps later
    # ([[reference_antimeridian]]).
    minx, miny, maxx, maxy = g.total_bounds
    print(f"  bbox {minx:.2f},{miny:.2f} .. {maxx:.2f},{maxy:.2f}")
    if not (2.4 < minx < 2.8 and 49.4 < miny < 49.8 and 6.2 < maxx < 6.6
            and 51.3 < maxy < 51.7):
        sys.exit("!! bbox is not Belgium's — check the CRS and the country filter")

    per = g.groupby("unit").agg(communes=("lau", "size"), pop=("pop", "sum"))
    print(f"  population per province (mean {per['pop'].mean():,.0f} people per unit):")
    for u in sorted(per.index):
        print(f"    {u} {NUTS2[u]:<22} {per.loc[u, 'pop']:>10,.0f}   "
              f"{per.loc[u, 'communes']:>3} communes")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["lau", "unit", "pop", "name", "geometry"]].reset_index(drop=True)
    out.to_file(OUT, driver="GPKG", layer="lau")
    print(f"wrote {OUT}  ({len(out):,} communes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="accepted for symmetry; both inputs are already on disk")
    ap.parse_args()
    main()
