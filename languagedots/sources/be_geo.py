"""Belgium: the placement layer, data/geo/be/be_place.gpkg.

    python sources/be_geo.py

religiondots' data/geo/be/be_lau.gpkg (read-only): GISCO LAU 2021, Belgium's 581 communes with
their 2021 population, keyed there to NUTS 2. Re-keyed here to NUTS 3 (the 44 arrondissements,
Verviers in two) from the same GISCO correspondence workbook, which is languagedots' counting
unit. The communes only place dots: sources/be_census.py writes each commune's own figure per
language (data/normalized/be_communes.csv), and countries/be.py weights by it.
"""
import os
import sys

import geopandas as gpd
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RD = os.path.join(os.path.dirname(ROOT), "religiondots")
RD_LAU = os.path.join(RD, "data", "geo", "be", "be_lau.gpkg")
LAU_XLSX = os.path.join(RD, "data", "geo", "lau2021", "EU-27-LAU-2021-NUTS-2021.xlsx")
OUT = os.path.join(ROOT, "data", "geo", "be", "be_place.gpkg")


def main():
    g = gpd.read_file(RD_LAU)
    x = pd.read_excel(LAU_XLSX, sheet_name="BE")
    x["lau"] = (x["LAU CODE"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
                .str.zfill(5))
    nuts3 = dict(zip(x["lau"], x["NUTS 3 CODE"].astype(str).str.strip()))
    only_g = sorted(set(g["lau"]) - set(nuts3))
    only_x = sorted(set(nuts3) - set(g["lau"]))
    print(f"religiondots be_lau.gpkg {len(g)} communes; workbook {len(nuts3)}; "
          f"layer-only {only_g[:5]}, workbook-only {only_x[:5]}")
    if only_g or only_x or len(g) != 581:
        sys.exit("!! the two do not agree commune for commune")
    g["nuts2"] = g["unit"]
    g["unit"] = g["lau"].map(nuts3)
    if (g["unit"].str[:4] != g["nuts2"]).any():
        sys.exit("!! a commune's NUTS 3 does not sit in its NUTS 2")
    if g["unit"].nunique() != 44:
        sys.exit(f"!! expected 44 NUTS 3, got {g['unit'].nunique()}")
    print(f"units: {g['unit'].nunique()} arrondissements, population {g['pop'].sum():,.0f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp.gpkg"
    g[["lau", "unit", "nuts2", "name", "pop", "geometry"]].to_file(tmp, driver="GPKG",
                                                                   layer="place")
    os.replace(tmp, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
