"""Uzbekistan boundaries for helper1m.

Reads OCHA COD-AB uzb_admbnda_adm2_2018b (199 districts and cities) and the
names fetch.py wrote to data/uzbekistan/units.csv. Writes
  data/uzbekistan/boundaries/adm2.gpkg   code, name, name_cn, parent, group
  data/uzbekistan/boundaries/adm1.gpkg   the 14 regions, dissolved from adm2 so
                                         the two levels nest exactly
name is the Statistics Committee's English name, name_cn (shown as the
tooltip's second line) its Uzbek Latin name. Run fetch.py first.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

HELPER = Path(__file__).resolve().parents[2]
SRC = HELPER.parent / "data" / "asia1m" / "uzbekistan" / "uzb_admbnda_adm2_2018b.shp"
UNITS = HELPER / "data" / "uzbekistan" / "units.csv"
OUT = HELPER / "data" / "uzbekistan" / "boundaries"


# COD 2018b has the two Kagan polygons the wrong way round: "Kagan district"
# (UZ06219) is a 1.9 km2 speck with nobody in it (Kontur 2023: 2 people) and
# "Kagan city" (UZ06403) is 470 km2 and holds the town and its farmland. OSM's
# Kogon shahri lies at 39.70-39.75 N and Kogon tumani spans 39.57-39.92 N.
SWAP_CODES = {"UZ06219": "UZ06403", "UZ06403": "UZ06219"}


def main():
    g = gpd.read_file(SRC).to_crs("EPSG:4326")
    g["ADM2_PCODE"] = g["ADM2_PCODE"].map(lambda c: SWAP_CODES.get(c, c))
    g["geometry"] = shapely.make_valid(g.geometry.values)
    units = pd.read_csv(UNITS, dtype=str).set_index("code")

    a2 = gpd.GeoDataFrame({
        "code": g["ADM2_PCODE"],
        "name": g["ADM2_PCODE"].map(units["name"]),
        "name_cn": g["ADM2_PCODE"].map(units["name_uz"]),
        "parent": g["ADM1_PCODE"],
        "group": g["ADM1_PCODE"],
    }, geometry=g.geometry, crs=g.crs)
    assert a2["name"].notna().all(), a2.loc[a2["name"].isna(), "code"].tolist()

    a1 = a2.dissolve(by="group", as_index=False)[["group", "geometry"]]
    a1["code"] = a1["group"]
    a1["name"] = a1["code"].map(units["name"])
    a1["name_cn"] = a1["code"].map(units["name_uz"])
    assert a1["name"].notna().all()
    a1 = a1[["code", "name", "name_cn", "group", "geometry"]]

    OUT.mkdir(parents=True, exist_ok=True)
    a2.to_file(OUT / "adm2.gpkg", driver="GPKG")
    a1.to_file(OUT / "adm1.gpkg", driver="GPKG")
    print(f"adm1 {len(a1)}, adm2 {len(a2)} -> {OUT}")


if __name__ == "__main__":
    main()
