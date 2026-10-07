"""Palika boundaries for helper1m Nepal: COD-AB admin3 without the park polygons.

COD-AB Nepal (v02, valid_on 2024-03-14) carries 775 admin3 features: the 753
local levels plus 22 polygons for national parks, a wildlife reserve, a hunting
reserve and the Lumbini area, which sit outside every palika. The third digit
of the unit code (character 6 of adm3_pcode) is the unit type; 5 is a park.
The census counts nobody in them, so they are dropped here and show as holes at
the palika level. Provinces and districts use COD's own admin1/admin2 files,
which include the park land.

Writes helper1m/data/nepal/boundaries/adm3.gpkg.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

from pathlib import Path  # noqa: E402

import geopandas as gpd  # noqa: E402

HELPER = Path(__file__).resolve().parents[2]
SRC = HELPER.parent / "data" / "asia1m" / "nepal" / "npl_admin3.shp"
OUT = HELPER / "data" / "nepal" / "boundaries" / "adm3.gpkg"
EXPECTED_TYPES = {"1": 6, "2": 11, "3": 276, "4": 460, "5": 22}


def main():
    g = gpd.read_file(SRC)
    g["unit_type"] = g["adm3_pcode"].str[6]
    got = g["unit_type"].value_counts().to_dict()
    if got != EXPECTED_TYPES:
        raise SystemExit(f"unit-type digits {got}, expected {EXPECTED_TYPES}")
    parks = g[g["unit_type"] == "5"]
    print(f"dropping {len(parks)} park polygons: "
          + ", ".join(sorted(set(parks["adm3_name"]))))
    g = g[g["unit_type"] != "5"]
    g = g[["adm3_pcode", "adm3_name", "adm2_pcode", "adm2_name", "adm1_pcode",
           "adm1_name", "geometry"]].copy()
    assert len(g) == 753 and g["adm3_pcode"].is_unique
    OUT.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(OUT, driver="GPKG", layer="adm3")
    print(f"wrote {OUT} ({len(g)} palikas)")


if __name__ == "__main__":
    main()
