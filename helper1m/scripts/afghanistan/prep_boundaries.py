"""Copy COD-AB v03 (AGCHO/NSIA via OCHA, cod-ab-afg, valid 2025-06-01) province
and district polygons into helper1m/data/afghanistan/boundaries/ with the
columns build_country.py reads. Source zip is religiondots' copy (read only).

v03 rather than asia1m's v02: v03's district codes are the ones COD-PS uses
(v02 has AF2110 where v03 and COD-PS have AF2008: Khulm moved from Balkh to
Samangan, as NSIA also has it).
"""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd

HERE = Path(__file__).parent
HELPER = HERE.parents[1]
REPO = HELPER.parent
SRC = REPO / "religiondots" / "data" / "raw" / "af" / "afg_admin_boundaries.geojson.zip"
OUT = HELPER / "data" / "afghanistan" / "boundaries"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a1 = gpd.read_file(f"zip://{SRC}!afg_admin1.geojson")
    a2 = gpd.read_file(f"zip://{SRC}!afg_admin2.geojson")
    a1 = a1[["adm1_pcode", "adm1_name", "geometry"]].rename(
        columns={"adm1_pcode": "code", "adm1_name": "name"})
    a1["group"] = a1["code"]
    a2 = a2[["adm2_pcode", "adm2_name", "adm1_pcode", "adm1_name", "unittype", "geometry"]].rename(
        columns={"adm2_pcode": "code", "adm2_name": "name", "adm1_pcode": "parent",
                 "adm1_name": "parent_name"})
    a2["group"] = a2["parent"]
    assert len(a1) == 34 and len(a2) == 401
    assert a2["code"].is_unique and set(a2["parent"]) == set(a1["code"])
    a1.to_file(OUT / "adm1.gpkg", driver="GPKG")
    a2.to_file(OUT / "adm2.gpkg", driver="GPKG")
    print(f"wrote {len(a1)} provinces, {len(a2)} districts -> {OUT}")


if __name__ == "__main__":
    main()
