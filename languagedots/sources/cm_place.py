"""Cameroon placement layer: religiondots' cm_hexes (12 units) with each hex's department added.

    python sources/cm_place.py      -> data/geo/cm/cm_hexes.gpkg  (unit, pop, department)

religiondots' layer (read-only) is COD-AB over Kontur 2023 400 m hexes, `unit` = its geo_id (ten
regions, Mfoundi and Wouri apart). Each hex gets the COD-AB department pcode its centroid falls
in. countries/cm.py weights each language's dots inside a unit by its share in the hex's
department (data/normalized/cm_department.csv, from sources/cm_afro.py). The counts stay the
unit's.

CHECKS: every hex gets a department of its own unit; all 58 departments are hit; the population
is religiondots' to the person.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

ADM2 = HERE.parent / "religiondots" / "data" / "raw" / "cm" / "shp" / "cmr_admin2.shp"
OUT = HERE / "data" / "geo" / "cm" / "cm_hexes.gpkg"
UTM = 32633


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def main():
    hexes = gpd.read_file(RD_GEO / "cm" / "cm_hexes.gpkg", engine="pyogrio")
    say(hexes["unit"].nunique() == 12, f"{len(hexes):,} hexes in 12 units")
    adm2 = gpd.read_file(ADM2, engine="pyogrio")[["adm2_pcode", "geometry"]]
    d = pd.read_csv(RD_GEO / "cm" / "cm_departments.csv", dtype=str)
    adm2["unit"] = adm2["adm2_pcode"].map(dict(zip(d["adm2_pcode"], d["unit"])))
    say(len(adm2) == 58 and adm2["unit"].notna().all(), "58 departments, each with a unit")
    adm2 = adm2.to_crs(hexes.crs)

    pts = hexes[["unit", "geometry"]].copy()
    pts["geometry"] = hexes.to_crs(UTM).centroid.to_crs(hexes.crs)
    j = gpd.sjoin(pts, adm2.rename(columns={"unit": "u2"}), how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    dep = j["adm2_pcode"].reindex(hexes.index)
    bad = dep.isna() | (j["u2"].reindex(hexes.index) != hexes["unit"])
    if bad.any():
        p = pts.loc[bad].to_crs(UTM)
        a = adm2.to_crs(UTM)
        for u, grp in p.groupby("unit"):
            cand = a[a["unit"] == u][["adm2_pcode", "geometry"]]
            near = gpd.sjoin_nearest(grp[["geometry"]], cand, how="left")
            near = near[~near.index.duplicated(keep="first")]
            dep.loc[near.index] = near["adm2_pcode"]
    print(f"  {int(bad.sum()):,} of {len(hexes):,} hexes put on the nearest department of their "
          "own unit (centroid off ADM2 or over a unit line)")
    hexes["department"] = dep.astype(str)
    pu = dict(zip(adm2["adm2_pcode"], adm2["unit"]))
    say(hexes["department"].map(pu).eq(hexes["unit"]).all(), "every hex's department lies in its unit")
    missing = sorted(set(adm2["adm2_pcode"]) - set(hexes["department"]))
    say(not missing, f"all 58 departments hit (missing: {missing})")
    rd = gpd.read_file(RD_GEO / "cm" / "cm_hexes.gpkg", engine="pyogrio", ignore_geometry=True)
    say(abs(hexes["pop"].sum() - rd["pop"].sum()) < 1, f"population {hexes['pop'].sum():,.0f} "
        "= religiondots'")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    hexes[["unit", "pop", "department", "geometry"]].to_file(OUT, driver="GPKG", engine="pyogrio")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
