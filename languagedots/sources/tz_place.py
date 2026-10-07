"""Tanzania placement layer: religiondots' tz_hexes (30 units) with each hex's district added.

    python sources/tz_place.py      -> data/geo/tz/tz_hexes.gpkg  (unit, pop, district)

religiondots' layer (read-only) is COD-AB 2018 ADM1 over Kontur 2023 400 m hexes, `unit` = its
geo_id (TZ01..TZ55, Songwe inside TZ12). Each hex gets the COD-AB 2018 ADM2 pcode (`district`)
its centroid falls in. countries/tz.py weights each language's dots inside a unit by its share
in the hex's district (data/normalized/tz_district.csv, from sources/tz_afro.py). The counts
stay the unit's.

CHECKS: every hex gets a district of its own unit; all 170 districts are hit; the population
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

ADM2 = HERE.parent / "religiondots" / "data" / "raw" / "tz" / "adm2" / "tza_admbnda_adm2_20181019.shp"
OUT = HERE / "data" / "geo" / "tz" / "tz_hexes.gpkg"


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def main():
    hexes = gpd.read_file(RD_GEO / "tz" / "tz_hexes.gpkg", engine="pyogrio")
    say(hexes["unit"].nunique() == 30, f"{len(hexes):,} hexes in 30 units")
    adm2 = gpd.read_file(ADM2, engine="pyogrio")[["ADM2_PCODE", "ADM2_EN", "geometry"]]
    say(len(adm2) == 170, "170 districts")
    dl = pd.read_csv(RD_GEO / "tz" / "tz_districts.csv", dtype=str)
    adm2["unit"] = adm2["ADM2_EN"].map(dict(zip(dl["district"], dl["unit"])))
    say(adm2["unit"].notna().all(), "every district has a unit")
    adm2 = adm2.to_crs(hexes.crs)

    pts = hexes[["unit", "geometry"]].copy()
    pts["geometry"] = hexes.to_crs(32737).centroid.to_crs(hexes.crs)
    j = gpd.sjoin(pts, adm2[["ADM2_PCODE", "unit", "geometry"]].rename(columns={"unit": "u2"}),
                  how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    dist = j["ADM2_PCODE"].reindex(hexes.index)
    bad = dist.isna() | (j["u2"].reindex(hexes.index) != hexes["unit"])
    # centroids off the ADM2 polygons (coast, lake, border) or across a unit line: the nearest
    # district of the hex's own unit
    if bad.any():
        p = pts.loc[bad].to_crs(32737)
        a = adm2.to_crs(32737)
        for u, grp in p.groupby("unit"):
            cand = a[a["unit"] == u][["ADM2_PCODE", "geometry"]]
            near = gpd.sjoin_nearest(grp[["geometry"]], cand, how="left")
            near = near[~near.index.duplicated(keep="first")]
            dist.loc[near.index] = near["ADM2_PCODE"]
    print(f"  {int(bad.sum()):,} of {len(hexes):,} hexes put on the nearest district of their own "
          "unit (centroid off ADM2 or over a unit line)")
    hexes["district"] = dist.astype(str)
    pu = dict(zip(adm2["ADM2_PCODE"], adm2["unit"]))
    say(hexes["district"].map(pu).eq(hexes["unit"]).all(), "every hex's district lies in its unit")
    missing = sorted(set(adm2["ADM2_PCODE"]) - set(hexes["district"]))
    say(not missing, f"all 170 districts hit (missing: {missing})")
    rd = gpd.read_file(RD_GEO / "tz" / "tz_hexes.gpkg", engine="pyogrio", ignore_geometry=True)
    say(abs(hexes["pop"].sum() - rd["pop"].sum()) < 1, f"population {hexes['pop'].sum():,.0f} "
        "= religiondots'")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    hexes[["unit", "pop", "district", "geometry"]].to_file(OUT, driver="GPKG", engine="pyogrio")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
