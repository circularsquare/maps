"""Zimbabwe placement layer: religiondots' zw_hexes (10 provinces) with each hex's district.

    python sources/zw_place.py      -> data/geo/zw/zw_hexes.gpkg  (unit, pop, district)

religiondots' layer (read-only) is Kontur 2023 400 m hexes on COD-AB provinces, `unit` = the
adm1 pcode (ZW10..ZW19), Lake Kariba removed. Each hex gets the placement district its centroid
falls in: COD-AB ADM2 with urban councils folded into the rural district around them
(sources/zw_afro.py `adm2_key`, 65 districts). countries/zw.py weights each language's dots
inside a province by its share in the hex's district (data/normalized/zw_district.csv). The
counts stay the census's.

CHECKS: every hex gets a district of its own province; all 65 districts are hit; the population
is religiondots' to the person.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sources"))
from rdlink import RD_GEO  # noqa: E402
from zw_afro import adm2, say  # noqa: E402

OUT = HERE / "data" / "geo" / "zw" / "zw_hexes.gpkg"


def main():
    hexes = gpd.read_file(RD_GEO / "zw" / "zw_hexes.gpkg", engine="pyogrio")
    say(hexes["unit"].nunique() == 10, f"{len(hexes):,} hexes in 10 provinces")
    g = adm2()[["key", "adm1_pcode", "geometry"]].to_crs(hexes.crs)
    pts = hexes[["unit", "geometry"]].copy()
    pts["geometry"] = hexes.to_crs(32735).centroid.to_crs(hexes.crs)
    j = gpd.sjoin(pts, g, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    dist = j["key"].reindex(hexes.index)
    bad = dist.isna() | (j["adm1_pcode"].reindex(hexes.index) != hexes["unit"])
    if bad.any():
        p = pts.loc[bad].to_crs(32735)
        a = g.to_crs(32735)
        for u, grp in p.groupby("unit"):
            cand = a[a["adm1_pcode"] == u][["key", "geometry"]]
            near = gpd.sjoin_nearest(grp[["geometry"]], cand, how="left")
            near = near[~near.index.duplicated(keep="first")]
            dist.loc[near.index] = near["key"]
    print(f"  {int(bad.sum()):,} of {len(hexes):,} hexes put on the nearest district of their own "
          "province (centroid off ADM2 or over a province line)")
    hexes["district"] = dist.astype(str)
    say((hexes["district"].str[:4] == hexes["unit"]).all(), "every hex's district is in its province")
    missing = sorted(set(g["key"]) - set(hexes["district"]))
    say(not missing, f"all {g['key'].nunique()} districts hit (missing: {missing})")
    rd = gpd.read_file(RD_GEO / "zw" / "zw_hexes.gpkg", engine="pyogrio", ignore_geometry=True)
    say(abs(hexes["pop"].sum() - rd["pop"].sum()) < 1, f"population {hexes['pop'].sum():,.0f} "
        "= religiondots'")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    hexes[["unit", "pop", "district", "geometry"]].to_file(OUT, driver="GPKG", engine="pyogrio")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
