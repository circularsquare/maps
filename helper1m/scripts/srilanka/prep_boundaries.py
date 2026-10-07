"""Boundaries for helper1m Sri Lanka, from DCS's own 2024 census geography.

Source: the DS-division layer the DCS cartography unit publishes on ArcGIS Online
(DSD_POP_DATA_NEW_Update, 340 polygons, one per DS division of CPH 2024, with the census
count on each). download.py saves it as raw/arcgis/dsd.geojson.

Level 3 is the 340 DS divisions as they are. Districts and provinces are dissolved from
them, so the three levels nest exactly.

Codes are the census's: DS "2103" (district 21, DS 03), district "21", province "2".

Writes helper1m/data/srilanka/boundaries/adm{1,2,3}.gpkg.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import geopandas as gpd  # noqa: E402
import shapely  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HELPER = Path(__file__).resolve().parents[2]
REPO = HELPER.parent
RAW = HELPER / "data" / "srilanka" / "raw"
OUT = HELPER / "data" / "srilanka" / "boundaries"
COD_ADM0 = REPO / "data" / "asia1m" / "srilanka" / "lka_admin0.shp"

# Province names as the census writes them; the layer has the same.
PROVINCES = {"1": "Western", "2": "Central", "3": "Southern", "4": "Northern", "5": "Eastern",
             "6": "North Western", "7": "North Central", "8": "Uva", "9": "Sabaragamuwa"}


def load_dsd():
    fc = json.loads((RAW / "arcgis" / "dsd.geojson").read_text(encoding="utf-8"))
    g = gpd.GeoDataFrame.from_features(fc["features"], crs="EPSG:4326")
    g["code"] = g["ds_uid"].astype(int).astype(str)
    g["district"] = g["code"].str[:2]
    g["province"] = g["code"].str[0]
    if not (g["Province_C"].astype(str) == g["province"]).all():
        raise SystemExit("ds_uid's first digit is not the province code")
    if len(g) != 340 or not g["code"].is_unique:
        raise SystemExit(f"{len(g)} DS polygons, unique={g['code'].is_unique}; want 340")
    if g["district"].nunique() != 25:
        raise SystemExit(f"{g['district'].nunique()} districts, want 25")
    bad = ~g.geometry.is_valid
    if bad.any():
        print(f"  repairing {int(bad.sum())} invalid DS geometries")
        g.loc[bad, "geometry"] = shapely.make_valid(g.loc[bad, "geometry"].values)
    g["geometry"] = g.geometry.buffer(0)
    return g


def main():
    g = load_dsd()
    eq = "EPSG:5235"   # SLD99 / Sri Lanka Grid 1999, metres
    area = g.to_crs(eq).area
    # overlaps between DS polygons
    sidx = g.sindex
    over = 0.0
    for i, geom in enumerate(g.geometry):
        for j in sidx.query(geom, predicate="intersects"):
            if j > i:
                over += g.geometry.iloc[[i]].to_crs(eq).intersection(
                    g.geometry.iloc[[j]].to_crs(eq).values[0]).area.sum()
    print(f"DS layer: 340 polygons, {area.sum() / 1e6:,.0f} km2, pairwise overlap "
          f"{over / 1e6:.2f} km2")
    if COD_ADM0.exists():
        cod = gpd.read_file(COD_ADM0).to_crs(eq)
        u = g.to_crs(eq).union_all()
        print(f"  against COD-AB adm0 ({cod.area.sum() / 1e6:,.0f} km2): COD land outside the "
              f"DS layer {cod.difference(u).area.sum() / 1e6:,.1f} km2, DS land outside COD "
              f"{(g.to_crs(eq).difference(cod.union_all()).area.sum()) / 1e6:,.1f} km2")

    adm3 = g.assign(name=g["DSD_Name"].str.strip(), parent=g["district"],
                    parent_name=g["District_N"].str.strip(), group=g["province"])
    adm3 = adm3[["code", "name", "parent", "parent_name", "group", "geometry"]]
    adm2 = adm3.dissolve(by="parent", as_index=False)[["parent", "parent_name", "group",
                                                      "geometry"]]
    adm2 = adm2.rename(columns={"parent": "code", "parent_name": "name"})
    adm2["parent"] = adm2["group"]
    adm2["parent_name"] = adm2["group"].map(PROVINCES)
    adm1 = adm3.dissolve(by="group", as_index=False)[["group", "geometry"]]
    adm1["code"] = adm1["group"]
    adm1["name"] = adm1["code"].map(PROVINCES)
    OUT.mkdir(parents=True, exist_ok=True)
    for lv, df in ((1, adm1), (2, adm2), (3, adm3)):
        df = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        df.to_file(OUT / f"adm{lv}.gpkg", driver="GPKG", layer=f"adm{lv}")
        print(f"wrote adm{lv}.gpkg ({len(df)} units)")


if __name__ == "__main__":
    main()
