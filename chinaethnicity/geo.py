"""County polygons and province outlines, taken from religiondots' DataV build.

religiondots/sources/cn_geo.py already walks DataV GeoAtlas for all 2,848 county-level units
with their GB/T 2260 adcodes, and clips them to de facto administration (it removes ground
China claims but does not administer, such as Arunachal Pradesh). Rather than repeat that,
this reads its two outputs and adds the names the census join needs. religiondots/sources/
cn_geo.md is the record of how those files were made and why DataV and not geoBoundaries.

Writes:
    data/geo/counties.gpkg      adcode, name, city_code, city, prov_code, prov + polygon
    data/geo/provinces.geojson  31 simplified province outlines for the viewer

Usage:
    python geo.py
"""
import json
import os
import sys

import geopandas as gpd
import pandas as pd
import shapely

from common import GEO, PROC, PROVINCES, RELIGIONDOTS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC_COUNTIES = os.path.join(RELIGIONDOTS, "data", "geo", "cn", "cn_counties.gpkg")
SRC_INDEX = os.path.join(RELIGIONDOTS, "data", "raw", "cn", "datav", "county_index.json")

OUT_COUNTIES = os.path.join(GEO, "counties.gpkg")
OUT_PROVINCES = os.path.join(PROC, "provinces.geojson")


def main():
    for p in (SRC_COUNTIES, SRC_INDEX):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}\n  build it in religiondots: "
                             f"python sources/cn_geo.py --fetch && python sources/cn_geo.py")

    g = gpd.read_file(SRC_COUNTIES)
    g = g.rename(columns={"unit": "adcode"})[["adcode", "geometry"]]
    g["adcode"] = g["adcode"].astype(str)
    with open(SRC_INDEX, encoding="utf-8") as fh:
        idx = pd.DataFrame(json.load(fh)).rename(columns={"code": "adcode"})
    if idx["adcode"].duplicated().any():
        raise SystemExit("county_index.json has duplicate adcodes")

    g = g.merge(idx, on="adcode", how="left")
    unnamed = g["name"].isna()
    if unnamed.any():
        raise SystemExit(f"{int(unnamed.sum())} polygons have no index entry: "
                         f"{g.loc[unnamed, 'adcode'].tolist()[:10]}")
    g = g.sort_values("adcode").reset_index(drop=True)
    g = g[["adcode", "name", "city_code", "city", "prov_code", "prov", "geometry"]]

    os.makedirs(GEO, exist_ok=True)
    g.to_file(OUT_COUNTIES, layer="counties", driver="GPKG")
    print(f"wrote {OUT_COUNTIES}: {len(g):,} counties in {g['prov_code'].nunique()} provinces")

    print("dissolving province outlines…")
    p = g.dissolve(by="prov_code").reset_index()[["prov_code", "geometry"]]
    # ~2 km tolerance: these are drawn as hairlines under the dots and only need to read
    # as borders. At 0.01 the file was 5.9 MB, which the viewer fetches before anything.
    # Tiny islands are dropped from the outline only; their dots are unaffected. The parts
    # are written as separate features rather than re-dissolved, because unioning the
    # simplified pieces raises GEOS topology errors along Liaoning's coast.
    # County polygons do not share edges exactly, so the dissolve leaves thousands of hairline
    # holes along internal borders, and those, not the coast, were most of the 5 MB. A small
    # outward-then-inward buffer closes them before simplifying.
    healed = p.geometry.buffer(0.01).buffer(-0.01)
    p["geometry"] = shapely.make_valid(healed.simplify(0.02, preserve_topology=True).values)
    p = p.explode(index_parts=False)
    p = p[p.geom_type.isin(["Polygon", "MultiPolygon"]) & (p.geometry.area > 0.0004)]
    p["geometry"] = [shapely.Polygon(g.exterior,
                                     [r for r in g.interiors if shapely.Polygon(r).area > 0.01])
                     for g in p.geometry]
    p["code"] = p["prov_code"].str[:2]
    p["key"] = p["code"].map(lambda c: PROVINCES[c][0])
    p["name"] = p["code"].map(lambda c: PROVINCES[c][1])
    p = p[["code", "key", "name", "geometry"]]
    os.makedirs(PROC, exist_ok=True)
    p.to_file(OUT_PROVINCES, driver="GeoJSON", COORDINATE_PRECISION=4)
    print(f"wrote {OUT_PROVINCES}: {len(p)} provinces, "
          f"{os.path.getsize(OUT_PROVINCES) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
