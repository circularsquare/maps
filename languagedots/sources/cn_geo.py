"""China placement layer: religiondots' 3 km population grid re-keyed to chinaethnicity's counties.

    python sources/cn_geo.py      -> data/geo/cn/cn_grid_3km.gpkg (unit = county adcode, pop)

WHY NOT religiondots' layer as it is: its cn_grid_3km.gpkg is keyed to religiondots' own 2,793
county ids, and the nationality table (chinaethnicity, sources/cn_ethnic.py) is keyed to the
2,848 polygons of chinaethnicity/data/geo/counties.gpkg; 55 of those codes are districts created
or renumbered since (Shenzhen's Guangming and Pingshan, Hangzhou's 2021 districts...). The grid's
cells and `pop` are kept as they are and only re-keyed: each cell goes to the county its
CENTROID falls in (AGENT_BRIEF §4.2: re-key religiondots' hexes instead of fetching Kontur
again). Both files are read-only; nothing is written outside languagedots.

A county no cell centroid falls in (small urban districts, islands) gets one cell of its own: its
polygon, with pop 1, so its people are drawn inside it rather than lost. The script prints them.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path  # noqa: E402

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RD_GRID = HERE.parent / "religiondots" / "data" / "geo" / "cn" / "cn_grid_3km.gpkg"
COUNTIES = HERE.parent / "chinaethnicity" / "data" / "geo" / "counties.gpkg"
OUT = HERE / "data" / "geo" / "cn" / "cn_grid_3km.gpkg"


def main():
    grid = gpd.read_file(RD_GRID)
    cty = gpd.read_file(COUNTIES)[["adcode", "geometry"]].to_crs(grid.crs)
    cty["adcode"] = cty["adcode"].astype(str)
    if cty["adcode"].duplicated().any():
        raise SystemExit("counties.gpkg: duplicate adcodes")
    metric = 3857
    pts = gpd.GeoDataFrame(geometry=grid.to_crs(metric).geometry.centroid, crs=metric).to_crs(grid.crs)
    j = gpd.sjoin(pts, cty, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    outside = j["adcode"].isna()
    print(f"  {len(grid):,} cells, {grid['pop'].sum():,.0f} people; {int(outside.sum()):,} cells "
          f"({grid.loc[outside, 'pop'].sum():,.0f} people) outside every county, dropped")
    lay = gpd.GeoDataFrame({"unit": j.loc[~outside, "adcode"].to_numpy(),
                            "pop": grid.loc[~outside, "pop"].to_numpy()},
                           geometry=grid.geometry[~outside].to_numpy(), crs=grid.crs)
    have = set(lay.loc[lay["pop"] > 0, "unit"])
    miss = cty[~cty["adcode"].isin(have)]
    print(f"  {len(miss)} counties with no populated cell centroid get their own polygon "
          f"(pop 1): {sorted(miss['adcode'])}")
    lay = pd.concat([lay, gpd.GeoDataFrame({"unit": miss["adcode"].to_numpy(), "pop": 1.0},
                                           geometry=miss.geometry.to_numpy(), crs=grid.crs)],
                    ignore_index=True)
    lay = gpd.GeoDataFrame(lay, geometry="geometry", crs=grid.crs).to_crs(4326)
    if set(lay["unit"]) != set(cty["adcode"]):
        raise SystemExit("re-keyed layer does not cover every county")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lay.to_file(OUT, layer="cells", driver="GPKG")
    print(f"  wrote {OUT} ({len(lay):,} cells, {lay['unit'].nunique():,} counties)")


if __name__ == "__main__":
    main()
