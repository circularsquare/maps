"""
One-time: convert the 414 MB StatCan DA cartographic shapefile into a compact
GeoParquet keyed by DGUID (reprojected to EPSG:4326, geometries made valid).

Reading the raw .shp with an attribute filter takes ~4 min per build; the parquet
loads in a second or two. build_dots.py prefers this cache when it exists.

Usage:
    python convert_boundaries.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).parent
SHP = HERE / "data" / "shapefiles" / "lda_2021" / "lda_000b21a_e.shp"
OUT = HERE / "data" / "shapefiles" / "da_2021.parquet"


def main():
    print(f"Reading {SHP} (this is the slow step, ~4 min)...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        gdf = gpd.read_file(SHP, columns=["DGUID", "PRUID"])
    print(f"  {len(gdf)} DAs read; reprojecting to EPSG:4326 + make_valid...")
    gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf["geometry"].make_valid()
    gdf.to_parquet(OUT)
    print(f"  wrote {OUT} ({OUT.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
