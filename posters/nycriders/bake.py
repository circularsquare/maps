"""
Cut the basemap extracts the nycriders poster needs out of the big global
sources, once, into build/.

The OSM water and coastline files are 1.2 GB each; reading them with a bbox
filter takes about a minute, which is a minute the render should not spend
every time a colour changes.

    python bake.py

Outputs:
    build/nyc_water.gpkg   OSM water polygons (ocean, harbour, the rivers)
    build/nyc_coast.gpkg   the coastline as lines, for a hairline shore
"""

from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).parent
BUILD = HERE / "build"
DATA = HERE.parent.parent / "data"

# Generous enough that any reasonable reframing still lands inside it: the whole
# harbour, the Sound out past City Island, and the ocean south of the Rockaways.
BBOX = (-74.35, 40.40, -73.55, 41.05)


def cut(src, out, layer):
    t0 = time.time()
    g = gpd.read_file(src, bbox=BBOX)
    g = g.to_crs("EPSG:4326")
    g.to_file(out, layer=layer, driver="GPKG")
    print(f"  {out.name}: {len(g)} features ({time.time() - t0:.0f}s)")


def main():
    BUILD.mkdir(exist_ok=True)
    print("Cutting basemap extracts (this reads two 1.2 GB shapefiles)...")
    cut(DATA / "water-polygons-split-4326" / "water_polygons.shp",
        BUILD / "nyc_water.gpkg", "water")
    cut(DATA / "coastlines-split-4326" / "lines.shp",
        BUILD / "nyc_coast.gpkg", "coast")
    print("Done.")


if __name__ == "__main__":
    main()
