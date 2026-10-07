"""Kontur population 2023-11-01 for Pakistan, as hex centroids with population.

The source is religiondots' copy of the Kontur gz (read only). It is unpacked once into
helper1m/data/pakistan/ and the centroids cached as a small parquet-free CSV-like gpkg.
"""

import gzip
import os
import shutil

KONTUR_GZ = os.path.join(os.path.dirname(__file__), "..", "..", "..", "religiondots", "data",
                         "geo", "kontur", "kontur_population_PK_20231101.gpkg.gz")


def centroids(data_dir):
    import geopandas as gpd

    out = os.path.join(data_dir, "kontur_pk_centroids.gpkg")
    if os.path.exists(out):
        return gpd.read_file(out)
    gpkg = os.path.join(data_dir, "kontur_population_PK_20231101.gpkg")
    if not os.path.exists(gpkg):
        with gzip.open(os.path.abspath(KONTUR_GZ), "rb") as src, open(gpkg, "wb") as dst:
            shutil.copyfileobj(src, dst)
    hexes = gpd.read_file(gpkg)
    pts = gpd.GeoDataFrame({"pop": hexes["population"].astype(float).to_numpy()},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(4326)
    pts.to_file(out, driver="GPKG")
    os.remove(gpkg)          # the unpacked hex file is ~1 GB of scratch; the centroids stay
    return pts
