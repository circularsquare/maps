import geopandas as gpd
import networkx as nx
import osmnx as ox
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, NoNorm, PowerNorm
import pickle
import numpy as np
import pandas as pd
import math
from pyproj import Transformer
from shapely.geometry import Point, box, MultiLineString, LineString
from itertools import combinations
from shapely.ops import linemerge


def generate_random_rgb():
    return (random.random(), random.random(), random.random())
    
# Define your project's target CRS
TARGET_CRS = "EPSG:2263"

#lines_path = "data/nystreets/routes_nyc_subway_may2016.shp"
lines_path = 'data/nystreets/geo_export_6c9929cb-3ce9-4cb5-862d-dc06efa13f97.shp'
lines_gdf = gpd.read_file(lines_path)
lines_gdf['geometry'] = lines_gdf.geometry.apply(lambda x: linemerge(x) if isinstance(x, MultiLineString) else x)
lines_gdf = lines_gdf.to_crs(TARGET_CRS)

stations_df = pd.read_csv("data/nystreets/MTA_Subway_Stations.csv")
stations_gdf = gpd.GeoDataFrame(
    stations_df,
    geometry=gpd.points_from_xy(stations_df['GTFS Longitude'], stations_df['GTFS Latitude']),
    crs="EPSG:4326").to_crs(TARGET_CRS)
stations_gdf['routes'] = stations_gdf['Daytime Routes'].str.split(' ')
print(stations_gdf[['Stop Name', 'Daytime Routes']].head(40))

print('subway data loaded!')

imageSize = 80
fig, ax = plt.subplots(figsize=(imageSize, imageSize))
ax.set_facecolor('black')
ax.set_axis_off()

lines_gdf = lines_gdf[lines_gdf.service == 'A']
# station = stations_gdf[stations_gdf['Stop Name'] == 'Court Sq']
# print(station)
rgbs = [generate_random_rgb() for x in range(len(lines_gdf))]
lines_gdf.plot(
    ax=ax,
    linewidth = 5,
    color = rgbs,
    alpha=0.5,
)

stations_gdf.plot(ax=ax)
# station.plot(ax=ax)

plt.tight_layout()
plt.savefig('nystreets/subway.png', bbox_inches='tight', pad_inches=0,
facecolor='white')
print('plot saved!')

#lines_gdf[['objectid', 'service_na', 'service', 'shape_stle']].to_csv('subway_lines_data.csv', index=False)
print('meow')


# import geopandas as gpd
# import matplotlib.pyplot as plt


# kontur_filepath = "data/nystreets/kontur_population_US_20231101.gpkg"
# west, south, east, north = -74.253845, 40.497615, -73.656464, 40.981972
# bbox_gdf_latlon = gpd.GeoDataFrame(
#     geometry=[box(west, south, east, north)],
#     crs="EPSG:4326"
# )
# bbox_gdf_projected = bbox_gdf_latlon.to_crs("EPSG:3857")
# projected_bbox_coords = tuple(bbox_gdf_projected.total_bounds)
# print("\nReloading the file using the correctly projected bounding box...")
# pop_gdf = gpd.read_file(
#     kontur_filepath,
#     bbox=projected_bbox_coords
# )

# print(f"Data loaded successfully. Found {len(pop_gdf)} population grid cells.")
# print(pop_gdf.columns)
# pop_gdf.plot(column='population')
# plt.savefig('nystreets/population.png', bbox_inches='tight', pad_inches=0,
# facecolor='black')