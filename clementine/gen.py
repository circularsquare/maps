import geopandas as gpd
import networkx as nx
import osmnx as ox
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, NoNorm, PowerNorm
import pickle
import numpy as np
import math
import pandas as pd
from scipy.spatial import KDTree
from itertools import combinations
from pyproj import Transformer
from shapely import LineString
from shapely.geometry import Point, box

ox.settings.use_cache = True
ox.settings.log_console = False

bbox = (-74.067078,40.597792,-73.797226,40.796658)

# Plot the map
fig, ax = plt.subplots(figsize=(39,39))
ax.set_facecolor('#cbe6b5')
plt.rcParams['lines.antialiased'] = False
plt.rcParams['patch.antialiased'] = False

# fips_codes = ['36061', '36047', '36081', '36005', '36085', '34017', '34003']
# waters = []
# for code in fips_codes:
#       url = "https://www2.census.gov/geo/tiger/TIGER2023/AREAWATER/tl_2023_" + code + "_areawater.zip" 
#       water = gpd.read_file(url)
#       waters.append(water)
# water = pd.concat(waters)
# water = water.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
# water.plot(ax=ax, color='lightblue')

parks = ox.features_from_bbox(bbox=bbox,
                              tags={'leisure': ['park', 'garden', 'nature_reserve'],
                                    'landuse': ['cemetery', 'recreation_ground', 'forest']})
parks.plot(ax=ax, color='#90ee90', edgecolor='none', zorder=1)

water = ox.features_from_bbox(bbox=bbox, 
                               tags={'natural': ['water', 'bay', 'strait', 'wetland'],
                                     'waterway': ['river', 'stream', 'canal', 'riverbank', 'tidal_channel'],
                                     'water': True, 'harbor': True, 
                                     'place': 'sea'})
water.plot(ax=ax, color='lightblue', edgecolor='none', zorder=1)


G = ox.graph_from_bbox(bbox, network_type="walk")
ox.plot_graph(G, ax=ax, node_size=0, edge_linewidth=0.005, 
                        edge_color='#a2bd8e', bgcolor=None,
                        show=False, close=False)




fig.savefig('ashy/mapbackw.png', bbox_inches='tight', pad_inches=0,)



