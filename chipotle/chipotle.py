import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import geopandas
from cartopy import crs as ccrs 
import cartopy.feature as cfeature
import cartopy
import os
from geodatasets import get_path
import colorsys
import random
from PIL import ImageColor
import re
import cartopy.io.shapereader as shpreader


cache_dir = os.path.join(os.getcwd(), 'cartopycache')
cartopy.config['pre_existing_data_dir'] = cache_dir

crs = ccrs.AlbersEqualArea()
plat = ccrs.PlateCarree()

fig, ax = plt.subplots(subplot_kw={"projection": plat}, figsize=(10, 10))
ax.set_extent([-130, -62, 6, 75], crs=ccrs.PlateCarree())

df = pd.read_csv('data/chipotle_stores.csv')
print(df.head(2))

points = ax.projection.transform_points(plat, df.longitude, df.latitude)
ax.scatter(points[:, 0], points[:, 1], c='darkgreen', 
    s=2, zorder=5, alpha=0.5,
    rasterized=True, antialiased=False)

fig.tight_layout()
fig.savefig("chipotle/chipotle.png", dpi=800)


