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


rail = pd.read_csv('data/nyc/railprocessed.csv')
print(rail.sort_values('population').head(10))

plat = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw={"projection": plat}, figsize=(10, 10))
ax.scatter(rail.Longitude, rail.Latitude, s=rail.population / 1000, c=rail.color, alpha=0.8)

ax.set_facecolor('white')
plt.tight_layout()
fig.savefig("nycatchment/plotNyc.png", dpi=1000)

