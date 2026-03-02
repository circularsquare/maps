
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import geopandas
from cartopy import crs as ccrs 
import cartopy.feature as cfeature
import cartopy
import os
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
from shapely.geometry import box
import rasterio
from rasterio.plot import show
from rasterio.transform import from_bounds
from rasterio.features import rasterize as rio_rasterize



# final tuple is latitude levels of correctness
crs = ccrs.AlbersEqualArea(95, 0, 0, 0, (6, 42))
plat = ccrs.PlateCarree()

fig, ax = plt.subplots(subplot_kw={"projection": crs}, figsize=(95, 80))
#ax.set_extent([47.7, 129.2, -9, 65], crs=ccrs.PlateCarree())

with rasterio.open('waterdata/seasonality_120E_30Nv1_4_2021.tif') as src:
    img = src.read()
    print(img)
    show(img, transform=src.transform, ax=ax)

plt.tight_layout()
plt.savefig('waterdataplot.png')


def plotTif(df_or_path, coverage_threshold=0.25, scale=3):
    if isinstance(df_or_path, str):
        df = geopandas.read_file(df_or_path)
    else:
        df = df_or_path
    if df.crs is None:
        df = df.set_crs("EPSG:4326")
    df = df.to_crs(crs)
    
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    
    width, height = 9500, 8000  # your actual output size
    
    transform_hr = from_bounds(x0, y0, x1, y1, width*scale, height*scale)
    
    hr_mask = rio_rasterize(
        [(geom, 1) for geom in df.geometry],
        out_shape=(height*scale, width*scale),
        transform=transform_hr,
        fill=0, dtype=np.uint8
    )
    
    hr_mask = hr_mask.reshape(height, scale, width, scale)
    coverage = hr_mask.mean(axis=(1, 3))
    water_mask = coverage >= coverage_threshold
    
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[water_mask] = [150, 200, 250, 255]
    
    ax.imshow(rgba, extent=[x0, x1, y0, y1], origin='upper', zorder=10, interpolation='none')

