
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import geopandas
from cartopy import crs as ccrs 
import cartopy.feature as cfeature
import cartopy
import os
from geodatasets import get_path

resolution = '10m'

# save coastline 
import cartopy.io.shapereader as shpreader
# shpreader.natural_earth(category='physical', name='land', resolution=resolution)
cache_dir = os.path.join(os.getcwd(), 'cartopycache')
cartopy.config['pre_existing_data_dir'] = cache_dir


crs = ccrs.AlbersEqualArea()
plat = ccrs.PlateCarree()

fig, ax = plt.subplots(subplot_kw={"projection": crs}, figsize=(50, 40))

ax.set_extent([-17, 40, 33, 70], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.NaturalEarthFeature(
    category='cultural', name='admin_1_states_provinces',
    scale=resolution, edgecolor='#535D8D', facecolor='none', zorder=10,
    rasterized = True, antialiased = False, linewidth = 0.005))
ax.add_feature(cfeature.NaturalEarthFeature(
    category='physical', name='rivers_lake_centerlines',
    scale=resolution, facecolor='none', edgecolor='#80ACCB', zorder=5,
    rasterized = True, antialiased = False, linewidth = 0.005))
ax.add_feature(cfeature.NaturalEarthFeature(
    category='physical', name='lakes',
    scale=resolution, facecolor='#D9F6FF', edgecolor='none', zorder=5,
    rasterized = True, antialiased = False, linewidth = 0.005))
ax.add_feature(cfeature.NaturalEarthFeature(
    category='physical', name='ocean',
    scale=resolution, facecolor='#D9F6FF', edgecolor='none', zorder=5,
    rasterized = True, antialiased = False, linewidth = 0.005))


cities = pd.read_csv('worldcities.csv', sep=',', lineterminator='\n')
bigCities = cities[cities.population > 100000]

points = ax.projection.transform_points(plat, bigCities.lng, bigCities.lat)
ax.scatter(points[:, 0], points[:, 1], c='r', 
    s=(72./fig.dpi)**2, zorder=5, marker=',',
    rasterized=True, antialiased=False)

plt.tight_layout()
plt.savefig('geopEuroDivs2.png')


# just coastline
fig, ax = plt.subplots(subplot_kw={"projection": crs}, figsize=(50, 40))
ax.set_extent([-17, 40, 33, 70], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.NaturalEarthFeature(
    category='physical',name='coastline',
    scale=resolution, facecolor='none', edgecolor='#535D8D',
    rasterized = True, antialiased = False, linewidth = 0.005))
plt.tight_layout()
plt.savefig('geopEuroCoastline.png')



    

# image = m.render(zoom=3)
# image.save('meow.png')
