
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


crs = ccrs.AlbersEqualArea(-95, 0, 0, 0, (21, 52))
plat = ccrs.PlateCarree()

<<<<<<< HEAD
fig, ax = plt.subplots(subplot_kw={"projection": crs}, figsize=(50, 40))
=======
#fig, ax = plt.subplots(subplot_kw={"projection": crs}, figsize=(50, 40))
fig, ax = plt.subplots(subplot_kw={"projection": crs}, figsize=(100, 80))
>>>>>>> 6bd637e490e1961f85600d98d1427f80e8ae73b5

ax.set_extent([-130, -62, 6, 75], crs=ccrs.PlateCarree())
# ax.add_feature(cfeature.NaturalEarthFeature(
#     category='cultural', name='admin_1_states_provinces',
#     scale=resolution, edgecolor='#535D8D', facecolor='none', zorder=10,
#     rasterized = True, antialiased = False, linewidth = 0.005))
ax.add_feature(cfeature.NaturalEarthFeature(
    category='physical', name='lakes',
    scale=resolution, facecolor='#D9F6FF', edgecolor='#535D8D', zorder=5,
    rasterized = True, antialiased = False, linewidth = 0.004))
ax.add_feature(cfeature.NaturalEarthFeature(
    category='physical', name='ocean',
    scale=resolution, facecolor='#D9F6FF', edgecolor='none', zorder=5,
    rasterized = True, antialiased = False, linewidth = 0.004))
ax.add_feature(cfeature.NaturalEarthFeature(
    category='physical',name='coastline',
    scale=resolution, facecolor='none', edgecolor='#535D8D', zorder=10,
    rasterized = True, antialiased = False, linewidth = 0.004))


# def plotShapefile(path):
#     df = geopandas.read_file(path)
#     df = df.to_crs(crs)
#     df.plot(ax=ax, edgecolor = '#63538d', facecolor='none', zorder = 10, 
#         rasterized=True, antialiased = False, linewidth = 0.004)
# plotShapefile('data/na1m/Muni_2012gw.shp')
# plotShapefile('data/na1m/cb_2018_us_county_5m.shp')
# plotShapefile('data/na1m/CAN_adm2.shp')


# cities = pd.read_csv('data/worldcities.csv', sep=',', lineterminator='\n')
# bigCities = cities[cities.population > 100000]

# points = ax.projection.transform_points(plat, bigCities.lng, bigCities.lat)
# ax.scatter(points[:, 0], points[:, 1], c='black', 
#     s=(72./fig.dpi)**2, zorder=5, marker='1',
#     rasterized=True, antialiased=False)
# ax.scatter(points[:, 0], points[:, 1], c='black', 
#     s=(72./fig.dpi)**2, zorder=5, marker='2',
#     rasterized=True, antialiased=False)

plt.tight_layout()
<<<<<<< HEAD
plt.savefig('na1m/nacoast.png')
=======
plt.savefig('na1m/nacoast2x.png')
>>>>>>> 6bd637e490e1961f85600d98d1427f80e8ae73b5



# just coastline
# fig, ax = plt.subplots(subplot_kw={"projection": crs}, figsize=(50, 40))
# ax.set_extent([-17, 40, 33, 70], crs=ccrs.PlateCarree())
# ax.add_feature(cfeature.NaturalEarthFeature(
#     category='physical',name='coastline',
#     scale=resolution, facecolor='none', edgecolor='#535D8D',
#     rasterized = True, antialiased = False, linewidth = 0.005))
# plt.tight_layout()
# plt.savefig('nacoastline.png')
