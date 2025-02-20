
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle
import geopandas
from cartopy import crs as ccrs 
import cartopy.feature as cfeature
import cartopy
import os
from geodatasets import get_path

resolution = '50m'

# save coastline 
import cartopy.io.shapereader as shpreader
# shpreader.natural_earth(category='physical', name='land', resolution=resolution)
cache_dir = os.path.join(os.getcwd(), 'cartopycache')
cartopy.config['pre_existing_data_dir'] = cache_dir


#crs = ccrs.AlbersEqualArea(-95, 0, 0, 0, (21, 52))
plat = ccrs.PlateCarree()

fig, ax = plt.subplots(subplot_kw={"projection": plat}, figsize=(100, 60))

ax.set_extent([-180, 180, -85, 90], crs=ccrs.PlateCarree())
# ax.add_feature(cfeature.NaturalEarthFeature(
#     category='cultural', name='admin_1_states_provinces',
#     scale=resolution, edgecolor='#535D8D', facecolor='none', zorder=10,
#     rasterized = True, antialiased = False, linewidth = 0.005))
# ax.add_feature(cfeature.NaturalEarthFeature(
#     category='physical', name='lakes',
#     scale=resolution, facecolor='#D9F6FF', edgecolor='none', zorder=5,
#     rasterized = True, antialiased = False, linewidth = 0.004))
# ax.add_feature(cfeature.NaturalEarthFeature(
#     category='physical', name='ocean',
#     scale=resolution, facecolor='#D9F6FF', edgecolor='none', zorder=5,
#     rasterized = True, antialiased = False, linewidth = 0.004))
# ax.add_feature(cfeature.NaturalEarthFeature(
#     category='physical',name='coastline',
#     scale=resolution, facecolor='none', edgecolor='#535D8D', zorder=10,
#     rasterized = True, antialiased = False, linewidth = 0.004))
# ax.add_feature(cfeature.NaturalEarthFeature(
#     category='physical',name='coastline',
#     scale=resolution, facecolor='none', edgecolor='#535D8D', zorder=10,
#     rasterized = True, antialiased = False, linewidth = 0.004))

# cities = pd.read_csv('data/worldcities.csv', sep=',', lineterminator='\n')
# bigCities = cities[cities.population > 100000]

# geocities = pd.read_csv('data/geonamescities.csv', sep=';')
# print(geocities.columns)
# geocities = geocities[['Geoname ID', 'Name', 'ASCII Name', 'Feature Class', 'Population', 'Coordinates']]
# geocities['lat'] = geocities.Coordinates.apply(lambda x: x.split(', ')[0]).astype('float')
# geocities['lng'] = geocities.Coordinates.apply(lambda x: x.split(', ')[1]).astype('float')
# geocities = geocities.rename(columns = {'Population': 'population', 'Name': 'name'})
# cities = geocities

# def plotCities(cities, color = 'black'):
#     points = ax.projection.transform_points(plat, cities.lng, cities.lat)
#     ax.scatter(points[:, 0], points[:, 1], c=color, 
#         s=(72./fig.dpi)**2 * (cities.population/20000)**0.7 * 2, zorder=100, marker='.',
#         rasterized = True, antialiased=False, linewidth=0)

# popThresholds = [20000, 50000, 100000, 150000, 250000, 400000, 600000, 1000000, 1500000, 2500000, 4000000, 6000000, 1e7, 1.5e7, 2e7, 1e9]
# popColors = ['#22c753', '#69c722', '#9ec722', '#c9cc29', '#ccb929', '#cc9e29', '#cc8b29', '#cc7727', '#cc5327', '#cc3a27', '#cc2763', '#b8238d', '#9323b8', '#7223b8', '#5223b8']
# for i in range(len(popThresholds) - 1):
#     print(popThresholds[i])
#     plotCities(cities[(cities.population > popThresholds[i]) & (cities.population < popThresholds[i+1])], popColors[i])

# plt.tight_layout()
# plt.savefig('worldtrain/worldtrainheight.png')




from PIL import Image
Image.MAX_IMAGE_PIXELS = 300000000 
# Open the TIFF file
image = Image.open('data/NE1_HR_LC.tif')

# Resize the image to a certain resolution
shrunk_image = image.resize((10000, 5000))

# Save the image as a PNG
shrunk_image.save('worldtrainheight.png', format='PNG')