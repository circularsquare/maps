
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

fig, ax = plt.subplots(subplot_kw={"projection": plat}, figsize=(200, 100))

ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
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

geocities = pd.read_csv('data/geonamescities.csv', sep=';')
print(geocities.columns)
geocities = geocities[['Geoname ID', 'Name', 'ASCII Name', 'Feature Class', 'Population', 'Coordinates']]
geocities['lat'] = geocities.Coordinates.apply(lambda x: x.split(', ')[0]).astype('float')
geocities['lng'] = geocities.Coordinates.apply(lambda x: x.split(', ')[1]).astype('float')
geocities = geocities.rename(columns = {'Population': 'population', 'Name': 'name'})
cities = geocities

def plotCities(cities, color = 'black'):
    points = ax.projection.transform_points(plat, cities.lng, cities.lat)
    ax.scatter(points[:, 0], points[:, 1], c=color, 
        s=(72./fig.dpi)**2 * (cities.population/20000)**0.7 * 2, zorder=100, marker='o',
        rasterized = True, antialiased=False, linewidth=0)

    # for _, city in cities.iterrows():
    #     circle = Circle((city.lng, city.lat), 0.0005, edgecolor=color, facecolor='none', lw=0.1, zorder=100)
    #     ax.add_patch(circle)

popThresholds = [50000, 100000, 150000, 250000, 400000, 600000, 1000000, 1500000, 2500000, 4000000, 6000000, 1e7, 1.5e7, 2e7, 1e9]
popColors = ['#69c722', '#9ec722', '#c9cc29', '#ccb929', '#cc9e29', '#cc8b29', '#cc7727', '#cc5327', '#cc3a27', '#cc2763', '#b8238d', '#9323b8', '#7223b8', '#5223b8']
for i in range(len(popThresholds) - 1):
    plotCities(cities[(cities.population > popThresholds[i]) & (cities.population < popThresholds[i+1])], popColors[i])

plt.tight_layout()
plt.savefig('worldtrain/worldtrain.png')


