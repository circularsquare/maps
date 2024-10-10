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

resolution = '10m'

# save coastline 
import cartopy.io.shapereader as shpreader
# shpreader.natural_earth(category='physical', name='land', resolution=resolution)
cache_dir = os.path.join(os.getcwd(), 'cartopycache')
cartopy.config['pre_existing_data_dir'] = cache_dir

crs = ccrs.AlbersEqualArea()
plat = ccrs.PlateCarree()

# df = geopandas.read_file('nycensus/tl_2020_36_tabblock20.shp')
# df = df.to_crs("EPSG:4326")

# pop = pd.read_csv('nycensus/DECENNIALDHC2020.P12-Data.csv') # block is DECENNIALDHC2020.P12-Data.csv
# pop = pop.rename(columns={'NAME': 'name', 'P12_001N': 'population'})
# pop['GEOID20'] = pop.GEO_ID.apply(lambda x: x[-15:])
# pop = pop[['GEOID20', 'name', 'population']]

# print('pop ', pop.iloc[1:3, :])
# print('blocks ', df.iloc[1, :])
# df = pd.merge(df, pop, left_on='GEOID20', right_on='GEOID20')
# df.population = df.population.astype(int)

# counties = ['061', '047', '081', '005', '085']
# df = df[(df.COUNTYFP20 == '061') | (df.COUNTYFP20 == '047') | (df.COUNTYFP20 == '081') | (df.COUNTYFP20 == '005') | (df.COUNTYFP20 == '085')]
# print('merged, ', df.iloc[:3, :])

# # df['geometry'] = df['geometry'].to_wkt()
# # df.to_csv('nycensus/blockswithpop.csv')

# # df = pd.read_csv('nycensus/blockswithpop.csv')
# # df['geometry'] = df['geometry'].apply(geopandas.from_wkt)
# # df = geopandas.GeoDataFrame(df, crs="EPSG:4326")
# # print(df.iloc[:4, :])

fig, ax = plt.subplots(subplot_kw={"projection": plat}, figsize=(10, 10))
ax.set_extent([-74.2, -73.7, 40.5, 41.0])

# df.plot(column='population', cmap='magma', ax=ax)

# subway = pd.read_csv('nycensus/subwaystations.csv')
# subway['population'] = 0
# df['station'] = 0

# def nearestStation(lng, lat, population):
#     lowestDist2 = 100
#     lowestStation = 0
#     lowestIndex = 0
#     for index, station in subway.iterrows():
#         dist2 = (float(lng) - float(station.Longitude))**2 + (float(lat) - float(station.Latitude))**2
#         if dist2 < lowestDist2:
#             lowestDist2 = dist2
#             lowestStation = station 
#             lowestIndex = index
#     subway.loc[lowestIndex, "population"] += population  # add block population to station.
#     return lowestIndex

# print(df.size)
# i = 0
# for _, block in df.iterrows():
#     if block.population > 0:
#         block.station = nearestStation(block.INTPTLON20, block.INTPTLAT20, block.population)
#         # add station id to block.
#     i += 1
#     if i % 1000 == 0:
#         print(i)

subway = pd.read_csv('nycensus/subwaystations2.csv')
subway['route1'] = subway['Daytime Routes'].apply(lambda x: x.split(' ')[0])
colorDict = {
    'A': '#0039A6', 'C': '#0039A6', 'E': '#0039A6', 
    'B': '#FF6319', 'D': '#FF6319', 'F': '#FF6319', 'M': '#FF6319', 
    'G': '#6CBE45',   'L': '#A7A9AC', 
    'J': '#996633', 'Z': '#996633',   
    'N': '#FCCC0A', 'Q': '#FCCC0A', 'R': '#FCCC0A', 
    '1': '#EE352E', '2': '#EE352E', '3': '#EE352E', 
    '4': '#00933C', '5': '#00933C', '6': '#00933C', 
    '7': '#B933AD',   'S': '#808183', 'SIR': '#0039A6'
}
def randomizeHue(color):
    r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = r/255, g/255, b/255

    r = max(0, min(1, (random.random() - 0.5) * 0.2 + r))
    g = max(0, min(1, (random.random() - 0.5) * 0.2 + g))
    b = max(0, min(1, (random.random() - 0.5) * 0.2 + b))

    hex_code = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
    return hex_code

subway['color'] = subway.route1.apply(lambda x: randomizeHue(colorDict[x]))

ax.scatter(subway.Longitude, subway.Latitude, s=subway.population / 1000, c=subway.color, alpha=0.8)

print(subway.sort_values('population').head(3))

ax.set_facecolor('black')
plt.tight_layout()
fig.savefig("geopNyc.png", dpi=300)


# df.to_csv('nycensus/blockswithpop.csv')
subway.to_csv('nycensus/subwaystations2.csv')


# todo
# color each station
# average color for stations with multiple lines?
# color each block based on nearest station (unless 0 pop)
# size each station according to popultion


