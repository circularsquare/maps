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

resolution = '10m'

# save coastline 
import cartopy.io.shapereader as shpreader
# shpreader.natural_earth(category='physical', name='land', resolution=resolution)
cache_dir = os.path.join(os.getcwd(), 'cartopycache')
cartopy.config['pre_existing_data_dir'] = cache_dir

crs = ccrs.AlbersEqualArea()
plat = ccrs.PlateCarree()

df = geopandas.read_file('data/tl_2020_36_tabblock20.shp')
df = df.to_crs("EPSG:4326")

pop = pd.read_csv('data/DECENNIALDHC2020.P12-Data.csv') # block is DECENNIALDHC2020.P12-Data.csv
pop = pop.rename(columns={'NAME': 'name', 'P12_001N': 'population'})
pop['GEOID20'] = pop.GEO_ID.apply(lambda x: x[-15:])
pop = pop[['GEOID20', 'name', 'population']]

df = pd.merge(df, pop, left_on='GEOID20', right_on='GEOID20')
df.population = df.population.astype(int)


counties = ['061', '047', '081', '005', '085']
df = df[(df.COUNTYFP20 == '061') | (df.COUNTYFP20 == '047') | (df.COUNTYFP20 == '081') | (df.COUNTYFP20 == '005') | (df.COUNTYFP20 == '085')] # 5 boroughs
#df = df[df.COUNTYFP20 == '061'] #just manhattan
print('merged, ', df.iloc[:3, :])

fig, ax = plt.subplots(subplot_kw={"projection": plat}, figsize=(10, 10))
ax.set_extent([-74.15, -73.7, 40.55, 40.92])

df['density'] = df.population/(df.ALAND20+df.AWATER20+1) 

print('pop ', pop.iloc[1:3, :])
print('blocks ', df.iloc[1, :])

# df.plot(column='density', cmap='magma', ax=ax)



subway = pd.read_csv('data/subwaystations.csv')
subway['population'] = 0
df['station'] = -1

def nearestStation(lng, lat, population):
    lowestIndex = nearestStation2(lng, lat)
    subway.loc[lowestIndex, "population"] += population  # add block population to station.
    return lowestIndex
def nearestStation2(lng, lat):
    lowestDist2 = 100
    lowestStation = -1
    lowestIndex = 0
    for index, station in subway.iterrows():
        # cos(41 deg)
        dist2 = ((float(lng) - float(station.Longitude)) * .755)**2 + (float(lat) - float(station.Latitude))**2
        if dist2 < lowestDist2:
            lowestDist2 = dist2
            lowestStation = station 
            lowestIndex = index
    return lowestIndex

print(df.shape)
i = 0
for index, block in df.iterrows():
    if (block.ALAND20/(block.AWATER20+0.001) > 1.0): # if less than 5, black
        # don't use the version with population cuz thats already calculated (subwaystations2)
        df.loc[index, 'station'] = nearestStation2(block.INTPTLON20, block.INTPTLAT20)
        # add station id to block.
    i += 1
    if i % 1000 == 0:
        print(i)
    if i > 2000000:
        break;




subway = pd.read_csv('data/subwaystations2.csv')
subway['route1'] = subway['Daytime Routes'].apply(lambda x: x.split(' ')[0])
subway['routes'] = subway['Daytime Routes'].apply(lambda x: x.split(' '))
colorDict = {
    'A': '#005ba6', 'C': '#0039A6', 'E': '#0042a6', 
    'B': '#ff6c26', 'D': '#ff8426', 'F': '#FF6319', 'M': '#ff7519', 
    'G': '#6CBE45', 'L': '#A7A9AC', 
    'J': '#996633', 'Z': '#945e28',   
    'N': '#FCCC0A', 'Q': '#fcd40a', 'R': '#fcdc0a', 'W': '#fcf40a', 
    '1': '#e32b24', '2': '#EE352E', '3': '#ee2e4b', 
    '4': '#00a336', '5': '#00a357', '6': '#00933C', 
    '7': '#B933AD', 'S': '#808183', 'SIR': '#0039A6'
}
def randomizeHue(color):
    r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255);
    r, g, b = colorsys.hsv_to_rgb(
        (h + (random.random()-0.5)*0.07)%1, 
        min(max(s + (random.random()-0.5)*0.1, 0), 1), 
        v)
    hex_code = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
    return hex_code
def getStationColor(routes):
    rtot, gtot, btot, stot = (0, 0, 0, 0)
    for route in routes:
        r, g, b = ImageColor.getcolor(colorDict[route], "RGB")
        rtot += r/255;
        gtot += g/255;
        btot += b/255;
        stot += colorsys.rgb_to_hsv(r/255, g/255, b/255)[1]
    r, g, b, s = (rtot / len(routes), gtot / len(routes), btot / len(routes), stot / len(routes))
    h, sMerged, v = colorsys.rgb_to_hsv(r, g, b)
    
    noiseFactor = 0.06
    if len(routes) > 4:
        noiseFactor = 0.01
    elif routes[0] in ('B', 'D', 'F', 'M', 'N', 'Q', 'R', 'W'):
        noiseFactor = 0.045
    elif routes[0] in ('A', 'C', 'E', '1', '2', '3', '7', 'G', 'SIR'):
        noiseFactor = 0.07
    elif routes[0] in ('4', '5', '6'):
        noiseFactor = 0.08
    elif routes[0] in ('L'):
        noiseFactor = 0.12
    vnoiseFactor = 0.02
    if routes[0] == 'L':
        vnoiseFactor = 0.05
    hnew = h + ((random.random()-0.5)*noiseFactor)%1
    snew = min(max((s + sMerged)/2 + (random.random()-0.5)*0.1 - 0, 0), 1)
    vnew = min(max(v + (random.random()-0.5)*vnoiseFactor - 0, 0), 1)
    r, g, b = colorsys.hsv_to_rgb(hnew, snew, vnew)
    hex = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
    return hex

def muteHue(hex, density):
    r, g, b = ImageColor.getcolor(hex, "RGB")
    densityFactor = ((1 / (1 + np.exp(-density*1.4)) - 0.5) * 2) * 0.73 + 0.12
    return '#{:02x}{:02x}{:02x}'.format(int(r*densityFactor), int(g*densityFactor), int(b*densityFactor))

subway['color'] = subway.routes.apply(getStationColor)

df['color'] = df.station.apply(lambda x: subway.iloc[x].color)
df.loc[df.station == -1, 'color'] = '#000000' # 0 pop or unpopulated for other reason
df.color = df.apply(lambda x: muteHue(x.color, x.density * 30), axis=1)
df.plot(color = df.color, ax=ax)


ax.scatter(subway.Longitude, subway.Latitude, s=subway.population / 1000, c=subway.color, alpha=0.8)

print(subway.sort_values('population').head(3))

ax.set_facecolor('black')
plt.tight_layout()
fig.savefig("nycensus/geopNyc.png", dpi=500)


# df.to_csv('nycensus/blockswithpop.csv')
#subway.to_csv('data/subwaystations2.csv')


# todo
# color each station
# average color for stations with multiple lines?
# brighter colors for express
# color each block based on nearest station (unless 0 pop)


