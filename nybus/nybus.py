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


import hashlib
import random

class RandomHashMap:
    def __init__(self, seed):
        self.seed = seed
        self.random = random.Random(seed)
    def hash(self, input):
        # Use a hash function to convert the input to a fixed-size integer
        hash_value = int(hashlib.sha256(input.encode()).hexdigest(), 16)
        # Use the random number generator to map the hash value to a random output
        output = self.random.random()  # adjust the range as needed
        return output

# Create a RandomHashMap instance with a seed
random_hash_map = RandomHashMap(42)



# convert mta excel to csv
# mta = pd.read_excel("nybus/2023 MTA Bus Tables.xlsx")
# mta.columns = mta.iloc[0]
# mta = mta[1:]
# mta.to_csv('nybus/mta.csv')
# nyct = pd.read_excel("nybus/2023 NYCT Bus Tables.xlsx")
# nyct.columns = nyct.iloc[0]
# nyct = nyct[1:]
# nyct.to_csv('nybus/nyct.csv')

def splitRoute(s):
    match = re.search(r'^[a-zA-Z]+', s)
    if match:
        alphabetic_portion = match.group()
    else:
        alphabetic_portion = ""

    match = re.search(r'\d+', s)
    if match:
        numeric_portion = match.group()
    else:
        numeric_portion = ""
    return alphabetic_portion, numeric_portion

mta = pd.read_csv('nybus/mta.csv')
nyct = pd.read_csv('nybus/nyct.csv')
mta = pd.concat([mta, nyct])
mta['riders'] = mta['2023.0']
mta = mta[['Route', 'riders']]
mta = mta.dropna() # also drops stuff that doesnt exist in 2023 ig.

def routeToColor(route, sbs):
    rand = random.Random(route + 'seed26') #16
    h = rand.random()
    r2 = rand.random()
    s = 1 - (r2*r2)*0.8
    l = 0.6
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    hex_code = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
    return hex_code
    # default '#4287f5'

#print(mta.Route.unique())
for i in range(len(mta)):
    route = str(mta.iloc[i].Route)
    route = route.replace("Lcl", "").replace("SBS", "").replace(" ", "").replace("Bx", "BX")
    if len(route.split('/')) > 1:
        if len(route.split('/')[1]) > 0:
            riders = mta.iloc[i].riders
            route1 = route.split('/')[0]
            route2 = route.split('/')[1]
            route1a, route1n = splitRoute(route1)
            route2a, route2n = splitRoute(route2)
            mta.iloc[i, 0] = route1a + route1n 
            mta.iloc[i, 1] = riders/2
            mta = pd.concat([mta, pd.DataFrame([[route1a + route2n, riders/2]], columns=['Route', 'riders'])])
        else:
            mta.iloc[i, 0] = route.split('/')[0]
    else:
        mta.iloc[i, 0] = route
mta.riders = mta.riders.astype(int)
mta.to_csv('nybus/mtaprepped.csv')
    

            

cache_dir = os.path.join(os.getcwd(), 'cartopycache')
cartopy.config['pre_existing_data_dir'] = cache_dir

crs = ccrs.AlbersEqualArea()
plat = ccrs.PlateCarree()

df = geopandas.read_file('data/bus_routes_nyc_dec2019.shp')
df = df.to_crs("EPSG:4326")
df['isSBS'] = df.route_id.apply(lambda x: x[-1] == '+')
df['route'] = df.route_id.apply(lambda x: re.sub(r'[a-zA-Z]+$', '', x.replace('+', '')))
df['color'] = df.apply(lambda x: routeToColor(x.route, x.isSBS), axis=1)

def findRidership(r):
    if mta[mta.Route == r].size > 0:
        return mta[mta.Route == r].iloc[0, 1]
    else:
        return 0

df['ridership'] = df.route.apply(findRidership)
df = df.sort_values('ridership', ascending=False)
df[['route', 'route_id', 'route_dir', 'route_shor', 'route_long', 'ridership', 'color']].to_csv('nybus/bus.csv')

print(df.iloc[1, :])
print(mta.iloc[1, :])

fig, ax = plt.subplots(subplot_kw={"projection": plat}, figsize=(10, 10))
ax.set_extent([-74.15, -73.7, 40.55, 40.92])
ax.set_facecolor('black')


for i in range(df.shape[0]):
    row = df.iloc[i]
    df.iloc[[i], -4].plot(ax=ax, 
        linewidth=row.ridership/20000 + 0.1,
        alpha = 0.5, color = row.color, #capstyle='round'
        zorder = i)
fig.tight_layout()
fig.savefig("nybus/bus.png", dpi=500)




