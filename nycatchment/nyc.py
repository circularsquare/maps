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
from scipy.spatial import cKDTree

resolution = '10m'

# save coastline 
import cartopy.io.shapereader as shpreader
# shpreader.natural_earth(category='physical', name='land', resolution=resolution)
cache_dir = os.path.join(os.getcwd(), 'cartopycache')
cartopy.config['pre_existing_data_dir'] = cache_dir

crs = ccrs.AlbersEqualArea()
plat = ccrs.PlateCarree()

ny = geopandas.read_file('data/nyc/tl_2020_36_tabblock20.shp') # new york
nj = geopandas.read_file('data/nyc/tl_2025_34_tabblock20.shp') # nj
ct = geopandas.read_file('data/nyc/tl_2025_09_tabblock20.shp') # ct

ny_counties = ['005', '047', '061', '081', '085',  # 5 boroughs
               '119', '059', '103', '087', '079',  # Westchester, Nassau, Suffolk, Rockland, Putnam
               '071', '027', '111']                # Orange, Dutchess, Ulster
nj_counties = ['003', '013', '017', '023', '025',  # Bergen, Essex, Hudson, Middlesex, Monmouth
               '027', '031', '035', '037', '039',  # Morris, Passaic, Somerset, Sussex, Union
               '015', '021', '029', '041']         # Hunterdon, Mercer, Ocean
ct_counties = ['001', '009', '005']                 # Fairfield, New Haven, Litchfield
ny = ny[ny.COUNTYFP20.isin(ny_counties)] 
nj = nj[nj.COUNTYFP20.isin(nj_counties)]  
ct = ct[ct.COUNTYFP20.isin(ct_counties)]  

df = pd.concat([nj, ny, ct]).reset_index(drop=True)
df = df.to_crs("EPSG:4326")
df = df[df['POP20'] > 0]

df['population'] = df.POP20.astype(int)
df['INTPTLON20'] = df['INTPTLON20'].astype(float)
df['INTPTLAT20'] = df['INTPTLAT20'].astype(float)

fig, ax = plt.subplots(subplot_kw={"projection": plat}, figsize=(10, 10))
ax.set_extent([-74.926758,-72.4,40.0,41.783601])

df['density'] = df.population/(df.ALAND20+df.AWATER20+1) 
df = df.drop(['TRACTCE20', 'BLOCKCE20', 'GEOIDFQ20', 'NAME20', 'MTFCC20', 'UR20', 'UACE20', 'FUNCSTAT20'], axis=1)

print('blocks ', df.iloc[1, :])

print ('block set up complete.')

# run nyc2.py first to generate station colors!
rail = pd.read_csv('data/nyc/allrail.csv')
rail['population'] = 0

# ============================================================================
# ULTRA-FAST KDTREE APPROACH
# ============================================================================

print('Building KDTree for nearest neighbor search...')

# Maximum distance in degrees (approximately 5 miles at 41° latitude)
# 5 miles ≈ 0.073 degrees at this latitude
# Formula: miles / (69 * cos(41°)) for longitude, miles / 69 for latitude
MAX_DISTANCE_MILES = 5
MAX_DISTANCE_DEG = MAX_DISTANCE_MILES / 69.0  # Approximate degrees

# Create station coordinates with lat-scaling (cos 41°)
station_coords = np.column_stack([
    rail.Longitude.values * 0.755,  # Apply cos(41°) scaling to longitude
    rail.Latitude.values
])

# Build KDTree
tree = cKDTree(station_coords)

# Convert coordinate columns to float
df['INTPTLON20'] = df['INTPTLON20'].astype(float)
df['INTPTLAT20'] = df['INTPTLAT20'].astype(float)

# Filter land blocks
land_mask = df.ALAND20 / (df.AWATER20 + 0.001) > 1.0
land_blocks = df[land_mask].copy()

print(f'Processing {len(land_blocks)} land blocks using KDTree...')

# Get block coordinates (also with lat-scaling)
block_coords = np.column_stack([
    land_blocks.INTPTLON20.values * 0.755,
    land_blocks.INTPTLAT20.values
])

# Query all nearest neighbors at once
distances, indices = tree.query(block_coords)

# Filter out blocks that are too far from any station
within_range = distances <= MAX_DISTANCE_DEG
print(f'Blocks within {MAX_DISTANCE_MILES} miles of a station: {within_range.sum()} / {len(land_blocks)}')
print(f'Excluded {(~within_range).sum()} blocks as too far from stations')

# Assign station indices only for blocks within range
df['station'] = -1

# Now with clean sequential indices, this will work!
land_indices_within_range = land_blocks.index[within_range]
station_indices_within_range = indices[within_range]
df.loc[land_indices_within_range, 'station'] = station_indices_within_range

# Calculate station populations
print('Calculating station populations...')
station_pops = np.zeros(len(rail))
for station_idx in range(len(rail)):
    mask = df.station == station_idx
    station_pops[station_idx] = df.loc[mask, 'population'].sum()

rail['population'] = station_pops

print('station populations calculated!')

def randomizeHue(color):
    r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255);
    r, g, b = colorsys.hsv_to_rgb(
        (h + (random.random()-0.5)*0.07)%1, 
        min(max(s + (random.random()-0.5)*0.1, 0), 1), 
        v)
    hex_code = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
    return hex_code
def muteHue(hex, density):
    r, g, b = ImageColor.getcolor(hex, "RGB")
    densityFactor = ((1 / (1 + np.exp(-density*2.4)) - 0.5) * 2) * 0.91 + 0.03
    return '#{:02x}{:02x}{:02x}'.format(int(r*densityFactor), int(g*densityFactor), int(b*densityFactor))

df['color'] = df.station.apply(lambda x: rail.iloc[x].color)
df.loc[df.station == -1, 'color'] = '#000000' # 0 pop or unpopulated for other reason
df.color = df.apply(lambda x: muteHue(x.color, x.density * 30), axis=1)

print(rail.sort_values('population').tail(10))

rail = rail.sort_values('population')
rail.to_csv('data/nyc/railprocessed.csv')
print('data saved.')

df.plot(color = df.color, ax=ax)
ax.scatter(rail.Longitude, rail.Latitude, s=rail.population / 1000 * 0.12, c=rail.color, alpha=0.8, edgecolor='none')

ax.set_facecolor('black')
plt.tight_layout(pad=0)
fig.savefig("nycatchment/geopNyc.png", dpi=1400, facecolor='black')


# df['color'] = '#555555'
# df.color = df.apply(lambda x: muteHue(x.color, x.density * 30), axis=1)
# df.loc[df.ALAND20/(df.AWATER20+0.001) < 1.0, 'color'] == '#000000'
# df.plot(color = df.color, ax=ax)

# ax.set_facecolor('black')
# plt.tight_layout()
# fig.savefig("nycensus/geopNycPlus.png", dpi=500)


