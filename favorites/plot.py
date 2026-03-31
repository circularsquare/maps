import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import cartopy.io.shapereader as shpreader
import osmnx as ox
from pyproj import CRS, Transformer
from shapely.ops import transform as shapely_transform

# --- config ---
LAT0, LON0 = 40.720908, -73.946968
R0_M = 10000
K    = 4   # log scale factor: larger = slower compression, more display space per octave
USE_OSM = False  # set True for detailed NYC coastline (slow first run)

# --- log radial transform ---
def log_r(r):
    return np.where(r <= R0_M, r, R0_M * (1 + K * np.log(np.maximum(r / R0_M, 1e-10))))

# autoscale display radius to fit halfway around the earth
HALF_EARTH_M  = np.pi * 6_371_000
DISPLAY_RADIUS = float(log_r(HALF_EARTH_M))
print(f"Display radius: {DISPLAY_RADIUS/1000:.1f} km (fits {HALF_EARTH_M/1000:.0f} km actual)")

# --- projection setup ---
aeqd_crs = CRS(f"+proj=aeqd +lat_0={LAT0} +lon_0={LON0} +units=m +ellps=WGS84")
to_aeqd = Transformer.from_crs(CRS("EPSG:4326"), aeqd_crs, always_xy=True)

def log_transform_coords(xs, ys, zs=None):
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    r = np.sqrt(xs**2 + ys**2)
    scale = np.where(r > 1e-6, log_r(r) / r, 1.0)
    return xs * scale, ys * scale

def log_transform_geom(geom):
    return shapely_transform(log_transform_coords, geom)

def load_ne(name, category='physical', resolution='10m'):
    gdf = gpd.read_file(shpreader.natural_earth(resolution=resolution, category=category, name=name))
    gdf = gdf.to_crs(aeqd_crs)
    gdf['geometry'] = gdf['geometry'].apply(log_transform_geom)
    return gdf

def load_osm(gdf):
    gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    gdf = gdf.to_crs(aeqd_crs)
    gdf['geometry'] = gdf['geometry'].apply(log_transform_geom)
    return gdf

# --- load global basemap (Natural Earth 10m) ---
print("Loading Natural Earth...")
ocean  = load_ne('ocean')
land   = load_ne('land')
lakes  = load_ne('lakes')
states = load_ne('admin_1_states_provinces', category='cultural')

if USE_OSM:
    print("Fetching OSM water features (may take a moment)...")
    water_osm = ox.features_from_place("New York City, New York, USA", tags={"natural": "water"})
    water_osm = load_osm(water_osm)

# --- transform favorites ---
df = pd.read_csv('coords.csv').dropna(subset=['lat', 'lon'])
x_ae, y_ae = to_aeqd.transform(df['lon'].values, df['lat'].values)
x_log, y_log = log_transform_coords(x_ae, y_ae)

# --- plot ---
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect('equal')
ax.set_xlim(-DISPLAY_RADIUS, DISPLAY_RADIUS)
ax.set_ylim(-DISPLAY_RADIUS, DISPLAY_RADIUS)
ax.set_axis_off()
fig.patch.set_facecolor('white')

ocean.plot (ax=ax, facecolor='#c9dff0', edgecolor='none', zorder=1)
land.plot  (ax=ax, facecolor='#f0ece4', edgecolor='none', zorder=2)
lakes.plot (ax=ax, facecolor='#c9dff0', edgecolor='none', zorder=3)
states.plot(ax=ax, facecolor='none',    edgecolor='#b0a898', linewidth=0.4, zorder=4)
load_ne('coastline').plot(ax=ax, edgecolor='#8a9aaa', linewidth=0.4, zorder=5)

if USE_OSM:
    water_osm.plot(ax=ax, facecolor='#c9dff0', edgecolor='none', zorder=6)

# favorites
ax.scatter(x_log, y_log,
           s=18, color='#5b9bd5', alpha=0.6,
           edgecolors='#2c5f8a', linewidths=0.5, zorder=10)

# clip to circle
clip_circle = mpatches.Circle((0, 0), DISPLAY_RADIUS, transform=ax.transData)
for coll in ax.collections:
    coll.set_clip_path(clip_circle)
ax.add_patch(mpatches.Circle((0, 0), DISPLAY_RADIUS,
             fill=False, edgecolor='#8a9aaa', linewidth=0.8, zorder=11))

plt.tight_layout()
plt.savefig('map' + str(K) + '.png', dpi=200, bbox_inches='tight')
print(f"Saved map.png ({len(df)} points)")
