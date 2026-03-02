
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
from rasterio.transform import from_bounds
from rasterio.features import rasterize as rio_rasterize


resolution = '10m'

# save coastline 
import cartopy.io.shapereader as shpreader
# shpreader.natural_earth(category='physical', name='land', resolution=resolution)
cache_dir = os.path.join(os.getcwd(), 'cartopycache')
cartopy.config['pre_existing_data_dir'] = cache_dir

# final tuple is latitude levels of correctness
crs = ccrs.AlbersEqualArea(95, 0, 0, 0, (6, 42))
plat = ccrs.PlateCarree()

fig, ax = plt.subplots(subplot_kw={"projection": crs}, figsize=(95, 80))
ax.set_extent([47.7, 129.2, -9, 65], crs=ccrs.PlateCarree())

# for plotting the distortion circles
# lons = np.arange(45, 130, 5)
# lats = np.arange(-10, 80, 5)
# ax.tissot(rad_km=40, lons=lons, lats=lats, edgecolor='gray', linewidth=1, facecolor='none')


# ax.add_feature(cfeature.NaturalEarthFeature(
#     category='cultural', name='admin_1_states_provinces',
#     scale=resolution, edgecolor='#535D8D', facecolor='none', zorder=10,
#     rasterized = True, antialiased = False, linewidth = 0.004))
# ax.add_feature(cfeature.NaturalEarthFeature(
#     category='physical', name='lakes',
#     scale=resolution, facecolor='#D9F6FF', edgecolor='#535D8D', zorder=5,
#     rasterized = True, antialiased = False, linewidth = 0.0035))
# ax.add_feature(cfeature.NaturalEarthFeature(
#     category='physical', name='ocean',
#     scale=resolution, facecolor='#D9F6FF', edgecolor='none', zorder=5,
#     rasterized = True, antialiased = False, linewidth = 0.004))
# ax.add_feature(cfeature.NaturalEarthFeature(
#     category='physical',name='coastline',
#     scale=resolution, facecolor='none', edgecolor='#535D8D', zorder=10,
#     rasterized = True, antialiased = False, linewidth = 0.004))


# cities = pd.read_csv('data/worldcities.csv', sep=',', lineterminator='\n')
# bigCities = cities[cities.population > 100000]

# points = ax.projection.transform_points(plat, bigCities.lng, bigCities.lat)
# ax.scatter(points[:, 0], points[:, 1], c='black', 
#     s=(72./fig.dpi)**2, zorder=5, marker='1',
#     rasterized=True, antialiased=False)
# ax.scatter(points[:, 0], points[:, 1], c='black', 
#     s=(72./fig.dpi)**2, zorder=5, marker='2',
#     rasterized=True, antialiased=False)


def plotShapefile(path):
    df = geopandas.read_file(path)
    if df.crs is None:
        df = df.set_crs("EPSG:4326")
    df = df.to_crs(crs)
    df.plot(ax=ax, edgecolor = '#63538d', facecolor='none', zorder = 10, 
        rasterized=True, antialiased = False, linewidth = 0.004)
def plotColor(path): 
    df = geopandas.read_file(path)
    if df.crs is None:
        df = df.set_crs("EPSG:4326")
    df = df.to_crs(crs)
    df.plot(ax=ax, edgecolor = 'none', facecolor='blue', zorder = 10, 
        rasterized=True, antialiased = False, linewidth = 0.004)
def plotColorThreshold(df_or_path, coverage_threshold=0.25, scale=3):
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


# plotColorThreshold("data/water-polygons-split-4326/water_polygons.shp")


lakes = geopandas.read_file("data/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp")
lakes = lakes[lakes['Lake_area'] >= 60]  # km²
plotColorThreshold(lakes)

#plotColorThreshold("data/ne_10m_lakes/ne_10m_ocean.shp")

#plotShapefile('data/asia1m/china/xiangzhen.shp')

#plotShapefile('data/asia1m/china/geoBoundaries-CHN-ADM3.shp')

# plotShapefile('data/asia1m/japan/jpn_admbnda_adm2_2019.shp')
#plotShapefile('data/asia1m/russia/geoBoundaries-RUS-ADM2.shp')
# plotShapefile('data/asia1m/indonesia/idn_admbnda_adm2_bps_20200401.shp')
# plotShapefile('data/asia1m/northkorea/prk_admin2.shp')
#plotShapefile('data/asia1m/northkorea/PRK_adm2.shp')
# plotShapefile('data/asia1m/taiwan/whosonfirst-data-admin-tw-macrocounty-polygon.shp')
# plotShapefile('data/asia1m/vietnam/vnm_admin1.shp')
# plotShapefile('data/asia1m/thailand/tha_admin1.shp')
# plotShapefile('data/asia1m/cambodia/khm_admin1.shp')
# plotShapefile('data/asia1m/laos/lao_admin1.shp')
# plotShapefile('data/asia1m/myanmar/mmr_admin2.shp')
# plotShapefile('data/asia1m/malaysia/mys_admin2.shp')
# plotShapefile('data/asia1m/philippines/phl_admbnda_adm2_psa_namria_20231106.shp')

# plotShapefile('data/asia1m/india/geoBoundaries-IND-ADM3.shp')
# plotShapefile('data/asia1m/bangladesh/bgd_admin2.shp')
# plotShapefile('data/asia1m/srilanka/lka_admin2.shp')
# plotShapefile('data/asia1m/pakistan/pak_admin2.shp')
# plotShapefile('data/asia1m/nepal/npl_admin2.shp')
# plotShapefile('data/asia1m/afghanistan/afg_admin2.shp')
# plotShapefile('data/asia1m/iran/irn_admin2.shp')
# plotShapefile('data/asia1m/iraq/irq_admin2.shp')
# plotShapefile('data/asia1m/turkey/tur_admin2.shp')
# plotShapefile('data/asia1m/yemen/yem_admin2.shp')
# plotShapefile('data/asia1m/yemen/omn_admin1.shp')
# plotShapefile('data/asia1m/saudiarabia/sau_admin1.shp')
# plotShapefile('data/asia1m/uae/are_admin1.shp')
# plotShapefile('data/asia1m/israel/whosonfirst-data-admin-il-region-polygon.shp')
# plotShapefile('data/asia1m/syria/syr_admin2.shp')
# plotShapefile('data/asia1m/lebanon/lbn_admin1.shp')
# plotShapefile('data/asia1m/jordan/geoBoundaries-JOR-ADM2.shp')
# plotShapefile('data/asia1m/azerbaijan/aze_admin1.shp')
# plotShapefile('data/asia1m/azerbaijan/arm_admin1.shp')
# plotShapefile('data/asia1m/azerbaijan/geo_admin1.shp')

# plotShapefile('data/asia1m/kazakhstan/kaz_admbnda_adm2_unhcr_2023.shp')
# plotShapefile('data/asia1m/uzbekistan/uzb_admbnda_adm2_2018b.shp')
# plotShapefile('data/asia1m/kyrgyzstan/kgz_admin1.shp')
# plotShapefile('data/asia1m/tajikistan/geoBoundaries-TJK-ADM1.shp')
# plotShapefile('data/asia1m/turkmenistan/geoBoundaries-TKM-ADM1.shp')
# plotShapefile('data/asia1m/mongolia/mng_admin1.shp')

# plotShapefile('data/asia1m/egypt/egy_admin2.shp')
# plotShapefile('data/asia1m/ethiopia/eth_admin2.shp')
# plotShapefile('data/asia1m/somalia/som_admin2.shp')
# plotShapefile('data/asia1m/ukraine/ukr_admin2.shp')

# url = "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_municipalities_geo_simple.json"
# sk_geo = geopandas.read_file(url)
# sk_geo = sk_geo.to_crs(crs)
# sk_geo.plot(ax=ax, edgecolor = '#63538d', facecolor='none', zorder = 10, 
#         rasterized=True, antialiased = False, linewidth = 0.004)

plt.tight_layout()
plt.savefig('asia1m/asialakesne.png')


