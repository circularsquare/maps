"""Shared grid, paths and tables for the walking surface build (docs/surface.md)."""
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data", "surface")
LC100 = os.path.join(RAW, "lc100", "cgls-lc100-2019-cog.tif")
GMTED = os.path.join(RAW, "gmted", "mn30_grd")
MAP_WALK = os.path.join(RAW, "unzipped", "2020_walking_only_friction_surface.geotiff")

# MAP's grid: 30 arc-seconds, 85N to 60S
W, H = 43200, 17400
NORTH, WEST = 85.0, -180.0
RES = 1.0 / 120
SPEED_SCALE = 1000      # stored uint16 = km/h * 1000; 0 = no walkable land
SHARE_SCALE = 250       # stored uint8 = share * 250

# LC100 2019 discrete classes -> off-road walking km/h (research/friction.md 3).
# MAP's survey values; where no MODIS class matches, the value MAP mostly gives those pixels
# (T-009 notes). Open needleleaf and open evergreen broadleaf remain guesses.
LC_SPEED = {
    111: 3.24, 113: 3.24,            # closed needleleaf
    112: 1.62,                       # closed evergreen broadleaf
    114: 4.00,                       # closed deciduous broadleaf
    115: 3.24, 116: 3.24,            # closed mixed, unknown
    121: 4.00, 123: 4.00, 125: 4.00,  # open needleleaf, open deciduous needleleaf, open mixed
    124: 4.86, 126: 4.86,            # open deciduous broadleaf, open unknown: MODIS savannas
    122: 3.00,                       # open evergreen broadleaf
    20: 4.20,                        # shrubs: MODIS open shrublands
    30: 4.86,                        # herbaceous
    40: 2.50,                        # cultivated
    50: 5.00,                        # urban
    60: 3.00,                        # bare / sparse
    70: 1.62,                        # snow and ice
    90: 2.00,                        # herbaceous wetland
    100: 4.86,                       # moss and lichen: MODIS grasslands
}
SNOW_KMH = 1.62                      # land with no LC100 class (north of 80 N, class 0)
LC_INLAND_WATER = 80
LC_SEA = 200
LC_NODATA = (0, 255)                 # unknown, outside


def lut_pace():
    """uint8 class -> minutes per metre (0 where not walkable land)."""
    lut = np.zeros(256, np.float32)
    for k, v in LC_SPEED.items():
        lut[k] = 0.06 / v
    return lut


def elev_factor(elev_m):
    return 1.016 * np.exp(-0.0001072 * elev_m)


def slope_factor(slope_deg):
    tobler = 6.0 * np.exp(-3.5 * np.abs(np.tan(0.01745 * slope_deg) + 0.05))
    return np.minimum(tobler, 5.0) / 5.0


def profile(dtype, count=1, nodata=None):
    from rasterio.transform import from_origin
    p = dict(driver="GTiff", width=W, height=H, count=count, dtype=dtype, crs="EPSG:4326",
             transform=from_origin(WEST, NORTH, RES, RES), tiled=True, blockxsize=512,
             blockysize=512, compress="deflate", predictor=2, zlevel=6, BIGTIFF="IF_SAFER")
    if nodata is not None:
        p["nodata"] = nodata
    return p
