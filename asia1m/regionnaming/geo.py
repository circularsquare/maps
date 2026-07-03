"""Georeferencing for asia1m main layer (9500x8000)."""
import numpy as np
from affine import Affine
from cartopy import crs as ccrs
from pyproj import Transformer

W, H = 9500, 8000
# from reproduce of generateAsia.py axes
POS = [0.010366174312245553, 0.00187499999999996, 0.9896338256877545, 0.998125]
XLIM = [-5646264.06927001, 4115969.424358185]
YLIM = [-838408.2719087065, 7524984.604275155]
PROJ4 = "+ellps=WGS84 +proj=aea +lon_0=95 +lat_0=0 +x_0=0 +y_0=0 +lat_1=6 +lat_2=42 +no_defs"

left_px  = POS[0]*W
right_px = POS[2]*W
top_px   = (1-POS[3])*H
bot_px   = (1-POS[1])*H

xspan = XLIM[1]-XLIM[0]
yspan = YLIM[1]-YLIM[0]
a = xspan/(right_px-left_px)
c = XLIM[0] - left_px*a
e = -yspan/(bot_px-top_px)
f = YLIM[1] - top_px*e
# rasterio affine: (col,row)->(x,y) at pixel corner; use +0.5 for centers when sampling
TRANSFORM = Affine(a, 0, c, 0, e, f)

_to_lonlat = Transformer.from_crs(PROJ4, "EPSG:4326", always_xy=True)
_to_proj   = Transformer.from_crs("EPSG:4326", PROJ4, always_xy=True)

def px_to_lonlat(px, py):
    x, y = TRANSFORM*(np.asarray(px)+0.5, np.asarray(py)+0.5)
    return _to_lonlat.transform(x, y)

def lonlat_to_px(lon, lat):
    x, y = _to_proj.transform(lon, lat)
    col, row = ~TRANSFORM*(x, y)
    return col-0.5, row-0.5
