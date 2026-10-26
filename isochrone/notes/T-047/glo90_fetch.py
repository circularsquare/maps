"""Fetch the GLO-90 tiles the T-047 sample uses into data/raw/glo90/ (public bucket, no account).

    python isochrone/notes/T-047/glo90_fetch.py
"""
import os
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(HERE, "data", "raw", "glo90")
sys_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path)
from tiles import TILES


def name(lat, lon):
    return "Copernicus_DSM_COG_30_%s%02d_00_%s%03d_00_DEM" % ("N" if lat >= 0 else "S", abs(lat), "E" if lon >= 0 else "W", abs(lon))


os.makedirs(OUT, exist_ok=True)
for lat, lon, label, kind in TILES:
    n = name(lat, lon)
    path = os.path.join(OUT, n + ".tif")
    if os.path.exists(path):
        continue
    url = "https://copernicus-dem-90m.s3.amazonaws.com/%s/%s.tif" % (n, n)
    try:
        urllib.request.urlretrieve(url, path + ".tmp")
        os.replace(path + ".tmp", path)
        print(label, os.path.getsize(path) // 1000, "KB", flush=True)
    except Exception as e:
        print(label, "none:", e, flush=True)
