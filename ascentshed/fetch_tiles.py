"""Fetch + merge AWS open elevation tiles (Tilezen terrain-tiles, no auth) into a
single GeoTIFF for a lon/lat bbox. Web Mercator, int16 metres.

  python fetch_tiles.py 6.0 45.2 8.2 46.2 9 ../data/ascentshed/alps.tif

Endpoint: https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif
Land+bathy globally; for a regional land test Mercator distortion is negligible.
"""
import sys, os, math, urllib.request
import numpy as np
import rasterio
from rasterio.merge import merge

EP = "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif"


def deg2tile(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def main():
    lon0, lat0, lon1, lat1, z, out = (float(sys.argv[1]), float(sys.argv[2]),
                                      float(sys.argv[3]), float(sys.argv[4]),
                                      int(sys.argv[5]), sys.argv[6])
    x0, y1 = deg2tile(lon0, lat0, z)      # lat0 (south) -> larger y
    x1, y0 = deg2tile(lon1, lat1, z)      # lat1 (north) -> smaller y
    xs = range(min(x0, x1), max(x0, x1) + 1)
    ys = range(min(y0, y1), max(y0, y1) + 1)
    cache = os.path.join(os.path.dirname(out) or '.', 'tilecache')
    os.makedirs(cache, exist_ok=True)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    paths = []
    n = len(xs) * len(ys)
    print(f"{n} tiles  x={xs.start}..{xs.stop-1}  y={ys.start}..{ys.stop-1}  z={z}")
    for i, x in enumerate(xs):
        for y in ys:
            p = os.path.join(cache, f"{z}_{x}_{y}.tif")
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                urllib.request.urlretrieve(EP.format(z=z, x=x, y=y), p)
            paths.append(p)
        print(f"  col {i+1}/{len(xs)} done")
    srcs = [rasterio.open(p) for p in paths]
    mosaic, transform = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)
    with rasterio.open(out, 'w', **meta) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()
    a = mosaic[0]
    print(f"wrote {out}  {a.shape}  elev {int(a.min())}..{int(a.max())} m")


if __name__ == "__main__":
    main()
