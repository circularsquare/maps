"""Population per map pixel, from Kontur's global H3 r6 hexagons (2023-11-01).

Each hexagon's population is spread evenly over the map pixels it covers, so summing a
region's pixels gives its people. Hexagons too small to cover a pixel centre put their
population on the pixel under their centroid instead, so nothing is dropped.

    python asia1m/naming/pop_raster.py <kontur_population_20231101_r6.gpkg>

The source ships gzipped in religiondots/data/geo/kontur/; unpack it somewhere scratch first.
Writes pop.npy (float32, 8000x9500) next to this file.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "regionnaming"))
import numpy as np
import geopandas as gpd
from pyproj import Transformer
from rasterio.features import rasterize
from geo import TRANSFORM, PROJ4, W, H, lonlat_to_px

t0 = time.time()
src = sys.argv[1]
to3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
x0, y0 = to3857.transform(40, -12)
x1, y1 = to3857.transform(180, 80)
g = gpd.read_file(src, bbox=(x0, y0, x1, y1), columns=["population"])
print(f"[{time.time()-t0:.0f}s] read {len(g)} hexagons, total {g.population.sum()/1e9:.2f}bn")
g = g.to_crs(PROJ4).reset_index(drop=True)

hexid = rasterize(((geom, i + 1) for i, geom in enumerate(g.geometry)),
                  out_shape=(H, W), transform=TRANSFORM, fill=0, dtype=np.int32)
print(f"[{time.time()-t0:.0f}s] rasterized")
n = len(g)
counts = np.bincount(hexid.ravel(), minlength=n + 1)
pop = g.population.to_numpy(np.float64)
per_px = np.zeros(n + 1)
covered = counts[1:] > 0
per_px[1:][covered] = pop[covered] / counts[1:][covered]
out = per_px[hexid].astype(np.float32)

# hexagons that hit no pixel centre (map edge, slivers): drop onto their centroid pixel
miss = np.flatnonzero(~covered)
if len(miss):
    c = g.geometry.iloc[miss].centroid.to_crs("EPSG:4326")
    px, py = lonlat_to_px(c.x.to_numpy(), c.y.to_numpy())
    px = np.round(px).astype(int); py = np.round(py).astype(int)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    np.add.at(out, (py[ok], px[ok]), pop[miss][ok].astype(np.float32))
print(f"[{time.time()-t0:.0f}s] {len(miss)} hexagons placed by centroid; "
      f"on map {out.sum()/1e9:.2f}bn people")
np.save(os.path.join(HERE, "pop.npy"), out)
print(f"[{time.time()-t0:.0f}s] saved pop.npy")
