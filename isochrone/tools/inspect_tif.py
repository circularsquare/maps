"""Print the basic facts of the walking-only friction surface."""
import numpy as np
import rasterio

SRC = r"C:\Users\anita\projects\maps\isochrone\data\raw\unzipped\2020_walking_only_friction_surface.geotiff"

with rasterio.open(SRC) as ds:
    a = ds.read(1, out_shape=(ds.height // 10, ds.width // 10))
    # values are minutes per metre, so km/h = 0.06 / f
    print("share exactly 0.06:", (a == np.float32(0.06)).mean())
    vals, counts = np.unique(np.round(a, 4), return_counts=True)
    order = np.argsort(-counts)[:15]
    for i in order:
        print("f=%.4f  kmh=%.2f  share=%.4f" % (vals[i], 0.06 / vals[i], counts[i] / a.size))
    land = a != np.float32(0.06)
    print("land-ish share", land.mean())
    print("land kmh percentiles 1,10,50,90,99:", np.percentile(0.06 / a[land], [1, 10, 50, 90, 99]))
    # a few probes: Sahara (23N, 12E), Paris, Pacific
    for name, lon, lat in [("Sahara", 12, 23), ("Paris", 2.35, 48.85), ("Pacific", -150, 0),
                           ("Himalaya", 86.9, 28.0), ("Amazon", -62, -4), ("Nile Cairo", 31.2, 30.0)]:
        r, c = ds.index(lon, lat)
        print(name, ds.read(1, window=((r, r + 1), (c, c + 1)))[0, 0])
