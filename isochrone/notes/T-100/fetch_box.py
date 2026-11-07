"""Read a lon/lat box of four Hydrography90m global layers by HTTP range (no full download).
    python fetch_box.py <name> <west> <south> <east> <north>
-> isochrone/data/raw/hydrography90m/box_<name>.npz: diff (int16, m above the basin outlet, MERIT Hydro
elevation), dist (int32, m to the basin outlet along the flow path), acc (float32, km2), basin (uint32)."""
import os
import sys
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import rasterio
from rasterio.windows import Window

OUT = r"C:\Users\anita\projects\maps\isochrone\data\raw\hydrography90m"
U = "/vsicurl/https://public.igb-berlin.de/public.php/webdav/global/"
ENV = dict(GDAL_HTTP_USERPWD="agciopgzXjWswF4:", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
           CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MULTIRANGE="YES",
           GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_NUM_THREADS="4", GDAL_HTTP_MAX_RETRY="5",
           GDAL_HTTP_RETRY_DELAY="3", GDAL_CACHEMAX=256)
LAYERS = {"diff": "outlet_diff_dw_basin_ovr.tif", "dist": "outlet_dist_dw_basin_ovr.tif",
          "acc": "accumulation_ovr.tif", "basin": "basin_ovr.tif"}
MIN_FREE_GB = 45


def read(name, win):
    with rasterio.Env(**ENV):
        with rasterio.open(U + LAYERS[name]) as ds:
            return ds.read(1, window=win)


def main():
    name = sys.argv[1]
    w, s, e, n = map(float, sys.argv[2:6])
    os.makedirs(OUT, exist_ok=True)
    free = shutil.disk_usage(OUT).free / 1e9
    if free < MIN_FREE_GB:
        raise SystemExit("only %.1f GB free on C:" % free)
    dst = os.path.join(OUT, "box_%s.npz" % name)
    if os.path.exists(dst):
        print("have", dst)
        return
    c0 = int(round((w + 180) * 1200))
    r0 = int(round((85 - n) * 1200))
    c1 = int(round((e + 180) * 1200))
    r1 = int(round((85 - s) * 1200))
    win = Window(c0, r0, c1 - c0, r1 - r0)
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        res = dict(zip(LAYERS, ex.map(lambda k: read(k, win), LAYERS)))
    tmp = dst + ".tmp.npz"
    np.savez_compressed(tmp, c0=c0, r0=r0, **res)
    os.replace(tmp, dst)
    print("%s %dx%d in %.0f s, %.1f MB" % (name, win.width, win.height, time.time() - t0, os.path.getsize(dst) / 1e6))


if __name__ == "__main__":
    main()
