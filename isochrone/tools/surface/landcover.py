"""LC100 2019 (100 m) -> land cover layers on MAP's 30" grid.

Outputs in data/surface/:
  lc_speed.tif  uint16 km/h * 1000 of the land part of each 1 km pixel: the area-weighted
                mean of time per metre over its 100 m land pixels (water and sea left out),
                turned back into a speed. 0 = no land.
  lc_water.tif  uint8 x 2 bands, share of the pixel * 250: band 1 inland water (class 80),
                band 2 open sea (class 200). Wetland (90) is land.
  lc_class.tif  uint8 LC100 class at the pixel centre (255 = none), for checks.

LC100 cells are 1/1008 degree, the grid 1/120: 8.4 to one, so pixels are split by area
with exact overlap weights. North of 80 N LC100 has no data; those rows stay 0 / 255.
Run: python tools/surface/landcover.py [workers]   (default 4; ~28 min on 4)
"""
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

N_IN, N_OUT = 1008, 120          # per degree
CH = 20                          # degrees of longitude per chunk
ROW0 = int((C.NORTH - 80.0) * 120)   # grid row of 80 N
WORKERS = 4


def weights(n_deg):
    """Sparse (n_deg*120, n_deg*1008) area-overlap matrix, rows sum to 1."""
    rows, cols, vals = [], [], []
    ratio = N_IN / N_OUT
    for j in range(n_deg * N_OUT):
        a, b = j * ratio, (j + 1) * ratio
        for i in range(int(np.floor(a)), int(np.ceil(b))):
            w = min(i + 1, b) - max(i, a)
            if w > 1e-9:
                rows.append(j)
                cols.append(i)
                vals.append(w / ratio)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_deg * N_OUT, n_deg * N_IN))


def centre_idx(n_deg):
    j = np.arange(n_deg * N_OUT)
    return np.floor((j + 0.5) * N_IN / N_OUT).astype(np.int64)


_WR = _WCT = _CI = _LUT = None


def init():
    global _WR, _WCT, _CI, _LUT
    _WR = weights(1)
    _WCT = weights(CH).T.tocsc()
    _CI = centre_idx(CH)
    _LUT = C.lut_pace()


def agg(x):
    # (1008, CH*1008) -> (120, CH*120)
    y = _WR @ x
    return np.asarray(y @ _WCT)


def do_row(d):
    """One degree row (d = 0 at 80 N). Returns arrays of shape (120, 43200)."""
    speed = np.zeros((N_OUT, C.W), np.uint16)
    water = np.zeros((2, N_OUT, C.W), np.uint8)
    cls = np.full((N_OUT, C.W), 255, np.uint8)
    lut_land = (_LUT > 0).astype(np.float32)
    with rasterio.open(C.LC100) as ds:
        for c in range(0, 360, CH):
            a = ds.read(1, window=Window(c * N_IN, d * N_IN, CH * N_IN, N_IN))
            pace = agg(_LUT[a])
            land = agg(lut_land[a])
            inland = agg((a == C.LC_INLAND_WATER).astype(np.float32))
            sea = agg((a == C.LC_SEA).astype(np.float32))
            valid = agg(((a != 0) & (a != 255)).astype(np.float32))
            sl = slice(c * N_OUT, (c + CH) * N_OUT)
            with np.errstate(divide="ignore", invalid="ignore"):
                kmh = 0.06 * land / pace
                s = np.where(land > 1e-6, np.rint(kmh * C.SPEED_SCALE), 0)
                speed[:, sl] = np.clip(np.nan_to_num(s), 0, 65535).astype(np.uint16)
                for k, x in enumerate((inland, sea)):
                    sh = np.where(valid > 1e-6, x / valid, 0)
                    water[k, :, sl] = np.rint(np.clip(sh, 0, 1) * C.SHARE_SCALE).astype(np.uint8)
            cls[:, sl] = a[_CI[:N_OUT]][:, _CI]
    return d, speed, water, cls


def main():
    from multiprocessing import Pool
    t0 = time.time()
    speed = np.zeros((C.H, C.W), np.uint16)
    water = np.zeros((2, C.H, C.W), np.uint8)
    cls = np.full((C.H, C.W), 255, np.uint8)
    n_rows = 140
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else WORKERS
    with Pool(workers, initializer=init) as pool:
        for k, (d, s, w, c) in enumerate(pool.imap_unordered(do_row, range(n_rows))):
            r = ROW0 + d * N_OUT
            speed[r:r + N_OUT] = s
            water[:, r:r + N_OUT] = w
            cls[r:r + N_OUT] = c
            print("degree row %d done (%d/%d) %.0f s" % (d, k + 1, n_rows, time.time() - t0),
                  flush=True)
    os.makedirs(C.OUT, exist_ok=True)
    for name, arr, dt, cnt in (("lc_speed.tif", speed, "uint16", 1),
                               ("lc_water.tif", water, "uint8", 2),
                               ("lc_class.tif", cls, "uint8", 1)):
        path = os.path.join(C.OUT, name)
        with rasterio.open(path, "w", **C.profile(dt, cnt)) as out:
            if cnt == 1:
                out.write(arr, 1)
            else:
                out.write(arr)
        print("wrote", path, os.path.getsize(path), flush=True)
    print("total %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
