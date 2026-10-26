"""Fetch the sea inputs for sea.py (docs/canoe.md 4): drifter currents and an ERA5 summer climatology.

  python tools/canoe/sea_fetch.py drifter      # ~100 MB of ERDDAP .nc into data/raw/gdp_drifter/
  python tools/canoe/sea_fetch.py era5 [threads]

drifter: the NOAA Global Drifter Program climatology (Laurindo et al. 2017, CC BY 4.0) from AOML
ERDDAP: monthly U, V for June-August and December-February, and the annual U, V, N.

era5: streams ARCO-ERA5 (Google public bucket, ERA5 is CC BY 4.0) one hour at a time and keeps only
running sums, so nothing hourly is stored. Years 1991-2020; one hour every 73 h (so the hour of day
cycles) for wave height, every third of those for sea ice and sea temperature; only hours in June-
August or December-February. Output data/raw/era5_summer/clim.npz: per season (jja, djf) the mean
of each variable on ERA5's 721 x 1440 grid (90N..90S, 0..359.75E), NaN where never valid, and the
sample counts.
"""
import datetime
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import requests

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(HERE, "data", "raw")
ERDDAP = "https://erddap.aoml.noaa.gov/gdp/erddap/griddap/"
ARCO = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/"
EPOCH = datetime.datetime(1940, 1, 1)       # ARCO time index 0
VARS = {"hs": "significant_height_of_combined_wind_waves_and_swell",
        "ice": "sea_ice_cover", "sst": "sea_surface_temperature"}
SEASON = {6: "jja", 7: "jja", 8: "jja", 12: "djf", 1: "djf", 2: "djf"}
STRIDE_H = 73
SLOW_EVERY = 3                              # ice and sst on every third wave sample
ICE_STRIDE_H = 49


def get(url, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, timeout=180)
            if r.status_code == 200:
                return r.content
            if r.status_code in (403, 404):
                raise RuntimeError("%d for %s" % (r.status_code, url))
        except requests.RequestException as e:
            if k == tries - 1:
                raise
            print("retry", url, e, flush=True)
        time.sleep(2 + 3 * k)
    raise RuntimeError("failed " + url)


def drifter():
    out = os.path.join(RAW, "gdp_drifter")
    os.makedirs(out, exist_ok=True)
    jobs = {"monthly_jun_aug.nc": "drifter_monthlymeans.nc?U[5:7][0:1439][0:631],V[5:7][0:1439][0:631]",
            "monthly_jan_feb.nc": "drifter_monthlymeans.nc?U[0:1][0:1439][0:631],V[0:1][0:1439][0:631]",
            "monthly_dec.nc": "drifter_monthlymeans.nc?U[11:11][0:1439][0:631],V[11:11][0:1439][0:631]",
            "annual.nc": "drifter_annualmeans.nc?U[0:1439][0:631],V[0:1439][0:631],N[0:1439][0:631]"}
    for name, q in jobs.items():
        path = os.path.join(out, name)
        if os.path.exists(path):
            continue
        t0 = time.time()
        data = get(ERDDAP + q)
        with open(path + ".tmp", "wb") as fh:
            fh.write(data)
        os.replace(path + ".tmp", path)
        print("%s %.1f MB %.0f s" % (name, len(data) / 1e6, time.time() - t0), flush=True)


def era5(threads=4, ice_only=False):
    """ice_only: sea ice alone for the least-ice months (aug, feb), one hour every 49 h,
    to data/raw/era5_summer/clim_ice.npz."""
    from numcodecs import Blosc
    codec = Blosc()
    out = os.path.join(RAW, "era5_summer")
    os.makedirs(out, exist_ok=True)
    t_start = int((datetime.datetime(1991, 1, 1) - EPOCH).total_seconds() // 3600)
    t_end = int((datetime.datetime(2021, 1, 1) - EPOCH).total_seconds() // 3600)
    seasons = {8: "aug", 2: "feb"} if ice_only else SEASON
    names = sorted(set(seasons.values()))
    jobs = []                                   # (season, var, hour index)
    n = 0
    for t in range(t_start, t_end, ICE_STRIDE_H if ice_only else STRIDE_H):
        season = seasons.get((EPOCH + datetime.timedelta(hours=t)).month)
        if not season:
            continue
        if ice_only:
            jobs.append((season, "ice", t))
            continue
        jobs.append((season, "hs", t))
        if n % SLOW_EVERY == 0:
            jobs += [(season, "ice", t), (season, "sst", t)]
        n += 1
    sums = {(s, v): np.zeros((721, 1440), np.float64) for s in names for v in VARS}
    cnts = {(s, v): np.zeros((721, 1440), np.int32) for s in names for v in VARS}
    lock = threading.Lock()
    done = [0, 0]
    t0 = time.time()

    def work(job):
        season, v, t = job
        raw = get("%s%s/%d.0.0" % (ARCO, VARS[v], t))
        a = np.frombuffer(codec.decode(raw), dtype="<f4").reshape(721, 1440)
        ok = np.isfinite(a)
        with lock:
            sums[season, v][ok] += a[ok]
            cnts[season, v] += ok
            done[0] += 1
            done[1] += len(raw)
            if done[0] % 100 == 0:
                print("%d / %d chunks, %.2f GB streamed, %.0f s" % (done[0], len(jobs), done[1] / 1e9, time.time() - t0), flush=True)

    print("%d chunks to stream" % len(jobs), flush=True)
    with ThreadPoolExecutor(threads) as ex:
        for f in [ex.submit(work, j) for j in jobs]:
            f.result()
    res = {}
    for (s, v), sm in sums.items():
        c = cnts[s, v]
        if not c.any():
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            res["%s_%s" % (v, s)] = np.where(c > 0, sm / np.maximum(c, 1), np.nan).astype(np.float32)
        res["n_%s_%s" % (v, s)] = c
    name = "clim_ice.npz" if ice_only else "clim.npz"
    np.savez_compressed(os.path.join(out, name + ".tmp.npz"), **res)
    os.replace(os.path.join(out, name + ".tmp.npz"), os.path.join(out, name))
    print("done: %d chunks, %.2f GB streamed, %.0f s" % (done[0], done[1] / 1e9, time.time() - t0), flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "drifter":
        drifter()
    elif what == "era5":
        era5(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    elif what == "era5ice":
        era5(int(sys.argv[2]) if len(sys.argv) > 2 else 4, ice_only=True)
