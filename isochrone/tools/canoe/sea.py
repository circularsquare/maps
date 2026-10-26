"""The 0.25 degree sea grid for the canoe (docs/canoe.md 4): data/canoe/sea.bin.gz.

    python tools/canoe/sea.py          # seconds; inputs from sea_fetch.py

Local summer everywhere: June-August for cells north of the equator, December-February south.
Currents: the drifter monthly climatology averaged over those months (months without data
skipped), else the annual mean, else 0. Wave height, sea ice and sea temperature: the ERA5
summer climatology (sea_fetch.py era5), the mean of the four ERA5 points round each cell centre;
cells with no ERA5 value (land, or waves under ice) take the nearest cell that has one.
Layout: notes/T-021.md.
"""
import gzip
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import xarray as xr
from scipy import ndimage

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data", "canoe")

STEP = 0.25
NORTH, SOUTH = 85.0, -60.0
H, W = int((NORTH - SOUTH) / STEP), int(360 / STEP)        # 580 x 1440, row 0 at 85N, col 0 at 180W
LAT = NORTH - STEP / 2 - STEP * np.arange(H)                 # cell centres
LON = -180 + STEP / 2 + STEP * np.arange(W)
NORTHERN = LAT > 0

CUR_STEP, HS_STEP = 0.02, 0.05                              # m/s, m per unit


def drifter():
    d = os.path.join(RAW, "gdp_drifter")

    def grid(da):
        # (month?, longitude, latitude ascending) -> our rows north to south
        a = da.sel(latitude=LAT, longitude=LON, method="nearest", tolerance=0.01).values
        return np.moveaxis(a, -1, -2)

    jja = xr.open_dataset(os.path.join(d, "monthly_jun_aug.nc"))
    djf = xr.concat([xr.open_dataset(os.path.join(d, "monthly_dec.nc")),
                     xr.open_dataset(os.path.join(d, "monthly_jan_feb.nc"))], "ClimatologicalMonth")
    ann = xr.open_dataset(os.path.join(d, "annual.nc"))
    out = {}
    for v in ("U", "V"):
        with np.errstate(invalid="ignore"), _quiet():
            s_n = np.nanmean(grid(jja[v]), axis=0)
            s_s = np.nanmean(grid(djf[v]), axis=0)
        s = np.where(NORTHERN[:, None], s_n, s_s)
        a = grid(ann[v])
        filled_annual = np.isnan(s) & np.isfinite(a)
        s = np.where(np.isnan(s), a, s)
        out[v] = s
        out[v + "_from_annual"] = filled_annual
    out["has"] = np.isfinite(out["U"]) & np.isfinite(out["V"])
    for v in ("U", "V"):
        out[v] = np.where(out["has"], out[v], 0.0)
    return out


class _quiet:
    def __enter__(self):
        import warnings
        self.w = warnings.catch_warnings()
        self.w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self.w.__exit__(*a)


def era5():
    c = np.load(os.path.join(RAW, "era5_summer", "clim.npz"))
    # ERA5 points: lat 90 - 0.25 i, lon 0.25 j (0..359.75). Our cell centre sits between four.
    i0 = np.rint((90 - (LAT + STEP / 2)) / STEP).astype(int)
    j0 = np.rint(((LON - STEP / 2) % 360) / STEP).astype(int)
    j1 = (j0 + 1) % 1440
    out = {}
    for v in ("hs", "ice", "sst"):
        f = np.where(NORTHERN[:, None], 0, 1)
        stack = []
        for s in ("jja", "djf"):
            a = c["%s_%s" % (v, s)]
            stack.append(np.stack([a[i0][:, j0], a[i0][:, j1], a[i0 + 1][:, j0], a[i0 + 1][:, j1]]))
        with _quiet():
            m = np.where(f == 0, np.nanmean(stack[0], axis=0), np.nanmean(stack[1], axis=0))
        out[v + "_has"] = np.isfinite(m)
        out[v] = fill_nearest(m)
    out["sst"] = out["sst"] - 273.15
    return out


def fill_nearest(a):
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    # tile three wide so the nearest value may lie across the 180 degree seam
    t = np.concatenate([bad, bad, bad], axis=1)
    idx = ndimage.distance_transform_edt(t, return_distances=False, return_indices=True)
    r, cidx = idx[0][:, W:2 * W], idx[1][:, W:2 * W] % W
    return a[r, cidx]


def main():
    cur = drifter()
    sea = era5()
    u8 = lambda x, step, lo, hi, dt: np.clip(np.rint(x / step), lo, hi).astype(dt)
    planes = [u8(cur["U"], CUR_STEP, -127, 127, np.int8), u8(cur["V"], CUR_STEP, -127, 127, np.int8),
              u8(sea["hs"], HS_STEP, 0, 255, np.uint8), u8(sea["ice"] * 100, 1, 0, 100, np.uint8),
              u8(sea["sst"], 1, -127, 127, np.int8)]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "sea.bin.gz")
    data = gzip.compress(b"".join(p.tobytes() for p in planes), compresslevel=9)
    with open(path + ".tmp", "wb") as fh:
        fh.write(data)
    os.replace(path + ".tmp", path)
    np.savez_compressed(os.path.join(RAW, "sea_debug.npz"), U=cur["U"], V=cur["V"], has_cur=cur["has"],
                        U_from_annual=cur["U_from_annual"], hs=sea["hs"], ice=sea["ice"], sst=sea["sst"],
                        hs_has=sea["hs_has"], ice_has=sea["ice_has"])
    ip = os.path.join(OUT, "index.json")
    idx = json.load(open(ip)) if os.path.exists(ip) else {}
    idx["sea"] = {"file": "sea.bin.gz", "west": -180.0, "north": NORTH, "step": STEP, "width": W, "height": H,
                  "bands": ["u", "v", "hs", "ice", "sst"]}
    with open(ip + ".tmp", "w") as fh:
        json.dump(idx, fh, separators=(",", ":"))
    os.replace(ip + ".tmp", ip)
    sp = np.hypot(cur["U"], cur["V"])
    print("sea.bin.gz %.2f MB; currents from drifters in %.1f%% of cells (%.1f%% from the annual mean), max %.2f m/s"
          % (len(data) / 1e6, 100 * cur["has"].mean(), 100 * cur["U_from_annual"].mean(), sp.max()))
    r, c = int((NORTH - 36) / STEP), int((180 - 73) / STEP)
    box = sp[r - 4:r + 4, c - 4:c + 4]
    print("Gulf Stream box 35-37N 72-74W: max %.2f m/s, mean %.2f" % (box.max(), box.mean()))


if __name__ == "__main__":
    main()
