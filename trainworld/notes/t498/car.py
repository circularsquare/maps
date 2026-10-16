"""T-498: real car minutes between a pack's 2 km zones, the game's drive model (kernel.rs drive_min_f,
place_min_per_km) on the pack's routed factors (drive_near / drive_far), plus zone positions.
    python car.py <packdir> <city> <out.npz>
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json, sys
import numpy as np

packdir, city, out = sys.argv[1:4]
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
from read_pack import load_pack  # noqa

h, a = load_pack(city, packdir)
g = h["gravity"]
zm = g["zone_m"]
(x0, y0) = g["grid_origin_m"]
GW, GH = g["grid"]
gx = np.asarray(a["zone_gx"], np.int64); gy = np.asarray(a["zone_gy"], np.int64)
Z = len(gx)
reg = None
dr = h["routes"]["drive"]
W, nreg = dr["window"], dr["regions"]
assert nreg == 1, "prototype: one region"
# zone positions: grid centres (what the gravity's distance uses) and commuter-weighted centroids
cx = x0 + (gx + 0.5) * zm; cy = y0 + (gy + 0.5) * zm
zone = np.asarray(a["zone"], np.int64)
wt = np.asarray(a["commute_home"], float) + np.asarray(a["commute_work"], float) + 1e-6
X, Y = np.asarray(a["x_m"], float), np.asarray(a["y_m"], float)
sw = np.bincount(zone, wt, Z)
mx = np.bincount(zone, wt * X, Z) / sw; my = np.bincount(zone, wt * Y, Z) / sw
ncell = np.bincount(zone, minlength=Z)
dens = np.bincount(zone, wt, Z) / (np.maximum(ncell, 1) * 0.1053)
# frame check: local equirectangular about the origin
lon0, lat0 = h["origin"]["lon"], h["origin"]["lat"]
R = 6371008.8
import h3
k = np.arange(0, len(X), max(1, len(X) // 50))
ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in np.asarray(a["h3"])[k]])
lon_e = lon0 + np.degrees(X[k] / (R * np.cos(np.radians(lat0)))); lat_e = lat0 + np.degrees(Y[k] / R)
print("frame check, max error m:", float(np.max(np.hypot((lon_e - ll[:, 1]) * 111320 * np.cos(np.radians(lat0)), (lat_e - ll[:, 0]) * 111320))))

# decode drive_near
near = np.asarray(a["drive_near"], np.uint8)
exists = np.full((GW, GH), -1, np.int64); exists[gx, gy] = np.arange(Z)
F = np.zeros((Z, Z), np.float32)
kk = 0
for i in range(Z):
    for dx in range(-W, W + 1):
        X2 = gx[i] + dx
        if X2 < 0 or X2 >= GW:
            continue
        for dy in range(-W, W + 1):
            Y2 = gy[i] + dy
            if Y2 < 0 or Y2 >= GH:
                continue
            j = exists[X2, Y2]
            if j >= 0:
                F[i, j] = np.exp(near[kk] / dr["q"]); kk += 1
assert kk == len(near), (kk, len(near))
sup = np.asarray(a["drive_super"], np.int64)
far = np.asarray(a["drive_far"], np.uint8).reshape(dr["supercells"], dr["supercells"])
farF = np.exp(far[sup][:, sup] / dr["q"]).astype(np.float32)
inwin = (np.abs(gx[:, None] - gx[None, :]) <= W) & (np.abs(gy[:, None] - gy[None, :]) <= W)
F = np.where(inwin, F, farF) * (1.3 / 1.210)
# the game's drive (kernel.rs), straight distance between grid centres, intrazonal 0.52 x side
d = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
d[np.arange(Z), np.arange(Z)] = 0.52 * zm
ff = 1.0 / (1.0 + dens / 5000.0)
loc = 60.0 / (15.0 + (40.0 - 15.0) * ff); main = 60.0 / (15.0 + (80.0 - 15.0) * ff)
road = d * 0.001 * F
e = np.minimum(0.5 * road, 2.0)
mn = road - 2 * e
kq = np.minimum(0.5 * mn, 8.0)
car = e * (loc[:, None] + loc[None, :]) + kq * (main[:, None] + main[None, :]) + (mn - 2 * kq) * 0.6 + 3.0  # + 3 min parking
car = car.astype(np.float32)
lonz = lon0 + np.degrees(mx / (R * np.cos(np.radians(lat0)))); latz = lat0 + np.degrees(my / R)
np.savez(out, sup=sup, win=W, car=car, F=F, cx=cx, cy=cy, mx=mx, my=my, lon=lonz, lat=latz, dens=dens, lon0=lon0, lat0=lat0)
print("zones", Z, "median routed factor", float(np.median(F)), "car min: median", float(np.median(car)))
