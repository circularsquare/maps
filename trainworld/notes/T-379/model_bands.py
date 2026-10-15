"""T-379: the market's ground departures a weekday per resident by band, for a pack's residents,
over the world field (the normaliser's own sum, binned), against NHTS by CBSA (nhts_cbsa.py).

    python -I model_bands.py <trainworld root> <pack id> [<pack id> ...]
"""
import json, math, sys, os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import numpy as np
root = sys.argv[1]
sys.path.insert(0, f"{root}/pipeline")
import world_field as W

reg = json.load(open(f"{root}/app/src/game/registry.json", encoding="utf-8"))
RULE = reg["market"]["rule"]
EDGES = np.array([0, 25, 50, 80, 100, 120, 150, 200, 250, 300, 500, 1000, 5000, 40000.0])
wf = W.load()
y_us = RULE["y_us"]
inc5 = np.minimum(wf.r5.gdp / y_us, 1)
a5 = wf.r5.pop * inc5 ** RULE["eps"]
a4 = np.add.reduceat(a5, wf.r4.start)
a3 = np.add.reduceat(a4, wf.r3.start)
R_E = 6371.0088


def bands_at(lon, lat):
    d3 = W.km(lon, lat, wf.r3.lon, wf.r3.lat)
    far = np.nonzero(d3 > W.MID_KM)[0]
    i4 = W._children(wf.r3, np.nonzero(d3 <= W.MID_KM)[0])
    d4 = W.km(lon, lat, wf.r4.lon[i4], wf.r4.lat[i4])
    mid, i5 = i4[d4 > W.NEAR_KM], W._children(wf.r4, i4[d4 <= W.NEAR_KM])
    d5 = W.km(lon, lat, wf.r5.lon[i5], wf.r5.lat[i5])
    d = np.concatenate([d5, d4[d4 > W.NEAR_KM], d3[far]])
    a = np.concatenate([a5[i5], a4[mid], a3[far]])
    c = np.concatenate([wf.r5.country[i5], wf.r4.country[mid], wf.r3.country[far]])
    us = c == c[np.argmin(d)]
    b = np.where(us, 1.0, RULE["border"])
    on = np.clip((d - RULE["onset_km"][0]) / (RULE["onset_km"][1] - RULE["onset_km"][0]), 0, 1)
    w = b * a * on * np.maximum(d, RULE["d0_km"]) ** -RULE["A"]
    h, _ = np.histogram(d, bins=EDGES, weights=w)
    return h / w.sum()


for pid in sys.argv[2:]:
    pk = json.load(open(f"{root}/data/packs/{pid}.market.json", encoding="utf-8"))
    f = pk["frame"]
    S = np.array(pk["sites"])
    nz = len(pk["zones"])
    z = S[:, 4].astype(int)
    pop = np.bincount(z, S[:, 2], nz)
    zx = np.bincount(z, S[:, 0] * S[:, 2], nz) / np.maximum(pop, 1e-9)
    zy = np.bincount(z, S[:, 1] * S[:, 2], nz) / np.maximum(pop, 1e-9)
    lon = f["lon"] + np.degrees(zx / (R_E * 1000 * math.cos(math.radians(f["lat"]))))
    lat = f["lat"] + np.degrees(zy / (R_E * 1000))
    tot = np.zeros(len(EDGES) - 1)
    for k in range(nz):
        if pop[k] > 0:
            tot += pop[k] * bands_at(lon[k], lat[k])
    rate = RULE["R"] * tot / pop.sum()
    print(f"{pid}: {pop.sum():,.0f} people; ground departures a weekday per resident by band")
    for i in range(len(EDGES) - 1):
        print(f"  {EDGES[i]:>5.0f}-{EDGES[i + 1]:<6.0f} {rate[i]:.5f}")
