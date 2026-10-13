"""T-276: fit the regional market's trip rate and distance curve to NHTS 2017 (nhts_long.json).

    python -I market_fit.py <intercity.json> <outside.json> <nhts_long.json>

The market (sim/src/demand/intercity.rs) gives each resident R departures a weekday, shared among
every represented zone and the outside world (pipeline/outside.py) in proportion to attraction x
f(d), f(d) = onset(d) x exp(-d / L) / max(d, D0)^A, onset 0 at ON0 km rising linearly to 1 at ON1.
Here the zones are rebuilt from the pack's sites as the game builds them, distances are great
circles, and for every zone the share of its departures in each NHTS band follows; the model's rate
per band is R x the population-weighted mean share. Fitted on Northeast residents' weekday
non-commute trips of 50-1,000 km by ground (air trips taken out), least squares on log rates.
"""
import json, math, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
from scipy.optimize import minimize

pack = json.load(open(sys.argv[1], encoding="utf-8"))
outside = json.load(open(sys.argv[2], encoding="utf-8"))
nhts = json.load(open(sys.argv[3], encoding="utf-8"))
R_KM = 6371.0088
lon0, lat0 = pack["frame"]["lon"], pack["frame"]["lat"]
ZM = pack.get("zone_m", 6000)

sites = np.array(pack["sites"], dtype=float)
x, y, pop, jobs, m = sites.T
attr = pop + 2 * jobs
keys = np.stack([np.floor(x / ZM), np.floor(y / ZM), m], 1)
uk, inv = np.unique(keys, axis=0, return_inverse=True)
inv = inv.ravel()
nz = len(uk)
w = pop + attr
zx = np.bincount(inv, x * w, nz) / np.bincount(inv, w, nz)
zy = np.bincount(inv, y * w, nz) / np.bincount(inv, w, nz)
zpop = np.bincount(inv, pop, nz)
zattr = np.bincount(inv, attr, nz)
R_E = 6371008.8
zlat = lat0 + np.degrees(zy / R_E)
zlon = lon0 + np.degrees(zx / (R_E * math.cos(math.radians(lat0))))


def unit(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


alpha = attr.sum() / pop.sum()
op = np.array(outside["points"], dtype=float)
uz = unit(zlon, zlat)
uo = unit(op[:, 0], op[:, 1])
dests_u = np.concatenate([uz, uo])
dests_a = np.concatenate([zattr, alpha * op[:, 2]])
is_zone = np.concatenate([np.ones(nz, bool), np.zeros(len(op), bool)])
print(f"{nz} zones, {zpop.sum() / 1e6:.1f}M people, {len(op)} outside points ({op[:, 2].sum() / 1e6:.0f}M people), alpha {alpha:.3f}")

# each zone's destinations' attraction by distance, in 0.5 km bins (the self pair left out), apart
# for represented zones and the outside world: every evaluation is then a product with f per bin
BIN, NB = 0.5, 10000
H_zone = np.zeros((nz, NB))
H_out = np.zeros((nz, NB))
for a in range(0, nz, 128):
    c = np.linalg.norm(uz[a:a + 128, None, :] - dests_u[None, :, :], axis=2)
    d = 2 * R_KM * np.arcsin(np.clip(c / 2, 0, 1))
    b = np.minimum((d / BIN).astype(int), NB - 1)
    for i in range(d.shape[0]):
        z = a + i
        wz = dests_a.copy()
        wz[z] = 0
        H_zone[z] = np.bincount(b[i][is_zone], wz[is_zone], NB)
        H_out[z] = np.bincount(b[i][~is_zone], wz[~is_zone], NB)
dmid = (np.arange(NB) + 0.5) * BIN

edges = np.array(nhts["edges_km"], float)
ne = nhts["northeast"]
rate = np.array(ne["noncommute"]["rate"]) - np.array(ne["noncommute_air"]["rate"])
FIT = (edges[:-1] >= 50) & (edges[1:] <= 1000)
bin_band = np.digitize(dmid, edges) - 1


def model(params, on0=25.0, on1=50.0, d0=25.0, detail=False):
    L, A = params
    onset = np.clip((dmid - on0) / (on1 - on0), 0, 1)
    f = onset * np.exp(-dmid / L) / np.maximum(dmid, d0) ** A
    Wz, Wo = H_zone * f, H_out * f
    tot = Wz.sum(1) + Wo.sum(1)
    W = (Wz + Wo) / np.maximum(tot, 1e-300)[:, None]
    per_bin = (zpop[:, None] * W).sum(0) / zpop.sum()
    mean_share = np.bincount(np.clip(bin_band, 0, len(edges) - 2), per_bin, len(edges) - 1)
    if not detail:
        return mean_share
    out_share = (zpop * Wo.sum(1) / np.maximum(tot, 1e-300)).sum() / zpop.sum()
    return mean_share, out_share


def loss(p):
    L, A = p
    if L <= 20 or A < 0 or A > 3:
        return 1e9
    s = model(p)
    # the best rate for these shapes, in closed form on logs
    lr = np.log(rate[FIT]) - np.log(np.maximum(s[FIT], 1e-30))
    R = math.exp(lr.mean())
    return float(((lr - math.log(R)) ** 2).sum())


best = None
for L0 in (150, 500):
    for A0 in (0.7, 1.4):
        r = minimize(loss, [L0, A0], method="Nelder-Mead", options={"xatol": 1, "fatol": 1e-5, "maxiter": 200})
        if best is None or r.fun < best.fun:
            best = r
L, A = best.x
s, out_share = model(best.x, detail=True)
R = math.exp((np.log(rate[FIT]) - np.log(s[FIT])).mean())
print(f"fit: L {L:.1f} km, A {A:.3f}, R {R:.4f} departures a weekday per resident; loss {best.fun:.3f}; share leaving the map {out_share:.3f}")
for k in range(len(edges) - 1):
    if edges[k] >= 20 and edges[k] < 2500:
        print(f"  {edges[k]:>5.0f}-{edges[k + 1]:<5.0f} km  NHTS ground {rate[k]:.5f}  model {R * s[k]:.5f}  x{R * s[k] / max(rate[k], 1e-9):.2f}{'  (fitted)' if FIT[k] else ''}")
# the same with the old formula for comparison: onset 40-80, L 180, A 1.2, d0 40, R 0.015, no outside world
old = model((180.0, 1.2), 40, 80, 40)
print("old formula, rate 0.015, with the outside world in its normaliser:")
for k in range(len(edges) - 1):
    if edges[k] >= 20 and edges[k] < 2500:
        print(f"  {edges[k]:>5.0f}-{edges[k + 1]:<5.0f} km  model {0.015 * old[k]:.5f}")
json.dump({"L_km": L, "A": A, "R": R, "onset_km": [25, 50], "d0_km": 25, "share_leaving": out_share,
           "bands": edges.tolist(), "nhts_ground": rate.tolist(), "model": (R * s).tolist()},
          open(os.path.join(os.path.dirname(sys.argv[3]), "market_fit.json"), "w"), indent=1)
