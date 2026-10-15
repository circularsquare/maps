"""T-379 experiment: the ground curve with nearby destinations competing in the normaliser.

Today a zone's ground departures are shared by N = sum a b on(d) max(d,25)^-A over the world field,
so every destination under the 25 km onset is invisible: a resident of Manhattan, with 12M people
within 25 km, chooses among long destinations as if they were not there. Here the normaliser also
counts the near ones at kappa (1 - on(d)) max(d,25)^-A (they are the local layer's trips, so the
market's trips keep on(d)); kappa 0 is today's rule.

Fitted, for each kappa, R and A to NHTS 2017 Northeast ground non-commute bands 50-1,000 km (as
research/intercity/market_fit.py), over the Northeast packs' 6 km zones against the world field;
reported against NHTS by household CBSA (nhts_cbsa.json) for New York and Philadelphia.

    python -I fit_compete.py <trainworld root> <nhts_cbsa.json> <out.npz cache>
"""
import json, math, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np
from scipy.optimize import minimize
root, cbsa_path, cache = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, f"{root}/pipeline")
from shapely.geometry import shape, Point
from shapely.strtree import STRtree

reg = json.load(open(f"{root}/app/src/game/registry.json", encoding="utf-8"))
RULE = reg["market"]["rule"]
NE = [c["id"] for c in reg["cities"] if c.get("market", -1) >= 0]
R_E = 6371.0088
BIN, NB = 0.5, 4000  # to 2,000 km; beyond in the last bin
NY = {"36005", "36047", "36061", "36081", "36085", "36059", "36103", "36119", "36087", "36079", "36027", "36071",
      "34003", "34013", "34017", "34019", "34023", "34025", "34027", "34029", "34031", "34035", "34037", "34039", "42103"}
PH = {"42017", "42029", "42045", "42091", "42101", "34005", "34007", "34015", "34033", "10003", "24015"}

if os.path.exists(cache):
    z = np.load(cache)
    H, zpop, zlon, zlat, zplace = z["H"], z["zpop"], z["zlon"], z["zlat"], z["zplace"]
    zcty = list(z["zcty"])
else:
    import world_field as W
    wf = W.load()
    inc5 = np.minimum(wf.r5.gdp / RULE["y_us"], 1)
    a5 = wf.r5.pop * inc5 ** RULE["eps"]
    a4 = np.add.reduceat(a5, wf.r4.start)
    a3 = np.add.reduceat(a4, wf.r3.start)
    counties = json.load(open(f"{root}/data/work/tiger_counties_simple.json"))
    keep = {k: shape(v) for k, v in counties.items() if k[:2] in ("09", "10", "11", "24", "25", "33", "34", "36", "42", "44", "50", "51", "54", "23")}
    cids = list(keep)
    tree = STRtree([keep[c] for c in cids])

    def county_of(lon, lat):
        p = Point(lon, lat)
        for i in tree.query(p):
            if keep[cids[i]].contains(p):
                return cids[i]
        return cids[tree.nearest(p)]

    Hs, P, LO, LA, PL, CT = [], [], [], [], [], []
    for pid in NE:
        pk = json.load(open(f"{root}/data/packs/{pid}.market.json", encoding="utf-8"))
        f = pk["frame"]
        S = np.array(pk["sites"])
        nz = len(pk["zones"])
        zi = S[:, 4].astype(int)
        pop = np.bincount(zi, S[:, 2], nz)
        zx = np.bincount(zi, S[:, 0] * S[:, 2], nz) / np.maximum(pop, 1e-9)
        zy = np.bincount(zi, S[:, 1] * S[:, 2], nz) / np.maximum(pop, 1e-9)
        lon = f["lon"] + np.degrees(zx / (R_E * 1000 * math.cos(math.radians(f["lat"]))))
        lat = f["lat"] + np.degrees(zy / (R_E * 1000))
        for k in range(nz):
            if pop[k] <= 0:
                continue
            d3 = W.km(lon[k], lat[k], wf.r3.lon, wf.r3.lat)
            far = np.nonzero(d3 > W.MID_KM)[0]
            i4 = W._children(wf.r3, np.nonzero(d3 <= W.MID_KM)[0])
            d4 = W.km(lon[k], lat[k], wf.r4.lon[i4], wf.r4.lat[i4])
            mid, i5 = i4[d4 > W.NEAR_KM], W._children(wf.r4, i4[d4 <= W.NEAR_KM])
            d5 = W.km(lon[k], lat[k], wf.r5.lon[i5], wf.r5.lat[i5])
            d = np.concatenate([d5, d4[d4 > W.NEAR_KM], d3[far]])
            a = np.concatenate([a5[i5], a4[mid], a3[far]])
            c = np.concatenate([wf.r5.country[i5], wf.r4.country[mid], wf.r3.country[far]])
            b = np.where(c == c[np.argmin(d)], 1.0, RULE["border"])
            Hs.append(np.bincount(np.minimum((d / BIN).astype(int), NB - 1), b * a, NB))
            P.append(pop[k]); LO.append(lon[k]); LA.append(lat[k]); PL.append(pk["place"]); CT.append(county_of(lon[k], lat[k]))
        print(pid, nz, flush=True)
    H, zpop, zlon, zlat, zplace, zcty = np.array(Hs), np.array(P), np.array(LO), np.array(LA), np.array(PL), CT
    np.savez(cache, H=H, zpop=zpop, zlon=zlon, zlat=zlat, zplace=zplace, zcty=np.array(zcty))

nhts = json.load(open(f"{root}/research/intercity/nhts_long.json"))
edges = np.array(nhts["edges_km"], float)
ne = nhts["northeast"]
rate = np.array(ne["noncommute"]["rate"]) - np.array(ne["noncommute_air"]["rate"])
FIT = (edges[:-1] >= 50) & (edges[1:] <= 1000)
dmid = (np.arange(NB) + 0.5) * BIN
band = np.clip(np.digitize(dmid, edges) - 1, 0, len(edges) - 2)
cb = json.load(open(cbsa_path))
CE = np.array(cb["edges_km"], float)
cband = np.clip(np.digitize(dmid, CE) - 1, 0, len(CE) - 2)
isNY = np.array([c in NY for c in zcty])
isPH = np.array([c in PH for c in zcty])
print(f"{len(zpop)} zones, {zpop.sum() / 1e6:.1f}M; NY MSA {zpop[isNY].sum() / 1e6:.1f}M, Philadelphia MSA {zpop[isPH].sum() / 1e6:.1f}M")


def shares(A, kappa):
    on = np.clip((dmid - 25) / 25, 0, 1)
    g = np.maximum(dmid, 25.0) ** -A
    ft, fn = on * g, (on + kappa * (1 - on)) * g
    Nz = H @ fn
    return (H * ft) / Nz[:, None], Nz


def loss(p, kappa):
    A = p[0]
    W, _ = shares(A, kappa)
    s = np.bincount(band, (zpop @ W) / zpop.sum(), len(edges) - 1)
    lr = np.log(rate[FIT]) - np.log(np.maximum(s[FIT], 1e-30))
    return float(((lr - lr.mean()) ** 2).sum()), math.exp(lr.mean()), s


def cbsa_bands(W, R, mask):
    s = np.bincount(cband, (zpop[mask] @ W[mask]) / zpop[mask].sum(), len(CE) - 1)
    return R * s


out = {}
for kappa in (0.0, 0.05, 0.1, 0.2, 0.4, 1.0):
    r = minimize(lambda p: loss(p, kappa)[0], [2.37], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6})
    A = float(r.x[0])
    L, R, s = loss([A], kappa)
    W, Nz = shares(A, kappa)
    ny, ph = cbsa_bands(W, R, isNY), cbsa_bands(W, R, isPH)
    out[kappa] = dict(A=A, R=R, loss=L, Nz=Nz)
    print(f"kappa {kappa}: A {A:.3f}, R {R:.4f}, loss {L:.3f}")
    for name, m, key in (("NY", ny, "New York CBSA"), ("PHL", ph, "Philadelphia CBSA")):
        t = np.array(cb[key]["rate"])
        segs = [(50, 80), (80, 150), (150, 300)]
        txt = []
        for lo, hi in segs:
            k = (CE[:-1] >= lo) & (CE[1:] <= hi)
            txt.append(f"{lo}-{hi}: {m[k].sum():.4f} vs {t[k].sum():.4f} (x{m[k].sum() / t[k].sum():.2f})")
        print(f"  {name}: " + "; ".join(txt))
np.savez(cache.replace(".npz", "_fits.npz"), **{f"Nz_{k}": v["Nz"] for k, v in out.items()}, zplace=zplace, zlon=zlon, zlat=zlat)
json.dump({str(k): dict(A=v["A"], R=v["R"], loss=v["loss"]) for k, v in out.items()}, open(cache.replace(".npz", "_fits.json"), "w"), indent=1)
