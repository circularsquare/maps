"""T-1150: the homes fixes through the real code path (tier0.signals, the population hooks), with
data/work and data/packs redirected to ../artifacts/trainworld/t1150/check/ (nothing of the real
tree written; no download: a missing tile stops it). Total-volume tiles from data/raw/tier0/ghsl or,
before the rebuild moves them there, data/raw/t1150_ghsl.

    python trainworld/notes/t1150/check.py <city>        sha, bei, lon, par, mos
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(os.path.dirname(TW), "artifacts", "trainworld", "t1150", "check")
sys.path.insert(0, os.path.join(TW, "pipeline"))
import numpy as np
import h3
import shapely
from shapely import union_all, make_valid
from shapely.geometry import shape
import build_city as b

b.WORK = os.path.join(HERE, "work")
b.PACKS = os.path.join(HERE, "packs")
os.makedirs(b.WORK, exist_ok=True)
os.makedirs(b.PACKS, exist_ok=True)
import tier0, england, china
from read_pack import load_pack

TOT = os.path.join(TW, "data", "raw", "t1150_ghsl")


def tile_path(kind, epoch, r, c):
    path = tier0.raw("ghsl", f"{tier0.PRODUCTS[kind][1].format(epoch=epoch)}_V1_0_R{r}_C{c}.zip")
    if kind == "tot_v" and not os.path.exists(path):
        path = os.path.join(TOT, os.path.basename(path))
    if os.path.exists(path):
        return path
    if os.path.exists(path + ".none"):
        return None
    raise SystemExit(f"would download {path}")


tier0._tile_path = tile_path
city = sys.argv[1]
cfg = b.CITIES[city]
t0 = time.time()
if city in ("sha", "lon", "par"):
    boundary = tier0.build_boundary(city, cfg)
    mask, _ = tier0.build_water(cfg, boundary)
else:  # no water cache for these extracts: the pack's own outline, people left on water pixels
    with open(os.path.join(TW, "data", "packs", f"{city}.boundary.geojson")) as f:
        boundary = union_all([make_valid(shape(ft["geometry"])) for ft in json.load(f)["features"]])
    mask = None
    tier0.land_function = lambda m: None
cache = os.path.join(b.WORK, f"{city}.tier0_signals.npz")
t1 = time.time()
sig = tier0.signals(city, cfg, boundary, mask)
print(f"signals {time.time() - t1:.0f} s", flush=True)
area = np.array([h3.cell_area(h3.int_to_str(int(c)), "km^2") for c in sig["h3"]])
lat, lon = sig["lat"], sig["lon"]


def dens(p, label):
    d = p / area
    print(f"  {label}: {p.sum():,.0f} people; densest cell {d.max():,.0f}/km2, over 100k/km2 {(d > 1e5).sum()}, "
          f"over 60k {(d > 6e4).sum()}", flush=True)


def with_weight(w):
    """sig with w as the weight inside townships and outside them (the rules before the decision)."""
    s = dict(sig)
    s["pop_ghsl"] = s["res_v"] = w
    return s


def town_score(p, ts, label):
    units = [(t[0], t[1]) for t in ts]
    truth = np.array([t[2] for t in ts])
    idx, _ = england.cell_units(lat, lon, units)
    ok = idx >= 0
    m = np.bincount(idx[ok], weights=p[ok], minlength=len(ts))
    m *= truth.sum() / m.sum()
    big = truth > 20000
    off = np.abs(m / np.maximum(truth, 1) - 1)
    print(f"  {label}: homes in the wrong township {0.5 * np.abs(m - truth).sum() / truth.sum():.1%}; "
          f"townships over 20k off by more than 30%: {(off[big] > 0.3).sum()} of {big.sum()}", flush=True)
    return m


def moved(a, bb, m=2000.0):
    lon0, lat0 = cfg["origin"] if "origin" in cfg else (lon.mean(), lat.mean())
    x = 6371008.8 * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = 6371008.8 * np.radians(lat - lat0)
    k = np.floor(x / m).astype(np.int64) * 100003 + np.floor(y / m).astype(np.int64)
    u, inv = np.unique(k, return_inverse=True)
    pa = np.bincount(inv, weights=a, minlength=len(u)); pa /= pa.sum()
    pb = np.bincount(inv, weights=bb, minlength=len(u)); pb /= pb.sum()
    return 0.5 * np.abs(pa - pb).sum()


def vs_pack(w, label):
    h, a = load_pack(city)
    lon0, lat0 = h["origin"]["lon"], h["origin"]["lat"]
    sx = 6371008.8 * np.radians(lon - lon0) * np.cos(np.radians(lat0)); sy = 6371008.8 * np.radians(lat - lat0)
    P = a["pop"].astype(float)
    out = []
    for m in (1000.0, 2000.0):
        kp = np.floor(a["x_m"] / m).astype(np.int64) * 100003 + np.floor(a["y_m"] / m).astype(np.int64)
        ks = np.floor(sx / m).astype(np.int64) * 100003 + np.floor(sy / m).astype(np.int64)
        u, inv = np.unique(np.concatenate([kp, ks]), return_inverse=True)
        t = np.bincount(inv[:len(kp)], weights=P, minlength=len(u)); t /= t.sum()
        p = np.bincount(inv[len(kp):], weights=w, minlength=len(u)); p /= p.sum()
        out.append(f"{m / 1000:g} km {0.5 * np.abs(t - p).sum():.1%}")
    print(f"  {label} against the pack's census homes: " + ", ".join(out), flush=True)


raw, bnd = sig["pop_ghs"], sig["pop_ghsl"]

print(f"{city}: {len(raw):,} cells; GHSL bounded moves {moved(raw, bnd):.1%} of people between 2 km squares", flush=True)
res = {"raw": raw, "bounded": bnd}
if city == "sha":
    ds = china.districts(city)
    units = [(d[0], d[1]) for d in ds]
    dt = {d[0]: d[2] for d in ds}
    res["now"] = england.spread_totals(lat, lon, units, dt, raw)  # the pack's rule before T-1150
    res["fix1"] = england.spread_totals(lat, lon, units, dt, bnd)
    res["fix12"], name = cfg["population"](city, boundary, sig)
    print("  source:", name)
    ts = [(t[0], t[1], t[2] * t[3]) for t in china.townships(boundary) if t[3] > 0]
    for k in ("now", "fix1", "fix12"):
        dens(res[k], k)
        town_score(res[k], ts, k)
    res["town_raw"], _ = china.township_population(city, boundary, with_weight(raw), held=ds)
    dens(res["town_raw"], "townships, GHS-POP inside (fix 2 alone)")
elif city == "bei":
    import plan_config
    res["fix12"], name = cfg["population"](city, boundary, sig)
    print("  source:", name)
    ts = [(t[0], t[1], t[2] * t[3]) for t in china.townships(boundary) if t[3] > 0]
    for k in ("raw", "bounded", "fix12"):
        dens(res[k], k)
        town_score(res[k], ts, k)
    res["town_raw"], _ = china.township_population(city, boundary, with_weight(raw))
    dens(res["town_raw"], "townships, GHS-POP inside (fix 2 alone)")
if city in ("sha", "bei"):
    res["town_bnd"], _ = china.township_population(city, boundary, with_weight(bnd),
                                                   held=china.districts(city) if city == "sha" else None)
    dens(res["town_bnd"], "townships, bounded GHSL inside (the first built rule)")
else:
    for k in ("raw", "bounded"):
        dens(res[k], k)
        vs_pack(res[k], f"GHSL {k}")
    if cfg.get("population"):
        res["census_now"], _ = cfg["population"](city, boundary, with_weight(raw))
        res["census_new"], _ = cfg["population"](city, boundary, sig)
        print(f"  census homes spread by bounded GHSL against by GHS-POP: {moved(res['census_now'], res['census_new']):.2%} "
              f"of people change 2 km square; {moved(res['census_now'], res['census_new'], 1000.0):.2%} at 1 km", flush=True)
np.savez_compressed(os.path.join(HERE, f"{city}_check.npz"), h3=sig["h3"], **res)
print(f"done in {time.time() - t0:.0f} s", flush=True)
