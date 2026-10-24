"""T-1150: the jobs proxy variants where a pack's jobs come from a table by small area (London BRES
by LSOA, Paris INSEE by commune): the proxy alone against the pack's jobs, as the share of jobs in
the wrong 2 km square and 6 km square.   python s18_outsample.py <city>"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")
from t1150lib import *
import tier0
from read_pack import load_pack
from scipy.signal import fftconvolve
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
city = sys.argv[1]
sig = load_sig(city)
h, a = load_pack(city)
g = h["gravity"]; zm = g["zone_m"]; x0, y0 = g["grid_origin_m"]
lon0, lat0 = h["origin"]["lon"], h["origin"]["lat"]
R = 6371008.8
sx = R * np.radians(sig["lon"] - lon0) * np.cos(np.radians(lat0)); sy = R * np.radians(sig["lat"] - lat0)


def sq(x, y, m):
    return np.floor((x - x0) / m).astype(np.int64) * 1000003 + np.floor((y - y0) / m).astype(np.int64)


orig = tier0._tile_path
def scratch_path(kind, epoch, r, c):
    if kind == "tot_v":
        p = os.path.join(HERE, "dl", f"GHS_BUILT_V_E{epoch}_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.zip")
        return p if os.path.exists(p) else None
    return orig(kind, epoch, r, c)
tier0._tile_path = scratch_path
cache = os.path.join(HERE, f"{city}_bi.npz")
if not os.path.exists(cache):
    bnd = shape(json.load(open(os.path.join(TW, "data", "packs", f"{city}.boundary.geojson")))["features"][0]["geometry"])
    tv, X0, Y0 = tier0.ghsl_window("tot_v", bnd.bounds)
    near = tier0.near_outline(bnd)
    r = 10
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = ((xx ** 2 + yy ** 2) <= r * r).astype(float)
    dens = np.maximum(fftconvolve(tv, disk, mode="same"), 0) / (disk.sum() * 0.01) / 1e6  # Mm3 a km2 within 1 km
    out = {}
    for nm, arr in (("totv", tv), ("bi1", tv * dens), ("bi2", tv * dens ** 2)):
        gg = tier0.grid_to_cells(arr, X0, Y0, None, None, near=near)
        out[nm] = np.array([gg.get(int(c), 0.0) for c in sig["h3"]])
    np.savez_compressed(cache, **out)
bi = dict(np.load(cache))
osm = sig["osm_errand"] + sig["osm_leisure"] + sig["osm_job"] + sig["osm_land"]
pop = sig["pop"]
truth = {m: None for m in (zm, 3 * zm)}
for m in truth:
    k = sq(a["x_m"].astype(float), a["y_m"].astype(float), m)
    u, inv = np.unique(np.concatenate([k, sq(sx, sy, m)]), return_inverse=True)
    truth[m] = (inv[:len(k)], inv[len(k):], len(u))
J = a["jobs"].astype(float)


def score(w):
    out = []
    for m, (ip, is_, n) in truth.items():
        t = np.bincount(ip, weights=J, minlength=n); t /= t.sum()
        p = np.bincount(is_, weights=w, minlength=n); p /= p.sum()
        out.append(0.5 * np.abs(t - p).sum())
    return out


print(f"{city}: pack jobs {J.sum():,.0f}; {h['sources'][1]['name'][:120]}")
nr2 = tier0.nres_signal(sig); nr3 = tier0.nres_signal(sig, 3.0)
build = {"nres h^2 (now)": nr2, "nres h^3": nr3, "intensity p1": bi["bi1"], "intensity p2": bi["bi2"],
         "half nres, half intensity p2": 0.5 * nr2 / nr2.sum() + 0.5 * bi["bi2"] / bi["bi2"].sum()}
for bn, b in build.items():
    for wn in (0.3, 0.5):
        w = wn * b / b.sum() + (0.8 - wn) * osm / osm.sum() + 0.2 * pop / pop.sum()
        s2, s6 = score(w)
        print(f"  {bn}, building weight {wn}: misplaced at 2 km {s2:.1%}, at 6 km {s6:.1%}")
