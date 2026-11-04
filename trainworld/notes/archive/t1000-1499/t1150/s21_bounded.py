"""T-1150: the homes fix bounded: GHS-POP's people per residential m3 held within a factor k of its
2 km Gaussian mean (r' = clip(r, rs / k, rs * k)), homes = residential volume x r', rescaled to
GHS-POP's total. k = inf is GHS-POP itself, k = 1 is ratio2. Scored against census homes:
Shanghai by township (district totals applied first, as the pack), London and Paris at 1 and 2 km
squares (the pack's census homes). Also the densest cell.   python s21_bounded.py <city>"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")
from t1150lib import *
import tier0, england
from read_pack import load_pack
from scipy import ndimage
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
city = sys.argv[1]
sig = load_sig(city)
orig = tier0._tile_path
def scratch_path(kind, epoch, r, c):
    if kind == "tot_v":
        p = os.path.join(HERE, "dl", f"GHS_BUILT_V_E{epoch}_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.zip")
        return p if os.path.exists(p) else None
    return orig(kind, epoch, r, c)
tier0._tile_path = scratch_path
bnd = shape(json.load(open(os.path.join(TW, "data", "packs", f"{city}.boundary.geojson")))["features"][0]["geometry"])
near = tier0.near_outline(bnd)
pop, X0, Y0 = tier0.ghsl_window("pop", bnd.bounds)
nv, _, _ = tier0.ghsl_window("nres_v", bnd.bounds)
tv, _, _ = tier0.ghsl_window("tot_v", bnd.bounds)
resv = np.maximum(tv - nv, 0)
g = lambda x: ndimage.gaussian_filter(x, 20.0, mode="constant")
rs = g(pop) / np.maximum(g(resv), 1e-9)
r = np.where(resv > 0, pop / np.maximum(resv, 1e-9), 0.0)
area = cell_area_km2(sig["h3"])
var = {}
for k in (np.inf, 3.0, 2.0, 1.5, 1.0):
    rk = np.clip(r, rs / k, rs * k) if np.isfinite(k) else r
    arr = np.where(resv > 0, resv * rk, 0.0)
    arr *= pop.sum() / arr.sum()
    gg = tier0.grid_to_cells(arr, X0, Y0, None, None, near=near)
    var[f"k{k:g}"] = np.array([gg.get(int(c), 0.0) for c in sig["h3"]])
np.savez_compressed(os.path.join(HERE, f"{city}_bounded.npz"), **var)
if city == "sha":
    ds = sha_units(); units = [(d[0], d[1]) for d in ds]
    fc = json.load(open(r"C:\Users\anita\projects\maps\helper1m\countries\china\adm4\01.geojson", encoding="utf-8"))
    truth = np.array([f["properties"]["populations"]["2020"] for f in fc["features"]], float)
    tidx = np.load(os.path.join(HERE, "sha_tidx.npy"))
    for nm, w in var.items():
        p = england.spread_totals(sig["lat"], sig["lon"], units, {d[0]: d[2] for d in ds}, w)
        m = np.bincount(tidx[tidx >= 0], weights=p[tidx >= 0], minlength=len(truth))
        m *= truth.sum() / m.sum()
        print(f"sha {nm}: townships misplaced {0.5 * np.abs(m - truth).sum() / truth.sum():.1%}; densest cell {(p / area).max():,.0f}/km2, "
              f"cells over 100k/km2 {(p / area > 1e5).sum()}")
else:
    h, a = load_pack(city)
    lon0, lat0 = h["origin"]["lon"], h["origin"]["lat"]
    R = 6371008.8
    sx = R * np.radians(sig["lon"] - lon0) * np.cos(np.radians(lat0)); sy = R * np.radians(sig["lat"] - lat0)
    P = a["pop"].astype(float)
    for nm, w in var.items():
        out = []
        for m in (1000.0, 2000.0):
            kp = np.floor(a["x_m"] / m).astype(np.int64) * 100003 + np.floor(a["y_m"] / m).astype(np.int64)
            ks = np.floor(sx / m).astype(np.int64) * 100003 + np.floor(sy / m).astype(np.int64)
            u, inv = np.unique(np.concatenate([kp, ks]), return_inverse=True)
            t = np.bincount(inv[:len(kp)], weights=P, minlength=len(u)); t /= t.sum()
            p = np.bincount(inv[len(kp):], weights=w, minlength=len(u)); p /= p.sum()
            out.append(f"{m / 1000:g} km {0.5 * np.abs(t - p).sum():.1%}")
        print(f"{city} {nm}: misplaced against the census " + ", ".join(out) + f"; densest cell {(w / area).max():,.0f}/km2 (unscaled)")
