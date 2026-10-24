"""T-1150: homes variants for Shanghai on a scratch copy, scored against the 2020 census by township
(helper1m's Shanghai townships: ASPECT summed over each township, census-exact by county).

Variants (each spread to the 16 districts' 2020 census totals, as the pack does):
  ghsl      GHS-POP as the pack has it
  smooth3   GHS-POP smoothed by a 3x3 pixel mean (spike damping only)
  cap       pixels above 3x their 8 neighbours' median clipped to 3x it
  resv      residential built volume (GHS-BUILT-V total - NRES), no GHS-POP
  ratioK    resv x the people per residential m3 of GHS-POP smoothed over a Gaussian of K km
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")
from t1150lib import *
import tier0, england, china
from scipy import ndimage
import shapely
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
city = "sha"
sig = load_sig(city)
ds = sha_units()
units = [(d[0], d[1]) for d in ds]
tot = {d[0]: d[2] for d in ds}

# ---- the grids over the pack's box
import build_city as b
bnd = shape(json.load(open(os.path.join(TW, "data", "raw", "tier0", "china", "sha_outline.geojson")))["features"][0]["geometry"])
box = bnd.bounds
pop, X0, Y0 = tier0.ghsl_window("pop", box)
nv, _, _ = tier0.ghsl_window("nres_v", box)
# total volume from the scratch tiles (same window arithmetic as ghsl_window)
orig = tier0._tile_path
def scratch_path(kind, epoch, r, c):
    if kind == "tot_v":
        p = os.path.join(HERE, "dl", f"GHS_BUILT_V_E{epoch}_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.zip")
        return p if os.path.exists(p) else None
    return orig(kind, epoch, r, c)
tier0._tile_path = scratch_path
tv, X0b, Y0b = tier0.ghsl_window("tot_v", box)
assert (X0b, Y0b) == (X0, Y0) and tv.shape == pop.shape
resv = np.maximum(tv - nv, 0.0)
print(f"window {pop.shape}: GHS-POP {pop.sum():,.0f}; residential volume {resv.sum() / 1e9:,.2f} bn m3; "
      f"people on pixels with no residential volume {pop[resv <= 0].sum():,.0f}")

# ---- pixel -> res-9 cell map (4 points a pixel), once
_, inv = tier0._moll()
rows, cols = np.nonzero((pop > 0) | (resv > 0))
off = (np.arange(2) + 0.5) / 2 * tier0.PIX_M
dx, dy = np.meshgrid(off, off)
x = (X0 + cols[:, None] * tier0.PIX_M + dx.ravel()[None, :]).ravel()
y = (Y0 - rows[:, None] * tier0.PIX_M - dy.ravel()[None, :]).ravel()
lon, lat = inv.transform(x, y)
ids = np.fromiter((h3.str_to_int(h3.latlng_to_cell(la, lo, 9)) for la, lo in zip(lat, lon)), np.uint64, len(x))
pos = {int(c): i for i, c in enumerate(sig["h3"])}
cell_of = np.array([pos.get(int(c), -1) for c in ids])
print(f"points {len(x):,}, on pack cells {np.mean(cell_of >= 0):.1%}")


def to_cells(grid):
    v = np.repeat(grid[rows, cols] / 4.0, 4)
    ok = cell_of >= 0
    return np.bincount(cell_of[ok], weights=v[ok], minlength=len(sig["h3"]))


def spread(w):
    return england.spread_totals(sig["lat"], sig["lon"], units, tot, w)


def gauss(a, km):
    return ndimage.gaussian_filter(a, sigma=km * 10.0, mode="constant")


variants = {"pack": sig["pop_ghsl"], "ghsl": to_cells(pop)}
variants["smooth3"] = to_cells(ndimage.uniform_filter(pop, 3, mode="constant"))
fp = np.ones((3, 3), bool); fp[1, 1] = False
med = ndimage.median_filter(pop, footprint=fp, mode="constant")
variants["cap3"] = to_cells(np.where(pop > 3 * med, 3 * med, pop))
variants["resv"] = to_cells(resv)
for km in (0.5, 1.0, 2.0, 3.0, 5.0):
    r = gauss(pop, km) / np.maximum(gauss(resv, km), 1e-9)
    g = resv * r
    g = np.where(resv > 0, g, 0.0)
    variants[f"ratio{km:g}"] = to_cells(g)
rr = np.where(resv > 0, pop / np.maximum(resv, 1e-9), 0.0)
sel = resv > 0
o = np.argsort(rr[sel]); cw = np.cumsum(resv[sel][o]) / resv[sel].sum()
for lo, hi in ((0.1, 0.9), (0.25, 0.75)):
    qlo, qhi = rr[sel][o][np.searchsorted(cw, lo)], rr[sel][o][np.searchsorted(cw, hi)]
    rc = np.clip(rr, qlo, qhi)
    variants[f"clip{int(lo * 100)}"] = to_cells(np.where(sel, resv * rc, 0.0))
    g = resv * gauss(resv * rc, 2.0) / np.maximum(gauss(resv, 2.0), 1e-9)
    variants[f"clip{int(lo * 100)}_r2"] = to_cells(np.where(sel, g, 0.0))
    print(f"clip {lo}-{hi}: people per 1,000 m3 {qlo * 1000:.2f}-{qhi * 1000:.2f}")
np.savez_compressed(os.path.join(HERE, "sha_variants_w.npz"), **variants)

# ---- truth: townships
fc = json.load(open(r"C:\Users\anita\projects\maps\helper1m\countries\china\adm4\01.geojson", encoding="utf-8"))
tw = [(f["properties"]["code"], shape(f["geometry"]), f["properties"]["populations"]["2020"], f["properties"]["name"]) for f in fc["features"]]
print(f"townships: {len(tw)}, 2020 people {sum(t[2] for t in tw):,}")
tidx, _ = england.cell_units(sig["lat"], sig["lon"], [(t[0], t[1]) for t in tw])
truth = np.array([t[2] for t in tw], float)
area = np.load(os.path.join(HERE, "sha_area.npy"))
tarea = np.bincount(tidx[tidx >= 0], weights=area[tidx >= 0], minlength=len(tw))
central = np.array([t[2] / max(a_, 1e-9) > 15000 for t, a_ in zip(tw, tarea)])
res = {}
print("\nvariant | misplaced across townships | central townships (>15k/km2) misplaced | densest cell /km2 | cells over 100k/km2 | top 0.1% cell /km2")
for name, w in variants.items():
    p = spread(w)
    m = np.bincount(tidx[tidx >= 0], weights=p[tidx >= 0], minlength=len(tw))
    k = truth.sum() / m.sum()  # cells outside every township polygon (shore slivers) set aside
    mis = 0.5 * np.abs(m * k - truth).sum() / truth.sum()
    mc = 0.5 * np.abs(m[central] * k - truth[central]).sum() / truth[central].sum()
    d = p / area
    res[name] = p
    print(f"{name} | {mis:.1%} | {mc:.1%} | {d.max():,.0f} | {(d > 100000).sum()} | {np.quantile(d[p > 0], 0.999):,.0f}")
np.savez_compressed(os.path.join(HERE, "sha_variants_pop.npz"), **res)
np.save(os.path.join(HERE, "sha_tidx.npy"), tidx)
json.dump([[t[0], t[3], t[2]] for t in tw], open(os.path.join(HERE, "sha_townships.json"), "w", encoding="utf-8"), ensure_ascii=False)
