"""T-1150: the same checks on another tier 0 pack: densest cells, how even the jobs are, and what
T-1150's homes fix (ratio2: residential volume x GHS-POP's people per residential m3 smoothed over
2 km) would move.   python s14_city.py <city>"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")
from t1150lib import *
import tier0
from read_pack import load_pack
from scipy import ndimage
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
city = sys.argv[1]
h, a = load_pack(city)
src = {s["what"]: s["name"] for s in h["sources"]}
print(f"{city}: pop {h['totals']['pop']:,} jobs {h['totals']['jobs']:,}\n  homes: {src['population'][:200]}\n  jobs: {src['jobs'][:260]}")
pa = cell_area_km2(a["h3"])
d = a["pop"] / pa
print(f"  pack cells: {len(pa):,}; densest homes {d.max():,.0f}/km2, cells over 100k/km2 {(d > 1e5).sum()}, over 60k {(d > 6e4).sum()}, top 0.1% {np.quantile(d[a['pop'] > 0], 0.999):,.0f}")
dj = a["jobs"] / pa
print(f"  densest jobs {dj.max():,.0f}/km2 ({a['jobs'].max():,.0f} in a cell)")
for k in ("pop", "jobs"):
    o = np.sort(a[k])[::-1]
    print(f"  {k}: top 1% of cells hold {o[:len(o) // 100].sum() / o.sum():.1%}, top 5% {o[:len(o) // 20].sum() / o.sum():.1%}")
z = a["zone"]
zp = np.bincount(z, weights=a["pop"].astype(float)); zj = np.bincount(z, weights=a["jobs"].astype(float))
print(f"  2 km zones: corr(homes, jobs) {np.corrcoef(zp, zj)[0, 1]:.2f}; jobs in the top 2% of zones {np.sort(zj)[::-1][:max(1, len(zj) // 50)].sum() / zj.sum():.1%}")

# homes fix on the signal cells
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
print(f"  people on pixels with no residential volume: {pop[resv <= 0].sum() / pop.sum():.2%}")
g = lambda x: ndimage.gaussian_filter(x, 20.0, mode="constant")
fix = np.where(resv > 0, resv * g(pop) / np.maximum(g(resv), 1e-9), 0.0)
fix *= pop.sum() / fix.sum()
cells = {}
for name, arr in (("ghsl", pop), ("ratio2", fix)):
    gg = tier0.grid_to_cells(arr, X0, Y0, None, None, near=near)
    cells[name] = np.array([gg.get(int(c), 0.0) for c in sig["h3"]])
area = cell_area_km2(sig["h3"])
k = cells["ghsl"].sum() / cells["ratio2"].sum()
cells["ratio2"] *= k
for name, v in cells.items():
    dd = v / area
    print(f"  {name} (unscaled GHSL people): densest cell {dd.max():,.0f}/km2, over 100k/km2 {(dd > 1e5).sum()}, over 60k {(dd > 6e4).sum()}")
# homes moved at 2 km (squares on the local grid)
lon0, lat0 = h["origin"]["lon"], h["origin"]["lat"]
x = 6371008.8 * np.radians(sig["lon"] - lon0) * np.cos(np.radians(lat0)); y = 6371008.8 * np.radians(sig["lat"] - lat0)
key = np.floor(x / 2000).astype(np.int64) * 100000 + np.floor(y / 2000).astype(np.int64)
u, inv = np.unique(key, return_inverse=True)
s1 = np.bincount(inv, weights=cells["ghsl"]); s2 = np.bincount(inv, weights=cells["ratio2"])
print(f"  ratio2 moves {0.5 * np.abs(s1 - s2).sum() / s1.sum():.1%} of homes between 2 km squares, "
      f"{0.5 * np.abs(cells['ghsl'] - cells['ratio2']).sum() / cells['ghsl'].sum():.1%} between cells")
np.savez_compressed(os.path.join(HERE, f"{city}_ratio2.npz"), pop=cells["ratio2"], ghsl=cells["ghsl"])
