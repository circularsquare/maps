"""T-1150: GHS-POP's people per residential cubic metre, by 100 m pixel: flat inside each of GHSL's
input census units, so the units show as patches.   python s7_ratio_map.py <city> <half km>"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from t1150lib import *
import tier0
import build_city as b
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
city, half = sys.argv[1], float(sys.argv[2])
lon0, lat0 = map(float, sys.argv[3:5]) if len(sys.argv) > 4 else b.CITIES[city]["origin"]
orig = tier0._tile_path
def scratch_path(kind, epoch, r, c):
    if kind == "tot_v":
        p = os.path.join(HERE, "dl", f"GHS_BUILT_V_E{epoch}_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.zip")
        return p if os.path.exists(p) else None
    return orig(kind, epoch, r, c)
tier0._tile_path = scratch_path
tier0.PRODUCTS.setdefault("tot_v", ("GHS_BUILT_V_GLOBE_R2023A", "GHS_BUILT_V_E{epoch}_GLOBE_R2023A_54009_100"))
fwd, inv = tier0._moll()
cx, cy = fwd.transform(lon0, lat0)
# a square window in Mollweide metres round the origin
box_x = (cx - half * 1000, cx + half * 1000)
box_y = (cy - half * 1000, cy + half * 1000)
lons, lats = inv.transform([box_x[0], box_x[0], box_x[1], box_x[1]], [box_y[0], box_y[1], box_y[0], box_y[1]])
bb = (min(lons), min(lats), max(lons), max(lats))
pop, X0, Y0 = tier0.ghsl_window("pop", bb)
nv, _, _ = tier0.ghsl_window("nres_v", bb)
tv, _, _ = tier0.ghsl_window("tot_v", bb)
c0, r0 = int((box_x[0] - X0) / 100), int((Y0 - box_y[1]) / 100)
n = int(2 * half * 10)
sl = (slice(r0, r0 + n), slice(c0, c0 + n))
pop, nv, tv = pop[sl], nv[sl], tv[sl]
resv = np.maximum(tv - nv, 0)
ratio = np.where(resv > 0, pop / np.maximum(resv, 1e-9) * 1000, np.nan)
print(f"{city}: ratio quantiles (people per 1,000 m3):", np.nanquantile(ratio, [0.05, 0.25, 0.5, 0.75, 0.95]).round(2))
fig, axs = plt.subplots(1, 3, figsize=(27, 9), dpi=100)
ext = [-half, half, -half, half]
im = axs[0].imshow(np.where(pop > 0, pop * 100, np.nan), extent=ext, cmap="magma_r", norm=LogNorm(300, 150000))
axs[0].set_title("GHS-POP 2020: people a km² (100 m pixels)")
fig.colorbar(im, ax=axs[0], shrink=0.7)
im = axs[1].imshow(np.where(resv > 0, resv, np.nan), extent=ext, cmap="magma_r", norm=LogNorm(1e3, 1e6))
axs[1].set_title("GHS-BUILT-V residential volume: m³ a hectare")
fig.colorbar(im, ax=axs[1], shrink=0.7)
im = axs[2].imshow(ratio, extent=ext, cmap="RdYlBu_r", norm=LogNorm(1, 30))
axs[2].set_title("GHS-POP over residential volume: people per 1,000 m³\n(flat inside each of GHSL's input census units)")
fig.colorbar(im, ax=axs[2], shrink=0.7)
for ax in axs:
    ax.set_facecolor("#e8eef2")
    ax.set_xlabel("km (GHSL's Mollweide grid)")
out = os.path.join(ART, f"{city}-ghsl-units.png")
fig.savefig(out, bbox_inches="tight")
print(out)
