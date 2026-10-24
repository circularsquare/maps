"""T-1150: density maps of a pack's homes and jobs, at 2 km zones and at the cell level.

    python s4_maps.py <city> [half-width km of the cell-level window, default 15] [tag]
Optional env T1150_ALT=<npz with pop/jobs per res-9 signal cell> draws a scratch variant at the
cell level instead of the pack (its res-9 cells)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from t1150lib import *
from read_pack import load_pack
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

city = sys.argv[1]
half = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
tag = sys.argv[3] if len(sys.argv) > 3 else ""
h, a = load_pack(city)
lon0, lat0 = h["origin"]["lon"], h["origin"]["lat"]
R = 6371008.8
k = np.cos(np.radians(lat0))
g = h["gravity"]
zm, (x0, y0), (W, H) = g["zone_m"], g["grid_origin_m"], g["grid"]
side = a["zone_size"].astype(int) if "zone_size" in a else np.ones(h["zones"], int)


def zone_maps():
    for what in ("pop", "jobs"):
        grid = np.full((H, W), np.nan)
        zs = np.bincount(a["zone"], weights=a[what].astype(float), minlength=h["zones"])
        for zi in range(h["zones"]):
            s = side[zi]
            d = zs[zi] / (s * s * (zm / 1000) ** 2)
            gx, gy = int(a["zone_gx"][zi]), int(a["zone_gy"][zi])
            cur = grid[gy:gy + s, gx:gx + s]
            grid[gy:gy + s, gx:gx + s] = np.where(np.isnan(cur), d, np.nan_to_num(cur) + d)
        fig, ax = plt.subplots(figsize=(10, 10 * H / W + 0.5), dpi=130)
        ext = [x0 / 1e3, (x0 + W * zm) / 1e3, y0 / 1e3, (y0 + H * zm) / 1e3]
        im = ax.imshow(np.where(grid > 0, grid, np.nan), origin="lower", extent=ext, cmap="magma_r",
                       norm=LogNorm(vmin=100, vmax=50000))
        ax.set_facecolor("#e8eef2")
        ax.set_title(f"{city}: {'homes' if what == 'pop' else 'jobs'} a km² by 2 km zone (pack)")
        ax.set_xlabel("km east of origin"); ax.set_ylabel("km north")
        fig.colorbar(im, ax=ax, shrink=0.7, label="per km²")
        out = os.path.join(ART, f"{city}{tag}-{what}-2km.png")
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(out, f"max {np.nanmax(grid):,.0f}/km2")


def cells_map(ids, vals, what, title, out, vmax=150000):
    polys, dens = [], []
    for c, v in zip(ids, vals):
        s = h3.int_to_str(int(c))
        la, lo = h3.cell_to_latlng(s)
        x = R * np.radians(lo - lon0) * k / 1e3
        y = R * np.radians(la - lat0) / 1e3
        if abs(x) > half or abs(y) > half or v <= 0:
            continue
        b = np.array(h3.cell_to_boundary(s))
        polys.append(np.c_[R * np.radians(b[:, 1] - lon0) * k / 1e3, R * np.radians(b[:, 0] - lat0) / 1e3])
        dens.append(v / h3.cell_area(s, "km^2"))
    fig, ax = plt.subplots(figsize=(11, 10), dpi=140)
    pc = PolyCollection(polys, array=np.array(dens), cmap="magma_r", norm=LogNorm(vmin=300, vmax=vmax), linewidths=0)
    ax.add_collection(pc)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
    ax.set_facecolor("#e8eef2")
    ax.set_title(title)
    ax.set_xlabel("km east of origin"); ax.set_ylabel("km north")
    fig.colorbar(pc, ax=ax, shrink=0.75, label="per km²")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(out, f"max {max(dens):,.0f}/km2")


if not os.environ.get("T1150_ALT"):
    zone_maps()
    for what in ("pop", "jobs"):
        cells_map(a["h3"], a[what], what, f"{city}: {'homes' if what == 'pop' else 'jobs'} a km² by cell (pack), central {2 * half:.0f} km",
                  os.path.join(ART, f"{city}{tag}-{what}-cells.png"))
else:
    z = np.load(os.environ["T1150_ALT"])
    sig = load_sig(city)
    for what in ("pop", "jobs"):
        if what in z.files:
            cells_map(sig["h3"], z[what], what, f"{city}{tag}: {'homes' if what == 'pop' else 'jobs'} a km² by res-9 cell, central {2 * half:.0f} km",
                      os.path.join(ART, f"{city}{tag}-{what}-cells.png"))
