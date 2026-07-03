"""Run ascentshed on a real DEM tile and render equal-heft territories.

  python run_dem.py ../data/ascentshed/alps.tif 3 18

args: tif path, integer downsample, N territories.
Renders hillshade + translucent territories + white divides + seed-peak dots.
"""
import sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LightSource
from scipy import ndimage

import ascentshed as A
import terrain as T


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '../data/ascentshed/alps.tif'
    ds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 18
    min_prom = float(sys.argv[4]) if len(sys.argv) > 4 else 150.0

    z, cellarea = T.load_dem(path, downsample=ds)
    print(f"DEM {z.shape}  elev {z.min():.0f}..{z.max():.0f} m", flush=True)
    # sea level for topo+bathy grids; valley floor for a land-only tile
    base = 0.0 if z.min() < -10 else float(z.min())

    t = time.time(); label, peaks = A.ascent_basins(z, land=(z > base))
    print(f"basins: {len(peaks)}  ({time.time()-t:.1f}s)", flush=True)
    t = time.time(); tree = A.build_tree(z, label, cellarea=cellarea, base=base)
    print(f"build_tree  ({time.time()-t:.1f}s)", flush=True)
    t = time.time()
    label = A.simplify_basins(label, tree, min_prom)
    tree = A.build_tree(z, label, cellarea=cellarea, base=base)
    print(f"simplify (>={min_prom:.0f}m): {len(tree['peaks'])} basins  "
          f"({time.time()-t:.1f}s)", flush=True)
    t = time.time(); lab, seeds = A.balanced_regions(label, tree, N, base=base)
    v = np.array(sorted(A.territory_volumes(lab, z, tree, base=base,
                        cellarea=cellarea).values(), reverse=True))
    print(f"regions N={N}: CV={A.cv(v):.3f}  max/min={v.max()/v.min():.2f}  "
          f"({time.time()-t:.1f}s)", flush=True)

    # ---- render ----
    import os
    m = z.shape[1]
    seamask = z <= base                              # true ocean (by elevation)
    landterr = (lab >= 0) & ~seamask                 # cells coloured by territory
    land_ids = [i for i in np.unique(lab) if i >= 0]
    remap = {val: k for k, val in enumerate(land_ids)}
    comp = np.zeros_like(lab)
    for val, k in remap.items():
        comp[lab == val] = k
    rng = np.random.default_rng(5)
    cols = plt.cm.tab20(np.linspace(0, 1, 20))
    rng.shuffle(cols)
    cmap = ListedColormap(cols)

    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(z, vert_exag=2.0, dx=1, dy=1)

    fig, ax = plt.subplots(figsize=(z.shape[1] / 130, z.shape[0] / 130), dpi=130)
    ax.imshow(np.where(seamask, np.nan, hs), cmap='gray', vmin=0, vmax=1)
    terr = np.ma.masked_where(~landterr, comp % 20)
    ax.imshow(terr, cmap=cmap, interpolation='nearest', alpha=0.55)
    # ocean: bathymetry-shaded blue
    ocean = np.ma.masked_where(~seamask, hs)
    ax.imshow(ocean, cmap='Blues_r', vmin=0.2, vmax=1.1, alpha=0.95)
    # divides between adjacent land territories only
    edges = np.zeros_like(lab, bool)
    diff_h = (lab[:, :-1] != lab[:, 1:]) & landterr[:, :-1] & landterr[:, 1:]
    diff_v = (lab[:-1, :] != lab[1:, :]) & landterr[:-1, :] & landterr[1:, :]
    edges[:, :-1] |= diff_h
    edges[:-1, :] |= diff_v
    ax.contour(edges, levels=[0.5], colors='white', linewidths=0.6)
    sy = [s // m for s in seeds]
    sx = [s % m for s in seeds]
    ax.scatter(sx, sy, s=14, c='black', marker='^', linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"ascentshed  {os.path.basename(path)}  N={len(land_ids)}  CV={A.cv(v):.2f}")
    fig.tight_layout()
    out = os.path.splitext(os.path.basename(path))[0] + '_regions.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
