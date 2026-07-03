"""Diagnostic: are territories actually contiguous, and what are the 'blue' cells?

Runs the pipeline on a small inland crop (no sea) and a coastal crop, then:
  - counts 4-connected components per territory (──> contiguity violation if >1)
  - reports how many 'blue' (z<=base) cells sit inland vs at the coast
  - checks the raw ascent basins are themselves contiguous
"""
import sys
import numpy as np
from scipy import ndimage
import ascentshed as A
import terrain as T


def contiguity(lab, conn=1):
    """For each territory id >=0, count connected components (conn=1 -> 4-nbr)."""
    structure = ndimage.generate_binary_structure(2, conn)
    bad = []
    for t in np.unique(lab):
        if t < 0:
            continue
        _, n = ndimage.label(lab == t, structure=structure)
        if n > 1:
            bad.append((int(t), n))
    return bad


def run_crop(z, ca, base, N, min_prom, tag):
    print(f"\n=== {tag}: {z.shape}  z {z.min():.0f}..{z.max():.0f}  base={base}")
    label, _ = A.ascent_basins(z, land=(z > base))
    tree = A.build_tree(z, label, cellarea=ca, base=base)
    # raw-basin contiguity (8-conn, since ascent uses 8 neighbours)
    raw_bad4 = contiguity(label, conn=1)
    raw_bad8 = contiguity(label, conn=2)
    print(f"raw basins: {len(tree['peaks'])}  "
          f"non-contig(4-conn): {len(raw_bad4)}  non-contig(8-conn): {len(raw_bad8)}")
    label2 = A.simplify_basins(label, tree, min_prom)
    tree2 = A.build_tree(z, label2, cellarea=ca, base=base)
    lab, roots = A.balanced_regions(label2, tree2, N, base=base)
    terr_bad4 = contiguity(lab, conn=1)
    terr_bad8 = contiguity(lab, conn=2)
    print(f"territories: {len(roots)}  "
          f"non-contig(4-conn): {len(terr_bad4)}  non-contig(8-conn): {len(terr_bad8)}")
    if terr_bad8:
        print(f"  8-conn-broken territories (id, #pieces): {terr_bad8[:6]}")
    # blue cells
    sea = z <= base
    interior_sea = sea & ndimage.binary_erosion(sea, iterations=2)
    print(f"sea cells z<=base: {sea.sum()}  ({100*sea.mean():.1f}%)  "
          f"of which interior(eroded): {interior_sea.sum()}")
    return label, label2, lab, tree2


if __name__ == "__main__":
    z, ca = T.load_dem('../data/ascentshed/gebco/japan.tif', downsample=2)
    base = 0.0
    # locate the highest cell (central Honshu Alps) and crop a window around it
    iy, ix = np.unravel_index(np.argmax(z), z.shape)
    print("highest cell at", (iy, ix), "elev", z[iy, ix])
    h = 220
    y0, x0 = max(0, iy - h), max(0, ix - h)
    crop = z[y0:y0 + 2 * h, x0:x0 + 2 * h]
    ccrop = ca[y0:y0 + 2 * h, x0:x0 + 2 * h]
    print("crop z>0 fraction:", round((crop > 0).mean(), 3))
    run_crop(crop, ccrop, base, 8, 80, "central-honshu crop")
    # synthetic, no sea at all
    import run_synthetic as RS
    zs = RS.fractal_terrain(240)
    run_crop(zs, np.ones_like(zs), 0.0, 12, 0.0, "synthetic (no sea)")
