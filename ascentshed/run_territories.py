"""Faithful inverse-watershed map: ascent basins, coarsened by VOLUME merging.

No region-growing, no equal-heft. Take the raw ascent basins (one per peak), then
repeatedly merge the smallest-volume region into the neighbour across its highest
saddle (the massif it is a shoulder of) until N regions remain -- agglomeration on
the basin ADJACENCY graph, so every region stays a single connected blob. Every
cell is coloured by the dominant peak its steepest-ascent path flows up to.
Region sizes are set by topography (unequal). Sea level is NOT special.

  python run_territories.py ../data/ascentshed/gebco/japan.tif 2 30
  (args: tif, downsample, N regions)
"""
import sys, os
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
# register a CJK-capable font so Japanese/Chinese peak names render (not boxes)
for _fp in ['../asia1m/chinese.msyh.ttf', '../asia1m/Unifontexmono-2vrqo.ttf']:
    if os.path.exists(_fp):
        try:
            font_manager.fontManager.addfont(_fp)
            plt.rcParams['font.family'] = font_manager.FontProperties(
                fname=_fp).get_name()
            break
        except Exception:
            pass

import ascentshed as A
import terrain as T


def assign_hues(regions, adj, seed=0, min_sep=0.04, tries=50):
    """Random hue per region; reroll if it lands within `min_sep` (circular) of an
    already-coloured bordering region. Gives up after `tries` rolls."""
    rng = np.random.default_rng(seed)
    order = sorted(regions, key=lambda r: -len(adj.get(r, ())))
    hue = {}
    for r in order:
        nh = [hue[n] for n in adj.get(r, ()) if n in hue]
        h = float(rng.random())
        for _ in range(tries):
            if all(min(abs(h - x), 1 - abs(h - x)) >= min_sep for x in nh):
                break
            h = float(rng.random())
        hue[r] = h
    return hue


# elevation -> lightness (value), piecewise-linear; 0.0 = black, 1.0 = white
_ELEV_XP = [-12000, -6000, -4000, -1000, -100, 0, 500, 1000, 2000, 3000, 4000, 5000, 8000]
_ELEV_FP = [0.0, 0.1, 0.15, 0.25, 0.35, 0.4, 0.5, 0.58, 0.7, 0.78, 0.83, 0.88, 1.0]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '../data/ascentshed/alps.tif'
    ds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    do_names = (sys.argv[4] != '0') if len(sys.argv) > 4 else True

    z, ca = T.load_dem(path, downsample=ds)
    base = 0.0 if z.min() < -10 else float(z.min())
    print(f"DEM {z.shape}  elev {z.min():.0f}..{z.max():.0f} m", flush=True)

    label, peaks = A.ascent_basins(z)            # raw ascent basins (1 per peak)
    tree = A.build_tree(z, label, cellarea=ca, base=base)
    print(f"{len(peaks)} raw peaks", flush=True)

    terr, regions = A.agglomerate(label, tree, N, score='volume')   # adjacency merge
    tree2 = A.build_tree(z, terr, cellarea=ca, base=base)
    print(f"agglomerated -> {len(regions)} regions", flush=True)

    # ---- name each region after its highest OSM-named peak ----
    import csv
    import rasterio
    import peaknames as PN
    stem = os.path.splitext(os.path.basename(path))[0]
    with rasterio.open(path) as rds:
        transform, bounds = rds.transform, rds.bounds
    names = {}
    try:
        if not do_names:
            raise RuntimeError("naming disabled (arg)")
        osm = PN.fetch_peaks(bounds, cache=f"{stem}_peaks.json")
        names = PN.name_regions(osm, terr, transform, ds, z)
        print(f"named {len(names)}/{len(regions)} regions from {len(osm)} OSM peaks",
              flush=True)
        with open(f"{stem}_names.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["region_id", "name", "name_en", "ele_m", "lat", "lon", "px_cells"])
            area = {int(r): int((terr == r).sum()) for r in regions}
            for r in sorted(regions, key=lambda r: -area[int(r)]):
                d = names.get(int(r), {})
                wr.writerow([r, d.get("name", ""), d.get("name_en", ""),
                             round(d["ele"]) if d.get("ele") else "",
                             round(d.get("lat", 0), 4) if d else "",
                             round(d.get("lon", 0), 4) if d else "", area[int(r)]])
        print(f"wrote {stem}_names.csv", flush=True)
    except Exception as ex:
        print(f"naming skipped: {ex}", flush=True)

    # contiguity: each territory should be one connected blob (the user's test)
    s8 = ndimage.generate_binary_structure(2, 2)
    pieces = []
    for r in regions:
        _, n = ndimage.label(terr == r, structure=s8)
        pieces.append(n)
    pieces = np.array(pieces)
    print(f"contiguity: {(pieces==1).sum()}/{len(regions)} territories are a single "
          f"connected blob;  max pieces={pieces.max()}", flush=True)

    # ---- render: region = HUE, elevation = LIGHTNESS (global scale) ----
    from matplotlib.colors import hsv_to_rgb
    m = z.shape[1]
    hue = assign_hues(regions, tree2.get('adj', {}))
    # per-cell hue via a lookup over region representative ids
    hue_lut = np.zeros(int(terr.max()) + 1)
    for r in regions:
        hue_lut[r] = hue[r]
    H = hue_lut[np.where(terr >= 0, terr, 0)]
    # value (lightness) from elevation via the fixed piecewise-linear curve
    V = np.interp(z, _ELEV_XP, _ELEV_FP)              # global scale, 0=black 1=white
    S = np.full(z.shape, 0.60)
    rgb = hsv_to_rgb(np.dstack([H, S, V]))

    fig, ax = plt.subplots(figsize=(m / 130, z.shape[0] / 130), dpi=130)
    ax.imshow(rgb, interpolation='nearest')
    # coastline: thin black line where land meets sea
    land = (z > base).astype(float)
    ax.contour(land, levels=[0.5], colors='black', linewidths=0.5)
    # a small marker at each region's peak (its highest summit)
    ax.scatter([r % m for r in regions], [r // m for r in regions],
               s=7, c='black', marker='^', linewidths=0.3, edgecolors='white')
    # label the largest regions with their peak name (English, to avoid CJK fonts)
    import matplotlib.patheffects as pe
    area = {int(r): int((terr == r).sum()) for r in regions}
    labelled = [int(r) for r in regions
                if names.get(int(r)) and (names[int(r)].get("name_en")
                                          or names[int(r)].get("name"))]
    biggest = sorted(labelled, key=lambda r: -area[r])[:60]   # biggest NAMED regions
    for r in biggest:
        d = names[r]
        txt = d.get("name_en") or d.get("name")
        ax.text(d["col"], d["row"] - 4, txt, fontsize=5.5, ha="center",
                va="bottom", color="white",
                path_effects=[pe.withStroke(linewidth=1.4, foreground="black")])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"ascent territories  {os.path.basename(path)}  "
                 f"N={len(regions)}  (volume-merged)  lighter = higher")
    fig.tight_layout()
    out = os.path.splitext(os.path.basename(path))[0] + '_territories.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
