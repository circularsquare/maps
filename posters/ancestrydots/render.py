"""
Render the ancestrydots poster from the baked dot array.

Renders an *index* raster (one uint16 ancestry index per pixel) rather than RGB,
so the hand-tuned palette in ancestry_colors.csv can be re-applied without
re-rendering, and so the output stays palette-limited — which is what keeps an
indexed PNG under Lumaprints' 100 MB upload cap.

    python render.py --scale 0.2                 # quick preview
    python render.py --scale 0.2 --theme light   # light-mode test
    python render.py                             # full 36x20.1in @300ppi

Outputs into build/:
    base_<theme>_<tag>.png    land / ocean / lakes
    dots_<theme>_<tag>.png    dots on transparency
    preview_<theme>_<tag>.png the two composited
    index_<tag>.npy           raw index raster, for recolouring
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pyproj import CRS, Transformer

from dotraster import splat, over

HERE = Path(__file__).parent
BUILD = HERE / "build"
NE = HERE.parent.parent / "data" / "ne_10m_lakes"

# Albers equal-area conic, EPSG:5070-style parameters. See posters/NOTES.md.
MAIN = CRS.from_proj4("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 "
                      "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
# Anita's bounds: Key West/Brownsville -> Edmonton, Campbell River -> Cape Breton
LAT_S, LAT_N, LON_W, LON_E = 24.55, 53.55, -125.24, -59.75

EMPTY = 0xFFFF

THEMES = {
    "dark":  {"land": "#0d0d0f", "water": "#141c24", "coast": "#2a3038"},
    "light": {"land": "#f2ede2", "water": "#d6e3ec", "coast": "#b9c6d0"},
}


def extent(tf):
    lons = np.linspace(LON_W, LON_E, 400)
    lats = np.linspace(LAT_S, LAT_N, 400)
    x, y = tf.transform(
        np.concatenate([lons, lons, np.full(400, LON_W), np.full(400, LON_E)]),
        np.concatenate([np.full(400, LAT_S), np.full(400, LAT_N), lats, lats]))
    return x.min(), x.max(), y.min(), y.max()


def render_base(W, H, box, theme, out):
    x0, x1, y0, y1 = box
    c = THEMES[theme]
    dpi = 100.0
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(c["land"])
    ax.set_axis_off()

    ocean = gpd.read_file(NE / "ne_10m_ocean.shp").to_crs(MAIN)
    lakes = gpd.read_file(NE / "ne_10m_lakes.shp").to_crs(MAIN)
    lw = max(W / 6000, 0.3)
    ocean.plot(ax=ax, color=c["water"], ec=c["coast"], lw=lw, zorder=1)
    lakes.plot(ax=ax, color=c["water"], ec=c["coast"], lw=lw * 0.7, zorder=2)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    fig.savefig(out, dpi=dpi, facecolor=c["land"])
    plt.close(fig)
    # matplotlib can land a pixel short of the requested size, so force it
    img = Image.open(out).convert("RGB")
    if img.size != (W, H):
        img = img.resize((W, H), Image.NEAREST)
        img.save(out)
    return np.array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-in", type=float, default=36.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="fraction of full size, for quick previews")
    ap.add_argument("--theme", choices=[*sorted(THEMES), "both"], default="dark",
                    help="'both' reuses the one (expensive) dot pass for each")
    ap.add_argument("--layers", action="store_true",
                    help="also write the dots-on-transparency layer for Aseprite")
    ap.add_argument("--view-width", type=int, default=0,
                    help="also write a downscaled copy this wide. Off by "
                         "default: shrinking the sheet averages each 1-2px dot "
                         "with ~20 background pixels and washes it out, so the "
                         "result misrepresents the print. Use --crop instead.")
    ap.add_argument("--dot-radius", type=float, default=1.0,
                    help="dot radius in px at FULL resolution")
    ap.add_argument("--ss", type=int, default=3,
                    help="supersample factor for antialiasing; 1 = hard edges")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", metavar="LON,LAT,W_IN,H_IN",
                    help="render only a W_IN x H_IN inch window of the final "
                         "poster centred on LON,LAT, at full --dpi. Use this to "
                         "check print texture without rendering the whole sheet.")
    args = ap.parse_args()

    t0 = time.time()
    tf = Transformer.from_crs("EPSG:4326", MAIN, always_xy=True)
    x0, x1, y0, y1 = box = extent(tf)
    aspect = (x1 - x0) / (y1 - y0)

    km_in = (x1 - x0) / 1000 / args.width_in
    m_per_px_full = (x1 - x0) / (args.width_in * args.dpi)

    if args.crop:
        clon, clat, cw, ch = (float(v) for v in args.crop.split(","))
        cx, cy = tf.transform(clon, clat)
        half_w, half_h = cw * args.dpi * m_per_px_full / 2, ch * args.dpi * m_per_px_full / 2
        x0, x1, y0, y1 = cx - half_w, cx + half_w, cy - half_h, cy + half_h
        box = (x0, x1, y0, y1)
        W, H = int(round(cw * args.dpi)), int(round(ch * args.dpi))
        radius = args.dot_radius
        tag = f"crop{clon:g}_{clat:g}_{W}x{H}"
        print(f"CROP {cw} x {ch} in of the {args.width_in:.0f}in sheet at "
              f"{clon:g},{clat:g} — {W} x {H} px at {args.dpi} dpi")
    else:
        W = int(round(args.width_in * args.dpi * args.scale))
        H = int(round(W / aspect))
        radius = max(args.dot_radius * args.scale, 0.5)
        tag = f"{W}x{H}"
        print(f"{args.width_in:.0f} x {args.width_in / aspect:.2f} in, "
              f"{W} x {H} px, scale {args.scale:g}, dot r={radius:.2f}px")
    print(f"  {km_in:.0f} km/inch, 1 px = {m_per_px_full:.0f} m at full res")

    palette = json.loads((BUILD / "palette.json").read_text(encoding="utf-8"))
    lut = np.zeros((EMPTY + 1, 3), dtype=np.uint8)
    for i, e in enumerate(palette):
        h = e["color"].lstrip("#")
        lut[i] = [int(h[j:j + 2], 16) for j in (0, 2, 4)]

    d = np.load(BUILD / "dots_na.npz")
    lon, lat, idx = d["lon"], d["lat"], d["idx"]
    print(f"  {len(lon):,} dots loaded ({time.time() - t0:.1f}s)")

    X, Y = tf.transform(lon, lat)
    px = (X - x0) / (x1 - x0) * W
    py = (y1 - Y) / (y1 - y0) * H
    keep = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    print(f"  {keep.sum():,} inside the crop, {(~keep).sum():,} outside")

    rng = np.random.default_rng(args.seed)
    rgb, alpha = splat(px[keep], py[keep], idx[keep], W, H, radius, lut,
                       ss=args.ss, rng=rng)
    print(f"  splat done, {(alpha > 0).mean() * 100:.1f}% coverage "
          f"({time.time() - t0:.1f}s)")

    if args.layers:
        Image.fromarray(np.dstack([rgb, alpha])).save(BUILD / f"dots_{tag}.png")
        print(f"  wrote dots_{tag}.png ({time.time() - t0:.1f}s)")

    for theme in (sorted(THEMES) if args.theme == "both" else [args.theme]):
        base = render_base(W, H, box, theme, BUILD / f"base_{theme}_{tag}.png")
        print(f"  base_{theme} done ({time.time() - t0:.1f}s)")

        comp = over(rgb, alpha, base)
        out = BUILD / f"preview_{theme}_{tag}.png"
        Image.fromarray(comp).save(out)
        print(f"  wrote {out.name} ({out.stat().st_size / (1 << 20):.1f} MB, "
              f"{time.time() - t0:.1f}s)")

        if args.view_width and W > args.view_width:
            vw = args.view_width
            vh = int(round(H * vw / W))
            view = BUILD / f"view_{theme}_{tag}.png"
            Image.fromarray(comp).resize((vw, vh), Image.LANCZOS).save(view)
            print(f"  wrote {view.name} ({vw}x{vh})")

    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
