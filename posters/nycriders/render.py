"""
Render the nycriders poster at print resolution.

Stripe width is riders per day through a track segment and bubble area is
boardings at a station complex — the same two numbers the interactive map
draws, on the same curves, so the poster and the web map agree about what a
given thickness means. Everything comes from vector sources; nothing is a
MapLibre screenshot.

    python render.py --scale 0.12                    # composition check, ~15 s
    python render.py                                 # full sheet
    python render.py --theme light
    python render.py --crop " -73.978,40.760,4,3"    # 1:1 texture check

Note the leading space inside the quotes on --crop: argparse otherwise reads a
negative longitude as a flag. Same habit as ancestrydots and japanrail.

Outputs into build/, layered so Aseprite can take them straight:
    base_<theme>_<tag>.png     water, coastline                   (opaque)
    lines_<tag>.png            the network                        (alpha)
    bubbles_<theme>_<tag>.png  station bubbles                    (alpha)
    preview_<theme>_<tag>.png  the above composited
    frame_<tag>.json           the framing, for a later layout pass
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from PIL import Image

import chains
import defaults as D
import frame as fr
import palette
import ribbons

HERE = Path(__file__).parent
BUILD = HERE / "build"
STATS = HERE.parent.parent / "riders" / "nycriders" / "stats.json"

THEMES = {
    # Land is the lighter ground and water the darker, as on the interactive
    # map. The separation has to carry the whole basemap here: there is no
    # street grid on the sheet, so the shoreline is the only thing telling a
    # reader which borough they are looking at.
    "dark": {"land": "#15171a", "water": "#0e1520", "coast": "#313b47",
             "bubble": "#e6ebf2", "bubble_alpha": 0.80, "bubble_edge": None},
    "light": {"land": "#f5f2ec", "water": "#dde6ee", "coast": "#9dafbd",
              "bubble": "#ffffff", "bubble_alpha": 0.75,
              "bubble_edge": "#1b2027"},
}


# ── framing ────────────────────────────────────────────────────────────────
def build_frame(args, feats):
    lon, lat = [], []
    for f in feats.values():
        for c in f["coords"]:
            lon.append(c[0])
            lat.append(c[1])
    lon, lat = np.array(lon), np.array(lat)
    w_in, h_in = (float(v) for v in args.sheet.lower().split("x"))
    return fr.fit_sheet(lon, lat, args.rot, w_in, h_in,
                        pad_km=args.pad_km, dpi=args.dpi)


# ── drawing helpers ────────────────────────────────────────────────────────
def new_fig(W, H, facecolor=None):
    dpi = 100.0
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    if facecolor:
        ax.set_facecolor(facecolor)
    return fig, ax, dpi


def finish(fig, ax, f, out, W, H, dpi, facecolor=None, transparent=False):
    ax.set_xlim(f.x0, f.x1)
    ax.set_ylim(f.y0, f.y1)
    fig.savefig(out, dpi=dpi, facecolor=facecolor or "none",
                transparent=transparent)
    plt.close(fig)
    # matplotlib can land a pixel short of the requested size; force it so the
    # layers broadcast against each other
    img = Image.open(out)
    if img.size != (W, H):
        img.convert("RGBA" if transparent else "RGB") \
           .resize((W, H), Image.NEAREST).save(out)


def read_rotated(path, rot, bbox=None):
    g = gpd.read_file(path, bbox=bbox)
    if not len(g):
        return None
    g = g.to_crs(fr.PROJ)
    if rot:
        g = g.rotate(-rot, origin=(0, 0)).to_frame("geometry")
    return g


# ── layers ─────────────────────────────────────────────────────────────────
def render_base(f, W, H, theme, out):
    c = THEMES[theme]
    fig, ax, dpi = new_fig(W, H, c["land"])
    bbox = f.lonlat_bbox()

    water = read_rotated(BUILD / "nyc_water.gpkg", f.rot, bbox)
    # Fill AND stroke in the fill colour: OSM's water file is split into tiles,
    # and an unstroked fill leaves an antialiased hairline along every shared
    # edge that reads as a grid across open water.
    if water is not None:
        water.plot(ax=ax, color=c["water"], ec=c["water"], lw=0.7, zorder=1)

    coast = read_rotated(BUILD / "nyc_coast.gpkg", f.rot, bbox)
    if coast is not None:
        coast.plot(ax=ax, color=c["coast"], lw=max(W / 9000, 0.3), zorder=2)

    finish(fig, ax, f, out, W, H, dpi, facecolor=c["land"])


def build_ribbons(f, args, feats):
    """Every stripe as a polyline in projected metres, in draw order, with the
    linewidth each is drawn at (inches on the sheet).

    The automatic placement in ribbons.py gets baked down into per-route control
    points by chains.py, and the sheet is drawn from those — so a hand edit and
    the automatic result are the same kind of thing, and a route's line cannot
    come apart at a junction. See chains.py.
    """
    runs, node_runs, graph = ribbons.build_runs(feats)

    vmax = max(fv["value"] for fv in feats.values())
    lo, hi = args.min_width, args.max_width
    widths_in = {k: 0.0 if fv["value"] < 0.5
                 else lo + (hi - lo) * (fv["value"] / vmax) ** args.width_gamma
                 for k, fv in feats.items()}
    widths_m = {k: w * f.m_per_inch for k, w in widths_in.items()}
    gap_m = args.gap * hi * f.m_per_inch

    rel = ribbons.fan_relative(runs, widths_m, gap_m)
    centres = ribbons.solve_centres(runs, node_runs, rel, reg=args.centre_reg)
    offs = ribbons.absolute(rel, centres)

    # project the shared graph once, round its corners, then read each run's
    # geometry back out of it — smoothing the graph rather than the runs is what
    # keeps two runs meeting at a node still meeting there
    nodes = list(graph["coord"].keys())
    ll = np.array([graph["coord"][n] for n in nodes], float)
    px, py = fr.project(ll[:, 0], ll[:, 1], f.rot)
    xy_node = {n: np.array([px[i], py[i]]) for i, n in enumerate(nodes)}
    if args.smooth_passes > 0:
        xy_node = ribbons.smooth_nodes(
            xy_node, graph, ribbons.node_half_widths(runs, offs),
            passes=args.smooth_passes, cap=args.smooth_cap)
    xy = [np.array([xy_node[n] for n in run["nodes"]]) for run in runs]
    joints = ribbons.joint_normals(runs, node_runs, xy)

    nudges = (ribbons.load_nudges(HERE / "nudge" / "nudges.json") if args.nudges
              else {"move": {}, "add": set(), "drop": set()})
    forced, dropped = ribbons.nudge_sets(nudges)
    cs = chains.build(runs, offs, xy, joints, forced=forced, dropped=dropped)
    report = ribbons.edit_report(nudges, cs)
    if report:
        print(report)

    paths, cols, lws, prio = [], [], [], []
    for c in cs:
        for pts, w_m in chains.draw_pieces(c, nudges['move']):
            paths.append(pts)
            cols.append(c["color"])
            lws.append(w_m / f.m_per_inch)
            prio.append(c["priority"])

    # priority ascending, ties broken by width so the busier stripe of a pair
    # lands on top — the same order the web map draws in
    order = np.lexsort((np.asarray(lws), np.asarray(prio)))
    return ([paths[i] for i in order], [cols[i] for i in order],
            np.asarray(lws)[order], len(cs))


def render_lines(f, W, H, scale, out, args, feats):
    t0 = time.time()
    paths, cols, lw_in, n_runs = build_ribbons(f, args, feats)
    pt = lw_in * args.dpi * scale * 72.0 / 100.0

    fig, ax, dpi = new_fig(W, H)
    ax.add_collection(LineCollection(
        paths, colors=cols, linewidths=pt, capstyle="round",
        joinstyle="round", alpha=args.line_alpha, zorder=1))
    finish(fig, ax, f, out, W, H, dpi, transparent=True)
    print(f"  {len(paths):,} pieces over {n_runs:,} route chains, "
          f"{lw_in.min() * 1000:.1f}–{lw_in.max() * 1000:.1f} mil "
          f"({time.time() - t0:.1f}s)")


def render_bubbles(f, W, H, scale, out, args, data):
    c = THEMES[args.theme_one]
    st = ribbons.station_totals(data, args.hour)
    lon = np.array([s[0] for s in st])
    lat = np.array([s[1] for s in st])
    val = np.array([s[2] for s in st])
    rx, ry = fr.project(lon, lat, f.rot)

    # area proportional to boardings, as on the interactive map
    r_in = args.bubble_min + (args.bubble_max - args.bubble_min) * np.sqrt(val / val.max())
    pt = r_in * args.dpi * scale * 72.0 / 100.0

    fig, ax, dpi = new_fig(W, H)
    face = to_rgba(c["bubble"], c["bubble_alpha"])
    kw = {"linewidths": 0, "edgecolors": "none"}
    if c["bubble_edge"]:
        kw = {"linewidths": 0.004 * args.dpi * scale * 72 / 100,
              "edgecolors": [to_rgba(c["bubble_edge"], 0.55)]}
    ax.scatter(rx, ry, s=(pt ** 2) * np.pi, facecolors=[face], zorder=1, **kw)
    finish(fig, ax, f, out, W, H, dpi, transparent=True)
    print(f"  {len(st):,} station bubbles, largest r={r_in.max() * 25.4:.2f} mm")


def composite(base, layers, out):
    im = Image.open(base).convert("RGBA")
    for p in layers:
        if p.exists():
            im = Image.alpha_composite(im, Image.open(p).convert("RGBA"))
    im.convert("RGB").save(out)
    return out


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=D.SHEET, help="inches, WxH")
    ap.add_argument("--rot", type=float, default=fr.DEFAULT_ROT,
                    help="clockwise tilt in degrees; 0 is north-up")
    ap.add_argument("--pad-km", type=float, default=D.PAD_KM,
                    help="slack around the network on the binding axis")
    ap.add_argument("--dpi", type=int, default=D.DPI)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="render at a fraction of full size, for quick checks")
    ap.add_argument("--crop", type=str, default=None,
                    help="'lon,lat,w_in,h_in' — 1:1 window, leading space")
    ap.add_argument("--theme", default="dark",
                    choices=["dark", "light", "both"])
    ap.add_argument("--hour", type=int, default=None,
                    help="0-23 for one hour; default is the whole day")
    ap.add_argument("--tag", default="v1")

    ap.add_argument("--min-width", type=float, default=D.MIN_WIDTH,
                    help="thinnest stripe, inches")
    ap.add_argument("--max-width", type=float, default=D.MAX_WIDTH,
                    help="thickest stripe, inches")
    ap.add_argument("--width-gamma", type=float, default=D.WIDTH_GAMMA)
    ap.add_argument("--gap", type=float, default=D.GAP,
                    help="gap between stripes, as a fraction of --max-width")
    ap.add_argument("--line-alpha", type=float, default=1.0)
    ap.add_argument("--no-nudges", dest="nudges", action="store_false",
                    help="ignore nudge/nudges.json, to see the unedited geometry")
    ap.add_argument("--smooth-passes", type=int, default=D.SMOOTH_PASSES,
                    help="corner-rounding passes on the shared centrelines; 0 "
                         "draws the raw GTFS shapes and lets tight curves fold")
    ap.add_argument("--smooth-cap", type=float, default=D.SMOOTH_CAP,
                    help="how far a vertex may move, as a fraction of the "
                         "widest bundle through it")
    ap.add_argument("--centre-reg", type=float, default=D.CENTRE_REG,
                    help="how hard a bundle is pulled back onto its own track; "
                         "lower lets a route hold one offset over more of its "
                         "length, at the cost of the bundle sitting off-centre")
    ap.add_argument("--bubble-min", type=float, default=D.BUBBLE_MIN,
                    help="smallest bubble radius, inches")
    ap.add_argument("--bubble-max", type=float, default=D.BUBBLE_MAX,
                    help="largest bubble radius, inches")
    ap.add_argument("--no-bubbles", action="store_true")
    args = ap.parse_args()

    BUILD.mkdir(exist_ok=True)
    t0 = time.time()
    data, feats = ribbons.load_features(STATS, args.hour)
    print(f"stats.json: {len(feats):,} route/stop-pair features "
          f"({time.time() - t0:.1f}s)")

    f = build_frame(args, feats)
    scale = args.scale
    tag = args.tag

    if args.crop:
        lon, lat, w_in, h_in = (float(v) for v in args.crop.strip().split(","))
        cx, cy = fr.project(lon, lat, f.rot)
        half_w, half_h = f.m_per_inch * w_in / 2, f.m_per_inch * h_in / 2
        f = fr.Frame(f.rot, cx - half_w, cx + half_w, cy - half_h, cy + half_h,
                     w_in, f.dpi)
        scale = 1.0
        tag = f"{args.tag}_crop"

    W, H = f.size_px(scale)
    print(f"sheet {f.width_in:.2f} x {f.height_in:.2f} in, {W} x {H} px, "
          f"{f.km_per_inch:.2f} km/inch, 1 px = {f.m_per_inch / f.dpi / scale:.1f} m")
    (BUILD / f"frame_{tag}.json").write_text(json.dumps(f.to_dict(), indent=1))

    lines_png = BUILD / f"lines_{tag}.png"
    render_lines(f, W, H, scale, lines_png, args, feats)

    themes = ["dark", "light"] if args.theme == "both" else [args.theme]
    for theme in themes:
        args.theme_one = theme
        base_png = BUILD / f"base_{theme}_{tag}.png"
        render_base(f, W, H, theme, base_png)
        layers = [lines_png]
        if not args.no_bubbles:
            bub = BUILD / f"bubbles_{theme}_{tag}.png"
            render_bubbles(f, W, H, scale, bub, args, data)
            layers.append(bub)        # bubbles over the lines, as on the web map
        out = composite(base_png, layers, BUILD / f"preview_{theme}_{tag}.png")
        print(f"  -> {out.name}")

    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
