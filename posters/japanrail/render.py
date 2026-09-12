"""
Render the national japanrail map at print resolution.

Line width is passenger throughput per segment, on the same knee curve the
interactive map uses, so the poster and the web map agree about what a given
thickness means. Everything is drawn from vector sources — no MapLibre
screenshot — so the sheet can go to 300 ppi and the basemap can be as quiet as
a printed map wants.

    python render.py --scale 0.12                  # composition check, ~20 s
    python render.py --rot 25 --scale 0.12         # try another tilt
    python render.py --theme light                 # full sheet, ~40 s
    python render.py --crop " 139.75,35.69,6,4"    # 1:1 texture check, ~5 s

Outputs into build/, always layered so Aseprite can take them straight:
    base_<theme>_<tag>.png     ocean, lakes, coastline            (opaque)
    lines_<tag>.png            the rail network                   (alpha)
    bubbles_<tag>.png          station bubbles, with --bubbles    (alpha)
    preview_<theme>_<tag>.png  the above composited
    frame_<tag>.json           the framing, for layout.py/insets.py
"""

from __future__ import annotations

import argparse
import json
import math
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

import frame as fr
import glowfx
import palette

HERE = Path(__file__).parent
BUILD = HERE / "build"
SEGMENTS = HERE.parent.parent / "riders" / "japanriders" / "data" / "segments.geojson"
STATIONS = HERE.parent.parent / "riders" / "japanriders" / "data" / "stations.geojson"

THEMES = {
    # Land is the lighter ground and water the darker, as on the interactive
    # map. An earlier pass had them 6 levels apart and Tokyo Bay was invisible;
    # at this separation the bays read without the sea becoming a shape of its
    # own that competes with the network.
    #
    # The HUES are the way round they are in the light theme: land neutral,
    # water blue. The first dark theme had it backwards — a blue-grey land
    # against a neutral black sea, which reads as the sea being the ground and
    # the land being the water.
    "dark":  {"land": "#16181b", "water": "#111c2a", "coast": "#36414e",
              "lake": "#111c2a", "casing": "#f2f6fb",
              "bubble": "#e8eef6", "bubble_edge": None, "bubble_alpha": None},
    # On cream a dark bubble reads as a hole punched in whatever line it sits
    # on. White with a translucent dark rim reads as a bead on top of the line,
    # which is what it does on the dark sheet.
    "light": {"land": "#f4f1ea", "water": "#dae5ee", "coast": "#9fb0be",
              "lake": "#dae5ee", "casing": "#ffffff",
              "bubble": "#ffffff", "bubble_edge": "#151a20",
              # white on cream needs more body than white on near-black
              "bubble_alpha": 0.62},
}

# ── width curve, ported from index.html ────────────────────────────────────
# Linear below a knee throughput and logarithmic above it, slopes matched at
# the knee so there is no kink. The knee is where the scale stops rewarding
# extra passengers linearly; at the default 1e6 that is just under the busiest
# segment, so the whole national network sits on the linear branch and the
# handful of Tokyo trunk segments compress.
W_CEIL = 1.3e6
LOG_CEIL = math.log10(W_CEIL)
LOG10E = math.log10(math.e)


def width_t(d, knee):
    """0..1 position on the width curve. d may be an array."""
    knee_x = 10.0 ** knee
    a = 1.0 / (LOG_CEIL - knee + LOG10E)
    b = a * (LOG10E - knee)
    c = a * LOG10E / knee_x
    v = np.clip(np.asarray(d, float), 1.0, W_CEIL)
    t = np.where(v >= knee_x, a * np.log10(v) + b, c * v)
    return np.clip(t, 0.0, 1.0)


# ── data ───────────────────────────────────────────────────────────────────
def load_segments(field):
    gj = json.loads(SEGMENTS.read_text(encoding="utf-8"))
    lines, dens, props = [], [], []
    for f in gj["features"]:
        p = f["properties"]
        v = p.get(field)
        if v is None:
            v = p.get("density")
        if v is None:
            continue
        g = f["geometry"]
        parts = ([g["coordinates"]] if g["type"] == "LineString"
                 else g["coordinates"])
        for part in parts:
            if len(part) < 2:
                continue
            lines.append(np.asarray(part, float))
            dens.append(float(v))
            props.append(p)
    return lines, np.asarray(dens), props


def load_stations():
    gj = json.loads(STATIONS.read_text(encoding="utf-8"))
    xy = np.array([f["geometry"]["coordinates"] for f in gj["features"]], float)
    riders = np.array([f["properties"]["riders"] for f in gj["features"]], float)
    return xy, riders


# ── drawing ────────────────────────────────────────────────────────────────
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
    # matplotlib can land a pixel short of the requested size (a 94.5 in
    # figure at 100 dpi comes out 9449, not 9450); force it so the layers
    # broadcast against each other.
    img = Image.open(out)
    if img.size != (W, H):
        img.convert("RGBA" if transparent else "RGB") \
           .resize((W, H), Image.NEAREST).save(out)


def read_rotated(path, rot, bbox=None):
    """Read a layer, reproject and rotate it. Returns None when the bbox holds
    nothing — geopandas' .plot() on an empty frame in a geographic CRS dies
    computing an aspect ratio from a NaN latitude, which is what a tight inset
    with no lakes in it does (Fukuoka)."""
    g = gpd.read_file(path, bbox=bbox)
    if not len(g):
        return None
    return g.to_crs(fr.PROJ).rotate(-rot, origin=(0, 0)).to_frame("geometry")


# The COD adm1 file is Japan's own claim, so Hokkaido's geometry carries the
# Southern Kurils / Northern Territories as separate island polygons: Iturup,
# Kunashir, Shikotan and the Habomai group. The map has no data there and takes
# no position on them, so they are dropped from the land mask and blacked out
# along with Korea and Sakhalin. Selected by CENTROID rather than by bounds:
# Kunashir's western tip (145.40E) overlaps Yururi and Moyururi (to 145.35E),
# which are undisputed Japanese islands off Nemuro and must stay.
DISPUTED_CENTROID = (145.6, 43.3)   # lon >= , lat >=


def japan_land(rot, simplify_m=200.0):
    """Japan's land as one rotated geometry, minus the disputed islands."""
    from shapely.ops import unary_union

    g = gpd.read_file(BUILD / "jp_pref.gpkg")
    keep = []
    for geom in g.geometry:
        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for p in parts:
            c = p.centroid
            if c.x >= DISPUTED_CENTROID[0] and c.y >= DISPUTED_CENTROID[1]:
                continue
            keep.append(p)
    s = gpd.GeoSeries(keep, crs=g.crs).to_crs(fr.PROJ)
    return unary_union(list(s.rotate(-rot, origin=(0, 0)).simplify(simplify_m)))


def veil_polygon(f, buffer_km, simplify_m=200.0):
    """The frame minus Japan's land — everything the map is not about.

    Korea sits in the top-left corner of any tilted framing and Sakhalin in the
    top-right, and with no rail data they would otherwise render as the same
    land tone as Japan, reading as 'nobody travels here'. Painting them out
    says out-of-scope without drawing a border.
    """
    from shapely.geometry import box

    land = japan_land(f.rot, simplify_m).buffer(buffer_km * 1000.0)
    pad = (f.x1 - f.x0) * 0.02
    rect = box(f.x0 - pad, f.y0 - pad, f.x1 + pad, f.y1 + pad)
    return gpd.GeoSeries([rect.difference(land)], crs=fr.PROJ)


def render_base(f, W, H, theme, out, prefectures=False, foreign="veil",
                veil_alpha=0.72, veil_buffer_km=2.0):
    c = THEMES[theme]
    fig, ax, dpi = new_fig(W, H, c["land"])
    bbox = f.lonlat_bbox()
    lw = max(W / 7000, 0.25)

    water = read_rotated(BUILD / "jp_water.gpkg", f.rot, bbox)
    # Fill only, and stroke in the fill colour: OSM's water file is split into
    # tiles, and an unstroked fill leaves an antialiased hairline along every
    # shared edge that reads as a grid across open sea.
    if water is not None:
        water.plot(ax=ax, color=c["water"], ec=c["water"], lw=0.7, zorder=1)

    lakes = read_rotated(BUILD / "jp_lakes.gpkg", f.rot, bbox)
    if lakes is not None:
        lakes.plot(ax=ax, color=c["lake"], ec=c["lake"], lw=0.5, zorder=2)

    if prefectures:
        pref = read_rotated(BUILD / "jp_pref.gpkg", f.rot, bbox)
        if pref is not None:
            pref.boundary.plot(ax=ax, color=c["coast"], lw=lw * 0.5, alpha=0.5,
                               zorder=3)

    coast = read_rotated(BUILD / "jp_coast.gpkg", f.rot, bbox)
    if coast is not None:
        coast.plot(ax=ax, color=c["coast"], lw=lw, zorder=4)

    if foreign != "show":
        veil = veil_polygon(f, veil_buffer_km)
        veil.plot(ax=ax, color=c["water"], ec="none",
                  alpha=1.0 if foreign == "hide" else veil_alpha, zorder=5)

    finish(fig, ax, f, out, W, H, dpi, facecolor=c["land"])
    return np.array(Image.open(out).convert("RGB"))


def render_lines(f, W, H, scale, out, args, theme):
    c = THEMES[theme]
    lines, dens, props = load_segments(args.field)
    t0 = time.time()

    xs = [fr.project(g[:, 0], g[:, 1], f.rot) for g in lines]
    verts = [np.column_stack(p) for p in xs]
    print(f"  {len(verts):,} paths, {sum(len(v) for v in verts):,} vertices "
          f"projected ({time.time() - t0:.1f}s)")

    t = width_t(dens, args.knee)
    w_in = args.min_width + (args.max_width - args.min_width) * t
    # matplotlib linewidth is in points and the figure is saved at 100 dpi,
    # so a point is 100/72 px; the sheet is args.dpi, hence the scale factor.
    pt = w_in * args.dpi * scale * 72.0 / 100.0

    if args.color == "mono":
        cols = [args.mono_color] * len(verts)
    elif args.color == "operator":
        cols = [palette.operator_color(p["op"]) for p in props]
    else:
        cols = [palette.line_color(p["op"], p["line"]) for p in props]

    # Thickest LAST, so the busy lines sit on top of the quiet ones. This
    # matches the interactive map and is the right way round for a flow map:
    # thickness is the message, so the trunk should not be interrupted by every
    # branch that crosses it. Drawing widest-first was tried and reads worse —
    # it protects rural branches at the cost of chopping the Tokaido into
    # segments wherever something crosses.
    order = np.argsort(pt)
    verts = [verts[i] for i in order]
    cols = [cols[i] for i in order]
    pt = pt[order]

    fig, ax, dpi = new_fig(W, H)
    if args.casing > 0:
        cpt = pt + args.casing * args.dpi * scale * 72.0 / 100.0
        ax.add_collection(LineCollection(
            verts, colors=c["casing"], linewidths=cpt, capstyle="round",
            joinstyle="round", alpha=args.casing_alpha, zorder=1))
    ax.add_collection(LineCollection(
        verts, colors=cols, linewidths=pt, capstyle="round",
        joinstyle="round", alpha=args.line_alpha, zorder=2))
    finish(fig, ax, f, out, W, H, dpi, transparent=True)
    print(f"  lines drawn, {w_in.min() * 1000:.1f}–{w_in.max() * 1000:.1f} mil "
          f"({time.time() - t0:.1f}s)")


def render_bubbles(f, W, H, scale, out, args, theme):
    c = THEMES[theme]
    xy, riders = load_stations()
    keep = riders >= args.min_riders
    xy, riders = xy[keep], riders[keep]
    rx, ry = fr.project(xy[:, 0], xy[:, 1], f.rot)
    # area proportional to ridership, as on the interactive map
    r_in = np.sqrt(riders) * args.bubble_scale
    pt = r_in * args.dpi * scale * 72.0 / 100.0

    fig, ax, dpi = new_fig(W, H)
    # Face and edge carry their own alphas, so scatter's own `alpha` is left
    # off — it would apply one value to both.
    face = to_rgba(c["bubble"], c.get("bubble_alpha") or args.bubble_alpha)
    edge = c.get("bubble_edge")
    kw = {"linewidths": 0, "edgecolors": "none"}
    if edge:
        kw = {"linewidths": args.bubble_edge_width * args.dpi * scale * 72 / 100,
              "edgecolors": [to_rgba(edge, args.bubble_edge_alpha)]}
    ax.scatter(rx, ry, s=(pt ** 2) * np.pi, facecolors=[face], zorder=1, **kw)
    finish(fig, ax, f, out, W, H, dpi, transparent=True)
    print(f"  {keep.sum():,} bubbles >= {args.min_riders:,} riders, "
          f"largest r={r_in.max() * 25.4:.2f} mm")


def composite(base, layers, out):
    im = Image.fromarray(base).convert("RGBA")
    for p in layers:
        if p.exists():
            im = Image.alpha_composite(im, Image.open(p).convert("RGBA"))
    im.convert("RGB").save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rot", type=float, default=fr.DEFAULT_ROT,
                    help="clockwise rotation in degrees; 0 is north-up, 51 is "
                         "the minimum-bounding-box tilt")
    ap.add_argument("--width-in", type=float, default=36.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="fraction of full size, for quick previews")
    ap.add_argument("--theme", choices=[*sorted(THEMES), "both"], default="dark")
    ap.add_argument("--pad-km", default="40,40,40,40",
                    help="margin west,east,south,north of the rail network, in "
                         "km, measured in the ROTATED frame (so west is the "
                         "left edge of the sheet, not compass west)")
    # Extra sheet, in INCHES, added after the fit at the same km/inch — so the
    # map does not shrink, the paper grows. --extend-right 2.6 is what buys the
    # Tokyo inset a column clear of Japan: land reaches x=27.38 in across the
    # rows the box occupies, and a 10.45 in box needs to start right of that.
    ap.add_argument("--extend-right", type=float, default=2.6)
    ap.add_argument("--extend-left", type=float, default=0.0)
    ap.add_argument("--extend-top", type=float, default=0.0)
    ap.add_argument("--extend-bottom", type=float, default=0.0)
    ap.add_argument("--field", default="density",
                    choices=["density", "d_base", "d_census"],
                    help="which throughput figure to draw")
    # Knee 5 (100,000/day) on the national sheet. 4.3 was tried first and
    # spread the low end further, but Anita reads the flatter curve as less
    # honest: it makes a 40,000/day branch look like a serious trunk. At 5 the
    # median segment is a third of the width it had, and the ranking between a
    # main line and a country branch is much closer to the real ratio. Rural
    # lines survive on the --min-width floor rather than on curve shape.
    # The insets run a different knee — see insets.py.
    ap.add_argument("--knee", type=float, default=5.0,
                    help="log10 of the knee throughput; lower is more "
                         "logarithmic. 6 = 1M is the web map's default.")
    # The floor is deliberately fine: 0.0055 in is 1.65 px at 300 ppi, which is
    # 0.14 mm on a 38.6 in sheet and 0.11 mm if it is printed at 30. That is
    # about as thin as an inkjet will hold on fine art paper, so it is the
    # number to revisit first if the proof loses the rural network. Widening
    # the gap between floor and ceiling is what makes the busy lines read as
    # busy rather than merely thicker.
    ap.add_argument("--min-width", type=float, default=0.0061,
                    help="inches — the quietest rural line")
    ap.add_argument("--max-width", type=float, default=0.145,
                    help="inches — the busiest segment. At 51 km/inch this is "
                         "7.5 km wide, so Tokyo's core fills solid.")
    ap.add_argument("--color", choices=["line", "operator", "mono"],
                    default="line")
    ap.add_argument("--mono-color", default="#7fd8ff")
    # Opaque. 0.70 was tried and rejected: a line's own segments overlap each
    # other at every join and at every doubled-back stretch, so a partly
    # transparent line goes blotchy along its own length — the transparency
    # reads as a rendering fault rather than as depth.
    ap.add_argument("--line-alpha", type=float, default=1.0)
    ap.add_argument("--casing", type=float, default=0.0,
                    help="inches of casing on EACH side; 0 turns it off")
    ap.add_argument("--casing-alpha", type=float, default=0.9)
    # The interactive map had a glow and dropped it: too slow per frame, and
    # drawn per segment it notched at every joint. Blurring the finished raster
    # has neither problem — see glowfx.py.
    # Defaults settled on a 1:1 Tokyo crop. A first pass at 0.85/0.45 read as a
    # coloured haze lying over the whole Kanto plain — the wide alpha of forty
    # lines simply adds to 1 and the ground stops being dark. Backing the wide
    # pass down to 0.18 and capping at 0.8 keeps it a rim on each line.
    ap.add_argument("--glow", type=float, default=0.018,
                    help="inches — radius of the tight glow pass; 0 turns the "
                         "whole effect off")
    ap.add_argument("--glow-strength", type=float, default=0.30)
    ap.add_argument("--glow-wide", type=float, default=4.0,
                    help="the second, broader pass, as a multiple of --glow")
    ap.add_argument("--glow-wide-strength", type=float, default=0.09)
    ap.add_argument("--glow-cap", type=float, default=0.80,
                    help="ceiling on glow alpha; below 1 keeps Tokyo from "
                         "burning out to a solid coloured disc")
    ap.add_argument("--no-bubbles", dest="bubbles", action="store_false",
                    help="leave the station bubbles off the national map")
    # Half the inset scale (insets.BUBBLE_SCALE is 9.0e-5). A third was tried
    # first and the stations barely registered at national scale; at a half
    # the big cities read as clusters without the country becoming a string of
    # beads with no line left showing.
    ap.add_argument("--bubble-scale", type=float, default=4.5e-5,
                    help="inches of radius per sqrt(rider)")
    ap.add_argument("--bubble-alpha", type=float, default=0.45)
    ap.add_argument("--bubble-edge-alpha", type=float, default=0.30,
                    help="only used by themes that define bubble_edge")
    ap.add_argument("--bubble-edge-width", type=float, default=0.005,
                    help="inches")
    ap.add_argument("--min-riders", type=float, default=3000)
    ap.add_argument("--prefectures", action="store_true")
    ap.add_argument("--foreign", choices=["veil", "hide", "show"],
                    default="hide",
                    help="what to do with Korea, Sakhalin and the rest of the "
                         "land that has no data on this map")
    ap.add_argument("--veil-alpha", type=float, default=0.72)
    ap.add_argument("--veil-buffer-km", type=float, default=2.0,
                    help="keep the wash this far off Japan's own coastline, so "
                         "a generalised admin boundary cannot nibble it")
    ap.add_argument("--crop", metavar="LON,LAT,W_IN,H_IN",
                    help="render only a W_IN x H_IN inch window of the sheet, "
                         "centred on LON,LAT, at full --dpi. A reduced --scale "
                         "preview thins every line by the same factor and so "
                         "understates the print badly; this is the only honest "
                         "way to judge texture.")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    BUILD.mkdir(exist_ok=True)
    t0 = time.time()

    # frame from the rail network's own vertices, mainland only — Okinawa is
    # 1,000 km off the southwest end and gets a relocated float instead.
    lines, dens, props = load_segments(args.field)
    pts = np.vstack(lines)
    pts = pts[pts[:, 1] >= fr.OKINAWA_MAX_LAT]
    pad = tuple(float(v) for v in args.pad_km.split(","))
    f = fr.fit(pts[:, 0], pts[:, 1], args.rot, args.width_in, pad, args.dpi)
    mpi = f.m_per_inch                       # fixed before the sheet grows
    f.x1 += args.extend_right * mpi
    f.x0 -= args.extend_left * mpi
    f.y1 += args.extend_top * mpi
    f.y0 -= args.extend_bottom * mpi
    f.width_in = round((f.x1 - f.x0) / mpi, 4)

    print(f"rot {args.rot:g} deg — sheet {f.width_in:.2f} x {f.height_in:.2f} in "
          f"(aspect {f.width_in / f.height_in:.2f})")
    print(f"  {f.km_per_inch:.1f} km/inch, 1 px = "
          f"{f.m_per_inch / args.dpi:.0f} m at {args.dpi} ppi")
    bb = f.lonlat_bbox(0)
    print(f"  covers lon {bb[0]:.2f}..{bb[2]:.2f}, lat {bb[1]:.2f}..{bb[3]:.2f}"
          + ("   <- reaches Korea/Russia, which have no data here"
             if bb[0] < 129.5 or bb[3] > 46.0 else ""))

    scale = args.scale
    if args.crop:
        clon, clat, cw, ch = (float(v) for v in args.crop.split(","))
        cx, cy = fr.project(clon, clat, args.rot)
        mpi = f.m_per_inch
        f = fr.Frame(args.rot, cx - cw / 2 * mpi, cx + cw / 2 * mpi,
                     cy - ch / 2 * mpi, cy + ch / 2 * mpi, cw, args.dpi)
        scale = 1.0
        print(f"  CROP {cw} x {ch} in at {clon:g},{clat:g} — 1:1 at {args.dpi} ppi")

    W, H = f.size_px(scale)
    tag = args.tag or (f"crop{args.crop.replace(',', '_').strip()}_rot{args.rot:g}"
                       if args.crop else f"rot{args.rot:g}_{W}x{H}")
    print(f"  {W} x {H} px")

    fd = f.to_dict()
    fd["extend"] = {"left": args.extend_left, "right": args.extend_right,
                    "top": args.extend_top, "bottom": args.extend_bottom}
    # compose.py draws the thickness ladder from these, so the legend cannot
    # disagree with the map about what a width means
    fd["width_scale"] = {"knee": args.knee, "min_width": args.min_width,
                         "max_width": args.max_width, "field": args.field,
                         "line_alpha": args.line_alpha}
    fd["bubble_scale"] = args.bubble_scale if args.bubbles else None
    (BUILD / f"frame_{tag}.json").write_text(json.dumps(fd, indent=1),
                                             encoding="utf-8")

    lines_png = BUILD / f"lines_{tag}.png"
    render_lines(f, W, H, scale, lines_png, args, "dark")

    glow_png = BUILD / f"glow_{tag}.png"
    if args.glow > 0:
        glowfx.glow(lines_png, glow_png, args.glow, args.dpi, scale,
                    args.glow_strength, args.glow_wide, args.glow_wide_strength,
                    cap=args.glow_cap)
        print(f"  glow r={args.glow * args.dpi * scale:.0f}px + "
              f"{args.glow * args.glow_wide * args.dpi * scale:.0f}px "
              f"({time.time() - t0:.1f}s)")

    themes = sorted(THEMES) if args.theme == "both" else [args.theme]
    for theme in themes:
        # Bubbles are the one data layer whose colour depends on the theme, so
        # they are rendered per theme rather than once and reused.
        bub_png = BUILD / f"bubbles_{theme}_{tag}.png"
        if args.bubbles:
            render_bubbles(f, W, H, scale, bub_png, args, theme)
        base = render_base(f, W, H, theme, BUILD / f"base_{theme}_{tag}.png",
                           args.prefectures, args.foreign, args.veil_alpha,
                           args.veil_buffer_km)
        print(f"  base_{theme} done ({time.time() - t0:.1f}s)")
        # glow under the lines, so the line itself stays its own clean colour
        out = composite(base, [glow_png, lines_png, bub_png],
                        BUILD / f"preview_{theme}_{tag}.png")
        print(f"  wrote {out.name} "
              f"({out.stat().st_size / (1 << 20):.1f} MB, {time.time() - t0:.1f}s)")

    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
