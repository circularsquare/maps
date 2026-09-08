"""
Render the city insets for the ancestrydots poster.

Each inset is its own little map: a local Albers centred on the city, so the
projection is accurate at city scale and Honolulu works at all (it is 60 deg of
longitude off the main map's central meridian, where CONUS Albers is unusable).

Scale is a fixed magnification of the main sheet, so dot density stays
comparable across every inset — magnifying is an honest zoom, not a re-scatter.

Water comes from the OSM water polygons, not Natural Earth: at 5x the main
scale 1 px is ~24 m, and NE 10m coastlines would read as visibly polygonal.

Also counts the dots inside each window to produce a per-city top-ancestry
mini-legend (data as JSON, plus a rendered starting-point PNG).

    python insets.py                    # all 16
    python insets.py --only nyc,la      # just those
    python insets.py --ss 1             # hard-edged, no antialiasing

Outputs into build/insets/.
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
from matplotlib.patches import Circle
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer

from fontTools.ttLib import TTFont

from dotraster import splat, over
from typeface import FAMILY, REGULAR

HERE = Path(__file__).parent
BUILD = HERE / "build"
OUT = BUILD / "insets"
WATER = HERE.parent.parent / "data" / "water-polygons-split-4326" / "water_polygons.shp"
NE = HERE.parent.parent / "data" / "ne_10m_lakes"

THEME = {"land": "#0d0d0f", "water": "#141c24", "coast": "#2a3038",
         "ink": "#e8e6e1", "dim": "#8a929c", "faint": "#6b727b",
         "rule": "#39434f"}

# main sheet is 36 in wide covering 6589 km -> 183.0 km/inch
KM_PER_IN_MAIN = 183.0

# name, label, lon, lat, width_in, height_in  (inches on the final sheet)
# in Anita's priority order; window shapes follow each metro's footprint
CITIES = [
    ("nyc",      "New York City", -73.95,  40.72, 3.0, 3.0),
    # recentred east and south to take in the Inland Empire and Orange County
    ("la",       "Los Angeles",  -117.90,  33.90, 3.4, 3.3),
    ("chicago",  "Chicago",       -87.72,  41.87, 3.0, 2.7),
    # shifted north — the old centre carried dead space below San Jose
    ("sf",       "San Francisco",-122.15,  37.71, 2.2, 3.9),
    ("toronto",  "Toronto",       -79.40,  43.72, 3.0, 2.7),
    ("miami",    "Miami",         -80.20,  26.05, 1.8, 3.1),
    ("dc",       "Washington DC", -77.03,  38.90, 2.6, 2.3),
    ("montreal", "Montreal",      -73.60,  45.53, 3.0, 2.7),
    ("atlanta",  "Atlanta",       -84.39,  33.78, 2.6, 2.5),
    ("detroit",  "Detroit",       -83.10,  42.38, 3.0, 2.7),
    ("seattle",  "Seattle",      -122.25,  47.58, 2.0, 3.1),
    ("philly",   "Philadelphia",  -75.15,  39.98, 2.6, 2.3),
    # recentred and grown to hold the whole of Oahu (71 x 48 km)
    ("honolulu", "Honolulu",     -157.98,  21.47, 2.2, 2.2),
    ("boston",   "Boston",        -71.08,  42.36, 2.6, 2.3),
    ("houston",  "Houston",       -95.40,  29.78, 3.0, 2.7),
    ("dallas",   "Dallas",        -96.85,  32.83, 3.0, 2.7),
    ("vancouver", "Vancouver",   -122.95,  49.22, 2.6, 2.2),
    ("phoenix",  "Phoenix",      -112.07,  33.45, 3.0, 2.7),
    ("minneapolis", "Minneapolis", -93.27, 44.97, 3.0, 2.7),
    # Places the main frame cannot reach. Hawaii sits at true main-map scale;
    # Puerto Rico is doubled because at 1x it is 1.2 x 0.4 in and saturates
    # solid; Alaska is *reduced*, which is why its scale has to be stated.
    # Sized to stop short of the antimeridian: wider than this and the window
    # wraps past -180 and picks up the Chukotka coast, which reads as an
    # artifact. This reaches Unalaska in the west and Ketchikan in the east,
    # and drops the far Aleutians, which is the usual compromise.
    # 80 px (0.267 in) shorter than it was, taken off the empty ocean at the
    # bottom — the centre moves north by half that so the cut lands there
    # rather than being split between top and bottom.
    ("alaska",   "Alaska",      -149.00, 63.88, 3.17, 3.43),
    ("hawaii",   "Hawaii",      -157.50, 20.60, 3.17, 3.00),
    ("pr",       "Puerto Rico",  -66.25, 18.22, 2.44, 1.70),
]

# rendered at something other than the global --mag
MAG = {"hawaii": 1.0, "pr": 2.0, "alaska": 0.24}


def local_crs(lon, lat):
    """Albers equal-area centred on the city — equal-area keeps dots-per-km
    meaningful, and centring keeps distortion negligible at this scale."""
    return CRS.from_proj4(
        f"+proj=aea +lat_1={lat - 2:.4f} +lat_2={lat + 2:.4f} +lat_0={lat:.4f} "
        f"+lon_0={lon:.4f} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def top_ancestries(sidx, palette, n, exclude=()):
    """Rank ancestries by dot count inside the window.

    Percentages are of ancestry *responses* in the window, not of people — ACS
    ancestry is multiple-response, so the shares are 'of all responses given'.
    The denominator is always every response in the window, so excluded groups
    still count toward the total and the listed shares stay honest.
    """
    counts = np.bincount(sidx, minlength=len(palette))
    total = int(counts.sum())
    rank = np.argsort(counts)[::-1]
    rows = []
    for i in rank:
        if len(rows) >= n or counts[i] == 0:
            break
        e = palette[int(i)]
        if e["group"] in exclude:
            continue
        rows.append({"label": e["label"], "group": e["group"],
                     "color": e["color"], "dots": int(counts[i]),
                     "people": int(counts[i]) * 100,
                     "pct": round(100 * counts[i] / total, 1)})
    return rows, total


PAD_IN = 0.10          # keeps caption text off the inset's frame
CAP_FS = 8.0           # smallest type in the project, by intent
NAME_GAP = 0.055       # city name to first ancestry row
NAME_FS = 9.5
# where the label starts, measured from the column's left edge. The swatch is
# centred at 0.026 with radius 0.026, so this is the swatch's right edge plus
# the gap. One constant because the label-fitting maths has to use the same
# number — two copies would drift.
LABEL_X = 0.095

# Two-column captions are narrow, and these labels overrun into their
# percentage. The band legend carries the full wording, so shortening here
# costs nothing.
SHORT = {
    "Black, no ancestry reported": "Black (no ancestry)",
    "White, no ancestry reported": "White (no ancestry)",
    "All other American Indian": "Other American Indian",
    "Other Hispanic or Latino": "Other Hispanic/Latino",
    "Trinidadian and Tobagonian": "Trinidadian & Tobagonian",
    "AIAN, not specified": "AIAN, unspecified",
}


_F = TTFont(str(REGULAR))
_UPM = _F["head"].unitsPerEm
_HMTX, _CMAP = _F["hmtx"], _F.getBestCmap()


def text_w(s, fs):
    """Advance width in inches — measured, because a character count is a bad
    proxy: 'Illinois' and 'Wisconsin' are the same length and very different
    widths, and the caption columns have no slack to absorb the error."""
    return sum(_HMTX[_CMAP[ord(c)]][0] for c in s if ord(c) in _CMAP) \
        / _UPM * fs / 72


# Last resort before an ellipsis. "Black (no ances…" is worse than a clean
# abbreviation, and the band legend carries the full wording either way.
ABBREV = {
    "Black, no ancestry reported": "Black (n.a.)",
    "White, no ancestry reported": "White (n.a.)",
    "All other American Indian": "Other Am. Indian",
    "Other Hispanic or Latino": "Other Hisp./Latino",
    "Trinidadian and Tobagonian": "Trinidadian",
    "Other Subsaharan African": "Other Subsaharan",
}


def short(label, fs, avail):
    """Shorten to fit `avail` inches, preferring a real abbreviation over a
    mid-word ellipsis. The band legend carries the full wording."""
    for s in (SHORT.get(label, label), ABBREV.get(label)):
        if s and text_w(s, fs) <= avail:
            return s
    s = ABBREV.get(label) or SHORT.get(label, label)
    while s and text_w(s + "…", fs) > avail:
        s = s[:-1]
    return s.rstrip() + "…"


def _half(fs):
    """Half the visual height of a line, in inches. Nunito's caps and ascenders
    both sit at 0.72 em, so that is the height that reads as the text block."""
    return 0.72 * fs / 72 / 2


def render_legend(name, label, rows, path, width_in, h_in, people, dpi=300):
    """Caption strip under an inset: city name, ancestries counted in bounds,
    and the top ancestries in two columns.

    Padding is uniform — the gap above the city name, left of everything, and
    below the last row are all PAD_IN. Rows are distributed so the last one
    lands exactly on the bottom padding rather than wherever a fixed step ends
    up. Both columns' percentages right-align, the second flush with the
    ancestry count above it.
    """
    fig = plt.figure(figsize=(width_in, h_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_in)
    ax.set_ylim(0, h_in)
    ax.set_axis_off()
    fig.patch.set_alpha(0)

    # divider separating the caption from the map region above it
    ax.plot([0, width_in], [h_in - 0.008, h_in - 0.008],
            color=THEME["rule"], lw=1.1, solid_capstyle="butt")

    y_name = h_in - PAD_IN - _half(NAME_FS)
    ax.text(PAD_IN, y_name, label, color=THEME["ink"], fontsize=NAME_FS,
            fontweight="bold", va="center", family=FAMILY)
    ax.text(width_in - PAD_IN, y_name, f"{people / 1e6:.1f}m ancestries",
            color=THEME["faint"], fontsize=CAP_FS, va="center", ha="right",
            family=FAMILY)

    ncol, gut = 2, 0.12
    per = -(-len(rows) // ncol)
    col_w = (width_in - 2 * PAD_IN - gut) / ncol
    hh = _half(CAP_FS)
    y_first = y_name - _half(NAME_FS) - NAME_GAP - hh
    y_last = PAD_IN + hh
    step = (y_first - y_last) / (per - 1) if per > 1 else 0

    for i, r in enumerate(rows):
        cx = PAD_IN + (i // per) * (col_w + gut)
        yy = y_first - (i % per) * step
        pct = f"{r['pct']:.1f}%"
        avail = col_w - LABEL_X - text_w(pct, CAP_FS) - 0.05
        ax.add_patch(Circle((cx + 0.026, yy), 0.026, color=r["color"], ec="none"))
        ax.text(cx + LABEL_X, yy, short(r["label"], CAP_FS, avail),
                color=THEME["dim"], fontsize=CAP_FS, va="center", family=FAMILY)
        ax.text(cx + col_w, yy, pct, color=THEME["faint"],
                fontsize=CAP_FS, va="center", ha="right", family=FAMILY)
    fig.savefig(path, dpi=dpi, transparent=True)
    plt.close(fig)


def render_city(name, label, lon, lat, w_in, h_in, dots, lut, palette, args, rng):
    t0 = time.time()
    mag = MAG.get(name, args.mag)
    km_in = KM_PER_IN_MAIN / mag
    W, H = int(round(w_in * args.dpi)), int(round(h_in * args.dpi))
    half_w = w_in * km_in * 1000 / 2
    half_h = h_in * km_in * 1000 / 2

    crs = local_crs(lon, lat)
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x0, x1, y0, y1 = -half_w, half_w, -half_h, half_h   # city is at the origin

    # coarse lon/lat prefilter so we only project the dots that can matter
    dlat = half_h / 111_000 * 1.3
    dlon = half_w / (111_000 * np.cos(np.radians(lat))) * 1.3
    lon_a, lat_a, idx_a = dots
    sel = ((lon_a > lon - dlon) & (lon_a < lon + dlon) &
           (lat_a > lat - dlat) & (lat_a < lat + dlat))
    sx, sy = tf.transform(lon_a[sel], lat_a[sel])
    px = (sx - x0) / (x1 - x0) * W
    py = (y1 - sy) / (y1 - y0) * H
    keep = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py, sidx = px[keep], py[keep], idx_a[sel][keep]

    rgb, alpha = splat(px, py, sidx, W, H, args.dot_radius, lut,
                       ss=args.ss, rng=rng)

    exclude = {g for g in args.exclude_groups.split(",") if g}
    rows, total = top_ancestries(sidx, palette, args.top, exclude)
    (OUT / f"{name}_top.json").write_text(json.dumps(
        {"city": label, "window_in": [w_in, h_in], "mag": mag,
         "dots": total, "people": total * 100, "top": rows},
        indent=2, ensure_ascii=False), encoding="utf-8")
    render_legend(name, label, rows[:args.caption_top],
                  OUT / f"{name}_legend.png", w_in, args.caption_in,
                  total * 100)

    # --- base: OSM water, bbox-filtered in lon/lat -------------------------
    bbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    try:
        water = gpd.read_file(WATER, bbox=bbox)
    except Exception as exc:                              # noqa: BLE001
        print(f"    water read failed ({exc}); falling back to Natural Earth")
        water = gpd.read_file(NE / "ne_10m_ocean.shp", bbox=bbox)
    lakes = gpd.read_file(NE / "ne_10m_lakes.shp", bbox=bbox)

    dpi = 100.0
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(THEME["land"])
    ax.set_axis_off()
    # The OSM water file is split into tiles. Stroking them in the coast colour
    # draws the tile grid across open water; filling with no edge leaves
    # antialiased hairlines along every shared edge, which reads as the same
    # grid. Stroking each tile in its own fill colour closes the seams without
    # drawing anything. NE lakes are single polygons and take a real edge.
    if len(water):
        water.to_crs(crs).plot(ax=ax, color=THEME["water"],
                               ec=THEME["water"], lw=0.7, zorder=1)
    if len(lakes):
        lakes.to_crs(crs).plot(ax=ax, color=THEME["water"],
                               ec=THEME["coast"], lw=0.3, zorder=2)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    base_path = OUT / f"{name}_base.png"
    fig.savefig(base_path, dpi=dpi, facecolor=THEME["land"])
    plt.close(fig)
    # matplotlib can land a pixel short (9.45 in x 100 dpi -> 944, not 945),
    # so force the base to the dot raster's exact size before compositing
    base_img = Image.open(base_path).convert("RGB")
    if base_img.size != (W, H):
        base_img = base_img.resize((W, H), Image.NEAREST)
        base_img.save(base_path)
    base = np.array(base_img)

    comp = over(rgb, alpha, base)
    Image.fromarray(comp).save(OUT / f"{name}.png")
    Image.fromarray(np.dstack([rgb, alpha])).save(OUT / f"{name}_dots.png")

    top2 = ", ".join(f"{r['label']} {r['pct']:.0f}%" for r in rows[:2])
    print(f"  {name:9s} {W:4d}x{H:4d}px {len(px):7,} dots "
          f"{(alpha > 0).mean() * 100:4.1f}% cover {time.time() - t0:5.1f}s   {top2}")
    return comp, rows, total


def contact_sheet(rendered, args):
    cols = 4
    pad, label_h = 16, 22
    cw = max(im.shape[1] for _, im in rendered) + pad
    ch = max(im.shape[0] for _, im in rendered) + pad + label_h
    rows_n = (len(rendered) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rows_n * ch), "#1a1a1e")
    draw = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(rendered):
        r, c = divmod(i, cols)
        x = c * cw + (cw - im.shape[1]) // 2
        y = r * ch + label_h + (ch - label_h - im.shape[0]) // 2
        sheet.paste(Image.fromarray(im), (x, y))
        draw.text((c * cw + 8, r * ch + 5), name.upper(), fill="#8a929c")
    out = BUILD / f"insets_contact_{args.mag:g}x_ss{args.ss}.png"
    sheet.save(out)
    print(f"\ncontact sheet -> {out.name} ({sheet.width}x{sheet.height})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mag", type=float, default=5.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--dot-radius", type=float, default=0.91,
                    help="output-pixel radius; 0.91 is ~35%% under the old 1.4")
    ap.add_argument("--ss", type=int, default=3,
                    help="supersample factor for antialiasing; 1 = hard edges")
    ap.add_argument("--top", type=int, default=8,
                    help="how many ancestries go in the JSON; must be at least "
                         "--caption-top or the caption comes up short")
    ap.add_argument("--caption-top", type=int, default=6,
                    help="how many ancestries fit in the on-poster caption "
                         "(the JSON still gets --top)")
    ap.add_argument("--caption-in", type=float, default=0.72,
                    help="caption strip height; must match layout.CAPTION_IN")
    ap.add_argument("--exclude-groups", default="",
                    help="comma-separated groups to keep out of the mini-legend, "
                         "e.g. no_ancestry — they still count in the denominator")
    ap.add_argument("--only", help="comma-separated city names")
    ap.add_argument("--from-layout", action="store_true",
                    help="take window sizes from build/layout.json, which "
                         "normalises them so each rail squares off")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    palette = json.loads((BUILD / "palette.json").read_text(encoding="utf-8"))
    lut = np.zeros((0x10000, 3), dtype=np.uint8)
    for i, e in enumerate(palette):
        h = e["color"].lstrip("#")
        lut[i] = [int(h[j:j + 2], 16) for j in (0, 2, 4)]

    d = np.load(BUILD / "dots_na.npz")
    dots = (d["lon"], d["lat"], d["idx"])
    rng = np.random.default_rng(args.seed)

    wanted = set(args.only.split(",")) if args.only else None
    sizes = {}
    if args.from_layout:
        lay = json.loads((BUILD / "layout.json").read_text(encoding="utf-8"))
        sizes = {p["name"]: (p["w_in"], p["h_in"]) for p in lay["insets"]}
        print(f"window sizes from layout.json ({len(sizes)} normalised)")
    todo = [(n, lb, lo, la, *sizes.get(n, (w, h)))
            for n, lb, lo, la, w, h in CITIES
            if wanted is None or n in wanted]

    km_in = KM_PER_IN_MAIN / args.mag
    area = sum(w * h for *_, w, h in todo)
    print(f"{len(todo)} insets at {args.mag:g}x = {km_in:.1f} km/inch, "
          f"r={args.dot_radius} ss={args.ss}, {area:.1f} sq in of insets\n")

    rendered, summary = [], []
    for name, label, lon, lat, w_in, h_in in todo:
        im, rows, total = render_city(name, label, lon, lat, w_in, h_in,
                                      dots, lut, palette, args, rng)
        rendered.append((name, im))
        summary.append((label, rows, total))

    lines = ["# Top ancestries per inset window",
             "",
             "Shares are of ancestry *responses* inside each window, not of "
             "people — ACS ancestry is multiple-response.", ""]
    for label, rows, total in summary:
        lines.append(f"## {label}  ({total * 100:,} responses)")
        for r in rows:
            lines.append(f"  {r['pct']:5.1f}%  {r['label']:32s} {r['color']}")
        lines.append("")
    (BUILD / "insets_top_ancestries.md").write_text("\n".join(lines),
                                                    encoding="utf-8")
    print(f"\nwrote {BUILD / 'insets_top_ancestries.md'}")

    if len(rendered) > 1:
        contact_sheet(rendered, args)


if __name__ == "__main__":
    main()
