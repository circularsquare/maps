"""
City insets for the japanrail poster.

Each inset is a **north-up** window — the national map is tilted 25 deg, the
insets are not. A magnified window is read as a city map, and a city map that
is also rotated makes the reader do two things at once for no gain; the frame
around it already says it is a different scale.

Windows are declared as a geographic scope in km, not as a magnification, so
"how much country does Tokyo need" is one number to argue about. The
magnification falls out of that and the box size the layout gives it.

    python insets.py --only tokyo                # one inset, default size
    python insets.py --theme both                # every inset, both themes
    python insets.py --from-layout               # sizes from build/layout.json

Writes build/insets/<name>_<theme>.png plus <name>.json (its window, for
compose.py's locator boxes and scale bars).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

import frame as fr
import glowfx
import render as R

HERE = Path(__file__).parent
BUILD = HERE / "build"
OUT = BUILD / "insets"

# scope_km is the window's (width, height) on the ground. These are deliberately
# tight: an early Tokyo window of 310 x 206 km carried most of Shizuoka, which
# is a different story from the one the inset is there to tell.
CITIES = {
    "tokyo": dict(
        center=(139.735, 35.663), scope_km=(78, 66), box_in=(10.45, 10.95),
        label="Tokyo", label_ja="東京",
        note="Tachikawa to Chiba across, Omiya down past Kamakura. The centre "
             "was at 35.700 while the box was square, which spent 11 km of "
             "empty Saitama above Omiya and cut Fujisawa off the bottom by "
             "1 km; 35.575 overcorrected, so this is 12% of the window height "
             "back north. Omiya is the hard northern constraint — the Joetsu "
             "Shinkansen branches off the Tohoku there and that junction has "
             "to be on the sheet. At 35.663 the window is lat 35.295–36.031, "
             "so Zushi is the southern limit and Yokosuka-chuo (35.276) falls "
             "1.5 km outside; going 8% rather than 12% would keep it."),
    "osaka": dict(
        center=(135.570, 34.770), scope_km=(78, 66), box_in=(11.0, 7.65),
        label="Kansai", label_ja="関西",
        note="the Keihanshin triangle: Kobe left, Kyoto top right, Nara right. "
             "An earlier 88 km window centred 0.04 deg north spent its left "
             "third on the empty Tanba hills and still lost the Nara and "
             "Wakayama lines off the bottom. The box is wider than the "
             "declared scope's aspect, so inset_frame widens the window to "
             "about 95 km; the centre is nudged east to keep the core centred "
             "rather than letting the extra width fall on empty Hyogo."),
    "nagoya": dict(
        center=(136.920, 35.140), scope_km=(56, 46), box_in=(6.5, 5.6),
        label="Nagoya", label_ja="名古屋",
        note="Gifu to Toyohashi's near side"),
}

# Fukuoka was built and dropped. At the 72 km 'north Kyushu' scope that reached
# Kitakyushu it was two long red JR lines and a lot of empty ground; tightened
# to the city it was better but still far sparser than the other three, and
# Fukuoka is the largest of the remaining candidates — Sapporo and Sendai would
# be sparser still. Three insets, deliberately.

# Insets run 6-7x the national scale, so the same printed width covers roughly a
# seventh of the ground and the metro cores stop blobbing. A slightly bolder
# maximum than the national sheet uses the extra room.
INSET_MIN_W = 0.0061
INSET_MAX_W = 0.167
# A HIGHER knee than the national sheet (1M against 100k), which is the right
# way round: inside a metro almost every segment is between 100k and 1.4M, so
# on the national curve they would all sit near the top and flatten into one
# weight. Putting the knee at the top of that range spends the whole width
# ladder on the range that actually varies here.
INSET_KNEE = 6.0
INSET_LINE_ALPHA = 1.0

# Station bubbles: area proportional to daily 乗降客数, as on the web map.
# 9.0e-5 puts Shinjuku's 2.20M at 0.13 in of radius — about 3.4 mm, a little
# over twice the widest line. A first pass at 1.9e-5 was far too timid: at
# 0.7 mm the bubbles read as noise on the line rather than as stations.
# They are translucent so a run of them along a line builds up rather than
# masking it, which is how the clusters at Shinjuku and Umeda appear.
# The 3,000/day floor drops roughly half the country's stations, all of them
# dots too small to print, and saves drawing them.
BUBBLE_SCALE = 9.0e-5
BUBBLE_ALPHA = 0.45
BUBBLE_MIN_RIDERS = 3000


def inset_frame(name, box_in=None):
    c = CITIES[name]
    w_in, h_in = box_in or c["box_in"]
    kmw, kmh = c["scope_km"]
    # The box's aspect wins: a wider box just shows more country at the same
    # magnification, so nothing is lost, and the rails stay square-edged.
    m_per_in = kmw * 1000.0 / w_in
    cx, cy = fr.project(*c["center"], 0.0)
    half_w = w_in * m_per_in / 2
    half_h = h_in * m_per_in / 2
    want_h_km = h_in * m_per_in / 1000.0
    if want_h_km < kmh:
        # the declared scope is taller than the box allows — grow the window to
        # hold it and let the width overshoot instead of cropping the city
        m_per_in = kmh * 1000.0 / h_in
        half_w = w_in * m_per_in / 2
        half_h = h_in * m_per_in / 2
    return fr.Frame(0.0, cx - half_w, cx + half_w, cy - half_h, cy + half_h,
                    w_in, 300)


def render_one(name, theme, box_in, dpi, bubbles=True, scale=1.0, glow=0.018,
               glow_strength=0.30, glow_wide=4.0, glow_wide_strength=0.09,
               glow_cap=0.80):
    c = CITIES[name]
    f = inset_frame(name, box_in)
    f.dpi = dpi
    W, H = f.size_px(scale)
    # magnification is relative to whatever the national sheet currently is,
    # so it comes from layout.json rather than from a hardcoded render tag
    mag = None
    for src, key in ((BUILD / "layout.json", "km_per_inch"),
                     (BUILD / "frame_v2.json", "km_per_inch")):
        if src.exists():
            mag = json.loads(src.read_text(encoding="utf-8"))[key] / f.km_per_inch
            break

    args = SimpleNamespace(
        field="density", knee=INSET_KNEE,
        min_width=INSET_MIN_W, max_width=INSET_MAX_W,
        color="line", mono_color="#7fd8ff", line_alpha=INSET_LINE_ALPHA,
        casing=0.0, casing_alpha=0.9, dpi=dpi,
        bubbles=bubbles, bubble_scale=BUBBLE_SCALE,
        bubble_alpha=BUBBLE_ALPHA, min_riders=BUBBLE_MIN_RIDERS,
        bubble_edge_alpha=0.30, bubble_edge_width=0.005)

    OUT.mkdir(parents=True, exist_ok=True)
    lines_png = OUT / f"{name}_lines.png"
    R.render_lines(f, W, H, scale, lines_png, args, theme)
    layers = []
    if glow > 0:
        # The glow radius is in inches of printed sheet, so an inset at 7x gets
        # the same halo *on paper* as the national map — which is what makes
        # the two read as one poster rather than two.
        gp = OUT / f"{name}_glow.png"
        glowfx.glow(lines_png, gp, glow, dpi, scale, glow_strength, glow_wide,
                    glow_wide_strength, cap=glow_cap)
        layers.append(gp)
    layers.append(lines_png)
    if bubbles:
        bub = OUT / f"{name}_bubbles_{theme}.png"
        R.render_bubbles(f, W, H, scale, bub, args, theme)
        layers.append(bub)

    base = R.render_base(f, W, H, theme, OUT / f"{name}_base_{theme}.png",
                         prefectures=False, foreign="hide")
    R.composite(base, layers, OUT / f"{name}_{theme}.png")

    (OUT / f"{name}.json").write_text(json.dumps({
        "name": name, "label": c["label"], "center": c["center"],
        "scope_km": [round(f.km_per_inch * f.width_in, 1),
                     round(f.km_per_inch * f.height_in, 1)],
        "box_in": [round(f.width_in, 3), round(f.height_in, 3)],
        "km_per_inch": round(f.km_per_inch, 3),
        "mag": round(mag, 2) if mag else None,
        "frame": f.to_dict(),
    }, indent=1), encoding="utf-8")
    return f, mag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="render just these; default is all of CITIES")
    ap.add_argument("--theme", choices=["dark", "light", "both"], default="dark")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--no-bubbles", dest="bubbles", action="store_false",
                    help="leave the station bubbles off")
    ap.add_argument("--glow", type=float, default=0.018,
                    help="inches — glow radius; 0 turns it off")
    ap.add_argument("--from-layout", action="store_true",
                    help="take each box size from build/layout.json, so the "
                         "renders match what the layout actually reserved")
    args = ap.parse_args()

    sizes = {}
    if args.from_layout:
        lay = json.loads((BUILD / "layout.json").read_text(encoding="utf-8"))
        sizes = {p["name"]: (p["w_in"], p["h_in"]) for p in lay["insets"]}

    names = args.only or list(CITIES)
    themes = ["dark", "light"] if args.theme == "both" else [args.theme]
    t0 = time.time()
    for name in names:
        for theme in themes:
            f, mag = render_one(name, theme, sizes.get(name), args.dpi,
                                args.bubbles, args.scale, args.glow)
            print(f"{name} {theme}: {f.width_in:.2f} x {f.height_in:.2f} in, "
                  f"{f.km_per_inch:.2f} km/inch"
                  + (f", {mag:.1f}x the national map" if mag else "")
                  + f", 1 px = {f.m_per_inch / args.dpi:.0f} m "
                    f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
