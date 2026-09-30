"""
Put the text on the nycriders sheet and write the print file.

    python render.py --theme dark --tag n4      # the map layers
    python compose.py --tag n4                  # -> build/poster_flat_dark_n4.png

Everything sits in one column in the empty right-hand side of the sheet
(eastern Queens and the Sound, nothing drawn there from the top down to about
20 in): title, a two-sentence description, the width and circle keys, a
scale bar, then the small print.

The keys are drawn from the same numbers the map is: widths from defaults.py
against the busiest segment in the data, circles with render.THEMES' own fill,
so the legend cannot quietly disagree with the lines it explains.

Writes, at full sheet size, tagged sRGB (Lumaprints reads an untagged file as
Adobe RGB and it prints oversaturated):

    poster_1_base_<theme>_<tag>.png    water, coastline               (opaque)
    poster_2_map_<tag>.png             lines and station circles      (alpha)
    poster_3_text_<theme>_<tag>.png    everything typeset             (alpha)
    poster_flat_<theme>_<tag>.png      all of it; this is the one to upload

The run fails loudly if any text lands on a line or circle, since the free
space moves whenever the nudges do.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont

import defaults as D
import ribbons
from render import THEMES, STATS

Image.MAX_IMAGE_PIXELS = None
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

HERE = Path(__file__).parent
BUILD = HERE / "build"
FONTS = HERE.parent / "ancestrydots"
DPI = D.DPI

INK = {"dark": {"text": "#e6ecf3", "dim": "#8a929c", "rule": "#3a434e"},
       "light": {"text": "#1b2027", "dim": "#5d6772", "rule": "#c9c3b6"}}

# The text column, in sheet inches. Nothing on the map is right of 17.5 in above
# y = 20 in; the column keeps a quarter inch off that and three quarters off the
# trim.
COL_X = 18.0
COL_W = 5.25
TOP = 1.6

# Matches the About panel on the web map.
TITLE = ["NYC Riders"]
BODY = [
    "This map shows the 4.3 million NYC subway trips taken on an average "
    "Wednesday in September 2025. Thicker lines have more traffic. Larger "
    "station bubbles have more boardings (including transfers).",
    "The MTA publishes origin-destination ridership estimates, as well as "
    "train timetables. In this map, riders navigate from their origins to "
    "their destinations, taking into account transfer times and when trains "
    "are scheduled.",
]
# A line is one route, in whichever direction is busier. The note sits after
# its heading on the same line, in the lighter colour; None leaves it out.
WIDTH_HEAD = "Lines (people per day)"
WIDTH_NOTE = "(in busier direction)"
WIDTH_KEY = [10_000, 50_000]                   # plus the busiest (111k), added below
CIRCLE_HEAD = "Stations (people per day)"
CIRCLE_NOTE = None
CIRCLE_KEY = [10_000, 100_000]                 # plus the busiest station
# Shorter names for the busiest segment, where the two station names together
# run past the column. Both of these stations are in Jackson Heights.
SEG_NAMES = {"7|709|710": "7 train at Jackson Heights"}
SMALL = [
    "Data from the MTA Subway Origin-Destination Ridership Estimate (2025) and "
    "MTA GTFS timetables, published at data.ny.gov and new.mta.info. Coastline "
    "© OpenStreetMap contributors. The Staten Island Railway is not shown. "
    "This is not an official MTA map.",
]
# at the legend's label size, above the small print
LINK = "live version at anita.garden/nycriders"

TITLE_PT = 52
BODY_PT = 16
HEAD_PT = 13
LABEL_PT = 12
SMALL_PT = 10


def px(inches):
    return int(round(inches * DPI))


def rgba(h, a=255):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (a,)


def font(pt, bold=False):
    path = FONTS / ("Nunito-Bold.ttf" if bold else "Nunito-Regular.ttf")
    return ImageFont.truetype(str(path), int(round(pt / 72 * DPI)))


def step(pt, leading=1.42):
    return pt * leading / 72


def wrap(d, text, f, max_px):
    out, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and d.textlength(trial, font=f) > max_px:
            out.append(line)
            line = word
        else:
            line = trial
    if line:
        out.append(line)
    return out


def fmt(n):
    return f"{int(round(n, -3)):,}" if n >= 10_000 else f"{int(round(n)):,}"


def save(im, path):
    """Temp file and rename, so an image viewer holding the old file costs a
    clear message rather than a half-written print file."""
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    im.save(tmp, icc_profile=SRGB)
    for attempt in range(4):
        try:
            os.replace(tmp, path)
            return path
        except OSError:
            time.sleep(0.4 * (attempt + 1))
    print(f"  !! {path.name} is open somewhere; new version left as {tmp.name}")
    return tmp


def heading(d, x, y, head, note, text, dim):
    """A key's bold heading with an optional lighter note after it on the same
    baseline. Returns y below it."""
    y += HEAD_PT * 0.74 / 72
    f = font(HEAD_PT, True)
    d.text((x, px(y)), head, fill=text, font=f, anchor="ls")
    if note:
        d.text((x + d.textlength(head + " ", font=f), px(y)), note, fill=dim,
               font=font(LABEL_PT), anchor="ls")
    return y + step(HEAD_PT, 1.25) - HEAD_PT * 0.74 / 72


def busiest(feats, data):
    """(value, label) for the thickest stripe and the biggest circle, named,
    so the top rung of each key is a real place rather than a round number."""
    k, f = max(feats.items(), key=lambda kv: kv[1]["value"])
    stops = data["stops"]
    a, b = stops.get(f["from"], {}).get("name"), stops.get(f["to"], {}).get("name")
    seg = SEG_NAMES.get(k) or (f"{f['route']} train, {a} to {b}" if a and b
                               else f"{f['route']} train")
    s = max(ribbons.station_totals(data), key=lambda s: s[2])
    return (f["value"], seg), (s[2], s[3])


def draw_text(d, theme, frame, feats, data):
    ink, bub = INK[theme], THEMES[theme]
    text, dim = rgba(ink["text"]), rgba(ink["dim"])
    x, w = px(COL_X), px(COL_W)
    y = TOP

    # title, sized down if a line would not fit the column
    pt = TITLE_PT
    while max(d.textlength(t, font=font(pt, True)) for t in TITLE) > w:
        pt -= 1
    f = font(pt, True)
    for t in TITLE:
        y += pt * 0.74 / 72
        d.text((x, px(y)), t, fill=text, font=f, anchor="ls")
        y += step(pt, 1.12) - pt * 0.74 / 72
    y += 0.24

    f = font(BODY_PT)
    for para in BODY:
        for line in wrap(d, para, f, w):
            y += step(BODY_PT)
            d.text((x, px(y)), line, fill=text, font=f, anchor="ls")
        y += 0.12
    y += 0.45

    (seg_v, seg_name), (stn_v, stn_name) = busiest(feats, data)

    # width key: bars at exactly the width render.py draws that value at
    y = heading(d, x, y, WIDTH_HEAD, WIDTH_NOTE, text, dim) + 0.22
    vmax = D.WIDTH_REF
    lo, hi = D.MIN_WIDTH, D.MAX_WIDTH
    bar = px(0.9)
    fl = font(LABEL_PT)
    rows = [(v, fmt(v)) for v in WIDTH_KEY] + [(seg_v, fmt(seg_v))]
    for i, (v, label) in enumerate(rows):
        wi = lo + (hi - lo) * (v / vmax) ** D.WIDTH_GAMMA
        t = max(1, px(wi))
        cy = px(y)
        d.rectangle([x, cy - t // 2, x + bar, cy - t // 2 + t], fill=text)
        d.text((x + bar + px(0.18), cy), label, fill=text, font=fl, anchor="lm")
        if i == len(rows) - 1:
            lx = x + bar + px(0.18) + d.textlength(label + "  ", font=fl)
            d.text((lx, cy), seg_name, fill=dim, font=fl, anchor="lm")
        y += 0.30
    y += 0.34

    # circle key, with the map's own fill and radius formula
    y = heading(d, x, y, CIRCLE_HEAD, CIRCLE_NOTE, text, dim) + 0.12
    smax = max(s[2] for s in ribbons.station_totals(data))
    rmax = D.BUBBLE_MAX
    face = rgba(bub["bubble"], int(round(255 * bub["bubble_alpha"])))
    edge = rgba(bub["bubble_edge"], 140) if bub["bubble_edge"] else None
    cx = x + px(rmax)
    for v, label in [(v, fmt(v)) for v in CIRCLE_KEY] + [(stn_v, fmt(stn_v))]:
        r_in = D.BUBBLE_MIN + (D.BUBBLE_MAX - D.BUBBLE_MIN) * np.sqrt(v / smax)
        r = px(r_in)
        y += max(r_in, 0.10)
        cy = px(y)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=face, outline=edge,
                  width=max(1, px(0.004)) if edge else 0)
        lx = x + px(2 * rmax + 0.18)
        d.text((lx, cy), label, fill=text, font=fl, anchor="lm")
        if v == stn_v:
            d.text((lx + d.textlength(label + "  ", font=fl), cy), stn_name,
                   fill=dim, font=fl, anchor="lm")
        y += max(r_in, 0.10) + 0.10
    y += 0.34

    # scale bars, miles first for a New York reader
    km_in = frame["km_per_inch"]
    t = max(3, px(0.016))
    for length_in, label in ((1.609344 / km_in, "1 mile"), (1.0 / km_in, "1 km")):
        cy = px(y)
        d.rectangle([x, cy - t // 2, x + px(length_in), cy - t // 2 + t], fill=dim)
        for end in (x, x + px(length_in) - t):
            d.rectangle([end, cy - px(0.05), end + t, cy + px(0.05)], fill=dim)
        d.text((x + px(length_in) + px(0.15), cy), label, fill=dim, font=fl, anchor="lm")
        y += 0.28
    y += 0.34

    y += step(LABEL_PT)
    d.text((x, px(y)), LINK, fill=text, font=font(LABEL_PT), anchor="ls")
    y += 0.22

    f = font(SMALL_PT)
    for para in SMALL:
        for line in wrap(d, para, f, w):
            y += step(SMALL_PT)
            d.text((x, px(y)), line, fill=dim, font=f, anchor="ls")
        y += 0.10
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="n3", help="which render.py output to use")
    ap.add_argument("--theme", default="dark", choices=["dark", "light"])
    args = ap.parse_args()
    theme, tag = args.theme, args.tag

    base_p = BUILD / f"base_{theme}_{tag}.png"
    lines_p = BUILD / f"lines_{tag}.png"
    bub_p = BUILD / f"bubbles_{theme}_{tag}.png"
    for p in (base_p, lines_p, bub_p):
        if not p.exists():
            raise SystemExit(f"missing {p.name}; run `python render.py --theme "
                             f"{theme} --tag {tag}` first")
    frame = json.loads((BUILD / f"frame_{tag}.json").read_text(encoding="utf-8"))
    data, feats = ribbons.load_features(STATS)

    base = Image.open(base_p).convert("RGBA")
    W, H = base.size
    mp = Image.alpha_composite(Image.open(lines_p).convert("RGBA"),
                               Image.open(bub_p).convert("RGBA"))

    tx = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bottom = draw_text(ImageDraw.Draw(tx), theme, frame, feats, data)
    print(f"sheet {W} x {H} px; text column {COL_X:.2f}-{COL_X + COL_W:.2f} in, "
          f"y {TOP:.2f}-{bottom:.2f} in")

    # anything typeset on top of a line or circle, with a margin of 0.08 in
    ta = np.asarray(tx.getchannel("A")) > 0
    ma = np.asarray(mp.getchannel("A")) > 8
    m = px(0.08)
    ys, xs = np.nonzero(ta)
    y0, y1, x0, x1 = ys.min() - m, ys.max() + m, xs.min() - m, xs.max() + m
    hit = ma[max(0, y0):y1, max(0, x0):x1].sum()
    if hit:
        print(f"  !! the text block overlaps {hit:,} map pixels; move COL_X / TOP")

    flat = Image.alpha_composite(Image.alpha_composite(base, mp), tx).convert("RGB")
    for name, im in ((f"poster_1_base_{theme}_{tag}.png", base.convert("RGB")),
                     (f"poster_2_map_{tag}.png", mp),
                     (f"poster_3_text_{theme}_{tag}.png", tx),
                     (f"poster_flat_{theme}_{tag}.png", flat)):
        out = save(im, BUILD / name)
        mb = out.stat().st_size / (1 << 20)
        cap = "" if "flat" not in name else ("   OK" if mb < 100 else "   OVER the 100 MB upload cap")
        print(f"  wrote {out.name:34s} {mb:6.1f} MB{cap}")


if __name__ == "__main__":
    main()
