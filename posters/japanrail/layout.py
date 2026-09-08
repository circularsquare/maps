"""
Plan the japanrail poster sheet: where the insets and the title block sit.

Unlike ancestrydots, this is **not** an optimiser. The composition is Anita's:
title block top-left, Osaka and Nagoya in the Sea of Japan above the cities
they magnify, Tokyo out in the Pacific to the right of the country, Fukuoka
tucked under the title. So the boxes are declared, and the script's job is to
*check* them — how much of the map each one covers, whether it stays on the
sheet, and where its window really is so a locator can be drawn.

Coverage is measured off the rendered layers rather than from geometry: the
lines PNG's alpha is exactly the ink that would be hidden, and the base PNG's
land colour is exactly the land, foreign land already painted out.

    python layout.py                       # check + write layout.json
    python layout.py --plan-only           # just redraw the plan image

Writes build/layout.json and build/layout_plan.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import frame as fr
from insets import CITIES

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).parent
BUILD = HERE / "build"

MARGIN_IN = 0.40           # keep everything this far off the trim edge
CAPTION_IN = 0.62          # strip under each inset for its name and figures

# Declared composition, in sheet inches. y is measured down from the top.
# Boxes are the MAP window only; the caption strip is added underneath, so the
# frame drawn around an inset is h_in + CAPTION_IN tall.
PLAN = [
    # region is (xmin, xmax, ymin, ymax) for --search: the part of the sheet
    # this box is allowed to live in, which is the composition talking. The
    # x_in/y_in are the chosen answer.
    # Osaka sits left of Nagoya because their cities do. With the boxes the
    # other way round the two leader lines crossed over the Chugoku coast,
    # which is the defect ancestrydots' free placement kept producing.
    #
    # Tokyo may not touch Japan at all. Land reaches x = 27.38 in across the
    # rows this box occupies, which is what `--extend-right 2.6` on the render
    # is for: the sheet grows to 38.6 in at the same km/inch so the box has a
    # clear column, rather than the map shrinking to make room.
    #
    # Osaka and Nagoya are a single top-aligned row under the text blocks, with
    # Osaka pushed to the left margin. That row can be at most about 8.5 in
    # tall: land starts at y = 12.5 (Noto and Sado) and y = 13.0 (west Kyushu),
    # and no inset may touch land at all.
    dict(name="tokyo",   x_in=27.75, y_in=9.15,  w_in=10.45, h_in=10.95,
         region=(27.5, 28.2, 9.0, 9.8)),
    # The row's floor is the San'in coast of Honshu, which the box bottoms run
    # into at y = 13.85 (measured: lon 130.9–133.5, lat 34.4–35.6). The island
    # exemptions bought 1.35 in over the previous 12.5 limit.
    dict(name="osaka",   x_in=0.40,  y_in=5.00,  w_in=11.60, h_in=8.20,
         region=(0.4, 2.0, 4.8, 5.4)),
    dict(name="nagoya",  x_in=12.50, y_in=5.00,  w_in=8.60,  h_in=7.20,
         region=(12.0, 20.0, 4.8, 5.4)),
]

# Text blocks. Not insets — no window, no locator, no caption strip. They are a
# row across the top: title on the left margin, legend to its right. compose.py
# reports if either overruns the height reserved here.
# The title block is a measure, not a container: 9.2 in is about 100 characters
# at the body size, which is as far as an eye should have to travel back.
TITLE = dict(name="title", x_in=0.40, y_in=0.40, w_in=9.20, h_in=4.40)
LEGEND = dict(name="legend", x_in=10.20, y_in=0.40, w_in=7.40, h_in=4.40)


# Islands an inset is allowed to sit on: (lon, lat, radius km). Anita's call —
# covering Tsushima, Iki or the Oki islands costs nothing a reader wants, and
# insisting on zero land anywhere was what held the Osaka/Nagoya row up and
# kept it small. Honshu, Kyushu, Shikoku and Hokkaido stay off limits.
LAND_EXEMPT = [
    (129.33, 34.40, 48),    # Tsushima
    (129.70, 33.78, 24),    # Iki
    (133.10, 36.22, 38),    # Oki
]


def _rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def content_grid(tag, f, cells_per_in=8):
    """Two grids of what is already on the sheet, 0..1 per cell: rail ink, and
    Japanese land.

    They are kept apart deliberately. Hiding a rail line is a real loss; a box
    sitting on empty Tohoku hillside costs almost nothing, and a single summed
    figure would rank those the same and push every inset out to sea for no
    reason. Ancestrydots could use one number because there the content *was*
    the dots.
    """
    import render as R

    lines = Image.open(BUILD / f"lines_{tag}.png").split()[-1]
    base = Image.open(BUILD / f"base_dark_{tag}.png").convert("RGB")
    W, H = lines.size
    gw = round(W / 300 * cells_per_in)
    gh = round(H / 300 * cells_per_in)
    ink = np.asarray(lines.resize((gw, gh), Image.BOX), float) / 255.0
    bub = BUILD / f"bubbles_dark_{tag}.png"
    if bub.exists():
        ink = np.maximum(ink, np.asarray(
            Image.open(bub).split()[-1].resize((gw, gh), Image.BOX),
            float) / 255.0)

    # Land by nearest theme colour rather than by a channel threshold. The
    # threshold version read the blue channel, which silently inverted the
    # moment the dark theme's hues were swapped so that water is the blue one.
    b = np.asarray(base.resize((gw, gh), Image.BOX), float)
    lc, wc = _rgb(R.THEMES["dark"]["land"]), _rgb(R.THEMES["dark"]["water"])
    land = (((b - lc) ** 2).sum(-1) < ((b - wc) ** 2).sum(-1)).astype(float)

    yy, xx = np.mgrid[0:gh, 0:gw]
    for lon, lat, km in LAND_EXEMPT:
        cx, cy = f.to_inches(lon, lat)
        r = km / f.km_per_inch * cells_per_in
        inside = ((xx - cx * cells_per_in) ** 2
                  + (yy - cy * cells_per_in) ** 2) <= r * r
        land[inside] = 0.0
    return ink, land, cells_per_in


def locator_quad(f, name):
    """The inset's window drawn on the main map, as four corners in sheet
    inches. It is a rotated square, not an axis-aligned box: the sheet is
    turned 25 deg and the insets are north-up, so the window lands on the map
    tilted by exactly that much."""
    from insets import inset_frame
    w = inset_frame(name)
    xs = [w.x0, w.x1, w.x1, w.x0]
    ys = [w.y1, w.y1, w.y0, w.y0]
    # inset frames are unrotated projected metres; put them back on the tilted map
    px, py = fr.unrotate(np.array(xs), np.array(ys), 0.0)
    rx, ry = fr.rotate(px, py, f.rot)
    return [[round((x - f.x0) / f.m_per_inch, 3),
             round((f.y1 - y) / f.m_per_inch, 3)] for x, y in zip(rx, ry)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v8",
                    help="which national render to measure against "
                         "(build/*_<tag>.png)")
    ap.add_argument("--cells-per-in", type=int, default=8)
    ap.add_argument("--plan-width", type=int, default=1900)
    ap.add_argument("--search", action="store_true",
                    help="report the cheapest placements inside each box's "
                         "declared region, then stop. The composition stays "
                         "hand-chosen; this only says what it costs.")
    args = ap.parse_args()

    f = fr.Frame.from_dict(json.loads(
        (BUILD / f"frame_{args.tag}.json").read_text(encoding="utf-8")))
    SW, SH = f.width_in, f.height_in
    print(f"sheet {SW:.2f} x {SH:.2f} in at {f.km_per_inch:.1f} km/inch")

    ink, land, cpi = content_grid(args.tag, f, args.cells_per_in)
    gh, gw = ink.shape

    def make_sat(g):
        s = np.zeros((gh + 1, gw + 1))
        s[1:, 1:] = np.cumsum(np.cumsum(g, 0), 1)
        return s

    sat_ink, sat_land = make_sat(ink), make_sat(land)
    total_ink, total_land = ink.sum(), land.sum()

    def cost(b, extra=0.0):
        """(rail ink, land) hidden by this box, in cells."""
        r0 = max(0, round((b["y_in"] - extra) * cpi))
        c0 = max(0, round((b["x_in"] - extra) * cpi))
        r1 = min(gh, round((b["y_in"] + b["h_in"] + CAPTION_IN + extra) * cpi))
        c1 = min(gw, round((b["x_in"] + b["w_in"] + extra) * cpi))
        out = []
        for s in (sat_ink, sat_land):
            out.append(s[r1, c1] - s[r0, c1] - s[r1, c0] + s[r0, c0])
        return out[0], out[1]

    if args.search:
        for p in PLAN:
            x0, x1, y0, y1 = p["region"]
            hh = p["h_in"] + CAPTION_IN
            out = []
            for y in np.arange(y0, min(y1, SH - MARGIN_IN - hh) + 1e-9, 0.20):
                for x in np.arange(x0, min(x1, SW - MARGIN_IN - p["w_in"])
                                   + 1e-9, 0.20):
                    ci, cl = cost({"x_in": x, "y_in": y, "w_in": p["w_in"],
                                   "h_in": p["h_in"]})
                    out.append((ci, cl, x, y))
            out.sort()
            print(f"\n{p['name']} ({p['w_in']} x {hh:.2f} in) in region "
                  f"{p['region']}")
            shown = []
            for ci, cl, x, y in out:
                if any(abs(x - a) < 1.0 and abs(y - b) < 1.0 for a, b in shown):
                    continue
                shown.append((x, y))
                print(f"   x={x:5.2f} y={y:5.2f}  hides {ci:7.1f} cells of "
                      f"rail, {cl:7.1f} of land")
                if len(shown) >= 5:
                    break
        return

    boxes, covered = [], 0.0
    for p in PLAN:
        b = dict(p)
        b["box_h_in"] = b["h_in"] + CAPTION_IN
        c = CITIES[b["name"]]
        b["label"] = c["label"]
        b["label_ja"] = c.get("label_ja", c["label"])
        b["center"] = c["center"]
        b["locator_in"] = locator_quad(f, b["name"])
        from insets import inset_frame
        w = inset_frame(b["name"], (b["w_in"], b["h_in"]))
        b["km_per_inch"] = round(w.km_per_inch, 3)
        b["mag"] = round(f.km_per_inch / w.km_per_inch, 2)
        ci, cl = cost(b)
        b["hides_rail"], b["hides_land"] = round(ci, 1), round(cl, 1)
        covered += ci

        right = b["x_in"] + b["w_in"]
        bottom = b["y_in"] + b["box_h_in"]
        fits = (b["x_in"] >= MARGIN_IN - 1e-6 and b["y_in"] >= MARGIN_IN - 1e-6
                and right <= SW - MARGIN_IN + 1e-6
                and bottom <= SH - MARGIN_IN + 1e-6)
        print(f"  {b['name']:<9} {b['w_in']:5.2f} x {b['box_h_in']:5.2f} in at "
              f"({b['x_in']:5.2f},{b['y_in']:5.2f})  {b['mag']:4.1f}x  hides "
              f"{ci:6.1f} rail ({ci / total_ink * 100:5.2f}%), "
              f"{cl:6.1f} land ({cl / total_land * 100:5.2f}%)"
              + ("" if fits else "   <- OFF THE SHEET"))
        boxes.append(b)

    texts = []
    for src_block in (TITLE, LEGEND):
        b = dict(src_block)
        b["box_h_in"] = b["h_in"]
        ci, cl = cost(b)
        b["hides_rail"], b["hides_land"] = round(ci, 1), round(cl, 1)
        print(f"  {b['name']:<9} {b['w_in']:5.2f} x {b['h_in']:5.2f} in at "
              f"({b['x_in']:5.2f},{b['y_in']:5.2f})        hides "
              f"{ci:6.1f} rail, {cl:6.1f} land")
        texts.append(b)
    t, legend = texts

    # overlaps
    allb = boxes + texts
    for i in range(len(allb)):
        for j in range(i + 1, len(allb)):
            a, b = allb[i], allb[j]
            if (a["x_in"] < b["x_in"] + b["w_in"]
                    and b["x_in"] < a["x_in"] + a["w_in"]
                    and a["y_in"] < b["y_in"] + b["box_h_in"]
                    and b["y_in"] < a["y_in"] + a["box_h_in"]):
                print(f"  !! {a['name']} overlaps {b['name']}")

    print(f"  total rail hidden {covered:.0f} of {total_ink:.0f} cells "
          f"({covered / total_ink * 100:.2f}%)")

    lay = {"sheet_in": [round(SW, 4), round(SH, 4)],
           "map_in": [round(SW, 4), round(SH, 4)],
           "km_per_inch": round(f.km_per_inch, 3),
           "caption_in": CAPTION_IN,
           "margin_in": MARGIN_IN,
           "frame": json.loads(
               (BUILD / f"frame_{args.tag}.json").read_text(encoding="utf-8")),
           "title": t,
           "legend": legend,
           "insets": boxes}
    (BUILD / "layout.json").write_text(json.dumps(lay, ensure_ascii=False,
                                                  indent=1), encoding="utf-8")

    # --- plan picture -----------------------------------------------------
    src = Image.open(BUILD / f"preview_dark_{args.tag}.png").convert("RGB")
    pw = args.plan_width
    ph = round(src.height * pw / src.width)
    plan = src.resize((pw, ph), Image.LANCZOS)
    d = ImageDraw.Draw(plan, "RGBA")
    s = pw / SW

    def rect(b, col, width=3):
        d.rectangle([b["x_in"] * s, b["y_in"] * s,
                     (b["x_in"] + b["w_in"]) * s,
                     (b["y_in"] + b["box_h_in"]) * s], outline=col, width=width)

    for b in boxes:
        rect(b, (0xba, 0x9a, 0x55, 0xff))
        d.line([b["x_in"] * s, (b["y_in"] + b["h_in"]) * s,
                (b["x_in"] + b["w_in"]) * s, (b["y_in"] + b["h_in"]) * s],
               fill=(0xba, 0x9a, 0x55, 0x88), width=2)
        d.polygon([(x * s, y * s) for x, y in b["locator_in"]],
                  outline=(0x5b, 0xb6, 0xff, 0xff))
        # The plan image keeps a leader so the pairing is obvious while
        # placing boxes; the poster itself does not draw one — see compose.py.
        lx = sum(p[0] for p in b["locator_in"]) / 4 * s
        ly = sum(p[1] for p in b["locator_in"]) / 4 * s
        d.line([lx, ly, (b["x_in"] + b["w_in"] / 2) * s,
                (b["y_in"] + b["h_in"] / 2) * s],
               fill=(0x5b, 0xb6, 0xff, 0x44), width=2)
    for b in texts:
        rect(b, (0x8a, 0xd8, 0x9a, 0xff))

    out = BUILD / "layout_plan.png"
    plan.save(out)
    print(f"  wrote {out.name} ({pw} x {ph})")


if __name__ == "__main__":
    main()
