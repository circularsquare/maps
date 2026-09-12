"""
Assemble the ancestrydots poster into layered full-sheet PNGs for Aseprite.

Everything is placed from build/layout.json, so the layers are guaranteed to
register with each other and with the plan.

    python compose.py

Writes, all the full sheet size (34 x 24 in at 300 ppi = 10200 x 7200):

    poster_1_base.png     land, water, and the legend band ground   (opaque)
    poster_2_dots.png     the dots                                  (alpha)
    poster_3_insets.png   all 20 insets and their captions          (alpha)
    poster_4_frames.png   inset borders and on-map locator boxes    (alpha)
    poster_5_text.png     title block and the ancestry legend       (alpha)
    poster_flat.png       all five flattened, for looking at

Frames are their own layer so they can be restyled or switched off without
re-rendering anything.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageCms

from typeface import REGULAR

# Everything here is authored in sRGB — the palette is sRGB hex, and matplotlib
# and PIL both write sRGB values. Without an embedded profile the printer has
# to guess, and Lumaprints recommends Adobe RGB, so an untagged file risks
# being read as Adobe RGB and coming out oversaturated. Tag it explicitly.
# Converting to Adobe RGB instead would gain nothing: no colour in the palette
# lies outside sRGB, so the wider space would carry the same colours.
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

HERE = Path(__file__).parent
BUILD = HERE / "build"
INSETS = BUILD / "insets"

DPI = 300
LAND = (0x0d, 0x0d, 0x0f)
BAND = (0x0d, 0x0d, 0x0f)
FRAME = (0xba, 0x9a, 0x55, 0xff)     # muted gold, thin
LOCATOR = (0xba, 0x9a, 0x55, 0xcc)
SCALE = (0x8a, 0x92, 0x9c, 0xdd)
# match the caption type: 5.4 pt at 300 ppi = 22.5 px
SCALE_PT = 5.4
SCALE_FONT = (ImageFont.truetype(str(REGULAR), int(round(SCALE_PT / 72 * DPI)))
              if REGULAR.exists() else ImageFont.load_default())
MAP_SCALE_PT = 9.0
MAP_SCALE_FONT = (ImageFont.truetype(str(REGULAR),
                                     int(round(MAP_SCALE_PT / 72 * DPI)))
                  if REGULAR.exists() else ImageFont.load_default())
# insets whose busy corner forces the scale bar to the other side
LEFT_SCALE = {"la"}


def px(inches):
    return int(round(inches * DPI))


def matching(pattern, want_w):
    """The render whose width matches layout.json. Deliberately not 'the
    widest' — after the frame is trimmed, the widest file on disk is the stale
    one from the previous extent."""
    hits = list(BUILD.glob(pattern))
    if not hits:
        raise SystemExit(f"missing {pattern} in {BUILD} — run render.py first")
    for p in hits:
        if abs(int(p.stem.split("_")[-1].split("x")[0]) - want_w) <= 2:
            return p
    have = ", ".join(sorted(p.name for p in hits))
    raise SystemExit(
        f"no {pattern} at {want_w}px wide (have: {have}). Re-run render.py — "
        f"it reads the frame from layout.json, so these are out of step.")


def main():
    lay = json.loads((BUILD / "layout.json").read_text(encoding="utf-8"))
    SW, SH = lay["sheet_in"]
    MW, MH = lay["map_in"]
    W, H = px(SW), px(SH)
    print(f"sheet {SW} x {SH} in -> {W} x {H} px at {DPI} ppi")

    base_src = matching("base_dark_*x*.png", W)
    dots_src = matching("dots_*x*.png", W)
    print(f"  base {base_src.name}\n  dots {dots_src.name}")

    base_img = Image.open(base_src).convert("RGB")
    want = (px(MW), px(MH))
    off = (abs(base_img.width - want[0]), abs(base_img.height - want[1]))
    if max(off) > 2:
        raise SystemExit(
            f"{base_src.name} is {base_img.size}, but layout.json says the map "
            f"is {want}. Re-run render.py — it reads the frame from "
            f"layout.json, so a gap this size means the two are out of step.")
    # a pixel or two is just inches-to-px rounding (20.13 in -> 6038 vs 6039)
    MAP_W, MAP_H = base_img.size

    # 1 — base: map, then the band ground below it
    l1 = Image.new("RGB", (W, H), BAND)
    l1.paste(base_img, (0, 0))

    # 2 — dots
    l2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    l2.paste(Image.open(dots_src).convert("RGBA"), (0, 0))

    # 3 — insets and captions
    l3 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for p in lay["insets"]:
        x, y = px(p["x_in"]), px(p["y_in"])
        im = Image.open(INSETS / f"{p['name']}.png").convert("RGBA")
        want = (px(p["w_in"]), px(p["h_in"]))
        if im.size != want:
            im = im.resize(want, Image.LANCZOS)
        l3.paste(im, (x, y))
        cap = INSETS / f"{p['name']}_legend.png"
        if cap.exists():
            ci = Image.open(cap).convert("RGBA")
            l3.paste(ci, (x, y + im.height), ci)

    # 4 — frames, locators, and each inset's own scale bar
    l4 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d4 = ImageDraw.Draw(l4)
    km_in_main = lay["km_per_inch"]
    for p in lay["insets"]:
        x, y = px(p["x_in"]), px(p["y_in"])
        d4.rectangle([x, y, x + px(p["w_in"]) - 1, y + px(p["box_h_in"]) - 1],
                     outline=FRAME, width=3)

        # scale bar in the bottom-right of the map region. Insets run at
        # different magnifications, so pick a round distance rather than a
        # fixed one. A twenty-fifth of the window keeps it a discreet tick —
        # at a fifth it read as a graphic element in its own right.
        km_in = km_in_main / p.get("mag", 5.0)
        target = p["w_in"] / 12.5 * km_in
        nice = min((n for n in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250,
                                500, 1000) if n >= target), default=1000)
        bar = px(nice / km_in)
        by = y + px(p["h_in"]) - px(0.12)
        if p["name"] in LEFT_SCALE:
            # LA's bottom-right corner is dense; the bottom-left is ocean
            bx = x + px(0.10)
            d4.text((bx, by - 6), f"{nice} km", fill=SCALE, anchor="ls",
                    font=SCALE_FONT)
        else:
            bx = x + px(p["w_in"]) - bar - px(0.10)
            d4.text((bx + bar, by - 6), f"{nice} km", fill=SCALE, anchor="rs",
                    font=SCALE_FONT)
        d4.rectangle([bx, by, bx + bar, by + 3], fill=SCALE)
        loc = p.get("locator_in")
        if loc:
            lx, ly, lw, lh = loc
            d4.rectangle([px(lx), px(ly), px(lx + lw), px(ly + lh)],
                         outline=LOCATOR, width=2)
    # hairline between the map and the band
    d4.line([0, MAP_H, W, MAP_H], fill=(0x2a, 0x30, 0x38, 0xff), width=2)

    # 5 — legend band text
    l5 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    band = Image.open(BUILD / "legend_band.png").convert("RGBA")
    if band.width != W:
        band = band.resize((W, int(band.height * W / band.width)), Image.LANCZOS)
    l5.paste(band, (0, MAP_H), band)

    # Main scale bar, on the map rather than in the band. It used to sit in the
    # bottom-left corner, but the left rail now reaches the bottom margin, so
    # it starts clear of whatever occupies the bottom-left instead.
    d5 = ImageDraw.Draw(l5)
    bottom_edge = max(
        (p["x_in"] + p["w_in"] for p in lay["insets"]
         if p["x_in"] < 12.0 and p["y_in"] + p["box_h_in"] > MH - 2.0),
        default=0.0)
    sx = px(max(bottom_edge + 0.45, 0.34))
    sy = MAP_H - px(0.34)
    sbar = px(500 / km_in_main)
    d5.rectangle([sx, sy, sx + sbar, sy + 5], fill=(0xba, 0xc2, 0xcc, 0xff))
    d5.text((sx + sbar + px(0.10), sy + 5), "500 km", fill=(0xba, 0xc2, 0xcc, 0xff),
            anchor="ls", font=MAP_SCALE_FONT)

    layers = [("poster_1_base.png", l1), ("poster_2_dots.png", l2),
              ("poster_3_insets.png", l3), ("poster_4_frames.png", l4),
              ("poster_5_text.png", l5)]
    for name, im in layers:
        im.save(BUILD / name, icc_profile=SRGB)
        print(f"  wrote {name:22s} {(BUILD / name).stat().st_size / (1 << 20):6.1f} MB")

    flat = l1.convert("RGBA")
    for _, im in layers[1:]:
        flat = Image.alpha_composite(flat, im)
    flat.convert("RGB").save(BUILD / "poster_flat.png", icc_profile=SRGB)
    sz = (BUILD / "poster_flat.png").stat().st_size / (1 << 20)
    print(f"  wrote poster_flat.png        {sz:6.1f} MB"
          f"   {'OK' if sz < 100 else 'OVER the 100 MB upload cap'}")


if __name__ == "__main__":
    main()
