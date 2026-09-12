"""
Assemble the japanrail poster into layered full-sheet PNGs for Aseprite.

Everything is placed from build/layout.json, so the layers register with each
other and with the plan by construction.

    python compose.py --theme dark
    python compose.py --theme both

Writes, all at the full sheet size:

    poster_1_base_<theme>.png    ocean, lakes, coastline          (opaque)
    poster_2_glow_<theme>.png    the glow behind the lines        (alpha)
    poster_3_lines_<theme>.png   the national network             (alpha)
    poster_4_insets_<theme>.png  the city insets and captions     (alpha)
    poster_5_frames_<theme>.png  inset borders, locators, leaders (alpha)
    poster_flat_<theme>.png      all of it, for looking at

Frames are their own layer so they can be restyled or dropped without
re-rendering anything.

The file is tagged sRGB. Everything here is authored in sRGB — the line colours
come from the web map's hex — and Lumaprints recommends Adobe RGB, so an
untagged file risks being read as Adobe RGB and printing oversaturated.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageCms, ImageDraw, ImageFont

from typeface import REGULAR, BOLD, JP_REGULAR, JP_BOLD, JP_INDEX

Image.MAX_IMAGE_PIXELS = None

SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

HERE = Path(__file__).parent
BUILD = HERE / "build"
INSETS = BUILD / "insets"
DPI = 300

THEME = {
    # bubble / bubble_edge / bubble_alpha mirror render.THEMES so the key in
    # the legend is drawn the same way the map draws them.
    "dark": dict(ground="#16181b", frame="#b9964f", locator="#5bb6ff",
                 leader="#5bb6ff", text="#e6ecf3", dim="#8a929c",
                 bubble="#e8eef6", bubble_edge=None, bubble_alpha=0.45),
    "light": dict(ground="#f4f1ea", frame="#8a6d28", locator="#1f6fb0",
                  leader="#1f6fb0", text="#1b2027", dim="#5d6772",
                  bubble="#ffffff", bubble_edge="#151a20", bubble_alpha=0.62),
}


# Targets whose rename was refused this run, reported together at the end.
LOCKED = []


def px(inches):
    return int(round(inches * DPI))


def save(im, path, **kw):
    """Write via a temp file and rename into place.

    Windows refuses to truncate a file another process has open — an image
    viewer left on one of these layers is enough — and that aborted a whole
    two-theme compose halfway through, leaving the run's earlier outputs
    stale and the later ones missing. Writing beside the target and renaming
    means a lock costs one file with a clear message, not the run.
    """
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    im.save(tmp, **kw)
    # Retry briefly. The file that fails moves around between runs, which is
    # the signature of a scanner (search indexer, antivirus) holding whichever
    # freshly-written file it happens to be on rather than a viewer parked on
    # one layer. A second or two is enough for those; a viewer needs closing.
    exc = None
    for attempt in range(4):
        try:
            os.replace(tmp, path)
            return True
        except OSError as e:
            exc = e
            time.sleep(0.4 * (attempt + 1))
    if exc is not None:
        # One held file must not cost the other eleven. The new version is
        # left beside the target under its .tmp name, so the run is complete
        # apart from a rename the user can do once the viewer is closed.
        LOCKED.append((path, tmp, exc))
        return False
    return True


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgba(h, a=255):
    return rgb(h) + (a,)


# Type floor. Everything is authored at the sheet's design width, but the sheet
# may well be ordered smaller, and the binding case is a 30 in print: at
# 38.6 in of design that is a 0.777 scale, so a 9.5 pt line lands at 7.4 pt on
# paper. Nothing is allowed below MIN_PT once printed that small, so the floor
# in design points is MIN_PT * sheet_width / MIN_PT_AT_IN and every size passes
# through font().
MIN_PT = 8.0
MIN_PT_AT_IN = 30.0
_FLOOR_PT = MIN_PT

# Inset caption: the city name alone on the left, the scale bar on the right
# with the magnification sitting over it. The name was briefly derived from the
# strip height, which filled it exactly and came out at 34 pt — too heavy
# against the poster's own 46 pt title. 28 pt is set, and both the name and the
# scale group are centred in the strip rather than padded from its top.
CAP_PAD_IN = 0.14
# 28 pt measured correct but read large, because a descender ("Nagoya", "Tokyo")
# takes the line to 105 px of the strip's 186 where a name without one takes 85.
CAP_NAME_PT = 20.0


def set_type_floor(sheet_w_in):
    global _FLOOR_PT
    _FLOOR_PT = MIN_PT * sheet_w_in / MIN_PT_AT_IN
    return _FLOOR_PT


_LANG = "en"
# Fraction of the em that is visible ink. Nunito puts caps and ascenders at
# 0.72; a Japanese face fills far more of its em box, so the same point size
# reads noticeably bigger and the padding maths has to know which it is.
EM_VISUAL = {"en": 0.72, "ja": 0.88}


def set_lang(lang):
    global _LANG
    _LANG = lang
    return lang


def S(key, **fmt):
    """A sheet string in the current language."""
    v = STRINGS[_LANG][key]
    return v.format(**fmt) if fmt else v


def font(pt, bold=False):
    # Scale down first, clamp second: the floor exists so nothing falls under
    # 8 pt at a 30 in print, and shrinking past it would defeat that.
    if _LANG == "ja":
        pt *= JA_PT_SCALE
    pt = max(pt, _FLOOR_PT)
    size = int(round(pt / 72 * DPI))
    if _LANG == "ja" and JP_REGULAR and JP_BOLD:
        path = JP_BOLD if bold else JP_REGULAR
        return ImageFont.truetype(str(path), size, index=JP_INDEX)
    path = BOLD if (bold and BOLD) else REGULAR
    if not path or not path.exists():
        return ImageFont.load_default()
    return ImageFont.truetype(str(path), size)


def small(bold=False):
    """The smallest type allowed on the sheet."""
    return font(0, bold)


def visual_h(pt):
    """Inches of visible line height — the number to lay padding out against,
    rather than the em or the full ascender-to-descender box."""
    return EM_VISUAL[_LANG] * max(pt, _FLOOR_PT) / 72.0


# Characters that may not begin a line (kinsoku shori). Without this the
# character wrap happily starts a line with a full stop or a closing bracket.
JA_NO_LINE_START = "、。，．・：；？！ゝゞヽヾー）」』】〕〉》”’%℃"
JA_NO_LINE_END = "（「『【〔〈《“‘"

# The Japanese copy is not purely Japanese: the sources paragraph carries
# "OpenStreetMap contributors", "HydroLAKES", "Messager et al. 2016",
# "gtfs-gis.jp". A per-character wrapper breaks straight through the middle of
# those ("Mes" / "sager"), which the Latin branch would never do. Alphanumerics
# only, so a break may still fall at a hyphen, dot or slash — those are real
# break opportunities inside "gtfs-gis.jp" and "anita.garden/japanrail/".
JA_LATIN_RUN = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                   "abcdefghijklmnopqrstuvwxyz0123456789")


def wrap(d, text, f, max_px):
    """Break a paragraph to a measured width.

    Word-wrap by advance width for Latin — the copy used to be hand-broken into
    fixed lines, which silently overran the block the moment the type size
    moved. Japanese has no spaces, so a whole paragraph is one "word" and the
    word wrapper returns it as a single long line; it breaks per character
    instead, with enough kinsoku not to strand punctuation at a line start.
    """
    if _LANG == "ja":
        out, line = [], ""
        for ch in text:
            if line and d.textlength(line + ch, font=f) > max_px:
                if ch in JA_NO_LINE_START:
                    out.append(line + ch)      # overhang rather than lead
                    line = ""
                elif line[-1] in JA_NO_LINE_END:
                    out.append(line[:-1])      # carry the opener down with it
                    line = line[-1] + ch
                elif ch in JA_LATIN_RUN and line[-1] in JA_LATIN_RUN:
                    # Mid-token break. Back up over the trailing Latin run and
                    # carry the whole token down, unless the run *is* the line
                    # — a single token longer than the measure has to overhang.
                    cut = len(line)
                    while cut > 0 and line[cut - 1] in JA_LATIN_RUN:
                        cut -= 1
                    # Backing up can land just after an opening bracket, which
                    # strands it at the line end — the very thing
                    # JA_NO_LINE_END prevents, reintroduced one step later.
                    # "HydroLAKES（" / "Messager" was the real case.
                    while cut > 0 and line[cut - 1] in JA_NO_LINE_END:
                        cut -= 1
                    if cut == 0:
                        out.append(line)
                        line = ch
                    else:
                        out.append(line[:cut])
                        line = line[cut:] + ch
                else:
                    out.append(line)
                    line = ch
                continue
            line += ch
        if line:
            out.append(line)
        return out

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


def nice_distance(target):
    return min((n for n in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000)
                if n >= target), default=1000)


def text_w(d, s, f):
    return d.textlength(s, font=f)


def scale_bar(d, x, y, km_per_in, target_in, col, pt=7.0, right_edge=None):
    """A plain bar with its distance written beside it. One bar, not a two-tone
    ladder: the ladder read as two bars and left it ambiguous whether the
    number covered one or both.

    y is the bar's CENTRE and the label is vertically centred on it. Sitting
    the bar on the label's baseline made a 2 px rule read as a long underscore
    under the text rather than as a measurement.

    Pass right_edge to right-align the whole bar+label group inside a box. The
    first version placed the bar at a fixed offset and let the label run past
    the frame, which is exactly what it did in the inset captions.
    """
    km = nice_distance(target_in * km_per_in)
    w = px(km / km_per_in)
    f = font(pt)
    label = S("km", km=km)
    gap = px(0.09)
    if right_edge is not None:
        x = right_edge - text_w(d, label, f) - gap - w
    t = max(3, px(0.022))
    d.rectangle([x, y - t // 2, x + w, y - t // 2 + t], fill=col)
    d.text((x + w + gap, y), label, fill=col, anchor="lm", font=f)
    return x, w


# ── text blocks ────────────────────────────────────────────────────────────
# Set to a narrow measure and at a size that reads across a room, rather than
# filling the block's width: one 12.6 in line of small type was a very long
# way for an eye to travel back along.
BODY_PT = 15.0
SRC_PT = 11.5
BODY_STEP_IN = 0.285
SRC_STEP_IN = 0.225

# Every string that appears on the sheet, per language. The Japanese edition is
# the native one — the data's own line, operator and station names are Japanese
# and the English map is the translation, not the other way round.
#
# NOTE: the Japanese copy here is DRAFT and wants a native read before this is
# printed. The terms of art (輸送密度, 乗降客数, 通過人員, 大都市交通センサス) are
# taken from the sources themselves and should be right; the connecting prose
# is mine. No em dashes in the small print — they read as gaps at this size.
STRINGS = {
    "en": {
        "title": "Japan Rail Flow",
        "body": [
            "Line thickness is proportional to passenger throughput. Passenger "
            "counts are per day, averaged over a year. Station circle size is "
            "also proportional to daily boardings and alightings.",
        ],
        "sources": [
            "Sources: gtfs-gis.jp Railway Transport Density data (Akira "
            "Nishizawa, Regional and Transport Data Institute), FY2024 for the "
            "JR companies that publish per segment and FY2023 elsewhere. "
            "Station-to-station detail across Greater Tokyo, Nagoya and "
            "Keihanshin from the 12th Metropolitan Transportation Census "
            "(2015), rescaled to each line's fiscal-year total. Geometry and "
            "station ridership from National Land Numerical Information N02 "
            "and S12 (MLIT); station figures are FY2021. Coastline (c) "
            "OpenStreetMap contributors. Lakes from HydroLAKES (Messager et "
            "al. 2016), CC BY 4.0.",
            "Projection: Lambert conformal conic rotated 25°",
        ],
        "credit": "interactive version at anita.garden/japanrail/",
        "seg_head": "Segment passengers per day",
        "stn_head": "Station passengers per day",
        "on_main": "on the main map",
        "on_inset": "in the city insets",
        "busiest": "Kanda–Tokyo, the busiest",
        "mag": "{mag:.1f}x scale",
        "km": "{km} km",
    },
    "ja": {
        "title": "日本の鉄道輸送密度",
        "body": [
            "線の太さは輸送密度（区間ごとの1日あたり通過人員）に比例します。"
            "人数は年度平均の1日あたりの値です。"
            "駅の円の大きさは1日あたりの乗降客数に比例します。",
        ],
        "sources": [
            "出典：gtfs-gis.jp 鉄道輸送密度データ（西澤明、地域・交通データ研究所）。"
            "区間別の値を公表しているJR各社は2024年度、その他は2023年度。"
            "首都圏・中京圏・近畿圏の駅間通過人員は第12回大都市交通センサス"
            "（2015年）を各線の年度合計に合わせて調整。"
            "路線形状と駅別乗降客数は国土数値情報 N02・S12（国土交通省）、"
            "駅の数値は2021年度。海岸線は (c) OpenStreetMap contributors。"
            "湖沼は HydroLAKES（Messager et al. 2016）、CC BY 4.0。",
            "投影法：ランベルト正角円錐図法、時計回りに25度回転",
        ],
        "credit": "インタラクティブ版 anita.garden/japanrail/",
        "seg_head": "区間別 1日あたり通過人員",
        "stn_head": "駅別 1日あたり乗降客数",
        "on_main": "全国図",
        "on_inset": "都市図",
        "busiest": "神田〜東京（最多区間）",
        "mag": "{mag:.1f}倍",
        "km": "{km} km",
    },
}

# Point sizes that need to come down for Japanese, because a CJK face fills far
# more of its em than Nunito does at the same size.
JA_PT_SCALE = 0.86

# Where the map's own scale bar sits, in sheet inches (the bar's left end and
# its vertical centre). Open Pacific: land in these rows stops at x = 21.0.
MAP_SCALE_AT = (22.30, 20.35)


def draw_title(d, lay, T):
    t = lay["title"]
    x, y = px(t["x_in"]), px(t["y_in"])
    measure = px(t["w_in"])
    d.text((x, y), S("title"), fill=rgba(T["text"]), font=font(46, bold=True),
           anchor="la")
    d.line([x, y + px(0.80), x + measure, y + px(0.80)],
           fill=rgba(T["dim"], 0x66), width=max(2, px(0.005)))

    f_body, f_src = font(BODY_PT), font(SRC_PT)
    yy = y + px(1.06)
    for para in S("body"):
        for line in wrap(d, para, f_body, measure):
            d.text((x, yy), line, fill=rgba(T["text"]), font=f_body,
                   anchor="la")
            yy += px(BODY_STEP_IN)
        yy += px(0.10)
    yy += px(0.10)
    for para in S("sources"):
        for line in wrap(d, para, f_src, measure):
            d.text((x, yy), line, fill=rgba(T["dim"]), font=f_src, anchor="la")
            yy += px(SRC_STEP_IN)
        yy += px(0.06)
    d.text((x, yy), S("credit"), fill=rgba(T["dim"]), font=f_src, anchor="la")
    return (yy + px(SRC_STEP_IN) - y) / DPI      # inches actually used


# Throughput values the ladder names. The top one is the busiest segment in the
# data (東北線 神田–東京), so the ladder's top rung is a real line rather than a
# round number the map never reaches.
LADDER = [(2000, "2,000"), (20000, "20,000"), (100000, "100,000"),
          (400000, "400,000"), (1442714, None)]   # None -> "…, the busiest"
BUBBLES = [(50000, "50,000"), (500000, "500,000"), (2204829, "2,200,000")]
# The ladder is deliberately tight — 0.62 in of bar and 0.235 in between rows,
# against 1.70 and 0.46 in the first pass, which sprawled over a third of the
# block and read as five separate rules rather than as one scale. It follows
# the interactive map's own legend, where the swatches are 34 px wide.
LADDER_BAR_IN = 0.62
LADDER_STEP_IN = 0.215
# Bubble rows. The inset key's 2.2M circle is 0.27 in across, so this is the
# floor before the rows start touching.
BUBBLE_ROW_IN = 0.34


def draw_legend(d, lay, T, km_in):
    """Thickness ladder and the two bubble keys, drawn from the same numbers
    the map used — width_scale comes out of the render's own frame JSON, so the
    legend cannot quietly disagree with the lines it explains.

    The scale bar is NOT here: sitting under the ladder it read as one more
    rung. It goes on the map, as it does on the ancestrydots sheet.
    """
    import render as R
    from insets import BUBBLE_SCALE

    g = lay["legend"]
    ws = lay["frame"].get("width_scale", {})
    knee = ws.get("knee", 5.0)
    minw, maxw = ws.get("min_width", 0.008), ws.get("max_width", 0.10)
    main_bubble = lay["frame"].get("bubble_scale") or BUBBLE_SCALE / 3.0

    x, y = px(g["x_in"]), px(g["y_in"])
    f_head, f_lab = font(15, bold=True), font(SRC_PT)
    d.text((x, y), S("seg_head"), fill=rgba(T["text"]), font=f_head,
           anchor="la")
    bar_x = x + px(0.08)
    bar_len = px(LADDER_BAR_IN)
    yy = y + px(0.42)
    for value, label in LADDER:
        if label is None:
            label = f"1,440,000  {S('busiest')}"
        t = float(R.width_t(value, knee))
        w = max(1, px(minw + (maxw - minw) * t))
        d.rectangle([bar_x, yy - w // 2, bar_x + bar_len, yy - w // 2 + w],
                    fill=rgba(T["text"]))
        d.text((bar_x + bar_len + px(0.16), yy), label, fill=rgba(T["text"]),
               font=f_lab, anchor="lm")
        yy += px(LADDER_STEP_IN)

    # Two bubble keys side by side, because the two scales are different: the
    # national map draws them at half the inset size, and a single key would be
    # wrong for one of the two places a reader looks.
    yy += px(0.26)
    d.text((x, yy), S("stn_head"), fill=rgba(T["text"]), font=f_head,
           anchor="la")
    yy += px(0.38)
    cols = [(S("on_main"), main_bubble, x + px(0.08)),
            (S("on_inset"), BUBBLE_SCALE, x + px(3.60))]
    for head, scale, cx in cols:
        d.text((cx, yy), head, fill=rgba(T["dim"]), font=f_lab, anchor="la")
    yy += px(0.42)
    row = px(BUBBLE_ROW_IN)
    # Match however render.py drew them for this theme, outline included.
    b_fill = rgba(T["bubble"], int(round(255 * T["bubble_alpha"])))
    b_edge = (rgba(T["bubble_edge"], int(round(255 * 0.30)))
              if T.get("bubble_edge") else None)
    for i, (value, label) in enumerate(BUBBLES):
        for head, scale, cx in cols:
            r = px(value ** 0.5 * scale)
            cy = yy + i * row
            d.ellipse([cx + px(0.18) - r, cy - r, cx + px(0.18) + r, cy + r],
                      fill=b_fill, outline=b_edge,
                      width=max(1, px(0.005)) if b_edge else 0)
            d.text((cx + px(0.54), cy), label, fill=rgba(T["text"]),
                   font=f_lab, anchor="lm")
    return (yy + 3 * row - y) / DPI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", choices=["dark", "light", "both"], default="dark")
    ap.add_argument("--lang", choices=["en", "ja", "both"], default="en",
                    help="ja needs a CJK face; see typeface.py. Japanese "
                         "output is suffixed _ja, English is not.")
    ap.add_argument("--tag", default="v8",
                    help="which national render to use (build/*_<tag>.png)")
    args = ap.parse_args()

    lay = json.loads((BUILD / "layout.json").read_text(encoding="utf-8"))
    SW, SH = lay["sheet_in"]
    W, H = px(SW), px(SH)
    km_in = lay["km_per_inch"]
    cap_in = lay["caption_in"]
    floor = set_type_floor(SW)
    print(f"sheet {SW} x {SH} in -> {W} x {H} px at {DPI} ppi")
    print(f"  type floor {floor:.1f} pt, which is {MIN_PT:.0f} pt if this "
          f"sheet is printed {MIN_PT_AT_IN:.0f} in wide")

    themes = ["dark", "light"] if args.theme == "both" else [args.theme]
    langs = ["en", "ja"] if args.lang == "both" else [args.lang]
    if "ja" in langs and not JP_REGULAR:
        raise SystemExit("no Japanese face available — see typeface.py")
    for theme, lang in ((t, l) for l in langs for t in themes):
        set_lang(lang)
        suffix = "" if lang == "en" else f"_{lang}"
        T = THEME[theme]
        base_src = BUILD / f"base_{theme}_{args.tag}.png"
        if not base_src.exists():
            raise SystemExit(f"missing {base_src.name} — run "
                             f"`python render.py --theme {theme} --tag "
                             f"{args.tag}` first")
        base_img = Image.open(base_src).convert("RGB")
        if abs(base_img.width - W) > 2 or abs(base_img.height - H) > 2:
            raise SystemExit(
                f"{base_src.name} is {base_img.size} but layout.json says the "
                f"sheet is {(W, H)}. Re-run render.py and layout.py together — "
                f"they are out of step.")
        W2, H2 = base_img.size

        l1 = Image.new("RGB", (W2, H2), rgb(T["ground"]))
        l1.paste(base_img, (0, 0))

        def load(p):
            im = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
            if p.exists():
                im.paste(Image.open(p).convert("RGBA"), (0, 0))
            return im

        l2 = load(BUILD / f"glow_{args.tag}.png")
        l3 = load(BUILD / f"lines_{args.tag}.png")
        bub = BUILD / f"bubbles_{theme}_{args.tag}.png"
        if bub.exists():
            # bubbles ride on the lines layer so the two stay one object when
            # the sheet is taken apart in Aseprite
            l3 = Image.alpha_composite(l3, load(bub))

        # ---- 4: insets and captions --------------------------------------
        l4 = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
        d4 = ImageDraw.Draw(l4)
        for p in lay["insets"]:
            src = INSETS / f"{p['name']}_{theme}.png"
            if not src.exists():
                print(f"  !! {src.name} missing — run insets.py --from-layout")
                continue
            x, y = px(p["x_in"]), px(p["y_in"])
            im = Image.open(src).convert("RGBA")
            want = (px(p["w_in"]), px(p["h_in"]))
            if im.size != want:
                im = im.resize(want, Image.LANCZOS)
            l4.paste(im, (x, y))

            # Caption strip, on the inset's own ground so it reads as one
            # plate. Placed off BASELINES computed from CAP_PAD and the visual
            # line heights, so the space above the name, below it, and inside
            # both edges is the same. Anchoring by ascender instead left the
            # bottom padding visibly adrift from the other three.
            cy0 = y + im.height
            d4.rectangle([x, cy0, x + px(p["w_in"]) - 1,
                          cy0 + px(cap_in) - 1], fill=rgba(T["ground"]))

            name_h = visual_h(CAP_NAME_PT)
            name_base = cy0 + px((cap_in - name_h) / 2 + name_h)
            label = p.get("label_ja", p["label"]) if _LANG == "ja" \
                else p["label"]
            d4.text((x + px(CAP_PAD_IN), name_base), label,
                    fill=rgba(T["text"]), font=font(CAP_NAME_PT, bold=True),
                    anchor="ls")

            # Right: "6.9x scale" stacked over the bar, the pair centred in the
            # strip. The km-per-inch figure is gone — it is the same fact as
            # the bar, said less usefully.
            f_mag = small()
            bar_t = max(3, px(0.022))
            group = px(visual_h(0) + 0.055) + bar_t
            top = cy0 + (px(cap_in) - group) // 2
            right = x + px(p["w_in"]) - px(CAP_PAD_IN)
            bx, bw = scale_bar(d4, 0, top + group - bar_t // 2,
                               p["km_per_inch"], 0.9, rgba(T["dim"]), pt=0,
                               right_edge=right)
            # right-aligned to the BAR's end, not the box's, so it sits over
            # the bar rather than over the "10 km" that follows it
            d4.text((bx + bw, top + px(visual_h(0))), S("mag", mag=p["mag"]),
                    fill=rgba(T["dim"]), font=f_mag, anchor="rs")

        # ---- 5: frames, locators, leaders --------------------------------
        l5 = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
        d5 = ImageDraw.Draw(l5)
        lw = max(3, px(0.012))
        for p in lay["insets"]:
            x, y = px(p["x_in"]), px(p["y_in"])
            bw, bh = px(p["w_in"]), px(p["box_h_in"])
            quad = [(px(a), px(b)) for a, b in p["locator_in"]]
            # No leader line from the locator to its box. A faint hairline
            # running across the country reads as one more railway, which on a
            # map whose entire content is faint hairlines is the one thing it
            # must not do. The locator square and the caption's city name
            # carry it instead.
            d5.polygon(quad, outline=rgba(T["locator"], 0xdd))
            d5.rectangle([x, y, x + bw - 1, y + bh - 1],
                         outline=rgba(T["frame"]), width=lw)
            # rule between the map window and its caption
            d5.line([x, y + px(p["h_in"]), x + bw - 1, y + px(p["h_in"])],
                    fill=rgba(T["frame"], 0x99), width=max(2, px(0.005)))

        used_t = draw_title(d5, lay, T)
        used_l = draw_legend(d5, lay, T, km_in)
        for name, used, box in (("title", used_t, lay["title"]),
                                ("legend", used_l, lay["legend"])):
            if used > box["h_in"] + 1e-6:
                print(f"  !! {name} block needs {used:.2f} in but layout.py "
                      f"reserved {box['h_in']:.2f} — it will run into whatever "
                      f"is below")
        # Map scale bar, on the map rather than in the legend block — the same
        # placement ancestrydots settled on. It sits in the Pacific south of
        # the Kii peninsula, clear of both the country and the Tokyo inset.
        scale_bar(d5, px(MAP_SCALE_AT[0]), px(MAP_SCALE_AT[1]), km_in, 3.0,
                  rgba(T["dim"]), pt=11)

        names = [(f"poster_1_base_{theme}{suffix}.png", l1),
                 (f"poster_2_glow_{theme}{suffix}.png", l2),
                 (f"poster_3_lines_{theme}{suffix}.png", l3),
                 (f"poster_4_insets_{theme}{suffix}.png", l4),
                 (f"poster_5_frames_{theme}{suffix}.png", l5)]
        for name, im in names:
            ok = save(im, BUILD / name, icc_profile=SRGB)
            shown = (BUILD / name) if ok else LOCKED[-1][1]
            print(f"  {'wrote' if ok else 'LOCKED, left as .tmp':21s} "
                  f"{name:28s} {shown.stat().st_size / (1 << 20):6.1f} MB")

        flat = l1.convert("RGBA")
        for _, im in names[1:]:
            flat = Image.alpha_composite(flat, im)
        out = BUILD / f"poster_flat_{theme}{suffix}.png"
        ok = save(flat.convert("RGB"), out, icc_profile=SRGB)
        shown = out if ok else LOCKED[-1][1]
        sz = shown.stat().st_size / (1 << 20)
        print(f"  {'wrote' if ok else 'LOCKED, left as .tmp':21s} "
              f"{out.name:28s} {sz:6.1f} MB"
              f"   {'OK' if sz < 100 else 'OVER the 100 MB upload cap'}")

    if LOCKED:
        print(f"\n{len(LOCKED)} file(s) could not be replaced — something has "
              f"them open (an image viewer on that layer will do it). The new "
              f"version is beside each one as .tmp.png; close the viewer and "
              f"re-run, or rename them by hand:")
        for path, tmp, exc in LOCKED:
            print(f"  {path.name}  <-  {tmp.name}")


if __name__ == "__main__":
    main()
