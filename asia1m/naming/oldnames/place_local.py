"""Chinese-character labels for the local-script map, checked against her current local layer.

    python asia1m/naming/oldnames/place_local.py [--all] [--avoid-english] [--fresh]

Reads her local text layer as it stands (textlocal.png, the .ase layer `textlocal`, exported
from a copy of the file), works out which of her labels agree with the data, and places only
what is missing. Writes, in this folder:
  local_labels_new.png     transparent 9500x8000 layer with only the labels she needs to add
  local_placements.csv     those labels: key, zh, x, y, w, h, layout, placed_inside, dist, anchor, where
  local_unplaced.csv       regions that need a label but got none, and why
  her_labels.csv           every label found on her layer: colour, box, region, what it reads as
  changes_vs_current.csv   what she would add, remove or fix on her layer
  changes_vs_current.png   a 2 px red box (3 px margin) round each of those, to lay over her file

Names: oldnames.csv `zh` (labels read off the map, China) and ../local_names.csv `zh` (picks).
`native` (Tibetan, Uyghur, Kazakh, Kirghiz) is not drawn: those scripts need shaping and this
PIL has no libraqm; such regions are listed in local_unplaced.csv.

Style: Ark Pixel 12px (monospaced zh_cn) at 12 px, 1-bit (SimSun 12 px for characters Ark Pixel
lacks); colour = the region's colour x 0.75, rounded up.

Her labels: her layer's text is cut into labels (pixels of one colour within 6 px of each
other). A label's colour names its region (as in assign.py), and it is read by comparing it
with the rendered characters of every region within 300 px: all three layouts, in Ark Pixel
zh_cn, zh_hk and ja and in Unifont, 1 px of play, 80% of pixels agreeing, characters Ark Pixel
lacks left out, a list key ("a.") dropped first. A label under 9 px tall is a key or mark. A
label is then: right; off its region (right, but more than 40 px from the region and its inset
copy, list entries aside); another region's name (in the colour of a region that has a
name of its own: rename); in no named region's colour but reading as a nearby region's name
(recolour); the first round's old characters (rename); unread but the size of the right number
of characters (another font or hand-drawn: counted as hers, not flagged); or different
characters (flagged to check). A city core whose English label is on an inset needs its
characters on the inset; a label of hers only on the main map is listed to move.

Placement, for regions with no label on her layer (one that reads as the region's name, or one
in the region's colour that reads as anything), greedy, smallest regions first:
near the English label (oldnames.csv), or the region's most interior point when the name is in
a list. A city core goes on its inset copy (insets.png) whenever her English label is on an
inset, or when the name is in a list and the inset copy is bigger than the main-map region.
Layouts tried: one line; one character per line; two lines for names of four or more. A spot is
allowed when its box, grown by one pixel, touches none of her text and stays 3 px clear of
labels placed before it; a main-map label also keeps off the inset footprints. Among allowed
spots within 60 px of the anchor the best is fully inside the region and nearest the anchor;
failing that, the one with most of its box on the region. A spot from the first round
(local_placements_v1.csv) is kept when it is still free (--fresh to ignore it). A first-round
label she has removed from both her layer and the imported layer (layer1.png, the .ase's
`Layer 1`) is not put back; it is listed for her to decide.
"""
import os, sys, csv, json, math, argparse, collections, functools
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy.ndimage as ndi
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
NAMING = os.path.join(HERE, "..")
ASIA = os.path.join(NAMING, "..")
RN = os.path.join(ASIA, "regionnaming")
FONT = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts",
                    "ark-pixel-12px-monospaced-zh_cn.ttf")
SEARCH = 60
GAP = 3          # blank pixels kept between two new labels (one more than the box check adds)
NATIVE_SHAPED = {"tibetan", "uyghur", "kazakh", "kirghiz"}

ap = argparse.ArgumentParser()
ap.add_argument("--all", action="store_true", help="also regions whose label on her layer is right")
ap.add_argument("--avoid-english", action="store_true", help="keep clear of the English text layer")
ap.add_argument("--fresh", action="store_true", help="ignore local_placements_v1.csv spots")
a = ap.parse_args()

regions = json.load(open(os.path.join(NAMING, "regions.json"), encoding="utf-8"))
R = {r["key"]: r for r in regions}
prelim = list(csv.DictReader(open(os.path.join(RN, "regions_prelim.csv"))))
by_color = collections.defaultdict(list)
for r in regions: by_color[r["color"]].append(r)
key2rid = {}
for p in prelim:
    cands = by_color.get(p["color"], [])
    if not cands: continue
    lon, lat = float(p["lon"]), float(p["lat"])
    r = min(cands, key=lambda r: (r["lon"] - lon) ** 2 + (r["lat"] - lat) ** 2)
    key2rid[r["key"]] = int(p["rid"])

def dark(h):
    return tuple(int(math.ceil(int(h[i:i + 2], 16) * 0.75)) for i in (1, 3, 5))
def hexc(c): return "#%02x%02x%02x" % tuple(int(v) for v in c)

# ---- names
want = {}      # key -> dict(zh, anchor or None, lbox, src)
MAPNAME = {}   # key -> the English name read off the map
skipped = []
for row in csv.DictReader(open(os.path.join(HERE, "oldnames.csv"), encoding="utf-8")):
    if not row["zh"]: continue
    off = ("list" in row["notes"] or "name search" in row["notes"])
    anchor = None if off else (int(row["label_x"]) + int(row["label_w"]) / 2,
                               int(row["label_y"]) + int(row["label_h"]) / 2)
    lbox = None if off else (int(row["label_x"]), int(row["label_y"]),
                             int(row["label_x"]) + int(row["label_w"]), int(row["label_y"]) + int(row["label_h"]))
    want[row["key"]] = dict(zh=row["zh"], anchor=anchor, lbox=lbox, src="oldnames " + row["confidence"])
    MAPNAME[row["key"]] = row["name"]
for row in csv.DictReader(open(os.path.join(NAMING, "local_names.csv"), encoding="utf-8-sig")):
    lang = (row.get("lang") or "").lower()
    if lang in NATIVE_SHAPED and row.get("native"):
        skipped.append(dict(key=row["key"], zh=row["zh"], why=f"{lang} script needs shaping; not drawn (zh would be {row['zh']})"))
        want.pop(row["key"], None)
        continue
    if not row["zh"]: continue
    prev = want.get(row["key"])
    want[row["key"]] = dict(zh=row["zh"], anchor=prev["anchor"] if prev else None,
                            lbox=prev["lbox"] if prev else None, src="picks")
    MAPNAME.setdefault(row["key"], row.get("name") or "")
want = {k: v for k, v in want.items() if k in R}
V1_PLACED = {}
if os.path.exists(os.path.join(HERE, "local_placements_v1.csv")):
    V1_PLACED = {p["key"]: p for p in csv.DictReader(open(os.path.join(HERE, "local_placements_v1.csv"), encoding="utf-8"))}

# ---- rasters
rid = np.load(os.path.join(RN, "region_id.npy"), mmap_mode="r")
H, W = rid.shape
ins = np.array(Image.open(os.path.join(HERE, "insets.png")).convert("RGBA"))
inset_mask = ins[:, :, 3] > 0
ins_rgb = ins[:, :, :3].copy(); ins_rgb[~inset_mask] = 0
del ins
tl = np.array(Image.open(os.path.join(HERE, "textlocal.png")).convert("RGBA"))
tl_mask = tl[:, :, 3] > 0
tl_rgb = tl[:, :, :3].copy()
del tl
l1 = np.array(Image.open(os.path.join(HERE, "layer1.png")).convert("RGBA")) if os.path.exists(os.path.join(HERE, "layer1.png")) else None

# inset copies: region colour -> boxes of its pieces on the insets (pieces 300 px apart are
# separate copies)
ys_, xs_ = np.nonzero(inset_mask)
code = (ins_rgb[ys_, xs_, 0].astype(np.int64) << 16) | (ins_rgb[ys_, xs_, 1].astype(np.int64) << 8) | ins_rgb[ys_, xs_, 2]
o_ = np.argsort(code, kind="stable"); code, ys_, xs_ = code[o_], ys_[o_], xs_[o_]
cuts = np.flatnonzero(np.diff(code)) + 1
colours = {r["color"] for r in regions}
inset_copies = {}
for a_, b_ in zip(np.r_[0, cuts], np.r_[cuts, len(code)]):
    h_ = "#%06x" % int(code[a_])
    if h_ not in colours or b_ - a_ < 4: continue
    bxs = []
    for x, y in sorted(zip(xs_[a_:b_].tolist(), ys_[a_:b_].tolist())):
        for bb in bxs:
            if bb[0] - 300 <= x <= bb[2] + 300 and bb[1] - 300 <= y <= bb[3] + 300:
                bb[0] = min(bb[0], x); bb[1] = min(bb[1], y); bb[2] = max(bb[2], x + 1); bb[3] = max(bb[3], y + 1)
                bb[4] += 1; break
        else:
            bxs.append([x, y, x + 1, y + 1, 1])
    inset_copies[h_] = bxs
del ys_, xs_, code, o_

def bbox_dist(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0); dy = max(b[1] - a[3], a[1] - b[3], 0)
    return (dx * dx + dy * dy) ** 0.5

@functools.lru_cache(maxsize=None)
def copies_of(k):
    """inset copies of region k: those whose nearest region of that colour is k"""
    out = []
    for bb in inset_copies.get(R[k]["color"], []):
        near = min(by_color[R[k]["color"]], key=lambda r: bbox_dist(bb, r["bbox"]))
        if near["key"] == k and bbox_dist(bb, R[k]["bbox"]) <= 1500: out.append(tuple(bb))
    return tuple(out)

def region_dist(box, k):
    d = bbox_dist(box, R[k]["bbox"])
    for bb in copies_of(k): d = min(d, bbox_dist(box, bb))
    return d

# ---- font
font = ImageFont.truetype(FONT, 12)
# Ark Pixel lacks some rarer characters (颍): those are drawn with SimSun 12 px, fitted to the cell
FALLBACK = ImageFont.truetype(r"C:\Windows\Fonts\simsun.ttc", 12)
NOTDEF = bytes(font.getmask("\U000F0001", mode="1"))
fallback_used = set()

@functools.lru_cache(maxsize=None)
def glyph(ch):
    if bytes(font.getmask(ch, mode="1")) != NOTDEF:
        im = Image.new("L", (12, 13), 0)
        d = ImageDraw.Draw(im); d.fontmode = "1"
        d.text((0, -1), ch, fill=255, font=font)
        return np.array(im)[:11, :11] > 127
    fallback_used.add(ch)
    im = Image.new("L", (24, 24), 0)
    d = ImageDraw.Draw(im); d.fontmode = "1"
    d.text((4, 4), ch, fill=255, font=FALLBACK)
    g = np.array(im) > 127
    out = np.zeros((11, 11), bool)
    if g.any():
        ys, xs = np.nonzero(g)
        g = g[ys.min():ys.min() + 11, xs.min():xs.min() + 11]
        oy, ox = (11 - g.shape[0]) // 2, (11 - g.shape[1]) // 2
        out[oy:oy + g.shape[0], ox:ox + g.shape[1]] = g
    return out

@functools.lru_cache(maxsize=None)
def render_t(lines):
    """bitmap of stacked lines, each glyph 11x11 on a 12 px pitch"""
    n = max(len(l) for l in lines)
    w, h = n * 12 - 1, len(lines) * 12 - 1
    out = np.zeros((h, w), bool)
    for i, l in enumerate(lines):
        off = (n - len(l)) * 6                  # centre shorter lines
        for j, ch in enumerate(l):
            out[i * 12:i * 12 + 11, off + j * 12:off + j * 12 + 11] = glyph(ch)
    return out
def render(lines): return render_t(tuple(lines))

def layouts(zh):
    out = [("line", [zh])]
    if len(zh) >= 2: out.append(("column", list(zh)))
    if len(zh) >= 4:
        k = math.ceil(len(zh) / 2)
        out.append(("two lines", [zh[:k], zh[k:]]))
    return out

# her own labels are not all in Ark Pixel zh_cn: some are in its zh_hk / ja variants, some in
# Unifont (16 px). Each is tried; a character Ark Pixel lacks (drawn here with SimSun) is left
# out of the comparison, since she may have drawn it some other way.
FONTS_DIR = os.path.dirname(FONT)
ALT = []
for fn, size, cell, pitch, dy in [("ark-pixel-12px-monospaced-zh_hk.ttf", 12, 11, 12, -1),
                                   ("ark-pixel-12px-monospaced-ja.ttf", 12, 11, 12, -1),
                                   ("unifont-17.0.04.otf", 16, 16, 16, 0)]:
    pth = os.path.join(FONTS_DIR, fn)
    if os.path.exists(pth): ALT.append((ImageFont.truetype(pth, size), cell, pitch, dy))

@functools.lru_cache(maxsize=None)
def alt_glyph(fi, ch):
    f, cell, pitch, dy = ALT[fi]
    im = Image.new("L", (cell + 1, cell + 2), 0)
    d = ImageDraw.Draw(im); d.fontmode = "1"
    d.text((0, dy), ch, fill=255, font=f)
    return np.array(im)[:cell, :cell] > 127

@functools.lru_cache(maxsize=None)
def variants(zh):
    """(layout, bitmap, mask of the cells that count) for every font and layout, cut to the ink"""
    out = []
    fonts = [None] + list(range(len(ALT)))
    for fi in fonts:
        cell, pitch = (11, 12) if fi is None else ALT[fi][1:3]
        for lname, lines in layouts(zh):
            n = max(len(l) for l in lines)
            h, w = len(lines) * pitch - (pitch - cell), n * pitch - (pitch - cell)
            bm = np.zeros((h, w), bool); valid = np.zeros((h, w), bool)
            for i, l in enumerate(lines):
                off = (n - len(l)) * pitch // 2
                for j, ch in enumerate(l):
                    g = glyph(ch) if fi is None else alt_glyph(fi, ch)
                    y0, x0 = i * pitch, off + j * pitch
                    bm[y0:y0 + cell, x0:x0 + cell] = g
                    if fi is not None or ch not in fallback_used:
                        valid[max(y0 - 1, 0):y0 + cell + 1, max(x0 - 1, 0):x0 + cell + 1] = True
            ys, xs = np.nonzero(bm)
            # leaving characters out is only safe when most of the name is still compared
            n_out = sum(1 for ch in zh if fi is None and ch in fallback_used)
            if not len(ys) or not valid.any() or (n_out and (len(zh) < 3 or n_out > 1)): continue
            sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
            out.append((lname, bm[sl], valid[sl]))
    return out

def reads_as(crop, zh):
    """does her label's ink (a bool crop cut to the ink) read as zh in one of the fonts?
    80% of the inked pixels agree, 1 px of play"""
    for lname, bm, valid in variants(zh):
        if abs(bm.shape[0] - crop.shape[0]) > 2 or abs(bm.shape[1] - crop.shape[1]) > 2: continue
        hh = max(bm.shape[0], crop.shape[0]) + 4; ww = max(bm.shape[1], crop.shape[1]) + 4
        B = np.zeros((hh, ww), bool); V = np.zeros((hh, ww), bool)
        B[2:2 + bm.shape[0], 2:2 + bm.shape[1]] = bm; V[2:2 + bm.shape[0], 2:2 + bm.shape[1]] = valid
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                A = np.zeros((hh, ww), bool)
                A[2 + dy:2 + dy + crop.shape[0], 2 + dx:2 + dx + crop.shape[1]] = crop
                inter = (A & B & V).sum(); uni = ((A | B) & V).sum()
                if uni and inter / uni >= 0.8: return lname
    return None

def looks_like_n_chars(crop, n):
    """for a label no font reads: is its size that of n characters of 10 to 16 px (or twice that)?"""
    h, w = crop.shape
    for s in list(range(10, 17)) + list(range(20, 33)):
        if abs(h - s) <= 2 and abs(w - (n * s + (n - 1))) <= n + 1: return True     # one line
        if abs(w - s) <= 2 and abs(h - (n * s + (n - 1))) <= n + 1: return True     # a column
    return False

# ---- her labels, read
by_dark = collections.defaultdict(list)
for r in regions: by_dark[dark(r["color"])].append(r["key"])
OFFS = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]

def colour_owner(c, box, radius=1000):
    cands = []
    for o in OFFS:
        for k in by_dark.get((c[0] + o[0], c[1] + o[1], c[2] + o[2]), []):
            d = region_dist(box, k)
            if d <= radius: cands.append((o != (0, 0, 0), d, k))
    return min(cands)[2] if cands else None

want_xy = [(k, R[k]["bbox"]) for k in want]
def near_wanted(box, pad=300):
    out = []
    for k, bb in want_xy:
        if bbox_dist(box, bb) <= pad or any(bbox_dist(box, c) <= pad for c in copies_of(k)): out.append(k)
    return out

def off_region(k, box, pad=40):
    """True when no pixel of region k (main map or inset copy) lies within pad px of the box"""
    X0, Y0 = max(box[0] - pad, 0), max(box[1] - pad, 0); X1, Y1 = min(box[2] + pad, W), min(box[3] + pad, H)
    if k in key2rid and (np.asarray(rid[Y0:Y1, X0:X1]) == key2rid[k]).any(): return False
    col = np.array([int(R[k]["color"][i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
    return not (ins_rgb[Y0:Y1, X0:X1] == col).all(axis=2).any()

def strip_key(crop):
    """a list entry ("a. 黄浦"): drop the short key and dot at the left, if there is one"""
    xs = np.flatnonzero(crop.any(axis=0))
    # the column runs left to right; drop runs while they are shorter than a character
    gaps = np.flatnonzero(np.diff(xs) > 1)
    starts = [xs[0]] + [xs[g + 1] for g in gaps]; ends = [xs[g] for g in gaps] + [xs[-1]]
    cut = None
    for s_, e_ in zip(starts, ends):
        ys = np.flatnonzero(crop[:, s_:e_ + 1].any(axis=1))
        if ys.max() - ys.min() + 1 < 9: cut = e_ + 1
        else: break
    if cut is None or cut > xs[-1]: return None
    rest = crop[:, cut:]
    xs2 = np.flatnonzero(rest.any(axis=0)); ys2 = np.flatnonzero(rest.any(axis=1))
    return rest[ys2.min():ys2.max() + 1, xs2.min():xs2.max() + 1]

# her text in one colour within 3 px is one label (the strokes of 八 or 川 can be 5 px apart)
lab, n_lab = ndi.label(ndi.binary_dilation(tl_mask, np.ones((7, 7), bool)), np.ones((3, 3), bool))
her = []          # her labels: dict(colour, box, crop, owner, reads, kind, key)
for i, sl in enumerate(ndi.find_objects(lab), 1):
    if sl is None: continue
    m = (lab[sl] == i) & tl_mask[sl]
    cols = tl_rgb[sl][m]
    for c, n in collections.Counter(map(tuple, cols.tolist())).items():
        if n < 6: continue
        cm = m & (tl_rgb[sl] == np.array(c, np.uint8)).all(axis=2)
        ys, xs = np.nonzero(cm)
        box = (sl[1].start + xs.min(), sl[0].start + ys.min(), sl[1].start + xs.max() + 1, sl[0].start + ys.max() + 1)
        crop = cm[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        her.append(dict(colour=c, box=box, crop=crop, px=int(n)))
print(len(her), "labels on her layer")

for h_ in her:
    c, box, crop = h_["colour"], h_["box"], h_["crop"]
    k = colour_owner(c, box)
    h_["owner"] = k
    if crop.shape[0] < 9 and crop.shape[1] < 9:
        h_["kind"], h_["reads"] = "key or mark", None; continue
    # read it: the colour's region first, then every region with a name within 300 px
    cand = ([k] if k in want else []) + [j for j in near_wanted(box) if j != k]
    h_["reads"] = next((j for j in cand if reads_as(crop, want[j]["zh"])), None)
    if not h_["reads"]:
        sk = strip_key(crop)
        if sk is not None and sk.shape[0] >= 9:
            h_["reads"] = next((j for j in cand if reads_as(sk, want[j]["zh"])), None)
            if h_["reads"]: h_["list_entry"] = True
    old_zh = V1_PLACED.get(k, {}).get("zh") if k else None
    if not h_["reads"] and crop.shape[0] < 9:
        h_["kind"] = "key or mark"; continue
    if h_["reads"] and h_["reads"] == k:
        h_["kind"] = "off its region" if off_region(k, box) and not h_.get("list_entry") else "right"
    elif h_["reads"]:
        # it reads as a nearby region's name: in a region's colour that has a name of its own, the
        # name is what is wrong (the first round put names on the wrong regions); otherwise the
        # colour is
        h_["kind"] = "another region's name" if k in want else "another region's colour"
    elif k in want and old_zh and old_zh != want[k]["zh"] and reads_as(crop, old_zh):
        h_["kind"] = "old characters"; h_["old_zh"] = old_zh
    elif k in want and looks_like_n_chars(sk if (sk := strip_key(crop)) is not None and sk.shape[0] >= 9 else crop, len(want[k]["zh"])):
        h_["kind"] = "unread, right length"      # another font or hand-drawn; the size fits
    elif k in want:
        h_["kind"] = "different characters"
    else:
        h_["kind"] = "no data" if k else "no region's colour"

# which regions she has dealt with: a label of hers that reads right (wherever it sits), or one
# in the region's colour that reads as something else (hers to judge, not ours to duplicate)
have = collections.defaultdict(list)
for h_ in her:
    if h_["kind"] in ("right", "off its region", "another region's colour"): have[h_["reads"]].append(h_)
    elif h_["kind"] in ("old characters", "different characters", "unread, right length", "another region's name"):
        have[h_["owner"]].append(h_)

# a city core whose English label is on an inset wants its characters on the inset too: a label
# of hers only on the main map doesn't count (it is listed to move)
def on_ins(box): return inset_mask[box[1]:box[3], box[0]:box[2]].mean() >= 0.5
to_inset = []
for k in list(have):
    w_ = want.get(k)
    if not w_ or not w_.get("lbox") or not copies_of(k) or not on_ins(w_["lbox"]): continue
    if not any(on_ins(h_["box"]) for h_ in have[k]):
        to_inset += [(k, h_) for h_ in have[k] if h_["kind"] in ("right", "off its region")]
        del have[k]

# ---- placement of what is missing
occupied = ndi.binary_dilation(tl_mask, np.ones((3, 3), bool))
if a.avoid_english:
    t = np.array(Image.open(os.path.join(NAMING, "text.png")).convert("RGBA"))[:, :, 3] > 0
    occupied |= ndi.binary_dilation(t, np.ones((3, 3), bool))
    del t
inset_foot = ndi.binary_dilation(inset_mask, np.ones((5, 5), bool))

def box_sum(ii, h, w):
    """sums over every h x w box with top-left in the integral image ii (padded)"""
    return ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]

def integral(m):
    ii = np.zeros((m.shape[0] + 1, m.shape[1] + 1), np.int32)
    ii[1:, 1:] = m.astype(np.int32).cumsum(0).cumsum(1)
    return ii

def l1_has(k, p):
    """is the first-round label for k still on the imported layer (layer1.png)?"""
    if l1 is None: return False
    x, y, w, h = int(p["x"]), int(p["y"]), int(p["w"]), int(p["h"])
    sub = l1[y:y + h, x:x + w]
    return ((sub[:, :, 3] > 0) & (np.abs(sub[:, :, :3].astype(int) - np.array(dark(R[k]["color"]))).max(axis=2) <= 1)).sum() >= 10

out = np.zeros((H, W, 4), np.uint8)
rows, unplaced, deleted = [], list(skipped), []
order = sorted(want, key=lambda k: R[k]["area_px"])
done = set()
for k in order:
    if have.get(k) and not a.all: done.add(k)
    elif k in V1_PLACED and V1_PLACED[k]["zh"] == want[k]["zh"] and not l1_has(k, V1_PLACED[k]) and not a.all:
        # she removed the first-round label from both layers: not put back, listed
        deleted.append(k); done.add(k)

for pass_, k in [(p_, k_) for p_ in ("keep", "search") for k_ in order]:
    if k in done: continue
    w_ = want[k]
    if k not in key2rid:
        if pass_ == "search":
            unplaced.append(dict(key=k, zh=w_["zh"], why="region not in regions.json / region_id")); done.add(k)
        continue
    r = R[k]; rid_k = key2rid[k]
    pad = SEARCH + 40
    col = np.array([int(r["color"][i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
    def window(bb, inset):
        X0, Y0 = max(bb[0] - pad, 0), max(bb[1] - pad, 0)
        X1, Y1 = min(bb[2] + pad, W - 1), min(bb[3] + pad, H - 1)
        if inset:
            reg = (ins_rgb[Y0:Y1 + 1, X0:X1 + 1] == col).all(axis=2)
        else:
            reg = np.asarray(rid[Y0:Y1 + 1, X0:X1 + 1]) == rid_k
        return X0, Y0, X1, Y1, reg
    copies = copies_of(k)
    # where to draw: on the inset whenever her English label is on one, or when the name sits
    # in a list and the inset copy is the bigger one; else on the main map
    where, target = "main", (r["bbox"], False)
    lb = w_.get("lbox")
    lb_on_inset = bool(lb) and inset_mask[lb[1]:lb[3], lb[0]:lb[2]].mean() >= 0.5
    if copies and lb_on_inset:
        where, target = "inset", (min(copies, key=lambda bb: bbox_dist(lb, bb)), True)
    elif copies and w_["anchor"] is None:
        big = max(copies, key=lambda bb: bb[4])
        if big[4] > r["area_px"]: where, target = "inset", (big, True)
    X0, Y0, X1, Y1, reg = window(*target)
    anchor = w_["anchor"]
    if where == "main" and lb and inset_mask[lb[1]:lb[3], lb[0]:lb[2]].any():
        anchor = None     # English label on an inset this region has no copy on
    if anchor is None:
        dist = ndi.distance_transform_edt(reg)
        yy, xx = np.unravel_index(dist.argmax(), dist.shape)
        ax, ay = X0 + xx, Y0 + yy; anchor_kind = "interior"
    else:
        ax, ay = anchor; anchor_kind = "english label"
    occ = occupied[Y0:Y1 + 1, X0:X1 + 1]
    if where == "main": occ = occ | inset_foot[Y0:Y1 + 1, X0:X1 + 1]
    best = None
    if pass_ == "keep":
        # a first-round spot still free is kept (it may already sit on her imported layer)
        p = V1_PLACED.get(k)
        if a.fresh or not p or p["zh"] != w_["zh"]: continue
        x, y, w, h = int(p["x"]), int(p["y"]), int(p["w"]), int(p["h"])
        if not (X0 <= x and x + w <= X1 + 1 and Y0 <= y and y + h <= Y1 + 1): continue
        lay = dict(layouts(w_["zh"])).get(p["layout"])
        if lay is None: continue
        bm = render(lay)
        if bm.shape != (h, w) or occ[max(y - 1 - Y0, 0):y - Y0 + h + 1, max(x - 1 - X0, 0):x - X0 + w + 1].any(): continue
        frac = reg[y - Y0:y - Y0 + h, x - X0:x - X0 + w].mean()
        if frac <= 0: continue
        best = (0, p["layout"], bm, y, x, frac, math.hypot(x + w / 2 - ax, y + h / 2 - ay))
        anchor_kind += ", first-round spot"
    else:
        ii_reg, ii_occ = integral(reg), integral(occ)
        for li, (lname, lines) in enumerate(layouts(w_["zh"])):
            bm = render(lines)
            h, w = bm.shape
            if h + 2 > reg.shape[0] or w + 2 > reg.shape[1]: continue
            # box grown by one pixel must be free; box itself scored by how much lies on the region
            free = box_sum(ii_occ, h + 2, w + 2) == 0          # top-left (y-1, x-1)
            onreg = box_sum(ii_reg, h, w)                       # top-left (y, x)
            fr = np.zeros_like(onreg, bool)
            fr[1:1 + free.shape[0], 1:1 + free.shape[1]] = free[:onreg.shape[0] - 1, :onreg.shape[1] - 1]
            cy = np.arange(onreg.shape[0])[:, None] + Y0 + h / 2
            cx = np.arange(onreg.shape[1])[None, :] + X0 + w / 2
            d = np.hypot(cx - ax, cy - ay)
            ok = fr & (d <= SEARCH)
            if not ok.any(): continue
            frac = onreg / float(h * w)
            score = np.where(frac >= 0.999, d, 1000 + (1 - frac) * 500 + d) + li * 3
            score = np.where(ok & (frac > 0), score, np.inf)
            i, j = np.unravel_index(np.argmin(score), score.shape)
            if np.isfinite(score[i, j]) and (best is None or score[i, j] < best[0]):
                best = (score[i, j], lname, bm, Y0 + i, X0 + j, frac[i, j], d[i, j])
        if best is None:
            unplaced.append(dict(key=k, zh=w_["zh"], why=f"no free spot within {SEARCH} px of the {anchor_kind} ({where})"))
            done.add(k); continue
    sc, lname, bm, y, x, frac, dist = best
    h, w = bm.shape
    sub = out[y:y + h, x:x + w]
    sub[bm] = dark(r["color"]) + (255,)
    occupied[max(y - GAP, 0):y + h + GAP, max(x - GAP, 0):x + w + GAP] = True
    rows.append(dict(key=k, zh=w_["zh"], x=x, y=y, w=w, h=h, layout=lname,
                     placed_inside="yes" if frac >= 0.999 else ("partly" if frac > 0 else "no"),
                     dist=round(float(dist), 1), anchor=anchor_kind, where=where, source=w_["src"],
                     on_layer1=pass_ == "keep" and l1_has(k, V1_PLACED[k])))
    done.add(k)

Image.fromarray(out, "RGBA").save(os.path.join(HERE, "local_labels_new.png"), optimize=True)
with open(os.path.join(HERE, "local_placements.csv"), "w", encoding="utf-8", newline="") as f:
    wr = csv.DictWriter(f, ["key", "zh", "x", "y", "w", "h", "layout", "placed_inside", "dist", "anchor", "where", "source", "on_layer1"])
    wr.writeheader(); wr.writerows(rows)
with open(os.path.join(HERE, "local_unplaced.csv"), "w", encoding="utf-8", newline="") as f:
    wr = csv.DictWriter(f, ["key", "zh", "why"]); wr.writeheader(); wr.writerows(unplaced)

# ---- her labels and what to change on her layer
def nm(k): return MAPNAME.get(k) or (R[k]["units"][0]["name"] if k in R and R[k]["units"] else "")
with open(os.path.join(HERE, "her_labels.csv"), "w", encoding="utf-8", newline="") as f:
    wr = csv.DictWriter(f, ["x", "y", "w", "h", "colour", "colour_region", "colour_region_name", "data_zh",
                            "reads_as_region", "reads_as", "kind"])
    wr.writeheader()
    for h_ in sorted(her, key=lambda h_: (h_["box"][1], h_["box"][0])):
        x0, y0, x1, y1 = h_["box"]; k = h_["owner"]; j = h_["reads"]
        wr.writerow(dict(x=x0, y=y0, w=x1 - x0, h=y1 - y0, colour=hexc(h_["colour"]), colour_region=k or "",
                         colour_region_name=nm(k) if k else "", data_zh=want[k]["zh"] if k in want else "",
                         reads_as_region=j or "", reads_as=(want[j]["zh"] if j else h_.get("old_zh", "")), kind=h_["kind"]))

ch = []
for h_ in her:
    x0, y0, x1, y1 = h_["box"]; k = h_["owner"]; j = h_["reads"]
    base = dict(her_x=x0, her_y=y0, her_w=x1 - x0, her_h=y1 - y0, her_colour=hexc(h_["colour"]),
                new_x="", new_y="", new_w="", new_h="", new_colour="")
    if h_["kind"] == "another region's name":
        ch.append(dict(base, change="rename", key=k, name=nm(k), zh=want[k]["zh"], new_colour=hexc(h_["colour"]),
                       note=f"her label reads {want[j]['zh']}, the name of {j} {nm(j)}; this colour is {nm(k)}"))
    elif h_["kind"] == "another region's colour":
        ch.append(dict(base, change="recolour", key=j, name=nm(j), zh=want[j]["zh"], new_colour=hexc(dark(R[j]["color"])),
                       note=f"drawn in the colour of {k} {nm(k)}" if k else "drawn in no region's colour"))
    elif h_["kind"] == "old characters":
        ch.append(dict(base, change="rename", key=k, name=nm(k), zh=want[k]["zh"], new_colour=hexc(h_["colour"]),
                       note=f"her label reads {h_['old_zh']} (the first round's characters)"))
    elif h_["kind"] == "different characters":
        ch.append(dict(base, change="check characters", key=k, name=nm(k), zh=want[k]["zh"], new_colour=hexc(h_["colour"]),
                       note="in this region's colour but not readable as the data's characters in Ark Pixel"))
    elif h_["kind"] == "off its region":
        ch.append(dict(base, change="move", key=k, name=nm(k), zh=want[k]["zh"], new_colour=hexc(h_["colour"]),
                       note="right characters and colour, more than 40 px from its region"))
for k, h_ in to_inset:
    x0, y0, x1, y1 = h_["box"]
    ch.append(dict(change="move to inset", key=k, name=nm(k), zh=want[k]["zh"], her_x=x0, her_y=y0, her_w=x1 - x0,
                   her_h=y1 - y0, her_colour=hexc(h_["colour"]), new_x="", new_y="", new_w="", new_h="", new_colour="",
                   note="her English label is on the inset; the new inset label is in local_labels_new.png"))
for p in rows:
    ch.append(dict(change="add", key=p["key"], name=nm(p["key"]), zh=p["zh"], her_x="", her_y="", her_w="", her_h="",
                   her_colour="", new_x=p["x"], new_y=p["y"], new_w=p["w"], new_h=p["h"],
                   new_colour=hexc(dark(R[p["key"]]["color"])),
                   note=f"on the {p['where']}" + ("; already on her imported Layer 1, same spot" if p["on_layer1"] else "")))
for k in deleted:
    p = V1_PLACED[k]
    ch.append(dict(change="removed by her?", key=k, name=nm(k), zh=want[k]["zh"], her_x="", her_y="", her_w="", her_h="",
                   her_colour="", new_x=p["x"], new_y=p["y"], new_w=p["w"], new_h=p["h"],
                   new_colour=hexc(dark(R[k]["color"])), note="first-round label gone from both her layers; not put back"))
ov = np.zeros((H, W, 4), np.uint8)
def box_out(x, y, w, h):
    m_ = 3
    x0, y0, x1, y1 = max(x - m_ - 2, 0), max(y - m_ - 2, 0), min(x + w + m_ + 2, W), min(y + h + m_ + 2, H)
    ov[y0:y1, x0:x0 + 2] = ov[y0:y1, x1 - 2:x1] = (255, 0, 0, 255)
    ov[y0:y0 + 2, x0:x1] = ov[y1 - 2:y1, x0:x1] = (255, 0, 0, 255)
for c in ch:
    if c["her_x"] != "": box_out(int(c["her_x"]), int(c["her_y"]), int(c["her_w"]), int(c["her_h"]))
    if c["new_x"] != "" and c["change"] == "add": box_out(int(c["new_x"]), int(c["new_y"]), int(c["new_w"]), int(c["new_h"]))
Image.fromarray(ov, "RGBA").save(os.path.join(HERE, "changes_vs_current.png"), optimize=True)
ch.sort(key=lambda c: (c["change"], int(c["her_y"] if c["her_y"] != "" else c["new_y"])))
with open(os.path.join(HERE, "changes_vs_current.csv"), "w", encoding="utf-8", newline="") as f:
    wr = csv.DictWriter(f, ["change", "key", "name", "zh", "her_x", "her_y", "her_w", "her_h", "her_colour",
                            "new_x", "new_y", "new_w", "new_h", "new_colour", "note"])
    wr.writeheader(); wr.writerows(ch)

print("her labels:", dict(collections.Counter(h_["kind"] for h_ in her)))
print("regions with zh:", len(want), "| dealt with on her layer:", sum(1 for k in want if have.get(k)),
      "| removed by her (not put back):", len(deleted))
cnt = collections.Counter(r["placed_inside"] for r in rows)
print(f"to add: {len(rows)} placed ({dict(cnt)}; {dict(collections.Counter(r['where'] for r in rows))};"
      f" {sum('first-round' in r['anchor'] for r in rows)} on first-round spots); {len(unplaced)} not placed")
print("changes:", dict(collections.Counter(c["change"] for c in ch)))
if fallback_used: print("drawn with SimSun (not in Ark Pixel):", "".join(sorted(fallback_used)))
