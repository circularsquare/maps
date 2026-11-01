"""Data check of the China labels on textlocal. Reuses place_local.py's label cutting, colour
owner and reads_as matcher, with more fonts. Writes data_labels.csv and data_issues.csv here."""
import os, csv, json, math, collections, functools
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy.ndimage as ndi
Image.MAX_IMAGE_PIXELS = None
S = os.path.dirname(os.path.abspath(__file__))
NAMING = r"C:\Users\anita\projects\maps\asia1m\naming"
RN = r"C:\Users\anita\projects\maps\asia1m\regionnaming"
UF = os.path.join(os.environ["LOCALAPPDATA"], "Microsoft", "Windows", "Fonts")

regions = json.load(open(os.path.join(NAMING, "regions.json"), encoding="utf-8"))
R = {r["key"]: r for r in regions}
by_color = collections.defaultdict(list)
for r in regions: by_color[r["color"]].append(r)
prelim = list(csv.DictReader(open(os.path.join(RN, "regions_prelim.csv"))))
key2rid = {}
for p in prelim:
    cands = by_color.get(p["color"], [])
    if not cands: continue
    lon, lat = float(p["lon"]), float(p["lat"])
    r = min(cands, key=lambda r: (r["lon"] - lon) ** 2 + (r["lat"] - lat) ** 2)
    key2rid[r["key"]] = int(p["rid"])

def dark(h): return tuple(min(191, int(math.ceil(int(h[i:i + 2], 16) * 0.75))) for i in (1, 3, 5))
def hexc(c): return "#%02x%02x%02x" % tuple(int(v) for v in c)

names = {r["key"]: r for r in csv.DictReader(open(os.path.join(NAMING, "all_names.csv"), encoding="utf-8-sig"))}
nonhan = {r["key"]: r for r in csv.DictReader(open(os.path.join(NAMING, "nonhan_regions.csv"), encoding="utf-8-sig"))}
CN = [r["key"] for r in regions if r["country"] == "China"]
CNs = set(CN)
def zh_of(k): return (names.get(k) or {}).get("zh") or ""
def ename(k):
    n = (names.get(k) or {}).get("name")
    if n: return n
    return R[k]["units"][0]["name"] if R[k].get("units") else ""
# names to read against: every region with a zh (any country), for "reads as a neighbour"
want = {k: zh_of(k) for k in R if zh_of(k)}

rid = np.load(os.path.join(RN, "region_id.npy"), mmap_mode="r")
H, W = rid.shape
ins = np.array(Image.open(os.path.join(S, "insets.png")).convert("RGBA"))
inset_mask = ins[:, :, 3] > 0
ins_rgb = ins[:, :, :3].copy(); ins_rgb[~inset_mask] = 0
del ins
tl = np.array(Image.open(os.path.join(S, "textlocal.png")).convert("RGBA"))
tl_mask = tl[:, :, 3] > 0
tl_rgb = tl[:, :, :3].copy()
del tl

# inset copies (as place_local)
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
    out = []
    for bb in inset_copies.get(R[k]["color"], []):
        near = min(by_color[R[k]["color"]], key=lambda r: bbox_dist(bb, r["bbox"]))
        if near["key"] == k and bbox_dist(bb, R[k]["bbox"]) <= 1500: out.append(tuple(bb))
    return tuple(out)

def region_dist(box, k):
    d = bbox_dist(box, R[k]["bbox"])
    for bb in copies_of(k): d = min(d, bbox_dist(box, bb))
    return d

# ---- fonts: each rendered at a fixed origin, cell = union ink box of sample characters
SAMPLE = "国一丨中青藏疆湖州县山江河"
FONTS = [("ark zh_cn", os.path.join(UF, "ark-pixel-12px-monospaced-zh_cn.ttf"), 12),
         ("asia1mpixel", r"C:\Users\anita\projects\maps\asia1m\fonts\asia1mpixel\asia1mpixel-Regular.ttf", 12),
         ("fusion zh_hans", r"C:\Users\anita\projects\maps\asia1m\fonts\asia1mpixel\sources\fusion-pixel-12px-monospaced-zh_hans.ttf", 12),
         ("ark zh_hk", os.path.join(UF, "ark-pixel-12px-monospaced-zh_hk.ttf"), 12),
         ("ark ja", os.path.join(UF, "ark-pixel-12px-monospaced-ja.ttf"), 12),
         ("simsun", r"C:\Windows\Fonts\simsun.ttc", 12),
         ("unifont", os.path.join(UF, "unifont-17.0.04.otf"), 16)]
F = []
for nm_, pth, size in FONTS:
    if not os.path.exists(pth): print("missing font", pth); continue
    f = ImageFont.truetype(pth, size)
    def raw(ch, f=f, size=size):
        im = Image.new("L", (size * 3, size * 3), 0)
        d = ImageDraw.Draw(im); d.fontmode = "1"
        d.text((size, size), ch, fill=255, font=f)
        return np.array(im) > 127
    u = np.zeros((size * 3, size * 3), bool)
    for ch in SAMPLE: u |= raw(ch)
    ys, xs = np.nonzero(u)
    cell = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    notdef = raw("\U000F0001").tobytes()
    F.append((nm_, raw, cell, notdef, size))
    print(nm_, "cell", cell[1] - cell[0], "x", cell[3] - cell[2])

@functools.lru_cache(maxsize=None)
def gl(fi, ch):
    nm_, raw, cell, notdef, size = F[fi]
    g = raw(ch)
    if g.tobytes() == notdef or not g.any(): return None
    return g[cell[0]:cell[1], cell[2]:cell[3]]

def layouts(zh):
    out = [("line", [zh])]
    if len(zh) >= 2: out.append(("column", list(zh)))
    if len(zh) >= 4:
        k = math.ceil(len(zh) / 2)
        out.append(("two lines", [zh[:k], zh[k:]]))
    if len(zh) >= 3:
        for k in range(1, len(zh)):
            if k != math.ceil(len(zh) / 2): out.append(("two lines %d" % k, [zh[:k], zh[k:]]))
    return out

@functools.lru_cache(maxsize=None)
def variants(zh):
    out = []
    for fi in range(len(F)):
        if any(gl(fi, ch) is None for ch in zh): continue
        ch_h, ch_w = gl(fi, zh[0]).shape
        for pitch_extra in (1, 0, 2):
            for lname, lines in layouts(zh):
                for align in ("centre", "left"):
                    if align == "left" and lname == "line": continue
                    n = max(len(l) for l in lines)
                    ph, pw = ch_h + pitch_extra, ch_w + pitch_extra
                    h, w = len(lines) * ph - pitch_extra, n * pw - pitch_extra
                    bm = np.zeros((h, w), bool)
                    for i, l in enumerate(lines):
                        off = (n - len(l)) * pw // 2 if align == "centre" else 0
                        for j, ch in enumerate(l):
                            bm[i * ph:i * ph + ch_h, off + j * pw:off + j * pw + ch_w] |= gl(fi, ch)
                    ys, xs = np.nonzero(bm)
                    if not len(ys): continue
                    bm = bm[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                    out.append((F[fi][0] + " " + lname, bm))
                    if pitch_extra == 1 and fi < 2:
                        out.append((F[fi][0] + " " + lname + " 2x", np.kron(bm, np.ones((2, 2), bool))))
                        out.append((F[fi][0] + " " + lname + " 3x", np.kron(bm, np.ones((3, 3), bool))))
    return out

SUFFIX = ["特别行政区", "自治县", "自治旗", "新区", "街道", "地区", "林区", "市", "区", "县", "旗", "镇", "乡", "省", "盟", "州"]
def strip_suffix(s):
    if not s: return ""
    for suf in SUFFIX:
        if s.endswith(suf) and len(s) - len(suf) >= 2: return s[:-len(suf)]
    return s
@functools.lru_cache(maxsize=None)
def altnames(k):
    """other names of region k's own area: units, prefectures, parents, towns"""
    r = R[k]; out = []
    for u in r.get("units") or []:
        out += [u.get("local"), u.get("prefecture_local")]
    for p in r.get("parents") or []: out.append(p.get("local"))
    for t in (r.get("towns") or [])[:30]: out.append(t.get("town"))
    s = []
    for n in out:
        n2 = strip_suffix(n)
        if n2 and n2 != zh_of(k) and n2 not in s and len(n2) <= 6: s.append(n2)
    return tuple(s)

def score(crop, zh):
    """best IoU of the label's ink with zh in any font / layout, 1 px of play"""
    best, bl = 0.0, None
    for lname, bm in variants(zh):
        if abs(bm.shape[0] - crop.shape[0]) > 2 or abs(bm.shape[1] - crop.shape[1]) > 2: continue
        hh = max(bm.shape[0], crop.shape[0]) + 4; ww = max(bm.shape[1], crop.shape[1]) + 4
        B = np.zeros((hh, ww), bool); B[2:2 + bm.shape[0], 2:2 + bm.shape[1]] = bm
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                A = np.zeros((hh, ww), bool)
                A[2 + dy:2 + dy + crop.shape[0], 2 + dx:2 + dx + crop.shape[1]] = crop
                inter = (A & B).sum(); uni = (A | B).sum()
                if uni and inter / uni > best: best, bl = inter / uni, lname
    return best, bl

def strip_key(crop):
    xs = np.flatnonzero(crop.any(axis=0))
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

by_dark = collections.defaultdict(list)
for r in regions: by_dark[dark(r["color"])].append(r["key"])
OFFS = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
def colour_owner(c, box, radius=1000):
    cands = []
    for o in OFFS:
        for k in by_dark.get((c[0] + o[0], c[1] + o[1], c[2] + o[2]), []):
            d = region_dist(box, k)
            if d <= radius: cands.append((o != (0, 0, 0), d, k))
    if not cands: return None, None
    m = min(cands)
    return m[2], not m[0]

# China extent: China regions plus their inset copies
ext = [W, H, 0, 0]
for k in CN:
    for bb in [R[k]["bbox"]] + [c[:4] for c in copies_of(k)]:
        ext = [min(ext[0], bb[0]), min(ext[1], bb[1]), max(ext[2], bb[2]), max(ext[3], bb[3])]
print("China extent", ext)
json.dump(dict(ext=ext), open(os.path.join(S, "extent.json"), "w"))

# ---- her labels, cut as place_local does
lab, n_lab = ndi.label(ndi.binary_dilation(tl_mask, np.ones((7, 7), bool)), np.ones((3, 3), bool))
her = []
for i, sl in enumerate(ndi.find_objects(lab), 1):
    if sl is None: continue
    m = (lab[sl] == i) & tl_mask[sl]
    cols = tl_rgb[sl][m]
    for c, n in collections.Counter(map(tuple, cols.tolist())).items():
        cm = m & (tl_rgb[sl] == np.array(c, np.uint8)).all(axis=2)
        ys, xs = np.nonzero(cm)
        box = (sl[1].start + xs.min(), sl[0].start + ys.min(), sl[1].start + xs.max() + 1, sl[0].start + ys.max() + 1)
        if box[2] < ext[0] - 50 or box[0] > ext[2] + 50 or box[3] < ext[1] - 50 or box[1] > ext[3] + 50: continue
        crop = cm[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        her.append(dict(colour=c, box=box, crop=crop, px=int(n), comp=i))
print(len(her), "colour pieces in the China extent")
exec(open(os.path.join(S, "tail.py"), encoding="utf-8").read())
raise SystemExit

want_xy = [(k, R[k]["bbox"]) for k in want]
def near_wanted(box, pad=300):
    return [k for k, bb in want_xy if bbox_dist(box, bb) <= pad or any(bbox_dist(box, c) <= pad for c in copies_of(k))]

def on_ins(box): return inset_mask[box[1]:box[3], box[0]:box[2]].mean() >= 0.5

def region_under(box):
    """region (main map) or inset colour under most of the box"""
    x0, y0, x1, y1 = box
    if on_ins(box):
        sub = ins_rgb[y0:y1, x0:x1].reshape(-1, 3); msk = inset_mask[y0:y1, x0:x1].reshape(-1)
        cs = collections.Counter(map(tuple, sub[msk].tolist()))
        for c, _ in cs.most_common():
            h_ = hexc(c)
            if h_ in by_color: return min(by_color[h_], key=lambda r: bbox_dist(box, r["bbox"]))["key"], "inset"
        return None, "inset"
    sub = np.asarray(rid[y0:y1, x0:x1]).reshape(-1)
    cs = collections.Counter(sub.tolist())
    rid2key = {v: k for k, v in key2rid.items()}
    for v, _ in cs.most_common():
        if v in rid2key: return rid2key[v], "main"
    return None, "main"

def off_region(k, box, pad=40):
    X0, Y0 = max(box[0] - pad, 0), max(box[1] - pad, 0); X1, Y1 = min(box[2] + pad, W), min(box[3] + pad, H)
    if k in key2rid and (np.asarray(rid[Y0:Y1, X0:X1]) == key2rid[k]).any(): return False
    col = np.array([int(R[k]["color"][i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
    return not (ins_rgb[Y0:Y1, X0:X1] == col).all(axis=2).any()

for h_ in her:
    c, box, crop = h_["colour"], h_["box"], h_["crop"]
    k, exact = colour_owner(c, box)
    h_["owner"], h_["exact"] = k, exact
    h_["small"] = crop.shape[0] < 9 and crop.shape[1] < 9
    h_["under"], h_["where"] = region_under(box)
    if h_["small"]: h_["reads"] = None; continue
    cand = ([k] if k in want else []) + [j for j in near_wanted(box) if j != k]
    best = (0.0, None, None)
    for j in cand:
        sc, ln = score(crop, want[j])
        sk = strip_key(crop)
        if sk is not None and sk.shape[0] >= 9:
            sc2, ln2 = score(sk, want[j])
            if sc2 > sc: sc, ln = sc2, ln2 + " (list key)"
        if sc > best[0]: best = (sc, j, ln)
        if j == k and sc >= 0.8: break
    h_["best"] = best
    own = score(crop, want[k]) if k in want else (0.0, None)
    h_["own_score"] = own[0]
    h_["reads"] = best[1] if best[0] >= 0.8 else None

# group pieces: same owner, boxes within 13 px of each other -> one label (characters spaced apart)
def merge_groups():
    groups = []
    by_owner = collections.defaultdict(list)
    for idx, h_ in enumerate(her):
        if not h_["small"]: by_owner[(h_["owner"], h_["colour"])].append(idx)
    for (k, c), idxs in by_owner.items():
        parent = {i: i for i in idxs}
        def f(i):
            while parent[i] != i: i = parent[i]
            return i
        for a in idxs:
            for b in idxs:
                if a < b and bbox_dist(her[a]["box"], her[b]["box"]) <= 13: parent[f(a)] = f(b)
        cl = collections.defaultdict(list)
        for i in idxs: cl[f(i)].append(i)
        groups += list(cl.values())
    return groups
groups = merge_groups()
labels = []
for g in groups:
    hs = [her[i] for i in g]
    box = (min(h["box"][0] for h in hs), min(h["box"][1] for h in hs), max(h["box"][2] for h in hs), max(h["box"][3] for h in hs))
    k = hs[0]["owner"]
    if len(hs) == 1:
        h = hs[0]; reads, best, own = h["reads"], h["best"], h["own_score"]
    else:
        c = np.array(hs[0]["colour"], np.uint8)
        crop = (tl_rgb[box[1]:box[3], box[0]:box[2]] == c).all(axis=2) & tl_mask[box[1]:box[3], box[0]:box[2]]
        own = score(crop, want[k])[0] if k in want else 0.0
        cand = ([k] if k in want else []) + [j for j in near_wanted(box) if j != k]
        best = (0.0, None, None)
        for j in cand:
            sc, ln = score(crop, want[j])
            if sc > best[0]: best = (sc, j, ln)
        reads = best[1] if best[0] >= 0.8 else None
    labels.append(dict(box=box, colour=hs[0]["colour"], owner=k, exact=hs[0]["exact"], reads=reads,
                       best=best, own=own, pieces=len(hs), under=hs[0]["under"], where="inset" if on_ins(box) else "main",
                       px=sum(h["px"] for h in hs)))
small = [h for h in her if h["small"]]

with open(os.path.join(S, "data_labels.csv"), "w", encoding="utf-8", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["x", "y", "w", "h", "colour", "owner", "owner_name", "owner_zh", "exact_colour", "where", "under",
                 "reads", "reads_zh", "best_score", "best_region", "best_layout", "own_score", "pieces", "px"])
    for L in sorted(labels, key=lambda L: (L["box"][1], L["box"][0])):
        b = L["box"]; k = L["owner"]
        wr.writerow([b[0], b[1], b[2] - b[0], b[3] - b[1], hexc(L["colour"]), k or "", ename(k) if k else "",
                     zh_of(k) if k else "", L["exact"], L["where"], L["under"] or "", L["reads"] or "",
                     want.get(L["reads"], "") if L["reads"] else "", round(L["best"][0], 3), L["best"][1] or "",
                     L["best"][2] or "", round(L["own"], 3), L["pieces"], L["px"]])

# ---- issues per China region
issues = []
def add(kind, key, saw, box, conf, note=""):
    issues.append(dict(kind=kind, key=key or "", name=ename(key) if key else "", zh=(zh_of(key) if key else ""),
                       native=(nonhan.get(key, {}).get("native", "") if key else ""), saw=saw,
                       x=int(box[0]), y=int(box[1]), w=int(box[2] - box[0]), h=int(box[3] - box[1]), found="data", confidence=conf, note=note))

own_labels = collections.defaultdict(list)
reads_labels = collections.defaultdict(list)
for L in labels:
    if L["owner"]: own_labels[L["owner"]].append(L)
    if L["reads"]: reads_labels[L["reads"]].append(L)

def bxs(L): b = L["box"]; return "(%d,%d %dx%d %s)" % (b[0], b[1], b[2] - b[0], b[3] - b[1], L["where"])

for k in CN:
    mine = own_labels.get(k, [])
    is_nh = k in nonhan
    if not mine:
        other = [L for L in reads_labels.get(k, [])]
        bb = R[k]["bbox"]
        if other:
            o = other[0]
            add("wrong_colour", k, "a label reading %s %s is drawn in the colour of %s %s" % (zh_of(k), bxs(o), o["owner"] or "no region", ename(o["owner"]) if o["owner"] else ""), o["box"], "high")
        else:
            cps = copies_of(k)
            add("missing", k, "no label in this region's colour (main map%s)" % (" or its inset" if cps else ""), bb,
                "high" if zh_of(k) or is_nh else "medium",
                "no zh in all_names.csv" if not zh_of(k) else "")
        continue
    if len(mine) >= 2:
        wh = sorted(L["where"] for L in mine)
        pair = len(mine) == 2 and wh == ["inset", "main"]
        add("duplicate", k, "%d labels in this colour: %s" % (len(mine), "; ".join(bxs(L) + (" reads " + want[L["reads"]] if L["reads"] else " unread") for L in mine)),
            (min(L["box"][0] for L in mine), min(L["box"][1] for L in mine), max(L["box"][2] for L in mine), max(L["box"][3] for L in mine)),
            "medium" if pair else "high", "main map + inset pair: hers to decide" if pair else "")
    for L in mine:
        if is_nh and nonhan[k].get("native"):
            continue      # a native-script label: accepted
        z = zh_of(k)
        if L["reads"] == k:
            if off_region(k, L["box"]):
                add("other", k, "right characters, more than 40 px from its region %s" % bxs(L), L["box"], "medium")
            continue
        if L["reads"] and L["reads"] != k:
            j = L["reads"]
            add("wrong_colour", k, "label %s reads %s (%s %s) but is in this region's colour" % (bxs(L), want[j], j, ename(j)), L["box"], "high" if j in own_labels and False else "medium",
                "or wrong characters; %s has %d label(s) of its own colour" % (j, len(own_labels.get(j, []))))
            continue
        if not z:
            continue
        add("wrong_characters", k, "label %s does not read as %s (best %.2f %s)" % (bxs(L), z, L["own"], L["best"][2] or ""), L["box"],
            "medium" if L["own"] < 0.6 else "low")

# labels in no region's colour inside the China extent
for L in labels:
    if L["owner"] is None:
        u = L["under"]
        if u and R[u]["country"] != "China": continue
        add("wrong_colour", L["reads"] or u, "label %s in colour %s that matches no region%s" % (bxs(L), hexc(L["colour"]), (", reads " + want[L["reads"]]) if L["reads"] else ""), L["box"], "high" if L["reads"] else "medium")
    elif not L["exact"] and L["owner"] in CNs:
        add("other", L["owner"], "label %s colour %s is 1 off the region's %s" % (bxs(L), hexc(L["colour"]), hexc(dark(R[L["owner"]]["color"]))), L["box"], "low")
# small pieces in a China region colour that sit apart from any label of theirs
for h in small:
    k = h["owner"]
    if k in CNs and not any(bbox_dist(h["box"], L["box"]) <= 3 for L in own_labels.get(k, [])):
        pass  # keys and marks are counted, not flagged

with open(os.path.join(S, "data_issues.csv"), "w", encoding="utf-8", newline="") as f:
    wr = csv.DictWriter(f, ["kind", "key", "name", "zh", "native", "saw", "x", "y", "w", "h", "found", "confidence", "note"])
    wr.writeheader(); wr.writerows(issues)
print("labels", len(labels), "small pieces", len(small))
print(collections.Counter(i["kind"] for i in issues))
print("China regions with own label:", sum(1 for k in CN if own_labels.get(k)), "of", len(CN))
