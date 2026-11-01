"""Labels -> regions -> names (and Chinese characters for China) -> oldnames.csv, review.csv.

    python asia1m/naming/oldnames/assign.py

Reads labels.json from read.py. See README.md for the rules.
"""
import os, sys, json, csv, re, collections, unicodedata, difflib, itertools, math
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
HERE = os.path.dirname(os.path.abspath(__file__))
NAMING = os.path.join(HERE, "..")
RN = os.path.join(NAMING, "..", "regionnaming")
sys.path.insert(0, RN)
Image.MAX_IMAGE_PIXELS = None
from pypinyin import pinyin, Style

# ---------------------------------------------------------------- text helpers
def norm(s):
    """compare form: lowercase latin letters only, accents and ü folded."""
    s = unicodedata.normalize("NFD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s)

def norm_py(s):
    return norm(s).replace("v", "u")

SUFFIXES = sorted("""自治县 自治旗 自治州 特区 林区 新区 地区 街道 苏木 民族乡 民族镇 回族乡 回族镇 满族乡 满族镇
蒙古族乡 彝族乡 苗族乡 土家族乡 瑶族乡 壮族乡 藏族乡 镇 乡 县 市 区 旗 盟 州 府 郡 路 军 监 厅 卫 所 道 司 场 农场 林场 开发区 管委会 办事处""".split(),
                  key=len, reverse=True)
ETHNIC = re.compile(r"(各族|.{1,6}?族)+(自治县|自治旗|自治州|乡|镇)$")

PEOPLES = sorted("""蒙古 回 藏 维吾尔 苗 彝 壮 布依 朝鲜 满 侗 瑶 白 土家 哈尼 哈萨克 傣 黎 傈僳 佤 畲 高山 拉祜 水 东乡
纳西 景颇 柯尔克孜 土 达斡尔 仫佬 羌 布朗 撒拉 毛南 仡佬 锡伯 阿昌 普米 塔吉克 怒 乌孜别克 俄罗斯 鄂温克 德昂 保安 裕固 京
塔塔尔 独龙 鄂伦春 赫哲 门巴 珞巴 基诺 各""".split(), key=len, reverse=True)


def zh_core(z):
    """(core, suffix): 元江哈尼族彝族傣族自治县 -> (元江, 自治县)"""
    z0 = (z or "").strip()
    suf = ""
    for s_ in SUFFIXES:
        if z0.endswith(s_) and len(z0) - len(s_) >= 1:
            z0, suf = z0[:-len(s_)], s_; break
    while z0.endswith("族"):
        for p in PEOPLES:
            if z0.endswith(p + "族") and len(z0) - len(p) - 1 >= 2:
                z0 = z0[:-len(p) - 1]; break
        else:
            break
    return z0, suf

def py_variants(z, cap=24):
    """pinyin spellings of a Chinese string, all heteronym readings (capped)."""
    if not z: return []
    ps = pinyin(z, style=Style.NORMAL, heteronym=True, errors="ignore")
    if not ps: return []
    out = set()
    for combo in itertools.islice(itertools.product(*[p[:3] for p in ps]), cap):
        out.add(norm_py("".join(combo)))
    return [o for o in out if o]

def forms(z):
    """compare forms for a place name in characters: core (no suffix), and core+suffix when
    the core is one character or the suffix is part of how it is said (xian, zhou ...)."""
    core, suf = zh_core(z)
    out = {}
    for v in py_variants(core): out[v] = core
    if suf and (len(core) == 1 or suf in ("州", "县", "江", "山", "河", "城", "关", "口")):
        for v in py_variants(core + suf): out.setdefault(v, core + suf)
    for v in py_variants(z): out.setdefault(v, z)
    return out

FEATURE_ZH = sorted("山脉 山区 山地 山 岭 峰 江 河 川 水 湖 泊 淀 水库 海 湾 岛 群岛 列岛 半岛 盆地 平原 高原 沙漠 草原 峡 溪".split(),
                    key=len, reverse=True)
FEATURE_PY = {"shanmai", "shan", "ling", "feng", "jiang", "he", "hu", "dian", "shuiku", "hai", "wan", "dao",
              "qundao", "liedao", "bandao", "pendi", "pingyuan", "gaoyuan", "shamo", "caoyuan", "xia", "xi",
              "chuan", "shui", "po", "shandi", "shanqu"}


def simplified_first(zhs):
    for z in zhs:
        try:
            z.encode("gb2312"); return z
        except UnicodeEncodeError:
            pass
    return zhs[0]


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Features:
    """GeoNames rivers, lakes, mountains, areas (features.csv), searched near a label"""
    def __init__(self, path=os.path.join(HERE, "features.csv")):
        self.grid = collections.defaultdict(list)
        if not os.path.exists(path): return
        for r in csv.DictReader(open(path, encoding="utf-8")):
            zh = simplified_first(r["zh"].split("|"))
            fs = {}
            core = zh
            for suf in FEATURE_ZH:
                if zh.endswith(suf) and len(zh) > len(suf):
                    core = zh[:-len(suf)]; break
            for v in py_variants(core): fs.setdefault(v, core)
            for v in py_variants(zh): fs.setdefault(v, zh)
            words = r["ascii"].lower().split()
            if len(words) > 1 and words[-1] in FEATURE_PY:
                fs.setdefault(norm_py("".join(words[:-1])), core)
            fs.setdefault(norm_py(r["ascii"]), zh)
            x, y = int(r["x"]), int(r["y"])
            e = dict(src="geonames", zh=zh, en=r["name"], code=r["code"], x=x, y=y)
            self.grid[(x // 200, y // 200)].append((e, fs))

    def exact(self, n, cx, cy):
        """features named n near (cx, cy): rivers within 600 px (the point is one place on a
        long river), mountain ranges within 400, the rest within 250"""
        out = []
        gx, gy = int(cx // 200), int(cy // 200)
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                for e, fs in self.grid.get((gx + dx, gy + dy), []):
                    radius = 600 if e["code"].startswith("STM") else 400 if e["code"] in ("MTS", "HLLS", "RDGE", "PLAT", "UPLD") else 250
                    if n in fs and abs(e["x"] - cx) <= radius and abs(e["y"] - cy) <= radius:
                        out.append(((e["x"] - cx) ** 2 + (e["y"] - cy) ** 2, e, fs[n]))
        out.sort(key=lambda t: t[0])
        return [(e, z) for _, e, z in out]


# ---------------------------------------------------------------- inputs
def load():
    regions = json.load(open(os.path.join(NAMING, "regions.json"), encoding="utf-8"))
    prelim = list(csv.DictReader(open(os.path.join(RN, "regions_prelim.csv"))))
    by_color = collections.defaultdict(list)
    for r in regions: by_color[r["color"]].append(r)
    rid2key = {}
    for p in prelim:
        cands = by_color.get(p["color"], [])
        if not cands: continue
        lon, lat = float(p["lon"]), float(p["lat"])
        r = min(cands, key=lambda r: (r["lon"] - lon) ** 2 + (r["lat"] - lat) ** 2)
        rid2key[int(p["rid"])] = r["key"]
    picks = {}
    for p in csv.DictReader(open(os.path.join(NAMING, "picks.csv"), encoding="utf-8")):
        picks[p["key"]] = p["name"]
    return regions, rid2key, picks


def region_admin(rid, rid2key, path, names_path, ids_wanted=None):
    """for each region key: {admin id: pixels}, from an admin raster."""
    adm = np.load(path, mmap_mode="r")
    names = json.load(open(names_path, encoding="utf-8"))
    out = collections.defaultdict(dict)
    H = rid.shape[0]
    step = 1000
    acc = collections.Counter()
    for y in range(0, H, step):
        a = np.asarray(rid[y:y + step]).astype(np.int64).ravel()
        b = np.asarray(adm[y:y + step]).astype(np.int64).ravel()
        ok = (a > 0) & (b > 0)
        k, c = np.unique(a[ok] * 1000000 + b[ok], return_counts=True)
        for kk, cc in zip(k.tolist(), c.tolist()): acc[kk] += cc
    for kk, cc in acc.items():
        r, b = divmod(kk, 1000000)
        key = rid2key.get(r)
        if key: out[key][b] = cc
    return out, names


def chgis_by_region(rid, rid2key):
    import pyogrio
    from geo import lonlat_to_px
    CH = os.path.join(NAMING, "..", "..", "data", "asia1m", "chgis")
    # border pixels (rid 0 between regions) go to the nearest region, as sources.py does
    out = collections.defaultdict(list)
    for z, shp, level in [("v6_time_pref_pts_utf_wgs84.zip", "v6_time_pref_pts_utf_wgs84.shp", "pref"),
                          ("v6_time_cnty_pts_utf_wgs84.zip", "v6_time_cnty_pts_utf_wgs84.shp", "cnty")]:
        df = pyogrio.read_dataframe(f"zip://{CH}/{z}!{shp}", read_geometry=False)
        x, y = lonlat_to_px(df["X_COOR"].to_numpy(float), df["Y_COOR"].to_numpy(float))
        x = np.round(x).astype(int); y = np.round(y).astype(int)
        H, W = rid.shape
        for i in range(len(df)):
            if not (0 <= x[i] < W and 0 <= y[i] < H): continue
            r = int(rid[y[i], x[i]])
            if r == 0:
                win = np.asarray(rid[max(y[i] - 3, 0):y[i] + 4, max(x[i] - 3, 0):x[i] + 4])
                v = win[win > 0]
                if len(v) == 0: continue
                r = int(np.bincount(v).argmax())
            key = rid2key.get(r)
            if not key: continue
            row = df.iloc[i]
            out[key].append(dict(zh=row["NAME_CH"], py=row["NAME_PY"], type=row["TYPE_CH"],
                                 beg=int(row["BEG_YR"]), end=int(row["END_YR"]), level=level))
    return out


def gazetteer(regions, rid2key):
    """per region: candidate names (units, prefectures, towns, CHGIS seats), cached."""
    path = os.path.join(HERE, "gazetteer.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    rid = np.load(os.path.join(RN, "region_id.npy"), mmap_mode="r")
    cnty, cnames = region_admin(rid, rid2key, os.path.join(RN, "adm_all.npy"), os.path.join(RN, "adm_all_names.json"))
    print("counties done", flush=True)
    twn, tnames = region_admin(rid, rid2key, os.path.join(RN, "adm_twn.npy"), os.path.join(RN, "adm_twn_names.json"))
    print("townships done", flush=True)
    ch = chgis_by_region(rid, rid2key)
    print("chgis done", flush=True)
    G = {}
    for r in regions:
        k = r["key"]; tot = max(r["area_px"], 1); e = []
        for b, c in sorted(cnty.get(k, {}).items(), key=lambda t: -t[1]):
            n = cnames[b]
            if n and c / tot >= 0.002:
                e.append(dict(src="unit", zh=n.get("local"), en=n.get("name"), share=round(c / tot, 4)))
        for u in r["units"]:
            e.append(dict(src="unit", zh=u.get("local"), en=u.get("name"), share=u.get("people_share", 0)))
        for p in r["parents"]:
            if p["level"] in ("prefecture", "province"):
                e.append(dict(src="prefecture" if p["level"] == "prefecture" else "province",
                              zh=p.get("local"), en=p.get("name"), share=p.get("people_share", 0)))
        for c in r.get("cities", []):
            e.append(dict(src="city", zh=None, en=c.get("name"), share=0))
        for b, c in sorted(twn.get(k, {}).items(), key=lambda t: -t[1]):
            n = tnames[b]
            if n and c / tot >= 0.002:
                e.append(dict(src="town", zh=n["town"], en=None, share=round(c / tot, 4), county=n["county"]))
        seen = set()
        for c in ch.get(k, []):
            if c["zh"] in seen: continue
            seen.add(c["zh"])
            e.append(dict(src="chgis", zh=c["zh"], en=c["py"], share=0, years=f"{c['beg']}-{c['end']}", type=c["type"]))
        G[k] = e
    json.dump(G, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return G


TITLE_WORDS = {"divided", "regions", "people", "projection", "million", "asia", "albers", "equal", "area", "by"}
ENTRY = re.compile(r"^(\d{1,2}|[a-z])\.\s*(.+)$")
MARKER = re.compile(r"^(\d{1,2}|[a-z])$")
TWO_ENTRIES = re.compile(r"^(\d{1,2}|[a-z])\.\s*(.+?)\s+(\d{1,2}\.\s*\S.*)$")
JOINED = {"China", "Taiwan", "Hong Kong", "South Korea", "North Korea", "Japan"}
TIER_OWN = ["unit", "prefecture", "town", "chgis", "geonames", "city", "province"]


def clean_marks(s, country):
    """keep the accents a country's romanisation uses; the rest are other labels' pixels"""
    keep = {"Vietnam": None, "South Korea": {"̆"}, "North Korea": {"̆"},
            "Japan": {"̄"}}.get(country, {"̈"})
    s = s.replace("¿", "?" if country == "Vietnam" else "")
    if keep is None: return s
    d = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in d if not unicodedata.combining(c) or c in keep))


def titlecase(s):
    out = []
    for w in s.split(" "):
        out.append(w[:1].upper() + w[1:] if w else w)
    return " ".join(out)


class Index:
    """compare-form index of one region's gazetteer entries"""
    def __init__(self, entries, china):
        self.by = collections.defaultdict(list)
        for e in entries:
            fs = {}
            zh = (e.get("zh") or "").split("|")[0].strip()
            if china and zh:
                fs.update(forms(zh))
                if e["src"] == "chgis" and e.get("en"):
                    py = e["en"]
                    words = py.split()
                    if len(words) > 1: fs.setdefault(norm_py("".join(words[:-1])), zh_core(zh)[0])
                    fs.setdefault(norm_py(py), zh)
            if e.get("en"):
                zc = zh_core(zh)[0] if zh else ""
                n = norm_py(e["en"])
                if n: fs.setdefault(n, zc)
                # 'Sylhet Sadar' -> 'sylhet'; 'X District' -> 'x'
                w = e["en"].split()
                if len(w) > 1: fs.setdefault(norm_py(w[0]), zc)
            for f, z in fs.items():
                self.by[f].append((e, z))

    def exact(self, n):
        return self.by.get(n, [])

    def fuzzy(self, n, cut=0.85):
        """closest name one edit away (two for names of 8+ letters), names of 5+ letters"""
        if len(n) < 5: return None
        best = None
        for f, lst in self.by.items():
            if abs(len(f) - len(n)) > 2 or len(f) < 5: continue
            d = levenshtein(n, f)
            if d > (2 if len(n) >= 8 else 1): continue
            r = difflib.SequenceMatcher(None, n, f).ratio()
            if r >= cut and (best is None or r > best[0]): best = (r, f, lst)
        return best


CONFUSABLE = {frozenset(p) for p in ("ij", "od", "lb", "li", "nh", "ce")}

def glyph_slip(a, b):
    """one letter swapped for a look-alike that another label's pixels can produce (aoiiang/aojiang)"""
    if len(a) != len(b): return False
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    return len(diff) == 1 and frozenset(diff[0]) in CONFUSABLE


def pick_entry(lst):
    order = {s: i for i, s in enumerate(TIER_OWN)}
    lst = sorted(lst, key=lambda ez: (not ez[1], order.get(ez[0]["src"], 9), -ez[0].get("share", 0)))
    return lst[0]


def main():
    regions, rid2key, picks = load()
    R = {r["key"]: r for r in regions}
    G = gazetteer(regions, rid2key)
    labels = json.load(open(os.path.join(HERE, "labels.json"), encoding="utf-8"))
    rid = np.load(os.path.join(RN, "region_id.npy"), mmap_mode="r")
    t = np.array(Image.open(os.path.join(NAMING, "text.png")).convert("RGBA"))
    m = t[:, :, 3] > 0
    idx_cache = {}
    def index(k):
        if k not in idx_cache:
            idx_cache[k] = Index(G.get(k, []), R[k]["country"] == "China")
        return idx_cache[k]

    def centre(b): return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    F = Features()

    # ---- label colour: her labels are drawn in their own region's colour x 0.75, rounded up
    # (255 comes out 191), so the ink colour names the owning region even when the label
    # sits on a neighbour, in the sea, on an inset or in a list. Matched within 1 per channel.
    def dark(h):
        return tuple(min(math.ceil(int(h[i:i + 2], 16) * 0.75), 191) for i in (1, 3, 5))
    by_dark = collections.defaultdict(list)
    for r in regions:
        by_dark[dark(r["color"])].append(r["key"])
    by_color_key = {r["color"] for r in regions}
    OFFS = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]

    def ink_colour(box):
        x0, y0, x1, y1 = box
        px = t[y0:y1, x0:x1].reshape(-1, 4)
        px = px[px[:, 3] > 0]
        if not len(px): return None
        return collections.Counter(map(tuple, px[:, :3].tolist())).most_common(1)[0][0]

    def bbox_dist(a, b):
        dx = max(b[0] - a[2], a[0] - b[2], 0); dy = max(b[1] - a[3], a[1] - b[3], 0)
        return (dx * dx + dy * dy) ** 0.5

    # the insets layer (insets.png, exported from the .ase): city cores drawn magnified, each in
    # its main-map region's exact colour. A label on an inset is near the inset copy, so the
    # distance to a region is to the nearer of its main-map pixels and its inset copy.
    ins = np.array(Image.open(os.path.join(HERE, "insets.png")).convert("RGBA"))
    inset_mask = ins[:, :, 3] > 0
    ys_, xs_ = np.nonzero(inset_mask)
    code = (ins[ys_, xs_, 0].astype(np.int64) << 16) | (ins[ys_, xs_, 1].astype(np.int64) << 8) | ins[ys_, xs_, 2]
    inset_copies = collections.defaultdict(list)     # region colour -> boxes of its inset pieces
    order_ = np.argsort(code, kind="stable")
    code, ys_, xs_ = code[order_], ys_[order_], xs_[order_]
    cuts = np.flatnonzero(np.diff(code)) + 1
    for a_, b_ in zip(np.r_[0, cuts], np.r_[cuts, len(code)]):
        if b_ - a_ < 4: continue
        h_ = "#%06x" % int(code[a_])
        if h_ not in by_color_key: continue
        # split into separate copies 300 px apart (a colour can recur on two insets)
        pts = sorted(zip(xs_[a_:b_].tolist(), ys_[a_:b_].tolist()))
        boxes_ = []
        for x, y in pts:
            for bb in boxes_:
                if bb[0] - 300 <= x <= bb[2] + 300 and bb[1] - 300 <= y <= bb[3] + 300:
                    bb[0] = min(bb[0], x); bb[1] = min(bb[1], y); bb[2] = max(bb[2], x + 1); bb[3] = max(bb[3], y + 1)
                    break
            else:
                boxes_.append([x, y, x + 1, y + 1])
        inset_copies[h_] = boxes_
    del ins, ys_, xs_, code, order_

    keys_of_colour = collections.defaultdict(list)
    for r in regions: keys_of_colour[r["color"]].append(r["key"])

    def near_inset_keys(boxes, pad=40):
        """regions whose inset copy lies under or within pad px of the boxes"""
        bx = [min(b[0] for b in boxes) - pad, min(b[1] for b in boxes) - pad,
              max(b[2] for b in boxes) + pad, max(b[3] for b in boxes) + pad]
        if not inset_mask[max(bx[1], 0):bx[3], max(bx[0], 0):bx[2]].any(): return []
        out = []
        for col, bbs in inset_copies.items():
            for bb in bbs:
                if bb[0] <= bx[2] and bb[2] >= bx[0] and bb[1] <= bx[3] and bb[3] >= bx[1]:
                    ks = keys_of_colour[col]
                    out.append(min(ks, key=lambda k: bbox_dist(bb, R[k]["bbox"]))); break
        return out

    def on_inset(box):
        x0, y0, x1, y1 = box
        return inset_mask[y0:y1, x0:x1].mean() >= 0.5 if (x1 > x0 and y1 > y0) else False

    def region_dist(box, k):
        d = bbox_dist(box, R[k]["bbox"])
        for bb in inset_copies.get(R[k]["color"], []):
            d = min(d, bbox_dist(box, bb))
        return d

    def colour_owner(box, text="", radius=1000):
        """the region whose colour the ink is drawn in: an exact colour before one off by 1,
        then a region whose names match the text, then the nearest"""
        c = ink_colour(box)
        if c is None: return None
        cands = []
        for o in OFFS:
            for k in by_dark.get((c[0] + o[0], c[1] + o[1], c[2] + o[2]), []):
                d = region_dist(box, k)
                if d <= radius: cands.append((o != (0, 0, 0), d, k))
        if not cands: return None
        cands.sort()
        if len(cands) > 1 and text:
            hit = [cc for cc in cands if name_hits(cc[2], text)]
            if hit: return hit[0][2]
        return cands[0][2]

    def feat_hits(text, box):
        n = norm_py(text)
        if len(n) < 4: return []
        cx, cy = centre(box)
        return F.exact(n, cx, cy)

    def key_of_rid(r): return rid2key.get(int(r)) if r else None

    def name_hits(k, text):
        """does text match region k's gazetteer exactly?"""
        if not k or k not in R: return False
        n = norm_py(text)
        return bool(n) and bool(index(k).exact(n))

    def ink_regions(boxes):
        cnt = collections.Counter()
        for (x0, y0, x1, y1) in boxes:
            ids = np.asarray(rid[y0:y1, x0:x1])[m[y0:y1, x0:x1]]
            cnt.update(ids.tolist())
        return cnt

    def near_regions(boxes, pad=40):
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
        win = np.asarray(rid[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad])
        return collections.Counter(win[win > 0].tolist())

    def span_match(lines, boxes):
        """keys of regions (under or near the text) whose names match the joined lines"""
        texts = {"".join(lines), " ".join(lines)}
        if lines[0].endswith("-"): texts.add(lines[0][:-1] + "".join(lines[1:]))
        out = []
        for r in list(ink_regions(boxes)) + [r for r, _ in near_regions(boxes).most_common(12)]:
            k = key_of_rid(r)
            if k and k not in out and any(name_hits(k, t) for t in texts): out.append(k)
        for k in near_inset_keys(boxes):
            if k not in out and any(name_hits(k, t) for t in texts): out.append(k)
        if not out:
            bx = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
            if any(feat_hits(t, bx) for t in texts): out.append("@geo")
        return out

    # ---- per label: kind; stacked lines split where the names say so
    items = []
    for li, l in enumerate(labels):
        lines = [x.strip(" .") if not ENTRY.match(x) else x for x in l["lines"]]
        txt = " ".join(lines)
        words = set(re.findall(r"[a-z]+", txt))
        if len(words & TITLE_WORDS) >= 2: continue
        if not re.search(r"[a-z0-9]", txt): continue
        boxes = l["boxes"]
        keep = [i for i, x in enumerate(lines) if re.search(r"[a-z0-9]", x)]
        lines = [lines[i] for i in keep]; boxes = [boxes[i] for i in keep]
        # two list columns read as one line ("12. pesanggrahan 18. jati asih"): two entries
        l2, b2 = [], []
        for x, b in zip(lines, boxes):
            mm = TWO_ENTRIES.match(x)
            if mm:
                cut = mm.start(3)
                per = (b[2] - b[0]) / max(len(x), 1)
                l2 += [x[:cut].rstrip(), x[cut:]]
                b2 += [[b[0], b[1], int(b[0] + (cut - 1) * per), b[3]], [int(b[0] + cut * per), b[1], b[2], b[3]]]
            else:
                l2.append(x); b2.append(b)
        lines, boxes = l2, b2
        ent = [ENTRY.match(x) for x in lines]
        mk = [MARKER.match(x) for x in lines]
        # entries and markers stand alone; the runs of other lines between them are names
        runs, cur = [], []
        for j, x in enumerate(lines):
            if ent[j]:
                if cur: runs.append(cur); cur = []
                items.append(dict(kind="entry", key=ent[j].group(1), text=ent[j].group(2), box=boxes[j], label=li))
            elif mk[j]:
                if cur: runs.append(cur); cur = []
                items.append(dict(kind="marker", key=mk[j].group(1), box=boxes[j], label=li,
                                  rids=ink_regions([boxes[j]])))
            elif re.match(r"^(\d{1,2}|[a-z])( +(\d{1,2}|[a-z]))+$", x):
                # several keys read as one line ("c d"): each key is its own marker
                if cur: runs.append(cur); cur = []
                x0, y0, x1, y1 = boxes[j]
                per = (x1 - x0) / max(len(x), 1)
                for mm in re.finditer(r"\d{1,2}|[a-z]", x):
                    b = [int(x0 + mm.start() * per), y0, int(x0 + mm.end() * per), y1]
                    items.append(dict(kind="marker", key=mm.group(0), box=b, label=li, rids=ink_regions([b])))
            else:
                cur.append(j)
        if cur: runs.append(cur)
        # colour first: consecutive lines drawn in the same region's colour are one label, a
        # change of colour starts another; lines whose colour names no region fall back to
        # the name split below
        merged = []
        for run in runs:
            # a line holding words in two regions' colours ("gaobei gou" over "dian", where
            # "gou" ends the bai / gou label beside it) is cut into its words first
            run2 = []
            for j in run:
                if " " in lines[j] and len(run) > 1:
                    words = lines[j].split(" ")
                    x0, y0, x1, y1 = boxes[j]
                    per = (x1 - x0) / max(len(lines[j]), 1)
                    wb, pos = [], 0
                    for wd in words:
                        wb.append([int(x0 + pos * per), y0, int(x0 + (pos + len(wd)) * per), y1]); pos += len(wd) + 1
                    wo = [colour_owner(b, wd) if len(norm_py(wd)) >= 2 else None for wd, b in zip(words, wb)]
                    if len({w for w in wo if w}) >= 2:
                        for wd, b in zip(words, wb):
                            lines.append(wd); boxes.append(b); run2.append(len(lines) - 1)
                        continue
                run2.append(j)
            run = run2
            own = [colour_owner(boxes[j], lines[j]) for j in run]
            seg = [0]
            for q in range(1, len(run)):
                if own[q] != own[q - 1] or own[q] is None: seg.append(q)
            seg.append(len(run))
            mine = []
            for a, b in zip(seg, seg[1:]):
                # a colourless stretch (several lines) is still name-split as a whole
                if own[a] is None and mine and mine[-1][1] is None:
                    mine[-1] = (mine[-1][0] + run[a:b], None, True)
                else:
                    mine.append((run[a:b], own[a], len(seg) > 2))
            merged += mine
        for run, owner, csplit in merged:
            if owner:
                sl = [lines[q] for q in run]; sb = [boxes[q] for q in run]
                box = [min(b[0] for b in sb), min(b[1] for b in sb), max(b[2] for b in sb), max(b[3] for b in sb)]
                # one line holding two labels side by side, in two colours: split at the space
                if len(sl) == 1 and " " in sl[0]:
                    words = sl[0].split(" ")
                    x0, y0, x1, y1 = sb[0]
                    per = (x1 - x0) / max(len(sl[0]), 1)
                    wb, pos = [], 0
                    for wd in words:
                        wb.append([int(x0 + pos * per), y0, int(x0 + (pos + len(wd)) * per), y1]); pos += len(wd) + 1
                    wo = [colour_owner(b, wd) if len(norm_py(wd)) >= 2 else None for wd, b in zip(words, wb)]
                    if len({w for w in wo if w}) >= 2:
                        # words without a colour of their own go with the word before (or after)
                        for q in range(len(wo)):
                            if wo[q] is None: wo[q] = next((w for w in wo[q::-1] if w), None) or next(w for w in wo if w)
                        q = 0
                        while q < len(words):
                            e = q
                            while e + 1 < len(words) and wo[e + 1] == wo[q]: e += 1
                            b = [wb[q][0], y0, wb[e][2], y1]
                            items.append(dict(kind="name", lines=[" ".join(words[q:e + 1])], boxes=[b], box=b, label=li,
                                              rids=ink_regions([b]), s=l["s"], matched=[], split=True, owner=wo[q]))
                            q = e + 1
                        continue
                # two labels stacked in one colour ("pamp- / anga" over "maca- / bebe", where
                # Macabebe is drawn in Pampanga's colour): split when the whole names nothing
                # and every part names a region; a part that doesn't name the colour's region
                # is placed by its name
                if len(sl) >= 2 and not name_hits(owner, "".join(x.rstrip("-") for x in sl)):
                    nn = len(sl)
                    memo2 = {}
                    def allmatch(i):
                        if i == nn: return []
                        if i in memo2: return memo2[i]
                        res = None
                        for j in range(i + 1, nn + 1):
                            if i == 0 and j == nn: continue
                            mt = [k_ for k_ in span_match(sl[i:j], sb[i:j]) if k_ != "@geo"]
                            if not mt: continue
                            rest = allmatch(j)
                            if rest is not None and (res is None or len(rest) + 1 < len(res)):
                                res = [(i, j, mt)] + rest
                        memo2[i] = res
                        return res
                    parts2 = allmatch(0)
                    if parts2 and len(parts2) >= 2:
                        for i, j, mt in parts2:
                            pb = sb[i:j]
                            bx = [min(b[0] for b in pb), min(b[1] for b in pb), max(b[2] for b in pb), max(b[3] for b in pb)]
                            own_ = owner if owner in mt else None
                            items.append(dict(kind="name", lines=sl[i:j], boxes=pb, box=bx, label=li, rids=ink_regions(pb),
                                              s=l["s"], matched=mt, split=True, owner=own_))
                        continue
                items.append(dict(kind="name", lines=sl, boxes=sb, box=box, label=li, rids=ink_regions(sb),
                                  s=l["s"], matched=[], split=csplit, owner=owner))
                continue
            # best split of the run into consecutive spans: lines in spans whose joined text
            # matches a region's names count; fewer spans break ties; no match keeps one span
            n = len(run)
            memo = {}
            def best(i):
                if i == n: return (0, 0, [])
                if i in memo: return memo[i]
                cands = []
                for j in range(i + 1, n + 1):
                    sl = [lines[run[q]] for q in range(i, j)]
                    sb = [boxes[run[q]] for q in range(i, j)]
                    mt = span_match(sl, sb) if n > 1 else []
                    sc, sp, parts = best(j)
                    cands.append((sc + (j - i if mt else 0), sp - 1, [(i, j, mt)] + parts))
                memo[i] = max(cands, key=lambda c: (c[0], c[1]))
                return memo[i]
            sc, sp, parts = best(0) if n > 1 else (0, 0, [(0, 1, [])])
            if sc == 0: parts = [(0, n, [])]
            for i, j, mt in parts:
                sl = [lines[run[q]] for q in range(i, j)]
                sb = [boxes[run[q]] for q in range(i, j)]
                box = [min(b[0] for b in sb), min(b[1] for b in sb), max(b[2] for b in sb), max(b[3] for b in sb)]
                # one line holding two labels side by side ("shenbei fushun"): split at the
                # spaces when the whole doesn't match a name and every word does
                if len(sl) == 1 and " " in sl[0] and not mt:
                    words = sl[0].split(" ")
                    x0, y0, x1, y1 = sb[0]
                    per = (x1 - x0) / max(len(sl[0]), 1)
                    wb, pos = [], 0
                    for wd in words:
                        wb.append([int(x0 + pos * per), y0, int(x0 + (pos + len(wd)) * per), y1]); pos += len(wd) + 1
                    wm = [span_match([wd], [b]) if len(norm_py(wd)) >= 3 else [] for wd, b in zip(words, wb)]
                    if all(wm):
                        for wd, b, mm in zip(words, wb, wm):
                            items.append(dict(kind="name", lines=[wd], boxes=[b], box=b, label=li,
                                              rids=ink_regions([b]), s=l["s"], matched=mm, split=True))
                        continue
                items.append(dict(kind="name", lines=sl, boxes=sb, box=box, label=li, rids=ink_regions(sb),
                                  s=l["s"], matched=mt, split=len(parts) > 1))
    print(collections.Counter(i["kind"] for i in items))
    # ---- pieces of one label that the reader kept apart ("ding" + "tao", "xing" over "ping"):
    # name pieces in the same region's colour that touch (same line with a gap of up to 10 px,
    # or stacked up to 6 px apart) are one label, read top to bottom and left to right, unless
    # only the other order matches a name of the region
    def touching(a, b):
        ov_y = min(a[3], b[3]) - max(a[1], b[1])
        ov_x = min(a[2], b[2]) - max(a[0], b[0])
        if ov_y >= 4 and ov_x > -10: return True
        if ov_x > -4 and max(a[1], b[1]) - min(a[3], b[3]) <= 6: return True
        return False
    joined_pieces = 0
    by_owner = collections.defaultdict(list)
    for it in items:
        if it["kind"] == "name" and it.get("owner"): by_owner[it["owner"]].append(it)
    for k, its in by_owner.items():
        if len(its) < 2: continue
        # clusters of touching pieces
        groups = []
        for it in its:
            hit = [g for g in groups if any(touching(it["box"], o["box"]) for o in g)]
            g = [it]
            for h in hit: g += h; groups.remove(h)
            groups.append(g)
        for g in groups:
            if len(g) < 2: continue
            # rows: pieces overlapping vertically share a row
            g.sort(key=lambda i: (i["box"][1], i["box"][0]))
            rows_ = []
            for it in g:
                for row_ in rows_:
                    if min(row_[0]["box"][3], it["box"][3]) - max(row_[0]["box"][1], it["box"][1]) >= 4:
                        row_.append(it); break
                else:
                    rows_.append([it])
            order = [it for row_ in rows_ for it in sorted(row_, key=lambda i: i["box"][0])]
            # two whole labels in one colour ("pampanga" over "macabebe") stay apart: when the
            # join names nothing of the region and some piece does on its own
            jt = "".join(x.rstrip("-") for it in order for x in it["lines"])
            def whole_elsewhere(it):
                t_ = "".join(x.rstrip("-") for x in it["lines"])
                return len(norm_py(t_)) >= 5 and any(kk != k for kk in span_match(it["lines"], it["boxes"]))
            if not name_hits(k, jt) and any(name_hits(k, "".join(x.rstrip("-") for x in it["lines"])) for it in order) \
                    and any(whole_elsewhere(it) for it in order):
                continue
            if len(order) == 2 and not name_hits(k, "".join(order[0]["lines"] + order[1]["lines"])) \
                    and name_hits(k, "".join(order[1]["lines"] + order[0]["lines"])):
                order = order[::-1]
            lines_, boxes_ = [], []
            for it in order: lines_ += it["lines"]; boxes_ += it["boxes"]
            rids_ = collections.Counter()
            for it in order: rids_.update(it["rids"])
            box = [min(b[0] for b in boxes_), min(b[1] for b in boxes_), max(b[2] for b in boxes_), max(b[3] for b in boxes_)]
            new = dict(order[0], lines=lines_, boxes=boxes_, box=box, rids=rids_, joined=len(order))
            for it in order: items.remove(it)
            items.append(new); joined_pieces += 1
    print(joined_pieces, "labels joined from pieces in one colour")
    # ---- markers -> list entries (same key; nearest, a name match counts the distance 3x shorter)
    markers = [i for i in items if i["kind"] == "marker"]
    entries = [i for i in items if i["kind"] == "entry"]
    for mk in markers:
        rc = collections.Counter({r: c for r, c in mk["rids"].items() if r})
        # keys are drawn in their region's colour too (the region is often tinier than the key)
        ow = colour_owner(mk["box"], radius=60)
        mk["region"] = ow or (key_of_rid(rc.most_common(1)[0][0]) if rc else None)
    for en in entries:
        en["owner"] = colour_owner(en["box"], en["text"], radius=1500)
    # first the pairs of one colour or whose entry names the marker's region (up to 2000 px
    # apart), then the rest by distance (up to 600 px; a letter key only up to 400 px,
    # letters are noisier); an entry whose colour names another region is not paired
    def pairing():
        pairs = []
        for a, mk in enumerate(markers):
            if not mk["region"] or a in used_m: continue
            ca = centre(mk["box"])
            for b, en in enumerate(entries):
                if b in used_e: continue
                if en["key"] != mk["key"] and {en["key"], mk["key"]} != {"9", "g"}: continue
                clash = en["owner"] and en["owner"] != mk["region"]
                if clash: continue
                cb = centre(en["box"])
                d = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
                hit = (en["owner"] == mk["region"]) or name_hits(mk["region"], en["text"])
                if hit and d <= 2000: pairs.append((0, d, a, b))
                elif d <= (600 if mk["key"].isdigit() else 400): pairs.append((1, d, a, b))
        pairs.sort()
        for tier, d, a, b in pairs:
            if a in used_m or b in used_e: continue
            used_m.add(a); used_e.add(b)
            markers[a]["entry"] = entries[b]; entries[b]["marker"] = markers[a]; markers[a]["dist"] = d
            markers[a]["name_checked"] = tier == 0
    used_m, used_e = set(), set()
    pairing()
    # then, among what is left, an entry drawn in the colour of a region whose own entry is
    # already paired (a slip: "1. shenyang" is in entry 2's colour) may pair with its key
    taken = {entries[b]["owner"] for b in used_e if entries[b]["owner"]}
    for b, en in enumerate(entries):
        if b not in used_e and en["owner"] in taken: en["owner"] = None
    pairing()

    # ---- name labels -> regions
    def assign_name(it):
        """the region under most of the label's ink, unless a region under or near it has
        the label's name (a label sitting partly on a neighbour)"""
        total = sum(it["rids"].values()) or 1
        share = {key_of_rid(r): c / total for r, c in it["rids"].items() if r and key_of_rid(r)}
        if it.get("owner"):
            s = share.get(it["owner"], 0)
            if s >= 0.5: return it["owner"], s, "under"
            if on_inset(it["box"]): return it["owner"], s, "label colour, on an inset"
            return it["owner"], s, "label colour, text off the region"
        matched =[k for k in (it.get("matched") or span_match(it["lines"], it["boxes"])) if k != "@geo"]
        cands = set(share) | set(matched)
        if not cands:
            near = near_regions(it["boxes"])
            if near:
                return key_of_rid(near.most_common(1)[0][0]), 0, "nearest"
            return None, 0, "none"
        text = "".join(it["lines"])
        def bonus(k):
            # a county, town or historical seat of that name is strong evidence; a parent
            # prefecture's name is shared by all its regions, so it counts for less
            if k not in matched: return 0
            srcs = {e["src"] for e, z in index(k).exact(norm_py(text))} or {"unit"}
            return 0.45 if srcs & {"unit", "town", "chgis"} else 0.2
        k = max(cands, key=lambda k: share.get(k, 0) + bonus(k))
        s = share.get(k, 0)
        if k in matched and s < 0.5: how = "nearby name match" if s == 0 else "name match, part of label on it"
        elif s >= 0.5: how = "under"
        else: how = "most ink"
        return k, s, how

    rc_xy = {r["key"]: ((r["bbox"][0] + r["bbox"][2]) / 2, (r["bbox"][1] + r["bbox"][3]) / 2) for r in regions}

    def name_search(text, box, radius=700):
        """labels on city insets (drawn on another layer, so over sea here): the nearest region
        within radius with a county, prefecture or historical seat of that name; a township
        name only counts within 250 px"""
        cx, cy = centre(box)
        n = norm_py(text)
        if len(n) < 3: return None, ""
        near = sorted((((x - cx) ** 2 + (y - cy) ** 2) ** 0.5, k) for k, (x, y) in rc_xy.items()
                      if abs(x - cx) < radius and abs(y - cy) < radius)
        for d, k in near:
            srcs = {e["src"] for e, z in index(k).exact(n)}
            if srcs - {"town"} or (srcs and d <= 250): return k, "exact"
        return None, ""

    def prefecture_of(k):
        ps = [p for p in R[k]["parents"] if p["level"] == "prefecture"]
        return ps[0]["name"] if ps else R[k]["country"]

    def list_search(group, radius=1600):
        """a numbered list whose keys aren't on the map (they're on an inset): the prefecture
        within radius where most of the list's names are found, then each name's region there"""
        cx, cy = centre(group[0]["box"])
        near = [k for k, (x, y) in rc_xy.items() if abs(x - cx) < radius and abs(y - cy) < radius]
        votes = collections.defaultdict(set)
        found = collections.defaultdict(dict)
        for i, en in enumerate(group):
            n = norm_py(en["text"])
            if len(n) < 3: continue
            for k in near:
                ex = index(k).exact(n)
                if ex:
                    p = prefecture_of(k)
                    votes[p].add(i)
                    rank = 0 if {e["src"] for e, z in ex} - {"town"} else 1
                    if i not in found[p] or rank < found[p][i][0]: found[p][i] = (rank, k)
        if not votes: return {}
        p = max(votes, key=lambda p: len(votes[p]))
        if len(votes[p]) < max(2, len(group) // 3) and len(group) > 1: return {}
        return {i: k for i, (rank, k) in found[p].items()}

    per_region = collections.defaultdict(list)
    unassigned = []
    paired_rk = {(m_["region"], m_["key"]) for m_ in markers if m_.get("entry")}
    paired_by_key = collections.defaultdict(list)
    for m_ in markers:
        if m_.get("entry"): paired_by_key[m_["key"]].append(m_)
    for en in entries:
        if en.get("owner"): paired_by_key[en["key"]].append(en)   # placed by its colour
    for it in items:
        if it["kind"] == "name":
            k, share, how = assign_name(it)
            if k is None or how == "nearest":
                k2, how2 = name_search("".join(it["lines"]), it["box"])
                if k2: k, share, how = k2, 0, f"name search ({how2}), text off the regions"
            it.update(region=k, share=share, how=how)
            if k: per_region[k].append(it)
            else: unassigned.append(it)
        elif it["kind"] == "marker":
            if it.get("entry") and it["region"]:
                en = it["entry"]
                per_region[it["region"]].append(dict(kind="keyed", lines=[en["text"]], key=it["key"], marker=it,
                                                     box=en["box"], share=1.0, how=f"key {it['key']}"))
            elif it["region"] and len(it["key"]) > 0 and it["key"].isdigit() and \
                    (it["region"], it["key"]) not in paired_rk and not any(
                        bbox_dist(it["box"], m["box"]) <= 600 for m in paired_by_key.get(it["key"], [])):
                # (the same key again, on the main map and its inset, is fine)
                unassigned.append(dict(it, why="numbered key with no list entry"))
    groups = collections.defaultdict(list)
    for it in items:
        if it["kind"] == "entry" and not it.get("marker") and it.get("owner"):
            # a list entry whose key is on an inset or lost: its colour names the region
            per_region[it["owner"]].append(dict(kind="name", lines=[it["text"]], boxes=[it["box"]], box=it["box"],
                                                share=0, how=f"list entry {it['key']}, by label colour",
                                                rids={}, owner=it["owner"]))
        elif it["kind"] == "entry" and not it.get("marker"): groups[it["label"]].append(it)
    for li, group in groups.items():
        hit = list_search(group)
        for i, it in enumerate(group):
            if i in hit:
                per_region[hit[i]].append(dict(kind="name", lines=[it["text"]], boxes=[it["box"]], box=it["box"],
                                               share=0, how=f"list entry {it['key']}, name search (exact)", rids={}))
            else:
                unassigned.append(dict(it, why="list entry with no key on the map"))

    # ---- a region holding several labels passes the extra ones to an unlabelled region under
    # or next to them whose names match (or, failing that, that holds a quarter of the ink)
    moved = 0
    for k in list(per_region):
        its = per_region[k]
        if len(its) < 2: continue
        its.sort(key=lambda i: (i["kind"] != "keyed", i.get("owner") != k, not name_hits(k, "".join(i["lines"])),
                                -i.get("share", 0), -sum(i.get("rids", {}).values())))
        for it in its[1:]:
            if it["kind"] != "name" or not it.get("boxes"): continue
            text = "".join(it["lines"])
            # a second label in this region's colour that names another region, unlabelled,
            # under or beside it, is a colour slip ("gucheng" drawn in Wuqiao's colour): it
            # moves there, but only on a name match
            slip = bool(it.get("owner")) and not name_hits(k, text)
            if it.get("owner") and not slip: continue
            under = it.get("rids") or {}
            total = sum(under.values()) or 1
            cands = []
            inset_near = [(kk_, 0) for kk_ in near_inset_keys(it["boxes"], pad=10)]
            for r_, c in list(under.items()) + [(r_, 0) for r_, _ in near_regions(it["boxes"]).most_common(8)] + inset_near:
                kk = r_ if isinstance(r_, str) else key_of_rid(r_)
                if not kk or kk == k or per_region.get(kk): continue
                hit = name_hits(kk, text)
                if slip and not hit:
                    # a near-twin colour (within 4 a channel) under the label also counts:
                    # "kunyu" over Rushan-Wendeng, drawn in Haiyang's all but equal colour
                    ic = ink_colour(it["box"])
                    hit = bool(ic) and c / total >= 0.25 and max(abs(p - q) for p, q in zip(ic, dark(R[kk]["color"]))) <= 4
                cands.append((hit, c / total, kk))
            cands.sort(reverse=True)
            if cands and (cands[0][0] or (cands[0][1] >= 0.25 and not slip)):
                hit, sh, kk = cands[0]
                per_region[k].remove(it)
                it = dict(it, share=sh, owner=None, how=(
                    "name match, label drawn in a neighbour's colour" if slip else
                    "name match, moved from a region with two labels" if hit
                    else "most ink after the first region kept another label"))
                per_region[kk].append(it); moved += 1
    print(moved, "extra labels moved to unlabelled regions")
    placed = [i for its in per_region.values() for i in its]
    print("labels placed:", len(placed), "by colour:", sum(1 for i in placed if i.get("owner")),
          "by key:", sum(1 for i in placed if i["kind"] == "keyed"),
          "by ink or name (fallback):", sum(1 for i in placed if not i.get("owner") and i["kind"] != "keyed"))

    # ---- names and characters
    rows, review = [], []
    for k, its in per_region.items():
        r = R[k]
        country = r["country"] or ""
        china = country == "China"
        # primary label: a keyed entry, else the label with most of its ink on the region
        its.sort(key=lambda i: (i["kind"] != "keyed", i.get("owner") != k, not name_hits(k, "".join(i["lines"])),
                                -i.get("share", 0), -sum(i.get("rids", {}).values())))
        it = its[0]
        others = [" / ".join(o["lines"]) for o in its[1:]]
        lines = [clean_marks(x, country) for x in it["lines"]]
        as_read = " / ".join(it["lines"]).replace("¿", "?" if country == "Vietnam" else "")
        # join stacked lines
        name = lines[0]
        for nxt in lines[1:]:
            if name.endswith("-"): name = name[:-1] + nxt
            elif china or country in JOINED: name = name + nxt
            else:
                spaced, solid = name + " " + nxt, name + nxt
                name = solid if (name_hits(k, solid) and not name_hits(k, spaced)) else spaced
        if china and " " in name and name_hits(k, name.replace(" ", "")):
            name = name.replace(" ", "")
        notes = []
        ocr_bad = "?" in "".join(lines)
        if ocr_bad: notes.append("unread glyph (?)")
        if it["how"] not in ("under",) and not it["how"].startswith("key"):
            notes.append(f"placed by {it['how']}")
        if it["kind"] == "keyed": notes.append(f"from numbered list, key {it['key']}")
        # the same name again (a map label and its inset copy) is not other text
        others = [o for o in others if norm_py(o) != norm_py(name)]
        if others: notes.append("other text on region: " + "; ".join(others[:4]))
        n = norm_py(name)
        idx = index(k)
        zh, src, conf, alt = "", "none", "", ""
        ex = idx.exact(n)
        matched_en = ""
        if ex:
            e, z = pick_entry(ex)
            src = e["src"]
            zh = z if china else (e.get("zh") or "")
            matched_en = e.get("en") or ""
            if ocr_bad:
                # the unread bit was stray ink (another label, a border patch): the rest matches
                ocr_bad = False; name = name.replace("?", "")
                notes.append("stray ink between letters ignored")
            others_zh = sorted({zz for ee, zz in ex if zz and zz != z})
            if china and others_zh: alt = "also " + ",".join(others_zh[:3])
            if china and not zh:
                # the name is known (an English-only list) but has no characters here
                fh = feat_hits(name, it["box"])
                if fh: zh = fh[0][1]; notes.append("characters from GeoNames")
                else:
                    for nb in r["neighbours"]:
                        ex2 = [ez for ez in index(nb["key"]).exact(n) if ez[1]] if nb["key"] in R else []
                        if ex2: zh = pick_entry(ex2)[1]; notes.append("characters from a neighbour's unit"); break
        if src == "none" and "?" in name:
            # an unread glyph: let it stand for one or two letters
            d = unicodedata.normalize("NFD", name.lower())
            d = "".join(c for c in d if not unicodedata.combining(c))
            pat = re.compile("".join("[a-z]{1,2}" if c == "?" else c for c in re.sub(r"[^a-z0-9?]", "", d)) + "$")
            wl = [(f, lst) for f, lst in idx.by.items() if pat.match(f)]
            if wl:
                f, lst = min(wl, key=lambda t: len(t[0]))
                e, z = pick_entry(lst)
                src = "wildcard " + e["src"]; zh = z if china else ""
                notes.append(f"unread glyph filled from {f}")
                if china: name = f
        if src == "none" and china:
            fh = feat_hits(name, it["box"])
            if fh:
                e, z = fh[0]
                src, zh = "geonames", z
                notes.append(f"GeoNames {e['en']} ({e['code']})")
        if src == "none":
            # neighbours
            for nb in r["neighbours"]:
                if nb["key"] not in R: continue
                ex2 = index(nb["key"]).exact(n)
                if ex2:
                    e, z = pick_entry(ex2)
                    src = "neighbour " + e["src"]; zh = z if china else ""
                    notes.append(f"name found in neighbour {nb['key']}")
                    break
            if src == "none":
                fz = idx.fuzzy(n)
                if fz:
                    rr, f, lst = fz
                    e, z = pick_entry(lst)
                    src = "fuzzy " + e["src"]; zh = z if china else ""
                    notes.append(f"fuzzy match {f} ({rr:.2f})")
                    if china and it.get("owner") == k and not ocr_bad and glyph_slip(n, f):
                        # the colour pins the region, and one of its own names is a letter
                        # away: a misread letter (aoiiang = aojiang). Take the name's spelling.
                        notes.append(f"read as {name}"); name = f; src = "fuzzy, colour " + e["src"]
        if alt: notes.append(alt)
        strong = it["kind"] == "keyed" or it.get("share", 0) >= 0.5 or "name match" in it["how"] or "(exact)" in it["how"] or "colour" in it["how"] or it.get("owner") == k
        if src.startswith("wildcard") and strong: conf = "medium"
        elif src.startswith("fuzzy, colour"): conf = "medium"
        elif ocr_bad or src.startswith("fuzzy"): conf = "low"
        elif (src in TIER_OWN and strong and (zh or not china) and src != "geonames"
              and not (china and len(zh) == 1 and it["kind"] != "keyed")):
            conf = "high"    # (a river from GeoNames, or one character, is only medium)
        elif china and not zh: conf = "low"
        elif src == "none" and not china and strong: conf = "medium"
        elif src == "none": conf = "low"
        else: conf = "medium"
        if k in picks:
            if norm(picks[k]) == norm(name): notes.append("matches pick")
            else: notes.append(f"pick is {picks[k]}")
        x0, y0, x1, y1 = it["box"]
        row = dict(key=k, country=country, text_as_read=as_read, name=titlecase(name),
                   zh=zh if china else "", zh_source=src if china else ("" if src == "none" else src),
                   confidence=conf, notes="; ".join(notes), label_x=int(x0), label_y=int(y0),
                   label_w=int(x1 - x0), label_h=int(y1 - y0), in_picks="yes" if k in picks else "")
        rows.append(row)
        if conf == "low" or (china and src == "none"):
            review.append(dict(kind="low or unmatched", **row))
    labelled = set(per_region)
    for it in unassigned:
        txt = " / ".join(it.get("lines") or [it.get("text") or it.get("key", "")])
        b = it["box"]
        review.append(dict(kind=it.get("why", "text not assigned to a region"), key="", country="",
                           text_as_read=txt, name="", zh="", zh_source="", confidence="", notes="",
                           label_x=int(b[0]), label_y=int(b[1]), label_w=int(b[2] - b[0]),
                           label_h=int(b[3] - b[1]), in_picks=""))
    for r in regions:
        if r["key"] in labelled or r["key"] in picks: continue
        x0, y0, x1, y1 = r["bbox"]
        review.append(dict(kind="no label found", key=r["key"], country=r["country"] or "", text_as_read="",
                           name="", zh="", zh_source="", confidence="", notes=f"{r['people']} people",
                           label_x=x0, label_y=y0, label_w=x1 - x0, label_h=y1 - y0, in_picks=""))
    cols = ["key", "country", "text_as_read", "name", "zh", "zh_source", "confidence", "notes",
            "label_x", "label_y", "label_w", "label_h", "in_picks"]
    rows.sort(key=lambda r: (r["country"], r["label_y"], r["label_x"]))
    with open(os.path.join(HERE, "oldnames.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
    with open(os.path.join(HERE, "review.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["kind"] + cols); w.writeheader(); w.writerows(review)
    # summary
    by = collections.Counter((r["country"], r["confidence"]) for r in rows)
    print(len(rows), "regions with a label;", len(review), "review rows")
    print(collections.Counter(r["kind"] for r in review))
    for c in sorted({r["country"] for r in rows}):
        print(f"  {c}: " + ", ".join(f"{cf} {by[(c, cf)]}" for cf in ("high", "medium", "low") if by[(c, cf)]))
    ch = [r for r in rows if r["country"] == "China"]
    print("China with characters:", collections.Counter((r["confidence"], bool(r["zh"])) for r in ch))
    pk = [r for r in rows if r["in_picks"]]
    print("in picks:", len(pk), "matching pick:", sum("matches pick" in r["notes"] for r in pk))


if __name__ == "__main__":
    main()
