"""Read the text layer into labels -> labels.json

    python asia1m/naming/oldnames/read.py

1. glyphs.find: exact template hits (1x, 2x, 4x). These find the lines: a line is a top row
   (and scale) with hits next to each other.
2. Each 1x line is then read again from end to end by a best-path search over its columns
   (decode_line): every glyph placed must have all its template pixels inked; ink that no glyph
   explains costs points, ink inside a placed glyph's body that isn't its own costs more. This
   reads letters that touch another label's pixels, which exact matching can't.
3. Accents: ink over an x-height letter (and one pixel under it) is read against marks.txt and
   added as combining marks (Vietnamese, Korean ŏ/ŭ). The horn of ơ/ư is the single pixel over
   the letter's right edge.
4. Lines read twice at overlapping places (a stray line one row off) keep the better reading.
5. Words: a gap of 3+ columns is a space; 7+ columns ends the segment. Segments whose top rows
   differ by one pixel and sit end to end are joined (labels nudged by a pixel).
6. Labels: segments stacked under one another (next top row 7-13 rows lower, overlapping
   horizontally, centres close) are one label.
"""
import os, sys, json, collections, unicodedata
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glyphs
Image.MAX_IMAGE_PIXELS = None

TEXT = os.path.join(HERE, "..", "text.png")
SPACE_GAP = 3     # columns between glyphs that make a space (at 1x)
SPLIT_GAP = 7     # columns that end a segment
UP = 4            # rows over the line's top row that accents reach
NB = UP + 9       # bits per column code in decode_line
POP = [bin(i).count("1") for i in range(1 << NB)]


def load_marks(path=os.path.join(HERE, "marks.txt")):
    out = {}
    lines = open(path, encoding="utf-8").read().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("="):
            name = lines[i][1:]
            j = i + 1; rows = []
            while j < len(lines) and lines[j].strip():
                rows.append(lines[j]); j += 1
            g = np.array([[c == "#" for c in r] for r in rows], bool)
            out[g.tobytes() + bytes(str(g.shape), "ascii")] = name
            i = j
        else:
            i += 1
    return out


def crop(a):
    ys, xs = np.nonzero(a)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1], (int(ys.min()), int(xs.min()))


class Font:
    def __init__(self, T):
        self.items = []
        for ch, gs in T.items():
            for g in gs:
                asc = g[:2].any(); desc = g[7:].any()
                top = 0 if asc else 2
                bot = 8 if desc else 6
                codes = [sum(int(g[k, x]) << (k + UP) for k in range(9)) for x in range(g.shape[1])]
                ext = sum(1 << (k + UP) for k in range(top, bot + 1))
                ink = int(g.sum())
                gain = ink if ch.isalnum() or ch in "ūōü" else ink - 1.5
                if ch == "ı": gain -= 3     # dotless i only where a mark replaces the dot
                # unexplained ink in the glyph's columns is charged like a skipped column's,
                # except over an x-height letter (rows 0-1 are accent room)
                charge = ROWS06 if asc else ROWS26
                self.items.append(dict(ch=ch, codes=codes, w=g.shape[1], ext=ext, ink=ink, gain=gain,
                                       asc=asc, desc=desc, g=g, charge=charge))


ROWS06 = sum(1 << (k + UP) for k in range(0, 7))
ROWS26 = sum(1 << (k + UP) for k in range(2, 7))


def decode_line(win, font):
    """win: NB x W bool (rows y-UP .. y+8). Returns (score, glyphs [(x, w, item)], covered)."""
    W = win.shape[1]
    wc = [int(sum(int(win[k, x]) << k for k in range(NB))) for x in range(W)]
    NEG = -1e9
    # best[x][f]: f=1 when a glyph ended exactly at x (the next column must be skipped)
    best = [[NEG, NEG] for _ in range(W + 1)]
    back = [[None, None] for _ in range(W + 1)]
    best[0][0] = 0.0
    for x in range(W):
        for f in (0, 1):
            b = best[x][f]
            if b == NEG: continue
            # skip column x: unexplained ink in rows 0-6 costs
            c = b - 1.5 * POP[wc[x] & ROWS06]
            if c > best[x + 1][0]:
                best[x + 1][0] = c; back[x + 1][0] = (x, f, None)
            if f == 1: continue
            for it in font.items:
                w = it["w"]
                if x + w > W: continue
                ok = True; foreign = 0; unexpl = 0
                for j in range(w):
                    tc = it["codes"][j]; c_ = wc[x + j]
                    if tc & ~c_: ok = False; break
                    foreign += POP[(c_ & ~tc) & it["ext"]]
                    unexpl += POP[(c_ & ~tc) & it["charge"]]
                if not ok: continue
                s = b + it["gain"] - 1.5 * unexpl - 0.75 * foreign
                if s > best[x + w][1]:
                    best[x + w][1] = s; back[x + w][1] = (x, 0, it)
    f = 0 if best[W][0] >= best[W][1] else 1
    score = best[W][f]
    out = []
    x = W
    while x > 0:
        px, pf, it = back[x][f]
        if it is not None: out.append((px, it["w"], it))
        x, f = px, pf
    out.reverse()
    # ink between two glyphs (or right next to the first/last) that no glyph explains: a letter
    # drawn by hand or crossed by something -> '?'
    BODY = sum(1 << (k + UP) for k in range(2, 7))
    covered = [False] * W
    for px, w, it in out:
        for j in range(px, px + w): covered[j] = True
    qs = []
    x = 0
    while x < W:
        if covered[x] or not (wc[x] & BODY): x += 1; continue
        e = x
        while e < W and not covered[e] and (wc[e] & BODY): e += 1
        ink = sum(POP[wc[j] & BODY] for j in range(x, e))
        left = any(px + w in (x, x - 1) for px, w, _ in out)
        right = any(px in (e, e + 1) for px, w, _ in out)
        if ink >= 4 and (left or right) and e - x <= 9:
            qs.append((x, e - x, dict(ch="?", g=np.zeros((9, e - x), bool), asc=True, desc=False)))
        x = e
    out = sorted(out + qs, key=lambda t: t[0])
    return score, out


def accents(win, gl, MK):
    """read marks over/under each placed x-height glyph -> list of strings (one per glyph)."""
    used = np.zeros_like(win)
    for x, w, it in gl:
        used[UP:, x:x + w] |= it["g"]
    res = win & ~used
    out = []
    for x, w, it in gl:
        ch = it["ch"]
        if ch == "ı": ch = "i"
        if not ch.isalpha() or it["asc"] or ord(ch) > 127:
            out.append(ch); continue
        zone = np.zeros_like(res)
        zone[:UP + 2, x:x + w] = res[:UP + 2, x:x + w]
        if not it["desc"]: zone[UP + 7, x:x + w] = res[UP + 7, x:x + w]
        if not zone.any():
            out.append(ch); continue
        L, n = ndi.label(zone, np.ones((3, 3), bool))
        base = ch
        horn, above, below, bad = [], [], [], False
        pieces = []
        for k in range(1, n + 1):
            c, (y0, x0) = crop(L == k)
            pieces.append((y0 + c.shape[0], y0, x0, c))
        # nearest the letter first: a mark must touch the accent rows (its bottom on row 0 or 1
        # of the line) or sit right over a mark already read (the tone over a circumflex)
        lowest_top = UP + 2
        for bot, y0, x0, c in sorted(pieces, key=lambda p: -p[0]):
            if ch in "ou" and c.shape == (1, 1) and y0 == UP + 1 and x0 == x + w - 1:
                horn.append("̛"); continue
            if y0 == UP + 7 and c.shape == (1, 1):
                below.append("̣"); continue
            if y0 >= UP + 7: continue
            key = c.tobytes() + bytes(str(c.shape), "ascii")
            near = bot >= UP + 1 or bot >= lowest_top - 1
            if key in MK and near:
                above.append((y0, MK[key])); lowest_top = min(lowest_top, y0)
            elif y0 < UP:
                continue          # reaches over the top row and is no mark: another label
            else:
                bad = True
        if bad:
            out.append(ch + "\u0000"); continue    # flagged: something unread over the letter
        above.sort(reverse=True)
        s = base + "".join(horn) + "".join(a for _, a in above) + "".join(below)
        out.append(unicodedata.normalize("NFC", s))
    return out


TINY = {  # the 3x5 digits used as keys inside small city regions
    "0": [".#.", "#.#", "#.#", "#.#", ".#."], "1": [".#.", "##.", ".#.", ".#.", "###"],
    "2": ["##.", "..#", ".#.", "#..", "###"], "3": ["###", "..#", ".##", "..#", "###"],
    "4": ["#.#", "#.#", "###", "..#", "..#"], "5": ["###", "#..", "###", "..#", "##."],
    "6": [".##", "#..", "###", "#.#", "###"], "7": ["###", "..#", ".#.", ".#.", ".#."],
    "8": ["###", "#.#", "###", "#.#", "###"], "9": ["###", "#.#", "###", "..#", "##."],
}


def tiny_numbers(m):
    """3x5 digits standing alone (blank all round) -> labels with s=0; digits one or two
    columns apart on the same rows are one number."""
    shapes = {np.array([[c == "#" for c in r] for r in v], bool).tobytes() + bytes([len(v[0])]): k
              for k, v in TINY.items()}
    lab, n = ndi.label(m, np.ones((3, 3), bool))
    found = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        h = sl[0].stop - sl[0].start; w = sl[1].stop - sl[1].start
        if h != 5 or w != 3: continue
        y0, x0 = sl[0].start, sl[1].start
        g = lab[sl] == i
        d = shapes.get(g.tobytes() + bytes([w]))
        if d is None: continue
        ring = m[max(y0 - 1, 0):y0 + 6, max(x0 - 1, 0):x0 + w + 1].copy()
        ring[1:6, 1:w + 1] = False
        if ring.any(): continue
        found.append((y0, x0, d))
    found.sort()
    out = []
    for y0, x0, d in found:
        if out and out[-1]["y"] == y0 and 0 <= x0 - out[-1]["x1"] <= 2:
            out[-1]["text"] += d; out[-1]["x1"] = x0 + 3
        else:
            out.append(dict(y=y0, x0=x0, x1=x0 + 3, text=d))
    return [dict(lines=[o["text"]], s=0, boxes=[[o["x0"], o["y"], o["x1"], o["y"] + 5]]) for o in out]


def main():
    t = np.array(Image.open(TEXT).convert("RGBA"))
    m = t[:, :, 3] > 0
    H, W = m.shape
    T = glyphs.load_templates()
    MK = load_marks()
    hits = glyphs.find(m, T)
    print(len(hits), "exact glyph hits", flush=True)
    # 'i' for dotless use in accents: the font's i with its dot is a template; accents replace
    # the dot, so add a dotless i the decoder can place under a mark
    T2 = dict(T)
    T2["ı"] = [np.array([[c == "#" for c in r] for r in ["..", "..", "##", ".#", ".#", ".#", ".#", "..", ".."]], bool)]
    font = Font(T2)

    rows = collections.defaultdict(list)
    for ch, x, y, w, s in hits:
        if ch.isalnum(): rows[(s, y)].append((x, w * s, ch))
    # candidate segments from exact hits
    cands = []
    for (s, y), gl in rows.items():
        gl.sort()
        cur = None
        for x, w, ch in gl:
            if cur and x - cur[1] <= SPLIT_GAP * s * 2:
                cur[1] = x + w; cur[2] += 1
            else:
                if cur: cands.append((s, y) + tuple(cur))
                cur = [x, x + w, 1]
        if cur: cands.append((s, y) + tuple(cur))
    print(len(cands), "candidate segments", flush=True)
    segs = []
    for s, y, x0, x1, n in cands:
        if s != 1:
            gl = sorted(g for g in rows[(s, y)] if x0 <= g[0] < x1)
            segs.append(dict(s=s, y=y, glyphs=[[x, w, ch] for x, w, ch in gl], score=sum(9 for _ in gl), pix=None))
            continue
        if y < UP or y + 9 > H: continue
        # extend to both sides until SPLIT_GAP blank columns in rows 0-6
        a = x0
        blank = 0
        while a > 0 and blank < SPLIT_GAP:
            a -= 1
            blank = blank + 1 if not m[y:y + 7, a].any() else 0
        b = x1; blank = 0
        while b < W - 1 and blank < SPLIT_GAP:
            blank = blank + 1 if not m[y:y + 7, b].any() else 0
            b += 1
        win = m[y - UP:y + 9, a:b]
        score, gl = decode_line(win, font)
        if not gl: continue
        chars = accents(win, gl, MK)
        items = [[a + x, w, c, it["g"]] for (x, w, it), c in zip(gl, chars)]
        ink = sum(int(i[3].sum()) for i in items if i[2][:1].isalnum())
        segs.append(dict(s=1, y=y, items=items, score=score, ink=ink))
    print(len(segs), "decoded", flush=True)
    # overlapping readings (a line read again one or two rows off, or reaching into another
    # label): lines with more letter ink go first; a later line loses each glyph whose
    # pixels are already taken, and the whole line if it loses half its letter ink
    segs.sort(key=lambda g: -(g.get("ink") or g["score"]))
    claimed = np.zeros_like(m)
    kept = []
    for g in segs:
        if g["s"] != 1:
            kept.append(g); continue
        y = g["y"]
        keep, lost = [], 0
        for x, w, c, gg in g["items"]:
            reg = claimed[y:y + 9, x:x + w]
            if c[:1].isalnum() and (reg & gg).sum() > 0.3 * gg.sum():
                # another line already read these pixels: keep a '?' so the gap shows
                lost += int(gg.sum()); keep.append([x, w, "?", np.zeros_like(gg)]); continue
            keep.append([x, w, c, gg])
        if lost > 0.5 * g["ink"]: continue
        # punctuation only next to a kept letter or digit
        al = [k for k in keep if k[2][:1].isalnum()]
        if not al: continue
        keep = [k for k in keep if k[2][:1].isalnum() or
                any(min(abs(k[0] - (q[0] + q[1])), abs(q[0] - (k[0] + k[1]))) <= 2 for q in al)]
        for x, w, c, gg in keep:
            if c[:1].isalnum(): claimed[y:y + 9, x:x + w] |= gg
        g["glyphs"] = [[x, w, c] for x, w, c, gg in keep]
        kept.append(g)
    print(len(kept), "lines kept", flush=True)
    # split into segments at wide gaps; spaces at narrow gaps
    out = []
    for g in kept:
        s = g["s"]
        cur = None
        for x, w, ch in sorted(g["glyphs"]):
            if cur is not None:
                gap = (x - cur["x1"]) // s
                if gap >= SPLIT_GAP:
                    out.append(cur); cur = None
                elif gap >= SPACE_GAP:
                    cur["text"] += " "
            if cur is None:
                cur = dict(s=s, y=g["y"], x0=x, x1=x, text="")
            cur["text"] += ch; cur["x1"] = x + w
        if cur: out.append(cur)
    # tidy: punctuation only beside letters/digits, strip stray punctuation at the ends
    segs = []
    for g in out:
        t_ = g["text"].replace("\u0000", "¿")   # ¿ = unread ink over the letter before it
        t_ = t_.strip(" '")
        if not any(c.isalnum() for c in t_): continue
        g["text"] = t_
        segs.append(g)
    # join pieces nudged a pixel up or down that sit end to end
    segs.sort(key=lambda g: (g["x0"], g["y"]))
    joined = True
    while joined:
        joined = False
        segs.sort(key=lambda g: g["x0"])
        for i, g in enumerate(segs):
            for h in segs:
                if h is g or h["s"] != g["s"]: continue
                if abs(h["y"] - g["y"]) == 1 * g["s"] and 0 < h["x0"] - g["x1"] < SPLIT_GAP * g["s"]:
                    gap = (h["x0"] - g["x1"]) // g["s"]
                    g["text"] += (" " if gap >= SPACE_GAP else "") + h["text"]
                    g["x1"] = h["x1"]; segs.remove(h); joined = True
                    break
            if joined: break
    print(len(segs), "segments", flush=True)
    # stacked labels
    idx = collections.defaultdict(list)
    for i, g in enumerate(segs): idx[(g["s"], g["y"])].append(i)
    below = {}
    for i, g in enumerate(segs):
        s = g["s"]; best = None
        for dy in range(7 * s, 13 * s + 1):
            for j in idx.get((s, g["y"] + dy), []):
                h = segs[j]
                ov = min(g["x1"], h["x1"]) - max(g["x0"], h["x0"])
                cg = (g["x0"] + g["x1"]) / 2; ch_ = (h["x0"] + h["x1"]) / 2
                if ov > 0 and abs(cg - ch_) <= max(g["x1"] - g["x0"], h["x1"] - h["x0"]) / 2 + 2:
                    if best is None or dy < best[0]: best = (dy, j)
            if best: break
        if best: below[i] = best[1]
    above = {}
    for i, j in below.items():
        if j in above:
            k = above[j]
            cj = (segs[j]["x0"] + segs[j]["x1"]) / 2
            if abs((segs[i]["x0"] + segs[i]["x1"]) / 2 - cj) < abs((segs[k]["x0"] + segs[k]["x1"]) / 2 - cj):
                above[j] = i
        else:
            above[j] = i
    below = {i: j for j, i in above.items()}
    labels = []
    for i, g in enumerate(segs):
        if i in above: continue
        chain = [i]
        while chain[-1] in below and len(chain) < 6: chain.append(below[chain[-1]])
        parts = [segs[k] for k in chain]
        labels.append(dict(lines=[p["text"] for p in parts], s=g["s"],
                           boxes=[[p["x0"], p["y"], p["x1"], p["y"] + 9 * p["s"]] for p in parts]))
    labels += tiny_numbers(m & ~claimed)
    json.dump(labels, open(os.path.join(HERE, "labels.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(len(labels), "labels;", sum(len(l["lines"]) > 1 for l in labels), "multi-line;",
          sum("?" in "".join(l["lines"]) for l in labels), "with ?")


if __name__ == "__main__":
    main()
