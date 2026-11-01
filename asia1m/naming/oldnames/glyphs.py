"""Find every glyph of the label font on the text layer by exact template matching.

The labels use one small proportional pixel font (x-height 5, ascenders and descenders 2),
sometimes drawn at 2x or 4x. glyphs.txt holds one template per character, each 9 rows tall
(rows 0-1 ascender, 2-6 x-height, 7-8 descender). A glyph matches at (x, y) when its window
in the text mask equals the template exactly and the columns just left and right of it are
blank. The window is rows 0-8 for letters with a descender and rows 0-6 for the rest, so the
ascenders of a line stacked tightly underneath don't spoil a match; the column checks either
side always use rows 0-6.

find(mask) -> list of hits (char, x, y, w, scale) where (x, y) is the top-left of the 9*scale
window in map pixels (y is the ascender top; the x-height top is y + 2*scale).
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = (1 << 9) - 1
SHORT = (1 << 7) - 1      # rows 0-6


def load_templates(path=os.path.join(HERE, "glyphs.txt")):
    t = {}
    lines = open(path, encoding="utf-8").read().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("="):
            ch = lines[i][1:]
            rows = lines[i + 1:i + 10]
            g = np.array([[c == "#" for c in r] for r in rows], bool)
            t.setdefault(ch, []).append(g)
            i += 10
        else:
            i += 1
    return t


def row_mask(g):
    return FULL if g[7:].any() else SHORT


def colcodes(m, rows=9):
    H, W = m.shape
    code = np.zeros((H - rows + 1, W), np.int32)
    for k in range(rows):
        code |= m[k:H - rows + 1 + k].astype(np.int32) << k
    return code


def tcodes(g):
    return [int(sum(int(g[k, x]) << k for k in range(g.shape[0]))) for x in range(g.shape[1])]


def match(m, templates):
    """hits at scale 1 on mask m: list of (char, x, y, w)."""
    code = colcodes(m)
    Hc, W = code.shape
    hits = []
    groups = {}
    for ch, gs in templates.items():
        for g in gs:
            groups.setdefault(row_mask(g), []).append((ch, tcodes(g)))
    side = np.zeros((Hc, W + 2), np.int32)
    side[:, 1:-1] = code & SHORT
    for mask, tl in groups.items():
        cm = code & mask if mask != FULL else code
        by_first = {}
        for ch, c in tl: by_first.setdefault(c[0], []).append((ch, c))
        firsts = np.array(sorted(by_first))
        ys, xs = np.nonzero(np.isin(cm, firsts))
        fv_all = cm[ys, xs]
        lb = side[ys, xs] == 0                     # column x-1 blank in rows 0-6
        ys, xs, fv_all = ys[lb], xs[lb], fv_all[lb]
        for fv in np.unique(fv_all):
            sel = fv_all == fv
            yy, xx = ys[sel], xs[sel]
            for ch, c in by_first[int(fv)]:
                w = len(c)
                ok = np.ones(len(yy), bool)
                for j in range(1, w):
                    xj = xx + j
                    inb = xj < W
                    ok &= inb
                    ok[inb] &= cm[yy[inb], xj[inb]] == c[j]
                ok &= side[yy, np.minimum(xx + w + 1, W + 1)] == 0   # column x+w blank
                for y, x in zip(yy[ok], xx[ok]):
                    hits.append((ch, int(x), int(y), w))
    return hits


def find(m, templates=None, scales=(1, 2, 4)):
    if templates is None:
        templates = load_templates()
    out = []
    for s in scales:
        if s == 1:
            out += [(ch, x, y, w, 1) for ch, x, y, w in match(m, templates)]
            continue
        seen = set()
        H, W = m.shape
        for oy in range(s):
            for ox in range(s):
                sub = m[oy::s, ox::s]
                for ch, x, y, w in match(sub, templates):
                    X, Y = x * s + ox, y * s + oy
                    if X - s < 0 or X + (w + 1) * s > W: continue
                    ok = False
                    for gg in templates[ch]:
                        if gg.shape[1] != w: continue
                        rows = 9 if row_mask(gg) == FULL else 7
                        big = np.zeros((rows * s, (w + 2) * s), bool)
                        big[:, s:s + w * s] = np.kron(gg[:rows], np.ones((s, s), bool))
                        win = m[Y:Y + rows * s, X - s:X + (w + 1) * s]
                        if win.shape == big.shape and (win == big).all():
                            ok = True
                    if ok and (X, Y) not in seen:
                        seen.add((X, Y))
                        out.append((ch, X, Y, w, s))
    return out
