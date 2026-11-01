"""Build "unixel Uyghur" from unixel-Regular.ttf.

unixel (Rooyin / Unixel project, OFL 1.1) is a Persian pixel font. This script adds the
Arabic-script letters Uyghur, Kazakh and Kirghiz need, drawn on the font's own pixel grid
(1 pixel = 160 font units, baseline at y=0) by copying existing glyphs and moving dots.

    C:\\Python39\\python.exe build_font.py [path\\to\\unixel-Regular.ttf]

Default source is the installed font; it is only read. Output: unixel-Uyghur-Regular.ttf
beside this script. To add more letters later (e.g. Urdu), add an entry to LETTERS below:
each form is (base glyph to copy, pixels to add, pixels to remove), pixels as (col, row)
in font pixels with row 0 the pixel row sitting on the baseline.
"""
import os
import sys

from fontTools.ttLib import TTFont
from fontTools.pens.pointInsidePen import PointInsidePen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables import otTables

U = 160  # font units per pixel
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars(
    r'%LOCALAPPDATA%\Microsoft\Windows\Fonts\unixel-Regular.ttf')
OUT = os.path.join(HERE, 'unixel-Uyghur-Regular.ttf')

# ---------------------------------------------------------------------------------------
# Marks, as pixel lists relative to the glyph origin.
DOTS3_WAW = [(2, 7), (1, 5), (3, 5)]                    # three dots over waw (as on theh)
DAMMA_WAW = [(2, 8), (3, 8), (2, 7), (3, 7), (2, 6), (1, 5)]   # small damma: blob + tail
SMALL_V_WAW = [(1, 6), (3, 6), (2, 5)]                  # v
SMALL_INV_V_WAW = [(2, 6), (1, 5), (3, 5)]              # inverted v (Kirghiz yu)
SMALL_YEH_WAW = [(3, 7), (4, 7), (0, 6), (2, 6), (1, 5)]  # tiny alef maksura shape
BAR_WAW = [(3, -1), (5, -1)]                            # bar through the tail (Kirghiz oe)
HAMZA_HIGH = [(2, 8), (3, 8), (1, 7), (2, 6), (3, 6), (1, 5)]  # same hamza as on waw

# Each letter: codepoint, joining ('R' right-joining, 'D' dual, 'U' none), and per form
# (glyph name, presentation-form codepoint or None, base glyph, add, remove, advance or None).
LETTERS = [
    # Uyghur / Kazakh / Kirghiz waw family: isolated + final
    dict(cp=0x06C7, join='R', forms=dict(
        isol=('u06C7', 0xFBD7, 'u0648', DAMMA_WAW, [], None),
        fina=('uFBD8', 0xFBD8, 'uFEEE', DAMMA_WAW, [], None))),
    dict(cp=0x06C6, join='R', forms=dict(
        isol=('u06C6', 0xFBD9, 'u0648', SMALL_V_WAW, [], None),
        fina=('uFBDA', 0xFBDA, 'uFEEE', SMALL_V_WAW, [], None))),
    dict(cp=0x06C8, join='R', forms=dict(
        isol=('u06C8', 0xFBDB, 'u0648', SMALL_YEH_WAW, [], None),
        fina=('uFBDC', 0xFBDC, 'uFEEE', SMALL_YEH_WAW, [], None))),
    dict(cp=0x06CB, join='R', forms=dict(
        isol=('u06CB', 0xFBDE, 'u0648', DOTS3_WAW, [], None),
        fina=('uFBDF', 0xFBDF, 'uFEEE', DOTS3_WAW, [], None))),
    dict(cp=0x06C5, join='R', forms=dict(
        isol=('u06C5', 0xFBE0, 'u0648', BAR_WAW, [], None),
        fina=('uFBE1', 0xFBE1, 'uFEEE', BAR_WAW, [], None))),
    dict(cp=0x06C9, join='R', forms=dict(
        isol=('u06C9', 0xFBE2, 'u0648', SMALL_INV_V_WAW, [], None),
        fina=('uFBE3', 0xFBE3, 'uFEEE', SMALL_INV_V_WAW, [], None))),
    # E: dotless yeh with two dots stacked vertically below
    dict(cp=0x06D0, join='D', forms=dict(
        isol=('u06D0', 0xFBE4, 'u0649', [(4, -5), (4, -7)], [], None),
        fina=('uFBE5', 0xFBE5, 'uFBFD', [(3, -5), (3, -7)], [], None),
        init=('uFBE6', 0xFBE6, 'uFBE8', [(1, -3), (1, -5)], [], None),
        medi=('uFBE7', 0xFBE7, 'uFBE9', [(1, -3), (1, -5)], [], None))),
    # NG: kaf with three dots. Isolated/final: the dots replace kaf's small inner mark and
    # sit higher than it, so a separated triangle reads apart from kaf's joined zigzag.
    dict(cp=0x06AD, join='D', forms=dict(
        isol=('u06AD', 0xFBD3, 'u0643', [(4, 6), (6, 6), (5, 8)], [(5, 6), (4, 5), (5, 4)], None),
        fina=('uFBD4', 0xFBD4, 'uFEDA', [(4, 6), (6, 6), (5, 8)], [(5, 6), (4, 5), (5, 4)], None),
        init=('uFBD5', 0xFBD5, 'uFEDB', [(3, 9), (5, 9), (4, 11)], [], None),
        medi=('uFBD6', 0xFBD6, 'uFEDC', [(3, 9), (5, 9), (4, 11)], [], None))),
    # Heh doachashmee (Uyghur h): the two-eyed shape in every form, so it never looks
    # like ae (U+06D5), which uses the round heh.
    dict(cp=0x06BE, join='D', forms=dict(
        isol=('u06BE', 0xFBAA, 'uFEEB', [], [(0, 0)], None),
        fina=('uFBAB', 0xFBAB, 'uFEEB', [(5, 0), (6, 0), (7, 0)], [(0, 0)], None),
        init=('uFBAC', 0xFBAC, 'uFEEB', [], [], None),
        medi=('uFBAD', 0xFBAD, 'uFEEB', [(5, 0), (6, 0), (7, 0)], [], None))),
    # Kazakh high hamza: a spacing letter (the source maps it to the zero-width hamza mark)
    dict(cp=0x0674, join='U', forms=dict(
        isol=('u0674', None, None, HAMZA_HIGH, [], 5 * U))),
]

# U+06CC Farsi yeh gets its own copy of the yeh glyph so U+0649 can go dotless.
KERN_AFTER_REH = ['u06AD', 'uFBD5', 'u06BE', 'uFBE6']   # like u0643, uFEDB, u0647, uFBE8

FAMILY = 'unixel Uyghur'
PS_NAME = 'unixelUyghur-Regular'
VERSION = '1.001'
MOD_NOTE = ('Modified 2026-10-23: added Uyghur, Kazakh and Kirghiz letters '
            '(U+06AD, 06BE, 06C5-06C9, 06CB, 06D0, 0674) and made U+0649 dotless in '
            'initial and medial forms. Built with build_font.py from unixel 1.000.')
DESCENT = 7 * U   # one pixel deeper than the source, for the lower dot of E


# ---------------------------------------------------------------------------------------
def pixels(gs, gn):
    g = gs[gn]
    bp = BoundsPen(gs)
    g.draw(bp)
    if bp.bounds is None:
        return set()
    x0, y0, x1, y1 = bp.bounds
    cells = set()
    for j in range(int(y0 // U) - 1, int(y1 // U) + 1):
        for i in range(int(x0 // U) - 1, int(x1 // U) + 1):
            p = PointInsidePen(gs, (i * U + U / 2, j * U + U / 2))
            g.draw(p)
            if p.getResult():
                cells.add((i, j))
    return cells


def trace(cells):
    """Union outline of pixel squares; outer contours clockwise (TrueType)."""
    edges = {}
    for (i, j) in cells:
        for a, b in (((i, j), (i, j + 1)), ((i, j + 1), (i + 1, j + 1)),
                     ((i + 1, j + 1), (i + 1, j)), ((i + 1, j), (i, j))):
            if (b, a) in edges:
                del edges[(b, a)]
            else:
                edges[(a, b)] = True
    out = {}
    for a, b in edges:
        out.setdefault(a, []).append(b)
    contours = []
    while out:
        start = next(iter(out))
        pts = [start]
        cur = start
        prev_dir = None
        while True:
            nexts = out[cur]
            if len(nexts) == 1 or prev_dir is None:
                nxt = nexts[0]
            else:
                # turn right first (keeps diagonal-touching pixels as separate loops)
                def turn(n):
                    d = (n[0] - cur[0], n[1] - cur[1])
                    return prev_dir[0] * d[1] - prev_dir[1] * d[0]   # <0 = right turn
                nxt = min(nexts, key=turn)
            nexts.remove(nxt)
            if not nexts:
                del out[cur]
            prev_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
            cur = nxt
            if cur == start and (start not in out):
                break
            if cur == start:
                break
            pts.append(cur)
        # drop collinear points
        simp = []
        n = len(pts)
        for k in range(n):
            p0, p1, p2 = pts[k - 1], pts[k], pts[(k + 1) % n]
            if (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0]) != 0:
                simp.append(p1)
        contours.append(simp)
    return contours


def make_glyph(font, name, cells, adv):
    pen = TTGlyphPen(None)
    for c in trace(cells):
        pen.moveTo((c[0][0] * U, c[0][1] * U))
        for p in c[1:]:
            pen.lineTo((p[0] * U, p[1] * U))
        pen.closePath()
    g = pen.glyph()
    font['glyf'][name] = g
    g.recalcBounds(font['glyf'])
    lsb = g.xMin if cells else 0
    font['hmtx'][name] = (adv, lsb)
    if name not in font.getGlyphOrder():
        font.getGlyphOrder().append(name)


def set_name(font, nid, text):
    font['name'].setName(text, nid, 3, 1, 0x409)


def main():
    font = TTFont(SRC)
    gs = font.getGlyphSet()
    order = list(font.getGlyphOrder())
    hmtx = font['hmtx']

    # 1. draw the new glyphs
    built = {}
    plan = []
    for L in LETTERS:
        for form, (gn, pcp, base, add, rem, adv) in L['forms'].items():
            cells = set(pixels(gs, base)) if base else set()
            for c in rem:
                if c not in cells:
                    raise SystemExit('%s: pixel %r to remove is not in %s' % (gn, c, base))
                cells.discard(c)
            for c in add:
                if c in cells:
                    raise SystemExit('%s: pixel %r already set in %s' % (gn, c, base))
                cells.add(c)
            plan.append((gn, cells, adv if adv is not None else hmtx[base][0]))
            built.setdefault(L['cp'], {})[form] = (gn, pcp)
    # Farsi yeh copy
    plan.append(('u06CC', pixels(gs, 'u0649'), hmtx['u0649'][0]))

    glyph_order = order + [p[0] for p in plan]
    font.setGlyphOrder(glyph_order)
    font['glyf'].glyphOrder = glyph_order
    for gn, cells, adv in plan:
        make_glyph(font, gn, cells, adv)

    # 2. cmap
    for t in font['cmap'].tables:
        if not t.isUnicode():
            continue
        for cp, forms in built.items():
            t.cmap[cp] = forms['isol'][0]
            for form, (gn, pcp) in forms.items():
                if pcp is not None and (t.format != 4 or pcp <= 0xFFFF):
                    t.cmap[pcp] = gn
        t.cmap[0x06CC] = 'u06CC'

    # 3. GSUB: find the init/medi/fina lookups through the feature list
    gsub = font['GSUB'].table
    feat_lookups = {}
    for fr in gsub.FeatureList.FeatureRecord:
        feat_lookups.setdefault(fr.FeatureTag, []).extend(fr.Feature.LookupListIndex)
    lk = gsub.LookupList.Lookup

    def single(tag):
        idx = [i for i in feat_lookups[tag] if lk[i].LookupType == 1]
        assert len(idx) == 1, (tag, idx)
        return lk[idx[0]].SubTable[0].mapping

    init_m, medi_m, fina_m = single('init'), single('medi'), single('fina')
    for cp, forms in built.items():
        iso = forms['isol'][0]
        if 'fina' in forms:
            fina_m[iso] = forms['fina'][0]
        if 'init' in forms:
            init_m[iso] = forms['init'][0]
        if 'medi' in forms:
            medi_m[iso] = forms['medi'][0]
    # U+0649 (Uyghur/Kazakh/Kirghiz i): dotless in initial and medial forms
    init_m['u0649'] = 'uFBE8'
    medi_m['u0649'] = 'uFBE9'
    # U+06CC (Persian yeh) keeps the dotted joined forms the source gave U+0649
    init_m['u06CC'] = 'uFBFE'
    medi_m['u06CC'] = 'uFBFF'
    fina_m['u06CC'] = 'uFBFD'
    # ccmp: yeh + hamza mark -> yeh with hamza, for the Persian yeh too
    for i in feat_lookups['ccmp']:
        if lk[i].LookupType == 4:
            ligs = lk[i].SubTable[0].ligatures
            if 'u0649' in ligs and 'u06CC' not in ligs:
                lig = otTables.Ligature()
                lig.Component = ['u0654']
                lig.CompCount = 2
                lig.LigGlyph = 'u0626'
                ligs['u06CC'] = [lig]

    # 4. GDEF: all new glyphs are base glyphs
    cd = font['GDEF'].table.GlyphClassDef.classDefs
    for gn, _, _ in plan:
        cd[gn] = 1

    # 5. GPOS: letters that tuck under a preceding reh/zain get the same 1px kern
    for lookup in font['GPOS'].table.LookupList.Lookup:
        for st in lookup.SubTable:
            if getattr(st, 'Format', None) == 2 and 'u0631' in st.Coverage.glyphs:
                for gn in KERN_AFTER_REH:
                    st.ClassDef2.classDefs[gn] = 1

    # 6. metrics: one more pixel of descent
    font['hhea'].descent = -DESCENT
    font['OS/2'].sTypoDescender = -DESCENT
    font['OS/2'].usWinDescent = DESCENT

    # 7. names (no Reserved Font Name in unixel's OFL.txt; renamed anyway so the two
    # fonts can be installed side by side)
    set_name(font, 1, FAMILY)
    set_name(font, 2, 'Regular')
    set_name(font, 3, '%s:Version %s' % (FAMILY.replace(' ', ''), VERSION))
    set_name(font, 4, FAMILY)
    set_name(font, 5, 'Version %s' % VERSION)
    set_name(font, 6, PS_NAME)
    set_name(font, 10, MOD_NOTE)
    font['head'].fontRevision = float(VERSION)

    # 8. integer pixel size: unixel's 2000 units/em with 160-unit pixels lands on 12.5 px.
    # 1920 = 12 x 160, so the font is pixel-exact at 12, 24, 36 (outlines unchanged).
    font['head'].unitsPerEm = 12 * U

    font.save(OUT)
    print('wrote', OUT, len(font.getGlyphOrder()), 'glyphs')


if __name__ == '__main__':
    main()
