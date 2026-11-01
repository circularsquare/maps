"""Build asia1mpixel-Regular.ttf: one pixel font for the asia1m map labels.

    C:\\Python39\\python.exe build.py

Sources (read only):
  m5x7 (Daniel Linssen, CC0)                    Latin, digits, punctuation   1 px = 64 of 1024 units (16 px em)
  Ark Pixel 12px Mono zh_cn (TakWolf, OFL)      Chinese                      1 px = 100 of 1200 units (12 px em)
  Fusion Pixel 12px Mono zh_hans (TakWolf, OFL) Chinese characters Ark's zh_cn lacks (sources/), same grid
  Pixeloid Sans (GGBotNet, OFL)                 Cyrillic (sources/)          1 px = 204 of 1836 units (9 px em);
     same caps 7 / x-height 5 / descender 2 as m5x7
  unixel Uyghur (../unixel-Uyghur-Regular.ttf,  Arabic script, with GSUB/GPOS 1 px = 160 of 1920 units (12 px em)
     built by ../build_font.py from unixel, OFL)

Output: one grid, 1 px = 160 units, 1920 units per em, so at font size 12 every design pixel
is one screen pixel (and 2x2 at 24). Every source glyph is rendered 1-bit at its native pixel
size with FreeType and rebuilt as square-pixel contours; nothing is resampled. After saving,
the font is rendered again and compared with the pixels it was built from.

To add letters later, add a table entry:
  LATIN_ART             char -> (top row, pixel art)      drawn on the m5x7 grid
  COMPOSE               char -> (base char, extra pixels, x shift)
  COPY                  char -> source char               (same glyph as an existing one)
  CYRILLIC              the code point range taken from Pixeloid Sans
  ARABIC                one dict per letter: joining type and, per form, (base glyph, add, remove)
Pixels are (column, row), row 0 is the row sitting on the baseline, rows go up.
"""
import os
import sys

import freetype
from fontTools.ttLib import TTFont
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables import otTables
from fontTools import subset

HERE = os.path.dirname(os.path.abspath(__file__))
WINFONTS = os.path.expandvars(r'%LOCALAPPDATA%\Microsoft\Windows\Fonts')
SRC_LATIN = os.path.join(WINFONTS, 'm5x7.ttf')
SRC_CJK = os.path.join(WINFONTS, 'ark-pixel-12px-monospaced-zh_cn.ttf')
SRC_CJK_FALLBACK = os.path.join(HERE, 'sources', 'fusion-pixel-12px-monospaced-zh_hans.ttf')
SRC_ARAB = os.path.join(HERE, '..', 'unixel-Uyghur-Regular.ttf')
SRC_CYR = os.path.join(HERE, 'sources', 'PixeloidSans.ttf')
CYR_PPEM = 9       # Pixeloid Sans: 1 px = 204 of 1836 units
CYRILLIC = (0x0400, 0x052F)
OUT = os.path.join(HERE, 'asia1mpixel-Regular.ttf')

U = 160            # font units per pixel
UPM = 12 * U       # 12 px per em
FAMILY = 'asia1mpixel'
VERSION = '1.000'
ASCENT_PX = 12     # line box: 12 px above the baseline, 7 below (19 px line)
DESCENT_PX = 7


def art(top, text):
    rows = [r for r in text.strip('\n').split('\n')]
    return {(x, top - i) for i, r in enumerate(rows) for x, ch in enumerate(r.strip()) if ch == '#'}


# =======================================================================================
# Latin additions on the m5x7 grid (m5x7: caps 7 px, x-height 5, descender 2, 1 px gap)
CARON_CAP = [(1, 9), (3, 9), (2, 8)]     # m5x7 draws caron and breve the same (Č, Ğ)
CARON_LOW = [(1, 7), (3, 7), (2, 6)]
MACRON_CAP = [(1, 8), (2, 8), (3, 8)]
MACRON_LOW = [(1, 6), (2, 6), (3, 6)]
ACUTE_CAP = [(3, 9), (2, 8)]
ACUTE_LOW = [(3, 7), (2, 6)]
# tone marks over ü (its dieresis is on row 6)
MACRON_UU = [(1, 8), (2, 8), (3, 8)]
ACUTE_UU = [(3, 9), (2, 8)]
CARON_UU = [(1, 9), (3, 9), (2, 8)]
GRAVE_UU = [(1, 9), (2, 8)]

LATIN_ART = {
    'Ə': (6, """
.###.
#...#
....#
#####
#...#
#...#
.###."""),
    'ə': (4, """
.###.
....#
#####
#...#
.###."""),
    '‘': (6, """
.#
#.
#."""),
    '’': (6, """
.#
.#
#."""),
    '“': (6, """
.#.#
#.#.
#.#."""),
    '”': (6, """
.#.#
.#.#
#.#."""),
    '–': (3, "######"),
    '—': (3, "#########"),
    '…': (0, "#.#.#"),
    '•': (3, """
##
##"""),
}

# char -> (base char, extra pixels, x shift of the base)
COMPOSE = {
    # pinyin
    'Ǎ': ('A', CARON_CAP, 0), 'ǎ': ('a', CARON_LOW, 0),
    'Ǐ': ('I', CARON_CAP, 0), 'ǐ': ('ı', [(0, 7), (2, 7), (1, 6)], 0),
    'Ǒ': ('O', CARON_CAP, 0), 'ǒ': ('o', CARON_LOW, 0),
    'Ǔ': ('U', CARON_CAP, 0), 'ǔ': ('u', CARON_LOW, 0),
    'ǖ': ('ü', MACRON_UU, 0), 'ǘ': ('ü', ACUTE_UU, 0),
    'ǚ': ('ü', CARON_UU, 0), 'ǜ': ('ü', GRAVE_UU, 0),
    'Ǵ': ('G', ACUTE_CAP, 0), 'ǵ': ('g', ACUTE_LOW, 0),
    # Tajik Ӯ ӯ, which Pixeloid Sans lacks: its own У у plus the macron it puts on Ӣ ӣ
    'Ӯ': ('У', [(1, 8), (2, 8), (3, 8)], 0),
    # lowercase у is m5x7's y (below), so its marked forms use m5x7's y and marks
    'ӯ': ('y', MACRON_LOW, 0), 'ў': ('y', CARON_LOW, 0),   # m5x7's breve = its caron (ğ)
}

# letters that look the same as an existing glyph
COPY = {
    'ʻ': '‘', 'ʼ': '’',
    # Cyrillic lowercase taken from m5x7's Latin twins instead of Pixeloid (her choice)
    'а': 'a', 'е': 'e', 'р': 'p', 'с': 'c', 'у': 'y', 'ё': 'ë',
}
ZERO_WIDTH = [0x200B, 0x200D, 0x2060, 0xFEFF]

# Chinese characters neither Ark nor Fusion has, spliced from parts of ones they do have:
# char -> list of (source char, columns x0..x1, rows y0..y1), plus pixels to add
SPLICE = {
    '颍': ([('潁', 0, 4, -2, 9), ('颖', 5, 10, -2, 9)], []),        # Ying river
    '枞': ([('樅', 0, 3, -2, 9), ('纵', 5, 10, -2, 9)], []),        # Zongyang
    '鳌': ([('鰲', 0, 10, -1, 9)], [(x, -2) for x in range(11)]),  # 鰲 with 鱼's bottom bar
}

# =======================================================================================
# Arabic additions on unixel's grid. Marks as pixel lists:


def TAH(x, y):        # small tah, as on ڈ ڑ in unixel
    return [(x, y + 3), (x, y + 2), (x + 1, y + 2), (x, y + 1), (x + 2, y + 1), (x, y), (x + 1, y)]


def RING(x, y):
    return [(x + 1, y + 2), (x, y + 1), (x + 2, y + 1), (x + 1, y)]


def V(x, y):          # small v
    return [(x, y + 1), (x + 2, y + 1), (x + 1, y)]


def UP3(x, y):        # three dots, point up
    return [(x + 1, y + 2), (x, y), (x + 2, y)]


def DOWN3(x, y):      # three dots, point down
    return [(x, y + 2), (x + 2, y + 2), (x + 1, y)]


def FOUR(x, y):
    return [(x, y), (x + 2, y), (x, y + 2), (x + 2, y + 2)]


def HAMZA(x, y):      # the hamza unixel draws on ئ
    return [(x + 1, y + 3), (x + 2, y + 3), (x, y + 2), (x + 1, y + 1), (x + 2, y + 1), (x, y)]


TEH_I = [(3, 5), (5, 5)]          # teh's dots, isolated/final
TEH_J = [(0, 6), (2, 6)]          # teh's dots, initial/medial
CHE_I = [(3, 0), (5, 0), (4, -2)]  # tcheh's dots, isolated/final
CHE_J = [(1, -2), (3, -2), (2, -4)]

HEH_GOAL_FINA = (2, """
..##...
.#..#..
#...###""")
HEH_GOAL_INIT = (2, """
...#
...#
####
.#.#
..#.""")
HEH_GOAL_MEDI = (1, """
...#..
######
..#.#.
...#..""")
SWASH_KAF_ISOL = (4, """
.......###
.....##...
...##.....
.##.......
##########""")
SWASH_KAF_FINA = (4, """
.......###
.....##...
...##.....
.##.......
############""")
SWASH_KAF_INIT = (4, """
......##
....##..
..##....
.#......
#######.""")
SWASH_KAF_MEDI = (4, """
......##
....##..
..##....
.#......
#########""")


def L(cp, join, lang, **forms):
    return dict(cp=cp, join=join, lang=lang, forms=forms)


# per form: (base glyph or None, pixels to add, pixels to remove[, advance px, art])
ARABIC = [
    # ---- Urdu (also Punjabi Shahmukhi, Balochi)
    L(0x0679, 'D', 'Urdu',
      isol=('u062A', TAH(3, 5), TEH_I), fina=('uFE96', TAH(3, 5), TEH_I),
      init=('uFE97', TAH(0, 5), TEH_J), medi=('uFE98', TAH(0, 5), TEH_J)),
    L(0x06BA, 'D', 'Urdu',                       # noon ghunna: add the joined forms
      isol='u06BA', fina='uFB9F',
      init=('uFEE7', [], [(0, 6)]), medi=('uFEE8', [], [(0, 6)])),
    L(0x06C1, 'D', 'Urdu',                       # heh goal
      isol=('u0647', [], []),                  # as ه; without its top tick it reads as a ring
      fina=(None, [], [], 6, HEH_GOAL_FINA),
      init=(None, [], [], 4, HEH_GOAL_INIT),
      medi=(None, [], [], 5, HEH_GOAL_MEDI)),
    L(0x06C2, 'D', 'Urdu',                       # heh goal with hamza
      isol=('u06C1', HAMZA(1, 5), []), fina=('u06C1.fina', HAMZA(1, 4), []),
      init=('u06C1.init', HAMZA(1, 4), []), medi=('u06C1.medi', HAMZA(2, 3), [])),
    # ---- Pashto
    L(0x067C, 'D', 'Pashto',
      isol=('u062A', RING(3, -4), []), fina=('uFE96', RING(3, -4), []),
      init=('uFE97', RING(0, -4), []), medi=('uFE98', RING(0, -4), [])),
    L(0x0681, 'D', 'Pashto',
      isol=('u062D', HAMZA(3, 5), []), fina=('uFEA2', HAMZA(3, 5), []),
      init=('uFEA3', HAMZA(1, 5), []), medi=('uFEA4', HAMZA(1, 5), [])),
    L(0x0685, 'D', 'Pashto',
      isol=('u062D', UP3(2, 5), []), fina=('uFEA2', UP3(2, 5), []),
      init=('uFEA3', UP3(1, 5), []), medi=('uFEA4', UP3(1, 5), [])),
    L(0x0689, 'R', 'Pashto',
      isol=('u062F', RING(2, -4), []), fina=('uFEAA', RING(2, -4), [])),
    L(0x0693, 'R', 'Pashto',
      isol=('u0631', RING(1, -6), []), fina=('uFEAE', RING(1, -6), [])),
    L(0x0696, 'R', 'Pashto',
      isol=('u0631', [(3, 5), (2, -5)], []), fina=('uFEAE', [(3, 5), (2, -5)], [])),
    L(0x069A, 'D', 'Pashto',
      isol=('u0633', [(8, 5), (9, -2)], []), fina=('uFEB2', [(8, 5), (9, -2)], []),
      init=('uFEB3', [(3, 5), (3, -2)], []), medi=('uFEB4', [(3, 5), (3, -2)], [])),
    L(0x06AB, 'D', 'Pashto',
      isol=('u06A9', RING(4, -4), []), fina=('uFB8F', RING(4, -4), []),
      init=('uFB90', RING(1, -4), []), medi=('uFB91', RING(1, -4), [])),
    L(0x06BC, 'D', 'Pashto',
      isol=('u0646', RING(2, -7), []), fina=('uFEE6', RING(2, -7), []),
      init=('uFEE7', RING(0, -4), []), medi=('uFEE8', RING(0, -4), [])),
    L(0x06CD, 'R', 'Pashto',
      isol=('u0649', [(7, 4), (7, 5)], []), fina=('uFBFD', [(6, 3), (6, 4)], [])),
    # ---- Sindhi
    L(0x067A, 'D', 'Sindhi',
      isol=('u062A', [(4, 5), (4, 7)], TEH_I), fina=('uFE96', [(4, 5), (4, 7)], TEH_I),
      init=('uFE97', [(1, 6), (1, 8)], TEH_J), medi=('uFE98', [(1, 6), (1, 8)], TEH_J)),
    L(0x067B, 'D', 'Sindhi',
      isol=('u0628', [(5, -2), (5, -4)], [(5, -3)]), fina=('uFE90', [(5, -2), (5, -4)], [(5, -3)]),
      init=('uFE91', [(0, -2), (0, -4)], [(0, -3)]), medi=('uFE92', [(0, -2), (0, -4)], [(0, -3)])),
    L(0x067D, 'D', 'Sindhi',
      isol=('u062A', DOWN3(3, 5), TEH_I), fina=('uFE96', DOWN3(3, 5), TEH_I),
      init=('uFE97', DOWN3(0, 6), TEH_J), medi=('uFE98', DOWN3(0, 6), TEH_J)),
    L(0x067F, 'D', 'Sindhi',
      isol=('u062A', FOUR(3, 5), TEH_I), fina=('uFE96', FOUR(3, 5), TEH_I),
      init=('uFE97', FOUR(0, 6), TEH_J), medi=('uFE98', FOUR(0, 6), TEH_J)),
    L(0x0680, 'D', 'Sindhi',
      isol=('u0628', FOUR(4, -4), [(5, -3)]), fina=('uFE90', FOUR(4, -4), [(5, -3)]),
      init=('uFE91', FOUR(0, -4), [(0, -3)]), medi=('uFE92', FOUR(0, -4), [(0, -3)])),
    L(0x0683, 'D', 'Sindhi',
      isol=('u0686', [(3, -1), (5, -1)], CHE_I), fina=('uFB7B', [(3, -1), (5, -1)], CHE_I),
      init=('uFB7C', [], [(2, -4)]), medi=('uFB7D', [], [(2, -4)])),
    L(0x0684, 'D', 'Sindhi',
      isol=('u0686', [(4, 0), (4, -2)], CHE_I), fina=('uFB7B', [(4, 0), (4, -2)], CHE_I),
      init=('uFB7C', [(2, -2), (2, -4)], CHE_J), medi=('uFB7D', [(2, -2), (2, -4)], CHE_J)),
    L(0x0687, 'D', 'Sindhi',
      isol=('u0686', FOUR(3, -2), CHE_I), fina=('uFB7B', FOUR(3, -2), CHE_I),
      init=('uFB7C', FOUR(1, -4), CHE_J), medi=('uFB7D', FOUR(1, -4), CHE_J)),
    L(0x068A, 'R', 'Sindhi', isol=('u062F', [(3, -2)], []), fina=('uFEAA', [(3, -2)], [])),
    L(0x068C, 'R', 'Sindhi',
      isol=('u062F', [(2, 6), (4, 6)], []), fina=('uFEAA', [(2, 6), (4, 6)], [])),
    L(0x068D, 'R', 'Sindhi',
      isol=('u062F', [(2, -2), (4, -2)], []), fina=('uFEAA', [(2, -2), (4, -2)], [])),
    L(0x068E, 'R', 'Sindhi', isol=('u062F', UP3(2, 6), []), fina=('uFEAA', UP3(2, 6), [])),
    L(0x068F, 'R', 'Sindhi', isol=('u062F', FOUR(2, 6), []), fina=('uFEAA', FOUR(2, 6), [])),
    L(0x0699, 'R', 'Sindhi', isol=('u0631', FOUR(2, 5), []), fina=('uFEAE', FOUR(2, 5), [])),
    L(0x06A6, 'D', 'Sindhi',
      isol=('u0641', FOUR(5, 6), [(6, 7)]), fina=('uFED2', FOUR(5, 6), [(6, 7)]),
      init=('uFED3', FOUR(1, 6), [(2, 7)]), medi=('uFED4', FOUR(1, 6), [(2, 7)])),
    L(0x06AA, 'D', 'Sindhi',                     # swash kaf
      isol=(None, [], [], 11, SWASH_KAF_ISOL), fina=(None, [], [], 11, SWASH_KAF_FINA),
      init=(None, [], [], 8, SWASH_KAF_INIT), medi=(None, [], [], 8, SWASH_KAF_MEDI)),
    L(0x06B1, 'D', 'Sindhi',
      isol=('u06AF', [(4, 10), (6, 10)], []), fina=('uFB93', [(4, 10), (6, 10)], []),
      init=('uFB94', [(1, 10), (3, 10)], []), medi=('uFB95', [(1, 10), (3, 10)], [])),
    L(0x06B3, 'D', 'Sindhi',
      isol=('u06AF', [(4, -2), (4, -4)], []), fina=('uFB93', [(4, -2), (4, -4)], []),
      init=('uFB94', [(2, -2), (2, -4)], []), medi=('uFB95', [(2, -2), (2, -4)], [])),
    L(0x06BB, 'D', 'Sindhi',
      isol=('u06BA', TAH(3, 4), []), fina=('uFB9F', TAH(3, 4), []),
      init=('uFEE7', TAH(0, 5), [(0, 6)]), medi=('uFEE8', TAH(0, 5), [(0, 6)])),
    # ---- Kurdish Sorani (ۆ ە ھ are already there)
    L(0x0695, 'R', 'Sorani', isol=('u0631', V(1, -6), []), fina=('uFEAE', V(1, -6), [])),
    L(0x06A4, 'D', 'Sorani',
      isol=('u0641', UP3(5, 6), [(6, 7)]), fina=('uFED2', UP3(5, 6), [(6, 7)]),
      init=('uFED3', UP3(1, 6), [(2, 7)]), medi=('uFED4', UP3(1, 6), [(2, 7)])),
    L(0x06B5, 'D', 'Sorani',
      isol=('u0644', V(4, 9), []), fina=('uFEDE', V(4, 9), []),
      init=('uFEDF', V(0, 9), []), medi=('uFEE0', V(0, 9), [])),
    L(0x06CE, 'D', 'Sorani',
      isol=('u0649', V(5, 5), []), fina=('uFBFD', V(5, 4), []),
      init=('uFBE8', V(1, 5), []), medi=('uFBE9', V(1, 5), [])),
]
# extra glyphs the teh/noon contextual alternates should treat as tall followers
CALT_LOOKAHEAD_EXTRA = ['u0699', 'u0699.fina']
# ccmp compositions: base + combining hamza above
CCMP_HAMZA = {'u06C1': 'u06C2', 'u06D2': 'u06D3'}


def fname(cp, form):
    return 'u%04X' % cp if form == 'isol' else 'u%04X.%s' % (cp, form)


# =======================================================================================
# CJK selection from Ark Pixel: Chinese only.
def cjk_wanted(cp):
    if cp < 0x2000:
        return False                       # Latin/Greek/Cyrillic come from m5x7 or are drawn
    excluded = [
        (0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xAC00, 0xD7FF),  # Hangul
        (0x3200, 0x321E), (0x3260, 0x327F), (0xFFA0, 0xFFDC),
        (0x3040, 0x30FA), (0x30FC, 0x30FF), (0x31F0, 0x31FF), (0x32D0, 0x32FE),  # kana
        (0x3300, 0x3357), (0xFF65, 0xFF9F), (0x1B000, 0x1B16F),
        (0xE000, 0xF8FF), (0xF0000, 0x10FFFF),                                   # private use
    ]
    return not any(a <= cp <= b for a, b in excluded)


# =======================================================================================
def raster(path, ppem):
    """FreeType 1-bit render of every glyph: {glyph name: (cells, advance px)}."""
    tt = TTFont(path)
    unit = tt['head'].unitsPerEm / ppem
    hmtx = tt['hmtx']
    face = freetype.Face(path)
    face.set_char_size(int(round(ppem * 64)))
    flags = freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_MONO | \
        freetype.FT_LOAD_NO_HINTING | freetype.FT_LOAD_NO_BITMAP
    out = {}
    bad_adv = []
    for gid, gn in enumerate(tt.getGlyphOrder()):
        face.load_glyph(gid, flags)
        g = face.glyph
        bm = g.bitmap
        cells = set()
        buf = bm.buffer
        for y in range(bm.rows):
            for x in range(bm.width):
                if buf[y * bm.pitch + x // 8] & (0x80 >> (x % 8)):
                    cells.add((g.bitmap_left + x, g.bitmap_top - 1 - y))
        adv = hmtx[gn][0] / unit
        if abs(adv - round(adv)) > 1e-6:
            bad_adv.append(gn)
        out[gn] = (cells, int(round(adv)))
    if bad_adv:
        print('  %s: %d advances off the pixel grid, rounded (%s...)' %
              (os.path.basename(path), len(bad_adv), bad_adv[:5]))
    return tt, out


def check_grid(tt, unit, names):
    """Count glyphs whose outline points are not on the source's pixel grid."""
    glyf = tt['glyf']
    off = 0
    for gn in names:
        g = glyf[gn]
        if g.numberOfContours > 0:
            coords, _, _ = g.getCoordinates(glyf)
            if any(c[0] % unit or c[1] % unit for c in coords):
                off += 1
    return off


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
                    return prev_dir[0] * d[1] - prev_dir[1] * d[0]
                nxt = min(nexts, key=turn)
            nexts.remove(nxt)
            if not nexts:
                del out[cur]
            prev_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
            cur = nxt
            if cur == start:
                break
            pts.append(cur)
        simp = []
        n = len(pts)
        for k in range(n):
            p0, p1, p2 = pts[k - 1], pts[k], pts[(k + 1) % n]
            if (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0]) != 0:
                simp.append(p1)
        contours.append(simp)
    return contours


def tt_glyph(cells):
    pen = TTGlyphPen(None)
    for c in trace(cells):
        pen.moveTo((c[0][0] * U, c[0][1] * U))
        for p in c[1:]:
            pen.lineTo((p[0] * U, p[1] * U))
        pen.closePath()
    return pen.glyph()


def m5x7_adv(cells):
    return (max(x for x, _ in cells) + 2) if cells else 0


# =======================================================================================
def main():
    glyphs = {}        # name -> (cells, adv px)
    cmap = {}          # cp -> name
    order = ['.notdef']
    box = {(x, y) for x in range(5) for y in range(7) if x in (0, 4) or y in (0, 6)}
    glyphs['.notdef'] = (box, 6)

    def add(name, cells, adv, cp=None):
        if name not in glyphs:
            order.append(name)
        glyphs[name] = (set(cells), adv)
        if cp is not None:
            cmap[cp] = name

    def uname(cp):
        return 'uni%04X' % cp if cp <= 0xFFFF else 'u%05X' % cp

    # ---- 1. Latin from m5x7 -----------------------------------------------------------
    lat_tt, lat = raster(SRC_LATIN, 16)
    lat_cmap = lat_tt.getBestCmap()
    print('m5x7: %d glyphs off the 64-unit grid' % check_grid(lat_tt, 64, set(lat_cmap.values())))
    for cp, gn in sorted(lat_cmap.items()):
        if cp == 0x3000:
            continue                       # ideographic space comes from Ark Pixel
        cells, adv = lat[gn]
        add(uname(cp), cells, adv, cp)
    n_m5x7 = len(cmap)

    def cells_of(ch):
        return glyphs[cmap[ord(ch)]]

    drawn = {}
    for ch, (top, text) in LATIN_ART.items():
        cells = art(top, text)
        add(uname(ord(ch)), cells, m5x7_adv(cells), ord(ch))
        drawn[ch] = 'drawn'
    # Cyrillic from Pixeloid Sans (same baseline, caps, x-height and descender as m5x7)
    cyr_tt, cyr = raster(SRC_CYR, CYR_PPEM)
    cyr_names = set()
    for cp, gn in sorted(cyr_tt.getBestCmap().items()):
        if CYRILLIC[0] <= cp <= CYRILLIC[1] and cp not in cmap:
            cells, adv = cyr[gn]
            add(uname(cp), cells, adv, cp)
            cyr_names.add(gn)
    cyr_unit = cyr_tt['head'].unitsPerEm // CYR_PPEM
    print('Pixeloid Sans: %d Cyrillic glyphs, %d off the %d-unit grid' %
          (len(cyr_names), check_grid(cyr_tt, cyr_unit, cyr_names), cyr_unit))
    for ch, src in COPY.items():
        cells, adv = cells_of(src)
        add(uname(ord(ch)), cells, adv, ord(ch))
    pending = dict(COMPOSE)
    while pending:
        progress = False
        for ch, (base, extra, dx) in list(pending.items()):
            if ord(base) not in cmap:
                continue
            bc, _ = cells_of(base)
            cells = {(x + dx, y) for x, y in bc}
            for p in extra:
                if p in cells:
                    raise SystemExit('%s: pixel %r already set in %s' % (ch, p, base))
                cells.add(p)
            add(uname(ord(ch)), cells, m5x7_adv(cells), ord(ch))
            drawn[ch] = 'composed'
            del pending[ch]
            progress = True
        if not progress:
            raise SystemExit('COMPOSE: missing bases for %s' % ''.join(pending))
    for cp in ZERO_WIDTH:
        add(uname(cp), set(), 0, cp)
    n_latin_extra = sum(1 for c in list(LATIN_ART) + list(COMPOSE) + list(COPY) if ord(c) < 0x400 or ord(c) > 0x52F)
    n_cyr = sum(1 for cp in cmap if 0x400 <= cp <= 0x52F)

    # ---- 2. Arabic from unixel Uyghur, plus new letters ------------------------------
    ar_tt, ar = raster(SRC_ARAB, 12)
    ar_cmap = ar_tt.getBestCmap()
    print('unixel Uyghur: %d glyphs off the 160-unit grid' %
          check_grid(ar_tt, 160, ar_tt.getGlyphOrder()))
    for gn in ar_tt.getGlyphOrder():
        if gn in ('.notdef', 'space'):
            continue
        cells, adv = ar[gn]
        if gn == 'u200C':
            cells, adv = set(), 0          # unixel draws ZWNJ visibly; it should not show
        add(gn, cells, adv)
    for cp, gn in ar_cmap.items():
        if cp in cmap or gn in ('.notdef', 'space'):
            continue                       # m5x7 wins for Latin, digits and punctuation
        cmap[cp] = gn
    built = {}
    new_arabic = []
    for Lt in ARABIC:
        cp = Lt['cp']
        for form, spec in Lt['forms'].items():
            if isinstance(spec, str):      # existing glyph
                built.setdefault(cp, {})[form] = spec
                continue
            base, plus, minus = spec[:3]
            name = fname(cp, form)
            if base is None:
                cells = art(*spec[4])
                adv = spec[3]
            else:
                cells, adv = glyphs[base]
                cells = set(cells)
            for p in minus:
                if p not in cells:
                    raise SystemExit('%s: pixel %r to remove is not in %s' % (name, p, base))
                cells.discard(p)
            for p in plus:
                if p in cells:
                    raise SystemExit('%s: pixel %r already set in %s' % (name, p, base))
                cells.add(p)
            if len(spec) > 3 and spec[3] is not None:
                adv = spec[3]
            add(name, cells, adv)
            new_arabic.append((name, base))
            built.setdefault(cp, {})[form] = name
        cmap[cp] = built[cp]['isol']

    gsub = ar_tt['GSUB'].table
    feat = {}
    for fr in gsub.FeatureList.FeatureRecord:
        feat.setdefault(fr.FeatureTag, []).extend(fr.Feature.LookupListIndex)
    lk = gsub.LookupList.Lookup
    gdef_extra = []

    def single(tag):
        idx = [i for i in feat[tag] if lk[i].LookupType == 1]
        assert len(idx) == 1, (tag, idx)
        return lk[idx[0]].SubTable[0].mapping
    maps = {t: single(t) for t in ('init', 'medi', 'fina')}
    for cp, forms in built.items():
        for form in ('init', 'medi', 'fina'):
            if form in forms:
                maps[form][forms['isol']] = forms[form]
    for i in feat['ccmp']:
        if lk[i].LookupType == 4:
            ligs = lk[i].SubTable[0].ligatures
            for b, out in CCMP_HAMZA.items():
                lig = otTables.Ligature()
                lig.Component = ['u0654']
                lig.CompCount = 2
                lig.LigGlyph = out
                ligs.setdefault(b, []).append(lig)
    # calt: teh/noon alternates before tall letters. unixel's alternate is the joined form
    # moved 1 px right with the gap on the baseline filled; new letters built on those
    # forms get the same treatment.
    calt_st = lk[feat['calt'][0]].SubTable[0]
    alt_map = lk[calt_st.SubstLookupRecord[0].LookupListIndex].SubTable[0].mapping
    inp = calt_st.InputCoverage[0].glyphs
    for name, base in list(new_arabic):
        if base in inp:
            cells, adv = glyphs[name]
            alt = name + '.alt'
            add(alt, {(x + 1, y) for x, y in cells} | {(0, 0)}, adv + 1)
            alt_map[name] = alt
            inp.append(name)
            gdef_extra.append(alt)
    la = calt_st.LookAheadCoverage[0].glyphs
    for name, base in new_arabic:
        if base in la and name not in la:
            la.append(name)
    for name in CALT_LOOKAHEAD_EXTRA:
        if name not in la:
            la.append(name)
    gdef = ar_tt['GDEF'].table
    for name in [n for n, _ in new_arabic] + gdef_extra:
        gdef.GlyphClassDef.classDefs[name] = 1
    # GPOS: the 1 px tuck after reh/zain
    gpos = ar_tt['GPOS'].table
    for lookup in gpos.LookupList.Lookup:
        for st in lookup.SubTable:
            if getattr(st, 'Format', None) == 2 and 'u0631' in st.Coverage.glyphs:
                c2 = st.ClassDef2.classDefs
                for name, base in new_arabic:
                    if base in st.Coverage.glyphs and name not in st.Coverage.glyphs:
                        st.Coverage.glyphs.append(name)
                    if base is not None and c2.get(base) == 1:
                        c2[name] = 1
                for name in ('u06AA', 'u06AA.init'):
                    c2[name] = 1
    n_arabic = sum(1 for cp in cmap if 0x600 <= cp <= 0x6FF or 0x750 <= cp <= 0x77F or
                   0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFF)

    # ---- 3. Chinese from Ark Pixel ------------------------------------------------------
    cjk_tt, cjk = raster(SRC_CJK, 12)
    cjk_cmap = cjk_tt.getBestCmap()
    by_src = {}
    n_cjk = 0
    for cp, gn in sorted(cjk_cmap.items()):
        if cp in cmap or not cjk_wanted(cp):
            continue
        if gn in by_src:
            cmap[cp] = by_src[gn]
        else:
            cells, adv = cjk[gn]
            name = uname(cp)
            add(name, cells, adv, cp)
            by_src[gn] = name
        n_cjk += 1
    print('Ark Pixel: %d glyphs off the 100-unit grid (of the used ones)' %
          check_grid(cjk_tt, 100, set(by_src)))
    # Fusion Pixel 12px (same author and grid) for the Chinese characters Ark's zh_cn lacks
    fu_tt, fu = raster(SRC_CJK_FALLBACK, 12)
    by_src = {}
    n_fusion = 0
    for cp, gn in sorted(fu_tt.getBestCmap().items()):
        if cp in cmap or not cjk_wanted(cp):
            continue
        if gn in by_src:
            cmap[cp] = by_src[gn]
        else:
            cells, adv = fu[gn]
            name = uname(cp)
            add(name, cells, adv, cp)
            by_src[gn] = name
        n_fusion += 1
    print('Fusion Pixel: %d glyphs off the 100-unit grid (of the used ones)' %
          check_grid(fu_tt, 100, set(by_src)))
    for ch, (pieces, extra) in SPLICE.items():
        if ord(ch) in cmap:
            continue                       # a source has it now; use that
        cells = set()
        for src, x0, x1, y0, y1 in pieces:
            sc, adv = glyphs[cmap[ord(src)]]
            cells |= {(x, y) for x, y in sc if x0 <= x <= x1 and y0 <= y <= y1}
        cells |= set(extra)
        add(uname(ord(ch)), cells, 12, ord(ch))
        n_fusion += 1

    # ---- 4. assemble --------------------------------------------------------------------
    print('assembling %d glyphs, %d code points' % (len(order), len(cmap)))
    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    ttg = {n: tt_glyph(glyphs[n][0]) for n in order}
    fb.setupGlyf(ttg)
    glyf = fb.font['glyf']
    metrics = {}
    ymax = ymin = 0
    for n in order:
        g = glyf[n]
        metrics[n] = (glyphs[n][1] * U, g.xMin if g.numberOfContours else 0)
        if g.numberOfContours:
            ymax, ymin = max(ymax, g.yMax), min(ymin, g.yMin)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ASCENT_PX * U, descent=-DESCENT_PX * U, lineGap=0)
    copyright_ = ('m5x7 by Daniel Linssen (CC0). '
                  'Copyright (c) 2021, TakWolf (https://takwolf.com), with Reserved Font Name '
                  '"Ark Pixel". Fusion Pixel Font copyright (c) 2022, TakWolf. Copyright 2023 The Rooyin Project Authors '
                  '(https://github.com/MDarvishi5124/unixel). '
                  'Cyrillic: Copyright 2020-2025 GGBotNet, with Reserved Font Name "Pixeloid". '
                  'Tajik U+04EE/04EF, extra Latin and extra Arabic letters made for asia1mpixel, 2026.')
    fb.setupNameTable({
        'copyright': copyright_,
        'familyName': FAMILY,
        'styleName': 'Regular',
        'uniqueFontIdentifier': '%s:Version %s' % (FAMILY, VERSION),
        'fullName': FAMILY,
        'version': 'Version %s' % VERSION,
        'psName': FAMILY + '-Regular',
        'description': ('Pixel font for the asia1m map: m5x7 Latin, Ark Pixel 12px Chinese and '
                        'unixel Uyghur Arabic script on one grid, 1 px = 1/12 em. '
                        'Use at 12 or 24 px with anti-aliasing off.'),
        'licenseDescription': ('This Font Software is licensed under the SIL Open Font '
                               'License, Version 1.1. This license is available with a FAQ '
                               'at: https://openfontlicense.org'),
        'licenseInfoURL': 'https://openfontlicense.org',
    })
    fb.setupOS2(
        version=4, usWeightClass=400, fsType=0, achVendID='NONE',
        sTypoAscender=ASCENT_PX * U, sTypoDescender=-DESCENT_PX * U, sTypoLineGap=0,
        usWinAscent=max(ymax, ASCENT_PX * U), usWinDescent=max(-ymin, DESCENT_PX * U),
        fsSelection=0x40 | 0x80,            # REGULAR, USE_TYPO_METRICS
        sxHeight=5 * U, sCapHeight=7 * U,
        ulCodePageRange1=(1 << 0) | (1 << 1) | (1 << 2) | (1 << 4) | (1 << 6) | (1 << 7) | (1 << 18),
        ulCodePageRange2=0,
    )
    fb.setupPost(keepGlyphNames=True)
    font = fb.font
    font['head'].fontRevision = float(VERSION)
    for tag in ('GSUB', 'GPOS', 'GDEF'):
        font[tag] = ar_tt[tag]

    # drop glyphs nothing reaches any more (unixel's Latin, its Latin kerning)
    opts = subset.Options()
    opts.layout_features = ['*']
    opts.layout_scripts = ['*']
    opts.name_IDs = ['*']
    opts.name_languages = ['*']
    opts.name_legacy = True
    opts.glyph_names = True
    opts.notdef_outline = True
    opts.notdef_glyph = True
    opts.hinting = False
    opts.prune_unicode_ranges = True
    opts.legacy_kern = False
    sub = subset.Subsetter(opts)
    sub.populate(unicodes=list(cmap))
    sub.subset(font)
    font['OS/2'].recalcUnicodeRanges(font)
    # coverage tables must be in glyph-id order (HarfBuzz binary-searches them); the ones
    # extended above are independent lists (chain-context format 3, class pair format 2)
    gid = font.getReverseGlyphMap()
    for lookup in font['GSUB'].table.LookupList.Lookup:
        for st in lookup.SubTable:
            if lookup.LookupType == 6 and st.Format == 3:
                for cov in st.BacktrackCoverage + st.InputCoverage + st.LookAheadCoverage:
                    cov.glyphs.sort(key=gid.get)
    for lookup in font['GPOS'].table.LookupList.Lookup:
        for st in lookup.SubTable:
            if lookup.LookupType == 2 and st.Format == 2:
                st.Coverage.glyphs.sort(key=gid.get)
    font.save(OUT)
    kept = font.getGlyphOrder()
    print('wrote %s: %d glyphs (%d bytes)' % (OUT, len(kept), os.path.getsize(OUT)))
    print('code points: m5x7 %d, drawn/composed/copied Latin %d, Cyrillic %d, Arabic script %d, '
          'Chinese: Ark Pixel %d + Fusion Pixel %d' %
          (n_m5x7, n_latin_extra, n_cyr, n_arabic, n_cjk, n_fusion))
    print('new Arabic glyphs: %d for %d letters' % (len(new_arabic), len(ARABIC)))
    print('outline bounds: %.1f px above, %.1f px below the baseline' % (ymax / U, -ymin / U))

    # ---- 5. self-check: FreeType 1-bit at 12 px reproduces every glyph's pixels ---------
    face = freetype.Face(OUT)
    face.set_char_size(12 * 64)
    flags = freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_MONO | freetype.FT_LOAD_NO_HINTING
    bad = []
    for gid, n in enumerate(kept):
        face.load_glyph(gid, flags)
        g = face.glyph
        bm = g.bitmap
        got = set()
        for y in range(bm.rows):
            for x in range(bm.width):
                if bm.buffer[y * bm.pitch + x // 8] & (0x80 >> (x % 8)):
                    got.add((g.bitmap_left + x, g.bitmap_top - 1 - y))
        want = glyphs[n][0] if n in glyphs else got
        if got != want or g.advance.x != glyphs.get(n, (0, g.advance.x // 64))[1] * 64:
            bad.append(n)
    print('pixel check at 12 px: %d of %d glyphs differ %s' % (len(bad), len(kept), bad[:10]))


if __name__ == '__main__':
    main()
