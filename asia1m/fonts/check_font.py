"""Check unixel-Uyghur-Regular.ttf and draw test_sheet.png.

    C:\\Python39\\python.exe check_font.py

1. opens the font in fontTools, PIL and FreeType;
2. shapes every uyghur/kazakh/kirghiz name in ../naming/local_native.csv plus sample
   words with HarfBuzz, and checks each letter got a real glyph (no .notdef) in the
   joining form its position needs (isol/init/medi/fina, from Unicode joining types);
3. checks FreeType's 1-bit rendering at 12 px/em matches the design pixels of every
   glyph this build added;
4. draws test_sheet.png: each name next to its Latin name at 1x and 4x, plus a table of
   the added letters in all their forms (tatweel ـ shows the joins).
"""
import csv
import os
import sys
import unicodedata

import freetype
import uharfbuzz as hb
from fontTools.ttLib import TTFont
from PIL import Image, ImageFont, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_font import pixels, LETTERS, U  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, 'unixel-Uyghur-Regular.ttf')
CSV = os.path.join(HERE, '..', 'naming', 'local_native.csv')
SHEET = os.path.join(HERE, 'test_sheet.png')

EXTRA = [
    ('Kucha', 'كۇچا'), ('Payzawat', 'پەيزاۋات'), ('Maralbexi', 'مارالبېشى'),
    ('Teklimakan', 'تەكلىماكان'), ('Urumchi', 'ئۈرۈمچى'), ('Shinjang', 'شىنجاڭ'),
    ('Heyt', 'ھېيت'), ('Sheher', 'شەھەر'), ('Mehelle', 'مەھەللە'), ('Olke', 'ئۆلكە'),
    ('Ile (Kazakh)', 'ٴىلە'), ('Tengri', 'تەڭرى'), ('Yengi', 'يېڭى'), ('Ghulja', 'غۇلجا'),
    ('Kirghiz ok', 'كۅل ۉ'), ('Iran (Persian ye)', 'ایران'),
]

# Unicode joining types for the Arabic block (enough for these languages)
R = set(range(0x0622, 0x0626)) | {0x0627, 0x0629} | set(range(0x062F, 0x0633)) | {0x0648} | \
    set(range(0x0671, 0x0674)) | set(range(0x0675, 0x0678)) | set(range(0x0688, 0x069A)) | \
    {0x06C0} | set(range(0x06C3, 0x06CC)) | {0x06CD, 0x06CF, 0x06D2, 0x06D3, 0x06D5}
NONJOIN = {0x0621, 0x0674}
TRANSP = set(range(0x064B, 0x0660)) | {0x0670} | set(range(0x06D6, 0x06EE))
JOINC = {0x0640, 0x200D}


def jtype(cp):
    if cp in TRANSP:
        return 'T'
    if cp in JOINC:
        return 'C'
    if cp in R:
        return 'R'
    if cp in NONJOIN:
        return 'U'
    if 0x0620 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
        if 0x0660 <= cp <= 0x066D or 0x06F0 <= cp <= 0x06F9 or cp in (0x060C, 0x061B, 0x061F, 0x06D4):
            return 'U'
        return 'D'
    return 'U'


def expected_forms(cps):
    types = [jtype(c) for c in cps]
    out = []
    for i, t in enumerate(types):
        if t in ('T', 'U', 'C'):
            out.append(None)
            continue
        p = next((types[k] for k in range(i - 1, -1, -1) if types[k] != 'T'), None)
        n = next((types[k] for k in range(i + 1, len(types)) if types[k] != 'T'), None)
        pj = p in ('D', 'C')
        nj = n in ('D', 'R', 'C') and t == 'D'
        if t == 'R':
            out.append('fina' if pj else 'isol')
        else:
            out.append({(1, 1): 'medi', (0, 1): 'init', (1, 0): 'fina', (0, 0): 'isol'}[(pj, nj)])
    return out


def main():
    tt = TTFont(FONT)
    cmap = tt.getBestCmap()
    order = tt.getGlyphOrder()
    gsub = tt['GSUB'].table
    lk = gsub.LookupList.Lookup
    feat = {}
    for fr in gsub.FeatureList.FeatureRecord:
        feat.setdefault(fr.FeatureTag, []).extend(fr.Feature.LookupListIndex)
    maps = {t: lk[[i for i in feat[t] if lk[i].LookupType == 1][0]].SubTable[0].mapping
            for t in ('init', 'medi', 'fina')}
    alt = {}   # contextual alternates (calt) back to their plain form
    for i in range(len(lk)):
        if lk[i].LookupType == 1 and i not in sum((feat[t] for t in ('init', 'medi', 'fina', 'locl')), []):
            for a, b in lk[i].SubTable[0].mapping.items():
                alt[b] = a

    ligs = set()   # ligature outputs (lam-alef, yeh+hamza) are fine wherever they appear
    for lookup in lk:
        if lookup.LookupType == 4:
            for lst in lookup.SubTable[0].ligatures.values():
                ligs.update(l.LigGlyph for l in lst)

    blob = hb.Blob.from_file_path(FONT)
    hbfont = hb.Font(hb.Face(blob))

    def shape(text):
        if not text:
            return []
        buf = hb.Buffer()
        buf.add_codepoints([ord(c) for c in text])
        buf.guess_segment_properties()
        hb.shape(hbfont, buf, {})
        return [(order[i.codepoint], i.cluster, p.x_advance, p.x_offset, p.y_offset)
                for i, p in zip(buf.glyph_infos, buf.glyph_positions)]

    samples = []
    for r in csv.DictReader(open(CSV, encoding='utf-8')):
        if r['lang'] in ('uyghur', 'kazakh', 'kirghiz'):
            # the font has no accented Latin, so drop accents from the labels
            latin = ''.join(c for c in unicodedata.normalize('NFKD', r['name'])
                            if not unicodedata.combining(c))
            samples.append((latin + ' (' + r['lang'][:2] + ')', r['native']))
    samples += EXTRA

    problems = []
    used_forms = {}
    for latin, text in samples:
        cps = [ord(c) for c in text]
        exp = expected_forms(cps)
        glyphs = shape(text)
        by_cluster = {}
        for g in glyphs:
            by_cluster.setdefault(g[1], []).append(g[0])
        for k, cp in enumerate(cps):
            if cp == 0x20:
                continue
            if k not in by_cluster:
                continue   # merged into a ligature cluster (lam-alef)
            gl = by_cluster[k]
            if '.notdef' in gl:
                problems.append('%s: .notdef for U+%04X' % (latin, cp))
                continue
            if exp[k] is None:
                continue
            want = cmap[cp] if exp[k] == 'isol' else maps[exp[k]].get(cmap[cp], cmap[cp])
            got = alt.get(gl[0], gl[0])
            if got != want and gl[0] not in ligs:
                problems.append('%s: U+%04X %s wants %s got %s' % (latin, cp, exp[k], want, gl[0]))
            used_forms.setdefault(cp, set()).add(exp[k])
    print('shaped %d names' % len(samples))
    for cp in sorted(used_forms):
        if cp >= 0x0600:
            print('  U+%04X %s forms used: %s' % (cp, chr(cp), ','.join(sorted(used_forms[cp]))))
    print('problems:', len(problems))
    for p in problems:
        print('  ', p)

    # FreeType 1-bit check of every added glyph
    gs = tt.getGlyphSet()
    face = freetype.Face(FONT)
    face.set_char_size(int(12 * 64))  # 12 px/em: one design pixel = one pixel
    new = [f[0] for L in LETTERS for f in L['forms'].values()] + ['u06CC', 'uFBE8', 'uFBE9']
    bad = 0
    for gn in new:
        face.load_glyph(order.index(gn), freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_MONO
                        | freetype.FT_LOAD_NO_HINTING)
        bm = face.glyph.bitmap
        left, top = face.glyph.bitmap_left, face.glyph.bitmap_top
        ft = set()
        for y in range(bm.rows):
            for x in range(bm.width):
                byte = bm.buffer[y * bm.pitch + x // 8]
                if byte & (0x80 >> (x % 8)):
                    ft.add((left + x, top - 1 - y))
        if ft != pixels(gs, gn):
            bad += 1
            print('  FreeType differs for', gn)
    print('FreeType 1-bit check: %d of %d added glyphs differ' % (bad, len(new)))

    # PIL opens it
    pf = ImageFont.truetype(FONT, 24)
    ImageDraw.Draw(Image.new('1', (200, 40))).text((0, 0), 'unixel Uyghur', font=pf, fill=1)
    print('PIL opened', pf.getname())

    # ---- test sheet -----------------------------------------------------------------
    cache = {}

    def gpix(gn):
        if gn not in cache:
            cache[gn] = pixels(gs, gn)
        return cache[gn]

    def draw(img, x, baseline, text):
        pen = 0
        for gn, cl, adv, xo, yo in shape(text):
            for (i, j) in gpix(gn):
                px, py = x + (pen + xo) // U + i, baseline - j - yo // U
                if 0 <= px < img.width and 0 <= py < img.height:
                    img.putpixel((px, py), 0)
            pen += adv
        return pen // U

    def width(text):
        return sum(g[2] for g in shape(text)) // U

    table = []
    for L in LETTERS:
        c = chr(L['cp'])
        if L['join'] == 'D':
            cells = [c, c + 'ـ', 'ـ' + c + 'ـ', 'ـ' + c]
        elif L['join'] == 'R':
            cells = [c, 'ـ' + c]
        else:
            cells = [c]
        table.append(('U+%04X' % L['cp'], '   '.join(reversed(cells))))
    table.append(('U+0649 i', '   '.join(reversed(['ى', 'ىـ', 'ـىـ', 'ـى']))))
    table.append(('U+064A y', '   '.join(reversed(['ي', 'يـ', 'ـيـ', 'ـي']))))
    table.append(('U+0643 k', '   '.join(reversed(['ك', 'كـ', 'ـكـ', 'ـك']))))
    table.append(('U+06D5 e', '   '.join(reversed(['ە', 'ـە']))))
    table.append(('U+0626', '   '.join(reversed(['ئ', 'ئـ', 'ـئـ', 'ـئ']))))

    rows = [('', '')] + table + [('', '')] + samples
    LH = 22
    lw = max(width(a) for a, _ in rows) + 8
    aw = max(width(b) for _, b in rows) + 4
    W, H = lw + aw + 8, LH * len(rows) + 8
    img = Image.new('1', (W, H), 1)
    for k, (a, b) in enumerate(rows):
        base = 4 + k * LH + 13
        draw(img, 4, base, a)
        draw(img, W - 4 - width(b), base, b)
    big = img.resize((W * 4, H * 4), Image.NEAREST)
    sheet = Image.new('1', (W * 4, H + 12 + H * 4), 1)
    sheet.paste(img, (0, 0))
    sheet.paste(big, (0, H + 12))
    sheet.save(SHEET)
    print('wrote', SHEET, sheet.size)


if __name__ == '__main__':
    main()
