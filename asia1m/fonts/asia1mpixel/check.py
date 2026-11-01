"""Check asia1mpixel-Regular.ttf and draw test_sheet.png.

    C:\\Python39\\python.exe check.py

1. shapes sample words of every script with HarfBuzz (plus every uyghur/kazakh/kirghiz name in
   ../../naming/local_native.csv and every Chinese name in ../../naming/all_names.csv) and
   checks no glyph is .notdef;
2. for Arabic script, checks every letter got the joining form its position needs
   (isol/init/medi/fina, from Unicode joining types);
3. draws test_sheet.png from FreeType's 1-bit rendering at 12 px: 1x on top, 4x below.
"""
import csv
import os
import sys

import freetype
import uharfbuzz as hb
from fontTools.ttLib import TTFont
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build import ARABIC, LATIN_ART, COMPOSE  # noqa: E402

FONT = os.path.join(HERE, 'asia1mpixel-Regular.ttf')
NAMING = os.path.join(HERE, '..', '..', 'naming')
SHEET = os.path.join(HERE, 'test_sheet.png')
sys.stdout.reconfigure(encoding='utf-8')

LATIN = [
    ('', 'The quick brown fox jumps over the lazy dog.'),
    ('', 'THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 0123456789'),
    ('Turkish', 'Şanlıurfa İstanbul Muğla Çorum Ağrı Gümüşhane'),
    ('Azerbaijani', 'Gəncə Şəki Lənkəran Mingəçevir Ərdəbil'),
    ('Uzbek', 'Oʻzbekiston Gʻijduvon Qarshi Toshkent Ma’no'),
    ('Turkmen', 'Türkmenistan Aşgabat Daşoguz Türkmenbaşy Žeň'),
    ('Kurmanji', 'Diyarbekir Wan Êlih Sêrt Şirnex Hekarî Mûş'),
    ('Kazakh Latin', 'Qazaqstan Öskemen Şymkent Ūlytau Ñ Äuliyeata'),
    ('Karakalpak', 'Nókis Qaraqalpaqstan Ǵárezsiz'),
    ('Pinyin', 'Xīnjiāng Ürümqi lǚ nǚ lǜ lǘ Guǎngzhōu Chóngqìng Hǎinán Ǎ Ǐ Ǒ Ǔ'),
    ('Accents', 'ü ö ş ç ğ ı İ ə Ə ä ñ é è ê á à â í ó ú ū ā ī ō ž č š ň ý'),
    ('Punctuation', '“Quote” ‘single’ a–b a—b wait… • (x) [y] {z} 1,2.3; 4:5 ?! # % & *'),
]
CYRILLIC = [
    ('Her labels', 'Даурия Сэлэнгэ Улаанбаатар Говь'),
    ('Russian', 'Москва Новосибирск Иркутск Хабаровск Якутия'),
    ('Russian', 'Забайкальский край, Бурятия, съезд, объём'),
    ('Russian', 'ЧИТА ЩЁЛКОВО ЭЛИСТА ЮЖНО-САХАЛИНСК ЯКУТСК ЖИГАНСК'),
    ('Pangram', 'Съешь же ещё этих мягких французских булок, да выпей чаю'),
    ('Pangram', 'СЪЕШЬ ЖЕ ЕЩЁ ЭТИХ МЯГКИХ ФРАНЦУЗСКИХ БУЛОК, ДА ВЫПЕЙ ЧАЮ'),
    ('Mongolian', 'Өвөрхангай Хөвсгөл Дорнод Сүхбаатар Баян-Өлгий'),
    ('Kazakh', 'Қазақстан Өскемен Ақтөбе Қарағанды Әулиеата Ұлытау'),
    ('Kazakh', 'Маңғыстау Іле жиһан ҚАЗАҚСТАН ҰЛЫТАУ ӘУЛИЕАТА'),
    ('Kirghiz', 'Кыргызстан Бишкек Ысык-Көл Нарын Жалал-Абад Таң'),
    ('Tajik', 'Тоҷикистон Душанбе Хуҷанд Кӯлоб Ғарм Қӯрғонтеппа Ҳисор Ӣ'),
    ('Uzbek', 'Ўзбекистон Тошкент Қарши Ғиждувон Ҳ ў'),
    ('Ukrainian', 'Київ Ґ ґ Є є Ї ї'),
]
ARABIC_WORDS = [
    ('Persian', 'ایران تهران مشهد افغانستان کابل هرات'),
    ('Urdu', 'پاکستان لاہور کراچی پشاور راولپنڈی'),
    ('Urdu', 'ڈیرہ غازی خان حیدرآباد ملتان سیالکوٹ ٹانک'),
    ('Urdu', 'بہاولپور اٹک ہنزہ گلگت کوہاٹ چترال'),
    ('Urdu', 'سرگودھا جہلم ٹھٹھہ کوئٹہ گوجرانوالہ جلوۂ'),
    ('Punjabi', 'پنجاب لہور ساہیوال فیصل آباد'),
    ('Balochi', 'بلوچستان گوادر تربت پنجگور'),
    ('Pashto', 'خیبر پښتونخوا ننګرهار کونړ لغمان پکتیا'),
    ('Pashto', 'ډېره اسماعیل خان ځای څلور ږیره غونډۍ ښار ټول'),
    ('Pashto', 'کندهار هلمند کوټه ګردېز ټانک'),
    ('Sindhi', 'سنڌ ڪراچي حيدرآباد ٺٽو لاڙڪاڻو ٿر'),
    ('Sindhi', 'ڄامشورو ٻيلو ڀٽ شاهه ڳوٺ ڏيپلو ڊگهو'),
    ('Sindhi', 'ڦل ڇور اڱڻ ڍڪ ڃڃ ڎڎ سکر ميرپورخاص'),
    ('Sorani', 'کوردستان سلێمانی هەولێر کەرکووک دهۆک'),
    ('Sorani', 'ڕانیە هەڵەبجە زاخۆ ڤیان شێخ'),
    ('Uyghur', 'كۇچا پەيزاۋات مارالبېشى تەكلىماكان ئۈرۈمچى شىنجاڭ'),
    ('Uyghur', 'ھېيت شەھەر مەھەللە ئۆلكە تەڭرى يېڭى غۇلجا'),
    ('Kazakh', 'ٴىلە قازاقستان'),
    ('Kirghiz', 'قىرعىزستان كۅل ۉ'),
]

# Unicode joining types (enough for these languages)
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
    joined = set(sum((feat[t] for t in ('init', 'medi', 'fina', 'locl')), []))
    alt = {}
    for i in range(len(lk)):
        if lk[i].LookupType == 1 and i not in joined:
            for a, b in lk[i].SubTable[0].mapping.items():
                alt[b] = a
    ligs = set()
    for lookup in lk:
        if lookup.LookupType == 4:
            for lst in lookup.SubTable[0].ligatures.values():
                ligs.update(lg.LigGlyph for lg in lst)

    hbfont = hb.Font(hb.Face(hb.Blob.from_file_path(FONT)))

    def shape(text):
        if not text:
            return []
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(hbfont, buf, {})
        return [(i.codepoint, i.cluster, p.x_advance, p.x_offset, p.y_offset)
                for i, p in zip(buf.glyph_infos, buf.glyph_positions)]

    # ---- samples ------------------------------------------------------------------------
    uy = []
    for r in csv.DictReader(open(os.path.join(NAMING, 'local_native.csv'), encoding='utf-8')):
        if r['lang'] in ('uyghur', 'kazakh', 'kirghiz'):
            uy.append((r['name'] + ' (' + r['lang'][:2] + ')', r['native']))
    zh = []
    for r in csv.DictReader(open(os.path.join(NAMING, 'all_names.csv'), encoding='utf-8-sig')):
        if r['zh'].strip():
            zh.append((r['name'], r['zh'].strip()))
    print('names: %d Uyghur/Kazakh/Kirghiz, %d Chinese' % (len(uy), len(zh)))

    # ---- 1. .notdef and joining forms -------------------------------------------------
    problems = []
    arabic_samples = ARABIC_WORDS + uy
    forms_table = []
    for Lt in ARABIC:
        c = chr(Lt['cp'])
        if Lt['join'] == 'D':
            forms_table.append(('U+%04X %s' % (Lt['cp'], Lt['lang']),
                                '  '.join([c, c + 'ـ', 'ـ' + c + 'ـ', 'ـ' + c, c + c + c])))
        else:
            forms_table.append(('U+%04X %s' % (Lt['cp'], Lt['lang']),
                                '  '.join([c, 'ـ' + c, 'ب' + c, c + c])))
    every = LATIN + CYRILLIC + arabic_samples + forms_table + zh
    for label, text in every:
        glyphs = shape(text)
        if any(g[0] == 0 for g in glyphs):
            missing = sorted({text[g[1]] for g in glyphs if g[0] == 0})
            problems.append('%s: .notdef for %s' % (label, ' '.join('U+%04X' % ord(m) for m in missing)))
    used = {}
    for label, text in arabic_samples + forms_table:
        cps = [ord(c) for c in text]
        exp = expected_forms(cps)
        by_cluster = {}
        for g in shape(text):
            by_cluster.setdefault(g[1], []).append(order[g[0]])
        for k, cp in enumerate(cps):
            if exp[k] is None or k not in by_cluster or cp not in cmap:
                continue
            gl = by_cluster[k]
            want = cmap[cp] if exp[k] == 'isol' else maps[exp[k]].get(cmap[cp], cmap[cp])
            got = alt.get(gl[0], gl[0])
            if got != want and gl[0] not in ligs:
                problems.append('%s: U+%04X %s wants %s got %s' % (label, cp, exp[k], want, gl[0]))
            used.setdefault(cp, set()).add(exp[k])
    for Lt in ARABIC:
        need = {'isol', 'init', 'medi', 'fina'} if Lt['join'] == 'D' else {'isol', 'fina'}
        if not need <= used.get(Lt['cp'], set()):
            problems.append('U+%04X: forms not exercised: %s' % (Lt['cp'], need - used.get(Lt['cp'], set())))
    print('shaped %d lines; problems: %d' % (len(every), len(problems)))
    for p in problems:
        print('  ', p)

    # ---- 2. test sheet -------------------------------------------------------------------
    face = freetype.Face(FONT)
    face.set_char_size(12 * 64)
    flags = freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_MONO | freetype.FT_LOAD_NO_HINTING
    cache = {}

    def gbits(gid):
        if gid not in cache:
            face.load_glyph(gid, flags)
            g = face.glyph
            bm = g.bitmap
            cells = []
            for y in range(bm.rows):
                for x in range(bm.width):
                    if bm.buffer[y * bm.pitch + x // 8] & (0x80 >> (x % 8)):
                        cells.append((g.bitmap_left + x, g.bitmap_top - 1 - y))
            cache[gid] = cells
        return cache[gid]

    def width(text):
        return sum(g[2] for g in shape(text)) // 160

    def draw(img, x, base, text):
        pen = 0
        for gid, cl, adv, xo, yo in shape(text):
            for (i, j) in gbits(gid):
                px, py = x + (pen + xo) // 160 + i, base - j - yo // 160
                if 0 <= px < img.width and 0 <= py < img.height:
                    img.putpixel((px, py), 0)
            pen += adv

    extra_cyr = 'ӨөҮүӘәҒғҚқҢңҰұҺһІіӢӣҲҳҶҷЎўЁёЙйЄєҐґЇїЅѕЈј'
    tailed = [c for c in COMPOSE if 0x400 <= ord(c) <= 0x52F]
    lat_new = [c for c in list(LATIN_ART) + list(COMPOSE) if not 0x400 <= ord(c) <= 0x52F]
    rus = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    rows = [('# Latin (m5x7)', '')] + LATIN + \
        [('New Latin', ' '.join(lat_new))] + \
        [('# Cyrillic', '')] + CYRILLIC + \
        [('Russian', rus), ('', rus.lower()),
         ('Other', ' '.join(extra_cyr)),
         ('Made here', ' '.join(tailed))] + \
        [('# Chinese (Ark Pixel; Fusion Pixel fills gaps)', '')] + \
        [('From Fusion', '溪 梁 犁 廊 滨 朔 慈 施 徽'), ('Spliced', '颍 枞 鳌')] + \
        [('', '  '.join(z for _, z in zh[k:k + 8])) for k in range(0, min(len(zh), 64), 8)] + \
        [('# Arabic script', '')] + ARABIC_WORDS + \
        [(a.replace('\u00ed', 'i'), b) for a, b in uy[:12]] + \
        [('# New Arabic letters: isol init medi fina, run', '')] + forms_table
    LH = 19
    lw = 110
    tw = max(width(b) for _, b in rows)
    W, H = lw + tw + 12, LH * len(rows) + 8
    img = Image.new('1', (W, H), 1)
    for k, (a, b) in enumerate(rows):
        base = 4 + k * LH + 12
        draw(img, 4, base, a.lstrip('# ') if a.startswith('#') else a)
        if a.startswith('#'):
            for x in range(4, W - 4):
                img.putpixel((x, base + 4), 0)
        if b:
            rtl = any(0x600 <= ord(c) <= 0x6FF for c in b)
            draw(img, (W - 4 - width(b)) if rtl else lw, base, b)
    big = img.resize((W * 4, H * 4), Image.NEAREST)
    sheet = Image.new('1', (W * 4, H + 16 + H * 4), 1)
    sheet.paste(img, (0, 0))
    sheet.paste(big, (0, H + 16))
    sheet.save(SHEET)
    print('wrote', SHEET, sheet.size)


if __name__ == '__main__':
    main()
