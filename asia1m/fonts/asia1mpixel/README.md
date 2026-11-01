# asia1mpixel

One pixel font for the asia1m map labels: m5x7 Latin, Pixeloid Sans Cyrillic, Ark Pixel
12px Chinese and unixel Uyghur Arabic script on a single grid, plus extra Latin and Arabic
letters drawn to match. At font size **12**, one design pixel is one screen pixel for every script (at 24 it
is 2x2). Each script keeps its own native size:

| Script | Source | Size at 12 px |
|---|---|---|
| Latin, digits, punctuation | m5x7 | caps 7 px, x-height 5, descenders 2 (as m5x7 at 16 px) |
| Cyrillic | Pixeloid Sans | same as Latin (caps 7, x-height 5, descenders 2) |
| Chinese | Ark Pixel 12px Mono zh_cn; Fusion Pixel 12px for characters Ark lacks | 12 px em, 11 px glyphs |
| Arabic script | unixel Uyghur (../unixel-Uyghur-Regular.ttf) + new letters | as unixel |

Line box: 12 px above the baseline and 7 below (19 px). The harakat marks and two vertical
kana marks reach a little higher; the Windows clipping box covers them.

## Contents

About 25,800 code points:

- **Latin (m5x7):** 318 code points: Basic Latin, Latin-1, Latin Extended-A (Turkish,
  Turkmen, Kurmanji, Indonesian and so on are all there already).
- **Latin drawn to match:** Ə ə (Azerbaijani), pinyin Ǎ ǎ Ǐ ǐ Ǒ ǒ Ǔ ǔ ǖ ǘ ǚ ǜ,
  Ǵ ǵ (Karakalpak), ʻ ʼ (Uzbek oʻ gʻ), ‘ ’ “ ” – — … •.
- **Cyrillic:** 176 code points, from Pixeloid Sans: full Russian, Mongolian Ө Ү,
  Kazakh Ә Ғ Қ Ң Ұ Һ І, Kirghiz, Tajik Ғ Ӣ Қ Ҳ Ҷ, Uzbek Ў, Ukrainian Є Ґ Ї and more.
  Exceptions:
  - lowercase а е р с у ё are m5x7's a e p c y ë, so they match the Latin exactly;
  - ў ӯ are m5x7's y with m5x7's breve and macron;
  - Ӯ (which Pixeloid lacks) is Pixeloid's У with the macron it uses on Ӣ.
  `cyrillic_options.png` shows the candidates that were compared, with a width check
  against m5x7.
- **Chinese:** 22,640 code points from Ark Pixel (CJK ideographs, CJK punctuation and
  symbols, fullwidth forms, Bopomofo) and 2,326 more from Fusion Pixel, for characters
  Ark's zh_cn build has not drawn (溪 梁 犁 廊 滨 and so on), three of them spliced (颍 枞 鳌).
  No Hangul or kana.
- **Arabic script:** everything in unixel Uyghur (Persian, Uyghur, Kazakh, Kirghiz) plus 37
  new letters with all their joining forms:
  - Urdu / Punjabi Shahmukhi / Balochi: ٹ ہ ۂ, joined forms of ں (ڈ ڑ ے ۓ ھ were in unixel)
  - Pashto: ټ ځ څ ډ ړ ږ ښ ګ ڼ ۍ (ې ی were there)
  - Sindhi: ٺ ٻ ٽ ٿ ڀ ڃ ڄ ڇ ڊ ڌ ڍ ڎ ڏ ڙ ڦ ڪ ڱ ڳ ڻ
  - Kurdish Sorani: ڕ ڤ ڵ ێ (ۆ ە ھ were there)

Not included: Tibetan, Yi and the Indic scripts (no pixel font with a usable licence and
shaping fits this grid); Urdu-style digits; the Chinese characters 滹 (Hutuo) and 溆 (Xupu),
which no source has; presentation-form code points for the new Arabic letters (text uses
the normal letters and the font's joining rules).

Chinese and Latin share the curly quotes, dashes and ellipsis. One code point can only
have one glyph, so these are the small m5x7-style ones; Chinese text needs fullwidth
punctuation (，。、「」（）), which comes from Ark Pixel.

## Install

1. Right-click `asia1mpixel-Regular.ttf` > **Install for current user**.
2. Restart Aseprite. The font shows up as **asia1mpixel**.
3. Use size **12** (or 24 for 2x) with **anti-aliasing off**.

## Rebuild

    C:\Python39\python.exe build.py
    C:\Python39\python.exe check.py

`build.py` reads m5x7 and Ark Pixel 12px from the installed fonts, Pixeloid Sans and
Fusion Pixel from `sources/`
and ../unixel-Uyghur-Regular.ttf (rebuild that first with ../build_font.py if you change
it). It writes `asia1mpixel-Regular.ttf` and then renders every glyph again to confirm it
matches the pixels it was built from. It takes under a minute.

To add letters, add a table entry in `build.py`:

- `LATIN_ART`: a letter drawn as pixel art on the m5x7 grid (`#` = pixel), with the row of
  its top line (caps start at row 6, small letters at row 4).
- `COMPOSE`: an existing letter plus accent pixels (`CARON_CAP`, `MACRON_LOW`, ...).
- `COPY`: a letter that looks the same as an existing one.
- `CYRILLIC`: the code point range taken from Pixeloid Sans.
- `ARABIC`: per joining form, the unixel glyph to start from, pixels to add and pixels to
  remove (helpers `TAH`, `RING`, `V`, `UP3`, `DOWN3`, `FOUR`, `HAMZA` place the marks).
- `SPLICE`: a Chinese character from column ranges of others.

Pixels are (column, row) with row 0 the row sitting on the baseline.

`check.py` shapes sample words in every script with HarfBuzz, plus every
Uyghur/Kazakh/Kirghiz name in `../../naming/local_native.csv` and every Chinese name in
`../../naming/all_names.csv`. It reports any missing glyph and any Arabic letter that did
not get the joining form its position needs, then redraws `test_sheet.png` (1-bit at 12 px,
1x on top and 4x below).

Needs fontTools, uharfbuzz, freetype-py and Pillow.

## Licence

SIL Open Font License 1.1 (`OFL.txt`); the credits are in `FONTLOG.txt`. You can use it
for anything, including the map and prints of it, and share or sell it bundled with other
things, but not sell the font on its own. Any copy must keep `OFL.txt`.

- m5x7 is CC0, so it can go under the OFL with the rest.
- Ark Pixel, Fusion Pixel, Pixeloid Sans and unixel are OFL. Ark Pixel and Pixeloid
  reserve their names ("Ark Pixel", "Pixeloid"), which this font does not use. Fusion's own sources are OFL or more permissive
  (licences in `sources/fusion-pixel-licenses/`).
