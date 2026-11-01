# unixel Uyghur

A copy of the pixel font **unixel** (Persian, from the Unixel/Rooyin project,
https://github.com/MDarvishi5124/unixel) with the Arabic-script letters Uyghur, Kazakh
and Kirghiz need. New letters are drawn on unixel's own pixel grid (1 pixel = 160 units,
2000 units per em) by copying its existing letters and moving dots.

## What was added

| Letter | Name | Forms | Built from |
|---|---|---|---|
| ۇ U+06C7 | U | isolated, final | waw + small damma |
| ۆ U+06C6 | OE | isolated, final | waw + small v |
| ۈ U+06C8 | YU | isolated, final | waw + small yeh |
| ۋ U+06CB | VE | isolated, final | waw + three dots |
| ۅ U+06C5 | Kirghiz OE | isolated, final | waw + bar through the tail |
| ۉ U+06C9 | Kirghiz YU | isolated, final | waw + inverted v |
| ې U+06D0 | E | all four | dotless yeh + two dots stacked below |
| ڭ U+06AD | NG | all four | kaf + three dots |
| ھ U+06BE | heh doachashmee (h) | all four | two-eyed heh in every form, so it never looks like ە |
| ٴ U+0674 | Kazakh high hamza | spacing letter | hamza (the source mapped it to the zero-width mark) |

Also changed:

- **U+0649 ى is dotless in initial and medial forms** (Uyghur i). unixel gave it the dotted
  Persian ی forms. U+06CC ی (Persian yeh) now has its own glyph and keeps the dotted forms.
- The presentation-form codepoints for the new letters (FBAA-FBAD, FBD3-FBD6, FBD7-FBE7)
  are mapped too.
- Descent is one pixel deeper (1120 units instead of 960) so the lower dot of ې fits.
- ئ U+0626, ە U+06D5, ى U+0649 and ي U+064A were already in unixel.

Not added: Kazakh ٵ ٶ ٷ ٸ (precomposed high-hamza letters; Kazakh text normally types ٴ
as its own character) and accented Latin (ü, é).

## Install

Right-click `unixel-Uyghur-Regular.ttf` > **Install for current user**, then restart
Aseprite. It shows up as **unixel Uyghur** and sits next to the original unixel, which it
does not replace.

One design pixel is 1/12 of the font size (units per em set to 1920 = 12 x 160; the original
unixel's 2000 gave 12.5), so the font is pixel-exact at 12, 24, 36 ... Turn anti-aliasing off
for crisp pixels.

## Rebuild

    C:\Python39\python.exe build_font.py [path\to\unixel-Regular.ttf]
    C:\Python39\python.exe check_font.py

`build_font.py` reads the installed unixel by default (read only) and writes
`unixel-Uyghur-Regular.ttf`. To add letters (for Pakistan: Urdu ٹ ڈ ڑ ں ے ہ, some of which
unixel already has), add an entry to `LETTERS`: per form, the glyph to copy, pixels to add
and pixels to remove, in (column, row) font pixels with row 0 sitting on the baseline.

`check_font.py` shapes every uyghur/kazakh/kirghiz name in `../naming/local_native.csv`
plus sample words with HarfBuzz, checks there are no missing glyphs and that every letter
gets the joined form its position needs, checks FreeType's 1-bit rendering matches the
drawn pixels, and redraws `test_sheet.png` (1x on top, 4x below; the top block of the 4x
part shows each new letter in all its forms, with ـ to show the joins).

Needs fontTools, Pillow, uharfbuzz and freetype-py.

## Licence

SIL Open Font License 1.1, same as unixel (`OFL.txt`). unixel declares no Reserved Font
Name; the copy is renamed "unixel Uyghur" anyway so both can be installed. The original
copyright line is kept, and the font's description field records the modification.
