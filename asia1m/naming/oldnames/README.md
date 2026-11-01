# oldnames

The names already written on the asia1m text layer, read back by code, with Chinese characters
for China. Second part: a draft layer of Chinese labels for the local-script map.

## Run

    python asia1m/naming/oldnames/read.py         # text.png -> labels.json (about 2 minutes)
    python asia1m/naming/oldnames/assign.py       # labels.json -> oldnames.csv, review.csv (about 1 minute;
                                                  #   the first run also builds gazetteer.json)
    python asia1m/naming/oldnames/place_local.py  # her layer textlocal.png -> her_labels.csv, local_labels_new.png,
                                                  #   local_placements.csv, local_unplaced.csv,
                                                  #   changes_vs_current.csv, changes_vs_current.png (under 1 minute)

Inputs: `../text.png`, `insets.png`, `textlocal.png`, `layer1.png`, `../regions.json`, `../picks.csv`, `../local_names.csv`,
`../../regionnaming/` (main.png, region_id.npy, regions_prelim.csv, adm_all and adm_twn rasters),
CHGIS v6 points in `data/asia1m/chgis/`, and `features.csv` (rivers, lakes, mountains and areas in
China with Chinese names, from GeoNames CN.zip, CC BY 4.0; `geonames_extract.py <CN.zip>` remakes
it from https://download.geonames.org/export/dump/CN.zip).

The .ase layers are exported from a copy of the file, never the original:

    "C:/Program Files/Aseprite/Aseprite.exe" -b --all-layers --layer <layer> <copy of asia1m.ase> --save-as <file>

- `../text.png` = layer `text` (alpha set to 255; the export carries the layer's opacity). The
  2026-10-19 export is kept as `../text_2026-10-19.png`; the one in use is from 2026-10-22, 23:33.
- `insets.png` = layer `insets`: the magnified city cores, each region in its main-map colour. The
  inset labels themselves are on the `text` layer.
- `textlocal.png` = layer `textlocal`, her local-script text as it stands (alpha set to 255).
- `layer1.png` = layer `Layer 1`: the first-round draft (local_labels_v1.png) as she imported it,
  with her edits; used only to tell a first-round label she deleted from one not yet merged.

## How the reader works

- **Font.** The labels are m5x7 at 16 px (x-height 5, ascenders and descenders 2), sometimes at 2x
  or 4x; keys inside small city regions are a 3x5 digit font. `glyphs.txt` holds the templates
  (taken from the map itself), `marks.txt` the accent marks.
- **Lines** (`glyphs.py`, `read.py`): exact template hits find the text lines. Each line is then
  read end to end by a best-path search: every glyph placed must have all its template pixels
  inked, and ink nothing explains costs points. That reads letters touched by another label's
  pixels. Ink between letters that nothing explains becomes `?`.
- **Accents**: marks over x-height letters (Vietnamese tones, circumflex, breve, horn, dot below;
  Korean ŏ ŭ) are read and kept for Vietnam, Korea (breve) and Japan (macron); elsewhere stray
  marks are dropped as other labels' pixels.
- **Label colour** (`assign.py`): every label, key and list entry is drawn in its own region's
  colour x 0.75, rounded up (a channel of 255 comes out 191). Fitted on the 1,734 lines that sit
  squarely inside one region: 1,529 match that rule exactly; nearly all the rest are list
  entries and keys, which are drawn in the colour of the region they name, not the one they sit
  on. So the ink colour names a label's region even when it sits on a neighbour, in the sea, on
  an inset or in a list. Colours are matched within 1 per channel, to regions within 1,000 px
  (1,500 for list entries, 60 for keys) of the label, counting a region's inset copy as part of
  it. Among several regions of that colour: an exact colour first, then one whose names match
  the text, then the nearest.
- **Labels**: lines stacked 7 to 13 rows apart and overlapping are one label; joined without a
  space in China, Taiwan, Hong Kong, Korea and Japan, with a space elsewhere, and without the
  hyphen after a line ending in `-`. Colour decides the grouping first: in a stack, a change of
  colour starts a new label ("rui'an" over "aojiang"), and a line with words in two colours is
  cut at the space ("gaobei gou" over "dian", where "gou" ends the bai / gou label beside it).
  Pieces the reader kept apart that touch and share a colour are joined ("ding" + "tao",
  "xing" over "ping"), in the order that matches a name of the region if only one does. Two
  whole labels in one colour stay apart when one names the region on its own and the other
  names a region nearby ("pampanga" over "macabebe"). Lines whose colour names no region fall
  back to the name split: a stack is split where its parts match different regions' names
  better ("mianchi / luoyang"), and a line holding two names side by side is split at the space
  ("shenbei fushun"). Two list columns read as one line ("12. pesanggrahan 18. jati asih") are
  two entries.
- **Regions**: a label goes to the region its colour names. Keys are paired with a list entry of
  the same key and colour, or one whose name matches the key's region; an entry whose colour
  names another region is not paired, except that an entry in the colour of a region whose own
  entry is already paired is treated as a slip ("1. shenyang" is drawn in entry 2's colour).
  List entries with no key on the map go to the region of their colour. When a region ends up
  with two labels in its colour and one names neither it nor its units, that one is a colour
  slip and moves to an unlabelled region under or beside it that has the name, or whose colour
  is within 4 of the ink ("gucheng" in Wuqiao's colour, "kunyu" in Haiyang's near twin). Only
  text whose colour names no region (23 labels) falls back to the old rule: the region under
  most of its ink, unless one under or beside it has a county, town or historical seat of that
  name; inset text is then matched by name within 700 px, and a region holding two such labels
  passes the second to an unlabelled region under it.
- **Characters** (China): the cleaned name, folded to bare pinyin, is compared with pypinyin
  readings (all readings of polyphones) of, in order: the region's counties and county-level
  units, its prefectures, its townships (all of them, from the township raster), CHGIS
  historical seats in the region, GeoNames rivers and mountains near the label, then the
  neighbours' units. Admin suffixes and ethnic-group parts are dropped (元江哈尼族彝族傣族自治县 ->
  元江), but kept for one-character names (息县). One unread glyph may stand for one or two letters.
  Fuzzy matches (one letter off) are given but marked low. When the label's colour pins the
  region and the one letter is a look-alike another label's pixels can make (i/j, o/d, l/b,
  l/i, n/h, c/e), the name takes the region's spelling ("aoiiang" -> Aojiang 鳌江, medium).

## Outputs

- `oldnames.csv`: one row per region with a label. `text_as_read` is the lines as read (` / `
  between stacked lines), `name` the joined, capitalised name, `zh` and `zh_source` (unit,
  prefecture, town, chgis, geonames, neighbour ..., wildcard ..., fuzzy ..., none), `confidence`,
  `notes` (how it was placed, other text on the region, whether it matches picks.csv), and the
  label's box on the map. Rows for regions in picks.csv are included; `in_picks` marks them.
- `review.csv`: low-confidence rows and China rows with no characters; text that went to no
  region; list entries whose key wasn't found; keys with no list entry; regions outside picks with
  no label at all.
- `labels.json`: every label read, with its boxes (used by place_local.py).

Confidence: **high** = the name matches the region's own county, prefecture, township or
historical seat exactly (China: with characters), placed by the label's colour, ink or a key;
**medium** = matched by a river or mountain from GeoNames, one character only, a neighbour's
name, an unread glyph filled in, or (outside China) a clean read with no admin name to check
against; **low** = unread glyphs, fuzzy matches, or no match in China.

## Summary (2026-10-22, third round: fresh text export)

2,224 regions have a label (all countries, picks included); the first round had 1,801, the
second 2,039. The new text export adds 185 China regions (labels she drew since Oct 19) and
changes two names (#eec7cc "an" is now Jianmen 剑门). Of the 2,376 labels placed, 2,062 went by
their colour, 291 by a key, and 23 by the old ink and name rule (text in no region's colour:
sea and area names, scale bars, some overlapping text). 38 rows note other text on the region;
review.csv has 415 rows (232 regions with no label, 80 in China; 129 low or unmatched; 50 keys
with no list entry; 4 bits of text that went nowhere). all_names.csv: 2,237 names, 1,768 from the
map. audit.csv flags the same 10 rows as the second round.

The table and the paragraphs after it are the second round's, except the China row.

| country | high | medium | low |
|---|---|---|---|
| China | 1156 | 87 | 95 |
| Indonesia | 221 | 43 | 6 |
| Japan | 69 | 31 | 5 |
| Philippines | 63 | 33 | 1 |
| Vietnam | 59 | 23 | 5 |
| Thailand | 55 | 12 | 0 |
| Myanmar | 43 | 6 | 3 |
| India | 23 | 16 | 1 |
| Malaysia | 22 | 11 | 1 |
| South Korea | 19 | 20 | 6 |
| North Korea | 16 | 9 | 0 |
| Taiwan | 14 | 10 | 0 |
| Cambodia | 11 | 4 | 2 |
| Laos | 5 | 0 | 3 |
| others (Russia, East Timor, Bhutan, Mongolia, Singapore, Hong Kong) | 3 | 11 | 1 |

China rows with characters (third round): 1,156 high, 87 medium, 16 low; 79 have none (mostly river, mountain,
area and old-region names that no source here has, e.g. jirim, xar moron, beijiang, nankun).

Against the first round: rows noting "other text on region" fell from 232 to 34 (most of the 34
are a name in brackets with a stray mark under it in Indonesia and Japan, or a river or area name
in the region's own colour); regions with no label fell from 443 to 233 (81 in China). 230
regions that had no label now have one: 72 by a label on an inset, 82 by a key (most keys are on
insets), 62 by a label in their colour sitting off the region, 13 by a list entry's colour. 20
regions lost the label they had because its colour names another region: mostly a city core that
now carries its own name (Gaoyou, Zunyi, Xinbei, Jiading, Hong Kong) where the first round had put
it on the larger region beside it. Picks matching the map: 250 of 267 (was 206 of 239).

review.csv has 390 rows: 233 regions with no label, 112 low or unmatched rows, 41 keys with no
list entry (most are keys 7 to 21 on the Guangzhou inset, whose list has only entries 1 to 6 on
text.png, and the 9 / g confusion), and 4 bits of text that went to no region. Every list entry
now finds its region.

The five hand fixes in ../name_fixes.csv (Aojiang, Longwan, Rui'an, Yongjia, Wenzhou) now come
out the same without them.

**Checked by hand** (second round): 30 rows drawn at random from the 232 that had "other text" in
the first round. 28 are clearly right (e.g. Luc Nam, Tongxu, Pinghe, Zhengding, Cibadak, Taishan
with Jiangmen moved to its core, Yongning with Wuxiang moved off). Haiyu 海虞 on a region that is
66% Zhangjiagang follows her colour (a Changshu town name). Baohe: key 5 names Luogang 骆岗 and the
region's own label says Baohe; the pick is Binhu, so neither is certain.

**Accuracy measured** (first round, reading): 60 rows drawn at random, each label crop checked by
eye against what was read: 57 read exactly (95%). The 3 misses: a 'g' lost under another
label's pixels (treng-galek read "tren-galek"; such losses now show as `?`), a 'g' lost under a
stray mark (shangshui read "shanshui", fixed to 商水 as a low fuzzy match), and a Vietnamese name
whose second word wasn't joined (lục nam read "luc"; now joined by colour). Letter for letter it
is above 99%. 50 random China high rows checked for characters: 47 right; the 3 wrong were a
river name with two possible rivers (攸水 v 酉水) and two one-syllable fragments, now joined into
whole names.

## Local-script draft (place_local.py)

Works against her local text layer as it stands (`textlocal.png`), not asia1mloc.png (an old
export). Style: Ark Pixel 12px (monospaced zh_cn) at 12 px, colour = region colour x 0.75
rounded up; characters Ark Pixel lacks (颍 徽 溪 梁 慈 ...) are drawn with SimSun 12 px. Tibetan,
Uyghur, Kazakh and Kirghiz names are not drawn: this PIL has no libraqm, so no shaping; they are
in local_unplaced.csv.

**Reading her layer.** Her text is cut into labels: pixels of one colour within 6 px of each
other. A label's colour names its region, as in assign.py. Each label is compared with the
rendered characters of every region within 300 px: three layouts, in Ark Pixel zh_cn, zh_hk and
ja and in Unifont 16 px, with 1 px of play and 80% of pixels agreeing. A character Ark Pixel
lacks is left out when the name has three or more characters; a list key ("a.") is dropped
first. `her_labels.csv` lists every label with its kind:

- right: reads as its colour's region's name;
- off its region: right, but more than 40 px from the region and its inset copy (list entries
  aside);
- another region's name: in the colour of a region that has a name of its own, reading as a
  nearby region's name. These follow the first round's wrong assignments (惠山 in Xishan's
  colour, 太仓 in Jiading's), so the fix is the characters;
- another region's colour: reads as a nearby region's name, in a colour of a region with no name
  in the data, or of no region;
- old characters: the first round's characters for that region (包头 where the data now says 东河);
- unread, right length: no font reads it, but it is the size of the right number of characters
  (another font, a custom layout, or characters she drew). Counted as hers, not flagged;
- different characters: in the region's colour, but neither readable nor the right size. Some are
  real differences (she wrote 松山 where the data says 松山湖, 科尔沁 for 科尔沁左翼后, 白龙 for
  白龙江), some are staggered layouts or two copies of one label run together;
- key or mark: under 9 px tall; no data (a colour of a region with no Chinese name, e.g. Japan,
  Korea); no region's colour.

**What she needs to do** is in `changes_vs_current.csv` (change, key, map name, data characters,
her label's box and colour, the new box and colour, a note) and `changes_vs_current.png` (a 2 px
red box, 3 px margin, round each of her labels to fix and each new label). Kinds: add, rename,
recolour, check characters, move, move to inset. `local_labels_new.png` holds only the labels to
add.

**Placing what is missing.** A region gets a new label when her layer has none: no label of hers
reads as its name, and none in its colour reads as anything. A city core whose English label is
on an inset gets its characters on that inset, near the English label, whether or not the inset
had local text before. A label of hers for such a core that is only on the main map is listed as
"move to inset". When the name is in a list, the core goes on the inset if its copy there is
bigger than the main-map region. Placement is greedy, smallest regions first: near the English
label (or the region's most interior point), inside the region if it fits, else as much on it as
possible. A new label never touches her text, keeps 3 px from other new labels, and on the main
map keeps off the inset footprints. A first-round spot is reused when still free (`--fresh`
ignores it). A first-round label she deleted from both `textlocal` and `Layer 1` is not put back
but listed ("removed by her?").

Last run (third round): 2,768 labels on her layer: 1,051 right, 5 off their region, 17 another
region's name, 17 another region's colour, 2 old characters, 136 unread but the right length,
45 different characters, 534 keys and marks, 902 in colours with no data, 59 in no region's
colour. Of 1,275 regions with characters, 1,201 have a label on her layer. To add: 67 (46 on
insets, 21 on the main map; 1 sits on a first-round spot already on her Layer 1). Not placed: 33
(7 with no free spot, 26 native script). changes_vs_current.csv: 67 add, 19 rename, 17 recolour,
45 check characters, 5 move, 1 move to inset. No first-round label was found deleted from both of
her layers.

The second round's files (`local_labels.png`, `changes.csv`, `changes.png`) were made against
asia1mloc.png and v1; they are kept as they were but are superseded by the files above.
`local_labels_v1.png` and `local_placements_v1.csv` are the first round, unchanged.

To add the PNG as a layer to a copy of the .ase:

    "C:\Program Files\Aseprite\Aseprite.exe" -b <copy>.ase --script-param png=asia1m/naming/oldnames/local_labels_new.png
        --script-param name="local draft" --script-param out=<copy with layer>.ase
        --script asia1m/naming/oldnames/add_layer.lua

Tested on a scratch copy: the new layer exported back equals local_labels.png pixel for pixel.
