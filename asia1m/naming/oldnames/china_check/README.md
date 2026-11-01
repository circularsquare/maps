# china_check

A proofread of the China part of the `textlocal` layer (Chinese-character labels), as of the
asia1m.ase saved on 2026-10-24. Read only: nothing in the .ase or the naming files was changed.

## Files

- `issues.csv`: one row per issue. `kind` (missing, duplicate, wrong_colour, wrong_characters,
  illegible, other), `key` (region colour), `name`, `zh` (all_names.csv, or the native spelling for
  non-Han regions), `saw`, the box on the full 9500x8000 map (`x, y, w, h`), `found` (data, visual,
  both), `confidence`, `note`, and `crop`. Sorted high confidence first.
- `issues.png`: transparent 9500x8000 overlay, a 2 px box (3 px margin) round each issue, coloured
  by kind; legend at the bottom left. Lay it over the .ase as a layer.
- `crops/`: one 2x crop per issue with its box drawn, named `kind_key_name.png`. For main-map +
  inset pairs that are far apart the crop shows the first copy's end of the span.
- `scripts/`: the scripts used (see Method). They read layer exports from the folder they sit in.

## Counts

157 issues: 19 missing, 71 duplicate, 11 wrong colour, 41 wrong characters, 8 illegible, 7 other.

| kind | high | medium | low |
|---|---|---|---|
| missing | 1 | 0 | 18 |
| wrong_colour | 10 | 1 | 0 |
| wrong_characters | 3 | 6 | 32 |
| duplicate | 3 | 68 | 0 |
| illegible | 0 | 6 | 2 |
| other | 2 | 0 | 5 |

Found by: both 118, data only 28, visual only 11.

Most of the 68 medium duplicates are the same name drawn on the main map and again on a city
inset (Beijing-Tianjin 18, Shanghai and Hangzhou-Ningbo 20, Pearl River 14, Chengdu-Chongqing 10,
Wuhan and Xi'an 5). They are listed so you can decide per inset whether the main-map copy stays.
Most of the 32 low wrong_characters are your own names that differ from all_names.csv on purpose
(shortenings like 盘龙 for 盘龙城, 科尔沁 for 科尔沁左翼后, or other local names like 塔城, 宁安, 骆岗),
plus six where all_names.csv itself looks wrong (锦州, 红枫, 邵水, 耒水, 灌河, 临沂). The 18 low
missing are slivers and tiny cores of 0 to 300 px with no name, nearly all with nothing visible to
label.

## Most certain issues

1. Ili (#f9eaf5), 4083,2280: the big region west of Ghulja has no label at all.
2. Jincheng (#5bd7f4), 6205,3228: reads 晋城, should be 金城 (Lanzhou).
3. Qujiang (#e8dc43), 6168,3005: Xi'an list entry e. 曲江 is in Weiyang's colour; Qujiang's colour
   appears nowhere.
4. Shanghai list entry o, 8288,3549: key o (list and inset) is in Minhang's colour and sits on
   Minhang, but the name is 崇明 in Chongming's colour. Probably meant 闵行.
5. Taicang (#8990df), 8096,3443: labelled 千灯 on the Shanghai inset; 千灯 is also Qiandeng's own label.
6. 烟台 (7629,2905) and 黄岛 (7549,3102) are each drawn twice, slightly offset.
7. 李沧 (Qingdao) is on the map and in the list; the Qingdao core is a pile of overlapping vertical
   labels and keys (7555,3055), with 青岛 in Fushan's colour.
8. Xi'an and Chongqing insets: the lists a-e and a-f have no keys on the inset cores.
9. 丹阳 is drawn in the colour of a 10 px sliver (#e69adf) instead of Danyang's #e69acd.
10. Liaoyang (#8ae995), 7679,2533: the 海城 label is in Liaoyang's colour (Haicheng's is very close).
11. List and label colours that are off by a few units: 老山, 茅山, 湛江, and the whole Guiyang list
    (云岩, 南明, 观山湖, 白云). They read right; only the colour does not match the region.

## Data flags the sweep showed are fine (left out of issues.csv)

18 flags: 齐齐/哈尔 (two lines), 昆嵛, 紫金 (main and inset), 藤州, 鉴江, 溆浦, 海虞, 本溪, 花溪, 安溪,
玉溪, 岑溪, 紫帽 (other glyph forms the font matcher missed, mostly 溪 which Ark Pixel lacks), the
Zhengzhou leader arrow, 曹县 drawn staggered, lạng sơn on the cross-border Langson region, and the
Wuhan inset title 武汉 in Hankou's colour. Also left out: 39 small pieces in a region's own fill
colour on `textlocal`. Every one of them covers a few pixels of region border (or river) under a
label, so they are deliberate border gaps, not labels.

The sweep found 11 things the data check could not: the Harbin keys c/d, Nanjing key a, Xiamen
keys 9/10 and the Pearl River keys 1 and 5 (which read as "15") are illegible; the Hangzhou inset
core is low contrast; the missing keys on the Xi'an and Chongqing insets; the Guangzhou and Shenzhen
lists whose rows touch; and the traditional 連 in 大連湾 and 關 in 西關 (興 in 興京 was flagged by both).

## Method

1. A copy of asia1m.ase was exported layer by layer with Aseprite (`-b --all-layers --layer <name>`):
   the visible stack is coastglobal, main, insets, textlocal (bottom to top). They were composited
   in that order (textlocal alpha set to 255) into one image. The copy was deleted afterwards.
2. Data check (`scripts/datacheck.py` + `scripts/tail.py`), built on place_local.py's reader:
   textlocal is cut into pieces of one colour (pixels within 6 px), stacked same-colour list lines
   are split at blank rows, and pieces under 9 px tall are keys. A piece's colour names its region
   (region colour x 0.75 rounded up, 255 giving 191; within 1 per channel; nearest region or inset
   copy within 1000 px). Each piece is read against the region's `zh` and the names of regions
   within 300 px: whole-label match in Ark Pixel zh_cn / zh_hk / ja, Fusion Pixel, asia1mpixel,
   SimSun 12 px and Unifont, at 1x, 2x and 3x, in one line, a column or two lines, 80% IoU with
   1 px of play; then character by character anywhere in the piece (each character 60% IoU, 80% of
   the ink explained). Non-Han regions in nonhan_regions.csv with a native spelling are accepted
   whatever they read. Inset frames are the solid pieces of the insets layer; a piece inside one
   is on the inset, a big piece above one is its title.
3. Visual sweep: the China extent (3749-8423 x 1276-5310) was cut into 110 tiles of 500x400 map px
   (12 px overlap) at 2x. 78 tiles hold China pixels and were each looked at in full; the other 32
   (Kazakhstan, Mongolia, Russia, the seas, Korea, Japan, Bangladesh, Thailand, Laos, the
   Philippines, south Taiwan) were looked at as 1x contact sheets. Dense cores were zoomed to 4-7x.
4. `scripts/merge.py` joins both: visual notes that confirm a data flag replace it (`found = both`),
   false alarms and border gaps are dropped, and the overlay and crops are drawn.
