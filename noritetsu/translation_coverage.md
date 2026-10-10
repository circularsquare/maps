# English and local names: coverage and how to fill the gaps

Measured 2026-10-07 on `dist/data/<cc>/lines.json` (`name`, `name_en`) and `stations.json`
(`n` local, `e` English). The language menu and a first `translit()` (Cyrillic, Greek,
Georgian, Armenian) are already in `dist/index.html`; this note measures what that leaves and
what would fill the rest. Scratch scripts were not kept in the project.

## Headline

- 73 regions, **20,294 lines** (4,455 with `name_en`, 22%) and **131,248 stations** (33,397
  with `e`, 25%).
- Those percentages mostly measure Latin-script countries, where no English name is fine:
  the local name shows. 53 of the 73 regions are Latin-script.
- The real gap is a local name in a non-Latin script with no English: **15,559 stations and
  4,273 lines**. Russia alone is 10,215 stations and 3,101 lines.
- After the client's current `translit()`, **about 1,300 stations and 690 lines** still show
  non-Latin text in English mode: cn 545 / 137, jp 202 / 407, tn 138, kz 132 / 42, rs 77, ma
  62, dz 49, eg 43, ir 30, kr 4 / 67, tw 0 / 13. About 215 of the stations (kz, rs, kg, tj,
  mk, xa) are only letters missing from the table.
- `name_en` is a copy of `name` in only a few places: rs 38 lines, gb 36, sg 11, dz 11 lines.
  Stations: in 471, gb 198, sg 162, ie 139, dz 99. All Latin-script, so harmless.

## Per-country table

Script is the script of most local station names. "Gap" = local name not in Latin script and
no English name (stations / lines). Register lines are `src != "osm"`.

| cc | script | lines | en % | register lines | reg en % | stations | en % | gap st / lines |
|---|---|---|---|---|---|---|---|---|
| al | Latin | 9 | 89 | 7 | 100 | 38 | 13 | 0 / 0 |
| am | Armenian (some Cyrillic) | 18 | 39 | 10 | 50 | 101 | 94 | 5 / 11 |
| ar | Latin | 51 | 47 | 23 | 100 | 574 | 0 | 0 / 0 |
| at | Latin | 358 | 14 | 125 | 35 | 2216 | 0 | 0 / 0 |
| au | Latin | 268 | 12 | 141 | 23 | 3014 | 1 | 0 / 0 |
| az | Latin | 29 | 28 | 18 | 17 | 143 | 18 | 0 / 0 |
| ba | Latin | 13 | 38 | 5 | 100 | 170 | 2 | 2 / 0 |
| be | Latin | 299 | 44 | 148 | 87 | 1398 | 1 | 0 / 0 |
| bg | Cyrillic | 61 | 64 | 33 | 94 | 883 | 60 | 338 / 18 |
| br | Latin | 79 | 33 | 19 | 100 | 712 | 0 | 0 / 0 |
| by | Cyrillic (Belarusian) | 163 | 9 | 75 | 16 | 1089 | 9 | 990 / 149 |
| ca | Latin | 197 | 29 | 131 | 38 | 1487 | 1 | 0 / 0 |
| ch | Latin | 797 | 0 | 404 | 0 | 2825 | 0 | 0 / 0 |
| cl | Latin | 25 | 28 | 7 | 100 | 252 | 0 | 0 / 0 |
| cn | Han | 868 | 80 | 423 | 82 | 10060 | 94 | 545 / 137 |
| cz | Latin | 515 | 0 | 244 | 0 | 3556 | 0 | 0 / 0 |
| de | Latin | 2470 | 2 | 1108 | 1 | 13746 | 0 | 0 / 0 |
| dk | Latin | 109 | 0 | 46 | 0 | 606 | 1 | 0 / 0 |
| dz | Latin / Arabic | 42 | 60 | 24 | 100 | 414 | 41 | 46 / 4 |
| ee | Latin | 40 | 2 | 13 | 0 | 165 | 4 | 0 / 0 |
| eg | Arabic | 38 | 89 | 30 | 100 | 367 | 84 | 43 / 1 |
| es | Latin | 423 | 41 | 171 | 100 | 2755 | 0 | 0 / 0 |
| fi | Latin | 78 | 12 | 31 | 0 | 455 | 2 | 0 / 0 |
| fr | Latin | 979 | 7 | 292 | 22 | 5484 | 0 | 0 / 0 |
| gb | Latin | 871 | 7 | 472 | 4 | 4040 | 5 | 0 / 0 |
| ge | Georgian | 34 | 68 | 18 | 39 | 239 | 86 | 32 / 11 |
| gr | Greek | 39 | 90 | 12 | 100 | 387 | 60 | 150 / 4 |
| hk | Han (lines Latin) | 33 | 91 | 14 | 100 | 255 | 100 | 0 / 0 |
| hr | Latin | 130 | 30 | 39 | 100 | 639 | 1 | 0 / 0 |
| hu | Latin | 352 | 38 | 133 | 100 | 1951 | 0 | 0 / 0 |
| id | Latin | 157 | 14 | 38 | 0 | 651 | 8 | 0 / 0 |
| ie | Latin | 48 | 0 | 17 | 0 | 237 | 59 | 0 / 0 |
| in | Latin | 877 | 1 | 724 | 1 | 8935 | 6 | 1 / 0 |
| ir | Arabic (Persian) | 57 | 96 | 26 | 100 | 448 | 93 | 30 / 1 |
| it | Latin | 661 | 6 | 310 | 0 | 4458 | 0 | 0 / 0 |
| jp | Han / kana | 1099 | 62 | 597 | 41 | 9073 | 98 | 202 / 407 |
| kg | Cyrillic | 8 | 12 | 3 | 0 | 46 | 17 | 35 / 6 |
| kr | Hangul | 135 | 50 | 85 | 27 | 1216 | 100 | 4 / 67 |
| kz | Cyrillic | 120 | 16 | 80 | 16 | 1065 | 14 | 891 / 95 |
| lt | Latin | 34 | 0 | 16 | 0 | 146 | 1 | 1 / 5 |
| lu | Latin | 50 | 32 | 16 | 100 | 116 | 1 | 0 / 0 |
| lv | Latin | 52 | 19 | 10 | 100 | 315 | 1 | 0 / 0 |
| ma | Latin / Arabic | 30 | 43 | 13 | 100 | 243 | 6 | 62 / 6 |
| md | Latin (some Cyrillic) | 18 | 28 | 15 | 27 | 186 | 4 | 3 / 0 |
| me | Latin | 6 | 67 | 3 | 100 | 46 | 4 | 0 / 1 |
| mk | Cyrillic | 8 | 100 | 6 | 100 | 107 | 94 | 5 / 0 |
| mx | Latin | 29 | 97 | 27 | 100 | 379 | 0 | 0 / 0 |
| my | Latin | 25 | 76 | 17 | 100 | 330 | 87 | 0 / 0 |
| nl | Latin | 288 | 1 | 97 | 2 | 1239 | 0 | 0 / 0 |
| no | Latin | 76 | 36 | 25 | 100 | 583 | 1 | 0 / 0 |
| nz | Latin | 32 | 6 | 15 | 13 | 157 | 20 | 0 / 0 |
| pl | Latin | 892 | 42 | 363 | 100 | 5766 | 1 | 9 / 2 |
| pt | Latin | 103 | 7 | 25 | 8 | 798 | 1 | 0 / 0 |
| ro | Latin | 202 | 64 | 86 | 98 | 2349 | 0 | 0 / 0 |
| rs | Cyrillic | 91 | 75 | 26 | 100 | 539 | 63 | 194 / 11 |
| ru | Cyrillic | 3546 | 12 | 930 | 42 | 14873 | 31 | 10215 / 3101 |
| se | Latin | 156 | 0 | 56 | 0 | 1013 | 0 | 0 / 0 |
| sg | Latin | 13 | 92 | 11 | 100 | 197 | 82 | 0 / 0 |
| si | Latin | 36 | 61 | 22 | 100 | 291 | 1 | 0 / 0 |
| sk | Latin | 174 | 2 | 70 | 0 | 1086 | 1 | 0 / 0 |
| th | Thai | 33 | 97 | 16 | 100 | 824 | 99 | 6 / 1 |
| tj | Cyrillic | 6 | 0 | 5 | 0 | 30 | 23 | 22 / 6 |
| tm | Latin / Cyrillic | 12 | 33 | 12 | 33 | 145 | 14 | 52 / 5 |
| tn | Arabic | 19 | 100 | 11 | 100 | 251 | 33 | 138 / 0 |
| tr | Latin | 145 | 30 | 41 | 78 | 1372 | 2 | 0 / 0 |
| tw | Han | 56 | 77 | 36 | 100 | 519 | 100 | 0 / 13 |
| ua | Cyrillic | 513 | 59 | 320 | 79 | 4199 | 63 | 1519 / 202 |
| us | Latin | 923 | 12 | 503 | 21 | 5745 | 1 | 0 / 0 |
| uz | Latin / Cyrillic | 49 | 20 | 37 | 11 | 335 | 48 | 74 / 13 |
| vn | Latin | 20 | 100 | 10 | 100 | 306 | 35 | 0 / 0 |
| xa | Cyrillic (Abkhaz) | 4 | 25 | 1 | 100 | 33 | 88 | 3 / 3 |
| xk | Latin | 5 | 80 | 3 | 100 | 48 | 10 | 0 / 0 |
| za | Latin | 98 | 5 | 53 | 9 | 532 | 3 | 0 / 0 |

## Sources for the gaps, cheapest first

**1. OSM `name:en` the build already had and dropped.** Gap stations with a matching OSM stop
(same node, or same name within 3 km) that carries `name:en`: cn 81, ru 60, jp 46, ua 23, gr 7,
by 5, rs 5. About 230 stations. These are station merges in `build_model.py` where the kept
station had no `name:en` and the merged one did. A fix in the build, no fetch.

**2. Wikidata English labels by the `wikidata` tag on OSM stops and route relations.**
`extract.py` keeps `wikidata` on stops and relations (`STOP_TAGS`, `REL_TAGS`; not on ways).
Gap items with a Wikidata id, and the share with an English label in a sample of 500 of the
2,198 ids (Wikidata started rate-limiting this IP partway through):

| cc | gap stations with QID | gap lines with QID | English label in sample |
|---|---|---|---|
| by | 737 | 18 | 20 of 24 |
| ru | 457 | 255 | 88 of 247 (many are "137 km" style or tram routes) |
| ua | 170 | 87 | 23 of 74 |
| bg | 100 | 15 | 16 of 17 (line labels are bare route numbers) |
| gr | 97 | 2 | 21 of 21 |
| cn | 97 | 44 | 68 of 71 |
| jp | 50 | 14 | 28 of 28 |
| rs | 28 | 2 | 3 of 5 |
| kz | 14 | 0 | 1 of 1 |
| ma, mk, dz, ir, pl, kr, tn, md | 5-11 each | 0-3 | mixed; dz 0 of 6 |

Estimated gain: by about 600 stations, cn about 90 + 40 lines, gr about 95, bg about 95, jp
about 50, ru about 160, ua about 50. Labels need the cleaning `ru_register.py --convert`
already does: strip "railway station", "halt", "Station", and check that the label reads as a
romanisation of the local name (`en_score`).

**3. Wikidata by a register's own code** (the Russian approach: ESR code P2815). Already used
for ru (3,419 register stops), ua and by (ESR as well). kz, uz, kg, tj, tm also use ESR
codes, and Wikidata items carry P2815 for them; kz has 891 gap stations and only 14 OSM QIDs,
so an ESR join is the only Wikidata route there. Expect a low hit rate (ru got 41% of
register stops with a passing label), so translit stays the main fill.

**4. Japan lines by name to Wikidata line items.** The colour fetch
(`data/raw/wikidata_colours_jp.json`) already holds Wikidata ids for Japanese lines with their
Japanese label. 124 of jp's 412 lines without English match one item by name exactly (kr: 21
of 67). Japanese line items almost always have an English label (28 of 28 in the sample).
One `wbgetentities` call per 50 ids. spec.md section 5 already plans this, matching on name
plus operator.

**5. Line names made from station names already in English.** CJK line names are often
"<place>線": 165 of jp's 412 gap lines have a stem that is a station whose English we have
(函館線 -> "Hakodate Line", 根室本線 -> "Nemuro Main Line"); kr 21 of 67 (중앙선 -> "Jungang
Line"; drop "Station" from the station name first, or 대구선 becomes "Daegu Station Line");
cn only 4, because Chinese line names abbreviate two places (京广, 沪昆). Russia's register
lines already do this from their two ends ("A — B"). With transliteration as a fallback for an
end with no English, the remaining 320 Russian "A — B" lines and 42 Ukrainian ones get a
name as well; the client does this today only because it transliterates the whole string.

**6. Tags not extracted yet.** No `.pbf` is left on disk, so these need either a re-extract
or a targeted Overpass fetch of the gap node ids (a few thousand nodes, cheap):
- `name:fr` for tn (138), ma (62), dz (49): Maghreb stations are commonly tagged in French,
  which is what their signs and timetables use in Latin script. Better than any Arabic
  romanisation.
- `name:ja-Hira` / `name:ja_kana` for jp: kana readings convert to Hepburn by rule.
- `int_name`, `name:latin`, `name:zh-Latn` / `name:zh_pinyin`, `name:ko-Latn`: low counts
  expected, but free in the same fetch.
- `name:ru` for kz, kg, tj, uz: many local names are Kazakh, Kyrgyz or Uzbek forms; Russian
  forms transliterate better with a plain table and match English-language sources.

**7. Registers' own English fields** are already used where they exist: kr (RAFIS, 영문역사명),
tw (TRA stations), hk (MTR), gr (`id_name`), sg, th (SRT), cn (Wikidata), ir (iranrail). No
register found for the CIS or Balkan gaps.

## Transliteration

The client already transliterates by a per-letter table. Problems found on test names:

| input | current output | should be |
|---|---|---|
| Λουτράκι | Loytraki | Loutraki (ELOT: ου = ou) |
| Ευαγγελισμός | Eyaggelismos | Evangelismos (ευ = ev, γγ = ng) |
| Μπράλος | Mpralos | Bralos (initial μπ = b) |
| Київ | Kiyiv | Kyiv (Ukrainian и = y, г = h) |
| Гомель | Gomel | Homiel (Belarusian г = h) |
| Търново | Trnovo | Tarnovo (Bulgarian ъ = a, щ = sht) |
| Ђевђелија | Ђevђeliјa | Đevđelija (Serbian letters missing) |
| Қарағанды | Қaraғandy | Qaraghandy (Kazakh letters missing) |
| ტყიბული | t'q'ibuli | Tqibuli (national system has no apostrophes; Georgian has no capitals, so the first letter needs upper-casing) |
| Երևան | Erevan | Yerevan (initial ե = ye; ու = u, not "ou") |

So the table needs to be per language, not per script. Pick the table from the region, or from
the letters present (і/ї/є/ґ = Ukrainian, ў = Belarusian, region bg = Bulgarian,
ђ/ћ/љ/њ/џ/ј = Serbian, ѓ/ќ/ѕ = Macedonian, қ/ғ/ү/ұ/ә/ө/ң/һ = Kazakh).

Recommended schemes. All of these fit in a small client-side table (under 2 KB each) and run
letter by letter, with a few context rules:

| language | scheme | context rules |
|---|---|---|
| Russian | BGN/PCGN (what English atlases use: "Yekaterinburg") or the passport/ICAO 2013 scheme (what RZD signage and tickets use: "Ekaterinburg") | BGN: е/ё = ye/yë at start, after a vowel, ъ, ь |
| Ukrainian | national 2010 (KMU 55) | є/ї/й/ю/я differ at start of word; зг = zgh |
| Belarusian | national 2023 Latin or BGN/PCGN Belarusian | г = h, ў = w (BGN) or ŭ |
| Bulgarian | Streamlined System (law, 2009) | ъ = a, щ = sht, final ия = ia |
| Serbian | Serbian Latin (Gaj), one letter for one | none; this is the country's own second alphabet |
| Macedonian | national/BGN | ѓ = gj, ќ = kj, ѕ = dz |
| Kazakh, Kyrgyz, Tajik, Uzbek, Turkmen | each country's official Latin alphabet (Kazakh 2021, Uzbek 1995, Turkmen 1993), or BGN/PCGN where none is settled (Kyrgyz, Tajik) | Uzbek and Turkmen Cyrillic map almost one to one onto the Latin forms already used beside them in our data |
| Greek | ELOT 743 (UN-adopted) | αυ/ευ = av/ev or af/ef before voiceless, ου = ou, γγ/γκ = ng, initial μπ = b, ντ = d |
| Georgian | national system 2002 | no apostrophes; capitalise the first letter |
| Armenian | BGN/PCGN 1981 | initial ե = ye, ո = vo; ու = u; և = yev |
| Korean | Revised Romanization | Hangul decomposes by arithmetic (U+AC00); needs the main sound-change rules (ㄴ+ㄹ = ll, final consonant before a vowel), about 50 lines of code. Only 4 stations and 67 lines need it |

Need data, not a table:

- **Chinese**: pinyin needs a character dictionary and has polyphones (重庆 Chongqing, not
  Zhongqing). Do it at build time with `pypinyin`, joined without tones and spaced per word
  the way station signs are ("Beijingnan"). Client-side libraries exist (`pinyin-pro`, about
  300 KB) but aren't needed for 545 stations.
- **Japanese**: kana converts by rule (Hepburn), and 208 line names have some kana. Kanji
  needs readings, and place-name readings are irregular, so a dictionary guesses wrong often.
  Use OSM `name:en` / `name:ja-Latn` / kana tags and Wikidata; do not guess readings.
- **Thai**: RTGS needs syllable parsing; only 6 stations, leave them.
- **Arabic and Persian**: vowels are not written, so any romanisation is a guess. Use
  `name:fr` (Maghreb), `name:en` (eg, ir), Wikidata. Do not transliterate.

## Generic words ("Ligne 10" -> "Line 10")

A phrase table on the start of a line name, applied after the English name is missing and
before transliteration: replace the generic prefix, keep the rest (number, route), and
transliterate the rest only if it is non-Latin. Measured on lines with no English name, a
table of about 60 prefixes matches **3,256 of 15,824**:

- ru 2,079 (Пригородный электропоезд 1,037, Трамвай 410, Пригородный дизельпоезд 279,
  Скорый поезд 239, Пассажирский поезд 106); ua 117; by 51; kz 18; bg 16.
- fr 277 ("Ligne de A à B" -> "A–B line" 169, Ligne 70); pl 82 (Tramwaj, Linia tramwajowa);
  hr 63 (Vlak); se 61 (Tåg, Spårvagn); es 55; it 53; br 38; de 102.
- cn 31 ("<city>地铁N号线" -> "<city> Metro Line N", where the city is a station we have in
  English), kr 8 ("N호선" -> "Line N"), tw 9 (區間 -> "Local", 自強 -> "Tze-Chiang", 太魯閣 ->
  "Taroko").

Examples of the table: Ligne/Linie/Linea/Línea/Línia/Linha/Linija/Lijn/Linja/Proga/Secția/
Linia kolejowa nr/Линия/Лінія/Линија/خط -> "Line"; Tram/Tramwaj/Tramvaj/Tramvai/Tranvía/
Spårvagn/Трамвай/Трамвај/Τραμ -> "Tram"; Zug/Treno/Tren/Trem/Vlak/Tåg/Juna/Pociąg/Поезд/
Потяг/Влак/Воз/Τρένο -> "Train"; Пригородный электропоезд -> "Suburban train"; Скорый поезд ->
"Express train"; Métro/Metrô/Метро/Μετρό -> "Metro"; "N号线/號線/호선" -> "Line N". Keep it in
one shared table (build-time or client), keyed by language, matched at the start only.

In Latin-script countries the rest of the missing English is mostly codes ("S5", "RE 7",
"TER C76") and proper names ("Tjustbanan", "Esk Valley Line"), which read fine as they are.

## Recommended order

1. **Fix the client table** (a few hours): per-language Cyrillic tables (ru, uk, be, bg, sr,
   mk, kk/ky/tg, uz), Greek digraphs, Georgian without apostrophes and capitalised, Armenian
   context rules. Improves about 14,300 gap stations that are already shown transliterated, and fixes about
   215 still showing raw letters (kz, rs, kg, tj, mk, xa).
2. **Generic-word table** for line names (3,256 lines, 2,079 of them Russian services).
3. **Recover dropped `name:en`** in the build's station merges (about 230 stations, no fetch).
4. **Wikidata labels by the OSM `wikidata` tags we already hold** (about 1,200 stations and
   100 lines; by, cn, gr, bg, ru, ua, jp), with the `en_score` romanisation check. Pace the
   requests: this IP got 429s from `wbgetentities` today.
5. **Japan and Korea lines**: Wikidata by name from the cached colour fetch (124 jp, 21 kr),
   then "<station>線" from station English (up to 165 jp, 21 kr; overlapping), then Hangul RR
   in the client.
6. **Chinese stations by pypinyin** at build time (545 stations; 81 also have a dropped
   `name:en`), and "N号线" lines.
7. **Overpass fetch of extra tags** for the remaining gap nodes: `name:fr` for tn/ma/dz,
   kana for jp, `name:ru` for Central Asia.
8. Thai, Arabic and Persian leftovers (about 270 stations): leave in the local script.

Note: ru_sources.md says "Nothing is transliterated by us" and spec.md section 5 says not to
"fix" a Japanese line name into a transliteration. The client's English mode now
transliterates Cyrillic at display time. That fits, as long as the build data keeps `e` /
`name_en` for real English names only and transliteration stays a display fallback.

## Filled 2026-10-07

Steps 3-7 of the order above, done without touching the build: `tools/english_names.py`
writes `dist/data/<cc>/names_en.json` (`{"st": {id: English}, "ln": {id: English}, "src":
{counts}}`, every country, empty where nothing was found) and `dist/data/names_en.json`
(all countries in one file, 80 KB, for the search index). The shipped `stations.json` /
`lines.json` are unchanged, so `e` / `name_en` there still hold only real English; pinyin and
Korean romanisation live only in names_en.json and its `src` counts say which is which.
Rerun: `python tools/english_names.py` (a few minutes, reads the data/proc pickles); add
`--fetch` to ask Wikidata for ids not yet in `tools/english_names_wikidata.json` (2,487 ids
cached, 10 batched SPARQL queries, one 429 on the way) and, once, for every Japanese line
item (`tools/english_names_jplines.json`, 18,274 items with Japanese labels and aliases,
English label, operators); `--overpass` asks overpass-api.de for the full tags of what the
first pass leaves (`tools/english_names_osm.json`: 509 nodes and 106 relations by id, and
368 stations by name within 1.5 km, 40 per query, 5 s apart; the server answered 504 and
429 often, the backoff and the kumi mirror got everything through); `--sample 20` prints a
spot-check. Needs `pypinyin` and `jieba` (word splitting for pinyin), both pip-installed
for the user.

The app side is not landed: the exact diff is `handoff_notes/english_names_index.diff`
(names_en.json fetched in `loadRegion` beside aliases.json, applied in `mergeRegion` before
the line-id fold and the search keys, `enFill` marking a filled name; the search index reads
the combined file). Local mode is unaffected: `stName` and `lineName` already put the local
name first.

**Added: 2,147 stations and 650 lines** (the first pass, without Overpass and the full
Japanese line list, gave 1,910 and 466). Only items whose local name has letters of a
non-Latin script and no English; Latin-script items were left alone (Azerbaijani ə counts as
Latin here, though the app's `LATIN` test says otherwise).

| source (`src` key) | stations | lines |
|---|---|---|
| OSM `name:en` the build dropped (own node, a folded OSM station, or a rail stop of the same name within 3 km) (`osm`) | 217 | 1 |
| Wikidata label via the OSM `wikidata` tag (`wd`) | 1,316 | 85 |
| Wikidata label by exact name and operator, jp/kr colour cache (`wdname`) | | 125 |
| Wikidata, every Japanese line item: label or alias, 線/本線 variants, operator-prefixed names (東武野田線 for 野田線) (`wd_ja`) | | 164 |
| "<station>線" / "<station>선", the station on that line (`stem`) | | 87 |
| Overpass: `name:fr` (tn, ma, dz, eg) (`fr`) | 230 | |
| Overpass: `name:en` / `name:fr` / `name:ja-Latn` on nodes and relations stops.pkl lacked (`osm_en`, `osm_full`) | 1 | 9 |
| Overpass: `name:ja-Latn` (`romaji`), kana readings by Hepburn (`kana`), `int_name` | 1, 3, 2 | |
| Japanese service names made of known words and stations ("普通 郡山<=>福島" -> "Local Koriyama – Fukushima") (`service`) | | 10 |
| pinyin (cn, tw) | 375 | 137 |
| Revised Romanization (kr) | 2 | 32 |

Per country, non-Latin items, "English" = has `e` / `name_en` (before -> after), "Latin" =
shows in Latin letters in English mode, counting what the client already transliterates:

| cc | stations | English | Latin | lines | English | Latin |
|---|---|---|---|---|---|---|
| by | 1077 | 93 -> 813 | 1068 -> 1077 | 163 | 14 -> 15 | 163 |
| cn | 10048 | 9503 -> 10048 | 9503 -> 10048 | 811 | 674 -> 811 | 674 -> 811 |
| jp | 9073 | 8871 -> 9017 | 8871 -> 9017 | 1087 | 680 -> 1055 | 680 -> 1055 |
| tn | 219 | 81 -> 217 | 81 -> 217 | 7 | 7 | 7 |
| dz | 84 | 35 -> 83 | 35 -> 83 | 4 | 0 -> 4 | 0 -> 4 |
| ma | 67 | 5 -> 60 | 5 -> 60 | 6 | 0 -> 6 | 0 -> 6 |
| eg | 347 | 304 -> 306 | 304 -> 306 | 35 | 34 -> 35 | 34 -> 35 |
| kr | 1216 | 1212 -> 1216 | 1212 -> 1216 | 135 | 68 -> 135 | 68 -> 135 |
| tw | 519 | 519 | 519 | 56 | 43 -> 56 | 43 -> 56 |
| ru | 14314 | 4407 -> 4597 | all | 3536 | 435 -> 461 | all |
| ua | 4121 | 2609 -> 2752 | 4120 | 504 | 302 -> 321 | all |
| gr | 381 | 231 -> 322 | all | 39 | 35 | all |
| rs | 530 | 340 -> 369 | all | 41 | 30 | all |
| bg | 870 | 532 -> 543 | all | 58 | 39 | all |
| kz | 1018 | 149 -> 161 | all | 114 | 19 | 113 |
| mk, pl, md, ir, in, th | | +4, +4, +2, +3, +1, +1 | | | pl +1 | |
| all 73 | 46129 | 30871 -> 33018 | 45048 -> 45998 | 6842 | 2553 -> 3203 | 6200 -> 6803 |

Still not in Latin letters in English mode: 131 stations and 39 lines, down from 1,081 and
642 before this work.
- jp 56 stations, 32 lines. The stations are N02 stops whose OSM nodes carry no English,
  romaji or kana (Hiroshima and Kumamoto tram stops such as 本町一丁目, Hokkaido halts such
  as 抜海, 雄信内); kanji are not read by guess. The lines are mostly funiculars and cable
  cars (高尾鋼索線, 十国鋼索線), bare "本線" / "支線" / "N号線" with no operator match,
  katakana train names (ソニック, μSKY), and 山陰線, which matches more than one Wikidata item.
- eg 41: Egyptian stations carry no `name:en`, `name:fr` or `int_name` in OSM; ir 14, th 5:
  likewise. These stay in the local script (no transliteration of Arabic, Persian or Thai).
- Russia's 9,700 stations without English still show transliterated; only 714 had a
  Wikidata id.

Spot-check, 20 random per source (all 86 Wikidata line labels read). Wrong or doubtful:
- Wikidata lines: Трамвай Т (Zhytomyr) -> "Zhytomyr tramway" (the system's item, not the
  route); Автозаводско-Нагорная линия -> "Avtozavodskaya" (label cut short); ma "TNR Kenitra-
  Casablanca" -> "Tangier - Casablanca"; two Russian narrow-gauge lines share "Apsheronsk
  narrow-gauge railway". Rejected before writing: tram routes labelled "Line 46" (the app's
  "Tram 46" says more), class labels in lower case ("limited express of Meitetsu"), named
  trains tagged with their line's item (つがる -> "Ohu North Line"), labels that drop the
  train number ("Express" for Скорый поезд 003/004).
- Wikidata stations: Локомотив -> "Locomotive" (translated, not romanised); a few Belarusian
  labels are in Łacinka ("Cimkavičy") beside BGN neighbours. The romanisation check
  (ru_register's `en_score`, a pinyin and a Greek version) turned down about 430 labels, mostly
  right ("135 km" on Химфармзавод, "Yasnaya Polyana" on Ветеранская), some real English
  lost ("Thebes" for Θήβα, "Luga" for Луга I).
- OSM name:en: one OSM typo kept ("Huquishidigongyuan" for 虎丘湿地公园); 銭座町 ->
  "Stadium City North" from its own node, maybe the 2024 renaming, unchecked; the Hanzomon /
  Den-en-toshi through service gets OSM's odd "Tokyo Metro - Denentoshi bypass line".
- pinyin: polyphones read the common way, not the place's (长店堡 -> "Zhangdianbao", likely
  Changdianbao); jieba splits some names badly (庐阳经开区 -> "Luyangjing Kaiqu",
  华侨城恐龙水世界 -> "Huaqiaocheng Konglongshui Shijie"). Line abbreviations of two places
  come out as one word ("Midong Line" for 密东线), which is how China Railway writes them.
- Korean: compounds stay one word where the parts are not stations ("Busansinhang Line",
  "Jungbunaeryuk Line", "Cheonanjikgyeol Line"); official forms would split them.
- Stems: none wrong in the sample once the station had to be on the line itself (before that,
  北条線 took Niigata's 北条 "Kitajo").

Second pass (Overpass, every Japanese line item), 20 random per new source and all of the
smaller ones read:
- `fr` (230): right throughout the sample; these are French forms, as Maghreb signs use
  ("Hammam Chatt", "Béja", "Cinq Maisons" for الديار الخمس, "27 février 1962", "Martyrs").
  They are not English, which is fine for a Latin-letter fallback but worth knowing.
- `wd_ja` (164, 30 read): right, with operator names where Wikidata puts them ("Tōkyū
  Den'en-Toshi Line", "Hankyū Kyōto Main Line" for 京都線). Doubtful: 山鼻線 -> "Sapporo
  Streetcar" (the system's item; 山鼻線 is one of its lines), 西神延伸線 -> "Seishin-Yamate
  Line" (the extension is part of it). A bare "Main Line" for 本線 was refused; seven 本線
  lines got their operator's ("Keisei Main Line", "Hanshin Main Line").
- `kana` (3) and `romaji` (1): 乙原 -> "Otobaru", 田辺島通 -> "Tabeshimadōri", 雲泉寺 ->
  "Unsenji", 山頂 -> "Sanchō". Hepburn reads ou/oo as ō, which misreads a compound across a
  word boundary; none in these.
- `service` (10): "Local Koriyama – Fukushima", "Through Limited Express", "Uzushio",
  "Local train" (OSM's "Train 普通"). Fine.
- `osm_full` lines (9): Algerian and Moroccan routes by `name:fr` ("Annaba ↔ Chihani",
  "Fes - Marrakech"); ふじかわ -> "Fujikawa" by `name:ja-Latn`.
