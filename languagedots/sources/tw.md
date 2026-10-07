# Taiwan: 2020 census, language learned earliest in childhood, by township

Drawn as `tw` (countries/tw.py). The queue's row is `cn-tw`, Natural Earth's ISO_A2 for Taiwan;
`countries.py` loads two-letter files only and religiondots draws Taiwan as `tw`, so the files are
`tw`, and `claim.py done tw` added a separate `tw` row (drawn). The `cn-tw` row is left for the
supervisor to retire.

## Source

DGBAS (Directorate-General of Budget, Accounting and Statistics), 109年人口及住宅普查 (2020
Population and Housing Census, reference date 8 November 2020), 總報告統計結果表 > 縣市別報告統計表:
one page per county or city on www.stat.gov.tw, each with ~46 XLSX tables, two of them on language
by township (按鄉鎮市區別分) for the resident population of Taiwanese nationality aged 6 and over:

- table 6 (`t006`), 使用語言情形: the language used most now (主要, one answer) and the second
  (次要);
- table 7 (`t007`), 兒時最早學會語言情形: the language learned earliest in childhood (one answer).

Five answers each: 國語 (Mandarin), 閩南語 (Taiwanese / Southern Min), 客語 (Hakka), 原住民族語
(indigenous languages), 其他語言 (other) and 不知或無 (does not know, or none). Shares per hundred
to one decimal, with the township's 6+ count. The county pages' `s` ids are pinned in
`sources/tw_census.py`; files land in `data/raw/tw/<s>_t00{6,7}.xlsx`.

The coverage sweep found only SEGIS's county dataset and thought the township table might not
exist. It does: the county reports carry every township. The census was register-based, with the
detailed items (language among them) asked in a cluster sample of about 16% of census areas and
weighted to the register, so township shares are estimates. Licence: the tables are public
downloads with no registration; DGBAS's site terms were not read (the SEGIS copy is under the
Government Open Data Licence, per the coverage sweep).

**Why table 7, not table 6.** The map is of first languages; "learned earliest in childhood" is
the mother-tongue question, single answer, and the census's own. Table 6's main language is what
people use now and would put Taiwan at 66% Mandarin. Both are normalised (`question` = earliest /
main in `data/normalized/tw.csv`); `note_public` gives the main-language figures.

ws.dgbas.gov.tw's certificate chain does not verify from here (curl exit 60); the fetch skips
verification for those public files only.

## Checks (`python sources/tw_census.py`), all asserted

- every file's title names its county and table;
- townships sum to the county row in both tables; the two tables agree on every township's
  population; every township's shares sum to 100 within 0.6;
- 368 townships; 21,784,369 resident nationals aged 6+, equal to the sum of SEGIS's county
  dataset (MOI's copy of the same census);
- SEGIS's county main-language shares equal table 6's county rows exactly (max difference 0.00).
  This is a transcription check, not an independent one: SEGIS also publishes shares.
- National results from the townships (count = share x population, so each township carries
  up to +-0.05% of its population of rounding): earliest 閩南語 53.2%, 國語 40.4%, 客語 4.6%,
  原住民族語 0.8%, 其他語言 0.8%, 不知或無 0.3%; main 國語 66.4%, 閩南語 31.7%, 客語 1.5%,
  原住民族語 0.2%, 其他 0.2%. Both match DGBAS's published national figures (earliest 53.2 /
  40.4 / 4.6; main 66.4 / 31.7, the 總報告 summary and SEGIS).

## Labels (`taxonomy/tw2020.py`, `taxonomy/tree.d/tw.txt`)

- 國語 `mandarin`, 閩南語 `min_nan`, 客語 `hakka` (existing nodes, repeated in the fragment with
  identical labels and colours).
- 原住民族語 on a new areal group `austronesian.taiwan`, "Taiwan indigenous languages". Not a
  genealogical node: Formosan is several primary branches of Austronesian, and Yami (Tao) is
  Malayo-Polynesian (Batanic), so the narrowest true node would be `austronesian` itself; an areal
  group, as `australian` and `americas_other` are, keeps the reading "Taiwan indigenous". Its
  members are Glottolog's fifteen languages (Amis, Atayal, Paiwan, Bunun, Puyuma, Rukai, Tsou,
  Saisiyat, Yami, Thao, Kavalan, Seediq, Sakizaya, Kanakanavu, Saaroa; all checked in
  `data/raw/glottolog/languages.csv`, all aust1307; Truku and Sediq are Seediq dialects there)
  plus Truku (below). Since ask 012 the members are drawn; the group keeps only people whose
  register entry names no people. Colour 0.76 0.15 150.
- 其他語言 on the root `other`: the table's note says it holds Taiwan Sign Language, "dialects of
  other places", foreign languages and other countries' sign languages.

## Naming the indigenous languages (ask 012, Anita 2026-10-05)

The census's 原住民族語 (169,306 people from the township shares) is shared in each township
across the indigenous peoples registered there, in proportion, every resulting row tier
`derived`; the census count per township is unchanged (asserted in `countries/tw.py`).

- Source (`python sources/tw_cip.py [--fetch]` -> `data/normalized/tw_peoples.csv`): Council of
  Indigenous Peoples, 原住民人口數統計資料, October 2020 release, 10910台閩縣市鄉鎮市區原住民族
  人口-按性別族別.xls (the household register's RCRPC1F0 report 7: resident indigenous population
  by township, people and sex, all ages). End of October is the month-end nearest the census
  date. Only the sheet's first block (不分平地山地, all statuses) and its 計 rows are read.
- Checks, asserted: national 575,967; the 16 peoples plus 尚未申報 sum to the total on every row;
  townships sum to their county, counties to the nation, in every column; 368 townships, joined
  both ways to the census keys with no fold. Every township with census indigenous speakers has
  registered indigenous residents, so nothing stays on the group for want of a register row.
- 尚未申報 (10,622 registered, not yet declared) stays on `austronesian.taiwan`: 2,982 people.
- Drawn: Amis 66,031, Paiwan 31,802, Atayal 24,953, Bunun 17,189, Truku 8,651, Rukai 4,245,
  Puyuma 3,825, Seediq 3,127, Saisiyat 2,101, Tsou 1,994, Yami 1,175, Kavalan 484, Sakizaya 299,
  Thao 184, Saaroa 139, Kanakanavu 124. The smallest draw no dot at 1:1000.
- What the proxy assumes: a township's indigenous-language speakers split like its registered
  peoples (all ages, against the census's 6+). A people that keeps its language better than its
  neighbours is under-drawn, and one that keeps it less over-drawn, but only within a township; in the mountain and east-coast townships one people dominates and
  the split is near-certain. Mixed urban districts are where it is weakest, and there the census
  figures are small.
- Truku and Seediq: the register counts 太魯閣 Truku (32,717) and 賽德克族 Seediq (10,602) as two
  peoples, Taiwan recognises two languages, and Glottolog has one (Seediq, taro1264, with Truku as
  a dialect). Two leaves, sibling to each other, with that said in `tree.d/tw.txt`.

## Not drawn (`gap`)

不知或無, 60,672 people (0.3%). Outside the question: children under 6 and residents without
Taiwanese nationality (migrant workers, foreign spouses not naturalised).

## Geography (`python sources/tw_geo.py` -> `data/geo/tw/`)

- Units: MOI/NLSC TOWN_MOI_1120317 (2023-03-17). The official download (tgos.tw via data.gov.tw
  dataset 7441) answers 403 outside Taiwan, as it did for religiondots; the file comes from the
  kiang/taiwan_basecode GitHub mirror of the NLSC release. The township set has not changed since
  2014. Join on county|town name in the census's own characters: 368 both ways, no fold needed.
  Unit id = TOWNCODE (`tw_lookup.csv`).
- Placement: Kontur 2023, religiondots' TW extract read in place, centroid join in TWD97 TM2;
  343 offshore centroids snapped within 1 km (42,819 people), 4 hexes dropped (1,128). Every
  township has populated hexes. Median township 54.7 km2 (74 hexes); smallest 0.92 km2.
- Kontur against the census 6+ per township (national ratio 1.098): p10 0.81, median 1.01, p90
  1.38; log r 0.981 against 0.153 for the best of 500 shuffles. Outside a factor of 3: Taipei
  Zhongshan 0.28 and Keelung Zhongshan 4.03 (Kontur holds 67,085 and 180,722; the polygons are
  right, so Kontur looks to have put some of Taipei's Zhongshan into Keelung's, a same-name
  error in its calibration), and Taoyuan Fuxing 6.88 (mountain district). Since the census
  decides every township's count and Kontur only places inside a township, none of this moves a
  dot between townships; Keelung Zhongshan's dots sit a little more on its densest hexes.
- Scatter: Kontur cap 2 blocks, registered real cores; water clip 2.41% of polygon area; 21,723
  dots at 1:1000.

## Worth knowing (not drawn)

- The 2010 census asked the same items; the Hakka Affairs Council surveys (2016, 2021) give Hakka
  identity by township, a different question.

## Calls someone might reverse

- Table 7 (earliest learned) rather than table 6 (main language now).
- `austronesian.taiwan` as an areal group with unused member nodes, rather than `austronesian`.
- Files as `tw`, not `cn-tw`.
- Indigenous split by registered people at October 2020 (not December 2020, or the lowland /
  mountain status blocks); Truku and Seediq as two leaves rather than one Seediq.

## Colours

Mandarin #ec6d3a, Min Nan #9c443b, Hakka #bc516f, Taiwan indigenous #61cb7c (washed #a7c8ac).
Indigenous leaves hand-picked in `tree.d/tw.txt` (hues 95-260, Austronesian's side of the
wheel). Neighbour pairs in OKLab: Amis-Paiwan 0.115 (first try was 0.054, Paiwan moved to a pale
lime), Amis-Truku 0.113, Tsou-Saaroa 0.113, Amis-Puyuma 0.144, Bunun-Rukai 0.146, Atayal-Seediq
0.155; every other bordering pair over 0.17. Amis to the washed group colour 0.076.

OKLab distance Min Nan to Hakka 0.097, the closest pair, and they share the Taoyuan-Hsinchu-
Miaoli and Pingtung plains; left as is because both nodes are drawn in hk, sg and others.
