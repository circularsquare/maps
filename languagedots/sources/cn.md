# China: 2020 census nationality, read as language with retention shares, 2,846 counties

Drawn 2026-10-05 (session edd42a8c-cn); Chinese split into dialect groups the same day (session
edd42a8c-cnd, §6). `sources/cn_ethnic.py`, `sources/cn_geo.py`, `sources/cn_dialect.py`,
`taxonomy/cn2020.py`, `taxonomy/tree.d/cn.txt`, `countries/cn.py`.

**1,409,778,724 people (the 31 provinces' census total), 2,846 counties, 76 nodes, 1,409,741
dots at 1:1000, every row `derived` except the 119.4M Chinese speakers moved onto their home
province's dialects (§8, `modelled`). 1,337,961,619 (94.9%) speak Chinese, drawn on their county's
main dialect group (19 Sinitic leaves); 71.8M on minority languages. Every county total is the
2020 census's, the 15 estimated provinces' since §9.**

## 1. The table

China's census asks nationality (minzu), never language: tier D, built under AGENT_BRIEF §2's
ethnicity rule. The county x 56-nationality table is **chinaethnicity's** (maps/chinaethnicity,
Anita's 2020 nationality dot map), read-only from its `data/work/`:

- `leaves_2020.csv`: 16 provinces measured, each province's own 2020 census yearbook table 1-4
  (population by region, sex and nationality), county rows. Parsed and checked there.
- `leaves_fallback.csv`: 15 provinces estimated (Hebei, Liaoning, Hunan, Sichuan to 2020
  prefecture rows; Tianjin, Shanxi, Anhui, Jiangxi, Guangdong, Chongqing, Guizhou, Tibet,
  Shaanxi, Gansu, Xinjiang to 2020 province totals) from the 2000 census county pattern,
  Anita's approved method there. 712,996,895 people, half the country. Of each group's 2020
  people, the share in estimated provinces: Uyghur 99%, Kazakh 99%, Bouyei 84%, Dong 81%,
  Tibetan 76%, Miao 72%, Tujia 69%, Yi 45%, Hui 39%, Zhuang 12%. Since 2026-10-06 these
  provinces' county totals are the 2020 census's and only the nationality mix is estimated (§9).

Each row's `adcodes` gives its split over county polygons; explicit shares (development zones)
are applied as given, and the few bare code lists (rows on several polygons) are split by the
placement grid's population, where chinaethnicity uses ASPECT's. Checks in `cn_ethnic.py`: 16 +
15 provinces and none in both, 58 group columns, every row's groups add to its total, every
adcode is a polygon of chinaethnicity's `counties.gpkg`, no county filled from two provinces,
country total 1,409,778,724 against 1,409,778,723 (float rounding in the estimated rows). The
national 1,411,778,724 adds 2 million serving military, counted nationally only (`gap`).

## 2. Retention (§2: check retention first)

One source for every group, so they are measured alike: the **National Language Resource
Monitoring and Research Center, Minority Languages** (Minzu University of China), series
"少数民族语言文字使用、发展和保护情况", one page per nationality, January 2024,
`https://nmlr.muc.edu.cn/info/1119/2132.htm` (Mongol) to `2682.htm` (undetermined), each page
linked to the next. All 55 pages read. Each gives a **site** share ("在调研点使用本民族语言的人数
占...总人口平均比例", the average over its case-study sites), usually a **national speaker count**
(some dated, some from the Language Resources Protection Project, 语保工程), and the 2010 census
population.

**The rule: retention = the lower of the site share and (national count / 2010 population).**
The sites sit in each group's own areas, so the site share runs high for a dispersed or shifting
group: She are 89.2% at the sites while the same page says about 1,000 people speak She at all;
Gelao 59.9% against 6,400 speakers; Tujia 19.9% against 170,000. A count above the site share is
usually stale (counted when the group was smaller: Korean 1.92M against 1.83M people) or takes in
speakers of other nationalities (Pumi). Where no national count, the site share alone. Hui and
Manchu are 0 because their pages say so ("回族主要转用汉语"; "满族通用汉语汉文", about 500 Manchu
speakers in 1982). The rest of each group goes on Chinese.

| nationality | 2020 people | drawn on | speaks it | basis |
|---|---:|---|---:|---|
| Han | 1,284,446,389 | Chinese (unnamed) | 100% | |
| Zhuang | 19,568,546 | Zhuang | 76.7% | site |
| Uyghur | 11,774,538 | Uyghur | 97.1% | site |
| Hui | 11,377,914 | Chinese | 0% | page; Hainan: Tsat 35.1% (6,000 / Hainan's 17,089) |
| Miao | 11,067,929 | Hmongic (unnamed) | 73.6% | site (count 7.11M / 9.43M = 75.4%) |
| Manchu | 10,423,303 | Chinese | 0% | page |
| Yi | 9,830,327 | Loloish (unnamed) | 91.3% | site |
| Tujia | 9,587,732 | Tujia | 2.0% | 170,000 (1980s) / 8,353,912 |
| Tibetan | 7,060,731 | Tibetic (unnamed) | 79.5% | site; Sichuan: Qiangic 14.6% first (below) |
| Mongol | 6,290,204 | Mongolian | 45.8% | 2,740,000 / 5,981,840 (site 85.3%) |
| Bouyei | 3,576,752 | Bouyei | 45.2% | site (count 2.0M / 2.87M = 69.7%) |
| Dong | 3,495,993 | Dong (Kam) | 41.7% | 1,200,000 / 2,879,974 (site 70.5%) |
| Yao | 3,309,341 | Hmong-Mien (unnamed) 86.2%, Bunu 9.6% | 95.8% | site; Bunu 268,000 / 2,796,003 |
| Bai | 2,091,543 | Bai | 67.2% | 1,300,000 / 1,933,510 |
| Hani | 1,733,166 | Hani | 88.7% | site |
| Korean | 1,702,479 | Korean | 84.2% | site |
| Li | 1,602,104 | Li (Hlai) | 68.3% | 1,000,000 / 1,463,064 |
| Kazakh | 1,562,518 | Kazakh | 92.7% | site |
| Dai | 1,329,985 | Kra-Dai (unnamed) | 94.7% | site |
| Dongxiang | 774,947 | Dongxiang (Santa) | 40.2% | 250,000 / 621,500 |
| Lisu | 762,996 | Lisu | 95.8% | site |
| She | 746,385 | Chinese | 0% | ~1,000 speakers; Guangdong: She 2.4% (1,000 / 42,080) |
| Gelao | 677,521 | Gelao | 1.2% | 6,400 / 550,746 |
| Lahu | 499,167 | Lahu | 82.3% | 400,000 / 485,966 |
| Sui | 495,928 | Sui | 100% | site |
| Wa | 430,977 | Wa | 83.8% | 360,000 / 429,709 |
| Naxi | 323,767 | Naxi | 99.8% | site |
| Qiang | 312,981 | Qiang | 19.4% | 60,000 / 309,576 (page: 306,000 use Chinese) |
| Tu | 281,928 | Tu (Monguor) | 64.8% | site |
| Mulao | 277,233 | Mulam | 81.6% | site |
| Kyrgyz | 204,402 | Kyrgyz | 100% | site |
| Xibe | 191,911 | Xibe | 26.0% | 49,500 / 190,481 |
| Salar | 165,159 | Salar | 82.8% | site |
| Jingpo | 160,471 | Kachin languages (unnamed) | 97.5% | site |
| Daur | 132,299 | Daur | 62.9% | site |
| Blang, Maonan, Deang, Jino | 300,000 | own leaves | 89-99% | site |
| Tajik | 50,896 | Iranian (unnamed) | 72.2% | site |
| Pumi, Achang, Nu, Evenki | 160,000 | own leaves (Nu: Sino-Tibetan) | 57-80% | site |
| Gin | 33,112 | Vietnamese | 21.4% | site |
| Bonan | 24,434 | Bonan | 49.8% | 10,000 / 20,074 |
| Russian | 16,136 | Russian | 48.2% | site |
| Yugur | 14,706 | other (unnamed) | 70.5% | site |
| Uzbek, Tatar | 16,286 | own leaves | 17.1%, 13% | site |
| Monpa | 11,143 | Monpa 12.3%, Tshangla 66.3% | 78.6% | 1,300 and 7,000 / 10,561 |
| Oroqen, Derung, Hezhen | 21,851 | own leaves (Hezhen: Nanai) | 37%, 76%, 4.1% | site; Hezhen 220 (1982) / 5,354 |
| Lhoba | 4,237 | Tani (unnamed) | 100% | site |
| Gaoshan | 3,479 | Taiwan indigenous (unnamed) | 100% | no survey; §2 default |
| undetermined | 836,488 | other (unnamed) | | languages not known |
| naturalised | 16,595 | other (unnamed) | | |

**Sichuan's Tibetans**: the Tibetan page counts Qiangic and rGyalrongic languages spoken by
Tibetans, all in Sichuan: Baima 10,000, Ergong 45,000, Ersu 20,000, Guiqiong 6,000, Gyalrong
100,000, Lavrung 10,000, Minyak 10,000, Namuyi 5,000, Queyu 7,000, Shixing 1,800, Zhaba 20,000
(1995-2008) = 234,800, over Sichuan's 2020 Tibetans 1,604,629 = 14.6%, on Qiangic unnamed; the
79.5% Tibetan share applies to the rest there.

Every share is national (or provincial for the four splits), so each county of a nationality
gets the same split: Mongols in Liaoning get Inner Mongolia's 45.8%, Tujia speakers are spread
over all Tujia counties though they live in western Hunan and Laifeng.

## 3. Tree and calls

New nodes in `tree.d/cn.txt` (its header carries the Glottolog checks): groups Loloish, Qiangic,
Hmongic; leaves Lisu, Lahu, Hani, Jino, Qiang, Pumi, Tujia, Bai, Naxi, Derung, Tshangla, Achang,
Salar, Dongxiang, Tu, Daur, Bonan, Xibe, Oroqen, Bouyei, Dong (Kam), Li (Hlai), Sui, Mulam,
Maonan, Gelao, Bunu, She, Wa, Blang, De'ang, Tsat. Hand-picked colours: Naxi (generated, it sat on
Lisu's blue) and Dong (generated, it sat between Zhuang's and Mulam's light greens).

- Han (and everyone else put on Chinese) on the county's main dialect group, §6. Before that
  session they sat on `sinotibetan.sinitic` itself ("Chinese, language not named").
- Multi-language nationalities on group nodes (cn2020.py's docstring lists them). Nu goes as far
  up as Sino-Tibetan; Yugur on `other`, as no node holds Turkic and Mongolic.
- **Jingpo moved to the Kachin group** (2026-10-06, group-node audit, 5d7dac7e-aud). Jingpo
  (160,471 drawn) sat on the Sino-Tibetan root. mm.txt's `sinotibetan.kachin` ("Kachin languages
  (Jingpho, Zaiwa, Lhaovo, Rawang)", with Lashi as `lachik`) holds every language the nationality
  speaks, so it is the narrowest node (spec §3); the line is repeated in cn.txt. Still a group
  node, still drawn as "not named": the census counts a nationality, not Jingpho against Zaiwa.
  Re-scattered; check_country ok, total unchanged.
- Uzbek and Tatar non-speakers go on Chinese, per §2's default. In Xinjiang they more likely
  speak Uyghur or Kazakh; no source gives that share.
- Undetermined nationalities on `other`: the largest (Chuanqing in Guizhou) speak Chinese, others
  Hmongic or Austroasiatic; nothing splits them.
- Small languages of larger nationalities not drawn: Mongols' Kazhuo (4,800, Yunnan) and Tuvan
  (3,000), Bouyei's Mo (10,000), Yao's Lakkja, Pa-Hng, Younuo and Jiongnai, Hui's Kangjia (487).

## 4. Geography

`sources/cn_geo.py` re-keys religiondots' `cn_grid_3km.gpkg` (166,958 cells with `pop`, read-only)
to chinaethnicity's 2,848 county polygons by cell centroid, because religiondots' grid is keyed to
its own 2,793 county ids and 55 of chinaethnicity's codes are newer districts. 7 cells (223,312
people) fall outside every county and are dropped; 2 counties with no cell centroid get their own
polygon at pop 1 (140303 Yangquan Kuangqu, 460302 Nansha). Kinmen (350527) and Nansha have no
people, as in chinaethnicity (Sansha's people on the Paracels, Anita's call there). Not raw
Kontur, so the cap check does not apply. The scatter placed every row on population.

## 5. Room for improvement

- A real language table: China's 2020 census has none. The 1990s survey book
  《中国少数民族语言使用情况》 (1994) and Sun Hongkai's 《中国的语言》 (2007) give speakers per
  language and would split Tibetan, Yi, Miao, Dai and Yao into named languages; neither is open.
- Chinese varieties: drawn by county from the Language Atlas (§6), as areas, not counts. Any
  survey of home dialect by county would replace the whole-county rule; none is open.
- Retention by province or county instead of one national share (a dispersed Mongol in Liaoning
  is not an Inner Mongolian herder). The nmlr pages do not break it down.
- 15 provinces' nationality mixes are 2000 patterns (their county totals are 2020's, §9);
  chinaethnicity's queue (`provinces.md`) is the place to fix that, and this map follows by
  rerunning `sources/cn_ethnic.py`.

## 6. Chinese dialect groups (Anita, 2026-10-05; session edd42a8c-cnd)

Anita: draw China's Chinese speakers by dialect group, "it won't be accurate in terms of quantities
but it'll help a lot in terms of representation". Approved proxy: everyone the retention step put
on Chinese (Han, Hui, Manchu, She, and the non-speaking share of every other nationality) goes on
the **county's main Sinitic group**, the whole county as one group, `derived`. Minorities of other
dialects inside a county vanish (Hakka villages in Cantonese counties, Mandarin-speaking migrants
in Guangdong and Shanghai until §8, Min and Hakka pockets in Zhejiang and Guangxi), so the counts are a
picture of where each group is the local speech, not of its speakers. Hakka come out at 27M
against the usual 40-50M for that reason.

**The table.** `dan-qqq/Chinese_dialect_distance` on GitHub, `data/CH_dialect_county_compl.csv`
(raw copy `data/raw/cn/dialect/`): 2,855 county-level units (modood's codes, about 2018-2019),
each with one 方言大区 / 方言区 / 方言片, compiled by hand from the **Language Atlas of China, 2nd ed.
(2012)** and the Chinese Dialect Dictionary (1991). It never marks a county as mixed, so no county
is split between groups (until §7). 2,712 rows are Sinitic; 143 carry a minority language (99 Tibetic, 33
Mongolian, 7 Turkic...). **No licence file**: it is a compilation of the atlas's published
classification, which is fact rather than expression; recorded here, not asked. Other leads,
not used: Crissman's Digital Language Atlas of China (Harvard Dataverse doi:10.7910/DVN/OHYYXH,
shapefiles) and the 1990-county coding of the atlas (doi:10.7910/DVN/QPUONU); both on 1990 units,
a harder join. A wrong row seen in passing (harmless here, as only Sinitic rows are used): 五家渠
(an XPCC city, nearly all Han) is filed as Uyghur and 格尔木 as Mongolian; both take the fallback.

**The join** (`sources/cn_dialect.py`, asserted both ways against chinaethnicity's 2,848 polygons):
2,612 polygons by code; 91 through yescallop/areacodes' code history (`result.csv`, 新代码 for
changes from 2010: renamed counties, Yichun's 2019 districts, Hangzhou's 2021 merges, Sanya's
districts), 5 of them finished by name inside the prefecture because chinaethnicity's polygons
carry DataV's codes (孟津区 410306, officially 410308; 三元区, 崇川区, 繁昌区, 南沙区); 9 polygons
carved out of a county since are given their parent's row (`CARVED`: Shenzhen's 龙华, 光明, 坪山,
临平, 龙港, 红谷滩, 西沙, 加格达奇, 胡杨河). No polygon received two old counties of different groups.
Every table code reaches a polygon except Kinmen (no people).
**Counties with no Sinitic group** (the table gives their minority language), 145 polygons, 3.4M
Han: the Han-weighted main group of the other counties in the prefecture (49), else the province
(19: Qinghai's Tibetan prefectures go to Zhongyuan Qinlong, Xining's), Alxa by hand to Lan-Yin
(Yinchuan's side, not Inner Mongolia's Jin), and Tibet, with no Sinitic county, to Southwestern
Mandarin (74), since most of its Han come from Sichuan and Chongqing.

**Level and nodes** (`cn2020.DIALECT`): Mandarin whole, Min by the table's subgroups (the atlas's
闽南区, 闽东区...), every other group whole; the 片 below are not drawn. All eight Mandarin groups
(区: Southwestern, Zhongyuan, Ji-Lu, Northeastern, Jiang-Huai, Jiao-Liao, Beijing, Lan-Yin) go on
the existing `sinitic.mandarin` leaf. They had their own leaves for a few hours; Anita, the same
day: "we probably shouldn't split all the different Mandarins", so they were folded back and
their cn.txt lines removed (Laos keeps its own `mandarin_southwestern` for the Hor, defined in
la.txt). The fallbacks above still pick an atlas group, which now draws as Mandarin. New flat
leaves under Sinitic: Jin, Gan, Xiang, Hui, Pinghua and Tuhua, Min Bei, Min Zhong, Pu-Xian,
Shao-Jiang, Leizhou, Hainanese (Glottolog codes in cn.txt). Reused: Mandarin, Wu, Hakka, Min Nan,
Min Dong; Teochew for Min Nan in Shantou, Chaozhou and Jieyang (潮汕片; Shanwei's Hailufeng stays
Min Nan); Sze Yap for Yue's 四邑片; all other Yue (广府, 高阳, 勾漏, 邕浔, 钦廉, 吴化, and 儋州话) on
Cantonese. The bare `sinitic` draws nothing in China now.

| group | people | | group | people |
|---|---:|---|---|---:|
| Mandarin | 845,297,035 | | Teochew | 16,750,746 |
| Wu | 119,992,415 | | Min Dong | 11,710,960 |
| Cantonese (Yue) | 96,491,613 | | Hainanese | 8,104,361 |
| Jin | 70,578,118 | | Leizhou Min | 6,866,180 |
| Gan | 52,644,312 | | Sze Yap | 5,698,076 |
| Xiang | 37,183,042 | | Pinghua and Tuhua | 4,162,009 |
| Hakka | 27,095,618 | | Hui, Pu-Xian, Min Bei, Min Zhong, Shao-Jiang | 10,377,293 |
| Min Nan | 25,009,844 | | | |

Mandarin's atlas groups, for the record (all on Mandarin): Southwestern 242,913,418, Zhongyuan
228,394,828, Ji-Lu 106,703,642, Northeastern 89,095,480, Jiang-Huai 82,150,390, Jiao-Liao
41,407,642, Beijing 33,036,101, Lan-Yin 21,595,534.

Sum 1,337,961,619, equal to what sat on Chinese before. Colours: all hand-picked, checked over
shared county borders and against Uyghur, Kazakh and Mongolian (`COLOURS.md`, 2026-10-05 China).

Calls someone might reverse: Yue other than Sze Yap drawn as "Cantonese" (Goulou and Gao-Yang
could be their own leaves); Wu keeps the shared label "Wu (Shanghainese)", odd over Wenzhou
(relabelling means editing ca.txt and hk.txt too); Tibet's Han as Mandarin (Southwestern).

## 7. Counties the atlas splits (2026-10-06, session 5d7dac7e-edge)

Anita (2026-10-06): dialect groups snapping at county, prefecture and province borders were one of
the map's most visible false edges; she accepts it is partly unavoidable with this data.

**Where the snapping comes from.** §6 gives every county one group, so each group's edge is a
county line by construction. Prefecture and province lines show only where they are also county
lines (the 145 fallback counties take a prefecture or province group, but those are minority-
language counties with few Chinese speakers). The minority-language layers add their own edges
at prefecture and province lines in the 15 estimated provinces (§1), which is not a dialect issue.

**What was done** (`sources/cn_dlac.py`, `countries/cn.py` `_CnWeighter`). Crissman's Digital
Language Atlas of China (ACASIAN 1995; Harvard Dataverse doi:10.7910/DVN/OHYYXH, **CC0**; raw in
`data/raw/cn/dlac/`) vectorises the 1987 first edition's map: 307 polygons, field CHINESE_GR. Each
3 km grid cell takes the group its representative point falls in. A county with one table row
whose own group holds at least 25% of its people in the polygons, and another group at least 15%,
has its Chinese split between them by those people; the other group's subgroup (Min, Yue) is the
nearest county's that the table files under it. Its dots then go on the cells the polygons give
each group. Counties where the polygons disagree outright (own group under 25%) are left whole:
that is the two editions classifying differently (Guangxi's Yue against Pinghua, the Jin border)
or digitising slop along the county line, and the 2012 table wins.

Each county's Chinese total is unchanged (asserted in cn.py); only its split between groups moves.
This changes the dialect node counts, which were already §6's approved proxy, not census figures.

**Result.** 122 of 2,677 single-group counties split; 27.6M Han moved to a second group. Largest:
Yue to Hakka 18 counties, 9.3M (Dongguan 27% Hakka, Shenzhen's Longgang and Longhua, Gaozhou, Xinyi,
Lianjiang, Bobai 61%); Mandarin to Wu 9, 2.5M (Nantong's Tongzhou, Taixing); Jin to Mandarin 14, 1.9M; Gan
to Mandarin 10, 1.5M; Min to Hakka 3, 1.4M; Pinghua and Mandarin both ways in Guangxi and Hunan,
2.6M. Hakka 27.1M before, 38.0M after, closer to the usual 40-50M. Cantonese 96.5M to 85.8M.
scatter: 248 (county, group) rows placed on the atlas's area, 1,409,741 dots.

**A quirk of the polygons.** Where the atlas overlays a minority language on Chinese, the polygon
often carries only the minority: central Hunan's Xiang country is filed "Miao-Yao Languages" with
no Chinese group, and 76M people's cells have none. They are left out of the shares, so Xiang is
almost never split. Within a split county those cells take any of its groups.

**Not done, and why.** A blend across county lines (pulling a group's dots towards a neighbouring
county of another group) would draw speakers no source places there: the atlas itself draws its
lines along counties in most of China, so those edges are what it says. The county-by-county
1990 coding of the atlas (doi:10.7910/DVN/QPUONU) would give a second opinion on mixed counties;
not fetched. A finer source (township dialect tables in county gazetteers, 县志) exists only in
print.

Calls someone might reverse: the 25% / 15% thresholds (set by reading the list of splits, which
are well-known mixed counties); using 1987 polygons inside a 2012 classification.

## 8. People from other provinces (2026-10-06, session 5d7dac7e-cnmig)

Anita (2026-10-06): "for china, is there internal migration data? would that be usable for mixing
dialects?" Yes. `sources/cn_migrants.py` -> `data/normalized/cn_migrants.csv`; `countries/cn.py`
`_migrants`.

**The tables.** The 2020 census asks every resident where their hukou is registered. Table 1-3
(各地区分性别的户口登记地在外乡镇街道的人口状况) splits those registered elsewhere into this county,
another county of the province and another province (省外); table 7-3 (按现住地、性别分的户口登记地在
外省的人口) gives the 省外 people by province of registration. Both are in the national yearbook by
province (`stats.gov.cn/sj/pcsj/rkpc/7rp/zk/html/A0103.xls`, `A0703.xls`; the .xls beside the
JPG, as for table 1-4) and in the provincial yearbooks by county or prefecture. Raw copies in
`data/raw/cn/migration/`; Hubei and Yunnan read from chinaethnicity's downloaded yearbooks.

**Grain** (the `grain` column), every province's total checked against the national 1-3:
- **county, origins by county** (7-3 by county): Beijing, Heilongjiang, Shanghai, Jiangsu, Fujian,
  Shandong, Hubei, Yunnan, Qinghai. **County, origins by prefecture**: Jilin, Henan (1-3 by county,
  7-3 by prefecture). **County, origins by province**: Guangxi (1-3 only; its 7-3 is not online).
  Rows joined to polygons through chinaethnicity's `leaves_2020.csv` (the same yearbooks' table 1-4
  rows, with its development-zone folds), every leaf matched once, in table order. One 7-3 row the
  1-4 table lacks (Mudanjiang's 经济技术开发区, 0 people) is spread over its prefecture. Every one of
  these provinces sums to the national figure to the person.
- **district**: Shenzhen, from 《深圳市人口普查年鉴-2020》 (one 795-page PDF on sz.gov.cn, table 7-3
  on PDF pages 614-661, read by word position; every district's 30 provinces sum to its total,
  8,228,763 in all; 大鹏新区 added to Longgang, whose polygon holds it; 深汕合作区's 7,557 left to
  Shanwei). Guangzhou, from 《广州市人口普查年鉴-2020》 (tjj.gz.gov.cn, volume 7 as 55 xls sheets),
  4,934,998.
- **prefecture**: Hebei, Liaoning, Hunan, Sichuan (their 7-3 stops at prefectures; Hebei's whole
  石家庄市 and 保定市 rows kept, the ① rows and 辛集 and 定州 dropped as already inside them; Liaoning's
  沈抚新区 row added to Fushun). Shanxi's 11 cities and 13 Guangdong cities from each city's census
  bulletin ("外省流入人口为 N 人", preliminary counts), read on tjgb.hongheiku.com (page ids in the
  script); Shanxi's 11 sum to its national 1,620,518 exactly. Zhejiang's 11 cities from the
  provincial bureau's analysis 浙江省第七次人口普查系列分析之七 (tjj.zj.gov.cn, 2022-07-22), table 7-4,
  in 万人, which sums to the national figure within 54 people. Inside a prefecture every county gets
  the prefecture's share.
- **Guangdong's six cities with no figure** (their bulletins print no 外省 number): the province's
  29,622,110 less Shenzhen, Guangzhou and the 13 bulletins leaves 4,955,531. Maoming, Shaoguan and
  Shanwei, outside the delta, take the share of the nine measured non-delta cities (4.0%); Foshan
  takes 3,000,000, the floor Yicai reports from 《2020中国人口普查分县资料》 (one of ten cities over
  3 million); Zhuhai (501,012) and Jiangmen (983,561) split the rest by population. The weakest
  numbers in the file; Foshan's real figure would replace all three.
- **province**: Tianjin, Inner Mongolia, Anhui, Jiangxi, Hainan, Chongqing, Guizhou, Tibet,
  Shaanxi, Gansu, Ningxia, Xinjiang, the national share in every county. Not looked for further:
  most are Mandarin speaking with mostly Mandarin-speaking migrants, where the split changes
  nothing drawn. Hainan (10.8%, Hainanese locally) and Anhui's and Jiangxi's southern counties are
  the ones a county table would move. Searched and not found: Zhejiang's and Inner Mongolia's
  1-3 and 7-3 (not in the Wayback Machine, live hosts 403), Hainan's (403).

**The population base.** A county's share is its people from other provinces over its population.
When this section was written cn.csv's county totals were a 2000 pattern in the 15 estimated
provinces (Shenzhen's districts 10.5M against a census 17.6M, Bao'an's share 107%), so the base
there was Dong and Wang's county panel by code. Since §9 cn.csv's totals are the 2020 census in
every province and the base is simply cn.csv's (`cn_migrants.py` no longer reads the panel).

**The rule** (`countries/cn.py` `_migrants`). For each county, f = people from other provinces /
base (capped at 0.9; the highest is Bao'an at 56%). The county's Chinese speakers (§2's Chinese:
Han and every non-retaining share) keep (1 - f) on the county's own dialect groups (§6, §7), and f
goes onto the origin provinces' mixes, weighted by where that county's migrants come from. An
origin province's mix is its own counties' local dialect groups summed (before this step), so a
migrant from Guangdong is 51% Cantonese, 22% Hakka, 14% Teochew, as Guangdong's locals are. These
rows are `modelled`; locals stay `derived`. Asserted: every county's Chinese total unchanged, no
county has migrants from its own province. Placement unchanged: a migrant row on a group the
county already has (Mandarin migrants in a county split between Mandarin and Wu) follows that
group's atlas area, the rest follow population.

**What it assumes.** Migrants speak their home province's average dialect: a Hunanese worker in
Dongguan is drawn as Hunan's mix (Xiang, Mandarin, Gan), not as whichever of those he speaks; the
children of migrants who have learnt the local dialect are drawn on the home one. People from other
provinces are assumed to be Chinese speakers in the same proportion as the county (a Uyghur in
Shenzhen is still counted as Uyghur by §2, not moved). People who moved within their province stay
on the local mix: Shenzhen's 4.2 million from the rest of Guangdong (many Teochew and Hakka) draw as
Shenzhen's Cantonese and Hakka, because no table gives the city of registration inside the province.

**Result.** 114,482,618 people moved (124.8M people from other provinces, times each county's
Chinese share). National: Mandarin 845.3M to 882.5M, Wu 121.0M to 92.7M, Cantonese 85.8M to 73.5M,
Gan +4.8M, Xiang +3.0M, Jin +2.6M, Hakka 38.0M to 35.5M, Min Nan 25.2M to 22.7M.

| place | before | after |
|---|---|---|
| Shenzhen | Cantonese 86.6, Hakka 13.4 | Cantonese 52.9, Mandarin 25.6, Hakka 8.1, Gan 5.0, Xiang 4.6, Min Nan 0.6 |
| Dongguan | Cantonese 73.0, Hakka 27.0 | Cantonese 40.1, Mandarin 32.1, Hakka 13.0, Xiang 5.8, Gan 4.8 |
| Guangzhou | Cantonese 95.7, Hakka 4.3 | Cantonese 74.6, Mandarin 13.3, Hakka 4.4, Xiang 2.9, Gan 2.5 |
| Shanghai | Wu 100 | Wu 62.9, Mandarin 30.2, Gan 2.3, Jin 1.3 |
| Suzhou | Wu 98.3, Mandarin 1.7 | Wu 69.3, Mandarin 26.1, Jin 1.6, Gan 1.4 |
| Hangzhou | Wu 97.2, Hui 2.8 | Wu 71.8, Mandarin 21.0, Gan 2.4, Hui 2.3 |
| Xiamen | Min Nan 100 | Min Nan 73.1, Mandarin 19.5, Gan 3.3 |
| Beijing | Mandarin 100 | Mandarin 92.1, Jin 5.0, Wu 0.9 |
| Urumqi | Mandarin 100 | Mandarin 98.5, Jin 0.7 (Xinjiang's Han come from Mandarin provinces) |

(Percent of each place's Chinese speakers. Shenzhen's Min is small because Teochew and Hakka from
eastern Guangdong are intra-province migrants, above.)

scatter: 1,409,741 dots, 414 (county, group) rows placed on the atlas's area; check_country ok.

**A problem this exposed, fixed in §9.** The estimated provinces' county totals were the 2000
pattern (Shenzhen 10.5M against a census 17.6M). The figures above are from before that fix; after
it, 119,419,521 people are moved (more of them in the grown cities), and Shenzhen's Chinese are
Cantonese 50.3, Mandarin 26.7, Hakka 9.0, Gan 5.1, Xiang 5.0; Dongguan Cantonese 40.1, Mandarin
32.2, Hakka 13.0; Guangzhou Cantonese 73.9, Mandarin 14.1.

Calls someone might reverse: Foshan's 3,000,000 floor and the Zhuhai and Jiangmen split; applying
the origin province's whole mix rather than its rural or out-migrating counties'; leaving
intra-province migrants local.

## 9. 2020 county totals in the estimated provinces (2026-10-06, session 5d7dac7e-cntot)

Anita approved: chinaethnicity's 15 estimated provinces (§1) drew each county at its 2000 share of
the 2020 province or prefecture, so fast-growing cities were far short (Shenzhen 10.7M against a
census 17.6M). Fixed inside languagedots; chinaethnicity is untouched. `sources/cn_totals.py`,
called from `sources/cn_ethnic.py`.

**The totals.** Dong and Wang's 2020 county census panel (github.com/leiii/census), helper1m's
copy, with helper1m's two repairs: 衡东县 565,423 (a 3,000 slip in the panel) and Xinjiang, which
the panel leaves empty, from helper1m's `scripts/china/xinjiang_counties.csv` (hongheiku.com's
county pages, summing to Xinjiang's census bulletin in every prefecture).

**The join**, by code, asserted both ways: every panel row of the 15 provinces reaches a polygon,
every polygon of theirs gets a row, no polygon in two groups. 1,503 panel rows to 1,533 polygons.
Three codes aliased (繁昌区 340212 to DataV's 340211, 水城区 520204 to 520221, 嘉峪关市 620201 to
620200). 36 development-zone codes in pooled rows are not polygons and are dropped (their people
are in the row's figure). 19 groups pool several polygons under one panel row (Hebei's city
districts with their zones, Xiong'an's three counties, Xi'an's 灞桥+未央 and 雁塔+长安, Anshan,
Bengbu, Zhanjiang, Jieyang...); a group's figure is split over its polygons by their old shares.
No polygon was left unmatched, so nothing had to be folded or split beyond those groups.

**Held to the census totals cn.csv already had**: the prefecture in Hebei, Liaoning, Hunan and
Sichuan (chinaethnicity scaled those to 2020 prefecture rows), the province elsewhere. The panel
matches exactly in 9 provinces and in every Hebei and Hunan prefecture. Where it falls short
(people the census books to no county row) the shortfall goes to the area's counties where the
placement grid holds more people than the panel, by that excess and never above it (helper1m's
rule): Shaanxi 769,627 (Xixian New Area, booked to Xi'an, living largely in Xianyang's counties),
Dalian 629,664 (its development zones), Anhui 460,257 (Wuhu and Huainan rows short), Fushun
42,312, Dazhou 45,463. Where it is over the area is scaled down: Deyang 238,197 (x0.936),
Guangdong 4,493.

**Mixes, raked.** Rescaling each county with its mix fixed moved the nationality totals the
census gives per province (Kazakh -15%, Guizhou's Tujia -15%, Guangdong's Miao +25%: the counties
that grew are not where those groups live). So the estimated provinces are then raked (iterative
proportional fitting) to both the county totals and each area's census nationality totals, the
ones chinaethnicity scaled to: both hold to within a person. Each county's Han share moves 0.4
points on average (people-weighted); the most, 7-9 points, in autonomous counties of Chongqing,
Guizhou and Sichuan, where the rake restores the minority totals. This departs from keeping the
mixes exactly as they were; it is the smallest change that keeps every census total.

**Checks.** National 1,409,778,724 before and after. Every province equal to helper1m's
`census2020_provinces.csv` to the person, before and after. Minority languages 71,817,105,
unchanged (retention shares are national or provincial, and the nationality totals are held).

| city | before | after | census |
|---|---:|---:|---:|
| Shenzhen | 10,707,073 | 17,493,774 | 17,560,061; the panel's districts 17,494,398, the other 65,663 most likely the Shenzhen-Shanwei zone, inside Shanwei's polygon |
| Dongguan | 9,816,431 | 10,466,252 | 10,466,625 |
| Guangzhou | 14,695,572 | 18,675,939 | 18,676,605 |
| Foshan | 8,064,552 | 9,498,524 | 9,498,863 |
| Suzhou | 12,748,262 | 12,748,262 | measured province, unchanged |
| Hangzhou | 11,936,010 | 11,936,010 | measured province, unchanged |
| Chengdu | 20,937,757 | 20,937,757 | 20,937,757 (Sichuan's prefecture rows already held it) |
| Xi'an | 8,139,851 | 12,332,418 | 12,952,907; the panel's rows 12,183,280, the rest Xixian's people drawn where they live |
| Hefei | 6,615,221 | 9,439,757 | 9,369,881; plus 69,876 of Anhui's unbooked 460k by the grid |
| Urumqi | 3,186,779 | 4,054,369 | 4,054,369 |

Bao'an 2,358,870 to 4,476,394. Largest factors: 瑶海区 x5.3, 蜀山区 x4.7 (Hefei), 鲅鱼圈区 x3.9;
smallest 乌鲁木齐县 x0.16, 北屯市 x0.26, 简阳市 x0.48 (land moved to Chengdu's new districts since
2000). 81.2M people moved between counties in all.

**Measured provinces, compared and not changed.** Their totals are the yearbooks' own; the panel
agrees to the person in Beijing, Shanghai, Jiangsu, Zhejiang, Shandong, Guangxi, Yunnan, Qinghai
and Ningxia. 1,121 of 1,211 counties that share a code agree within 1%. Where they differ it is
the panel short (Fujian 551,250, mostly Zhangzhou -13%; Hainan 85,474, Danzhou; Jilin, Inner
Mongolia, Heilongjiang's Da Hinggan Ling -16%), the panel over (Henan +188,161, Hubei +71,833), or
chinaethnicity splitting a yearbook row over polygons differently from the panel's units (余杭/临平,
殷都/安阳县, 中牟 with Zhengzhou's airport zone). Nothing points to an error in the measured tables.

**After.** `cn_migrants.py` rerun on the new base (124,829,596 people from other provinces);
119,419,521 Chinese speakers moved onto origin mixes (was 114,482,618). National dialect groups:
Mandarin 886.5M, Wu 92.7M, Cantonese 78.4M (73.5M before), Jin 73.7M, Gan 54.3M, Xiang 41.4M,
Hakka 33.0M, Min Nan 22.0M. scatter: 1,409,741 dots; check_country ok.

Calls someone might reverse: raking the mixes rather than keeping them fixed; the grid-excess rule
for Shaanxi's 770k (they land across the province, not only around Xixian); Xinjiang's XPCC city
figures taken as the cities' own though their regiments are scattered around them.

## 10. Putonghua in the non-Mandarin south (2026-10-06, session 5d7dac7e-clds)

Anita (2026-10-06) approved tabulating CLDS 2016 for this ("not personal data... these cities are
pretty big... this will not be a commercial product"), with WVS as a check. Until now nobody in
the south was drawn on Mandarin unless they came from a Mandarin province (§8), so Guangdong's
drawn Mandarin was 12.8% of its Chinese speakers. `sources/cn_putonghua.py` ->
`data/normalized/cn_putonghua.csv` (aggregates only); `countries/cn.py` `_putonghua`.

**The survey.** China Labor-force Dynamics Survey 2016 (Center for Social Survey, Sun Yat-sen
University), individual file, read-only from religiondots' raw tree, never copied here. Terms:
non-commercial, no raw data passed on, cite CLDS. Item `I1_8_4`, main language used after work or
school: Putonghua / local dialect / home-town dialect / other. Ages 15-64 in sampled households,
weight `wpp`. Migrant status from the hukou place (`I1_3_1_psu`; its county part is scrambled by
the release, the prefecture part is real) against the interview city: local (same prefecture; same
province in Shanghai), intra (another prefecture of the province), inter (another province).
20,516 of 21,086 have an answer, a city and a hukou place.

**Only non-Mandarin cities count.** In Mandarin areas people call their own Mandarin "local
dialect" (Henan 86%), so a CLDS city feeds the shares only when its counties are under half
Mandarin in the atlas (§6, §7): 44 cities, 6,798 people. Left out as Mandarin: Nanjing, Xuzhou,
Yancheng, Yangzhou, Suqian, every sampled Anhui city, Xiangxi, Baise, Laibin.

**Province x status** (weighted %, non-Mandarin cities only, n = respondents):

| province | locals: n; Putonghua, local, home-town | intra-province: n; Putonghua | other provinces: n; Putonghua, local, home-town |
|---|---|---|---|
| Guangdong | 2,615; 4.3, 91.1, 4.0 | 247; 21.0 (home-town 50.2) | 479; 29.9, 8.1, 61.3 |
| Guangxi | 321; 11.9, 84.9, 2.9 | 10; 23.3 | 0 |
| Fujian | 706; 21.5, 74.9, 2.4 | 40; 44.1 | 25; 26.2, 0.9, 73.0 |
| Zhejiang | 746; 13.4, 82.8, 2.8 | 14; 52.2 | 82; 31.8, 17.4, 49.0 |
| Shanghai | 135; 23.0, 71.5, 5.5 | 0 | 11; 30.1, 0, 69.9 |
| Jiangsu (Wu cities) | 211; 5.1, 93.8, 1.1 | 3 | 25; 7.5, 36.7, 55.9 |
| Jiangxi | 474; 3.4, 83.1, 13.4 | 2 | 5 |
| Hunan | 610; 7.6, 80.5, 11.8 | 30; 22.8 | 7 |

All statuses together, Putonghua %: Shenzhen 39.8 (n 160), Guangzhou 11.9 (426), Dongguan 26.6
(166), Foshan 5.5 (245), Shanghai 23.4 (146), Xiamen 38.2 (150), Hangzhou 19.1 (208), Suzhou 4.7
(75). The script prints every city x status cell. **Age** (record only, not used): settled
Putonghua at 15-30 / 31-45 / 46-64 is Shanghai 41 / 36 / 10, Zhejiang 32 / 19 / 5, Fujian 44 /
28 / 13, Guangdong 12 / 8 / 2. Guangdong's people from other provinces by origin: Hunan 40.5%
Putonghua (n 125), Henan 31.7, Jiangxi 31.1, Hubei 29.8, Sichuan 18.9, Guangxi 14.9 (n 118). No
sign that people from Mandarin provinces answer Putonghua more often, so one share per
destination serves every origin.

**The rule.** Two shares per place: `settled` (locals and intra-province migrants pooled, as the
map draws both on the county's own groups, §8) and `inter`. Each is shrunk towards the level above
with 30 pseudo-cases: pool of the 44 cities (settled 10.2%, inter 29.4%) -> province -> city. A
county takes its prefecture's share where CLDS sampled it, else its province's; Hainan (not
sampled) and Anhui (no non-Mandarin city sampled) take the pool. In every county of the ten
provinces (Guangdong, Guangxi, Hainan, Fujian, Zhejiang, Shanghai, Jiangsu, Jiangxi, Hunan,
Anhui) that share of each non-Mandarin row moves onto Mandarin: the settled share from the
county's own groups (`derived` rows), the inter share from migrants' home groups (`modelled`
rows). Mandarin rows are untouched, so a north Jiangsu county changes only for its migrants from
Wu or Gan areas. Moved rows are `modelled`. Asserted: every county's Chinese total unchanged.
43,555,751 people moved: 36,148,517 settled, 7,407,234 from other provinces.

**Result**, Mandarin share of Chinese speakers (before = after §8 and §9):

| place | before | after | moved by this step | CLDS Putonghua, all statuses | WVS 2018 Putonghua at home (n) |
|---|---:|---:|---:|---:|---:|
| Guangdong | 12.8 | 22.1 | 9.3 | 10.6 | 25.4 (263) |
| Shenzhen | 26.7 | 49.8 | 23.1 | 39.8 | |
| Guangzhou | 14.1 | 23.3 | 9.3 | 11.9 | |
| Dongguan | 32.2 | 46.4 | 14.2 | 26.6 | |
| Foshan | 17.2 | 22.2 | 5.0 | 5.5 | |
| Shanghai | 30.4 | 46.9 | 16.5 | 23.4 | 27.5 (42) |
| Zhejiang | 20.1 | 32.9 | 12.8 | 15.8 | 19.2 (111) |
| Fujian | 9.5 | 27.5 | 18.0 | 23.1 | 44.2 (110) |
| Xiamen | 19.6 | 48.1 | 28.5 | 38.2 | |
| Jiangsu | 69.9 | 71.8 | 1.9 | 11.1 | 40.7 (104) |
| Guangxi | 24.3 | 33.4 | 9.1 | 8.4 | 2.6 (153) |
| Hainan | 6.5 | 16.8 | 10.4 | not sampled | 51.5 (71) |
| Jiangxi | 12.1 | 15.3 | 3.2 | 3.3 | 26.0 (44) |
| Hunan | 22.1 | 29.1 | 7.0 | 7.9 | 1.8 (72) |
| Anhui | 89.4 | 90.6 | 1.2 | 4.5 | 7.0 (130) |

(CLDS and WVS province columns are over every sampled city, Mandarin ones included. WVS: wave 7,
Q272 by N_REGION_ISO, the route of `sources/ir_wvs.py`, saved to `data/raw/cn/wvs/`; the file is
weighted, so its printed percentages are used as they are.) National Mandarin 886.5M to 930.1M;
Wu 92.7M to 80.1M, Cantonese 78.4M to 69.4M, Hakka 33.0M to 29.9M, Min Nan 22.0M to 17.8M.
scatter: 1,409,743 dots; check_country ok.

**Reading the comparison.** "Moved by this step" is the figure comparable to CLDS and lands near
it (Guangdong 9.3 against 10.6). The map's total Mandarin is higher than either survey because it
also holds migrants from Mandarin provinces, whom both surveys file under home-town dialect or
"other Chinese dialects". WVS is well above CLDS in Guangdong, Fujian, Jiangsu, Jiangxi and Hainan:
it asks about home, adults 18+, its province samples are small, and in Jiangsu it includes the
Mandarin north, which calls its speech Putonghua. It is below CLDS in Guangxi and Hunan. Neither is
clearly right; CLDS has 6,800 people in these cities and the city grain, so it is the one used.
Hainan is the weak spot: unsampled by CLDS, drawn at the pool's 10%, against WVS's 52% from 71
people.

**Shenzhen.** Its local dialect in the map: the atlas table files every Shenzhen district as Yue
(Guangfu); §7's 1987 polygons add Hakka in Longgang (29%) and Longhua (58%), so Shenzhen's locals
draw 85% Cantonese, 15% Hakka. That undercounts Hakka. The old Bao'an county (today's Shenzhen)
was, by the county gazetteer as quoted online, about 56% Hakka, 35% Cantonese (Weitou) and 9%
Dapeng speech before 1980, and "over 60% Hakka" in 1978 (sznews.com, 2019-05-15): Cantonese in
the west (Nantou, Xixiang, Shajing, the old villages of Futian and Luohu), Hakka in the east and
north (Longgang, Pingshan, Longhua, Guanlan). Pingshan is 100% Hakka in the 1987 polygons and
Guangming 88%, but both were left whole Cantonese: they were carved out of Longgang and Bao'an
after the table was compiled, inherit their parent's row, and §7 keeps the table where the
polygons disagree outright. The table never classified them, so that rule should not apply there;
not changed in this session (§7's logic). It matters less than it looks: CLDS's 23 Shenzhen-hukou
respondents answer 81% Putonghua (mostly people who moved there and took a Shenzhen hukou, not
natives), and its 47 people from elsewhere in Guangdong answer 73% home-town dialect (Teochew,
Hakka, Cantonese from their own cities), which the map still draws as Shenzhen's local mix (§8).

**The "80% Mandarin" figures for Shenzhen** found online are not home language. The 94.5% (2014)
is the Putonghua 普及率, the share able to communicate in Putonghua; "80%" is usually the share of
residents from outside the city. A 1985 sample survey found under 20% speaking Putonghua, Cantonese
and Hakka 25% each, Min and other dialects 30% (sznews, same article). CLDS's after-work item is
narrower than ability and broader than home: 39.8% Putonghua over all statuses, the rest mostly
home-town dialects. The map now draws Shenzhen at 49.8% Mandarin (Putonghua users plus migrants
from Mandarin provinces).

**Assumptions.** One share per destination for every origin (checked above); intra-province
migrants share the locals' rate and otherwise stay on the local mix (no table of their origin
prefecture); 2016 rates on 2020 counts; CLDS samples households, so workers in factory dormitories
are probably under-represented; under-15s and over-64s take the 15-64 rate. Not applied: the
Mandarin provinces' non-Mandarin areas (Shanxi's Jin, southeast Hubei's Gan), where the same
reading would work; migrants answering "local dialect" (Guangdong inter 8%, Zhejiang 17%), who
stay on their home groups.

(The Pingshan and Guangming issue above was fixed in §11.)

Calls someone might reverse: the 30 pseudo-case shrinkage (Shenzhen settled 38.8% raw, 29.1%
used); pooling locals with intra-province migrants; Hainan on the pool rather than WVS; applying
to every non-Mandarin row in the ten provinces, Mandarin-majority counties included.

## 11. Softer dialect edges from MCPDict points; carved districts by the polygons (2026-10-06, session 5d7dac7e-mcp)

Anita (2026-10-06), on western and northern Guangdong and Guangxi, where whole counties flip from
Hakka to Cantonese to Pinghua: "not too sure how the mcpdict blend would work really but if you
think you can make it work lets do it"; county gazetteers judged not worth it. Lead 3 of
`sources/cn_dialect_sources.md`. `sources/cn_mcpdict.py` -> `data/normalized/cn_dialect_mcp.csv`
(read by `countries/cn.py` in place of `cn_dialect_dlac.csv`) and `cn_mcp_points.csv`; its
`Placer` is now the weighter (replacing §7's hard polygon mask).

**The points.** 汉字音典 (MCPDict, github.com/osfans/MCPDict), `tools/info.geojson`, fetched
2026-10-06 to `data/raw/cn/mcpdict/` with its **MIT licence** (`LICENSE`, (c) 2019 Yun Wang).
3,147 points at township or village, each filed under its Language Atlas 2nd ed. group
(地圖集二分區). Dropped: foreign readings, opera, minority languages, the standard, Waxiang (no
node): 3,097 left, 3,066 inside a county (the rest Hong Kong, Macau, Taiwan, abroad). Each takes its
node through `cn2020.dialect_node` (Min Nan in Chaoshan = Teochew, Yue 四邑 = Sze Yap). Islands:
the `方言島` field or a label ending 方言島 (286).

**(a) Counts, in six provinces only** (Guangdong, Guangxi, Fujian, Jiangxi, Hunan, Zhejiang). A
point of a node the county's table rows lack is evidence some of its people speak it. Per node:
`0.25 x (1 - e^-E_edge) + 0.12 x (1 - e^-E_pocket)`, all added nodes together at most 25% of the
county's Chinese, taken from its own groups in proportion. An **edge** point is one whose node is
the table group of a neighbouring county and that is not marked an island: the group's area runs
over the county line, the case asked about. Anything else is a **pocket** (island, migrant
village) worth at most 12%; Mandarin pockets, mostly garrison 军话, at most 3%. E counts points
**per group**: each node's points are weighted by its people per point across the six provinces
against the median node, clipped to 0.25-1, so the oversampled groups shrink (Pinghua 224 points
for 4.8M people: 0.25; Shaojiang, Min Bei, Hui also low) and no big group's point counts more than
one. The county's own points are ignored: the atlas already says its group is there.
200 edge points and 179 pockets; 173 counties gain a group; 10.3M people change group across the
six provinces (Guangdong 2.7M, Hunan 2.2M, Guangxi 2.3M, Zhejiang 1.4M, Jiangxi 0.9M, Fujian 0.8M).
**This changes counts** (Anita's "keep it small and say so"); every county's Chinese total is
unchanged.

**(b) Held towards the published province totals.** The gazetteer figures (lead 4) as shares of
their sum, so their 1990s/2000 vintage and "+" lower bounds drop out: Guangxi 《广西通志·汉语方言志》
(1998) Yue 12M, Hakka 5.6M, Mandarin 5M, Pinghua 4M, Xiang 1.5M, Min 0.25M; Guangdong
《广东省志·方言志》 (2004) Yue 40M, Hakka 15M, Min 17M, plus Shaozhou Tuhua 0.8M (Zhuang Chusheng).
Each coarse group may move only towards its published share (target clipped between today's total
and the published one), then the province's county x node table is raked to its county totals and
those targets. Neither binds much: the changes are small next to the gaps.

| millions, locals before migrants (§8) | Guangxi before | after | published share | Guangdong before | after | published share |
|---|---:|---:|---:|---:|---:|---:|
| Yue | 22.11 | 20.90 | 15.06 | 75.08 | 74.38 | 67.53 |
| Hakka | 2.82 | 3.42 | 7.03 | 26.29 | 26.07 | 25.33 |
| Mandarin | 8.38 | 7.59 | 6.28 | 0 | 0.05 | |
| Pinghua / Tuhua | 1.42 | 2.53 | 5.02 | 0.53 | 0.65 | 1.35 |
| Xiang | 0.86 | 0.93 | 1.88 | | | |
| Min | 0 | 0.22 | 0.31 | 21.00 | 21.76 | 28.70 |

Guangxi stays far from its gazetteer: too much Yue and Mandarin, too little Hakka and Pinghua.
The 25% cap is why; the table (§6) files every Guangxi Pinghua county under Yue or Mandarin, and
Guangxi's Hakka live in villages scattered through Yue and Mandarin counties, which MCPDict files
as islands. A rake to the published totals with a larger cap would close the gap; not done
without Anita's say (it would move about 5M more people). (Tried in §12 and reverted at Anita's
request; this table stands.)

**Placement, nationally** (no counts move). In every county with several local groups (268), each
group's dots lean towards its points: cell seed = base + 4 x exp(-d^2 / 2 x 8 km^2), d the
distance to the group's nearest point in any county; base 1 for the table's groups (in a §7 split
county, 1 inside that group's 1987 polygons or no polygon, 0.05 outside), 0.05 for a group only
the points gave. The county's cell x group matrix is then raked from seed x population to the
cells' population and the groups' totals, so cells near a Hakka point fill with Hakka first and
every cell keeps its population. Migrants' home groups the county has no local share of follow
population, as before. `cn_mcpdict_before_after.png` (project root) shows Hakka, Cantonese and
Pinghua shares of local Chinese per cell, 108-114.2 E, 20.9-25.6 N, before and after, points in red.

**Carved districts** (coordinator, 2026-10-06; §10's Shenzhen note). §7 keeps the table's group
where the 1987 polygons disagree outright, but 14 units have no table row of their own: they were
carved from a county after the table was compiled and inherit its row (`cn_dlac.inherited()`: every
row's code also reaches another unit). The table never classified them, so the polygons now decide
there (groups at 15% or more kept): 平城区, 加格达奇区, 临平区, 钱塘区, 龙港市, 无为市, 红谷滩区,
龙华区, 坪山区, 光明区, 西沙区, 南沙区, 叙州区, 胡杨河市. Four change: Pingshan Yue to Hakka (100%
in the polygons; MCPDict's Weitou point at 田头 then adds 16% Cantonese), Guangming Yue to Hakka
(88%, its 12% Yue under the 15% floor), Longhua as before (58/42), and **Longgang (Zhejiang) Min
Nan to Wu** (polygons 88% Wu; MCPDict's own Longgang point is Oujiang Wu, and two 蛮话 points add
Min Dong as pockets). §7 now splits 125 counties.

**Shenzhen**, locals' dialect (the `derived` rows): Cantonese 86, Hakka 14 before; Cantonese 77,
Hakka 23 after (old Bao'an county: about 56 Hakka, 35 Cantonese, 9 Dapeng, §10). All Chinese
speakers: Mandarin 50, Cantonese 35, Hakka 6 before; Mandarin 50, Cantonese 32, Hakka 10 after.

**Before and after**, percent of each county's Chinese speakers (all steps, migrants and §10
included):

| county | before | after |
|---|---|---|
| Yangchun (Yangjiang) | Cantonese 94, Mandarin 5 | Cantonese 88, Hakka 6, Mandarin 5 |
| Luoding (Yunfu) | Cantonese 92, Mandarin 8 | Cantonese 79, Mandarin 7, Min Nan 7, Hakka 6 |
| Qingxin (Qingyuan) | Hakka 88, Mandarin 10 | Hakka 67, Cantonese 22, Mandarin 10 |
| Yingde (Qingyuan) | Hakka 88, Mandarin 10 | Hakka 74, Cantonese 15, Mandarin 10 |
| Lechang (Shaoguan) | Hakka 90, Mandarin 9 | Hakka 74, Pinghua 16, Mandarin 9 |
| Dianbai (Maoming) | Leizhou 53, Hakka 41, Mandarin 6 | Leizhou 48, Hakka 37, Mandarin 7, Sze Yap 6 |
| Gaozhou (Maoming) | Hakka 48, Cantonese 46, Mandarin 6 | unchanged counts; placement only |
| Wuchuan (Zhanjiang) | Leizhou 50, Cantonese 42, Mandarin 8 | unchanged counts; placement only |
| Zhongshan (Hezhou) | Cantonese 87, Mandarin 13 | Cantonese 82, Mandarin 13, Pinghua 6 |
| Binyang (Nanning) | Cantonese 87, Mandarin 12 | Cantonese 66, Pinghua 14, Mandarin 14, Hakka 6 |
| Xixiangtang (Nanning) | Cantonese 83, Mandarin 16 | Cantonese 64, Pinghua 19, Mandarin 16 |
| Bobai, Luchuan (Yulin); Hepu (Beihai) | §7 splits | unchanged counts; placement only |

National: Hakka 29.9M to 32.3M, Pinghua 4.75M to 5.72M, Cantonese 69.4M to 67.4M, Mandarin 930.1M
to 928.9M. check_country ok; scatter 1,409,742 dots, 925 (county, group) rows leaned.

**Found in passing.** 陆河县 (Shanwei) is filed Yue (Goulou) in the county table; Luhe is
Hakka-speaking (its MCPDict point is Hakka, and it now gets 14%). A table fix belongs in
`cn_dialect.py`; not made here. (Fixed in §12.) The MCPDict "Southwestern" placeholder subgroup on added Mandarin
rows is a label only (all Mandarin is one node).

Calls someone might reverse: the 25% / 12% / 3% caps and E0 = 1 (set by reading the county list:
Yangchun comes out 6% Hakka against the gazetteer's ~15%, so they err low); counting pockets at all;
letting the provinces move only towards the published shares rather than raking to them; the 8 km
kernel; Longgang (Zhejiang) as Wu.

## 12. Guangxi's gazetteer checked, a full rake tried and reverted; Luhe is Hakka (2026-10-06, session 5d7dac7e-gx)

Anita (2026-10-06): "we can push guangxi if you think gazetteer is somewhat trustworthy and it's not
from super long ago". §11 had moved Guangxi only partway towards its published dialect totals.

**Which source the Guangxi totals are.** The figures in `PUB` reach us through one article,
「闲话广西汉语方言」, 广西日报 (Guangxi Daily), reprinted by chinanews.com 2011-04-12 (fetched and read
in full this session): 粤方言 "1200多万", 平话 "400万左右", 北方方言 (Southwestern Mandarin, 桂柳话)
"500万以上", 客家方言 "560多万", 湘方言 "150万左右" (全州, 灌阳, 资源, 兴安), 闽南方言 "25万人左右".
The article cites no source and gives no year. Its figures, county lists and wording match what is
quoted elsewhere as 《广西通志·汉语方言志》 (Guangxi regional gazetteer, dialect volume, 1998; the
regional office's catalogue lists it as volume 88, but the Wayback copies of lib.gxdfz.org.cn stop
at volume 84, so the attribution is not checked against the book). They are **round "使用人口"
estimates by dialect area**, not counted speakers: they sum to 28.35M, about Guangxi's Han
population around 1990-2000 (some 26-27 million in those censuses) plus Zhuang and others who speak
a Chinese dialect. Vintage: the 1990s.

**Newer figures.** Only one: the Language Atlas of China 2nd ed. (2012), whose group totals use
2004 populations (Xiong Zhenghui and Zhang Zhenxing, 汉语方言的分区, 方言 2008(2), p.105-107, open PDF
on ling.cass.cn): Guangxi's Hakka "分布于79个县市，大约有420万人", Min "大约有14万人"; Yue and Pinghua
are given only for all provinces together (Yue 58.82M in 141 counties; Pinghua and Tuhua 7.78M in
60 counties, 42 of them Guangxi's). The popular figures online (Yue 15-20M, Hakka 7M, Mandarin
12-15M or 25M) carry no source and are not used. Wikipedia's Pinghua "3-4 million (2013)" agrees
with the gazetteer. Searched and not found: a 2010s Guangxi language survey with dialect totals,
the atlas's B2-2 text itself (print only), 谢建猷《广西汉语方言研究》 (print).

**Verdict: trustworthy enough, used as shares.** It is the region's own dialect survey written by
its dialectologists, its six groups and their county lists agree with the 2012 atlas's map, and
its numbers are round estimates of area populations, the same kind of number the atlas gives. The
1990s vintage is handled as Anita asked: only the shares are used (Yue 42.3%, Hakka 19.8%,
Mandarin 17.6%, Pinghua 14.1%, Xiang 5.3%, Min 0.9%), applied to today's locals (each county's
Chinese speakers before §8's migrants and §10's Putonghua). The one real disagreement is Hakka: the
2012 atlas's 4.2M on a 2004 base is about 14.5% of the same total against the gazetteer's 19.8%.
The gazetteer is used, as asked; the atlas figure is the obvious reversal.

**Reverted at Anita's request** (2026-10-06, after seeing it): "i dont wanna just grow hakka where its already present, cuz it already looks quite sharply split. we can leave it underrepresented." The rake below scales a group's existing cells together, so it deepened the counties already split (Bobai, Luchuan, Hepu) instead of spreading Hakka. It is off in code (`GX_FULL_RAKE = False` in `cn_mcpdict.py`; `gx_rake` kept for the record) and Guangxi is back to §11's partway hold: locals Yue 20.90M, Mandarin 7.59M, Hakka 3.42M, Pinghua 2.53M, Xiang 0.93M, Min 0.22M. **Hakka and Pinghua in Guangxi are knowingly underrepresented** against the gazetteer's shares (7.0M and 5.0M). The Luhe fix below stays.

**The rake as tried** (`sources/cn_mcpdict.py`, `FULL`, `gx_rake`; off). Guangxi's county x node table after
§11's points (with the 25% combined cap lifted, which changes almost nothing: it rarely bound) is
raked (IPF) to every county's Chinese total and the gazetteer's shares. IPF keeps zeros at zero, so
the extra Hakka, Pinghua and Min first need seeds:
- the 1987 atlas polygons (§7's layer on the 3 km grid): the group's share of the county's people
  inside its polygons, times 0.5; 0.1 in the city-core districts of Nanning, Liuzhou and Guilin,
  where the polygons paint the villages' speech over the city; 0.15 for Pinghua or Min polygons in
  a county neither the gazetteer's list nor an MCPDict point names (兴宾: 87% Pinghua in the
  polygons, no point, not in the list);
- floors from the gazetteer's own lists: Hakka 2% in every county except 全州, 兴安, 资源, 凤山 (which
  it names as the only Hakka-free counties); Pinghua 5% in its named 桂南 and 桂北 counties (南宁郊区,
  now Nanning's districts, 宾阳, 横县, 上林, 马山, 浦北, 扶绥, 宁明, 龙州, 百色郊区, 平果, 田东, 田阳, 柳江,
  柳城, 融水, 融安, 三江, 罗城, 桂林郊区, 临桂, 灵川, 永福, 龙胜, 八步 with 平桂, 富川, 钟山); Min Nan 1% in
  its ten named counties;
- the table's groups and §7's and §11's rows as they were (a seed only ever raises a cell).
Xiang is not raked to its share: the gazetteer puts all of it in four northern counties that have
lost people since (their Chinese today 1.14M against its 1.5M), so its target is those four at
their polygon share (兴安 69%; the other three were already all Xiang) plus Xiang elsewhere as §11
left it, 1.09M; the other five groups share the rest by the gazetteer. Rows the rake created, or
raised from the polygons over a points-only row, have basis `Guangxi gazetteer rake`, and their
dots lean to the group's 1987 polygons as in a §7 split county (`Placer`). Asserted: the rake
converges to within a person on every county and group; every county's Chinese total unchanged.

| Guangxi locals, millions (as tried, reverted) | §11 | raked | gazetteer share |
|---|---:|---:|---:|
| Yue | 20.90 | 15.42 | 15.06 |
| Mandarin | 7.59 | 6.42 | 6.28 |
| Hakka | 3.42 | 7.19 | 7.03 |
| Pinghua | 2.53 | 5.14 | 5.02 |
| Xiang | 0.93 | 1.09 | 1.88 (not reachable, above) |
| Min | 0.22 | 0.32 | 0.31 |

(Each group lands 2.5% over its share because Xiang's shortfall is spread over the others.) 8.8M of
Guangxi's 35.6M local Chinese speakers are now on a group other than the table's, against 2.3M
after §11. All layers (migrants and §10 included), Guangxi draws Cantonese 17.97M to 13.25M,
Mandarin 11.23 to 10.23, Hakka 2.99 to 6.23, Pinghua 2.17 to 4.42, Xiang 0.88 to 1.03. Nationally
Cantonese 67.4M to 61.6M, Hakka 32.3 to 36.3, Pinghua 5.7 to 8.4 (Guangxi's migrants in Guangdong
carry its mix, §8), Mandarin 928.9 to 927.7.

Percent of each county's Chinese speakers, all layers, as tried (reverted; the map draws the left column):

| county | before | after |
|---|---|---|
| Bobai (Yulin) | Hakka 53, Cantonese 34, Mandarin 12 | Hakka 71, Cantonese 16, Mandarin 12 |
| Luchuan (Yulin) | Cantonese 57, Hakka 31, Mandarin 12 | Hakka 53, Cantonese 34, Mandarin 12 |
| Hepu (Beihai) | Cantonese 49, Hakka 38, Mandarin 13 | Hakka 60, Cantonese 27, Mandarin 13 |
| Babu (Hezhou) | Cantonese 79, Mandarin 13, Pinghua 7 | Hakka 40, Cantonese 35, Mandarin 13, Pinghua 10 |
| Pinggui (Hezhou) | Cantonese 75, Mandarin 13, Pinghua 11 | Hakka 40, Cantonese 32, Pinghua 15, Mandarin 13 |
| Zhongshan (Hezhou) | Cantonese 82, Mandarin 13, Pinghua 6 | Cantonese 34, Pinghua 30, Hakka 23, Mandarin 13 |
| Xixiangtang (Nanning) | Cantonese 64, Pinghua 19, Mandarin 16 | Cantonese 41, Pinghua 38, Mandarin 16, Hakka 3 |
| Qingxiu (Nanning) | Cantonese 78, Mandarin 16, Pinghua 4 | Cantonese 59, Pinghua 19, Mandarin 16, Hakka 4 |
| Binyang (Nanning) | Cantonese 66, Mandarin 14, Pinghua 14, Hakka 6 | Pinghua 54, Cantonese 23, Mandarin 13, Hakka 10 |
| Hengzhou (Nanning) | Cantonese 74, Pinghua 14, Mandarin 12 | Pinghua 50, Cantonese 32, Mandarin 12, Hakka 6 |
| Wuming (Nanning) | Mandarin 53, Pinghua 47 | Pinghua 58, Mandarin 40 |
| Lingui (Guilin) | Mandarin 54, Pinghua 33, Hakka 6, Xiang 6 | Mandarin 42, Pinghua 40, Xiang 9, Hakka 7 |
| Lingchuan (Guilin) | Mandarin 48, Pinghua 44, Min Nan 6 | Pinghua 54, Mandarin 37, Min Nan 4 |
| Xiufeng (Guilin) | Mandarin 57, Pinghua 41 | Pinghua 52, Mandarin 44 |
| Diecai (Guilin) | Mandarin 83, Pinghua 15 | Mandarin 73, Pinghua 23, Hakka 3 |
| Xing'an (Guilin) | Mandarin 80, Xiang 20 | Xiang 58, Mandarin 39, Pinghua 3 |

What to watch: the extra goes mostly where Hakka and Pinghua already were, as IPF scales a
column's cells together, so Bobai, Hepu and Gangnan (85% Hakka among locals) may now be on the high
side; Liuzhou's 城中 and 柳北 get 14% Hakka from 1987 polygons that paint the whole city Hakka;
every pure-Yue county in the southeast now has about 5% Hakka from the 2% floor.

**Luhe.** 陆河县 (Shanwei) was filed in the county table as 粤/勾漏片, a Yulin (Guangxi) group 400 km
away. The 2012 atlas makes 客家话 海陆片 of "海丰、陆丰以及1988年从陆丰析出的陆河县" (Xiong and Zhang 2008,
p.103), and MCPDict's 陆河 point is 客家話－海陸片. Fixed in `cn_dialect.py` (`TABLE_FIX`, Hakka,
subgroup Hailu): Luhe's Chinese go from Cantonese 78, Hakka 13 to Hakka 90 (Mandarin 9 from
migrants and §10). A side effect: Haifeng's MCPDict Yue point is no longer at an edge (its
neighbour is now Hakka), so Haifeng's Cantonese falls from 15% to 7%. Haifeng and Lufeng stay Min
Nan in the table. The same atlas paragraph shows the table has no 海陆片 or 粤西片 Hakka county at
all; Luhe was the only outright misfiling found.

After the revert: check_country ok; scatter 1,409,742 dots. Guangxi's hold matches §11's figures above.

If Guangxi is revisited: a rake that spreads Hakka thinly (by the 1987 polygons and the gazetteer's
floors only, not scaling the existing Hakka counties) would answer Anita's objection; the 2012
atlas's Hakka 4.2M is the newer total to aim at.