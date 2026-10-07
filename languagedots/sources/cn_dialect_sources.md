# China: sources for Putonghua at home and finer dialect geography (research only)

Scout session 5d7dac7e-cnscout, 2026-10-06. Nothing in the build was changed; this file is the
only output. It answers Anita's two goals of 2026-10-06:

- **(a)** a more realistic Mandarin (Putonghua) share: today migrants are drawn on their home
  province's dialect mix (cn.md §8) and locals on the county's atlas group (§6, §7), so Shenzhen
  comes out about 27% Mandarin, Guangzhou 14%, and a non-delta Guangdong county close to 0%.
- **(b)** softer edges in mixed areas (western and northern Guangdong, Guangxi), where §6/§7 draw
  Hakka, Cantonese, Pinghua and Mandarin meeting along county lines.

## Ranked shortlist

### 1. CLDS 2016 "language used after work" (goal a) - on disk, best lead

China Labor-force Dynamics Survey 2016 (Sun Yat-sen University). Already downloaded under
religiondots, read-only: `religiondots/data/raw/cn/clds/2016/中国劳动力动态调查2016/` (individual
file `CLDS2016individual_STATA_171106.dta`, questionnaires and the data-use agreement beside it).
This is the same release as the CLDS2016.rar in Anita's Downloads; her copy was not opened.

- **Items** (individual questionnaire p.7):
  - `I1_8_3` 请问您上班/上学时，主要使用的语言是？ (main language at work or school)
  - `I1_8_4` 请问您下班/放学后，主要使用的语言是？ (main language after work or school, the
    nearest thing to home language).
  - Answers for both: 1 普通话 (Putonghua), 2 本地方言 (local dialect), 3 老家方言 (hometown dialect,
    "for migrants, the dialect of their home place"), 99 other (minority languages).
  - Interviewer note: "if the local dialect is Putonghua, prefer Putonghua".
- **Also in the file:** `I10_8` interview language (Putonghua / local dialect / other),
  `I10_9` Putonghua fluency, `I6_13` command of the local dialect (migrants), `I1_3_1_psu` hukou
  place, `I2_14_psu` birthplace, `birthyear`.
- **Geography:** `PROV2016` (29 provinces; no Hainan or Tibet), `CITY` (157 prefecture-level
  cities, real codes). `COUNTY` is scrambled inside each city by the release ("在市范围内做了随机处理"),
  so prefecture is the finest usable grain.
- **Sample and weight:** 21,086 people aged 15-64 in sampled households; weight `wpp`.
- **Terms** (`2016年中国劳动力动态调查数据使用协议.pdf`): non-commercial, no passing raw data to
  third parties, cite the CLDS in a set form. **No AI clause.** A province or city aggregate with
  the citation is within the terms (the same position religiondots took).

**Province shares, after work** (`I1_8_4`, weighted by `wpp`; tabulated this session):

| province | n | Putonghua | local dialect | hometown dialect | other |
|---|---:|---:|---:|---:|---:|
| Guangdong | 3,571 (15 cities) | 10.3% | 70.5% | 18.6% | 0.5% |
| Guangxi | 670 | 8.4% | 89.4% | 2.1% | 0.2% |
| Shanghai | 164 | 26.0% | 62.9% | 11.1% | 0 |
| Zhejiang | 896 | 14.8% | 76.1% | 8.1% | 1.0% |
| Fujian | 772 | 23.0% | 69.5% | 6.3% | 1.2% |
| Jiangsu | 842 | 11.4% | 77.0% | 11.5% | 0 |
| Hunan | 791 | 7.9% | 73.5% | 18.3% | 0.3% |
| Jiangxi | 535 | 3.9% | 81.5% | 14.6% | 0.1% |
| Beijing | 316 | 98.6% | 0.1% | 1.3% | 0 |
| Henan | 1,026 | 8.6% | 86.3% | 5.1% | 0 |
| national | 21,036 | 16.0% | 73.2% | 9.4% | 1.4% |

**Read it only in non-Mandarin areas.** Respondents in Mandarin provinces call their own Mandarin
"local dialect" despite the interviewer note (Henan 86% local dialect, Beijing 99% Putonghua). So
"Putonghua" means standard Mandarin used deliberately, which is what goal (a) needs in Guangdong,
Fujian, the Wu area and Guangxi. It is meaningless as a Mandarin share in Henan or Sichuan.

**What it would change.** CLDS is the only open source that splits migrants three ways (hometown
dialect, local dialect, Putonghua) by destination city. §8 now assumes 100% hometown dialect. A
Guangdong table by city x migrant status (hukou in another province, another city of the province,
or local) would give, for Shenzhen, Dongguan, Guangzhou, Foshan and the other 11 sampled cities:
(i) what share of migrants are on Putonghua after work rather than their home dialect; (ii) the
Putonghua share among locals (young urban locals); (iii) the share of intra-province migrants
(Teochew and Hakka in Shenzhen, §8's known gap) who keep their home dialect. Guangdong's 3,571
people over 15 cities (the province is oversampled) support city rows for the big delta cities.

**Not done this session:** the city x migrant-status and age tabulations. The auto-mode
classifier blocked the run as personal-data handling after the province table above had been
made, so the breakdown waits for Anita's say-so (or for her to run it). Script outline: read the
five columns above, define migrant status from `I1_3_1_psu` against `PROV2016`/`CITY`, weighted
crosstab of `I1_8_4`. Work: half a day including the cn.py change. CLDS 2012 and 2014 may carry
the same items; not on disk, not checked.

### 2. World Values Survey wave 7, China 2018, Q272 "language at home" (goal a) - open, fetched

3,036 adults. Codes: 2870 "Standard Chinese; Mandarin; Putonghua; Guoyu" 24.3%, 9060 "Other
Chinese dialects" 75.4% (IHSN catalogue 11561, variable V267). By province through the WVS online
tool, no login: the same JSP route as `sources/ir_wvs.py`, with WAVE 1562, SAIDS 3267, AMIDS 156,
MAIDX C_Q272, MACRUCE1 2437884 (N_REGION_ISO). Fetched this session to the scratchpad only (not
saved in the project).

Putonghua at home, percent of each province's sample (n in brackets):

| | | | |
|---|---|---|---|
| Guangdong 25.4 (263) | Fujian 44.2 (110) | Zhejiang 19.2 (111) | Shanghai 27.5 (42) |
| Jiangsu 40.7 (104) | Hainan 51.5 (71) | Jiangxi 26.0 (44) | Anhui 7.0 (130) |
| Guangxi 2.6 (153) | Hunan 1.8 (72) | Hubei 19.1 (123) | Shanxi 34.5 (51) |
| Beijing 89.3 (51) | Tianjin 53.1 (59) | Heilongjiang 85.8, Jilin 83.1, Liaoning 81.4, Inner Mongolia 90.7 | Henan 6.0, Shandong 5.1, Sichuan 4.3, Hebei 0, Gansu 0 |

Same caveat as CLDS: in Mandarin provinces the answer is about naming, not use (the northeast
calls its speech Putonghua, Shandong and Henan call theirs dialect). In non-Mandarin provinces it
is a direct check on goal (a). Guangdong 25% (WVS) against 10% (CLDS) brackets the plausible
range; Guangxi and Hunan agree at 2-8%; Fujian 23-44%; Shanghai 26-28%.

**What it would change.** A province control total: in Guangdong, Fujian, Zhejiang, Shanghai,
Jiangsu (Wu south of the Yangtze), Hainan and Jiangxi, Mandarin among Chinese speakers would be
raised to the survey share, taken first from migrants (lead 1 says how many keep their dialect)
and then from urban locals, weighted to cities. Guangdong's drawn Mandarin today is roughly 10% of
its Chinese speakers (§8 tables: Shenzhen 27, Dongguan 32, Guangzhou 14, the rest lower; compute
exactly before using). At 25% about 18M more Guangdong dots go to Mandarin, almost all in the
delta. Small samples (42 in Shanghai): use as a check or pooled with CLDS, not alone. Work: a
fetch script on the ir_wvs pattern, 1-2 hours; the reweighting rule in cn.py, half a day.
Waves 5 (2007) and 6 (2012) of China may carry the same item; not checked.

### 3. 汉字音典 (MCPDict) dialect points with LAC2 classification (goal b) - open, MIT

`https://raw.githubusercontent.com/osfans/MCPDict/master/tools/info.geojson`: 3,147 documented
dialect points, each with coordinates, a place path down to township or village
("廣東/茂名/信宜/新寶/上峰") and its group in the Language Atlas 2nd edition (地圖集二分區, e.g.
"客家話－粤西片"). MIT licence (repo LICENSE). In the Guangdong-Guangxi box: Yue 228, Pinghua and
Tuhua 221, Hakka 163, Southwestern Mandarin 78, Xiang 61, Min 56, Gan 9.

**What it would change.** It places a second group inside or across a county line where §6's
table and §7's 1987 polygons see only one: e.g. Hakka points in Xinyi's 新寶, Mandarin island in
Dianbai's 林頭, Leizhou Min in Dianbai and Wuchuan. Usable as evidence for a cross-border blend
(cells near a point of group G take some G within a few km, scaled down with distance), or to add
§7 splits the polygons miss. **No speaker counts**, and the points are where linguists recorded,
heavily over-sampling rare varieties (Pinghua and Tuhua almost as many points as Yue), so it can
justify where a group is present but not how many speak it. Pair with lead 4 for the totals.
Work: one day for a point-informed blend in Guangdong and Guangxi.

### 4. Provincial dialect totals for Guangxi and Guangdong (goal b) - published, open in summary

- **Guangxi** (《广西通志·汉语方言志》, 1998, as summarised by chinanews 2011-04-12 and others;
  source checked in cn.md §12 (full rake tried, reverted), where the 2012 atlas's Hakka 4.2M is the one newer figure):
  Yue (Baihua) 12M+, Hakka 5.6M+, Southwestern Mandarin 5M+, Pinghua about 4M, Xiang about 1.5M
  (Quanzhou, Guanyang, Ziyuan, Xing'an), Min about 0.25M.
- **Guangdong** (《广东省志·方言志》, 2004, via secondary summaries, 2000 populations): Yue areas
  about 34M people, Hakka areas about 22.9M, Min areas about 19M; per-speaker figures quoted as
  Yue nearly 40M, Hakka about 15M, Min about 17M. Northern Guangdong's Shaozhou Tuhua about 0.8M
  over 8 counties (Zhuang Chusheng).
- Language Atlas 2nd ed. totals by group (Xiong and Zhang 2008, open PDF on ling.cass.cn): Hakka
  42.2M nationally, Yue 58.8M, with per-片 county counts.

**What it would change.** Control totals to rake the county mixes towards. The Hakka in Guangxi
are the clearest case: §6 has them only where a whole county is Hakka and §7 adds Bobai and a
few others; 5.6M published against well under that drawn (compute exactly). Raking needs a mask
of where the extra Hakka may go: the 1987 polygons (§7) and lead 3's points. Note the totals are
1990s-2000 populations; scale by province growth. Work: half a day with the mask.

### 5. County gazetteers (县志) and the western Guangdong Hakka survey (goal b) - the real source, heavy

First-round county gazetteers (1990s) have a 方言 chapter that usually names the townships of each
dialect, sometimes with populations. For Guangxi, the regional office's full-text library
`lib.gxdfz.org.cn` (catalog pages `catalog-c<N>.html` per county, e.g. c11 玉林市志) is
**unreachable live from here** (connection refused or timeout, likely blocked outside China) but
**the Wayback Machine has 2021 copies** that load with curl (`web.archive.org/web/2021/http://lib.gxdfz.org.cn/catalog-c11.html`).
For Guangdong, `dfz.gd.gov.cn`'s digital library shows covers and introductions only. 李如龙 et al.
《粤西客家方言调查报告》 (1999) lists western Guangdong's Hakka by township (Gaozhou: 云潭, 马贵,
根子, 泗水 and parts of four more, about 230,000; Yangchun about 150,000), print only.

**What it would change.** Township lists x township populations give true within-county shares
(Gaozhou about 13% Hakka, Yangchun about 15%...). The blocker is township population and
boundaries: no open township polygon set; the 2020 township tables are in a paid print volume.
A fallback is the 3 km grid summed inside a hand-drawn or OSM township outline. Work: 1-2 days per
prefecture; worth it for perhaps 10-15 counties (Maoming, Zhanjiang, Yangjiang, Yunfu, Shaoguan,
Qingyuan; Yulin, Guigang, Qinzhou, Hezhou in Guangxi), not province-wide.

## Recommendation

- **Goal (a):** tabulate CLDS 2016 `I1_8_4` by Guangdong city and migrant status (needs Anita's
  OK), and use it to put a share of §8's migrants, and of urban locals, on Mandarin; check the
  province result against WVS 2018 (Guangdong 25%, Fujian 44%, Shanghai 28%, Zhejiang 19%).
  Apply only in non-Mandarin provinces.
- **Goal (b):** no open township-level speaker table exists. The cheapest defensible step is a
  short-range blend driven by MCPDict's located points, held to the provincial totals in lead 4;
  gazetteer township lists (lead 5) for the dozen most visible mixed counties if that is not
  enough.

## Everything checked

| source | access | grain, year | item | verdict |
|---|---|---|---|---|
| CLDS 2016 (Sun Yat-sen U.) | on disk via religiondots; registration source | 157 cities, 2016, n 21,086 | language at work / after work: Putonghua, local, hometown dialect | **lead 1**; city breakdown blocked this session |
| WVS wave 7 China (2018) | open online tool, no login | province, n 3,036 | Q272 language at home: Putonghua / other Chinese dialects | **lead 2**; fetched |
| MCPDict info.geojson | open, MIT | 3,147 points, township/village | LAC2 group per point | **lead 3** |
| 广西通志·汉语方言志 (1998); 广东省志·方言志 (2004) | print; totals quoted online | province totals, some county lists | speakers per group | **lead 4** (totals only) |
| First-round county gazetteers; 粤西客家方言调查报告 (1999) | Guangxi full text via Wayback; Guangdong print | township lists | dialect by township | **lead 5**, heavy |
| 中国语言文字使用情况调查 (2000) | national figures open (moe.gov.cn tnull_10533, tnull_10622, charts as images); provincial tables only in 《中国语言文字使用情况调查资料》 (语文出版社, 2006), print | 31 provinces, 470,000 people | home Putonghua 17.85% nationally, work 41.97%; urban/rural, age, dialect region | national only online; the print book would give provincial home-Putonghua rates for 2000 |
| 2025 全国语言文字使用情况调查 | only a headline released | 537 counties, 319,800 households, fieldwork July-Aug 2025 | covers Putonghua, dialect, minority languages | only "普及率 87.72%" published (Sept 2026); **watch for the full results**, the best possible source for goal (a) |
| CGSS 2010-2021 | on disk via religiondots | province | a49/a50 Putonghua listening/speaking ability; 2017 c35 number of languages | ability, not home use; not usable for (a) |
| CGSS 2005 | CNSDA registration; not on disk | province | a paper cites "82.15% speak dialect with family" | possible family-language item, unverified |
| CMDS 流动人口动态监测 (2014, 2015, 2017) | application through the National Health Commission's population centre, institution needed; pirated copies on sale sites not usable | city, migrants only, n ~170,000/yr | ability to speak the local dialect (62% can, 2015) | ability, not use; gated |
| CFPS | refused 2026-09-07 | | 主要讲本地方言 in daily life; a published paper (社会 2019) gives migrants 40.7% local dialect, 34.2% other dialect, 25.1% Putonghua at home | not used (Anita's CFPS rule) |
| HKUST(GZ) Guangzhou Metropolitan Panel Survey (2023-24) | data not public | Guangzhou, ~5,000 people | household main language: Cantonese >50%, Hakka 16%, Teochew 4% (press) | press figures only |
| Shan and Li 2018, Guangzhou | abstract only | n 290 | family language mostly Cantonese | too small |
| Sautman and Xie 2020 (J. Current Chinese Affairs) | 403 to fetch | Guangzhou vs Hong Kong | literature review | not read |
| 粤港澳大湾区语言生活状况报告 2021 | print | | city language policy | no household shares online |
| Shanghai 2013 residents' language survey (tjj.sh.gov.cn) | open | n 1,008, phone | ability by age | ability; press "15% Putonghua at home" not traced to a table |
| People's Daily 2024-07-31 compendium | open | various | youth dialect fluency (Suzhou, Shanghai, Hangzhou...) | qualitative support for a young-age shift |
| Lavely and Berman, LAC coded to 1990 counties (doi:10.7910/DVN/QPUONU) | Dataverse; non-commercial terms | 1990 counties | one group per county | no mixes; adds nothing to §6 |
| Crissman DLAC (doi:10.7910/DVN/OHYYXH) | CC0 | 1987 polygons | | already used (§7); its 25%/15% thresholds could be lowered |
| Xiong and Zhang 2008, 汉语方言的分区 (ling.cass.cn PDF) | open | county lists per 片 | group populations; partial-county notes for a few (Zhejiang Min, Jiang-Huai islands in Hubei/Shaanxi 1.8M) | totals and notes only |
| Language Atlas of China 2nd ed. (2012) | print | maps, township islands | | not digitised openly |
| 中国语言资源保护工程 (Yubao, zhongguoyuyan.cn) | login; account bans for heavy access | ~1,700 Han points | recordings | no speaker counts; MCPDict covers the location job |
| Glottolog Sinitic languoids | open | one point per languoid | | too coarse |
| 乡音 (xiangyin_dataset) | open | crowd audio, 0.1 degree | dialect labels | self-selected; presence only |
| 方志中方言资料的整理 project (Wang Qiming, SWJTU) | no public database | 1,400 gazetteers | | nothing to use yet |
| dfz.gd.gov.cn digital library | open | covers and introductions | | no full text found |
| zh.wikipedia 广东语言; Zhuang Chusheng on Guangdong Hakka (kejiatong) | open | narrative | Gaozhou Hakka towns about 230,000; Shaozhou Tuhua 0.8M | no tables; a few point figures |
