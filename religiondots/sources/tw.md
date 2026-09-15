# Taiwan — `sources/tw.py`, `sources/tw_geo.py`, `taxonomy/tw2018.py`

Built 2026-09-14 by session `d743fc47-tw`. §11i (2026-09-06) closed Taiwan on "nobody asks, and no
survey fills the gap"; `sources.md` §scout-2026-09-14-taiwan-belarus-gabon reopened the survey half.

| | |
|---|---|
| counting geography | 22 counties and cities (縣市), ISO 3166-2; 19 drawn |
| placement | Kontur 400 m hexagons (2023-11-01), keyed to geoBoundaries TWN ADM1 by `sources/tw_geo.py` |
| basis | self-identification, adults, sample survey |
| population base | Ministry of the Interior household register, end of 2025 (民國114年), 23,299,132 |
| tier | `modelled` throughout |
| instrument | Taiwan Social Change Survey (TSCS), 7 rounds 1994-2018, 13,395 placed respondents who answered |
| ask | none |

---

## 1. Route, and what was checked

**The census does not ask** (§11i, the 2020 census read). The MOI's religious-body roll is 6.9% of
the population and certifies its own members (§11i); not used.

**ARDA's open copies of TSCS.** `thearda.com/data-archive?fid=<id>&tab=3` is a click-through
agreement (acknowledge ARDA and the principal investigators; data "as is"; Indiana law), no login,
read 2026-09-14. The files are on OSF. Stata files used:

| round | ARDA id | OSF file | rows | religion | residence | weight (ARDA's) |
|---|---|---|---:|---|---|---|
| 1994 | TSC94 | `q2yvw` | 1,862 | `RELIGION` (short card) | `LIVES`, postcode | none |
| 1999 | TSC99 | `bv7nk` | 1,925 | `RELIGION` (long card) | `LIVES`, postcode | none |
| 2004 | TSC04 | `vgr7e` | 1,881 | `RELBEL` (long) | `LIVES`, postcode | `WEIGHT` |
| 2009 | TSC09 | `c7vw3` | 1,927 | `RELBEL` (long) | `WHRLIVEP` postcode, `WHRLIVEM` county | `WEIGHT` |
| 2014 religion | TSCS142 | `u8kxr` | 1,934 | `V15` (long) | `ZIP`, county of sampling area | `WR_19_5` |
| 2015 | TSCS151 | `t48sd` | 2,034 | `V11` (short) | `ZIP` postcode, `ZIP2` county | `WSEL` |
| 2018 religion | TSCS181 | `xgn86` | 1,842 | `V29` (long) | `ZIP` postcode, `V4CITY` present residence | none printed; `WR_19_5` used, as 2014 |

ARDA's 2009 codebook says 1,928 cases; the Stata file has 1,927.

**Not used, and why.** SRDA's linked religion-module file (7,595 cases, `zip`) and its restricted
village codes need an SRDA membership (§6b: open mirrors first; the seven ARDA rounds are more
than SRDA's four). WVS wave 7 Taiwan and ISSP Taiwan need a download form or a GESIS login; not
opened. The MOI temple registry is a building layer (§8.3 refused).

## 2. Where each respondent is

- **Postcodes to counties**: Chunghwa Post's `1050812_行政區經緯度(toPost).xml` (data.gov.tw
  dataset 25489): 368 three-digit codes, each in exactly one county, all 22 reached.
- **Two witnesses that the postcode is read right**: in 2009 the postcode and `WHRLIVEM` agree on
  all 1,926 placed respondents; in 2015 `ZIP` and `ZIP2` agree on all 2,032. Both asserted.
- **2018's two columns differ on purpose**: `ZIP` is the sampling area and `V4CITY` present
  residence (`V4`: 16.3% interviewed at a current address that is not the registered one). The
  sampling area is used because it is the household register's geography, which is also the
  population base's.
- **Unplaced, dropped and asserted**: 1994 postcodes 316 and 531 (2), 1999 117, 254 and 99 (4),
  2009 one `Other foreign country` (1).
- **The 2010 mergers are whole-county unions** (Taipei County to New Taipei; Taichung, Tainan and
  Kaohsiung counties into their cities; Taoyuan upgraded 2014), so old labels map one to one.
- **Never sampled in any round**: Penghu, Kinmen, Lienchiang. Not drawn (§2 of `sources/ec.md`'s
  line, Anita 2026-09-08 on Galápagos: nothing measured the place).
- **Sampled in only some rounds**: Chiayi City (not 1994), Nantou (not 1994, 2014), Hsinchu County
  (not 1999, 2014, 2018), Hualien (not 2004), Taitung (not 2004, 2009, 2015). The standardised
  index below is built for exactly this; see §4.

## 3. The cards, and the trap

The **short card** (1994, 2015) lists religions; the **long card** (the other five) records the
answer and the interviewer codes it, splitting folk religion into four codes (self-identified,
*worships the gods*, not clearly specified, other) and Buddhism into sects.

**Trap: code 9 is None in 1994 and Cihui Tang in 2015** (None is 10 there). Each round has its own
code table in `tw.py`, and every code's label is matched against the answer it is mapped to, so a
renumbered card stops the build. Two labels do not name their answer and are pinned exactly:
2014's `Dual practicing of Chan and Pure Land` (matched by pattern) and 2018's code 39 `Do not
know`, which sits in the Buddhist block (`LABEL_EXCEPTIONS`).

**The card moves the Buddhist and folk shares**, and that is why the level is read from two rounds
only (§4).

## 4. Construction

- **Level**: 2014 and 2018 pooled and weighted, the two religion-module rounds on the same long
  card.
- **Pattern**: all seven rounds, as observed over expected per county, where each respondent is
  expected at their own round's national shares (indirect standardisation). A county sampled
  mostly on the 1994 card is not read as Buddhist because the card was.
- **Test**: townships nest in counties, so the null regroups townships (spec §12, Puerto Rico).
  200 within-round halvings of the townships; statistic the median Spearman of the two halves'
  county indices; null deals townships to counties within round, keeping each county's township
  count per round, 400 draws; chi-square of observed against expected as a veto; one township
  holding over half an answer refuses (`CELL_CAP`). 2014 has no postcode, so its cluster is
  stratum and PSU, asserted to sit in one county.
- **A second cell cap, inside the top county.** The national cap cannot see one township setting
  the county that tops the rank. `Buddhism and Taoism, or the three teachings` passed the split-half
  (p 0.022, chi-square 2e-61, largest township 10% nationally) on Hsinchu County, whose 35 answers
  are 26 from 2009 postcode 303 (Hukou; 25 of them code 102, three teachings in one) and 9 from 2004
  postcode 310. Refused when the top county's largest township holds over `CELL_CAP` of its answers.
  The five carried answers' top counties: Yilan 26% (none), Chiayi County 13% (folk), Taipei 4%
  (Buddhism), Yilan 32% (Taoism), Taitung 33% (Protestant).
- **Compose**: seed = level x county index, the indices first pulled toward 1 by gamma-Poisson
  empirical Bayes (prior strength 4 to 15 expected answers; Taitung's Protestant index 2.86 to
  1.89, Nantou's 0.00 to 0.45); then IPF to county populations and the 2014-2018 level, 7 passes.
  A plain level x index did not close: seven counties' carried answers came to over 100%, because
  the index is against each round's own shares and the level is 2014-2018's. The tail shares each
  county's remainder at national proportions; the 2x rule's worst is 1.16x, so not flat.

## 6. Results (2026-09-14)

Split-half on 515 townships (70, 76, 83, 88, 67, 66, 65 per round), 19 counties:

| answer | n | median | null 95th | p | chi-square p | verdict |
|---|---:|---:|---:|---:|---:|---|
| No religious belief | 1,991 | +0.493 | +0.325 | 0.010 | 7e-27 | carried |
| Folk religion | 5,184 | +0.696 | +0.289 | 0.0025 | 3e-95 | carried |
| Buddhism | 2,999 | +0.471 | +0.288 | 0.005 | 5e-37 | carried |
| Taoism | 1,826 | +0.596 | +0.274 | 0.0025 | 2e-41 | carried |
| Buddhism and Taoism, or the three teachings | 264 | +0.400 | +0.325 | 0.022 | 2e-61 | refused, top county one township |
| Yiguan Dao | 266 | +0.229 | +0.320 | 0.117 | 0.027 | national |
| Protestant Christianity | 585 | +0.573 | +0.254 | 0.0025 | 4e-24 | carried |
| Catholicism | 179 | +0.269 | +0.321 | 0.095 | 2e-67 | national (Hualien tops both halves in 66%) |
| Japanese, other Chinese, Islam, Other | 101 | | | | | national |

Printed, not deciding: 1994-2004 against 2009-2018 county indices, Spearman +0.42 to +0.55 for the
five carried answers.

National shares by round (weighted where the round has weights), %:

| | 1994 | 1999 | 2004 | 2009 | 2014 | 2015 | 2018 |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 13.0 | 13.6 | 20.9 | 12.8 | 10.4 | 19.3 | 13.8 |
| folk | 31.0 | 33.4 | 29.6 | 42.7 | 48.2 | 37.1 | 48.8 |
| Buddhism | 38.5 | 26.3 | 24.9 | 19.6 | 15.2 | 19.3 | 13.5 |
| Taoism | 9.1 | 12.7 | 15.4 | 13.8 | 15.4 | 16.6 | 12.8 |
| Protestant | 4.2 | 4.8 | 3.3 | 3.9 | 4.2 | 4.1 | 5.5 |

Drawn (2014+2018 level): folk 48.47%, Buddhism 14.42%, Taoism 14.14%, none 12.06%, Protestant
4.84%, Yiguan Dao 2.06%, Buddhism and Taoism 1.96%, Catholic 1.12%, Japanese 0.44%, Other 0.39%,
other Chinese 0.10%. Islam is 3 respondents in seven rounds and 0 in the level rounds, so draws
nothing.

Highest and lowest drawn, carried answers: folk Chiayi County 73.8%, Yunlin 68.2% / Taipei 38.1%,
Yilan 36.8%; Buddhism Taipei 19.5% / Yunlin 7.0%; Taoism Yilan 25.0%, Kaohsiung 21.4% / Chiayi
County 6.0%; Protestant Hualien 11.6%, Taipei 9.5%, Taitung 9.4% / Miaoli 1.7%; none Taoyuan 17.9%
/ Chiayi County 6.3%.

## 7. Calls someone might reverse

- **Sampling area over present residence in 2018** (§2). 124 of 1,842 differ.
- **Level from 2014 and 2018 only.** Pooling all seven would put Buddhism near 22% and folk near
  39%, on a mix of cards.
- **`Buddhism and Taoism` on `chinesefolk`**, not split between `buddhism` and `daoism`
  (`taxonomy/tw2018.py` REVIEW). **Yiguan Dao on `eastasiannew`**, no Chinese child node.
- **Kinmen and Penghu blank**, not given a neighbour's or the national rate (Ecuador's line).
- **geoBoundaries (OSM) polygons** because the NLSC host 403s; both witnesses pass.
- **Not done**: an adults-only age lean; a Hakka or indigenous proxy for the counties (TSCS asks
  father's ethnicity; not used, §3.5b would need a published split to weight by).

## 5. Geography

- **Boundaries**: the NLSC county file (data.gov.tw dataset 7442, `直轄市、縣(市)界線1140318.zip`
  on `www.tgos.tw`) answered **403** to a scripted GET on 2026-09-14; not retried. geoBoundaries
  TWN ADM1 (OSM via Wambacher, 2017, ODbL) is used, 22 features keyed by `shapeISO`.
- **Witness 1, area**: every polygon within 0.97-1.03 of the MOI's land area for its county
  (ODRP048); Spearman +0.999; none of 20,000 shuffled pairings reaches it.
- **Witness 2, people**: Kontur 2023-11-01 per county against the register, Spearman +0.963, none
  of 20,000 shuffles reaches it. Drawn counties 0.81 (Yunlin) to 1.71 (Keelung); Chiayi City 1.39.
  **Kinmen reads 0.40**: its register holds many people who live on Taiwan proper. Kinmen is not
  drawn, and the ratio moves no dot between counties anywhere, since counts come from the register.
- **Grid**: 23,059 hexes; the extract reaches Taiping Island (114.4 E) and Dongsha, which fall in no
  county polygon. 546 centroids outside every county, 510 snapped within 1 km, 36 hexes (2,583
  people) dropped. Median 1,102 hexes per county.
- **Density cap**: two blocks at 46,200/km2, both `real` in `kontur_cap.csv`: the Taipei basin core
  (165 hexes, 56.3% of Taipei City's weight; the register averages 37,022/km2 over all of Yonghe)
  and Sanchong and Luzhou (53 hexes, 20.9% of New Taipei's).

## 8. Review, 2026-09-15 (session `d743fc47-rev3`)

A second reader, working from `countries/tw.py`, `taxonomy/tw2018.py`, `data/normalized/tw.csv` and
the split-half table in §6. `check_md`, `built_countries --check` and `check_rollup tw` are clean
(23,039,482 people, all `modelled`, nothing orphaned). Every figure in `note_public` recomputes from
`tw.csv`: folk religion Chiayi County 73.78%, Yunlin 68.19%, Taipei 38.10%; Taoism Tainan 20.77%,
Kaohsiung 21.41%, Pingtung 20.80%, Yilan 25.02%; Protestant Taipei 9.45%, Hualien 11.59%, Taitung
9.43%. Screenshot: dots on the western plain and the east-coast towns, the Central Mountain Range
empty, Penghu, Kinmen and Matsu blank. Nothing rebuilt or changed.

- **`Buddhism` -> `buddhism.mahayana` goes against spec §2.6, and splits the Chinese-speaking
  countries two against four.** Spec §2.6 (Anita, 2026-09-07) says no country's Buddhists are
  given a school from outside the source, and `kr2015.py` keeps Korea on bare `buddhism`. `hk2021.py`
  also puts a survey's `Buddhism` on `buddhism.mahayana`; `sg2020.py`, `my2020.py`, `vn2009.py` and
  `kr2015.py` put it on `buddhism`. The long card does code sects (Pure Land, Chan, Mi Zong) in five
  of the seven rounds, so this is closer to the source saying it than Hong Kong is. But the 1994 and
  2015 short cards do not, and the REVIEW entry itself says Mi Zong includes Tibetan lineages. It
  makes Taiwan's 3.3 million the largest `buddhism.mahayana` fill on the map, from a survey, which is
  the "more specific as the evidence gets worse" pattern §2.6 describes. The REVIEW entry cites
  neither §2.6 nor Hong Kong. Moves no count; changes which legend row the Buddhists sit on. Left
  for Anita.
- **Catholics at the national rate is an error with a direction, not only a noisy one.** The test
  was applied as written (rank p 0.095) and the note says plainly that Catholics are spread at the
  national rate. But the same row prints chi-square p 2e-67 with Hualien topping both halves in 66%
  of splits, and Taiwan's Catholics are widely reported to be heavily indigenous (not checked
  against a source here). Drawn: Hualien 1.26%, Taitung 1.14%, against 1.12% nationally, so if that
  is right the flat share under-draws the east and over-draws the west. TSCS asks father's ethnicity
  (§7); the Catholic share among respondents with an indigenous father against the rest would size
  the miss without drawing from it. Not done here.
- **The top-county cell cap exists only in `tw.py`.** `stability.CELL_CAP` is national, and the
  spec §12 entry added with Taiwan covers the card index, not the cap. Any survey country that
  carries a category on the national cap alone (LAPOP, ESS, Afrobarometer, `pr.py`) could have its
  top unit set by one cluster, the way Hsinchu County's `three teachings` was 26 of 35 answers from
  Hukou. Whether a drawn country fails it is a supervisor's question; if one does, it changes drawn
  numbers and is an ask.
- **`note_public` opens "Taiwan's census has never asked about religion".** §11i read the 2020
  questionnaire. "Never" also covers the earlier ROC censuses and the Japanese-era ones, which
  nobody here opened. "does not ask" would be safe. Left for the builder.
- **On the map most of the legend is pink.** `Chinese religions`, `East Asian new religions`,
  `Daoism` and `Mahayana Buddhism` read as shades of one colour, and the first two looked almost
  the same in the legend at 1400 px. Worth a human eye; ask 019 left a similar pair alone.
