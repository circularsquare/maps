# Chinese folk religion as practice: China, Taiwan, Hong Kong (2026-09-15)

**Why this file exists.** Anita, 2026-09-15 (`ask/RULINGS.md`): China's ~88% `unknown` and Taiwan's
48% `chinesefolk` both misstate practice, so the three should move to one consistent middle-ground
measure. **Decided and built the same day as spec §3.13**: section 4's rule, with Hong Kong on Pew's
21% of the unaffiliated directly (no bridge from Taiwan). Built by `sources/cn_altar.py`,
`sources/tw_altar.py` (with `sources/folk_altar.py`) and `taxonomy/hk2021.py`'s `ALTAR_FOLK_SHARE`;
the results are section 8.

The same day, `chinesefolk` was relabelled "Chinese folk religion" (it was "Chinese religions").

Three research passes are kept beside this file with exact Chinese wording, URLs and every dead end:
- `folk_practice_notes_china.md`
- `folk_practice_notes_tw_hk.md`
- `folk_practice_notes_instruments.md`

They were written as scratch notes, so "coordinator computing" in them means section 3 below.

All figures are % of adults.

---

## 1. Why the map jumps at the Taiwan Strait

**China.** Pooled CGSS self-identification: ~3% choose `民间信仰（拜妈祖、关公等）`, ~88% are grey.

**Taiwan.** TSCS 2014+2018 long card: 48% folk religion. The respondent answers in their own words
and the interviewer codes it.

| TSCS folk codes, unweighted (2018 report 附錄四) | 2009 | 2014 | 2018 |
|---|---:|---:|---:|
| 021 自己認為的, volunteers "folk religion" | 3.1 | 2.5 | 1.5 |
| 022 拜神的, worships the gods | 35.9 | 42.7 | 46.4 |
| 023 沒有清楚認定, not clearly stated | 3.6 | 3.1 | 1.3 |

- The 2018 report's interviewer rules (pp. 103-104) put 自認無宗教信仰但會跟著家人一起拜 on 022: people
  who say they have no religion but worship along with their family.
- 031 信佛、拜佛 (~11%) is the Buddhist twin of 022.
- Weighted (ARDA files, `wr_19_5`): 021 is 2.40 / 1.46, 022 is 42.72 / 45.92.

**Hong Kong.** HKPCS 2021 offered no folk box: 65.8% no religion, drawn `unknown`.

**So people who name folk religion themselves are ~2-3% on both sides.** The difference is the card.

## 2. Practice measures found

| measure | China | Taiwan | Hong Kong |
|---|---|---|---|
| names Buddhism, Taoism or folk religion | ~6-8 (CGSS) | 77 long card; ~57 Pew 2023 card | 15-18 (HKPCS 2021, Pew 2023, HKCC 2021) |
| home shrine, altar or religious object, "for religious reasons" (ISSP item) | **19.8** (CGSS 2010, section 3) | **72.5 / 69.5 / 71.3** (TSCS 2009/2014/2018 report) | no ISSP item |
| "an altar in your home" (Pew 2023) | not asked | 49 | 25 |
| worshipped gods or spirits at a temple or at home, past year | **27.2** weighted, 25.1 unweighted (CSLS 2007, `data/raw/cn/slsc2007.dta`) | | |
| deity image or ancestral tablet at home | 28.0 (CSLS 2007) | ancestral tablet 61 (TSCS 1994/99) | |
| burns incense to Buddha or deities a few times a year or more | **26**, monthly 11 (CFPS 2016 via Pew 2023, **CFPS-derived**) | | |
| went somewhere to pray for good luck, past year | **24** (CGSS 2018 via Pew 2023) | | |
| prays to gods, or temple for a religious purpose, several times a year or more | | 61, non-Christian ~59 (TSCS 2018); monthly ~37 | |
| prays to or honours Guanyin | | 69 (Pew) | 30 (Pew) |
| ancestor rite, past year | 72-75 (CSLS 2007, CGSS 2018) | 81-83 (Pew, TSCS) | 57 (Pew) |

- Ancestor rites are high everywhere, so they do not separate practitioners from anyone else.
- CSLS 2007's weight moves villages from 24.8% of the sample to 63.1% of the weight, which raises the
  rural items.

## 3. The ISSP home-altar item in CGSS 2010, by province

**Item.** `na`: 您家里是否有出于信仰宗教的原因而设的神龛、祭坛、或者摆放宗教物品.

**Sample.** The ISSP 2008 module went to 4,203 of 11,783 respondents. Weight `WEIGHT`, province `s41`.

**National, by religion answer:**

| answer | share | altar |
|---|---:|---:|
| no religion | 85.4 | 13.9 |
| Buddhism | 6.2 | 62.2 |
| folk religion | 2.9 | 67.9 |
| all adults | 100 | 19.8 |

- "No religion" and an altar together are **11.9%** of all adults.
- Names a religion, or has an altar: **26.4%**.

**By province, sorted by altar share.** "Named" is % who name any religion; "folk" is % naming folk
religion; "none + altar" is % of all adults.

| province | n | named | altar | temple several/yr+ | folk | none + altar |
|---|---:|---:|---:|---:|---:|---:|
| Guangdong | 187 | 20.1 | 50.4 | 17.0 | 9.1 | 33.2 |
| Xinjiang | 40 | 85.5 | 48.2 | 36.0 | 0.0 | 0.0 |
| Fujian | 100 | 33.9 | 42.0 | 23.0 | 19.5 | 18.9 |
| Jiangxi | 151 | 13.7 | 37.0 | 7.0 | 0.4 | 24.3 |
| Tibet | 17 | 51.0 | 35.9 | 17.0 | 0.0 | 0.0 |
| Ningxia | 40 | 96.2 | 32.1 | 52.5 | 0.0 | 2.1 |
| Guangxi | 167 | 20.1 | 28.1 | 4.9 | 16.3 | 13.8 |
| Shanghai | 174 | 18.8 | 26.5 | 7.9 | 0.0 | 17.7 |
| Hunan | 164 | 6.9 | 23.7 | 4.7 | 2.5 | 21.6 |
| Jiangsu | 187 | 7.8 | 23.3 | 7.7 | 0.4 | 18.9 |
| Guizhou | 122 | 15.5 | 21.5 | 7.3 | 6.2 | 13.2 |
| Inner Mongolia | 31 | 5.2 | 20.6 | 0.0 | 0.0 | 15.5 |
| Qinghai | 35 | 23.1 | 20.4 | 7.4 | 3.4 | 8.8 |
| Hainan | 36 | 18.6 | 19.9 | 1.3 | 6.6 | 6.3 |
| Zhejiang | 195 | 20.0 | 19.8 | 7.1 | 0.0 | 10.9 |
| Henan | 203 | 19.2 | 18.7 | 9.4 | 1.3 | 6.7 |
| Shandong | 222 | 8.9 | 18.2 | 3.4 | 2.7 | 14.3 |
| Yunnan | 151 | 23.0 | 16.8 | 11.7 | 5.1 | 12.6 |
| Hebei | 115 | 9.8 | 14.7 | 6.9 | 1.6 | 8.8 |
| Sichuan | 224 | 6.6 | 14.7 | 4.9 | 2.0 | 9.6 |
| Anhui | 137 | 2.9 | 14.7 | 2.2 | 0.0 | 13.9 |
| Heilongjiang | 220 | 8.7 | 13.0 | 3.7 | 0.0 | 5.5 |
| Chongqing | 117 | 1.3 | 12.7 | 1.8 | 1.3 | 12.7 |
| Jilin | 175 | 8.9 | 11.6 | 6.1 | 0.0 | 5.2 |
| Tianjin | 150 | 11.7 | 11.4 | 2.4 | 0.0 | 4.3 |
| Shaanxi | 148 | 13.5 | 10.0 | 3.8 | 0.0 | 6.6 |
| Beijing | 162 | 4.8 | 8.7 | 1.8 | 0.0 | 5.2 |
| Hubei | 197 | 5.1 | 7.7 | 5.1 | 0.0 | 6.8 |
| Liaoning | 147 | 6.9 | 5.1 | 3.7 | 2.1 | 1.4 |
| Shanxi | 104 | 10.6 | 5.1 | 5.6 | 1.0 | 2.6 |
| Gansu | 85 | 7.6 | 3.9 | 2.3 | 0.0 | 1.0 |

- In Xinjiang, Ningxia and Tibet the item picks up Muslim and Tibetan Buddhist objects. There,
  "none + altar" is near zero.
- Zhejiang's 19.8% sits low against its ~34,000 registered folk temples (2013). Its cell is one wave
  and a few sampling units.

## 4. A candidate rule, and what it would draw

**The rule.** Draw `chinesefolk` for people who name folk religion, and for people who name no religion
but keep a religious home shrine or altar. In Taiwan, "no religion" includes codes 022 and 023.
Everyone who names a religion stays on it. People who name nothing and keep no altar stay grey.

**China, CGSS 2010 module weights**
- Folk: 2.9 + 11.9 ≈ **14.8%** nationally.
- Grey: from ~85% to ~73%.
- Guangdong ≈ 42, Fujian ≈ 38, Jiangxi ≈ 25, Shanghai ≈ 18, Beijing ≈ 5, Gansu ≈ 1.

**Taiwan, weighted**
- 2014: 2.4 + 32.1 (022/023 with an altar) + 4.8 (none with an altar) = **39.3%**.
- 2018: 1.5 + 34.5 + 6.5 = **42.5%**.
- Grey or none without an altar: ~19-21%.

**Level check, China.** "Names a religion or keeps an altar" is 26.4%. The independent measures:
- CSLS 2007, worshipped at a temple or at home: 27.2%
- CFPS 2016, incense a few times a year or more: 26% (**CFPS-derived**)
- CGSS 2018, prayed for luck: 24%

**Four instruments, three houses, 2007-2018, within three points.**

**Hong Kong.** It has no ISSP item. Either leave it on naming, or bridge Taiwan's ISSP share by Pew's
Hong Kong/Taiwan altar ratio (25/49). The bridge assumes Pew's wording shifts the answer the same way
in both places.

## 5. Caveats for whoever builds it

- **The altar item is not worded identically.**
  - TSCS Q55 prompts with examples: 八卦、符咒、神像、避邪鏡、金錢豹、蟾蜍、門神、劍獅、照壁、照片、真言、咒語、護身符、十字架、經典.
    Those invite a yes for protective objects.
  - CGSS 2010 says only 神龛、祭坛、或者摆放宗教物品.
  - So Taiwan's ~70% probably runs high against China's figure.
  - CGSS 2018's C21 adds 观音像、神位、神符 but is not open (CNSDA application).
- **CGSS 2010 is one wave.** Provinces hold 17 to 224 module respondents, and the §14.10 split-half test
  has not been run. Six provinces are under 50.
- **ARDA's English variable labels for TSCS 2018 are not the Chinese questions.**
  - `v28` is ancestors only (祭拜祖先).
  - `v32` is religious activities (進香、禪修、做禮拜…), not temple visits.
  - That is why "several times a year or more" reads 75% in 2014 (`v24`) and 32% in 2018. Read the
    Academia Sinica 報告書 before trusting a label.
- **China's `note_public` may have the 2021 folk collapse wrong.** Pew, *Measuring Religion in China*
  (2023) pp. 118-119, attributes CGSS 2021's 0.2% mainly to nine higher-folk provinces going
  unsampled (6% against 1% in 2018). The note credits a sixteenfold swing to question wording alone.
  Not re-checked here.
- **The rule reverses spec §3.1 for this node**, and `sources/hk.md` §7's line that §3.1 forbids mixing
  practice with naming. If it is adopted it needs its own spec section.

## 6. Compiler figures, traced

- **CIA Factbook, China** (folk 21.9, Buddhist 18.2, unaffiliated 52.1, "2021 est.") is Pew's *Global
  Religious Landscape* (2012) row for 2010.
  - Built from CSLS 2007 with a loose folk definition: anyone not in another group who worshipped gods
    or spirits, or burned incense in a temple, or believed in gods, spirits, heaven, hell, the
    afterlife or reincarnation.
  - Buddhist 18 is a belief item.
  - Pew 2025 dropped the method for CGSS 2010 and 2018: China 2020 is unaffiliated 89.6, Buddhist 3.7,
    other including folk 3.0.
- **CIA Factbook, Taiwan** (Buddhist 35.3, Taoist 33.2, "2005 est."): the Interior Ministry's estimate
  as quoted in the 2006 government yearbook. Snippet only, not opened.
- **CIA Factbook, Hong Kong** (2016 est.): matches religious bodies' own counts over population, which
  `hk.md` §2 rejects.
- **Pew 2012** folk religion: Taiwan 44.2 (TSCS 2009, adjusted), Hong Kong 12.8 (Asian Barometer
  2001, adjusted).
- **World Religion Database** folk religion (via ARDA): China 29.56, Taiwan 42.50, Hong Kong 42.50. The
  identical Taiwan and Hong Kong shares look assigned, not measured. No method found.

## 7. Not reached, and the next leads

- **Yang and Hu 2012**, "Mapping Chinese Folk Religion in Mainland China and Taiwan", JSSR 51(3):
  505-521. The direct cross-strait comparison; paywalled everywhere tried.
- **EASS 2018 integrated file** (ICPSR 38489): grave visits, praying for luck and charm on identical
  wording in China and Taiwan, coded by province and county. 403 to fetchers; probably a free login.
- **AsiaBarometer 2006** (openICPSR 163441): the only questionnaire fielded in all three. Religion items
  unchecked.
- **CGSS 2018**: ISSP altar item with examples, by province. CNSDA application only.
- **Surveys that might split by province:** Li Xiangping's 2011 Yangtze Delta survey (ECNU) and Yu Tao's
  2008 six-province rural survey. Results not online.
- **East China Normal University 2005** (31.4% "religious", ~200M Buddhist, Daoist or folk-deity
  worshippers): press reports only, no wording.

## 8. What was built, 2026-09-15

| | `chinesefolk` before | after | grey before | grey after |
|---|---:|---:|---|---|
| China | ~3.1% | **16.1%** (215M) | `unknown` 88.1% | `unknown` 75.0% |
| Taiwan | 50.4% | **43.1%** | `unknown` 0, `unaffiliated` 12.1% | `unknown` 19.4% (13.0% until code 010 without an altar left `unaffiliated` the same day) |
| Hong Kong | 0 | **13.7%** | `unknown` 65.4% | `unknown` 51.7% |

- **China**: the CGSS 2010 altar rate among no-religion respondents passes the province test (p 0.0025)
  and is shrunk with a prior of 16 respondents. Folk religion is 46.9% of Guangdong, 34.5% of Fujian,
  25.1% of Jiangxi, 6.0% of Beijing and 3.1% of Liaoning.
- **Taiwan**: code 021 at its national 4.0% of the folk answer; codes 022-024 by county altar rate
  (p 0.032); code 010 at the national 47.4% (p 0.052). Folk religion is 65.1% of Yunlin, 61.6% of
  Chiayi County and 32.7% of Taipei.
- **Hong Kong**: Pew's 21% of the unaffiliated, the same in every district.
- **The strait**: Guangdong 46.9% and Fujian 34.5% against Taiwan's 43.1%, where it had been about 18%
  against 50%.
- Section 4's figures were first estimates from unshrunk rates; these are the built ones.
- Section 10 revises China's and Hong Kong's rows the same day: 15.1% (201M) and 13.1%.

## 9. Review, 2026-09-15 (session `cb8b206e-rev5`)

A second reader over `countries/cn.py`, `countries/hk.py`, `countries/tw.py`, `taxonomy/tw2018.py`,
`taxonomy/hk2021.py`, the three altar modules, the normalized CSVs and the raw files, against spec §3.13
and `ask/RULINGS.md` (2026-09-15). `check_md` clean, `built_countries --check` ok (187). `check_rollup`:
tw clean; cn (77.2%) and hk (51.2%) flag derived `unknown` that vanishes with inferred dots hidden, which
predates the redraw and is ruled (2026-09-08 for China, `hk.md` §5a); the altar rows are `modelled` like
the survey layers. Screenshots of all three clean.

**Every figure recomputes.**
- China, `_cn_counts` as built against the same call with `cn_altar.csv` zeroed: `chinesefolk`
  40,887,561 (3.07%) to 215,032,470 (16.13%), `unknown` 88.09% to 75.03%; Guangdong 46.85%, Fujian
  34.52%, Jiangxi 25.13%, Beijing 5.96%, Liaoning 3.10%. `cn_altar.csv` rebuilt from `cgss2010.dta` with a
  separate reader: `na` 有 = 1, 没有 = 2 (28 refusals); 4,203 in the module, 3,622 eligible, 454 with an
  altar, 133 counties, 13.97% weighted, prior 15.99 respondents; every province's n, k, raw and shrunk
  share match to six decimals.
- Taiwan, off `tw.csv`: `chinesefolk` 43.14%, `unknown` 13.00%, `unaffiliated` 6.35% (reassembled
  before: 50.42% and 12.06%); code 021 is 4.0% of the folk answer and code 010 splits at 47.37% in every
  county; Yunlin 65.06%, Chiayi County 61.61%, Taipei 32.75%. Off the raw rounds: `havshrin` 1 = Yes; the
  2014 and 2018 files' value labels do not load (duplicate occupation labels), but code 1 gives 69.5% and
  71.4% unweighted against the report's 69.5 and 71.3, so the coding holds. Codes 022-024 are 2,530
  respondents, 1,844 with an altar, 189 townships, 72.42% weighted; code 010 is 693, 328, 162 townships,
  47.37%.
- Hong Kong: `chinesefolk` 1,017,774 (13.73%), `unknown` 65.39% to 51.66%. Pew's Practices chapter
  (published 2024-06-17, fieldwork June to September 2023) gives the 25% of Hong Kong adults and says
  altars are less common among the unaffiliated; the 21% is in a chart the fetch could not read, and the
  report PDF URL tried returned 404. Unconfirmed, not contradicted.

**The rule is applied the same way in all three**: the share is carved from the `unknown` residual after
the named answers, the new rows are `modelled`, and nobody who named Buddhism or Daoism is counted twice.
Three differences, none of them a rebuild:

- **The altar rate reaches everyone in the residual, not only the kind of person the survey asked.**
  China: of the 174.1M added, 163.2M are Han and 10.9M are minorities, Zhuang 2.73M, Miao 1.45M, Tujia
  1.20M, Yi 1.14M, Mongols 0.74M, Manchu 0.60M and smaller groups. Inner Mongolia's 15.5% is 30
  respondents in one county, and a Mongol's religious altar is more likely Tibetan Buddhist, as a Yi, Hani
  or Bai one is more likely their own tradition, so some of these read as Chinese folk religion and are
  not. It is the CGSS layer's arithmetic, about 5% of the layer. `note_public`'s "The Mongols are
  deliberately absent too" was already loose after the CGSS layer and is looser now. Hong Kong: `hk.md`
  §10 says about 9,000 non-Chinese are drawn on `chinesefolk`, which counts only the three derived rows.
  The residual also holds every `NOT_ASSERTED` row, and the total is about 44,800 (OtherEthnicity 11,550,
  White 8,821, Indian 6,098, Filipino 6,089, Nepalese 4,263, Indonesian 2,542, Thai 1,860, Japanese 1,471,
  Korean 1,246, other South Asian 762, Pakistani 123), 4.4% of the layer. That includes about 10,400
  Indians and Nepalese, whom `note_public` calls "counted and left grey on purpose". Applying
  `ALTAR_FOLK_SHARE` to the `Chinese` row's residual alone would end it for Hong Kong. Not rebuilt.
- **Where no-religion people without an altar go differs**, as RULINGS records: `unknown` in China and
  Hong Kong and for Taiwan's codes 022-024, `unaffiliated` for Taiwan's code 010 (6.35%).
- **The node's public note says China's and Taiwan's surveys "ask the same altar question"**, where
  §3.13 and section 5 here say the TSCS version lists folk objects as examples and probably reads high.
  Something like "carry the same international altar question, Taiwan's with examples of folk objects"
  would match. In `taxonomy/branches.py`, not edited.

**Spec §3.13 against `ask/RULINGS.md`.** The quotes, the exception to §3.1 for one node in three places,
the label, and Hong Kong on Pew's unaffiliated share all agree. Two sentences say more than the ruling:
- The rule paragraph's "People who name nothing and keep no altar stay grey" states as rule what RULINGS
  lists as not decided and what §3.13's own weak-points list calls not ruled on. On the map Taiwan's 010
  people are the "No religion" colour, a different grey from "Religion unknown". "are not drawn on
  `chinesefolk`" would say only what was decided.
- "It is not a precedent for any other node or country" closes what RULINGS leaves open ("whether the
  rule reaches any other country"). Vietnam's `unknown` residual (RULINGS 2026-09-06, splitting it into
  folk practice not decided) is the obvious candidate.
Not edited.

**Smaller things.**
- `basis` in all three still reads as self-identification only, and `countries.py` says basis is never
  silently mixed. The altar half is 39 of Taiwan's 43 points, 13 of China's 16 and all of Hong Kong's
  13.7. A clause such as "with a home-altar question for folk religion" would say it. Not edited.
- The rename reaches every country on `chinesefolk`, so Indonesia's Khong Hu Chu (106,568, a
  state-recognised religion), New Zealand's Falun Gong and the Confucianism rows in au, cz, nz and uk now
  read "Chinese folk religion". The rename is Anita's (`all`), and no `note_public` quotes the old label.
  Recorded only.
- `taxonomy/cn2000.py` `NOT_ASSERTED["Han"]` still ends on §3.1 saying "pick a basis" and "`chinesefolk`
  is still the node waiting for them if that ever changes"; stale since §3.13. Not edited.
- **Fixed**: `countries/cn.py` `gap` said "a religion for 88% of these dots"; it now says 75%, with the
  comment above it dated. Reaches the map on the next tail. `check_md` clean after.

No ask: the spec wording is a record fix, the minority rows are mapping calls, and where no-altar people
go is already listed in RULINGS as not decided.

## 10. Fixes on section 9, 2026-09-15 (session `cb8b206e-folkfix`)

On the supervisor's calls over section 9.

- **Hong Kong**: `ALTAR_FOLK_SHARE` now reaches the `Chinese` row's residual only
  (`taxonomy/hk2021.py` `ALTAR_FOLK_ROWS`). 44,825 non-Chinese came off `chinesefolk`, 10,361 of them
  Indians and Nepalese. `chinesefolk` 13.13% (972,949), `unknown` 52.26%. `hk.md` §10's "about 9,000"
  corrected; the note says 13.1% and that nobody outside the Chinese population is counted this way.
- **China**: the altar share reaches the Han row only, and `cn_altar.py` now computes the rate from Han
  respondents (3,382, 131 counties, median +0.508, p 0.0025). The minorities keep their grey, including
  the Mongols; the reasons are in `cn.md` §11. `chinesefolk` 15.08% (201,035,265), `unknown` 76.08%;
  Guangdong 46.35%, Fujian 34.35%, Jiangxi 25.07%, Beijing 6.14%, Liaoning 2.85%. The CGSS naming layer
  still reaches every row.
- **Spec §3.13**: where people who name nothing and keep no altar go, and whether the rule reaches any
  other node or country, now both read as undecided. The table and the Xinjiang figure are updated.
- **Wording**: `basis` in cn, tw and hk now says folk religion also counts a home altar. The node's
  public note says the two surveys carry the same international altar question but Taiwan's lists
  folk objects as examples and probably reads higher, and that in China and Hong Kong the altar group
  is drawn only among Han Chinese.
- `taxonomy/cn2000.py`'s Han entries now point at §3.13.
- Not touched: which other countries' rows sit on `chinesefolk` (Indonesia's Khong Hu Chu, New
  Zealand's Falun Gong, the Confucianism rows), which is with Anita.

## 11. Rows split off `chinesefolk`, 2026-09-15 (session `cb8b206e-folksplit`)

Anita's ruling on ask 036 (`ask/RULINGS.md`, `cn`, `all`): split rows off the node where the
source's own label justifies it. Confucianism goes to `confucianism`, and Falun Gong under East Asian
new religions. Every other row was not decided, so each one stayed unless its label plainly names
something else.

**Moved.** People as each country draws them.

| country | source row | people | now |
|---|---|---:|---|
| Indonesia | `Khong Hu Chu` | 116,916 | `confucianism` |
| Canada | `Confucian` | 961 (995 national) | `confucianism` |
| Australia | `Confucianism` | 330 | `confucianism` |
| New Zealand | `Confucianism` | 110 | `confucianism` |
| New Zealand | `Falun Gong` | 116 | `eastasiannew` |
| England and Wales | `Other religion: Confucianist` | 76 | `confucianism` |
| Czechia | `konfucianismus` | 13 | `confucianism` |

- **Falun Gong is on the `eastasiannew` parent, and no leaf was added.** There is no Chinese grouping
  node under it. Taiwan's Yiguan Dao and `Other Chinese religions` already sit on the parent
  (`tw2018.py`), as does Brazil's `Outras novas religiões orientais`. 116 people do not justify a
  legend row of their own.
- **Only Indonesia's row draws dots**: 11 of Indonesia's 23,670 at 1:10,000, with totals unchanged
  at both editions. Czechia's 13 are measured and draw a ring in both editions. The other rows are
  allocated and under one dot, so they change legend counts only.
- Thailand (`ขงจื้อ`) and Japan (`jp2024.py`) already mapped Confucianism to `confucianism`.
  Indonesia now holds the node's largest population, ahead of South Korea's 75,703.

**Kept on `chinesefolk`.** Each label names Chinese religion in general, ancestor veneration, or a
declared combination.

| country | source row | people | why it stays |
|---|---|---:|---|
| Australia | `Ancestor Veneration`; `Chinese Religions, nfd`; `nec` | 409; 6; 49 | general or practice labels |
| Canada | `Ancestor veneration`; `Chinese religions and spiritual traditions, n.i.e.` | 1,660; 4,360 (national) | general or practice labels |
| New Zealand | `Chinese Religions nfd`; `nec` | 306; 20 | general labels |
| England and Wales | `Other religion: Chinese Religion` | 111 | general label |
| Mauritius | `Buddhist/Chinese` | 5,053 | one cell for two things; the split (Buddhist 2,178, Chinese 2,434, Other Chinese 441) is national only, so separating it would mean allocating (`mu.md` §5) |
| Singapore | `Taoism` | 303,962 | the census footnotes it as including Chinese Traditional Beliefs (`sg.md` §6) |
| Taiwan | `Buddhism and Taoism, or the three teachings`, and the folk and altar rows | 451,462 for the first | a declared combination; Taiwan is claimed by `c035b00d-folk` and was not touched |
| China, Hong Kong | the CGSS folk layer and the altar rows | | folk religion, section 8 |

- Foreign-born halves built on `taxonomy/origin_religion.py` still put 92% of Chinese and 80% of
  Taiwanese nationals on `chinesefolk`. That is a modelled share, not a labelled row, and was left.
- **Vietnam has nothing on the node.** `vn2009.py` names `chinesefolk` only in a REVIEW sentence,
  and its folk practice is in `unknown`. The ask's list of countries came from a text search.

**Also changed.**
- Node notes in `taxonomy/branches.py`:
  - `confucianism` no longer says Korea is the only source drawn on it;
  - `chinesefolk` says what moved and what stays;
  - `eastasiannew` says Chinese foundings sit on the parent.
- Tree rebuilt: 729 nodes.
- REVIEW text updated in `id2010.py` and `nz2023.py`, with comments at the moved lines in
  `au2021.py`, `ca2021.py`, `cz2021.py` and `uk2021.py`.
- Records added: `au.md` §9, `ca.md` §11, `cz.md` §10, the dated section at the end of `id.md`,
  `nz.md` §11, and `uk.md` §11.

**Checks.**
- `check_mapping.py`:
  - clean for id and cz;
  - au, nz and uk fail on the same parent and total categories as before. Those countries draw
    from allocated files, and no mapping key was removed.
  - Canada is outside the tool (registry SPECIAL), so its rows were checked by resolving every
    national row through the parent chain.
- Both editions rescattered for au, ca, cz, id, nz and uk.
- `coverage.py` reports `id chinesefolk — draws dots but is not in the country's coverage`. That is
  the documented pre-tail case (COMMANDS.txt step 9): coverage compares against the last build's
  `counts.json`, so it shows the node Indonesia used to draw until the tail rebuilds.
