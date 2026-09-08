# China — 2000 census nationality by county, on 2010 provincial totals

Drawn 2026-09-05; **rewritten 2026-09-07, when 2.3% of the country became 100% of it — §7 below
is that change and spec §14.13/§14.14 are the decisions.** `sources/cn.py` rebuilds
`data/normalized/cn.csv` from `data/raw/cn/`; `data/` is gitignored, so this file is the record.

| | |
|---|---|
| structure | *2000 Population Census Data Assembly* (中国2000年人口普查统计资料汇编), 31 provincial volumes, table A0106 |
| totals | 2010 census, NBS *中国2010年人口普查资料* table 1-6, province × sex × nationality |
| geography | **county (县级行政区), 2,768 drawn units**, mean 481,000 people |
| categories | 59 — the 56 nationalities plus unidentified, naturalised and the unit total |
| basis | `ethnicity_derived`. **Nobody in China has ever been asked about religion in a census** |
| tier | **`derived` on 99.86% of rows, `modelled` on the rest; nothing is `measured`** (spec §7) |
| licence | CC0 on the Dataverse volumes; the NBS table is a government work. Joshua Project's PGIC file is free and keyless |
| drawn | **1,332,807,363 — 100%**, of which 91.65% is `unknown` |

> **Two changes on 2026-09-08 and the numbers below predate both.** spec §14.16 added a
> `self_id` layer from the pooled Chinese General Social Survey — 58.4M Mahayana Buddhists and
> 21.3M Protestants carved out of the grey at province grain, which took China from 2.5%
> coloured to 8.4% (see `sources/cn_cgss.md`). spec §14.17 repaired the county join: **168
> census county names matched no adcode and were being read and then discarded, 67.8M people,
> 5.47% of the country, 142 of them urban districts.** Zhongshan alone was 2.36M, absent
> because the volume romanises 中山 as ZHONGZHAN. Every province now reconciles to its 2010
> total and the national figure lands 17 people from the published census.

## 1. Why this country exists at all, and the rule that lets it

China has never carried a religion question in any census. So unlike every other country on
this map there is no answer to read, and what is drawn is **spec §14.5's permitted
derivation**: an ethnic category may imply a religion where the category was itself
constituted religiously, at no finer geography than the ethnicity is published at, and never
where the group is religiously mixed.

`taxonomy/cn2000.py` argues each of the 56 nationalities and is the file to read before
touching anything here. Fifteen clear that bar; six more carry a **fractional** Christian share
under §14.9, which §14.5 would have forbidden and no longer does (see §7); and the rest claim
nothing and are drawn on `unknown`:

| node | nationalities | 2010 people |
|---|---|---|
| `islam` | Hui, Uyghur, Kazakh, Dongxiang, Salar, Kyrgyz, Tajik, Uzbek, Bonan, Tatar | 23,068,296 |
| `buddhism.vajrayana` | Tibetan, Yugur, Monba, Pumi | 6,345,861 |
| `buddhism.theravada` | Dai | 1,259,298 |
| `christianity.protestant` | a share of Lisu, Lahu, Va, Jingpo, Nu, Derung — **and the Korean nationality since 2026-09-08** | 875,255 + ~549,000 |
| `unknown` | Han and 39 others — counted, nothing claimed | 1,227,791,820 |

**The load-bearing argument is reflect-versus-reveal, not obscurity.** The only input is the
Chinese state's own published county tabulation of its own territory, and no compilation of a
government's published tables can tell that government something it does not already hold.
spec §14.2 drew that line before anyone went looking for China, and §14.5 is where Anita
decided it.

## 2. What is NOT CLAIMED, and every absence is a decision

**The Han, 1.14 billion people — DRAWN since 2026-09-07, on `unknown`, and nothing claimed
about them.** spec §14.5 closed with *"distributing han needs more thought"*; §14.7 decided to show
them without dividing them and §14.13 unblocked it when CFPS refused. What follows is why they are
not divided, which has not changed. Han religion is §14.5's religiously-mixed row, where the derivation does all the work.
The specific problem is that the folk-religion / irreligious boundary is largely an artefact
of the question: ask people in China to name a religion and ~85% name none, ask instead about
grave-tending, temple visits and belief in deities and most of it comes back. spec §3.1 says
pick a basis, and there is no basis on which both answers are true at once. `chinesefolk` is
the node waiting for them; what is undecided is what geography it may claim.

**The Mongols, 5.98M, and this one reverses spec §12 and §14.5's own table.** Both send
Mongol → Tibetan Buddhist. At 5.98M in 2010 that is **more people than Tibetans (6.28M is
close, and in 2000 Mongols outnumbered them outright)**, so Inner Mongolia rather than Tibet
would have been the largest block of Vajrayana dots in China. §14.5 requires the coefficient
to be *"near 1 and documented rather than fitted"*; for Tibetans that documentation is
everywhere and for Mongols there is nothing comparable to point at, decades after the Gelug
monastic system they would have been counted through was dismantled. **Anita's call,
2026-09-05.** Tu (241k) goes with them for the same reason.

**Pumi (33.6k) is drawn and probably should not be** — Pumi religion is Hangui held alongside
Gelug, which is the mixed row by the same argument used against Mongol. Flagged in
cn2000.py's REVIEW rather than quietly kept; it is 34 dots.

## 3. The source, which is far better than sources.md §12 expected

Harvard's **`chinacensus` dataverse** holds the 2000 assembly as 31 datasets, one per
province, **CC0, no auth, no bot wall**. Table `J<gb>A0106.tab` in each is *"Population by
sex, nationality and by township"*: 179 columns = 2 name columns + 59 groups × (total, male,
female), with **county subtotal rows already in it** (`V2 == 'Total'`). So county level needs
no aggregation and the ~40,000 townships underneath come free.

**Township is deliberately not used.** The source reaches it; Anita's call is county, and
§14.5's "no finer than the state publishes" is a ceiling rather than a target. A
township-level map of Uyghur settlement is a different object from a county one.

### The column order is verified, not assumed

The `.tab` files have no header — columns are `V1…V179`. The order is GB/T 3304 census order,
and the check is that it reproduces published figures on groups whose true shares are known:

| | drawn | published 2000 |
|---|---|---|
| Ningxia Hui | 33.9% | 33.9% |
| Xinjiang Uyghur / Han | 45.2% / 40.6% | 45.2% / 40.6% |
| Xizang Tibetan | 92.8% | 92.8% |
| Guangxi Zhuang | 32.4% | 32.4% |
| national Han | 91.6% | 91.6% |

A wrong order would put those on the wrong groups and be obvious immediately, which is worth
more than a column header would have been.

### The province sums prove the parse, and name the one hole

The 31 files sum to **1,239,452,849** against the published 2000 provincial sum of
1,242,612,226. The entire 3,159,377 difference is **Hainan**, whose file carries 14 of its 24
county-level units as name-only rows with no data — the Li and Miao autonomous counties plus
the Paracel, Spratly and Macclesfield groups. Every other province is exact **to the person**,
which is a far stronger check than any tolerance. Sanya, which holds the Utsul Muslims, is
present, so the hole costs the drawn population essentially nothing.

`cn.py` asserts this identity and says so if it ever stops holding.

## 4. The vintage split, and why the totals are 2010 rather than 2020

No county-level ethnic table newer than 2000 is in the open. So spec §3.4 applies — structure
from the detailed source, totals from the recent one — and each group's county figures are
scaled by `province_2010[g] / province_2000[g]`.

**2020 exists and is unusable.** The NBS publishes the 2020 yearbook's table 1-4 as a **3 MB
JPEG scan**; the 2010 edition's table 1-6 is HTML. Anita's call, 2026-09-05, after the
observation that the choice **moves magnitude and not geography** — the shape is 2000's
either way, and 2010 → 2020 is close to a uniform per-group rescale (Uyghurs 10.07M → 11.77M).
Trusting OCR for a country's magnitudes to buy a ~15% size correction was not worth it.

The 2010 parse is checked against published national figures and hits all four exactly: Hui
10,586,087, Uyghur 10,069,346, Tibetan 6,282,187, Mongol 5,981,840.

**The denominator is the file's own province sum, not the resolved subset.** Using only the
counties that resolved would silently redistribute an unresolved county's people into its
neighbours, which is spec §8.1's Connecticut failure in a different hat. The consequence is
that the written rows sum to slightly less than the 2010 provincial total, by exactly the
stranded share — dropped people are dropped, not spread (spec §3.5).

## 5. The county join, which is the fragile part

The census carries romanised names and **no codes**; DataV carries Chinese names and the
**GB/T 2260 adcode**. So this is a name join, and §12 is right that those are where the traps
are. **2,691 of 2,859 counties resolve:**

| how | n |
|---|---|
| by name (pinyin, tiered) | 2,561 |
| by code order | 30 |
| by hand override | 100 |
| **unresolved** | **168**, holding 69,728 drawn people = **0.26%** |

Four things made the difference between 59.9% and 94.1%, and three of them generalise:

- **The file is in GB/T 2260 code order**, and so is DataV's. That is the only thing
  separating **Yining city from Yining county** — the census drops the administrative suffix,
  so both romanise to `YINING`, as do Hetian, Linxia and a few dozen more. The nth occurrence
  of a repeated name is the nth candidate by adcode. Nothing else resolves these, and taking
  the first candidate would have put a prefecture's whole urban Uyghur population in its rural
  half half the time.
- **Two transliteration rules, each fixing a class rather than a case.** The source writes 藏
  as `CANG` where pinyin has `zang` (Tianzhu, Muli, the Tibetan autonomous counties), and 什
  as `SHEN` where pinyin has `shi` (Kashgar). Both **add** a candidate rather than replacing
  one — Shandong's 莘县 really is `SHEN`, and rewriting every trailing `-shen` to `-shi` lost
  a county to fix Kashgar.
- **Keys are tried in priority tiers, never pooled.** Henan's 固始县 is `GUSHI`, whose stem
  `gu` hits half the province, and pooling the full pinyin with the stem made an exact match
  look ambiguous.
- **A city-prefixed district scopes to that city.** The census disambiguates by prefixing —
  `JINAN SHIZHONG` — and Shandong has three 市中区, so the province-wide lookup is ambiguous
  while the same lookup inside Jinan is exact.

**Ambiguity is refused, not guessed.** A name matching two adcodes returns unresolved and
shows up in the report, where it earns an override line if it carries anybody. That took the
stranded figure *up* from 0.86% to 1.27% before the overrides brought it to 0.26% — the 0.86%
was fake, because it counted coin tosses as successes.

**The overrides are validated against the index.** A target that is not a real adcode joins
fine, finds no polygon, and the people vanish — §8.1 exactly. `cn.py` fails on it now, and it
has already caught one: 340203 for Wuhu's Yijiang district, which DataV numbers 340209.

**What remains unresolved is 168 eastern urban districts** holding a few thousand Hui each and
no more than 2,020 in any one. They are dropped rather than spread.

## 6. What to distrust

**The geography is 25 years old and the cities have grown.** Dots are placed by the 2000
pattern, so urban Hui communities are understated relative to rural ones everywhere, and the
coastal migrant communities are placed by a pattern that predates them. `cn.py` reports every
(province, group) rescale factor outside 0.5–2.0, and the ones that fire are exactly this:
**Zhejiang Dai ×11.06, Zhejiang Uyghur ×6.85, Shanghai Uyghur ×3.09, Beijing Uyghur ×2.23.**
Those are real migration and the magnitudes are right; what is wrong is *where inside the
province* those few thousand people are drawn. Read them as "this province, somewhere".

**An IPF against modern county totals was considered and rejected.** Constraining to both
margins — 2010 provincial × group totals and a modern county population — would push minority
mass into counties that grew, which is the right correction in the abstract. It is the wrong
one here, because urban growth in Xinjiang and Tibet was disproportionately Han, so the method
would inflate the Uyghur and Tibetan share of exactly the cities where that number is most
contested. Recorded so it is not rediscovered as a good idea.

**Nineteen (province, group) pairs are 2000-zero and 2010-nonzero** — a group that arrived in
a province after 2000, 1 to 39 people each. They cannot be placed and are not.

**The derivation is the whole claim.** Every dot is `derived` or `modelled`, and China is the
**first country on this map that is 100% not-measured** — turning on `inferred dots: hidden`
empties it completely. That is the honest test of the country and it is worth performing. It
survives the 2026-09-07 rewrite: `unknown` is `derived` for spec §3.4's reason, not for a reason
about religion, so a billion grey dots vanish with the coloured ones.

## 7. The 2026-09-07 rewrite — 2.3% became 100%, and Christianity arrived

spec §14.13 and §14.14 are the decisions; this is what the files now hold.

| | people | share | node | tier |
|---|---|---|---|---|
| §14.5 religio-ethnic, 15 nationalities | 30,673,455 | 2.44% | `islam`, `buddhism.vajrayana`, `buddhism.theravada` | `derived` |
| §14.9 mission peoples, 6 nationalities | 875,255 | 0.07% | `christianity.protestant` | `modelled` |
| everyone else, 40 categories | 1,227,791,820 | **97.50%** | `unknown` | `derived` |

**Superseded twice on 2026-09-08.** Pumi left the religio-ethnic list (§14.16), so it is 14
nationalities and 30.75M; the CGSS layer took 58.4M Mahayana Buddhists and 21.3M Protestants out
of `unknown`; and §14.17's join repair added 73.5M people to every row's denominator. Current:

| | people | share | node | tier |
|---|---|---|---|---|
| §14.5 religio-ethnic, 14 nationalities | 30,706,864 | 2.30% | `islam`, `buddhism.vajrayana`, `buddhism.theravada` | `derived` |
| §14.16 CGSS, 29 provinces | **58,392,846** | 4.38% | `buddhism.mahayana` | `modelled` |
| §14.16 CGSS + §14.9 mission peoples | **22,148,236** | 1.66% | `christianity.protestant` | `modelled` |
| everyone else | 1,221,559,417 | **91.65%** | `unknown` | `derived` |

`taxonomy/cn2000.py` now exposes **`shares(cat) -> [(node, share, tier)]`** rather than a single
node, and `countries.py::_cn_counts` fans one source row out to several rows. `resolve()` is kept
for `tools/check_mapping.py`, which therefore reports each mission nationality's WHOLE population
against `christianity.protestant` — 1.81M, not the 876k actually drawn. That is the tool showing
the mapping rather than the magnitude, and it is worth knowing before it looks like a bug.

**The scatter is 1,259,338 dots at 1:1,000**, which makes China the largest country on the map.
2,530 people nationally fall under one dot and draw nothing; 194 units have polygons and no rows.

### The Christian layer, in one paragraph

Six Yunnan border nationalities carry a fractional Protestant share: **Lisu 79.7%, Lahu 42.0%,
Derung 28.1%, Jingpo 23.9%, Nu 23.2%, Va 15.3%**, the remainder of each going to `unknown` and both
halves marked `modelled`. Coefficients are Joshua Project's `PercentAdherents`, population-weighted
over the people-groups the Chinese state classifies under each nationality — which is why Jingpo is
24% and not its headline group's 54%: the Zaiwa are the larger half at 0.25%. **Selection is not
JP's**: a mechanical threshold over JP returns 121 million Han Christians in the Wu- and Min-speaking
southeast, so the groups are chosen on evidence outside the missionary literature and only the
number comes from JP. §14.14 has the full argument.

**A seventh nationality joined them on 2026-09-08 (spec §14.18): the Korean, at JP's 30%,
~549,000 people, concentrated in Yanbian.** It is the only one of the seven whose selection rests
on attestation that is analogical rather than local, and its coefficient lands within two points
of South Korea's own self-identified Christian share (2015 census, 27.6%) — which may be
convergence or may be that figure carried across the border. If it is the latter the split is
wrong too, since a third of South Korea's Christians are Catholic and this row is Protestant-only.
`taxonomy/cn2000.py`'s REVIEW keeps the whole pre-decision argument.

**The Lisu row is the one to distrust**, and it is 560,283 people, two thirds of the Christian
layer: JP says 80% where the figure usually cited as official — 300,000 Christian Lisu in Yunnan —
is 43%, and the churches claim essentially 100%. The map takes the high end and says so in
`note_public`.

**And ~590,000 Christians are knowingly left in the grey.** The A-Hmao and Gha-Mu of northwestern
Guizhou are at JP 80% and are officially `Miao`, 6% of a 9.4-million nationality whose centroid is
427 km away. A source that places the A-Hmao by county is the highest-value missing input for this
country.

### What the refusal of CFPS cost, for the record

Nothing that a survey could have supplied. §14.13 has the argument: Pew's *Measuring Religion in
China* puts Buddhism at 4% by self-identification and 33% by belief, from CGSS and CFPS in the same
year, so the layer CFPS was blocking was ~4% of the grey and its size was a choice of question
rather than a measurement. **CGSS 2021 was obtained and checked** — openly mirrored on figshare, CC
BY, 8 MB, carrying `provinces` and the religion question — and rejected: 19 provinces of 31, and a
per-province Buddhist cell running from 1 respondent to 55.
