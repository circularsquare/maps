# cn_clds — China Labor-force Dynamics Survey 2016, self-identified religion by province

Obtained 2026-09-08, by Anita, from Science Data Bank after registering a free account — the
route §14.16 identified and could not walk itself. `sources/cn_clds.py` rebuilds
`data/normalized/cn_clds.csv` from `data/raw/cn/clds/`; `data/` is gitignored, so this file is
the record.

**Nothing from this survey is drawn, and that is settled** — spec §14.20, Anita, 2026-09-08.
This source exists to **check** the CGSS layer, not to feed it, and it is not mentioned in
`note_public` either, which keeps the published map free of a source whose licence bans
commercial use.

**Do not read that as the two surveys agreeing.** They agree on the ordering of the provinces
and differ on the levels by **+34% on Buddhism and +56% on Protestantism**, which would be
78.1M Buddhist dots against the 58.4M drawn. The decision rests on nobody being able to say
which level is right, and on CLDS being unable to add geography in any case: its county codes
are randomised by the depositor and only 11 of its 157 prefectures reach ten Protestant
respondents, so province is its ceiling, which is where CGSS already sits.

| | |
|---|---|
| what | *中国劳动力动态调查* (China Labor-force Dynamics Survey), 2016 wave, 试用版 (trial release) |
| who | Center for Social Science Survey, Sun Yat-sen University |
| n | **21,086 individuals, 29 provinces, 402 communities, 157 prefecture-level cities** |
| religion variable | **`I7_1 宗教信仰`**, single choice, nine answers |
| basis | **`self_id`** — the same question family as CGSS, Vietnam's census and Russia's Arena |
| weight | `wpp`, summing to 965,087,488 |
| licence | **restrictive; see below. Not CC BY, whatever ScienceDB's landing page says** |

## The archive, and the one thing that made it look empty

`CLDS2016.rar`, 37 MB, twelve files. The three `.dta` are individual (230 MB, 1,515
variables), household (103 MB) and community (2.3 MB); the rest are the questionnaires, the
codebook and the data-use agreement as PDFs.

**A grep for 宗教 over the variable labels returns nothing, and the file has a religion
question anyway.** The `.dta` carries GB18030 label bytes and pandas hands them back decoded
as latin-1, so every Chinese label is mojibake until it is round-tripped
(`s.encode("latin-1").decode("gb18030")`). The first pass here reported *0 variables mention a
religion word* on a file with four of them. `cn_clds.py::fix` is that round-trip and the
docstring says why it exists.

The four are `I7_1 宗教信仰`, `I7_2 宗教活动次数` (frequency of religious activity),
`I6_7_3_9 宗教组织` (membership of a religious organisation) and one attitude item. Only
`I7_1` is used.

### The answer set, which differs from CGSS in both directions

`I7_1a`: 天主教 / 基督教 / 佛教 / **藏传佛教** / 道教 / 伊斯兰教 / 东正教 / 其他宗教 /
无宗教信仰.

**It separates Tibetan Buddhism from Buddhism, which CGSS does not** — 61 respondents, 0.86%
weighted, against the ethnic derivation's 6.35M over 1.33bn = 0.48%. Same order of magnitude,
and the first independent number anybody has put beside §14.5's Vajrayana row.

**It offers no folk-religion option, which CGSS does.** That is a §3.1a answer-set difference
and it was tested rather than assumed — see below, because the obvious conclusion from it is
wrong.

| | n | weighted % |
|---|---|---|
| 无宗教信仰 none | 18,270 | 87.80 |
| 佛教 Buddhism | 1,348 | 6.52 |
| 伊斯兰教 Islam | 821 | 1.93 |
| 基督教 Protestant | 417 | 2.10 |
| 藏传佛教 Tibetan Buddhism | 61 | 0.86 |
| 道教 Daoism | 53 | 0.28 |
| 天主教 Catholic | 49 | 0.28 |
| 其他宗教 other | 50 | 0.23 |
| 东正教 Orthodox | 0 | 0.00 |
| **any religion** | **2,799** | **12.20** |

## What it is worth, in three findings

### 1. The national level lands on CGSS's trend line, from a different survey

CGSS reads any-religion at 14.47% (2012), 10.61% (2017), 7.50% (2021), and §14.16 could not
separate a real decline from the multi-select-to-single-choice instrument change at 2021.
**CLDS 2016 is single-choice, a different house, a different sample, and returns 12.20%** —
between the 2012 and 2017 readings, in the right year, on the right slope. A fourth point from
a fourth instrument agreeing with the 2012→2017 segment says that segment is mostly real.

### 2. The Protestant layer is replicated across surveys, and that is what it was chased for

§14.16 drew Protestantism flagged, on a CGSS 2012↔2021 rank correlation of **+0.17** its own
module calls *"a failure to demonstrate signal"*. Across surveys the picture is different:

| | Spearman, CLDS 2016 vs CGSS pooled, 29 provinces |
|---|---|
| `buddhism.mahayana` | **+0.596** |
| `christianity.protestant` | **+0.595** |

**CLDS finds Henan first at 10.9%, with 97 Protestant respondents — the largest Protestant
cell in either survey** — then Zhejiang 7.5% and Jiangsu 7.2%. CGSS pooled has Henan first at
4.6%. Two surveys sharing no fieldwork agreeing on the ordering is the evidence §14.10's fifth
condition was asking for; CGSS's wave-to-wave wobble measures the temporal instability of
*reporting*, which is a different thing from the absence of a geography.

Caveat kept: 19 of 29 provinces have fewer than 10 Protestant respondents in CLDS too. The
agreement is on the ordering of the provinces that do have them.

### 3. Buddhism agrees on shape and disagrees on level, and it is NOT folk religion

Zhejiang and Fujian are first and second in both surveys, but CLDS puts them at 36.1% and
32.1% against CGSS's 14.8% and 11.5%. Population-weighted national Buddhism is 5.93% (CLDS)
against 4.44% (CGSS pooled).

**The obvious explanation is that CLDS offers no folk option so Mazu and Guandi worshippers
pick 佛教. It is wrong, and it was tested three ways:**

- the gap correlates with CGSS's provincial folk share at only **+0.10**;
- adding folk to CGSS *lowers* agreement with CLDS, Pearson +0.662 → +0.492;
- **Guangdong has the highest folk share in CGSS at 22.5% and a gap of −0.55 points; Zhejiang
  has almost the lowest at 0.24% and the largest gap in the country at +21.4.**

Nor is it the universe. CLDS under-represents the over-65s, who are usually assumed the most
religious cohort — and in CLDS they are not: any-religion runs 12.1 / 13.0 / 12.0 / 11.7%
across 15-29, 30-44, 45-59 and 60+. **Restricting CLDS to CGSS's 18+ universe moves Buddhism
from 6.50% to 6.51%.** The difference is between the instruments, not their populations.

Nor is it a handful of communities: Zhejiang's 36% is spread across all seventeen of them, at
74, 65, 62, 59, 52, 49, 41, 38, 35, 23, 22, 16, 16, 9, 8, 3 and 3 per cent.

**So the level difference is unexplained and is the reason this cannot simply be averaged into
the existing layer.** It is exactly §3.4's shape-versus-level split, arriving as a choice
rather than a vintage.

## THE COMMUNITY COUNT IS THE THING TO CONDITION ON, AND THIS SURVEY PROVES IT BOTH WAYS

§14.16 found CGSS's provincial cut of Islam was a lottery and used the census margin to show
it. CLDS fails the same test in the *opposite direction*, which turns a suspicion about one
survey into a rule about survey design:

| | census, Muslim nationalities | CGSS pooled | CLDS 2016 | CLDS communities |
|---|---|---|---|---|
| Xinjiang | 58.3% | 92.0% | **61.3%** | 17 |
| Ningxia | 34.5% | 90.4% | **1.1%** | **4** |
| Qinghai | 16.9% | 1.1% | 20.5% | **4** |
| Gansu | 7.2% | 0.6% | 8.3% | 16 |
| Yunnan | 1.5% | 10.4% | 0.2% | 12 |

**Ningxia's four communities returned four Muslims between them** — one at 5%, three at 0% —
in a province a third of whose people are Hui. Those same four communities produce its
**18.4% Buddhist** figure, which is the largest CLDS-CGSS disagreement anywhere in China and
is pure sampling accident. It is third on the drawn Buddhism list and it is worthless.

**Where the community count is high, the survey reproduces the census margin almost exactly:**
Xinjiang 1.05×, Beijing 1.02×, Gansu 1.15×, its seventeen Xinjiang communities running from
97% Muslim down to 3%, which is the real north-south structure of the province.

`cn_clds.py::check` prints this table on every run and names the thin provinces:
**Chongqing, Inner Mongolia, Jilin, Ningxia, Qinghai and Tianjin**, all under eight
communities. `MIN_PSU` is where a threshold would go if one is ever wanted; nothing is
filtered on it yet.

## THE BUILDINGS AGREE WITH THE ANSWERS, AND THIS IS THE BEST EVIDENCE IN THE COUNTRY

The community questionnaire records, for each of the 402 sampled communities, whether the
interviewer found a **church (C68), a temple (C69), a mosque (C70), a Daoist temple (C71) and
an ancestral hall (C67)**, with a count and the date of the earliest one.

§14.15 ruled out the registered-venue registry as a **magnitude** source and was right: §2.6 is
the Thailand mistake, and China's registry omits house churches, which is most of what is
there. **This is a different object.** It is an interviewer's observation of a place, in the
same community whose residents answered the religion question, and it is used here as a
**coherence** check rather than a count of anybody:

| the interviewer found | communities | share of the matching self-report | |
|---|---|---|---|
| a church | 37 of 398 | **6.46%** Protestant vs 1.53% | **4.2×** |
| a temple | 106 | 11.97% Buddhist vs 4.31% | 2.8× |
| a mosque | 16 | **51.73%** Muslim vs 0.88% | **59×** |
| an ancestral hall | 69 | 12.96% Buddhist vs 4.97% | 2.6× |

**The Protestant row is the one that matters**, because Protestantism is the weakest thing
drawn in this country. People who say they are Protestant live, four times over, in the
communities that have a church in them — and that is not a survey agreeing with a survey, it
is a survey agreeing with a building. Nothing else on the Chinese map has a check of this kind.

The mosque row at 59× is the ethnic derivation seen from the other side, and it is the same
spatial clustering that makes a four-community province worthless.

Two cautions. Only 37 communities have a church at all, and **within those, the NUMBER of
churches predicts nothing** (r = +0.017), so it is presence and not intensity that carries the
signal. And the ancestral-hall row at 2.6× is a reminder that 佛教 in a Chinese answer set is
not cleanly separable from lineage and folk practice, which is §14.16's whole caveat arriving
from a new direction.

**The yes/no coding is `1 = 没有`, `2 = 有`, the reverse of the obvious guess.** Taking the
guess turns all four ratios into their reciprocals and reads as a perfect *anti*-correlation
between religion and religious buildings, which is how this check first came out here. The
Stata label-SET names in this file collide with unrelated variable names (the set called `C68`
is about migration reasons), so the coding has to be read off the converted categoricals.
`cn_clds.py::venues` does that and reproduces the table above.

## LICENCE — AND THIS IS THE PART THAT CONSTRAINS THE PROJECT, NOT JUST THE FILE

ScienceDB's landing page advertises **CC BY 4.0 and `conditionsOfAccess: PUBLIC`**. **The
agreement shipped inside the archive says otherwise**, and §6b's rule — the badge is the
depositor's claim, the terms are the origin's — applies with more force here than anywhere
else on this map. `2016年中国劳动力动态调查数据使用协议.pdf`, clause 2:

- **(2) the data is for the applicant's own use, may not be given to any third party, and the
  raw data may not be published or released in any form.** Aggregates derived from it are
  analysis rather than data, which is the ordinary reading and the one this project relies on.
- **(3) it may not be used for any commercial (profit-making) or political purpose.**
- (4) no alteration or distorting use, and no fabricated results under the CLDS name.
- (6) a required citation, verbatim: *"Data used in this paper is from the China Labor-force
  Dynamics Survey (CLDS) by the Center for Social Science Survey at Sun Yat-sen University in
  Guangzhou, China. The opinions are the author's alone. Please refer to http://css.sysu.edu.cn
  for more information about the CLDS data"*.
- (7) an obligation to register any resulting publication with `css.sysu.edu.cn/Thesis/Upload`.

**Clause (3) is a new constraint on this map and not just on this file.** It puts CLDS in the
same box as [[reference_poster_commercial_licences]]'s existing blockers: a religiondots poster
that is ever *sold* cannot carry a CLDS-derived layer. The existing CGSS layer has no such
clause. **So drawing CLDS trades a commercially clean country for a better-evidenced one**,
and that is a decision rather than a detail.

Cite **Sun Yat-sen University CSS** as the origin, never the ScienceDB depositor, who is a
third party at Hebei University.

## What was NOT obtained, so it is not re-searched

Per [[feedback_nothing_is_truly_dead]]:

- **CLDS 2012, 2014, 2018.** The ScienceDB deposit advertises all four waves plus the 2011
  Guangdong pilot; the archive Anita retrieved is **2016 only**. Whether the others are
  separate downloads behind the same account or simply absent is unchecked, and it is the
  cheapest remaining lead on this country — a second CLDS wave would give the same
  cross-survey test a time dimension.
- **The household file is not read.** 103 MB, and the individual file has no 民族 column, so
  if an ethnicity variable exists anywhere it is there. That would let the ethnic derivation
  and the self-identification be checked against each other *within one survey*, which is the
  check neither source can currently give.
- **The community file is read only for its five venue questions** (above). It has 711
  variables and the rest are unexamined; `C7_1 少数民族人口比例`, the minority share of each
  community, looked like the direct diagnostic for the lottery problem but correlates with
  self-reported Islam at only −0.02 across the 258 communities that answer it, which does not
  make sense and probably means the variable is banded or conditional rather than a
  percentage. Not chased.
- **`I7_2 宗教活动次数`** — frequency of religious activity, asked of everyone. Untouched, and
  it is the one variable here that could say something about the §14.16 gap between naming a
  religion and practising one. It is not a `self_id` answer, so §3.1 keeps it out of the map,
  but it could inform `note_public`.
