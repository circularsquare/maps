# Kyrgyzstan — drawn from the Life in Transition Survey, because nothing else asks

Built 2026-09-08. `sources/kg.py`, `sources/kg_geo.py`, `sources/kg_grid.py`,
`sources/lits.py`, `taxonomy/kg2016.py`. 7,404,329 people, 9 units, 8 drawn nodes, basis
`self_id`, every row `modelled`.

The queue sent Kyrgyzstan here for the survey. The instruction was to run Kazakhstan's route
first (§9cd: an office that has ruled out a cross-tabulation on the evidence of its
*publications* has ruled out nothing), and it was run in full. **Kyrgyzstan does not hold the
cross-tabulation, and unlike Kazakhstan the reason is upstream of the office: the question has
never been on a Kyrgyz census form.**

| | |
|---|---|
| office | National Statistical Committee of the Kyrgyz Republic, `stat.gov.kg` |
| censuses since independence | **1999, 2009, 2022** |
| religion question | **none, on any of them.** Form 2 of the 2022 census runs questions 1 to 17 and is printed in full in Book I |
| published table catalogue | **806 downloads across 22 subject pages, no religion table**; and 546 of 1,400 swept `dynamic` ids return a named file, none of them religion |
| national open data portal | `data.gov.kg`, CKAN, **1,410 packages enumerated in full**; religion appears only as *religious organisations*, which are buildings |
| dashboards | none. No iframe on any `stat.gov.kg` page, and no Qlik, Power BI, Tableau, Superset, Metabase or ArcGIS anywhere on the domain |
| licence | **CC BY-NC-SA 4.0**, linked from the footer of every page |
| what is drawn instead | **EBRD Life in Transition Survey III**, 2015-16, n=1,500, 75 PSUs, **9 of 9 oblasts** |

---

## 1. The census does not ask, and that is a fact about the questionnaire

`stat.gov.kg/ru/statisticheskie-perepisi/` lists two population censuses, 2009 and 2022. The
2009 page links eleven volumes: territory and population, sex and age, **nationality and
language**, education, marital status, fertility, sources of income, households and families,
housing, migration, economic activity, plus a per-oblast volume for each region. There is no
religion volume. The 2009 Book I in English (72 pages, cached from UNSD) contains the strings
`religio`, `faith` and `confession` zero times.

**The 2022 census settles it properly, because Book I prints the census forms as an annex.**
`publicationarchive/5cf96b00-ef31-4c8c-845c-ec7f38bbc670.pdf`, 101 pages, 172,600 characters,
`вероисповед` and `религи` zero times across the whole document. Its §V, *Методологические
принципы — Программа переписи*, names five forms and says Form 2 carries the demographic,
economic, educational and geographic characteristics; and then the annex prints Form 2 itself,
question by question:

```
     1 Родственные отношения           10 Владение языками (10.1-10.5)
     2 Ваш пол                         11 Миграция (11.1-11.7)
     3 Временное отсутствие            12 Образование и обучение (12.1-12.6)
     4 Дата рождения                   13 Функциональные ограничения (13.1-13.6)
     5 Свидетельство о рождении        14 Источники средств к существованию
     6 Семейное положение              15 Занятость и безработица (15.1-15.6)
     7 ВАША НАЦИОНАЛЬНОСТЬ             16 Использование современных средств связи
     8 Страна рождения                 17 Рождаемость
     9 Страна гражданства
```

Seventeen questions. **Nationality at 7, native language at 10.1, and no religion item on any
of the five forms.** `[[reference_census_questionnaire]]` — the questionnaire is the only
thing that separates "the office does not publish it" from "nobody asked".

The Cabinet of Ministers resolution that ordered the census (no. 228 of 22 October 2021,
`media/files/c0dae0f6-...docx`) contains `религи` exactly once, and it is an instruction to the
**State Commission for Religious Affairs** to help enumerate people in its institutions.

## 2. The office's own catalogue was enumerated rather than searched

§11aj's lesson from Uzbekistan: *a country closed on an enumerated database stays closed until
the database changes; a country closed on documents was never closed at all.* Kyrgyzstan's
office has no SDMX endpoint and no indicator API, but it does have something that enumerates
just as well.

Each of the twenty-two subject pages under `/ru/statistics/<topic>/` carries its whole list of
downloads as server-rendered HTML, in three routes: `download/dynamic/<n>/` (the time-series
tables, numbered in a national classification like `5.01.00.15`), `download/operational/<n>/`
and `download/methodology/<n>/`. **Harvesting all twenty-two gives 806 entries with their
titles**, and grepping the lot for `вероисповед|религи|конфесси|мусульман|ислам|правосл|
христиан|буддис|атеис` returns **zero**. Ethnicity returns two, and both are the same table:
`5.01.00.15 Внешняя миграция населения по этническому составу`, external migration by ethnic
composition. There is no nationality-by-oblast **stock** table in the catalogue at all.

**And the unlinked files were checked too.** `[[reference_cms_download_id_sweep]]`: the
`dynamic` route takes a bare integer, so ids 1 to 1,400 were swept and the filename read out of
`Content-Disposition`. **546 return a named file** and not one filename names religion or
ethnicity. So the catalogue is not merely the linked subset.

`data.gov.kg`, the national open data portal, is a CKAN instance; `package_search` paged out
**all 1,410 packages**. Four match religion and all four are the State Commission's quarterly
register of **religious organisations and religious buildings**. That is a count of mosques and
churches, not of people, and `[[feedback_proxy_residual_nameable]]` rules it out as a proxy: the
non-matching part is not a published number anyone can weight by.

## 3. There is no BI engine here, and one subdomain is firewalled

Fetched as **raw HTML** rather than through WebFetch, which strips iframes (§9cd): `stat.gov.kg`
in all three languages, `/ru/opendata/`, `/ru/bazy-dannyh/`, `/ru/statistics/naselenie/`, the
census pages and the publications archive. **Not one `<iframe>` on any of them**, and no Qlik,
Power BI, Tableau, Superset, Metabase or ArcGIS token anywhere. The site is a server-rendered
Django application whose only JavaScript is jQuery, Bootstrap and Chart.js.

Subdomain probe: `www`, `forum` and `reg` answer. `reg.stat.gov.kg` is a Drupal 10 *personnel
registration* system returning 403. **`census.stat.gov.kg` resolves to 212.42.101.110 and
times out on every port tried** — 80, 443, 3000, 5000, 8000, 8080, 8083, 8443, 9000, 22, 1433,
5432, all `timeout` rather than `refused`. That is a drop rule, and it is a different signature
from Uzbekistan's §11aj geo-fence, which *accepted* every SYN and reset the TLS handshake.
Nothing is known to be behind it.

> **The site search is broken and is not evidence.** `/ru/search/?q=` returns a page of
> identical size for `население` and for `zzzqqq`; the only difference between the two
> responses is the echoed query string. A negative from it means nothing, which is why the
> catalogue was harvested instead.

## 4. So the survey, and what it can and cannot carry

EBRD Life in Transition Survey III, fielded late 2015 into early 2016. Open download, no auth
and no terms gate (§11ag), already on disk as `data/raw/lits/lits_iii.dta`.

**Kyrgyzstan is the one Central Asian country in this file that clears coverage.** 75 PSUs
across **all nine** oblasts and republican cities, n=1,500, against Uzbekistan's ten of
fourteen (§11aj) and Tajikistan's five of five with 99.5% of respondents in one category.

```
    89.23%  MUSLIM                                  n=1331
     6.76%  ORTHODOX CHRISTIAN                      n= 107
     1.94%  ATHEISTIC / AGNOSTIC / NONE             n=  24
     0.91%  BUDDHIST                                n=  16
     0.78%  OTHER CHRISTIAN, INCLUDING PROTESTANT   n=  14
     0.19%  OTHER                                   n=   4
     0.14%  JEWISH                                  n=   2
     0.03%  CATHOLIC                                n=   1
     0.02%  Refusal                                 n=   1
```

**The card has eight substantive codes and not six.** §11ag read the printed questionnaire and
recorded six; the delivered `q922` value labels carry `BUDDHIST` and `CATHOLIC` as well, and
both are used here. Corrected in `sources/lits.py`.

### 4.1 The split-half, and why the usual bar is the wrong instrument at nine units

`sources/lapop.py` splits waves in half and compares each category's ordering across them
against `1.96/sqrt(n-1)`. Neither half transfers. LiTS III is one wave, so the split is on
**PSUs** — two respondents in the same PSU are the same twenty-household cluster — and one
random split of 75 PSUs is noisy enough to change the answer run to run, so the statistic is
the **median over 400 random PSU halves**.

And that statistic cannot be compared with `1.96/sqrt(n-1)`, which is the standard error of
**one** correlation. So the null was built: the PSU-to-region labels are shuffled and the whole
median-of-400 recomputed, 400 times.

```
    category                                 national   median rho   null 95th        p
    MUSLIM                                     89.23%       +0.548      +0.450    0.020   drawn
    ORTHODOX CHRISTIAN                          6.76%       +0.752      +0.468    0.008   drawn
    ATHEISTIC / AGNOSTIC / NONE                 1.94%       +0.185      +0.524    0.342
    BUDDHIST                                    0.91%       -0.151      +0.460    0.758
    OTHER CHRISTIAN, INCLUDING PROTESTANT       0.78%       +0.394      +0.550    0.170
    OTHER                                       0.19%       +0.540      +0.750    0.344
    JEWISH                                      0.14%       -0.125      +1.000    0.973
```

**The fixed bar at nine units is +0.693, and Islam would have failed it.** The null's own 95th
percentile is +0.450, so the bar was rejecting a category the data distinguishes from chance at
p=0.020. That is not a licence to move bars: it is `[[reference_check_needs_power]]` in the
other direction, and the fix is to measure the null rather than to assume it.

**There is no single new bar. Seven categories get seven nulls, and two of them are stricter
than the +0.693 they replaced.** Read the `null 95th` column above rather than the +0.450:
`OTHER` sits at +0.750 and `JEWISH` at +1.000, both above the fixed bar, while the five denser
categories run +0.450 to +0.550. That is the whole answer to "is this just weaker", and it is
stronger than an argument, because a permutation null is looser exactly where a category is
dense enough for a median-of-400 to be stable and stricter exactly where it is not, which is
what a calibrated null does and what a moved bar cannot do. `OTHER` fails at p=0.34 *because
its own null sits at +0.750*, not because +0.450 was applied to it and missed. §9.2 sets the
seven side by side.

**Two more things make that checkable rather than assertable, and both are on disk.** The
significance level was never chosen here: `alpha` is `lits.stability`'s inherited 0.05 default,
the same 95% that `lapop.py`'s `1.96` already encodes, so what changed is the instrument and
not the level. And the instrument is the size it claims to be: 200 datasets with the geography
destroyed by that same shuffle, each then run through `lits.stability` as if it were the
observed data with a null of its own, give 1,400 p-values rejecting at **0.046** against a
nominal 0.05, and the two categories the country turns on come out at 0.040 and 0.055. §9.1 has
the per-category table and the method. That calibration was a one-off run on the second read
rather than something the build prints, so read it as a measurement made on a particular day
and not as a check that will fail if the code changes.

### 4.2 The Buddhist cell is an artefact, and the test found it

Sixteen respondents answer `BUDDHIST`. **Six are in Osh oblast and four in Batken** — the two
most rural and most uniformly Muslim oblasts in the country — while Bishkek, which has whatever
Buddhist community Kyrgyzstan has, returns one. There is no reading of Kyrgyzstan in which that
is a geography. Its split-half median is the worst of any category here at -0.151, with 68% of
halves negative, so it is spread at the national rate and `taxonomy/kg2016.py` records the
national figure as a ceiling. **This is what a keying error looks like from the inside of a
thin survey**, and it is the argument for running the stability test on every category rather
than on the ones that look wrong.

### 4.3 What the survey cannot see at all

Kyrgyz shamanic and ancestor practice — the *bübü* and *bakshy* healers, spring and grave
veneration, the *mazar* pilgrimage sites — has no answer on this card, and the people who take
part in it answer Muslim, which is also how most of them would describe themselves. Nothing
here counts it and `other.kg`, at four respondents, is not where it is hiding. Same shape as
`other.do` for Dominican Vodú and `other.st` for São Toméan djambi.

## 5. The population is the office's own and not COD-PS

COD-PS for Kyrgyzstan is dated **2018** and totals 6,140,200. The office's own resident
population is **7,404,329 at 1 January 2026**, published down to the individual village at
`/ru/statistics/download/operational/825/`. That is a 20.6% gap and it is eight years of real
growth, not a projection disagreement. Ecuador (§9bn) took the office's own over a COD error of
3.4%; this one is six times larger.

COD-PS is still read, for the age bands only, and those are used as a ratio.

**One thing to know about the licence.** Every `stat.gov.kg` page footers a link headed
*Условие лицензирования распространения данных* pointing at **CC BY-NC-SA 4.0**, so the
population figures this country is drawn on carry a NON-COMMERCIAL and share-alike
condition. Nothing about the map as it stands conflicts with that, and attribution is in
`source`. It is written down here because it would bind a printed edition, which is the
one place this project has met that question before. The EBRD survey itself has no terms
gate at all (§11ag).

**Two parser traps in that file.** Talas oblast's territory code is written `41707 000 000 00 0`
with spaces where the other eight have none, so digits are extracted rather than strings
compared. And COD-PS's own CSV is **cp1251**, not UTF-8, so a UTF-8 read raises on the first
oblast name.

## 6. Three witnesses on two different joins, and the population check is not one of them

There are two joins here and they are not the same join. The office's population table has to
reach COD's polygons, and separately the survey's own region labels have to reach them. The
first two witnesses close the first join and say nothing at all about the second, which is the
gap §9.5 found and which witness 3 now closes.

- **Witness 1, arithmetic, office to COD.** The office keys its rows on the 14-digit SOATE
  territory code and COD keys its polygons on a pcode, and `417NN000000000` is `KGNN000000000`
  for all nine. No names involved.
- **Witness 2, names, office to COD.** COD's `adm1_name1` and the office's own Russian name
  agree for all nine, independently of the codes.
- **Witness 3, the abbreviation, LiTS to COD.** LiTS's nine `region_name` strings are
  abbreviated (`Д-АБАДСКАЯ`, `И-КУЛЬСКАЯ`) and the two cities carry `горкенеш`, the Kyrgyz for
  city council, so the decode is written out in `kg_geo.LITS_REGION` rather than matched. What
  closes it is `kg_geo.lits_decode_witness`: drop the noise words, split both the LiTS label
  and COD's own Russian name on `-`, `.` and space, and require every LiTS token to equal or to
  begin the name's token in the same position, `И` for `Иссык` and `КУЛЬСКАЯ` for `Кульская`.
  Each of the nine labels then abbreviates **exactly one** of the nine names, so of all 362,880
  pairings of labels to names exactly one satisfies the rule and it is the one written down.
  Nothing in it is a correlation or a population figure.

  The old form of this witness was `set(LITS_REGION.values()) != set(g["pcode"])`, which is a
  coverage test that every permutation of the dict passes identically. §9.5 has the
  demonstration that it had to be replaced, and the swap it constructed is what the new
  assertion was tested against: `И-КУЛЬСКАЯ` written against Batken fails on the token count
  before it reaches the letters.

### THE POPULATION PERMUTATION CANNOT SEPARATE ISSYK-KUL FROM BATKEN, AND SAYS SO

The held-out check correlates the survey's weighted share of respondents per oblast against the
office's population share, and ranks it against **all 362,879 other orderings** of the nine
units. Observed r = **+0.9866**. Exactly one ordering beats it, at +0.9870: **the one that
swaps Issyk-Kul and Batken.**

Those two are 7.49% and 8.17% of Kyrgyzstan. **The quantity being correlated is each unit's
share of the survey's 1,500 respondents**, a multinomial proportion over the whole sample, so
the error on it is computed on n=1,500 and not on either oblast's own interviews: near 7.5%
that is **±1.33 percentage points** at 95%, which is the figure the run prints. The two units
are **0.69pp apart** on it, half the criterion. No correlation computed on that quantity can
tell them apart, and calling the swap a catch would be reporting sampling noise as a finding.

Issyk-Kul and Batken are n=100 and n=140 in this survey, and an error computed on 120
interviews instead would be about ±4.9pp. That is not the number to use here, and §9.4 has why
the whole-sample denominator is the right one and why it is the conservative one: clustering
and weighting both inflate the true error, so a narrower `se` forgives less than a
design-correct one would.

So `lits.held_out` forgives a beating ordering **only** when every unit it moves goes to a unit
within 1.96 standard errors of its own population share, prints which pairs those are on every
run, and stops the build on anything else. That is a computed criterion rather than a widened
bar. It is also not a substitute for a witness, and the witness it needs is **witness 3**:
witnesses 1 and 2 separate Issyk-Kul from Batken for the office-to-COD join and are silent on
the LiTS decode, which is the join this check is testing. §9.5 is that finding and the third
bullet above is the fix.

## 7. §3.5, and the hole that matters is the age cut

The refusal cell is **one respondent**, in Chui oblast. With a single non-zero unit there is no
correlation to compute and the lean check says so rather than producing a number, exactly as in
Uzbekistan.

**The real exclusion is children, and it is large.** LiTS III interviews adults, and Kyrgyzstan
is one of the youngest countries in the former Soviet Union: COD-PS 2018 puts the average
oblast at about a third under 18, and the share is not flat across the nine. The adult
composition is applied to the whole resident population, which is what every survey-drawn
country here does, and `countries.py` says so in `gap`.

## 8. What would improve this country

- **A nationality-by-oblast table from the 2022 census.** It was collected (question 7) and it
  is not in the office's catalogue. With it, Kyrgyzstan becomes a §9aq ethnicity model that
  could be scored against this survey, the way §9cd scored Kazakhstan's — and the two together
  would be much stronger than either. The regional volumes (Book III, one per oblast, December
  2023 to January 2024) are the place to look first and were not opened here.
- **A second survey to pool with.** LiTS IV has the country and not the question (§11aj). The
  Life in Kyrgyzstan panel study (DIW Berlin, 2010-2019, ~3,000 households, oblast
  representative) was not tested and sits behind a registration form.
- **`census.stat.gov.kg`**, if it answers from anywhere else.

---

## 9. Second read, 2026-09-08 — the permutation null checked from outside the builder's account

A review pass. Nothing here needs rebuilding and no ask was filed. `check_md.py` clean,
`built_countries.py --check` names nothing, `check_rollup.py kg` is 7,404,329 all `modelled`
with no derived rows and no orphans. `sources/kg.py` re-run and `data/normalized/kg.csv` came
back **byte-identical**, so the country is reproducible from the file on disk.

### 9.1 The shuffle destroys the geography and nothing else, and the test is correctly sized

`lits.stability` permutes `assign`, the PSU-to-oblast index array. That preserves three things
and destroys one:

- each PSU's own category weights, so the twenty-household cluster stays intact;
- the **number** of PSUs each oblast receives, so unit sizes survive (Osh keeps 16, Talas 3);
- the same 400 splits, reused for the observed statistic and for every null draw, so the null
  is conditioned on the identical split structure rather than resampling it.

What it destroys is only the PSU-to-oblast association. Under the null the observed assignment
is one arrangement among equally likely ones, so the test is exact by exchangeability.

**Measured rather than argued.** 200 datasets with the geography destroyed by that same
shuffle, each one then run through `lits.stability` as if it were the observed data, with its
own 200-draw null:

```
    category                                 reject at .05     median p
    MUSLIM                                       0.040           0.495
    ORTHODOX CHRISTIAN                           0.055           0.575
    ATHEISTIC / AGNOSTIC / NONE                  0.085           0.493
    BUDDHIST                                     0.055           0.557
    OTHER CHRISTIAN, INCLUDING PROTESTANT        0.055           0.527
    OTHER                                        0.035           0.975
    JEWISH                                       0.000           0.930
    ---------------------------------------------------------------------
    1,400 p-values     0.0464 at .05     0.0864 at .10     0.1829 at .20
```

Nominal within a standard error at all three levels, and the two categories the country turns
on are 0.040 and 0.055. **So p=0.020 for Islam and p=0.008 for Orthodoxy are the sizes they
claim to be.** The one category above nominal is the merged irreligion box at 0.085, about 2.3
standard errors, and it fails on the real data anyway at p=0.34, so nothing drawn depends on it.

### 9.2 The `OTHER` demonstration holds, and the stronger form of it is not written down

Each category's own null 95th against the fixed `1.96/sqrt(n-1)` of +0.693:

```
    MUSLIM                       +0.450   ORTHODOX CHRISTIAN            +0.468
    BUDDHIST                     +0.460   ATHEISTIC / AGNOSTIC / NONE   +0.524
    OTHER CHRISTIAN              +0.550   OTHER                         +0.750   STRICTER
                                          JEWISH                        +1.000   STRICTER
```

**Two of the seven testable categories are held to a HARDER bar than the one they replaced.**
That is the answer to "is this just weaker": the permutation is looser exactly where the
category is dense enough for a median-of-400 to be stable and stricter exactly where it is not,
which is what a calibrated null does and what a moved bar cannot do. `OTHER` fails at p=0.34
*because its own null sits at +0.750*, not because +0.450 was applied to it and missed.
`countries.py`'s note and `runlog.md` both say "the new bar", singular; there are seven, one per
category, and the seven-ness is the whole argument.

### 9.3 Provenance: the record does not claim what it cannot show, and two things would settle it

Nothing on disk dates the choice. `sources/lits.py` is untracked and `~/.claude/file-history`
holds no snapshot of it or of `kg.py`, so the order in which the null was written relative to
Islam's +0.548 is not recoverable. **The record does not claim a chronological provenance.**
§4.1 here and `kg.py`'s docstring both state plainly that the fixed bar was known to reject
Islam, and then argue from the statistic rather than from the outcome, which is the right shape.

Two things make that argument checkable and neither is currently written down:

- **The level was inherited and only the instrument changed.** `alpha` is `stability`'s default
  0.05, the same 95% that `lapop.py`'s 1.96 already encodes. Nobody picked a significance level
  here; the question was only what statistic that 95% is measured against.
- **9.1 and 9.2.** A calibration at nominal size, and two categories held to a stricter bar, are
  what turn "this is a re-calibration, not a loosening" from an assertion into a demonstration.

**Both are now written in, 2026-09-08.** §4.1 carries them, and `countries.py`'s `note` carries
a compressed form: the seven nulls with `OTHER` and `JEWISH` above the fixed bar, the inherited
`alpha`, and the 1,400 p-values at 0.046. The judgement was that this record's problem is the
opposite of Panama's — Panama asserted a provenance nothing on disk could show, while these two
are a default argument value anyone can read off `lits.stability`'s signature and a measurement
whose method §9.1 states in full. Both are labelled for what they are: the calibration is
recorded as a one-off review run and not as a check the build performs, because nothing on disk
reproduces it and the record should not imply otherwise.

### 9.4 The forgiveness criterion is computed, and errs in the safe direction

`se = 1.96 * sqrt(b(1-b)/len(df))` uses simple random sampling on the whole 1,500. That is the
right denominator, because a unit's share of respondents is a multinomial proportion over the
entire sample, and it **understates** the true error, since 75 clusters of twenty and a weighted
share both inflate it. A narrower `se` forgives less, so the criterion is stricter than a
design-correct one would be. The gap is 0.68pp against 1.33pp, half the criterion; at a plain
one-sigma criterion it would be 0.68 against 0.68, exactly marginal. **A tuned z would have been
about 1.0 rather than 1.96**, so nothing here looks fitted to the answer.

**But §6's prose attaches the number to the wrong sample.** "the survey measures each on about
120 respondents, where the 95% error on a share near 8% is ±1.37 percentage points" — the ±1.37
is n=1,500; on n=120 it is ±4.9pp. The figure is right and the sentence points at the wrong n,
which is the one thing a reader checking the criterion would recompute. The same sentence shape
is in `lits.py`'s docstring. And §6's ±1.37 and 0.68pp are a generic p=0.08 rounding: the run
prints **±1.33pp and 0.69pp**, which is what `countries.py` and `sources.md` §9co carry.

**Fixed, 2026-09-08.** §6 now names n=1,500 as the denominator and why, quotes ±1.33pp and
0.69pp, and says separately that the two oblasts' own samples are n=100 and n=140 and that an
error computed on about 120 interviews would be roughly ±4.9pp. `lits.held_out`'s docstring is
corrected the same way.

### 9.5 THE TWO WITNESSES NAMED DO NOT TOUCH THE JOIN THE HELD-OUT CHECK IS TESTING

The held-out check runs on the LiTS `region_name` to pcode decode, which is `kg_geo.LITS_REGION`,
a hand-written dict. Witness 1 (SOATE `417NN` = COD `KGNN`) and witness 2 (the Russian names)
both pin **the office to COD**. Neither says anything about which pcode `И-КУЛЬСКАЯ` goes to.
Witness 3 as written is `set(LITS_REGION.values()) != set(g["pcode"])`, a coverage test that any
permutation of the dict passes identically.

Demonstrated rather than asserted. Swap `И-КУЛЬСКАЯ` and `БАТКЕНСКАЯ` in `LITS_REGION` and
re-run:

```
    TRUE join    r = +0.9866   1 of 362,879 orderings reaches it   -> forgiven, passes
    SWAPPED      r = +0.9870   0 of 362,879 orderings reach it     -> passes outright
```

**The wrong join passes more cleanly than the right one**, and it needs no forgiveness to do it.
So no automated check in this country separates the two units, and §6's *"witnesses 1 and 2 both
separate Issyk-Kul from Batken outright"* is true of a different join from the one at issue.

The decode is correct — `И-КУЛЬСКАЯ` cannot abbreviate `Баткенская` — but the whole evidence for
it is a human reading a hand-written table, which is `[[reference_name_join_wrong_neighbour]]`'s
exact shape. It is one assertion to close, against the `name_ru` already sitting in
`kg_lookup.csv`: strip `горкенеш`, split each LiTS string on `-` and `.`, and require every token
to equal or to prefix the token in the same position of the office's own Russian name
(`И` prefixes `Иссык`, `КУЛЬСКАЯ` equals `Кульская`; `БАТКЕНСКАЯ` equals `Баткенская`). Not
written here because `kg_geo.py` rebuilds `kg_lookup.csv` and `kg_oblasts.gpkg`, and that is not
a reviewer's edit to make unasked.

**Written, 2026-09-08, as `kg_geo.lits_decode_witness`, and §6's witness 3 is now it.** Built to
the rule above, with one strengthening: the match is required to be **unique** across all nine
names, so the assertion says not merely that the written pairing is consistent but that no other
pairing of the nine is. Exactly 1 of the 362,880 pairings satisfies it, and it is the one in
`LITS_REGION`.

Tested against this section's own construction rather than assumed to work, because an assertion
that cannot fail is worse than none:

```
    TRUE join     each of the 9 labels abbreviates exactly one name  -> PASSES
    SWAPPED       'БАТКЕНСКАЯ' written against KG02 (Иссык-Кульская) but abbreviates KG05
                  'И-КУЛЬСКАЯ' written against KG05 (Баткенская)     but abbreviates KG02
                                                                     -> FAILS, stops the build
```

The swap fails on the token count before it reaches the letters: `и-кульская` is two tokens and
`баткенская` is one. `kg_geo.py` re-run afterwards, `kg_lookup.csv` and `kg_codps_ages.csv` both
byte-identical; `kg_oblasts.gpkg` differs only in the `gpkg_contents` timestamp, its nine rows
and their WKB identical to the byte. `sources/kg.py` re-run, `data/normalized/kg.csv`
byte-identical at md5 `b896b370239784b91967ba7fcaefab3d`.

### 9.6 The age lean reproduces, and `note_public` quotes the half that survives

```
    under-18 share vs the drawn MUSLIM              r = +0.724   leave-one-out +0.372 to +0.893
    under-18 share vs the drawn ORTHODOX CHRISTIAN  r = -0.846   leave-one-out -0.960 to -0.740
```

`note_public` gives -0.85 with its full leave-one-out range and does not quote Islam's +0.72,
which is the right choice: Islam's halves when one oblast is dropped and Orthodoxy's never
leaves a narrow band. §7 here reports neither number, only the 31.3% to 41.0% spread.

One small step in the argument is not measured. The -0.85 is a **between**-oblast correlation,
and the bias it is used to establish, that an adult composition applied to children draws the
country more Orthodox than it is, is a **within**-oblast fertility differential. The between is
good evidence for the within, and is almost certainly right in Kyrgyzstan, but `note_public`'s
"therefore" is doing a step of ecological inference that nothing here measures. Worth one
softening word if that note is edited for another reason; not worth editing it for.

### 9.7 The Buddhist cell: the test really did find it unprompted

`stability` runs over every category before anything is drawn, and `CARRIES` is asserted against
its output afterwards, so nothing selected Buddhism for inspection first. It reproduces: 16
respondents, 6 in Osh oblast, 4 in Batken, 1 in Bishkek, median -0.151, the worst of the eight
and the only negative besides Judaism's -0.125. It is drawn at the national rate, the ceiling is
stated in three reader-facing places, and the drawn 66,738 matches `note_public`. Honest.

### 9.8 Reader-facing figures, recomputed from `data/normalized/kg.csv`

All reproduce: 7,404,329 total; Orthodoxy 509,117 nationally at 6.876% and 331,243 in Bishkek,
which is 65.1% of them and 24.4% of the city; Chui 10.14%; Islam and Orthodoxy 96.071% together;
Talas's tail 8.84% against Bishkek's 7.92% and Osh oblast's 2.30%; Buddhism 66,738; the other
cell 0.183%; Talas n=60; Osh and Batken n=320 and n=140, 460 with no Orthodox answer; grain
7,404,329 / 9 = 822,703. **Superlatives are by people and the pairing is fair**: Bishkek is both
the most Orthodox people and the highest Orthodox share. The six tail categories all come out
"most people Bishkek, highest share Talas" by construction of the residual, and `note_public`
names none of them as a superlative, which is right.

Three rounding notes, none of them wrong enough to change:

- "**4.0%** nationally" for the tail is the survey's `small_total`; the drawn national tail is
  3.93%.
- "zero out of that many still leaves room for about one percent" is generous: the exact
  one-sided 95% bound on 0 of 460 is **0.65%**. Clustering across 23 PSUs would widen it, so the
  direction of the slack is the honest one.
- §4's table above is the **with-refusal** share (89.23% Islam, Refusal 0.02%) and §4.1's table
  immediately below it is the **without-refusal** one the code prints (89.252%). The two tables
  are stitched from different runs of the same thing.

### 9.9 The licence, read at the source

`https://stat.gov.kg/ru/` and `/en/` footer a link *Условие лицензирования распространения
данных* / *License condition for data dissemination* to
`/{ru,en}/ATTRIBUTION-NONCOMMERCIAL-SHAREALIKE-4-INTERNATIONAL/`, whose text is *"All materials
posted on the site are available under the Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International license"*, with attribution, non-commercial use, share-alike and
no-additional-restrictions each spelled out. Last updated 09.06.2024. **§5 records it
accurately**, share-alike included.

**One thing §5 does not have, and a printed edition would want it.** The same footer, on every
page and independent of the CC grant, carries a statutory obligation: *"Пользователи при
использовании данных официальной статистики и соответствующих метаданных обязаны ссылаться на их
источник. (ст. 30 Закона об официальной статистике)"* — users of official statistics and the
corresponding metadata must cite their source, Article 30 of the Law on Official Statistics. That
is law rather than licence, so it does not lapse if the Creative Commons page is changed or
withdrawn.

### 9.10 One stale cross-reference, left alone

`sources/lits.py`'s module docstring says *"§9cm is the Kyrgyz build"*. §9cm is Panama;
Kyrgyzstan is **§9co**, which `sources/kg.py` has right. Not corrected here because another
session was building on `lits.py` at the time.

**Corrected, 2026-09-08**, once the Tajikistan session had finished with the file.

### 9.11 The map

Screenshotted over CDP at the `view` bbox. Dots inside the border, none in the water, the
distribution sitting on the Fergana rim, the Chui valley and the Issyk-Kul shore with Naryn
almost empty, which is where Kyrgyzstan's people are. Legend totals match the CSV:
Christianity 568k, Islam 6.6m, no religion 141k, Buddhism 66k, Judaism 9k, other 13k.
