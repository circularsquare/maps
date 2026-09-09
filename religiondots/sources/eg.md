# Egypt — Arab Barometer, waves III, IV, V and VII (2013–2022)

Built 2026-09-08. **The first country on this map drawn from the Arab Barometer**, and the
first from anywhere in the Arab world. `sources.md` §11af is the assessment of the source
across the whole region; this file is Egypt's own record. Read §11af first if you are about to
do Lebanon, Iraq, Jordan or Yemen — most of the traps are the source's, not Egypt's.

## Why this country needed a ruling before it needed a build

Egypt is the one country here where **whether to draw at all, and how finely, was a decision
somebody had to take rather than a technical question.** Spec §14.4 rule 2 says no resolution
finer than the state's own publication, and it is aimed at exactly this: a persecuted minority
being mapped by outsiders at a grain its own government withholds.

* CAPMAS **collected** religion in the censuses of 1986, 1996, 2006 and 2017.
* It has **published** it once, from 1986: Christians 5.7–5.8% of the country.
* The governorate figures that exist in the academic literature exist because a researcher
  obtained them, not because CAPMAS released them.

So `ask/answered/001-eg-copts-at-governorate-level-from-the-arab-bar.md` was filed with three
options: draw at governorate, draw Egypt as one national unit with the concentration described
in prose but not drawn, or leave it closed. **Anita ruled on 2026-09-08 for governorate**, on
the ground that governorates are pretty big. She also asked whether anything finer was
possible, and it is not; that answer is recorded below so nobody re-derives it.

**Nothing in this build re-opens either question.**

## THE CEILING, WHICH IS THE SAME TIER AS THE RULING

Two independent reasons there is nothing below governorate to have:

1. **The instrument.** Arab Barometer's `Q1` is Governorate and the file carries no finer
   geography in any wave. There is no district, no markaz, no PSU identifier that resolves to
   a place.
2. **The state.** §11af read CAPMAS's own NADA catalogue record for `EGY-CAPMAS-CENSUS-2017`
   off the Wayback Machine (`censusinfo.capmas.gov.eg` is DNS-dead since about March 2026).
   The deposited data file is 7,948,075 cases and **thirteen variables**: `GOV_CODE`
   `GOV_NAME` `STATION_CODE` `STATION_NAME` `MARITAL_STATUS_CODE` `MARITAL_STATUS_NAME`
   `GENDER_CODE` `GENDER_NAME` `AGE_YEAR` `WORK_STATUS_CODE` `WORK_STATUS_NAME` `EDU_CODE`
   `EDU_NAME`. Religion is not among them, at any tier.

## The files

| file | what it does |
|---|---|
| `sources/arabbarometer.py` | the construction, the split-half, the held-out check and the wave table, shared with every Arab Barometer country after this one |
| `sources/eg.py` | the `.sav` files → `data/normalized/eg.csv`. Egypt's governorate harmoniser, its `CARRIES`, its `DROPPED` and its two census cross-checks. |
| `sources/eg_geo.py` | COD-AB ADM1 → `data/geo/eg/eg_governorates.gpkg` + `eg_lookup.csv`, with **CAPMAS's own governorate populations** joined on a four-witness pairing. |
| `sources/eg_grid.py` | Kontur 400 m hexes → `data/geo/eg/eg_hexes.gpkg`, the placement layer. |
| `taxonomy/eg2022.py` | the two answers → the tree. Named for the last wave in the pool, per the registry convention. |

## Getting the data

Everything here fetches itself and nothing needs an account.

```
python sources/eg.py --fetch          # ~46 MB of Arab Barometer zips, all ten waves
python sources/eg_geo.py --fetch      # 15 MB COD-AB shapefile bundle + 27 small CAPMAS calls
python sources/eg_grid.py --fetch     # 6.8 MB gzipped Kontur gpkg
python sources/eg.py                  # -> data/normalized/eg.csv
```

**The terms were read before anything was downloaded**, because of Nişanyan (§11ac, where a
retrieval clause killed a better source than this one). Arab Barometer's FAQ: *"Anyone can
download the Arab Barometer data for analysis at no cost"*, the data are *"publicly available
and free of charge"*. The name/email/purpose form on the downloads page is a request step
rather than a licence, no retrieval restriction, no redistribution clause, no stated citation
requirement, and the real file URLs are printed in the page's own HTML.

## THE CAPMAS API, AND WHY THE POPULATION IS NOT COD-PS

**HDX's `cod-ps-egy` is the 2012 COMPAS estimate: 81,395,541 people.** Egypt passed 100
million in 2020. Drawing today's Copts on a fourteen-year-old denominator would put a quarter
of the country's people nowhere, and unevenly, because the governorates have not grown at the
same rate. §9bn's Ecuador made the same call against a COD-PS error of **3.4%**; this one is
**34%**.

So the magnitude comes from CAPMAS instead, and finding it is the reusable part.

**§11d and §11af both concluded CAPMAS has no API**, on a sound test run against the wrong
thing: `www.capmas.gov.eg` is a React shell that returns the same 1,421 bytes with HTTP 200
for every path, including `/nonsense-path-xyz-123`. `[[reference_spa_hidden_apis]]` says to
grep the bundle, and `/static/js/main.81637733.js` has the constants block:

```
API_EXTERNAL_TRADE_ENDPOINT_URL: "https://www.capmas.gov.eg:8090"
API_ENDPOINT_URL:                "https://www.capmas.gov.eg:8080"
SEARCH_API_ENDPOINT_URL:         "https://www.capmas.gov.eg:8087"
```

**The API is on a different PORT, not a different path.** That is the generalisable line and
it is not in the memory note yet: a probe that walks paths on the 443 host cannot find it, and
the negative it returns is indistinguishable from a dead portal. Port 8080 answers with no
key, and among its 120 routes:

```
api/Governorate                            27 governorates, Arabic + English names, ids, areas
api/GovernoratePopulation                  ?governorateId=<id>&date=<YYYY-MM-DD>
api/GovernoratePopulation/MinAndMaxDate     2018-01-01 .. now
```

Three things checked before it was used:

* **It is deterministic for a fixed date.** The same governorate asked twice for 2026-01-01
  returns the same integer, so `POP_DATE` is pinned and the response cached to `data/raw/eg/`.
  A clock that moved under the build would re-level the country by a few months of growth on
  every run.
* **It is a per-governorate series, not one national clock apportioned.** The governorate
  shares move between 2018 and 2026 (Cairo 10.039% → 9.647%, Minya 5.819% → 6.093%), which a
  fixed-weight split could not do.
* **The 2018-01-01 total is 96,364,295**, against the 2017 census's 94.8 million two months
  earlier. It is anchored where it should be.

Total at the pin: **108,528,518**.

## THE JOIN IS ON THE ARABIC, AND THERE ARE FOUR WITNESSES

COD and CAPMAS transliterate differently and neither is wrong — COD says `Suhag`,
`Kalyoubia`, `Kafr El-Shikh`, `Sharkia`, `Behera`; CAPMAS says `Sohag`, `Qalyubia`,
`Kafr ElSheikh`, `ElSharqeya`, `Behaira`. **English is not a key here.** `sources/eg_geo.py`
requires all four of:

1. an authored `GOVERNORATES` table, pcode → (CAPMAS id, display name);
2. the **Arabic** name on each pcode against CAPMAS's Arabic for that id, folded for the alef
   forms, final ya and ta marbuta only;
3. **CAPMAS's own area against COD's polygon area**, ρ = +0.950, and **0 of 5,000 random
   pairings** reach it (best random +0.634);
4. the five desert governorates coming out the five sparsest.

Witness 3 is a rank correlation rather than a per-unit tolerance because two of the
twenty-seven genuinely disagree and neither is an error; see the Luxor section.

**Witness 4 was written the wrong way round first, and it is worth the line.** The obvious
form is *"New Valley sparsest, Cairo densest"* and **Cairo is not the densest governorate in
Egypt** — the Cairo governorate carries 3,085 km² of desert expansion east of the city, while
Qalyubia is 1,336 km² containing Shubra El Kheima. An assertion that encodes a plausible
belief rather than a measured fact fails on correct data, and the time it costs is spent
hunting a join error that is not there.

## COD-AB'S LUXOR IS THE CITY, NOT THE GOVERNORATE

COD's Arabic for it is `مدينة الأقصر`, *City of Luxor*, and its polygon is **596 km²** against
CAPMAS's **5,428**. Checked geometrically rather than inferred: Luxor city falls inside COD's
Luxor, and **Esna and Armant, both in Luxor governorate, fall inside COD's Qena.**

The consequence is a placement one and not a count one. Luxor's 1,459,385 dots are packed into
the city polygon, and the rest of Luxor governorate's territory takes Qena's colours. Both
governorates draw at 8–9% Christian, so the visible cost is small, and it is recorded rather
than patched because the alternative is editing a COD boundary by hand.

It is also why `sources/eg_grid.py` prints Luxor at **2.42x** and Qena at **0.80x** on the
per-governorate Kontur ratio. That table is printed and not asserted (§9t) and this is the
case it was printed for.

## THE LABELS, WHICH ARE `[[reference_pooled_survey_labels]]` TWICE OVER

### The geography, which §11af found

45 distinct `Q1` labels for a country with 27 governorates, because each wave brings its own
label set. Three spellings of Kafr El Sheikh, three of Faiyum, `Sohag`/`Souhag`,
`Asyut`/`Assiut`, `Minya`/`Menia`. Two prove it is a mechanism and not untidiness:

* **`The Lake` is Beheira.** al-Buhayra means *the lake*, as al-Sharqiyya means *the eastern*
  (which appears as `Eastern`) and al-Gharbiyya *the western* (as `Western`). A
  transliteration join misses all three; a translation join finds them.
* **`The West Bank` appears in the Egyptian rows of wave III**, n=60, and is dropped.

**Harmonise before testing.** On the raw labels the split-half fails at +0.416 against a +0.566
bar, because only 13 of 27 units appear in both halves and the test is being run on mangled
units. Harmonised it passes. The lesson is not that Egypt passes; it is that a stability test
run before the units exist reports noise as a negative, and a negative is what this project
treats as evidence.

### The religion codes, which is new here and is the worse trap

**`Q1012`'s numeric codes are re-used for different answers between waves:**

| code | wave III | wave IV | wave V | wave VII |
|---|---|---|---|---|
| 1 | Muslim | Muslim | Muslim | Muslim |
| 2 | Christian | Christian | Christian | Christian |
| 3 | Other | Other | **Jewish** | Other |
| 4 | Jewish (Yemen only) | Jewish | **Atheist** | **No religion** |
| 5 | — | — | other | — |

A pooled frame keyed on the code silently merges three different answers and nothing about the
result looks wrong. `arabbarometer.load()` therefore decodes each wave through **that wave's
own** label set before anything is pooled, and the label string is the category from then on.
This did not bite here because §11af's scratch scripts happened to decode per wave too, but it
is one line away from biting Lebanon or Iraq, where codes 3, 4 and 5 are not empty.

## The two atheists, and why they are dropped rather than spread

Pooled, Egypt's answers are Muslim, Christian, and **two wave V respondents who chose
`Atheist`**. `sources/gt.py` spreads its one Guatemalan Jew at the national rate, so the
default here would be to do the same. It is not done, for a reason that is about the card
rather than the count:

**the option was on one of the four questionnaires.** Wave V offers `Atheist`, wave VII offers
`No religion` (a different answer), and waves III and IV offer neither. A share pooled across
all four for a box that existed on one of them measures which questionnaire was in the field.
LAPOP's card is stable across its waves, so Guatemala's case is not this case. Applying
2/6,840 to 108 million people would draw about thirty thousand irreligious Egyptians on the
strength of two interviews in one round.

And **a floor is not a magnitude**: in Egypt, saying this to a stranger with a clipboard
carries a real risk, so two is a lower bound of unknown depth. §14.4 rule 1 forbids inventing
the correction and nothing published supplies one, so `note_public` says it instead.

## `The West Bank` is probably Gharbia, and it is still not mapped

Wave III's Egyptian rows carry `The West Bank` on code 2010, n=60, and the code appears on **no
non-Egyptian row**, so it is not two countries' code ranges overlapping in a shared label set.
Three things say it is Gharbia:

1. wave III's codes 2001–2023 reproduce **Egypt's own governorate numbering in order**, and
   position 10 in that numbering is Gharbia;
2. **Gharbia is otherwise absent from wave III** and present in all three other waves, and it
   is Egypt's fifth most populous governorate;
3. the mechanism is already visible two rows above. al-Gharbiyya is *the western*, and a
   translator handed it alone offers *the West Bank* for al-Diffa al-Gharbiyya, the same way
   the same file offers `The Lake` for al-Buhayra.

**It is not mapped anyway.** It buys 60 respondents, 0.9% of the pool, in a governorate that
has 287 more from the other three waves. Being wrong about it is
`[[reference_name_join_wrong_neighbour]]` exactly: sixty people assigned to a governorate they
are not in, preserving every total, invisible to every arithmetic check the build runs. A small
gain is not worth the project's own worst failure mode. Recorded here so nobody re-derives it,
and so a later source can settle it.

## The checks, re-run rather than cited

`sources/eg.py` fails the build on the first two of these.

| check | this build | bar or comparison |
|---|---|---|
| weighted national Christian share | **5.93%** | 1986 census, the last published: **5.7–5.8%** |
| Cairo, pooled | **8.51%** | census Cairo: 9.3% (1986), **8.57% (1996)** |
| governorates differ, chi-square | p = **1.5e-37** | — |
| Upper Egypt vs the rest | **11.9% vs 4.7%** | p = 2.9e-26 |
| split-half rank correlation, 23 units | **+0.518** weighted, **+0.495** unweighted | bar **+0.418**, so both pass |
| held-out: unit share of respondents vs CAPMAS | r = **+0.996** over 24 | 0 of 20,000 random pairings reach it |

### Two numbers here do not match §11af's, and both are accounted for

**The split-half.** §11af reports **+0.495** and this build reports **+0.518**. The difference
is entirely that §11af's was **unweighted** and this one is weighted, because the weighted
share is what the map draws; putting the weights back reproduces +0.495 to three decimals. Same
23 units, same bar, same verdict.

**Kafr El Sheikh.** §11af's pooled table prints **1.9%** and this build prints **1.56%**. Its
harmoniser had `"kafir el-sheikh"` as a `NORM` key while its own key function stripped hyphens
first, so wave III's 40 Kafr El Sheikh respondents could never match and were dropped as
unmapped. Removing them here reproduces 1.85%, which is what §11af rounded. The bug did not
touch the split-half, because Kafr El Sheikh has zero Christians in waves III and IV either
way and its rank does not move. Every other row of §11af's ordering reproduces exactly.

## Weighted and unweighted disagree about which governorate is top

This is the one thing in this country a reader could be misled by, so it is stated in
`note_public` and here.

| | unweighted | weighted, which is what is drawn |
|---|---|---|
| Sohag | 16.9% | **13.90%** |
| Minya | 16.1% | **16.41%** |
| Asyut | 14.8% | **13.58%** |

The weights are the survey's own and they are what makes the national reading agree with the
census: **5.93% weighted against 6.68% unweighted**, and the 1986 census published 5.7–5.8%.
So they are used. But the three top governorates sit inside one another's 95% intervals either
way (about ±3.7 pp on samples of 317–392), so **the order among them is not a finding** and
`note_public` says to read them as a group.

## What is drawn, from `data/normalized/eg.csv`

24 of 27 governorates, **107,658,120 people**, two categories, 48 rows, every one `modelled`.

```
93.98%  Muslim      101,173,209   -> islam
 6.02%  Christian     6,484,911   -> christianity
```

Christian share as drawn, top and bottom:

```
Minya          16.41%   n=392    1,084,788      Kafr El Sheikh   1.94%   n=256
Sohag          13.90%   n=337      834,170      Beheira          1.83%   n=468
Asyut          13.58%   n=317      718,078      Ismailia         1.70%   n=112
Aswan          12.41%   n=104      213,417      Sharqia          1.06%   n=526
Port Said      10.97%   n= 44       88,108      Red Sea          0.00%   n= 28
```

**The Red Sea is drawn as entirely Muslim** because none of its 28 respondents answered
Christian. On 0/28 the 95% upper bound is about 11%, so that is not a finding that no Copts
live in Hurghada, and `note_public` says so. Matrouh (n=16) and Port Said (n=44) are the other
thin ones against a median of 271; Guatemala drew Zacapa on n=40 and named it, which is the
same call.

## Not drawn

**New Valley, North Sinai and South Sinai**: no respondents in any of the four waves, so no
share to apply. **870,398 people, 0.80% of Egypt, on 48% of its land.** `gap_share=0.0080`,
hand-written, because these people were never in any table and `tools/gap_share.py` correctly
refuses (*"the mapping excludes nothing"*).

Filling them at the national rate was considered and rejected: it would assert that the
Western Desert oases and the Sinai look like Egypt's average, and §14.4 rule 1 forbids
inventing a magnitude. Russia's `ru_fill.py` fills its four unsurveyed subjects because a
published census ethnicity table predicts them at R² 0.964; Egypt has no such predictor.

## Not chased

* **`dialogueacrossborders.com/.../AWRpapers/paper52.pdf`** (*Discrepancies Between Coptic
  Statistics in the Egyptian Census and Estimates*) is **403 to every client tried**, here and
  in §11af, and `web.archive.org` is not reachable from this harness. It is a browser job. The
  Cairo series it carries (10.1% in 1976, 9.3% in 1986, 8.57% in 1996) is what `note_public`
  and the Cairo check quote, on §11af's record plus two independent corroborations, and the
  primary has not been opened by anybody here.
* **IPUMS** holds `eg1986a`, `eg1996a` and `eg2006a` with a `RELIGION` variable and would
  settle the governorate series outright. `[[reference_ipums_account]]`: still blocked.
* **`Q1012A_CHRISTIAN`** (wave VII only) would split Coptic Orthodox from the rest. Sixty-six
  Egyptian Christians answer it, which cannot carry a national split let alone a governorate
  one. `taxonomy/eg2022.py`'s REVIEW has the argument for staying on `christianity`.
* **`Q1012A_MUSLIM`** is not a madhhab column; §11af measured 45–82% answering *just a Muslim*
  across five Arab countries.

---

## Review, 2026-09-08 (`rd-review`, second perspective)

Recomputed from `data/raw/arabbarometer/*.sav` with a scratch script that does **not** import
`sources/eg.py` or `sources/arabbarometer.py` — its own reader, its own copy of `NORM`, its own
`key()`. Everything below is what that independent pass found.

### Both validation discrepancies reproduce exactly

| claim | independent recompute |
|---|---|
| split-half **+0.518** weighted, **+0.495** unweighted, 23 units, bar +0.418 | +0.5183 / +0.4950, 23 units, bar +0.4180. Identical for `Christian` and `Muslim`, which a two-box card forces |
| §11af dropped **40** Kafr El Sheikh respondents | wave III's label is `Kafir el-Sheikh`, and wave III contributes **exactly 40** of the governorate's 256 |
| removing them reproduces **1.85%** | 1.852% unweighted, against §11af's rounded 1.9% |
| the drop did not move the split-half | Kafr El Sheikh has 0 Christians in waves III and IV either way |
| `Q1012` code re-use, wave by wave | the four `.sav` label sets match the table in `sources/arabbarometer.py` character for character, including wave V's code 3 `Jewish` / code 4 `Atheist` |
| `The West Bank`, n=60, wave III only, code 2010 | confirmed, and it is the only unmapped label in the pool |
| 2 atheists, wave V only | confirmed |
| Minya 16.41 / Sohag 13.90 / Asyut 13.58 weighted; Sohag 16.91 / Minya 16.07 / Asyut 14.83 unweighted | confirmed to two decimals, so the order really does flip on the weighting |
| Matrouh n=16, Red Sea n=28, Port Said n=44, median 271 | confirmed (median 271.5) |
| Sharqia 1.06%, Beheira 1.83%, Cairo 8.19% | confirmed |

**And the failure class is now guarded rather than described.** §11af's harmoniser could drop a
key without erroring; `sources/eg.py` raises `SystemExit` on any `Q1` label that reaches it
without a governorate, and `ab.stability` raises if the both-halves overlap is not the 23 its
bar was computed for. Those two assertions are what make the Kafr El Sheikh bug a build failure
next time instead of a silent 0.4 pp.

### The Kafr El Sheikh paragraph compares unweighted to unweighted without saying so

*"§11af's pooled table prints 1.9% and this build prints 1.56%"* — 1.56% is the **unweighted**
pooled reading, which is what `sources/eg.py` prints in its chi-square block. What the build
actually **draws** is 1.94%, which this same file's own *What is drawn* table gives and which
`data/normalized/eg.csv` confirms (74,175 of 3,817,182). The comparison against §11af is sound
because both sides are unweighted; the wording is what makes a reader meet 1.56% and 1.94% in
one file with nothing to tell them apart.

The same conflation appears twice more, and is worth a sweep before Lebanon rather than a patch
here:

* `sources/eg.py`'s module docstring heads with `93.3% Muslim / 6.7% Christian` and
  `split-half +0.495`. Both are the unweighted pool; the build draws 93.98/6.02, and
  `ab.stability` is weighted, so it prints +0.518. `taxonomy/eg2022.py` has the drawn pair right.
* `ab.stability`'s own docstring says *"Harmonised, the same data give +0.495"* — again the
  unweighted figure, attributed to a function that computes the weighted one. That one is
  shared-module documentation that four more countries will read.

Nothing drawn is affected. The figures in `countries.py` are the weighted ones throughout and
they are right.

### `44% of its land` was wrong, and is corrected to 48%

`data/raw/eg/capmas_governorate_population_2026-01-01.json` carries `area_sqkm` per governorate.
New Valley 429,151 + North Sinai 26,121 + South Sinai 30,948 = **486,221 km² of 1,004,484**, so
the three unsampled governorates are **48.4%** of Egypt, not 44%. Corrected in `note_public`, in
`_eg_counts`'s docstring, in `sources/eg.py`'s `UNSAMPLED` comment and in the *Not drawn*
section above. The population side (870,398 people, 0.802%, `gap_share=0.0080`) recomputes
exactly and is untouched.

### `note_public` handles the headline honestly

It names Minya, Sohag and Asyut in weighted order and then says in the same breath that they
*"sit inside one another's margins, so read them as a group and not as an order"*, and puts the
claim it can actually support, the middle Nile against the Delta, in its place. No single winner
is named. That is the right handling of a three-way tie. It does not mention that the order
flips under the survey's unweighted reading; that belongs here rather than in the note, and it
is here.

### A trap in `sources/arabbarometer.py` that Egypt escapes and Lebanon does not

The question put to this review was whether the `Q1012` code re-use is guarded by an assertion
or only by a comment. **The code side is guarded twice over** — `load()` decodes each wave
through that wave's own `variable_value_labels` before anything is pooled, so a numeric code
never survives into the frame, and `undecoded` raises on any code the wave has no label for.
That is construction plus an assertion, and it is right.

**The label side is guarded by neither, and it is the same failure one column across.** After
the decode, pooling is on the raw label string: `load()` strips the country name but not the
answer, and nothing asserts that a given answer is worded the same way in every wave. Measured
across all eight waves for the four countries §11af lines up next:

```
lebanon   Other   100 (IV) + 80 (VI-2) + 14 (VIII) = 194
          other   190 (V)
jordan    Refused to answer 6  ...  refused 1
iraq, yemen: no clash
```

So a Lebanon pool built on this module today would carry `Other` and `other` as **two
categories, 194 and 190 respondents**, for one box on one card. The consequences, in order of
how quietly they happen: `national()` reports two answers where the card has one; each is tested
by the split-half on half the respondents, which halves the power of the test that decides
whether a category carries its own geography; each is measured against `ELIGIBLE_FLOOR`
separately, so a category that clears 1% whole can fail it twice; and `taxonomy/lb*.py`'s `MAP`
would need both spellings or one of them resolves to `None`. Every one of those preserves the
totals.

Not fixed here, because the fix is a judgement rather than a typo: case-folding would be wrong
if two waves ever use one wording for two different answers, so the right guard is probably an
assertion that the decoded label sets of the pooled waves agree once folded, raising with the
clash rather than silently merging. That is the next builder's call, and it is one line from
biting. `sources/lb.py` should not be written before it is made.

Also noticed while in there, not chased: `ab.held_out` raises when **any** of 20,000 random
pairings reaches the observed r. Egypt has 24 units and clears it, and `sources/lapop.py` has
the identical rule, so this is precedent rather than something new. But Lebanon's Arab Barometer
cut is about eight units and 8! = 40,320, so the identity permutation gets drawn roughly every
other run: a perfect decode can hard-fail that check on nothing but a small unit count.

### The map, one glance

Screenshotted at country fit and again over Asyut to Aswan. Dots follow the Nile ribbon and the
Delta, nothing in the sea, Sinai and New Valley correctly blank, Christian yellow visibly mixed
through Upper Egypt and thinning in the Delta. **The Luxor/Qena boundary is not visible** — no
seam, no colour step, and the Luxor packing reads as ordinary settlement density, because the
Kontur weights put the dots where the people are anyway. Recording it rather than patching a COD
boundary was the right call. The Red Sea coast strip does read as a hard 100% green, which is
the 0/28 sample; `note_public` names it.

Nothing here is a §14 question and no ask was filed.

---

## The label guard the review asked for, 2026-09-08

The two things the review above left open in `sources/arabbarometer.py` were taken up on the
same day, ahead of Lebanon, Iraq, Jordan and Yemen. Egypt is untouched by both:
`data/normalized/eg.csv` re-runs to the same SHA-256 it had before, `5ab6494d…6299`, 48 rows
and 107,658,120 people. Only `held_out`'s printed lines changed, and only in wording.

### `ab.assert_one_wording` — one answer, one spelling, or it will not pool

`load()` now ends by grouping the pooled `category` values under a **deliberately narrow**
fold — Unicode NFKC, collapsed whitespace, stripped edge punctuation, `casefold` — and raising
if one folded form has more than one raw spelling. On the real `.sav` files it gives exactly
what the review predicted:

```
Lebanon's pooled waves spell one answer more than one way, so the pool would carry it as
two categories with the totals still adding up:
    'other' arrives as 2 categories:
      'Other'  194 respondents  (wave IV n=100, wave VI-2 n=80, wave VIII n=14)
      'other'  190 respondents  (wave V n=190)
```

**It does not fold the data, and refusing to was the whole point.** Merging those two is
almost certainly right for Lebanon and is still a reading of two showcards rather than a typo
correction, and from inside the module a wave that spells one answer two ways is
indistinguishable from a wave that spells two answers similarly. So the module names the
clash, the counts and the waves, and hands the call to whoever writes `sources/lb.py`, where
it can be re-worded next to its reason. Egypt's pool passes untouched, which it must: **the
guard is not that every wave offers the same answers.** They do not, the module docstring says
why, and asserting set equality would fail Egypt on wave V's `Atheist` alone. A missing answer
is a different card; two spellings of one answer are the same card twice.

**What it will not catch, on purpose.** Jordan pools `Refused to answer` (6, waves VI-2 and
VII) beside `refused` (1, wave V), and those are two different sentences, not two spellings of
one. Verified: `load("Jordan")` returns both and does not raise. Widening the fold far enough
to merge them would also merge answers that are genuinely different — Lebanon's own
`Something else: SPECIFY_______` (66, wave VI-3) against its `Other` is the case sitting right
there — and a guard that hard-fails on a judgement is worse than one that misses it. Jordan's
pair is non-response and will be dropped by that country's `DROPPED` anyway; it is written
down here so it is met on purpose rather than discovered.

### `held_out` at eight units: the observed ordering is not one of the wrong answers

The review's second item. The old rule failed if **any** of 20,000 sampled orderings reached
the observed r, which is right when the orderings vastly outnumber the draws and breaks when
they do not. Measured on a synthetic eight-unit decode that is perfect apart from 3% sampling
noise, r = +0.99994: **the old rule hard-fails it in 3 runs out of 10**, on nothing but unit
count. Two changes, and neither is a lowered bar:

* The observed ordering is excluded from the null, **by value rather than by index**, which
  also excludes orderings that only swap units of equal population. Those reproduce the
  observed r exactly, so no correlation can tell them from the truth and counting them as
  beating it is a false alarm rather than a catch.
* Under `EXACT_PERM_MAX` = 50,000 orderings, which is 8 units and fewer, every ordering is
  checked instead of 20,000 being sampled. The result is then a proof rather than a sample —
  *none of the 40,319 other ways of pairing these units reaches this r* — and it costs about a
  second. It is also strictly harder to pass: the exhaustive pass on the synthetic case found
  one beating ordering that 20,000 draws had missed, on a variant whose two smallest units
  were near-tied.

Still fails what it should: the same eight units against a deliberately mispaired population
table give r = +0.358, which 7,643 of the 40,319 orderings match or beat, and it raises.
`held_out` now also prints the ceiling on its own evidence — `n` units can express at best 1
in n!−1 — and says so loudly under seven units, where the check cannot carry a join on its own
however cleanly it passes. That is `sources/lapop.py`'s own "reported, not asserted" argument
about its age check, applied to itself.

**`sources/lapop.py` is not changed, and its docstring now says why.** Its three drawn
countries are El Salvador 14 units, Guatemala 22, Ecuador 23; 14! = 8.7e10 against 20,000
draws is a probability of 2e-7, so none of them can have this failure and editing a module
three built countries import to fix it would be the worse trade. The note in its `held_out`
docstring names the boundary and the countries that will hit it — Costa Rica has 7 provinces,
Panama 10 — and points at arabbarometer's version.

### Found on the way, not fixed: wave I stops `load()` for three of the four

`ABI_English.sav` has **no `Q1012` column at all**, for any country. Wave I fielded Jordan,
Lebanon and Yemen, so `load()` reaches `wave I has <country> rows but no Q1012` and raises
before it can get to anything else; those three cannot currently be loaded at all, and only
Iraq and Egypt, absent from wave I, get through. The assertion is doing its job — a wave with
respondents and no religion question is worth stopping on — but the module has no way to say
*"wave I did not ask this one"*, because `expect_waves` is checked only after every wave has
been read. Left for whoever builds `lb`, `jo` or `ye`, because the fix is a judgement about
whether a missing column means a wave that never asked or a re-release that renamed something,
and that is better made by someone with a country in front of them. It is the first thing they
will hit, in the first minute.
