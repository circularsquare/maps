# Jordan — Arab Barometer, waves II to VIII (2010–2024)

Built 2026-09-09. The second country on this map drawn from the Arab Barometer, after Egypt
(`sources/eg.md`, `sources.md` §9bz). `sources.md` §11af is the assessment of the source across
the region and §11ag's row A is the group Jordan came out of. **Read §9bz before Lebanon, Iraq
or Yemen; read this file too, because three of its findings are the survey's rather than
Jordan's** and all three will bite those countries.

## 1. The state route, tested first and closed rather than missing

Jordan is the clean case of a country that **asks and does not publish**, and the evidence is
DOS's own paper rather than an absence of search results.

**Both censuses ask religion, and the questionnaires prove it.** DOS publishes them:

* **2015**, `dosweb.dos.gov.jo/DataBank/census2015/Questionare_en.pdf`. The household-members
  block runs items 201 to 216. Counting the column headers, 201 Population Category, 202
  Serial number, 203 Name, 204 Relation to head, 205 Sex, 206 Date of birth, 207 Age, **208
  `Religion`, `1.mustim 2.chistian 3.other`** in DOS's own spelling, 209 Nationality.
* **2004**, `dosweb.dos.gov.jo/DataBank/Census2004/questionare/private_e.pdf`, the Private
  Household Register. The same block, the same position: **`الديانة`, `1. الإسلام 2. المسيحية
  3. أخرى`**.

**And neither answer has been published, at any geography including the nation.**

| where it was looked for | what is there |
|---|---|
| 2015 census statistical tables | **133 tables in ten sections** (buildings, housing units, general population, education, economic, Jordanians abroad, refugees, non-Jordanians, deaths, difficulties). No religion table. |
| 2004 census statistical tables | **165 tables in eight sections.** No religion table. |
| `jorinfo.dos.gov.jo` PxWeb databank | **520 tables, enumerated in full.** Thirteen mention a religion word and every one of them is ISIC division 94, *activities of religious organisations*, inside an economic table. |
| the census `Main Result` and `Demographic features` analytical reports | 115 pages between them; the only occurrence of the word is `religious education` in a definition of secondary-school streams. |
| UNSD Demographic Yearbook table 28 (`tools/oracle.py`) | **absent**, which is what §11r's oracle proves and no more. |

**The 2015 figures were withheld deliberately and there is a public trace of it.** Eissa
Masarweh, writing at `sahafi.jo`, estimates **197,000 resident Jordanian Christians at mid-2021,
2.7% of the Jordanian population**, and says plainly that he is carrying the 2004 census's ratio
forward *because the 2015 religion figures were not released*. That is the only external number
this country has, it is not a state publication, and it is used here as a band rather than as a
target. See §5.

**The PxWeb API is at `/Databank/api/v1/en/` and not under `/pxweb/`** — Ghana's trap (§9's
`sources/gh.md` 1) on a different host, and `/Databank/pxweb/api/v1/en/` answers **HTTP 500**,
which reads as a broken API rather than a wrong path. It also rate-limits hard, **429 with no
`Retry-After`** after about thirty requests; a sweep needs a sleep and a backoff or it silently
returns a partial tree. The first pass of the walk here found 48 tables and looked complete.

**There is no COD for Jordan at all.** `cod-ab-jor`, `cod-ps-jor` and `cod-em-jor` all 404 on
HDX, and the Jordan group there carries no administrative or population COD of any kind. So the
usual question of whether to prefer the office's figures over COD-PS (§9bn's Ecuador, §9bz's
Egypt) does not arise: DOS is the only source and it is also the better one. Boundaries are
geoBoundaries `gbOpen JOR/ADM1`, pinned to commit `9469f09`.

## 2. What is drawn

**11,937,000 people, twelve governorates, two nodes, every row `modelled`.** No gap: all twelve
governorates are sampled in all nine waves, so every person DOS counts gets a dot.

| | as drawn |
|---|---:|
| Muslim | 11,770,050 (98.60%) |
| Christian | 166,950 (1.40%) |

Christian share by governorate, as drawn, with the pooled respondents behind each:

| governorate | share | n | Christians drawn |
|---|---:|---:|---:|
| Balqa | **4.33%** | 766 | 26,557 |
| Ajloun | 3.67% | 467 | 8,069 |
| Karak | 2.32% | 712 | 9,156 |
| Amman | 1.57% | 5,280 | **78,590** |
| Mafraq | 1.54% | 784 | 10,558 |
| Aqaba | 0.94% | 447 | 2,371 |
| Irbid | 0.85% | 2,642 | 18,771 |
| Zarqa | 0.66% | 2,039 | 11,192 |
| Madaba | 0.51% | 455 | 1,196 |
| Ma'an | 0.25% | 448 | 490 |
| Jerash | 0.00% | 480 | 0 |
| Tafilah | 0.00% | 386 | 0 |

**Amman holds 47.1% of all the Christians drawn and is not the most Christian governorate.**
That is the shape of the country and it is worth saying explicitly, because the map's colour
gradient shows Balqa and the map's dot count shows Amman.

**Madaba comes out low and it is the one figure here that reads wrong.** Madaba town is one of
the oldest Christian centres in Jordan; the governorate draws at 0.51% on 455 interviews, five
of which were Christian. The 95% interval is ±0.65 points, so the survey cannot separate Madaba
from 2% and the drawn figure should not be read as a finding about Madaba. It is in
`note_public`.

## 3. The three things that are the survey's rather than Jordan's

### 3a. Wave II was invisible to `ab.load` and is 1,188 of these respondents

Wave II, and only wave II, prints its `country` value labels as **`8. Jordan`, `5. Egypt`,
`17. Saudi Arabia`** — the numeric code repeated inside the label. `ab.load` selected a country
with `label.lower() == country.lower()`, found nothing, and fell through the `continue` that
exists for a wave which did not field the country. **No error, and a smaller pool than the file
contains, for every Arab Barometer country ever built here.**

Wave II has `q1012 Religion`, `q1 Province/Governorate/State` and a weight. Jordan: 1,188
respondents over all twelve governorates, 39 of them Christian. Egypt: 1,219 over 21
governorates, 71 Christian.

`ab.country_key` now strips a leading ordinal and nothing else, and its docstring is the
write-up. **This is `[[reference_pooled_survey_labels]]` on a third column of the same survey**
— the module docstring has it on the religion codes, `assert_one_wording` on the religion
wordings, and this is it on the country.

**Egypt is NOT rebuilt on the deeper pool.** That would move an already-drawn country's numbers,
which is Anita's call and not a side effect of a shared-module fix (AGENT_BRIEF §3), so
`sources/eg.py` now passes `waves=` explicitly with the reason written at the point of use, and
`data/normalized/eg.csv` is byte-identical after the change. For whoever picks it up: wave II
reads **5.83% Christian for Egypt against the 5.93% Egypt is drawn on**, so it does not look
like a correction waiting to happen; what it would change is every governorate cell, the
split-half and the `2013-2022` vintage.

### 3b. `ab.fold` could not see wave II's answer labels either, and now can

Wave II spells its answers `1. muslim`, `2. christian`, `99999. declined to answer`. `fold`
strips case, spacing and edge punctuation, so `1. muslim` and `Muslim` **do not collide** and
`assert_one_wording` says nothing: the pool would carry Islam as two categories with every total
still adding up, which is the exact failure that function exists to catch. `fold` now strips a
leading `<n>.` as well, on the same argument as `country_key`, and the collision is detected.

That is deliberately not the same as widening the fold to merge `Refused to answer` with
`refused` — which is the thing §11af's note warns against, because a fold loose enough for that
pair would also merge Lebanon's `Something else: SPECIFY_______` into its `Other`.

### 3c. The refusals: one box, three spellings, resolved in the country module

Jordan's pool spells one interviewer-coded refusal three ways: `refused` (wave V, 1),
`Refused to answer` (waves VI-2 and VII, 6) and `99999. declined to answer` (wave II, 4).
`ab.load` grew a `recode=` argument for exactly this — applied before `assert_one_wording`,
raising on a key the pool does not contain, so the merge is a written decision in
`sources/jo.py` and not a loosened rule in the shared module.

All eleven are then **dropped**, not spread: a refusal is not a religion, and spreading it at
the national rate would draw about 9,600 Jordanians on the strength of eleven people who
declined to say.

**§3.5's lean check, with leave-one-out, because eleven refusals over twelve governorates has
no power at all.** The refused share per governorate correlates **r = +0.361** with the
Christian share, so on its face the refusals sit slightly where the Christians are and dropping
them makes the map marginally less Christian than Jordan is. **Leave-one-out is what settles
it: the correlation runs +0.031 to +0.683 depending on which single governorate is dropped.**
Eleven people cannot carry a correlation across twelve units and that range is what it looks
like when they try. So the direction is reported and not acted on, and it is in the record so
that nobody reads the drop as demonstrated-neutral either.

**And the grain of that check is worth one line, because the first version of it was wrong.**
Run on the raw `Q1` label it reported 44 units and r = −0.051, because the refusals were spread
across forty-four spellings of twelve places. The lean check has to run on harmonised units like
every other test here, and the sign flipped when it did.

## 4. Wave I asks the question and still cannot be used

`ab.load` raises on a wave that has the country's rows and no `Q1012`, and wave I is that wave
for every country. It was left open deliberately for whoever built the first country that wave I
fields, and the answer is neither of the two obvious ones.

**Wave I asks religion.** Its item is **`q711. religion`**, `1 muslim, 2 christian, 3 sunni
muslim (lebanon & bahrain), 4 shiite muslim (lebanon & bahrain), 5 druze (lebanon), 97 not
clear, 100 not provided/not usable`, answered by 1,142 of Jordan's 1,143 respondents. It is not
a re-release that renamed something and it is not a wave that skipped the question: wave I
predates the questionnaire renumbering that put the demographics block at `q10xx` from wave II
onwards.

**And it still cannot enter this pool, for a reason that has nothing to do with the decode.**
Wave I has **181 columns and not one of them is subnational.** No `q1`, no governorate, no
region, no district, no PSU. `country` is the finest geography in the file. A wave with no
geography cannot be cut by governorate however cleanly its religion column reads.

So it is excluded through `waves=` rather than allowed to fail, and it is used once, as a free
external check: **unweighted, wave I reads 1.57% Christian for Jordan**, inside the band the
other nine waves occupy.

## 5. The checks

**The pooled national share, against the only external figures that exist.** 14,906 usable
respondents; **1.64% Christian unweighted, 1.41% weighted**. Masarweh's 2.7% is of *Jordanians*,
which is about **1.8% of the resident population** DOS's denominator counts. So the survey reads
roughly **0.8x** the only outside estimate. That is a real undercount and it is stated rather
than corrected; §11ad's warning case is LAPOP at **0.21x** a census on a comparable minority
cell in Suriname, so 0.8x is a different order of problem from that one, and Egypt's equivalent
ratio was about 1.0x.

`CHRISTIAN_BAND` in `sources/jo.py` is **0.8% to 3.0%**, recorded before the build ran and
recorded honestly: it was set knowing the per-wave shares from a scratch probe, so it is a guard
against a future re-release rather than a test this build passed blind.

**The governorate decode, tested three ways.**

1. **The code witness, which is the strong one.** The `Q1` code means three different things
   across the nine waves: waves II and III number the governorates **3501 to 3512 in Jordan's
   official order**; waves V, VII and VIII use **800 followed by Jordan's own governorate
   number**, so 80011 is Amman and 80034 is Aqaba; waves IV, VI-1, VI-2 and VI-3 use an
   arbitrary 1 to 12 order that is **not even the same between IV and VI**. So the code can
   never be the pooling key — and in the six waves where it decodes, it is an independent test
   of the harmonised names. **10,175 respondents, all agreeing, zero disagreements.** A
   permutation of the name table preserves every total and is invisible to arithmetic
   (`[[reference_name_join_wrong_neighbour]]`); it could not survive this.
2. **The held-out population check.** The survey's governorate shares of respondents against
   DOS's governorate populations: **r = +1.000 over twelve units, and none of 20,000 random
   pairings reaches it.** The Arab Barometer quota-samples by governorate, so a high correlation
   is expected and is not evidence about the survey; what the permutation tests is that the
   names were joined to the right polygons.
3. **The label harmonisation itself.** 44 distinct `Q1` strings over nine waves for twelve
   governorates. Two are the mechanism rather than untidiness: **`The capital` and `العاصمة` are
   Amman** (the governorate's name is Muhafazat al-Asima, and DOS itself prints العاصمة in
   Arabic against `Amman` in English), which is §9bz's `The Lake` again; and **wave VII prints
   its governorates in Arabic** while the other eight print them in English. `Azurqa` is
   az-Zarqa with the article run in, `Ajioun` is Ajloun with the l read as an i. Transcribed,
   never repaired.

**The geography of the boundaries, four witnesses** (`sources/jo_geo.py`): the authored ISO
3166-2 table covers geoBoundaries' twelve `shapeISO` values exactly; **DOS's published area
against geoBoundaries' measured polygon area, rho = +0.993 with 0 of 5,000 random pairings
reaching it** (Jordan's governorates run from 410 km² to 32,832 km², a factor of 80); the three
desert governorates come out the three sparsest; and nine of twelve names agree letter for
letter with the three that do not being the known romanisation splits (`Jerash`/`Jarash`,
`Ajloun`/`Ajlun`, `Tafilah`/`Tafiela`).

**The split-half, which passes thinly, and the leave-one-out the brief asked for.**

    Muslim      98.59%   spearman +0.617   pearson +0.708   bar +0.591 on 12 units
    Christian    1.41%   spearman +0.617   pearson +0.708   bar +0.591 on 12 units

The two are identical **by construction and not by corroboration**: Jordan's card has two
answers, so a governorate's Muslim share is one minus its Christian share, the rankings are
exact reverses, and the correlations must agree. Egypt is the same shape.

Leave-one-out over the twelve governorates gives **+0.509 to +0.717**; eight of twelve clear the
12-unit bar of +0.591 and six clear the 11-unit bar of +0.620 that an eleven-unit test actually
faces. **The minimum is a tie and the top two governorates of the ordering are each
individually load-bearing**: dropping Balqa gives +0.5093 and dropping Ajloun gives +0.5093 as
well, to four decimals, and either one alone takes the verdict under all three bars. This
section said "the unit that matters is Balqa" until 2026-09-09, which was `leave_one_out`
reporting `min()`'s first-of-a-tie rather than the tie; §9.4 found it and the function now names
every unit on the minimum. It is a slightly worse fragility than one unit's leverage and it does
not change the verdict.

**The bar was applied as written and not moved**, in either direction: §14.16's bar is shared,
and moving it silently changes every other country (AGENT_BRIEF §3).

**And thin against that bar is not the same as marginal, which `ask/007-cr` is the reason to
check.** That ask, filed the day before this build, establishes that `1.96/sqrt(n-1)` is not
the 0.05-level test its docstring claims: it is stricter than the exact permutation null at
every unit count, and worst where units are fewest. Asked of Jordan's counts, on the ask's own
script:

| n | fixed bar | exact 95th | exact 97.5th | what the fixed bar really is |
|---:|---:|---:|---:|---|
| 12 | +0.5910 | **+0.4965** | +0.5804 | a 0.023-level test |
| 11 | +0.6198 | +0.5273 | +0.6091 | a 0.022-level test |

So **Jordan's +0.617 clears the fixed bar, the exact 95th and the exact 97.5th**, and this
country does not move whichever way that ask is ruled. The pass is more comfortable on the
honest null than on the one that was applied. **What remains fragile is Balqa's leverage, and
no choice of bar addresses that**: the worst leave-one-out value, +0.509 on eleven units, fails
all three bars.

The fragility buys a sentence in `note_public` rather than a different verdict, and three
things stand behind the verdict that the split-half cannot see:

* the twelve governorates differ at **p = 2.6e-18** (chi-square, 11 dof);
* **Balqa, Madaba and Ajloun together read 3.50% against 1.41% elsewhere, p = 4.1e-10**;
* the ordering the survey returns is the one Jordan's own Christian geography would predict.

And the alternative is not a more careful map. With two answers and neither carrying,
`ab.build` gives every governorate the national rate, so the map would say 1.41% in Balqa and
1.41% in Tafilah, which the data contradicts at p = 2.6e-18.

**Why the split-half is as thin as it is, which is worth knowing before Lebanon.** The cut falls
after wave IV, so the early half is waves II, III and IV (n=4,479) and the late half is V, VI-1,
VI-2, VI-3, VII and VIII (n=10,427). The early half contains **wave III, whose Jordanian
Christian share is 0.34% weighted against 1.0% to 2.1% in every other wave** — ten Christians in
1,795 respondents. That single anomalous wave is a third of the early half's sample and most of
its noise. The wave is kept: dropping a wave because its answer is inconvenient is the one move
this file must not make.

## 6. The universe, which is not the same universe in every wave

**DOS's 11,937,000 counts everybody; the survey mostly counts citizens.** Jordan's 2015 census
counted 9,531,712 residents of whom **2,918,125 were non-Jordanian**, so roughly three in ten
are not citizens: Syrians above all, then Egyptians, Palestinians without citizenship and
Iraqis.

**Wave IV is the only wave that sampled them, and it measures them as different.** Its
`q1020jo` records **303 Syrians among 1,500 Jordanian respondents**, 20.2%, which is about the
non-citizen share of the country. Every other wave returns Jordanian or Palestinian origin for
all but a handful (14 `Other` in wave VII, 11 in wave V, 4 in wave IV).

| wave IV, by origin | n | Christian |
|---|---:|---:|
| Syrian | 303 | **0** (0.00%) |
| Jordanian or Palestinian | 1,197 | 28 (2.34%) |

Chi-square **p = 0.014**. `sources/jo.py::origins` re-reads this out of the `.sav` on every
build and raises if wave IV stops carrying either variable, so the finding cannot be quoted
after it has stopped being true.

**Nothing corrects for it**, because no published source gives the religion of Jordan's
non-citizens at any geography and §14.4 rule 1 forbids inventing one. What is done is what §3.5
asks: the shares are applied to the whole resident population, and `note_public` says the
Christian figure is a **ceiling** rather than a middle estimate.

The alternative considered and rejected was to draw only the citizen population and put the
other three million in `gap`. It was rejected because DOS publishes no citizen population by
governorate in the current estimates, so the denominator would have to be constructed from the
2015 census and aged forward, and because leaving 30% of a country blank is what §3.5 treats as
a last resort. If somebody wants to revisit it, the input is the 2015 census's non-Jordanians
section, tables 8.1 to 8.9.

## 7. Why this was drawn at governorate without a new ask

`ask/answered/001-eg` asked the §14.4 rule 2 question for Egypt and Anita ruled **draw at
governorate, because governorates are pretty big**. She then ruled on this group directly in
queue.md §11ag row A: **build them, and decide what to show once there is something to show**,
with Lebanon named as the one to raise with her because its confessional balance is the reason
it has had no census since 1932.

Jordan is the weakest case in the group for rule 2 and the calculation is not close:

* **twelve governorates for 11.9 million people**, against Egypt's twenty-seven; the smallest,
  Tafilah, holds 120,300 people and Amman holds 5 million;
* Jordan's Christians are **not situated as Egypt's Copts are**. The constitution reserves
  parliamentary seats for them, the churches publish their own counts, and the community's
  geography is not a secret in Jordan;
* the resolution is the instrument's ceiling anyway: `Q1` is Governorate and there is nothing
  below it in any wave, and DOS has published nothing at any tier including the nation.

So the call was mine and it is recorded here rather than filed. **If a reviewer disagrees, the
reversal is cheap**: the same build at one national unit is a two-line change to
`sources/jo.py`'s `units`, and nothing else on the map depends on Jordan.

## 8. What would replace this

* **The 2015 census religion table.** It exists inside DOS and would replace the whole
  construction with counts, probably at district (`liwa`) rather than governorate, since
  DOS's population estimates go to sub-district. That is the thing to re-check whenever DOS
  releases anything new.
* **IPUMS `jo2004a`.** The World Bank microdata catalogue lists *Jordan Population and Housing
  Census 2004 - IPUMS Subset* as study 505, and the 2004 questionnaire asks religion, so the
  variable is very likely in it. **Blocked**, `[[reference_ipums_account]]`.
* **A denominational split.** Nothing in this survey has one, and the Greek Orthodox / Melkite
  Greek Catholic split is the interesting thing about Jordanian Christianity. `taxonomy/jo2024.py`'s
  `REVIEW` explains why it is not guessed at.
* **The Jordan Population and Family Health Survey**, DOS's own DHS, runs 1990 to 2023 and is on
  the World Bank catalogue. Its data dictionary shows no religion variable and DHS microdata is
  institutionally walled anyway (§11ag row D), so it is not a route.

## 9. Review, 2026-09-09

*Session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-jo-rev`, `.claude/commands/rd-review.md`. Read
from the `.sav` files, `countries.py`, `data/normalized/jo.csv` and `ask/007-cr` rather than
from §§1-8. **Nothing needs rebuilding and no dot moved.** `check_md.py` clean,
`built_countries.py --check` names nothing, `check_rollup.py jo` 11,937,000 all modelled and 0
orphaned, `sources/jo.py` re-run and `data/normalized/jo.csv` byte-identical at sha256
`6313786b…7bc8fe67`. Screenshot clean: dots on the western highlands, the Jordan valley and the
Aqaba spur, Amman dense, the eastern desert nearly empty, none in the sea.*

### 9.1 The wave II finding is real, and it is general to every country in the file

Checked by dumping every wave's `country` value labels straight out of `pyreadstat`, without
importing `ab`. **Wave II is the only wave of the ten that prefixes them**: 22 of 22 labels
carry an ordinal, and 0 of the other waves' 103 labels do. Replaying the old exact-match filter
against the new `country_key` over every country key in the file: **all 23 gain wave II, with
one exception**, and five of them (`bahrain`, `comoros`, `djibouti`, `somalia`, `syria`) had no
Q1012 wave at all before it. So §3a's *"for every Arab Barometer country ever built here"* is if
anything understated: it was also every country that could ever have been built here.

**The exception is Saudi Arabia and it is the next country to trip on this.** Wave II spells it
`17. Saudi Arabia` and wave V spells it `Kingdom of Saudi Arabia`; `country_key` strips the
ordinal and nothing else, by design, so those remain two keys. A Saudi build that passes either
spelling gets one wave and no warning, which is the same silence in a new place. Declining to
fold further is the right call (`country_key`'s own docstring gives the reason) and this is the
line that pays for it: **whoever builds `sa` must pass both spellings and check the count.**

**Corrected 2026-09-09, and the correction matters more than the original.** Everything above
was measured on the *declared* value labels, and declared is not present. Re-measured on the
rows, per country per wave:

* **Eight countries gain wave II, not twenty-three**: Algeria, Egypt, Iraq, Jordan, Lebanon,
  Palestine, Sudan and Tunisia, between 1,188 and 1,538 respondents each. The other fifteen keys
  are labels a wave declares over zero rows, or rows with an empty `Q1012`.
* **`comoros`, `djibouti`, `somalia`, `syria`, `oman` and `emirates` have no rows anywhere in
  the ten files.** They are entries in wave II's code list and nothing else. `bahrain` has 435
  respondents and they are all in wave I, the wave with no `Q1012` at all. So the sentence
  *"five of them had no Q1012 wave at all before it"* was right about the five and wrong about
  what they gained, which was nothing.
* **Saudi Arabia is not a country that gets one wave. It is a country this survey cannot
  draw.** Wave II has 1,404 Saudi respondents with a governorate (`8001. Riyadh` through
  `8006. Asser`) and a weight and **zero `Q1012` answers**, and wave V declares `Kingdom of
  Saudi Arabia` at code 17 against **zero rows**. Both spellings, pooled, give an empty pool.
  Mauritania is the same shape: 3,200 respondents in waves VII and VIII, no religion answer.
  Qatar's only religion wave is IV, 518 respondents.

The alias was still added, because the near-miss is real and the next one will not be a country
with an empty column: `ab.COUNTRY_ALIASES` names both Saudi spellings and `ab.load` now raises
on an empty pool with the reason above written into the message, so nobody re-derives it.

### 9.2 The `fold` widening merges nothing that is genuinely different, checked on every country

This was the thing worth checking, and it is clean. Every distinct `Q1012` value label anywhere
in the ten waves is 28 strings. Under the new fold they collapse into 21 keys, and **the ordinal
strip is responsible for exactly four of those merges**: `1. muslim`/`Muslim`,
`2. christian`/`Christian`, `10001. jewish`/`Jewish`, `0. missing`/`Missing`. In all four the
text after the ordinal is character-identical to the other spelling. **There is no label
anywhere in the file that the new fold merges with a label of different wording.**

Run per country rather than globally, because that is the set `assert_one_wording` actually
sees, the new fold raises collisions in eight of the fourteen countries and every one of them is
that same `<n>. <name>` against `<Name>` pattern. **Lebanon's known case survives untouched**:
`Something else: SPECIFY_______` folds to `something else: specify` and `Other`/`other` fold to
`other`, three separate wordings and three separate keys, exactly as before. The three refusal
wordings stay three as well: `declined to answer`, `refused to answer` and `refused` are
distinct keys, which is what forces `sources/jo.py`'s `RECODE` to be a written decision.

### 9.3 Egypt's pin is real and Egypt did not move, and here is what rebuilding would cost

`sources/eg.py` pins `WAVES = ["III", "IV", "V", "VII"]` and passes it as `waves=`, so wave II
never enters `load()`; and since none of those four labels carries an ordinal in either the
country column or the answer column, **both changes are provably no-ops for Egypt's pool**.
Verified rather than reasoned: `sources/eg.py` re-run here, `data/normalized/eg.csv` unchanged at
sha256 `5ab6494d…6bb36299`.

**But §3a's *"nothing about it looks like a correction waiting to happen"* is a statement about
the national share only, and the national share is not where the movement is.** Egypt was built
twice here, pinned and deeper, writing nothing:

| | pinned (III, IV, V, VII) | deeper (plus II) |
|---|---:|---:|
| respondents | 6,778 | 7,767 |
| national Christian, as drawn | **6.024%** | **5.831%** |
| Christians drawn | 6,484,912 | 6,277,561 |
| split-half spearman / bar | +0.518 / +0.418 | +0.541 / +0.418 |

So the country total moves **0.19 points, about 207,000 people, 3.2% of the Christians drawn**,
and the split-half gets slightly *better*. Per governorate the median move is 0.41pp and the
ordering is nearly preserved, rank correlation +0.987. **The whole of the change is in one
governorate and it is at the top of the ordering.** Asyut goes 13.58% to 17.45%, +3.87pp, and
**displaces Minya as Egypt's most Christian governorate**, which is the one sentence about Egypt
a reader is most likely to carry away. Nothing else moves more than 1.6pp.

**And that flip rests on 28 interviews.** Wave II reads Asyut at **40.0% Christian on n=70**,
against 13.6% in the pooled four waves. Arab Barometer clusters its PSUs, so one or two Coptic
neighbourhoods drawn in Asyut in 2011 produces exactly that, and no test in this module can tell
that from a real difference. **My view, recorded and not filed: the pin was the right call and
the case for keeping it is stronger than §3a makes it**, because the deeper pool does not just
add depth, it moves a published superlative on a single clustered cell. Two things pull the
other way and Anita should have them: 5.831% is **closer to the 1986 census's 5.7-5.8%** than
the drawn 6.024% is, and 1,219 respondents is **18% more sample than Egypt uses**.

**A rebuild is also not the two-line change it sounds like.** Wave II's `Q1` spellings include
three that `eg.py`'s `NORM` has no key for: `2008. Qaliubiya` (90), `2009. Kafr el-Sheikh` (50)
and **`2007. East` (89), which is not an Egyptian governorate name at all** and would need a
decision rather than a spelling. That is 229 of wave II's 1,219 respondents, so the table above
is a floor on the movement. The split-half halves also re-partition, the cut moving from after
wave IV to after wave III so the early half becomes II plus III rather than III plus IV, and the
`2013-2022` vintage becomes 2010-2022.

### 9.4 The split-half, independently, including the part the record does not say

`+0.617` against `+0.591` reproduces, and so does the whole leave-one-out range, +0.509 to
+0.717, 8 of 12 clearing the 12-unit bar and 6 of 12 the 11-unit one.

**§5 says "the unit that matters is Balqa" and two governorates matter equally.** Dropping Balqa
gives +0.5093 and **dropping Ajloun gives +0.5093 as well**, to four decimals; they are ranks 11
and 12 of the early half and either one alone takes the verdict under all three bars at eleven
units. `leave_one_out` prints only Balqa because `min(rs, key=rs.get)` returns the first of a
tie, so the tool is not wrong, it is silent about a tie. The honest statement is **"the top two
governorates of the ordering are each individually load-bearing"**, which is a slightly worse
fragility than the record describes and does not change the verdict. Jerash and Tafilah each
take it to +0.5457, which also fails the 11-unit fixed bar.

**Fixed 2026-09-09.** `leave_one_out` now names every unit within the printed precision of the
minimum instead of the first of them, §5 says the tie rather than Balqa, and the build prints
*"2 units matter equally and each is individually load-bearing: Balqa, Ajloun"*. The function is
Jordan's own and no other country calls it, so the change is local; the shared `ab.stability`
has no equivalent `min()` and needed nothing. `data/normalized/jo.csv` is byte-identical after
it, at sha256 `6313786b…7bc8fe67`.

**One caveat on the null that §5's table does not carry, and it runs the safe way.** Jordan's
observed statistic has **tied ranks**: three governorates read exactly 0.000% Christian in the
early half and three in the late half, so it is a Spearman with ties compared against an untied
permutation null. Rebuilt properly, permuting the tied rank vector itself, the null is
indistinguishable: 95th **+0.4965** (identical), 97.5th **+0.5816** against +0.5804. Jordan's
exact one-sided p under the correct null is **0.0175**. So the tie is worth knowing about and
changes nothing here.

### 9.5 `ask/007-cr`'s appended arithmetic is right

Re-derived independently, 4,000,000 Monte Carlo orderings per unit count, without using that
ask's script:

| n | fixed bar | my 95th | ask says | my 97.5th | ask says | P(rho >= fixed bar) |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | +0.6198 | +0.5273 | +0.5273 | +0.6091 | +0.6091 | 0.0220 |
| 12 | +0.5910 | +0.4965 | +0.4965 | +0.5804 | +0.5804 | 0.0228 |

Every figure in the 2026-09-09 appendix reproduces, and so do the ask's original rows (n=7 at
0.0172, n=22 at 0.0238) and its correction's n=20 at +0.3789. **Jordan's +0.617 does clear all
three bars and the country does not move whichever way the ask is ruled.**

### 9.6 Reader-facing figures, recomputed from `data/normalized/jo.csv` by summing people

All of them check. 1.40% Christian; Balqa 4.33%, Ajloun 3.67%, Karak 2.32%, Amman 1.57%, Madaba
0.51%, Zarqa 0.66%, Ma'an 0.25%; Amman's 78,590 is **47.07%** of the 166,950 drawn, so "nearly
half" is right and it is a count of people rather than of dots. The two rule-of-three bounds are
right as well: 0 of 480 gives 0.62% and 0 of 386 gives 0.78%, which `note_public` rounds to
"about 0.6% and 0.8%". The three survey-only figures are asserted by `note_public_figures` on
every build.

**`note_public` does not overclaim, and it is the part of this country that most easily could.**
It says to read the top three as a group rather than as a ranking, names both zero governorates
as the sample rather than the country, and singles out Madaba as the figure not to trust. Two
small things rather than defects. **14,906** appears in the note and is the one figure of the
four that `note_public_figures` does not assert; it is 14,917 minus the eleven refusals, so it
goes stale only if `DROPPED`'s count changes, but the constant is worth adding beside the other
three. And *"Balqa's figure is Fuheis, Mahis and Salt"* attributes a governorate measurement to
three towns the survey cannot resolve; it is almost certainly true and it is the one sentence in
the note asserting more than the survey or a cited source supports.

**Both done 2026-09-09.** `NOTE_NAMED = 14_906` sits beside the other three constants and
`note_public_figures` asserts it as `NOTE_POOLED` minus the `DROPPED` count, which is the moving
part: it goes stale the moment the refusal count changes, and that is the change nobody would
think to re-read the note for. The note now reads **"14,906 people who named a religion across
nine rounds"** rather than "14,906 people interviewed", which also removes a real oddity — the
same note calls 14,917 *"people interviewed"* five paragraphs later, and both cannot be that.
And the Balqa sentence now reads *"Balqa contains Fuheis, Mahis and Salt, three of the best
known Christian towns in the country, although the survey asks nobody which town they live in
and the 4.33% is the governorate's"*, which keeps the fact and drops the resolution the
instrument does not have. `check_md.py` clean, `tiles.py --refresh-meta` run, no dot moved.

### 9.7 Two shared-module notes, neither touched because builders are live

**`expect_waves` cannot do what its docstring says once `waves=` is passed.** `ab.load` computes
`got` from the frame it has already filtered by `want`, so `expect_waves=WAVES, waves=WAVES`,
which is what `sources/jo.py` does and `sources/eg.py` in effect, is a tautology. It can catch a
wave disappearing from the files but **never a wave being added**, which is the re-release its
docstring names. Fixing it means computing `got` before the `want` filter, which changes the
assertion's meaning for every country on the module and is not a reviewer's edit to make while
`ug` is building. Written here so the next Arab Barometer country can decide it deliberately.

**The mapping and the boundary join both hold.** `Muslim -> islam` and `Christian ->
christianity` are the bare family nodes, which matches Egypt and eleven other countries in
`taxonomy/`; no new node and no new legend row. `jo_lookup.csv`'s twelve populations sum to
11,937,000 exactly, and `note_public` and `source=` both state the end-2025 vintage where a
reader meets it. No em dashes and no markup anywhere in the entry's plain-text fields; the
bolded lead sentences in `note_public` are the paragraph-break convention the `countries.py`
docstring asks for, not the listicle voice it warns about. `gap_share.py jo` returns *the mapping
excludes nothing*, so the absent `gap=` is correct rather than missing.

## 10. The shared module, repaired 2026-09-09

*Session `967ffe99-93b1-4ff9-8169-4a6d5ffa084e-abfix2`, working from §9. Only
`sources/arabbarometer.py`, `sources/jo.py` and Jordan's `countries.py` entry were touched;
`sources/eg.py` was left alone on purpose and `data/normalized/eg.csv` is byte-identical at
sha256 `5ab6494d…6bb36299`, `jo.csv` at `6313786b…7bc8fe67`.*

### 10.1 `expect_waves` is now read from the files, and the failure it missed is constructible

§9.7's finding, fixed. `ab.wave_coverage(country)` reads every wave's metadata and two of its
columns, about a second and a half over all ten files, and returns the waves that hold at least
one respondent of this country with a non-null `Q1012` — **what the files offer, computed
before and independently of `waves=`**. `expect_waves` is asserted against that.

**The failure it missed, constructed and run.** Its shape is the wave II bug's own: a wave in
the file, absent from the pool, and a builder who wrote `expect_waves` from the pool it could
see. That is `ab.load("Jordan", expect_waves=E, waves=E)` with `E` the nine waves minus II.

* the old rule: the pool returns those eight, `got == list(expect_waves)` is `True`, and it
  **passes in silence** while 1,188 wave II Jordanians sit in the file;
* the repaired module **raises** — *"the files offer Jordan a religion answer in waves ['II']
  and `waves=` leaves them out of the pool, saying nothing"*;
* and the real call, `expect_waves=WAVES, waves=WAVES`, passes and writes the same file.

Three further cases were run and all behave: `expect_waves` short by a wave that IS pooled
raises; an `omit` entry naming a wave the files do not offer raises, the way a stale `recode`
key does; and the two reads of the same files, `wave_coverage`'s scan and the full read, are
cross-checked against each other on every build.

**A pool may still be narrower than the files, but not silently.** `omit={wave: reason}` is the
new argument and the reason is printed at build time. Egypt's is in `ab.OMITTED` rather than in
`sources/eg.py`, for the one reason that the pin is Anita's to rule on (§9.3) and the country
module is not to be edited while that is open; the constant says so and says to move it into
`sources/eg.py` when the ruling lands. Egypt's build now prints its own pin, which it did not
before.

### 10.2 The near-miss guard, and what it says about Saudi Arabia

`ab.COUNTRY_ALIASES` now names `Saudi Arabia` and `Kingdom of Saudi Arabia` as one country, and
`ab.assert_no_near_miss` is the guard that makes the next one cheap. Two country keys are a near
miss when the words of one are a **strict subset** of the words of the other and every extra
word is in `_FORM_OF_STATE`: `saudi arabia` inside `kingdom of saudi arabia`, `egypt` inside
`arab republic of egypt`, `emirates` inside `united arab emirates`. **`sudan` inside `south
sudan` is not**, because `south` is not a form of state and those are two countries; that is the
whole reason the rule is a word list rather than a fuzzy match. It reads the *declared* labels
across all ten waves, not the observed ones, because wave V declares `Kingdom of Saudi Arabia`
over zero rows and the observed set would have missed it entirely.

Removing the alias and asking for `Saudi Arabia` raises, naming both keys and the waves each is
declared in. With the alias in place the country resolves to both spellings and then fails on
the honest ground: an empty pool, with §9.1's corrected measurement in the message.

**The sweep the guard was written from found nothing else.** The ten waves declare 103
`country` value labels between them, folding to 23 distinct keys; Saudi Arabia is the only
near-miss pair under the rule, and no other key nests inside another at all. So the alias table
has one entry and is expected to stay small.
