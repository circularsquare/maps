# Sweden — `sources/se.py`, `sources/se_geo.py`, `taxonomy/se2024.py`

Drawn 2026-09-11; rebuilt 2026-09-14 with Catholics, Orthodox and Jews at their riksområde's
rate (§2). **21 units, 51 nodes, 10,405,479 people, 99.55% of the country.**
No Swedish census has asked religion and the population register has not carried it since
1991, when the Church of Sweden handed folkbokföringen to the tax agency.

| | |
|---|---|
| counting geography | **NUTS 3, the 21 län** — 495,000 people each; Catholics, Orthodox and Jews at their NUTS 2 riksområde's rate inside each län |
| placement | 290 kommuner, GISCO LAU 2021, weighted by kommun population |
| basis | self-identification (survey) for citizens; nationality-derived for foreign residents |
| tier | **`modelled` throughout** |
| vintage | ESS 2010–2016 pooled (län), 2010–2023 pooled (riksområde), census 2021, Pew 2020, church register 2021 |

---

## 1. The custom-tabulation shelf, which is §9cu's instruction and which pays off sideways

`queue.md`'s ESS block carries a standing instruction added when the Netherlands was drawn:
**check the office's custom-table shelf before building any of these from ESS.** CBS publishes
religion per gemeente as a *maatwerk* table that is in neither its OData catalogue nor any
StatLine page, and it beat the ESS route by thirty times the resolution. The named Swedish
equivalent is SCB's *beställd statistik*.

**Sweden has no such shelf, and that is a real negative rather than a failure to look.**

- `scb.se/vara-tjanster/bestall-data-och-statistik/` lists six ordering services — the
  business register, regional statistics products, statistics tables, survey design,
  microdata, available data sources — and every one is framed as work done *"efter dina
  önskemål"*. There is no archive of previous orders and no catalogue of past
  specialbearbetningar. Ordering is bespoke and paid.
- `scb.se/hitta-statistik/sok/?query=trossamfund` returns ten results and none is a religion
  table: civil-society accounts, the organisations'-economy survey, three occupational
  median-age tables, and tax-assessment pages that mention religious-community fees.
- The reason underneath is legal rather than editorial. Sweden does not register religious
  affiliation for anybody, so SCB holds no religion variable of its own to tabulate. This is
  the difference from the Netherlands that makes the two cases opposite rather than parallel:
  CBS could produce a gemeente table because the Enquête Beroepsbevolking **asks** religion of
  460,000 adults. No Swedish official survey does.

**And then the thing SCB does hold turns up on somebody else's website.** The Church of
Sweden's `Medlemsutveckling 2020-2021, per församling, kommun och län samt riket` says in its
own header:

> *Uppgifter om folkmängd, medlemmar i Svenska kyrkan samt avlidna medlemmar tas fram av SCB
> på uppdrag av Svenska kyrkan.*

Population, membership and deceased members, produced by SCB on the Church of Sweden's
commission, for every parish, all 290 kommuner and all 21 län. So the commissioned religion
tabulation exists; it is simply published by the customer and not by the office.

**§11k's line was `Sweden. SCB carries nothing.`** That is true of SCB's own catalogue and
false of what SCB produces, and the generalisable form is in spec §12: *a statistics office
that publishes no religion table may still produce one on commission, and the place to look
for it is the customer's website.*

The other tier was checked too and is genuinely national-only. **MUCF** (which absorbed SST,
the Myndigheten för stöd till trossamfund, in 2025 — `myndighetensst.se` now 301-redirects
into `mucf.se` and its old `/kunskap/statistik-om-trossamfund.html` path 404s there) publishes
`betjänade` per faith community at
`mucf.se/rapporter-och-statistik/trossamfund-antal-betjanade`, as HTML tables with no
download: 47 named bodies, 844,694 people in the 2024 column, **and no geography at all**.
That is §11k's Nordic pattern exactly — every denomination nowhere, or one denomination
everywhere. Those figures are used here, but for a mapping decision rather than for a map; §5
is where.

## 2. Which rounds, and the level question, which the queue got wrong in a useful direction

Sweden is in ESS rounds 1–9 and 11 and **is absent from round 10**. Rounds 1–4 have no
`region` variable at all (`E201VariableNotFound`), the wall Greece, France, Italy and Finland
all hit. That leaves six, and `regunit` splits them:

| rounds | `regunit` | units | citizens |
|---|---|---|---:|
| 5, 6, 7, 8 | NUTS level 3 | **21 län** | 6,449 |
| 9, 11 | NUTS level 2 | 8 riksområden | 2,657 |

`queue.md` priced Sweden at *"NUTS-2 (8) to verify"*. Verifying it moves the country a level
finer for four of six rounds, and those four hold 71% of the sample. **This is Italy's problem
(sources.md §9as) with the halves swapped, and Italy's answer applies: split by category.** The
first build (2026-09-11) called it a straight choice between two whole builds and drew
everything at the län; it was rebuilt on 2026-09-14, with Anita's approval, as below.

**Each level wins different categories.** Run §3's test (median rho over every round split,
permutation null, the same function the build uses) at all three:

| | 21 län, r5-8 | 8 units, r5-8 | 8 units, all six |
|---|---:|---:|---:|
| No religion | +0.253 | **+0.643** | **+0.655** |
| Svenska kyrkan | **+0.414** | **+0.857** | **+0.762** |
| Islam | **+0.321** | **+0.756** | **+0.714** |
| **Annan protestantisk församling** | **+0.333** | **−0.119** | +0.381 |
| Katolska kyrkan | +0.011 | **+0.814** | **+0.833** |
| Ortodoxa kyrkan | +0.140 | **+0.796** | **+0.888** |
| Judisk | +0.266 | +0.679 | **+0.875** |

(bold = passes both its own permutation null, which differs by unit count, and the spatial
chi-square, at 0.05)

**A whole coarse build would give geography to more categories and lose the one thing about
Sweden worth drawing.** At 8 units the Catholics and the Orthodox come back, and `Annan protestantisk
församling` goes to **−0.119** — not weakened, reversed. The reason is visible in the
geography: NUTS 2 puts Jönköping (6.25%) in `Småland med öarna` with Kalmar (0.69%) and
Gotland, so the free-church belt is averaged against its own opposite. That is aggregation
destroying a signal, not a sample failing to find one.

**So the construction is Italy's (`sources/it.py::_composition`): a category takes the finest
level at which it passes both of §3's tests, and a riksområde's share is applied inside each of
its län, times that län's own citizen population.** Counting stays at the 21 län, where the
census measures the foreign half, and every citizen row's `note` in `se.csv` names the level
its share came from.

| level | pool | categories |
|---|---|---|
| 21 län | rounds 5-8, 6,449 citizens | Svenska kyrkan, Islam, Annan protestantisk församling |
| 8 riksområden | all six rounds, 9,106 citizens | Katolska kyrkan, Ortodoxa kyrkan, Judisk |
| national rate inside each län's residual | rounds 5-8 | No religion, Annan kristen församling, Österländsk religion, Annan icke-kristen religion |

Three calls inside that:

1. **The riksområde pool is all six rounds.** Rounds 9 and 11 were only ever left out because
   they have no NUTS 3. For a category drawn at NUTS 2 that reason is gone, and they add 41%
   to categories that have 70 and 40 respondents in rounds 5-8. Italy did the same: each
   level takes every round that carries it.
2. **`No religion` passes at the riksområde and stays the residual.** It is 68% of citizens
   and the complement of the län categories. Fixed at its riksområde's share, the small tail
   left over has to absorb each län's own departure in Svenska kyrkan, Islam and the free
   churches, and it goes negative in 6 of 21 län (Kronoberg −5.27%). As the residual it still
   runs from 50.79% of Kronoberg to 76.43% of Gävleborg. `KEEP_AS_RESIDUAL` in `se.py` holds
   it there and `_compose` prints the check on every build.
3. **`Judisk` passes at the riksområde on ten respondents and is drawn there, as the rule is
   written.** Seven of the ten are in Stockholm, which is what the chi-square detects (0.011;
   0.013 on a Monte Carlo version, since the expected counts are about one per unit). The rest
   of the ordering is two respondents in Småland, one in Sydsverige and none in Västsverige.
   So Stockholms län draws 0.28% Jewish, Kalmar and Gotland 0.13%, and Västra Götaland 0.01%
   (its foreign residents only), although the Jewish population is in Stockholm, Gothenburg
   and Malmö. Before the rebuild every län drew about 0.09%. Adding `Judisk` to
   `KEEP_AS_RESIDUAL` reverses it.

What moved: Catholics now run from 4.37% of Stockholms län to 0.94% of Västernorrland (before
3.53% to 1.69%), the Orthodox family from 2.65% to 0.49% (before 2.06% to 1.07%). Nothing at
the län level moved.

**An earlier version of this section said the opposite on a worse statistic.** With one
chronological halving against `spearman_null`'s fixed bar, the eight NUTS 2 regions looked
excellent on rounds 5-8 (+0.952, +0.881) and looked to fail outright once rounds 9 and 11 were
pooled in (+0.381, +0.548), and the conclusion drawn was that adding 2018 and 2023 destroyed
the ordering. **It does not.** That was one arbitrary halving reading as a result; the median
over every split shows the all-six-round build is the strongest of the three on five of six
categories.

`build()` prints rounds 9 and 11's national composition beside rounds 5-8's: **every category
moves by less than 3.5 points**. Unaffiliated 68.08% → 64.71%, Church of Sweden 23.38% →
26.04%, Islam 2.59% → 2.96%.

**And one halving is not a statistic, which Sweden shows better than any country so far.**
Four rounds admit three distinct halvings. On `Svenska kyrkan` at 21 län they give **+0.125,
+0.434 and +0.458** against a fixed bar of +0.3701: one verdict of "no" and two of "yes" from
the same data, decided by which two rounds happen to be put together. spec §12 carries it.

## 3. Which categories carry their own geography — §9cy's test, plus one requirement

Belgium brought §14.16's split-half to ESS on the same day and **rebuilt its null to do it**:
resample by ROUND rather than by PSU (the API returns cross-tabs and no PSU), take the MEDIAN
Spearman over every split, and compare it to a permutation of the unit labels drawn **per
round**. `sources/se.py::_stability` is `sources/be.py`'s function, copied unchanged, because
two countries on one instrument on one day must not use two tests. Greece, Finland, France,
Germany and Italy still draw every ESS category where it was measured.

**Sweden adds a spatial chi-square at 0.05 as a second requirement.** §9bi states it in
the direction of overrides — *"Before proposing one, run the chi-square: if the units do not
differ, there is nothing to draw"* — and this is the same warning from the direction nobody
had hit. Median rho over 3 splits, 2,000-draw permutation null, seed 0:

| category | n | median rho | null 95th | p | chi² p | verdict |
|---|---:|---:|---:|---:|---:|---|
| No religion | 4,397 | +0.253 | +0.309 | 0.0925 | 3.1e-08 | national rate |
| **Svenska kyrkan** | 1,551 | **+0.414** | +0.305 | **0.0135** | 5.1e-15 | **own geography** |
| **Islam** | 136 | **+0.321** | +0.317 | **0.0475** | 2.1e-04 | **own geography** |
| **Annan protestantisk församling** | 130 | **+0.333** | +0.317 | **0.0440** | 5.0e-08 | **own geography** |
| Annan kristen församling | 74 | +0.173 | +0.333 | 0.1989 | 1.5e-01 | national rate |
| Katolska kyrkan | 70 | +0.011 | +0.337 | 0.4988 | 3.0e-01 | national rate |
| Ortodoxa kyrkan | 40 | +0.140 | +0.316 | 0.2439 | 3.0e-01 | national rate |
| Annan icke-kristen religion | 21 | +0.447 | +0.357 | **0.0180** | 3.2e-01 | **REFUSED on the chi-square** |
| Österländsk religion | 23 | +0.431 | +0.355 | **0.0215** | 4.0e-01 | **REFUSED on the chi-square** |
| Judisk | 7 | +0.266 | +0.352 | 0.1169 | 8.7e-01 | national rate |

Three of ten carry at the län. **At the 8 riksområden**, all six rounds, ten
three-against-three splits, the same null and the same chi-square:

| category | n | median rho | null 95th | p | chi² p | drawn at |
|---|---:|---:|---:|---:|---:|---|
| No religion | 6,087 | +0.655 | +0.440 | 0.0025 | 3.0e-11 | residual (§2, call 2) |
| Svenska kyrkan | 2,303 | +0.762 | +0.452 | 0.0005 | 1.9e-16 | län |
| Islam | 191 | +0.714 | +0.452 | 0.0025 | 1.2e-06 | län |
| Annan protestantisk församling | 182 | +0.381 | +0.476 | 0.0930 | 1.1e-05 | län |
| Annan kristen församling | 108 | −0.060 | +0.453 | 0.5852 | 4.2e-01 | residual |
| **Katolska kyrkan** | 107 | **+0.833** | +0.476 | **0.0005** | 1.4e-04 | **riksområde** |
| **Ortodoxa kyrkan** | 59 | **+0.888** | +0.470 | **0.0005** | 1.0e-03 | **riksområde** |
| Annan icke-kristen religion | 31 | +0.087 | +0.460 | 0.3748 | 8.5e-01 | residual |
| Österländsk religion | 28 | +0.368 | +0.460 | 0.1029 | 3.5e-01 | residual |
| **Judisk** | 10 | **+0.875** | +0.515 | **0.0040** | 1.1e-02 | **riksområde** (§2, call 3) |

The four left are still **drawn**, at the national rate inside each län's own residual
(§9bi's construction, not a flat rate), so the partition stays closed and what is withdrawn is
only the claim to know where those people are.

**THE TWO REFUSALS ARE THE FINDING THAT GENERALISES.** 21 and 23 respondents, both clearing a
permutation test built for exactly this problem, both with spatial chi-squares saying the 21
län are indistinguishable. **A Spearman over a column that is zero in most units is decided by
how the ties break**, so a rank test does not merely lose power on a tiny category, it can be
*passed* by one — and the permutation null does not rescue it, because the null is permuting
the same mostly-zero column. Ten tests at alpha 0.05 expect half a false pass; these are two
candidates sitting exactly where one would look. Adding the chi-square can only make a
category fail, so it is not the forbidden move of tuning a bar until something passes. It also
replaces `lapop.ELIGIBLE_FLOOR`'s 1% size gate, which would have refused the same two for a
worse reason: **size is eligibility, the chi-square is evidence, and only the second one is
about whether there is a geography there.**

**`No religion` fails at the län and it barely matters**, which is §9cy's reading of the same
result. It is 95% of the residual the passing categories leave, so it still moves with each
län's own measured Lutheran, Muslim and free-church share and comes out from 50.79% of
Kronoberg to 76.43% of Gävleborg. What it is not is a direct estimate the map stands behind, and
`note_public` says so.

**The free-church result is the best thing in this country.** `Annan protestantisk församling`
is 130 respondents in four rounds and it passes both tests: **7.71% of Örebro, 6.25% of
Jönköping, 4.22% of Västerbotten** against 3.05% nationally, with Kalmar last at 0.69%. That is
Örebromissionen, the Svenska Alliansmissionen and the EFS coast, three revivals a century apart,
recovered in the right three places from a sample nobody would have bet on. It is the strongest
evidence here that the instrument works, and it is why the two noise passes had to be refused
individually rather than the whole small-category tier being distrusted.

## 4. The register, which is the country's subject and is not a validation

The Church of Sweden PDF gives membership and total population for all 21 län at 31 December
2021: **5,627,932 members, 53.94% of 10,434,479 people.** This map's `Svenska kyrkan` is
**21.47%**. The 32-point gap is real and is Finland's gap (§9by, 17 points) at twice the size:
membership is conferred by infant baptism and ended by filling in a form, and about three and
a half million Swedes have not filled it in and do not describe themselves as belonging.

**The more interesting half is that the two instruments do not order the län the same way.**
Spearman **+0.247** over 21 län, p=0.28.

(survey column = the ESS `Svenska kyrkan` share among citizens, which is what the
register is comparable to. `note_public` quotes the DRAWN share instead, e.g. Kronoberg
35.56%, because its denominator includes the foreign half.)

| | register top | survey top |
|---|---|---|
| 1 | Norrbotten 67.6% | Kronoberg 39.4% |
| 2 | Jämtland 65.8% | Kalmar 34.9% |
| 3 | Värmland 65.5% | Halland 31.3% |
| 4 | Blekinge 64.0% | Norrbotten 32.5% |
| … | | |
| 21 | Stockholm 45.0% | Stockholm 15.0% |

Nominal membership is highest in the sparse north; stated belonging is highest in the Småland
and south-western belt where the free churches and the high-church revival are. Both ends of
the register's list and both ends of the survey's are stable; the middle shuffles. Gävleborg
is the largest single disagreement, 8th on membership and 19th on belonging.

**This is not a failed validation and it must not be written up as one.** The two instruments
measure different quantities, and a person who is on the rolls and says they belong to no
religion answers one way to the registrar and the other to the interviewer. What it does do is
put a ceiling on how much weight the survey's Lutheran geography can bear, and `note_public`
says so in those words.

The join to the PDF is by an explicit 21-row table and not by name: the church spells two län
differently from ESS and GISCO (`Dalarna län` for Dalarnas, `Kalmars län` for Kalmar), which
is precisely the shape that slips through a name join. `_register()` then **checks each row's
own population against the census's for that code** and exits if any differs by more than 2%,
which is a join check that does not look at names at all
(`[[reference_name_join_wrong_neighbour]]`).

## 5. The Orthodox answer, split three ways, and why Sweden is the exception

ESS offers one `Ortodoxa kyrkan` box. Every other country on this map with an
undifferentiated Orthodox answer sends it to `christianity.orthodox` on an explicit
arithmetic argument — Austria's note says the Armenians are 1.0% of its cell, the UK's says
filing it on `oriental` instead "would be wrong for 98%", and Canada sends its unspecified
remainder to Eastern because that is where the overwhelming majority of a Canadian Orthodox
answer belongs.

**In Sweden that argument runs the other way.** MUCF's 2024 `betjänade`, grouped by communion:

| | | |
|---|---:|---:|
| **Eastern Orthodox** | 69,153 | 45.85% |
| **Oriental Orthodox** | 72,133 | **47.82%** |
| **Church of the East** | 9,542 | 6.33% |

The non-Chalcedonian half is the larger one, because of the two Syriac jurisdictions in
Södertälje at 26,045 and 20,550. Putting the whole cell on `christianity.orthodox` would
assert the wrong communion for about half the people in it, across the schism that
`christianity.oriental`'s own node text calls the commonest error in religion taxonomies.

So the answer is **split on MUCF's counts**, per spec §3.11, into `christianity.orthodox`,
`christianity.oriental` and `christianity.churchofeast`. Three existing nodes, no new legend
row. `sources/se.py` emits three source categories off the one ESS code, so the operation is
visible in `se.csv` and `tools/check_mapping.py` sees all three rather than it being hidden
inside `resolve()`; and it happens **after** the split-half, so the test and the respondent
counts are on the answer the survey actually collected.

Two things named rather than fitted:

- **`betjänade` counts residents and this ratio is borrowed across the citizen/foreign line.**
  It is conservative in the direction that matters. The Syriac and Assyrian population came
  from Tur Abdin in the 1960s–80s and later from Lebanon, Syria and Iraq, is heavily
  naturalised and now third-generation, so it sits in the citizen half; the Romanian,
  Bulgarian and Polish Orthodox are post-2007 EU movers with little reason to naturalise and
  sit in the foreign half. The citizen cell is therefore if anything **more** Oriental than
  47.82%. Left unadjusted because nothing publishes the adjustment.
- **Some Assyrians will have answered `Annan kristen församling` instead.** The Church of the
  East is not called Orthodox in its own name, so 6.33% is a floor for that node.

The split is a national constant applied inside every län, so it moves nobody geographically.
What it does is stop about 75,000 people being filed in a communion half of them left in 451.

This call was put to a second agent with the files and not the reasoning, as `AGENT_BRIEF` §3
prescribes; it returned the MUCF grouping independently and recommended the split, and the
figures above were then read off the MUCF page directly rather than taken from its report.

## 6. What the build cannot do, in one list

- **The placement weight is population and not religion.** Stockholms län is one composition
  over 2.4M people, so its 6.3% Muslim share spreads by where anyone lives and the real
  geography — Järva and the north-western suburbs against the inner city — is invisible.
  Södertälje gets its Oriental Orthodox dots at the county rate like everywhere else.
  **SCB publishes population by country of birth per kommun, openly**, which would let Sweden
  use the Italy weighter for the foreign half and put the origin-derived dots where the
  origin-born people are. That is a named improvement, not a thing this build does, and it is
  the single highest-value follow-up here.
- **The free churches are one cell.** MUCF counts Equmeniakyrkan (96,460), Pingst (101,925),
  Evangeliska Frikyrkan (41,255), EFS (29,571) and Svenska Alliansmissionen (17,619)
  separately. ESS has one box and no geography for any of them.
- **EFS is invisible twice over.** It works inside the Church of Sweden, so its members answer
  `Svenska kyrkan`, which is Finland's Laestadian problem in Swedish.
- **The Shia are not separable in the citizen half.** MUCF puts Islamiska Shiasamfunden at
  35,388 against roughly 168,000 across the Sunni organisations, so something like a sixth of
  organised Swedish Islam is Shia. ESS offers one Islam code. Sweden's Shia dots come from the
  foreign half only, where `origin_religion.py` splits Iraq, Iran, Lebanon and Afghanistan.
- **Alevis have nowhere to go.** Alevitiska Riksförbundet reports 3,624 betjänade, ESS has no
  box, and `islam.alevi` does not exist on the tree — Türkiye's §11ac left them inside the
  Islam parent for the same reason.
- **The residual is most of every län.** §9bi's construction gives each län's tail as what is
  left after its six measured categories, and since `No religion` stays in it the tail is
  55.93% of Kronoberg to 82.94% of Gävleborg against 70.04% nationally. That is
  the house construction working as designed, and it means the unaffiliated share on this map
  is the complement of the measured religious shares rather than a reading of its own.

## 7. Numbers to check a rebuild against

```
6,449 answered Swedish citizens, rounds 5-8, 21 län, median 203 per län
9,106 answered Swedish citizens, all six rounds, 8 riksområden
10,452,316 people in cens_21ctz_r3 = 9,571,492 NAT + 856,212 FOR + 24,612 unknown
5,627,932 Church of Sweden members at 31/12/2021, 53.94%
drawn 10,405,479 of 10,452,316 — 99.55%, 51 nodes
unaffiliated 63.47%  christianity.lutheran 21.47%  islam family 5.31%
christianity.protestant 3.05%  christianity.catholic.latin 2.57%  orthodox family 1.66%
judaism 0.09%
at the lan: Svenska kyrkan, Islam, frikyrka; at the riksomrade: Katolska, Ortodoxa, Judisk
```

## 8. One thing the checklist got wrong, fixed here

`COMMANDS.txt` ran `coverage.py` at step 7, before the scatter. **The check reads the dots**,
so a country with none on disk yet passes vacuously. Sweden passed at step 7 and then failed
with **eighteen uncovered nodes** once its dots existed, every one of them the foreign half's:
eight national Orthodox churches, two Oriental ones, Eastern Catholicism, Shia Islam, Alevism,
Hinduism and Buddhism. The fix is one line in `coverage.py`'s foreign-half tuple, `("se",
"other.se")`, and the step is now numbered after the scatter with the reason written beside it.
`tiles.py` reads coverage, so a fix there means the build tail runs again.
