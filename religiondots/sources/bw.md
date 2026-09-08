# Botswana — 2011 PHC religion by named locality

Drawn 2026-09-08. See `sources.md` §9bu, `sources/bw.py`, `sources/bw_geo.py`,
`taxonomy/bw2011.py`.

**1,384,276 people aged 12 and over, at 485 of the country's 519 ADM3 localities.**

---

## 1. Why the 2011 census and not the 2022 one

`sources.md` §11p swept Africa on 2026-09-06 and recorded Botswana as *"religion x LANGUAGE
only, no geography found in the volume that has the categories"*. **That reading of the 2022
census is correct and was checked again here.** The 2022 *Analytical Report Volume 1* runs to
168 pages; its religion chapter cross-tabulates religion with sex, marital status, locality
type (town / urban village / rural) and employment status, and there is a second table
crossing religion with language spoken at home. The same volume has district tables for
population, density, sex, youth and land ownership. It never crosses religion with district.
Volumes 2, 3, 4 and 5, the Administrative and Technical Report, the four dissemination
papers, the 2024 dissemination conference set, the Gender Monograph 2025 and the
`/census-2022-data` CSV were all read: **the 2022 census publishes no subnational religion of
any kind.**

**The 2011 census is published the other way round.** Its national volumes are also
geography-free (the *National Statistical Tables 2015*, 512 pages, has exactly one religion
table, national by age; the 2011 Analytical Report's Chapter 21 crosses religion with age,
sex, marital status, education, employment and migration, and with nothing spatial). But
Statistics Botswana also issued a separate ***Population and Housing Census 2011 Selected
Indicators*** booklet per census district, and **every one of them prints a religion table by
named village.**

So the choice was an eleven-year-old census at ~500 localities against the current one at
nothing, and it was taken without much hesitation: this project already draws Jamaica 2011,
Saint Vincent 2012, Benin 2013 and Laos 2015. **The cost is stated in `note_public`** and it
is not only the age of the figures; see §4.

## 2. The eighteen booklets, and the two that do not exist

All under `https://www.statsbots.org.bw/sites/default/files/publications/`. The series is
numbered Vol 1 to Vol 11.1 and covers 26 of the 28 census districts:

| booklet | covers | localities | 12+ answering |
|---|---|---|---|
| Cities and Towns | all seven towns, one row each | 7 | 350,901 |
| Ngwaketse (Southern) | BW0801 | 28 | 91,111 |
| Barolong | BW0802 | 33 | 38,403 |
| Ngwaketse West | BW0803 | 10 | 9,268 |
| South East | BW0901 | 6 | 67,524 |
| Kweneng East | BW1001 | 28 | 190,265 |
| Kweneng West | BW1002 | 23 | 32,001 |
| Kgatleng | BW1101 | 21 | 67,801 |
| Central Serowe/Palapye | BW1201 | 39 | 129,219 |
| Central Mahalapye | BW1202 | 36 | 82,371 |
| Central Tutume | BW1205 | 42 | 100,172 |
| North East | BW1301 | 35 | 41,696 |
| Ngamiland East | BW1401 | 21 | 63,474 |
| Ngamiland West | BW1402 | 29 | 39,149 |
| Chobe | BW1501 **and the Ngamiland Delta, BW1403** | 18 | 19,130 |
| Ghanzi | BW1601 **and the CKGR, BW1602** | 19 | 30,658 |
| Kgalagadi South | BW1701 | 18 | 20,839 |
| Kgalagadi North | BW1702 | 11 | 15,232 |

**Central Boteti (BW1204) and Central Bobonong (BW1203) have no booklet.** They are the gaps
at 6.1 and 6.3 in the series numbering. Twenty-six plausible filenames were probed under
`/sites/default/files/publications/` and every one 404s; the Wayback CDX listing for the whole
of `statsbots.org.bw` holds the same eighteen files and no more. **They were never put
online.** 129,312 people in 2011, 6.4% of the census, not drawn and named in `gap=`.

`Pop%20Indicators%20South%20East%202015.pdf` is a byte-identical duplicate of the South East
booklet. Skip it.

**The obvious next improvement, if anyone wants it:** those two districts could be filled
from the 2011 urban/rural religion split, which UNSD carries for Botswana as an exact
partition (urban 1,297,287, rural 727,617) and which is the same census and the same
instrument. It was not done here because 93.7% measured at village level seemed a better
object than 100% with a modelled hole in it, and because the fill would be worth very little:
it would give those two districts two profiles between them.

## 3. The parse, and the three ways the booklets disagree

They were typeset separately and every difference fails silently.

* **The row-total column is in three different places.** Ten booklets put it first, six put
  it last (Barolong, Kgatleng, Kweneng East, Ngwaketse, Ngwaketse West, South East), and
  Ghanzi has none at all. `sources/bw.py` decides arithmetically, on every row of the table:
  the total column is the one that equals the sum of the others. **A parser that assumed nine
  columns reads the total as `Christian` and shifts every category by one**, which reconciles
  against nothing and looks plausible.
* **The captions lie, so they are never read.** Central Serowe/Palapye captions *both* halves
  of its count/percentage pair `(%)`, and the counts are in the one labelled 12a. The Cities
  and Towns booklet captions its religion counts *"Number of people by marital status"*.
  Ngwaketse West prints every caption on the page in a block at the very end of the reading
  order. Chobe writes `Table: 7A`. **Counts are told from percentages on the values** (a count
  table contains no token with a decimal point) and tables are found by anchoring on the
  literal column header `Christian`.
* **`Other` is a row label as well as a column name.** Every booklet ends its list with an
  `Other` residual holding the localities it does not name. A parser that drops header words
  anywhere in the table drops that row too, and with it everyone who lives in a village too
  small to print. Header words are skipped only *before* the table's first number, and the
  header is treated as over as soon as a column slot is seen twice.

**Chobe and Ngamiland East print no `Not stated` column at all.** Those two booklets write a
blank rather than a fabricated zero.

**Two booklets do not add up, and both are the office's arithmetic rather than the parse.**
Ghanzi's printed Total covers 18 of its 19 rows: the CKGR is listed as one of its localities
but is its own census district and is left out of the district total. With the CKGR row
removed, seven of nine columns agree to the person and `Other` and `Not stated` are one and
two out. Central Tutume's 42 rows are each internally consistent and their row totals sum
exactly, but the Total row's own category cells are 16 out on Christian and 8 each the other
way on Badimo and No religion, netting to zero; the office evidently built that row from a
separate pass. **Both are asserted with a tight explicit bound** (`KNOWN_SLIPS`), not
tolerated by a loose tolerance. Widening the bound is the wrong repair.

**Ngamiland West prints the same table twice**, as 10A and again as 11A, with identical values
and an identical village list. It is one district table, not two halves, and the duplicate is
dropped on a value signature.

## 4. The thing that should worry the next person: no religion halved

Botswana's answer that should be checked hardest is **no religion at 14.8%** of the 2011
answering population, which is the highest figure on this map anywhere in sub-Saharan Africa.
It is well evidenced within 2011: it appears at district level in all eighteen booklets, it is
15.3% in UNSD's national table for the same census, and its geography is coherent (rural high,
urban low, with village-level extremes above 50%).

**But the 2022 census puts the same cell at 6.9%**, and Volume 1 says so in as many words:
*"a significant reduction among the non-religious population, from 15.3 percent in 2011 to 6.9
percent in 2022"*. That is a fall of more than half in eleven years. **No plausible amount of
conversion accounts for it**, and the report does not treat it as a finding needing
explanation. Two things could be going on and this file cannot separate them:

1. the 2011 question or its coding admitted a "no religion" answer more readily than 2022 did;
2. or the 2011 enumeration recorded a non-response as `no religion` in some districts. The
   printed `Not stated` cell is only 5,146 people, 0.37%, which is *implausibly low* for a
   question of this kind and is itself a hint that non-response went somewhere else.

**Read (2) as the live risk.** If it is right, some of the map's no-religion geography is
really an enumeration-quality map. The 2022 locality-type table is a weak check and it points
the same way as 2011 on the town/country gradient (towns 4.4% against rural 10.9% in 2022,
towns 9.8% against the national 14.8% in 2011), so the *shape* survives into the newer census
even though the *level* does not.

## 5. Badimo

3.86% of the answering population, 53,439 people. **`indigenous.african`, no child node.**
Badimo is the census's own word, the Setswana plural for the ancestors, and `taxonomy/bw2011.py`
argues the §2.4 case for not minting `indigenous.badimo`: a single national cell naming one
tradition does not need its own leaf when `indigenous.african` already holds exactly this, and
`zw2022.py` made the same call for Zimbabwe.

**It is a floor**, per `sources.md` §11b's continental rule: the box is exclusive of the
Christian one, so it counts people who gave the ancestors *instead of* a church.

**And its national figure is understated here by more than the average.** The eight drawn
categories come to 93.7% of UNSD's national 2011 total, but Badimo comes to only 88.2% of its
own national figure. So Central Boteti and Central Bobonong, the two missing districts, are
more traditional than the country as a whole. That is stated in `note_public`.

Its geography is Kweneng West (9.7%), Central Mahalapye (6.7%) and Central Serowe/Palapye
(6.2%), against 2.0% in the seven towns. Village extremes: Sorilatholo 46.3%, Mmanxotae 41.8%,
Moremi 39.0%.

## 6. The geography, and what actually tests it

**Unit = COD-AB ADM3, 519 polygons, which tile the country exactly** (their area is 1.0000 of
ADM0). The census counts and the polygons are at the same grain, which is unusual here.

**The join is on name and there is no code on the census side.** Names are matched WITHIN the
district and never across the country, which is load-bearing: **eight ADM3 names occur twice
in Botswana** (TULI three times, in Bobonong, Mahalapye and Serowe Palapye; also MAKALAMABEDI,
BOROTSI, CHADIBE, OTSE, PHUDUHUDU, SESUNG and TOTENG). A country-wide lookup would pair some of
them with the wrong district's polygon and every total would still reconcile.
[[reference_name_join_wrong_neighbour]].

398 of 404 named localities match one-to-one. Thirteen needed an alias, each with exactly one
plausible candidate inside its own district and each checked against the 2022 locality
report's spelling of the same village; they are listed in `bw_geo.py`'s `ALIAS`. The six that
still do not match are the Okavango Delta villages, and they land correctly anyway because the
Ngamiland Delta has exactly one ADM3 polygon.

**What tests the join is Kontur, not the names.** Its modelled population per ADM3 is built
from building footprints and knows nothing about the census, so the log-log correlation
between the two is a quantity the join does not determine. Measured rather than asserted
([[reference_check_needs_power]]): **r = 0.744 over 397 paired localities, against 0.174 for
the best of 500 random pairings** and a median of 0.033.

**A district's `Other` residual is spread over the ADM3 polygons in that district that no
named locality claimed**, weighted by Kontur population, and those rows are `derived` in §7's
sense while the name-matched ones are `measured`. The tier is decided in `bw_geo.py` and rides
in the lookup, because deriving it from the weight in `countries.py` would call a residual
that happened to land on a single polygon `measured`.

**The Chobe booklet is two districts and the split is a partition, not a guess.** Its 18 rows
are the nine Chobe villages the 2022 locality report names for district 72, plus seven
Okavango Delta rows, plus the booklet's own `Other`. The Delta seven are re-homed to BW1403.

## 7. One thing that was considered and not filed as an ask

The village-level detail makes it possible to read the map as an ethnic one, which is §14.2's
standing risk. It was checked before drawing and **the pattern does not line up that way**:
the highest no-religion districts are Kweneng West (29.8%) but also Central Mahalapye (21.6%)
and Central Serowe/Palapye (22.4%), which are Bangwato and Bakwena country, while **Kgalagadi
South, deep in the desert, is the most Christian district in Botswana at 90.8%**. The gradient
is village size and distance from a town, not community. `note_public` says that in as many
words, with Kgalagadi South named, so a reader is not left to guess.

## 8. What did not need doing

`microdata.statsbots.org.bw` is a NADA instance carrying the 2011 and 2022 censuses; the
catalogue and the DDI metadata are open but the microdata itself needs a free account, which
this session did not create. It was not needed. `botswana.opendataforafrica.org` is a Knoema
host and returns a Cloudflare 403, which is §11p's finding unchanged and a fact about Knoema
rather than about Statistics Botswana. The USCB country-geodatabase series does not include
Botswana. HDX's Botswana group has no religion dataset.

## 9. Review pass, 2026-09-08

A second agent recomputed every reader-facing figure from `data/normalized/bw.csv` without
reading this file's framing first. **Everything reproduces**, including the superlatives, so
the pattern of the six reviews before this one did not recur here.

| claim in `note_public` | recomputed | |
|---|---|---|
| no religion 14.8% | 14.807% of the eight drawn categories | ok |
| highest in sub-Saharan Africa on this map | bw 14.80%, next is Angola 12.92% | ok |
| Zimbabwe's is 8.3% | 8.27% from `counts.json` | ok |
| seven cities and towns 9.8% | 9.76% | ok |
| Kweneng West **29.8%** | 29.78%, and it is the maximum of the eighteen | ok |
| Mahalapye and Serowe/Palapye above 21% | 21.64% and 22.42% | ok |
| Monwane, Tsetseng, Leologane above half | 55.8%, 53.6%, 54.9% | ok |
| Kgalagadi South most Christian, 90.8% | 90.76%, and it is the maximum | ok |
| Badimo 3.9% | 3.860% | ok |
| 2011 national no religion 15.3% | 15.253% of the UNSD table | ok |
| Christianity 79.9% | 79.929% | ok |
| 93.7% of the national total | 93.671% | ok |
| Badimo only 88.2% | 88.164% | ok |
| grain, 2,900 people on average | 485 drawn units, 2,854 each | ok |
| twenty-eight districts | 28 ADM2 codes in `bw_localities.gpkg` | ok |

Three notes on the ones that are literally correct but read wider than they compute.

**`eastern Central District` was wrong and has been corrected to `Central District`.** Only
Bobonong is eastern. Boteti is the *western* end of the Central District: mean locality
centroid 24.78°E against Bobonong's 28.39°E, which puts it west of Kweneng West (24.86°E) and
Chobe (24.84°E), out toward Rakops and the Makgadikgadi. This is not a nitpick, because the
note's whole argument is a town-to-desert gradient and Boteti sits on the desert side of it —
which is *why* the shortfall has the shape it has. One word changed in `note_public`;
`check_md.py` clean and `tiles.py --refresh-meta` run.

**The direction of the Badimo bias is stated correctly, and it is half the story.** Per
category, drawn against the UNSD national table:

| | drawn | national | covered | undrawn people | share of the undrawn |
|---|---|---|---|---|---|
| Christian | 1,106,439 | 1,171,537 | 94.4% | 65,098 | 69.60% |
| No religion | 204,965 | 225,416 | **90.9%** | **20,451** | 21.87% |
| Badimo | 53,439 | 60,613 | **88.2%** | 7,174 | 7.67% |
| everything else | 19,433 | 20,235 | 96.0% | 802 | 0.86% |
| TOTAL | 1,384,276 | 1,477,801 | 93.7% | 93,525 | 100% |

So the missing pair implies a composition of 69.6% Christian, 21.9% no religion, 7.7% Badimo,
against 79.9 / 14.8 / 3.9 drawn. Badimo is understated, exactly as `note_public` says. But no
religion is understated too, at 90.9% coverage, and it is the larger hole in people by nearly
three to one. **"More traditional than the country as a whole" is true and incomplete**: the
pair is *less Christian*, and both Badimo and irreligion run above the national rate there.
That is precisely what this note's own "irreligion rises as you leave town" thesis predicts
for a Kalahari-edge district, so the finding corroborates the argument rather than
complicating it. Left as the builder wrote it — the claim is not false and rewriting someone's
`note_public` paragraph is not a reviewer's call — but if it is ever revised, saying "less
Christian" instead of "more traditional" would be both more accurate and a stronger sentence.
It also means the headline 14.8% errs *low* against the true national 15.3%, so the
sub-Saharan superlative is conservative.

**`129,312` and `93.7%` are on different universes and sit one sentence apart.** 129,312 is
the all-ages 2011 census population of the two districts; 93.7% is a share of the 12-and-over
religion universe, where the shortfall is 93,525. A reader who divides gets 91.5%, not 93.7%.
Both figures are right and the note never actually equates them, but the juxtaposition invites
the arithmetic. Not changed.

**Checks run.** `check_md.py` clean; `built_countries.py --check` names nothing;
`check_rollup.py bw` reports 101,032 people (7.3%) as derived-and-orphaned, which is bw alone
above 2%. That is not a defect: here `derived` marks a *spatial* spread (a district residual or
a Delta village put across polygons), not an inferred category, so there is no coarser cell for
the tree to roll up to and no `COLUMNS` entry would help. `_bw_counts`'s docstring already says
the tier is spatial and explains why deriving it from the weight would be the failure mode.
Worth knowing that the measured-only view therefore drops 7.3% of Botswana for a reason the
check's wording does not describe.

**The join reconciles exactly.** Every one of the eighteen booklets' named localities sums to
its own printed district total to the person, the sole exception being Ghanzi's +210, which is
the CKGR row the printed total omits and which §3 already documents. One screenshot taken: dots
along the eastern corridor from Gaborone through Molepolole, Mahalapye and Serowe to
Francistown, Kalahari sparse, nothing in the sea, Badimo drawing visibly purple. Nothing that
needs a human eye.
