# Saint Lucia — 2022 census, via CSO's own provisional report

`sources/lc.py` -> `data/normalized/lc.csv`. Boundaries: `sources/lc_geo.py`, placement:
`sources/lc_grid.py`. Taxonomy: `taxonomy/lc2022.py`.

**171,834 people on 10 districts, 22 named categories, 95.89% of that universe drawn.** The
most Catholic country on this map — and the one whose census documents the fastest religious
change any source here can show: **92.4% Catholic in 1960, 50.6% in 2022**.

| | |
|---|---|
| counting geography | **district, 10 units** — 17,200 people each |
| placement | Kontur H3 r8 hexes, 772 of them, snapped to the coastline |
| basis | self-identification, census |
| tier | `measured` throughout |
| vintage | census 2022, provisional release 2 |
| universe | household population — CSO's own **weighted** estimate; see §3 |

---

## 1. THE DOCUMENT, AND THE TWO REVISIONS

CSO serves *Provisional Census Report 2022, Release 2, Rev 2.5* from its own WordPress media
library. The CARICOM mirror (`sources.md` §11v) serves **Rev 2.7**, which is newer than
anything the publisher's own listing offers — the inverse of §11v's Bahamas case, where the
mirror carried a filename and the publisher carried the file.

**It does not matter, and `lc.py` proves it rather than asserting it.** Both files are
downloaded and the drawn table's page is compared:

    OK  Table D.2 is textually identical in Rev 2.5 and Rev 2.7, so the revision
        this file reads cannot affect the map

The two revisions differ on 33 of ~150 pages — the employment chapter, and where the national
religion table breaks across a page — and on nothing this map reads. The publisher's own copy
is the one parsed.

**A third file is downloaded and it is the important one**: `St-Lucia-Census-2022.pdf`, the
Survey Solutions **questionnaire**, also on CSO's site. See §2.

## 2. THE `Mennonite` ROW IS THE FORM'S `Evangelical` OPTION

**3,760 people, 2.19% of Saint Lucia, filed under the wrong religion in the census's own
report.** This is the largest reading decision on any Caribbean source in this project, and
it is machine-checked on every run rather than argued in prose.

**The questionnaire.** CSO publishes the 2022 instrument — *St Lucia Census 2022, Version 4* —
and its question 1.5, *What is %rostertitle%'s religion/denomination?*, offers 22 options:

    1 Anglican          7 Hindu               13 Nazarene         19 Hinduism
    2 Baptist           8 Jehovah's Witnesses 14 Rastafarian      20 Atheist - Do not believe in God
    3 Bahai Faith       9 Methodist           15 Roman Catholic   21 None - No Religion but believe in God
    4 Brethren         10 Mormon              16 Salvation Army   22 Other
    5 Buddhism         11 Islam (Muslim)      17 Seventh-Day Adventist
    6 EVANGELICAL      12 Pentecostal         18 Universal Church

Table D.2's 23 rows are those 22 options **in the same order**, plus `Not reported`.
**Twenty-two of the twenty-three match one for one.** The one that does not is option 6:
the questionnaire says `Evangelical` and the report prints `Mennonite`.

**The 2010 census.** Its Table 40, *Census Population by Religious Affiliation*, gives a
share per decade back to 1960. `Evangelical` is **2.2%** in 2010 — the same share the 2022
report gives `Mennonite` — sitting in the same place in a similar list. **There is no
Mennonite row in 2010 and no Evangelical row in 2022.**

**The region.** Grenada's 2021 form has BOTH cells: `MENNONITE` 280 people (0.26%) beside
`EVANGELICAL` 2,553 (2.36%). That is what a real Mennonite count next to a real Evangelical
count looks like in the eastern Caribbean. Saint Lucia has one cell, at Evangelical's
magnitude, under Mennonite's name.

**And the geography fits.** 6.33% of Micoud, 5.07% of Laborie, 3.51% of Dennery — the rural
south and east — against 0.93% in Castries and **0.05% in Anse La Raye**. That is where
Caribbean evangelical churches grow. It is not what an Anabaptist settlement looks like
anywhere.

### What is done about it, and what is not

- **`lc.csv` carries the label the report prints.** A normalised file is a record of what
  the source says (§12).
- **`taxonomy/lc2022.py` resolves it to `christianity.evangelical`** — the node for an
  answer of *Evangelical* that names no body, which is `ke2019.py`'s call for Kenya and
  `gd2021.py`'s for Grenada's own cell. Not `christianity.protestant`, which would throw
  away the one thing the form establishes.
- **Nothing is moved between cells.** The magnitude is the source's; only the label is read
  differently, which is the distinction §9as drew for Italy: *a category can be wrong about
  its label and right about its magnitude.*
- **`lc.py` refuses to build if the shape of the discrepancy changes.** It asserts the option
  list verbatim, the position of the single disagreement, and that there is exactly one.

> **The rule, and it is new here.** §12 says never map a category on its string alone.
> This adds: **when a statistics office publishes its own questionnaire, the questionnaire
> is evidence about the table, and it is the only place a mislabelled column can be caught.**
> Nothing internal to Table D.2 is wrong — it sums, it reconciles, it agrees with the
> national table — and no amount of checking it against itself would ever have found this.

## 3. THE FIGURES ARE ALREADY WEIGHTED UP FOR UNDERCOUNT, AND THAT IS CSO'S DOING

The report says so in as many words, under *Reading these tables*:

> *"the Household Population is the sum of all people recorded on visitation records and/or
> electronic census questionnaires, **on which a set of geographically dependent weight
> factors has been applied to arrive at estimated full values**"*

and `Table 2` prints the factors:

| district | undercount | weight | estimated household population |
|---|---|---|---|
| **Laborie** | **33.7%** | **1.507** | 8,507 |
| Micoud | 30.9% | 1.446 | 16,693 |
| Choiseul | 25.3% | 1.338 | 7,122 |
| Castries | 25.2% | 1.337 | 60,614 |
| Vieux Fort | 22.4% | 1.288 | 19,669 |
| Canaries | 20.8% | 1.263 | 2,171 |
| Dennery | 19.2% | 1.238 | 12,943 |
| Gros Islet | 18.8% | 1.231 | 29,953 |
| Soufriere | 16.3% | 1.195 | 8,322 |
| **Anse La Raye** | **9.7%** | **1.107** | 5,841 |
| **Saint Lucia** | **23.3%** | **1.304** | **171,834** |

**This is the exact inverse of Barbados** (`bb.md` §3). BSS publishes the raw tabulable count,
warns that its parish tables are understated, and this project declines to scale them (§14.4).
CSO scales its own, per district, before publishing, and every table in the report holds the
scaled figure. **Neither map does any scaling; the difference is entirely on the publisher's
side of the line, and it has to be said out loud because the two countries' numbers look the
same kind of thing and are not.**

`lc.py` asserts Table 2's `Estimated Household Population` against Table D.2's district
totals, so the claim in this section is checked rather than quoted.

### The universe

    171,834   in private households   <- every table, Table D.2 included
      1,114   in institutions (194 hospitals, 684 prisons, 236 other)
    172,948   total RESIDENT population
      5,859   visitors in hotels and guesthouses
    178,807   total population on census night
    182,289   mid-year estimate for 2022, which the report contrasts with its own count

Nothing is scaled to any of the lower rows.

## 4. THE TABLE MISSES ITS OWN MARGINS, BY ONE TO THREE PEOPLE

Measured, printed on every run, and bounded:

- a category row's ten districts miss the row's own `Total` by **-2 to +2**;
- a district column's categories miss the column's own `Total` by **-3 to +3**;
- and **every drawn cell together is 171,829 against a published 171,834** — five people,
  0.003%.

Two-sided, single-digit, on weighted estimates that were each rounded on their own. That is
what independent rounding looks like and is nothing a parse error looks like: a dropped row
or a mis-assigned column would be out by hundreds. Cayman's tables (§9at) miss by one to five
for the same kind of reason; Grenada's (`gd.md`) reconcile to the person.

## 5. THE FORM ASKS ABOUT HINDUISM TWICE

Options **7 `Hindu`** and **19 `Hinduism`**, 253 and 66 people. Both are drawn, at one node,
and neither is dropped — 319 people, 0.19%.

`Hinduism` sits at the end of the option list beside the non-religious answers, where late
additions go, so the most likely history is that it was appended without anyone noticing
option 7. The two cells even have different geographies, which is what an arbitrary split of
one small population between two adjacent boxes produces. Merging them is not a judgement
about what respondents meant; it is the only reading under which the same word does not
appear twice in one legend.

## 6. THE BOUNDARIES: COD'S AREAS ARE WRONG AND ITS BOUNDARIES ARE NOT

COD-AB's `lca_admbnda_adm1_gov_2019` is the government's own file and its ADM1 is CSO's
district tier exactly — **10 polygons, 10 columns, joined 10/10 both ways**, with one name
differing by a hyphen (`Vieux-Fort` / `Vieux Fort`) and the pcode asserted from the boundary
side as an independent check on the pairing.

**But its district AREAS do not match the census's own.** CSO's `Table A.6` publishes a land
area per district, summing to the island's official 238.2 sq mi:

| district | COD | CSO Table A.6 | ratio |
|---|---|---|---|
| **Dennery** | 123.3 km² | 72.3 km² | **1.71x** |
| Vieux Fort | 57.9 | 49.7 | 1.16x |
| Anse La Raye | 37.5 | 37.6 | 1.00x |
| Choiseul | 25.2 | 25.9 | 0.97x |
| Laborie | 30.3 | 33.9 | 0.89x |
| Gros Islet | 86.1 | 100.2 | 0.86x |
| Castries | 87.4 | 102.6 | 0.85x |
| Micoud | 93.5 | 112.1 | 0.83x |
| Canaries | 19.3 | 24.3 | 0.79x |
| **Soufriere** | **42.8** | **58.5** | **0.73x** |
| SAINT LUCIA | 603.3 | 616.9 | 0.98x |

**In Cayman (§9at) a ratio table like that was a boundary error that moved dots.** Here it
is not, and three measurements say so:

1. **geoBoundaries** (gbOpen LCA ADM1, sourced from Wikimedia Commons, CC0) reproduces
   Table A.6 within **8.3%** on every district — so a set matching the census does exist.
2. **The two sets agree on 528 of COD's own 547 ADM2 settlements.** The nineteen that move
   are `Central Forest Reserve`, `Forest Reserve`, `La Sorciere` and hamlets on a shared
   edge.
3. **And they place the same people.** Summing Kontur's population grid inside each and
   comparing with the census district by district gives an rms deviation from 1 of **0.320
   under COD and 0.317 under geoBoundaries**. **Dennery holds 10,581 people under COD and
   10,571 under geoBoundaries** — despite COD's Dennery being 51 km² larger. The extra land
   is the Central Forest Reserve and nobody lives in it.

**COD is kept.** It is the government's own file, it is what every other country here uses,
and geoBoundaries has defects of its own: it spells Anse La Raye `Anse la Raya`, and it puts
the populated **Babonneau** settlements in Gros Islet where COD's government-sourced ADM2
puts them in Castries.

**OSM is not a third opinion**: it has no `admin_level=6` relations for Saint Lucia at all.

> **A disagreement about area is not a disagreement about people, and only one of the two
> can be checked cheaply.** The area table is what makes the problem visible; the population
> grid is what settles whether it matters. Run the second before acting on the first.

## 7. THE GRID DISAGREES WITH THE CENSUS AND BOTH DISAGREE WITH THE CORRECTION

Nationally Kontur and the census agree — 185,296 against a resident 172,948, ratio 1.07 —
and per district they run **0.58x in Laborie to 1.54x in Anse La Raye**. §6 establishes that
this is not the boundaries. It has a direction:

- **Kontur over-attributes to the built-up north-west** (Castries 1.44x, Anse La Raye 1.54x,
  Canaries 1.33x) and **under-attributes to the dispersed rural south** (Laborie 0.58x,
  Choiseul 0.67x). That is the standard bias of a building-footprint model.
- **And CSO's heaviest undercount corrections were in the same places Kontur sees fewest
  people.** Laborie was weighted 1.507 and Micoud 1.446, the two largest in the country; they
  come out at 0.58x and 0.80x here. **Two methods that share no inputs both find fewer people
  in rural Saint Lucia than the census asserts.** That is recorded, not resolved.

None of it changes how many dots a district gets (§9t): the counts are the census's, and only
the shape *within* a district comes from the grid.

## 8. A FINER GEOGRAPHY EXISTS AND CARRIES NO RELIGION

COD's ADM2 is **547 settlements**, ~310 people each, which would be among the finest tiers on
this map. Religion is published at the district and nowhere below it, so ADM2 would be a
placement layer with no counts to place — which is Kontur's job, and Kontur does it without
implying a counting tier the census does not have.

## 9. WHAT THE COUNTRY SHOWS

- **50.61% Roman Catholic**, the highest on this map, in an island Britain and France traded
  fourteen times and where the French church outlasted the British navy.
- **And the fastest fall this map can document.** 92.4% (1960), 90.5% (1970), 85.6% (1980),
  79.0% (1991), 67.5% (2001), 61.1% (2010), **50.6% (2022)** — from the census's own
  back-series. About half of what left went to two churches (Adventists 1.8% -> 10.8%,
  Pentecostals 0.0% -> 9.0%) and about half to no church.
- **The Catholic geography is a north-south divide**: 71.7% of Choiseul, 70.1% of Soufriere,
  66.1% of Canaries against 44.1% of Anse La Raye, 44.8% of Castries and 45.8% of Gros Islet
  — the north-west, where two thirds of the country now lives.
- **14.11% report no religion but belief in God and 0.30% report atheism** — a 47-fold gap,
  from a form that asked both. The non-affiliated mirror the Catholics: 16.2% of Castries
  against **4.8% of Choiseul**.
- **Anglicans are 1.28% here and 23.87% in Barbados**, and where they are is odd: Choiseul
  (3.71%) and Laborie (2.79%), in the Catholic south.
- **71% of the country's 310 Nazarenes are in Gros Islet** — one congregation, visible in a
  national census.
- **`Not reported` is 4.11% and it is urban**: 6.90% of Castries and 4.49% of Gros Islet
  against 0.41% of Canaries.

## 10. §14, briefly

Saint Lucia's census asks religion of everyone, publishes it by district, and the country has
no religious-minority protection issue this project needs to weigh. The only §14 question
here is §14.4, and it is answered in §3: the scaling was done by the publisher and is not
undone or extended.
