# The Bahamas — 2022 census, via BNSI's All-Island Report

`sources/bs.py` -> `data/normalized/bs.csv`. Boundaries: `sources/bs_geo.py`, placement:
`sources/bs_grid.py`. Taxonomy: `taxonomy/bs2022.py`.

**398,165 people on 18 islands, 24 named religions, 95.21% drawn.** The most Baptist country
on this map by a factor of four, the first census outside the United States to count African
Methodists under their own name, and the first here whose category list changes from unit to
unit.

| | |
|---|---|
| counting geography | **island, 18 units** — 22,000 people each, but see §6 |
| placement | Kontur H3 r8 hexes, 3,380 of them, snapped to the coastline |
| basis | self-identification, census, whole population |
| tier | `measured` throughout |
| vintage | census 2022 — the most recent census on this map |

---

## 1. Why it was still open, and how it was found

`sources.md` §11t swept Latin America and the Caribbean on 2026-09-07 and listed the Bahamas
as reporting religion for 2000, 2010 and 2022 — an open lead, not a build. The reason it
stayed a lead is worth keeping:

> **BNSI's own publications page does not list the report that has the data.**

`bnsistats.gov.bs/publications` lists four 2022 census releases. The one a search finds, and
the one every news story about Bahamian religion is written from, is the **First Release**
(February 2025), whose Table 6.0 is *Leading Religious Denominations: All Bahamas* — religion
by age and sex, **for the country as a whole**. Its preface says so outright:

> *"The graphs and tables presented in this report show data for the Total Population and are
> classified by Island, Supervisory Districts, Age Group and Sex. The other Census topics
> (such as Religion, Marital Status, etc.) are reported by Age group and Sex and are for All
> Bahamas."*

Read that and the country is national-only. It is not: a separate **579-page All-Island
Report**, published June 2026, carries `Table 12.1` to `12.18` — *Total Population by Sex,
Age-Group and Religion* for each island in turn. It is absent from the publications listing.

It was found on the **CARICOM regional mirror** (`sources.md` §11v) — and it is live on BNSI's
own CDN, which is what `bs.py` fetches. The mirror is how the file becomes findable, not where
it has to be cited from.

> **A publisher's own index is not a listing of what the publisher serves.** The same finding
> as Zimbabwe's WordPress media library (§11p) from the other direction: there the API was
> empty and the files were on a page, here the page is short and the file is on the CDN.

## 2. The one structural surprise: the category list is not fixed

**Only New Providence prints all 24 named bodies.** Every other island folds its smallest
answers into a single `OTHER RELIGION*` cell — and a starred footnote under each table names
exactly which ones went in.

```
  Ragged Island,  4 categories:   *"Other Religion" includes the following: Assemblies of
  God, Church of God, Church of God of Prophecy, Pentecostal, Roman Catholic, Other
  Christian Denomination (including non-denominational groups) and None.
```

So the parser cannot walk a fixed row sequence the way `sources/tt.py` does. It validates
each label against a known set instead, and reads the footnotes, so the collapsing is
**recorded rather than inferred**.

**What it costs is small nationally and not small locally.**

| | |
|---|---|
| the residual, nationally | **372 people, 0.093%** |
| islands with one | 17 of 18 |
| worst island | **Ragged Island, 30.4%** of its 56 people |
| second worst | **Mayaguana, 11.8%** of its 203 |
| islands whose footnote folds in `None` | **2** — Mayaguana and Ragged Island |

Those last two are the ones that matter: **on Mayaguana and Ragged Island the census reports
no irreligion at all**, because `None` went into the pooled cell. At most 41 people. Drawn as
`other.bs` and recorded in `taxonomy/bs2022.py` rather than corrected (§14.4). The alternative
was to drop the cell, which would delete a third of Ragged Island from the map rather than
misfile it.

> **A suppression that names what it suppressed is a better source than one that does not,
> and it is still a suppression.** Nothing else on this map publishes the contents of its own
> residual, per unit, in a footnote.

## 3. Four reconciliations, and the outer ones are cross-document

1. **The seven age bands sum to each cell's own TOTAL column** — 753 cells, exact. This is a
   check on *every figure read*, not on the margins, which is unusual and which is why it is
   listed first.
2. **Male + Female == TOTAL**, 251 cells, exact.
3. **The categories sum to each island's own TOTAL row**, 18 islands, exact.
4. **The 18 islands sum to the FIRST RELEASE's Table 6.0, category by category** — and the
   shortfall on each category is exactly what the island footnotes say was pooled. The
   national total is 398,165 both ways.

Number 4 is two separately published tabulations, in two documents fourteen months apart,
agreeing to the person. Nothing about the parse could produce that by accident.

**Two cross-document aliases were needed and both are the publisher's.** The first release
writes `Church of God (including Church of God of Prophecy)` where the All-Island Report
writes `... AND Church of God of Prophecy`, and it spells the atheist row **`Athiest`**. Held
in a two-entry table rather than solved by fuzzy matching — a near-match rule would happily
pair `Church of God` with `Church of God of Prophecy` if BNSI ever split them.

**And `JEHOVAH'S WITNESS` is printed with a curly apostrophe on five islands and a straight
one on three**, which is `tt.py`'s trap exactly. Everything is matched on uppercase
alphanumerics only.

## 4. The island codes are BNSI's own

The 2022 questionnaire (first release, p93) pre-fills `Name of Island` from a numbered list:

```
  1 New Providence   2 Grand Bahama   3 Abaco   4 Acklins   5 Andros   6 Berry Islands
  7 Bimini   8 Cat Island   9 Crooked Island   10 Eleuthera   11 Exuma and Cays
  12 Harbour Island   13 Inagua   14 Long Island   15 Mayaguana   16 Ragged Island
  17 San Salvador & Rum Cay   18 Spanish Wells
```

Tables 12.1–12.18 come in exactly that order, so **the table number is the census's island
code** and `geo_id` is not invented here. `bs.py` checks each table's printed caption against
the code rather than trusting the order.

## 5. The geography had to be built, and six cays decided it

**Nobody publishes the census's own tier.** COD-AB and geoBoundaries are the same geometry and
both give the **32 local-government districts** of the Local Government Act 1996. They nest
inside the 18 islands exactly, so the island tier is a dissolve rather than an approximation.

Twenty-six districts carry their island in the name. The six that do not were each checked
against BNSI's own publications rather than against a map:

| district | island | evidence |
|---|---|---|
| Hope Town, Grand Cay, Moore's Island | Abaco | named in the 2010 ABACO settlement sheets |
| Mangrove Cay | Andros | `ANDROS 2010 CENSUS REPORT` p106 settlement list |
| **Black Point** | **Exuma** | **enumeration district 420201** in `EXUMA AND CAYS POPULATION BY SETTLEMENT: 2010` |
| City of Freeport | Grand Bahama | — |

Black Point is the one that needed it: there is a Black Point settlement on Andros too, and a
settlement-name search returns both. The ED code settles it, because BNSI's own 2010
enumeration districts are prefixed by island.

**Harbour Island and Spanish Wells are census islands in their own right**, not part of
Eleuthera, though both sit a few kilometres off its northern tip and are reached from it.
BNSI numbers them 12 and 18. Where the census tier and the geographic intuition disagree, the
census wins — and §6 shows it was the right call by a mile.

## 6. Three quarters of the country is one polygon

**New Providence holds 296,732 of 398,165 people — 74.5%.** No other country on this map is
this concentrated in one counting unit. So the honest reading of the granularity line is not
"22,000 people per unit": it is *one unit holding three quarters of the country, and seventeen
small ones holding the rest*. What the map shows for most Bahamians is Nassau's composition
spread across Nassau by where people live.

**The seventeen Family Islands are where the geography is real, and they are not alike.**

| island | the thing about it |
|---|---|
| **Harbour Island** | **32.0% Roman Catholic, 1.2% Baptist** — the only place in the country where the national religion is a rounding error. 38.4% Church of God. |
| **Spanish Wells** | **23.2% Brethren** against 1.5% nationally — a sixteenfold concentration — plus 25.6% Methodist, 21.8% Pentecostal, and 1.0% Catholic. |
| Mayaguana | 73.9% Baptist |
| Long Island | 43.2% Anglican |
| Crooked Island | 29.7% Seventh Day Adventist |
| Acklins | **23.2% did not state a religion** — five times the national rate, unexplained |

**Harbour Island and Spanish Wells are three kilometres apart** and are religiously nothing
like each other or like the Bahamas. Spanish Wells is a Loyalist fishing settlement; Harbour
Island's Dunmore Town was the colonial capital. Both histories are still legible in the
answers, and both would have been erased by folding the two into Eleuthera.

## 7. What the country shows

* **34.13% Baptist — 135,875 people**, which is **35.84% of the population actually drawn**.
  Measured the same way across every country here that could plausibly compete, the next
  three are **Saint Vincent 9.29%, the United States 7.29% and Jamaica 6.90%** — so the
  Bahamas is nearly four times the runner-up. Being Caribbean and Anglophone does not predict
  this: Jamaica is 500 km away with a Baptist history at least as old.
* **Anglicanism is a Family Island religion**, not a Nassau one: 43.2% of Long Island, 34.9%
  of Inagua, 34.7% of the Berry Islands, against 11.2% of New Providence.
* **African Methodists get their own cell** — 1,028 people, and `christianity.methodist
  .african` had been reachable only from the U.S. Religion Census until now. 282 of them are
  on Eleuthera, 3.1% of that island against 0.02% of Grand Bahama.
* **`None` and `Atheist` are separate boxes** — 24,668 and 281. That is exactly
  `branches.py`'s distinction between `unaffiliated` and `secular`, and most censuses here
  collapse it.
* **Rastafari runs opposite to the stereotype**: highest on Cat Island (1.02%) and Andros
  (1.01%), lowest in Nassau (0.24%). The fourth census count of Rastafari on this map, after
  Jamaica, Saint Vincent and Trinidad — and the first where it is a rural answer.
* **The third largest Christian answer is a residual.** `Other Christian Denomination
  (including non-denominational groups)` is 35,278 people, 8.86%, ahead of the Roman Catholics
  at 8.72%. BNSI's own commentary reads this cell as the non-denominational answer overtaking
  Catholicism. It maps to the branch root `christianity` — see `bs2022.py`, which argues why
  `christianity.other` would be wrong.

## 8. Placement, and the 6% that missed the country

A plain `within` join of Kontur's 400 m grid leaves **836 hexes and 25,095 modelled people —
6.08% — outside every island**, because COD-AB's Bahamas geometry is generalised GDAMS 2009
and a hex is 400 m across. Measured before deciding what to do with it:

```
  <100 m from an island   303 hexes   12,606 people
  100-250 m               367 hexes   10,595 people
  250-500 m               163 hexes    1,891 people
  500 m and beyond          3 hexes        3 people
```

**Three people are genuinely offshore. The rest is the coast.** Nine of the twelve heaviest
unclaimed hexes are Nassau's own waterfront. In a country where the population *is* the
coastline, dropping that tilts every island's dots inland, so the hexes are snapped to the
nearest island within **1 km** — a threshold sitting in an empty gap in the measured
distribution rather than through the middle of it. The histogram prints on every run, so a
future vintage with a real offshore population would show up rather than be absorbed.

> **Generalised coastline against a fine grid is not the same failure as a border overrun,
> and needs the opposite treatment.** Trinidad (§9ak) drops its unclaimed hexes and should;
> its loss is small and its units are big. Here the loss is 6% and concentrated on exactly
> the settlements that matter.

**The per-island Kontur/census ratio runs 0.80x to 1.92x** around a national 1.036x. Exuma at
1.92x and Abaco at 1.32x are second homes and resorts read as population by a building-
footprint model — not a grouping error, since no island shows a compensating deficit. Only
the within-island shape is used, so no island gets the wrong number of dots (§9t).

## 9. One bug worth keeping, because the symptom was almost invisible

The snap above was first written with the surviving-hex mask as `unit != None`. `unit` is a
pandas Series whose unmatched entries are **NaN, not None**, so the comparison was true for
them and the two genuinely-offshore hexes survived with a null unit. Every printed number was
right — *"still outside after the snap: 0 hexes"* — and the only symptom anywhere downstream
was one line of `scatter.py`:

```
  1 units have polygons but no religion rows
```

`bs_grid.py` now asserts that the built layer carries exactly the units `bs_geo.py` made and
no others, which is the check that would have caught it at the source.

## 10. What is in the report and not used

Seven age bands and both sexes on every cell, read and used as checks but not drawn. The
other twenty-two per-island table families — age-sex, marital status, education, school
attendance, employment, housing, water and toilet facilities, internet access, agriculture.
And the 2010 per-island census reports, whose `Table 7.0` is the same religion table for the
previous census; they are on `stats.gov.bs` and would support a §3.4 change-over-time build,
which nothing here does.
