# Eswatini — 2017 Population and Housing Census, Volume 3

Central Statistical Office, Mbabane, December 2019. Drawn: **20 categories on 4 regions,
1,093,238 people, 97.81% of the census**. `sources/sz.py`, `sources/sz_geo.py`,
`sources/sz_grid.py`, `taxonomy/sz2017.py`.

The country is worth drawing for one cell. **`Zionists` are 367,290 people, 33.60% of
Eswatini**, which makes `christianity.africaninstituted` the plurality religion of a country
for the first time on this map, and with `Apostles` the node reaches 38.48%.

---

## 1. The office is dead and the census is on the government portal

`sources.md` §11w (2026-09-07) ranked Eswatini the most valuable undrawn African country on
the oracle's category depth and closed it as **"no reachable host"**. That was right about the
office and wrong about the country, and the difference is worth stating because it generalises.

What is true:

* `eswatinistats.org.sz` resolves (102.23.132.23) and times out on http and https alike. It
  is not a bot wall returning 418 ([[reference_dead_stats_office]]) and not a redirect; the
  port does not answer. A browser User-Agent changes nothing.
* `swazistats.org.sz` does not resolve at all.
* The Wayback CDX has **61 captures** across `eswatinistats.org.sz` and **not one PDF**. Fifty
  of the 61 are static assets of `ndcc.eswatinistats.org.sz`, a login-gated ASP.NET data
  portal (the National Data Coordination Centre), whose only reachable pages are
  `Account/Login`, `Account/Register` and `Account/ForgotPassword`.

**So there is no route through the statistics office, and there did not need to be.** The
census volumes are Joomla articles on `www.gov.sz`, the government portal, which answers
normally. The article is `2455-eswatini-census-documents`; it 404s today and its last good
capture is 2024-07-24. Its three links are:

    https://www.gov.sz/images/FinanceDocuments/Volume-3.pdf   3.0 MB, religion   <- drawn
    https://www.gov.sz/images/FinanceDocuments/Volume-5.pdf   1.8 MB, literacy, economic activity, youth
    https://www.gov.sz/images/FinanceDocuments/Volume-6.pdf   0.8 MB, albinism, disability, epilepsy

All three still serve from the live host even though the article listing them does not. There
is no Volume 1 or 4 at that path; **Volume 2, the Census Atlas, is somewhere else entirely**,
at `/images/planningministry/census2017.pdf` (20 MB), and `/images/planningministry/Volume-3-1.pdf`
is a byte-identical duplicate of Volume 3.

### The transferable part

**A dead statistics office is not a dead census, and the census volumes are as likely to be
filed under the ministry that paid for them as under the office that wrote them.** These sat
in the *finance* ministry's upload folder, which no amount of probing `eswatinistats.org.sz`
would ever have reached. The route that found them, and which cost about ten minutes:

1. `web.archive.org/cdx/search/cdx?url=gov.sz&matchType=domain&limit=60000&fl=original` —
   one GET, 5 MB, every URL the archive has ever seen on the whole government domain.
2. `grep -i census` over it. The livestock censuses are the noise; the signal is a single
   Joomla article id.
3. Fetch the article's last 200 capture with the `id_` suffix and read its hrefs.

Two notes on doing this again. The CDX endpoint **refuses `https://web.archive.org` over port
80** (connection refused, which reads as "the archive is down") and it answers a `filter=`
regex it does not like with **HTTP 500 and an empty body**, which reads as "no matches". Ask
for the unfiltered list and grep locally; it is one request either way.

## 2. Region is the CSO's ceiling, and four tables prove it rather than one

Religion by region is Table 3.2.4, thirteen Christian denominations on four regions. There is
nothing finer, and this was checked rather than assumed:

* **Volume 3's chapter 3 is the entire religion output of the census.** Five tables, four
  pages, and the chapter's own closing section is a recommendation for the next census.
* **Volume 2 is a Census Atlas of 37 maps, 29 of them at tinkhundla**, including population,
  density, growth rate, dependency, sanitation, water, television, orphanhood, unemployment,
  housing tenure, maize, cattle, goats and dogs. **No religion.** So the CSO holds religion at
  tinkhundla and chose not to publish it, which is Uganda's and Togo's shape (§11w).
* Volumes 5 and 6 have no religion table at all.
* OCHA's **COD-PS for Eswatini stops at ADM1** as well, so there is no external population
  file at tinkhundla either.

Four regions for 1.09M is ~273,000 each. Drawn under spec §3.9b, which sets no minimum unit
count: take the finest geography a country publishes and say what it therefore cannot show.

## 3. The regional table is Christians only, and the residual is still counted

This is the shape of the country and the thing to understand before changing anything.

| table | what it gives | geography |
|---|---|---|
| 3.2.1 | 9 top-level religions | **national only** |
| 3.2.2 | 13 Christian denominations | national only |
| 3.2.3 | the same 13 | urban / rural |
| **3.2.4** | **the same 13** | **4 regions — the drawn table** |
| 3.2.5 | 3.2.4 as percentages | 4 regions |
| 5.2.2 | population | 4 regions |

So a region's **Christians** are measured thirteen ways, and its **non-Christians** are a
single number: its Table 5.2.2 population minus its Table 3.2.4 Christian total. That number
is a count, not an estimate. What is carried down from the national table is only how it
splits across Islam, Hindu, Baha'i, Traditionalist, Judaism, Other, No religion and Not Stated.

**89.25% of the country is `measured` and 10.75% is `derived`**, and no magnitude is estimated
at a finer level than the source publishes it (§14.4 rule 1, `sources/kz.py`'s shape).

**The cost is concentrated in one cell and it should be stated plainly.** `No religion` is
80,861 people, 7.40%, and it is 69% of everything derived here. Its regional variation is
invisible on this map. Next door in Zimbabwe the same answer runs 4.5% to 13.5% across
provinces, a threefold spread, so the flatness drawn here is an artefact of what the CSO
published and not a finding about Eswatini.

### The rounding is controlled on both margins, on purpose

Allocating each region's residual independently by largest remainder reproduces every region
total and misses the national column by a person or two (`No religion` came out 80,860 against
a published 80,861). Nothing visible changes at that size; **the check does**. A column that no
longer equals the figure Table 3.2.1 prints cannot be asserted against it, and an assertion
given up is a class of error let through. `_allocate` therefore floors the 4x8 matrix and
hands out the shortfall subject to both margins at once.

## 4. Six tables are read and one is drawn, because Zimbabwe's warning applies exactly

Every identity inside Table 3.2.4 — the four regions summing to its Total column, the Total
column summing to 975,757 — **survives a consistent permutation of its four region columns**.
That is `sources/zw.py`'s lesson and it is why five other tables are parsed. The checks that a
column swap cannot survive are all cross-table:

* 3.2.4's `Total` column == 3.2.2's `Total` column, 13 rows. Different page.
* 3.2.3's urban + rural == 3.2.2's total, 13 rows. Different page.
* 3.2.2's 13 denominations == 3.2.1's `Christian` cell, 975,757.
* Male + Female == Total on every row of 3.2.1 and 3.2.2.
* 3.2.5's printed percentage reproduces 3.2.4 over the region's Christians, **52 cells**.

And the one that actually pins the columns, which crosses two chapters of the volume:

> **Each region's Christians over its Table 5.2.2 population must land near the national
> 89.25%.** The true pairing gives 89.36 / 89.77 / 89.05 / 88.43. `sz.py` enumerates all 24
> orderings of the four columns and asserts that **exactly one** lands inside two points of
> the national share. Four units is few enough to enumerate rather than sample, so this is an
> exact statement and not a p-value.

### The UNSD oracle as a second transcription

`tools/oracle.py Eswatini` returns twenty categories partitioning 1,093,238 exactly, forwarded
by the CSO to the Demographic Yearbook and merged (the 13 denominations, 7 non-Christian
answers, and `Other Christians` = the Christian `Other` + the Christian `Not Stated`). **Every
figure is asserted against it.** This is a third use of §0.5's oracle: not *does a table
exist* and not *how deep is it*, but **a second pair of eyes on the parse, keyed to a file
nobody in this project produced**. Cheap, and it would catch a whole page read from the wrong
place.

One caveat that matters for the taxonomy: **the oracle labels the Zionist cell `Zion Christian
Church` and the volume labels it `Zionists`, and the volume is right.** The ZCC proper is
Engenas Lekganyane's church at Moria in Limpopo; what Table 3.2.2 counts is the whole Swazi
Zionist stream, dozens of bodies. The UNSD label is a tidy-up made in transmission and reading
it as one denomination would be a real error. §2.4 — the source's own word is what goes in
`source_category`.

## 5. Kontur is wrong about this country

Full numbers and the argument are in `sources/sz_grid.py`'s docstring. The short form:

| region | census | Kontur | norm | WorldPop | norm |
|---|---:|---:|---:|---:|---:|
| Hhohho | 320,651 | 136,475 | **0.38** | 331,687 | 0.98 |
| Manzini | 355,945 | 281,632 | **0.71** | 386,326 | 1.02 |
| Shiselweni | 204,111 | 265,051 | 1.17 | 206,382 | 0.95 |
| Lubombo | 212,531 | 527,600 | **2.24** | 233,856 | 1.04 |

Kontur puts **43% of Eswatini in Lubombo**, which the census counts at 19%. Its fifteen
largest hexes are almost all in the northern Lowveld around Simunye, Mhlume and Tshaneni —
the sugar-estate belt, mapped building by building in OSM — while the Highveld *imiti*, the
dispersed homesteads most Swazis live in, are barely mapped at all. **Kontur is built from
building footprints, so it inherits where the mapping happened rather than where the people
are, and in a small country one mapping campaign is enough to tip a whole region.**

The boundaries were cleared before Kontur was blamed, three ways: Mbabane, Piggs Peak,
Manzini, Matsapha, Nhlangano, Hlatikulu, Siteki, Big Bend and Simunye each fall inside the
region they belong to; COD's four polygons reproduce the CSO's published areas to within 0.4%;
and an independent second spatial join reproduces the same four Kontur totals.

Placement is **WorldPop's constrained 100 m `maxar_v1` raster for 2020**, summed in 4x4 blocks
to ~370 m, which is Kontur's resolution. It is built from machine-extracted Maxar footprints
rather than volunteered mapping and does not have the blind spot. **A second WorldPop release
is carried as a control rather than as a second opinion**: the unconstrained 2017 raster is a
different model of the census's own year and agrees to within 0.037 on every region. Both are
checked on every run, and two independently built grids agreeing is what makes the weight
worth trusting — a single grid agreeing with the census could be luck, and Kontur shows what
disagreement looks like.

The constrained raster is drawn on and the 2017 one is not, deliberately: *constrained* means
WorldPop places nobody outside mapped built-up land, so no dots land in Malolotja, Hlane or
Mlawula. The better year loses to the better placement and the control keeps the choice
honest.

Two small things for whoever touches this next. WorldPop ships the **constrained** rasters as
**BigTIFF** (`II+\0`, version 43) and the unconstrained ones as classic TIFF (`II*\0`), so a
magic check that only accepts `*` rejects the file the country is built on. And `scatter.py`
still prints *"rows placed on Kontur hex population"* for Eswatini: that string is baked into
twelve weighter classes in `countries.py` and rewriting all twelve is not a surgical edit, so
it was left. It names the right builder file (`sources/sz_grid.py`) and it is the only place
in the project that says something untrue about this country.

## 6. Two things about the numbers that are not obvious

**COD-PS disagrees with the census about the baseline, and the census wins.** COD-PS 2022's
`Growth rates since last census` sheet gives a 2017 census population of 1,106,451 against the
CSO's published 1,093,238 — 13,213 more, spread across all four regions. The office's own
published figure is what is drawn, which is §9bn's Ecuador call made the same way.

**`Traditionalist` at 0.45% is a floor and a low one.** The box is exclusive of the Christian
ones, so it counts people who put nothing else first. Swazi ancestral practice — the
*emadloti*, consultation of an *inyanga* or *sangoma*, and the *Incwala* and *Umhlanga*
rituals the monarchy conducts annually — very commonly accompanies church membership rather
than replacing it, and the Zionist churches grew out of exactly that overlap while being
counted at seventy-five times the size. This is §11b's standing caveat for the whole continent
and Eswatini is one of its sharper cases.

## 7. What is not drawn

**23,925 people, 2.19%**, who gave no religion at all in Table 3.2.1. Excluded per §3.5 and
`tt2011.py`'s Trinidad precedent: a non-response is not an answer. It is the whole of the
country's `gap`.

Note this is a different cell from `Christian: Not Stated`, thirteen people who answered
Christian and named no denomination. Those thirteen **are** drawn, on `christianity.other`,
because an answer with a missing second level is still an answer.

## 8. Review pass — 2026-09-08

A second reader, not the builder. Nothing here changes the country; it is recorded so the next
reviewer does not repeat the checks.

**Checked and clean.** `check_md.py` clean; `built_countries.py --check` names nothing;
arithmetic in `note_public` reconciles against `data/normalized/sz.csv` at every figure
(33.60%, 38.48%, 7.40%, 2.19%, and the "half as large again" against `Evangelical`'s 22.71%).
The four plain-text fields carry no markup and no em dashes, and `note_public`'s seven bold
runs are four paragraph-break topic sentences plus three figures, which is the structure the
`countries.py` docstring prescribes rather than the listicle voice it warns about.

**`check_tiles.py sz` reports DIFFERENT and it is a flag mismatch, not a defect.** Run it as
`python tools/check_tiles.py sz --no-atomic`: 0 missing, and the only "unexpected" features
are `zw`, `ke`, `ao` and `ru` dots sharing the six world-spanning tiles at z0 to z5. `zw`
produces the identical six-tile failure, with `sz` dots as its unexpected ones. Without
`--no-atomic` the reference invents an `atomic` layer the step-11 archive never had.

**Geometry sane.** 1,059 dots, bbox 30.83 to 32.07 E and -27.31 to -25.75 S, inside the
country with no spill into South Africa or Mozambique; a 6x6 histogram puts the mass on the
Mbabane-Manzini corridor rather than flat across the four regions, which is what the WorldPop
switch was for. 90 of 1,059 dots are tier 1, matching the 8.7% `check_rollup.py` reports.

**`rings_sz.geojson` is empty and that is required, not a miss.** `Bahai Faith`'s REVIEW entry
says the cell "may ring (§4.3)"; it cannot. All eight non-Christian cells are `derived`, and
spec §4.3 bars an allocated count from a ring outright, so `bahai`, `hinduism` and `judaism`
draw nothing at 1:1,000 and correctly leave nothing behind. The REVIEW wording is the only
inaccuracy found in the country and it is in a comment, not in output.

**`COLUMNS = {}` is the right answer to `check_rollup.py`.** The tool asks whether the source
published a cell holding the orphaned 93,556 at the drawn unit. It did not: what Table 5.2.2
minus Table 3.2.4 leaves is "not a Christian", which is not a religion and has no node. The
module says so in a comment already. Left alone.

**Precedent swept across all 97 `MAP` dicts, and the two claims that assert a convention both
hold.** `Christian: Seventh Day Adventist` to the bare `christianity.adventist`: 46 non-US
census cells do this and `christianity.adventist.sda` is used by `usrc2020.py` alone, plus
`ca2021.py`'s `LEAF`, so "the convention every non-US mapping follows" is accurate.
`Christian: Methodist` to the bare parent: 42 files agree. `Zionists` and `Apostles` to
`christianity.africaninstituted` agrees with `zw2022.py` and `bj2013.py`; the Pacific and Irish
`Apostolic` cells that went to `christianity.pentecostal` (`ck2011`, `fj2007`, `vu2020`,
`ie2022`) are all non-African and are a geography split rather than a contradiction.

**One corpus-wide inconsistency the sweep surfaced, which is NOT Eswatini's.** Identical
denomination strings are split between a Holiness parent and its child across the map.
`Nazarene` goes to `christianity.holiness.nazarene` in `bb2010`, `lc2022`, `py2002` and here,
and to the bare `christianity.holiness` in `bz2022`, `nz2023`, `ph2020`, `au2021` and `ws2021`.
`Salvation Army` is worse: the child in seven files, the parent in five, and
`christianity.methodist.holiness` in `vc2012`. Eswatini took the child and gave the strongest
reason any of the twelve gives, so it is on the better side of the split and nothing should
change here. Recorded because harmonising it would touch a dozen already-drawn countries and
is a legend decision rather than a mapping one; whoever next files a Holiness cell should raise
it rather than adding a thirteenth precedent.

No ask filed. Nothing rebuilt, nothing re-scattered.
