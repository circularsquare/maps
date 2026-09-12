# Timor-Leste — Population and Housing Census 2022

Drawn 2026-09-08. 1,248,705 people on 14 municipalities, 8 nodes.
Modules: `sources/tl.py`, `sources/tl_geo.py`, `sources/tl_grid.py`, `taxonomy/tl2022.py`.
See also `sources.md` §9cj.

---

## 1. Where the religion question is published, and where it is not

INETL (Instituto Nacional de Estatística de Timor-Leste, I.P.) asked religion in 2004, 2010,
2015 and 2022. The publications are not parallel:

| census | national | subnational |
|---|---|---|
| 2004 | UNSD table 28, 7 categories | not found |
| 2010 | reprinted in the yearbooks | municipality, in the yearbooks; posto for Viqueque |
| 2015 | Volume 2 table 11 (`3_2015-V2-Nationality-Citizenship-Religion.xls`) | **municipality, in that table** |
| 2022 | main report table 4.07 | **municipality, only in the thirteen yearbooks** |

**The 2022 main report has no religion geography at all.** `Final-Main-Report_TLPHC-Census_WEB.pdf`
gives religion a numbered section (3.2.2) and one table, 4.07, which is religion by five-year
age group and sex for the country. None of the 24 tables in chapter 4 crosses religion with
municipality, administrative post or suco, and none of the eight thematic reports (gender,
disability, children and youth, education, migration, mortality, fertility, labour) does either.
A session that read the main report and stopped would close Timor-Leste as national-only, and
the queue row that priced it at "6 categories" from the oracle would look right.

**What carries the geography is `<Municipality> em Números`.** Each of the thirteen Serviços de
Estatística Municipais publishes an annual volume of its own, and the 2022 edition of every one
of them prints, under *Proteção Social*, a table headed `Distribuição População por Religião`
with 2010, 2015 and 2022 side by side, split male / female / total. Thirteen volumes, one table
each, at `inetl-ip.gov.tl/wp-content/uploads/<yyyy>/<mm>/<Name>-em-Numeros-2022.pdf`. They were
issued between June 2025 and March 2026, three years after the count, by a different directorate
from the one that wrote the main report.

This is Botswana's shape (§9bu) and Rwanda's (§9cb) for the third time: the national report stops
at the nation, and a per-unit booklet series filed away from the report set carries the fine
geography.

## 2. Atauro, the fourteenth municipality, which has no volume

The 2022 census tabulates **fourteen** municipalities. Atauro, the island north of Dili, was a
posto of the capital until 2022 and is listed separately in table 4.03 at 10,295 people. No
`em Números` was ever written for it, and the Dili volume's 2022 column excludes it.

So its religion row is **table 4.07 minus the thirteen volumes**. That is subtraction between two
published tables rather than a model, and four things say the residual really is Atauro:

* it is 9,622 people, which is **0.935** of Atauro's census population where the country's
  religion universe is 0.931 of its own;
* every category comes out non-negative;
* Islam, Buddhism and indigenous religion come out at **exactly zero**, without being told to;
* **5,332 of the 9,622 are Protestant**, 55.4%, which is the island's known signature and is
  nothing like any of the thirteen (Aileu, the next highest, is 7.9%).

It is drawn `derived`, so it renders desaturated. The arithmetic is exact, but no publication
anywhere prints the numbers, and a reader who wants to check Atauro against a source cannot.
`countries.py` `_tl_counts` says the same thing in its docstring.

## 3. Why the map stops at the municipality

Four routes to a finer tier were tried and all four are closed.

* **2015 Volume 4, the Suco Reports.** Twelve worksheets at suco level (population by age group,
  school attendance, level of education, main economic activity, and seven housing
  characteristics). No religion. Volume 2, which has religion, is municipality only.
* **The 452 Sensu Fó Fila Fali suco reports.** Launched nationally on 26 July 2024 by the Prime
  Minister and the Minister of Finance, one report per suco, and INETL's own news item about the
  launch is on the site. The reports themselves are not: the WordPress media library holds 735
  documents and none of them is one, and the site's search finds only the news item.
* **The REDATAM population dashboard.** INETL's menu and its census pages both link
  `http://20.6.104.113/redatam/`. The host answers on **no port at all** (80, 443, 8080, 8083,
  8000 all time out), so this is not a bot wall or an SPA, it is off. Worth retrying: a REDATAM
  server over the 2022 microdata would give religion by suco directly.
* **The other editions of the yearbooks.** The 2021, 2023 and 2024 editions of all thirteen were
  downloaded and scanned. Only **Viqueque** ever prints religion below the municipality: its 2022
  edition gives 2010 and 2015 by administrative post, five units, seven categories, and 2022 by
  municipality on the next page. The Covalima volume looks like a second case and is not, because
  the `Posto Administrativo` table on the same page as its religion table is Bolsa da Mãe
  recipients; the two are told apart on column count and not on the heading.

Fourteen units for 1.25M people is 89,000 each, which is finer per person than Georgia's regions
and Armenia's marzes, both drawn.

## 4. The volumes are typed by hand, and the 2015 column proves it

These are not extracts of a central file. The thirteen disagree about section numbering (2.4.1
or 2.5.1), about the header language (`Religião` or `Religious`), about the last two category
labels (`No religion` / `No answer`, or Baucau's Tetum `Laiha Rligiaun` / `La hatan`), about
whether a `Total` row is printed at all, and Ermera heads its 2022 column **`Censo 2020`**.

But every one of them reprints 2015, and 2015 was published centrally. So `sources/tl.py` checks
**all 252 cells** of the 2015 column, male, female and total, against Volume 2 table 11.
**Four are wrong, in three volumes, and they are two different mistakes:**

| volume | cell | prints | workbook says |
|---|---|---|---|
| Ainaro | 2015 Catholic total | 62,988 | 62,388 |
| Manatuto | 2015 `Seluk` total | 97 | 93 |
| Oecusse | 2015 Catholic female | 31,962 | 33,962 |
| Oecusse | 2015 Catholic total | 66,402 | 68,402 |

Ainaro's is the municipality's whole population pasted into the Catholic row; Manatuto's is four
people over; in both cases **the male and female cells beside them are exactly right**. That is
why `read_volume` refuses any 2022 row whose sexes do not sum to its own total. Oecusse's is the
other kind, a mistyped female cell carried into the total, so the row is internally consistent
and only the workbook catches it.

A 1.6% cell error rate is the calibration the 2022 column needed, and it is why that column is
reconciled rather than trusted.

## 5. The reconciliations

Three, and the last two are the ones that matter.

1. **Internal.** Every 2022 row's male + female = its own total; where a volume prints a `Total`
   row, the categories sum to it.
2. **Against table 4.07, category by category.** The thirteen volumes plus the Atauro residual
   reproduce all nine of the main report's national figures and its 1,248,705 total exactly.
   Nothing in the volumes could have been copied from 4.07, which was published two years before
   most of them and never at this geography.
3. **Against UNSD Demographic Yearbook table 28**, which is INETL's own return to New York and a
   different publication from anything else read here. All seven 2015 figures reproduce from
   Volume 2's thirteen municipalities and all five named 2022 figures from table 4.07.

**Ask the oracle by name.** `tools/oracle.py` matches UNSD's country string, so `oracle.py tl`
comes back as a miss that reads like an absence; `oracle.py "Timor-Leste"` returns 2004, 2015 and
2022. The `fm` review found this on 2026-09-08.

**And the oracle's 2022 row is a witness, not a source.** It names five categories and puts
everything else in one `Other` cell of **95,328** against a total of **1,341,737**. That total is
the resident population, not the religion universe (1,248,705, the population aged 3 and over in
private households), so the residual is mostly the under-threes. Drawing the oracle's row would
put 7.1% of Timor-Leste into an unclassified cell that does not exist.

## 6. Geography

COD-AB `tls_admin_boundaries` (HDX, `valid_on 2020-09-11`) has 13 ADM1, 65 ADM2 and 442 ADM3.
It is one municipality behind the census, and **the fix is a tier down inside the same file**:
Atauro is already there as ADM2 `TL0604` at 139.91 km² against the census's 140.55, so
`tl_geo.py` subtracts that polygon from Dili's ADM1 and adds it as the fourteenth unit. All
fourteen come within 1.8% of table 4.03's own area column, and the subtraction is asserted to
have removed exactly one connected piece of Dili.

**Nothing joins on a name anywhere in this country.** `sources/tl.py` writes the p-code into
`geo_id` and `tl_geo.py` builds the polygons on the same p-codes, so the join is an identity and
`[[reference_name_join_wrong_neighbour]]` does not apply. The one place a name join would have
been tempting is the 2015 workbook, which prints `SAR1 OF OECUSSE` and `LIQUIÇA` where COD-AB has
`Oecussi` and `Liquiçá`; those are handled by a hand-written table in `PRINTED_2015` and are used
only for the check.

Placement is Kontur's November 2023 400 m grid (`kontur_population_TL_20231101`), 10,442 hexes
inside the country after 402 are dropped as the extract's overrun into Indonesian West Timor.
Kontur holds 1,337,005 people against the census's 1,341,737, a ratio of 0.996, and its per-unit
shares are within 1.2 points of the census's everywhere.

## 7. The §3.5 lean check

Two holes, both tested against every drawn category across the fourteen municipalities, with a
20,000-draw permutation test ([[reference_check_needs_power]]).

**`No answer`, 239 people, 0.019% of the universe.** Highest in Dili at 0.038% and lowest in
Lautém at 0.005%. It correlates **+0.776** with the Hindu share (p = 0.006) and with nothing else
that survives; both of those are Dili and so is the hole, so what the check has found is that
non-response is urban, not that it is Hindu. At 239 people it cannot move a drawn share.

**One correlation was dropped on the leave-one-out**, which spec §12 asks for at this many units
and which matters more here than at Cabo Verde's 22. `No answer` against the Buddhist share is
+0.570 over all fourteen and **−0.201 without Dili**, so it is Dili and nothing else and is not
quoted. The Hindu one survives: recomputed fourteen times, it runs +0.454 (dropping Dili) to
+0.914 (dropping Baucau) and never changes sign.

**The people the question never reached, 93,032, 6.93% of the population.** Under 3, or outside
private households. It runs 6.52% in Dili to 7.66% in Aileu, so the whole spread is 1.1 points.
Its only significant correlation is −0.488 with the Hindu share (p = 0.045), which is the same
Dili effect with the sign flipped, because Dili has the fewest children and the most Hindus.
Catholicism is +0.300 (p = 0.37) and not significant.

**Neither hole leans in a way that changes anything drawn**, and `note_public` says so with the
figures rather than only with the sizes.

## 8. What is worth coming back for

* **The REDATAM host.** If `20.6.104.113` ever comes up, it is religion by suco.
* **The 452 suco reports.** They exist on paper in every suco in the country. If INETL ever puts
  them online, or if a copy surfaces, that is 452 units.
* **A 2027 edition of the yearbooks.** Viqueque already publishes religion by administrative
  post for the older censuses; if any municipality does it for 2022, the tier moves.

---

## 9. Review, 2026-09-08

A second pass, not the builder. Everything below was recomputed from the PDFs, the workbooks and
the built data rather than read off §1-§8.

### The Atauro subtraction reproduces, off a different parser

The 2022 column was re-extracted from the thirteen volumes by a second reader that walks the
page's text stream in reading order and takes the numeric run after each label, rather than
clustering words on `y0` the way `sources/tl.py` does. **116 of the 117 cells in
`data/normalized/tl.csv` come back identical.** The one it could not read is Aileu's `No religion`,
whose label is the bare word `No` in the text layer; read by hand off the raw stream it is
`18 | 17 | 35`, which is what the file has. Table 4.07 and table 4.03 were re-read out of
`tl_2022_ch4_basic.xlsx` and match `NATIONAL_2022`, `UNIVERSE_2022` and `ATAURO_POPULATION_2022`
to the person, and 4.07's nine categories sum to its own 1,248,705.

So the residual is confirmed: thirteen volumes 1,239,083, national 1,248,705, **Atauro 9,622** as
Catholicism 4,283, Protestantism 5,332, Hinduism 1, Other 1, `No religion` 3, `No answer` 2, with
Islam, Buddhism and indigenous religion at zero. 55.41% Protestant, 44.51% Catholic.

**A fifth witness the record does not use, and it is tighter than the band.** The 2022 religion
universe as a share of table 4.03's population runs 0.9228 (Aileu) to 0.9348 (Dili) across the
thirteen. Atauro's residual is **0.9346**, inside that range and next to Dili, which is where a
municipality with the country's fewest children should sit. `ATAURO_BAND` at 0.85-1.00 is a much
looser test than the volumes themselves offer.

### What the add-up guard catches in 2022, and what only the band catches

Ainaro's and Manatuto's 2015 errors are total cells with correct sexes beside them, so
`read_volume`'s `m + f != t` would refuse a 2022 row of that shape. **Oecusse's shape is not
caught and nothing at municipality level can catch it**, because a mistyped sex cell carried into
its own total leaves the row internally consistent and there is no 2022 workbook to check against.

What does catch it is the residual, and it is worth knowing how much slack there is. The residual
must land in 8,751-10,295 people and it is 9,622, so **the thirteen volumes may be wrong by +673
or -871 people in aggregate before the build refuses to write**. An Oecusse-sized error, 2,000
people, breaks it (share 0.740). An error under about 700 people does not, and it would move that
many people silently between a volume and Atauro without changing any national total. That is the
residual's real cost and it is small; nothing drawn depends on the 2015 column at all.

### The two polygons are clean

Recomputed off `data/geo/tl/tl_municipalities.gpkg`: **worst pairwise overlap across all fourteen
units is 0.0 km²**, and the union of the fourteen equals the union of COD-AB's thirteen ADM1 to
0.0 km² in *both* directions. So the ADM2 polygon is vertex-identical to its share of the ADM1
one, and the difference left neither a sliver nor a gap.

Dili has two parts after the subtraction and the second is 0.0046 km². It is not an artefact: it
lies inside Dili's own latitude band, which ends at -8.485, while Atauro spans -8.309 to -8.127.
It is a COD-AB islet off the mainland that was there before.

All 111 Kontur hexes assigned to `TL0604` are on the island, and 10 drawn dots fall inside the
Atauro polygon. One Catholic dot in Viqueque sits 15 m off the coastline, which is a hex
straddling the shore and nothing else.

### The lean check reproduces exactly

`No answer` against the Hindu share **+0.776**, permutation p = 0.004 on 20,000 draws; leave-one-out
**+0.454 (drop Dili) to +0.914 (drop Baucau)**, never changing sign. Against the Buddhist share
+0.570 and **-0.201 without Dili**, so dropping it was right. The never-reached hole runs 6.516%
(Dili) to 7.660% (Aileu), correlates -0.488 with the Hindu share (p = 0.044) and +0.300 with the
Catholic one (p = 0.38). `gap_share` recomputes to 0.06952 against the 0.06951 in the entry.

**The "non-response is urban" reading holds and is the honest one.** `No answer` correlates
positively with *every* small category at once, Islam +0.439, `No religion` +0.443, Buddhism
+0.570, Hinduism +0.776, which is the signature of a single urban unit rather than of any
religion; and with Dili removed the Hindu correlation falls to +0.454, which at thirteen units is
not significant. 77% of the country's Hindus and 48% of its non-response are both in Dili.

### Two reader-facing figures were wrong, and they had one cause — FIXED

Both were computed off `counts.json`'s **dot** counts rather than off people, and a dot is 1,000
people floored per node.

* *"ahead of Paraguay at 90.8%"*. Paraguay is 3,488 of 3,843 dots, which is 90.76%, but
  3,488,086 of 3,855,397 people, which is **90.47%**. Corrected to 90.5% in `countries.py`,
  `taxonomy/tl2022.py` and `sources.md` §9cj. The Poland figure and Timor-Leste's own 97.5% are
  both right.
* *"Only the Philippines at 0.040% and Myanmar at 0.058% are lower"*. **Kiribati is lower**, at
  0.046%, and Myanmar is 0.060% and not 0.058%. Timor-Leste's 0.064% is fourth lowest of the 84
  drawn countries that record any irreligious count, with Samoa at 0.064% just above it. Kiribati
  was invisible to a dots-based scan because its 51 irreligious people are less than one dot.
  Corrected in `countries.py` and `taxonomy/tl2022.py`; the generalisation is in spec §12.

Everything else in `note_public` checks out against the built data: Covalima 99.802%, twelve units
above 96%, Aileu second at 7.89% Protestant with Manufahi third at 3.38%, Atauro the only unit
under 90% Catholic, Islam 3,202 with 1,964 in Dili and 363 in Lautém, indigenous 240 at 0.019%,
`No religion` 797 at 0.064%. `counts.json` was one edit behind `countries.py` on the note and was
brought back in step with `tiles.py --refresh-meta`.
