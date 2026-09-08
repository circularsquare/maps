# Guatemala — LAPOP AmericasBarometer, waves 2010–2023

Built 2026-09-08. **The first country on this map drawn from the AmericasBarometer.**
`sources.md` §11ad is the assessment of LAPOP as a source across all nine countries it could
serve; this file is Guatemala's own record. Read §11ad first if you are about to do a second
one — most of the traps are the source's, not Guatemala's.

## Why a survey at all

`sources.md` §11x closed Guatemala on two independent witnesses: the UNSD oracle reports
**1964** as the only Guatemalan religion tabulation ever forwarded to it, and IPUMS's
`RELIGION` variable says Guatemala 1964 as well. INE's own site (`www.ine.gob.gt`) returns a
**Radware captcha page with HTTP 200 on every path**, and `censopoblacion.gt` — the 2018
census microsite — resolves, returns 200, and is a **parked domain** whose
`<meta name="description">` reads *"This domain may be for sale!"*. Neither of those is a
reason to think the question exists; §11x's point is that a 200 is not a page.

None of that is reopened here. This is a survey standing where a census would be, and every
row it produces is `modelled` in spec §7's sense.

## The files

| file | what it does |
|---|---|
| `sources/lapop.py` | the construction, the split-half, the held-out check and the category list, shared with `sv.py` and every LAPOP country after it |
| `sources/gt.py` | LAPOP `.dta` → `data/normalized/gt.csv`. This country's decode, its `CARRIES` and its one `OVERRIDE`. |
| `sources/gt_geo.py` | COD-AB ADM1 → `data/geo/gt/gt_departamentos.gpkg` + `gt_lookup.csv`, and COD-PS 2024 populations joined on the pcode. |
| `sources/gt_grid.py` | Kontur 400 m hexes → `data/geo/gt/gt_hexes.gpkg`, the placement layer. |
| `taxonomy/gt2023.py` | the eleven answers → the tree. Named for the last wave in the pool, per the registry convention. |

## Getting the data

**The `.dta` is a manual download and is not fetched by any script here.** LAPOP's Grand Merge
sits behind a click-through on `lapopsurveys.org` — free, no institutional affiliation, a name
and an email address. Ask for:

```
Grand_Merge_2004-2023_LAPOP_AmericasBarometer_v1.0_FREE.dta.zip     62 MB
```

and unzip it to `data/raw/lapop/` (1.12 GB open, 301,156 rows × 1,408 columns). Then:

```
python sources/gt.py --fetch          # slims it to 10 columns, ~1 min, writes a 2.9 MB feather
python sources/gt_geo.py --fetch      # 2.9 MB from HDX, plus a 7 KB CSV
python sources/gt_grid.py --fetch     # 5.7 MB from Kontur
python sources/gt.py
```

**Do not read the whole `.dta`.** `pyreadstat.read_dta(..., usecols=[...])` takes about a
minute for ten columns; a full read costs several GB of RAM for 1,398 columns nobody wants.
`metadataonly=True` gets you the value-label sets in a second, which is how the category list
in `gt2023.py` was transcribed.

## The construction

    share      LAPOP q3c, six waves pooled, weight1500, by prov     8,919 respondents
    magnitude  OCHA COD-PS 2024 by ADM1 pcode                       17,843,132 people
    output     religion x departamento, 22 units, tier `modelled`

**No magnitude is invented.** Every person drawn is a person COD-PS counts in that
department; the survey only decides the column. Same shape as `sources/kz.py`, and it is
spec §14.4 rule 1 holding by identity.

**The universe mismatch is real and is carried rather than fixed.** LAPOP interviews adults
18 and over; the shares are applied to the whole population. Drawing only the adults would
leave about 45% of a very young country blank, and §6.12 is about how badly a blank reads on
a dot map. `basis` says *adults 18 and over* so the reader knows whose answers these are.

## Which categories carry a geography, and the test that decides it

**A size threshold was the first answer and it was wrong.** The first version of `gt.py` cut
at 4% of the country, on §11ad's finding that this instrument's provincial cut fails below
about 1% of a province. That would have drawn `Protestante Tradicional` (5.43% national) on
its own department shares.

Its split-half rank correlation across the six waves is **−0.04**.

So the criterion is stability, not size. Rank the 22 departments on 2010–2014, rank them
again on 2016–2023, correlate. The bar is what it takes to be distinguishable from zero at
95% on 22 units, which is 1.96/√21 = **+0.43**:

| category | national | spearman | pearson | |
|---|---:|---:|---:|---|
| Católico | 51.97% | **+0.57** | +0.66 | own geography |
| Evangélica y Pentecostal | 34.52% | **+0.50** | +0.62 | own geography |
| Protestante Tradicional | 5.43% | **−0.04** | −0.12 | national rate |
| Ninguna (creyente) | 4.85% | **+0.21** | +0.18 | **own geography, UNDER THE BAR — Anita's call** |
| Otro | 1.14% | undefined | | national rate |
| the six under 1% | | | | national rate (§11ad) |

### `Ninguna (creyente)` IS DRAWN UNDER THE BAR — Anita, 2026-09-08

This file's first version spread it flat and flagged the call as hers. She took it: **draw it.**

**The split-half answers a narrower question than the decision needs.** It asks whether the
ORDERING replicates and returns +0.21 against a +0.43 bar. It never asks whether the
departments differ at all — and they do, decisively: **chi-square p = 3.6e-16** across the 22.
So +0.21 means the ordering is not pinned, not that the variation is fake. That is spec
§14.16's China exactly (Protestantism, rank stability +0.17, spatial p = 1.3e-84, drawn on
Anita's call with the weakness named), and drawing this keeps the two countries consistent.

**The stable half is also the legible half.** The capital tops both wave halves — 8.2% then
8.6%, on 188 respondents — and four of the five Maya highland departments sit stably at the
bottom: Quiché 0.9→1.3, Alta Verapaz 1.5→1.8, Huehuetenango 1.5→2.4, Sololá 2.2→2.1. It is
the middle eighteen that shuffle (Chiquimula 8.9→2.3, Sacatepéquez 0.0→7.9), and
`note_public` tells the reader to trust the two ends and not the middle.

**Flat was not the neutral option.** At 4.85% everywhere it asserted that Quiché and Guatemala
City have identical shares, which nothing believes. Switching moved **194,002 people**, 22% of
the category: the capital 178,300 → 307,597, Quiché 55,137 → 12,123.

**The bar was NOT moved.** `sources/gt.py`'s `OVERRIDE` names the one category, carries the
reason, and prints it on every run. Moving the bar would have silently redrawn every other
category in every LAPOP country; naming a category does not.

**A category that fails is still DRAWN.** Its people go at the national rate inside that
department's own residual, so the partition stays closed and nobody is deleted. What is
withdrawn is the claim to know where they are.

### And the tail is a residual, not a flat rate

An earlier version renormalised the drawn categories to leave a **fixed** national tail in
every department. That forces their combined share to be the same everywhere, which the data
contradicts (the residual runs 1.6% of Chiquimula to 14.2% of Suchitepéquez). The three drawn
shares now pass through untouched and the eight remaining categories divide each department's
own residual at their national relative proportions.

## The join, and why it is checked twice for something this easy

Guatemala's 22 departments have carried the same official numbering since the nineteenth
century and all three sides agree on it:

    LAPOP prov   201..222  ->  department 01..22
    COD-AB       GT01..GT22
    COD-PS       GT01..GT22

`gt_geo.py` requires the **code** join and the **name** join to succeed independently, on all
22, and refuses to run otherwise. `sources/ni_geo.py` is why: Nicaragua's code join matched
145 of 153 and silently sent Waspám's Moravians inland, and **a permutation preserves every
total**, so no arithmetic check can find one.

### The held-out check, and the one that had to be withdrawn — CORRECTED 2026-09-08

§11ad validated LAPOP's provincial cut against censuses in Mexico, Peru and Suriname. That is
evidence about the instrument, not about this join, so `lapop.held_out()` runs the local
version.

**The test is the permutation, not the correlation.** LAPOP's weighted department
distribution tracks COD-PS's population distribution at **r = +0.965**, and **none of 20,000
random pairings of the same 22 departments reaches it** — the best random pairing manages
+0.937, which is high, and is exactly why the raw correlation on its own would not have been
evidence. El Progreso is sampled at 1.65× its population share and Zacapa at 0.29×, which is
the design rather than an error.

> **THIS FILE ORIGINALLY CLAIMED A SECOND WITNESS AND IT WAS NOT ONE.** It reported
> *"mean adult age, LAPOP vs COD-PS's own 5-year bands: r = +0.52"* and said neither check
> would survive a permuted `prov`. The age comparison was then measured properly, when
> El Salvador failed it at r = −0.11 and the question became whether that was a broken join
> or a useless test:
>
>     Guatemala     between-unit variance 0.887  vs  mean sampling variance 1.013   F = 0.88
>     El Salvador                         0.218                            0.609    F = 0.36
>
> **F below 1 means the departments' mean ages are indistinguishable from that many draws on
> one distribution**, so the test cannot tell a good decode from a permuted one in either
> country. Guatemala's +0.52 was luck; El Salvador's −0.11 was the same non-result with the
> other sign, and failing a country on it would have been a false alarm.
>
> The comparison is still printed, with its F beside it, and it decides nothing. The general
> lesson is the one spec §3.10d already states in a different register: **a check that has
> never been shown to have power is not a check.** Measure the spread against the noise
> before reading anything into the correlation.

## What this country cannot show

**`Religiones Tradicionales` is 0.22%** — 39,521 people — in a country its own 2018 census
found **43.6% indigenous**. §11ad measured what this instrument does to folk practice in the
one place a census could check it: in Suriname LAPOP's traditional-religion cell is **0.21×**
the census's and the missing people come back as Christians. The card has one worldwide box
and never prints *costumbre*.

It is not corrected, because §14.4 forbids inventing a magnitude and nothing published says
what the right one is. **This is the single thing a second source would most improve**, and
Guatemala is on `queue.md`'s standing refinement list for it.

**`Zacapa` rests on 40 interviews**, a 95% interval of ±15.5pp on its Catholic share. Every
other department has at least 129 and the median has 279.

**The level is a fourteen-year average.** Guatemalan Catholicism runs 55.3% (2010) → 49.8%
(2023) across the pooled span, so the map is a couple of points more Catholic than the 2023
wave alone would draw. Pooling is what buys the department-level precision the split-half
depends on; a single wave is 1,500 people over 22 departments and cannot carry ADM1 at all.
spec §14.16's rule, and the cost is stated in `note_public`.

## What was drawn

    9,243,243  christianity.catholic.latin      51.8%
    6,221,650  christianity.evangelical         34.9%
      955,906  christianity.protestant           5.4%
      852,792  unchurched                        4.8%
      255,032  other.gt                          1.4%
      113,121  secular                           0.6%
       85,748  christianity.witnesses            0.5%
       74,110  christianity.latterday            0.4%
       39,521  indigenous                        0.2%
        2,009  judaism                           0.01%

17,839 dots at 1:1,000 over 80,578 Kontur hexes. Evangelical runs **50.7% of Izabal** to
**12.5% of Chiquimula**, four to one; Catholic runs 35.4% of Retalhuleu to 80.0% of
Chiquimula.

## Open

- ~~The `Ninguna (creyente)` call.~~ **Decided 2026-09-08: drawn.** See above.
- **`Protestante Tradicional` stays flat at -0.04**, and unlike `Ninguna` its chi-square is the weakest of the four eligible categories (p = 1.6e-06 against 3.6e-16). If a later wave moves it, revisit it the same way: chi-square first, then the split-half, then a named override.
- **A second source for the non-Christian tail**, which is what would move this country most.
  Nothing has been searched for yet. The USCB per-country geodatabases were checked and
  Guatemala is **not** among the 34 (Haiti, the Dominican Republic and Colombia are, and none
  of those three carries religion either).
- **Municipio geography.** LAPOP samples about 55 Guatemalan municipalities per wave and
  `municipio` is a PSU list rather than a partition, so there is no finer tier to reach for
  in this source. COD-AB ships ADM2 (340 municipios) and it has no counts to put on it.
