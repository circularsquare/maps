# Barbados — 2010 census, via BSS's own tables workbook

`sources/bb.py` -> `data/normalized/bb.csv`. Boundaries: `sources/bb_geo.py`, placement:
`sources/bb_grid.py`. Taxonomy: `taxonomy/bb2010.py`.

**226,193 people on 11 parishes, 22 named categories, 98.77% of that universe drawn.** The
most Anglican country on this map, and the only census outside the United States that counts
Nazarenes, Wesleyans and the Salvation Army as three separate answers.

| | |
|---|---|
| counting geography | **parish, 11 units** — 20,600 people each |
| placement | Kontur H3 r8 hexes, 705 of them, snapped to the coastline |
| basis | self-identification, census |
| tier | `measured` throughout |
| vintage | census 2010 — **and 2021 exists; see §1** |

---

## 1. WHY 2010 AND NOT 2021

BSS ran a 2021 census and published the same table — `02.06`, *Total Population by Parish,
Sex and Religion* — with the same 23 categories, in `Census-2020-Tables.xlsx`. It is not
used.

| | 2010 | 2021 |
|---|---|---|
| estimated resident population | 277,821 | 269,090 |
| tabulated / tabulable population | **226,193** | **136,415** |
| estimated undercount | 49,115 | 130,993 |
| **percentage undercount** | **18%** | **48.7%** |

**The 2021 census reached about half the country**, and BSS says what that costs the
geography in its own §6.0, *Tabulated Results*:

> *"The tabulated results of the 2021 Census can be regarded as a large sample of the
> resident population in Barbados … **In most cases, disaggregation by area is not included
> – as most results at that level would be understated**, considering the significant size
> of the undercount."*

The workbook ships the parish cut regardless. **The publisher's own warning is taken over
the publisher's own spreadsheet.** Nothing is lost in depth by choosing 2010: the category
lists are the same 23, with 2021 renaming `Muslim` to `Islam` and `Jewish` to `Judaism` and
reordering.

> **A newer census is not automatically the better source, and the office will sometimes
> tell you so in prose while contradicting itself in a file.** Compare Saint Vincent
> (§11v), where the 2023 report is newer and *coarser* — 13 census divisions against the
> 221 enumeration districts the map already draws — and 2012 stands for a different reason.

## 2. THE `NO RELIGION` COLUMN HAS NO HEADER, AND IT IS A FIFTH OF THE COUNTRY

In the 2010 sheet, column 23 sits between `Other Non-Christian` and `Not Stated` and **its
header cell is blank**. It holds **46,562 people — 20.59%, the second largest answer in
Barbados**. A header-driven read names it `Unnamed: 23` or drops it, and the country loses a
fifth of itself with no error anywhere.

Two independent things identify it, and `bb.py` asserts both:

1. **Arithmetic.** The categories sum to each unit's own `Total` only when it is included —
   on all 36 rows (11 parishes plus the national row, each in Total/Male/Female). Drop it
   and every one of those fails by 20.6%.
2. **The 2021 workbook.** Its Table 02.06 publishes the same categories in the same relative
   order with that position labelled **`No Religious Affiliation`**.

So the 2021 file, rejected as a source in §1, is the *evidence* for reading the 2010 one.
The check lines the two category lists up as sets and stops if they ever stop matching —
four labels differ and all four are held in an explicit table (`Islam`/`Muslim`,
`Judaism`/`Jewish`, a curly apostrophe in `Baha’i`, and `Jehovah's Witness` against
`Jehovah Witness`), because the point of the check is that the two lists are the *same
list*, and anything that let two different lists pass would defeat it.

## 3. KONTUR INDEPENDENTLY REPRODUCES THE CENSUS'S OWN UNDERCOUNT

The 2010 report publishes an **estimated resident population by parish** (Table C) as well
as the tabulable counts, so the census's coverage can be computed per parish. It is not
uniform:

| parish | tabulable | estimated | coverage |
|---|---|---|---|
| St. James | 21,258 | 28,498 | **74.6%** |
| St. Philip | 23,788 | 30,662 | 77.6% |
| St. Michael | 69,604 | 88,529 | 78.6% |
| Christ Church | 43,127 | 54,336 | 79.4% |
| St. Thomas | 12,035 | 14,249 | 84.5% |
| St. Lucy | 8,609 | 9,758 | 88.2% |
| St. Joseph | 5,939 | 6,620 | 89.7% |
| St. Andrew | 4,631 | 5,139 | 90.1% |
| St. Peter | 10,382 | 11,300 | 91.9% |
| St. George | 18,203 | 19,767 | 92.1% |
| St. John | 8,617 | 8,963 | **96.1%** |

**Kontur's building-footprint grid knows nothing about any of that**, and its
modelled-to-tabulated ratio per parish tracks the implied undercount factor:

```
  Pearson r(1/coverage, Kontur ratio)   = +0.863
  Spearman rank r                       = +0.809
  shuffle null, 20,000 relabellings     : best r = +0.919, 0.05% reach the real one
```

Two entirely unrelated sources — a statistical office's own post-enumeration estimate, and a
satellite-derived building model — **agreeing about which parishes were under-enumerated**.
That is the strongest external check any country on this map has on its own coverage.

It also means the per-parish ratio table in `bb_grid.py` is a **coverage read rather than a
shape check**, and is expected to sit above 1 — around 1/0.814 ≈ 1.23 on average, and higher
where the census did worst. The band is asserted against the *estimated resident* population
(ratio 1.015), not against the tabulable one, so the census's own 18% gap does not read as a
grid error.

**Nothing is scaled up** (§14.4). Correcting the parishes would assume the missed 18% has
the same religion mix as the counted 82%, and nothing establishes that. So an under-covered
parish draws proportionally fewer dots than its true population warrants — St. James by
about a quarter — while the composition *inside* each parish is unaffected.

## 4. What the country shows

* **23.87% Anglican — the most Anglican country on this map**, against 13.9% in Saint
  Vincent, 11.9% in the Bahamas, 5.7% in Trinidad and 2.8% in Jamaica. **St. John is 37.5%.**
* **The two big answers are spatial opposites.** Anglicanism peaks in St. John (37.5%) and
  bottoms in St. Andrew (15.3%); `Other Pentecostal` does the reverse — **29.4% of
  St. Andrew** against 19.5% nationally. The established church holds the settled south and
  east; the Pentecostal churches hold the rugged, poorer Scotland District in the
  north-centre.
* **Four Holiness bodies, counted separately** — Nazarene 3.23%, Wesleyan 3.40%, Salvation
  Army 0.39%, Church of God 2.37%. **9.4% of the country in one family, split four ways**,
  which no other source here does. Three of the four nodes had been reachable only from the
  U.S. Religion Census.
* **And they are not evenly spread.** **St. Lucy is 13.4% Adventist** — more than twice the
  national 5.9% — and **2.3% Salvation Army against 0.4%**, a sixfold concentration, both in
  the northernmost parish. **St. Joseph is 7.2% Wesleyan.** **St. Thomas is 5.7% Moravian**
  against 1.2% nationally, with St. John at 4.3% — the old Moravian mission field.
* **Roman Catholics are 3.84%**, very low for the Caribbean. Barbados was never Spanish or
  French and never received the Irish, Portuguese or Hispanic migrations that made
  Catholicism large in Trinidad, Belize or the Bahamas.
* **20.59% report no religious affiliation**, highest in St. Joseph (25.8%) and St. Michael
  (23.6%), lowest in St. Philip (16.3%).
* **Rastafari is 1.03%** — the fifth census count on this map after Jamaica, Saint Vincent,
  Trinidad and the Bahamas — and rural again: 2.01% of St. Andrew against 1.38% of
  St. Michael.
* **Muslims are 0.71% and concentrated in Bridgetown** — 1.61% of St. Michael, which is the
  Indo-Guyanese and Gujarati commercial community.

## 5. Placement

**705 Kontur hexes.** Barbados is the densest country on this map, and its parishes are
unusually uniform in area — 23.9 km² (St. Joseph) to 62.5 km² (St. Philip), a 2.6-fold
spread against Trinidad's 71-fold and the Bahamas' four orders of magnitude. So the grid is
not here for empty land or for unit-size range.

It is here for **St. Michael**, which holds 69,604 of 226,193 people on 40.7 km², nearly all
of them in Bridgetown and the suburban belt running south and west. Uniform scatter would
spread a third of the country evenly over a parish that is built up at one end.

2.45% of the grid lands outside every parish, all of it within 500 m of one — COD's
coastline against a 400 m hex, on a single compact island with no outlying cays — and is
snapped to the nearest parish within 1 km, as in the Bahamas (§9ar) and Cayman (§9at).

## 6. Two notes on the source

**Neither workbook is linked from the census page.** `stats.gov.bb/census/` links the 2010
and 2021 *reports* as PDFs and neither xlsx; both are in the WordPress media library and
nowhere else. The PDF's Table 02.06 is the same table spread over six pages of scanned
layout — the workbook is the same data with no parsing risk, and it was found the way the
Bahamas' All-Island Report was (§11v).

**COD-AB's ADM1 is the parish tier exactly**, 11/11 both ways, and the parishes are the only
sub-national geography Barbados publishes at all — there is no tier below them in any
boundary set. Ten of the eleven names differ only as `St.` against COD's `Saint`, handled by
`fold()` rather than an alias table. BSS publishes no code, so `bb.py` carries COD's pcode by
name and `bb_geo.py` asserts the pairing from the other side.

## 7. What is in the workbook and not used

Male and female splits on every religion cell, read and used as a check. Table 02.05 is the
same religion list by five-year age group nationally. The other 78 sheets cover age, sex,
union status, fertility, education, employment, industry, occupation, migration, disability,
housing, tenure, water, sanitation and household amenities — all by parish.
