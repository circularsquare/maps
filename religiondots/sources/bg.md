# Bulgaria — NSI, Преброяване 2021

Wired 2026-09-08. 6,519,789 people, 265 municipalities, 8 categories.

| | |
|---|---|
| source | Национален статистически институт, Census 2021, `Census2021_Ethnocultural characteristics_BG.xlsx` sheet 4 |
| basis | `self_id` |
| geography | 265 общини (obshtini), which are Bulgaria's LAU units |
| categories | 5 drawn, 3 non-answers, plus the universe total |
| drawn | **5,171,267 people, 79.3%** |
| placement | `POPGRID2021_1000M`, the census's own 1 km grid, 6,461,591 people |
| licence | NSI general licence; the grid additionally forbids publishing individual cell values |

**The queue's entry for Bulgaria was wrong in a specific and instructive way.** §11o recorded
it as *"re-tested and it answers"*, naming two hosts. Both answer and neither is the route:

* `nsi.bg/en/content/6704/population-religion` returns **200 with the NSI homepage**. The
  content id is dead and the site falls through to the front page instead of 404ing, so a
  status-code probe scores it as live. This is the failure mode a probe cannot see, and the
  tell is that the `<title>` is the organisation's name rather than the page's.
* `censusresults.nsi.bg` is live and is **the 2011 census**, publishing religion by **oblast**
  (28 units) crossed with age and urban/rural. Ten years stale and nine times coarser.

---

## 1. The route, which is the site's own search box

```
https://www.nsi.bg/search?q=вероизповедание
   -> /press-release/etnokulturni-harakteristiki-na-naselenieto-kam-7-septemvri-2021-godina-6833
https://www.nsi.bg/sitemap
   -> /statistical-data/151/1349   "Резултати от Преброяване 2021"   nine .xlsx
   -> /statistical-data/151/1350   "Грид на населението, 1 кв.км"    the placement layer
```

`/sitemap` is 287 KB of static links and is the fastest way to see NSI's real structure; the
navigation on every content page is JavaScript and greps as nothing. **Search before
concluding a Drupal-shaped office is walled** — the whole country was two GETs away from a
sweep that had already declared it "not chased further".

The workbook is **78 KB**. Sheet 1 is ethnicity nationally, sheet 2 ethnicity by municipality,
sheet 3 mother tongue, **sheet 4 religion by municipality**. `sources/bg.py` asserts the sheet
4 header row verbatim, so a re-publication that reorders the columns fails the build rather
than relabelling a religion.

## 2. The partition is exact and is witnessed twice

The eight category columns sum to `Общо` in **every one** of the 265 obshtini, the 28 oblasti,
the six NUTS regions and the country, with a difference of zero. `bg.py` checks each row
rather than only the national one (§5a).

The second witness is NSI's own press release, `Census2021-ethnos.pdf`. Recomputing its
headline shares from the 265 municipal rows returns **71.48%, 10.82%, 5.17%, 4.39%, 8.01%**
against its published 71.5 / 10.8 / 5.2 / 4.4 / 8.0. Note the denominator NSI uses for those:
**5,903,108**, the enumerated population *minus* the 616,681 register additions, and
*including* both declining answers. Using "those who named a religion" instead gives 81.6%
Christian and contradicts the office's own number, so any note quoting a share must say which
denominator it is on.

## 3. The UNSD oracle returns values, and `sources.md` §11r documents the wrong column set

§11r introduced `data.un.org`'s Demographic Yearbook table 28 with

```
...&Format=csv&c=0,1,2,3,4,5,6
```

which yields **Table Code, Country, Year, Reference Date, Area Code, Area** and no numbers at
all. That is why the oracle has only ever been asked the yes/no question *"has this office
ever tabulated religion"*. The column set that carries the counts is

```
http://data.un.org/Handlers/DownloadHandler.ashx?DataFilter=tableCode:28
    &DataMartId=POP&Format=csv&c=0,2,3,6,8,10,15,16
    -> Country | Year | Area | Sex | Religion | Source Year | Value | Value Footnotes
```

**Column 16 is `Value`.** Three traps, all of which look like something else:

* asking for **17 or more** columns returns a **zero-byte body with a 200**, so it reads as a
  network fault rather than a bad request;
* adding `countryCode:100` to the `DataFilter` also returns zero bytes, so the filter is
  effectively table-code only and the whole 220 KB must be pulled and filtered locally;
* the country name is column **2**, not column 0. Filtering on `row[0]` matches nothing and
  looks exactly like "this country is absent from the oracle", which is the one conclusion
  §11r warns is expensive to get wrong.

Used properly it carries Bulgaria at **2001, 2011 and 2021**, urban and rural as well as
total, and it agrees with the workbook to the person. It also folds NSI's two declining boxes
into one `Not Specified`: 259,235 + 472,606 = **731,841** exactly, which is what establishes
the two publications are the same tabulation.

**This is worth re-running for every country in the file**, and especially for the ones §11r
and §11p closed on a category count they read off the index.

## 4. The denominations, which are DERIVED — and this section reverses the first build

**Settled 2026-09-08.** The first build drew `Християнско` and `Мюсюлманско` undivided, citing
§14.4 rule 1. On the map that put a flat Christianity colour between four Orthodox
neighbours. Anita: *"bulgaria looks kinda out of place as it's the only one in the area where
we dont have christianity breakdown to orthodox and not. so it displays as generic light
yellow. is there any way we could get the orthodox proportion?"* and, on being shown the
options, *"yes 2011 oblast composition is ideal."*

**The first reading of rule 1 was too narrow.** It forbids estimating a magnitude a source
does not publish. Every magnitude in `bg_split.py`'s output is NSI's: the national
denomination totals are the 2021 census's, each obshtina's Christian and Muslim totals are
the 2021 census's, and only the *distribution of the former across the latter* is modelled,
from the same state's 2011 census. That is §14.10's amendment, which permits the map to run
the model itself when the magnitude is the host state's, the coefficients documented and the
output checked against something independent.

**Method** (`bg_split.py`): seed a 28 x 5 oblast-by-denomination table from 2011; rake it
(IPF) onto two measured 2021 margins, each oblast's own Christian total and each
denomination's published national total; then split each obshtina by largest remainder so its
rows sum to its own measured column exactly, repairing the national totals afterwards with
single-person moves inside one obshtina. Muslims get the seed and the split but no raking,
because no 2021 Sunni/Shia national total exists.

**Three checks, and the first could have failed.** Eastern Orthodox is **97.44%** of
Christians in 2011 and **97.30%** in 2021, so the ten-year-old shape is sound; the build
fails if that drift ever exceeds half a point. Every obshtina sums to its measured column and
every denomination to its published national figure, both to the person. And the Shia total,
which nothing in the model constrained, comes out at **29,470** against the **27,579** that
UNSD's classification could not code out of Bulgaria's 2021 Muslims — a 6.9% difference from
a publication the model never saw.

**What it costs.** The composition is uniform inside an oblast, because 28 units is the
finest geography any Bulgarian census publishes it at. The clearest casualty is the
Catholics: most of the 38,709 are Banat Bulgarians in **Rakovski**, and this spreads them
across Plovdiv oblast. Everything is reversible in the interface: every derived row carries
`parent_column`, so `inferred dots: not shown` redraws the measured table exactly.

**Still worth chasing**, in order: a finer 2021 table (the census clearly holds the five
Christian categories at unit level since it publishes them nationally; `infostat.nsi.bg`
serves the NSI homepage for every path and was not cracked, and the EU Census Hub carries
Bulgarian 2021 hypercubes), which would turn all of this from derived into measured.

**What the municipal table has:** one `Християнско` column, 4,219,270 people, 64.7% of
Bulgaria.

**What exists elsewhere:**

| | Eastern Orthodox | Protestant | Catholic | Armenian Apostolic | other Christian |
|---|---:|---:|---:|---:|---:|
| 2021, **country only** | 4,091,780 | 69,852 | 38,709 | 5,002 | 13,927 |
| 2011, **per oblast** | 4,374,135 | 64,476 | 48,945 | 1,715 | — |

and the 2011 portal additionally splits Islam into **Sunni 546,004**, **Shia 27,407** and
unspecified 3,728, per oblast. Bulgaria's Shia are the Alevi (Kazalbash) of Razgrad,
Silistra, Targovishte and Sliven, the largest such population in Europe outside Turkey; this
map is the only place they appear, and they arrive on the weaker of the two splits.

## 5. A fifth of the country is not drawn, and none of it is recoverable

| column | people | share | why not drawn |
|---|---:|---:|---|
| `Не мога да определя` | 259,235 | 4.4% | offered box, "cannot determine". An answer in §9aq's sense, but not one that names a religion, and `unaffiliated` is a different box. hr2021.py's `Ne izjašnjavaju se`. |
| `Не желая да отговоря` | 472,606 | 8.0% | offered box, "do not wish to answer". The question has been voluntary since 1992. |
| `Непоказано1` | 616,681 | 9.5% | never asked. Imputed from administrative registers because the census could not reach them. mk2021.py's 132,260 case exactly. |

**This is the opposite of Slovakia's `ostatné`.** There a residual demonstrably contained
named churches and was drawn on §6.12 to avoid a hole reading as an absence of people. Here
`Друго` is its own column beside these three, so there is nothing hidden inside them to
recover, and drawing them would assert a religion for 1.35M people who declined to name one.

**But the hole is not evenly spread and the note says so.** Non-response runs **3.1% to
64.3%** between municipalities, median 15.9%. The five highest are all Rhodope municipalities
with large Bulgarian-speaking Muslim populations — Nedelino 64.3%, Banite 55.8%, Zlatograd
52.5%, Laki 43.0%, Devin 37.6% — and Sofia is 29.1%, against 3-5% across the Vidin and
Kyustendil villages. A thin patch of dots in the Rhodopes is a question left unanswered
rather than an empty valley.

## 6. Geometry and placement both cost nothing

**Boundaries: GISCO LAU 2021.** Bulgaria is 265 units whose `LAU_ID` is NSI's own obshtina
code verbatim (`VID09`, `SML31`, `KRZ07`), which are the strings the workbook keys on. The
crosswalk is the identity function and `bg_geo.py` asserts the two 265-element sets are equal,
so §12's shapes of failure 1 and 2 cannot arise. The shapefile was already on disk as a shared
asset from `es_geo.py`.

**Placement: the census's own 1 km grid**, the third measured placement layer in the project
after Germany and Slovakia. `POPGRID2021_1000M` is Census 2021 aggregated from the point
location of every record, 112,883 cells of which **20,912 hold anybody**, summing to
**6,461,591**. The 58,198 missing (0.9%) are NSI's own published "unallocated on map" figure,
records with no usable point location; `bg_geo.py` asserts the published number so a different
grid fails rather than silently reweighting the country.

Cells are split by **area of intersection**, not assigned by centre — sk_geo.py's rule, needed
here for the opposite reason. Slovakia's problem was municipalities smaller than a cell;
Bulgaria's obshtini are large, but the Danube and Black Sea edges are all border cells and a
centroid rule discards the share of a square lying over Romania or the sea. Normalising
against the area actually inside the country places **6,461,591 of 6,461,591**, nothing lost.

Grid/census ratio is within 0.5-2.0 for **100.00%** of obshtini, median 0.998, worst
Bozhurishte at 0.893. **That is not an independent check on the counts** (§9av): the grid is
the same enumeration as the workbook, so it validates the cell-to-obshtina assignment and
nothing else.

**The grid's licence is narrower than NSI's general one** and is worth carrying: NSI permits
"analytical developments and mapping products based on the statistical information in the
dataset ... but without showing the original values in the individual grid cells", and forbids
redistributing the dataset in its original form. Using it as a within-unit weight is inside
that; a choropleth of cell population would not be.

## 7. What the map gains

Bulgaria was the last hole in the Balkans, which is otherwise complete: Romania, Serbia,
Kosovo, North Macedonia, Greece, Croatia, Bosnia, Montenegro and Slovenia's neighbours are all
drawn. What it adds beyond contiguity is **47 municipalities where Muslims outnumber
Christians**, in three separate geographies that a national figure of 10.8% hides completely:
the Rhodopes along the Greek and Turkish border, the Ludogorie between Razgrad, Silistra and
Shumen, and the villages inland from Burgas (Ruen, 86.6%, is the highest Muslim share of any
municipality of its size).
