# Vanuatu — VNSO, 2020 National Population and Housing Census, Basic Tables Vol 1, Table 3.5

Wired 2026-09-08. 293,963 people, **66 area councils**, 12 drawn categories, **99.85% of the
country drawn**.

| | |
|---|---|
| source | Vanuatu National Statistics Office, **2020 National Population and Housing Census**, *Basic Tables Volume 1*, Table 3.5 — *Total population in private households by religion and region* |
| basis | `self_id`, **population in private households**, no age floor |
| geography | **66 units** — 64 rural area councils plus Port Vila and Luganville, ~4,500 people each |
| categories | **14 printed**, 12 drawn: ten named churches, customary beliefs, no religion, plus two residuals |
| drawn | **293,536 of 293,963, 99.85%**; the 428 not drawn are `Refuse to answer` (394) and `Not Stated` (34) |
| licence | published census table, open download, no account |
| witness | Volume 2's Table 30 reproduces all thirteen of its national figures exactly |

**The queue had Vanuatu down as six provinces. It is 66 units, and the categories survive the
fine tier intact** — which is the trade-off Fiji could not have (§9bd §3) and did not need to
be argued here at all.

---

## 1. Access — a Joomla site with no API, and the certificate does not verify

`vnso.gov.vu` answers, but three things about it are worth writing down:

* **`curl` fails with exit 60 before it fails with anything else.** The TLS chain does not
  verify from here. `sources/vu.py` passes `verify=False` and checks the payload instead: a
  `%PDF` header and an `%%EOF` trailer, per the truncated-at-source rule.
* **It is Joomla, not WordPress**, so none of §11v's or §9bd's CMS routes apply. Its search
  component returns **HTTP 500** on every query, including single words, so the library cannot
  be searched at all.
* **What works is walking the menu.** The files live under a predictable static tree,
  `/images/Public_Documents/Census_Surveys/Census/<year>/`, and a crawl of the nineteen
  reachable statistics pages found 417 files. The census ones:

```
  Census/1999/1999_PC_Report.pdf                         religion by ISLAND, tables 2.10-2.12
  Census/2009/2009 Census Basic Tables Report - Vol1.pdf  religion by province, table 3.5
  Census/2009/2009 Census Analytical Report - Vol2.pdf
  Census/2016/2016_Mini_Census_Main_Report_Vol_1.pdf      NO religion
  Census/2020/Basic_Tables/2020NPHC_Volume_1_-_Version_2.pdf   <- drawn, table 3.5
  Census/2020/..._Analytical_report_Volume_2.pdf               <- the witness, table 30
```

**No Excel anywhere.** The PDF is the source, and §3 is about getting the table out of it.

## 2. The tier, which cost nothing to take

Table 3.5's `Region` column is the **whole census hierarchy in one table**:

```
  VANUATU                             293,963
    URBAN                              65,867
      Port Vila                        48,461      <- drawn
      Luganville                       17,407      <- drawn
    RURAL                             228,095
      TORBA        11,215   7 area councils        <- drawn
      SANMA        42,245   9
      PENAMA       34,123  10
      MALAMPA      41,506  10
      SHEFA        54,108  17
      TAFEA        44,899  11
```

64 rural area councils plus the two urban municipalities, and **the same fourteen columns are
printed at every one of them**. `sources/vu.py` writes only the 66 leaf units and asserts the
hierarchy adds up first, at every level, column by column.

**This is the §9k trade-off simply not arising.** Fiji had to choose between 15 provinces with
23 categories and 86 tikina with 6; Vanuatu publishes the deep list and the fine geography in
the same table, so there is nothing to give up. Worth saying because the queue priced this
country at six provinces and it is eleven times finer than that.

## 3. Getting it off the page — right-aligned columns and two traps

The table is split across four PDF pages: **62-63 carry `Total` and eight religion columns,
64-65 the remaining six**, with the same row list repeated. The two blocks are parsed
separately and joined on row order, and `read()` asserts the labels match position for
position before it does.

Text order is useless here (the numbers are right-aligned and run into the labels), so rows
are rebuilt from **word geometry**: cluster words into visual rows by `y`, then recover the
columns by clustering the **right edge** of every numeric token, which is the stable one.

**Three things that break a naive parse, all of them silent:**

1. **`Canal - Fanafo` contains a hyphen, and a hyphen is this table's symbol for zero.** Parsed
   as a value it becomes a phantom column.
2. **`Central Pentecost 1` and `Central Pentecost 2` end in a digit** — two genuinely distinct
   area councils, whose suffixes become phantom values the same way.
   Both are fixed by the same rule: a value token must start right of `LABEL_X`.
3. **The header band is not at a fixed height.** Page 62 puts `Region` at y=117 and page 63 at
   y=105. A cutoff tuned on page 62 silently swallows **SHEFA** (y=118), which is the largest
   province in the country: the parse then returns 74 rows instead of 75, every other check
   still passes, and Shefa's 54,108 people quietly stop existing. The file anchors on the
   `Region` row itself.

## 4. THE PUBLISHED TABLE DOES NOT ADD UP, AND IT IS THE SOURCE

**41 of the 75 printed rows have their fourteen categories missing the printed `Total`.** The
distribution: `-2` on 4 rows, `-1` on 19, `+1` on 16, `+2` on 2. **Nothing misses by more than
two.**

This was assumed to be a parse bug and is not:

* Six rows were rendered at 200 dpi and read digit by digit — `VANUATU`, `URBAN`, `Port Vila`,
  `Luganville`, `RURAL` and `East Santo` — and all six agree with what the parser extracts.
* **East Santo prints a total of 5,788 beside categories that sum to 5,786.** The page is
  wrong, not the reader.
* Volume 2's Table 30 reproduces all thirteen national figures exactly, from a separate
  typesetting of the same census.

It is consistent with each cell being rounded independently, which is a common confidentiality
or weighting artefact. **So the category sum is each unit's universe here, and the printed
`Total` is carried in `vu.csv` as a `Total` row for reporting only.** At a maximum of two
people against units averaging 4,500 nothing visible changes, but `check()` asserts the bound:
if a row ever misses by more than 2, the build stops.

## 5. What is in the table

```
  Presbyterian                80,060   27.23%      Customary beliefs        9,080   3.09%
  Seventh Day Adventist       43,541   14.81%      Apostolic                6,894   2.35%
  Catholic                    35,602   12.11%      Latter Day Saints        5,174   1.76%
  Anglican                    35,339   12.02%      No Religion/Faith        4,023   1.37%
  Other churches              35,270   12.00%      ------------------------------------
  Churches of Christ          14,588    4.96%      Refuse to answer           394   0.13%
  Assemblies of God           14,450    4.92%      Not Stated                  34   0.01%
  Neil Thomas Ministry         9,515    3.24%      Total                  293,963
```

**No church is close to a majority, which is unusual in this region.** The largest is 27.2%;
Fiji's Methodists are 34.7% and the Tuvaluan and Tongan national churches far higher. Four
bodies sit between 12% and 15%.

**`Latter Day Saints` is new in 2020** — Volume 2's four-census table has a dash for it in
1989, 1999 and 2009.

## 6. `Other churches` is 12% and does not go to `christianity`

The two volumes label the same column differently, and it matters for 35,270 people:

| | header | what it implies |
|---|---|---|
| Volume 1, Table 3.5 | `Other churches` | Christian |
| Volume 2, Table 30 | `Other` | nothing |

And Volume 2 says what is in it: **"the category 'Other' includes 88 different religions
ranging from one member to more than 2,000 members."** *Religions*, and 88 of them. Vanuatu's
Baha'i, Muslim and Jehovah's Witness communities have nowhere else in this table to be, so they
are inside this cell and nothing published says in what proportion.

**So it goes to `other.vu` and not to `christianity`.** Claiming the whole cell for
Christianity on the strength of a column header that the other volume of the same census
contradicts would be §14.4 rule 1. It is the least comfortable call in the file and it is in
`taxonomy/vu2020.py`'s REVIEW.

The largest of the 88 is unnamed too: "more than 2,000 members" is all the census says.

## 7. Customary beliefs — a printed category, not a residual

**9,080 people, 3.09%, and Vanuatu has counted it in four consecutive censuses**: 6,484 in
1989, 10,365 in 1999, 8,600 in 2009, 9,080 in 2020. That is a national statistics office
listing customary belief beside the churches on equal terms, which is rare enough that
`indigenous.vanuatu` was added for it rather than folding it into a residual.

**It is almost entirely one island.** 7,757 of the 9,080 are in Tafea, 17.3% of the province,
and within Tafea it is Tanna:

```
  South West Tanna     1,967 of  6,487   30.3%
  Middle Bush Tanna    1,742 of  6,882   25.3%
  North Tanna            896 of  4,701   19.1%
  West Tanna           2,142 of 12,790   16.7%
  Aniwa                   75 of    465   16.1%
  South Pentecost        790 of  5,632   14.0%
```

That is where the **John Frum** movement and the Prince Philip movement are. The census names
neither, so neither is a node: `kastom` is what it counts.

**Read it as a floor**, for the reason `indigenous.african` and `indigenous.myanmar` carry:
the box is exclusive of the church boxes, and customary practice in Vanuatu runs alongside
church membership rather than instead of it.

## 8. The join is free, and it gets a witness that names cannot fake

COD-AB Vanuatu's **ADM2 is the census's own tier**: 66 polygons against 66 census units, and
**every name matches outright** on a folded comparison, nothing spare either way, no aliases.
Neither side publishes an id, and neither needs to.

**A name join that matches everything is exactly the one nobody checks**, and the failure it
cannot see is the wrong twin — two different places sharing a name, paired confidently
(`reference_name_join_wrong_neighbour`). So `sources/vu_geo.py` asserts something independent
of the names: **OCHA files each ADM2 under an ADM1 province, and the census files each area
council under a province by where it prints in Table 3.5.** Two organisations, and they agree
on **all 64** rural councils. Port Vila and Luganville sit outside the province blocks of the
table; COD files them under Shefa and Sanma, which is reported rather than asserted.

**No antimeridian problem.** Vanuatu is 166.5°E to 170.2°E, 3.7 degrees wide, so none of
§9bd §7's machinery is needed and plain EPSG:4326 is correct. The span is asserted anyway,
because a torn polygon is silent.

## 9. Placement — Kontur, and the 13% that had to be snapped rather than dropped

`sources/vu_grid.py`, 4,946 Kontur 400 m hexagons. Vanuatu needs a population grid because its
people are on the coasts: the interiors of Santo and Malekula are close to empty, Torres is six
islands, and an equal share per polygon would place dots by area.

**The vintage gap is three years** — counts 2020, grid 2023 — the smallest on this map outside
the countries whose grid and census share a year.

**645 hexes, 44,347 people, 13.25% of the grid, fall outside every council.** That is double
Fiji's 6%, and the reason it cannot simply be dropped is that **it is directional**:

```
  within  100 m   198 hexes   20,016 people    45.1% of the strays
  within  250 m   453         36,624           82.6%
  within  500 m   639         44,296           99.9%
  max distance 6,032 m, one cell
```

They are coastal cells whose centroid lands just seaward of a detailed island coastline on a
400 m grid, and the largest are **15 to 150 metres out, clustered on Port Vila and Luganville**
— which is where the people are. Dropping them would weight every shoreline light and pull the
dots inland. So each stray within 500 m is **snapped to the nearest council**; six cells and 51
people (0.015%) remain unplaced and are dropped.

Snapping is measurably the better choice: the census-against-Kontur correlation rose from
**r = 0.9607 to r = 0.9738**, and the worst per-unit ratios tightened from 0.50–1.92 to
0.65–1.75.

**And with 66 units that correlation is a real check on the join**, which Fiji's fifteen
provinces could not support:

```
  national      Kontur 334,530 vs census 293,963 (private households), ratio 1.138
  band          all 66 inside a factor of 4
  correlation   r = 0.9738, against a best of 0.4119 over 2,000 random pairings (0 reach it)
```

## 10. What the map shows

**Three missions divided the group between them in the 1850s and the boundaries have barely
moved.** Anglican is **77.3% of Torba** and **44.9% of Penama** against **0.2% of Tafea** and
0.4% of Malampa: the Melanesian Mission worked the Banks and Torres islands and northern
Pentecost and essentially nowhere else. Presbyterian is 43.9% of Malampa and 40.7% of Shefa,
the centre and south, and 0.5% of Torba. The Churches of Christ are 16.1% of Penama and 0.9%
of Malampa.

Sanma is the exception and the analytical report says so: it "showed the most diverse mix of
religions of all provinces".

**Tafea is the one province where a non-Christian category is large**, at 17.3% customary.

## 11. What else is on this server, unused

- **The 1999 census publishes religion by ISLAND** (tables 2.10–2.12, pages 93–98), with a
  slightly different category list and separate male/female tables. Finer in one sense than
  area councils and twenty-six years old.
- **The 2009 basic tables give religion by province and urban/rural only** (table 3.5, pages
  34–40), so 2020 is both newer *and* eleven times finer. No reason to prefer it.
- **The 2016 mini census does not ask religion at all** — searched, zero hits.
- SPC's **`vanuatu.popgis.spc.int` is live** and serves the 2020 census at area council,
  enumeration area and island level with **1,687 indicators — and no religion theme in any of
  its three trees**. It was checked first, on Fiji's precedent, and it is not the route here.
  It would be an excellent geography source if COD-AB ever stopped matching.
- The 2020 basic tables also cross religion with **sex** (tables 3.6, 3.7) and with **five-year
  age group** (table 3.8). Not read.
