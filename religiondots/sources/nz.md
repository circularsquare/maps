# New Zealand — Stats NZ census religious affiliation

Ingested 2026-08-30.  `sources/nz.py --fetch` downloads `data/raw/nz/`; `sources/nz.py`
writes `data/normalized/nz.csv`.

**The shape of the problem, stated first, because everything below is a consequence of it:
New Zealand publishes religion at depth, and it publishes religion at fine geography, but
never both in the same table.**  Level 3 of the classification — Anglican, Roman Catholic,
Rātana, Ringatū, Sikhism, Tenrikyo — stops at territorial authority.  Statistical area 2 gets
level 1: nine religion groups, one of which is "Christian" and covers a third of the country.
So this is a spec.md §3.4 country by construction, and the normalised file carries both
tables rather than pretending one of them is the other.

| | rows | geography | units | classification | categories | year |
|---|---|---|---|---|---|---|
| SA2 table | 29,809 | statistical area 2 | 2,395 | **level 1** | 11 + 2 aggregates | 2023 |
| national table | 166 | New Zealand | 1 | **level 3** | 163 + 3 aggregates | 2018 |

`basis` is `self_id` on every row: both are the census religious-affiliation question, which
asks the respondent about themselves.  See "What self_id is doing here" below — 15.6% of the
2023 answers did not come from a 2023 form, and that is a confidence question rather than a
basis question.

---

## 1. Where it comes from and how to re-fetch

### SA2, 2023 — ArcGIS feature service, no key

    https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/
        2023_Census_totals_by_topic_for_individuals_by_SA2/FeatureServer/1/query

Layer **1** is "2023 Census totals by topic for individuals by SA2 part 1", the unclipped
attribute table.  `sources/nz.py --fetch` pages it at the service's own `maxRecordCount` of
2000 — two requests, 1.7MB — and saves the responses verbatim as
`sa2_part1_religion_page0.json` / `page1.json`, alongside the layer metadata
(`sa2_part1_layer.json`) and the portal item record (`arcgis_item.json`).

Discovery notes worth keeping, because none of this is findable from the Stats NZ website:

- Religion is on **part 1**, not part 2.  Part 2 is housing, study, qualifications and travel;
  it has 862 fields and not one of them is religion.
- The hub page (`2023census-statsnz.hub.arcgis.com/maps/29a82d5a0ea24a3880219bcb3df126dc`)
  and the datafinder page (`datafinder.stats.govt.nz/layer/120898-…`) are both JavaScript
  shells — fetching them returns a title and nothing else.  The REST endpoint with `f=json`
  is the only readable description of what the table contains.
- **Field names carry no meaning.**  The religion columns are `VAR_1_403` … `VAR_1_415`; the
  category is only in the field *alias*.  `sources/nz.py` therefore reads the mapping out of
  the saved layer metadata rather than hardcoding the numbers, so a re-release that renumbers
  the columns fails loudly instead of silently relabelling Islam as Judaism.  The same table
  at SA1 numbers them `VAR_1_398` … `VAR_1_410` — different offsets for the same variable in
  the sibling dataset, which is exactly the trap.

The same service's layer **0** is the identical table **clipped to the coastline** and carries
geometry, so boundaries of guaranteed-matching vintage come from the same place as the data
(§4).  Layer 2/3 are part 2, clipped and unclipped.

### National, 2018 — one CSV inside a zip, no key

    https://www.stats.govt.nz/assets/Uploads/2018-Census-totals-by-topic/Download-data/
        2018-census-totals-by-topic-national-highlights-csv.zip
        -> religious-affiliation-total-responses-2018-census-csv.csv

69KB, 52 CSVs, one per variable.  Columns: `Code, Religious_affiliation,
Census_usually_resident_population_count`.  Codes are five characters and must be read as
strings — `00000` is No Religion.  The zip is stored untouched in `data/raw/nz/` and read with
`zipfile`.

**There is no 2023 equivalent.**  The 2018 census had a "totals by topic — national
highlights" CSV bundle; the 2023 census does not, and all 2023 detail lives in Aotearoa Data
Explorer instead (§6).

## 2. Licence

**CC BY 4.0**, from the portal item's own `licenseInfo`: "Deed - Attribution 4.0
International - Creative Commons", attribution "Stats NZ – Tatauranga Aotearoa".  The layer's
`copyrightText` is the same.  The 2018 CSV bundle sits under the stats.govt.nz site terms,
whose footer reads "Unless noted otherwise, all content on stats.govt.nz is licensed under the
Creative Commons 4.0 International Licence" and links to CC BY 4.0.  Attribution required, no
other restriction; safe to ship.

## 3. Categories and national counts

### The 11 level-1 categories, 2023, summed over 2,395 SA2s

| source_category | count | % of population |
|---|---:|---:|
| No Religion | 2,575,989 | 51.58 |
| Christian | 1,614,456 | 32.33 |
| Object to answering | 342,723 | 6.86 |
| Hinduism | 144,834 | 2.90 |
| Other Religions, Beliefs and Philosophies | 101,055 | 2.02 |
| Islam | 75,135 | 1.50 |
| Māori Religions, Beliefs and Philosophies | 65,151 | 1.30 |
| Buddhism | 57,102 | 1.14 |
| Spiritualism and New Age Religions | 21,180 | 0.42 |
| Judaism | 5,487 | 0.11 |
| Residual Categories | 0 | 0.00 |
| *Total* (aggregate) | *4,993,920* | |
| *Total stated* (aggregate) | *4,993,842* | |

"Residual Categories" — don't know, religion unidentifiable, response outside scope, not
stated — is **zero in every SA2 in the country**, and "Total stated" is 78 short of "Total"
rather than the hundreds of thousands one would expect.  That is not a clean census; it is
imputation, and §5 is about it.

### The 163 level-3 categories, 2018, national

This is the R2 payload and the reason the 2018 table is carried at all.  Names are verbatim
except that a bare `nfd` or `nec` is shorthand for the full category, which repeats the family
name — `data/normalized/nz.csv` has them in full, as spec.md §2 requires.

**No Religion** — 1 category, 2,264,601.

**Buddhism** — 7, 52,779: Buddhism nfd 44,355 · Theravada Buddhism 4,851 · Zen Buddhism 1,401 ·
Mahayana Buddhism 1,026 · Nichiren Buddhism 768 · Vajrayana Buddhism 327 · Buddhism nec 51.

**Christian** — 87, 1,738,638: Anglican 314,913 · Christian nfd 307,926 · Roman Catholic 295,743
· Presbyterian 221,199 · Catholicism nfd 173,016 · Latter-day Saints 54,123 · Methodist nfd
52,734 · Baptist nfd 35,967 · Born Again 33,486 · Pentecostal nfd 22,296 · Jehovah's Witnesses
20,061 · Christian Fellowship 18,042 · Seventh Day Adventist 17,799 · Assemblies of God 14,883 ·
Tongan Methodist 11,169 · Protestant nfd 8,544 · Samoan Congregational 7,932 · Salvation Army
7,929 · Plymouth or Exclusive Brethren 6,822 · Chinese Christian 6,660 · Open Brethren 5,640 ·
ACTS Churches 5,460 · Reformed 5,418 · Wesleyan Methodist 4,623 · Evangelical 4,554 · Orthodox
nfd 4,503 · Methodist nec 3,657 · Uniting/Union Church 3,624 · Lutheran 3,585 · Korean Christian
3,543 · Congregational 3,513 · Vineyard Christian Fellowship 3,399 · Greek Orthodox 3,162 · New
Life 3,132 · Elim 3,018 · Russian Orthodox 2,952 · Korean Presbyterian 2,820 · Liberal Catholic
2,115 · Christian and Missionary Alliance 2,094 · Pentecostal nec 1,911 · Christadelphian 1,758
· Destiny Church 1,722 · Cook Island Congregational 1,698 · Arise Church 1,641 · Jesus Follower
1,575 · Christian Outreach 1,554 · Brethren nfd 1,551 · Church of God 1,458 · Orthodox nec 1,440
· Church of Christ nfd 1,302 · Independent Baptist 1,218 · Associated Churches of Christ 1,176 ·
Catholicism nec 1,086 · Full Gospel 1,017 · Reformed Baptist 987 · Independent Pentecostal 954 ·
Religious Society of Friends (Quaker) 954 · Serbian Orthodox 936 · Bible Baptist 846 · Other
Church of Christ and Churches of Christ nec 780 · Independent Evangelical Churches 750 ·
Equippers Church 705 · United Pentecostal 690 · Christian nec 642 · Christian Science 639 ·
Christian Revival Crusade 576 · Nazarene 564 · Coptic Orthodox 546 · Chaldean Catholic 534 ·
Syro-Malabar Catholic 483 · City Impact Church 441 · Adventist nfd 378 · Unitarian 354 ·
Assyrian Orthodox 327 · Chinese Presbyterian 327 · Worldwide Church of God 279 · Revival Centres
180 · Brethren nec 147 · Maronite Catholic 96 · Fundamentalist 87 · Ecumenical 69 · Metropolitan
Community Church 63 · Adventist nec 54 · Ukrainian Catholic 39 · Melkite Catholic 33 · Baptist
nec 12 · Commonwealth Covenant Church 3.

**Hinduism** — 5, 123,534: Hinduism nfd 121,644 · Hare Krishna 645 · Hinduism nec 882 ·
**Yoga 327** · Arya Samaj 36.

**Islam** — 6, 61,455: Islam nfd 57,276 · Sunni 2,961 · Shi'a 651 · Ahmadiyya Muslim 369 ·
Sufi 162 · Islam nec 36.

**Judaism** — 4, 5,274: Judaism nfd 3,348 · Reformed Judaism 807 · Orthodox Judaism 792 ·
Conservative Judaism 327.

**Māori Religions, Beliefs and Philosophies** — 5, 62,634: Ratana 43,821 · Ringatū 12,336 ·
nfd 3,699 · nec 1,584 · Paimarire 1,194.  (Stats NZ's own spelling is inconsistent —
"Ringatū" has its macron, "Ratana" and "Paimarire" do not.  Kept verbatim in the CSV.)

**Spiritualism and New Age Religions** — 13, 19,695: Spiritualist 8,262 · Pagan 2,730 ·
Rastafarianism 1,707 · Wiccan 1,482 · Other New Age Religions nec 1,311 · Satanism 1,149 ·
Nature and Earth Based Religions nfd 807 · Nature and Earth Based Religions nec 648 ·
Pantheist 453 · New Age nfd 363 · Church of Scientology 321 · Animist 273 · Druid 189.

**Other Religions, Beliefs and Philosophies** — 30, 91,239: Sikhism 40,908 · **Jedi 20,409** ·
Atheism 7,068 · Agnosticism 6,516 · Church of the Flying Spaghetti Monster 4,248 · Baha'i
2,925 · Theism 2,607 · nfd 1,434 · Taoism 1,098 · Zoroastrian 1,068 · Humanism 663 · Jainism
612 · Shinto 387 · nec 297 · Chinese Religions nfd 276 · Deism 150 · Mahikari 138 · Falun Gong
105 · Confucianism 99 · Unification Church (Moonist) 93 · Japanese Religion nfd 33 ·
Japanese Religion nec 18 · Chinese Religions nec 18 · Socialism 15 · **Tenrikyo 12** · Marxism
12 · Libertarianism 9 · Rationalism 9 · **Cao Dai 6** · Maoism 6.

**Residual and object** — 5, 312,795: Object to answering 312,795; Don't know, Religion
unidentifiable, Response outside scope and Not stated are all zero.

Four things in that list are spec.md §2.3 problems rather than denominations, and none of them
should be given a taxonomy node without Anita deciding:

- **Jedi, 20,409** — larger than every Orthodox jurisdiction in the country combined, a
  protest answer rather than an affiliation, and it is *in* the classification with a code.
  Flying Spaghetti Monster, 4,248, is the same thing.
- **Yoga, 327, filed under Hinduism** — the same category the US Religion Census has as "Hindu
  Yoga and Meditation" (437k) and which spec.md §2.3 already holds out.  New Zealand's version
  is 1,300× smaller and identically wrong.
- **Atheism 7,068, Agnosticism 6,516, Humanism 663, Rationalism 9** sit under "Other
  Religions", *not* under "No Religion" — so New Zealand's irreligious are split across two
  level-1 branches by whether they named their irreligion.
- **Marxism 12, Maoism 6, Socialism 15, Libertarianism 9** are coded as religions.

`nfd` is "not further defined" and `nec` "not elsewhere classified" — these are the residual
nodes spec.md §3.2 wants computed, except that Stats NZ has already computed them and they are
enormous.  *Christian nfd* alone is 307,926, and *Catholicism nfd* 173,016 outweighs Roman
Catholic 295,743 by more than half.  A map that drew only the named denominations would be
showing about 60% of New Zealand's Christians.

## 4. Geography and vintage

**SA2 2023 (`SA22023_V1_00`), the vintage of the Statistical standard for geographic areas
2023, boundaries as at 1 January 2023.**  2,395 areas, and all 2,395 are present.

The trap here runs the *opposite* way to the Connecticut case in spec.md §8.1, and is worth
recording as the mirror image:

> "Address data from 2013 and 2018 Censuses was updated to be consistent with the 2023 areas."

So the 2013 and 2018 columns in this same table are **also on 2023 boundaries**, not on their
own.  Joining the 2018 column to SA2 2018 boundaries would be wrong, even though "match the
data's vintage" is the rule that usually saves you.  The rule is really *match the geography
the publisher coded the data to*, and here the publisher recoded history forwards.  Stats NZ
also warns that this makes 2013/2018 counts published in 2023 differ slightly from the ones
published at the time.

**Boundaries come from the same service**, layer 0, which is the identical table clipped to
the coastline and carries geometry — the NZ equivalent of choosing `cb_` over `tl_` for the US
so dots do not scatter offshore:

    .../2023_Census_totals_by_topic_for_individuals_by_SA2/FeatureServer/0/query
        ?where=1%3D1&outFields=SA22023_V1_00&returnGeometry=true&f=geojson

Generalised boundaries are also on `datafinder.stats.govt.nz` (layer 111218), which wants a
free API key; the feature service does not, so there is no reason to use datafinder.

**On placement (spec.md §8.2):** SA2s hold a median of 2,115 people, IQR-wise they are of the
same order as US census tracts, and the design targets are 2,000–4,000 in urban areas but only
500–3,000 in rural ones — so scattering an SA2's dots uniformly is a *rougher* population
weighting than the US tract trick, not an equal one.  **79 SA2s have zero population** and 141
have under 100; these are inlets, harbours, oceanic areas and forest parks, and they are the
same ones that carry the suppressed cells in §7.

## 5. Multiple responses — measured, twice

**A person could report up to four religions, and every count in both tables is a *total
responses* count.**  Stats NZ: "Religious affiliation is a multiple response variable so the
number of responses can be greater than the number of respondents", and "If more than one
religion is reported, each response up to a maximum of four responses is counted."

The size of it, which is the spec.md §3.1/§3.3 question, and the answer is **it depends on the
level, and by a factor of four**:

| | people | responses | inflation |
|---|---:|---:|---:|
| 2023, level 1, summed over SA2s | 4,993,920 | 5,003,112 | **+9,192 = 0.18%** |
| 2018, level 1, summed over SA2s | 4,699,731 | 4,708,512 | **+8,781 = 0.19%** |
| 2018, level 3, national, from Stats NZ's own aggregate row | 4,699,755 | 4,732,641 | **+32,886 = 0.70%** |

That is the whole nesting story in one table.  At level 1 a person who answered "Anglican and
Catholic" is one Christian; at level 3 they are two rows.  So **the deeper the classification,
the worse the double counting**, and the 2018 file's own `TotalResponse` row (4,732,641)
against its `Total` row (4,699,755) is Stats NZ saying so explicitly.

Per family, 2018, national level-3 roll-up against the level-1 SA2 sum for the same year:

| level-1 family | L3 categories | L3 roll-up | L1 sum | gap |
|---|---:|---:|---:|---:|
| No Religion | 1 | 2,264,601 | 2,264,487 | **−0.01%** |
| Judaism | 4 | 5,274 | 5,268 | −0.11% |
| Hinduism | 5 | 123,534 | 123,336 | −0.16% |
| Buddhism | 7 | 52,779 | 52,656 | −0.23% |
| Other Religions | 30 | 91,239 | 91,020 | −0.24% |
| Spiritualism and New Age | 13 | 19,695 | 19,506 | −0.96% |
| **Christian** | 87 | 1,738,638 | 1,717,164 | **−1.24%** |
| Islam | 6 | 61,455 | 60,666 | −1.28% |
| **Māori Religions** | 5 | 62,634 | 61,605 | **−1.64%** |

**No Religion is the control and it works**: one level-3 category means no room for a second
response inside the family, and the two figures agree to 0.01% — which is also the proof that
the rest of the column is a real effect and not a join error.  21,474 people gave two or more
different Christian denominations; 1,029 gave two or more Māori religions, which will be
Rātana and Ringatū; 789 gave more than one Islamic answer.

**Which level was kept, and why both.**  `data/normalized/nz.csv` carries the SA2 rows at
level 1 and the national rows at level 3, and they are *not* nested inside each other in the
file — different `year`, different `source_id`, different `geo_level`.  Nothing in the file
should ever be summed across those two groups.  The intended use is spec.md §3.4: 2023
level-1 SA2 totals, split by 2018 national level-3 shares, tagged `structure_year: 2018,
total_year: 2023`.  Doing that split will need the level-3 shares to be scaled down by the
per-family gap above, or the split parts will over-sum their parent by up to 1.6%.

## 6. What is behind a key, and what it would buy

**Aotearoa Data Explorer (`explore.data.stats.govt.nz`) has 2023 religion at level 3, and its
API needs a free subscription key from `portal.apis.stats.govt.nz`.**  Not attempted further,
per the standing rule about auth walls.

What is reachable without one: `https://api.data.stats.govt.nz/rest/dataflow/all/all/latest`
returns the full 911-dataflow catalogue unauthenticated.  Every other route — a specific
dataflow, a data query, `references=all` — returns `401 missing subscription key`.  The
catalogue is still worth having, because it names exactly what a key would unlock:

| dataflow | what | geography |
|---|---|---|
| `CEN23_ECI_025` | Religious affiliation **(total responses level 3)**, age, gender, 2013/2018/2023 | RC, TALB, Health |
| `CEN23_ECI_017` | Religious affiliation (total responses **common religions**), ethnicity, gender | RC, TALB, **SA2**, Health |
| `CEN23_ECI_024` | Religious affiliation (common religions), birthplace, age | RC, TALB, **SA2**, Health |
| `CEN23_MAO_077` | Iwi affiliation × religious affiliation **level 2**, Māori descent population | RC, TALB, Health |
| `CEN18_ECI_027` | the 2018 equivalent of ECI_025 | RC, TA, **SA2**, DHB |

So a key would buy **2023 level 3 at territorial authority** (67 TAs plus local boards) and
**2023 "common religions" at SA2** — a mid-depth list, more than the 11 we have.  That is the
single largest available improvement to New Zealand's coverage and it costs a registration.
The table of what exists comes from Stats NZ's own
[2023 Census product and release finder](https://www.stats.govt.nz/tools/2023-census-product-and-release-finder/)
(an .xlsx listing all 757 published products, geography by geography) — which is the right
tool to check first for any country that has one.

## 7. Confidentiality, and the sentinel that eats 4% of the country

Counts are **fixed random rounded to base 3 (FRR3)**, independently per cell, and counts under
six are suppressed where the table is fine-grained.  Two consequences:

1. **Rows do not add to their own totals.**  Expected, declared by Stats NZ, and small: across
   2,395 units the noise on a national category total is on the order of ±40.
2. **Suppressed cells are negative numbers in the same integer column as the counts** —
   `-999 Confidential`, `-997 Not available` — documented only in the layer's description
   blob.  For 2023 there are **1,326 suppressed cells across 112 of the 2,395 SA2s**, all in
   areas with 0–15 residents.

The second one is the trap, and it is a silent one.  Summing the column as delivered gives:

    Islam                     -36,753
    Hinduism                   32,946
    No Religion             2,472,093    (published: 2,576,049)

Islam goes negative, which is at least visible; No Religion comes out 4.0% low, which looks
entirely plausible and is wrong.  `sources/nz.py` drops the sentinels and reports how many it
dropped.  Because each suppressed cell is a count under six, the national sums are low by at
most 6,630 people (0.13%) and in practice by far less — No Religion, once the sentinels are
out, lands 60 short of the published figure across 104 suppressed cells.

## 8. Reconciliation

**2023 SA2 sums against Stats NZ's published national figures.**  Population reconciles
exactly: the 2,395 SA2 "Total" cells sum to **4,993,920** against the published census usually
resident population of **4,993,923** — three people, which is one FRR3 step.  Every category
share matches the published percentage to the published decimal:

| | this file | Stats NZ published |
|---|---:|---:|
| No Religion | 2,575,989 = 51.58% | 2,576,049 = 51.6% |
| Christian | 32.33% | 32.3% |
| Hinduism | 2.90% | 2.9% |
| Islam | 1.50% | 1.5% |
| Object to answering | 6.86% | 6.86% |

**2018 SA2 sums against Stats NZ's own 2018 national table**, which is the stronger check
because both sides are Stats NZ and neither is a rounded percentage: population 4,699,731
against 4,699,755 — **24 people, 0.0005%** — and No Religion 2,264,487 against 2,264,601, 114
people or **0.01%**, over a category with only one level-3 child so nothing can hide in it.
The remaining families differ by the multiple-response gap tabulated in §5, which is the point
of that table.

Nothing here needed a fudge, and nothing was dropped except the 1,326 suppressed cells, which
are counted in the script's own output.

## 9. What self_id is doing here, and the one thing that should worry us

Every row is `self_id`.  It is a census question about the respondent's own religion, so the
basis is not in doubt — but the *provenance* is, and Stats NZ says so plainly on the DataInfo+
page for the variable:

| source of the 2023 religious affiliation answer | share |
|---|---:|
| 2023 Census response | **84.4%** |
| a **2018** census form | 6.1% |
| a **2013** census form | 3.0% |
| statistical imputation (probabilistic, or CANCEIS donor) | **6.5%** |
| admin data | 0.0% |
| no information | **0.0%** |

**15.6% of New Zealand's 2023 religion data is not a 2023 answer**, and 6.5% is not an answer
at all — it is the religion of the household member closest in age, or a donor record.  This
is why "Residual Categories" is exactly zero in all 2,395 SA2s and why "Total stated" is 78
short of the population instead of hundreds of thousands: the not-stated slice that spec.md
§3.2 expects to compute has already been filled in by the statistical agency, invisibly and
without a flag.

The consequence for spec.md §7: New Zealand should not be drawn at full saturation on the
strength of being a census.  Something between `measured` and `derived` is honest, and the
unit panel should be able to say "6.5% imputed" the way it says a year.  A country that fills
its non-response by imputation and a country that reports it as "not stated" look identical in
the output file and are not the same claim.  Left as a decision for Anita rather than encoded,
since it affects the confidence tier definitions and not just New Zealand.

## 10. Surprises, in order of how much they cost

1. **The −999 sentinel.**  Half an hour spent believing the whole country was 4% short of its
   published figures.  A negative Islam is what gave it away; had the suppressed cells been
   coded 0 the error would have shipped.
2. **Depth and geography never coincide.**  Every other decision in this file follows from it.
3. **Backwards vintage.**  2013 and 2018 counts recoded onto 2023 boundaries, so the §8.1 rule
   has to be read as "the geography the publisher coded to", not "the geography of the year".
4. **The residual is already gone.**  Not stated is 0 everywhere, because it was imputed away.
5. **Jedi has an official code** and 20,409 adherents, which is more than any Orthodox
   jurisdiction in New Zealand.
6. **Atheists are filed under "Other Religions"**, structurally separate from "No Religion".
7. **The product and release finder** is a genuinely good artefact — one .xlsx listing all 757
   published 2023 products with their geography and variables.  Finding it first would have
   saved most of the search; it is the thing to look for in any national statistics office.
