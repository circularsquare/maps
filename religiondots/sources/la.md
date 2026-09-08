# Laos — Lao Statistics Bureau, Population and Housing Census 2015, at village level

`sources/la.py` → `data/normalized/la.csv`
`sources/la_geo.py` → `data/geo/la/la_units.gpkg`, `la_lookup.csv`
`sources/la_grid.py` → `data/geo/la/la_hexes.gpkg`
`taxonomy/la2015.py` → the mapping

**8,499 villages, 6,481,482 people, six categories, 99.83% of the census.** The finest
counting geography of any mainland Asian country on this map: 763 people per unit, against
Nepal's 38,400, India's 202,000 and Cambodia's 622,000.

Anita found the printed dot map in the 2008 atlas and asked whether its source was usable.
It was.

---

## 1. Laos was closed, and the verdict was about the report rather than the office

`sources.md` §9an closed Laos on 2026-09-07:

> **Laos.** The 2015 PHC report's `Table P2.9` is **national only** — no geography at all —
> with Buddhist 64.7%, Christian 1.7%, no religion 31.4%. […] **Park it.**

Every word of that is still true. `Table P2.9` is the whole published religion output of the
2015 census and it has no geography. LAOSIS is a KOSIS-platform database with no religion
dimension. What the paragraph could not know is that **LSB publishes the same census at
village level somewhere else**, and has since 2021.

The channel is the 2008 atlas's own data platform, still running. *Socio-Economic Atlas of
the Lao PDR — an analysis based on the 2005 Population and Housing Census* (Messerli,
Epprecht, Minot, Souksavath, Chanthalanouvong and Heinimann; NCCR North-South and Geographica
Bernensia, 2008) came with **DECIDE Info**, a spatial data service for the census. DECIDE
became **K4D** (`k4d.la`), run by the Ministry of Agriculture and Forestry with Swiss
Development Cooperation support, and the ArcGIS server behind it at `gis.cde.unibe.ch`
carries **1,365 services**, roughly 264 of them 2005 and 400 of them 2015.

This is [[feedback_nothing_is_truly_dead]] and [[reference_dead_stats_office]] in a new
shape. The office had not moved and was not blocked. **A second publication channel existed
that nobody had looked for**, because the search had been for a table and this is a map
server.

## 2. How to find it again, and what the discovery path actually was

Worth writing down because none of it needed a key or an account:

1. The printed map's credit line is `© NCCR North-South, DOS, LNMCS, 2008`. Searching the
   atlas title reaches `decide.k4d.la`, which 301s to `k4d.la`.
2. `k4d.la` links a **Data Catalogue** at `en.data.k4d.la`. It is an **ArcGIS Hub** site, so
   its HTML is a shell and its dataset list is not in the page ([[reference_spa_hidden_apis]]).
   `hub.arcgis.com/api/v3/domains/en.data.k4d.la` resolves the domain to
   `orgId=Hkc3bAqSR3WGz2YJ`, `orgKey=UNIBERN`.
3. `arcgis.com/sharing/rest/search?q=orgid:<that>` lists the items. Each carries the service
   URL, the licence and a description naming LSB.
4. `gis.cde.unibe.ch/gis/rest/services/Decide?f=json` lists all 1,365 services directly, which
   is faster than the item search and is what `sources/la.py` documents.

**Licence, off the item metadata:** *"With proper citation of the data sources, the data can
be used freely."* Attribution only. Every item's description names `Lao Population and
Housing Census 2015` and gives LSB's postal address and `lstats@lsb.gov.la`.

## 3. Six categories off five services and a residual

| religion | service | published as |
|---|---|---|
| Buddhist | `laos_2015_distribution_of_buddhists` | count, `UNPReAB27` |
| Christian | `laos_2015_distribution_of_christians` | count, `UNPReAB28` |
| Muslim | `laos_2015_distribution_of_muslims` | count, `UNPReAB30` |
| Baha'i | `laos_2015_percentage_of_bahais` | percentage, `urpreab36` |
| No religion | `laos_2015_percentage_of_no_religion` | percentage, `URPReAE26` |
| Others/not stated | — | **residual** |

There is no sixth service. `laos_2015_percentage_of_population_with_other_religion` exists
and is **misnamed**: its only attribute is a string holding the village's dominant-religion
class (`Buddhist 80-99%`, `Village not dominated by a group of more than 80%`). Its own item
description confirms it, being a truncated legend rather than a variable definition. So
`Others/not stated` is derived as each village's population less the five published
categories.

**The residual is not slop.** It is non-negative in all 8,499 villages, and adding the Muslims
and Baha'is back gives 137,020 against Table 3.5's printed `Others/not stated` of 137,640 —
99.55%, the difference being the villages missing from the file (§5).

**The percentage services carry eight significant digits**, which turns a convenience into a
check. `0.70600098%` of 1,983 is 13.999999, so the integer count is recovered exactly rather
than apportioned. Contrast Cambodia, whose one decimal place puts a ±0.0005 × population band
on every cell. And it means the percentage service and the population service can be asserted
to be describing the same village: a percentage read against the wrong row would miss an
integer by an arbitrary amount. **It lands on an integer in 16,997 of 16,998 cells.**

The one miss is LSB's own and is named in the module. `B. Nongbua` in Salavan
(`VCODE 1403002`) has a Baha'i share of `0.5533597%`, which is `7/1265` exactly while the
population service gives the village 1,244. Seven people either way. The band stays at 1e-3
and the cell is whitelisted, because widening it to 0.12 would admit a genuinely misjoined
row on any village under about 2,000 people.

## 4. The check is `Table 2.3`, and it is unusually sharp

There is no name join and no p-code join in this country: the religion figures are attributes
**on** the polygons. So the usual §12 reconciliation is unavailable and something else has to
do its work.

Table 2.3 of the published report (p.29, sourced there to `Table P1.2`) gives district count
and population for the 18 provinces. Aggregating the villages by the **province prefix of
their code**:

- **148 districts against 148 published**, in all 18 provinces;
- **14 of 18 provinces reproduce the published population to the person**;
- the other four are Savannakhet −7,324, Vientiane Capital −1,474, Khammuane −1,388 and
  Phongsaly −560;
- **nothing anywhere is in excess.**

One-signed differences are what a coverage gap looks like. A bad join produces both signs.

**Twelve villages have moved district since their code was minted, and two of them are why
the check uses the code rather than the attribute.** `B. Phou pard` (770) and `B. Phou lar`
(506) carry Louangnamtha `VCODE`s and the file places them in Namor district, Oudomxai.
Aggregating by the `PCODE` attribute therefore puts Louangnamtha 1,276 short and Oudomxai
1,276 over, and aggregating by the code prefix reconciles both exactly — 1,276 = 770 + 506.
**The attribute is right about where the village is and the code is right about where the
report counted it**, so the check uses the code and the drawn geography uses the polygon.

## 5. The gap: 10,746 people, 0.17%

Villages the census enumerated and the K4D layer does not carry. Concentrated as above, and
Savannakhet is two thirds of it. Nothing published says why; the likeliest reading is villages
whose GPS point was never captured, since the whole layer is built around those points (§6).

Drawn nowhere and stated in `countries.py`'s `gap=` row, per §3.5.

## 6. The polygons are travel-time catchments, not boundaries

Laos has never had official digital village boundaries. What the 2005 census had was **a GPS
point for each of 10,547 villages**, and CDE grew polygons around them with an accessibility
model — travel time rather than straight-line distance, which in that terrain is a large
difference. Section A.7 of the atlas describes the construction and adds:

> this atlas is by no means intended to be used as a planning tool at the level of single
> villages

So a polygon is the territory whose nearest village is that one, and the village's people are
at the point. They tile the country: **230,548 km² against Laos's 236,800, 97.4%**.

That is exactly §8.2's case for a population grid, and `la_grid.py` supplies one. The area
distribution is the argument: median 14.3 km², but p95 is 93.8 and **96 polygons over 200 km²
hold 68,769 people across 13.9% of the country**, and those are the upland districts of
Phongsaly, Houaphan, Xekong and Attapeu, which is exactly where the interesting cell is.

**Kontur's Laos extract is thin.** 69,696 hexes is about 9,600 km², so it covers the built-up
4% of the country. 8,134 villages (95.7%) get at least one populated hex; **365 get none**
(267,465 people, 4.13%) and fall back to their own polygon as a single placement shape, which
is an equal-share wash inside that village. Marked `src='polygon'` in the layer.

Both nulls discriminate, and both are far weaker than Nepal's for a reason that is about the
grain rather than the join:

| | real | null |
|---|---|---|
| outside a factor of 3 | 1,072 of 8,134 (13.2%) | shuffled median 2,995; 0 of 200 shuffles under 1,300 |
| r on log populations | 0.5906 | best of 2,000 shuffles 0.0450 |

Median ratio 0.97, quartiles 0.63–1.35. A village with one Kontur hex reading `1` against a
census `1,205` is a single 400 m cell being asked to carry a settlement's level, and **the
level being wrong there does not move a dot**: a unit with one polygon puts all its dots in
it whatever weight it carries. Kontur finding exactly one built-up cell in a mountain
catchment is a statement about *where*, which is the only thing the weight is used for.

## 7. The 31.45%, which is the whole argument

The census's own English label is `no religion`. It is **not** filed on `unaffiliated`, and
`taxonomy/la2015.py` and the `indigenous.laos` entry in `branches.py` carry the case. In
short:

1. **The instrument.** The atlas, §F.5: *"In the National Population and Housing Census of
   2005 religion was defined as any spiritual system with written doctrines. According to
   this definition only Buddhism, Christianity, Baha'i and Islam are therefore identified as
   religions."* Animism was defined out of the category, not measured and found absent.
2. **The publisher's own gloss**, same page: *"it might be suggested that a more appropriate
   term for the 'other' category would be Animism. The majority of non-Lao ethnic groups are
   essentially Animists."*
3. **The census report's summary**, 2015 PHC p.5: *"32 percent reported themselves as having
   no religion or being animist."* And the Lao subtitle on LSB's own `no religion` service
   reads *"following other religions or not following any religion"*.
4. **Pew**, *How the Global Religious Landscape Changed From 2010 to 2020* (2025), p.190
   table: Laos's **religiously unaffiliated `<0.1%`** and **other religions 34.2%, 2,510,000
   people** — the tenth-largest `other religions` population in the world. This is spec
   §3.11's first bullet: an external national estimate naming a category the census refuses
   to, and it bounds the genuinely non-religious part at a few thousand people.
5. **The geography.** Dakcheung 96.5%, Samuoi 93.1%, Ta Oi 92.1%, May 92.0%, against
   Vientiane Capital **5.9%** and Champasak **2.0%**. Irreligion concentrates in cities; this
   is at its minimum in the only city.
6. **The ethno-linguistic cross-tab**, from LSB's own ten category layers on the same 8,499
   villages: **9.2% among Lao-Tai against 65.1% Mon-Khmer, 79.1% Hmong-Mien, 77.4%
   Sino-Tibetan.** That is ecological and is offered as shape rather than measurement. The
   non-ecological version needs no assumption: **84.5% of the 2.04 million are in villages
   that are less than half Lao-Tai.**

### 7a. Two tests run afterwards, on Anita's question of whether 31% irreligion is realistic

Both come off data already on disk and both are stronger than anything in the list above.

**The 2005 census called the same cell `another religion`, and the box was renamed without
the people changing.** 2005: 31.04%. 2015: 31.45%. That could be coincidence at national
level, so it was checked per district — the 2005 `another religion` share against the 2015
`no religion` share across the **137 districts present in both censuses**:

    r = 0.9714,  and the median district moved -0.0 points between the two censuses

**A census does not relabel a box and find the new label distributed exactly like the old one
across 137 districts unless it is the same population.** In 2005 these people were recorded
as *following another religion*; the only thing that happened between the censuses is the
word on the form. The few districts that did move are named in the printout and are mostly
boundary changes (Thapangthong +48, Ngeun +52, Sanamxay −22).

**And the urban gradient runs backwards inside Vientiane Capital**, which is the one place in
Laos where a secular population could exist:

| district | population | `no religion` |
|---|---:|---:|
| Xaythany (outer, northern hills) | 196,565 | **11.99%** |
| Sangthong (rural west) | 29,509 | 8.90% |
| Naxaithong (outer north-west) | 75,228 | 8.78% |
| Mayparkngum | 49,211 | 5.49% |
| Sikhottabong | 120,999 | 3.73% |
| Sisattanak (inner) | 64,723 | 3.11% |
| Xaysetha (inner) | 116,576 | 3.03% |
| **Chanthabuly (the city centre)** | 69,046 | **2.41%** |
| Hadxaifong | 97,609 | 0.89% |

**A thirteen-fold gradient inside one municipality, and it is at its minimum in the historic
core.** Irreligion has never been measured anywhere on earth with that shape.

**What the real figure is, is unmeasured.** Pew bounds it at `<0.1%` nationally; the inner
Vientiane districts put a ceiling of about 2-3% on the one population that could plausibly be
secular, and even that is a ceiling because those districts hold upland migrants too. Laos is
in no comparative survey series that asks — not WVS, not the Asian Barometer — so nothing
measures it directly and this map does not draw a figure for it.

**Read it as a ceiling**, which reverses the reading every other node in the `indigenous`
family carries. Vanuatu's kastom and Myanmar's animist box stood beside the churches on the
form, so they undercount everyone who keeps both; this is a residual, so it also holds
whatever genuine irreligion Laos has.

**The alternative was `unknown`** (§6.3a-ii), Vietnam's node for a residual of the same kind
one border away. Rejected because the evidence above is a good deal more specific than *what
they practise is not determinable*, and because a grey block over the Lao uplands would say
nothing about one of the sharpest religious geographies on this map. **It is one line in
`taxonomy/la2015.py` to reverse and nothing else depends on it.**

## 8. Ethics (§14)

Laos regulates religious practice under **Decree 315 (2016)**, which requires registration and
central approval for congregations, construction and foreign contact. The US State
Department's annual religious freedom reporting documents village-level closures of Protestant
congregations and detentions of members, concentrated in the same northern and southern
uplands where this map draws both the Christian and the traditional-religion cells. **Both
should be read as floors** for that reason, and the `Christian` note in `taxonomy/la2015.py`
says so.

§14's question of whether to draw at all does not bite here: the state published these figures
itself, at village level, under an attribution-only licence, and the category that carries the
ethical weight is the one it declined to name rather than one it named and suppressed.

## 9. Not done

- **2005 is available at the same server and is not built.** 10,520 villages, 5,602,111
  people, five categories (`percentage_of_buddhists`, `_christians`, `_muslims`,
  `_bahai_follower`, `_follower_of_another_religion`) summing to 99.96%. It is the year the
  printed atlas map draws, and it would give Laos a **two-census time series at village
  level**, which almost nothing here has. The 2005 categories have no `no religion` cell at
  all: the residual is `another religion` at 31.04%, which is the cleanest evidence in the
  whole file for §7's reading.
- **Savannakhet has its own service folder** (`SVK/svk_2015_*`), a separate publication of
  the same data for one province. Unused; it would be an independent parse check.
- **The 2011 agricultural census** is on the same server with ethnicity per village, and
  1995 is not.
