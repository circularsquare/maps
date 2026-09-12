# Slovakia — Štatistický úrad SR, SODB 2021, religion by obec

**2,927 municipalities, 5,449,270 people, 11 categories, all of them drawn — 100% of the
country.** Built 2026-09-08 from `gis.scitanie.sk`, the census's own ArcGIS server.

`sources/sk.py` (counts), `sources/sk_geo.py` (polygons and the placement grid),
`taxonomy/sk2021.py` (the mapping), `sources.md` §11aa (how it was found).

---

## 1. Four sweeps recorded this country as walled, and the route is none of the ones they tried

`sources.md` §11 ("Slovakia, re-examined 2026-09-04 — three walls, each now named"), §11c,
§11k and §11o's *"untouched; four sweeps stand"* all went at two hosts:

| host | 2026-09-08 | |
|---|---|---|
| `slovak.statistics.sk` | **403** | still walled, as every sweep said |
| `datacube.statistics.sk` | **open** | and `data.statistics.sk/api/v2/collection` is a keyless JSON-stat endpoint with **678 datasets and no religion**. SODB 2021 is not in DATAcube at all. |
| `www.scitanie.sk` | **open** | the census's own site, and it links the thing nobody looked for |
| **`gis.scitanie.sk`** | **open** | **a public ArcGIS Server, 86 hosted services, no key, `capabilities: Query`** |

The generalising bit is not about Slovakia: **a statistical office's census results and its
statistical database can be two different products on two different hosts, and the sweep that
asks only "is the office reachable" answers the wrong question.** Every sweep here correctly
established that `slovak.statistics.sk` was shut, and the census had never been behind it.

## 2. There is no join, which is a first

    https://gis.scitanie.sk/server/rest/services/Hosted/obyv_ekchar_nabo_vekskup/FeatureServer/4

Layer 4 is `AR4318_obec_t_SK`: **2,927 polygons, each carrying its own religion counts as
fields**. Every other country on this map pairs a counts table from one publisher with
boundaries from another, and §12's two commonest failure shapes — a silent drop and a
confident wrong pairing — are both about that pairing. Here the unit a number belongs to is
not asserted by this build at all.

Peer layers, all in the same service, all free:

| layer | | n |
|---|---|---:|
| `/4` `AR4318_obec_t_SK` | **religion + polygons, obec** | **2,927** |
| `/5` `AR4318_okres_t_SK` | the same by district | 79 |
| `/3` `AR4318_kraj_t_SK` | the same by region | 8 |
| `/1` | religion × age group, obec, points | 50,265 rows |

and in `hranice`, unused here but worth knowing about: **`casti_obci` 4,114** and **`zsj`
7,246** boundary polygons — Slovakia publishes geography three levels below the drawn one.

## 3. The partition is exact and witnessed three times

1. **Within the layer.** The eleven category columns sum to `spolu`, per obec, on all 2,927
   rows. Nationally: **5,449,270, difference 0.**
2. **Across tiers.** Fetching the 8 kraje from layer 3 — a separate request against a
   separate layer — reproduces the total and all eleven category totals exactly.
3. **Across publishers.** UNSD Demographic Yearbook table 28 carries SODB 2021 independently
   and agrees **to the person** on the total and on all nine named churches.

## 4. `ostatné` is 7.83%, is drawn, and is not what its name says — the one real limit

The service publishes nine named churches, `bez náboženského vyznania`, and one residual
`ostatné` of **426,496 people**. That is too large to be other religions, and the third
witness above is what settles what is in it. UNSD's version of the same census has **21**
categories and breaks the tail out:

```
Not Stated                        353,797   6.493%
Other Religions                    64,990   1.193%
Christian                          10,811
Baptists Fraternity Union           3,883
Fraternity Church                   3,440
Seventh Day Adventist               3,001
Jewish                              2,007
Old Catholic Church                 1,778
Czechoslovak Hussite Church           581
Church of Jesus Christ of LDS         377
Baha'i                                311
New Apostolic                          73
                                  -------
                                  445,049
```

and **`Kresťanské zbory` (18,553) + `ostatné` (426,496) = 445,049**, exactly. So the two
publications agree on the mass and partition the tail differently, and `ostatné`
demonstrably contains the 353,797 who did not state a religion. `sk.py` asserts that identity
on every run.

**It was first built EXCLUDED on spec §3.5, and Anita reversed that on 2026-09-08:** *"we
should definitely draw these points as an other(slovakia), tons of countries have an other."*
The argument for drawing is **§6.12** — excluding it left 7.83% of Slovakia as a hole, and a
hole on a dot map reads as an absence of *people*, not of work. 426,496 people who exist and
were counted were being shown as nobody. So it goes to **`other.sk`**, and the country is
100% drawn.

> **The node is labelled `Other or not stated (Slovakia)`, not `Other religion`, and it is
> not comparable with `other.ro`, `other.me` or `other.mw`.** Those are residuals of
> religions somebody named. This one is that *plus* the non-response, merged by the publisher
> before release, and the non-response is five parts in six.

> **This is still not Kazakhstan's case (§9aq) even though both end up drawn.** There an
> offered `Отказываюсь указать` box that people actively ticked was drawn *as a refusal*,
> because an offered answer somebody picked is an answer. Slovakia's form has no refusal
> option — `nezistené` is what the office computes for a form left blank. Drawing it here is
> a §6.12 judgement about holes, not a §3.5 reclassification of non-response into an answer.
> Nothing in either tabulation distinguishes the two cases; only the form does
> ([[reference_census_questionnaire]]).

### The mix is not constant, and that is what would mislead

Measured on the drawn data, the cell runs **0.0% to 58.8%** between municipalities — median
3.8%, p90 8.9%, p99 15.1% — and the top of the list is not a religious geography at all:

| | people | `ostatné` |
|---|---:|---:|
| **Košice – Luník IX** | 7,037 | **58.8%** |
| Pavlovce nad Uhom | 4,620 | 28.7% |
| Jasov | 3,535 | 26.9% |
| Hnúšťa | 6,762 | 18.0% |
| **Bratislava – Staré Mesto** | 46,080 | 16.4% |
| Košice – Staré Mesto | 20,133 | 15.2% |
| *against* Lendak, Zákamenné, Rabča, Skalité (Orava/Kysuce) | | **0.7-2.7%** |

Those peaks are Slovakia's Roma settlements and its two city centres — places where a census
form comes back unanswered — so this node is **mostly *did not answer* where it is dense and
mostly *some other faith* where it is sparse**. Its density is a map of the census's reach.
`note_public` says so on the map, and it is why the national 83/17 split must never be applied
per unit to try to separate the two halves.

**What is folded in, stated rather than buried:** about **1.33%** of Slovakia is genuinely
other religions — Baptists, the Fraternity Church, Adventists, Jews, Old Catholics,
Czechoslovak Hussites, Latter-day Saints, Bahá'ís, New Apostolics. **Every one of those
bodies is already on this map's tree**, so if the office ever publishes its 21 categories by
obec they come back for free and `other.sk` shrinks to the non-response alone. That is the
single best upgrade available to this country.

Routes tried and closed for that finer table, 2026-09-08, so nobody walks them twice:
`data.statistics.sk` JSON-stat (678 cubes, no religion); `datacube.statistics.sk/api/v1/...`
(404); `data.gov.sk` and `data.slovensko.sk` (both SPAs, no CKAN JSON at the usual paths);
`www.scitanie.sk`'s three JS bundles (**no API path of any kind** — the SPA reads the same
ArcGIS services, so the GIS portal is the front door, not a back one); and the site's own
**OpenData menu items, which have empty `href`s** — advertised and never wired up.

## 5. Placement is measured, and it is still not an independent check

`obyv_grid_1km` layer 2 is SODB 2021 redistributed to **49,969 1 km cells**, and
`sum(obyv_tp_all)` over the whole layer is **5,449,270 — the census total to the person.** So
Slovakia joins Germany (§9g) as a country whose placement is measured rather than modelled,
and no Kontur extract is used.

> **But §9av's rule applies with full force.** The Central African Republic taught that a
> population grid can be downstream of the census it is checked against; here it is not merely
> downstream, it **is the same enumeration**. The grid-vs-census ratio band that catches a
> scrambled name join everywhere else **can say nothing about the counts** in Slovakia. It
> validates the cell-to-obec assignment and nothing else, and `sk_geo.py` prints that caveat
> beside the number so it cannot be read as a data check.

## 6. me_geo.py's centroid rule is wrong here, and it failed loudly

The established pattern (Kenya, Montenegro, Mauritius) assigns a grid cell to the unit
containing its centre. **Slovakia has obce far smaller than a 1 km cell**, so a cell whose
centre lands in a hamlet credits that hamlet with the cell's whole population:

| obec | census | centroid rule | ratio |
|---|---:|---:|---:|
| Záborie | 170 | 1,409 | **8.3x** |
| Mošurov | 180 | 1,338 | **7.4x** |
| Krížovany | 351 | 2,068 | 5.9x |

which would have pulled a neighbouring town's dots into a village. **Cells are split by area
of intersection instead**, each piece taking its share of the cell's population — 87,240
cell/obec pieces, and the in-band fraction goes from 95.7% to 98.3%.

**And then renormalised against the area inside Slovakia.** Dividing by the full cell area
throws away the part of a border cell lying over Austria, Hungary, Poland or Ukraine — and
with it 5,762 real Slovaks (0.106%), because a cell's people are all in Slovakia even where
its square is not. Normalising against the area actually inside the country places
**5,449,270 of 5,449,270, nothing lost.**

Six obce contain no grid-cell centre at all and would have had nowhere to put their dots;
§9af's Mauritius rule says `place_weight`'s equal-share fallback cannot fire for a unit
absent from the placement layer, so each is given its own polygon as a one-cell placement
area. After the switch to area splitting none are left, but the guard stays.

## 7. Three ArcGIS traps, all live

1. **`maxRecordCount` is 2000 and the layer has 2,927 rows.** An unpaged query returns 2,000
   of them with `exceededTransferLimit: true` buried in the response body and **no error**.
   Every per-row check passes on a Slovakia missing a third of its municipalities. §5a in a
   new disguise. `sk.py` asserts the paged count against the layer's own `returnCountOnly`
   endpoint, never against the length of one response.
2. **`supportsPagination` is not advertised and `resultOffset` works anyway.**
   `advancedQueryCapabilities.supportsPagination` is absent from the service metadata;
   `resultOffset=2000` returns exactly the remaining 927. *Do not read an unadvertised
   capability as an absent one.*
3. **`supportedQueryFormats` says `JSON` and `f=geojson` works**, honouring `outSR=4326`. The
   service's own CRS is Web Mercator 3857.

A fourth, smaller: the **kraj layer carries no `cislo`/`kraj`/`okres` fields** and asking for
them is a **500**, not an empty column — so the field list has to be per layer.

## 8. TLS: certifi, not `verify=False`

`gis.scitanie.sk` fails on the system certificate store with `unable to get local issuer
certificate` and **verifies fine against `certifi`'s current bundle**. That is the middle
case of the three this project has met: not `gh.py`'s server-omits-its-intermediate (where
curl, urllib and certifi all fail together and there is no option but to turn verification
off), and not §9h's one-client-only. Use `certifi.where()`.

## 9. What the country shows

- **The sharpest church/no-church contrast across a land border on this map.** Slovakia is
  **55.8% Roman Catholic**; Czechia, which it was one state with until 1993, is 7.0%.
- **The Lutherans are the middle of the country** — 5.3%, the church of the Slovak national
  revival, in Turiec, Liptov, Gemer and the Zvolen basin rather than the Catholic west.
- **The Greek Catholics are the east** — 218,235 people, 4.0%, **the largest Byzantine-rite
  Catholic population on this map after Romania's**, in Prešov and the Rusyn villages.
  Suppressed in 1950, restored in 1968.
- **The Reformed are Hungarian** — 85,271 in a strip along the southern border, the same
  church and the same minority as across the frontier in Hungary and in Romania.
- **No religion is 23.8%**, and it is not comparable with Czechia's 47.8% without care: see
  below.

## 10. What it cannot show

- **Which half of `other.sk` any given dot is**, per §4. It is drawn, so nobody is missing
  from the map, but 83% of it nationally is people who did not answer and the proportion
  varies from nothing to 58.8% between municipalities.
- **Anything below the obec**, though the office publishes 4,114 and 7,246 unit geographies
  for other variables.
- **`secular` and `unchurched` are unlit for Slovakia** (`coverage.py`, §6.12) and that is a
  property of the form: Slovakia offers one no-religion box, where Czechia offers *atheist*,
  *agnostic*, *deist* and *believing but belonging to no church* separately. **Spec §3.1a —
  compare the answer LISTS before reading a number across a border.** The 23.8%/47.8% gap
  between the two halves of former Czechoslovakia is real but is not all of it belief.
- **Where a religion sits inside an obec.** The placement grid is a population weight; a
  Catholic dot and a Lutheran dot spread identically within a unit.
