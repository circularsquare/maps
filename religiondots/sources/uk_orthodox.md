# England's Orthodox Christians, placed by country of birth

`sources/uk_orthodox.py` → `data/normalized/uk_orthodox.csv` (6,856 rows, one per English MSOA).

**`count` is a placement weight, not people.** The magnitude comes entirely from
`sources/uk_bes.md`; this file only says where.

---

## 1. Why Orthodoxy needs a proxy when nothing else does

It is the one leg in England's split that **neither placement source can find**.

| | |
|---|---|
| English Church Census 2005 | **49 Orthodox churches at a 7% response rate**, its worst by a factor of four. Zero of its 94 county-by-settlement cells reach three churches. |
| OpenStreetMap | **129 Orthodox churches in all of England.** |

Both undercounts have one cause, and it is not carelessness: **a 2005 survey and a map of
buildings are describing a country Orthodoxy had not yet arrived in.** The church census fell
on the Sunday after Orthodox Easter, when many were shut. Orthodox parishes here are mostly
post-2004; a great many rent an Anglican church on a Sunday afternoon and are mapped as the
Anglican church they meet in, and a congregation with no building of its own is usually not
mapped at all.

Meanwhile the British Election Study puts Orthodoxy at **2.17% of England's Christians, about
567,000 people — larger than the Baptists.** Placing that by OSM's 129 buildings put **6.0% of
Norfolk's Christians on Orthodoxy**, which is nonsense.

So the placement comes from the thing the 2021 census does record at fine geography and which
does track that population: where people born in Orthodox-majority countries live. Anita,
2026-09-07: *"okay we can place orthodox by origin."*

---

## 2. Two API limits, and telling them apart cost half an hour

ONS publishes country of birth at 60 categories down to local authority and at 190 categories
down to MSOA. Asking below MSOA returns:

```
HTTP 400  {"errors":[""]}          <- disclosure control refusing. A wall.
HTTP 403  {"errors":["Too many rows returned, ..."]}   <- a row cap. A queue.
```

**A 400 is a wall and a 403 is a queue**, and the empty error body on the 400 makes them look
alike. The 403 at MSOA is solved by asking for 200 areas at a time, 37 requests for all 7,264.
Server-side category filtering does not help: the cap counts rows before the filter.

**Nomis only has 60 categories, and 60 is not enough.** In the classification the bulk
downloads use:

- Greece sits inside `Other member countries in March 2001`, with the Netherlands and Sweden
- Bulgaria and Cyprus inside `Other EU countries`, with Czechia and Hungary
- Ukraine, Russia, Serbia and Moldova inside `Rest of Europe: Other Europe`, with Norway and
  Switzerland

**Romania is the only Orthodox country nomis separates.** Only `country_of_birth_190a`, which
is API-only, names the rest.

---

## 3. The weights, and why they are not optional

Born-in is not believes-in, and the error is not even across countries. Unweighted,
**Albania would carry 5.9% of England's Orthodox geography while being about 7% Orthodox** —
a Muslim-majority country whose diaspora here is substantially Kosovar. Weighted it carries
0.6%. That gap is the whole argument for the table.

Each origin is weighted by the Orthodox share of its own population, Eastern and Oriental
together because the anchor's single `Orthodox Christian` option does not separate them. What
each contributes to the final shape:

| origin | born in England | share of shape | unweighted would be |
|---|---:|---:|---:|
| Romania | 530,315 | **53.3%** | 46.4% |
| Bulgaria | 146,894 | 12.3% | 12.9% |
| Greece | 78,304 | 8.3% | 6.9% |
| Cyprus | 71,068 | 5.8% | 6.2% |
| Moldova | 54,994 | 5.8% | 4.8% |
| Russia | 54,942 | 4.5% | 4.8% |
| Ukraine | 38,945 | 3.4% | 3.4% |
| Eritrea | 35,641 | 1.7% | 3.1% |
| Ethiopia | 21,931 | 1.1% | 1.9% |
| Serbia | 11,532 | 1.1% | 1.0% |
| **Albania** | 67,966 | **0.6%** | **5.9%** |
| the rest | 25,105 | 1.9% | 2.7% |

The shares are each country's own census where it has a religion question (Romania 2021,
Bulgaria 2021, Serbia 2022, Moldova 2014, Georgia 2014, North Macedonia 2021, Bosnia 2013,
Albania 2011, Armenia 2011, Ethiopia 2007) and the conventional survey figure where it does
not (Greece, Russia, Ukraine, Belarus, Eritrea). Russia's uses the Sreda Arena atlas, the same
source `sources/ru.md` uses for Russia itself.

**Cyprus is the least certain number in the table.** The Republic is about 89% Orthodox, but
Britain's Cypriot population includes a large Turkish Cypriot minority that a birthplace
cannot separate, so it is set to 0.70. If that is wrong it is wrong by a few points on 5.8%
of the shape.

These are the only estimated numbers in England's split, they move the shape and not the size
— the anchor fixes the total either way — and `check()` prints the table above on every build.

---

## 4. What it produces, and how to read it

Tier is **`modelled`** (spec §7), the only leg in England's split that is. The other five are
`derived`: their coarse totals were counted, and only the placement is inferred. Here nothing
about Orthodoxy was counted at any level. The census counted birthplaces.

Where it puts people, as a share of each county's Christians:

| | |
|---|---:|
| Outer London | 6.5% |
| Inner London | 4.8% |
| West Midlands | 2.7% |
| Cambridgeshire | 1.5% |
| Cornwall | 0.7% |
| North Yorkshire | 0.5% |

Enfield and Haringey for the Greek Cypriots, Harrow and Wembley for the Romanians,
Peterborough and the fenland towns for the Romanians and Bulgarians in the agriculture. The
low end is as informative as the high: Cornwall and North Yorkshire are where this population
is not.

**A caveat the map cannot fix.** The proxy sees migrants and not their British-born children,
who are in the census as England-born. A second-generation Greek Cypriot in Enfield is
Orthodox and invisible to this variable. The effect is to place Orthodoxy slightly too much
where recent arrivals live and too little where the older diasporas settled, which for Greek
Cypriots is a fifty-year gap.

---

## 5. Re-fetch

Free, no key, no registration. The area list and the observations both come from the ONS
Census 2021 API. `scratchpad/fetch_cob.py` in the session that built this is the paging loop;
the shape of it is:

```
# 7,264 MSOAs, 500 at a time
https://api.beta.ons.gov.uk/v1/population-types/UR/area-types/msoa/areas?limit=500&offset=N

# observations, 200 areas at a time (403 above that)
https://api.beta.ons.gov.uk/v1/population-types/UR/census-observations
  ?dimensions=country_of_birth_190a&area-type=msoa,E02000001,E02000002,...
```

→ `data/raw/uk/msoa_country_of_birth.csv`. The OA→MSOA join comes from ONS Open Geography's
`OA_LSOA_MSOA_EW_DEC_2021_LU_v3`, an **exact-fit** lookup, in `data/raw/uk/oa_msoa.csv`.

## 6. Licence

Office for National Statistics, Census 2021, **Open Government Licence v3.0**, Crown
copyright. The same terms as everything else from ONS in this map, and unlike the English
Church Census this one can ship commercially.
