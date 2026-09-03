# Canada — Census of Population 2021 (Statistics Canada)

`source_id = ca_census_2021` · ingested 2026-08-30 · `sources/ca.py` → `data/normalized/ca.csv`

**Headline: Canada gives both halves of R2 but never in the same table.** The deep
classification — **168 categories, four levels, down to Old Order Mennonites, Mar Thoma,
Doukhobors, Wiccans and the Mission de l'Esprit Saint** — stops at province and metro area.
The fine geography — **5,161 census subdivisions and 6,247 census tracts** — carries only the
collapsed 25-category list. Three products are therefore ingested together, and the CSV mixes
geographic levels on purpose. `geo_level` is how you pick one.

---

## 1. What was downloaded

All three are free, direct HTTP, no login, no captcha, no API key.

| file in `data/raw/ca/` | product | what | size |
|---|---|---|---|
| `98100345-eng.zip` | **98-10-0345** *Religion by immigrant status and period of immigration: Canada, provinces and territories, census metropolitan areas and census agglomerations with parts* | **Religion (168)** × 174 geographies × age × gender × immigrant status | 12.8 MB |
| `98-401-X2021005_English_CSV.zip` | **98-401-X2021005** Census Profile — Canada, provinces, territories, CDs and CSDs | Religion (25), 2,631 characteristics in all | 197 MB |
| `98-401-X2021007_English_CSV.zip` | **98-401-X2021007** Census Profile — CMAs, tracted CAs and census tracts | Religion (25) | 250 MB |

Re-fetch:

```
curl -L -o data/raw/ca/98100345-eng.zip \
  "https://www150.statcan.gc.ca/n1/tbl/csv/98100345-eng.zip"

curl -L -o data/raw/ca/98-401-X2021005_English_CSV.zip \
  "https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/details/download-telecharger/comp/GetFile.cfm?Lang=E&FILETYPE=CSV&GEONO=005"

curl -L -o data/raw/ca/98-401-X2021007_English_CSV.zip \
  "https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/details/download-telecharger/comp/GetFile.cfm?Lang=E&FILETYPE=CSV&GEONO=007"
```

The `GEONO` codes on the profile download page are the only index of what each profile file
covers; the page carries no catalogue numbers. `GEONO=006` (2.2 GB, or six regional splits) adds
**dissemination areas** — 57,932 units of 400–700 people, finer still, same 25 categories. Not
taken: at DA scale the base-5 random rounding (§6) starts to bite, and CT already gives ~4,400
people per unit inside every metro area. It is the obvious next step if the dot value ever gets
fine enough to need it.

`python sources/ca.py` reads all three and rewrites the CSV. It streams two 2.6 GB CSVs and takes
about four minutes; it prints the whole of §7 below as it goes.

## 2. Geography, and its vintage

**Everything is 2021 Census geography (SGC 2021).** `geo_id` is the **DGUID**, StatCan's canonical
unique identifier, which states its own vintage in its first four characters — `2021A00051001101`
is the 2021 CSD 1001101. There is no vintage ambiguity to get wrong here, which is a mercy after
spec.md §8.1's Connecticut.

| `geo_level` | units | categories | product | notes |
|---|---|---|---|---|
| `country` | 1 | 168 | 98-10-0345 | |
| `province` | 13 | 168 | 98-10-0345 | 10 provinces + 3 territories; StatCan labels territories separately, we do not |
| `cma` | 41 | 168 | 98-10-0345 | census metropolitan areas |
| `ca` | 111 | 168 | 98-10-0345 | census agglomerations |
| `cma_part` | 2 | 168 | 98-10-0345 | Ottawa–Gatineau, Ont. and Que. parts |
| `ca_part` | 6 | 168 | 98-10-0345 | three cross-border CAs |
| `cd` | 293 | 25 | 98-401-X2021005 | census divisions |
| `csd` | **5,161** | 25 | 98-401-X2021005 | census subdivisions — municipalities, reserves, unorganised areas |
| `ct` | **6,247** | 25 | 98-401-X2021007 | census tracts, inside CMAs and tracted CAs only |

Country and province rows are taken **only** from 98-10-0345, whose 168 categories contain the
profile's 25 exactly, so those two levels are not duplicated. CMA and CA rows exist in the profile
too and were dropped for the same reason.

**Matching boundaries** (2021 cartographic — `b`, coastline-clipped, the analogue of the US `cb_`
files; the `a` digital versions extend into water and would scatter dots offshore):

```
https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/files-fichiers/lcsd000b21a_e.zip   CSD
https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/files-fichiers/lct_000b21a_e.zip   CT
https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/files-fichiers/lcd_000b21a_e.zip   CD
https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/files-fichiers/lcma000b21a_e.zip   CMA/CA
```

All four verified reachable 2026-08-30; not downloaded, since `data/geo/` is shared. They join on
DGUID (`DGUID` is a field in the shapefiles), the same join [[project_canadadots]] used for DAs.

**The CSD level alone is not good enough for a dot map, and that is why CTs are in here.** A
Canadian CSD is a municipality, and Toronto is one CSD of 2.6 million people; scattering its
religion uniformly over the city would erase the entire geography of the thing. CTs cover
**75.8% of the national population** at roughly 4,400 people each. The natural build is CT inside
metro areas, CSD outside — but **CSD and CT are different hierarchies and both are in this file**,
so summing the CSV without filtering `geo_level` double counts three-quarters of Canada. Same for
`cma`/`ca` against `province`, and for `cma_part` inside `cma`.

## 3. Basis: `self_id`, every row

Canada asks on the long-form questionnaire (25% sample), **question 30** in 2021: *"What is this
person's religion?"* — a write-in box, a "No religion" mark-in circle, and an instruction to
indicate a specific denomination or religion **"even if the person was not currently a practising
member of that group"**. It is a self-identification question about affiliation, explicitly *not*
about practice or attendance, and not a membership roll, so **every row is `self_id`** — there is
no per-row split of the kind ASARB forced in the US (spec.md §3.1).

Two consequences of it being a **25% sample**: the counts are weighted estimates, not a full
enumeration; and the universe is **population in private households**, not the whole population.
Canada's 2021 population is 36,991,981; the religion universe is **36,328,480**. The
**663,501 people (1.8%) in collective dwellings** — nursing homes, prisons, residences, work
camps — are outside every figure in this file. Worth remembering before computing a residual
against a population layer.

Non-response on the religion question itself: **1.4% non-response, 1.8% imputation** nationally
(Religion Reference Guide, 98-500-X2021016). Imputed values are in the counts, unflagged.

## 4. The categories, and how they nest

**They nest, four deep, and the CSV contains parents and children side by side.** Summing the
`source_category` column for one geography double counts everything. `note` carries
`parent=<the parent's verbatim source_category>` on every row precisely so the tree can be
rebuilt without a lookup table; the root has `parent=` empty.

The two products name their root differently — 98-10-0345 says `Total - Religion`, the Census
Profile says `Total - Religion for the population in private households - 25% sample data`. Both
are verbatim and both are the same universe.

`n.o.s.` = not otherwise specified (the respondent named the family but not the body — e.g.
"Protestant"), `n.i.e.` = not included elsewhere (a named body too small for its own code). Both
are genuine residual nodes and StatCan's own `…other/unspecified` in spec.md §3.2's sense —
they are already computed, and they are large: `Christian, n.o.s.` alone is **2,760,760 people,
7.6% of Canada**, more than every non-Christian religion in the country except Islam.

### The full 168, with national counts (2021, population in private households)

Indentation is the containment tree. Percentages are of `Total - Religion` = 36,328,480.

```
Total - Religion                                             36,328,480  100.000%
  Buddhist                                                      356,975   0.983%
  Christian                                                  19,373,325  53.328%
    Christian, n.o.s.                                         2,760,760   7.599%
    Anabaptist                                                  144,145   0.397%
      Anabaptist, n.o.s.                                          5,105   0.014%
      Amish                                                       3,535   0.010%
      Apostolic Christian Church (Nazarean)                         955   0.003%
      Be in Christ Church of Canada                               3,695   0.010%
      Mennonite                                                 130,585   0.359%
      Anabaptist, n.i.e.                                            260   0.001%
    Anglican                                                  1,134,315   3.122%
    Baptist                                                     436,940   1.203%
    Catholic                                                 10,880,360  29.950%
      Eastern Catholic                                           77,965   0.215%
        Eastern Catholic, n.o.s.                                 11,220   0.031%
        Armenian Catholic                                         1,535   0.004%
        Chaldean Catholic                                         5,075   0.014%
        Coptic Catholic                                             525   0.001%
        Maronite Catholic                                         9,125   0.025%
        Melkite Greek Catholic                                    2,210   0.006%
        Ukrainian Greek Catholic                                 45,040   0.124%
        Syriac Catholic                                           2,665   0.007%
        Eastern Catholic, n.i.e.                                    570   0.002%
      Roman Catholic                                         10,799,070  29.726%
      Other Catholic denominations                                3,325   0.009%
        Community Catholic Church of Canada                       1,830   0.005%
        Catholic, n.i.e.                                          1,495   0.004%
    Christian Orthodox                                          623,010   1.715%
      Christian Orthodox, n.o.s.                                229,465   0.632%
      Eastern Orthodox                                          326,855   0.900%
        Eastern Orthodox, n.o.s.                                 13,585   0.037%
        Albanian Orthodox                                           625   0.002%
        Antiochian Orthodox                                       2,145   0.006%
        Bulgarian Orthodox                                        3,825   0.011%
        Greek Orthodox                                          204,030   0.562%
        Macedonian Orthodox                                       6,050   0.017%
        Romanian Orthodox                                        16,120   0.044%
        Russian Orthodox                                         28,245   0.078%
        Serbian Orthodox                                         25,445   0.070%
        Ukrainian Orthodox                                       25,970   0.071%
        Eastern Orthodox, n.i.e.                                    825   0.002%
      Oriental Orthodox                                          65,305   0.180%
        Oriental Orthodox, n.o.s.                                   785   0.002%
        Armenian Orthodox                                        14,030   0.039%
        Coptic Orthodox                                          31,625   0.087%
        Eritrean Orthodox                                         2,370   0.007%
        Ethiopian Orthodox                                        7,815   0.022%
        Syriac Orthodox                                           6,780   0.019%
        Oriental Orthodox, n.i.e.                                 1,910   0.005%
      Christian Orthodox, n.i.e.                                  1,380   0.004%
    Jehovah's Witness                                           137,255   0.378%
    Latter Day Saints                                            87,725   0.241%
      Church of Jesus Christ of Latter-day Saints (Mormon)       85,320   0.235%
      Community of Christ                                         2,300   0.006%
      Latter Day Saints, n.i.e.                                     105   0.000%
    Lutheran                                                    328,040   0.903%
    Methodist and Wesleyan (Holiness)                           100,655   0.277%
      Methodist, n.o.s.                                          18,545   0.051%
      Canadian Church of God Ministries                             510   0.001%
      Church of the Nazarene                                      5,655   0.016%
      Evangelical Missionary Church                               5,165   0.014%
      Free Methodist Church                                       6,430   0.018%
      Salvation Army                                             51,935   0.143%
      United Methodist Church                                     1,475   0.004%
      Wesleyan Church                                            10,525   0.029%
      Methodist and Wesleyan (Holiness), n.i.e.                     420   0.001%
    Pentecostal and other Charismatic                           399,025   1.098%
      Pentecostal                                               392,570   1.081%
      Other Charismatic                                           6,455   0.018%
        Charismatic, n.o.s.                                       3,620   0.010%
        Catch the Fire/Partners in Harvest                          585   0.002%
        Victory Churches                                            660   0.002%
        Vineyard                                                    935   0.003%
        Charismatic, n.i.e.                                         660   0.002%
    Presbyterian                                                301,400   0.830%
    Reformed                                                     79,870   0.220%
      Reformed, n.o.s.                                           14,870   0.041%
      Canadian Reformed Church                                    9,780   0.027%
      Christian Reformed Church                                  45,110   0.124%
      Free Reformed Church                                        1,930   0.005%
      Netherlands Reformed                                        2,980   0.008%
      Reformed Church in America                                    700   0.002%
      United Reformed Church                                      4,070   0.011%
      Reformed, n.i.e.                                              420   0.001%
    United Church                                             1,214,185   3.342%
    Other Christian and Christian-related traditions            745,650   2.053%
      Apostolic, n.o.s.                                           6,795   0.019%
      Associated Gospel Churches                                  4,625   0.013%
      Brethren, n.o.s.                                            5,665   0.016%
      Calvinist, n.o.s.                                             780   0.002%
      Christadelphian                                             2,385   0.007%
      Christian and Missionary Alliance                          31,495   0.087%
      Christian Church (Disciples of Christ)                      1,140   0.003%
      Christian or Plymouth Brethren                              3,515   0.010%
      Christian Science                                           1,600   0.004%
      Church of God (Armstrong)                                   1,780   0.005%
      Church of God, n.o.s.                                       3,925   0.011%
      Church of God (Seventh Day)                                   630   0.002%
      Churches of Christ                                          5,740   0.016%
      Congregational Christian Churches in Canada                   555   0.002%
      Congregational, n.o.s.                                        595   0.002%
      Doukhobor                                                   1,675   0.005%
      Evangelical, n.o.s.                                        94,800   0.261%
      Evangelical Covenant Church                                 1,220   0.003%
      Evangelical Free Church                                     5,525   0.015%
      Grace Communion International                                 330   0.001%
      Iglesia ni Cristo                                          20,100   0.055%
      Interdenominational Christian                               2,020   0.006%
      Marthomite (Mar Thoma Church)                               1,735   0.005%
      Messianic Jewish                                            2,845   0.008%
      Mission de l'Esprit Saint                                     770   0.002%
      Moravian Church                                             3,660   0.010%
      New Apostolic Church                                        3,755   0.010%
      Non-denominational Christian                               54,455   0.150%
      Protestant, n.o.s.                                        398,210   1.096%
      Religious Society of Friends (Quakers)                      2,190   0.006%
      Seventh-day Adventist                                      68,305   0.188%
      Swedenborgian (New Church)                                    585   0.002%
      Other Christian and Christian-related traditions, n.i.e.       12,220   0.034%
  Hindu                                                         828,195   2.280%
  Jewish                                                        335,295   0.923%
  Muslim                                                      1,775,715   4.888%
  Sikh                                                          771,795   2.124%
  Traditional (North American Indigenous) spirituality           80,685   0.222%
  Other religions and spiritual traditions                      229,020   0.630%
    Animist                                                       1,575   0.004%
    Baha'i                                                       18,975   0.052%
    Chinese religions and spiritual traditions                   12,570   0.035%
      Ancestor veneration                                         1,660   0.005%
      Confucian                                                     995   0.003%
      Taoist                                                      5,550   0.015%
      Chinese religions and spiritual traditions, n.i.e.          4,360   0.012%
    Druze                                                         6,445   0.018%
    ECKist                                                        1,665   0.005%
    Gnostic                                                       1,180   0.003%
    Jain                                                          8,275   0.023%
    Japanese religions and spiritual traditions                   2,075   0.006%
      Shinto                                                      1,590   0.004%
      Japanese religions and spiritual traditions, n.i.e.           490   0.001%
    Multi-faith, n.o.s.                                           5,265   0.014%
    New Age                                                       2,665   0.007%
    New Thought-Unity-Religious Science                           1,720   0.005%
    Pagan beliefs and spiritual traditions                       45,320   0.125%
      Pagan, n.o.s.                                              24,615   0.068%
      Druidic                                                     1,645   0.005%
      Neopagan                                                    4,475   0.012%
      Wiccan                                                     12,625   0.035%
      Pagan beliefs and spiritual traditions, n.i.e.              1,960   0.005%
    Pantheist, n.o.s.                                             1,855   0.005%
    Personal faith or spiritual beliefs, n.o.s.                  60,190   0.166%
    Rastafarian                                                   2,110   0.006%
    Satanist                                                      5,895   0.016%
    Scientologist                                                 1,380   0.004%
    Shamanist                                                       745   0.002%
    Spiritualist                                                 12,310   0.034%
    Theist, n.o.s.                                                9,790   0.027%
    Unitarian/Unitarian Universalist                             10,930   0.030%
    Zoroastrian                                                   7,280   0.020%
    Other religions or spiritual traditions, n.i.e.               8,805   0.024%
  No religion and secular perspectives                       12,577,475  34.622%
    No religion                                              12,381,995  34.083%
    Secular perspectives                                        195,480   0.538%
      Secular perspectives, n.o.s.                                1,445   0.004%
      Agnostic                                                   83,780   0.231%
      Atheist                                                    86,385   0.238%
      Humanist                                                   11,395   0.031%
      Secular perspectives, n.i.e.                               12,470   0.034%
```

## 5. What is *not* in this list, and where the ceiling is

StatCan states the 2021 Census collected **"more than 200 religions and religious groups or
denominations"** and that the variable gained **"over 100 additional religions"** relative to
2011. The full classification is Appendix 2.14 of the Dictionary
(`.../2021/ref/dict/app/index-eng.cfm?ID=a2_14`) and it is visibly deeper than what ships: the
appendix splits **Amish** into *Amish, n.o.s.* / *Old Order Amish*, and **Mennonite** into
*Evangelical Mennonite Conference*, *Holdeman Mennonite*, *Mennonite Brethren*, *Mennonite Church
Canada*, *Old Colony Mennonite*, *Old Order Mennonite* and more. The disseminated **Religion (168)**
stops at `Amish` and `Mennonite`.

So **168 is the public ceiling, not the collected ceiling.** The rest exists and would need a
custom tabulation (cost recovery, weeks). I checked the alternatives before concluding this:

- every religion cube in the StatCan WDS catalogue (18 of them, from a full
  `getAllCubesListLite` dump) — only **98-10-0342** and **98-10-0345** carry Religion (168);
  everything else, including the two tables that go down to census subdivisions
  (98-10-0354, CD + CSDs of 5,000+), carries Religion (25);
- every Census Profile download product (`GEONO=001`…`029`) — all share one identical
  2,631-characteristic list, so the CSD file and the DA file have the *same* 25 religion
  categories as the national file. The profile does not get deeper at coarser geography;
- the Special Interest Profile "Religion" (98-26-0009) — also 25.

**98-10-0342** (*Religion by visible minority and generation status*) has the same 168 categories
over the same 174 geographies and would do as well; 98-10-0345 was taken because its zip is
**12.8 MB against 212 MB** for the same religion totals.

## 6. Rounding and suppression — quantified

**Random rounding to base 5, on every count.** Verified mechanically: **0 of 321,757 counts in
`ca.csv` is not a multiple of 5.** Each cell is rounded independently up or down to a neighbouring
multiple of 5, so the error on any one cell is at most ±4 and is unbiased, but **totals do not
reconcile exactly with their parts and never will**. That is the whole explanation of §7's
residuals, and it is why a nesting check on Canadian data must be a tolerance, not an equality.

Rounding is applied **per table**, so the same quantity disagrees across products: the Census
Profile's national Lutheran count is 328,045 and 98-10-0345's is 328,040. **3 of the 25 shared
national categories differ, always by exactly ±5.** Nobody is wrong; do not try to reconcile them.

**Suppression is by geography, not by cell** — a unit is either fully published or fully withheld
with symbol `x`, "suppressed to meet the confidentiality requirements of the *Statistics Act*".

| | CSD | CT |
|---|---|---|
| units | 5,161 | 6,247 |
| fully suppressed | **644 (12.5%)** | **89 (1.4%)** |
| population living in them | **7,260** | **2,900** |
| largest suppressed unit | 241 people | 1,318 people |
| of which incompletely enumerated reserves/settlements | **63** | 7 |

**0.02% of Canadians live in a suppressed unit**, so the hole is negligible in area terms and
politically specific: 63 of the 644 are the incompletely enumerated First Nations reserves, and
`Traditional (North American Indigenous) spirituality` is exactly the category they would have
carried. The suppressed rows are **kept in the CSV** with an empty `count` and
`note=…;suppressed=x`, rather than dropped, so a downstream join sees the hole instead of
silently missing 644 municipalities (spec.md §8.1's rule, applied to a hole in the data rather
than in the boundaries). No CD is suppressed.

**Quality suppression was discontinued in 2021**, and this one is a trap. In 2016 StatCan withheld
long-form data for any area with a global non-response rate over 50%; in 2021 it publishes them
and merely advises caution. So **241 CSDs (and 16 CTs) publish religion counts on a long-form
total non-response rate of 50% or worse**, up to 100%, and they look exactly like good data.
Median CSD TNR is 5.3%. Every profile row therefore carries `tnr_lf=` in its `note` — this is a
ready-made input to spec.md §7's confidence tier, and it is per unit, not per country.

`dq=` in the same note is the 5-digit `DATA_QUALITY_FLAG`: digit 1 is incomplete enumeration
(1 = incompletely enumerated reserve), digit 4 is the long-form quality flag (9 = suppressed).

## 7. Reconciliation

Printed by `sources/ca.py` on every run.

**Against StatCan's own published national figures** (*The Daily*, 26 October 2022, "Religion" —
the release rounds to the nearest 0.1 million):

| category | published | `ca.csv`, `geo_level=country` | published % | ours |
|---|---|---|---|---|
| Christian | 19.3 M, 53.3% | **19,373,325** | 53.3% | 53.33% |
| Catholic | 10.9 M, 29.9% | **10,880,360** | 29.9% | 29.95% |
| No religion and secular perspectives | 12.6 M, 34.6% | **12,577,475** | 34.6% | 34.62% |
| Muslim | 1.8 M, 4.9% | **1,775,715** | 4.9% | 4.89% |
| Hindu | 830,000, 2.3% | **828,195** | 2.3% | 2.28% |
| Sikh | 770,000, 2.1% | **771,795** | 2.1% | 2.12% |
| Buddhist | 360,000, 1.0% | **356,975** | 1.0% | 0.98% |
| Jewish | 335,000, 0.9% | **335,295** | 0.9% | 0.92% |

Every category matches to the precision the release publishes.

**Internally**, with `Total - Religion` = **36,328,480**:

| check | result |
|---|---|
| 21 parent nodes of the 168-tree vs the sum of their children | **12 differ, all by ±5 to ±25** — random rounding, largest overshoot 25 people (0.00007%) |
| 13 provinces + territories, summed | 36,328,490 — **+10** |
| 293 census divisions, summed | 36,328,505 — **+25** |
| 5,161 CSDs, summed | 36,322,855 — **99.985%**, the 0.015% being the 644 suppressed units |
| 6,247 CTs, summed | 27,544,565 — **75.821%**, which is the share of Canadians living in a CMA or tracted CA, as expected |
| 25 shared national categories, Census Profile vs 98-10-0345 | **3 differ, each by ±5** |

No negative residual anywhere, no unit over 100% of its parent: this is a `self_id` census, so
spec.md §3.6's roll-attribution failure cannot arise. The maximum discrepancy anywhere in the
national arithmetic is **25 people**.

## 8. Licence

**Statistics Canada Open Licence** (statcan.gc.ca/en/reference/licence) — a worldwide,
royalty-free, non-exclusive licence to "use, reproduce, publish, freely distribute, or sell the
Information", to make and sell value-added products, and to sublicence. Attribution is required
and, because this map adapts the figures rather than republishing them as-is, the required form is
the value-added one:

> Adapted from Statistics Canada, *Census of Population, 2021*, Statistics Canada Catalogue nos.
> 98-10-0345 and 98-401-X2021005/007, 2021. This does not constitute an endorsement by Statistics
> Canada of this product.

Prohibited: misrepresenting the source, implying StatCan endorsement, and merging the data with
anything else in order to identify individuals. Nothing here blocks shipping. The boundary files
are under the same licence.

## 9. Things that surprised me

1. **The depth is in the wrong table.** Every intuition says the fine-geography file is the
   compromise one; here the fine-geography file and the national file have the *identical*
   characteristic list, and the depth lives in a table about *immigration* that happens to carry
   religion as a dimension. It took a dump of the whole WDS cube catalogue to find it.
2. **Roman Catholic is 99.3% of Catholic** (10,799,070 of 10,880,360) but the Eastern Catholic
   churches are enumerated anyway — Ukrainian Greek Catholic 45,040, Maronite 9,125, Chaldean
   5,075, down to Coptic Catholic at 525. Similarly **9 named Eastern Orthodox jurisdictions**
   (Greek 204,030 down to Albanian 625) and **5 Oriental Orthodox**. This is the deepest Orthodox
   and Eastern Catholic split any census on the map is likely to give.
3. **Quebec is a category error waiting to happen.** 4,472,555 Catholics, 54% of the province,
   against 2,267,720 with no religion — and Quebec's Catholic identification is famously an
   ethno-cultural marker rather than a practising one. The number is right; a map that reads it
   as devotion is not. Nothing to do about it in the data, but the about panel will need a line.
4. **26 of the 168 categories are under 1,000 people nationally** — Latter Day Saints n.i.e. 105,
   Grace Communion International 330, Coptic Catholic 525, Confucian 995. At 1 dot per 1,000
   these are rings, all of them, and at province granularity there is nothing finer to place them
   with. Same shape as the US Anabaptist tail (spec.md §4.3), from an unrelated country and a
   completely different collection method.
5. **`Chinese religions and spiritual traditions` totals 12,570 people in a country with 1.7
   million people of Chinese origin**, and it is a *node with children* (Ancestor veneration,
   Confucian, Taoist). This is spec.md §3.3 seen from the census side: a self-identification
   question simply does not capture folk practice, and the count is a fact about the question,
   not about the practice. Do not let the tree's depth here imply confidence.
6. **`Multi-faith, n.o.s.` exists as a code** (5,265 people) — the classification admits a
   combination category even though the question asks for one religion. It is spec.md §3.3's
   syncretic node, arriving unbidden from a census that did not want it.
7. The Census Profile's `CHARACTERISTIC_NAME` column drops the leading-space indentation that the
   `_meta.txt` file inside the same zip uses to encode the hierarchy. `sources/ca.py` reads the
   tree from the meta file for that reason; a parser working from the data CSV alone cannot know
   that `Anglican` is a child of `Christian` and would double count by 53% of the country.

## 10. Notes for whoever wires this into the pipeline

- **`ca.csv` is 62 MB.** It is not committed: the maps-root `.gitignore` line `data/` catches
  every `data/` directory in the repo, `religiondots/data/normalized/` included, so the file is
  reproducible-only. Checked, not assumed — `git check-ignore -v` says so. Roughly a third of
  those bytes is the `parent=` token repeating the profile's 74-character root name; shortening
  it is the obvious saving if the size ever matters.
- The profile CSVs inside the zips are **latin-1**, not UTF-8 (Montréal, Trois-Rivières);
  98-10-0345 is UTF-8 with a BOM. `ca.csv` is written UTF-8.
- `geo_name` for a CT is just its number, e.g. `9320001.00` — census tracts have no names. The
  first three digits are the CMA/CA code, so the metro area is recoverable from the id.
- `count` is empty, never zero, for a suppressed unit. A real zero is written `0` and is common
  at CT level.
- Nothing here has been mapped to `taxonomy/`. `source_category` is verbatim, per the brief and
  spec.md §2.4; the 168 names are StatCan's own strings. Note for whoever does the mapping later
  that **`Messianic Jewish` sits under Christianity in StatCan's tree**, which is the same call
  the US mapping made and flagged in `REVIEW` — the two sources agree, for once.
