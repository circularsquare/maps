# Australia — ABS Census 2021, religious affiliation

`source_id: au_census_2021` · `basis: self_id` (every row) · `year: 2021`
Ingested 2026-08-30. `sources/au.py` → `data/normalized/au.csv`, **84,282 rows**.

| geo_level | units | categories | rows |
|---|---|---|---|
| `sa2` | 2,472 Statistical Areas Level 2 | 34 | 84,048 |
| `nation` | 1 (`AUS`) | 154 (150 ASCRG religious groups + 4 aggregates) | 154 |
| `state` | 8 states/territories | 10 (9 ASCRG broad groups + Total) | 80 |

**TableBuilder was not needed.** The ~150-category detailed classification is in a free
public xlsx. No login, no captcha, nothing blocked.

---

## 1. Where it came from, and how to re-fetch

Three files in `data/raw/au/`, all direct, all anonymous, ~42 MB total:

```
curl -L -o data/raw/au/2021_GCP_SA2_for_AUS_short-header.zip \
  https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA2_for_AUS_short-header.zip

curl -L -o data/raw/au/Census_article_Religious_affiliation_in_Australia.xlsx \
  "https://www.abs.gov.au/statistics/people/people-and-communities/cultural-diversity-census/2021/Census%20article%20-%20Religious%20affiliation%20in%20Australia.xlsx"

curl -L -o data/raw/au/2021_Religious_affiliation_classification.xlsx \
  "https://www.abs.gov.au/census/guide-census-data/census-dictionary/2021/variables-topic/cultural-diversity/religious-affiliation-relp/Religious%20affiliation%20classification.xlsx"
```

| file | what it is | what we take |
|---|---|---|
| `2021_GCP_SA2_for_AUS_short-header.zip` (42 MB) | 2021 Census **General Community Profile** DataPack, SA2, release R2. 65 tables. | `2021Census_G14_AUST_SA2.csv` — table **G14 "Religious Affiliation by Sex"**, the *only* religion table in the GCP. Plus `Metadata/2021Census_geog_desc_1st_2nd_3rd_release.xlsx` for SA2 names. |
| `Census_article_Religious_affiliation_in_Australia.xlsx` (69 KB) | Data behind the ABS article *Religious affiliation in Australia*, released 4 July 2022 with *Cultural diversity: Census 2021*. Seven tables. | **Table 4** (religious groups, Australia, 2016 + 2021) → the `nation` rows. **Table 1** (broad groups × state) → the `state` rows. **Table 3** (narrow groups, Australia) → the reconciliation target only, not emitted. |
| `2021_Religious_affiliation_classification.xlsx` (27 KB) | The **ASCRG 2021** structure itself: broad / narrow / religious group codes and official labels. | Every `source_category` label and every `ascrg=` code in `note`. Nothing is hardcoded from my reading of a table. |

DataPacks landing page: <https://www.abs.gov.au/census/find-census-data/datapacks>
(pattern: `2021_{GCP|IP|TSP|PEP|WPP}_{SA1|SA2|SA3|SA4|STE|LGA|POA|…}_for_{AUS|NSW|…}_short-header.zip`).

The DataPack readme's `Summary_of_changes.txt` lists the R1→R2 amendments. **G14 is not among
them** — only G08, G11, G19, G22 and G62 changed. So the R2 file is the final G14.

### Not used, and why

- **SA1.** `2021_GCP_SA1_for_AUS_short-header.zip` (365 MB) exists and **does contain G14** —
  verified by downloading the ACT-only SA1 pack, which has `2021Census_G14_ACT_SA1.csv`. Not
  taken: an SA1 holds ~400 people by design, so at any dot value this project can ship (spec §4.2
  says 1:1,000) an SA1 is a fraction of one dot, and ABS perturbation is proportionally brutal at
  that size. **SA1 is the right *placement* layer, not the right *data* layer** — it is Australia's
  exact analogue of the US census tract trick in spec §8.2: SA1s are built to a population target,
  so scattering an SA2's dots equally across its SA1s is already a population weighting and needs
  no population raster.
- **TableBuilder.** Would give religious group × SA2, i.e. the detailed classification at fine
  geography, which is the one thing missing here. It needs a registered login. Per the brief, not
  attempted. Worth someone's time later — see §5, the "Other Religious Groups" problem.
- **ABS Data API** (`data.api.abs.gov.au`). Checked: it publishes **no 2021 Census religion
  dataflow**. The API route is a dead end for this variable.

---

## 2. Geography and vintage — spec §8.1

**ASGS Edition 3, July 2021 – June 2026.** The column is literally `SA2_CODE_2021`. Boundaries
must be the 2021 edition, not ASGS 2016 and not whatever is newest when someone reads this.

ASGS 2016 had **2,310** SA2s; Edition 3 has **2,473**, and ABS records that ~8% of 2016 SA2s
changed shape — mostly inner-city and growth-corridor splits. Codes were reused across the
editions for unchanged areas, so **a 2016-boundary join against 2021 codes would silently
succeed on most rows and be wrong on the ones that moved**. That is the Connecticut failure of
spec §8.1 in its quieter form: no error, no missing state, just a few hundred areas with the
wrong outline.

Boundaries to fetch when the join is built (CC BY 4.0, same licence as the data):

```
https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA2020.zip
```
(48 MB; a GDA94 twin exists at `…_SHP_GDA94.zip`.)

**The join must be checked both directions and three things expected:**

1. **2,472 vs 2,473.** The DataPack's own census geography metadata lists **2,472** SA2s and G14
   has exactly those 2,472 — both directions clean, zero unmatched either way. ABS's ASGS Ed3
   publication quotes **2,473**. One unit of difference I did not chase; the DataPack metadata is
   the authority for *this* join, and the boundary file will name the extra one immediately.
2. **18 special-purpose SA2s with no boundary at all** — two per state and territory,
   `x97979799` "Migratory - Offshore - Shipping" and `x99999499` "No usual address". Zero area,
   **52,920 people = 0.208% of the population**. Every row for these carries
   `flag=pseudo_sa2_no_boundary` in `note`. They are real census units and must be dropped
   deliberately, not lost in a geometry join.
3. **37 SA2s with a real polygon and zero population** — Wollemi, Centennial Park, Royal Botanic
   Gardens Victoria, Mount Coot-tha, Kalgoorlie Airport, Lake Burley Griffin, Greenbank Military
   Camp, several reservoirs and industrial estates. An expected absence, named so it is not
   mistaken for an accidental one.

`geo_id` is the 9-character SA2 code as a **string**; leading digits are significant and the first
digit is the state. `state` rows use ASGS STE codes `1`–`8`. Everything is **place of usual
residence** and **excludes overseas visitors**.

---

## 3. The classification, and the double-counting trap

ABS uses the **Australian Standard Classification of Religious Groups (ASCRG) 2021**, three
levels deep:

| level | count | example |
|---|---|---|
| broad group (1 char) | 10 (7 + 3 supplementary) | `2` Christianity |
| narrow group (2–3 char) | 38 | `223` Eastern Orthodox |
| religious group (4 char) | 151 | `2233` Greek Orthodox |

**G14 contains all three levels in one file, side by side**, which is precisely the shape spec
§2.3 and §3.2 warn about: `Christianity_Anglican_P`, `Christianity_Tot_P` and `Tot_P` are all
columns of the same row, and adding them up counts several million Australians two and three
times. Table 4 at nation level has the same shape (150 leaves *plus* `Total Christian`,
`Total Other Religions`, `Total Secular…`, `Total`).

**What I kept: everything, tagged.** Dropping the parents would have been the tidier-looking
choice and the wrong one — spec §3.2 computes `residual = parent − Σ children` at every level,
and it needs the published parent, not a re-derived one (ABS perturbs them separately, so they
differ; see §4). So every row carries `level=` in `note`:

| `level` | what it is | safe to sum? |
|---|---|---|
| `leaf` | partitions the population exactly once at that geo_level | **yes — this is the one to use** |
| `narrow`/`broad`-group aggregates, tagged `group_total` | sums of leaves above them | no, not with leaves |
| `grand_total` | the whole population | no |

`note` also carries `parent=` (the ABS category directly above) and `ascrg=` (the official code),
so the source's own hierarchy is fully reconstructable without inventing anything. **No row is
mapped to the religiondots taxonomy** — spec §2.4, deferred.

`sa2` has 30 leaves + 3 `group_total` + 1 `grand_total` = 34.
`nation` has 150 leaves + 4 aggregates = 154.
`state` has 9 leaves-at-broad-level + 1 `grand_total` = 10; these are tagged `broad_group`
because at state level the broad group *is* the finest available, and they must not be mixed with
`sa2` leaves.

---

## 4. Reconciliation against the ABS published nationals

Summing the 2,472 SA2 cells against the ABS's own published national table (article Tables 1 and
3). `sources/au.py` prints this every run.

| ASCRG category | SA2 sum | ABS published | diff | % |
|---|---:|---:|---:|---:|
| Buddhism | 615,634 | 615,823 | −189 | −0.03 |
| Anglican | 2,496,317 | 2,496,273 | +44 | 0.00 |
| Assyrian Apostolic | 18,932 | 19,141 | −209 | −1.09 |
| Baptist | 347,381 | 347,334 | +47 | 0.01 |
| Brethren | 17,946 | 18,258 | −312 | −1.71 |
| Catholic | 5,075,884 | 5,075,907 | −23 | −0.00 |
| Churches of Christ | 35,809 | 35,928 | −119 | −0.33 |
| Eastern Orthodox | 535,308 | 535,470 | −162 | −0.03 |
| Jehovah's Witnesses | 84,145 | 84,405 | −260 | −0.31 |
| Latter-day Saints | 57,790 | 57,868 | −78 | −0.13 |
| Lutheran | 145,916 | 145,868 | +48 | 0.03 |
| Oriental Orthodox | 60,629 | 60,774 | −145 | −0.24 |
| Other Protestant | 112,378 | 112,474 | −96 | −0.09 |
| Pentecostal | 255,673 | 255,838 | −165 | −0.06 |
| Presbyterian and Reformed | 414,954 | 414,882 | +72 | 0.02 |
| Salvation Army | 35,305 | 35,356 | −51 | −0.14 |
| Seventh-day Adventist | 63,596 | 63,662 | −66 | −0.10 |
| Uniting Church | 673,383 | 673,260 | +123 | 0.02 |
| Christianity, nfd | 688,467 | 688,440 | +27 | 0.00 |
| Other Christian | 27,480 | 27,679 | −199 | −0.72 |
| **Christianity** *(group_total)* | 11,148,753 | 11,148,814 | −61 | −0.00 |
| Hinduism | 683,856 | 684,002 | −146 | −0.02 |
| Islam | 813,391 | 813,392 | −1 | −0.00 |
| Judaism | 99,608 | 99,956 | −348 | −0.35 |
| Australian Aboriginal Traditional Religions | 7,391 | 7,887 | −496 | **−6.29** |
| Sikhism | 210,415 | 210,400 | +15 | 0.01 |
| Other Religious Groups | 107,127 | 107,134 | −7 | −0.01 |
| **Other Religions** *(group_total)* | 325,546 | 325,421 | +125 | 0.04 |
| No Religion, (so described) | 9,767,363 | 9,767,448 | −85 | −0.00 |
| Secular Beliefs | 73,461 | 73,548 | −87 | −0.12 |
| Other Spiritual Beliefs | 45,888 | 45,784 | +104 | 0.23 |
| **Secular Beliefs and Other Spiritual Beliefs and No Religious Affiliation** *(group_total)* | 9,886,823 | 9,886,957 | −134 | −0.00 |
| Religious affiliation not stated | 1,848,513 | 1,848,428 | +85 | 0.00 |
| **Total** *(grand_total)* | **25,422,677** | **25,422,788** | **−111** | **−0.00** |

**Nation detail:** the 150 religious-group rows sum to **25,422,764** against the published Total
**25,422,788** — **−24**, i.e. 0.9 parts per million.

**Nothing here is a mapping error.** ABS applies a small random adjustment to **every cell of
every table independently**, so no two ABS tables add up to each other exactly and no ABS total
equals the sum of its own parts. Inside G14 itself:

```
30 leaf columns   25,419,940   vs Tot_P 25,422,677   diff  −2,737
Christianity      19 children  11,147,293  parent 11,148,753  diff −1,460
Other Religions    3 children     324,933  parent    325,546  diff   −613
Secular …          3 children   9,886,712  parent  9,886,823  diff   −111
```

**The bias is one-directional and worth knowing about**: it is downward for rare categories,
because a true count of 1 or 2 is perturbed to 0 in many of the 2,472 cells and never perturbed
upward enough to compensate. Australian Aboriginal Traditional Religions loses **6.3%** this way
(present in only 484 SA2s), Brethren 1.7%, Assyrian Apostolic 1.1%, Other Christian 0.7%; big
categories lose essentially nothing. **So an SA2-summed national figure for a small group is a
floor, not the number**, and the published national figure should be preferred wherever both
exist. The Secular Beliefs group_total also carries the 183 people in `7000 Secular Beliefs …
nfd`, which G14 gives no column of its own — another reason the parent exceeds its children.

For spec §3.6's yardstick (ASARB's roll overshoot ran at 0.0065% of population): this is a
`self_id` census, so **no SA2 can exceed its own population by construction** — `Tot_P` is the
denominator, not an independent figure. There is no catchment problem here at all.

---

## 5. The categories

### 5.1 What is available at SA2 (the map layer)

The 30 `leaf` categories, with the national sum of the 2,472 cells:

| ASCRG | category | parent | SA2 sum |
|---|---|---|---:|
| 101 | Buddhism | Buddhism | 615,634 |
| 200 | Christianity, nfd | Christianity | 688,467 |
| 201 | Anglican | Christianity | 2,496,317 |
| 203 | Baptist | Christianity | 347,381 |
| 205 | Brethren | Christianity | 17,946 |
| 207 | Catholic | Christianity | 5,075,884 |
| 211 | Churches of Christ | Christianity | 35,809 |
| 213 | Jehovah's Witnesses | Christianity | 84,145 |
| 215 | Latter-day Saints | Christianity | 57,790 |
| 217 | Lutheran | Christianity | 145,916 |
| 221 | Oriental Orthodox | Christianity | 60,629 |
| 222 | Assyrian Apostolic | Christianity | 18,932 |
| 223 | Eastern Orthodox | Christianity | 535,308 |
| 225 | Presbyterian and Reformed | Christianity | 414,954 |
| 227 | Salvation Army | Christianity | 35,305 |
| 231 | Seventh-day Adventist | Christianity | 63,596 |
| 233 | Uniting Church | Christianity | 673,383 |
| 24 | Pentecostal | Christianity | 255,673 |
| 28 | Other Protestant | Christianity | 112,378 |
| 29 | Other Christian | Christianity | 27,480 |
| 301 | Hinduism | Hinduism | 683,856 |
| 401 | Islam | Islam | 813,391 |
| 501 | Judaism | Judaism | 99,608 |
| 601 | Australian Aboriginal Traditional Religions | Other Religions | 7,391 |
| 615 | Sikhism | Other Religions | 210,415 |
| 603+605+607+611+613+617+69 | **Other Religious Groups** | Other Religions | 107,127 |
| 71 | No Religion, (so described) | Secular Beliefs … | 9,767,363 |
| 72 | Secular Beliefs | Secular Beliefs … | 73,461 |
| 73 | Other Spiritual Beliefs | Secular Beliefs … | 45,888 |
| 000+&&& | Religious affiliation not stated | Total | 1,848,513 |

plus `group_total` rows for Christianity (2), Other Religions (6), Secular Beliefs … (7) and a
`grand_total` row (Total).

Two G14 columns are **not** ASCRG categories and are labelled with the DataPack's own wording:

- **`Other Religious Groups`, 107,127 people.** A G14-only bucket over *seven* ASCRG narrow
  groups — Baha'i, Chinese Religions, Druse, Japanese Religions, Nature Religions, Spiritualism
  and Miscellaneous Religions. **This is the one real loss.** At SA2 you cannot separate Taoism,
  Shinto, Paganism, Wicca, Jainism, Zoroastrianism, Mandaean, Yezidi, Caodaism, Rastafari,
  Scientology or Satanism from each other — a third of a percent of the population, and almost
  every group in Australia that spec R2 exists to show, in one undifferentiated cell.
- **`Religious affiliation not stated`, 1,848,513.** G14 folds both ASCRG supplementary codes
  together: `&&&` Not stated (1,751,052) + `000` Inadequately described (97,376). They *are*
  separable at nation and state level.

Median SA2 has **26 of the 30** leaf categories present (mean 24.5, max 30); 18.2% of all
leaf cells are exactly zero. Coverage per category ranges from Catholic in 2,389 SA2s down to
Australian Aboriginal Traditional Religions in 484 and Assyrian Apostolic in 302 — a lot of
spec §4.3 rings.

### 5.2 What is available at nation only — the 150 religious groups

The detailed level, `geo_level=nation`, `geo_id=AUS`. Grouped by ASCRG narrow group; every row's
`note` carries its `parent=`.

**Single-group parents** (broad = narrow = religious group, one row each): Buddhism 615,823 ·
Christianity, nfd 688,440 · Baptist 347,334 · Brethren 18,258 · Jehovah's Witnesses 84,405 ·
Lutheran 145,868 · Salvation Army 35,356 · Seventh-day Adventist 63,662 · Uniting Church 673,260 ·
Hinduism 684,002 · Islam 813,392 · Judaism 99,956 · Australian Aboriginal Traditional Religions
7,887 · Baha'i 14,937 · Druse 4,268 · Sikhism 210,400 · Spiritualism 8,879 ·
No Religion, so described (7101) 9,767,448 · Secular Beliefs … nfd (7000) 183 ·
Inadequately described 97,376 · Not stated 1,751,052.

| parent | code | religious group | 2021 |
|---|---|---|---:|
| Anglican | 2012 | Anglican Church of Australia | 2,495,818 |
| | 2013 | Anglican Catholic Church | 402 |
| | 2019 | Anglican, nec | 63 |
| Catholic | 2070 | Catholic, nfd | 9 |
| | 2071 | Western Catholic | 4,994,188 |
| | 2072 | Maronite Catholic | 47,014 |
| | 2073 | Melkite Catholic | 3,086 |
| | 2074 | Ukrainian Catholic | 2,882 |
| | 2075 | Chaldean Catholic | 14,103 |
| | 2076 | Syro Malabar Catholic | 10,301 |
| | 2079 | Catholic, nec | 4,322 |
| Churches of Christ | 2110 | Churches of Christ, nfd | 5,266 |
| | 2111 | Churches of Christ (Conference) | 30,525 |
| | 2112 | Church of Christ (Non-denominational) | 90 |
| | 2113 | International Church of Christ | 39 |
| Latter-day Saints | 2150 | Latter-day Saints, nfd | 24 |
| | 2151 | The Church of Jesus Christ of Latter-day Saints | 57,253 |
| | 2152 | Community of Christ | 593 |
| Oriental Orthodox | 2210 | Oriental Orthodox, nfd | 75 |
| | 2212 | Armenian Apostolic | 7,870 |
| | 2214 | Coptic Orthodox Church | 33,091 |
| | 2215 | Syrian Orthodox Church | 13,089 |
| | 2216 | Ethiopian Orthodox Church | 5,888 |
| | 2219 | Oriental Orthodox, nec | 766 |
| Assyrian Apostolic | 2220 | Assyrian Apostolic, nfd | 2,172 |
| | 2221 | Assyrian Church of the East | 15,459 |
| | 2222 | Ancient Church of the East | 1,495 |
| | 2229 | Assyrian Apostolic, nec | 6 |
| Eastern Orthodox | 2230 | Eastern Orthodox, nfd | 1,365 |
| | 2231 | Albanian Orthodox | 16 |
| | 2232 | Antiochian Orthodox | 14,264 |
| | 2233 | Greek Orthodox | 390,963 |
| | 2234 | Macedonian Orthodox | 52,311 |
| | 2235 | Romanian Orthodox | 2,147 |
| | 2236 | Russian Orthodox | 22,631 |
| | 2237 | Serbian Orthodox | 48,287 |
| | 2238 | Ukrainian Orthodox | 2,627 |
| | 2239 | Eastern Orthodox, nec | 848 |
| Presbyterian and Reformed | 2250 | Presbyterian and Reformed, nfd | 85 |
| | 2251 | Presbyterian | 402,138 |
| | 2252 | Reformed | 10,323 |
| | 2253 | Free Reformed | 2,344 |
| Pentecostal | 2400 | Pentecostal, nfd | 228,916 |
| | 2401 | Apostolic Church (Australia) | 457 |
| | 2402 | Australian Christian Churches (Assemblies of God) | 14,361 |
| | 2403 | Bethesda Ministries International (Bethesda Churches) | 17 |
| | 2404 | C3 Church Global (Christian City Church) | 305 |
| | 2406 | International Network of Churches (Christian Outreach Centres) | 5,940 |
| | 2407 | CRC International (Christian Revival Crusade) | 1,057 |
| | 2411 | Foursquare Gospel Church | 271 |
| | 2412 | Full Gospel Church of Australia (Full Gospel Church) | 258 |
| | 2413 | Revival Centres | 735 |
| | 2414 | Rhema Family Church | 38 |
| | 2415 | United Pentecostal | 737 |
| | 2416 | Acts 2 Alliance | 123 |
| | 2417 | Christian Church in Australia | 7 |
| | 2418 | Pentecostal City Life Church | 3 |
| | 2421 | Revival Fellowship | 1,114 |
| | 2422 | Victory Life Centre | 16 |
| | 2423 | Victory Worship Centre | 3 |
| | 2424 | Worship Centre Network | 221 |
| | 2499 | Pentecostal, nec | 1,276 |
| Other Protestant | 2800 | Other Protestant, nfd | 35,912 |
| | 2801 | Aboriginal Evangelical Missions | 2,159 |
| | 2802 | Born Again Christian | 12,805 |
| | 2803 | Christian and Missionary Alliance | 1,663 |
| | 2804 | Church of the Nazarene | 942 |
| | 2805 | Congregational | 3,409 |
| | 2806 | Ethnic Evangelical Churches | 5,135 |
| | 2807 | Independent Evangelical Churches | 7,811 |
| | 2808 | Wesleyan Methodist Church | 3,455 |
| | 2811 | Christian Community Churches of Australia | 5,183 |
| | 2812 | Methodist, so described | 33,076 |
| | 2813 | United Methodist Church | 326 |
| | 2899 | Other Protestant, nec | 593 |
| Other Christian | 2900 | Other Christian, nfd | 171 |
| | 2901 | Apostolic Church of Queensland | 2,710 |
| | 2902 | Christadelphians | 9,734 |
| | 2903 | Christian Science | 975 |
| | 2904 | Gnostic Christians | 843 |
| | 2905 | Liberal Catholic Church | 161 |
| | 2906 | New Apostolic Church | 2,324 |
| | 2907 | New Churches (Swedenborgian) | 225 |
| | 2908 | Ratana (Maori) | 3,271 |
| | 2911 | Religious Science | 108 |
| | 2912 | Religious Society of Friends (Quakers) | 1,738 |
| | 2913 | Temple Society | 231 |
| | 2915 | Grace Communion International (Worldwide Church of God) | 204 |
| | 2999 | Other Christian, nec | 4,993 |
| Chinese Religions | 6050 | Chinese Religions, nfd | 6 |
| | 6051 | Ancestor Veneration | 409 |
| | 6052 | Confucianism | 330 |
| | 6053 | Taoism | 6,149 |
| | 6059 | Chinese Religions, nec | 49 |
| Japanese Religions | 6110 | Japanese Religions, nfd | 7 |
| | 6111 | Shinto | 1,395 |
| | 6112 | Sukyo Mahikari | 340 |
| | 6113 | Tenrikyo | 103 |
| | 6119 | Japanese Religions, nec | 10 |
| Nature Religions | 6130 | Nature Religions, nfd | 563 |
| | 6131 | Animism | 1,166 |
| | 6132 | Druidism | 819 |
| | 6133 | Paganism | 18,625 |
| | 6135 | Wiccan (Witchcraft) | 7,786 |
| | 6139 | Nature Religions, nec | 4,176 |
| Miscellaneous Religions | 6901 | Mandaean | 9,178 |
| | 6902 | Yezidi | 4,123 |
| | 6991 | Caodaism | 677 |
| | 6992 | Church of Scientology | 1,655 |
| | 6993 | Eckankar | 488 |
| | 6994 | Rastafari | 921 |
| | 6995 | Satanism | 4,995 |
| | 6996 | Theosophy | 855 |
| | 6997 | Jainism | 5,851 |
| | 6998 | Zoroastrianism | 2,986 |
| | 6999 | Religious Groups, nec | 5,339 |
| Secular Beliefs | 7200 | Secular Beliefs, nfd | 289 |
| | 7201 | Agnosticism | 31,676 |
| | 7202 | Atheism | 37,797 |
| | 7203 | Humanism | 2,194 |
| | 7204 | Rationalism | 750 |
| | 7299 | Secular Beliefs, nec | 830 |
| Other Spiritual Beliefs | 7300 | Other Spiritual Beliefs, nfd | 1,770 |
| | 7301 | Multi Faith | 3,579 |
| | 7302 | New Age | 951 |
| | 7303 | Own Spiritual Beliefs | 27,379 |
| | 7304 | Theism | 5,421 |
| | 7305 | Unitarian Universalism | 828 |
| | 7399 | Other Spiritual Beliefs, nec | 5,863 |

---

## 6. Religion is voluntary, and how much that costs

**Religion is the only optional question on the Australian census.** Section 14(3) of the
*Census and Statistics Act 1905* forbids compelling an answer, and has since 1911 — the exemption
was won in the Senate debate on the original Act over fears of "sectarian bitterness". Every other
question is mandatory. Worth knowing when the UK, Ireland and New Zealand rows land — several
other censuses on this map also make religion voluntary, but few by a statute this old.

| | 2021 | 2016 |
|---|---|---|
| **ABS-published non-response rate (RELP)** | **6.9%** | 9.1% |
| Not stated (`&&&`) | 1,751,052 = 6.89% | 2,132,167 = 9.11% |
| Inadequately described (`000`) — a religion not in the classification | 97,376 = 0.38% | 106,568 = 0.46% |
| **Together, which is what G14 gives at SA2** | **1,848,428 = 7.27%** | 2,238,735 = 9.57% |

So the SA2 layer has **7.27% of the population in a single unusable bucket**, and non-response
fell sharply between censuses — which by itself pushes every category's 2016→2021 change in the
positive direction and is worth remembering before reading any decline as secularisation.

Note the direction: this is `self_id`, so spec §3.5's undercount worry runs the *other* way from a
`roll` source. Nobody is counted twice; a lot of people simply declined. The 7.27% is a real,
irreducible slice of every SA2 and belongs on the map as its own residual (spec §3.2), not
redistributed.

---

## 7. Licence

**Creative Commons Attribution 4.0 International.** Stated in the DataPack's own
`Readme/CreativeCommons_Licensing_readme.txt` and on the ASGS boundary download page. No
registration, no redistribution restriction, commercial use fine.

The required attribution differs by whether the material is transformed, and **ours is**:

> Based on Australian Bureau of Statistics data.

(For untransformed material it would be "Source: Australian Bureau of Statistics". Since we
re-tabulate, scatter and derive, the "Based on" form is the correct one — ABS explicitly lists
"deriving new statistics from published ABS statistics" as the trigger.) A link back to
<https://www.abs.gov.au> is requested. The suggested full citation:

> Australian Bureau of Statistics (2022), *Cultural diversity: Census* and *2021 Census
> DataPacks, General Community Profile*, ABS Website, accessed 30 August 2026.

---

## 8. Things that surprised me

1. **TableBuilder is not the only route to the detailed classification.** The brief expected it
   to be; it is not. ABS published all 150 religious groups nationally in a 69 KB spreadsheet
   attached to a news article. The login wall guards the *cross-tabulation*, not the detail.
2. **ABS publishes counts of 3.** `2418 Pentecostal City Life Church: 3` and `2423 Victory
   Worship Centre: 3` are printed national figures. **107 of the 150 groups are under 10,000
   people, 58 under 1,000 and 19 under 100.** That is finer at the small end than ASARB, whose
   smallest was 26 (spec §4.3), and it is a straight R3 case — dozens of groups that can only ever
   be rings, but rings at *national* scale, since the geography does not go with them. Ten of the
   nineteen under 100 are `nfd`/`nec` residuals rather than named bodies; the nine that are real
   named denominations are Pentecostal City Life Church 3, Victory Worship Centre 3, Christian
   Church in Australia 7, Albanian Orthodox 16, Victory Life Centre 16, Bethesda Ministries
   International 17, Rhema Family Church 38, International Church of Christ 39 and Church of
   Christ (Non-denominational) 90.
3. **The detailed layer mostly describes people who did not answer in detail.** `Pentecostal, nfd`
   is 228,916 of 255,838 Pentecostals — **89% of the branch is "Pentecostal, unspecified"**, and
   the 18 named bodies share the other 11%. `Christianity, nfd` is 688,440, larger than every
   denomination except Western Catholic and the Anglican Church of Australia. `Other Protestant,
   nfd` is 32% of its parent. So spec §3.2's residual is not a tidy-up here — it is the biggest
   slice under several branches, and a map that drew only the named bodies would misrepresent
   Australian Pentecostalism by an order of magnitude.
4. **Yezidi went 63 → 4,123 in five years**, a 65× rise, the largest proportional change in the
   file. Not an artefact: Australia's 2015–2019 humanitarian intake resettled Yazidi survivors of
   the Sinjar massacre, concentrated in Wagga Wagga, Toowoomba and Coffs Harbour. `6901 Mandaean`
   (9,178) and `2075 Chaldean Catholic` (14,103) are the same story a decade earlier. Australia's
   religion tail is a refugee-policy record.
5. **Anglicans fell 3.10M → 2.50M and the Uniting Church 870k → 673k in five years**, while
   No Religion rose 6.93M → 9.77M. A 19% and 23% fall in one intercensal period, in absolute
   numbers, in a growing population.
6. **The one thing SA2 cannot show is the thing this project most wants.** Every non-Abrahamic
   minority except Buddhism, Hinduism and Sikhism — Baha'i, Taoism, Shinto, Paganism, Wicca,
   Jainism, Zoroastrianism, Mandaean, Yezidi, Druse, Caodaism, Spiritualism, Rastafari — collapses
   into the single `Other Religious Groups` column at SA2. The detail exists and the geography
   exists; ABS just never crosses them outside TableBuilder. If anyone ever gets a TableBuilder
   login, `RELP (religious group) × SA2` is the single highest-value pull available for Australia.
7. **Sikhism is the fastest-growing religion in the country** — 125,901 → 210,400, +67% — and is
   already larger than Judaism (99,956) and every Orthodox jurisdiction except Greek.
8. **The perturbation bias is systematic, not random.** I expected SA2 sums to scatter around the
   published nationals; they don't. Every rare category comes in *low* (§4). It is small in
   absolute terms, but it means a fine-geography build of a small group is always a floor.
9. **`Ratana (Maori)`, 3,271, has its own ASCRG code in the Australian census.** So does
   `Temple Society` (231) and `Grace Communion International` (204). The classification is
   remarkably generous with named bodies for a general-purpose national standard.
10. **ABS began reviewing the ASCRG in 2022**, so the 2026 census will not use this classification
    unchanged. Whatever cross-source matching happens later (spec §2.4) should not assume these
    codes are stable across censuses.
