# Pakistan — 2017 Population and Housing Census, via the U.S. Census Bureau

`sources/pk.py` → `data/normalized/pk.csv`. 207,684,626 people, **135 districts drawn**,
6 source categories → 5 nodes, 100% of the published tabulation.

| | |
|---|---|
| source | `PK_RELIGION_GEOG1_2017census_uscb_202401`, a layer of the USCB Pakistan geodatabase |
| publisher | **U.S. Census Bureau**, transcribing PBS *Table 9, Population by sex, religion and rural/urban* |
| url | `https://data.humdata.org/dataset/pakistan-subnational-population-and-housing-data-tables` |
| geography | **district (135 of 155)** — 1.54m people each. The file also has 585 tehsils and they are NOT drawn; see §3. |
| categories | 6: Islam, Christianity, Hinduism, Qadiani/Ahmadi, Scheduled Castes, Other |
| basis | `self_id` |
| year | 2017 |
| access | open, no login, no key. Two GETs, 8.7 MB. |
| licence | HDX, U.S. Census Bureau — public domain as a US Government work |

Second country off the seam in `sources.md` §11h; Ethiopia (§9u) was the first.
`sources/pk_geo.md` is the boundary and placement half.

**This country was discussed with Anita before it was built rather than after**, which is
what `spec.md` §14 asks for. Two of the six categories are not ordinary mapping decisions
and the whole of §3 and §4 below is that conversation, written down.

---

## 1. What the data is, and what it is not

A perfect partition, at every level:

| level | units with data | sums to national? |
|---|---|---|
| province (ADM1) | 6 of 8 | **yes, all 6 categories** |
| division (ADM2) | 30 of 36 | **yes** |
| district (ADM3) | **135 of 155** | **yes** |
| tehsil (ADM4) | 536 of 585 | **yes** |

and the national figure is **207,684,626**, PBS's own published census total — the one number
here that comes from outside the USCB file.

**No `-999`, and that is worth stating rather than assuming.** Ethiopia's file in the same
series used `-999` as its null sentinel and losing 0.03% of the country to it was the nastiest
trap in that ingest (§9u). Pakistan's file uses genuine nulls. **The convention is per country,
not per publisher**, so `pk.py` asserts that there are *zero* negative cells — a future release
that adopts the sentinel then fails loudly instead of silently subtracting.

## 2. Azad Kashmir and Gilgit-Baltistan have no data, and the file says why

20 of the 155 districts are all-null. They are not scattered: they are exactly
**Azad Kashmir (10)** and **Gilgit-Baltistan (10)**, and USCB's metadata is explicit —

> *Religion data for the autonomous and disputed regions of Azad Kashmir and
> Gilgit-Baltistan were not published by the Pakistan Bureau of Statistics.*

They are dropped, which costs nothing arithmetically (the 135 with data already sum to the
national total) and everything visually: about 6 million people and the entire north of the
country draw blank. `note_public` says what the hole is, because **an unexplained blank reads
as "nobody lives here"**, which is the one thing it must not.

There is a second loss inside that one. **Gilgit-Baltistan is where Pakistan's Shia population
is most concentrated** — it is the only Shia-majority region of the country — so the one place
where a Sunni/Shia division would be most visible is the one place with no data at all. The
census does not ask the question anywhere else either, so nothing is lost that could have been
drawn; it is worth knowing all the same.

## 3. The tier is district because of §14.4, and the file offered finer

**This is the most consequential decision in the ingest and it is not a data decision.**

The USCB file carries religion at **585 fourth-order units** — 366 thanas, 107 talukas, 56
sub-tehsils, 47 sub-divisions and a few others. `pk.py` normalises all of them.
`countries.py` draws **135 districts** and ignores the tehsils.

The reason is `spec.md` §14.4: *no resolution finer than the state's own publication*. So the
question is what PBS itself publishes, and it was checked rather than assumed:

- USCB's citation is **`census-2017-district-wise/`** — a district-level page. That URL now
  404s, and `pbs.gov.pk`'s 404 is a catch-all (identical 201,3xx-byte body for a real path and
  for `/nonexistent-path-xyz-12345`), so §9h's compare-the-404s test comes back negative and
  there is no API namespace to find.
- PBS's own **tehsil** release is on `pbs.gov.pk/censusarchive/` as one PDF per province —
  `punjab_tehsil.pdf`, `sindh_tehsil.pdf`, `kp_tehsil.pdf`, `balochistan_tehsil.pdf`,
  `fata_tehsil.pdf`. `sindh_tehsil.pdf` was downloaded and read: **6 pages, and it carries
  `TABLE-4 AREA, POPULATION BY SEX, SEX RATIO, POPULATION DENSITY, URBAN PROPORTION…` and
  nothing else.** Zero occurrences of "religion" in the whole file.

So PBS publishes religion at district and publishes area and headcount at tehsil. Drawing
religion on the tehsils would be finer than the state's own publication of that variable.

**What it costs is less than it sounds**, which is what made the decision easy:

| | tehsil (536) | district (135) |
|---|---|---|
| people per unit | 387,000 | 1,538,000 |
| Thar Hindu geography | Mithi 67.2% | **Umerkot 52.2%, Tharparkar 43.4%** |
| Punjab Christian belt | Lahore Cantt 8.3% | **Lahore 5.1%, Sheikhupura 3.8%** |
| Ahmadis | Lalian 13.6% | **Chiniot 4.4%** |

All three things worth seeing survive. 1.54m per unit is coarser than everything else on this
map except Kenya's 1.01m, which was accepted for its categories (§9o).

## 4. The two categories that needed a decision

### 4a. `Qadiani/Ahmadi` — kept, and filed under Islam

191,737 people, 0.09%. **Keeping them was Anita's call and the reasoning is worth recording,
because the instinct to protect by omitting is a good instinct that is wrong here.**

The Ahmadiyya community's central grievance is *erasure*: Pakistan's constitution declares
them non-Muslim, its penal code makes it a criminal offence for them to call themselves
Muslim, and the census accordingly prints `Qadiani/Ahmadi` as a peer of `Muslim` rather than
inside it. **Deleting them from a world religion map performs that same erasure from a
friendlier direction.** This map draws Guyana's 3,496 Rastafarians and Australia's 4,125
Yazidis by name; singling out Ahmadis for deletion would not be neutral. And there is no
informational novelty to protect — Rabwah being the movement's Pakistani headquarters is a
fact so public that the Punjab Assembly renamed the town over it.

**They are filed at `islam.ahmadiyya`, under Islam**, which is a substantive choice. `spec.md`
§2 says the tree is containment as it holds for people now, not as a state's law defines it.
Filing them outside Islam would make Pakistani constitutional law this map's taxonomy.

**The count is a floor by a large and unknown margin**, and the reasons are specific rather
than general: registering as Ahmadi carries a separate electoral roll and requires, for a
passport, a declaration disavowing the movement's founder — and the community has organised
census boycotts on exactly that ground since 1974. Every independent estimate is several times
191,737. Said in `note_public` as a floor, the way §11b says to read an African
`Traditionalist` cell; not corrected, because correcting it means inventing a magnitude
(§14.4).

**And this is the category that decided §3.** A third of all counted Ahmadis are in one place.
At tehsil the map would put Lalian at **13.6%**; at district it puts Chiniot at **4.4%**. The
district tier is what §14.4 permits and it is also, not coincidentally, the less pinpointed of
the two.

### 4b. `Scheduled Castes` — merged into Hinduism

849,614 people. **This is a caste category, not a religion**, printed by PBS as a peer of
`Hinduism` rather than inside it. These are Pakistan's Dalit communities — Meghwar, Bheel,
Kolhi, Bagri, Oad — and they are Hindu.

Filing a caste as a religion would be a category error; leaving it in a residual would erase
the largest Dalit population outside India. So both cells go to `hinduism`, and Pakistan's
Hindus come to **4,444,870** — the figure usually quoted for the country, which neither cell
gives alone.

**The merge costs something real and it is worth naming.** The two cells have different
geographies. Their shares correlate only **0.43** across tehsils, and the split is systematic:

```
irrigated Sindh  ->  records as "Hinduism"          Samaro taluka   51.0% Hindu,  0.4% SC
Thar desert      ->  records as "Scheduled Castes"  Islamkot taluka 15.4% Hindu, 43.8% SC
                                                    Nagarparkar     23.6% Hindu, 39.9% SC
```

That is caste Hindus of the canal belt against Dalit communities of the desert, and merging
flattens it.

**What decided it is that the 2017 split is unreliable and the publisher says so.** The 2023
National Census Report states that the only change from 2017 is *"improvement in reporting of
scheduled caste by clear differentiation between Hindu and scheduled caste"* — and Sindh's
Scheduled Caste count went from **831,562 to 1,325,559, up 59%**, while its Hindu count barely
moved. A boundary the publisher describes as newly fixed is not one to draw on the old side
of. `source_category` is kept verbatim, so a 2023 ingest can separate them (§2.4).

## 5. What the map shows

- **84 of 135 districts are over 99% Muslim.** Kohistan is 99.995%. That is the country, and
  the map should not pretend otherwise.
- **Umerkot is 52.2% Hindu — the only district in Pakistan without a Muslim majority** — with
  Tharparkar 43.4%, Mirpur Khas 38.7%, Tando Allahyar 34.2%, Badin 23.6%, Sanghar 21.8%. Sindh
  is 8.7% Hindu against a national 2.1%; Punjab is 0.19% and Khyber Pakhtunkhwa 0.02%. This is
  the part of Sindh that did not empty in 1947.
- **The Christian belt is central Punjab and it is urban and industrial**: Lahore 5.1%,
  Islamabad 4.3%, Sheikhupura 3.8%, Gujranwala 3.6%, Sialkot 3.5%, Kasur 3.5%, Faisalabad 3.4%.
- **Chiniot is 4.4% Ahmadi** and holds a third of the counted community.

## 6. What the map cannot show

- **No division of Islam**, for 200 million people. Pakistan's Muslims are overwhelmingly
  Sunni Hanafi, with the Barelvi and Deobandi movements inside that and a Shia minority usually
  put at 10–15%; the census asks none of it, and the region where it would be clearest has no
  data (§2).
- **No division of Christianity.** Roughly half Catholic, half the Church of Pakistan — a 1970
  union of Anglicans, Methodists, Lutherans and Presbyterians. One cell.
- **Sikhs and Parsis are invisible**, inside an `Other` cell of 43,253 people (0.021%, the
  smallest residual on this map). Nankana Sahib is the birthplace of Guru Nanak and Karachi
  holds one of the last Parsi communities; neither can be drawn. **The 2023 census gives both
  cells of their own**, so this is a fact about the 2017 form rather than about Pakistan.
- **No non-response category at all** — the six sum to the census population exactly. As in
  Ethiopia, that is a fact about the tabulation and does not mean nobody refused.

## 7. The 2023 census is better and is not obtainable

Checked before building, at Anita's request, and the answer is no:

- **It is better data.** 241.5 million; adds **Sikh** and **Parsi** cells; and fixes the
  Hindu/Scheduled-Caste differentiation (§4b).
- **It publishes religion by province.** Table 4.13 of the *National Census Report 2023* and
  §3.3.1 of each *Provincial Census Report* are province-level. Eight units is far below this
  map's floor.
- **The district tables exist and cannot be reached.** `Table 9, Population by sex, religion
  and rural/urban, census-2023` is listed in the report's own statistical-tables index, but
  the tables themselves live on **`census23.pbos.gov.pk`, which refuses connections** — tried
  twice, both schemes, both ports, on separate days. Wayback has **124 snapshots of that host
  and every one is the bare root page**; no table, XLSX or result URL was ever captured.
  §11f's oldest-snapshot trick has nothing to work on.
- **Only Islamabad has a 2023 district report.** `District-Census-Report-2023-Islamabad.pdf`
  exists; the same pattern 404s for Tharparkar, Umerkot, Lahore, Chiniot, Karachi and the rest.
  There is no per-district series to walk.
- **USCB has not published a 2023 file.** Theirs is `202401` and cites the 2017 census.

So 2017 at district is the current vintage.

### 7a. The 2023 data was located exactly, and it is behind a dead host — 2026-09-06

A second, harder look at Anita's request. **The 2023 table is now fully specified and still
unreachable**, which is a better place to leave it than "not found": when the host returns
this is an afternoon, not a search.

**What the portal was.** `census23.pbos.gov.pk` is an ASP.NET app whose analysis page ran on
one AJAX endpoint. Its own bundle, recovered from
`web.archive.org/…/census23.pbos.gov.pk/Scripts/Analysis/app.js`, gives the whole contract:

```
POST /Analysis/GetPopulation/
     Level:  1 = province | 2 = division | 3 = district | 4 = TEHSIL
     code1:  first  unit code      (page default '044' = Lahore District)
     code2:  second unit code      (page default '128' = Islamabad District)
POST /Misc/GetDropDownDataComp/   { Level: n }   -> the code list for that level
```

**It is a two-unit COMPARISON tool**, which is why the one archived response holds exactly two
records rather than a national table — and why a full harvest would be ~80 POSTs at district
or ~300 at tehsil. Entirely feasible against a live host.

**What the response carries.** Each record is ~600 fields, of which `r1`…`r8` are the eight
2023 religion categories — **two more than 2017, and they are the two that were missing**. The
one archived call (2026-08-06) yields two real districts, and both sum exactly:

| district | pop | r1 Muslim | r2 Christian | r3 Hindu | r4 Ahmadi | r5 Sch. Caste | r6 | r7 | r8 Other |
|---|---|---|---|---|---|---|---|---|---|
| Lahore | 12,978,661 | 12,363,149 | 602,431 | 2,487 | 7,139 | 324 | **715** | **77** | 2,339 |
| Islamabad | 2,283,244 | 2,181,663 | 97,281 | 839 | 2,398 | 45 | **60** | **10** | 948 |

`r6` and `r7` are Sikh and Parsi on every reading — they are new in 2023, they are tiny, and
their ratio to each other is right for both cities. **Not asserted as fact**: the category
labels were never captured, only the field order, so confirming which is which needs the
`Level=1` response checked against the *Provincial Census Report* percentages.

**And 2023 would move §14.4's ceiling.** The portal's own level selector offers **TEHSIL**, so
for the 2023 vintage the state publishes religion below district and §3's argument would have
to be re-made rather than reused.

**What was tried and failed**, so nobody repeats it: the host refuses connections on both
schemes and both ports across three days; `/Analysis/*` on `www.pbs.gov.pk`, `www.pbos.gov.pk`
and `census.pbos.gov.pk` all 404; `pbs.gov.pk/census-2023-district-wise/results/003` existed
and is now a 404, and its only two archive captures are 301s; the Wayback CDX for the whole
host is **13 distinct URLs**, of which the two `Analysis` ones above are the only data; PBS's
WordPress media library (5,407 items, enumerated exhaustively rather than searched) carries the
2017 `Table09p-*.xls` set and no 2023 equivalent.

**Two routes deliberately NOT taken**, both of which would have produced a Sikh and a Parsi
node today.

*A §3.4 re-base* — 2017 district structure carrying 2023 magnitudes, as `br_rescale.py` does
for Brazil. Refused: Sikhs and Parsis exist at province level only in that source, and
spreading Punjab's Sikhs evenly across its 41 districts would invent a geography for the exact
community the upgrade is meant to show. Nankana Sahib is not an average Punjabi district.

*Splitting the existing `Other` dots within each province by the 2023 shares* — the
`allocate.py --within` move. Refused for two reasons that are in the data rather than in
principle:

- **`Other` is a mixture of communities that do not overlap.** It runs 20.8 per 100,000
  nationally and **835 in Chitral — 40x, and 39.8% of Khyber Pakhtunkhwa's whole cell.** That
  is the Kalasha, who live in three valleys and nowhere else. Karachi South (129) is the
  Parsis; Nankana Sahib (122) the Sikhs. Splitting proportionally would give two fifths of
  KP's Sikhs to Chitral and relabel the Kalasha as Sikh — the largest and most visible thing
  in the layer, made wrong.
- **The two residuals do not reconcile.** 2023's `Others` alone is **90,341**, over twice
  2017's whole `Other` of 43,253, *and* 2023 has separate Sikh and Parsi cells besides. There
  is no defensible fraction of the 2017 cell to call Sikh.

**The rule worth carrying:** `--within` needs *no evidence against uniformity*, which is a
condition to test, not assume — compute the child units' rates for the category first. A unit
at 40x the mean is a different population wearing the same label. §14.4.

## 8. Not done

- **The tehsil tier is normalised and unused**, deliberately (§3). If PBS is ever found to
  have published religion below district, the rows are already in `pk.csv`.
- **The 1998 census** is the previous one with religion and would give a 19-year change map.
  Not checked; USCB's file does not carry it.
- **`PK_GEOG2_*_2010` layers** (a second geography vintage, 147 ADM2 units) are in the same
  geodatabase and are not used — they carry the 2010 agricultural census, not religion.
