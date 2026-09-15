# Pakistan — 2017 Population and Housing Census, via the U.S. Census Bureau

**Since 2026-09-14 the drawn build is the 2023 census, PBS Table 9, at 136 districts: see §9.** Sections 1 to
8 describe the 2017 build, which is kept and no longer drawn.

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

### 7b. The 2023 district and tehsil tables are on pbs.gov.pk after all, 2026-09-14

Found by the sect sweep (`sources/branches.md`), not ingested. **§7's "not obtainable" is wrong.**

    https://www.pbs.gov.pk/wp-content/uploads/census_tables/tables/table_9_kp_districts.pdf
    ...table_9_punjab_districts.pdf, ...table_9_sindh_districts.pdf, ...table_9_balochistan_districts.pdf

All four answer 200, `application/pdf`, 2.3 to 2.7 MB, `Last-Modified` 2025-01-22. The KP file was
read: *Table 9: Population by sex, religion and rural/urban, Census-2023*, province then **district
then tehsil**, each by all/rural/urban and sex, with columns Muslim, Christian, Hindu Jati,
Qadiani/Ahmadi, Scheduled Castes, **Sikh, Parsi**, Others. KP totals 40,641,120, with 4,050 Sikhs
and 36 Parsis. That header order is §7a's `r1`…`r8`, so §7a's reading of `r6` and `r7` as Sikh and
Parsi is confirmed.

**Why §7a's exhaustive media-library sweep missed it**: these files sit under
`wp-content/uploads/census_tables/`, and they are not WordPress media items. Islamabad's file name was
not probed. A rebuild on 2023 re-opens §3's §14.4 tier argument, because the state now prints Ahmadis
by tehsil.

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

---

## 9. Rebuilt on the 2023 census at district, 2026-09-14

**Pakistan is now drawn from the 2023 Digital Census, PBS's own Table 9, at 136 districts.** Anita
approved it the same day §7b found the tables: *"ok we can rebuild pakistan at district."* §1 to §8
above describe the 2017 build, which is kept whole and no longer drawn: `sources/pk.py` (now writing
`data/normalized/pk2017.csv`), `sources/pk_geo.py`, `taxonomy/pk2017.py` and `_pk_counts`.

| | 2017 (USCB) | 2023 (PBS) |
|---|---|---|
| modules | `pk.py`, `pk_geo.py`, `pk2017.py` | `pk_2023.py`, `pk_2023_geo.py`, `pk2023.py` (taxonomy) |
| people | 207,684,626 | **240,458,089** |
| districts drawn | 135 | **136** |
| categories -> nodes | 6 -> 5 | **8 -> 7** |
| boundaries | USCB geodatabase, identity join | COD-AB v01 tehsils + OSM for Karachi |

`taxonomy/registry.py` pins `OVERRIDE['pk'] = 'pk2023'`, because two vintages on disk otherwise make
`discover()` refuse for every consumer.

### 9.1 The source

Five PDFs under `https://www.pbs.gov.pk/wp-content/uploads/census_tables/tables/`: `table_9_kp_districts.pdf`,
`table_9_punjab_districts.pdf`, `table_9_sindh_districts.pdf`, `table_9_balochistan_districts.pdf`, and
**`table_9_islamabad.pdf`**, which is the name §7b had not probed (no `_districts`; `table_9_islamabad_districts.pdf`
and `table_9_ict_districts.pdf` both 404). All `Last-Modified` 2025-01-22. Browser User-Agent.

National, and every figure below reconciles to Table 4.13 of the *National Census Report 2023*:

| Table 9 cell | people | share | node |
|---|---|---|---|
| Muslim | 231,686,709 | 96.352% | `islam` |
| Hindu Jati | 3,867,729 | 1.609% | `hinduism` |
| Scheduled Castes | 1,349,487 | 0.561% | `hinduism` (merged, §9.4) |
| Christian | 3,300,788 | 1.373% | `christianity` |
| Qadiani/Ahmadi | 162,684 | 0.068% | `islam.ahmadiyya` |
| **Sikh** | **15,998** | 0.007% | `sikhism`, new cell |
| **Parsi** | **2,348** | 0.001% | `zoroastrianism`, new cell |
| Others | 72,346 | 0.030% | `other.pk` |

Both new nodes were already on the tree from the US build, so nothing was added.

### 9.2 What the parse asserts, and the two traps in it

`sources/pk_2023.py` reads the PDFs by span geometry with PyMuPDF. All of these pass:

- every block's 12 rows: Total = the eight religions; ALL SEXES = male + female + transgender; ALL
  LOCALITIES = rural + urban, on all nine columns;
- 136 district blocks, matching PBS's *List of Administrative Districts (as on 01-03-2023)* province by
  province (35 KP, 36 Punjab, 30 Sindh, 34 Balochistan, 1 Islamabad);
- 590 tehsil-tier units sum to their districts, and districts to their provinces, on all nine columns;
- the five provinces equal Table 4.13 cell by cell, and sum to its Pakistan row;
- Lahore and Islamabad equal §7a's two archived census23 portal records on all nine columns, which is an
  independent publication path.

**Table 4.13 misprints Punjab's Muslim cell as 24,462,897.** The row's own total less its other cells is
124,462,897, and the Pakistan row only adds up with that, so the check holds the corrected figure.

**Trap 1: the numbers are right-aligned in columns of different widths.** The first parser assigned cells by
midpoints between the printed column numbers `1`..`10`, and on KP page 1 a one-digit Scheduled Castes cell
landed in Sikh. The table's header draws its vertical rules as thin rectangles; reading those and assigning
each number by its RIGHT edge is exact. A cell that collides or overruns the last rule fails loudly.

**Trap 2: one district header does not say DISTRICT.** Malakand's block is headed `MALAKAND PROTECTED AREA`,
so a suffix rule filed it and its two sub-divisions under Lower Kohistan. The per-province district count
against PBS's list caught it; `DISTRICT_HEADERS` names it.

Page 1 of each file prints thousands separators and later pages do not, and the Muslim column is bold on
some pages; neither matters to a geometry read, and every block's sums would catch a misread digit.

### 9.3 Who is not in it

- **1,041,342 people in restricted areas, counted by head only.** NCR p.124: the 241,499,431 total
  *"includes individuals from restricted areas for whom only headcounts are available. Consequently, detailed
  demographic characteristics such as ... religion ... are available for only 240,458,089"*. They are in no
  table at any level. `gap_share=0.004312`, hand-written, since `tools/gap_share.py` cannot see a group that
  is in no column.
- **Azad Kashmir and Gilgit-Baltistan, which is §2 again, reworded.** The 2023 report says census districts
  were set up in both and the pilot census ran there (PDF pp. 24, 85, 87), but the published 241.5m and every
  table cover only the four provinces and Islamabad; its migration figure counts only out-migration from them
  (p.140). So "enumerated, not published", and `note_public` now says they are in no published table.

### 9.4 The mapping, `taxonomy/pk2023.py`

Every pk2017.py decision is kept. **Scheduled Castes stays merged into Hinduism, and one of §4b's two reasons
has weakened.** §4b merged partly because PBS called the 2017 split unreliable; 2023 is the fixed split, and it
is sharp: Tharparkar is 27.0% Scheduled Castes and 18.7% Hindu Jati, Umer Kot 11.3% and 43.3%. The merge now
stands on the other reason alone, that a caste is not a religion and a legend row for it would draw the state's
caste line as a line between faiths. Recorded in the module's REVIEW. `source_category` is verbatim, so the
split is one mapping line away if that is ever wanted.

### 9.5 The tier is district, by choice rather than by §14.4

§3 drew district because PBS published religion no finer. **That is no longer true**: 2023 Table 9 prints every
tehsil, sub-division, taluka and sub-tehsil, and Lalian tehsil is 13.41% Ahmadi where Chiniot district is 4.30%.
Anita chose district anyway on 2026-09-14. The 591 tehsil-tier rows are in `pk.csv` and `_pk2023_counts` does
not read them.

### 9.6 Boundaries: the 2023 district set is in no one file

`sources/pk_2023_geo.py`. The census's 136 districts are PBS's list as on 01-03-2023. **OCHA's COD-AB v01**
(HDX `cod-ab-pak`, `valid_on` 2022-09-09, reviewed 2024-09-27, resources re-uploaded 2026-08-14) has 160 ADM2,
of which 24 are AJK and GB. The remaining 136 differ from the census's 136 in exactly three places:

- **129 districts join by name, 1:1 in both directions, within province.** Eight need `ALIAS`: Dera Ismail
  Khan / `D. I. Khan`, Lower and Upper Chitral / `Chitral Lower`, `Chitral Upper`, Lower and Upper Kohistan /
  `Kohistan Lower`, `Kohistan Upper`, Malakand Protected Area / `Malakand`, Layyah / `Leiah`, and **Surab /
  `Shaheed Sikandarabad`**, which is the same district under its old name.
- **COD's Lehri is not a census district, and it went to two parents.** The census prints the LEHRI
  sub-division under Sibi and BHAG under Kachhi. COD's ADM3 has both as tehsils under Lehri, so the polygons
  are reassigned tehsil by tehsil. Nothing is dissolved.
- **Keamari is a census district (notified 2020) and not a COD one, and Karachi cannot be rebuilt from COD at
  all.** COD's ADM3 under Karachi is the 2001 towns, and 2023's Karachi West includes Manghopir (1.08m), which
  COD has inside Gadap Town under Malir. So **Karachi's seven districts come from OpenStreetMap**: the seven
  admin_level=6 relations (`OSM_KARACHI` in the script; OSM still labels West as *Orangi District* and South as
  *Karachi District*), clipped to COD's six-district footprint so the city's outer edge stays COD's. Every
  pairing is confirmed by OSM's subarea towns against the census's own sub-division names, e.g. Keamari's
  Baldia, SITE, Keamari and Mauripur. OSM covers 93.2% of COD's Karachi by area; the gaps hold 18 hexes and
  2,806 people, who go to the nearest district, and that population (0.01% of Karachi) is what is asserted.
  OSM's coastal districts run out over the sea (5,841 km2 against COD's 3,849), which is why an
  intersection-over-union bar failed at 0.588 and was replaced. ODbL, (c) OpenStreetMap contributors.

**So no census district was dissolved into a parent.** The checks, beyond the name join:

- **Second key**: 468 of 501 COD tehsil names (93%) appear among the census tehsils of the district they were
  paired with. The one pair with none is Malakand, whose COD tehsils are named for their towns (Bat Khela,
  Dargai) and whose census sub-divisions for the Ranizai areas those towns are seats of; it is named in
  `TEHSIL_NAMES_DIFFER` with that reason.
- **Kontur 2023-11, joined on hex centroids** (the same file as 2017, re-unpacked from the .gz and removed
  again): 364,283 hexes kept, 88 to 8,843 per district, 236,154,648 people against the census's 240,458,089,
  **ratio 0.982**, per district median 0.98, quartiles 0.77 to 1.08. The low tail is Balochistan (Surab 0.44,
  Quetta 0.46, Kharan 0.47, Kech 0.52) and it is not the join, since Quetta can pair with nothing else: the
  2023 census counts Balochistan far above what the grid expects. Karachi's central districts run 0.67 to
  0.74. The weight is within-district only, so neither moves a dot between districts.
- **§12's geography witness**: Hindu share against its 5 nearest districts r=0.92, Christian r=0.75, where
  200 shuffles of the same shares reach at most 0.49 and 0.39.
- 16,435 hexes (5.43m people) fall in no district: AJK, GB and border overrun.

### 9.7 What 2023 shows

- **78 of 136 districts are over 99% Muslim**; Umer Kot, 54.66% Hindu, is the only Muslim-minority district.
- **Hindus**: Umer Kot 54.7%, Tharparkar 45.6%, Mirpur Khas 41.5%, Tando Allahyar 36.6%, Badin 25.1%, Sanghar
  24.5%. Sindh 8.81%, Punjab 0.20%, KP 0.02%.
- **Christians**: Lahore 4.64%, Islamabad 4.26%, Sheikhupura 3.67%, Gujranwala 3.50%, Sialkot 3.46%, Kasur
  3.43%, Korangi 3.42%, Faisalabad 3.40%.
- **Ahmadis**: 162,684, down from 191,737; Chiniot 67,223 (4.30%).
- **Sikhs**: Nankana Sahib 1,887 (11.8% of the national count), Peshawar 1,481, Buner 1,023, Attock 769.
- **Parsis**: Karachi South 952 (40.5%), Karachi East 328, Karachi West 222, Rahim Yar Khan 175.
- **Others**: Lower Chitral 1,451 per 100,000 (4,617 people), which is the Kalasha, then Awaran 669, Gwadar 305
  and Panjgur 225, all Makran. The Zikri community is the likely reading of the Makran spike and nothing in the
  census confirms it; the node note says so in those terms. §7b's *"a unit at 40x the mean is a different
  population wearing the same label"* still describes this cell with Sikhs and Parsis taken out.

### 9.8 Not done

- **The tehsil tier is normalised and not drawn** (§9.5).
- **COD-AB v01 predates districts notified after 01-03-2023** (Murree, Talagang, Kot Addu, Taunsa, Wazirabad,
  Hub, Usta Muhammad). The 2023 census does not use them, so they do not matter until a later vintage does.
- **Karachi's outer edge is COD's and its internal lines are OSM's**; a hex near the join of the two sources can
  land one district over. 18 hexes needed the nearest-district rule, so this is small.
- **The 1998 census** is still unchecked.
