# Small territories: eight from §11ap

Built 2026-09-14, session `f95259a4-terr`, from the buildable Caribbean and Atlantic rows of
sources.md §11ap. The microstate tier's shape (sources/micro.md): a place small enough that one
polygon, or a few islands, carries no claim about placement at 1 dot = 1,000 people.

| code | territory | source | year | units drawn | cats | people | on nodes | dots 1:1k | rings |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `vg` | British Virgin Islands | census report Table 77 | 2010 | 4 islands | 22 | 28,054 | 27,371 | 21 | 11 |
| `fk` | Falkland Islands | Census 2016 Table 6 | 2016 | 3 locations | 8 | 3,198 | 3,006 | 2 | 5 |
| `kn` | Saint Kitts and Nevis | Dept of Statistics web table | 2011 | 1 | 21 | 47,195 | 47,139 | 39 | 9 |
| `sx` | Sint Maarten | UNSD table 28 (= census Table B-10) | 2011 | 1 | 17 | 33,609 | 32,813 | 26 | 5 |
| `ai` | Anguilla | UNSD table 28 | 2001 | 1 | 18 | 11,430 | 11,391 | 5 | 15 |
| `aw` | Aruba | census report Table P-A.5 | 2010 | 1 | 10 | 101,484 | 100,968 | 95 | 4 |
| `cw` | Curaçao | Census 2023 Table D-5 | 2023 | 1 | 14 | 155,826 | 151,738 | 146 | 4 |
| `bq` | Caribbean Netherlands | CBS 82868NED, Omnibus survey | 2021 | 3 islands | 12 | 27,726 | 27,510 | 23 | 4 |

Code: `sources/terr.py` (vg fk kn aw cw), `sources/bq.py`, and two rows added to
`sources/micro.py` (sx ai). Mappings `taxonomy/<cc><year>.py`; eight `other.<cc>` nodes in
`branches.py`, nothing else new on the tree. Nothing was parked or closed.

## 1. Routes, and what each table was checked against

- **`vg`**: `unstats.un.org/unsd/demographic/sources/census/wphc/BVI/VGB-2016-09-08.pdf`, report
  p.60 (PDF p.67). Transcribed; every island column sums to its printed total (Anegada 285,
  Cooper Island 26, Great Camanoe 6, Jost Van Dyke 298, Tortola 23,491, Virgin Gorda 3,930,
  Yachts 18). Compared with UNSD's 2010 national row: equal on 21 of 22 rows.
- **`fk`**: Falkland Islands Government, *Census 2016 Full Report*, Table 6 (PDF p.73). Each
  location sums to its total. 2016 is newer than UNSD's 2006 row.
- **`kn`**: `stats.gov.kn/topics/demographic-social-statistics/population/population-by-religious-belief-2011/`,
  a WordPress data table whose cells carry `data-original-value`; transcribed and compared cell by
  cell with the saved page on every run. The page says only *Source: Department of Statistics*.
- **`sx`**: UNSD's row, and the census report's Table B-10 (`stats.sintmaartengov.org/download.php?type=rep&section=CEN&nummer=9`,
  PDF pp.31-32) was read and matches it to the person.
- **`ai`**: UNSD's row, exact. Not re-opened; §11ap found the 2001 report matches.
- **`aw`**: Wayback copy of CBS Aruba's *Fifth Population and Housing Census* (report p.82, and
  the person form on p.270). The fetch needs Wayback's `id_` form or urllib saves a 9 KB redirect
  page as the PDF; `terr.py` now checks the file signature. Compared with UNSD's 2010 row.
- **`cw`**: CBS Curaçao's live workbook; the transcription is compared with it on every run.
- **`bq`**: CBS OData, `82868NED` (shares, margins) and `83774NED` (population). JSON, no key.

## 2. Where UNSD's row is wrong, and how

- **British Virgin Islands: a number.** UNSD prints Muslim **255**; Table 77 prints **266** on
  both its national and its by-island table, and the island columns close with 266. The 11
  people are exactly the gap that makes `oracle.py --list` report the country as `NOT a
  partition`. `terr.py` pins this one disagreement, so any other change fails the build.
- **Aruba: a label.** UNSD's `Pagan` 5,625 is the census's `No religion` 5,625. The person form's
  question 4 offers Roman Catholic, Protestant (reformed), Jehovah's witness, Methodist,
  Adventist, Anglican, Jewish, No religion, and Other with a line; there is no pagan box. The ten
  rows sum to 101,483 against a printed 101,484 in both the census and UNSD.
- **Sint Maarten: labels only**, and harmless: UNSD's `Christian` and `Unknown` are the report's
  `Christianity` and `Not reported`.

## 3. Placement

Kontur 2023-11-01 extracts, people against the census:

| | hexes | Kontur | census | ratio |
|---|---:|---:|---:|---:|
| `vg` Tortola (with the three folded units) | 185 | 26,091 | 23,541 | 1.11 |
| `vg` Virgin Gorda | 60 | 4,503 | 3,930 | 1.15 |
| `vg` Jost Van Dyke | 19 | 331 | 298 | 1.11 |
| `vg` Anegada | 37 | 330 | 285 | 1.16 |
| `fk` Stanley | 50 | 3,570 | 2,458 | 1.45 |
| `fk` Mount Pleasant Complex | 75 | 567 | 359 | 1.58 |
| `fk` Camp | 1,240 | 2,921 | 381 | **7.67** |
| `kn` | 373 | 47,775 | 47,195 | 1.01 |
| `sx` | 93 | 46,057 | 33,609 | 1.37 |
| `ai` | 161 | 15,618 | 11,430 | 1.37 |
| `aw` | 279 | 107,034 | 101,484 | 1.05 |
| `cw` | 503 | 190,649 | 155,826 | 1.22 |

- **BVI islands are assigned by coordinates** (`terr._unit_vg`): Anegada north of 18.65°N, Jost
  Van Dyke west of -64.705° and north of 18.43°N, Virgin Gorda east of -64.47°, Tortola the rest.
  The four ratios agree within 0.05 of each other, which is the check that the lines are right.
  **Cooper Island (26), Great Camanoe (6) and Yachts (18) get no hex** and `countries.py` folds
  them into Tortola; 50 people, no dot either way.
- **Falklands units are two 5 km circles** around Stanley and Mount Pleasant, and Camp is
  everything else, which is the census's own definition. Kontur spreads 7.7 times Camp's people
  over it. It changes nothing: at 1:1,000 the islands draw two dots, both in Stanley (checked in
  `dots_fk.geojson`), and at 1:10,000 none.
- **Caribbean Netherlands has no Kontur hexes at all.** The BQ extract is a 2.6 KB valid
  GeoPackage with zero features. The islands are Natural Earth's map unit `NLY` (three parts,
  split by centroid) with dots spread evenly inside each (spec §8.2). Bonaire's 20 or so dots
  therefore include some in Washington Slagbaai park and on the salt pans.
- **`country_shapes.py` did not know the Caribbean Netherlands**, because `admin_0_countries`
  folds it into the Netherlands and the script stops on a registered country it cannot find.
  `FROM_UNITS = {"bq": "NLY"}` reads it from the map-units file instead.
- `kontur_cap.py` finds no block at the cap in any of the seven layers.

## 4. Curaçao is one polygon at 156,000 people, and that is a decision

§11aa's permission is for places *under ~100k*, where placement asserts nothing. Curaçao is
155,826 on 444 km² and draws 146 dots. **It is drawn as one polygon anyway**, because:

- nothing finer exists: §11ap found no geozone or district table with religion in the 2011 or
  the 2023 census (not re-checked here), so the choice is one polygon or not drawing Curaçao;
- the island is small enough that Kontur's surface does the only placing there is to do, and
  the dots land on Willemstad and the inhabited west, which is where the people are;
- the note says in plain words that the dots follow population and not religion.

Aruba, at 101,484, sits on the line and is treated the same. What would reverse this: a Senso
table of religion by geozone or *bario*. Then Curaçao should be redrawn at that tier.

## 5. Curaçao's 2011 table has two labels swapped

§11ap flagged Hinduism at 3,058 (2011) against 1,211 (2023). The 2011 D-5
(`cuatro.sim-cdn.nl/cbscuracao/uploads/d-5-population-by-religion-age-group-and-sex.xls`) looks
wrong, not the 2023 one:

| 2011 row label | total | female | 65+ | looks like 2023's | total | female | 65+ |
|---|---:|---:|---:|---|---:|---:|---:|
| `Hinduidm` | 3,058 | 62% | 13% | Jehovah's Witness | 3,184 | 62% | 31% |
| `Jehova's Witness` | 1,222 | 46% | 5% | Hinduism | 1,211 | 51% | 12% |

Read with the labels swapped, both groups change slightly over twelve years instead of one
falling 60% and the other rising 160%. It is an inference from shape, not a correction the
office has published, and it does not touch the build, which draws 2023 only.

## 6. Caribbean Netherlands: the survey decisions

- **Round:** 2021 only (fieldwork October to December 2021; CBS marks it provisional). The
  2017/2018 round is printed beside it on every run and the island shapes hold: Bonaire Catholic
  59.7 to 60.3, Sint Eustatius Methodist 28.6 to 24.8 and Adventist 17.8 to 18.9, Saba Catholic
  43.7 to 50.1. Pooling was not done, because the two rounds withhold different cells.
- **Population base:** 1 January 2022 (Bonaire 22,573, Sint Eustatius 3,242, Saba 1,911), the
  date nearest fieldwork. The islands have grown about a fifth since, mostly by immigration.
- **Universe:** persons 15 and over; the shares are applied to all ages, as every survey build
  here does. Rows are `modelled`.
- **Withheld cells:** 100 minus the printed shares (0.7%, 0.7%, 1.8%) is `Niet gepubliceerd`,
  excluded, and is the whole of `gap_share` (216 of 27,726). It is not spread over the withheld
  religions, because CBS did not say how big each one is.
- No split-half test: three units cannot carry a rank correlation, and each island is measured
  directly.

## 7. Open threads, none blocking

- **St Kitts and Nevis, 47,195 against 46,398.** §11ap says the table's total is 797 above a
  census count quoted elsewhere; not resolved. The 2021-22 census summary at
  `nia.gov.kn/wp-content/uploads/2025/03/Census-Report-2021-2022.pdf` now 404s. A search
  summary attributes these same shares to *the 2011 census* (US State Department report); a
  lead, not evidence, but `how` says census on the strength of the year and the table's shape.
- **`Church of God` goes to `christianity.holiness`** in vg, kn and ai, with ag, bm, dm, ms and ky.
  BVI prints the New Testament Church of God (Cleveland) separately, so the bare cell there is
  plausibly the Holiness body; St Kitts and Anguilla give no such hint.
- **`Wesleyan Holiness` stays at `christianity.holiness`** in kn, as in ky, although a
  `christianity.holiness.wesleyan` node now exists from the US build. Move both or neither.
- **Protestant, three ways across the Dutch islands:** Aruba's form says *Protestant,
  reformed* and goes to `christianity.reformed.continental`; Curaçao and the Caribbean
  Netherlands print a bare `Protestant` and go to `christianity.protestant`.
- **Sint Maarten's paired rows** (`Islam / Judaism` 377, `Buddhism / Sikh` 88) are in `other.sx`.
- **Name:** `Falkland Islands`, as the census publisher and Natural Earth write it; UNSD's
  `Falkland Islands (Malvinas)` was not used.
