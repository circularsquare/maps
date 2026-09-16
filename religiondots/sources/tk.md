# Tokelau — 2016 Tokelau Census of Population and Dwellings, religion by atoll

**Drawn 2026-09-15** (session `cb8b206e-tk`). Three units, the atolls; 5 nodes; 1,188 people drawn
of the 1,197 usual residents present on census night; every row `measured`. `gap` is the 302 usual
residents who were overseas plus the 9 not stated, 20.75% of the 1,499 usual residents. **0 dots and
5 rings at 1:1,000 and at 1:10,000**: the largest body is 603 people, under one dot even nationally.

- `sources/tk.py` -> `data/normalized/tk.csv` (two workbooks and the profile report in `data/raw/tk/`,
  pinned)
- `sources/tk_geo.py` -> `data/geo/tk/tk_atolls.gpkg`, `tk_hexes.gpkg`, `tk_lookup.csv` (Kontur
  Boundaries TK in `data/raw/tk/`, Kontur population TK in `data/geo/kontur/`, both pinned)
- `taxonomy/tk2016.py` the mapping; `countries/tk.py` the entry
- `taxonomy/branches.py` gains `christianity.reformed.congregational.tokelau`;
  `country_shapes.py::FROM_UNITS` gains `"tk": "TKL"`; `sources/geo_checks.csv` a `grid_floor,tk` row
- sources.md **§tk-2026-09-15** is the summary; **§scout-2026-09-14-asia-oceania** was the scout's.

```
python sources/tk.py --fetch
python sources/tk_geo.py --fetch
```

## 1. What the Tokelau National Statistics Office publishes

Everything below is linked from the office's census page,
`tokelau.org.nz/Tokelau+Government/Government+Departments/Office+of+the+Council+for+the+Ongoing+Government+OCOG/Tokelau+National+Statistics+Office/census.html`.

| release | religion | tier |
|---|---|---|
| **2016 census, *Tables about social profile* (xlsx, 1 February 2017)** | **Table 5.8, religious affiliation by atoll of usual residence, 2011 and 2016, persons** | **3 atolls** |
| 2016 census, *Tables about demography* (xlsx) | none; 1.3.1 de jure and 1.3.2 absentees by atoll, used as checks | atolls |
| *Profile of Tokelau: 2016 Tokelau Census* (95 pp.) | prose and Figure 5.4 (a chart) on printed p.28; the religion classification on pp.76-77 | atolls |
| atoll profiles 2016 (one PDF per atoll) | not opened | atoll |
| 2006 census tables (xls) and analytical report | not opened; UNSD table 28 has 2006 nationally | |
| 2019 population count | no religion (the scout's reading) | |
| 2022 census | licensed microdata: Pacific Data Hub `spc_tkl_2022_phc_v01_m` (catalogue 834), ILO survey library 8528; SPC's SDD collection page. **All three answered 403 to WebFetch on 2026-09-15**, so whether the 2022 form asks religion is unchecked. A browser job, and the one lead for a newer vintage | |
| UNSD Demographic Yearbook table 28 | 2016, 2011, 2006, national | national |

The office's page lists no 2019 or 2022 files.

## 2. The files

| file | URL (under `tokelau.org.nz/site/tokelau/files/TokelauNSO/2016Census/` unless given) | digest | bytes |
|---|---|---|---|
| `tk2016_tables_social_profile.xlsx` | `2016 Tokelau Census of Population and Dwellings - Tables about social profile.xlsx` | `IAXB35RC7XF25GUELMAR5QHJAHOIDRPH` | 177,914 |
| `tk2016_tables_demography.xlsx` | `... - Tables about demography.xlsx` | `WJK4TS75ILLVUNRZAFBJIGNHTPIXFVPL` | 75,125 |
| `tk2016_profile_report.pdf` | `profile-tokelau-2016-census-final-to-print28jun17jj.pdf` | `2IZBMBTJFWAYOYOC4NWCX5THXAU4GTRM` | 2,395,295 (95 pages, `%%EOF`) |
| `kontur_population_TK_20231101.gpkg.gz` | Kontur S3 `kontur_datasets/` | `SNOBIGKEAJP5Y7R2A7TEYRGCX2YS6CIO` | 4,360 |
| `kontur_boundaries_TK_20230628.gpkg.gz` | Kontur S3 `kontur_datasets/` (HDX `kontur-boundaries-tokelau`, ODbL) | `C6CRWPADY2YQBSM5K72GTZ4JIINR5DZL` | 25,852 |

## 3. The checks (`sources/tk.py::check`)

| check | result |
|---|---|
| files pinned | all three |
| Table 5.8 parsed by label = transcription | 7 rows x 3 atolls, 2011 and 2016 |
| stray whitespace in labels | only `Congregational Christian ` (trailing space), stripped |
| caption | "For usually resident population present in Tokelau on census night", by atoll of usual residence |
| each atoll's rows sum to its total; atolls sum to each row's Total | both years: 413 / 399 / 385 = 1,197 (2016), 385 / 449 / 309 = 1,143 (2011) |
| demography 1.3.1 de jure minus 1.3.2 absentees = Table 5.8's atoll totals | both years, every atoll |
| profile printed p.15 | "The de jure usually resident population in 2016 was 1,499", "1,197 ... present ... and 302 ... overseas", 48 TPS in Apia and 254 elsewhere |
| profile printed p.16, Table 4.1 | present 413 / 399 / 385, absent 106 / 85 / 63, Samoa 48 |
| profile printed p.28 | the form offered "Congregational Christian, Roman Catholic, and Presbyterian" and "other, please specify" |
| profile printed pp.76-77 | classification codes 1-8, 99 Other, 999 Not Stated, based on RELIGAFF (NZ 1999) |
| UNSD 2011 | equal to Table 5.8's 2011 column |
| UNSD 2016 | three cells differ, 4 people in all (office, UNSD): Roman Catholic 463 / 460, Other Christian 50 / 54, No religion 1 / 0. Pinned; the office's table is the source |

The profile's prose gives Congregational shares that leave out not stated (Atafu 78.3%, which is
318 of 406), and says Presbyterian was 4.2% overall where the table gives 5.9% (4.2% is Other
Christian). The note uses the table's shares of everyone present (Atafu 77.0%).

## 4. The universe, the form and the gap

**The table covers the usual residents present on census night, 1,197.** Tokelau's official count
is the de jure usually resident population, 1,499 (the count that sets each atoll's seats in the
General Fono): the 1,197, plus 254 usual residents temporarily overseas (for under 12 months; 35%
of absentees were away for schooling, printed p.19), plus 48 Tokelau Public Service employees and
their families based in Apia. Heads of household gave "some basic information about absentees"
and the Apia staff answered a shorter form of "basic demographic questions" (printed pp.15, 75), so
none of the 302 has a religion. The census night count, 1,285, adds visitors and temporary
residents and is not used.

**The form** (printed p.28) named three churches with an "other, please specify" option. No religion
exists only as a write-in (classification code 8), and 1 person wrote it. The contents sheet of the
workbook says most `not stated` answers in 2016 came from people whose age was imputed, so the
questions after it went unanswered.

**`gap` and `gap_share`.** The 302 who were not asked and the 9 not stated are both inside the 1,499
usual residents, so they add in one universe: 311 / 1,499 = 0.2075, stated as 20.7% in `gap`.
`tools/gap_share.py` sees only the 9 (0.75% of the rows), because the 302 are in no religion table;
the authored figure is larger, which its `--check` allows. The viewer's bar total is then 1,188 /
0.7925 = 1,499, the official count. What reversing takes: `gap_share=0.0075` and a `gap` naming only
the 9, if the bar should be the residents present.

## 5. Geography

**No COD-AB exists** (HDX `cod-ps-tkl`: "No administrative boundaries common operational dataset
(COD-AB) is available for Tokelau"), and geoBoundaries gbOpen TKL has ADM0 only (a Sentinel-2 land
mask). Natural Earth's countries file has no Tokelau feature; its map-units file has `TKL` (GEOUNIT
Tokelau, ADMIN New Zealand), taken through `country_shapes.py::FROM_UNITS`.

**Units: Kontur Boundaries TK 2023-06-28** (OpenStreetMap), admin level 8: Atafu, Fakaofo and
Nukunonu, each a sea area of 2,039, 2,414 and 2,655 km2 around its atoll, not overlapping. They only
assign hexes to atolls; the nearest two atolls are about 60 km apart.

**Witness the name tag cannot decide.** The profile report (printed p.10): "Nukunonu lies 64
kilometres north-west of Fakaofo, and Atafu lies 92 kilometres north-west of Nukunonu." Between the
population-weighted hex centres of OSM's atolls: Nukunonu from Fakaofo 68.1 km at 286.6 degrees
(1.06x), Atafu from Nukunonu 103.9 km at 314.9 degrees (1.13x). The printed distances are between
atolls and the hexes sit on the villages, so the band is 0.80-1.25x and north-west is asserted.

**Placement: Kontur 2023 hexes.** 17 populated hexes, 1,837 people: Atafu 642 on 6 hexes (1.55x the
413 present), Fakaofo 662 on 5 (1.66x), Nukunonu 533 on 6 (1.38x); against the de jure count on the
atolls 1.27x. Kontur models a later, larger population (the scout read 1,647 usual residents in the
2019 count). The grid is under spec §8.2e's floor (median 6 hexes per unit) and kept: 7 of the 17
hexes hold 1 to 4 people and the other 10 are the four villages (one each on Atafu and Nukunonu, two
on Fakaofo, printed p.10), so equal shares would put a fifth to a half of each atoll's dots on
near-empty hexes. Registered `accepted` in `sources/geo_checks.csv`. No Kontur cap blocks
(`kontur_cap.py tk`).

**Natural Earth's New Zealand feature in the countries file reaches Tokelau** (bounds to 8.54 S).
`FROM_UNITS` appends `tk` after it, so the viewer's "which country is the camera over" will name `nz`
over the atolls, the same shape as `bq` inside `nl`, which Anita left as is on 2026-09-15 (playbook
`geography.md`). Not fixed here and not asked; it is one line in the report.

## 6. Mapping (`taxonomy/tk2016.py`)

| category | 2016 | node |
|---|---:|---|
| Congregational Christian | 603 (50.38%) | `christianity.reformed.congregational.tokelau`, new |
| Roman Catholic | 463 (38.68%) | `christianity.catholic.latin` |
| Presbyterian | 71 (5.93%) | `christianity.reformed.presbyterian` |
| Other Christian | 50 (4.18%) | `christianity` |
| No religion | 1 | `unaffiliated` |
| Not stated | 9 (0.75%) | EXCLUDED, in `gap` |
| Spiritualism and New Age religions | 0 | no row emitted |

**The Congregational Christian Church of Tokelau gets its own node.** `.ekt`, the scout's warning, is
Tuvalu's church. Wikipedia's article on the Congregational Christian Church of Samoa (read
2026-09-15) says the LMS left its Tokelau outposts to the CCCS and that the Tokelau district became
independent in 1996 as the Congregational Christian Church of Tokelau, *Ekalehia Fakapotopotoga
Kelihiano Tokelau* (its note 13); Wikipedia's *Religion in Tokelau* spells it *Ekalehia
Fakalapotopotoga Kelihiano Tokelau* with no citation. A search snippet from an International Bible
Reading Association partner page says the Tokelau church split from the Samoan one in 1997; the page
now answers 404. The Pacific Conference of Churches member list answered 403. None of this was
checked with the church, so the node description says "the mid-1990s" and gives Wikipedia's year.
Filed beside `.cccs` as the eighth of the Pacific Congregational set, following `.cccas` (American
Samoa, the same day). The census label names no church; the form offered it as one of three, and a
CCCS member living on the atolls would tick it too.

**Presbyterian: 54 on Atafu in 2016 against 5 in 2011.** Which body, and why it rose, is in none of
the documents read. Drawn as counted.

**Other Christian goes on `christianity`**, as Gibraltar's did: the write-ins beside three named
churches, a residual of the form. 24 of the 50 are on Nukunonu, which counted none in 2011.

## 7. §14

Considered, not escalated. The units are the atolls themselves, 385 to 413 people, and the only
non-Christian answer is one person with no religion on Atafu, which the office's own published
table prints and which draws at most a ring. No search for restrictions on religious minorities in
Tokelau was made.

## 8. Not opened or not checked

- The 2022 census form and microdata (403 everywhere, §1).
- The three 2016 atoll profiles; the 2006 census tables and report; the 2011 census report.
- The church's own account of its independence (§6), and the Presbyterian body on Atafu.
- The village split on Fakaofo (Fale and Fenua Fala): the form codes both; no religion table by
  village was seen.

## Review, 2026-09-15 (cb8b206e-rev3)

Full pass. `check_md` clean, `built_countries --check` ok, `check_rollup tk` clean (1,188 measured).
Every `note_public` figure recomputes off `tk.csv`: Atafu 318 of 413 (77.0%), Fakaofo 250 of 399
(62.7%), Nukunonu 315 of 385 Catholic (81.8%); `gap_share` 311 of 1,499 (0.2075). **`.tokelau`
agreed:** it is the eighth of a set that already gives each Pacific national church its own leaf at
depth 4 (ask 009's reason such leaves are cheap), and neither `.cccs`, which it left, nor the bare
parent would be more accurate. Presbyterian, Other Christian on `christianity` as Gibraltar, and
Latin Catholic as American Samoa: agreed. Counting the 302 absentees in `gap_share` is recorded
with its reversal and left as built. §14: agreed, nothing. Screenshot: the country selects, its
header reads right and no dots draw, as §0 says; a retake with rings on was not got (the probe
could not find the control) and was skipped. Nothing to change.
