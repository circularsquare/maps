# Gibraltar — Census of Gibraltar 2022, religion by major residential area

**Drawn 2026-09-15** (session `d743fc47-gi`). One unit, the territory; 8 categories; 37,936
people (the whole usually-resident population); every row `measured`. 34 dots and 2 rings (other
or not stated, Hindu) at 1:1,000; 2 dots and 7 rings at 1:10,000. The placement hexes are cut to
Gibraltar's land, so no dot can land in Spain or the sea (§5a, since the re-scatter of 2026-09-15;
the first build placed on whole hexes and one dot fell in La Línea).

- `sources/gi.py` -> `data/normalized/gi.csv` (the report in `data/raw/gi/`, pinned) and
  `data/geo/gi/gi_hexes.gpkg` (Kontur GI 2023-11-01, raw extract in `data/raw/micro/` as the
  microstate tier keeps it)
- `taxonomy/gi2022.py` -> the mapping; `countries/gi.py` -> the entry; `taxonomy/branches.py`
  `other.gi` is the one new node
- sources.md **§gi-2026-09-15** is the summary; **§scout-2026-09-15-europe** was the scout's.

```
python sources/gi.py --fetch
```

## 1. What HM Government of Gibraltar publishes

| release | religion | tier |
|---|---|---|
| **Census of Gibraltar 2022, report (528 pp., published 2025)** | **Table 42, major residential area x religion x sex, persons, p.174**; Table 43, enumeration area x religion x sex, pp.175-177; Tables 37-41 religion by age, nationality, birthplace, year of arrival, length of residence (territory); 48, 54, 201, 240, 251 (partnership, language, householder, gender identity, orientation) | **7 areas + Institutions; 78 EAs** |
| the same report, Figure 8's table, p.52 | religion 1970, 1981, 1991, 2001, 2012, 2022, eight answers | territory |
| Census of Gibraltar 2012 (the scout read it; not fetched here) | Tables 1.16c (EA) and 1.16cc (area), printed pp.52-53, with other and not stated apart | 7 areas, EAs |
| UNSD Demographic Yearbook table 28 | 2012 (9 categories) and 2001 (8), exact partitions | territory |

Appendix 9 (p.523) defines each area as a list of EAs, and Appendix 8 (pp.517-522) each EA as a
list of streets and housing estates. **No map of either is in the report**, and none was found:
citypopulation.de draws EA and area maps citing the Statistics Office for the figures, with no
source or licence stated for the boundaries; Wikipedia's article lists the areas with no
boundary source. The Statistics Office's own site and any Gibraltar government GIS were not
searched for polygons.

## 2. The file

| file | URL | digest | bytes |
|---|---|---|---|
| `census_of_gibraltar_2022_report.pdf` | `gibraltar.gov.gi/uploads/statistics/2025/Census/Census%20of%20Gibraltar%202022%20-%20Report.pdf` | `MMRHWPMMCBEKBISSO2X4V2SH7QRZWKSR` | 18,264,524 (528 pages, `%%EOF`) |

The copy laid before Parliament, `parliament.gi/uploads/contents/papers_laid/2025/census_of_gibraltar_report_2022.pdf`,
answered with the same Content-Length on 2026-09-15 and is `fetch()`'s fallback; its bytes were
not compared. The page text layer holds every table; nothing had to be rendered or transcribed
by eye beyond checking.

## 3. The checks (`sources/gi.py::check`)

| check | result |
|---|---|
| file pinned | digest above, 528 pages |
| Table 42 parsed off p.174 = transcription | 9 rows x 9 columns x persons, males, females |
| persons = males + females; answers sum to row totals; rows sum to the total row | every cell |
| p.44: 37,936 is the usually-resident population | yes; families of UK servicemen (260) and visitors (1,240) are persons present only, UK servicemen outside the census |
| p.52 series, 2022 column = Table 42's total row | yes |
| p.52 series, 2012 and 2001 columns = UNSD table 28 | to the person, with UNSD's Other and Not Stated merged (2012: 365 + 44 = 409) |
| Table 43's total row = Table 42's; its 78 EA rows close and sum to it | all 19 columns |
| Appendix 9's lists as printed | yes |
| Eastside = EA 1; North District = EAs 2-9, 12 | exact, all 19 columns |
| **Institutions = EAs 80-84, 86, 87, and EA 85 is in South District** | exact; see below |
| EA 11 splits Town Area 201 / Upper Town 56 | inside EA 11 in every column |
| EA 63 gives Sandpits 313; EA 64 gives the Reclamation Areas 4,081; South District closes on the rest | every cell inside its EA except **one pinned cell, 4 people** |
| form question 11 | eight boxes, no not-stated box, no write-in |

**EA 85 is tabulated in South District.** Appendix 8 names EAs 80-87 only as a group (religious
institutions, hotels, hostels, hospitals, prison, marinas, old people's homes, other
institutions) and Appendix 9 does not list them. Institutions in Table 42 is short of EAs 80-87
by exactly EA 85's row (70 people: 43 men, 27 women; 18 Roman Catholic, 19 Church of England, 26
no religion) in all 19 columns, and South District is over by the same row. EA 11, 63 and 64's
splits are each fixed by the one other area sharing them, so South District is the only area
that can take it. `check()` finds the EA from the shortfall and asserts it is 85.

**One cell is 4 people apart.** The Reclamation Areas are EAs 65-70 plus part of 64, so their
share of EA 64 is fixed in every column. For Church of England males it comes to 168 (452 in
Table 42 less 284 in EAs 65-70) against EA 64's 164. Two readings fit: four such men are
tabulated from somewhere outside the listed EAs (EA 85, with 12, is the only unlisted piece that
large), or the two tables differ by 4 in that cell. The report does not settle it. Pinned in
`check()`, so any other disagreement fails the build. Nothing drawn depends on the area table.

## 4. The questionnaire, and `Other/Not stated`

Individual question 11, *Religion?* (p.480): Roman Catholic, Church of England, Other Christian,
Muslim, Jewish, Hindu, Other, No religion. **No write-in and no not-stated box**, so a person who
did not state a religion left it blank. The 2022 tables print `Other/Not stated` as one column;
the 1970-2022 series uses the merged label for every year. The 2012 report printed them apart
(365 other, 44 not stated), so the merged pair was 89% other then.

Drawn on a new node, `other.gi`, not `unknown` and not `gap`, following the shape of Anita's
ruling on ask 023 (Iran's merged 1390 column stays on a coloured node). No `gap` field: nobody in
the table is left undrawn. `tools/gap_share.py` has nothing to compute.

Where it leans: **50 of the 779 are in Institutions, 9.1% of its 552 residents**, against 1.95% of
everyone else. Institutions were enumerated by a senior census officer with the managers
"providing the information requested" (p.37); that the blanks come from there is an inference.

Mapping: Roman Catholic -> `christianity.catholic.latin`; Church of England ->
`christianity.anglican`; **Other Christian -> `christianity`** (the parent: the form names two
churches, so this is a residual holding whole branches, and `christianity.other`'s note reserves
that node for bodies with no branch; Belgium's precedent, not Barbados's); Muslim -> `islam`;
Jewish -> `judaism`; Hindu -> `hinduism`; No Religion -> `unaffiliated`.

## 5. Why one unit, when the table has seven areas

This is the call most worth reversing if someone wants to, so the reasoning in full.

1. **No polygons exist for the areas.** They are lists of EAs, the EAs are lists of streets, and
   three EAs (11, 63, 64) are listed under two areas, split by an unpublished line. Building them
   means tracing street lists onto OpenStreetMap with no published outline to check against.
2. **The placement layer cannot separate them.** Kontur covers Gibraltar with 20 hexes of
   0.78 km2; 14 hold 126 people or more. Two hold 7,119 and 6,875, and a hex that size spans Town
   Area, Upper Town and the reclamation estates west of Main Street. Drawing the areas would need
   a finer layer as well (building footprints, or an EA-level population surface).
3. **At 38 dots the microstate ruling applies** (Anita, 2026-09-08: a national table is complete
   for a country of tens of dots). RULINGS records leaving a published district table undrawn as
   not decided, which is why this is spelled out.

What drawing the areas would show, from Table 42 (shares of each row, %):

| area | people | RC | CofE | other Chr. | Muslim | Jewish | Hindu | none | other/n.s. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Eastside | 497 | 71.63 | 11.07 | 2.62 | 2.82 | 0.80 | 0.40 | 10.46 | 0.20 |
| North District | 6,557 | 58.35 | 6.65 | 4.36 | 6.86 | 1.30 | 1.51 | 18.30 | 2.67 |
| Reclamation Areas | 14,965 | 69.13 | 6.05 | 3.88 | 2.26 | 2.21 | 2.54 | 12.11 | 1.83 |
| Town Area | 3,783 | 37.85 | 7.64 | 3.75 | **17.39** | **12.95** | 3.09 | 15.36 | 1.96 |
| Upper Town | 2,860 | 61.29 | 7.20 | 3.85 | 8.18 | 0.84 | 0.17 | 16.29 | 2.17 |
| Sandpits Area | 2,097 | 73.06 | 5.44 | 3.72 | 2.05 | 2.38 | 2.05 | 9.97 | 1.34 |
| South District | 6,625 | 68.09 | 7.85 | 4.15 | 1.75 | 1.21 | 0.65 | 14.57 | 1.74 |
| Institutions | 552 | 62.14 | 2.72 | 3.44 | 10.14 | 1.27 | 0.72 | 10.51 | 9.06 |
| **Total** | 37,936 | 63.52 | 6.70 | 3.96 | 5.03 | 2.82 | 1.83 | 14.08 | 2.05 |

Town Area holds 490 of the 1,070 Jews (45.8%) and 658 of the 1,909 Muslims (34.5%); the
Reclamation Areas 380 of the 693 Hindus (54.8%) and 1,812 of the 5,343 with no religion. At 1:1,000
Town Area is under four dots, so the area grain would move one or two Muslim and Jewish dots.
`note_public` states the Town Area figures in words instead.

**To reverse:** `sources/gi.py` already carries Table 42 checked against Table 43; `emit()` would
add area rows, and the work is a units layer (the seven areas traced from Appendix 8, EA 85 in
South District, Institutions spread over the areas or placed at named institutions) and a
placement layer finer than Kontur.

## 5a. The hexes are cut to Gibraltar's land (2026-09-15, session `d743fc47-fixes`)

The first build placed dots on the 20 whole Kontur hexes, and the scatter's water clip took off
only the sea. Along the isthmus the hexes run into La Línea, so the Roman Catholic dot the review
found at -5.3504, 36.1585 (§11) was in Spain. It sat in hex GI:8, 1,220 people, of whose land only
13.7% is Gibraltar's.

**The outline is OpenStreetMap relation 1278736** (Gibraltar, `admin_level=2`, ODbL), fetched
in full from the OSM API into `data/raw/gi/osm_relation_1278736.json`. Overpass returned a 504
and two mirrors timed out on the same day. The relation takes in the territorial sea. The OSM
water polygons, the layer `water.py` clips with, take it off again, which leaves 6.59 km2 of land
with its north edge at 36.15492 N on 5.35 W, the frontier fence. Two other outlines were
measured and not used. Natural Earth's 10m Gibraltar is 7 vertices and 3.69 km2, with 20 of the
first build's 34 dots outside it. geoBoundaries' gbOpen GIB ADM0, a Sentinel-2 land-cover trace
(CC BY 4.0), is 6.23 km2 of land and stops at 36.1527 N on 5.35 W, about 250 m short of the fence.
It would itself have put a dot and two rings outside.

**Each hex keeps the share of its people that its land inside Gibraltar is of all its land**,
Spanish land included and the sea excluded, as `mt_geo.py` now shares Malta's coastal hexes.
Two hexes are cut: GI:8 keeps 167 of 1,220 (13.7%) and GI:17 keeps 1,203 of 2,750 (43.8%). GI:9,
23 people, has no land in Gibraltar and is dropped. That leaves 19 pieces holding 32,285 of
Kontur's 34,908 people, 0.85 of the census where the whole hexes were 0.92. The weight only places
the territory's 34 dots, so this moves no count.

`geometry()` stops if the relation's tags change, if its outer ways do not close into one ring,
if the land is outside 6.3 to 7.0 km2, if the north edge on 5.35 W leaves 36.153 to 36.157, or if
any piece reaches outside the land. It writes the land as `data/geo/gi/gi_land.gpkg`, and
`python sources/gi.py --check-dots` asserts every dot and ring of both editions is on it, to 7.2 m.
That is how far the scatter's 4-decimal coordinates can round.

## 6. What the table shows

Roman Catholic 63.52% (72.1% in 2012), no religion 14.08% (7.1% in 2012, 2.9% in 2001), Church of
England 6.70%, Muslim 5.03%, other Christian 3.96%, Jewish 2.82%, other or not stated 2.05%,
Hindu 1.83%. The 1970-2022 series: Muslims 1,989 in 1970, 1,102 in 2001, 1,909 in 2022; Jews
552, 584, 1,070.

## 7. §14 was considered and no ask was filed

Drawn at one unit, so the map places no group anywhere a population layer would not; the Town
Area concentrations are stated in the note as the office's own published figures. The table is
the territory's government's publication. No search for restrictions on or attacks against
religious minorities in Gibraltar was made for this build.

## 8. Gotchas

- **Table 43's footnote, "Enumeration Areas - see Appendix 8", has a bare dash** that reads as a
  zero cell; `read_t43` stops at `Note:`.
- **Appendix 9 does not place EA 85**, and Table 42 counts it in South District (§3).
- **The series table on p.52 splits 2001's total as `27,49 5`** in the text layer; only the
  answers are read for 2001.
- **Printed page = PDF page index + 1** throughout this report.
- **2022 merges other with not stated**; 2012 did not (§4).

## 9. Reopen when

- Polygons for the EAs or areas appear (the Statistics Office, the Land Property Services or Town
  Planning GIS; none searched), together with a placement layer finer than Kontur (§5).
- The 2012 report's area table is wanted as a second vintage (Tables 1.16c and 1.16cc, other and
  not stated apart).

## 10. Terms

The report is a public document of HM Government of Gibraltar, also laid before Parliament; no
licence text was seen and none was looked for. Kontur Population is CC BY 4.0.

## 11. Review, 2026-09-15 (session `d743fc47-rev8`)

Light pass. `check_md.py` clean, `built_countries.py --check` ok, `check_rollup.py gi` clean
(37,936 measured, nothing derived).

- **Figures.** Every `note_public` figure recomputes off `gi.csv` and §5's table: 63.5, 14.1,
  6.7, 5.0, 4.0, 2.8 and 1.8%; 779 people at 2.1%; Town Area 490 and 658 of 3,783 (13.0% and
  17.4%); the Reclamation Areas 380 of 693 Hindus.
- **Mapping.** `Other Christian` on the parent: agreed. Across the mapping modules that label is
  filed about half on `christianity` and half on `christianity.other` (`ca` and `hu` use both).
  Gibraltar's form, two named churches and then one box, is the shape of the parent side (`be`,
  `fr`, `uk`, `lk`), not of `bb`'s long named list. `other.gi` with no `gap`, on the shape of ask
  023's ruling for Iran: agreed.
- **One unit rather than seven areas**: the builder's call, reasoned in §5, agreed.
- **Screenshot: one dot looks off, worth a human eye.** A Roman Catholic dot at -5.3504, 36.1585
  (`dots_gi.geojson`) draws north of the border line, in what reads as La Línea de la
  Concepción in Spain; the frontier runs just north of the runway. Not diagnosed, nothing
  rebuilt. The other 33 land on Gibraltar. **Diagnosed and fixed the same day** (session
  `d743fc47-fixes`): the dot was in a Kontur hex that straddles the frontier, and the hexes are
  now cut to Gibraltar's land (§5a). Both editions were re-scattered, and `--check-dots` puts every
  dot and ring on that land. The build tail was not run. Hindu (693) and `other.gi` (779) get no dot at
  1:1,000 and appear only as rings, which are off by default, so that view's legend lists six
  rows; that is the carry rule working, not a fault.
