# Montenegro: MONSTAT, Popis 2023, mother tongue

Drawn 2026-10-05 (session edd42a8c-me). 612,232 of 623,633 people in 24 nodes on the 25
municipalities of the 2023 census, placed inside each municipality by the census's own mother
tongue of each of its 1,462 settlements. 603 dots at 1:1000, 8 rings.

Files: `sources/me_census.py` (fetch + normalise + checks), `sources/me_geo.py` (municipality
polygons, settlement points, placement layer), `taxonomy/me2023.py`, `taxonomy/tree.d/me.txt`,
`countries/me.py`. Data: `data/raw/me/` (five xlsx, `osm_places.csv`, `wikidata_places.csv`),
`data/normalized/me.csv`, `data/geo/me/me_opstine.gpkg` (OSM polygons), `data/geo/me/me_hexes.gpkg`,
`data/geo/me/me_settlements.csv`. Read only from religiondots: Kontur ME, Natural Earth 10m.

## 1. The tables

The coverage sweep's lead was right: the national open-data portal has the 2023 census by
municipality as four xlsx, dataset
https://data.gov.me/en/dataset/stanovnistvo-crne-gore-prema-nacionalnoj-vjeri-maternjem-jeziku-i-jeziku-kojim-se-uobicajeno-govori
(Tabela 1 ethnicity, 2 religion, 3 mother tongue, 4 language usually spoken). MONSTAT's census page
`page.php?id=2342` (religiondots found it; the files sit under `uploads/files/popis 2021/`, the
census's first scheduled year) has the settlement workbooks, including mother tongue by settlement.

* **Drawn: Tabela 3, mother tongue by municipality.** 25 labels plus "Ne zeli da se izjasni"
  (does not wish to declare), for the country and 25 municipalities, count and percent columns.
  One answer per person.
* **Placement: `naselja maternji jezik popis 2023.(1).xlsx`**, the same 26 answers by 1,462
  settlements.
* Checks only: Tabela 4 (language usually spoken), Tabela 1 (ethnicity), `naselja popis 2023.xlsx`.

| answer | national | share | node |
|---|---|---|---|
| Srpski | 269,307 | 43.18% | serbian |
| Crnogorski | 215,299 | 34.52% | montenegrin |
| Bosanski | 43,470 | 6.97% | bosnian |
| Albanski | 32,725 | 5.25% | albanian |
| Ruski | 14,731 | 2.36% | russian |
| Srpsko-Hrvatski | 12,999 | 2.08% | serbocroatian |
| Ne zeli da se izjasni | 10,691 | 1.71% | gap |
| Romski | 4,658 | 0.75% | romani |
| Ostali jezici | 3,109 | 0.50% | other |
| Ukrajinski | 2,308 | 0.37% | ukrainian |
| Hrvatski | 2,193 | 0.35% | croatian |
| Bosnjacki | 2,030 | 0.33% | bosniak |
| Turski | 1,823 | 0.29% | turkish |
| Crnogorski-Srpski-Bosanski-Hrvatski | 1,721 | 0.28% | new leaf |
| Maternji | 1,408 | 0.23% | South Slavic (group) |
| Crnogorski-Srpski | 1,336 | 0.21% | new leaf |
| Srpski-Crnogorski | 1,210 | 0.19% | new leaf |
| Makedonski | 613 | 0.10% | macedonian |
| Engleski | 407 | | english |
| Njemacki | 292 | | german |
| Bjeloruski | 280 | | belarusian |
| Hrvatsko-Srpski | 233 | | croatoserbian |
| Goranski | 226 | | new leaf |
| Ostalo | 200 | | other |
| Bokeljski | 186 | | new leaf |
| Jugoslovenski | 178 | | yugoslav |

## 2. Checks, with numbers

`python sources/me_census.py`:
* national total 623,633, MONSTAT's published 2023 population; the national column has no
  suppressed cell and its 26 answers partition it exactly;
* the 25 municipal totals sum to the national total; published cells never exceed a total;
* **suppression**: municipal cells under 10 (and some others) are `z`. Per municipality the hidden
  people are total minus published: **716 people, 0.11%**, written as one `z (suppressed)` row per
  municipality and not drawn. Per answer it bites only the small ones: Jugoslovenski loses 19%,
  Hrvatsko-Srpski 16%, Goranski 15%, Njemacki 12%, Engleski 8%, Ostalo 69% (137 of 200); Bosnian
  0.06%, Albanian 0.03%, Serbian and Montenegrin nothing;
* Tabela 4 and Tabela 1 have the same total in all 25 municipalities;
* the settlement workbook has the same 25 municipality names and the same 26 answers, and its
  published cells never exceed the municipal cell. 219 settlements have their total suppressed;
  the other 1,243 hold 622,537 people (religiondots saw the same 219 in the religion workbook).

`python sources/me_geo.py`:
* **25 OSM admin_level=6 polygons** (Geofabrik extract, assembled with pyosmium) join the census's
  25 municipalities by name both ways, after stripping "Opstina", "Glavni grad", "Prijestolnica"
  and Ulcinj's Albanian half. 13,650 km² against a land area of 13,812;
* Kontur ME 10,095 hexes, 628,497 people; 284 hexes (18,041) fall outside the polygons, mostly on
  the coast and lake shores where OSM's municipalities stop at the water: 128 (15,964 people) in
  Natural Earth's Montenegro or the sea and over 2 km from a neighbour, snapped to the nearest unit
  within 1.5 km; 156 (2,077) near a border or in a neighbour, dropped;
* Kontur / census nationally 1.004; per unit normalised p10 0.80, median 1.06, p90 1.36; log
  correlation 0.976 against a best of 0.662 over 500 shuffles. Low: Petnjica 0.62, Budva 0.75;
  high: Plužine 1.81, Šavnik 1.92 (Kontur fills the empty mountain municipalities). These only move
  dots inside a unit.
* settlement points: 1,021 settlements matched to an OSM place node, 303 more to a Wikidata item
  (villages and settlements in Montenegro with coordinates, one SPARQL query, CC0), each by folded
  name inside the municipality buffered 1.5 km; 138 unplaced (57,478 people, 9.2%). The unplaced
  are mostly town quarters that neither source names: Kličevo 8,069, Dragova Luka 4,357,
  Straševina 2,202 (Nikšić), Rozino 4,631, Budva Centar 2,559, Dubovica 2,544 (Budva), Centar grada
  2,790 (Bijelo Polje). Budva is only 18% placed and Nikšić 69%; every other municipality 82% or
  more. Each hex goes to the nearest placed settlement of its own municipality, so an unplaced
  quarter is weighted by its neighbours' mix.
* spot check on the dots (share of the dot's own settlement speaking that language, median):
  Albanian 67%, Bosnian 73%, Serbian 45%, Montenegrin 42%. A quarter of Albanian dots sit in
  settlements under 10% Albanian; they are Podgorica's and Bar's town Albanians, whose
  settlement is the whole town.

## 3. Calls

* **Mother tongue, not language usually spoken.** The brief's order; Tabela 4 is the alternative,
  and nationally it differs little (Montenegrin 225,956, Serbian 271,422).
* **Four standards as four leaves**, as rs2022, hr2021 and ba2013. Which name people give follows
  nationality loosely: Tabela 1 has 256,436 Montenegrins and 205,370 Serbs, against 215,299
  Montenegrin and 269,307 Serbian speakers, so many Montenegrins by nationality name Serbian.
  `note_public` says the four are one language.
* **Every printed label a node.** Serbo-Croatian, Croato-Serbian, Bosniak and Yugoslav on their
  existing leaves (hr, ba, lu). New leaves: Montenegrin-Serbian, Serbian-Montenegrin (the two
  orders apart, as ba keeps its), Montenegrin-Serbian-Bosnian-Croatian, Bokelj (the Bay of Kotor's
  regional name; Kotor 64, Tivat 81, Herceg Novi 40) and Gorani (Glottolog gora1268, filed under
  Macedonian; a sibling of Macedonian so Macedonian stays a leaf; Podgorica 95, Bar 42, Berane 31,
  Rožaje 24).
* **Maternji ("mother tongue", 1,408) on the South Slavic group**, drawn as "language not named".
  It names no language. It is published only in Serbian- and Montenegrin-speaking municipalities
  (Podgorica 534, Nikšić 195, Herceg Novi 118, Kotor 103) and is absent from all six Albanian and
  Bosniak ones, so South Slavic is the narrowest node that safely holds it; not put into Serbian or
  Montenegrin, the choice the answer avoids. Someone could argue for `other`.
* **Ostali jezici (3,109) and Ostalo (200) both on `other`.** Never broken down; no indigenous
  remainder to keep apart.
* **Not drawn (gap)**: 10,691 who did not declare (6 of them inside a suppressed cell) and 710
  more in suppressed municipal cells, 11,401, 1.8%.
* **Own geography, not religiondots'.** Religiondots draws Montenegro on geoBoundaries' 23 polygons
  with Tuzi and Zeta folded into Podgorica. Tuzi is 60% Albanian, a quarter of Montenegro's
  Albanian speakers; averaging it into the capital is the one thing a language map cannot do.
  OSM has all 25 municipalities, so this map has Tuzi and Zeta as their own units. Religiondots
  could reuse `data/geo/me/me_opstine.gpkg` (its record lists this as its best upgrade).
* **Placement by the settlements' own mother tongue** (brief 4.4; counts unchanged, so no ask).
  Better than North Macedonia's ethnicity proxy: the settlement table asks the same question. Inside
  a settlement with a published total, its `z` people are split evenly over its `z` cells for
  placement; a settlement whose total is `z` takes its municipality's shares. Scatter: all 93
  (municipality, language) rows that drew a dot were placed this way, none on plain population.
* **Colour**: Gorani pinned to a darker olive (it was generated 0.044 from Montenegrin and 0.026
  from Ukrainian). That moved only this file's three compound leaves; Montenegrin-Serbian now sits
  0.044 from Montenegrin, left (a compound of it, 1,336 people scattered). The big pairs are far
  apart: Serbian teal, Montenegrin gold, Bosnian lime (0.102 from Montenegrin, ba.txt), Albanian
  brown, Russian green.

## 4. Not done

* The 2011 census (mother tongue by municipality, 21 units) was not looked at; the spec rules out
  a time slider.
* Settlement points for Budva's and Nikšić's town quarters: neither OSM nor Wikidata names them.
  Hand-placing about 20 quarters would lift Budva from 18% placed; it moves dots within a town only.

## 5. Cut from note_public (2026-10-06 text sweep)

* Russian and Ukrainian speakers are mostly recent arrivals on the coast.
* Settlement cells under 10 are withheld, so very small groups are placed less exactly.
