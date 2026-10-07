# Macau: 2021 census, usual language by statistical district

Session d9e44929-mo, 2026-10-05. Drawn: 663,005 people aged 3+ on 23 statistical districts
(zonas estatísticas), 659 dots, 7 languages, placed on 3,733 residential buildings by their
census population.

## Source

DSEC (Direcção dos Serviços de Estatística e Censos), 2021 Population Census, 16th census,
reference date August 2021, whole resident population 682,070. The question is **usual
language** (日常用語, Língua corrente), asked of everyone aged 3 and over; "Not applicable"
(18,288) is the under-3s. `how` = "census, 2021, usual language".

- **Language table**: the census's Population Statistics Database (https://www.dsec.gov.mo/CensosWebDB/)
  is an AngularJS page over an open JSON API, no key: `GET
  https://www.dsec.gov.mo/InReportApi/Censos2016/AllResidentMeta2021` lists the dimensions,
  `POST .../AllResidentData` with `{"dimensions": "zona,family_language", "filters": null,
  "year": 2021}` returns the cube (the "Censos2016" in the path is the app's name; `year` picks
  the census; 2011 and 2016 also answer). Found in the app's `scripts-e1f4d1f46c.js`
  (`apiService`). Dimensions available for residents include parish (7 + maritime), statistical
  district (23 + maritime), age, sex, nationality, ethnicity, education. Nothing finer than the
  district carries language.
- **Placement**: DSEC's Statistical Geographic Information System
  (https://www.dsec.gov.mo/gis/unidade/, "2021 Population Census" tab), an ArcGIS 10.0 server.
  Service `Production2021/Census-QueryZonaBuilding-2021`, layer 0 (district polygons) and layer
  1 (5,682 building polygons with the census's resident count per building). Responses saved in
  `data/raw/mo/` with their URLs.

The coverage sweep's lead (territory-wide figures only, "probably statistical zone or parish")
was right about the zone; the district table and the building layer were not in it.

Terms: DSEC's public database and GIS, no registration; cited as DSEC.

National totals (aged 3+): Cantonese 537,981 (81.0%), Other Chinese dialects 36,032, Mandarin
31,405, English 23,635, Tagalog 19,154, Others 11,626, Portuguese 3,949.

## Checks (`python sources/mo_census.py`, `python sources/mo_geo.py`), none a tolerance

- The district x language table sums, language by language, to the national usual-language
  table, whose total is the census's 682,070.
- District x language x nationality (1,344 cells), summed over nationality, equals district x
  language in all 192 cells; parish x language sums to the national table.
- **Second release, per unit**: the GIS's buildings, joined to the district polygons by their
  largest overlap, sum to the database's population for **all 23 districts exactly** (0 people
  off); their total is the census's land population, 681,293; the maritime feature holds 777,
  the database's maritime area. The buildings' geocode (first two digits) reproduces the seven
  parish totals exactly.
- District polygons: area equals the server's Shape.area; the GIS's density field times the
  polygon area is 0.988-1.020 of the census count (density is printed rounded).

## Geography (`python sources/mo_geo.py` -> `data/geo/mo/`)

- **Not in religiondots**; units and placement built here from DSEC's own GIS.
- **Use the query service, not the display services.** `Census2021_Density23_*` and
  `Census2021_Population_*` serve the same features generalised to a handful of vertices
  (district 7 as 5 points, 0.156 km2 against 0.212) and about 960 buildings stored in a broken
  second coordinate system; on those, the building join missed the census districts by 3.6%.
  `Census-QueryZonaBuilding-2021` has full geometry and the join is exact.
- **Projection**: the server labels the data wkid 3064 (IGM95 / UTM 32N) and its own reprojection
  to 4326 lands Macau at 4.7E 0.16N. The coordinates are the Macao Grid (EPSG:8433); reprojected
  locally with EPSG "Macao 1920 to WGS 84 (1)", about a 300 m shift. Bounds come out 113.529-113.598E,
  22.110-22.217N.
- **Placement**: the populated buildings, `pop` = census residents. 24 buildings straddle a
  district line and go to their larger part. Median 73 buildings per district (min 10, Pac On &
  Taipa Grande). Every language in a district is spread on the same building weights: nothing
  says where inside a district one language's speakers live.
- Not a Kontur layer, so no cap blocks and no grid-floor question. Water clip touched 16
  buildings (1% of their area).
- District 20 ("Universidade e Baía de Pac On", footnoted in the database) is drawn by the
  GIS in north Taipa only; no district polygon covers the University of Macau campus on
  Hengqin, and the building layer has one populated building near it (125 people, district
  23). Every district still reproduces its census count exactly, so whoever lives on the
  campus is counted, and placed, in buildings the GIS draws elsewhere. Not pursued.

## Labels (`taxonomy/mo2021.py`); no new nodes

Cantonese on `cantonese`, Mandarin on `mandarin`, Portuguese, English, Tagalog on their nodes.
"Other Chinese dialects" on `sinitic` (Chinese, language not named): the database names no
variety inside it in 2011, 2016 or 2021. "Others" on the root `other`.

## Not drawn (`gap`)

Children under 3, 18,288 (2.7%); 777 people on boats in the maritime area (733 Cantonese, 41
other Chinese, 3 Mandarin), which has no land polygon.

## Worth knowing (not drawn)

The nationality cut shows what the two remainders and the migrant groups hold: of 33,896
Filipino nationals, 18,718 named Tagalog and 14,209 English; of 12,217 Vietnamese, 5,199
Cantonese, 3,062 Mandarin and 3,755 "Others" (most likely Vietnamese); "Others" is 11,626 in
all, 3,755 Vietnamese, 3,453 other nationality, 2,617 Chinese, 1,330 Indonesian. Splitting
"Others" by nationality would change counts on a proxy, so it is not done (Anita's to allow).
Portuguese: 3,485 of 3,949 speakers are Portuguese nationals; 4,015 Portuguese nationals named
Cantonese.

## Calls someone might reverse

- District grain with building placement, over parish grain (7 units, same seven groups).
- "Other Chinese dialects" on `sinitic` rather than guessed into Hokkien.
- Building-population placement (the census's own count) rather than Kontur.

## Colours

All seven nodes already had colours from Hong Kong and elsewhere; not retuned.
