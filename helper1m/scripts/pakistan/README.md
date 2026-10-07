# Pakistan

Three levels: province/territory (7), district (156), tehsil (542). Years 2017 and 2023, both
from census counts, both on the 2023 units.

```
C:\Python39\python.exe helper1m\scripts\pakistan\fetch.py      # download, parse, boundaries, population.csv (a few minutes)
C:\Python39\python.exe helper1m\scripts\build_country.py pakistan
C:\Python39\python.exe helper1m\scripts\pakistan\check.py      # joins + Kontur comparison
C:\Python39\python.exe helper1m\scripts\pakistan\language.py   # mother-tongue pies (seconds)
```

## Mother-tongue pies (`language.py`)

Writes `countries/pakistan/composition.json` from data `maps/languagedots` already holds (read
only), with its label mapping (`taxonomy/pk2023.py`) and colours. Grouping, summing up and
writing are `scripts/language_common.py`, shared with Afghanistan; the palette is
`scripts/language_colors.csv`, hand-editable, appended to only.

- **Four provinces and Islamabad, at tehsil level.** 2023 census Table 11 (population by mother
  tongue: 14 named languages and Others), read by languagedots' own reader
  (`sources/pk_t11.py`) from the CRAN package PakPC2023. Table 11 lists the same 591 units as
  Table 1, so each goes through this map's `crosswalk.csv` to its adm3 unit: a join by census
  unit, not by polygon name. 577 match on folded district + unit name; 14 are in `ALIAS`
  (Table 1's PDF text drops "ll": AI = Allai, KAR KAHAR, KAG = Kallag, 18-HAZARI = Hazari; Quetta's
  units are written type-first; Tando Allahyar). Every tehsil-level unit in the five has a pie.
- **Table 11 leaves out the 1,041,342 counted by head only.** Table 1 minus Table 11 per unit
  shows where they were: 248 units, adding to exactly that figure, led by Quetta city 221,980,
  Islamabad 80,619, Rojhan 55,066, Rawalpindi 54,815, Kuchlak 47,091 and Palas (Kohistan)
  40,752, which loses 30% of its people that way. So a pie's total can be below the unit's
  population; the pie is sized by the population. Kandhkot taluka is one person larger in
  Table 11 than in Table 1.
- **Checked:** every unit's languages add to its TOTAL row; summed to districts, Table 11 equals
  languagedots' `pk.csv` in all 1,777 cells; each unit's groups add to its total, and every level
  holds every person (240,458,089; 246,500,586 with the north switched on).
- **Gilgit-Baltistan and Azad Kashmir draw no pie** (Anita, 2026-10-06): they are not in
  Table 11. `MODELLED_NORTH = True` in `language.py` brings back languagedots' models of them,
  GB per district (also its tehsil-level unit) and AJK per district only. Those models are: GB's 2023 census mother-tongue shares for the region as a whole, split by district
  with the GB MICS surveys (2016-17 district table, 2024-25 for Burushaski, Khowar and Wakhi),
  and the AJK yearbook's whole-percent district estimates (Statistical Year Book 2025, Table
  15.31) on the 2023 census population. The groups that come only from them say so on hover.
- **Groups (18):** a language gets its own colour at 0.1% of the country or 30% of some unit
  with 10,000 speakers there. Punjabi, Pashto, Sindhi, Saraiki, Urdu, Balochi, Hindko,
  Pahari-Pothwari (AJK), Brahui, Mewati, Kohistani, Shina, Gojri (AJK), Balti, Kashmiri,
  Burushaski, Khowar, and Other languages: Table 11's Others (3.4 million; Khowar, Gujari,
  Burushaski and Persian among them, not separable), Dogri (Bhimber's 30%, which rounds to just
  under the bar), Wakhi, Kalasha, Kundal Shahi.

## Sources

**PBS 2023 Digital Census, Table 1** — "Area, population by sex, sex ratio, population density,
urban population, household size and annual growth rate", one PDF per province, under
`https://www.pbs.gov.pk/wp-content/uploads/census_tables/tables/`:
`table_1_{kp,punjab,sindh,balochistan}_districts.pdf` and `table_1_islamabad.pdf`
(Last-Modified 2025-01-22; needs a browser User-Agent). Each runs province, district, then the
district's tehsils / talukas / sub-divisions / sub-tehsils, each with rural and urban rows.
Column 3 is the 2023 count, **column 11 is the 2017 count re-tabulated on the 2023 units**,
column 12 the 2017–23 growth rate. That makes it the pair helper1m wants: two census counts on
one boundary set at the finest level PBS prints.

Table 1 is used rather than Table 9 (religion, which religiondots parsed) because Table 1
includes the 1,041,342 people in restricted areas counted by head only. Its footnote says so;
Table 9 and everything from Table 4 on leave them out. Table 1's provinces are PBS's headline
totals.

**Azad Kashmir** — AJ&K Statistical Year Book 2025, Table 15.15 (PDF page 220),
`https://www.pndajk.gov.pk/uploadfiles/downloads/Statistical%20Year%20Book%202025.pdf`: 2023
census and 2017 census by tehsil, the 2017 figures re-tabulated on the 2023 tehsils (Mirpur and
Dudyal differ from the 2024 book's split). Cited as "Population & Housing Census Report 2023,
PBS". District totals checked against AJK At a Glance 2025, p.3.

**Gilgit-Baltistan** — GB at a Glance 2025, Statistical & Research Cell, P&DD GB, p.3,
`https://pnd.gog.pk/storage/downloads/AiRIlDEcscWPC1s58oXIgpjlVAS7jd-metaR0IgQVQgR2xhbmNlIDIwMjUuMS5wZGY=-.pdf`:
2017 and 2023 census by district (10 districts). No tehsil figures are published.

Both are transcribed in `ajk_gb.py` from the PDFs' text layer, with sum checks.

**Boundaries.** OpenStreetMap admin_level=7 relations (tehsils), fetched per province from
Overpass by `osm.py` (one all-Pakistan `out geom` query times out). OCHA COD-AB v01
(`data/asia1m/pakistan/pak_admin{1,2,3}.shp`, valid_on 2022-09-09) for the holes in OSM, for the
outline, and for Azad Kashmir and Gilgit-Baltistan. religiondots' census district polygons
(`religiondots/data/geo/pk2023/pk_districts.gpkg`) are read only to say which census district
each OSM tehsil sits in.

## How each level is made

**Tehsil, four provinces and Islamabad (500 units from 591 census units).** OSM's tehsils were
drawn in 2023 and carry most of the recent splits COD lacks (Lahore's five tehsils, Peshawar's
seven, Mohmand's seven, Tando Allahyar's three, Chaman City / Saddar...). `crosswalk.py` pairs
census units with OSM tehsils inside each district:

- 422 census units pair one-to-one with an OSM tehsil by folded name (difflib >= 0.75).
- `GROUPS` lists every hand pairing with its reason: renames (Wazir = Gumatti, AI = Allai,
  Mandanr = Chamla, Golarchi = Shaheed Fazil Rahu...), OSM tehsils newer than the census merged
  back into their parent (Tirah into Bara, Kalam into Behrain, Ali Pur Chatha into Wazirabad,
  Jalalpur Jattan into Gujrat, Multan Khurd into Talagang, Thana Baizai into Swat Ranizai...), and
  census sub-tehsils OSM lacks folded into the OSM tehsil that holds them. For the last kind the
  evidence is an OSM place node of the same name inside one tehsil (Yakmach, Khost, Hoshab,
  Greshak, Bostan, Kuchlak, Sambaza) or, failing that, the census areas in Table 1 leaving room
  in exactly one tehsil. Where neither works the census units are merged with every OSM tehsil
  they could be in (Upper Dir's three sub-divisions, Awaran, Musakhel, Panjgur).
- `WHOLE_DISTRICT`: 14 districts are one unit at tehsil level. Karachi's seven — its 31
  sub-divisions share names with OSM's 2022 local-government towns but not their lines
  (Baldia is 949k in the census and 215k in Kontur on OSM's Baldia Town), and COD has the 2001
  towns. Plus Barkhan, Dera Bugti, Duki, Gwadar, Nushki, Sohbatpur, Surab, where OSM has one
  or two tehsils against several census units.
- COD fills where OSM has nothing: Lakki Marwat (OSM has only Bettani; COD's Lakki Marwat holds
  the census's Lakki Marwat + Ghazni Khel), Islamabad. COD's finer line cuts an OSM tehsil in two
  places: Kot Chhutta out of Dera Ghazi Khan, Nowshera out of Khushab.
- Leftover pieces inside COD's outline of the five units (2,433 km2, all border slivers; the
  largest 320 km2 on the Iran border) go to the unit they touch most. Overlap between units:
  1.2 km2.

So 42 units hold more than one census unit (133 census units between them), and no census unit
is split.

**Tehsil, Azad Kashmir (32).** COD's 32 tehsils pair one-to-one with the yearbook's 32
(Karnah = COD Leepa, Darliah Jattan = Dulliya Jattan, Hari Gail = Harighel, Mang = Mong,
Dudyal = Dadyal). OSM has a newer Chakswari tehsil the census does not, so COD fits better here.

**Gilgit-Baltistan (10).** COD v01 already has 14 districts; the census counts the older ten,
so Darel and Tangir go back into Diamer, Gupis-Yasin into Ghizer, Rondu into Skardu. The
district polygons are also the tehsil-level units.

**District and province** are dissolved from the tehsil level, so the three levels nest and
every district and province figure is an exact sum.

## Checks (results of the 2026-10-02 build)

- Every Table 1 row: all = male + female + transgender; all = rural + urban for 2023 and 2017.
  Tehsils sum to districts and districts to provinces, both years, every district.
- Provinces equal PBS's headline totals: Punjab 127,688,922, Sindh 55,696,147, KP 40,856,097,
  Balochistan 14,894,402, Islamabad 2,363,863; national 241,499,431. 2017: 207,684,626, the 2017
  census total.
- 2017 column cross-check: the growth rate Table 1 prints in column 12 recomputed from columns 3
  and 11 agrees within 0.075 points on every unit, so the 2017 column is read right.
- AJK tehsils sum to 4,333,467 (2023) and 4,032,363 (2017) and to each district's 2023 total in
  At a Glance; GB districts to 1,709,049 and 1,492,924.
- All 136 districts at level 2 equal Table 1's own district rows.
- Joins: 542 tehsil-level, 156 district and 7 province polygons, every one with both years; no
  population row without a polygon.
- Kontur 2023-11 summed on hex centroids (`check.py`): national ratio 0.974, but Balochistan
  0.59, Gilgit-Baltistan 0.67, Punjab 1.04. Districts within 10% of their province's median
  ratio 60%, within 25% 86%. Tehsils within 10% 39%, within 25% 70%. Raw within 10%: 34% of
  tehsils. Kontur is a weak witness here: it runs low in every dense city core and high in the
  suburbs around it (Lahore City 0.64, Shalimar 0.65 against Raiwind 1.95; Gujranwala City 0.64
  against Gujranwala Saddar 1.71) although the census and OSM areas for those tehsils agree to a
  few percent, and it is low across all of Balochistan, as religiondots found. The census
  figures themselves are exact sums of published counts; the only error the map can carry is a
  boundary that does not match the census unit, and the Kontur test cannot separate that from
  its own urban bias.

## Known weaknesses

- **Karachi is seven units** of 2.3 to 3.9 million at tehsil level, and Quetta City + Sariab is
  1.9 million. Lahore City (4.1M), Faisalabad City (3.7M) and Rawalpindi (3.7M) are single census
  tehsils.
- **Fast growth between the two counts extrapolates.** 43 tehsil units grew over 5% a year
  2017–23, mostly ex-FATA (Razmak 21%), Kohistan (Bankand Ranolia 24%, while neighbouring Palas
  fell 4.5% a year — a reallocation between the two counts more than real change) and western
  Balochistan. Peshawar Tehsil shows a fall (2.18M to 2.11M) while the new tehsils around it
  doubled, which looks like the 2017 re-tabulation onto the 2021 tehsils. The viewer's straight
  line carries these on to 2026.
- **Dera Ghazi Khan / Kot Chhutta** are cut on COD's line inside OSM's single tehsil; Kontur puts
  1.21M in Kot Chhutta against 0.90M census and 0.88M in Dera Ghazi Khan against 1.44M, so COD's
  line may run too close to the city.
- **Kharan / Washuk**: the census's Kharan includes Tohmulk, which COD files under Washuk; OSM's
  Tohmulk polygon is used, but the census Kharan is 14,958 km2 and ours is about 10,700, so some
  Washuk land probably belongs to Kharan. Population is unaffected; the line is approximate.
- **Placements by area** (Kachhi, Kalat, Kech's Zamoran, Kharan's Patkain, Khuzdar, Killa
  Saifullah, Kohlu, Nasirabad, Pishin's Nana Sahib, Zhob, North Waziristan's Shawal, Torghar's
  Daur Mera, Lower Dir) rest on Table 1's areas, which are loose (Abbottabad Tehsil is 1,285 km2
  in Table 1 and 1,029 in OSM). If one is wrong, a sub-tehsil of 5k–70k people sits in the
  neighbouring tehsil.
- **GB has no tehsil level** and its districts are drawn as COD depicts them; GB at a Glance's
  areas differ (Shigar 4,173 km2 there against 8,532 in COD).
- **Restricted areas**: the 1.04 million counted by head only are inside the tehsil figures but
  PBS does not say which tehsils; they are wherever Table 1 put them.

## Tried and not used

- **COD-AB tehsils alone**: 125 Sindh, 139 Punjab, 153 KP, 103 Balochistan units against the
  census's 138, 146, 148, 158. COD still has Lahore as two tehsils (census five), Peshawar as
  four 2001 towns (census seven), Karachi as 2001 towns. OSM's tehsils matched the census
  one-to-one by name in 72 of 136 districts before any hand pairing.
- **OSM Karachi towns**: name-matched to sub-divisions, Kontur ratios ran 0.16 to 16 within one
  district (Orangi 0.16, Saddar 16), so the lines are not the census's.
- **PBS GIS page** (`pbs.gov.pk/gis/`): lists administrative-unit PDFs, no shapefiles.
- **GB at a Glance 2024** on `portal.pnd.gog.pk`: the host no longer resolves; the Wayback copy
  is cut off at exactly 5 MiB and unreadable. The 2025 edition on `pnd.gog.pk` replaced it.
- **USCB's 2017 workbook** (religiondots) has AJK and GB, but its GB tehsils are estimates spread
  from 1998 shares and its AJK total (4,045,367) is not the census's (4,032,363).

## Files

- `fetch.py` — driver: downloads, parses Table 1, runs `prep_boundaries.py`, writes `population.csv`.
- `table1.py` — Table 1 PDF reader (geometry-based; numbers right-aligned, column by right edge
  against the header's vertical rules; a long name squeezed onto two lines is rejoined).
- `osm.py` — Overpass fetch and polygon assembly. `kontur.py` — Kontur centroids for checks.
- `crosswalk.py` — every hand decision, with its reason. `ajk_gb.py` — AJK and GB figures.
- `prep_boundaries.py` — writes `data/pakistan/boundaries/adm{1,2,3}.gpkg` and `crosswalk.csv`.
- `check.py` — join and Kontur checks, writes `data/pakistan/check_kontur_adm{1,2,3}.csv`.
- `data/pakistan/raw/osm_pk_places.json` — OSM place nodes, fetched once by hand to locate
  sub-tehsil headquarters; not needed for a rebuild.

Gotchas in Table 1: a data row can hold a bold "1" at header height, so the header row is the
one y carrying all of 1..12 (the first version dropped MERYAN TEHSIL that way); Malakand's
district header is "MALAKAND DISTRICT" here but "MALAKAND PROTECTED AREA" in Table 9; Tando
Allahyar is spelt TANDO AHYAR; Table 9's KALLAG is Table 1's KAG; the last page of each file has
a footnote that must be cut off before rows are read.
