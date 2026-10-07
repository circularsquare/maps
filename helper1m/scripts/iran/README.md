# Iran

Three levels on the units of the 1395 (November 2016) census: 31 provinces (ostan), 429 counties
(shahrestan) and 1,049 districts (bakhsh; 1,057 census districts, eight pairs merged). Years
2011 (1390 census, carried onto the 2016 units), 2016 (1395 census) and 2024 (SCI's 1403
provincial estimate, spread over counties and districts by a model). The viewer's current-year
estimate is the line through 2016 and 2024.

Run order, from the repo root (all with `C:\Python39\python.exe`):

```
helper1m\scripts\iran\download.py      # SCI workbooks via Wayback, GEO1400, the 1403 estimate page
# Geofabrik iran-latest.osm.pbf (230 MB) into helper1m\data\iran\raw\osm\ by hand, then:
helper1m\scripts\iran\osm_admin.py     # -> data/iran/osm_admin.gpkg (admin polygons, levels 4-8)
helper1m\scripts\iran\osm_places.py    # -> data/iran/osm_places.gpkg (47,814 place nodes)
helper1m\scripts\iran\fetch.py         # provinces + counties (writes counties.csv the first time)
helper1m\scripts\iran\bakhsh.py        # district polygons -> data/iran/boundaries/adm{1,2,3}.gpkg
helper1m\scripts\iran\fetch.py         # again: now adds level 3
helper1m\scripts\iran\check.py
helper1m\scripts\build_country.py iran
```

The .pbf was deleted after the two OSM scripts ran (2026-10-06); the two gpkgs are kept.
`bakhsh.py` takes a few minutes (the overlay and the Wikidata name lookups, cached in
`data/iran/raw/wikidata_*.json`). Everything else runs in under a minute.

## Codes

| level | code | example | name |
|---|---|---|---|
| 1 province | COD-AB `adm1_pcode` | `IR028` | English, COD `adm1_name` (`Tehran`); Persian in `name_cn` |
| 2 county | COD-AB `adm2_pcode` | `IR028015` | English, COD `adm2_name`; Persian in `name_cn` |
| 3 district | county pcode + SCI district code(s) | `IR028015-02`, merged `IR029004-0102` | English (OSM / Wikidata); Persian in `name_cn` |

The province codes are COD's, numbered alphabetically by English name, not SCI's own province
codes (SCI: 00 Markazi ... 23 Tehran ... 30 Alborz). The full list: IR001 Alborz, IR002 Ardabil,
IR003 Bushehr, IR004 Chaharmahal and Bakhtiari, IR005 East Azerbaijan, IR006 Fars, IR007 Gilan,
IR008 Golestan, IR009 Hamadan, IR010 Hormozgan, IR011 Ilam, IR012 Isfahan, IR013 Kerman, IR014
Kermanshah, IR015 Khuzestan, IR016 Kohgiluyeh and Boyer-Ahmad, IR017 Kurdistan, IR018 Lorestan,
IR019 Markazi, IR020 Mazandaran, IR021 North Khorasan, IR022 Qazvin, IR023 Qom, IR024 Razavi
Khorasan, IR025 Semnan, IR026 Sistan and Baluchestan, IR027 South Khorasan, IR028 Tehran, IR029
West Azerbaijan, IR030 Yazd, IR031 Zanjan.

**For composition pies:** languagedots' and religiondots' Iran `geo_id` is the English province
name, which is exactly the level-1 `name` here (all 31 asserted equal on 2026-10-06). Join on
`name`, or map name to code with the list above. Both projects use the same 1395 province
units, so Tabas is in South Khorasan in all three.

`data/iran/counties.csv` holds each county's SCI code (`sci_code`, province + county, e.g.
`2301` Tehran) beside its COD pcode; `boundaries/adm3.gpkg` holds each district unit's SCI
district codes (`sci_districts`, province + county + district).

## Sources

**1395 census by settlement.** SCI (Statistical Centre of Iran), `CN95_HouseholdPopulationVillage_NN_r.xlsx`,
one per province, `amar.org.ir/Portals/0/census/1395/results/abadi/`. Every province, county,
district, rural district (dehestan), city and village row with households, population, men and
women. `_r` is the revised file and the only one for Kurdistan (12), so `_r` is used for all.
Rows: record type 1 province, 2 county, 3 district, 4 dehestan, 5 city, 6/8 village. A zoned
city has a total row plus one row per zone (`ShahrTop` filled); the zone rows are dropped.
Villages of three households or fewer print `*`; their people are inside every parent row.
Non-settled (nomadic) households are counted at county level only (the notes sheet says so):
48,798 people, which is why a county can exceed the sum of its districts.

**1390 census by settlement.** SCI, `ABADY-90/osNN.xls` (Fars, Mazandaran, West Azerbaijan only
as `osNN-r.xls`), `amar.org.ir/Portals/0/sarshomari90/Files/ABADY-90/`, on the 1390 divisions.
The address code is province (2) + county (2) + district (2) + dehestan or city (4) + census
block (3) + village (6). District `99` is a county's non-settled population. Zoned cities appear
only as their zones (`اراك 1`, `تبريز4-`), without a total row. Rows are not in hierarchy order
(a dehestan's row can come after its villages).

Both are read through the Wayback Machine (`web.archive.org/web/2020id_/...`), because
`amar.org.ir` resets every TLS connection from here (religiondots found the same).

**Counties.** OCHA COD-AB Iran (valid 2019-05-14, `data/asia1m/iran/irn_admin{1,2}.shp`, the
same files religiondots read). Its 429 counties are the 1395 census's: COD-PS ADM2 2016
(`irn_admpop_adm2_2016_v2.csv`, HDX) gives each pcode the census count, and each SCI county
is joined to the COD county of its province with the same 1395 population. All 429 are unique
matches and every Persian name agrees (one spelling, Haftgel/Haftkel).

**Districts.** OpenStreetMap district (bakhsh, `admin_level=6`) polygons from the Geofabrik
extract of 2026-10-05: 1,137 polygons, which are the 2026 districts. SCI's settlement file for
the 1400 divisions, `GEO1400.xlsx` (`amar.org.ir/Portals/0/GEO/2024/`, Wayback 2025), lists
every 1400 county, district, dehestan, city and village with the same six-digit village code the
1395 census uses, which is the bridge between OSM's names and the 1395 districts (below).

**2024 provincial estimate.** SCI's estimate of each province's population in 1403, in
thousands, as reproduced by Iran Open Data (`iranopendata.org`, dataset iod-06124, "برآورد جمعیت
ایران به تفکیک استان در سال ۱۴۰۳", source line "مرکز آمار ایران"). The live site sits behind a
Cloudflare challenge, so the Wayback copy of 2025-06-14 is parsed. The 31 rows sum to 85,964,000
against a printed national 85,961,000 (rounding); the province rows are used. SCI's own release
was not found in the archive. Treated as 2024 (the Iranian year 1403 runs March 2024 to March
2025).

## The 1390 carry (2011 on the 2016 units)

About 40 counties were split or reshaped between the two censuses (Varamin lost Pishva and
Qarchak, Ahvaz lost Karun, Karaj lost Fardis), so 1390 county rows cannot be used directly.
Every 1390 settlement unit is followed to the 1395 district (and so county) its people are in:

- **Villages**, by the six-digit village code, which both censuses share (unique within a
  province in both). 95% of listed 1390 village people match a 1395 village. Codes are matched
  inside the province first, then nationally with the name checked, for Tabas (below).
- **A dehestan** is spread over the 1395 districts its matched villages went to, by their 1390
  people. That carries the suppressed `*` villages and unmatched villages along with their
  neighbours.
- **Cities**, by name inside the province, with zone numbers stripped so a zoned city's zones
  come back together.
- **Unmatched units** (renamed cities, dehestans with no matched village) take the distribution
  of their 1390 district, or failing that their 1390 county. Carried by: own match 74,564,757;
  district 528,666; county 56,246.
- **Non-settled people** of a 1390 county follow its settled people.
- **One hand split** (`CITY_SPLITS` in `fetch.py`): Fardis (181,174 in 1395) was part of Karaj
  city until 2013; the 1390 file has no Fardis, and Karaj's 1390 zones held both. Karaj's 1390
  people are shared between 1395 Karaj and Fardis by their 1395 counts. Found by listing every
  1395 city of over 5,000 with no 1390 city or village of its name; the rest of that list are
  spelling changes (Chabahar, Talesh) or villages that became towns in their own county.

Rounding: provinces first (largest remainder to SCI's 75,149,669), then counties inside them, so
every province whose lines did not move equals its 1390 publication exactly. Two moves change
province totals: **Tabas** (69,658) was in Yazd in 1390 and in South Khorasan in 1395 (it went
back to Yazd in 2018), and 540 people moved from Tehran to Alborz. Since the units are 1395's,
Yazd's 2011 figure is 69,658 below the 1390 publication and South Khorasan's above it.

Check: 390 of the 429 counties carry a 1390 county's name; 311 of those equal their 1390 row
exactly and 347 are within 1%. The rest are the counties that lost land (Varamin x0.56,
Nikshahr x0.60, Kangan x0.62 and so on, each to a county created in between).

## 2024: provinces from SCI, counties and districts by model

No county figure after 1395 was found in an open file. SCI did publish county estimates for
1400 (469 counties, on the 1400 map, May 2022) and a rebuild of 1390 and 1395 onto those
counties, but neither file is in the Wayback Machine (CDX searched by keyword and by every
spreadsheet on `amar.org.ir`); only press quotes survive.

So each province's 1403 total is spread over its counties: each county is first grown by its
province's ratio, then half of its own 2011-2016 annual lead or lag over the province is carried
on for the eight years, and the province is raked back to SCI's total (`forward_cast`,
`FORWARD_DAMP = 0.5`). Districts are done the same way inside their county. Half, because six
county figures from SCI's 1400 estimate quoted in the press fit a line through 2016 and 2024
best at 0.5:

| county | SCI 1400 | model at 1400 (half) | flat province rate | full trend |
|---|---:|---:|---:|---:|
| Tehran | 9,039,000 | x1.005 | x1.019 | x0.989 |
| Mashhad | 3,619,000 | x1.003 | x0.992 | x1.013 |
| Isfahan | 2,178,000 | x1.058 | x1.068 | x1.047 |
| Sanandaj | 528,500 | x0.988 | x0.980 | x0.995 |
| Saqqez | 236,300 | x0.989 | x0.990 | x0.985 |
| Sarvabad | 43,700 | x0.974 | x1.062 | x0.895 |

(Isfahan is high at every setting; Isfahan county probably lost land by 1400 to the new
Jarquyeh, Kuhpayeh, Varzaneh and Harand counties, which is what the press figure is for.)

The full trend would also have run Taleghan down 10% a year for eight years. Projected county
growth 2016-2024 by this rule: median +0.4% a year, 5th-95th percentile -1.8% to +3.1%.
**The province totals are SCI's; the split inside a province is mine.** Fast-growing new towns
(Pardis, Parand in Robat Karim, Sahand in Osku) keep growing faster than their province, at half
the 2011-16 pace.

## Districts (level 3)

`bakhsh.py`. The 1395 districts need polygons, and the only open ones are OSM's of 2026, ten
years and about 40 new counties later. A district promoted to a county comes back as that
county's "Central District", and new districts have been cut from old ones. So:

1. Each OSM district is matched to a **1400 district** of `GEO1400.xlsx` by county name and
   district name inside its province (1,074 of 1,137; `بخش مرکزی شهرستان X` and `مرکزي` both
   fold to "Central"; county spellings are matched loosely, Chabahar / Chah Bahar).
2. A 1400 district's people are known in **1395 districts**: its villages by code and its cities
   by name, with their 1395 counts.
3. Each OSM polygon is cut on the **COD county lines**, so districts nest in counties. A piece
   goes to the 1395 district holding most of its 1400 district's people inside that county.
4. **Located settlements**, as a second route that needs no district names: 31,651 1395
   settlements (72.6 million people) whose name matches exactly one OSM place node inside their
   own county. On pieces matched by name, 99.7% of the located people sit in the district the
   name route chose. Pieces with no 1400 match (100, mostly districts created after 1400) take
   their composition from the located settlements, and on one piece they overruled the name
   route (OSM's "Jarquyeh Sofla", now its county's "Central", which a close-name match had sent
   to Jarquyeh Olya).
5. **Merges.** When a piece's people come from two 1395 districts at 15% or more, those
   districts become one unit (merge to the common piece, never split). A 1395 district with no
   piece joins the unit holding most of its people. Result: eight merged pairs, in Nir,
   Kazerun (Kuhmareh + Chenar Shahijan), Fahraj, Khoshab, Khash, Nehbandan, Chaypareh
   (Hajjilar + Central) and Sardasht; one 1395 district had no polygon of its own.
   1,057 districts -> 1,049 units.
6. **Slivers** (pieces under 2 km² or 2% of their OSM district; 2,607 of 4,359, the mismatch
   between OSM and COD lines) follow their own composition or their neighbour.

**Populations.** Each county's figure is shared over its units in every year: 2016 by the
districts' own counts (the county's non-settled people pro rata), 2011 by the carried 1390
settled people, 2024 as above. Every district sums to its county in every year.

**Names.** English from OSM `name:en`, the OSM polygon's Wikidata label, or the OSM county name
for a district that has become a county; then a Wikidata search on the exact Persian label
("بخش X", or "بخش X شهرستان Y" as Wikidata writes them). 13 units still have a part with no
English name anywhere; those parts are rough letter-for-letter romanisations marked `*`
("Ruddsht District*"), since Persian script leaves out short vowels. Persian names (SCI's
spelling) are in `name_cn`.

## Checks (check.py, 2026-10-06)

- **Totals.** 2016: 79,926,270, SCI's published total, at every level; provinces equal the
  census province rows and COD-PS ADM2 sums. 2011: 75,149,669, SCI's published 1390 total. 2024:
  85,964,000 = the sum of SCI's 31 province estimates. Levels nest exactly in every year.
- **Coverage.** 31/31, 429/429 and 1,049/1,049 units have all three years; no population row
  without a unit.
- **Counties against Kontur 2023** (religiondots' county table, read only): 55% within 25% after
  the national factor. Kontur is no witness here: it draws false cities at its density cap in
  rural counties (Sarvestan 47x, Kherameh 32x, Kavar 19x, Torghabe-o-Shandiz 20x the census),
  which religiondots documented and calibrated away.
- **Districts against religiondots' county-calibrated Kontur hexes**: 87% of people in
  districts within 25%. The big misses (Tukahur, Kuhsar, Kurin at 0.03-0.11) are Kontur's rural
  shape inside a county; the located-settlement check puts those districts' own villages inside
  their polygons.
- **Districts against located settlements**: 99.7% of the 72.6 million located people fall inside
  their own district's polygon. Five units hold more than 20% of other districts' located people
  (Lalehabad 70% own, Rudan's Central 74%, Yazdanabad 75%, Sarduiyeh 75%, Chavarzaq 77%):
  OSM's district lines there have moved since 1395 (a dehestan transferred between districts),
  so a few thousand people sit in the neighbouring unit's polygon.
- **Spot checks.** 2016: Tehran county 8,737,510, Mashhad 3,372,660, Shiraz 1,869,001 (the census
  figures). 2011 carried: Shahreza 149,555, equal to its 1390 county row.

## Known weaknesses

- **2024 inside a province is a model** (above). The provinces are SCI's.
- **Big cities are one district each.** Tehran county's Central District is 8.7 million, Mashhad's 3.3
  million; city zones (mantaqe) are counted in the 1395 file but have no open polygons.
- **Counties are 2016's.** About 40 counties created since (Eshqabad, Jarquyeh, Kuhpayeh, Harand,
  Dashtiari, ...) are inside their 2016 county. At district level many of them show up, because
  they were districts in 2016.
- **Taleghan** falls from 26,976 (1390) to 16,815 (1395). Both are SCI's settlement counts with
  no boundary change; it looks like a change in who was counted as resident (Taleghan fills with
  summer residents). Carried into 2024 at half pace.
- **Nomads** (48,798 in 2016) are spread pro rata over their county's districts.
- **Thirteen district names** are rough romanisations (marked `*`).

## Searched and not used

- SCI's 1400 county estimate and its rebuild of 1390/1395 on the 1400 counties (the best
  possible source): announced May 2022, "در درگاه ملی آمار"; no file in the Wayback CDX for
  `amar.org.ir` (every spreadsheet listed, keyword filters `baravord`, `bazsazi`, `shahrestan`,
  `1400`). Press articles quote a few counties only.
- iranstatis.com: provincial series 1365-1405 "reconstructed on the 1395 divisions", behind a
  paid subscription.
- SCI's 1390 county table PDF (`jamiat_shahrestan_keshvar3.pdf`): the Wayback copy is cut at
  1 MiB; not needed, since the settlement file's own county rows close.
- Fehrest_Taghsimat_Keshvari_1401.xlsx (SCI's 1401 division list): has no village codes, so it
  cannot bridge to 1395 the way GEO1400 does.
- Overpass: 504 on three endpoints on 2026-10-06, hence the Geofabrik extract.

## Files

- `download.py` — SCI workbooks (Wayback), GEO1400, the 1403 estimate page.
- `census.py` — readers for the two settlement workbooks; `fold()` for Persian names.
- `fetch.py` — 1395 tables, COD join, the 1390 carry, the 2024 forward cast, `population.csv`,
  `counties.csv`, `carry_1390.csv` (every 1390 unit, how it was carried, where to).
- `osm_admin.py`, `osm_places.py` — OSM admin polygons and place nodes out of the extract.
- `bakhsh.py` — district polygons and units; `bakhsh_units.csv`, `bakhsh_pieces.csv` (every OSM
  piece, its 1400 district, where it went).
- `check.py` — the checks above.
