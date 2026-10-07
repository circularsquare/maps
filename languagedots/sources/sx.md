# Sint Maarten (sx): record

Drawn 2026-10-05 (session edd42a8c-sx). Census 2011, language most spoken in the household,
persons in private households, whole country (the Dutch side), 32,813 people drawn in 23
languages; placed inside the country per census region (8) by birthplace.

Files: `sources/sx_census.py` (counts), `sources/sx_geo.py` (placement layer),
`taxonomy/sx2011.py`, `countries/sx.py`, `data/normalized/sx.csv`, `data/geo/sx/sx_hexes.gpkg`,
`data/raw/sx/` (Housing and Education 2011 workbooks, the 2011 census report PDF, COD-AB zip).
No tree fragment: every node exists.

## 1. What exists, and why 2011

- **Census 2011, Table F-07**, *Private households by most spoken language in the household*
  (STAT, Department of Statistics): households and persons, national only, 25 named answers plus
  "no response". One answer per household, given to every member. Published as the "Housing 2011"
  workbook (stats.sintmaartengov.org/tables.php?division=social&topic=cen ->
  `download.php?type=census&nummer=3`, sheet table_f_07) and printed in the census report
  (`download.php?type=rep&section=CEN&nummer=9`, PDF p.216). 33,162 persons in 12,854 households;
  the other 447 of the 33,609 counted lived in institutions and were not asked.
- **Census 2022** (census moment 11 November 2022, 41,901 people). The only language output is a
  chart in the *Sint Maarten Population 2022* presentation (`download.php?type=rep&section=CEN&
  nummer=18`, p.58): households' main language as shares, English 70.6, Spanish 13.2, Creole 7.8,
  Dutch 3.0, Papiamentu 1.1, Chinese 1.7, Hindi 1.0, French 0.9, other 0.8. Households, not
  persons (in 2011 the two differ: English 67.2% of households, 70.1% of persons), no counts, and
  "Creole" not split. Converting it to persons would need household sizes by language from
  elsewhere, which changes counts; 2011 drawn, 2022 quoted in `note_public`.
- **Nothing by region.** The 2011 tables pages (Education, Health, Housing, Labour workbooks) and
  the report hold language only nationally (F-07; D-13 by level of education). STAT's open-data
  portal (opendata.sintmaartengov.org, a StatLine clone) answered 503 Service Unavailable to every
  route on 2026-10-05; not retried through Wayback. The coverage sweep's lead (home language,
  2011) was right.

## 2. Checks (sources/sx_census.py)

- F-07's rows sum to 33,160 persons and 12,853 households, two persons and one household short of
  the printed totals, in the workbook and the PDF alike; accepted within 3.
- Each persons share agrees with count / 33,162 within 0.049.
- The 2022 presentation's 2011 column (English 67.2, Spanish 13.3, Creole 10.3, Dutch 3.7,
  Papiamentu 1.5, Hindi 1.3, Chinese 0.4, French 0.3) equals F-07's household shares, so the
  2022 chart's "Creole" is F-07's "french creole".
- Table D-13 (7,846 day-school attendees by household language, 19 languages) is a subset of
  F-07 row by row: no language has more attendees than persons.

## 3. Mapping (taxonomy/sx2011.py)

- **french creole (2,972) on Haitian Creole.** It could include Antillean Kweyol, but B-18 counts
  2,613 Haiti-born against 1,486 Dominica-born and 435 St Lucia-born, and most Dominicans on the
  island are English-speaking; the 2022 report's 2011 nationality chart has Haitians at 6.9%
  (about 2,300). Curaçao's 2011 check row made the same call.
- **creole (42) on `creole`**, the root: no base named, and the island has English-, French- and
  Portuguese-based creole speakers. Draws as "language not named".
- **libanese (1), jordanian (2) on Arabic**; **nigerian (8) on `africa_other`**: nationality words
  given as a language. **surnamese (5) on Sranan Tongo.** **chinese on `sinotibetan.sinitic`** as
  everywhere unsplit. **hindi** as printed, though much of the island's Indian community is Sindhi
  by origin. **other (4) on `other`.** No response (347) not drawn.
- Colours unchanged: English pale blue #c3e2fe, Spanish ochre #cbb76a, Haitian lilac #c9a3f5,
  Dutch blue #51c7f1, Hindi orange #f79133, Papiamento pale green #dceeb2. All tell apart.

## 4. Geography and placement (sources/sx_geo.py)

- **Regions.** COD-AB for Sint Maarten (HDX `cod-ab-sxm`, VROMI boundaries valid 2024-01-22),
  admin1 = the eight census regions, SX1 Low Lands to SX8 Upper Prince's Quarter; names asserted.
- **Table B-18** (population by region by country of birth, report PDF pp.48-51) **prints its
  column headers one place off**: the first column, headed Colebay, is the Not reported column
  (3,776). Asserted two ways: the Total row equals Table B-21's region totals (Colebay 5,594,
  Cul-de-sac 7,593, Little Bay 3,093, Low Lands 348, LPQ 8,143, Philipsburg 1,327, Simpson Bay 596,
  UPQ 3,139, Not reported 3,776), and the Sint Maarten-born row equals B-23's local-born, column
  by column. Every row's columns sum to its printed total within 2.
- **Hexes.** religiondots' Kontur hexes for SX (93, read only) cut along the regions; each piece
  keeps its hex's people by its share of the hex's area (4,470 Kontur people on hex area at sea or
  on the French side dropped; sharing among pieces instead put 84 people on a 540 m² sliver).
  Kontur per region against 2011, normalised: 0.90 to 1.17; log r = 0.996 against a 95th
  percentile of 0.704 over 500 shuffles (eight units, so a weak test). Pieces are scaled to each
  region's 2011 population; the 3,776 with no region are left out (only shares matter).
- **Language weights** (AGENT_BRIEF §4.4): piece people times the region's share born in:
  Dominican Republic + Colombia + Venezuela (Spanish), Haiti (Haitian Creole), India (Hindi,
  Gujarati, Telugu), China, Philippines, Netherlands (Dutch), Aruba + Bonaire + Curaçao
  (Papiamento), Suriname (Sranan), France + Guadeloupe (French); every other language by all
  foreign-born; English by population. Haiti-born are 20% of Cole Bay; Hispanic-born 19% of
  Philipsburg.

## 5. Calls someone might reverse

- 2011 persons over 2022 household shares (§1).
- Dutch placed by Netherlands-born only (683 in the regions against 1,192 Dutch speakers);
  Suriname-born and locally born Dutch speakers are left out of its weight.
- french creole all on Haitian Creole (§3).

## 6. Scatter

29 dots at 1:1000 over 24 polygons (English, Spanish, Haitian Creole, Dutch draw dots), 19 rings;
3,813 people fall under one dot per language and draw none. scatter.py's water check reported one
unit losing over 95% to the sea and left it unclipped, as on Aruba.
