# Barbados (bb): record

Drawn 2026-10-05 (session edd42a8c-mono3). No census language question. Built under the
2026-10-05 ruling for countries with no language question (AGENT_BRIEF §2), as the Bahamas
(`sources/bs.md`): the native-born on the national vernacular, immigrants by birthplace. 2010
census, tabulable population 226,193, 11 parishes, 16 nodes, every row `derived`. Placed on
religiondots' Kontur hexes (read-only). 219 dots, 8 rings.

Files: `sources/bb_census.py`, `taxonomy/bb2010.py`, `taxonomy/tree.d/bb.txt`, `countries/bb.py`,
`data/normalized/bb.csv`. Raw workbook is religiondots' `data/raw/bb/bb_census_tables_2010.xlsx`,
read only.

## 1. Sources

- No language question in 2010 or 2021 (religiondots' copies of both workbooks list every table).
- **2010 census workbook** (Barbados Statistical Service): 01.01 parish totals; 02.04 parish x
  ethnic origin (Black 209,109, White 6,135, East Indian 3,018, Mixed 7,034, ...); 04.02
  Barbadian-born by parish of usual residence (193,368); 04.01 country of birth, national only
  (32,825 foreign-born, of whom 12,164 "Countries Unknown").
- **2021 not used**: its tables cover 136,415 people, about half the resident population.
- The 2010 tabulation is itself 81.4% of the estimated resident population (BSS Table A);
  religiondots draws the same tabulable count and so does this (`gap`).

## 2. The model

- **Foreign-born per parish** = parish total - Barbadian-born residents, split by the national
  mix of known birthplaces (unknown left out of the mix). Birthplace by parish is not published.
- **Birthplace -> language** (`taxonomy/bb2010.py`): the English-speaking Caribbean on its own
  creole, new nodes here (Guyanese Creole creo1235, Vincentian vinc1243, Trinidadian trin1276,
  Grenadian gren1247, Antigua and Barbuda anti1245, which Glottolog also gives St Kitts); St Lucia
  and Dominica Antillean Creole; Jamaica Jamaican Creole; UK, US, Canada, Australia, Bermuda,
  Belize English; India **Gujarati** (Barbados's Indian community is overwhelmingly Gujarati,
  from Surat and Bharuch); Suriname Sranan Tongo (COUNTRY_LANG's Ndyuka is a Maroon language);
  pooled "Other Asia / Other Latin America / Other Countries" on `other`.
- **Barbadian-born -> Bajan** (new node, Glottolog baja1265, hand colour 0.58 0.13 128),
  **except white Barbadians -> English.** White Barbadian-born = parish White x parish
  Barbadian-born share: 5,205 nationally. The first try, White minus the UK/US/Canada-born,
  left 1,313 (zero in St Michael), because many UK- and US-born residents are the Black
  children of returning Barbadians; proportional is the less wrong assumption.
- East Indian, Mixed and other Barbadian-born are on Bajan.

## 3. Checks (asserted)

01.01, 02.04 and 04.02 agree on 226,193 and 193,368; 02.04's parish totals equal 01.01's and its
categories sum to each total; the known countries plus Unknown equal 32,825; each parish's rows
sum to its population; parish names join religiondots' ids both ways.

## 4. Calls someone might reverse

- All non-white Barbadian-born on Bajan (83%): Bajan and English form a continuum and no survey
  counts English-dominant Barbadians.
- White Barbadians on English, estimated proportionally.
- Caribbean immigrants on their home creoles rather than on English (COUNTRY_LANG). Applied to
  the Bahamas too, for consistency.

## 5. Room for improvement

Birthplace by parish (the census has it) would replace the one national mix. A 2021 tabulation
with full coverage would replace 2010.
