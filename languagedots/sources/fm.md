# Micronesia (fm): record

Drawn 2026-10-05 (session edd42a8c-last). 95,544 people aged 3+ (2010 census), 4 states,
15 nodes, rows `measured`. 90 dots, 6 rings.

Files: `sources/fm_census.py`, `taxonomy/fm2010.py`, `taxonomy/tree.d/fm.txt`, `countries/fm.py`,
`data/normalized/fm.csv`, `data/geo/fm/fm_hexes.gpkg`. Source workbook is religiondots'
`data/raw/fm/fsm_basic_tables_2010.xlsx` (stats.gov.fm), read-only.

## Source

The queue pointed at Table B08 (ethnicity). The same 2010 Basic Tables workbook also has Table
B10A, **"Language mainly spoken at home", persons 3+, by state**: a real home-language table,
so it is drawn instead of ethnicity. 15 labels; they sum to the 3+ total in every state and
nationally, and states sum to the national column (asserted). The 2023 census basic tables
(national and per state) have no language or ethnicity table.

Cross-check against B08 (ethnicity, single group, 2010): Chuukese + Mortlockese 46,888 vs
Chuukese home language 47,211; Pohnpeian 23,482 vs 24,373; Kosraean 2,657 single + 3,832 with
Kosraean main in multiple vs 6,077. They agree in shape.

## Mapping calls

- New leaves under Oceanic: Pohnpeian, Kosraean, Pingelapese, Mokilese, Sapwuahfik,
  "Yap outer-island languages" (Ulithian, Woleaian, Satawalese: one census answer, so one leaf),
  "Nukuoro and Kapingamarangi" (two Polynesian outliers, one census answer). Glottocodes left out.
- Chuukese includes Mortlockese (the census does not separate them at home).
- "Other Pacific Island Languages" (103) on `austronesian` (could include Palauan, outside
  Oceanic); "Chinese / Taiwanese" on the Chinese group node; Filipino on Tagalog.
- Pohnpeian, Kosraean and the Yap outer-island leaf hand-coloured apart from Chuukese.

## Placement

religiondots' hexes (municipalities in Yap and Pohnpei) re-keyed to states via
`fm_lookup.csv`. Within a state dots follow Kontur population, so Yapese and outer-island dots
mix across Yap proper and the outer atolls.

## Calls someone might reverse

- 2010 home language over 2023 ethnicity-free tables: the vintage is 13 years old.
- Children under 3 not drawn (the table's universe).

## Room for improvement

Home language by municipality (2010 microdata) would separate Yap's outer islands and Pohnpei's
atolls from the main islands.
