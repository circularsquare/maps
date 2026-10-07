# Maldives (mv): record

Drawn 2026-10-05 (session edd42a8c-mono2). No language question. 2022 census usual residents,
515,132 on 20 atolls and Male, every row `derived`: Dhivehi 74.3%, Bengali 15.1%, Hindi 2.0%,
Sinhala 1.6%, other 1.0%, Tamil 1.0%; 57 nodes. Placed on religiondots' Kontur hexes cut to
islands (read-only). 499 dots, 42 rings.

Files: `sources/mv_census.py`, `taxonomy/mv2022.py` (identity), `taxonomy/tree.d/mv.txt` (31 bare
repeats, generated), `countries/mv.py`, `data/raw/mv/` (2022 tables MG1-MG13, P1-P6 from the
Wayback Machine; the migration report PDF), `data/normalized/mv.csv`.

## Sources

`census.gov.mv` answers 404; its 2022 tables are in the Wayback Machine (CDX of
`census.gov.mv/2022/*`). No 2022 table gives nationality by atoll: MG9/MG10/MG11 split only
Maldivian and foreign. Nationality is in the migration report ("Population Movement & Migration
Dynamics", 2024), Table 6.1, for Male and the atolls together: Bangladesh 74,815, India 32,971,
Sri Lanka 11,338, Nepal 4,222, Indonesia 2,301, Philippines 1,723, Others 5,123. The script finds
each row's text in the PDF and asserts the columns sum to MG11's foreign totals (52,799 Male,
79,694 atolls).

## Method

- Maldivians (382,639) on Dhivehi.
- Each atoll's foreign residents (MG11) at the atolls' nationality mix; Male at Male's.
- Each nationality at its country's drawn mix on this map (sa.md's home-mix method, 1% cut):
  India's 18 languages, Sri Lanka Sinhala 75 / Tamil 25, Bangladesh Bengali, the Philippines,
  Indonesia, Nepal. "Others" on `other` (no breakdown; resort staff of many nationalities).
- No TeO2 retention: temporary workers (most stay 3-4 years, the report says), and Dhivehi is
  not a language they take up at home.
- Atoll codes (HA, HDh...) joined to religiondots' units by name, asserted against its
  `mv_lookup.csv`.

## Calls someone might reverse

- India at its whole-country mix (Hindi 27%). Indians in the Maldives are probably more
  Keralite and Tamil than that (teachers, nurses); no source by state was found.
- Same nationality mix in every atoll.
- Placement inside an atoll follows Kontur, not the resort/industrial islands where most of
  the atolls' foreigners live (MG11: 48,770 of 79,694 on non-administrative islands). A
  placement by island type would need MG11's split per atoll on the hex layer.

## Room for improvement

Nationality by atoll (the census microdata, a request form asking identity) or a work-permit
count by atoll.
