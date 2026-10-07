# Nauru (nr): record

Drawn 2026-10-05 (session edd42a8c-last). 11,680 people (2021 census), national, 49 nodes (most
1-person shares of foreign origin mixes), rows `derived`. 11 dots, 48 rings.

Files: `sources/nr_census.py`, `taxonomy/nr2021.py` (identity), `taxonomy/tree.d/nr.txt`,
`countries/nr.py`, `data/normalized/nr.csv`. Workbook: religiondots'
`data/raw/nr/population-housing-census-2021-tables-vol1.xlsx`, read-only.

## Source and checks

2021 PHC Table I-1, population by district and ethnicity: 16 ethnicities x 15 districts; the
districts sum to the TOTAL row in every column and the ethnicities to the total (asserted).
Nauruan 11,046 (94.6%), Kiribati 261, Fijian 153, Tuvaluan 68, Solomon Islander 42, others <25.

## How (AGENT_BRIEF section 2, ethnicity read as language)

- Retention: the 2011 census report (Republic of Nauru National Report on Population and
  Housing, 2011, nauru-data.sprep.org) has 95% of people 5+ speaking Nauruan at home (multiple
  answers; English 66%). That matches the Nauruan ethnic share, so ethnic Nauruans are drawn
  on Nauruan whole.
- Foreign ethnicities through `origin_mix.mix(iso, "nr")`; Kosraean on Kosraean (fm.txt);
  "Other ethnicity" (24) on `other`.
- National: religiondots' layer is one unit and the district split moves fractions of a dot.

## Calls someone might reverse

- Fijian ethnicity (153) on Fiji's home mix rather than iTaukei Fijian only (Fiji is not drawn
  at the time of writing, so it is Fijian whole; it will follow fj once drawn).
- District grain thrown away.

## Room for improvement

The 2011 census language table by district, if published as a table, would give a measured
alternative.
