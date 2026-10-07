# Tuvalu (tv): record

Drawn 2026-10-05 (session edd42a8c-last). 10,532 of 10,632 residents (2022 census), two units
(Nui, the rest), 3 nodes, rows `derived`. 9 dots, 2 rings.

Files: `sources/tv_census.py`, `taxonomy/tv2022.py`, `taxonomy/tree.d/tv.txt` (bare repeats),
`countries/tv.py`, `data/normalized/tv.csv`, `data/geo/tv/tv_hexes.gpkg`,
`data/raw/tv/tuvalu_2022_census_report.pdf` (spc.int/digitallibrary/get/zskjx; the SPC pages
403 to WebFetch, plain curl with a browser UA works).

## Source

Tuvalu 2022 Census on Population and Housing, analytical report (2025). No language question:
Table 10 is literacy (Tuvaluan 97%, English 89%, Nuian 19% of 5+). Figure 6, ethnicity:
Tuvaluan 94%, Tuvaluan/I-Kiribati 4%, Tuvaluan/Other 1%, Other 1%, not stated 1%. Table 9:
resident population 10,632 by island of enumeration (Nui 514).

## How (AGENT_BRIEF section 2, ethnicity read as language)

- Nui: Nuian is a Gilbertese dialect (Glottolog gilb1244 lists TV); all 514 enumerated on Nui on
  Gilbertese.
- Rest (10,118): Figure 6 shares; the three Tuvaluan answers on Tuvaluan, Other on `other`,
  not stated left out (100, the gap).
- No retention source exists: ethnicity is nearly universal Tuvaluan, so nothing to move.
- Placement: religiondots' one-unit hexes re-keyed by a box around Nui (7 hexes, 377 Kontur
  people; asserted).

## Calls someone might reverse

- Tuvaluan/I-Kiribati (4%) drawn as Tuvaluan rather than split.
- All of Nui on Gilbertese (some there will speak Tuvaluan first); Nui people on Funafuti on
  Tuvaluan.

## Room for improvement

A home-language item (none in 2012, 2017 or 2022); the 2022 microdata (Pacific Microdata
Library, catalog 829) might cross ethnicity with island and birthplace.
