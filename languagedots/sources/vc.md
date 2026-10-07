# Saint Vincent and the Grenadines (vc): record

Drawn 2026-10-05 (session edd42a8c-mono3). No language question: Vincentian Creole for everyone
except the 889 who gave their ethnicity as white (English), 219 populated enumeration districts,
2012 census, rows `derived`. Placed on religiondots' ED polygons by area (read-only; they are
small, 500 people on average, and religiondots draws them the same way). 108 dots.

Files: `sources/vc_census.py`, `taxonomy/vc2012.py`, `taxonomy/tree.d/vc.txt`, `countries/vc.py`,
`data/normalized/vc.csv`.

## Source

US Census Bureau subnational tables for SVG (2021-09; religiondots'
`data/raw/vc/saint_vincent_and_the_grenadines_uscb_202109.xlsx`): seven tables of the 2012
census by ED, none on language. "Ethnicity and Religion" gives Black 77,763, Mixed 25,111,
Indigenous 3,280, East Indian 1,199, White 889, Portuguese 753, Other 193. Checks: categories sum
to each ED's total; EDs sum to the national row per column; ED ids equal religiondots' both ways.

## Calls

- White on English, everyone else on Vincentian Creole (Glottolog vinc1243; node from `bb.txt`,
  hand-coloured here 0.62 0.12 185). Portuguese-descended Vincentians (753) are long-settled and
  on the creole. The creole/English continuum is not measured.
- Indigenous (Garifuna and Kalinago descent) on the creole: neither language survives on the
  island.
- Immigrants not separable (no birthplace table by ED in the USCB set).
