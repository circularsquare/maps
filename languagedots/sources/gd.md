# Grenada (gd): record

Drawn 2026-10-05 (session edd42a8c-mono3). No language question: Grenadian Creole for everyone
except the 973 who gave their ethnicity as white (English), 7 parishes, 2021 census, rows
`derived`. Placed on religiondots' Kontur hexes (read-only). 107 dots.

Files: `sources/gd_census.py`, `taxonomy/gd2021.py`, `taxonomy/tree.d/gd.txt`, `countries/gd.py`,
`data/normalized/gd.csv`.

## Sources

- 2021 census preliminary report (CSO; religiondots' `data/raw/gd/gd_census_2021_preliminary.pdf`):
  42 tables, none on language. Table 22 (ethnicity by parish) used for the white count; Table 19
  (place of birth) only says "abroad" (5,526), no country, national only.
- Parish totals: religiondots' parish TOTAL rows (Town of St George folded into St George); the
  script asserts each equals Table 22's total and that the typed White column sums to Table 16's
  973.

## Calls

- Everyone but white Grenadians on Grenadian Creole (Glottolog gren1247; node from `bb.txt`,
  hand-coloured here 0.64 0.13 100). As in Barbados and the Bahamas, the creole/English continuum
  is not measured, so no Afro-Grenadian is drawn as English.
- Immigrants (5.1% born abroad, no country) on Grenadian Creole: nothing says where they are from;
  most are from the neighbouring English-creole islands.
- Grenadian French Creole (Patois) not drawn: a few hundred mostly elderly speakers, no count.
- East Indians (1,467) on Grenadian Creole; Bhojpuri is no longer spoken.
