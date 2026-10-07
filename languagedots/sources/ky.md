# Cayman Islands (ky): record

Drawn 2026-10-05 (session edd42a8c-mono3). No language question: 2021 census country of birth by
district, 68,443 people over 6 districts (363 DK/NS left out), every row `derived`. Built as
Barbados (`sources/bb.md`). Placed on religiondots' Kontur hexes (read-only). 65 dots, 3 rings.
English 51.7%, Jamaican Creole 24.9%, Spanish 7.7%, `other` 6.4%, Tagalog 5.6%, Hindi 2.1%.

Files: `sources/ky_census.py`, `taxonomy/ky2021.py`, `taxonomy/tree.d/ky.txt` (bare repeats),
`countries/ky.py`, `data/normalized/ky.csv`.

## Source

ESO, *Cayman Islands' 2021 Census Report* (religiondots' `data/raw/ky/ky_census_report_2021.pdf`),
Tables 4.12C, 4.12D, 4.13E-4.13I (pp.128-134), country of birth x sex x status, one per district;
Cayman Brac and Little Cayman summed into religiondots' Sister Islands. About 20 named
birthplaces plus Other and DK/NS.

**The district tables do not all add up.** The ESO's figures are rounded from weighted counts
(George Town's rows sum one over its Total, three districts two under), and three tables leave
out rows their Totals include: East End has no Guyana row (7 short), Cayman Brac no India or
Costa Rica (26), Little Cayman no Canada or Other (23). Shortfalls over 3 are kept as "Not
listed in the district table" on `other`; the district Totals themselves sum to the census's
68,811 and match religiondots' within 6.

## Calls

- Cayman-born on English: Caymanian speech is an English dialect; Glottolog has no Cayman creole.
- Immigrants by birthplace, the Caribbean on their creoles (bb.txt's nodes), Jamaica the largest.
- Honduras (2,883) on Spanish, though some are English-creole-speaking Bay Islanders; nothing
  counts them apart.
- South Africa (on English, not COUNTRY_LANG's Zulu) and India (Hindi, COUNTRY_LANG) are guesses
  about who emigrates to the Caymans; both small.
- US- and UK-born on English, as in Barbados.
