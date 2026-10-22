# Jersey: the record

Drawn 2026-10-19 (session eca0185b-terr, "coverage gaps"). 103,267 residents (the whole census), one
unit, 265 nodes (mostly tiny origin-mix tails), 95 dots and 220 rings at 1:1000. Every row
`derived`. Scripts: `sources/je_census.py`, `sources/terr_common.py` (written-id helper),
`taxonomy/je2021.py`, `taxonomy/tree.d/je.txt` (origin_mix block generated), `countries/je.py`.
Geography is religiondots' `je_hexes.gpkg` (one unit `JE`, 219 hexes), read in place. No asks.

## 1. Source

Statistics Jersey, 2021 Census, open data on opendata.gov.je (files in `data/raw/je/`, URLs in
`je_census.py`): `ethnicity-2021-census.csv` (ethnic and cultural identity, one answer per person,
19 groups, total 103,267), `place-of-birth-2021-census.csv` (8 groups, 103,267),
`2021-census-otherplaceofbirth.csv` (countries of birth with 10+ residents, 8,590 people),
`2021-census-placeofbirth-age.csv` (8 groups by age, rounded to 10).

The 2021 census asked no language question (Chief Minister's answer WQ.28/2024: "the Census did not
ask about languages, so there is no data on the total number of Portuguese speakers"). The 2011
census had none either. The island's birthplace and identity tables are island-wide only.

## 2. Call: identity, not birthplace, for the named groups

Birthplace would put the Jersey-born children of Portuguese and Polish parents on English. Of
8,280 Portugal/Madeira-born residents only about 450 are under 20; yet the States answer says 10%
of school students answering a survey speak Portuguese at home. Identity (Portuguese / Madeiran
9,739 = 9.4%) carries the second generation. It is the larger count and may overstate Portuguese
at home (some Jersey-born Portuguese-identity residents speak English at home); the two figures
bracket it: 8,280 by birth, 9,739 by identity. Polish 3,080 against 2,808 born in Poland, Romanian
1,427 against 1,338 born in Romania.

## 3. Mapping (`sources/je_census.py`)

- Jersey (45,849), British (31,534), Irish (2,199), South African (716), Black Other (66) and the
  three Mixed groups (1,377): English. Bermudian rule (`sources/bm.md`): the island's own and the
  British-Irish. South Africans: no Afrikaans share is known for Jersey, so English.
- French (692), Polish, Portuguese / Madeiran, Romanian, Indian, Thai, Chinese: that country's
  `origin_mix` mix (dest `je`). Romania's brings 6% Hungarian.
- White Other (3,737): birthplaces in the detail table that are European (less Romania, Poland,
  Portugal, France, Ireland and the UK, which have their own groups) plus Australia, Canada, New
  Zealand and the US (drawn as English), weighted by head count, each through `origin_mix`.
- Asian Other (853), Black African (764), Black Caribbean (145): the Asian, African and Caribbean
  birthplaces in the detail table (lists in `je_census.py`), weighted the same way.
- `sources/terr_common.py::mix_written` turns `origin_mix`'s drawn ids back into written ids,
  because the fragment tool `origin_mix.py --fragment` expects written ids; without it the
  generated block carried regrouped groups (`zone_e`...) that `build.py` rejects.

## 4. Checks (asserted)

Both tables sum to 103,267; the detail table (8,590) is within "Other European country" +
"Elsewhere" (8,932; countries under 10 are not listed); Portuguese, Polish and Romanian identities
each exceed their country-born count; the rows sum to 103,267.

## 5. Not drawn

Jerriais (Norman French, Jersey's own language): no count. The 2001 census found 2,700 people
could speak it and 113 used it every day; the latter is a tenth of a dot, so it is not drawn.
Residents who speak it are drawn as English. Jersey French as an official language is not a home
language.

## 6. Geography

One unit, as religiondots (`je_hexes.gpkg`, Kontur: 111,802 people against 103,267 counted).
The parish population tables on opendata.gov.je (12 parishes) were not used; gov.je also has a
parish profile PDF for each parish, which may carry birthplace or identity by parish (not read).
Room for improvement: parish-level identity or birthplace.
