# United Arab Emirates (ae): record

Drawn 2026-10-05 (session edd42a8c-gulf). Method shared with the other Gulf states:
`sources/gulf.md`. 7 emirates, 11,294,243 people (2024), 101 nodes, every row `derived`, on
religiondots' Kontur 400 m hexes by population. 11,247 dots at 1:1,000; 10 rings.

```
python sources/ae_build.py
python taxonomy/build.py
python tools/check_country.py ae
python scatter.py --country ae
```

Files: `sources/ae_build.py`, `sources/gulf_mix.py`, `taxonomy/ae2024.py`,
`taxonomy/tree.d/ae.txt`, `countries/ae.py`, `data/normalized/ae.csv`.

## Tables

- **Population by emirate**: religiondots' layer (`../religiondots/sources/ae.md` §2, read through
  its `data/normalized/ae.csv` and `ae_foreign.csv`, read-only): each emirate's newest total, the
  four older ones scaled together to FCSC's 2024 national 11,294,243; each emirate's newest count
  of Emiratis (2005-2022) grown to 2024, 1,519,227 in all (1.09x GLMM's FCSC-based series).
  Asserted to both totals.
- **Origins**: UN DESA International Migrant Stock 2024, UAE column, 33 named origins and
  `Others` 248,004 (3.0%). DESA's UAE figures look modelled (religiondots: shares nearly constant
  across years) but are the only origin table; no nationality by emirate exists. One mix for both
  sexes (religiondots: DESA's UAE men and women nearly alike) and for every emirate.

## Results

Gulf Arabic 14.2% (Emiratis 1,519,227 plus 80,963 Gulf-born), Bengali 12.9%, Hindi 9.2%,
Egyptian Arabic 8.9%, Malayalam 8.7%, Punjabi 5.5%, Tamil 3.3%, Levantine Arabic 3.2%, Pashto
3.1%, other 2.7%. Indians: Keralites 831,550 = 25.6% of DESA's 3,248,545.

## Gaps and room for improvement

- Emiratis who are not Arabic-speaking at home (Persian-origin 'Ajam, Baluch, East African
  Swahili-speaking families): no count.
- Nationality by emirate (Dubai Statistics Center and SCAD hold it; DSC rejects scripted
  requests) would replace the one national mix; Dubai's labour camps and Abu Dhabi's Emirati
  suburbs certainly differ.

## Placement inside units (2026-10-06)

Citizens and foreign residents are now placed apart inside each unit (session 5d7dac7e-gulf): foreign dots lean to dense hexes and fill OSM industrial land and labour camps, citizens take the rest; one rule for the six Gulf states, fitted on Kuwait's areas and Oman's wilayat. Method, fit, data searched: `sources/gulf_place.md`. Abu Dhabi emirate is split into its three regions (2023 census totals, SCAD's 2016 Emiratis by region): non-Emirati 87.0% Abu Dhabi Region, 75.2% Al Ain, 87.8% Al Dhafra (was 83.9% everywhere). Placement layer is now `data/geo/ae/ae_hexes.gpkg`. Non-Emirati share before -> after: Dubai city 91.8 -> 93.0%, Sharjah city 90.0 -> 92.9%, Abu Dhabi island + Mussafah 83.9 -> 90.3%, Al Ain city 83.9 -> 79.3%. The cities stay mixed: they are 89-93% foreign and nationality is one national mix.
