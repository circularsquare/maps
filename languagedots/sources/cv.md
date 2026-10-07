# Cape Verde (cv): record

Drawn 2026-10-05 (session edd42a8c-mono2). No language source in the census: all 491,233 people
of the 2021 census on Kabuverdianu across the 22 concelhos, rows `derived`. Placed on
religiondots' Kontur hexes (read-only). Haiti's model (`sources/ht.md`). 491 dots.

Files: `sources/cv_pop.py`, `taxonomy/cv2021.py`, `taxonomy/tree.d/cv.txt` (bare repeats; the node
is us.txt's and st.txt's `creole.portuguese_based.kabuverdianu`), `countries/cv.py`,
`data/normalized/cv.csv`.

## Why everyone is drawn as Kabuverdianu

- RGPH-2021's 22 concelho workbooks (religiondots' raw `119-148.xlsx`) have no language and no
  nationality table (searched for língua, idioma, crioulo, nacionalidade, estrangeiro).
- Afrobarometer home language (`python sources/mono_afro.py "Cape Verde" "Cabo Verde"`): rounds
  4-9, 7,271 respondents; Crioulo / Creole 99.51-99.86% per round, Portuguese 0-0.45% (about 20
  respondents, mostly in Praia and Mindelo), one "Other". The survey samples only Santiago, Fogo,
  Sao Vicente and Santo Antao, so it corroborates rather than builds the map.
- Portuguese is learned (AGENT_BRIEF §2, supervisor's brief): not drawn, as Haiti's French.

## Population and checks

religiondots' `cv_lookup.csv`: RGPH-2021 resident population per concelho, 491,233, asserted 22
rows, geo_id == unit. `check_country.py cv`: ok.

## Calls someone might reverse

- Portuguese home speakers (0.2-0.3% in the survey) not drawn.
- Foreign residents (West Africans, Portuguese, Chinese; no count by concelho) drawn as
  Kabuverdianu, in `gap`.
