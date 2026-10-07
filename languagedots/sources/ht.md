# Haiti (ht): record

Drawn 2026-10-05 (session edd42a8c-carib). No language source: all 11,899,555 people on Haitian
Creole across the ten departments, rows `derived`, on the COD-PS 2024 department populations.
Placed on religiondots' Kontur hexes (read-only).

Files: `sources/ht_pop.py`, `taxonomy/ht2024.py`, `taxonomy/tree.d/ht.txt` (bare repeats),
`countries/ht.py`, `data/normalized/ht.csv`. Population file is religiondots'
`data/raw/ht/hti_admpop_adm1_2024.csv`, read only.

## Why everyone is drawn as Creole

- The 2003 RGPH asked literacy, not language; no fifth census has been held (scout 2026-10-05).
  religiondots' ECVMAS 2012 microdata was not searched for a language item: in a country where
  the answer is near-universal it could not change the map.
- Haitian Creole is the first language of nearly every Haitian; French is acquired through school
  and spoken fluently by a minority (commonly put at 5-10%), and AGENT_BRIEF §2 (2026-10-05) says
  learned second languages are not drawn. French-speaking-at-home families, a small elite, have no
  count anywhere and are drawn as Creole, as are Spanish speakers on the border; both said in
  `note_public`.

## Population and checks

COD-PS 2024 (IHSI projections on HDX), the same figures religiondots carries as `pop_2024` in
`ht_lookup.csv`; the script asserts the two agree per department and that the join is 1:1 on the
p-codes. Department, not commune: with one language there is nothing for a commune split to show,
and the hexes already place people inside each department by Kontur population.

## Calls someone might reverse

- French first-language speakers not drawn (no count exists; a guessed share would be invented).

## Scatter

11,899 dots over 7,531 hexes. Note: `taxonomy/build.py` currently fails on gr2021's two missing
nodes (another session's work in progress); Haitian Creole was already in languages.json, so ht
is unaffected.
