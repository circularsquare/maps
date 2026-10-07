# Bermuda (bm): record

Drawn 2026-10-05 (session edd42a8c-mono5). No census language question. Built as Barbados and
Antigua (`sources/bb.md`, `ag.md`): 2016 census country of birth, national, 63,743 people (36
not stated left out), 480 nodes (mostly tiny origin-mix tails), every row `derived`. Placed on
religiondots' Kontur hexes, one national unit (read-only). 57 dots, 273 rings.

Files: `sources/bm_census.py`, `taxonomy/bm2016.py`, `taxonomy/tree.d/bm.txt` (origin_mix block
generated), `countries/bm.py`, `data/normalized/bm.csv`, `data/raw/bm/bm_2016_census_report.pdf`.

## Source

2016 Population and Housing Census Report (Department of Statistics;
gov.bm/files/media-library/20260413/fdce7ebe-2016_census_report.pdf). No language question.
Table 1: total 63,779, Bermuda-born 44,411, not stated 36. Table 4.5: foreign-born by country of
birth, 19,332, every country named (171 rows). Parish breakdown of birthplace not published.

Checks (asserted): Table 4.5 rows sum to 19,332 and male + female = total per row; the three
parts sum to 63,779; Azores + Portugal = Table 1's 1,643; Table 1's UK and Canada as printed.

## Mapping

- Bermuda-born, and the UK-, US- and Canada-born, on English. Bermudian English is an English
  variety, not a creole. The UK/US/Canada call follows bb and ag: many are Bermudians' children,
  and the US and Canadian home mixes would put about 900 Spanish and French speakers in Bermuda.
- Azores on Portugal's mix; every other birthplace through `origin_mix.mix(iso, "bm")`.
  Three one-to-three-person oddities (South Georgia, Europa Island, Paracel Islands) mapped to
  GB, FR, CN.

## Calls someone might reverse

- Bermuda-born Portuguese-Bermudians on English: no count of Portuguese at home exists; most
  of the island-born grew up in English.
- UK/US/Canada-born all on English (above).
- National grain: one unit, so every language follows population.

## Room for improvement

Birthplace by parish (the census has it, unpublished) would place the Azorean and Filipino
communities instead of spreading them by population.
