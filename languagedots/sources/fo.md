# Faroe Islands (`fo`): Census 2011, first language by district

Built 2026-10-05 by session `edd42a8c-fo`. Code: `sources/fo_census.py` (fetch, normalise,
checks), `taxonomy/fo2011.py` (mapping), `taxonomy/tree.d/fo.txt` (no new node),
`countries/fo.py` (entry). Geography is religiondots' (read only).

## 1. Source

- **Hagstova Føroya statbank, Census 2011** (`H2/MT/MT01/MT0103`, "Language"), open PxWeb v1
  API, POST, no key. Census day 11 November 2011; the only Faroese census since 1977, and none
  since.
  - **MT16** *MT1.3.3 Population by primary language, country of birth of person, father and
    mother and place of usual residence* (table updated 2025-06-23). Drawn: first language by
    the 7 districts, birth totals. Fetched a second time by country of birth (national) for the
    mapping calls.
  - **MT17** *MT1.3.4 ... age and sex*: first language by 5-year age, national. Witness.
  - MT14/MT15 (Faroese language *skills*) exist beside them; not used.
  - Queries saved beside the CSVs: `data/raw/fo/*.query.json`.
- **The question.** One first language per person for the whole population, children included:
  no not-stated row, and the ten answers sum to each district's total. The table's variable is
  "first language", its title "primary language"; `how` says first language.
- Ten answers: Faroese, Danish, Other Nordic languages, Other European languages, Asian
  languages, Middle East/North African languages, Other African languages, South American
  languages, Sign language, No language. Nothing finer is published in the statbank.

## 2. Checks (all asserted in `fo_census.py`)

- MT16 total 48,346, equal to religiondots' MT1 resident count, and the districts sum to it.
- Every unsuppressed row's districts sum to its national cell; each district's ten answers sum
  to its total.
- MT17's eleven language totals equal MT16's.
- National: Faroese 45,361 (93.83%), Danish 1,546 (3.20%), Other Nordic 411, Other European 607,
  Asian 290, ME/North African 40, Other African 31, South American 1, Sign language 18,
  No language 41.
- No second table of this census gives language by district, so there is no per-unit witness
  beyond the arithmetic.

## 3. Suppression

Hagstova prints `...` for a cell under 3. The district query hides 22 cells in five rows, and
the South American national cell (1, by the total less the other nine rows). Both margins of the
hidden cells are known (row: national less printed districts; column: district total less printed
rows: Norðoyar 4, Eysturoy 1, N-streymoy 4, S-streymoy 0, Vágar 4, Sandoy 0, Suðuroy 2). Only
**two** integer tables with every hidden cell in 0..2 meet both margins; `fill()` enumerates them
and averages, so a few cells are 0.5 or 1.5. The two differ only in whether Norðoyar or Vágar
holds 1 ME/North African and 1 Other African speaker.

## 4. Mapping calls (`taxonomy/fo2011.py`)

- Faroese, Danish: the North Germanic leaves already in fi.txt (repeated in fo.txt so this
  country stands alone).
- **Other Nordic languages (411) on `other`, not North Germanic.** By country of birth (MT16):
  277 born in other Nordic countries, 56 in Greenland, 44 in the Faroes, 10 in Denmark. The
  Greenland-born have Danish and Faroese as separate answers, so their "other Nordic" language
  is almost surely Greenlandic (Eskimo-Aleut); Finnish and Sami would also be filed here. The
  narrowest node holding all of that is `other`.
- Other European (607), Asian (290), ME/North African (40): `other`, as bw2011, na2011, at2001.
  Other European takes 31 USA-born and 42 South-America-born speakers, so English, Spanish and
  Portuguese are in it.
- Other African (31): `africa_other` (at2001's precedent).
- **South American languages (1): `americas_other`.** Since Spanish and Portuguese are filed as
  European (42 South-America-born in Other European against at most 2 in this row), this row is
  an indigenous South American language. One person; recorded so nobody wonders.
- Sign language (18): `signlanguage`. No language (41, 22 of them under 5; MT17): not drawn, in
  `gap`.
- Colours: Faroese generates a sea green (#69b799) and Danish a sky blue (#61b1cf); distinct
  enough, no hand pick.

## 5. Geography and placement

Religiondots' `data/geo/fo/fo_hexes.gpkg` (its `sources/fo_geo.py`): Kontur hexes labelled with
the census's 7 districts, `unit` spelled exactly as MT16 prints it (asserted both ways in
`countries/fo.py`), `pop` the November 2011 village register calibrated over Kontur (religiondots
rejected raw Kontur, which halves Tórshavn). Plain `pop_weight`. No within-district proxy for the
immigrant languages: the register has no village by citizenship or birthplace table
(statbank `IB01` checked), so every language is placed on the village population.

check_country: 48,305 people, 7 units, 6 nodes, ok. Scatter at 1:1000: 47 dots, 3 rings;
1,305 people sit under one dot per language nationally and draw no dot.

## 6. Not done, where to reopen

- Hagstova's census publications might print the "other" languages by name nationally; not
  searched. A national split would not change the drawing at this grain much.
- MT14 (Faroese language skills) could note how many non-Faroese first-language speakers speak
  Faroese; not used.
