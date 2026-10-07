# Lesotho (ls): record

Drawn 2026-10-05 (session edd42a8c-mono). Afrobarometer R4-R9 (2008-2022) home language, pooled,
7,197 respondents, shares per district times the 2016 census district counts; every row
`modelled`. 2,007,201 people on 10 districts: Sesotho 98.18%, Xhosa 0.94%, English 0.47%, Phuthi
0.32%, other 0.08%. Placed on religiondots' Kontur hexes (read-only).

Files: `sources/ls_afro.py` (uses `sources/mono_afro.py`), `taxonomy/ls2022.py`,
`taxonomy/tree.d/ls.txt` (new node Phuthi; the rest bare repeats), `countries/ls.py`,
`data/normalized/ls.csv`.

## Why a survey

The 2016 census has no language or ethnicity item (IHSN catalog 8293; scout 2026-10-05). The
Afrobarometer asks home language in every round, one answer, so AGENT_BRIEF §2's survey route.
religiondots draws Lesotho's religion from the same six rounds on the same base.

## The survey's answers

| answer | R4 | R5 | R6 | R7 | R8 | R9 | pooled |
|---|---|---|---|---|---|---|---|
| Sesotho | 97.96 | 97.49 | 98.61 | 98.69 | 96.85 | 99.52 | 98.19 |
| Sethepu | 0.53 | 1.37 | 0.73 | 0.49 | 2.19 | 0.23 | 0.92 |
| English | 1.30 | 0.20 | 0.52 | 0.50 | 0.28 | 0.08 | 0.48 |
| Sephuthi | 0.00 | 0.84 | 0.14 | 0.16 | 0.68 | 0.17 | 0.33 |
| Other(s) | 0.21 | 0.10 | 0 | 0.16 | 0 | 0 | 0.08 |

The minorities sit where the literature puts them: Quthing 7.05% Xhosa and 2.94% Phuthi (432
respondents), Qacha's Nek 8.61% Xhosa (256), Mohale's Hoek 1.05% Xhosa and 0.55% Phuthi.

## Mapping calls

- **Sethepu -> Xhosa.** Sethepu (also Seqhotsa) is the Sesotho name for isiXhosa as spoken by
  Lesotho's Thembu-descended Xhosa of Quthing and Qacha's Nek (worldatlas "Languages of Lesotho";
  linguistlist lgpolicy 2011-02). A name for the language, not a different one, so merged as a
  spelling-variant case.
- **Sephuthi -> new node `nigercongo.bantu.nguni.phuthi`.** Glottolog phut1246, classified under
  Nguni. Hand colour 0.47 0.10 60 (#834b14), darker than Xhosa and far from Sesotho's pale yellow,
  since all three meet in Quthing.
- **English at R7's mother tongue**, nationally (section below; it was drawn as the pooled
  home-language answers, 0.95% of Maseru).
- "Other"/"Others" on `other`.

## Population, join and checks

2016 census district counts from religiondots' `ls_lookup.csv`, asserted 10 districts summing to
2,007,201. REGION -> district is religiondots' `sources/ls.py` NORM (Butha-Buthe has three
spellings, Mohale's Hoek and Qacha's Nek two apostrophes; `gkey` strips both). `unit_counts`
asserts every label maps and every district has respondents; largest-remainder rounding keeps
each district at its census figure. `check_country.py ls`: ok.

## English at R7's mother tongue (2026-10-05, session edd42a8c-r7e)

Anita ruled on ask 018: lingua francas at Afrobarometer R7's **mother tongue** question (Q2A).
`ls_afro.py` now reads Q2A for R7 (`extract(..., r7q="Q2A")` in `mono_afro.py`) and draws
English at its R7 Q2A national share in every district (`unit_counts(..., lf=["English"])`):
4 English answers of 1,200 are too few to place by district. Every other answer keeps its
pooled per-district share, scaled to what English leaves.

| | before (pooled home language) | after (R7 mother tongue) |
|---|---|---|
| English | 0.47% (9,500), 0.95% of Maseru | 0.29% (5,800), the same share in every district |

Sesotho 98.18% -> 98.31%, Xhosa 0.94% -> 0.98%, Phuthi 0.32% -> 0.33%.

## Calls someone might reverse

- Interviews were in Sesotho or English, which probably undercounts Xhosa and Phuthi; drawn as
  measured, said in `note_public`.
- Sethepu merged into Xhosa rather than given its own node.

## Scatter

2,004 dots over 1,835 hexes.
