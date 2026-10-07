# Eswatini (sz): record

Drawn 2026-10-05 (session edd42a8c-mono). Afrobarometer R5-R9 (2011-2022) home language, pooled,
6,000 respondents, shares per region times the 2017 census region counts; every row `modelled`.
1,093,238 people on 4 regions: siSwati 96.09%, English 2.94%, Zulu 0.62%, other 0.20%, Xitsonga
0.12%, Portuguese 0.02%. Placed on religiondots' WorldPop cells (read-only).

Files: `sources/sz_afro.py` (uses `sources/mono_afro.py`), `taxonomy/sz2022.py`,
`taxonomy/tree.d/sz.txt` (bare repeats), `countries/sz.py`, `data/normalized/sz.csv`.

## Why a survey

The 2007 questionnaire (unstats SWZ2007en.pdf) P15 asks literacy only; 2017's Volume 3 has no
language table (scout 2026-10-05). The Afrobarometer asks home language, one answer: §2's survey
route. Swaziland is in R5-R9 (not R4).

## The survey's answers (weighted %)

| answer | R5 | R6 | R7 | R8 | R9 | pooled |
|---|---|---|---|---|---|---|
| siSwati (R7 "Siswati") | 97.78 | 97.42 | 97.47 | 96.33 | 91.29 | 96.05 |
| English | 0.73 | 1.39 | 1.09 | 3.14 | 8.36 | 2.94 |
| Zulu (R5 "Isizulu") | 1.07 | 0 | 1.14 | 0.54 | 0.35 | 0.62 |
| Other | 0 | 0.87 | 0.13 | 0 | 0 | 0.20 |
| Shangaan | 0.22 | 0.32 | 0 | 0 | 0 | 0.11 |
| Portuguese | 0.12 | 0 | 0 | 0 | 0 | 0.02 |
| Tonga | 0.07 | 0 | 0 | 0 | 0 | 0.01 |
| Refused | 0 | 0 | 0.18 | 0 | 0 | 0.04 (dropped) |

Zulu is strongest in Shiselweni (2.45%, 25 of 1,096), on the KwaZulu-Natal border, as expected.

## Mapping calls

- **Shangaan -> Xitsonga**; **Tonga** (one R5 respondent, Hhohho) merged with it as "Thonga", the
  older name for the same people. No source counts a Zambian/Malawian Tonga community here.
- **English drawn as measured.** The question is home language, not ability, so §2's
  learned-language rule does not fold it. But R9's 8.36% against ~1% in R5-R7 looks like a change
  in how answers were taken rather than in households; the pooled 2.94% is said in `note_public`
  to be the least certain figure. Using R5-R7 only would put English near 1%.
- "Other" on `other`; refusals dropped (not drawn).

## English at R7's mother tongue (2026-10-05, session edd42a8c-r7e)

Anita ruled on ask 018: lingua francas at Afrobarometer R7's **mother tongue** question (Q2A).
`sz_afro.py` now reads Q2A for R7 (`extract(..., r7q="Q2A")` in `mono_afro.py`) and draws
English at its R7 Q2A national share, 0.556%, in every region (`unit_counts(..., lf=
["English"])`; 7 answers of 1,200, too few by region). Every other answer keeps its pooled
per-region share, scaled to what English leaves. This also settles the R9 jump above.

| | before (pooled home language) | after (R7 mother tongue) |
|---|---|---|
| English | 2.94% (32,100) | 0.56% (6,100) |
| siSwati | 96.09% | 98.52% |

Zulu 0.62% -> 0.56% (R7's Q2A Zulu now counts in the pool instead of its Q2B one).

## Population, join and checks

2017 census region totals (Volume 3, Table 5.2.2) hard-coded in `sz_afro.py` and asserted equal
to the `Total` rows of religiondots' `data/normalized/sz.csv`; 1,093,238 in all. REGION names
match the four regions exactly. Largest-remainder rounding keeps each region at its census
figure. `check_country.py sz`: ok. The placement layer's own `pop` is WorldPop, which religiondots
re-weights per region; here it only places people inside each region, so its region totals do not
matter.

## Calls someone might reverse

- English at R7's mother tongue, 0.56% (section above); it was 2.94% pooled, R9's jump included.
- Region grain: the survey has no finer usable geography for Eswatini.

## Scatter

1,091 dots over 1,090 cells.
