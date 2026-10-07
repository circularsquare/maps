# Burundi (bi): record

Drawn 2026-10-05 (session edd42a8c-mono); population base replaced the same day (session
edd42a8c-mono2, see "Population base" below). Afrobarometer R5 (2012) and R6 (2014) home
language, pooled, 2,400 respondents, shares per province times current province populations;
every row `modelled`. 12,332,788 people on 17 provinces: Kirundi 99.34%, Swahili 0.53%, French
0.13%. Placed on religiondots' Kontur hexes (read-only).

## Population base (2026-10-05, edd42a8c-mono2)

First built on the 2008 RGPH's 8,053,574 people, about 4M short of today. Now:

- **National total**: the 2024 RGPHAE preliminary count, 12,332,788 (5,901,069 men, 6,431,719
  women), released 27 March 2025 (reported by ADIP-Burundi; UNFPA Burundi on the launch). Its
  province figures are on the 2025 five-province map (Bujumbura 3,353,555, Butanyerera
  2,530,206, Gitega 2,278,215, Burunga 2,118,551, Buhumuza 2,052,261), which does not nest in
  the 2008 provinces, so it supplies only the total.
- **Split between the 17 provinces of 2008**: OCHA COD-PS 2022 commune projections (HDX
  `cod-ps-bdi`, `bdi_admpop_2022_adm2_v3.csv` in `data/raw/bi/`, 119 communes, 13,462,695 people).
  Communes summed onto their 2008 province by name against religiondots' `bi_communes.csv`
  (four spelling variants mapped in code; every non-Mairie commune asserted to fall in its own
  province); Rumonge province's five communes go back to Bururi (Burambi, Buyengero, Rumonge) and
  Bujumbura Rural (Bugarama, Muhuta), asserted; the three Mairie communes onto Bujumbura Mairie.
  Shares scaled to the 2024 total by largest remainder (asserted to sum exactly).
- Growth since 2008 per province runs x1.32 (Muramvya) to x1.71 (Makamba), and x2.06 for
  Bujumbura Mairie. The survey shares are unchanged; only the base moved.

Call someone might reverse: the projection (13.46M) is 9% above the census count; scaling to the
census assumes the overshoot is even across provinces. Placement inside a province still follows
religiondots' hexes, weighted to 2008 commune counts.

Files: `sources/bi_afro.py` (uses `sources/mono_afro.py`, shared with ls, sz), `taxonomy/bi2014.py`,
`taxonomy/tree.d/bi.txt` (bare repeats), `countries/bi.py`, `data/normalized/bi.csv`.

## Why a survey

The 2008 RGPH questionnaire (unstats BDI2008fr.pdf) asks only "quelle langue sait lire et
écrire" (literacy); the 2024 RGPHAE has published preliminary totals only (scout 2026-10-05). The
Afrobarometer asks home language with one answer, so this is AGENT_BRIEF §2's survey route. Only
R5 and R6 surveyed Burundi.

## The survey's answers

| answer | R5 | R6 | pooled | where |
|---|---|---|---|---|
| Kirundi | 99.46 | 99.27 | 99.37 | everywhere |
| Swahili (R5 "Kiswahili") | 0.25 | 0.73 | 0.48 | Bujumbura Mairie 8 of 176, Cibitoke 2, Bururi 1 |
| French | 0.29 | 0.00 | 0.15 | one each in Kayanza, Mwaro, Ngozi |

Bujumbura Mairie comes out 4.05% Swahili, which fits Swahili's known place as the language of
the city's Muslim quarters (Buyenzi, Bwiza); Rumonge, the other expected Swahili area, is inside
2008's Bururi, where one respondent named it. French is drawn as measured: the question is home
language, not ability, so §2's learned-language rule does not fold it.

## Lingua francas and ask 018 (2026-10-05, session edd42a8c-r7e)

Anita ruled that lingua francas are drawn at Afrobarometer R7's mother-tongue question (Q2A).
Burundi is not in R7 (only R5 and R6 surveyed it; R7's 34 countries checked), so there is no
mother-tongue answer to draw from. Swahili (0.53%) and French (0.13%) stay at the pooled R5/R6
"language of respondent" answers, the first-language wording of ask 018. Nothing redrawn.

## Population, join and checks

Units are religiondots' 17 provinces of 2008 (`bi_lookup.csv`, asserted 17 rows summing to the
2008 RGPH's 8,053,574); populations as in "Population base". REGION labels map onto units through religiondots' `sources/bi.py` NORM (Bujumbura =
Bujumbura Rural; Mairie/Marie, Cankuzo/Cankuza, Ruyigi/Ruyiga spellings); `unit_counts` asserts
every label maps and every province has respondents. Counts rounded by largest remainder, so each
province sums to its census figure. `check_country.py bi`: ok.

## Calls someone might reverse

- (Superseded) the 2008 census as the base; see "Population base".
- Small languages rest on one to eight respondents per province; Mwaro's 1.5% French is one
  person. Kept because it is what was measured; a national-share fallback would be the
  alternative.

## Room for improvement

R5/R6 microdata carry no commune for Burundi usable here; a 2024 RGPHAE language table (if the
full results ever add one) would replace this.

## Scatter

12,332 dots over 9,954 hexes (was 8,052 on the 2008 base).

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Kirundi is in a Rwanda-Rundi group with Kinyarwanda, Ha and Hangaza. Their colours are still far apart. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
