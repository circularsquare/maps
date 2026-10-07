# Malawi: 2018 census tribe, read as home language through Afrobarometer

Built 2026-10-05 (session edd42a8c-mw). 17,506,538 people (all Malawians), 32 units, 26
answers, every row `modelled`. Tier D (ethnicity only), built under AGENT_BRIEF §2's
2026-10-05 ruling with the retention check. Modelled on Ghana (`sources/gh.md`), simpler.

| | |
|---|---|
| census | NSO 2018 PHC Main Report, **Table E4** "Malawian Population by Tribe, Region and District" (pp. 132-133): 12 tribes + Other, Malawians only, 28 districts + 4 cities. PDF is religiondots' `data/raw/mw/mw_phc2018_main_report.pdf` (read-only; `cms.nsomalawi.mw/api/download/270/`) |
| survey | Afrobarometer R5, R7, R8, R9 (2012-2022), Malawi: 5,965 respondents with a tribe and a home language; R7 and R9 name a district (2,382). Religiondots' merged .sav files, read-only |
| geography | religiondots' 32 district units and Kontur 400 m hexes (`mw_hexes.gpkg`, `mw_lookup.csv`), read-only; population weight |
| scripts | `sources/mw_afro.py --fetch` (extract, all six rounds), `sources/mw_census.py`; `taxonomy/mw2018.py`, `taxonomy/tree.d/mw.txt`, `countries/mw.py` |

## 1. What was searched for a real language table

- **2018 PHC**: no language question (the literacy item says "in any language"). Tribe only.
  The 32 district reports carry no finer tribe table that was found; Traditional Authority
  level not seen.
- **1998 PHC** asked "the language most commonly used in this household". Published national
  figures (as quoted from the census report): Chichewa about 70%, Chiyao 10.1%, Chitumbuka
  9.5%, every other language under 3% including Chilomwe. A district table would exist only in
  IPUMS microdata (MW1998A_LANGUAG; IPUMS account blocked) or the 1998 report, not found online.
  Used here as the check in §3.
- **2008 PHC**: tribe again, no language (IPUMS ETHNICMW).

## 2. Census checks (Table E4)

Read as a token stream (area name, then 14 figures). The 13 tribes sum to Total on all 36
rows; each region's districts sum to the region row on all 14 columns; the three regions sum
to Malawi, 17,506,538. The four cities are peers of their districts, as in religiondots.
Table 3.5 (national, p. 20) differs slightly: 17,506,022 total; Tumbuka 1,614,955 vs E4's
1,614,577, Sukwa 93,762 vs 93,456, Other 186,319 vs 187,522. E4 is used throughout since it
is the district table and internally consistent. Non-Malawians (57,211 = 17,563,749 -
17,506,538) are not in the table and not drawn.

## 3. Retention check, and why rounds 4 and 6 are out

Each district's count of a tribe is shared over P(home language | tribe), estimated nationally,
then per region (3), then per district, each shrunk to the level above with K = 30 weighted
respondents (`sources/mw_census.py`).

Rounds 4 (2008) and 6 (2014) record the tribe's own language far more often than the others:

| tribe -> own language | R4 | R5 | R6 | R7 | R8 | R9 |
|---|---|---|---|---|---|---|
| Lomwe -> Chilomwe | 77% | 3% | 67% | 29% | 9% | 5% |
| Ngoni -> Chingoni | 65% | 12% | 52% | 8% | 9% | 6% |
| Mang'anja -> Chimang'anja | 76% | 28% | 82% | 41% | 23% | 13% |
| Chichewa, all respondents | 47% | 71% | 51% | 66% | 74% | 73% |

The question labels do not explain it ("language of respondent" in R4-R6, "language spoken
in home" R7-R9; R5 behaves like the later rounds). The 1998 census sides with R5, R7-R9: with
them the model gives **Chichewa 69.6%, Chitumbuka 9.7%, Chiyao 8.4%, Chilomwe 2.9%, Chisena
2.1%, Chitonga 1.5%**, against 1998's ~70 / 9.5 / 10.1 / <3 / 2.7 / 1.7. All six rounds pooled
gave Chichewa 59% and Chilomwe 8.4% (1.5M), which nothing corroborates. So R4 and R6 are left
out; the cost is district detail (R4 and R6 held 60% of the district-coded respondents).

Retention with R5, R7-R9 (national, census-weighted): Chewa 97% Chichewa; Tumbuka 87% own
(10% Chichewa); Yao 61% (38% Chichewa); Sena 53% (45% Chichewa); Tonga 56% (24% Tumbuka, 19%
Chichewa); Nkhonde 72% (15% Tumbuka); Lambya 61% (27% Tumbuka); Sukwa 73% (n = 10); Mang'anja
29% Chimang'anja, 65% Chichewa; Lomwe 15% (82% Chichewa); Ngoni 9% Chingoni (86% Chichewa, 4%
Tumbuka); Nyanja 21% Chinyanja, 79% Chichewa (n = 17). The Other tribe (95) is mostly Ndali,
Tumbuka, Chichewa, Nyiha.

Checks: every district's languages sum to its E4 total; the drawn total is 17,506,538; 32
units join religiondots' lookup both ways; every survey district name is a census district
and agrees with the respondent's region.

## 3a. Chichewa and English at R7's mother tongue (2026-10-05, session edd42a8c-r7e)

Anita ruled on ask 018: lingua francas at Afrobarometer R7's **mother tongue** question (Q2A).
`sources/mw_afro.py` now extracts Q2A for R7 (not Q2B, "language spoken in home"). In
`sources/mw_census.py`, Chichewa (for every tribe but the Chewa) and English take their shares
from R7 alone (`LF_ROUNDS`; national -> region -> district with K = 30, the national share
shrunk with K0 = 10 to R7's rate over all non-Chewa); every other answer comes from R5, R7-R9
among the non-lingua-franca answers, scaled to what is left. R7 Q2A gives Chichewa 50.2%
nationally (Q2B: 66%), which is R4/R6's level (47-51%), not R5/R8/R9's.

| drawn | before (R5, R7-R9 home) | after (R7 mother tongue) |
|---|---|---|
| Chichewa | 12.19M (69.6%) | 8.77M (50.1%) |
| English | 25,700 (0.15%) | 0 (no R7 respondent named it) |
| Chilomwe | 507,000 (2.9%) | 2.22M (12.7%) |
| Chiyao | 8.4% | 10.9% |
| Chingoni | 1.0% | 3.7% |
| Chimang'anja | 1.4% | 3.2% |

Lomwe now 67% Chilomwe (was 15%), Ngoni 36% Chingoni (9%), Yao 78% Chiyao (61%), Mang'anja
66% Chimang'anja (29%). The 1998 census check in §3 no longer holds: its item was "the
language most commonly used in this household", a use question, and R7's mother tongue sides
with R4/R6 instead. §3's reason for leaving R4 and R6 out (they disagreed with 1998) is
weaker now; they are still out, which costs district detail.

## 4. Calls

- **Chichewa, Chinyanja, Chimang'anja are three nodes** as respondents named them, though
  Glottolog has one language with Chewa and Mang'anja as dialects. The note says so.
- **Chingoni split by region**: Glottolog's Ngoni (Nyanja) ngon1270 for Central and Southern
  answers (zm.txt's `nyanja_sena.ngoni`), Ngoni (Tumbuka) ngon1272 for Northern answers (new
  `tumbuka.ngoni_tumbuka`). Region decides, per AGENT_BRIEF §3's place-dependent label rule.
- **Malawi Lomwe and Malawi Sena get their own nodes** (Glottolog mala1256, mala1475), apart
  from mz.txt's Mozambique ones. Tonga (Nyasa) tong1321 under Tumbuka, per Glottolog.
- **Nkhonde and Nyakyusa kept apart** (the survey names both; Glottolog: dialects of one
  language); Nkhonde and Sukwa hang from Bantu beside tz.txt's Nyakyusa and Ndali.
- R6 codes Chewa as "Chewu" (936 respondents, 98% Chichewa, no "Chewa" code that round): read
  as Chewa. Survey tribes the census lacks (Senga, Ndali, Khokhola, Wiza, Chikunda, Other)
  feed the census's Other; verbatims naming Nyanja or Nkhonde go to those tribes.
- Two Sena respondents answering "Chisenga" in the lower Shire are read as Chisena (Senga is
  spoken in the far north and Zambia).
- English at R7's mother tongue (§3a): none. "Other" with no verbatim on `other`.
- Lambya (bare in zm.txt) coloured here; hand colours for the new nodes, tree.d/mw.txt says why.

## 5. Room for improvement

- **Likoma**: its 10,649 Nyanja are drawn 79% Chichewa from 17 Nyanja respondents elsewhere
  (none on Likoma); the island's own speech is usually called Chinyanja.
- Small languages the survey rarely met (Sukwa, Kokola, Nyungwe) are drawn from a handful of
  answers (cut from `gap`, 2026-10-06).
- The district level rests on R7 and R9 only (2,382 respondents), so most small tribes take
  their region's shares.
- A real table would fix this: the 1998 census household-language item by district (IPUMS or
  the 1998 report), or a 2018 microdata cross of tribe by district with any language item.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Lomwe (Malawi) is in a Lomwe group with Mozambique's Elomwe, Sena (Malawi) in a Sena group with Mozambique's Cisena, Nkhonde under Nyakyusa (Glottolog's Ngonde dialect of Nyakyusa-Ngonde). Chewa, Nyanja and Mang'anja were already in the Nyanja group. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
