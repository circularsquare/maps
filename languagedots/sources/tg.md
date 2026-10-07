# Togo: Afrobarometer R5-R7 first-language answers by region, on 2022 census populations

Drawn 2026-10-05 (session edd42a8c-wafr). 8,095,498 people (RGPH-5 2022), 6 units, 24 answers,
every row `modelled`. 8,083 dots.

```
python sources/wafr_afro.py tg    # respondents from religiondots' Afrobarometer .sav (read-only)
python sources/tg_afro.py
python taxonomy/build.py
python tools/check_country.py tg
python scatter.py --country tg
```

## 1. What exists

- **No published census table.** RGPH-4 (2010) asked ethnicity (IPUMS `ETHNICTG`; IPUMS account
  blocked, and IPUMS has no Togo language variable). INSEED published no ethnic or language
  table from 2010 or RGPH-5 (2022) (coverage sweep, 2026-10-03). CLEAR Global has no Togo
  language dataset on HDX (`togo-languages` 404, 2026-10-05). DHS/MICS gated.
- So AGENT_BRIEF section 2's survey case: **Afrobarometer** R5 (2012), R6 (2014), R7 (2017),
  ~1,200 adults each, all six survey regions, which are religiondots' six units (Lomé commune
  = "Lomé (Golfe 1 to 5)", Maritime = the rest). R5-R6 asked "language of respondent", R7 Q2A
  "mother tongue": the first-language reading of ask 018. R8-R9 ask "language spoken in home".

## 2. Checks and the lingua-franca question

Ethnic group x language (R5-R7, unweighted): retention is high for every group: Ewe 95%, Kabiyè
90%, Moba 96%, Tem 93%, Mina 74% (rest Ewe), Ouatchi 70% (rest Ewe and Mina), Ikposo 83%,
Nawdm 84%. The survey asks ethnicity and language on the same card, so those who name another
language are drawn on it: no further retention move.

National %, by reading (the switch is `FIRST_ROUNDS` in `sources/tg_afro.py`):

| | R5-R7 drawn | R5-R7 French as given | R8-R9 home |
|---|---|---|---|
| Ewe | 33.7 | 33.4 | 35.3 |
| Kabiyè | 13.3 | 12.9 | 12.9 |
| Moba | 9.1 | 9.0 | 10.1 |
| Tem | 6.2 | 6.0 | 6.3 |
| Mina | 5.7 | 5.6 | 5.2 |
| Ouatchi | 3.6 | 3.5 | 1.6 |
| Nawdm | 2.9 | 2.8 | 1.2 |
| French | 0.02 | 1.7 | 4.2 |

R7 itself shows the effect: 245 of 1,200 give a home language different from their mother
tongue, mostly Kabiyè, Ouatchi, Ikposo, Ifè and Nawdm speakers naming Ewe, Mina or French.
French: 63 answers in R5-R7, 59 from French-language interviews; 62 moved to the respondent's
ethnic group's language (`MOVE_FRENCH`, Nigeria's rule).

Drawn per unit (top shares): Lomé Ewe 56%, Mina 23%, Kabiyè 6%; Maritime Ewe 61%, Ouatchi 9%,
Mina 8%; Plateaux Ewe 35%, Aja 12%, Ikposo 12%, Ifè 11%; Centrale Kabiyè 37%, Tem 28%, other
13%; Kara Kabiyè 39%, Tem 13%, Lama 12%, Ntcham 11%; Savanes Moba 58%, Gourma 22%, Ngangam 8%.

### French at R7's mother tongue (2026-10-05, ask 018)

Anita's ruling: lingua francas at Afrobarometer R7's separate **mother tongue** question (Q2A).
French answers are still moved to the ethnic language first (`MOVE_FRENCH`), then
`french_at_r7` (`FRENCH_AT_R7`) sets each region's French share to R7's Q2A share, shrunk to the
national 1.52% by 50 respondents (`wafr_afro.r7_mother`, `shrink`); the region's other answers
scale to what is left. 20 of R7's 1,200 named French as mother tongue, all in French
interviews. French 1.7k (0.02%) -> 131k (1.6%): Maritime 2.9%, Kara 1.9%, Lomé 1.2%, Savanes
1.1%, Plateaux 0.5%, Centrale 0.4%. Ewe 33.7 -> 33.0%, Kabiyè 13.3 -> 13.1%.

## 3. Calls someone might reverse

- R5-R7 over R5-R9 (one line).
- French at R7's mother-tongue share (Anita's ruling; the shrink is mine).
- "Tchamba" on Akaselem; "Ngam-gam" a new Gur leaf; the card's "Other" (13% of Centrale) on
  `africa_other`; one "Aklobo" answer folded into it.
- Region grain only, though R6, R7 and R9 name the prefecture: religiondots has region polygons
  only and ~10-30 respondents per prefecture.

## 4. Room for improvement

The 2010 census ethnicity by prefecture (in IPUMS, or an INSEED table if one appears) with this
survey's retention would give 39 prefectures instead of six regions; Togo's north is a mosaic
(Kara region's Kabiyè, Lama, Tem, Ntcham, Konkomba) that six units flatten.

## Terms

Afrobarometer: free download, citation requested. INSEED population quoted via religiondots.
Glottolog CC BY; Kontur CC BY 4.0.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Ewe, Mina (Gen), Ouatchi (Waci), Aja and Fon are in a Gbe group with Benin's Gbe languages; Ifè is in 'Yoruba and Ede'. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
