# Togo: MICS6 2017 language groups by region, split by Afrobarometer R5-R7, on 2022 census populations

**Since 2026-10-09 (session 32a047f0) Togo is drawn from MICS6 2017 microdata, with
Afrobarometer splitting MICS's groups** (§0). Sections 1-4 are the Afrobarometer-only build,
drawn 2026-10-05, still the source of the split (`sources/tg_afro.py`, now writing
`data/normalized/tg_afro.csv`).

```
python sources/wafr_afro.py tg    # Afrobarometer respondents (once)
python sources/tg_mics.py         # -> data/normalized/tg.csv
python tools/check_country.py tg
python scatter.py --country tg
```

## 0. MICS6 2017 (`sources/tg_mics.py`, `taxonomy/tg2017.py`, `countries/tg.py`)

Anita made a UNICEF MICS account and downloaded Togo's MICS6 (2017; INSEED, UNICEF). SPSS files
in `data/raw/tg/mics_2017/` (gitignored: research use, no redistribution; the readme asks that
copies of reports and publications go to INSEED and UNICEF Togo). 8,404 households sampled,
7,916 interviewed, 34,988 members, 420 clusters, 60 in each of seven strata (HH7).

**Item: HC1B**, mother tongue of the household head, read as every member's (hl.sav x
hhweight). Eleven answers: EWE/MINA, KABYE, MOBA-GOURMA, KOTOKOLI/TEM, BASSAR/KONKOMBA,
AKPOSSO/AKEBOU, IFE/ANA, TCHOKOSSI, AUTRES LANGUES NATIONALES, LANGUES ETRANGERES, FRANCAIS.
Cross-checks: HH16 (the respondent's mother tongue) equals HC1B in 96.4% of households and is
within 2.4 points of it for every group in every unit (largest: Lomé EWE/MINA 71.0 vs 73.4).
Unlike Iraq it does not slide to the interview language: 2,203 interviews were in French and 24
of those respondents named French; of 909 non-Ewe/Mina heads interviewed in Ewe/Mina, 76
respondents (8%) named Ewe/Mina, probably partly spouses. WM14 and MWM14 (women and men 15-49)
are printed and agree within a few points (Savanes "other national" 8% women, 17% men). HC1B
used, as in Iraq: the standard item, and the household reading.

**How it enters: MICS sets each group's share per region; Afrobarometer splits each group.**
MICS's eleven groups cannot replace Afrobarometer's 24 answers, and Afrobarometer cannot
outweigh MICS on the groups: MICS samples every household member, children and foreign
residents included (Afrobarometer samples adult citizens, which is why it finds almost no
foreign languages and MICS finds 4.7%), it is all 2017, and it has 60 clusters and about 4,400
persons in every region against Afrobarometer's 370 to 1,070 adults over 2012-2017. Pooling
would need MICS's groups as the common categories, where MICS dominates anyway. Inside each
group, the languages Afrobarometer files under it share MICS's figure in their own proportions
in that region (French answers moved to the ethnic language first, as before). Where fewer
than 5 Afrobarometer respondents in a region named any of a two-language group's languages,
the group takes Afrobarometer's national split (12 cases, all under 7% of their region:
e.g. Lomé BASSAR/KONKOMBA, Plateaux MOBA-GOURMA); a sparse "other national" would go on
`africa_other` (none happened).

**Groups.** EWE/MINA holds every Gbe answer (Ewe, Mina, Ouatchi, Aja, Fon): 2,854 of 2,935
Adja-Ewe-ethnic heads answer EWE/MINA and 26 "other national", and Maritime, where
Afrobarometer has 9% Ouatchi, has 5 of 1,096 households on "other national". Each Afrobarometer
language MICS does not name sits under AUTRES LANGUES NATIONALES: Nawdm, Lama, Akaselem
(Tchamba), Fulfulde, Hausa, Yoruba (Afrobarometer's respondents are citizens), the card's own
"Other", and Ngangam. Ngangam is a Gurma language, but MOBA-GOURMA does not name it and Savanes
fits that reading: Afrobarometer Moba + Gourma 79%, unnamed rest 13%; MICS MOBA-GOURMA 67%,
"other national" 12%. KABYE, KOTOKOLI/TEM, IFE/ANA, TCHOKOSSI and FRANCAIS are one node each.
No lumped answer is split by anything but Afrobarometer, and no new node was needed.

**LANGUES ETRANGERES** (308 heads, 4.7% of persons; 9.8% of Lomé, 2.6-4.9% elsewhere): one
label, "Foreign language", on `africa_other`. 236 of the 308 heads give "other nationalities"
as ethnic group (31 Adja-Ewe), 179 are Muslim, 103 live in rural households spread over every
region (40 clusters of 60 in Lomé, 14-27 elsewhere): residents from the neighbouring countries,
whose languages MICS does not name and Afrobarometer, a survey of citizens, does not reach.
Burkina Faso's "Autre langue africaine" is on the same node (taxonomy/bf2006.py).

**French is drawn at Afrobarometer R8-R9's home-language share** (`FRENCH = "home"` in
`tg_mics.py`; Anita's standing rule, 2026-10-09: the map leans towards the language spoken at
home). French is the one answer where mother tongue and home language part ways in Togo: MICS's
mother-tongue item gives 0.32% (24 heads), Afrobarometer R7's mother-tongue question (ask 018's
figure, drawn until this change) 1.52%, and R8-R9 (2021-22 and 2024, "language spoken in
home", 2,399 respondents) 4.29%. So R8-R9 it is: each unit's weighted share, shrunk to the
national share by 50 respondents (`french_home()`; 240 to 736 respondents per unit, so the
shrink is small), and each unit's other rows give up the difference pro rata. `FRENCH = "r7"`
or `"mics"` draws the other two readings.

| French, % of unit | Centrale | Kara | Maritime | Lomé | Plateaux | Savanes |
|---|---|---|---|---|---|---|
| MICS HC1B (mother tongue) | 0.20 | 0.61 | 0.29 | 0.50 | 0.38 | 0.01 |
| Afrobarometer R7 (mother tongue, drawn before) | 0.43 | 1.90 | 2.93 | 1.22 | 0.49 | 1.08 |
| Afrobarometer R8-R9 raw (home) | 3.0 | 4.7 | 5.5 | 6.3 | 2.8 | 1.7 |
| drawn (R8-R9, shrunk) | 3.20 | 4.63 | 5.45 | 6.08 | 2.95 | 2.16 |

National 4.22% drawn (341,934 people), against 1.62% (131,249) before.

**Units.** MICS's Lomé Commune is religiondots' "Lomé (Golfe 1 to 5)" (drawn on COD-AB's Lome
Commune); its Golfe Urbain (Lomé's suburbs in Golfe prefecture) and Maritime strata pool, with
their weights, into the rest of Maritime (120 clusters). Asserted: all seven strata map, all six
units are covered. Populations unchanged (2022 census, 8,095,498); largest remainder.

**Zeros in a cluster sample.** 60 clusters per region miss a group living in 2% of a region's
clusters 30% of the time, 5% about 5% of the time. No zero matters here: groups MICS misses in a
region are small Afrobarometer answers anyway, and the unnamed languages are split by
Afrobarometer. The big movement rests on many clusters: Savanes TCHOKOSSI is 110 households in
13 clusters (9 with 8 or more households, Oti prefecture around Mango), where Afrobarometer had
13 respondents.

Before (Afrobarometer, 2026-10-05) -> after, % of each unit:

| | before | after |
|---|---|---|
| Lomé | Ewe 55.7, Mina 22.7, Kabiyè 6.2, Ouatchi 3.9, Tem 3.4, French 1.2, foreign 0 | Ewe 44.6, Mina 18.2, foreign 9.2, French 6.1, Tem 4.1, Kabiyè 3.7, Ouatchi 3.1 |
| Maritime | Ewe 59.0, Ouatchi 9.2, Mina 8.2, Kabiyè 6.4, French 2.9, Tem 1.9 | Ewe 55.2, Ouatchi 8.6, Mina 7.7, French 5.4, Kabiyè 5.1, foreign 4.7, Tem 4.4 |
| Plateaux | Ewe 35.2, Aja 11.8, Ikposo 11.8, Ifè 11.5, Kabiyè 8.4, Akebu 6.2 | Ewe 27.1, Kabiyè 16.9, Ifè 9.5, Aja 9.1, Ikposo 6.3, Tem 4.8, Akebu 3.3, French 2.9 |
| Centrale | Kabiyè 36.5, Tem 28.1, Other 12.6, Nawdm 5.5, Lama 4.2, Tchamba 3.6 | Kabiyè 37.6, Tem 36.6, Ifè 3.3, French 3.2, Ntcham 3.0, Other 1.9, Nawdm 0.8 |
| Kara | Kabiyè 38.6, Tem 12.6, Lama 11.4, Ntcham 10.9, Konkomba 7.4, Nawdm 7.4 | Kabiyè 30.6, Ntcham 17.5, Konkomba 12.0, Lama 9.9, Tem 7.1, Nawdm 6.4, French 4.6 |
| Savanes | Moba 57.2, Gourma 21.6, Ngangam 7.5, Other 3.6, Anufo 2.5 | Moba 47.7, Gourma 18.0, Anufo 14.8, Ngangam 7.1, foreign 3.8, Other 3.4, French 2.2 |

National: Ewe 33.0 -> 28.8%, Kabiyè 13.1 -> 13.0, Moba 9.0 -> 8.7, Tem 6.1 -> 7.4, Mina 5.5 ->
4.8, foreign 0 -> 4.5, French 1.6 -> 4.2, Ouatchi 3.5 -> 3.2, Ntcham 2.0 -> 3.0, Anufo 0.5 ->
2.6, Ikposo 3.0 -> 1.9, Nawdm 2.9 -> 1.7, Other 2.7 -> 1.4. 8,083 dots.

Calls someone might reverse: group shares from MICS alone rather than pooled with
Afrobarometer; Ngangam under "other national" rather than MOBA-GOURMA; French at R8-R9's
home-language 4.2% (Anita's rule) rather than a mother-tongue reading (MICS 0.3%, R7 1.5%);
foreign languages on `africa_other` rather than `other` (Anita, 2026-10-09: keep).

## The Afrobarometer-only build (2026-10-05 to 2026-10-09; now the split inside MICS's groups)

Drawn 2026-10-05 (session edd42a8c-wafr). 8,095,498 people (RGPH-5 2022), 6 units, 24 answers,
every row `modelled`. 8,083 dots.

```
python sources/wafr_afro.py tg    # respondents from religiondots' Afrobarometer .sav (read-only)
python sources/tg_afro.py         # now -> data/normalized/tg_afro.csv (comparison only)
python taxonomy/build.py
python tools/check_country.py tg
python scatter.py --country tg
```

## 1. What exists

- **No published census table.** RGPH-4 (2010) asked ethnicity (IPUMS `ETHNICTG`; IPUMS account
  blocked, and IPUMS has no Togo language variable). INSEED published no ethnic or language
  table from 2010 or RGPH-5 (2022) (coverage sweep, 2026-10-03). CLEAR Global has no Togo
  language dataset on HDX (`togo-languages` 404, 2026-10-05). DHS/MICS gated (MICS6 2017
  since obtained and used, §0).
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

Afrobarometer: free download, citation requested. MICS6 2017: UNICEF account, research use, no
redistribution of the files; only shares are written. INSEED population quoted via religiondots.
Glottolog CC BY; Kontur CC BY 4.0.

## Cross-border groups (2026-10-06, 5d7dac7e-xb)

Ewe, Mina (Gen), Ouatchi (Waci), Aja and Fon are in a Gbe group with Benin's Gbe languages; Ifè is in 'Yoruba and Ede'. Groups only (taxonomy/regroup.txt): no label's node or count changed, each keeps its leaf and colour; dots rewritten in place. The full table of cases is in followups.md (2026-10-06, languages that stop at a border).
