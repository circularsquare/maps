# Liberia: Afrobarometer R4-R7 first-language answers by county, on 2022 census populations

Drawn 2026-10-05 (session edd42a8c-wafr). 5,250,187 people (2022 PHC), 15 counties, 19 answers,
every row `modelled`. 5,241 dots.

```
python sources/wafr_afro.py lr    # respondents from religiondots' Afrobarometer .sav (read-only)
python sources/lr_afro.py
python taxonomy/build.py
python tools/check_country.py lr
python scatter.py --country lr
```

## 1. What exists

- **Census ethnicity is national only.** 2008 *Final Results* (Wayback copy of lisgis.net,
  `data/raw/lr/nphc_2008_final_report.pdf`, 352 pp), Table 4.4 (pp. 115-117), ethnic
  affiliation by age and sex, Liberia only. religiondots read all 92 pages of the 2022 *Final
  Results* and the 15 thematic reports: social items are national only there too. No language
  question in either. IPUMS has ETHNICLR (account blocked), no language variable. The 2011
  Census Atlas (`lisgis_2008_loc.pdf`) has nothing on ethnicity. lisgis.gov.lr is now an SPA;
  its `/api/publications` lists no census report.
- So the survey case: **Afrobarometer** R4 (2008), R5 (2012), R6 (2015), R7 (2018), ~1,200
  each, all 15 counties; R4-R6 "language of respondent", R7 Q2A "mother tongue".

## 2. Checks

National %, drawn (R4-R7, English moved) / 2008 census ethnicity (Table 4.4, of 3,476,608):
Kpelle 19.3 / 20.3; Bassa 12.1 / 13.4; Grebo 10.4 / 10.0; Mano 7.5 / 7.9; Gio 7.4 / 8.0; Kru
7.1 / 6.0; Lorma 6.9 / 5.1; Gola 5.2 / 4.4; Vai 4.9 / 4.0; Kissi 4.8 / 4.8; Krahn 4.4 / 4.0;
Mandingo 2.6 / 3.2; Gbandi 2.5 / 3.0; Mende 1.8 / 1.3; Belle 1.2 / 0.8; Dei 0.7 / 0.3; **Sapo
0.16 / 1.25** (Sinoe's Sapo barely sampled). Other Liberian 0.6, other African 1.4, non-African
0.1 in the census have no counterpart here. Not raked to the census: the margins agree within
about two points everywhere but Sapo, and the census is 14 years older than the populations.

Retention (ethnic group x answer, R4-R7): 83-95% name their group's language; nearly all the
rest name English (Kpelle 57 of 960, Bassa 46 of 629, Grebo 47 of 455, Lorma 35 of 336).

## 3. English, the lingua franca (ask 018)

| | R4-R7 drawn | R4-R7 English as given | R8-R9 home |
|---|---|---|---|
| (Liberian) English | 0.8 | 8.5 | 39.8 |
| Kpelle | 19.3 | 17.9 | 14.8 |
| Bassa | 12.1 | 11.0 | 6.9 |

369 English answers in R4-R7 (299 interviewed in English, 69 in Liberian English, the only
interview languages); 337 moved to the respondent's ethnic group's language (`MOVE_ENGLISH`),
the 32 with no ethnic language (English, national identity, other) drawn as Liberian English.
R7 shows the gap directly: 601 of 1,200 give a home language different from their mother tongue,
almost all English. Under ask 018's other reading Liberia would be about 40% English.

**2026-10-05, ask 018 closed.** Anita's ruling: lingua francas at Afrobarometer R7's separate
**mother tongue** question (Q2A). English answers are still moved first (`MOVE_ENGLISH`), then
`english_at_r7` (`ENGLISH_AT_R7`) sets each county's Liberian English share to R7's Q2A share
("English" + "Liberian English", 10 of 1,200), shrunk to the national 0.79% by 50 respondents
(`wafr_afro.r7_mother`, `shrink`); the county's other answers scale to what is left. Liberian
English 40k (0.77%) -> 47k (0.89%): Grand Gedeh 1.7%, Montserrado 1.4% (was 1.9%), Lofa 1.3%,
Sinoe 1.1%, elsewhere 0.2-0.6% (was 0 in most counties). Every other language moves by under
0.1 point.

Drawn per county (top shares): Bomi Gola 75%; Bong Kpelle 69%; Gbarpolu Kpelle 75%, Gola 15%;
Grand Bassa Bassa 65%; Grand Cape Mount Vai 51%, Gola 20%, Mende 18%; Grand Gedeh Krahn 65%;
Grand Kru Kru 46%, Grebo 46%; Lofa Lorma 32%, Kissi 30%, Gbandi 15%; Margibi Kpelle 47%, Bassa
18%; Maryland Grebo 76%; Montserrado Kpelle 18%, Bassa 13%, Grebo 10%; Nimba Gio 46%, Mano 42%;
River Gee Grebo 83%; River Cess Bassa 88%; Sinoe Kru 54%, Grebo 19%.

## 4. Calls someone might reverse

- R4-R7 over R4-R9 (one line); English at R7's mother-tongue share (Anita's ruling; the shrink
  is mine).
- Unshrunk county shares: Grand Kru (64 respondents), River Gee and River Cess (96) are noisy.
- Mandingo on Maninka; Grebo one leaf; Gola under Atlantic.

## 5. Room for improvement

A county-level ethnic table from either census (or IPUMS 2008 microdata) would replace the
survey's county pattern with a count and lift Sapo to its real size.

## Terms

Afrobarometer: free download, citation requested. LISGIS publications quoted. Glottolog CC BY;
Kontur CC BY 4.0.
