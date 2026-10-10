# Guyana (gy): record

**Since 2026-10-09 (session 32a047f0) each region's Amerindian retention comes from MICS6
2019-20 microdata** (§0), replacing the flat 20% of §2. Sections 1-3 are the first build.

## 0. MICS6 2019-20 as retention (`sources/gy_mics.py`, `taxonomy/gy2019.py`, `countries/gy.py`)

Data: Guyana MICS6 2019-20 SPSS files from mics.unicef.org (Anita's UNICEF account), in
`data/raw/gy/mics_2019/` (gitignored; research use, no redistribution). 8,285 households, 7,072
interviewed, 26,209 members, 10 regions (HH7 = the census regions = religiondots' units 1-10).

**Item: HC1B, language of the household head** (English / Indigenous language / Spanish /
Portuguese / Other), read as every member's, person-weighted. HH16 (respondent's native
language) slides to English: of 264 households with an indigenous-language head, the respondent
said English in 190 and indigenous in 73. Every questionnaire was English (HH14) and 7,063 of
7,072 interviews (HH15). WM14 (women 15-49) has only English / Other and puts 193 of 216 women
in indigenous-headed households on English. So HC1B, the high reading, as in Iraq.

**How MICS enters: as retention.** The census counts Amerindians per region exactly; MICS's
Amerindian sample per region is 17 to 435 households. The census groups stay, and MICS gives
each region's share of Amerindian-headed persons whose head's language is indigenous (also
Spanish, Portuguese, other); everyone else gets the region's shares among non-Amerindian-headed
persons (99.7% English, read as Creole as before); white Guyanese stay on English. Region and
national totals are unchanged (746,955). Rows carrying MICS shares are `modelled`, the Creole and
English rows `derived`.

**Interviewers.** HC1B depends on who asked. In the same clusters some interviewers record an
indigenous language for most Amerindian heads and others for none: Potaro-Siparuni cluster 358,
interviewer 834 6 of 6, 833 0 of 6 (833: 0 of 33 Amerindian households overall); South
Rupununi clusters 382-395, 915/916 90 of 104, 912 4 of 53 in the same villages. The script
compares each interviewer with the rate their cluster-mates recorded in the same clusters and
drops five (912, 922, 926 in Region 9; 833, 815 in Region 8; expected >= 3, Poisson P < 0.05),
then rebuilds each cluster's rate from the rest, averaged over clusters by their Amerindian
population. Raw -> checked: Region 8 29.9 -> 36.9%, Region 9 27.1 -> 37.2%; elsewhere
unchanged. North Rupununi (clusters 364-377, one team) stays low: only interviewer 924 there
recorded any (6 of 47, in two villages), while the team's other four recorded 0 of 174, so the
north-south gap (Makushi vs Wapishana country) may be partly the team. It cannot be separated.

**Shrinkage.** Each region's rate is pulled towards its pool (interior: Regions 7, 8, 9; coast
and north-west: the rest) with the number of clusters holding Amerindian households as the
sample size and 10 as the prior weight, since households in a village answer alike. It matters
only for the thin coastal regions (Essequibo Islands-West Demerara: 1 indigenous household of
20, 17.2% raw, 11.8% drawn; Mahaica-Berbice 12.5 -> 7.0%).

**Other language with an Amerindian head** (19 households; 12 of them in three Upper Mazaruni
clusters, 319, 322, 323, beside that region's indigenous answers, 4 with an indigenous-speaking
respondent) is read as an unnamed indigenous language, also on `americas_other`. With any other
head "other" stays on `other` (7 households, 3 of them Chinese, in East Berbice).

| Amerindian language drawn | first build (flat 20%) | now | % of region now |
|---|---:|---:|---:|
| 1 Barima-Waini | 3,569 | 454 | 1.6 |
| 2 Pomeroon-Supenaam | 1,767 | 97 | 0.2 |
| 4 Demerara-Mahaica | 1,413 | 112 | 0.04 |
| 7 Cuyuni-Mazaruni | 1,367 | 3,132 | 17.0 |
| 8 Potaro-Siparuni | 1,602 | 3,295 | 29.7 |
| 9 Upper Takutu-Upper Essequibo | 4,162 | 8,334 | 34.4 |
| Guyana | 15,698 | 16,004 | 2.1 |

Retention drawn (indigenous only): Regions 7, 8, 9 about 37% of Amerindians, Barima-Waini 2.5%,
Pomeroon-Supenaam 1.1%, coast 1-12%; nationally 18.5% (raw 15.0%), beside the IDB 2013
survey's 20% of households fluent. So the national total barely moves but the geography does:
the north-west (Warao, Carib, Arawak) falls from 3,569 to 454 and the interior doubles.

National now: Guyanese Creole 728,556 (97.5%, was 730,842), Amerindian language 16,004, Spanish
1,308, English 415, Portuguese 382, other 290. 745 dots, 3 rings.

**Spanish and Portuguese** are drawn, being measured, but are 1-2 dots: Spanish 16 households
(mostly coastal East Indian and Mixed heads in Regions 3-5, plausibly returnees from Venezuela),
Portuguese 9 (Regions 7-9, mining country near Brazil). Per-region shares rest on 1-3 households
each. The survey's frame and the 2012 population base both predate the Venezuelan arrivals after
2018, so those are probably undercounted; nothing here measures them.

**Zeros.** Pomeroon-Supenaam has 111 Amerindian households in 23 clusters and no indigenous
head. That says retention there is low, not nil: a speaking village holding a few percent of the
region's 8,834 Amerindians would be missed by that sample more often than not. The shrinkage
draws 1.1%.

Calls someone might reverse: HC1B over HH16; dropping the five interviewers (without it,
national retention is 15.0% and Regions 8, 9 about 28%); Amerindian-headed "other" on
`americas_other`; drawing the 1-2 Spanish and Portuguese dots.

Still open: the indigenous languages stay unnamed (MICS has one code); placement inside a region
is by population, so Region 9's Wapishana south and Makushi north are not told apart.

Drawn 2026-10-05 (session edd42a8c-amer). Census 2012 ethnic background by region, read as
language (AGENT_BRIEF §2 ethnicity rule). 746,955 people on 10 regions, every row `derived`.
Guyanese Creole 97.8%, Amerindian language (unnamed) 15,698, English 415. 745 dots, 1 ring.

Files: `sources/gy_census.py`, `taxonomy/gy2012.py`, `taxonomy/tree.d/gy.txt` (bare repeats),
`countries/gy.py`, `data/normalized/gy.csv`.

## 1. Source

Bureau of Statistics, 2012 Census Compendium 2, Table 2.3 (ethnic background x region), from
religiondots' copy of the PDF (read-only), typed into the script. Includes prorated "not stated"
(321) and no-contact persons (16,331). Checks: rows and columns sum to the printed totals
(746,955); region totals equal religiondots' gy_lookup census column. No language question
and no language table in the compendium. Birthplace by region not used (foreign-born about
1.5% in 2012; Venezuelan arrivals came after).

## 2. Calls

- **Amerindian 78,492: 20% on `americas_other`, 80% on Guyanese Creole.** Retention from the
  IDB's "Guyana's Indigenous Peoples 2013 Survey: Final Report" (Bollers, Clarke, Johnny, Wenner,
  2019, doi 10.18235/0001591, pp. 71-72): 337 households in 11 villages, "only 20% of households
  were fluent in their own language", fluency higher further from the capital. Read through
  Wikipedia's citation; the IDB PDF answered 403 to curl and is not on Wayback, so the page was
  not seen directly. A flat 20% understates the Rupununi (Regions 8, 9) and overstates the coast.
  The census has one Amerindian category, so the language is unnamed (`americas_other`, the
  narrowest node holding Arawakan, Cariban and Warao).
- **White on English** (as bb, tt); African, East Indian, Mixed, Portuguese, Chinese, Other on
  Guyanese Creole (creo1235). Caribbean Hindustani (cari1275) not drawn: no figure.
- Berbice and Skepi Creole Dutch are extinct; not drawn.

## 3. Room for improvement

The IDB report's village table (pp. 71-72) would give a hinterland/coast split; a Guyana MICS
(2014, 2019-20) mother-tongue item, if it exists, would give regional retention (done
2026-10-09 with MICS6 2019-20, §0; MICS5 2014 not tried); a census with Amerindian nations would
let the languages be named.
