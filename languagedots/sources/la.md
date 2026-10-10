# Laos (la)

Drawn 2026-10-05 by session edd42a8c-la. Ethnicity read as language under AGENT_BRIEF section 2
(Anita, 2026-10-05). **Since 2026-10-09 (session 32a047f0) a retention correction from LSIS III
(MICS6) 2023 microdata moves part of each non-Tai group onto Lao (§0).** Rows for Lao and the
residual are `derived`; every other row is `modelled` (a split census category, a survey share,
or both). The sections after §0 describe the census build, which is unchanged underneath.

## 0. Retention from LSIS III 2023 (`sources/la_mics.py`, 2026-10-09)

Anita made a UNICEF MICS account and downloaded the Lao Social Indicator Survey III 2023 (MICS6;
LSB, UNICEF), SPSS files in `data/raw/la/mics_2023/` (gitignored; research use, no
redistribution). 20,993 households, 20,325 interviewed, 18 provinces (HH7 codes = the census's
PCODE: per province the non-Tai share of persons, MICS 2023 against census 2015, correlates 0.993,
largest gap Xaysomboun 79.9 / 70.6). HC2 is the head's ethnic group in the census's own 49-group
order (plus "Bu", 93 households in 2 provinces, not in the census and unused). There is no HC1B.

**HH16 cannot be used.** It is "native language of the respondent", Lao / other, and it follows
the interview language (HH15):

| HH15 interview | HH16 Lao | HH16 other |
|---|---:|---:|
| Lao | 18,269 | 1,713 |
| other | 5 | 338 |

Hmong-headed households interviewed in Lao without a translator answer Lao 90% of the time (of
1,556); those interviewed in another language 1% (of 331). So HH16 calls 86% of Hmong, 91% of
Khmu, 94% of Katang and 100% of Makong (Bru), Phong, Xingmoun and Oy Lao-native. WM14 and MWM14
(women's and men's own native language) are the same: Hmong 83% and 88%. The respondent was the
head in 54% of households and the spouse in 32%; HH16 differs between them by a few points
(Hmong 89% Lao where the head answered, 83% otherwise), so mixed households are not the
explanation.

**FL7 is used instead**: "language the child speaks most of the time at home", Lao / other, asked
of one child 7-14 per household with children (fs.sav, 7,308 answers). It is about someone else
and about use, and it disagrees with HH16 where HH16 is least believable: of 2,067 children who
speak another language at home, 1,524 have a caretaker recorded Lao-native (FS14).

| group | children | clusters | HH16 % Lao | FL7 % Lao | drawn % Lao |
|---|---:|---:|---:|---:|---:|
| Khmu | 970 | 225 | 90.8 | 32.8 | 37.0 |
| Hmong | 772 | 163 | 85.8 | 19.7 | 16.5 |
| Akha | 239 | 39 | 57.3 | 19.4 | 10.8 |
| Makong (Bru) | 102 | 25 | 100.0 | 54.2 | 61.2 |
| Katang | 98 | 17 | 94.4 | 36.9 | 35.9 |
| Katu | 92 | 20 | 25.7 | 41.5 | 35.0 |
| Trieng | 92 | 24 | 56.0 | 61.6 | 58.8 |
| Harak (Alak) | 85 | 20 | 74.3 | 52.4 | 54.1 |
| Pounoy (Phunoi) | 79 | 26 | 89.1 | 87.9 | 57.5 |
| Xuay (Kuy) | 56 | 16 | 92.8 | 82.5 | 60.2 |
| Yrou (Laven) | 40 | 10 | 91.9 | 81.9 | 77.7 |
| Ewmien | 41 | 11 | 94.7 | 32.9 | 26.3 |

(HH16 as persons in hhweight; FL7 in fsweight; drawn = after the model below, as a share of the
group's 2015 census people. Every group's row is in `data/normalized/la_mics_shares.csv`.)

**FL7 has its own trap: interviewers.** Among minority children in clusters where their census
category heads 90% or more of households, the 54 interviewers with 10+ such children record from
0% to 100% Lao at home (7 at 0%, 6 at 80% or more; interviewer 213 in Sekong, 43 children in 17
clusters, all Lao). In the 201 clusters where two such interviewers each saw 2+ of these
children, their shares differ by 41 points on average. So the model carries an interviewer
effect (sd 1.23 logits) and draws a typical interviewer's answer.

**The model** (children of the non-Tai groups, fsweight): logit P(Lao at home) = family +
group + b x own + interviewer, where own is the child's census category's share of its cluster's
households. Families (Mon-Khmer, Hmong-Mien, Sino-Tibetan, MICS's own grouping) are free; group
terms are shrunk towards the family (ridge 3, so a thin group leans on its family and an
unsampled one is its family); interviewers are shrunk like a random effect (ridge 1). No province
term: each team works in one province, so province and interviewer cannot be told apart. The
slope is -2.53: shift is strongest where a group is a small minority (Lao at home 66% where the
category heads under a quarter of the cluster, 23% where it heads all of it). For the 22 groups
with 20+ children, the intercept is then re-solved on the census villages so the group's national
total equals the model's share over its MICS children (census villages measure own on 2015
persons, MICS clusters on 2023 sampled households, and they differ: Akha 0.95 in MICS against
0.83 on the census). So **MICS sets each group's total and the slope only decides which villages
hold its Lao speakers**. In every village round(people x p) move to a row "Lao-speaking <group>",
drawn as Lao, remainders by largest remainder per group and province; village totals unchanged.

**Not moved:** Lao; the Tai groups (Tai, Phu Thai, Lue, Nhuan, Yang, Saek, Tai Nua), because the
answer list is only Lao / other and a Phu Thai or Lue speaker may well call their language Lao
(FL7 gives them about 90% "Lao", Nhuan 64%); "Other, not stated and foreigners".

**Before / after** (% of the 6,481,482 people in the village file):

| | Lao | Khmu | Hmong | Makong | Katang | Akha | Phunoi |
|---|---:|---:|---:|---:|---:|---:|---:|
| Laos | 52.8 -> 65.4 | 10.9 -> 6.9 | 9.2 -> 7.7 | 2.5 -> 1.0 | 2.2 -> 1.4 | 1.7 -> 1.6 | 0.6 -> 0.3 |
| Vientiane Capital | 91.6 -> 94.0 | 1.4 -> 0.4 | 3.4 -> 2.5 | | | | |
| Phongsaly | | 18.8 -> 12.1 | | | | 21.0 -> 19.1 | 18.5 -> 8.3 |
| Oudomxay | 12.1 -> 35.6 | 59.1 -> 40.2 | 14.9 -> 12.1 | | | | |
| Luang Prabang | 30.1 -> 50.7 | 46.9 -> 30.3 | 17.7 -> 14.2 | | | | |
| Xiengkhuang | 35.3 -> 46.1 | 8.7 -> 4.9 | 42.1 -> 36.0 | | | | |
| Savannakhet | 60.3 -> 72.0 | | | 11.3 -> 4.5 | 9.1 -> 6.0 | | |
| Sekong | 15.1 -> 54.3 | | | | | | |
| Attapeu | 37.5 -> 68.0 | | | | | | |

812,550 people move (Khmu 262,057, Hmong 98,144, Makong 99,991, Katang 51,831, Laven 43,849).
Sekong's Katu go 23.9 -> 15.6%, Trieng 21.3 -> 8.3%, Alak 13.2 -> 6.2%.

**How much to trust it.** The direction is not in doubt; the size is soft. (1) FL7 measures
children 7-14, who have shifted further than their parents, so drawing their share for everyone
overstates Lao. (2) Interviewer effects are large; the typical interviewer may still lean towards
Lao (the answer's code reads "reading test available: Lao"). (3) Small groups rest on a handful of
clusters (Tri 2 children, Nhaheun 1 cluster) and lean on their family. `RETENTION = False` in
`countries/la.py` draws the census groups unmoved, as before. A cluster sample's zero matters less
here than in Iraq: no group is drawn from a zero; unsampled groups take their family's model.

Checks asserted in the script: file sizes; HH16 follows HH15; the crosswalk (every province
within 15 points, r > 0.9); slope inside -3.5..-1; calibration to 1e-6; every village's total
unchanged; national Lao inside 52-70%. check_country: ok, 51 languages, 8,499 units. Scatter:
6,455 dots, 9 rings, no equal-share fallbacks.

## Sources

| | |
|---|---|
| village counts | 2015 Population and Housing Census, ten ethno-linguistic categories as % of village population, 8,499 villages, on the K4D atlas server: `https://gis.cde.unibe.ch/gis/rest/services/Decide/laos_2015_pcnt_population_ethno_linguistic_category_<cat>/MapServer/0` (lao, tai_thai, khmuic, palaungic, katuic, bahnaric_khmer, vietic, hmong, mien, tibeto_burman) and `laos_2015_total_population`. Licence field: "With proper citation of the data sources, the data can be used freely." Same channel and village roster as religiondots' Laos. |
| national groups | 2015 PHC report (LSB, 2016), Table P2.7, Lao citizens by sex and ethnicity, 49 groups, pdf p.121-122. `https://lao.unfpa.org/sites/default/files/pub-pdf/PHC-ENG-FNAL-WEB_0.pdf` (`data/raw/la/phc2015_report.pdf`) |
| within-category split | 2011 Lao Census of Agriculture on the same server: `laos_2011_main_ethnicity_in_the_village_code`, `_second_most_numerous_`, `_third_most_numerous_` (codes 1-49 in P2.7's order) and `laos_2011_pcnt_of_agricultural_hhs_of_the_main_ethnicity` |
| geography | religiondots' 400m Kontur hex layer for the same 8,499 villages (`RD_GEO/la/la_hexes.gpkg`, unit `LA-<VCODE>`), read-only |

Scripts: `sources/la_census.py` (-> `data/raw/la/*.json`, `data/normalized/la.csv`),
`taxonomy/la2015.py`, `taxonomy/tree.d/la.txt`, `countries/la.py`.

## What was searched

- **Finest table naming the 49 groups**: national only (P2.7). The report's appendix has no
  ethnicity by province; K4D has no 2015 per-group layer (its 1,365 services were listed and
  grepped; 2015 and 2005 carry only the ten categories and four families). No district table was
  found on the web. So: village categories, split as below.
- **Language question**: the 2015 census has none (the report never mentions a language item).
- **Retention**: LSIS II 2017 (MICS6/DHS FR356, `data/raw/la/lsis2017_FR356.pdf`) asks
  "native language of the respondent" (HH16, WM14) but only as Lao / other, and the report
  tabulates neither it nor home language by ethnicity (its learning-environment table drops the
  MICS "teacher uses home language" column). The microdata needs a MICS account with identity:
  not used. The World Bank STEP 2012 survey is registration-gated. No published study gives a
  per-group share. **No retention correction was applied** in the 2026-10-05 build; superseded
  by §0 (LSIS III 2023 microdata).

## Method

1. Each village's ten category counts = percentage x village population. The percentages carry
   eight significant digits, so the counts are recovered exactly: 6 misses in 84,990 cells, all in
   two villages (1403002, already known from religiondots as LSB's own denominator slip, and
   1501016), each off by a fraction of a person. The residual (population less the ten) is
   "other, not stated and foreigners", 115,924.
2. **The categories cut the 49 groups differently from P2.7 in two places, found by reconciling
   national totals** (village file vs P2.7 summed by category):

   | category | villages | P2.7 | |
   |---|---|---|---|
   | Lao | 2,835,458 | 3,427,665 | Lao 592k short |
   | Tai-Thai | 1,185,651 | 597,524 | 588k over: census "Lao" in Tai-Thai |
   | Khmuic | 770,488 | 779,144 | 8.7k short |
   | Vietic | 15,260 | 6,374 | 8.9k over: census "Phong" in Vietic |
   | Palaungic | 28,175 | 28,172 | with Bid (2,372) counted Palaungic |
   | Katuic | 505,481 | 505,394 | |
   | Bahnaric-Khmer | 208,186 | 208,141 | |
   | Hmong / Mien / Tibeto-Burman | 594,830 / 32,399 / 189,630 | 595,028 / 32,400 / 189,551 | |
   | other | 115,924 | 77,297 | + foreigners |

   So census Lao may sit in Lao or Tai-Thai (the 49-group list folds Phuan, Yoy, Kaleung and
   other Tai subgroups into Lao; K4D's categories evidently come from the finer subgroup codes),
   and census Phong in Khmuic or Vietic (Glottolog has both a Khmuic Phong-Kniang `phon1246` and
   a Vietic Phong dialect of Hung `phon1243`). Bid is Palaungic (Glottolog `bitt1240`).
   The service named `distribution_of_ethno_linguistic_category_mon_khmer` is mislabelled: it
   holds Bahnaric-Khmer only (208,188).
3. Within each village and category, people go to the village's 2011 main/second/third groups
   that the category can hold (main weighted by its share of farm households, the rest 2:1);
   cells with none borrow the district's mix for that category, else the province's, else P2.7.
   5,204,965 people are seeded from their own village, 602,339 from the district, 428,088 from
   the province, 130,166 from the nation.
4. Rake (IPF) per pool of categories (Lao+Tai-Thai, Khmuic+Vietic, each other one alone) so each
   group's national total is P2.7's x (village file / P2.7 for the pool, 0.999-1.0004) while
   every village's category count stays the census's. Converged to 1e-5; integer rounding by
   largest remainder per village category.

## Checks (asserted or printed by the script)

- 8,499 villages on every 2015 service, same roster; 18 provinces' worth of queries.
- Integer recovery as above; Tai-Thai, Mien and Sino-Tibetan count services agree with the
  percentages in every village.
- Every (village, category) adds back to the census; la.csv sums to 6,481,482 = the village
  file (10,746 short of the national 6,492,228: villages the K4D file lacks, as in religiondots).
- Group totals after the split are within 1% of P2.7 for every group above 5,000 people (worst Thai Neua, 14,009 against 14,148; the pool scale and rounding).
- 2011-2015 village join: 8,011 VCODEs in both; 1 differs in district; 575 have dissimilar
  romanised names (renumbered or merged villages). Not filtered: a wrong 2011 village in the
  same district seeds about what the district fallback would.
- check_country: ok, 51 languages, 8,499 units. Scatter: 6,455 dots, 6 rings, no equal-share
  fallbacks.

## Result

Lao 52.8%, Khmu 10.9%, Hmong 9.2%, Phu Thai 3.4%, Tai 3.1%, Bru (Makong) 2.5%, Katang 2.2%,
Tai Lue 1.9%, Akha 1.7%, other 1.8%.

## Calls someone might reverse

- **The split of categories into groups on 2011 village lists, raked to national totals.**
  Village category counts and national group totals are the census's; only which village of a
  category holds which group is modelled. Not filed as an ask: it moves people only between
  groups of one category within a village, against measured national totals. The alternative
  was drawing seven of the ten categories as washed-out group nodes.
- **Lao in the Tai-Thai category drawn as Lao** (about 590k), since the census calls them Lao.
- **Phong split into two leaves** by the census's own category.
- **"Tai" on Vietnam's `kradai.tai_vietnam`** (Tai Dam and Tai Don); Lao "Tai" also includes Tai
  Daeng. **Ewmien on `hmongmien.dao`** (includes the Lanten/Kim Mun). **Yang on `kradai.giay`**
  (Nhang). **Nhoaun on Northern Thai**. **Hayi on Hani**. **Hor on Southwestern Mandarin**.
- **Other, not stated and foreigners on `other`**, drawn, because the village file does not
  separate them (P2.7: 77,297 other/not stated; P2.8: 45,538 foreigners).
- Retention (§0, 2026-10-09): FL7 children's home language rather than HH16; drawn for all ages;
  interviewer effects modelled; Tai groups not moved.
- Colours hand-picked: Phu Thai (beside Lao in Savannakhet), Katang, Lamet and Prai (were close
  to Khmu's magenta).

## Room for improvement

- A per-group table below national (LSB's district tables, or IPUMS's 2005/2015 samples, gated
  and the account blocked) would replace the 2011-based split.
- Retention for adults: LSIS III's native-language items (HH16, WM14, MWM14) follow the interview
  language, so the only usable item is children's (§0). A survey asking adults' first language
  with the groups' languages as answers would fix the overstatement. LSIS II 2017's FL module
  (if its microdata has FL7) could be pooled to thin the interviewer noise.
