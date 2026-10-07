# Laos (la)

Drawn 2026-10-05 by session edd42a8c-la. Ethnicity read as language under AGENT_BRIEF section 2
(Anita, 2026-10-05). No retention correction: no published source gives one (below). Rows for
Lao, Hmong, Iu Mien and the residual are `derived`; every other group is `modelled`, because its
village counts come from splitting a census category.

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
  per-group share. **No retention correction is applied**; minorities are drawn wholly on their
  own languages.

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
- No retention correction (above).
- Colours hand-picked: Phu Thai (beside Lao in Savannakhet), Katang, Lamet and Prai (were close
  to Khmu's magenta).

## Room for improvement

- A per-group table below national (LSB's district tables, or IPUMS's 2005/2015 samples, gated
  and the account blocked) would replace the 2011-based split.
- A retention source: LSIS II's HH16 native language by ethnicity from the microdata would give
  the share of each group that is Lao-native, if someone with a MICS account tabulates it.
