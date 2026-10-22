# Gibraltar: the record

Drawn 2026-10-19 (session eca0185b-terr, "coverage gaps"). 31,958 people aged 14 and over (the
census's universe), one unit, 12 nodes, 28 dots and 11 rings at 1:1000. Scripts:
`sources/gi_census.py`, `taxonomy/gi2022.py`, `taxonomy/tree.d/gi.txt`, `countries/gi.py`.
Geography is religiondots' `gi_hexes.gpkg` (one unit `GI`, 19 hexes), read in place. No asks.

## 1. Source and question

Census of Gibraltar 2022 (HM Government of Gibraltar; religiondots' copy,
`religiondots/data/raw/gi/census_of_gibraltar_2022_report.pdf`, read in place). Q12: "What is your
main language?", options "English" and "Other (including British Sign Language)" with the language
written in, asked of people aged 14 and over (the report p. 34 says the question was new in
2022). Those not answering English were asked their fluency in English.

Table 51 (PDF p. 191), main language by nationality group and sex: English 27,984, Spanish 1,784,
Arabic 806, Other 1,384 (including 4 BSL users), total 31,958 aged 14+. Table 52 (p. 192) gives the
same by age group; its male and female totals equal Table 51's (checked). The report text (p. 55):
English 87.6%, Spanish 5.6%, Arabic 2.5%, BSL 0.01%; 12.4% (3,974) not English; 64.0% of them
fluent in English. Figure 11 shows ten languages other than English as bars; only the first two
are printed as numbers.

## 2. Calls

- The eight small bars (Portuguese, Polish, French, German, Hebrew, Russian, Chinese, Filipino) are
  read off the chart: the figure is an embedded image of 497 x 533 pixels, measured against its
  gridlines (0 to 6%). The same measurement gives Spanish 5.59% (tabled 5.58%) and Arabic 2.52%
  (tabled 2.52%), so the scale is right; the eight bars come to Portuguese 0.43%, Polish 0.30,
  French 0.26, German 0.22, Hebrew 0.19, Russian 0.18, Chinese 0.17, Filipino 0.15, rounded to 10
  people each (140, 90, 80, 70, 60, 60, 50, 50). These rows are `derived`; English, Spanish, Arabic
  and BSL are `measured`. The remaining 780 of "Other" are unnamed (`other`).
- "Arabic" is drawn as Moroccan Arabic (Darija). Of the 806, 576 are Gibraltarian and 160
  Moroccan nationals; Gibraltar's Arabic speakers are the Moroccan community and its descendants.
  Knowledge, not a table; reversing it moves 806 people to the Arabic group node.
- "Chinese" on the Sinitic group (variety not named); Hebrew on its node; BSL on `signlanguage.bsl`.
- Llanito (the local English-Spanish speech) was not an option on the form.
- Universe: aged 14 and over only. The 5,978 under 14 (of 37,936 usual residents) were not asked and
  are the gap, as Guam's under-5s are. Drawing them as the adults would add 16%.
- One unit: Table 55 gives main language for seven residential areas, but religiondots found the
  areas are defined only as lists of enumeration areas with no map (`religiondots/sources/gi.md`).

## 3. Geography

religiondots' layer: Kontur GI, 19 hexes, 32,285 people against 37,936 counted. The census total
includes under-14s; the table's universe does not.

## 4. Room for improvement

A numbers table for the eight chart languages (the Government Statistics Office may hold one); a
map of the seven residential areas, so Table 55's languages by area could be placed.
