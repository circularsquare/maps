# Falkland Islands: the record

Drawn 2026-10-19 (session eca0185b-terr, "coverage gaps"). 3,198 usual residents (2016 census), 3
locations, 32 nodes, 2 dots and 29 rings at 1:1000. Every row `derived`. Scripts:
`sources/fk_census.py`, `taxonomy/fk2016.py`, `taxonomy/tree.d/fk.txt`, `countries/fk.py`.
Geography is religiondots' `fk_hexes.gpkg` (Stanley, Camp, Mount Pleasant Complex), read in place.

## 1. Source

Falkland Islands Government, *Falkland Islands Census 2016, Full Report* (religiondots' copy,
`religiondots/data/raw/terr/Falkland_Islands_Census_2016_-_Full_Report.pdf`, read in place).
Table 5a (annex p. xi, PDF p. 76), "Population by language other than English, sex and location":
31 languages by Stanley, Camp and MPC. The report says (pp. 7, 31): English is the first language
for 85% of the population; 496 people speak a language other than English in their homes; 31
foreign languages, Spanish 325, Shona 73, Philippine languages 64. 86% of those 496 speak English
"very well" or "well". Table 4.2 groups the same 561 mentions; the script checks Table 5a against
it. The 561 mentions are more than the 496 people: about 65 named two languages.

## 2. Calls

- Mentions are scaled by 496/561 at every location (the report gives no people-by-location), so
  the people with another language are 412 in Stanley, 49 in Camp and 35 at MPC; everyone else is
  an English first-language speaker. All rows `derived`.
- "Castellano" is Spanish; "Pilipino" in the text is Filipino 31 + Tagalog 22 + Bisaya/Cebuano 11,
  each its own node. "Gaelic" (2) sits on the Celtic group, "Chinese" (2) on Sinitic, "French
  Arabic" (1) on Arabic. "Sri Lankan" and "Swiss" are nationalities, on `other` (1 each).
- Why 2016 and not 2021: the 2021 census counted 3,662 usual residents (Stanley 2,538, Camp about
  604 with MPC), and its report is on the Falkland Islands Government census page
  (falklands.gov.fk/policy/2021-census/census, "2021 Census Report" and a tables workbook). This
  session's machine does not resolve falklands.gov.fk and the National Archives mirror is behind a
  Cloudflare check or returns 404, so the 2021 tables were not read; whether the 2021 census
  asked language at home is not known. A 2021 language table would replace this build.

## 3. Geography

religiondots' layer, three units: Stanley 50 hexes (3,570 Kontur people), MPC 75 (567), Camp 1,240
(2,921; Kontur puts about 7.7x the census count in Camp, so the Camp dots lean towards where Kontur
guesses people are; Camp draws no dot at 1:1000 either way). Stanley's 2,046 English speakers make the 2 dots; every other row is
a ring.
