# Niue: the record

Drawn 2026-10-19 (session eca0185b-terr, "coverage gaps"). 1,564 usual residents, one unit, 3
nodes, no dots and 3 rings at 1:1000 (the island holds under one dot of any language). Every row
`derived` (multi-answer sharing). Scripts: `sources/nu_census.py`, `taxonomy/nu2022.py`,
`taxonomy/tree.d/nu.txt`, `countries/nu.py`. Geography is religiondots' `nu_hexes.gpkg` (one unit
`NU`, 91 hexes), read in place. No asks.

## 1. Source and question

Niue Statistics Office, 2022 Niue Census of Population and Housing Report
(`data/raw/nu/census2022.pdf`; niuestatistics.nu/download/35/census/2793/...). The household form
has a question on the languages spoken in the household (report p. 79, several may be named), and
every resident aged 5 and over is asked how well they speak, read, write and understand Vagahau
Niue (Table 3.8, 13 to 16). The household table is the home-language one: "Main language spoken in
Household" for private occupied dwellings by village (PDF p. 70, below Table 19, numbered Table 20
in the PDF): Vagahau Niue 395, English 328, Other languages 65, in 513 households (rows overlap:
788 mentions). The 2022 census counted 1,681 people, 1,564 of them usual residents (117 visitors).

Ability for contrast (Table 3.8, residents aged 4+): 69.7% proficient in speaking Vagahau Niue
(1,014 people), 22.5% basic, 7.8% cannot; down from 86.4% in 2006. Ability is not a home language,
so it is not drawn.

## 2. Drawing

Mentions shared across the 1,564 usual residents (spec §3.6): Vagahau Niue 784 (50.1%), English 651
(41.6%), other languages 129 (8.2%). The household table is the only language table by place; its
mentions cannot be turned into people without knowing how many households named two. The share
of English is therefore a share of mentions. "Other languages" is unnamed: `other` (the ethnicity
table shows Tuvaluan, Samoan, Tongan and Fijian residents among 411 non-Niuean).

By village (Vagahau Niue, English, other, households): Alofi South 75/99/22/139, Alofi North
35/36/11/50, Makefu 20/20/3/23, Tuapa 26/13/1/30, Namukulu 4/1/1/4, Hikutavake 13/13/0/13, Toi
10/6/0/12, Mutalau 29/21/1/30, Lakepa 26/12/0/30, Liku 24/13/2/27, Hakupu 49/36/6/56, Vaiea
17/17/14/17, Avatele 34/25/0/39, Tamakautoga 33/16/4/43. Alofi South and Vaiea lean towards English
and other; the island is drawn as one unit because 1,564 people give under two dots at 1:1000 and
no village boundary layer is in religiondots.

## 3. Checks (`sources/nu_census.py`)

The 14 villages sum to the printed Total row (395, 328, 65, 513); 513 equals Table 19's household
total; 1,564 usual residents is the figure in the report's text (pp. 10, 13).

## 4. Geography

religiondots' layer: Kontur NU, 91 hexes, 2,479 people against 1,564 (ratio 1.59; the widest in
this tier, noted in religiondots' `countries/nu.py`). It only weights where the rings sit.
