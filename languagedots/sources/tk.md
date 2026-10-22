# Tokelau: the record

Drawn 2026-10-19 (session eca0185b-terr, "coverage gaps"). 1,062 people aged 5 and over present on
census night, the three atolls as one unit, 6 nodes, no dots and 6 rings at 1:1000 (the whole
country is a little over one dot). Every row `measured`. Scripts: `sources/tk_census.py`,
`sources/tk_geo.py`, `taxonomy/tk2016.py`, `taxonomy/tree.d/tk.txt`, `countries/tk.py`. No asks.

## 1. Source and question

Tokelau National Statistics Office and Statistics New Zealand, *2016 Tokelau Census of Population
and Dwellings*, social profile tables (religiondots' copy,
`religiondots/data/raw/tk/tk2016_tables_social_profile.xlsx`, read in place), and the profile report
(`tk2016_profile_report.pdf`, ch. 5). The 2016 census extended its language questions: languages in
which the person can hold a conversation (several allowed; Table 5.4), other languages used
regularly (5.6), the language first learnt and still understood (5.7), and the language spoken
most often at home (5.5). Table 5.5, aged 5 and over present on census night, one answer each:
Tokelauan 877, English 90, Tuvaluan 54, Samoan 21, Other language 16, Kiribati 4, None (too young to
talk) 1, Not stated 6; total 1,069. It is by whether the person lived overseas six months or more
(620 yes, 428 no, 21 not stated), not by atoll; no language table is published by atoll.

The ability tables, for contrast: 88.1% could hold a conversation in Tokelauan (96.1% in 2006),
48.6% in English, 11.2% in Tuvaluan; 52.3% spoke two or more languages (profile report p. 25, 26).
First learnt (Table 5.7, all 1,197 present): Tokelauan 832, Samoan 96, English 96, Tuvaluan 85,
Kiribati 10, other 34. Not drawn: the home-language table is the one asked for.

## 2. Checks (`sources/tk_census.py`)

Table 5.5's rows sum to its printed totals (620, 428, 21; 1,069); Table 5.4's age rows sum to
the 1,197 present, 128 of them aged 0 to 4, leaving 1,069 aged 5 and over.

## 3. Calls

- Drawn: 1,062 (1,069 less the 6 not stated and the 1 who cannot talk). Gap: the 128 under 5, the
  302 usual residents overseas on census night (religiondots' `countries/tk.py`), and the 7.
- "Other language" is the unnamed remainder on `other`. "Kiribati" is Gilbertese.
- One unit, because the table is national. A proxy by atoll (Tuvaluan or Samoan ethnicity by
  atoll, Table 5.1) could place those languages, but would set per-atoll counts the census did not
  publish, so it is not used. Room for improvement: a language table by atoll from the Statistics
  Office; the 2016 profile says Nukunonu has the most multilingual residents.

## 4. Geography

religiondots' 17 Kontur hexes for Atafu (6), Fakaofo (5) and Nukunonu (6) re-keyed to one unit `TK`
(`sources/tk_geo.py`): Kontur has 642, 662 and 533 people on them (35%, 36%, 29%) against 413, 399
and 385 present on census night (34%, 33%, 32%).
