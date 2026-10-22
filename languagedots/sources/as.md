# American Samoa: the record

Drawn 2026-10-19 (session eca0185b-terr, "coverage gaps"). 44,174 people aged 5 and over in
households, 10 units, 5 nodes from 6 labels, 41 dots and 1 ring at 1:1000. Scripts:
`sources/ia2020.py` (shared reader), `sources/as_census.py` (table and checks),
`taxonomy/as2020.py`, `taxonomy/tree.d/as.txt`, `countries/as.py`. No asks. Geography is
religiondots' (`rd_geo` yes), read in place.

## 1. Source and question

2020 Island Areas Census of American Samoa, Demographic and Housing Characteristics Summary File
(U.S. Census Bureau, public domain), table PCT24 "Age by language spoken at home for the
population 5 years and over in households". A full count (long-form questionnaire for everyone).
The question: does this person speak a language other than English at home, and if so which.
One answer per person, so a Samoan-and-English home counts under Samoan and "English" is English
only.

Files (`data/raw/as/`): `as2020.dhc.zip`, `2020-iac-dhc-american-samoa-table-matrix.xlsx`; URLs in
`sources/ia2020.py`.

PCT24 has only 6 rows here (Guam and the CNMI get 12): English only, Samoan, Tongan, other Pacific
Island languages, Asian languages, other. Totals: Samoan 38,835 (87.9%), other Pacific Island
1,798 (4.1%), English only 1,444 (3.3%), Tongan 913 (2.1%), Asian languages 945 (2.1%), other 239
(0.5%). No finer detail exists for 2020 and there are no public Island Areas microdata, so the
groups cannot be split.

## 2. Grain

PCT24 is published for the territory, 5 districts, 16 counties, 18 tracts and block groups.
Nothing is suppressed. Counties are used, because religiondots' placement layer is on them
(`religiondots/data/geo/as/as_hexes.gpkg`, 10 units: nine counties of Tutuila, and Manu'a, which
religiondots draws as one unit and the Bureau publishes as five counties, Faleasao, Fitiuta, Ofu,
Olosega and Ta'u, summed here). Rose Island and Swains Island hold nobody. Tracts (18) would be
finer on Tutuila's west side (Tualauta county, 20,246 people, holds six tracts); not used.

## 3. Checks (`sources/as_census.py`)

1. PCT24 adds up (bands to total, languages to band) in all 426 records.
2. The 16 counties sum exactly to the 5 districts, and those to the territory, in all 6 rows.
3. The 18 tracts sum exactly to the 5 districts, in all 6 rows.
4. The join to religiondots' units: all 14 counties with people join, and each unit's total
   population (P1) equals religiondots' `census_2020` in every unit (both are the 2020 census).
5. A second table of the same census: PBG5 (the same universe in three groups, by block group)
   summed per tract equals PCT24 collapsed to those groups in all 18 tracts.

## 4. Mapping (`taxonomy/as2020.py`)

| label | node |
|---|---|
| Speak only English | indoeuropean.germanic.english |
| Speak Samoan | austronesian.oceanic.samoan (drawn under polynesian) |
| Speak Tongan | austronesian.oceanic.tongan (drawn under polynesian) |
| Speak other Pacific Island languages | pacific_other |
| Speak Asian languages | other |
| Speak other languages | other |

`pacific_other` as Guam (`sources/gu.md` §4): the Bureau's Pacific Island group also holds Papuan
languages, so no Austronesian node contains everything in it. In American Samoa it is mostly
Polynesian languages (Tuvaluan, Tokelauan, Niuean, Fijian, Cook Islands Maori), but the table
does not say. "Asian languages" spans Filipino languages, Chinese, Korean and others, so `other`,
as Guam's "other Asian languages". No new node; `tree.d/as.txt` repeats existing ones (Samoan,
Tongan from `be.txt`, `pacific_other` from `gu.txt`). Glottolog (`data/raw/glottolog`): Samoan
samo1305 and Tongan tong1325 are Austronesian.

## 5. Geography

religiondots' `as_hexes.gpkg`: Kontur AS 400 m hexes keyed to the 10 units (214 hexes, 43,906
people against 49,710 counted, ratio 0.88; per unit 0.58 (Ituau) to 1.93 (Saole) of the census; the weight only
spreads dots inside a unit; see `religiondots/data/geo/as/as_lookup.csv`). Kontur is only a weight
inside a unit here; the counts are the census's. Seven units "lose over 95% to the sea" in the
scatter's water step and are left unclipped (water.py's stilt-village rule); that comes from the
placement layer and is shared with religiondots' American Samoa. 3,174 people (7.2%) are under one
dot per language nationally and draw as 1 ring.

## 6. Second source

Not searched. religiondots' 2015 household survey (`religiondots/sources/as.py`) asks no language
question.
