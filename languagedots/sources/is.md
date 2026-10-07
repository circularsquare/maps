# Iceland (is): record

Drawn 2026-10-05 (session edd42a8c-mono2). No language question. Hagstofa MAN04203, population by
municipality and citizenship, 1 January 2026: 394,324 people on 62 municipalities, every row
`derived`. Icelandic 88.1%, Polish 4.0%, Spanish 0.8%, Lithuanian 0.7%, Romanian 0.6%, Ukrainian
0.6%, Russian 0.5%, English 0.5%; 190 nodes. Placed on religiondots' 400 m Kontur grid keyed by its
`muni` column (read-only). 375 dots, 179 rings.

Files: `sources/is_pop.py`, `taxonomy/is2026.py` (identity mapping), `taxonomy/tree.d/is.txt`
(214 bare repeats, generated from the CSV's nodes and their parents; no new nodes),
`countries/is.py`, `data/raw/is/man04203_2026.json`, `data/normalized/is.csv`.

## Method (AGENT_BRIEF §2, rich country with no language question)

- **Table**: PxWeb `Ibuar/mannfjoldi/3_bakgrunnur/Rikisfang/MAN04203.px`, both sexes, 2026.
  Asserted: total 394,324; citizenships partition every municipality; municipalities sum to
  Iceland for every citizenship; 62 municipalities, the same codes as religiondots' grid.
  The API answers 429 for minutes at a time; the fetch retries.
- **Icelandic citizens (323,769)** on Icelandic, naturalised immigrants included (`gap`).
- **Foreign or stateless (70,555)**: 22 multilingual origins with 200+ citizens in Iceland at their
  own drawn mix on this map (sa.md's home-mix method, 1% cut): UA, LV, LT, EE, ES, PH, IN, NG,
  AF, IQ, GH, PK, IR, CN, RO, SK, BG, RU, RS, IT, PS, SO. Germany, France, the US, Sweden and
  the Netherlands are left on their main language: their drawn mixes are mostly their own
  immigrants. Everyone else on `fr_build.COUNTRY_LANG`, with Portugal's splits for Canada,
  Switzerland and Belgium, France on French.
- **Retention**: TeO2 (`fr_build.TEO_FRENCH`, by TeO2 region) moved 23,702 people onto Icelandic,
  a third of the foreign citizens (Polish 23,573 before, 15,739 after).
- Stateless (31), unspecified foreign, ex-Yugoslavia unspecified (7): `other`.

## Calls someone might reverse

- TeO2 retention applied to foreign citizens, as Portugal applies it to nationals. Many of
  Iceland's foreign citizens are recent labour migrants; TeO2 measures settled families in
  France, so the third moved onto Icelandic is probably too high. Dropping it is one line.
- Home mixes bring in minority languages (Russian for Latvians 37%, Estonians 29%, Ukrainians
  27%; Catalan for Spaniards) that the citizens in Iceland may not mirror.
- Municipality grain, citizenship rather than country of birth (Hagstofa has birth country by
  municipality too, but citizenship is what COUNTRY_LANG and the other proxy countries use).

## Room for improvement

A survey of immigrants' home language in Iceland (none found) would replace TeO2.
