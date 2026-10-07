# Trinidad and Tobago (tt): record

Drawn 2026-10-05 (session edd42a8c-amer). No census language question; built as Barbados
(`sources/bb.md`). Census 2011, 1,322,547 people (the printed non-institutional total) on 15
municipalities, every row `derived`. Trinidadian Creole 91.2%, Tobagonian Creole 58,124, English
17,037, Guyanese 11,482, Grenadian 9,097, Vincentian 7,190, other 5,037. 1,317 dots, 29 rings.

Files: `sources/tt_census.py`, `taxonomy/tt2011.py`, `taxonomy/tree.d/tt.txt`, `countries/tt.py`,
`data/raw/tt/` (4 programs + outputs), `data/normalized/tt.csv`.

## 1. Sources

- **CSO Census 2011**, REDATAM base PHC2011 at prod.redatam.org/bintto (open; person entity
  `PERSON`). VARLIST has ETHNIC, PLABIRTH, CNTBIRTHR (12 named countries + Other), nothing on
  language. Tables: ETHNIC, PLABIRTH, CNTBIRTHR by municipality, and ETHNIC x PLABIRTH by
  municipality.
- **The REDATAM base holds 1,171,797 persons** against 1,322,547 in the printed non-institutional
  municipal totals (religiondots' normalized tt.csv, from the 2011 demographic report). The mix
  comes from the base, each municipality scaled to its printed total.
- Checks: 15 municipalities in every table; crosstab margins within each one-way table (the
  crosstab drops anyone missing either answer); country of birth given for 87-97% of each
  municipality's foreign-born; persons agree across two tables.

## 2. Calls

- Native-born: Trinidadian Creole (trin1276), Tobagonian Creole (toba1282, new node, hand colour)
  in Tobago; **Caucasian native-born on English** (as bb's white Barbadians).
- **Indo-Trinidadians on Trinidadian Creole.** Trinidad Bhojpuri (trin1268) survives among a few
  elderly speakers; no figure was found, so nothing drawn (task brief: only with a cited figure).
- Foreign-born by country: Caribbean creoles (bb.txt nodes), Saint Lucia on the Kweyol node
  (dm.md), UK/US/Canada English, China, India, Venezuela via origin_mix, CSO's "Other" on `other`.
  The country mix applies to everyone foreign-born in the municipality, scaled for missing
  countries.
- Standard English vs creole is a continuum; no source separates them, so only white natives
  are drawn on English. Venezuelans arriving after 2016 are not in 2011.

## 3. Room for improvement

A 2021/2024 census tabulation (none on REDATAM yet) would add the Venezuelan arrivals; any survey
of home language would split creole from English and count Bhojpuri.
