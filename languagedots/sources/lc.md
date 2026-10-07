# Saint Lucia (lc): record

Drawn 2026-10-05 (session edd42a8c-amer). No census language question; built as Dominica
(`sources/dm.md`). 2022 census birthplace per district, scaled to CSO's published weighted
district totals: 171,835 people (household population) on 10 districts, every row `derived`.
Kweyol 95.7%, other 6,009, English 1,328. 171 dots.

Files: `sources/lc_census.py` (a copy of tt_census.py with Saint Lucia's variables),
`taxonomy/lc2022.py`, `taxonomy/tree.d/lc.txt` (bare repeats), `countries/lc.py`,
`data/raw/lc/` (4 programs + outputs), `data/normalized/lc.csv`.

## 1. Sources

- **CSO 2022 census**, REDATAM base PHC2022 at prod.redatam.org/binlca (open). VARLIST has no
  language item. The public base folds ethnicity (P1_4) to "African Descent/Black", "Other",
  "Not reported", and country of birth (P2_2_2ABROAD_R) to world regions (Latin America and the
  Caribbean 3,420, North America 979, Europe 585, Other 410). It splits Castries into City,
  Suburban and Rural (12 areas), merged into religiondots' one Castries unit.
- The base is the unweighted enumeration (131,825); CSO weighted districts up for a 23%
  undercount (religiondots' lc record). Each district's mix is scaled to the published total
  (religiondots' normalized lc.csv "Total" rows, household population 171,834).
- Checks: 12 areas in every table; crosstab margins within one-way tables; country of birth given
  for every born-elsewhere person; persons agree across two tables.

## 2. Calls

- **Everyone born in Saint Lucia on Kweyol** (dm.txt's Antillean Creole node, per the task brief;
  Glottolog has Saint Lucian Creole French sain1246 separately). English-first Saint Lucians,
  surely many in Castries and Gros Islet, are uncounted; said in note_public. No survey with a
  language item was found (not searched beyond the census and the brief's leads).
- Born in North America on English; born in "Latin America and the Caribbean", "Europe", "Other"
  on `other` (pooled regions; many Caribbean-born speak Kweyol or an English creole, but nothing
  says which).
- White and Indo-Saint Lucians cannot be split out in the public base.

## 3. Room for improvement

A survey asking home language (Kweyol vs English) by district; country of birth by name (CSO's
full tables) would place Caribbean immigrants on their own creoles.
