# Guyana (gy): record

Drawn 2026-10-05 (session edd42a8c-amer). Census 2012 ethnic background by region, read as
language (AGENT_BRIEF §2 ethnicity rule). 746,955 people on 10 regions, every row `derived`.
Guyanese Creole 97.8%, Amerindian language (unnamed) 15,698, English 415. 745 dots, 1 ring.

Files: `sources/gy_census.py`, `taxonomy/gy2012.py`, `taxonomy/tree.d/gy.txt` (bare repeats),
`countries/gy.py`, `data/normalized/gy.csv`.

## 1. Source

Bureau of Statistics, 2012 Census Compendium 2, Table 2.3 (ethnic background x region), from
religiondots' copy of the PDF (read-only), typed into the script. Includes prorated "not stated"
(321) and no-contact persons (16,331). Checks: rows and columns sum to the printed totals
(746,955); region totals equal religiondots' gy_lookup census column. No language question
and no language table in the compendium. Birthplace by region not used (foreign-born about
1.5% in 2012; Venezuelan arrivals came after).

## 2. Calls

- **Amerindian 78,492: 20% on `americas_other`, 80% on Guyanese Creole.** Retention from the
  IDB's "Guyana's Indigenous Peoples 2013 Survey: Final Report" (Bollers, Clarke, Johnny, Wenner,
  2019, doi 10.18235/0001591, pp. 71-72): 337 households in 11 villages, "only 20% of households
  were fluent in their own language", fluency higher further from the capital. Read through
  Wikipedia's citation; the IDB PDF answered 403 to curl and is not on Wayback, so the page was
  not seen directly. A flat 20% understates the Rupununi (Regions 8, 9) and overstates the coast.
  The census has one Amerindian category, so the language is unnamed (`americas_other`, the
  narrowest node holding Arawakan, Cariban and Warao).
- **White on English** (as bb, tt); African, East Indian, Mixed, Portuguese, Chinese, Other on
  Guyanese Creole (creo1235). Caribbean Hindustani (cari1275) not drawn: no figure.
- Berbice and Skepi Creole Dutch are extinct; not drawn.

## 3. Room for improvement

The IDB report's village table (pp. 71-72) would give a hinterland/coast split; a Guyana MICS
(2014, 2019-20) mother-tongue item, if it exists, would give regional retention; a census with
Amerindian nations would let the languages be named.
