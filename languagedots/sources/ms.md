# Montserrat (ms): record

Drawn 2026-10-05 (session edd42a8c-last). 4,768 people (2011 census, place of birth), national,
rows `derived`. 3 dots, 35 rings.

Files: `sources/ms_census.py`, `taxonomy/ms2011.py` (identity), `taxonomy/tree.d/ms.txt`,
`countries/ms.py`, `data/normalized/ms.csv`, `data/raw/ms/ms_q47_born.htm`.

## Source

ECLAC REDATAM WebServer for the Montserrat 2011 PHC (`prod.redatam.org/binmsr`, base PHC2011;
linked from redatam.org/es/procesar-en-linea/caribe/montserrat). This base has no free-program
(PROGRED) item; the Frequency form (`RpWebStats.exe/Frequency`, ITEM FREQ1, ROW
PERSON.Q47_BORN, FORMAT HTML, PERCENT OFF) works by plain POST. Q47 codes the Montserrat-born by
village and the foreign-born by country. Total 4,775 (the base's persons; the published usual
resident count is 4,922); rows sum to it (asserted).

## How

- Montserrat-born (27 villages + "Elsewhere in Montserrat", 2,910) on the Leeward creole
  (`antiguan`, Glottolog anti1245 lists MS).
- Foreign-born: St Kitts's conventions (`sources/kn_census.py`): English-Caribbean creoles by
  name (Guyanese 597, Grenadian, Trinidadian, Virgin Islands; Antigua and St Kitts on the
  Leeward creole), US/Canada/UK on English, the rest through `origin_mix`. "Another country"
  (54) on `other`; 7 don't know / not stated not drawn.
- National: religiondots' layer is one unit; the island's whole population would make 5 dots.

## Calls someone might reverse

- UK- and US-born (238) on English, as St Kitts and Barbados did.
- 2011 vintage; the 2023 census has no open tables yet.
